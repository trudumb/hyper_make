# Regime Detection (HMM) Skill

## Purpose

Replace crude regime proxies (warmup multipliers, hard thresholds) with proper Bayesian belief tracking over market states. This enables:

- Smooth parameter blending between regimes
- Anticipating regime transitions
- Regime-specific model deployment
- Principled uncertainty handling

## When to Use

- Implementing regime-dependent parameters
- Building the "should I be aggressive or defensive" decision
- Debugging regime-specific losses
- Replacing hard-coded volatility thresholds

## Prerequisites

- `measurement-infrastructure` for validation
- Historical market data with regime labels (can be derived)
- Understanding of your loss patterns by market condition

---

## State Space Definition

```rust
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum MarketRegime {
    Quiet,      // Low volatility, balanced flow, normal fill rates
    Trending,   // Directional momentum, elevated adverse selection
    Volatile,   // High volatility, wide spreads, uncertain direction
    Cascade,    // Liquidation cascade, extreme toxicity
}
```

### Regime Characteristics

| Regime | Volatility | Trade Rate | Imbalance | Adverse Selection | Duration |
|--------|-----------|------------|-----------|-------------------|----------|
| Quiet | Low (σ < 0.01%) | Low-Normal | Balanced | Low (~20%) | Hours |
| Trending | Medium | Normal-High | Directional | Medium (~40%) | 10-60 min |
| Volatile | High (σ > 0.03%) | High | Variable | Medium (~35%) | Minutes-Hours |
| Cascade | Extreme | Very High | Extreme | Very High (~70%) | Minutes |

---

## Hidden Markov Model Specification

### Emission Model

For each regime, model the distribution of observables:

```rust
struct RegimeEmissionModel {
    // Volatility (log-normal)
    volatility_log_mean: f64,
    volatility_log_std: f64,
    
    // Trade intensity (gamma or log-normal)
    trade_intensity_mean: f64,
    trade_intensity_std: f64,
    
    // Imbalance (normal, can be signed)
    imbalance_mean: f64,
    imbalance_std: f64,
    
    // Adverse selection rate (beta distribution)
    adverse_selection_alpha: f64,
    adverse_selection_beta: f64,
}

impl RegimeEmissionModel {
    fn log_likelihood(&self, obs: &ObservationVector) -> f64 {
        let mut ll = 0.0;
        
        // Volatility (log-normal)
        let log_vol = obs.volatility.ln();
        ll += gaussian_log_pdf(log_vol, self.volatility_log_mean, self.volatility_log_std);
        
        // Trade intensity (log-normal for simplicity)
        let log_int = obs.trade_intensity.max(0.1).ln();
        let int_log_mean = self.trade_intensity_mean.ln();
        let int_log_std = self.trade_intensity_std / self.trade_intensity_mean;  // Approx
        ll += gaussian_log_pdf(log_int, int_log_mean, int_log_std);
        
        // Imbalance (normal)
        ll += gaussian_log_pdf(obs.imbalance, self.imbalance_mean, self.imbalance_std);
        
        // Adverse selection (beta, but use normal approx for speed)
        ll += gaussian_log_pdf(
            obs.adverse_selection_rate,
            self.adverse_selection_alpha / (self.adverse_selection_alpha + self.adverse_selection_beta),
            0.1,  // Fixed std for simplicity
        );
        
        ll
    }
}

fn gaussian_log_pdf(x: f64, mean: f64, std: f64) -> f64 {
    let z = (x - mean) / std;
    -0.5 * z * z - std.ln() - 0.5 * (2.0 * PI).ln()
}
```

### Transition Matrix

```rust
struct HMMParams {
    // Transition probabilities: trans[i][j] = P(state_t = j | state_{t-1} = i)
    transition: [[f64; 4]; 4],
    
    // Emission models for each state
    emissions: [RegimeEmissionModel; 4],
    
    // Initial state distribution
    initial: [f64; 4],
}

fn default_hmm_params() -> HMMParams {
    HMMParams {
        transition: [
            // From Quiet: mostly stays quiet
            [0.95, 0.03, 0.019, 0.001],
            // From Trending: can stay, revert to quiet, or escalate
            [0.10, 0.85, 0.04, 0.01],
            // From Volatile: can calm down, trend, or escalate
            [0.15, 0.10, 0.70, 0.05],
            // From Cascade: usually short, reverts to volatile
            [0.05, 0.05, 0.60, 0.30],
        ],
        emissions: [
            // Quiet
            RegimeEmissionModel {
                volatility_log_mean: -9.2,  // ~0.01%
                volatility_log_std: 0.5,
                trade_intensity_mean: 0.5,
                trade_intensity_std: 0.3,
                imbalance_mean: 0.0,
                imbalance_std: 0.15,
                adverse_selection_alpha: 2.0,
                adverse_selection_beta: 8.0,  // ~20%
            },
            // Trending
            RegimeEmissionModel {
                volatility_log_mean: -8.5,  // ~0.02%
                volatility_log_std: 0.4,
                trade_intensity_mean: 1.5,
                trade_intensity_std: 0.8,
                imbalance_mean: 0.0,  // Can be positive or negative
                imbalance_std: 0.3,   // But more variable
                adverse_selection_alpha: 4.0,
                adverse_selection_beta: 6.0,  // ~40%
            },
            // Volatile
            RegimeEmissionModel {
                volatility_log_mean: -7.8,  // ~0.04%
                volatility_log_std: 0.5,
                trade_intensity_mean: 3.0,
                trade_intensity_std: 1.5,
                imbalance_mean: 0.0,
                imbalance_std: 0.4,
                adverse_selection_alpha: 3.5,
                adverse_selection_beta: 6.5,  // ~35%
            },
            // Cascade
            RegimeEmissionModel {
                volatility_log_mean: -6.9,  // ~0.1%
                volatility_log_std: 0.6,
                trade_intensity_mean: 10.0,
                trade_intensity_std: 5.0,
                imbalance_mean: 0.0,
                imbalance_std: 0.6,
                adverse_selection_alpha: 7.0,
                adverse_selection_beta: 3.0,  // ~70%
            },
        ],
        initial: [0.7, 0.15, 0.14, 0.01],
    }
}
```

---

## Online Filtering (Forward Algorithm)

Maintain a belief state that updates with each observation:

```rust
struct OnlineHMMFilter {
    params: HMMParams,
    
    // Current belief: P(regime | all observations so far)
    belief: [f64; 4],
    
    // For debugging/analysis
    belief_history: RingBuffer<([f64; 4], u64)>,
}

#[derive(Clone)]
struct ObservationVector {
    timestamp_ns: u64,
    volatility: f64,           // Realized vol over last minute
    trade_intensity: f64,      // Trades per second
    imbalance: f64,            // Signed volume imbalance [-1, 1]
    adverse_selection_rate: f64,  // Fraction of recent fills that were toxic
}

impl OnlineHMMFilter {
    fn new(params: HMMParams) -> Self {
        OnlineHMMFilter {
            belief: params.initial,
            params,
            belief_history: RingBuffer::new(1000),
        }
    }
    
    fn update(&mut self, obs: &ObservationVector) {
        // === Prediction Step ===
        // P(state_t | obs_{1:t-1}) = Σ_i P(state_t | state_{t-1}=i) * P(state_{t-1}=i | obs_{1:t-1})
        let mut predicted = [0.0; 4];
        for j in 0..4 {
            for i in 0..4 {
                predicted[j] += self.params.transition[i][j] * self.belief[i];
            }
        }
        
        // === Update Step ===
        // P(state_t | obs_{1:t}) ∝ P(obs_t | state_t) * P(state_t | obs_{1:t-1})
        let mut updated = [0.0; 4];
        let mut normalizer = 0.0;
        
        for i in 0..4 {
            let log_likelihood = self.params.emissions[i].log_likelihood(obs);
            let likelihood = log_likelihood.exp().max(1e-100);  // Prevent underflow
            updated[i] = predicted[i] * likelihood;
            normalizer += updated[i];
        }
        
        // Normalize
        for i in 0..4 {
            self.belief[i] = updated[i] / normalizer;
        }
        
        // Store history
        self.belief_history.push((self.belief, obs.timestamp_ns));
    }
    
    fn most_likely_regime(&self) -> MarketRegime {
        let max_idx = self.belief.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap().0;
        
        match max_idx {
            0 => MarketRegime::Quiet,
            1 => MarketRegime::Trending,
            2 => MarketRegime::Volatile,
            3 => MarketRegime::Cascade,
            _ => unreachable!(),
        }
    }
    
    fn regime_probabilities(&self) -> HashMap<MarketRegime, f64> {
        let mut probs = HashMap::new();
        probs.insert(MarketRegime::Quiet, self.belief[0]);
        probs.insert(MarketRegime::Trending, self.belief[1]);
        probs.insert(MarketRegime::Volatile, self.belief[2]);
        probs.insert(MarketRegime::Cascade, self.belief[3]);
        probs
    }
    
    fn probability_of(&self, regime: MarketRegime) -> f64 {
        match regime {
            MarketRegime::Quiet => self.belief[0],
            MarketRegime::Trending => self.belief[1],
            MarketRegime::Volatile => self.belief[2],
            MarketRegime::Cascade => self.belief[3],
        }
    }
}
```

---

## Regime-Specific Parameters

Define optimal parameters for each regime:

```rust
struct RegimeParams {
    gamma: f64,              // Risk aversion
    kappa_multiplier: f64,   // Fill rate multiplier
    spread_floor_bps: f64,   // Minimum spread
    max_inventory: f64,      // Inventory limit (fraction of usual)
    quote_size_multiplier: f64,
}

fn get_regime_params(regime: MarketRegime) -> RegimeParams {
    match regime {
        MarketRegime::Quiet => RegimeParams {
            gamma: 0.3,
            kappa_multiplier: 1.0,
            spread_floor_bps: 5.0,
            max_inventory: 1.0,
            quote_size_multiplier: 1.0,
        },
        MarketRegime::Trending => RegimeParams {
            gamma: 0.5,               // More risk-averse
            kappa_multiplier: 0.7,    // Lower fill rate on wrong side
            spread_floor_bps: 10.0,
            max_inventory: 0.5,       // Tighter limits
            quote_size_multiplier: 0.8,
        },
        MarketRegime::Volatile => RegimeParams {
            gamma: 0.8,
            kappa_multiplier: 1.5,    // Higher fill rates
            spread_floor_bps: 15.0,
            max_inventory: 0.3,
            quote_size_multiplier: 0.6,
        },
        MarketRegime::Cascade => RegimeParams {
            gamma: 2.0,               // Extreme risk aversion
            kappa_multiplier: 5.0,    // Very high but toxic fills
            spread_floor_bps: 50.0,   // Wide spreads
            max_inventory: 0.1,       // Minimal inventory
            quote_size_multiplier: 0.3,
        },
    }
}
```

### Parameter Blending

Don't hard-switch between regimes—blend by belief probability:

```rust
fn blend_params_by_belief(filter: &OnlineHMMFilter) -> RegimeParams {
    let probs = filter.regime_probabilities();
    
    let mut blended = RegimeParams {
        gamma: 0.0,
        kappa_multiplier: 0.0,
        spread_floor_bps: 0.0,
        max_inventory: 0.0,
        quote_size_multiplier: 0.0,
    };
    
    for (regime, prob) in &probs {
        let params = get_regime_params(*regime);
        
        // Linear blend for most params
        blended.spread_floor_bps += prob * params.spread_floor_bps;
        blended.max_inventory += prob * params.max_inventory;
        blended.quote_size_multiplier += prob * params.quote_size_multiplier;
        
        // Log-space blend for gamma and kappa (multiplicative quantities)
        blended.gamma += prob * params.gamma.ln();
        blended.kappa_multiplier += prob * params.kappa_multiplier.ln();
    }
    
    // Convert back from log-space
    blended.gamma = blended.gamma.exp();
    blended.kappa_multiplier = blended.kappa_multiplier.exp();
    
    blended
}
```

---

## Parameter Learning (Baum-Welch)

Periodically re-estimate HMM parameters from historical data:

```rust
fn baum_welch_iteration(
    params: &mut HMMParams,
    observations: &[ObservationVector],
) {
    let n = observations.len();
    let k = 4;  // Number of states
    
    // === Forward Pass (α) ===
    let mut alpha = vec![[0.0f64; 4]; n];
    let mut scale = vec![0.0f64; n];  // For numerical stability
    
    // Initialize
    for i in 0..k {
        alpha[0][i] = params.initial[i] * params.emissions[i].log_likelihood(&observations[0]).exp();
    }
    scale[0] = alpha[0].iter().sum::<f64>();
    for i in 0..k {
        alpha[0][i] /= scale[0];
    }
    
    // Forward recursion
    for t in 1..n {
        for j in 0..k {
            let mut sum = 0.0;
            for i in 0..k {
                sum += alpha[t-1][i] * params.transition[i][j];
            }
            let likelihood = params.emissions[j].log_likelihood(&observations[t]).exp();
            alpha[t][j] = sum * likelihood;
        }
        scale[t] = alpha[t].iter().sum::<f64>().max(1e-100);
        for i in 0..k {
            alpha[t][i] /= scale[t];
        }
    }
    
    // === Backward Pass (β) ===
    let mut beta = vec![[1.0f64; 4]; n];
    
    for t in (0..n-1).rev() {
        for i in 0..k {
            let mut sum = 0.0;
            for j in 0..k {
                let likelihood = params.emissions[j].log_likelihood(&observations[t+1]).exp();
                sum += params.transition[i][j] * likelihood * beta[t+1][j];
            }
            beta[t][i] = sum / scale[t+1];
        }
    }
    
    // === Compute γ and ξ ===
    let mut gamma = vec![[0.0f64; 4]; n];
    let mut xi = vec![[[0.0f64; 4]; 4]; n-1];
    
    // γ[t][i] = P(state_t = i | observations)
    for t in 0..n {
        let mut sum = 0.0;
        for i in 0..k {
            gamma[t][i] = alpha[t][i] * beta[t][i];
            sum += gamma[t][i];
        }
        for i in 0..k {
            gamma[t][i] /= sum.max(1e-100);
        }
    }
    
    // ξ[t][i][j] = P(state_t = i, state_{t+1} = j | observations)
    for t in 0..n-1 {
        let mut sum = 0.0;
        for i in 0..k {
            for j in 0..k {
                let likelihood = params.emissions[j].log_likelihood(&observations[t+1]).exp();
                xi[t][i][j] = alpha[t][i] * params.transition[i][j] * likelihood * beta[t+1][j];
                sum += xi[t][i][j];
            }
        }
        for i in 0..k {
            for j in 0..k {
                xi[t][i][j] /= sum.max(1e-100);
            }
        }
    }
    
    // === M-Step: Update Parameters ===
    
    // Update transition probabilities
    for i in 0..k {
        let denom: f64 = (0..n-1).map(|t| gamma[t][i]).sum();
        for j in 0..k {
            let numer: f64 = (0..n-1).map(|t| xi[t][i][j]).sum();
            params.transition[i][j] = (numer / denom.max(1e-10)).max(0.001);
        }
        // Normalize
        let row_sum: f64 = params.transition[i].iter().sum();
        for j in 0..k {
            params.transition[i][j] /= row_sum;
        }
    }
    
    // Update emission parameters (simplified: just update means)
    for i in 0..k {
        let weight_sum: f64 = (0..n).map(|t| gamma[t][i]).sum();
        
        // Volatility (log-mean)
        let vol_log_sum: f64 = (0..n)
            .map(|t| gamma[t][i] * observations[t].volatility.ln())
            .sum();
        params.emissions[i].volatility_log_mean = vol_log_sum / weight_sum.max(1e-10);
        
        // Trade intensity
        let int_sum: f64 = (0..n)
            .map(|t| gamma[t][i] * observations[t].trade_intensity)
            .sum();
        params.emissions[i].trade_intensity_mean = int_sum / weight_sum.max(1e-10);
        
        // Imbalance std
        let imb_sq_sum: f64 = (0..n)
            .map(|t| gamma[t][i] * observations[t].imbalance.powi(2))
            .sum();
        params.emissions[i].imbalance_std = (imb_sq_sum / weight_sum.max(1e-10)).sqrt();
    }
    
    // Update initial distribution
    params.initial = gamma[0];
}

fn train_hmm(observations: &[ObservationVector], max_iterations: usize) -> HMMParams {
    let mut params = default_hmm_params();
    
    for iter in 0..max_iterations {
        let ll_before = compute_log_likelihood(&params, observations);
        baum_welch_iteration(&mut params, observations);
        let ll_after = compute_log_likelihood(&params, observations);
        
        println!("Iteration {}: log-likelihood {} -> {}", iter, ll_before, ll_after);
        
        // Convergence check
        if (ll_after - ll_before).abs() < 0.001 {
            break;
        }
    }
    
    params
}
```

---

## Integration with Quote Engine

```rust
impl QuoteEngine {
    fn generate_quotes(&mut self, market_data: &MarketData) -> QuoteSet {
        // Update HMM with current observation
        let obs = ObservationVector {
            timestamp_ns: market_data.timestamp_ns,
            volatility: self.volatility_estimator.get_sigma(),
            trade_intensity: market_data.trades_per_second,
            imbalance: market_data.trade_imbalance,
            adverse_selection_rate: self.recent_adverse_selection_rate(),
        };
        self.hmm_filter.update(&obs);
        
        // Get blended parameters
        let regime_params = blend_params_by_belief(&self.hmm_filter);
        
        // Use blended params in GLFT
        let gamma = regime_params.gamma;
        let kappa = self.base_kappa * regime_params.kappa_multiplier;
        
        let half_spread = (1.0 / gamma) * (1.0 + gamma / kappa).ln() + MAKER_FEE;
        let spread_floor = regime_params.spread_floor_bps / 10000.0;
        let final_half_spread = half_spread.max(spread_floor);
        
        // Inventory limits from regime
        let max_inv = self.base_max_inventory * regime_params.max_inventory;
        
        // Generate quotes...
    }
}
```

---

## Validation

### Regime Prediction Accuracy

```rust
fn validate_regime_detection(
    filter: &OnlineHMMFilter,
    observations: &[ObservationVector],
    true_regimes: &[MarketRegime],  // Ground truth from ex-post labeling
) -> ValidationReport {
    let mut correct = 0;
    let mut total = 0;
    let mut confusion = [[0usize; 4]; 4];
    
    let mut filter = filter.clone();
    
    for (obs, &true_regime) in observations.iter().zip(true_regimes) {
        filter.update(obs);
        let predicted = filter.most_likely_regime();
        
        let pred_idx = regime_to_idx(predicted);
        let true_idx = regime_to_idx(true_regime);
        
        confusion[true_idx][pred_idx] += 1;
        if predicted == true_regime {
            correct += 1;
        }
        total += 1;
    }
    
    ValidationReport {
        accuracy: correct as f64 / total as f64,
        confusion_matrix: confusion,
    }
}
```

### Economic Value

The real test: does regime detection improve PnL?

```rust
fn backtest_with_regime_detection(
    market_data: &[MarketData],
    with_hmm: bool,
) -> BacktestResult {
    let mut engine = if with_hmm {
        QuoteEngine::with_regime_detection()
    } else {
        QuoteEngine::without_regime_detection()
    };
    
    let mut pnl = 0.0;
    // ... run backtest ...
    
    BacktestResult { pnl, sharpe, max_drawdown }
}
```

---

## Dependencies

- **Requires**: measurement-infrastructure, historical market data
- **Enables**: Regime-specific parameters, smooth parameter blending

## Common Mistakes

1. **Hard switching**: Use probability blending, not argmax
2. **Too many states**: 4 is usually enough; more = overfitting
3. **Ignoring transition dynamics**: Time in regime matters
4. **Static parameters**: Retrain Baum-Welch periodically
5. **Observation selection**: Include adverse selection rate, not just vol/intensity

## Next Steps

1. Define 4 regimes based on your trading experience
2. Set reasonable initial emission parameters
3. Implement online filter
4. Implement parameter blending
5. Validate on historical data
6. Set up weekly Baum-Welch retraining
7. Monitor regime distribution in production
