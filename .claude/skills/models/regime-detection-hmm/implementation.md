# Regime Detection HMM - Implementation Details

Detailed Rust implementation code for the Regime Detection HMM skill. See [SKILL.md](./SKILL.md) for architecture overview and usage guidance.

---

## Emission Model

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

---

## HMM Parameters and Default Transition Matrix

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

## Online HMM Filter (Forward Algorithm)

```rust
struct OnlineHMMFilter {
    params: HMMParams,
    belief: [f64; 4],  // P(regime | all observations so far)
    belief_history: RingBuffer<([f64; 4], u64)>,
}

#[derive(Clone)]
struct ObservationVector {
    timestamp_ns: u64,
    volatility: f64,              // Realized vol over last minute
    trade_intensity: f64,         // Trades per second
    imbalance: f64,               // Signed volume imbalance [-1, 1]
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
        // Prediction step: propagate belief through transition matrix
        let mut predicted = [0.0; 4];
        for j in 0..4 {
            for i in 0..4 {
                predicted[j] += self.params.transition[i][j] * self.belief[i];
            }
        }

        // Update step: weight by emission likelihood and normalize
        let mut updated = [0.0; 4];
        let mut normalizer = 0.0;
        for i in 0..4 {
            let log_likelihood = self.params.emissions[i].log_likelihood(obs);
            let likelihood = log_likelihood.exp().max(1e-100);
            updated[i] = predicted[i] * likelihood;
            normalizer += updated[i];
        }
        for i in 0..4 {
            self.belief[i] = updated[i] / normalizer;
        }

        self.belief_history.push((self.belief, obs.timestamp_ns));
    }

    fn most_likely_regime(&self) -> MarketRegime { /* argmax over belief */ }
    fn regime_probabilities(&self) -> HashMap<MarketRegime, f64> { /* belief as map */ }
    fn probability_of(&self, regime: MarketRegime) -> f64 { /* belief[regime_idx] */ }
}
```

---

## Regime-Specific Parameters and Blending

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
        MarketRegime::Quiet    => RegimeParams { gamma: 0.3, kappa_multiplier: 1.0, spread_floor_bps: 5.0,  max_inventory: 1.0, quote_size_multiplier: 1.0 },
        MarketRegime::Trending => RegimeParams { gamma: 0.5, kappa_multiplier: 0.7, spread_floor_bps: 10.0, max_inventory: 0.5, quote_size_multiplier: 0.8 },
        MarketRegime::Volatile => RegimeParams { gamma: 0.8, kappa_multiplier: 1.5, spread_floor_bps: 15.0, max_inventory: 0.3, quote_size_multiplier: 0.6 },
        MarketRegime::Cascade  => RegimeParams { gamma: 2.0, kappa_multiplier: 5.0, spread_floor_bps: 50.0, max_inventory: 0.1, quote_size_multiplier: 0.3 },
    }
}

fn blend_params_by_belief(filter: &OnlineHMMFilter) -> RegimeParams {
    let probs = filter.regime_probabilities();
    let mut blended = RegimeParams { gamma: 0.0, kappa_multiplier: 0.0, spread_floor_bps: 0.0, max_inventory: 0.0, quote_size_multiplier: 0.0 };

    for (regime, prob) in &probs {
        let params = get_regime_params(*regime);
        // Linear blend for additive params
        blended.spread_floor_bps += prob * params.spread_floor_bps;
        blended.max_inventory += prob * params.max_inventory;
        blended.quote_size_multiplier += prob * params.quote_size_multiplier;
        // Log-space blend for multiplicative params (gamma, kappa)
        blended.gamma += prob * params.gamma.ln();
        blended.kappa_multiplier += prob * params.kappa_multiplier.ln();
    }

    blended.gamma = blended.gamma.exp();
    blended.kappa_multiplier = blended.kappa_multiplier.exp();
    blended
}
```

---

## Parameter Learning (Baum-Welch)

Full EM algorithm for offline HMM parameter estimation.

```rust
fn baum_welch_iteration(
    params: &mut HMMParams,
    observations: &[ObservationVector],
) {
    let n = observations.len();
    let k = 4;  // Number of states

    // === Forward Pass (alpha) ===
    let mut alpha = vec![[0.0f64; 4]; n];
    let mut scale = vec![0.0f64; n];  // For numerical stability

    // Initialize
    for i in 0..k {
        alpha[0][i] = params.initial[i] * params.emissions[i].log_likelihood(&observations[0]).exp();
    }
    scale[0] = alpha[0].iter().sum::<f64>();
    for i in 0..k { alpha[0][i] /= scale[0]; }

    // Forward recursion
    for t in 1..n {
        for j in 0..k {
            let mut sum = 0.0;
            for i in 0..k { sum += alpha[t-1][i] * params.transition[i][j]; }
            let likelihood = params.emissions[j].log_likelihood(&observations[t]).exp();
            alpha[t][j] = sum * likelihood;
        }
        scale[t] = alpha[t].iter().sum::<f64>().max(1e-100);
        for i in 0..k { alpha[t][i] /= scale[t]; }
    }

    // === Backward Pass (beta) ===
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

    // === Compute gamma and xi ===
    let mut gamma = vec![[0.0f64; 4]; n];
    let mut xi = vec![[[0.0f64; 4]; 4]; n-1];

    // gamma[t][i] = P(state_t = i | observations)
    for t in 0..n {
        let mut sum = 0.0;
        for i in 0..k { gamma[t][i] = alpha[t][i] * beta[t][i]; sum += gamma[t][i]; }
        for i in 0..k { gamma[t][i] /= sum.max(1e-100); }
    }

    // xi[t][i][j] = P(state_t = i, state_{t+1} = j | observations)
    for t in 0..n-1 {
        let mut sum = 0.0;
        for i in 0..k {
            for j in 0..k {
                let likelihood = params.emissions[j].log_likelihood(&observations[t+1]).exp();
                xi[t][i][j] = alpha[t][i] * params.transition[i][j] * likelihood * beta[t+1][j];
                sum += xi[t][i][j];
            }
        }
        for i in 0..k { for j in 0..k { xi[t][i][j] /= sum.max(1e-100); } }
    }

    // === M-Step: Update Parameters ===
    // Transition probabilities
    for i in 0..k {
        let denom: f64 = (0..n-1).map(|t| gamma[t][i]).sum();
        for j in 0..k {
            let numer: f64 = (0..n-1).map(|t| xi[t][i][j]).sum();
            params.transition[i][j] = (numer / denom.max(1e-10)).max(0.001);
        }
        let row_sum: f64 = params.transition[i].iter().sum();
        for j in 0..k { params.transition[i][j] /= row_sum; }
    }

    // Emission parameters (simplified: update means)
    for i in 0..k {
        let weight_sum: f64 = (0..n).map(|t| gamma[t][i]).sum();
        let vol_log_sum: f64 = (0..n).map(|t| gamma[t][i] * observations[t].volatility.ln()).sum();
        params.emissions[i].volatility_log_mean = vol_log_sum / weight_sum.max(1e-10);
        let int_sum: f64 = (0..n).map(|t| gamma[t][i] * observations[t].trade_intensity).sum();
        params.emissions[i].trade_intensity_mean = int_sum / weight_sum.max(1e-10);
        let imb_sq_sum: f64 = (0..n).map(|t| gamma[t][i] * observations[t].imbalance.powi(2)).sum();
        params.emissions[i].imbalance_std = (imb_sq_sum / weight_sum.max(1e-10)).sqrt();
    }

    params.initial = gamma[0];
}

fn train_hmm(observations: &[ObservationVector], max_iterations: usize) -> HMMParams {
    let mut params = default_hmm_params();
    for iter in 0..max_iterations {
        let ll_before = compute_log_likelihood(&params, observations);
        baum_welch_iteration(&mut params, observations);
        let ll_after = compute_log_likelihood(&params, observations);
        println!("Iteration {}: log-likelihood {} -> {}", iter, ll_before, ll_after);
        if (ll_after - ll_before).abs() < 0.001 { break; }
    }
    params
}
```

---

## Quote Engine Integration

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
        if predicted == true_regime { correct += 1; }
        total += 1;
    }

    ValidationReport {
        accuracy: correct as f64 / total as f64,
        confusion_matrix: confusion,
    }
}
```

### Economic Value Backtest

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
