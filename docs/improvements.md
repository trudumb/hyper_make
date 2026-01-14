Let me build this out properly. This will be comprehensive.

---

## Part 1: Measurement Infrastructure

Before improving any model, you need to know exactly how wrong your current models are and in what ways. This is the foundation everything else builds on.

### 1.1 The Prediction Log Schema

Every quote cycle, your system makes implicit predictions. These need to be recorded with enough granularity to diagnose failures.

```rust
struct PredictionRecord {
    // Timing
    timestamp_ns: u64,
    quote_cycle_id: u64,
    
    // Market state at prediction time
    market_state: MarketStateSnapshot,
    
    // Model outputs (what we predicted)
    predictions: ModelPredictions,
    
    // What actually happened (filled in async)
    outcomes: Option<ObservedOutcomes>,
}

struct MarketStateSnapshot {
    // L2 book state
    bid_levels: Vec<(f64, f64)>,  // (price, size) for top N levels
    ask_levels: Vec<(f64, f64)>,
    
    // Derived quantities your models use
    microprice: f64,
    microprice_std: f64,  // uncertainty on microprice
    
    // Kappa inputs
    kappa_book: f64,
    kappa_robust: f64,
    kappa_own: f64,
    kappa_final: f64,
    
    // Volatility
    sigma_bipower: f64,
    sigma_realized_1m: f64,
    
    // Gamma inputs
    gamma_base: f64,
    gamma_effective: f64,
    
    // External state
    funding_rate: f64,
    time_to_funding_settlement_s: f64,
    open_interest: f64,
    open_interest_delta_1m: f64,
    
    // Cross-exchange (if available)
    binance_mid: Option<f64>,
    binance_hl_spread: Option<f64>,
    
    // Your position
    inventory: f64,
    inventory_age_s: f64,
}

struct ModelPredictions {
    // For each quote level you're placing
    levels: Vec<LevelPrediction>,
    
    // Aggregate predictions
    expected_fill_rate_1s: f64,
    expected_fill_rate_10s: f64,
    expected_adverse_selection_bps: f64,
    regime_probabilities: HashMap<Regime, f64>,
}

struct LevelPrediction {
    side: Side,
    price: f64,
    size: f64,
    depth_from_mid_bps: f64,
    
    // Fill probability predictions at different horizons
    p_fill_100ms: f64,
    p_fill_1s: f64,
    p_fill_10s: f64,
    
    // Conditional predictions
    p_adverse_given_fill: f64,  // P(price moves against us | we get filled)
    expected_pnl_given_fill: f64,
    
    // Queue position estimate
    estimated_queue_position: f64,
    estimated_queue_total: f64,
}

struct ObservedOutcomes {
    // Fill outcomes
    fills: Vec<FillOutcome>,
    
    // Price evolution
    price_1s_later: f64,
    price_10s_later: f64,
    price_60s_later: f64,
    
    // Did we get adversely selected?
    adverse_selection_realized_bps: f64,
}

struct FillOutcome {
    level_index: usize,
    fill_timestamp_ns: u64,
    fill_price: f64,
    fill_size: f64,
    
    // Post-fill price evolution
    mark_price_at_fill: f64,
    mark_price_100ms_later: f64,
    mark_price_1s_later: f64,
    mark_price_10s_later: f64,
}
```

### 1.2 Calibration Analysis Pipeline

Once you have prediction logs, you need systematic analysis to find miscalibration.

**Step 1: Probability Calibration Curves**

For each prediction type (fill probability at different horizons), bin predictions and compare to realized frequencies:

```
Algorithm: BuildCalibrationCurve
Input: predictions[], outcomes[], num_bins=20

1. Sort (prediction, outcome) pairs by prediction value
2. Divide into num_bins equal-sized buckets
3. For each bucket:
   - mean_predicted = average of predictions in bucket
   - realized_frequency = fraction where outcome=True
   - count = number of samples in bucket
4. Return [(mean_predicted, realized_frequency, count), ...]
```

**Interpretation:**
- Perfect calibration: points lie on y=x diagonal
- Overconfident: curve below diagonal (you predict 70%, reality is 50%)
- Underconfident: curve above diagonal

**Step 2: Brier Score Decomposition**

The Brier score is the mean squared error of probability predictions:

```
BS = (1/N) Σ (pᵢ - oᵢ)²
```

Where pᵢ is predicted probability and oᵢ ∈ {0,1} is outcome.

Decompose into three components:

```
BS = Reliability - Resolution + Uncertainty

Reliability = (1/N) Σₖ nₖ(p̄ₖ - ōₖ)²
  - Measures calibration quality
  - Lower is better
  - If your 70% predictions hit 70%, this is 0

Resolution = (1/N) Σₖ nₖ(ōₖ - ō)²
  - Measures discrimination ability
  - Higher is better
  - Are your high predictions different from low predictions?

Uncertainty = ō(1 - ō)
  - Base rate variance
  - Not controllable, just the inherent difficulty
```

**Implementation:**

```rust
struct BrierDecomposition {
    brier_score: f64,
    reliability: f64,
    resolution: f64,
    uncertainty: f64,
}

fn compute_brier_decomposition(
    predictions: &[f64],
    outcomes: &[bool],
    num_bins: usize
) -> BrierDecomposition {
    let n = predictions.len() as f64;
    let o_bar: f64 = outcomes.iter().map(|&o| if o { 1.0 } else { 0.0 }).sum::<f64>() / n;
    
    // Bin predictions
    let mut bins: Vec<Vec<(f64, bool)>> = vec![vec![]; num_bins];
    for (&p, &o) in predictions.iter().zip(outcomes.iter()) {
        let bin_idx = ((p * num_bins as f64) as usize).min(num_bins - 1);
        bins[bin_idx].push((p, o));
    }
    
    let mut reliability = 0.0;
    let mut resolution = 0.0;
    
    for bin in &bins {
        if bin.is_empty() { continue; }
        
        let n_k = bin.len() as f64;
        let p_bar_k: f64 = bin.iter().map(|(p, _)| p).sum::<f64>() / n_k;
        let o_bar_k: f64 = bin.iter().map(|(_, o)| if *o { 1.0 } else { 0.0 }).sum::<f64>() / n_k;
        
        reliability += n_k * (p_bar_k - o_bar_k).powi(2);
        resolution += n_k * (o_bar_k - o_bar).powi(2);
    }
    
    reliability /= n;
    resolution /= n;
    let uncertainty = o_bar * (1.0 - o_bar);
    let brier_score = reliability - resolution + uncertainty;
    
    BrierDecomposition { brier_score, reliability, resolution, uncertainty }
}
```

**Step 3: Conditional Calibration Analysis**

Overall calibration can hide regime-dependent failures. Slice your calibration analysis by:

```
Conditioning Variables:
├── Volatility regime (σ quartiles)
├── Funding rate regime (positive/negative/extreme)
├── Time of day (funding settlement windows)
├── Inventory state (long/flat/short)
├── Recent fill rate (active/quiet)
├── Book imbalance (bid-heavy/balanced/ask-heavy)
└── Cross-exchange state (HL leading/lagging/synced)
```

For each slice, compute separate calibration curves and Brier decomposition. This reveals:
- "Model is well-calibrated except when funding is extreme"
- "Fill predictions are overconfident during high volatility"
- "Adverse selection predictions fail when we're already long"

**Step 4: Information Ratio by Prediction Type**

For each model output, compute how much information it provides:

```
Information Ratio = Resolution / Uncertainty

IR > 1.0: Model predictions carry useful information
IR ≈ 1.0: Model is roughly as good as predicting base rate
IR < 1.0: Model is adding noise
```

Track IR over time. If IR degrades, your model is becoming stale.

### 1.3 Outcome Attribution Pipeline

When you lose money, you need to know why. Build an attribution system:

```rust
struct CycleAttribution {
    cycle_id: u64,
    gross_pnl: f64,
    
    // Decomposition
    spread_capture: f64,      // Revenue from bid-ask spread
    adverse_selection: f64,   // Loss from fills before adverse moves
    inventory_cost: f64,      // Cost of holding inventory
    fee_cost: f64,            // Exchange fees
    
    // Model accuracy
    fill_prediction_error: f64,
    adverse_selection_prediction_error: f64,
    volatility_prediction_error: f64,
}

fn attribute_cycle_pnl(record: &PredictionRecord) -> CycleAttribution {
    let outcomes = record.outcomes.as_ref().expect("Need outcomes");
    
    let mut spread_capture = 0.0;
    let mut adverse_selection = 0.0;
    let mut fee_cost = 0.0;
    
    for fill in &outcomes.fills {
        let level = &record.predictions.levels[fill.level_index];
        
        // Spread capture: distance from mid at fill time
        let mid_at_fill = fill.mark_price_at_fill;
        let spread_earned = match level.side {
            Side::Bid => mid_at_fill - fill.fill_price,
            Side::Ask => fill.fill_price - mid_at_fill,
        };
        spread_capture += spread_earned * fill.fill_size;
        
        // Adverse selection: price move against us after fill
        let price_move = fill.mark_price_1s_later - fill.mark_price_at_fill;
        let adverse = match level.side {
            Side::Bid => -price_move,  // Bought, price dropped
            Side::Ask => price_move,   // Sold, price rose
        };
        adverse_selection += adverse.min(0.0) * fill.fill_size;
        
        // Fees
        fee_cost += fill.fill_price * fill.fill_size * 0.00015; // 1.5 bps maker
    }
    
    // Inventory cost: mark-to-market on held inventory
    let inventory_cost = compute_inventory_mtm(record);
    
    // Prediction errors
    let fill_prediction_error = compute_fill_prediction_error(record);
    let adverse_selection_prediction_error = compute_as_prediction_error(record);
    let volatility_prediction_error = compute_vol_prediction_error(record);
    
    CycleAttribution {
        cycle_id: record.quote_cycle_id,
        gross_pnl: spread_capture + adverse_selection - fee_cost - inventory_cost,
        spread_capture,
        adverse_selection,
        inventory_cost,
        fee_cost,
        fill_prediction_error,
        adverse_selection_prediction_error,
        volatility_prediction_error,
    }
}
```

**Daily Attribution Report:**

```
=== PnL Attribution: 2026-01-13 ===

Gross PnL:              -$127.45
├── Spread Capture:     +$284.30  (working correctly)
├── Adverse Selection:  -$312.80  (THIS IS THE PROBLEM)
├── Inventory Cost:     -$45.20   (acceptable)
└── Fees:               -$53.75   (fixed cost)

Model Accuracy:
├── Fill Prediction:    Brier=0.18, IR=1.34  (good)
├── Adverse Selection:  Brier=0.31, IR=0.89  (NEEDS WORK)
└── Volatility:         RMSE=0.000012        (acceptable)

Regime Breakdown:
├── Quiet (68% of time):    +$89.20
├── Active (28% of time):   -$45.60
└── Extreme (4% of time):   -$171.05  (INVESTIGATE)
```

This tells you exactly where to focus: adverse selection prediction during extreme regimes.

---

## Part 2: Information Source Audit

Before building models, systematically measure what signals contain predictive information.

### 2.1 Mutual Information Estimation

For each candidate signal X and target variable Y, estimate mutual information:

```
I(X; Y) = H(Y) - H(Y|X)
```

Where H is entropy. This measures how many bits of information X provides about Y.

**Continuous Variables: k-NN Estimator (Kraskov et al.)**

```rust
use kiddo::KdTree;  // or similar k-d tree

fn estimate_mutual_information(
    x: &[f64],  // Signal values
    y: &[f64],  // Target values
    k: usize    // Number of neighbors (typically 3-10)
) -> f64 {
    let n = x.len();
    assert_eq!(n, y.len());
    
    // Build k-d trees for joint and marginals
    let mut joint_tree: KdTree<f64, usize, 2> = KdTree::new();
    let mut x_tree: KdTree<f64, usize, 1> = KdTree::new();
    let mut y_tree: KdTree<f64, usize, 1> = KdTree::new();
    
    for i in 0..n {
        joint_tree.add(&[x[i], y[i]], i);
        x_tree.add(&[x[i]], i);
        y_tree.add(&[y[i]], i);
    }
    
    let mut mi_sum = 0.0;
    
    for i in 0..n {
        // Find k-th nearest neighbor distance in joint space
        let neighbors = joint_tree.nearest_n(&[x[i], y[i]], k + 1);  // +1 to exclude self
        let eps = neighbors.last().unwrap().distance;  // Chebyshev distance
        
        // Count points within eps in marginals
        let n_x = count_within_distance(&x_tree, x[i], eps);
        let n_y = count_within_distance(&y_tree, y[i], eps);
        
        mi_sum += digamma(k as f64) + digamma(n as f64) 
                  - digamma(n_x as f64) - digamma(n_y as f64);
    }
    
    mi_sum / n as f64
}

fn digamma(x: f64) -> f64 {
    // Digamma function approximation
    if x < 6.0 {
        digamma(x + 1.0) - 1.0 / x
    } else {
        x.ln() - 1.0 / (2.0 * x) - 1.0 / (12.0 * x.powi(2))
    }
}
```

### 2.2 Signal Audit Framework

**Step 1: Define Target Variables**

```rust
enum PredictionTarget {
    // Direction
    PriceDirection1s,    // sign(price[t+1s] - price[t])
    PriceDirection10s,
    PriceDirection60s,
    
    // Magnitude
    AbsReturn1s,
    AbsReturn10s,
    Volatility1m,
    
    // Fill-related
    FillProbability1s,
    FillProbability10s,
    TimeToNextFill,
    
    // Adverse selection
    AdverseSelectionOnNextFill,
    InformedFlowProbability,
}
```

**Step 2: Define Candidate Signals**

```rust
struct SignalCatalog {
    // Book-derived (you already have these)
    microprice_imbalance: f64,      // (bid_size - ask_size) / (bid_size + ask_size)
    book_pressure: f64,             // Integrated depth asymmetry
    spread_bps: f64,
    depth_at_1bps: f64,
    depth_at_5bps: f64,
    
    // Trade-derived
    trade_imbalance_1s: f64,        // Net signed volume last 1s
    trade_imbalance_10s: f64,
    trade_arrival_rate: f64,        // Trades per second
    avg_trade_size: f64,
    large_trade_indicator: bool,    // Trade > 2σ from mean
    
    // Hyperliquid-specific
    funding_rate: f64,
    funding_rate_change_1h: f64,
    time_to_funding_settlement: f64,
    open_interest: f64,
    open_interest_change_1m: f64,
    open_interest_change_10m: f64,
    
    // Cross-exchange
    binance_hl_spread: f64,
    binance_lead_indicator: f64,    // Recent price change on Binance
    binance_volume_ratio: f64,      // Binance volume / HL volume
    
    // Derived/Composite
    funding_x_imbalance: f64,       // funding_rate * trade_imbalance
    oi_momentum: f64,               // OI change acceleration
}
```

**Step 3: Run Systematic Analysis**

```rust
struct SignalAnalysisResult {
    signal_name: String,
    target_name: String,
    
    // Information content
    mutual_information_bits: f64,
    
    // Linear relationship
    correlation: f64,
    
    // Predictive power (if target is binary)
    auc_roc: Option<f64>,
    
    // Lag analysis
    optimal_lag_ms: i64,
    mi_at_optimal_lag: f64,
    
    // Regime dependence
    mi_by_regime: HashMap<Regime, f64>,
    
    // Stationarity
    half_life_of_predictability_s: f64,  // How quickly does signal decay?
}

fn analyze_signal(
    signal: &[f64],
    target: &[f64],
    timestamps: &[u64],
    regimes: &[Regime]
) -> SignalAnalysisResult {
    // Basic MI
    let mi = estimate_mutual_information(signal, target, 5);
    
    // Correlation
    let corr = pearson_correlation(signal, target);
    
    // Lag analysis: compute MI at different lags
    let mut best_lag = 0i64;
    let mut best_mi = mi;
    for lag_ms in [-500, -200, -100, -50, 0, 50, 100, 200, 500].iter() {
        let lagged_mi = estimate_mi_with_lag(signal, target, timestamps, *lag_ms);
        if lagged_mi > best_mi {
            best_mi = lagged_mi;
            best_lag = *lag_ms;
        }
    }
    
    // MI by regime
    let mut mi_by_regime = HashMap::new();
    for regime in [Regime::Quiet, Regime::Active, Regime::Volatile].iter() {
        let mask: Vec<bool> = regimes.iter().map(|r| r == regime).collect();
        let filtered_signal: Vec<f64> = signal.iter().zip(&mask)
            .filter(|(_, &m)| m).map(|(s, _)| *s).collect();
        let filtered_target: Vec<f64> = target.iter().zip(&mask)
            .filter(|(_, &m)| m).map(|(t, _)| *t).collect();
        if filtered_signal.len() > 100 {
            mi_by_regime.insert(*regime, estimate_mutual_information(&filtered_signal, &filtered_target, 5));
        }
    }
    
    // Half-life of predictability
    let half_life = estimate_predictability_decay(signal, target, timestamps);
    
    SignalAnalysisResult {
        signal_name: "...",
        target_name: "...",
        mutual_information_bits: mi,
        correlation: corr,
        auc_roc: None,  // Compute if target is binary
        optimal_lag_ms: best_lag,
        mi_at_optimal_lag: best_mi,
        mi_by_regime,
        half_life_of_predictability_s: half_life,
    }
}
```

**Step 4: Generate Signal Audit Report**

```
=== Signal Audit Report: 2026-01-13 ===

Target: PriceDirection1s

Signal                      MI (bits)  Corr    Opt Lag   Regime Var
─────────────────────────────────────────────────────────────────────
binance_lead_indicator      0.089      0.31    -150ms    High (2.3x in volatile)
trade_imbalance_1s          0.067      0.24    0ms       Medium
microprice_imbalance        0.045      0.19    0ms       Low
funding_x_imbalance         0.041      0.15    0ms       High (3.1x near settlement)
open_interest_change_1m     0.023      0.08    0ms       Low
book_pressure               0.018      0.11    0ms       Low
funding_rate                0.012      0.05    0ms       Medium

Target: AdverseSelectionOnNextFill

Signal                      MI (bits)  Corr    Opt Lag   Regime Var
─────────────────────────────────────────────────────────────────────
trade_arrival_rate          0.134      0.42    0ms       High (4.2x in cascade)
large_trade_indicator       0.098      0.38    -50ms     Medium
binance_hl_spread           0.076      0.29    -100ms    High
open_interest_change_1m     0.054      0.21    0ms       Medium
funding_rate_change_1h      0.031      0.12    0ms       Low

ACTIONABLE INSIGHTS:
1. binance_lead_indicator is your highest-value unused signal for direction
2. trade_arrival_rate strongly predicts adverse selection - use for dynamic kappa
3. funding_x_imbalance has 3x higher MI near settlement - regime-condition this
4. open_interest_change is useful for adverse selection but not direction
```

This tells you exactly which signals to incorporate and how.

---

## Part 3: Proprietary Fill Intensity Model

Your current kappa blending is frequency-based (fills per second). The upgrade: model the full intensity process conditional on exchange-specific state.

### 3.1 Hawkes Process Foundation

Standard Hawkes process for trade arrivals:

```
λ(t) = μ + ∫₀ᵗ α·e^(-β(t-s)) dN(s)
```

Where:
- μ = baseline intensity
- α = excitation from each event
- β = decay rate
- N(s) = counting process (number of trades by time s)

**Estimation via Maximum Likelihood:**

```rust
struct HawkesParams {
    mu: f64,      // Baseline intensity
    alpha: f64,   // Excitation
    beta: f64,    // Decay
}

fn hawkes_log_likelihood(
    params: &HawkesParams,
    event_times: &[f64],  // Event times in seconds
    t_max: f64            // Observation window end
) -> f64 {
    let HawkesParams { mu, alpha, beta } = *params;
    
    let mut log_lik = 0.0;
    let mut intensity_integral = mu * t_max;  // ∫₀^T λ(t) dt contribution from baseline
    
    for (i, &t_i) in event_times.iter().enumerate() {
        // Compute λ(tᵢ) = μ + Σⱼ<ᵢ α·exp(-β(tᵢ - tⱼ))
        let mut lambda_i = mu;
        for &t_j in &event_times[..i] {
            lambda_i += alpha * (-beta * (t_i - t_j)).exp();
        }
        
        log_lik += lambda_i.ln();  // + ln(λ(tᵢ)) for each event
        
        // Add contribution to integral: ∫ₜᵢ^T α·exp(-β(t-tᵢ)) dt = (α/β)(1 - exp(-β(T-tᵢ)))
        intensity_integral += (alpha / beta) * (1.0 - (-beta * (t_max - t_i)).exp());
    }
    
    log_lik -= intensity_integral;  // - ∫₀^T λ(t) dt
    
    log_lik
}

fn fit_hawkes(event_times: &[f64], t_max: f64) -> HawkesParams {
    // Use L-BFGS or similar optimizer
    let initial = HawkesParams { mu: 1.0, alpha: 0.5, beta: 1.0 };
    
    // Constraint: α/β < 1 for stationarity
    optimize_with_constraints(
        |p| -hawkes_log_likelihood(p, event_times, t_max),
        initial,
        |p| p.alpha / p.beta < 0.99 && p.mu > 0.0 && p.alpha > 0.0 && p.beta > 0.0
    )
}
```

### 3.2 Exchange-Specific Extensions

The standard Hawkes is too simple. Extend it with conditioning variables:

**Extension 1: State-Dependent Baseline**

```
μ(t) = μ₀ · exp(w_F · F(t) + w_OI · ΔOI(t) + w_τ · τ(t))
```

Where:
- F(t) = funding rate
- ΔOI(t) = OI change rate
- τ(t) = time to funding settlement (cyclical feature)

**Extension 2: Trade-Type-Dependent Excitation**

Different trade types have different excitation effects:

```
λ(t) = μ(t) + Σᵢ αᵢ(sᵢ, vᵢ, dᵢ) · K(t - tᵢ)
```

Where αᵢ depends on:
- sᵢ = side (buy/sell)
- vᵢ = volume
- dᵢ = whether trade was on "our side" of the book

```rust
fn compute_excitation(trade: &Trade, our_side: Side) -> f64 {
    let base_alpha = 0.3;
    
    // Size effect: larger trades excite more
    let size_mult = (trade.size / MEDIAN_TRADE_SIZE).sqrt().min(3.0);
    
    // Side effect: trades on our side are more relevant for our fills
    let side_mult = if trade.side == our_side { 1.5 } else { 0.8 };
    
    // Aggressor effect: market orders excite more than limit fills
    let aggressor_mult = if trade.is_aggressor { 1.2 } else { 1.0 };
    
    base_alpha * size_mult * side_mult * aggressor_mult
}
```

**Extension 3: Queue-Position-Dependent Kernel**

The kernel shouldn't just depend on time—it should depend on how the queue has moved:

```rust
fn adaptive_kernel(
    time_since_trade: f64,
    queue_change_since_trade: f64,  // How much queue ahead of us was consumed
    beta: f64
) -> f64 {
    // Standard temporal decay
    let temporal = (-beta * time_since_trade).exp();
    
    // Queue consumption effect: if queue ahead was eaten, we're more likely to fill
    let queue_mult = 1.0 + 0.5 * (queue_change_since_trade / TYPICAL_QUEUE_SIZE).min(1.0);
    
    temporal * queue_mult
}
```

### 3.3 Full Model Specification

```rust
struct HyperliquidFillIntensityModel {
    // Baseline parameters
    mu_0: f64,
    w_funding: f64,
    w_oi_change: f64,
    w_time_to_settlement: f64,  // Cyclical effect
    
    // Excitation parameters
    alpha_base: f64,
    alpha_size_power: f64,      // α ∝ size^power
    alpha_same_side_mult: f64,
    alpha_aggressor_mult: f64,
    
    // Decay parameters
    beta_time: f64,
    beta_queue_sensitivity: f64,
    
    // Regime-switching (optional)
    regime_multipliers: HashMap<Regime, f64>,
}

impl HyperliquidFillIntensityModel {
    fn intensity_at(
        &self,
        t: f64,
        recent_trades: &[Trade],
        queue_position: f64,
        queue_history: &[(f64, f64)],  // (time, queue_size) history
        market_state: &MarketState
    ) -> f64 {
        // State-dependent baseline
        let funding_effect = self.w_funding * market_state.funding_rate;
        let oi_effect = self.w_oi_change * market_state.oi_change_rate;
        let settlement_effect = self.w_time_to_settlement 
            * (market_state.time_to_settlement / 8.0 * std::f64::consts::TAU).sin();
        
        let mu_t = self.mu_0 * (funding_effect + oi_effect + settlement_effect).exp();
        
        // Excitation from recent trades
        let mut excitation = 0.0;
        for trade in recent_trades {
            let time_since = t - trade.timestamp;
            if time_since <= 0.0 || time_since > 60.0 { continue; }
            
            // Compute α for this trade
            let size_mult = (trade.size / MEDIAN_SIZE).powf(self.alpha_size_power);
            let side_mult = if trade.side == Side::Bid { self.alpha_same_side_mult } else { 1.0 };
            let aggr_mult = if trade.is_aggressor { self.alpha_aggressor_mult } else { 1.0 };
            let alpha_i = self.alpha_base * size_mult * side_mult * aggr_mult;
            
            // Compute kernel with queue adjustment
            let queue_at_trade = interpolate_queue(queue_history, trade.timestamp);
            let queue_consumed = (queue_at_trade - queue_position).max(0.0);
            let kernel = adaptive_kernel(time_since, queue_consumed, self.beta_time, self.beta_queue_sensitivity);
            
            excitation += alpha_i * kernel;
        }
        
        // Regime adjustment
        let regime_mult = self.regime_multipliers.get(&market_state.regime).copied().unwrap_or(1.0);
        
        (mu_t + excitation) * regime_mult
    }
    
    fn expected_fills_in_window(
        &self,
        t_start: f64,
        t_end: f64,
        depth_bps: f64,
        market_state: &MarketState,
        recent_trades: &[Trade],
        queue_position: f64
    ) -> f64 {
        // Numerical integration or closed-form approximation
        let n_steps = 20;
        let dt = (t_end - t_start) / n_steps as f64;
        
        let mut integral = 0.0;
        for i in 0..n_steps {
            let t = t_start + (i as f64 + 0.5) * dt;
            let lambda_t = self.intensity_at(t, recent_trades, queue_position, &[], market_state);
            
            // Depth adjustment: further from mid = lower fill intensity
            let depth_decay = (-0.5 * depth_bps / 5.0).exp();  // Half-life of 5 bps
            
            integral += lambda_t * depth_decay * dt;
        }
        
        integral
    }
}
```

### 3.4 Online Parameter Estimation

Parameters need to adapt in real-time:

```rust
struct OnlineHawkesEstimator {
    // Current parameter estimates
    params: HyperliquidFillIntensityModel,
    
    // Sufficient statistics for online updates
    event_count: usize,
    baseline_exposure: f64,  // Total time observed
    excitation_sum: f64,     // Σ contributions from self-excitation
    
    // Learning rate schedule
    learning_rate: f64,
    min_learning_rate: f64,
    decay_rate: f64,
}

impl OnlineHawkesEstimator {
    fn update_on_fill(&mut self, fill: &Fill, market_state: &MarketState) {
        self.event_count += 1;
        
        // Compute gradient of log-likelihood w.r.t. parameters
        // This is an approximation using the current fill
        
        let predicted_intensity = self.params.intensity_at(
            fill.timestamp,
            &fill.recent_trades,
            fill.queue_position,
            &fill.queue_history,
            market_state
        );
        
        // Innovation: actual fill vs predicted intensity
        let innovation = 1.0 - predicted_intensity * fill.time_since_last_fill;
        
        // Gradient updates (simplified)
        let lr = self.learning_rate;
        
        // Update baseline parameters
        self.params.mu_0 += lr * innovation * self.params.mu_0;
        self.params.w_funding += lr * innovation * market_state.funding_rate;
        
        // Decay learning rate
        self.learning_rate = (self.learning_rate * self.decay_rate).max(self.min_learning_rate);
    }
    
    fn update_on_no_fill(&mut self, time_window: f64, market_state: &MarketState) {
        // Update based on survival (no fill in window)
        self.baseline_exposure += time_window;
        
        let predicted_fills = self.params.expected_fills_in_window(
            0.0, time_window, 0.0, market_state, &[], 0.0
        );
        
        // If we predicted fills but got none, reduce intensity estimates
        if predicted_fills > 0.5 {
            let adjustment = -self.learning_rate * predicted_fills.min(1.0);
            self.params.mu_0 *= (1.0 + adjustment).max(0.5);
        }
    }
}
```

### 3.5 Converting Intensity to Kappa

Your GLFT formula uses kappa (fill rate per unit spread). Convert from Hawkes intensity:

```rust
fn intensity_to_kappa(
    fill_intensity_model: &HyperliquidFillIntensityModel,
    market_state: &MarketState,
    reference_depth_bps: f64  // Typically 5-10 bps
) -> f64 {
    // Kappa represents: additional fills per unit of spread tightening
    // κ = ∂(fill_rate)/∂(depth)
    
    let eps = 0.1;  // 0.1 bps perturbation
    
    let fill_rate_at_depth = fill_intensity_model.expected_fills_in_window(
        0.0, 1.0, reference_depth_bps, market_state, &[], 0.0
    );
    
    let fill_rate_tighter = fill_intensity_model.expected_fills_in_window(
        0.0, 1.0, reference_depth_bps - eps, market_state, &[], 0.0
    );
    
    // κ ≈ -Δ(fill_rate) / Δ(depth)
    let kappa = (fill_rate_tighter - fill_rate_at_depth) / (eps / 10000.0);  // Convert bps to fraction
    
    kappa.max(100.0)  // Floor to prevent division issues in GLFT
}
```

---

## Part 4: Adverse Selection Decomposition

The goal: predict which trades are informed before they hit you, and adjust quotes accordingly.

### 4.1 Mixture Model for Trade Classification

Every trade comes from one of several latent types:

```
Types: {noise, informed, liquidation, arbitrage}

P(type | observable_features) = softmax(W · φ(features))
```

**Feature Engineering:**

```rust
struct TradeFeatures {
    // Size features
    size_zscore: f64,           // (size - μ) / σ over rolling window
    size_quantile: f64,         // Percentile rank
    
    // Timing features
    time_since_last_trade_ms: f64,
    trades_in_last_1s: u32,
    trades_in_last_10s: u32,
    
    // Aggression features
    is_aggressor: bool,
    crossed_spread_bps: f64,    // How far into the book did it go
    
    // Directional features
    signed_volume_imbalance_1s: f64,
    signed_volume_imbalance_10s: f64,
    
    // Funding interaction
    funding_rate: f64,
    trade_aligns_with_funding: bool,  // Buying when funding positive = going against squeeze
    
    // Cross-exchange
    binance_price_change_100ms: f64,
    binance_hl_spread_at_trade: f64,
    
    // Book state
    book_imbalance_at_trade: f64,
    depth_consumed_pct: f64,    // What % of top level did this trade consume
    
    // Hyperliquid-specific
    oi_change_1m_before: f64,
    near_liquidation_price: bool,  // Is there significant OI near this price?
}
```

**Labeling Strategy (for training):**

You need labeled examples. Since you can't observe true intent, use ex-post outcomes as proxies:

```rust
enum TradeLabel {
    Informed,      // Price moved >X bps in trade direction within 10s
    Noise,         // Price stayed flat or reversed
    Liquidation,   // Part of a liquidation cascade (detect via OI drop)
    Arbitrage,     // Cross-exchange spread closed immediately after
}

fn label_trade(trade: &Trade, future_prices: &[f64], market_context: &MarketContext) -> TradeLabel {
    let price_10s_later = future_prices[100];  // Assuming 100ms resolution
    let price_move_bps = (price_10s_later - trade.price) / trade.price * 10000.0;
    let signed_move = if trade.side == Side::Bid { price_move_bps } else { -price_move_bps };
    
    // Check for liquidation cascade
    if market_context.oi_dropped_significantly && market_context.funding_extreme {
        return TradeLabel::Liquidation;
    }
    
    // Check for arbitrage
    if market_context.cross_exchange_spread_closed_within_500ms {
        return TradeLabel::Arbitrage;
    }
    
    // Informed vs noise based on ex-post price move
    if signed_move > 5.0 {  // Moved >5 bps in trade direction
        TradeLabel::Informed
    } else {
        TradeLabel::Noise
    }
}
```

### 4.2 Classifier Architecture

Use a small neural network or gradient boosted trees:

```rust
struct TradeClassifier {
    // Option 1: Logistic regression (interpretable)
    weights: HashMap<String, f64>,
    
    // Option 2: Small MLP (more expressive)
    layer1: Matrix,  // input_dim x 32
    layer2: Matrix,  // 32 x 16
    layer3: Matrix,  // 16 x 4 (output: probabilities for each class)
}

impl TradeClassifier {
    fn predict(&self, features: &TradeFeatures) -> ClassProbabilities {
        let x = features.to_vector();
        
        // Forward pass
        let h1 = relu(self.layer1.transpose() * x);
        let h2 = relu(self.layer2.transpose() * h1);
        let logits = self.layer3.transpose() * h2;
        
        softmax(logits)
    }
    
    fn informed_probability(&self, features: &TradeFeatures) -> f64 {
        self.predict(features).informed
    }
}
```

**Training Loop:**

```rust
fn train_classifier(
    labeled_trades: &[(TradeFeatures, TradeLabel)],
    validation_set: &[(TradeFeatures, TradeLabel)]
) -> TradeClassifier {
    let mut classifier = TradeClassifier::random_init();
    let mut optimizer = Adam::new(0.001);
    
    for epoch in 0..100 {
        // Shuffle and batch
        let batches = create_batches(labeled_trades, batch_size=64);
        
        for batch in batches {
            let mut grad_sum = Gradients::zeros();
            
            for (features, label) in batch {
                let probs = classifier.predict(features);
                let loss = cross_entropy_loss(&probs, label);
                let grad = backpropagate(&classifier, features, label);
                grad_sum += grad;
            }
            
            optimizer.step(&mut classifier, grad_sum / batch.len());
        }
        
        // Validation
        let val_accuracy = evaluate(&classifier, validation_set);
        println!("Epoch {}: validation accuracy = {:.3}", epoch, val_accuracy);
    }
    
    classifier
}
```

### 4.3 Real-Time Integration

Use the classifier to adjust kappa and spread in real-time:

```rust
struct AdverseSelectionAdjuster {
    classifier: TradeClassifier,
    
    // Running estimate of informed flow intensity
    informed_intensity: ExponentialMovingAverage,
    
    // Configuration
    kappa_discount_per_informed_pct: f64,  // How much to reduce kappa per 1% informed
    spread_premium_per_informed_pct: f64,  // How much to widen spread per 1% informed
}

impl AdverseSelectionAdjuster {
    fn on_trade(&mut self, trade: &Trade, features: &TradeFeatures) {
        let informed_prob = self.classifier.informed_probability(features);
        
        // Update running estimate
        self.informed_intensity.update(informed_prob);
    }
    
    fn get_kappa_adjustment(&self) -> f64 {
        // If recent trades are 30% informed, reduce kappa by 30% * discount_factor
        let informed_pct = self.informed_intensity.value() * 100.0;
        let adjustment = 1.0 - informed_pct * self.kappa_discount_per_informed_pct;
        adjustment.max(0.3)  // Don't reduce kappa by more than 70%
    }
    
    fn get_spread_adjustment_bps(&self) -> f64 {
        let informed_pct = self.informed_intensity.value() * 100.0;
        informed_pct * self.spread_premium_per_informed_pct
    }
}

// Integration into your quote engine
fn compute_adjusted_kappa(
    base_kappa: f64,
    adverse_selection_adjuster: &AdverseSelectionAdjuster
) -> f64 {
    base_kappa * adverse_selection_adjuster.get_kappa_adjustment()
}
```

### 4.4 Liquidation Detection Subsystem

Liquidations on Hyperliquid are particularly predictable and toxic. Build a dedicated detector:

```rust
struct LiquidationDetector {
    // OI tracking
    oi_history: RingBuffer<(u64, f64)>,  // (timestamp, open_interest)
    
    // Funding tracking
    funding_history: RingBuffer<(u64, f64)>,
    
    // Liquidation cascade detection
    cascade_threshold_oi_drop_pct: f64,   // OI drop that indicates cascade
    cascade_threshold_time_s: f64,         // Time window to measure drop
    
    // Current state
    liquidation_probability: f64,
}

impl LiquidationDetector {
    fn update(&mut self, current_oi: f64, current_funding: f64, timestamp: u64) {
        self.oi_history.push((timestamp, current_oi));
        self.funding_history.push((timestamp, current_funding));
        
        // Compute OI change rate
        let oi_1m_ago = self.oi_history.get_at_time(timestamp - 60_000);
        let oi_change_pct = (current_oi - oi_1m_ago) / oi_1m_ago * 100.0;
        
        // Funding extremity
        let funding_percentile = self.funding_history.percentile_rank(current_funding);
        
        // Liquidation probability model
        self.liquidation_probability = self.compute_liquidation_probability(
            oi_change_pct,
            funding_percentile,
            current_funding
        );
    }
    
    fn compute_liquidation_probability(
        &self,
        oi_change_pct: f64,
        funding_percentile: f64,
        funding_rate: f64
    ) -> f64 {
        // Heuristic model (replace with learned model)
        let mut prob = 0.0;
        
        // Rapid OI decrease is a strong signal
        if oi_change_pct < -2.0 {
            prob += 0.3;
        }
        if oi_change_pct < -5.0 {
            prob += 0.4;
        }
        
        // Extreme funding + OI drop = very likely cascade
        if funding_percentile > 0.95 || funding_percentile < 0.05 {
            prob += 0.2;
        }
        
        // Funding direction matters
        if (funding_rate > 0.0005 && oi_change_pct < -1.0) ||  // Longs getting squeezed
           (funding_rate < -0.0005 && oi_change_pct < -1.0) {  // Shorts getting squeezed
            prob += 0.3;
        }
        
        prob.min(0.95)
    }
    
    fn is_cascade_active(&self) -> bool {
        self.liquidation_probability > 0.5
    }
}
```

When liquidation probability is high, dramatically widen spreads or pull quotes entirely.

---

## Part 5: Regime Detection System

Your current warmup multiplier is a crude regime proxy. Build a proper HMM.

### 5.1 Hidden Markov Model Specification

**States:**

```rust
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum MarketRegime {
    Quiet,          // Low volatility, balanced flow, normal fill rates
    Trending,       // Directional momentum, elevated adverse selection
    Volatile,       // High volatility, wide spreads, uncertain direction
    Cascade,        // Liquidation cascade, extreme toxicity
}
```

**Emission Model:**

For each regime, define the distribution of observable features:

```rust
struct RegimeEmissionModel {
    // For each regime, parameters of the emission distribution
    volatility_mean: f64,
    volatility_std: f64,
    
    trade_intensity_mean: f64,
    trade_intensity_std: f64,
    
    imbalance_mean: f64,
    imbalance_std: f64,
    
    adverse_selection_mean: f64,
    adverse_selection_std: f64,
}

struct HMMParams {
    // Transition matrix (4x4)
    transition_probs: [[f64; 4]; 4],
    
    // Emission parameters for each regime
    emissions: [RegimeEmissionModel; 4],
    
    // Initial state distribution
    initial_probs: [f64; 4],
}
```

**Initialization:**

```rust
fn initialize_hmm_params() -> HMMParams {
    HMMParams {
        transition_probs: [
            // From Quiet: mostly stays quiet, sometimes transitions
            [0.95, 0.03, 0.019, 0.001],
            // From Trending: can stay or revert
            [0.10, 0.85, 0.04, 0.01],
            // From Volatile: can calm down or escalate
            [0.15, 0.10, 0.70, 0.05],
            // From Cascade: usually short, reverts to volatile
            [0.05, 0.05, 0.60, 0.30],
        ],
        emissions: [
            // Quiet
            RegimeEmissionModel {
                volatility_mean: 0.00005,
                volatility_std: 0.00002,
                trade_intensity_mean: 0.5,
                trade_intensity_std: 0.2,
                imbalance_mean: 0.0,
                imbalance_std: 0.15,
                adverse_selection_mean: 0.2,
                adverse_selection_std: 0.1,
            },
            // Trending
            RegimeEmissionModel {
                volatility_mean: 0.00015,
                volatility_std: 0.00005,
                trade_intensity_mean: 1.5,
                trade_intensity_std: 0.5,
                imbalance_mean: 0.3,  // Biased
                imbalance_std: 0.2,
                adverse_selection_mean: 0.5,
                adverse_selection_std: 0.15,
            },
            // Volatile
            RegimeEmissionModel {
                volatility_mean: 0.00030,
                volatility_std: 0.00010,
                trade_intensity_mean: 3.0,
                trade_intensity_std: 1.0,
                imbalance_mean: 0.0,
                imbalance_std: 0.4,
                adverse_selection_mean: 0.4,
                adverse_selection_std: 0.2,
            },
            // Cascade
            RegimeEmissionModel {
                volatility_mean: 0.00100,
                volatility_std: 0.00030,
                trade_intensity_mean: 10.0,
                trade_intensity_std: 3.0,
                imbalance_mean: 0.0,
                imbalance_std: 0.6,
                adverse_selection_mean: 0.8,
                adverse_selection_std: 0.1,
            },
        ],
        initial_probs: [0.7, 0.15, 0.14, 0.01],
    }
}
```

### 5.2 Online Filtering (Forward Algorithm)

Maintain a belief state over regimes that updates with each observation:

```rust
struct OnlineHMMFilter {
    params: HMMParams,
    
    // Current belief state: P(regime | observations so far)
    belief: [f64; 4],
    
    // Observation buffer for smoothing
    observation_buffer: RingBuffer<ObservationVector>,
}

#[derive(Clone)]
struct ObservationVector {
    volatility: f64,
    trade_intensity: f64,
    imbalance: f64,
    adverse_selection: f64,
}

impl OnlineHMMFilter {
    fn new(params: HMMParams) -> Self {
        OnlineHMMFilter {
            params,
            belief: params.initial_probs,
            observation_buffer: RingBuffer::new(100),
        }
    }
    
    fn update(&mut self, obs: &ObservationVector) {
        self.observation_buffer.push(obs.clone());
        
        // Prediction step: apply transition matrix
        let mut predicted = [0.0; 4];
        for j in 0..4 {
            for i in 0..4 {
                predicted[j] += self.params.transition_probs[i][j] * self.belief[i];
            }
        }
        
        // Update step: multiply by observation likelihood
        let mut updated = [0.0; 4];
        let mut normalizer = 0.0;
        
        for i in 0..4 {
            let likelihood = self.observation_likelihood(obs, i);
            updated[i] = predicted[i] * likelihood;
            normalizer += updated[i];
        }
        
        // Normalize
        for i in 0..4 {
            self.belief[i] = updated[i] / normalizer;
        }
    }
    
    fn observation_likelihood(&self, obs: &ObservationVector, regime: usize) -> f64 {
        let emission = &self.params.emissions[regime];
        
        // Assume independent Gaussian emissions
        let vol_ll = gaussian_pdf(obs.volatility, emission.volatility_mean, emission.volatility_std);
        let int_ll = gaussian_pdf(obs.trade_intensity, emission.trade_intensity_mean, emission.trade_intensity_std);
        let imb_ll = gaussian_pdf(obs.imbalance, emission.imbalance_mean, emission.imbalance_std);
        let as_ll = gaussian_pdf(obs.adverse_selection, emission.adverse_selection_mean, emission.adverse_selection_std);
        
        vol_ll * int_ll * imb_ll * as_ll
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
}

fn gaussian_pdf(x: f64, mean: f64, std: f64) -> f64 {
    let z = (x - mean) / std;
    (-0.5 * z * z).exp() / (std * (2.0 * std::f64::consts::PI).sqrt())
}
```

### 5.3 Regime-Dependent Parameters

For each regime, maintain separate GLFT parameters:

```rust
struct RegimeSpecificParams {
    gamma: f64,
    kappa_multiplier: f64,
    spread_floor_bps: f64,
    max_inventory: f64,
}

fn get_regime_params(regime: MarketRegime) -> RegimeSpecificParams {
    match regime {
        MarketRegime::Quiet => RegimeSpecificParams {
            gamma: 0.3,
            kappa_multiplier: 1.0,
            spread_floor_bps: 5.0,
            max_inventory: 1.0,
        },
        MarketRegime::Trending => RegimeSpecificParams {
            gamma: 0.5,              // More risk-averse
            kappa_multiplier: 0.7,   // Expect lower fill rates
            spread_floor_bps: 10.0,
            max_inventory: 0.5,      // Tighter inventory limits
        },
        MarketRegime::Volatile => RegimeSpecificParams {
            gamma: 0.8,
            kappa_multiplier: 1.5,   // Higher fill rates but toxic
            spread_floor_bps: 15.0,
            max_inventory: 0.3,
        },
        MarketRegime::Cascade => RegimeSpecificParams {
            gamma: 2.0,              // Extreme risk aversion
            kappa_multiplier: 5.0,   // Fills are certain but very toxic
            spread_floor_bps: 50.0,  // Wide spreads
            max_inventory: 0.1,      // Almost no inventory tolerance
        },
    }
}

fn blend_params_by_belief(
    hmm_filter: &OnlineHMMFilter
) -> RegimeSpecificParams {
    let probs = hmm_filter.regime_probabilities();
    
    let mut blended = RegimeSpecificParams {
        gamma: 0.0,
        kappa_multiplier: 0.0,
        spread_floor_bps: 0.0,
        max_inventory: 0.0,
    };
    
    for (regime, prob) in probs {
        let params = get_regime_params(regime);
        blended.gamma += prob * params.gamma;
        blended.kappa_multiplier += prob * params.kappa_multiplier;
        blended.spread_floor_bps += prob * params.spread_floor_bps;
        blended.max_inventory += prob * params.max_inventory;
    }
    
    blended
}
```

### 5.4 HMM Parameter Learning (Baum-Welch)

Periodically re-estimate HMM parameters from historical data:

```rust
fn baum_welch_iteration(
    params: &mut HMMParams,
    observations: &[ObservationVector]
) {
    let n = observations.len();
    let k = 4;  // Number of states
    
    // Forward pass
    let mut alpha = vec![[0.0; 4]; n];
    
    // Initialize
    for i in 0..k {
        alpha[0][i] = params.initial_probs[i] * observation_likelihood(&observations[0], i, params);
    }
    normalize(&mut alpha[0]);
    
    // Forward recursion
    for t in 1..n {
        for j in 0..k {
            let mut sum = 0.0;
            for i in 0..k {
                sum += alpha[t-1][i] * params.transition_probs[i][j];
            }
            alpha[t][j] = sum * observation_likelihood(&observations[t], j, params);
        }
        normalize(&mut alpha[t]);
    }
    
    // Backward pass
    let mut beta = vec![[1.0; 4]; n];
    
    for t in (0..n-1).rev() {
        for i in 0..k {
            let mut sum = 0.0;
            for j in 0..k {
                sum += params.transition_probs[i][j] 
                     * observation_likelihood(&observations[t+1], j, params) 
                     * beta[t+1][j];
            }
            beta[t][i] = sum;
        }
        normalize(&mut beta[t]);
    }
    
    // Compute gamma (state posteriors) and xi (transition posteriors)
    let mut gamma = vec![[0.0; 4]; n];
    let mut xi = vec![[[0.0; 4]; 4]; n-1];
    
    for t in 0..n {
        let mut sum = 0.0;
        for i in 0..k {
            gamma[t][i] = alpha[t][i] * beta[t][i];
            sum += gamma[t][i];
        }
        for i in 0..k {
            gamma[t][i] /= sum;
        }
    }
    
    for t in 0..n-1 {
        let mut sum = 0.0;
        for i in 0..k {
            for j in 0..k {
                xi[t][i][j] = alpha[t][i] 
                            * params.transition_probs[i][j]
                            * observation_likelihood(&observations[t+1], j, params)
                            * beta[t+1][j];
                sum += xi[t][i][j];
            }
        }
        for i in 0..k {
            for j in 0..k {
                xi[t][i][j] /= sum;
            }
        }
    }
    
    // M-step: update parameters
    
    // Update transition probabilities
    for i in 0..k {
        let mut denom = 0.0;
        for t in 0..n-1 {
            denom += gamma[t][i];
        }
        
        for j in 0..k {
            let mut numer = 0.0;
            for t in 0..n-1 {
                numer += xi[t][i][j];
            }
            params.transition_probs[i][j] = numer / denom.max(1e-10);
        }
    }
    
    // Update emission parameters (for Gaussian emissions)
    for i in 0..k {
        let mut weight_sum = 0.0;
        let mut vol_sum = 0.0;
        let mut vol_sq_sum = 0.0;
        // ... similar for other features
        
        for t in 0..n {
            weight_sum += gamma[t][i];
            vol_sum += gamma[t][i] * observations[t].volatility;
            vol_sq_sum += gamma[t][i] * observations[t].volatility.powi(2);
        }
        
        params.emissions[i].volatility_mean = vol_sum / weight_sum;
        params.emissions[i].volatility_std = 
            ((vol_sq_sum / weight_sum) - params.emissions[i].volatility_mean.powi(2)).sqrt();
        
        // ... update other emission parameters similarly
    }
}
```

---

## Part 6: Cross-Exchange Lead-Lag Model

This is potentially your highest-value proprietary signal.

### 6.1 Lead-Lag Estimation

**Observation:** Binance BTC perp leads Hyperliquid BTC perp by 50-500ms depending on conditions.

```rust
struct LeadLagEstimator {
    // Rolling buffers of price changes
    binance_changes: RingBuffer<(u64, f64)>,  // (timestamp_ns, price_change)
    hl_changes: RingBuffer<(u64, f64)>,
    
    // Estimated parameters
    lag_estimate_ms: f64,
    lag_estimate_std: f64,
    beta_estimate: f64,  // HL_change ≈ β × Binance_change(t - lag)
    beta_estimate_std: f64,
    
    // Model confidence
    r_squared: f64,
    sample_count: usize,
}

impl LeadLagEstimator {
    fn update(&mut self, binance_mid: f64, hl_mid: f64, timestamp_ns: u64) {
        // Compute price changes
        let binance_prev = self.binance_changes.latest().map(|(_, p)| p).unwrap_or(binance_mid);
        let hl_prev = self.hl_changes.latest().map(|(_, p)| p).unwrap_or(hl_mid);
        
        let binance_change = (binance_mid - binance_prev) / binance_prev;
        let hl_change = (hl_mid - hl_prev) / hl_prev;
        
        self.binance_changes.push((timestamp_ns, binance_change));
        self.hl_changes.push((timestamp_ns, hl_change));
        
        // Periodically re-estimate lag
        if self.sample_count % 100 == 0 {
            self.estimate_lag();
        }
        
        self.sample_count += 1;
    }
    
    fn estimate_lag(&mut self) {
        // Grid search over candidate lags
        let candidate_lags_ms: Vec<f64> = (-100..=500).step_by(10).map(|x| x as f64).collect();
        
        let mut best_lag = 0.0;
        let mut best_r2 = -1.0;
        let mut best_beta = 0.0;
        
        for &lag_ms in &candidate_lags_ms {
            let (beta, r2) = self.compute_regression_at_lag(lag_ms);
            if r2 > best_r2 {
                best_r2 = r2;
                best_lag = lag_ms;
                best_beta = beta;
            }
        }
        
        // Update estimates with smoothing
        let alpha = 0.1;
        self.lag_estimate_ms = alpha * best_lag + (1.0 - alpha) * self.lag_estimate_ms;
        self.beta_estimate = alpha * best_beta + (1.0 - alpha) * self.beta_estimate;
        self.r_squared = alpha * best_r2 + (1.0 - alpha) * self.r_squared;
    }
    
    fn compute_regression_at_lag(&self, lag_ms: f64) -> (f64, f64) {
        // Align Binance changes with HL changes at specified lag
        let mut x_vec = Vec::new();
        let mut y_vec = Vec::new();
        
        for (hl_ts, hl_change) in self.hl_changes.iter() {
            // Find Binance change at (hl_ts - lag_ms)
            let target_ts = *hl_ts - (lag_ms * 1_000_000.0) as u64;
            if let Some(binance_change) = self.binance_changes.interpolate_at(target_ts) {
                x_vec.push(binance_change);
                y_vec.push(*hl_change);
            }
        }
        
        if x_vec.len() < 50 {
            return (0.0, 0.0);
        }
        
        // Simple linear regression
        let n = x_vec.len() as f64;
        let x_mean: f64 = x_vec.iter().sum::<f64>() / n;
        let y_mean: f64 = y_vec.iter().sum::<f64>() / n;
        
        let mut num = 0.0;
        let mut denom = 0.0;
        for i in 0..x_vec.len() {
            num += (x_vec[i] - x_mean) * (y_vec[i] - y_mean);
            denom += (x_vec[i] - x_mean).powi(2);
        }
        
        let beta = num / denom.max(1e-10);
        
        // Compute R²
        let mut ss_res = 0.0;
        let mut ss_tot = 0.0;
        for i in 0..x_vec.len() {
            let y_pred = beta * x_vec[i];
            ss_res += (y_vec[i] - y_pred).powi(2);
            ss_tot += (y_vec[i] - y_mean).powi(2);
        }
        
        let r2 = 1.0 - ss_res / ss_tot.max(1e-10);
        
        (beta, r2)
    }
}
```

### 6.2 Adaptive Lead-Lag with Regime Conditioning

The lead-lag relationship changes with volatility:

```rust
struct RegimeConditionedLeadLag {
    // Separate estimators for different volatility regimes
    estimators: HashMap<VolatilityRegime, LeadLagEstimator>,
    
    // Current volatility regime
    current_regime: VolatilityRegime,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum VolatilityRegime {
    Low,    // Bottom quartile
    Medium, // Middle two quartiles
    High,   // Top quartile
}

impl RegimeConditionedLeadLag {
    fn get_current_estimate(&self) -> (f64, f64) {
        let estimator = &self.estimators[&self.current_regime];
        (estimator.lag_estimate_ms, estimator.beta_estimate)
    }
    
    fn predict_hl_move(&self, recent_binance_change: f64, time_since_change_ms: f64) -> f64 {
        let (lag_ms, beta) = self.get_current_estimate();
        
        // If Binance moved recently and we're within the lag window, predict HL move
        if time_since_change_ms < lag_ms {
            // Expected HL move hasn't fully materialized yet
            let completion_fraction = time_since_change_ms / lag_ms;
            let remaining_move = beta * recent_binance_change * (1.0 - completion_fraction);
            remaining_move
        } else {
            0.0
        }
    }
}
```

### 6.3 Integration into Quote Generation

Use lead-lag to adjust microprice and quote skew:

```rust
fn compute_adjusted_microprice(
    local_microprice: f64,
    lead_lag_model: &RegimeConditionedLeadLag,
    recent_binance_move: f64,
    time_since_binance_move_ms: f64
) -> f64 {
    let predicted_hl_move = lead_lag_model.predict_hl_move(
        recent_binance_move,
        time_since_binance_move_ms
    );
    
    // Adjust microprice by expected move
    local_microprice * (1.0 + predicted_hl_move)
}

fn compute_directional_skew(
    lead_lag_model: &RegimeConditionedLeadLag,
    recent_binance_move: f64,
    time_since_binance_move_ms: f64,
    base_skew: f64
) -> f64 {
    let (lag_ms, beta) = lead_lag_model.get_current_estimate();
    
    // If Binance moved up and lag hasn't elapsed, skew asks tighter
    if time_since_binance_move_ms < lag_ms {
        let expected_direction = recent_binance_move.signum();
        let confidence = lead_lag_model.estimators[&lead_lag_model.current_regime].r_squared;
        
        // Add skew in direction of expected move
        base_skew + expected_direction * confidence * 2.0  // 2 bps adjustment at full confidence
    } else {
        base_skew
    }
}
```

---

## Part 7: Integration Architecture

Putting it all together:

```rust
struct EnhancedQuoteEngine {
    // Existing components
    kappa_orchestrator: KappaOrchestrator,
    volatility_estimator: BipowerVariation,
    microprice_estimator: MicropriceEstimator,
    
    // New components
    prediction_logger: PredictionLogger,
    fill_intensity_model: HyperliquidFillIntensityModel,
    trade_classifier: TradeClassifier,
    adverse_selection_adjuster: AdverseSelectionAdjuster,
    hmm_filter: OnlineHMMFilter,
    liquidation_detector: LiquidationDetector,
    lead_lag_model: RegimeConditionedLeadLag,
    
    // Calibration tracking
    calibration_metrics: CalibrationMetrics,
}

impl EnhancedQuoteEngine {
    fn generate_quotes(&mut self, market_data: &MarketData) -> QuoteSet {
        // 1. Update all models with new data
        self.update_models(market_data);
        
        // 2. Log predictions for calibration
        let predictions = self.generate_predictions(market_data);
        self.prediction_logger.log(&predictions, market_data);
        
        // 3. Get regime-blended parameters
        let regime_params = blend_params_by_belief(&self.hmm_filter);
        
        // 4. Check for liquidation cascade
        if self.liquidation_detector.is_cascade_active() {
            return self.generate_defensive_quotes(market_data);
        }
        
        // 5. Compute adjusted microprice using lead-lag
        let adjusted_microprice = compute_adjusted_microprice(
            self.microprice_estimator.get_microprice(),
            &self.lead_lag_model,
            market_data.recent_binance_move,
            market_data.time_since_binance_move_ms
        );
        
        // 6. Compute fill-intensity-based kappa
        let intensity_kappa = intensity_to_kappa(
            &self.fill_intensity_model,
            &market_data.market_state,
            10.0  // Reference depth
        );
        
        // 7. Apply adverse selection adjustment
        let adjusted_kappa = intensity_kappa 
            * self.adverse_selection_adjuster.get_kappa_adjustment()
            * regime_params.kappa_multiplier;
        
        // 8. Compute GLFT optimal spread
        let gamma = regime_params.gamma;
        let glft_half_spread = (1.0 / gamma) * (1.0 + gamma / adjusted_kappa).ln() + MAKER_FEE;
        
        // 9. Apply floor and uncertainty premium
        let spread_floor = regime_params.spread_floor_bps / 10000.0;
        let uncertainty_mult = self.compute_uncertainty_multiplier();
        let final_half_spread = glft_half_spread.max(spread_floor) * uncertainty_mult;
        
        // 10. Compute inventory skew
        let inventory_skew = self.compute_inventory_skew(market_data);
        
        // 11. Add lead-lag directional skew
        let directional_skew = compute_directional_skew(
            &self.lead_lag_model,
            market_data.recent_binance_move,
            market_data.time_since_binance_move_ms,
            0.0
        ) / 10000.0;
        
        // 12. Generate final quotes
        let bid_depth = final_half_spread + inventory_skew - directional_skew;
        let ask_depth = final_half_spread - inventory_skew + directional_skew;
        
        QuoteSet {
            bid_price: adjusted_microprice * (1.0 - bid_depth),
            ask_price: adjusted_microprice * (1.0 + ask_depth),
            bid_size: self.compute_bid_size(regime_params.max_inventory),
            ask_size: self.compute_ask_size(regime_params.max_inventory),
            
            // Metadata for logging
            regime: self.hmm_filter.most_likely_regime(),
            kappa_used: adjusted_kappa,
            gamma_used: gamma,
            spread_bps: (bid_depth + ask_depth) * 10000.0,
        }
    }
    
    fn update_models(&mut self, market_data: &MarketData) {
        // Update HMM with current observations
        let obs = ObservationVector {
            volatility: self.volatility_estimator.get_sigma(),
            trade_intensity: market_data.trades_per_second,
            imbalance: market_data.trade_imbalance,
            adverse_selection: self.calibration_metrics.recent_adverse_selection,
        };
        self.hmm_filter.update(&obs);
        
        // Update liquidation detector
        self.liquidation_detector.update(
            market_data.open_interest,
            market_data.funding_rate,
            market_data.timestamp
        );
        
        // Update lead-lag model
        self.lead_lag_model.update(
            market_data.binance_mid,
            market_data.hl_mid,
            market_data.timestamp
        );
        
        // Update trade classifier with recent trades
        for trade in &market_data.recent_trades {
            let features = extract_trade_features(trade, market_data);
            self.adverse_selection_adjuster.on_trade(trade, &features);
        }
    }
}
```

---

## Implementation Priority

Based on expected edge per engineering effort:

1. **Week 1-2: Measurement Infrastructure**
   - Build prediction logging
   - Implement Brier decomposition
   - Set up daily calibration reports
   - This is foundational—everything else depends on measuring properly

2. **Week 3-4: Signal Audit**
   - Implement MI estimation
   - Catalog all available signals
   - Run information content analysis
   - Identify highest-value unused signals

3. **Week 5-6: Lead-Lag Model** 
   - Highest expected edge for moderate complexity
   - Implement cross-exchange data feed
   - Build regime-conditioned estimator
   - Integrate into microprice adjustment

4. **Week 7-8: Regime Detection**
   - Implement HMM filter
   - Define regime-specific parameters
   - Replace warmup multiplier with proper Bayesian belief
   - Set up Baum-Welch re-estimation

5. **Week 9-10: Adverse Selection Classifier**
   - Build labeled dataset from historical fills
   - Train trade classifier
   - Integrate into real-time kappa adjustment
   - Build liquidation detector

6. **Week 11-12: Fill Intensity Model**
   - Implement extended Hawkes process
   - Add queue-position conditioning
   - Integrate with exchange-specific features
   - Replace blended kappa with intensity-derived kappa

Each component should be validated against the measurement infrastructure before moving to the next. If a component doesn't improve calibration metrics, investigate why before building more complexity.