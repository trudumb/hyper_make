# Fill Intensity Hawkes - Implementation Details

All Rust code blocks extracted from [SKILL.md](./SKILL.md).

---

## State-Dependent Baseline Intensity

Computes the baseline intensity `mu(t)` as a function of funding rate, OI change, settlement phase, and volatility.

```rust
fn compute_baseline_intensity(
    params: &BaselineParams,
    market_state: &MarketState,
) -> f64 {
    let funding_effect = params.w_funding * market_state.funding_rate;
    let oi_effect = params.w_oi * market_state.oi_change_rate;

    // Cyclical effect: activity increases near funding settlement
    let settlement_phase = market_state.time_to_settlement_s / (8.0 * 3600.0) * TAU;
    let settlement_effect = params.w_settlement * settlement_phase.sin();

    let volatility_effect = params.w_volatility * market_state.sigma.ln();

    params.mu_0 * (funding_effect + oi_effect + settlement_effect + volatility_effect).exp()
}
```

---

## Trade-Type-Dependent Excitation

Computes excitation alpha for each trade, accounting for size, side, and aggressor status.

```rust
fn compute_excitation(
    params: &ExcitationParams,
    trade: &Trade,
    our_side: Side,
) -> f64 {
    let base_alpha = params.alpha_base;

    // Size effect: larger trades excite more (sublinear)
    let size_mult = (trade.size / params.median_trade_size)
        .powf(params.size_exponent)  // Typically 0.3-0.5
        .min(3.0);  // Cap at 3x

    // Side effect: trades on our side are more relevant
    let side_mult = if trade.side == our_side {
        params.same_side_mult  // Typically 1.5
    } else {
        1.0
    };

    // Aggressor effect: market orders signal more urgency
    let aggressor_mult = if trade.is_aggressor {
        params.aggressor_mult  // Typically 1.2
    } else {
        1.0
    };

    base_alpha * size_mult * side_mult * aggressor_mult
}
```

---

## Queue-Position-Dependent Kernel

Adaptive kernel combining temporal decay with queue consumption effects.

```rust
fn adaptive_kernel(
    params: &KernelParams,
    time_since_trade: f64,
    queue_consumed: f64,  // How much queue ahead of us was eaten
) -> f64 {
    // Standard temporal decay
    let temporal = (-params.beta * time_since_trade).exp();

    // Queue consumption effect
    // If queue ahead was eaten, we're closer to the front -> higher fill probability
    let queue_mult = 1.0 + params.queue_sensitivity
        * (queue_consumed / params.typical_queue_size).min(1.0);

    temporal * queue_mult
}
```

---

## Full Model Structs and Implementation

Core data structures and the full `HyperliquidFillIntensityModel` implementation including `intensity_at`, `expected_fills_in_window`, and `fill_probability`.

```rust
struct HyperliquidFillIntensityModel {
    // Baseline parameters
    baseline: BaselineParams,

    // Excitation parameters
    excitation: ExcitationParams,

    // Kernel parameters
    kernel: KernelParams,

    // Depth sensitivity
    depth_half_life_bps: f64,  // How quickly fill prob decays with distance from mid

    // Regime multipliers
    regime_multipliers: HashMap<Regime, f64>,
}

struct BaselineParams {
    mu_0: f64,
    w_funding: f64,
    w_oi: f64,
    w_settlement: f64,
    w_volatility: f64,
}

struct ExcitationParams {
    alpha_base: f64,
    size_exponent: f64,
    same_side_mult: f64,
    aggressor_mult: f64,
    median_trade_size: f64,
}

struct KernelParams {
    beta: f64,  // Temporal decay rate
    queue_sensitivity: f64,
}

impl HyperliquidFillIntensityModel {
    /// Compute instantaneous fill intensity at time t
    fn intensity_at(
        &self,
        t: f64,
        recent_trades: &[Trade],
        queue_position: f64,
        queue_history: &[(f64, f64)],  // (time, queue_size)
        market_state: &MarketState,
        our_side: Side,
    ) -> f64 {
        // State-dependent baseline
        let mu_t = compute_baseline_intensity(&self.baseline, market_state);

        // Excitation from recent trades
        let mut excitation = 0.0;
        for trade in recent_trades {
            let time_since = t - trade.timestamp_s;
            if time_since <= 0.0 || time_since > 60.0 { continue; }

            // Compute alpha for this trade
            let alpha_i = compute_excitation(&self.excitation, trade, our_side);

            // Queue consumption since this trade
            let queue_at_trade = interpolate_queue(queue_history, trade.timestamp_s);
            let queue_consumed = (queue_at_trade - queue_position).max(0.0);

            // Compute kernel
            let kernel = adaptive_kernel(&self.kernel, time_since, queue_consumed);

            excitation += alpha_i * kernel;
        }

        // Regime adjustment
        let regime = market_state.dominant_regime();
        let regime_mult = self.regime_multipliers.get(&regime).copied().unwrap_or(1.0);

        (mu_t + excitation) * regime_mult
    }

    /// Expected fills in a time window at a given depth
    fn expected_fills_in_window(
        &self,
        t_start: f64,
        t_end: f64,
        depth_bps: f64,
        market_state: &MarketState,
        recent_trades: &[Trade],
        queue_position: f64,
        our_side: Side,
    ) -> f64 {
        // Numerical integration (trapezoidal)
        let n_steps = 20;
        let dt = (t_end - t_start) / n_steps as f64;

        let mut integral = 0.0;
        for i in 0..n_steps {
            let t = t_start + (i as f64 + 0.5) * dt;
            let lambda_t = self.intensity_at(
                t, recent_trades, queue_position, &[], market_state, our_side
            );

            // Depth adjustment: further from mid = lower fill intensity
            let depth_decay = (-0.693 * depth_bps / self.depth_half_life_bps).exp();

            integral += lambda_t * depth_decay * dt;
        }

        integral
    }

    /// Fill probability in time window (1 - exp(-expected_fills))
    fn fill_probability(
        &self,
        t_start: f64,
        t_end: f64,
        depth_bps: f64,
        market_state: &MarketState,
        recent_trades: &[Trade],
        queue_position: f64,
        our_side: Side,
    ) -> f64 {
        let expected = self.expected_fills_in_window(
            t_start, t_end, depth_bps, market_state, recent_trades, queue_position, our_side
        );

        1.0 - (-expected).exp()
    }
}
```

---

## Batch Parameter Estimation (MLE)

Log-likelihood function and L-BFGS optimization for fitting model parameters from historical data.

```rust
fn hawkes_log_likelihood(
    params: &HyperliquidFillIntensityModel,
    fills: &[Fill],
    trades: &[Trade],
    market_states: &[MarketState],
    t_max: f64,
) -> f64 {
    let mut log_lik = 0.0;
    let mut intensity_integral = 0.0;

    // For each fill, add log(lambda(t_i))
    for fill in fills {
        let t = fill.timestamp_s;
        let recent_trades: Vec<Trade> = trades.iter()
            .filter(|tr| tr.timestamp_s < t && tr.timestamp_s > t - 60.0)
            .cloned()
            .collect();
        let market_state = interpolate_market_state(market_states, t);

        let lambda_i = params.intensity_at(
            t, &recent_trades, fill.queue_position, &[], &market_state, fill.side
        );

        log_lik += lambda_i.ln();
    }

    // Subtract integral_0^T lambda(t) dt (using numerical integration)
    let n_steps = 1000;
    let dt = t_max / n_steps as f64;
    for i in 0..n_steps {
        let t = (i as f64 + 0.5) * dt;
        let recent_trades: Vec<Trade> = trades.iter()
            .filter(|tr| tr.timestamp_s < t && tr.timestamp_s > t - 60.0)
            .cloned()
            .collect();
        let market_state = interpolate_market_state(market_states, t);

        // Average intensity over bid and ask
        let lambda_bid = params.intensity_at(
            t, &recent_trades, 0.0, &[], &market_state, Side::Bid
        );
        let lambda_ask = params.intensity_at(
            t, &recent_trades, 0.0, &[], &market_state, Side::Ask
        );

        intensity_integral += (lambda_bid + lambda_ask) / 2.0 * dt;
    }

    log_lik -= intensity_integral;

    log_lik
}

fn fit_hawkes_model(
    fills: &[Fill],
    trades: &[Trade],
    market_states: &[MarketState],
) -> HyperliquidFillIntensityModel {
    let initial = HyperliquidFillIntensityModel::default();

    // L-BFGS optimization
    let result = lbfgs_optimize(
        |params| -hawkes_log_likelihood(params, fills, trades, market_states, t_max),
        initial,
        LbfgsOptions {
            max_iterations: 100,
            tolerance: 1e-6,
        }
    );

    result.params
}
```

---

## Online Estimation

Real-time parameter updates using stochastic gradient on fill/no-fill events.

```rust
struct OnlineHawkesEstimator {
    params: HyperliquidFillIntensityModel,

    // Sufficient statistics
    fill_count: usize,
    time_observed: f64,

    // Learning rate
    learning_rate: f64,
    min_learning_rate: f64,
    decay_rate: f64,
}

impl OnlineHawkesEstimator {
    fn on_fill(&mut self, fill: &Fill, market_state: &MarketState, recent_trades: &[Trade]) {
        self.fill_count += 1;

        // Compute predicted intensity at fill time
        let predicted = self.params.intensity_at(
            fill.timestamp_s,
            recent_trades,
            fill.queue_position,
            &[],
            market_state,
            fill.side,
        );

        // Innovation: difference between observed (1 fill) and expected
        let dt = fill.time_since_last_fill_s;
        let expected = predicted * dt;
        let innovation = 1.0 - expected;

        // Gradient step on baseline
        let lr = self.learning_rate;
        self.params.baseline.mu_0 *= (1.0 + lr * innovation).max(0.5).min(2.0);

        // Decay learning rate
        self.learning_rate = (self.learning_rate * self.decay_rate)
            .max(self.min_learning_rate);
    }

    fn on_no_fill(&mut self, time_window: f64, market_state: &MarketState) {
        self.time_observed += time_window;

        // Expected fills in window
        let expected = self.params.expected_fills_in_window(
            0.0, time_window, 5.0, market_state, &[], 0.0, Side::Bid
        );

        // If we expected fills but got none, reduce intensity
        if expected > 0.5 {
            let lr = self.learning_rate;
            let adjustment = -lr * expected.min(1.0);
            self.params.baseline.mu_0 *= (1.0 + adjustment).max(0.5);
        }
    }
}
```

---

## Intensity-to-Kappa Conversion

Converts Hawkes intensity to the kappa parameter used by the GLFT formula, via finite difference on depth.

```rust
fn intensity_to_kappa(
    model: &HyperliquidFillIntensityModel,
    market_state: &MarketState,
    recent_trades: &[Trade],
    reference_depth_bps: f64,  // Typically 5-10 bps
    our_side: Side,
) -> f64 {
    // Kappa = d(fill_rate)/d(depth)
    // Estimate via finite difference

    let eps = 0.5;  // 0.5 bps perturbation

    let fill_rate_at_depth = model.expected_fills_in_window(
        0.0, 1.0,  // 1 second window
        reference_depth_bps,
        market_state,
        recent_trades,
        0.0,  // Queue position (assume front)
        our_side,
    );

    let fill_rate_tighter = model.expected_fills_in_window(
        0.0, 1.0,
        reference_depth_bps - eps,
        market_state,
        recent_trades,
        0.0,
        our_side,
    );

    // kappa ~ -Delta(fill_rate) / Delta(depth_fraction)
    let depth_change_fraction = eps / 10000.0;
    let kappa = (fill_rate_tighter - fill_rate_at_depth) / depth_change_fraction;

    kappa.max(100.0)  // Floor to prevent division issues in GLFT
}
```

---

## Validation

Calibration checks comparing predicted fill probabilities against actual fill outcomes, computing Brier score decomposition.

```rust
fn validate_fill_model(
    model: &HyperliquidFillIntensityModel,
    validation_data: &[PredictionRecord],
) -> ValidationReport {
    let mut predictions = Vec::new();
    let mut outcomes = Vec::new();

    for record in validation_data {
        // Extract fill predictions
        for level in &record.predictions.levels {
            predictions.push(level.p_fill_1s);

            let filled = record.outcomes.as_ref()
                .map(|o| o.fills.iter().any(|f| f.level_index == level.index))
                .unwrap_or(false);
            outcomes.push(filled);
        }
    }

    let brier = compute_brier_decomposition(&predictions, &outcomes, 20);

    ValidationReport {
        brier_score: brier.brier_score,
        information_ratio: brier.information_ratio,
        calibration_curve: build_calibration_curve(&predictions, &outcomes, 20),
    }
}
```

---

## Regime Multipliers

Default regime multiplier configuration.

```rust
fn default_regime_multipliers() -> HashMap<Regime, f64> {
    let mut m = HashMap::new();
    m.insert(Regime::Quiet, 1.0);
    m.insert(Regime::Trending, 0.8);
    m.insert(Regime::Volatile, 2.0);
    m.insert(Regime::Cascade, 5.0);
    m
}
```

---

## Quote Engine Integration

How the fill intensity model connects to the quote engine for kappa computation.

```rust
impl QuoteEngine {
    fn compute_kappa(&self, market_state: &MarketState) -> f64 {
        // Get intensity-based kappa
        let intensity_kappa = intensity_to_kappa(
            &self.fill_intensity_model,
            market_state,
            &self.recent_trades,
            10.0,  // Reference depth
            Side::Bid,  // Average bid and ask
        );

        // Apply adverse selection adjustment
        let adjusted_kappa = intensity_kappa
            * self.adverse_selection_adjuster.get_kappa_adjustment();

        // Apply regime blending
        let regime_params = self.hmm_filter.get_blended_params();
        let final_kappa = adjusted_kappa * regime_params.kappa_multiplier;

        final_kappa.max(MIN_KAPPA)
    }
}
```
