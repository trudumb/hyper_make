# Cross-Exchange Lead-Lag Model Skill

## Purpose

Exploit the fact that Binance BTC perpetual leads Hyperliquid by 50-500ms. This is potentially your highest-value proprietary signal because:

- It's based on physical reality (Binance has 10-100x more volume)
- It's predictive of short-term direction
- It enables microprice adjustment and quote skew
- It's continuously measurable and adaptable

## When to Use

- You have a cross-exchange data feed (Binance websocket)
- Building directional microprice adjustment
- Adding quote skew based on recent price moves elsewhere
- Improving adverse selection detection (arbitrage flow)

## Prerequisites

- Real-time Binance BTC perpetual market data feed
- Synchronized timestamps (NTP or better)
- `measurement-infrastructure` for validation
- `signal-audit` to quantify information content

---

## Core Observation

Price discovery happens on Binance first because:
1. Higher volume (10-100x Hyperliquid)
2. More sophisticated participants
3. Lower latency infrastructure
4. Deeper liquidity

When Binance price moves, Hyperliquid follows with a lag of 50-500ms depending on:
- Volatility regime (high vol = faster arbitrage = shorter lag)
- Time of day (more participants = shorter lag)
- Move size (larger moves propagate faster)

---

## Lead-Lag Estimation

### Basic Model

```
HL_return(t) = β × Binance_return(t - lag) + ε
```

Where:
- lag = optimal lead time (typically 100-300ms)
- β = transmission coefficient (typically 0.6-0.9)
- ε = noise (Hyperliquid-specific flow)

### Estimator Implementation

```rust
struct LeadLagEstimator {
    // Price change history
    binance_returns: RingBuffer<(u64, f64)>,  // (timestamp_ns, return)
    hl_returns: RingBuffer<(u64, f64)>,
    
    // Current estimates
    lag_estimate_ms: f64,
    lag_estimate_std: f64,
    beta_estimate: f64,
    beta_estimate_std: f64,
    r_squared: f64,
    
    // Sample tracking
    sample_count: usize,
    last_estimation_time: u64,
    
    // Configuration
    estimation_interval_ms: u64,
    min_samples: usize,
    return_window_ms: u64,  // Window for computing returns
}

impl LeadLagEstimator {
    fn new() -> Self {
        LeadLagEstimator {
            binance_returns: RingBuffer::new(10000),
            hl_returns: RingBuffer::new(10000),
            lag_estimate_ms: 200.0,  // Initial guess
            lag_estimate_std: 100.0,
            beta_estimate: 0.8,
            beta_estimate_std: 0.2,
            r_squared: 0.0,
            sample_count: 0,
            last_estimation_time: 0,
            estimation_interval_ms: 1000,  // Re-estimate every second
            min_samples: 100,
            return_window_ms: 100,  // 100ms returns
        }
    }
    
    fn on_binance_price(&mut self, price: f64, timestamp_ns: u64) {
        if let Some(&(prev_ts, prev_price)) = self.binance_returns.latest() {
            let dt_ms = (timestamp_ns - prev_ts) / 1_000_000;
            if dt_ms >= self.return_window_ms {
                let ret = (price - prev_price) / prev_price;
                self.binance_returns.push((timestamp_ns, ret));
            }
        } else {
            self.binance_returns.push((timestamp_ns, 0.0));
        }
    }
    
    fn on_hl_price(&mut self, price: f64, timestamp_ns: u64) {
        if let Some(&(prev_ts, prev_price)) = self.hl_returns.latest() {
            let dt_ms = (timestamp_ns - prev_ts) / 1_000_000;
            if dt_ms >= self.return_window_ms {
                let ret = (price - prev_price) / prev_price;
                self.hl_returns.push((timestamp_ns, ret));
                
                self.sample_count += 1;
                
                // Periodically re-estimate
                if timestamp_ns - self.last_estimation_time > self.estimation_interval_ms * 1_000_000 {
                    self.estimate_lag();
                    self.last_estimation_time = timestamp_ns;
                }
            }
        } else {
            self.hl_returns.push((timestamp_ns, 0.0));
        }
    }
    
    fn estimate_lag(&mut self) {
        if self.sample_count < self.min_samples {
            return;
        }
        
        // Grid search over candidate lags
        let candidate_lags: Vec<i64> = vec![
            -100, -50, 0, 50, 100, 150, 200, 250, 300, 400, 500
        ];
        
        let mut best_lag = 0i64;
        let mut best_r2 = -1.0;
        let mut best_beta = 0.0;
        let mut best_beta_std = 1.0;
        
        for &lag_ms in &candidate_lags {
            if let Some((beta, r2, beta_std)) = self.compute_regression_at_lag(lag_ms) {
                if r2 > best_r2 {
                    best_r2 = r2;
                    best_lag = lag_ms;
                    best_beta = beta;
                    best_beta_std = beta_std;
                }
            }
        }
        
        // Exponential smoothing update
        let alpha = 0.1;  // Slow adaptation
        self.lag_estimate_ms = alpha * best_lag as f64 + (1.0 - alpha) * self.lag_estimate_ms;
        self.beta_estimate = alpha * best_beta + (1.0 - alpha) * self.beta_estimate;
        self.beta_estimate_std = alpha * best_beta_std + (1.0 - alpha) * self.beta_estimate_std;
        self.r_squared = alpha * best_r2 + (1.0 - alpha) * self.r_squared;
    }
    
    fn compute_regression_at_lag(&self, lag_ms: i64) -> Option<(f64, f64, f64)> {
        // Align Binance returns with HL returns at specified lag
        let lag_ns = lag_ms * 1_000_000;
        
        let mut x_vec = Vec::new();  // Binance returns
        let mut y_vec = Vec::new();  // HL returns
        
        for &(hl_ts, hl_ret) in self.hl_returns.iter() {
            // Find Binance return at (hl_ts - lag)
            let target_ts = (hl_ts as i64 - lag_ns) as u64;
            if let Some(binance_ret) = self.binance_returns.interpolate_at(target_ts) {
                x_vec.push(binance_ret);
                y_vec.push(hl_ret);
            }
        }
        
        if x_vec.len() < 50 {
            return None;
        }
        
        // Linear regression: y = β*x
        let n = x_vec.len() as f64;
        let x_mean: f64 = x_vec.iter().sum::<f64>() / n;
        let y_mean: f64 = y_vec.iter().sum::<f64>() / n;
        
        let mut cov_xy = 0.0;
        let mut var_x = 0.0;
        let mut var_y = 0.0;
        
        for i in 0..x_vec.len() {
            let dx = x_vec[i] - x_mean;
            let dy = y_vec[i] - y_mean;
            cov_xy += dx * dy;
            var_x += dx * dx;
            var_y += dy * dy;
        }
        
        if var_x < 1e-20 {
            return None;
        }
        
        let beta = cov_xy / var_x;
        let r2 = (cov_xy * cov_xy) / (var_x * var_y + 1e-20);
        
        // Standard error of beta
        let residual_var: f64 = y_vec.iter().enumerate()
            .map(|(i, &y)| (y - beta * x_vec[i]).powi(2))
            .sum::<f64>() / (n - 2.0).max(1.0);
        let beta_std = (residual_var / var_x).sqrt();
        
        Some((beta, r2, beta_std))
    }
}
```

### Regime-Conditioned Lead-Lag

The lead-lag relationship varies with volatility:

```rust
struct RegimeConditionedLeadLag {
    // Separate estimators for volatility regimes
    estimators: HashMap<VolatilityRegime, LeadLagEstimator>,
    
    // Current regime (from HMM or vol classifier)
    current_regime: VolatilityRegime,
    
    // Volatility thresholds for regime classification
    vol_thresholds: [f64; 3],  // Low/Medium/High boundaries
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum VolatilityRegime {
    Low,     // Bottom quartile
    Medium,  // Middle two quartiles
    High,    // Top quartile
}

impl RegimeConditionedLeadLag {
    fn new() -> Self {
        let mut estimators = HashMap::new();
        estimators.insert(VolatilityRegime::Low, LeadLagEstimator::new());
        estimators.insert(VolatilityRegime::Medium, LeadLagEstimator::new());
        estimators.insert(VolatilityRegime::High, LeadLagEstimator::new());
        
        RegimeConditionedLeadLag {
            estimators,
            current_regime: VolatilityRegime::Medium,
            vol_thresholds: [0.0001, 0.0003, 0.001],  // 1bp, 3bp, 10bp per 100ms
        }
    }
    
    fn update_regime(&mut self, current_volatility: f64) {
        self.current_regime = if current_volatility < self.vol_thresholds[0] {
            VolatilityRegime::Low
        } else if current_volatility < self.vol_thresholds[1] {
            VolatilityRegime::Medium
        } else {
            VolatilityRegime::High
        };
    }
    
    fn on_prices(&mut self, binance_price: f64, hl_price: f64, timestamp_ns: u64) {
        // Update current regime's estimator
        if let Some(est) = self.estimators.get_mut(&self.current_regime) {
            est.on_binance_price(binance_price, timestamp_ns);
            est.on_hl_price(hl_price, timestamp_ns);
        }
    }
    
    fn get_current_estimate(&self) -> (f64, f64, f64) {
        let est = &self.estimators[&self.current_regime];
        (est.lag_estimate_ms, est.beta_estimate, est.r_squared)
    }
    
    /// Predict remaining HL move given recent Binance move
    fn predict_remaining_move(&self, binance_return: f64, time_since_binance_move_ms: f64) -> f64 {
        let (lag_ms, beta, r2) = self.get_current_estimate();
        
        // If R² is too low, signal is unreliable
        if r2 < 0.1 {
            return 0.0;
        }
        
        // If we're within the lag window, predict remaining move
        if time_since_binance_move_ms < lag_ms {
            let completion_fraction = time_since_binance_move_ms / lag_ms;
            let expected_total_move = beta * binance_return;
            let remaining_move = expected_total_move * (1.0 - completion_fraction);
            
            // Scale by confidence (R²)
            remaining_move * r2.sqrt()
        } else {
            0.0
        }
    }
}
```

---

## Integration into Quote Engine

### Microprice Adjustment

```rust
fn compute_adjusted_microprice(
    local_microprice: f64,
    lead_lag: &RegimeConditionedLeadLag,
    recent_binance_return: f64,
    time_since_binance_move_ms: f64,
) -> f64 {
    let predicted_move = lead_lag.predict_remaining_move(
        recent_binance_return,
        time_since_binance_move_ms,
    );
    
    local_microprice * (1.0 + predicted_move)
}
```

### Quote Skew

When Binance moves and the lag hasn't elapsed, skew quotes:

```rust
fn compute_lead_lag_skew_bps(
    lead_lag: &RegimeConditionedLeadLag,
    recent_binance_return: f64,
    time_since_binance_move_ms: f64,
    max_skew_bps: f64,
) -> f64 {
    let (lag_ms, beta, r2) = lead_lag.get_current_estimate();
    
    // Only skew if signal is reliable and we're in the lag window
    if r2 < 0.2 || time_since_binance_move_ms >= lag_ms {
        return 0.0;
    }
    
    // Expected direction
    let expected_direction = recent_binance_return.signum();
    
    // Skew magnitude based on move size and confidence
    let move_magnitude_bps = recent_binance_return.abs() * 10000.0;
    let confidence = r2.sqrt();
    let time_decay = 1.0 - time_since_binance_move_ms / lag_ms;
    
    let skew = expected_direction * move_magnitude_bps * confidence * time_decay * 0.5;
    
    // Cap at max skew
    skew.max(-max_skew_bps).min(max_skew_bps)
}
```

### Full Integration

```rust
impl QuoteEngine {
    fn generate_quotes_with_lead_lag(&mut self, market_data: &MarketData) -> QuoteSet {
        // Update lead-lag model
        self.lead_lag.update_regime(self.volatility_estimator.get_sigma());
        self.lead_lag.on_prices(
            market_data.binance_mid,
            market_data.hl_mid,
            market_data.timestamp_ns,
        );
        
        // Detect recent Binance move
        let binance_return = (market_data.binance_mid - self.last_binance_mid) / self.last_binance_mid;
        let time_since_move = if binance_return.abs() > 0.0001 {  // >1 bps
            self.last_significant_binance_move_time = market_data.timestamp_ns;
            0.0
        } else {
            (market_data.timestamp_ns - self.last_significant_binance_move_time) as f64 / 1_000_000.0
        };
        
        // Adjust microprice
        let adjusted_microprice = compute_adjusted_microprice(
            self.microprice_estimator.get_microprice(),
            &self.lead_lag,
            binance_return,
            time_since_move,
        );
        
        // Compute skew
        let lead_lag_skew_bps = compute_lead_lag_skew_bps(
            &self.lead_lag,
            binance_return,
            time_since_move,
            5.0,  // Max 5 bps skew
        );
        
        // Apply to quotes
        let base_half_spread = self.compute_base_half_spread();
        let inventory_skew = self.compute_inventory_skew();
        
        let bid_depth = base_half_spread + inventory_skew - lead_lag_skew_bps / 10000.0;
        let ask_depth = base_half_spread - inventory_skew + lead_lag_skew_bps / 10000.0;
        
        QuoteSet {
            bid_price: adjusted_microprice * (1.0 - bid_depth),
            ask_price: adjusted_microprice * (1.0 + ask_depth),
            // ...
        }
    }
}
```

---

## Monitoring and Decay Detection

### Signal Quality Tracking

```rust
struct LeadLagMonitor {
    // Historical R² values
    r2_history: RingBuffer<(u64, f64)>,
    
    // Alert thresholds
    min_r2_warning: f64,   // 0.15 typical
    min_r2_critical: f64,  // 0.10 typical
    decay_rate_warning: f64,  // R² dropping >10% per day
}

impl LeadLagMonitor {
    fn check_health(&self, current_r2: f64) -> SignalHealth {
        // Current R² check
        if current_r2 < self.min_r2_critical {
            return SignalHealth::Critical("Lead-lag R² below critical threshold".to_string());
        }
        if current_r2 < self.min_r2_warning {
            return SignalHealth::Warning("Lead-lag R² below warning threshold".to_string());
        }
        
        // Decay rate check
        if let Some(r2_24h_ago) = self.r2_history.get_at_age(Duration::from_secs(86400)) {
            let decay_rate = (r2_24h_ago - current_r2) / r2_24h_ago;
            if decay_rate > self.decay_rate_warning {
                return SignalHealth::Warning(format!(
                    "Lead-lag R² decaying at {:.1}% per day", decay_rate * 100.0
                ));
            }
        }
        
        SignalHealth::Ok
    }
}
```

### Expected Decay Pattern

Lead-lag signal decays as:
1. More arbitrageurs enter Hyperliquid
2. Technology improves (lower latency)
3. Liquidity deepens

**Typical half-life: 6-12 months**

Plan to:
- Monitor R² continuously
- Reduce reliance as R² drops
- Eventually this edge will be arbitraged away

---

## Validation

### Backtesting

```rust
fn backtest_lead_lag_value(
    market_data: &[MarketData],
    with_lead_lag: bool,
) -> BacktestResult {
    let mut engine = QuoteEngine::new();
    engine.set_lead_lag_enabled(with_lead_lag);
    
    let mut pnl = 0.0;
    let mut fills = 0;
    let mut adverse_selection_total = 0.0;
    
    for tick in market_data {
        let quotes = engine.generate_quotes(tick);
        
        // Simulate fills and track PnL
        // ...
    }
    
    BacktestResult {
        pnl,
        sharpe: pnl / pnl_std,
        adverse_selection_rate: adverse_selection_total / fills as f64,
    }
}
```

### Key Metrics

- **R² > 0.2**: Signal is useful
- **R² > 0.3**: Signal is strong
- **R² > 0.4**: Signal is exceptional (rare)

Compare PnL with/without lead-lag:
- **Expected improvement: 10-30%** in spread capture
- **Adverse selection should decrease** (you anticipate moves)

---

## Dependencies

- **Requires**: Binance websocket feed, synchronized timestamps
- **Enables**: Microprice adjustment, quote skew, arbitrage flow detection

## Common Mistakes

1. **Wrong timestamp sync**: Even 10ms error ruins the estimate
2. **Too aggressive skew**: Don't rely too heavily on a decaying signal
3. **Ignoring regime**: Low-vol has different lag than high-vol
4. **Not monitoring decay**: This edge will shrink over time
5. **Symmetric treatment**: Lag may differ for up vs down moves

## Data Requirements

- Binance BTC-PERP book ticker: ~10ms resolution
- Hyperliquid BTC book ticker: ~50ms resolution minimum
- Timestamps: NTP-synchronized or better
- Storage: ~10GB/month for tick data

## Next Steps

1. Set up Binance websocket feed
2. Implement basic LeadLagEstimator
3. Validate R² > 0.2 on historical data
4. Add regime conditioning
5. Integrate microprice adjustment
6. Add quote skew
7. Set up decay monitoring
8. Plan for eventual edge erosion
