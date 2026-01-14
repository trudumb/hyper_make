---
name: Signal Audit
description: Systematically measure the predictive information content of all available signals before building models
---

# Signal Audit Skill

## Purpose

Systematically measure the predictive information content of all available signals before building models. This prevents wasting effort on low-value features and identifies high-value signals you might be ignoring.

## When to Use

- Before building any new predictive model
- When adding a new data source (cross-exchange feed, new API field)
- Quarterly review of signal value decay
- Debugging why a model stopped working
- Deciding which features to include in a model

## Prerequisites

- `measurement-infrastructure` implemented (for outcome data)
- Historical market data with candidate signals
- Defined prediction targets

---

## Core Concept: Mutual Information

Mutual information measures how many bits of information signal X provides about target Y:

```
I(X; Y) = H(Y) - H(Y|X)
```

Where H is entropy.

**Key properties:**
- I(X; Y) ≥ 0 (always non-negative)
- I(X; Y) = 0 if and only if X and Y are independent
- Works for non-linear relationships (unlike correlation)
- Units are bits (or nats if using natural log)

---

## Signal Catalog

### Book-Derived Signals

```rust
struct BookSignals {
    // Basic
    spread_bps: f64,
    mid_price: f64,
    
    // Imbalance
    microprice_imbalance: f64,     // (bid_size - ask_size) / (bid_size + ask_size) at L1
    book_imbalance_l5: f64,        // Same but integrated over top 5 levels
    book_pressure: f64,            // Weighted depth asymmetry
    
    // Depth
    depth_at_1bps: f64,
    depth_at_5bps: f64,
    depth_at_10bps: f64,
    
    // Shape
    book_slope_bid: f64,           // How quickly depth increases away from mid
    book_slope_ask: f64,
}
```

### Trade-Derived Signals

```rust
struct TradeSignals {
    // Volume imbalance
    trade_imbalance_1s: f64,       // Net signed volume last 1s
    trade_imbalance_10s: f64,
    trade_imbalance_60s: f64,
    
    // Intensity
    trade_arrival_rate: f64,       // Trades per second
    volume_rate: f64,              // Volume per second
    
    // Size distribution
    avg_trade_size: f64,
    trade_size_std: f64,
    large_trade_count_1m: u32,     // Trades > 2σ from mean
    
    // Aggression
    aggressor_imbalance: f64,      // (aggressive_buys - aggressive_sells) / total
}
```

### Hyperliquid-Specific Signals

```rust
struct HyperliquidSignals {
    // Funding
    funding_rate: f64,
    funding_rate_change_1h: f64,
    funding_rate_change_8h: f64,
    predicted_funding_rate: f64,
    time_to_funding_settlement_s: f64,
    
    // Open Interest
    open_interest: f64,
    open_interest_change_1m: f64,
    open_interest_change_5m: f64,
    open_interest_change_1h: f64,
    oi_momentum: f64,              // Acceleration of OI change
    
    // Vault activity
    hlp_vault_position: f64,       // If available
}
```

### Cross-Exchange Signals

```rust
struct CrossExchangeSignals {
    // Binance
    binance_mid: f64,
    binance_spread_bps: f64,
    binance_hl_basis_bps: f64,     // Binance mid - HL mid
    
    // Lead indicators
    binance_return_100ms: f64,     // Binance price change last 100ms
    binance_return_500ms: f64,
    binance_return_1s: f64,
    
    // Volume ratio
    binance_volume_ratio: f64,     // Binance volume / HL volume
}
```

### Composite Signals

```rust
struct CompositeSignals {
    // Interactions
    funding_x_imbalance: f64,      // funding_rate * trade_imbalance
    oi_x_funding: f64,             // OI change * funding rate
    basis_x_imbalance: f64,        // Cross-exchange basis * book imbalance
    
    // Momentum
    price_momentum_1m: f64,
    price_momentum_5m: f64,
    volume_momentum: f64,
}
```

---

## Prediction Targets

```rust
enum PredictionTarget {
    // Direction
    PriceDirection1s,     // sign(price[t+1s] - price[t])
    PriceDirection10s,
    PriceDirection60s,
    
    // Magnitude
    AbsReturn1s,
    AbsReturn10s,
    Volatility1m,
    
    // Fill-related
    FillWithin1s,
    FillWithin10s,
    TimeToNextFill,
    
    // Adverse selection
    AdverseOnNextFill,    // Did price move against us?
    InformedFlow,         // Was the trade informed?
    
    // Regime
    RegimeTransition,     // Will regime change in next minute?
}
```

---

## Mutual Information Estimation

### k-NN Estimator (Kraskov et al.)

For continuous variables, use the k-nearest-neighbor estimator:

```rust
use kdtree::KdTree;

fn estimate_mutual_information(
    x: &[f64],
    y: &[f64],
    k: usize,  // Typically 3-10
) -> f64 {
    let n = x.len();
    assert_eq!(n, y.len());
    
    // Normalize to [0, 1] to handle different scales
    let x_norm = normalize(x);
    let y_norm = normalize(y);
    
    // Build k-d trees
    let mut joint_tree = KdTree::new(2);
    let mut x_tree = KdTree::new(1);
    let mut y_tree = KdTree::new(1);
    
    for i in 0..n {
        joint_tree.add(&[x_norm[i], y_norm[i]], i).unwrap();
        x_tree.add(&[x_norm[i]], i).unwrap();
        y_tree.add(&[y_norm[i]], i).unwrap();
    }
    
    let mut mi_sum = 0.0;
    
    for i in 0..n {
        // Find k-th nearest neighbor distance in joint space (Chebyshev/max norm)
        let neighbors = joint_tree.nearest(&[x_norm[i], y_norm[i]], k + 1, &chebyshev_distance).unwrap();
        let eps = neighbors.last().unwrap().0;  // Distance to k-th neighbor
        
        // Count points within eps in marginals
        let n_x = count_within_chebyshev(&x_tree, x_norm[i], eps);
        let n_y = count_within_chebyshev(&y_tree, y_norm[i], eps);
        
        mi_sum += digamma(k as f64) + digamma(n as f64) 
                  - digamma(n_x as f64) - digamma(n_y as f64);
    }
    
    (mi_sum / n as f64).max(0.0)  // MI is non-negative
}

fn digamma(x: f64) -> f64 {
    if x < 6.0 {
        digamma(x + 1.0) - 1.0 / x
    } else {
        x.ln() - 1.0 / (2.0 * x) - 1.0 / (12.0 * x.powi(2))
    }
}

fn normalize(x: &[f64]) -> Vec<f64> {
    let min = x.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = (max - min).max(1e-10);
    x.iter().map(|&v| (v - min) / range).collect()
}
```

### For Binary Targets

Use the simpler binned estimator:

```rust
fn estimate_mi_binary_target(
    x: &[f64],
    y: &[bool],
    num_bins: usize,
) -> f64 {
    let n = x.len() as f64;
    
    // Bin the continuous variable
    let x_min = x.iter().cloned().fold(f64::INFINITY, f64::min);
    let x_max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let bin_width = (x_max - x_min) / num_bins as f64;
    
    // Count joint and marginal frequencies
    let mut joint_counts = vec![[0usize; 2]; num_bins];  // [bin][outcome]
    let mut x_counts = vec![0usize; num_bins];
    let mut y_counts = [0usize; 2];
    
    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let bin = ((xi - x_min) / bin_width).floor() as usize;
        let bin = bin.min(num_bins - 1);
        let yi = if yi { 1 } else { 0 };
        
        joint_counts[bin][yi] += 1;
        x_counts[bin] += 1;
        y_counts[yi] += 1;
    }
    
    // Compute MI
    let mut mi = 0.0;
    for bin in 0..num_bins {
        for outcome in 0..2 {
            let p_xy = joint_counts[bin][outcome] as f64 / n;
            let p_x = x_counts[bin] as f64 / n;
            let p_y = y_counts[outcome] as f64 / n;
            
            if p_xy > 0.0 && p_x > 0.0 && p_y > 0.0 {
                mi += p_xy * (p_xy / (p_x * p_y)).ln();
            }
        }
    }
    
    mi.max(0.0)
}
```

---

## Signal Analysis Framework

```rust
struct SignalAnalysisResult {
    signal_name: String,
    target_name: String,
    
    // Information content
    mutual_information_bits: f64,
    mutual_information_normalized: f64,  // MI / H(Y), fraction of target entropy explained
    
    // Linear relationship (for comparison)
    correlation: f64,
    correlation_abs: f64,
    
    // Predictive power (if target is binary)
    auc_roc: Option<f64>,
    
    // Lag analysis
    optimal_lag_ms: i64,
    mi_at_optimal_lag: f64,
    
    // Regime dependence
    mi_by_regime: HashMap<String, f64>,
    regime_variance_ratio: f64,  // max(MI) / min(MI) across regimes
    
    // Stationarity
    mi_trend_30d: f64,  // Is MI increasing or decreasing over time?
}

fn analyze_signal(
    signal_name: &str,
    signal_values: &[f64],
    target_name: &str,
    target_values: &[f64],  // or &[bool] for binary
    timestamps: &[u64],
    regimes: &[String],
) -> SignalAnalysisResult {
    // Basic MI
    let mi = estimate_mutual_information(signal_values, target_values, 5);
    
    // Correlation
    let corr = pearson_correlation(signal_values, target_values);
    
    // Lag analysis
    let (optimal_lag, mi_at_lag) = find_optimal_lag(signal_values, target_values, timestamps);
    
    // MI by regime
    let mut mi_by_regime = HashMap::new();
    let unique_regimes: HashSet<_> = regimes.iter().collect();
    
    for regime in unique_regimes {
        let mask: Vec<bool> = regimes.iter().map(|r| r == regime).collect();
        let filtered_signal: Vec<f64> = signal_values.iter()
            .zip(&mask)
            .filter(|(_, &m)| m)
            .map(|(s, _)| *s)
            .collect();
        let filtered_target: Vec<f64> = target_values.iter()
            .zip(&mask)
            .filter(|(_, &m)| m)
            .map(|(t, _)| *t)
            .collect();
        
        if filtered_signal.len() >= 100 {
            let regime_mi = estimate_mutual_information(&filtered_signal, &filtered_target, 5);
            mi_by_regime.insert(regime.clone(), regime_mi);
        }
    }
    
    // Regime variance
    let mi_values: Vec<f64> = mi_by_regime.values().cloned().collect();
    let regime_variance_ratio = if mi_values.len() >= 2 {
        let max_mi = mi_values.iter().cloned().fold(0.0, f64::max);
        let min_mi = mi_values.iter().cloned().fold(f64::INFINITY, f64::min);
        max_mi / min_mi.max(0.001)
    } else {
        1.0
    };
    
    // Target entropy (for normalization)
    let target_entropy = compute_entropy(target_values);
    
    SignalAnalysisResult {
        signal_name: signal_name.to_string(),
        target_name: target_name.to_string(),
        mutual_information_bits: mi,
        mutual_information_normalized: mi / target_entropy.max(0.001),
        correlation: corr,
        correlation_abs: corr.abs(),
        auc_roc: None,  // Compute separately if needed
        optimal_lag_ms: optimal_lag,
        mi_at_optimal_lag: mi_at_lag,
        mi_by_regime,
        regime_variance_ratio,
        mi_trend_30d: 0.0,  // Compute from historical data
    }
}

fn find_optimal_lag(
    signal: &[f64],
    target: &[f64],
    timestamps: &[u64],
) -> (i64, f64) {
    let candidate_lags: Vec<i64> = vec![-500, -200, -100, -50, 0, 50, 100, 200, 500];
    
    let mut best_lag = 0i64;
    let mut best_mi = 0.0;
    
    for &lag_ms in &candidate_lags {
        let aligned = align_with_lag(signal, target, timestamps, lag_ms);
        if aligned.0.len() < 100 { continue; }
        
        let mi = estimate_mutual_information(&aligned.0, &aligned.1, 5);
        if mi > best_mi {
            best_mi = mi;
            best_lag = lag_ms;
        }
    }
    
    (best_lag, best_mi)
}
```

---

## Signal Audit Report

```rust
fn generate_signal_audit_report(
    signals: &HashMap<String, Vec<f64>>,
    target_name: &str,
    target: &[f64],
    timestamps: &[u64],
    regimes: &[String],
) -> String {
    let mut results: Vec<SignalAnalysisResult> = Vec::new();
    
    for (name, values) in signals {
        let result = analyze_signal(name, values, target_name, target, timestamps, regimes);
        results.push(result);
    }
    
    // Sort by MI descending
    results.sort_by(|a, b| b.mutual_information_bits.partial_cmp(&a.mutual_information_bits).unwrap());
    
    let mut report = format!("=== Signal Audit Report ===\nTarget: {}\n\n", target_name);
    
    report.push_str("Signal                      MI (bits)  Corr    Opt Lag   Regime Var\n");
    report.push_str("─────────────────────────────────────────────────────────────────────\n");
    
    for result in &results {
        report.push_str(&format!(
            "{:<26} {:.4}     {:.2}    {:>5}ms    {:.1}x\n",
            result.signal_name,
            result.mutual_information_bits,
            result.correlation,
            result.optimal_lag_ms,
            result.regime_variance_ratio,
        ));
    }
    
    // Actionable insights
    report.push_str("\nACTIONABLE INSIGHTS:\n");
    
    // Highest unused signal
    if let Some(top) = results.first() {
        report.push_str(&format!(
            "1. {} has highest MI ({:.4} bits) - prioritize if not already used\n",
            top.signal_name, top.mutual_information_bits
        ));
    }
    
    // Regime-conditional signals
    for result in &results {
        if result.regime_variance_ratio > 2.0 {
            report.push_str(&format!(
                "2. {} has {:.1}x higher MI in some regimes - consider regime conditioning\n",
                result.signal_name, result.regime_variance_ratio
            ));
            break;
        }
    }
    
    // Lagged signals
    for result in &results {
        if result.optimal_lag_ms != 0 && result.mi_at_optimal_lag > result.mutual_information_bits * 1.2 {
            report.push_str(&format!(
                "3. {} has 20%+ more MI at {}ms lag - incorporate lag in feature\n",
                result.signal_name, result.optimal_lag_ms
            ));
            break;
        }
    }
    
    // Correlated but low MI (non-linear relationship)
    for result in &results {
        if result.correlation_abs > 0.3 && result.mutual_information_bits < 0.01 {
            report.push_str(&format!(
                "4. {} has high correlation but low MI - relationship may be noisy or spurious\n",
                result.signal_name
            ));
            break;
        }
    }
    
    report
}
```

### Example Report Output

```
=== Signal Audit Report ===
Target: PriceDirection1s

Signal                      MI (bits)  Corr    Opt Lag   Regime Var
─────────────────────────────────────────────────────────────────────
binance_return_100ms        0.0890     0.31    -150ms    2.3x
trade_imbalance_1s          0.0670     0.24       0ms    1.4x
microprice_imbalance        0.0450     0.19       0ms    1.2x
funding_x_imbalance         0.0410     0.15       0ms    3.1x
open_interest_change_1m     0.0230     0.08       0ms    1.1x
book_pressure               0.0180     0.11       0ms    1.3x
funding_rate                0.0120     0.05       0ms    1.8x

ACTIONABLE INSIGHTS:
1. binance_return_100ms has highest MI (0.089 bits) - prioritize if not already used
2. funding_x_imbalance has 3.1x higher MI in some regimes - consider regime conditioning
3. binance_return_100ms has 20%+ more MI at -150ms lag - incorporate lag in feature
```

---

## Signal Quality Thresholds

```rust
struct SignalQualityThresholds {
    // Minimum MI to include in model
    min_mi_bits: f64,           // 0.01 typical
    
    // Minimum samples for reliable estimate
    min_samples: usize,         // 1000 typical
    
    // Maximum regime variance before requiring conditioning
    max_regime_variance: f64,   // 3.0 typical
    
    // Minimum correlation for sanity check
    min_correlation: f64,       // 0.05 typical
}

fn filter_signals(
    results: &[SignalAnalysisResult],
    thresholds: &SignalQualityThresholds,
) -> Vec<&SignalAnalysisResult> {
    results.iter()
        .filter(|r| {
            r.mutual_information_bits >= thresholds.min_mi_bits
            && r.correlation_abs >= thresholds.min_correlation
        })
        .collect()
}

fn flag_regime_conditional(
    results: &[SignalAnalysisResult],
    thresholds: &SignalQualityThresholds,
) -> Vec<&SignalAnalysisResult> {
    results.iter()
        .filter(|r| r.regime_variance_ratio > thresholds.max_regime_variance)
        .collect()
}
```

---

## Tracking Signal Decay

Signals lose value over time as:
- Other participants discover them
- Market structure changes
- Regime shifts

Track MI over rolling windows:

```rust
fn compute_signal_decay(
    signal_name: &str,
    historical_mis: &[(NaiveDate, f64)],  // (date, MI) pairs
) -> SignalDecayReport {
    // Linear regression on MI over time
    let n = historical_mis.len() as f64;
    let x: Vec<f64> = (0..historical_mis.len()).map(|i| i as f64).collect();
    let y: Vec<f64> = historical_mis.iter().map(|(_, mi)| *mi).collect();
    
    let x_mean = x.iter().sum::<f64>() / n;
    let y_mean = y.iter().sum::<f64>() / n;
    
    let slope = x.iter().zip(&y)
        .map(|(xi, yi)| (xi - x_mean) * (yi - y_mean))
        .sum::<f64>()
        / x.iter().map(|xi| (xi - x_mean).powi(2)).sum::<f64>();
    
    // Half-life: how long until MI drops by 50%?
    let current_mi = y.last().unwrap();
    let half_life_days = if slope < 0.0 {
        (current_mi * 0.5) / (-slope)
    } else {
        f64::INFINITY  // MI is increasing or stable
    };
    
    SignalDecayReport {
        signal_name: signal_name.to_string(),
        current_mi: *current_mi,
        mi_30d_ago: historical_mis.get(historical_mis.len().saturating_sub(30))
            .map(|(_, mi)| *mi)
            .unwrap_or(*current_mi),
        trend_per_day: slope,
        half_life_days,
        action: if half_life_days < 30.0 {
            "URGENT: Signal decaying rapidly. Investigate or replace.".to_string()
        } else if half_life_days < 90.0 {
            "WARNING: Signal decaying. Monitor closely.".to_string()
        } else {
            "OK: Signal stable.".to_string()
        },
    }
}
```

---

## Dependencies

- **Requires**: measurement-infrastructure (for outcome data), historical market data
- **Enables**: All model skills (by identifying which features to use)

## Common Mistakes

1. **Using correlation instead of MI**: Correlation misses non-linear relationships
2. **Not checking lag**: Some signals lead the target and are more valuable at a lag
3. **Ignoring regime conditioning**: A signal useless overall might be gold in specific regimes
4. **Not tracking decay**: Signals that worked last year might be worthless now
5. **Too few samples**: MI estimation needs 1000+ samples for reliability

## Next Steps

After signal audit:
1. Select top signals for your target (MI > 0.01 bits)
2. Flag regime-conditional signals for special handling
3. Incorporate optimal lags into feature engineering
4. Read the relevant model skill to build the predictor
5. Set up decay tracking for production monitoring
