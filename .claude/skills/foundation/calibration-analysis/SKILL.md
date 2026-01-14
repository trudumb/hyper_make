---
name: Calibration Analysis
description: Systematic analysis of model predictions vs realized outcomes to identify model weaknesses
---

# Calibration Analysis Skill

## Purpose

Systematic analysis of model predictions vs realized outcomes. This tells you exactly where your models are wrong and by how much.

## When to Use

- Daily model health checks
- Debugging unexpected PnL patterns
- Deciding which model component to improve next
- Validating changes before/after deployment
- Any time you want to know "is this model actually working?"

## Prerequisites

- `measurement-infrastructure` must be implemented and logging data
- At least 24 hours of prediction data (preferably 1 week+)

---

## Key Metrics

### 1. Brier Score

The mean squared error of probability predictions:

```
BS = (1/N) Σ (pᵢ - oᵢ)²
```

Where:
- pᵢ = predicted probability
- oᵢ ∈ {0, 1} = actual outcome

**Interpretation:**
- BS = 0: Perfect predictions
- BS = 0.25: Predicting 50% for everything (random)
- BS > 0.25: Worse than random

### 2. Brier Score Decomposition

Split Brier score into interpretable components:

```
BS = Reliability - Resolution + Uncertainty

Reliability = (1/N) Σₖ nₖ(p̄ₖ - ōₖ)²
  - Measures calibration quality
  - Lower is better
  - If your 70% predictions hit 70%, this is 0

Resolution = (1/N) Σₖ nₖ(ōₖ - ō)²
  - Measures discrimination ability
  - Higher is better
  - Do your high predictions differ from low predictions?

Uncertainty = ō(1 - ō)
  - Base rate variance
  - Not controllable, just inherent difficulty
```

### 3. Information Ratio

```
IR = Resolution / Uncertainty
```

**Interpretation:**
- IR > 1.0: Model predictions carry useful information
- IR ≈ 1.0: Model is roughly as good as predicting base rate
- IR < 1.0: Model is adding noise (REMOVE IT)

---

## Implementation

### Brier Score Decomposition

```rust
struct BrierDecomposition {
    brier_score: f64,
    reliability: f64,
    resolution: f64,
    uncertainty: f64,
    information_ratio: f64,
}

fn compute_brier_decomposition(
    predictions: &[f64],
    outcomes: &[bool],
    num_bins: usize,
) -> BrierDecomposition {
    let n = predictions.len() as f64;
    
    // Base rate
    let o_bar: f64 = outcomes.iter()
        .map(|&o| if o { 1.0 } else { 0.0 })
        .sum::<f64>() / n;
    
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
        let o_bar_k: f64 = bin.iter()
            .map(|(_, o)| if *o { 1.0 } else { 0.0 })
            .sum::<f64>() / n_k;
        
        reliability += n_k * (p_bar_k - o_bar_k).powi(2);
        resolution += n_k * (o_bar_k - o_bar).powi(2);
    }
    
    reliability /= n;
    resolution /= n;
    let uncertainty = o_bar * (1.0 - o_bar);
    let brier_score = reliability - resolution + uncertainty;
    let information_ratio = resolution / uncertainty.max(1e-10);
    
    BrierDecomposition {
        brier_score,
        reliability,
        resolution,
        uncertainty,
        information_ratio,
    }
}
```

### Calibration Curve

```rust
struct CalibrationPoint {
    bin_center: f64,        // Mean predicted probability in bin
    realized_rate: f64,     // Actual success rate in bin
    sample_count: usize,    // Number of samples in bin
    confidence_interval: (f64, f64),  // 95% CI on realized rate
}

fn build_calibration_curve(
    predictions: &[f64],
    outcomes: &[bool],
    num_bins: usize,
) -> Vec<CalibrationPoint> {
    // Sort by prediction
    let mut pairs: Vec<(f64, bool)> = predictions.iter()
        .zip(outcomes.iter())
        .map(|(&p, &o)| (p, o))
        .collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    
    // Divide into equal-sized bins
    let bin_size = pairs.len() / num_bins;
    let mut curve = Vec::new();
    
    for i in 0..num_bins {
        let start = i * bin_size;
        let end = if i == num_bins - 1 { pairs.len() } else { (i + 1) * bin_size };
        let bin = &pairs[start..end];
        
        if bin.is_empty() { continue; }
        
        let mean_pred: f64 = bin.iter().map(|(p, _)| p).sum::<f64>() / bin.len() as f64;
        let realized: f64 = bin.iter()
            .map(|(_, o)| if *o { 1.0 } else { 0.0 })
            .sum::<f64>() / bin.len() as f64;
        
        // Wilson score interval for confidence
        let ci = wilson_score_interval(realized, bin.len(), 0.95);
        
        curve.push(CalibrationPoint {
            bin_center: mean_pred,
            realized_rate: realized,
            sample_count: bin.len(),
            confidence_interval: ci,
        });
    }
    
    curve
}

fn wilson_score_interval(p: f64, n: usize, confidence: f64) -> (f64, f64) {
    let z = 1.96;  // 95% confidence
    let n = n as f64;
    
    let denominator = 1.0 + z * z / n;
    let center = (p + z * z / (2.0 * n)) / denominator;
    let margin = z * (p * (1.0 - p) / n + z * z / (4.0 * n * n)).sqrt() / denominator;
    
    ((center - margin).max(0.0), (center + margin).min(1.0))
}
```

### Conditional Calibration

Slice calibration by conditioning variables:

```rust
enum ConditioningVariable {
    VolatilityQuartile,
    FundingRegime,
    TimeOfDay,
    InventoryState,
    RecentFillRate,
    BookImbalance,
    Regime,
}

struct ConditionalCalibration {
    variable: ConditioningVariable,
    slices: HashMap<String, BrierDecomposition>,
}

fn compute_conditional_calibration(
    records: &[PredictionRecord],
    prediction_extractor: impl Fn(&PredictionRecord) -> f64,
    outcome_extractor: impl Fn(&PredictionRecord) -> bool,
    condition: ConditioningVariable,
) -> ConditionalCalibration {
    // Group records by condition
    let mut groups: HashMap<String, Vec<(f64, bool)>> = HashMap::new();
    
    for record in records {
        let slice_name = get_slice_name(record, &condition);
        let pred = prediction_extractor(record);
        let outcome = outcome_extractor(record);
        
        groups.entry(slice_name)
            .or_default()
            .push((pred, outcome));
    }
    
    // Compute Brier decomposition for each slice
    let mut slices = HashMap::new();
    for (name, pairs) in groups {
        if pairs.len() < 100 { continue; }  // Need minimum samples
        
        let preds: Vec<f64> = pairs.iter().map(|(p, _)| *p).collect();
        let outcomes: Vec<bool> = pairs.iter().map(|(_, o)| *o).collect();
        
        slices.insert(name, compute_brier_decomposition(&preds, &outcomes, 20));
    }
    
    ConditionalCalibration {
        variable: condition,
        slices,
    }
}

fn get_slice_name(record: &PredictionRecord, condition: &ConditioningVariable) -> String {
    match condition {
        ConditioningVariable::VolatilityQuartile => {
            let sigma = record.market_state.sigma_bipower;
            if sigma < 0.00005 { "Q1_low".to_string() }
            else if sigma < 0.00010 { "Q2_medium".to_string() }
            else if sigma < 0.00020 { "Q3_high".to_string() }
            else { "Q4_extreme".to_string() }
        }
        ConditioningVariable::FundingRegime => {
            let funding = record.market_state.funding_rate;
            if funding > 0.0005 { "high_positive".to_string() }
            else if funding > 0.0 { "low_positive".to_string() }
            else if funding > -0.0005 { "low_negative".to_string() }
            else { "high_negative".to_string() }
        }
        ConditioningVariable::Regime => {
            let state = &record.market_state;
            if state.regime_cascade_prob > 0.5 { "cascade".to_string() }
            else if state.regime_volatile_prob > 0.5 { "volatile".to_string() }
            else if state.regime_trending_prob > 0.5 { "trending".to_string() }
            else { "quiet".to_string() }
        }
        // ... other conditions
        _ => "unknown".to_string()
    }
}
```

---

## PnL Attribution

Decompose daily PnL to identify problem areas:

```rust
struct PnLAttribution {
    date: NaiveDate,
    gross_pnl: f64,
    
    // Decomposition
    spread_capture: f64,      // Revenue from bid-ask spread
    adverse_selection: f64,   // Loss from fills before adverse moves
    inventory_cost: f64,      // Cost of holding inventory (mark-to-market)
    fees: f64,                // Exchange fees
    
    // By regime
    pnl_by_regime: HashMap<Regime, f64>,
    time_by_regime: HashMap<Regime, f64>,  // Fraction of time
}

fn compute_pnl_attribution(records: &[PredictionRecord]) -> PnLAttribution {
    let mut spread_capture = 0.0;
    let mut adverse_selection = 0.0;
    let mut fees = 0.0;
    let mut pnl_by_regime: HashMap<Regime, f64> = HashMap::new();
    let mut time_by_regime: HashMap<Regime, f64> = HashMap::new();
    
    for record in records {
        let Some(outcomes) = &record.outcomes else { continue };
        let regime = record.market_state.dominant_regime();
        
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
            let price_move_1s = fill.mark_price_1s_later - fill.mark_price_at_fill;
            let adverse = match level.side {
                Side::Bid => -price_move_1s,  // Bought, price dropped
                Side::Ask => price_move_1s,   // Sold, price rose
            };
            adverse_selection += adverse.min(0.0) * fill.fill_size;
            
            // Fees
            fees += fill.fill_price * fill.fill_size * 0.00015;  // 1.5 bps
            
            // By regime
            *pnl_by_regime.entry(regime).or_default() += 
                spread_earned * fill.fill_size + adverse.min(0.0) * fill.fill_size;
        }
        
        *time_by_regime.entry(regime).or_default() += 1.0;
    }
    
    // Normalize time fractions
    let total_time: f64 = time_by_regime.values().sum();
    for v in time_by_regime.values_mut() {
        *v /= total_time;
    }
    
    // Inventory cost computed separately from position changes
    let inventory_cost = compute_inventory_mtm(records);
    
    PnLAttribution {
        date: records[0].timestamp.date(),
        gross_pnl: spread_capture + adverse_selection - fees - inventory_cost,
        spread_capture,
        adverse_selection,
        inventory_cost,
        fees,
        pnl_by_regime,
        time_by_regime,
    }
}
```

---

## Daily Report Template

```
=== Calibration Report: {date} ===

PnL Attribution
───────────────────────────────────────────
Gross PnL:              ${gross_pnl:>10.2}
├── Spread Capture:     ${spread_capture:>10.2}  {spread_status}
├── Adverse Selection:  ${adverse_selection:>10.2}  {as_status}
├── Inventory Cost:     ${inventory_cost:>10.2}
└── Fees:               ${fees:>10.2}

Model Calibration
───────────────────────────────────────────
                        Brier   IR      Status
Fill Prediction (1s):   {fp_brier:.3}  {fp_ir:.2}   {fp_status}
Fill Prediction (10s):  {fp10_brier:.3} {fp10_ir:.2}  {fp10_status}
Adverse Selection:      {as_brier:.3}  {as_ir:.2}   {as_status_model}
Volatility (RMSE):      {vol_rmse:.6}          {vol_status}

Regime Distribution
───────────────────────────────────────────
            Time    PnL         PnL/Hour
Quiet:      {quiet_time:>4.0%}   ${quiet_pnl:>8.2}   ${quiet_rate:>6.2}/hr
Trending:   {trend_time:>4.0%}   ${trend_pnl:>8.2}   ${trend_rate:>6.2}/hr
Volatile:   {vol_time:>4.0%}   ${vol_pnl:>8.2}   ${vol_rate:>6.2}/hr
Cascade:    {casc_time:>4.0%}   ${casc_pnl:>8.2}   ${casc_rate:>6.2}/hr

Conditional Calibration Issues
───────────────────────────────────────────
{conditional_issues}

Actionable Items
───────────────────────────────────────────
{action_items}
```

### Report Generation

```rust
fn generate_daily_report(date: NaiveDate) -> String {
    let records = load_predictions_for_date(date);
    
    // PnL attribution
    let pnl = compute_pnl_attribution(&records);
    
    // Model calibration
    let fill_1s = compute_brier_decomposition(
        &extract_fill_predictions_1s(&records),
        &extract_fill_outcomes_1s(&records),
        20
    );
    let fill_10s = compute_brier_decomposition(
        &extract_fill_predictions_10s(&records),
        &extract_fill_outcomes_10s(&records),
        20
    );
    let adverse = compute_brier_decomposition(
        &extract_adverse_predictions(&records),
        &extract_adverse_outcomes(&records),
        20
    );
    
    // Conditional calibration
    let conditions = [
        ConditioningVariable::Regime,
        ConditioningVariable::VolatilityQuartile,
        ConditioningVariable::FundingRegime,
    ];
    
    let mut issues = Vec::new();
    for cond in conditions {
        let cal = compute_conditional_calibration(&records, /* ... */, cond);
        for (slice, decomp) in &cal.slices {
            if decomp.information_ratio < 1.0 {
                issues.push(format!(
                    "- {} in {}: IR={:.2} (model adding noise)",
                    cond, slice, decomp.information_ratio
                ));
            }
        }
    }
    
    // Action items
    let actions = generate_action_items(&pnl, &fill_1s, &adverse, &issues);
    
    format_report(/* ... */)
}

fn generate_action_items(
    pnl: &PnLAttribution,
    fill_cal: &BrierDecomposition,
    adverse_cal: &BrierDecomposition,
    issues: &[String],
) -> Vec<String> {
    let mut actions = Vec::new();
    
    // Adverse selection is biggest PnL drag
    if pnl.adverse_selection < -100.0 && pnl.adverse_selection.abs() > pnl.spread_capture * 0.5 {
        actions.push("HIGH: Adverse selection > 50% of spread capture. Review adverse-selection-classifier.".to_string());
    }
    
    // Model IR below 1.0
    if fill_cal.information_ratio < 1.0 {
        actions.push(format!(
            "HIGH: Fill prediction IR={:.2}. Model adding noise. Consider removing or retraining.",
            fill_cal.information_ratio
        ));
    }
    
    if adverse_cal.information_ratio < 1.0 {
        actions.push(format!(
            "HIGH: Adverse selection IR={:.2}. Model adding noise. Review classifier.",
            adverse_cal.information_ratio
        ));
    }
    
    // Regime-specific issues
    if let Some(&cascade_pnl) = pnl.pnl_by_regime.get(&Regime::Cascade) {
        if cascade_pnl < -50.0 {
            actions.push(format!(
                "MEDIUM: Cascade regime PnL=${:.2}. Review liquidation detector thresholds.",
                cascade_pnl
            ));
        }
    }
    
    // Conditional calibration issues
    if !issues.is_empty() {
        actions.push(format!(
            "MEDIUM: {} conditional calibration issues. Review conditional slices.",
            issues.len()
        ));
    }
    
    actions
}
```

---

## Alert Thresholds

Set up automated alerts:

```rust
struct AlertThresholds {
    // Model health
    min_information_ratio: f64,      // 1.0 - below this, model is useless
    max_brier_score: f64,            // 0.25 - above this, worse than random
    
    // PnL
    max_daily_loss: f64,             // Dollar amount
    max_adverse_selection_ratio: f64, // AS / spread_capture
    
    // Regime
    max_cascade_loss: f64,           // Dollar amount in cascade regime
}

impl Default for AlertThresholds {
    fn default() -> Self {
        AlertThresholds {
            min_information_ratio: 1.0,
            max_brier_score: 0.25,
            max_daily_loss: 500.0,
            max_adverse_selection_ratio: 0.7,
            max_cascade_loss: 100.0,
        }
    }
}

fn check_alerts(report: &DailyReport, thresholds: &AlertThresholds) -> Vec<Alert> {
    let mut alerts = Vec::new();
    
    if report.fill_calibration.information_ratio < thresholds.min_information_ratio {
        alerts.push(Alert::Critical(format!(
            "Fill prediction IR={:.2} below threshold {}",
            report.fill_calibration.information_ratio,
            thresholds.min_information_ratio
        )));
    }
    
    if report.pnl.gross_pnl < -thresholds.max_daily_loss {
        alerts.push(Alert::Critical(format!(
            "Daily loss ${:.2} exceeds threshold ${:.2}",
            report.pnl.gross_pnl,
            thresholds.max_daily_loss
        )));
    }
    
    // ... more checks
    
    alerts
}
```

---

## Common Patterns

### "Model is well-calibrated overall but fails in regime X"

This is the most common pattern. Solution:
1. Identify which regime has IR < 1.0
2. Either: train regime-specific model, or
3. Fall back to wider spreads / simpler model in that regime

### "Calibration looks good but still losing money"

Possible causes:
1. Good calibration on the wrong metric (e.g., calibrated fill prediction but adverse selection is the real problem)
2. Execution slippage not captured in calibration
3. Latency effects (predictions stale by the time orders placed)

### "IR > 1 but Brier score is high"

Model has good discrimination (can tell high from low) but poor calibration (predictions don't match frequencies). Fix with isotonic regression or Platt scaling.

---

## Dependencies

- **Requires**: measurement-infrastructure (prediction logs with outcomes)
- **Enables**: All model improvement work, daily-calibration-report

## Next Steps

After analyzing calibration:
1. Identify weakest component (lowest IR or biggest PnL drag)
2. Read that component's skill file
3. Use signal-audit to identify better features
4. Implement improvement
5. Re-run calibration to validate
