---
name: Measurement Infrastructure
description: Build the prediction logging and outcome tracking system that all other model improvements depend on
---

# Measurement Infrastructure Skill

## Purpose

Build the prediction logging and outcome tracking system that all other model improvements depend on. This is the foundation - without proper measurement, you can't know if anything else is working.

**READ THIS SKILL FIRST before any model work.**

## When to Use

- Setting up a new trading system from scratch
- Adding prediction logging to an existing quote engine
- Building the data pipeline for calibration analysis
- Any time you're about to build a predictive model

## Core Principle

Every quote cycle, your system makes implicit predictions:
- "If I place a bid at this price, the probability of fill in 1s is X%"
- "If I get filled, the probability of adverse selection is Y%"
- "The price will move Z bps in the next 10 seconds"

These predictions must be **recorded** with enough granularity to diagnose failures.

---

## Schema Definitions

### 1. Prediction Record (Top Level)

```rust
struct PredictionRecord {
    // Timing
    timestamp_ns: u64,
    quote_cycle_id: u64,
    
    // Market state at prediction time (for conditioning analysis)
    market_state: MarketStateSnapshot,
    
    // Model outputs (what we predicted)
    predictions: ModelPredictions,
    
    // What actually happened (filled in async)
    outcomes: Option<ObservedOutcomes>,
}
```

### 2. Market State Snapshot

Capture everything the model could condition on:

```rust
struct MarketStateSnapshot {
    // === L2 Book State ===
    bid_levels: Vec<(f64, f64)>,  // (price, size) for top N levels
    ask_levels: Vec<(f64, f64)>,
    spread_bps: f64,
    
    // === Derived Quantities ===
    microprice: f64,
    microprice_uncertainty: f64,
    book_imbalance: f64,  // (bid_size - ask_size) / total
    
    // === Kappa Inputs ===
    kappa_book: f64,
    kappa_robust: f64,
    kappa_own: f64,
    kappa_final: f64,
    
    // === Volatility ===
    sigma_bipower: f64,
    sigma_realized_1m: f64,
    sigma_realized_5m: f64,
    
    // === Gamma Inputs ===
    gamma_base: f64,
    gamma_effective: f64,
    
    // === Hyperliquid-Specific ===
    funding_rate: f64,
    funding_rate_predicted: f64,
    time_to_funding_settlement_s: f64,
    open_interest: f64,
    open_interest_delta_1m: f64,
    open_interest_delta_5m: f64,
    
    // === Cross-Exchange ===
    binance_mid: Option<f64>,
    binance_spread_bps: Option<f64>,
    binance_hl_basis_bps: Option<f64>,
    
    // === Position State ===
    inventory: f64,
    inventory_age_s: f64,
    unrealized_pnl: f64,
    
    // === Regime ===
    regime_quiet_prob: f64,
    regime_trending_prob: f64,
    regime_volatile_prob: f64,
    regime_cascade_prob: f64,
}
```

### 3. Model Predictions

What the model actually predicted:

```rust
struct ModelPredictions {
    // Per-level predictions
    levels: Vec<LevelPrediction>,
    
    // Aggregate predictions
    expected_fill_rate_1s: f64,
    expected_fill_rate_10s: f64,
    expected_adverse_selection_bps: f64,
    
    // Direction predictions (if any)
    predicted_price_direction_1s: f64,  // [-1, 1]
    predicted_price_direction_10s: f64,
    direction_confidence: f64,
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
    p_adverse_given_fill: f64,
    expected_adverse_bps_given_fill: f64,
    expected_pnl_given_fill: f64,
    
    // Queue position estimate
    estimated_queue_position: f64,
    estimated_queue_total: f64,
}
```

### 4. Observed Outcomes

Filled in asynchronously after the fact:

```rust
struct ObservedOutcomes {
    // Fill outcomes
    fills: Vec<FillOutcome>,
    
    // Price evolution (for direction/volatility validation)
    price_100ms_later: f64,
    price_1s_later: f64,
    price_10s_later: f64,
    price_60s_later: f64,
    
    // Realized metrics
    realized_volatility_1m: f64,
    realized_adverse_selection_bps: f64,
}

struct FillOutcome {
    level_index: usize,  // Which LevelPrediction this corresponds to
    fill_timestamp_ns: u64,
    fill_latency_ns: u64,  // Time from quote to fill
    fill_price: f64,
    fill_size: f64,
    
    // Post-fill price evolution (for adverse selection measurement)
    mark_price_at_fill: f64,
    mark_price_100ms_later: f64,
    mark_price_1s_later: f64,
    mark_price_10s_later: f64,
}
```

---

## Implementation Checklist

### Step 1: Instrument Quote Generation

```rust
impl QuoteEngine {
    fn generate_quotes(&mut self, market_data: &MarketData) -> QuoteSet {
        // Capture state BEFORE computing quotes
        let market_state = self.snapshot_market_state(market_data);
        
        // Generate quotes (existing logic)
        let quotes = self.compute_optimal_quotes(market_data);
        
        // Capture predictions
        let predictions = self.extract_predictions(&quotes, market_data);
        
        // Log prediction record (outcomes filled later)
        let record = PredictionRecord {
            timestamp_ns: now_ns(),
            quote_cycle_id: self.next_cycle_id(),
            market_state,
            predictions,
            outcomes: None,  // Filled async
        };
        
        self.prediction_logger.log(record);
        
        quotes
    }
}
```

### Step 2: Build Async Outcome Matcher

```rust
struct OutcomeMatcher {
    // Pending predictions awaiting outcomes
    pending: HashMap<u64, PredictionRecord>,  // cycle_id -> record
    
    // Configuration
    max_horizon_s: f64,  // How long to wait for outcomes (60s typical)
}

impl OutcomeMatcher {
    fn on_fill(&mut self, fill: &Fill) {
        // Find the prediction record this fill corresponds to
        if let Some(record) = self.find_matching_record(fill) {
            record.outcomes.get_or_insert_with(Default::default)
                .fills.push(self.create_fill_outcome(fill));
        }
    }
    
    fn on_price_update(&mut self, price: f64, timestamp_ns: u64) {
        // Update price evolution for all pending records
        for record in self.pending.values_mut() {
            let elapsed_s = (timestamp_ns - record.timestamp_ns) as f64 / 1e9;
            
            if let Some(outcomes) = &mut record.outcomes {
                match elapsed_s {
                    t if t >= 0.1 && outcomes.price_100ms_later == 0.0 => {
                        outcomes.price_100ms_later = price;
                    }
                    t if t >= 1.0 && outcomes.price_1s_later == 0.0 => {
                        outcomes.price_1s_later = price;
                    }
                    // ... etc
                    _ => {}
                }
            }
        }
    }
    
    fn flush_completed(&mut self) -> Vec<PredictionRecord> {
        // Remove and return records that have all outcomes filled
        let now = now_ns();
        let mut completed = Vec::new();
        
        self.pending.retain(|_, record| {
            let age_s = (now - record.timestamp_ns) as f64 / 1e9;
            if age_s > self.max_horizon_s {
                if record.outcomes.is_some() {
                    completed.push(record.clone());
                }
                false  // Remove from pending
            } else {
                true   // Keep in pending
            }
        });
        
        completed
    }
}
```

### Step 3: Storage Layer

Use Parquet for columnar storage (efficient for analytical queries):

```rust
struct PredictionStorage {
    writer: ParquetWriter,
    buffer: Vec<PredictionRecord>,
    flush_threshold: usize,  // Flush every N records
}

impl PredictionStorage {
    fn log(&mut self, record: PredictionRecord) {
        self.buffer.push(record);
        
        if self.buffer.len() >= self.flush_threshold {
            self.flush();
        }
    }
    
    fn flush(&mut self) {
        // Convert to columnar format and write
        let batch = self.to_record_batch(&self.buffer);
        self.writer.write_batch(batch);
        self.buffer.clear();
    }
}
```

### Step 4: Query Interface

```rust
struct PredictionQuery {
    storage_path: PathBuf,
}

impl PredictionQuery {
    /// Load predictions for a date range
    fn load_range(&self, start: DateTime, end: DateTime) -> Vec<PredictionRecord> {
        // Parquet predicate pushdown for efficiency
        read_parquet_with_filter(
            &self.storage_path,
            |row| row.timestamp >= start && row.timestamp <= end
        )
    }
    
    /// Load predictions for a specific regime
    fn load_by_regime(&self, regime: Regime, date: Date) -> Vec<PredictionRecord> {
        self.load_range(date.start(), date.end())
            .into_iter()
            .filter(|r| r.market_state.dominant_regime() == regime)
            .collect()
    }
    
    /// Load predictions with fills (for adverse selection analysis)
    fn load_filled(&self, date: Date) -> Vec<PredictionRecord> {
        self.load_range(date.start(), date.end())
            .into_iter()
            .filter(|r| r.outcomes.as_ref().map(|o| !o.fills.is_empty()).unwrap_or(false))
            .collect()
    }
}
```

---

## What to Log vs What to Skip

### Must Log (Critical)
- All fill probability predictions
- All fill outcomes
- Post-fill price evolution (for adverse selection)
- Market state at prediction time
- Regime probabilities

### Should Log (Important)
- Kappa inputs and outputs
- Gamma inputs and outputs
- Queue position estimates
- Cross-exchange state

### Can Skip (Space Optimization)
- Full L2 book beyond top 5 levels
- Sub-100ms price updates
- Predictions for orders that were never placed

---

## Validation Checks

Before considering the infrastructure complete:

```rust
#[test]
fn prediction_outcome_match_rate() {
    let records = load_last_24h();
    
    // Every prediction should eventually get outcomes
    let with_outcomes = records.iter()
        .filter(|r| r.outcomes.is_some())
        .count();
    
    let match_rate = with_outcomes as f64 / records.len() as f64;
    assert!(match_rate > 0.99, "Outcome match rate too low: {}", match_rate);
}

#[test]
fn fill_outcome_completeness() {
    let records = load_last_24h();
    
    for record in &records {
        if let Some(outcomes) = &record.outcomes {
            for fill in &outcomes.fills {
                // Every fill should have post-fill prices
                assert!(fill.mark_price_1s_later > 0.0);
                assert!(fill.mark_price_10s_later > 0.0);
            }
        }
    }
}

#[test]
fn market_state_completeness() {
    let records = load_last_24h();
    
    for record in &records {
        let state = &record.market_state;
        
        // Critical fields must be populated
        assert!(state.microprice > 0.0);
        assert!(state.kappa_final > 0.0);
        assert!(state.sigma_bipower > 0.0);
        
        // Regime probabilities must sum to 1
        let regime_sum = state.regime_quiet_prob 
            + state.regime_trending_prob 
            + state.regime_volatile_prob 
            + state.regime_cascade_prob;
        assert!((regime_sum - 1.0).abs() < 0.01);
    }
}
```

---

## Dependencies

- **Requires**: Your existing quote engine, market data feed
- **Enables**: calibration-analysis, signal-audit, all model skills

## Common Mistakes

1. **Logging only aggregates**: You need per-prediction granularity to diagnose issues
2. **Missing market state**: Without conditioning variables, you can't do conditional calibration
3. **Synchronous outcome filling**: This blocks the hot path; must be async
4. **Not logging "boring" periods**: Quiet market data is just as important for calibration
5. **Forgetting to log predictions for orders that didn't fill**: These are negative examples

---

## Next Steps

Once this infrastructure is in place:
1. Read `calibration-analysis/SKILL.md` to analyze the logged data
2. Read `signal-audit/SKILL.md` to measure signal information content
3. Then proceed to specific model skills
