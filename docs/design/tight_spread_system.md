# Tight Spread Quoting System - Design Specification

**Version**: 1.0
**Date**: 2026-01-01
**Status**: Design Complete

---

## 1. Overview

### 1.1 Purpose

Enable the market maker to quote tighter spreads (3-5 bps) during favorable market conditions while maintaining the current 8 bps floor during adverse conditions.

### 1.2 Key Insight

Expert market makers don't quote tight all the time. They quote tight when:
- Market regime is CALM
- Toxicity is PREDICTABLY LOW
- Quote update latency is FAST
- Inventory is NEUTRAL

### 1.3 Expected Impact

| Metric | Current | With Tight Spreads |
|--------|---------|-------------------|
| Floor (calm) | 8 bps | 3 bps |
| Floor (normal) | 8 bps | 5 bps |
| Fill rate | Baseline | +30-50% |
| P&L per fill | Baseline | -40% (calm regime) |
| **Net P&L** | Baseline | **+15-25%** (via volume) |

---

## 2. Requirements

### 2.1 Functional Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| FR1 | Detect market regime (Calm/Normal/Volatile/Cascade) in real-time | P0 |
| FR2 | Predict toxicity score before each quote cycle | P0 |
| FR3 | Compute dynamic spread floor based on regime and toxicity | P0 |
| FR4 | Apply conditional gate for tight quote activation | P0 |
| FR5 | Export regime/toxicity metrics to Prometheus | P1 |
| FR6 | Support online learning for toxicity model | P2 |

### 2.2 Non-Functional Requirements

| ID | Requirement | Target |
|----|-------------|--------|
| NFR1 | Regime detection latency | < 1ms |
| NFR2 | Toxicity prediction latency | < 5ms |
| NFR3 | False positive rate (calm when actually volatile) | < 10% |
| NFR4 | Default behavior on uncertainty | Widen to Normal |
| NFR5 | Feature flag for production rollout | Required |

---

## 3. Architecture

### 3.1 Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      ParameterEstimator                          │
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐  │
│  │  Volatility  │  │    Kappa     │  │      Microprice        │  │
│  │  Estimator   │  │  Estimator   │  │      Estimator         │  │
│  └──────┬───────┘  └──────┬───────┘  └───────────┬────────────┘  │
│         │                 │                       │               │
│         └────────────┬────┴───────────────────────┘               │
│                      ▼                                            │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                MarketRegimeDetector (NEW)                  │  │
│  │  • Threshold-based classification                          │  │
│  │  • Inputs: σ, jump_ratio, book_imbalance, trade_rate       │  │
│  │  • Output: MarketRegime + confidence                       │  │
│  └────────────────────────┬───────────────────────────────────┘  │
│                           │                                       │
│                           ▼                                       │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                ToxicityPredictor (NEW)                     │  │
│  │  • Feature-based scoring                                   │  │
│  │  • Inputs: regime, imbalances, hour, recent_fills          │  │
│  │  • Output: toxicity_score + confidence + factor            │  │
│  └────────────────────────┬───────────────────────────────────┘  │
│                           │                                       │
│                           ▼                                       │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                DynamicSpreadFloor (NEW)                    │  │
│  │  • floor = f(regime, toxicity, inventory, hour)            │  │
│  │  • Output: min_spread_floor_bps                            │  │
│  └────────────────────────────────────────────────────────────┘  │
└───────────────────────────────┬───────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                         MarketParams                             │
│  + regime: MarketRegime                                          │
│  + regime_confidence: f64                                        │
│  + toxicity_score: f64                                           │
│  + dynamic_floor_bps: f64                                        │
│  + can_quote_tight: bool                                         │
└───────────────────────────────┬───────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       LadderStrategy                             │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                  TightSpreadGate (NEW)                     │  │
│  │  can_tight = ALL(                                          │  │
│  │    regime == Calm,                                         │  │
│  │    toxicity < 0.1,                                         │  │
│  │    hour NOT IN toxic_hours,                                │  │
│  │    |inventory| < 30% max,                                  │  │
│  │    confidence > threshold                                  │  │
│  │  )                                                         │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                   │
│  Use dynamic_floor_bps instead of static min_spread_floor        │
│  Apply regime.gamma_multiplier() to effective_gamma              │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 File Structure

```
src/market_maker/
├── estimator/
│   ├── mod.rs                    # Add regime, toxicity exports
│   ├── regime.rs                 # NEW: MarketRegimeDetector
│   └── toxicity.rs               # NEW: ToxicityPredictor
├── strategy/
│   ├── mod.rs                    # Add tight_spread exports
│   ├── dynamic_floor.rs          # NEW: DynamicSpreadFloor
│   ├── tight_gate.rs             # NEW: TightSpreadGate
│   ├── market_params.rs          # MODIFY: Add regime, toxicity fields
│   └── risk_config.rs            # MODIFY: Add TightSpreadConfig
└── quoting/ladder/
    └── depth_generator.rs        # MODIFY: Use dynamic floor
```

---

## 4. Interface Specifications

### 4.1 MarketRegime Enum

```rust
/// Market regime classification for spread adjustment
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MarketRegime {
    /// Low volatility, stable book, minimal jumps
    /// Safe for tight spreads (3 bps floor)
    Calm,

    /// Baseline market conditions
    /// Standard GLFT spreads (5 bps floor)
    Normal,

    /// High volatility or jump activity
    /// Wide spreads required (8 bps floor)
    Volatile,

    /// Liquidation cascade detected
    /// Pull all quotes
    Cascade,
}

impl MarketRegime {
    /// Get the minimum spread floor for this regime
    pub fn spread_floor_bps(&self, config: &TightSpreadConfig) -> f64 {
        match self {
            Self::Calm => config.calm_floor_bps,
            Self::Normal => config.normal_floor_bps,
            Self::Volatile => config.volatile_floor_bps,
            Self::Cascade => f64::INFINITY,
        }
    }

    /// Get the gamma multiplier for this regime
    pub fn gamma_multiplier(&self) -> f64 {
        match self {
            Self::Calm => 0.5,
            Self::Normal => 1.0,
            Self::Volatile => 1.5,
            Self::Cascade => 5.0,
        }
    }

    /// Check if this regime allows tight spreads
    pub fn allows_tight(&self) -> bool {
        matches!(self, Self::Calm)
    }
}
```

### 4.2 RegimeDetector Trait

```rust
/// Trait for market regime detection implementations
pub trait RegimeDetector: Send + Sync {
    /// Detect current market regime from inputs
    fn detect(&self, inputs: &RegimeInputs) -> RegimeOutput;

    /// Update internal state (for online learning variants)
    fn update(&mut self, _observation: &RegimeObservation) {}
}

/// Inputs for regime detection
#[derive(Debug, Clone)]
pub struct RegimeInputs {
    /// Current volatility estimate (per-second)
    pub sigma: f64,
    /// Baseline volatility for comparison
    pub sigma_baseline: f64,
    /// Jump ratio (RV/BV) - values > 1.5 indicate jump activity
    pub jump_ratio: f64,
    /// Order book imbalance (-1 to 1)
    pub book_imbalance: f64,
    /// Current trade rate (trades per second)
    pub trade_rate: f64,
    /// Baseline trade rate for comparison
    pub trade_rate_baseline: f64,
    /// Liquidation cascade severity (0 to 1+)
    pub cascade_severity: f64,
}

/// Output from regime detection
#[derive(Debug, Clone)]
pub struct RegimeOutput {
    /// Detected regime
    pub regime: MarketRegime,
    /// Confidence in classification (0.0 to 1.0)
    pub confidence: f64,
    /// Estimated probability of regime change in next 10 seconds
    pub transition_prob: f64,
}
```

### 4.3 ToxicityPredictor Trait

```rust
/// Trait for toxicity prediction implementations
pub trait ToxicityPredictor: Send + Sync {
    /// Predict toxicity for the current market state
    fn predict(&self, inputs: &ToxicityInputs) -> ToxicityOutput;

    /// Update model with fill observation (online learning)
    fn update(&mut self, fill: &FillObservation);
}

/// Inputs for toxicity prediction
#[derive(Debug, Clone)]
pub struct ToxicityInputs {
    /// Current market regime
    pub regime: MarketRegime,
    /// Order book imbalance (-1 to 1)
    pub book_imbalance: f64,
    /// Trade flow imbalance (-1 to 1)
    pub flow_imbalance: f64,
    /// Current hour in UTC
    pub hour_utc: u32,
    /// Recent fill sizes (USD notional)
    pub recent_fill_sizes: Vec<f64>,
    /// Time since last fill in milliseconds
    pub time_since_last_fill_ms: u64,
}

/// Output from toxicity prediction
#[derive(Debug, Clone)]
pub struct ToxicityOutput {
    /// Toxicity score (0.0 = safe, 1.0 = very toxic)
    pub toxicity_score: f64,
    /// Confidence in prediction (0.0 to 1.0)
    pub confidence: f64,
    /// Dominant factor driving toxicity
    pub dominant_factor: ToxicityFactor,
}

/// Factors that can drive toxicity
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToxicityFactor {
    BookImbalance,
    FlowImbalance,
    ToxicHour,
    LargeTrades,
    HighVolatility,
    Unknown,
}
```

### 4.4 TightSpreadConfig

```rust
/// Configuration for tight spread quoting system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TightSpreadConfig {
    // === Feature Flag ===
    /// Enable tight spread quoting (disabled by default for safety)
    pub enabled: bool,

    // === Regime-Based Spread Floors ===
    /// Spread floor during calm regime (bps)
    pub calm_floor_bps: f64,
    /// Spread floor during normal regime (bps)
    pub normal_floor_bps: f64,
    /// Spread floor during volatile regime (bps)
    pub volatile_floor_bps: f64,

    // === Regime Detection Thresholds ===
    /// Volatility ratio below which regime is Calm
    pub sigma_calm_ratio: f64,
    /// Volatility ratio above which regime is Volatile
    pub sigma_volatile_ratio: f64,
    /// Jump ratio above which regime is Volatile
    pub jump_ratio_threshold: f64,
    /// Cascade severity above which regime is Cascade
    pub cascade_severity_threshold: f64,

    // === Toxicity Thresholds ===
    /// Toxicity below which tight quoting is allowed
    pub toxicity_tight_threshold: f64,
    /// Toxicity above which spreads widen to volatile floor
    pub toxicity_wide_threshold: f64,

    // === Toxic Hours ===
    /// Hours in UTC when tight quoting is disabled
    pub toxic_hours: Vec<u32>,

    // === Inventory Constraint ===
    /// Maximum inventory utilization for tight quoting
    pub max_inventory_for_tight: f64,

    // === Confidence Requirements ===
    /// Minimum confidence for regime detection
    pub min_regime_confidence: f64,
    /// Minimum confidence for toxicity prediction
    pub min_toxicity_confidence: f64,

    // === Fail-Safe ===
    /// Default regime when confidence is low
    pub default_on_uncertainty: MarketRegime,
}

impl Default for TightSpreadConfig {
    fn default() -> Self {
        Self {
            enabled: false,  // DISABLED by default

            calm_floor_bps: 3.0,
            normal_floor_bps: 5.0,
            volatile_floor_bps: 8.0,

            sigma_calm_ratio: 0.7,
            sigma_volatile_ratio: 2.0,
            jump_ratio_threshold: 1.5,
            cascade_severity_threshold: 0.5,

            toxicity_tight_threshold: 0.1,
            toxicity_wide_threshold: 0.3,

            toxic_hours: vec![6, 7, 14],

            max_inventory_for_tight: 0.3,

            min_regime_confidence: 0.7,
            min_toxicity_confidence: 0.6,

            default_on_uncertainty: MarketRegime::Normal,
        }
    }
}
```

---

## 5. Algorithms

### 5.1 Regime Detection (Threshold-Based)

```
ALGORITHM DetectRegime(inputs: RegimeInputs, config: TightSpreadConfig) -> RegimeOutput

INPUT:
  - sigma: current volatility
  - sigma_baseline: reference volatility
  - jump_ratio: RV/BV ratio
  - book_imbalance: order book skew
  - trade_rate: current trade frequency
  - cascade_severity: liquidation cascade score

OUTPUT:
  - regime: MarketRegime
  - confidence: float [0, 1]
  - transition_prob: float [0, 1]

PROCEDURE:
  // Step 1: Check cascade (highest priority)
  IF cascade_severity > config.cascade_severity_threshold:
    RETURN (Cascade, 0.95, 0.1)

  // Step 2: Calculate volatility ratio
  vol_ratio = sigma / sigma_baseline

  // Step 3: Score calm vs volatile indicators
  calm_score = 0.0
  volatile_score = 0.0

  // Volatility component (weight: 0.4)
  IF vol_ratio < config.sigma_calm_ratio:
    calm_score += 0.4
  ELIF vol_ratio > config.sigma_volatile_ratio:
    volatile_score += 0.4

  // Jump ratio component (weight: 0.3)
  IF jump_ratio < 1.2:
    calm_score += 0.3
  ELIF jump_ratio > config.jump_ratio_threshold:
    volatile_score += 0.3

  // Book stability component (weight: 0.15)
  IF abs(book_imbalance) < 0.3:
    calm_score += 0.15
  ELIF abs(book_imbalance) > 0.6:
    volatile_score += 0.15

  // Trade rate component (weight: 0.15)
  rate_ratio = trade_rate / trade_rate_baseline
  IF rate_ratio < 0.8:
    calm_score += 0.15
  ELIF rate_ratio > 2.0:
    volatile_score += 0.15

  // Step 4: Determine regime
  IF calm_score > 0.6:
    regime = Calm
    confidence = calm_score
  ELIF volatile_score > 0.5:
    regime = Volatile
    confidence = volatile_score
  ELSE:
    regime = Normal
    confidence = 1.0 - max(calm_score, volatile_score)

  // Step 5: Estimate transition probability
  transition_prob = 0.1 + 0.3 * (1.0 - confidence)

  RETURN (regime, confidence, transition_prob)
```

### 5.2 Toxicity Prediction (Feature-Based)

```
ALGORITHM PredictToxicity(inputs: ToxicityInputs, config: TightSpreadConfig) -> ToxicityOutput

INPUT:
  - regime: current market regime
  - book_imbalance: order book skew
  - flow_imbalance: trade flow skew
  - hour_utc: current hour
  - recent_fill_sizes: list of recent fill sizes
  - time_since_last_fill_ms: time since last fill

OUTPUT:
  - toxicity_score: float [0, 1]
  - confidence: float [0, 1]
  - dominant_factor: ToxicityFactor

PROCEDURE:
  score = 0.0
  max_factor = (Unknown, 0.0)

  // Book imbalance component (0-0.25)
  book_tox = abs(book_imbalance) * 0.25
  score += book_tox
  IF book_tox > max_factor.value:
    max_factor = (BookImbalance, book_tox)

  // Flow imbalance component (0-0.25)
  flow_tox = abs(flow_imbalance) * 0.25
  score += flow_tox
  IF flow_tox > max_factor.value:
    max_factor = (FlowImbalance, flow_tox)

  // Toxic hour component (0 or 0.2)
  IF hour_utc IN config.toxic_hours:
    score += 0.2
    IF 0.2 > max_factor.value:
      max_factor = (ToxicHour, 0.2)

  // Large trade component (0-0.2)
  IF recent_fill_sizes.length > 0:
    avg_size = mean(recent_fill_sizes)
    IF avg_size > 2.0 * historical_median_size:
      score += 0.2
      IF 0.2 > max_factor.value:
        max_factor = (LargeTrades, 0.2)

  // Regime component (0-0.1)
  IF regime == Volatile:
    score += 0.1
    IF 0.1 > max_factor.value:
      max_factor = (HighVolatility, 0.1)

  // Normalize score
  score = min(score, 1.0)

  // Calculate confidence based on data availability
  confidence = 0.5 + 0.1 * min(recent_fill_sizes.length, 5)

  RETURN (score, confidence, max_factor.factor)
```

### 5.3 Dynamic Floor Calculation

```
ALGORITHM CalculateDynamicFloor(
  regime: MarketRegime,
  toxicity: ToxicityOutput,
  inventory_util: float,
  hour_utc: u32,
  config: TightSpreadConfig
) -> float

OUTPUT:
  - dynamic_floor_bps: minimum spread floor

PROCEDURE:
  // Start with regime-based floor
  base_floor = regime.spread_floor_bps(config)

  // If tight spread is disabled, return volatile floor always
  IF NOT config.enabled:
    RETURN config.volatile_floor_bps

  // Check toxicity
  IF toxicity.toxicity_score > config.toxicity_wide_threshold:
    RETURN config.volatile_floor_bps

  // Check inventory
  IF inventory_util > config.max_inventory_for_tight:
    RETURN max(base_floor, config.normal_floor_bps)

  // Check toxic hours (override regime)
  IF hour_utc IN config.toxic_hours:
    RETURN max(base_floor, config.normal_floor_bps)

  RETURN base_floor
```

### 5.4 Tight Spread Gate

```
ALGORITHM CanQuoteTight(
  regime: MarketRegime,
  regime_confidence: float,
  toxicity: ToxicityOutput,
  inventory_util: float,
  hour_utc: u32,
  config: TightSpreadConfig
) -> (bool, String)

OUTPUT:
  - can_tight: boolean
  - reason: explanation if false

PROCEDURE:
  // Check feature flag
  IF NOT config.enabled:
    RETURN (false, "tight_spread_disabled")

  // Check regime
  IF regime != Calm:
    RETURN (false, "regime_not_calm")

  // Check regime confidence
  IF regime_confidence < config.min_regime_confidence:
    RETURN (false, "low_regime_confidence")

  // Check toxicity
  IF toxicity.toxicity_score > config.toxicity_tight_threshold:
    RETURN (false, "high_toxicity")

  // Check toxicity confidence
  IF toxicity.confidence < config.min_toxicity_confidence:
    RETURN (false, "low_toxicity_confidence")

  // Check toxic hours
  IF hour_utc IN config.toxic_hours:
    RETURN (false, "toxic_hour")

  // Check inventory
  IF inventory_util > config.max_inventory_for_tight:
    RETURN (false, "high_inventory")

  RETURN (true, "all_checks_passed")
```

---

## 6. Metrics

### 6.1 Prometheus Metrics

```rust
// Regime metrics
mm_market_regime{regime="calm|normal|volatile|cascade"}: Gauge
mm_regime_confidence: Gauge
mm_regime_transition_prob: Gauge

// Toxicity metrics
mm_toxicity_score: Gauge
mm_toxicity_confidence: Gauge
mm_toxicity_dominant_factor{factor="book|flow|hour|size|vol|unknown"}: Gauge

// Floor metrics
mm_dynamic_floor_bps: Gauge
mm_tight_quote_active: Gauge  // 0 or 1
mm_tight_quote_blocked_reason{reason="..."}: Counter

// Performance metrics
mm_regime_detection_latency_us: Histogram
mm_toxicity_prediction_latency_us: Histogram

// Analysis metrics (for calibration)
mm_fills_by_regime{regime="..."}: Counter
mm_adverse_selection_by_regime{regime="..."}: Counter
mm_pnl_by_regime{regime="..."}: Counter
mm_regime_duration_seconds{regime="..."}: Histogram
```

### 6.2 Logging

```rust
// Regime change logging
debug!(
    old_regime = ?old_regime,
    new_regime = ?new_regime,
    confidence = %format!("{:.2}", confidence),
    vol_ratio = %format!("{:.2}", vol_ratio),
    jump_ratio = %format!("{:.2}", jump_ratio),
    "Market regime changed"
);

// Tight quote activation/deactivation
info!(
    can_tight = can_tight,
    reason = %reason,
    floor_bps = %format!("{:.1}", floor_bps),
    regime = ?regime,
    toxicity = %format!("{:.2}", toxicity_score),
    "[TightSpread] Quote mode changed"
);

// Fill observation for calibration
debug!(
    regime = ?regime,
    toxicity = %format!("{:.2}", toxicity),
    spread_bps = %format!("{:.1}", spread),
    adverse_selection_bps = %format!("{:.2}", as_bps),
    "[TightSpread] Fill observation"
);
```

---

## 7. Testing Strategy

### 7.1 Unit Tests

```rust
#[cfg(test)]
mod tests {
    // Regime detection
    #[test]
    fn test_regime_calm_conditions() {
        let inputs = RegimeInputs {
            sigma: 0.00005,          // Low vol
            sigma_baseline: 0.0001,
            jump_ratio: 1.0,         // No jumps
            book_imbalance: 0.1,     // Balanced
            trade_rate: 0.5,
            trade_rate_baseline: 1.0,
            cascade_severity: 0.0,
        };
        let output = detector.detect(&inputs);
        assert_eq!(output.regime, MarketRegime::Calm);
        assert!(output.confidence > 0.6);
    }

    #[test]
    fn test_regime_volatile_on_jumps() {
        let inputs = RegimeInputs {
            sigma: 0.0001,
            sigma_baseline: 0.0001,
            jump_ratio: 2.5,         // High jumps
            book_imbalance: 0.5,
            trade_rate: 3.0,
            trade_rate_baseline: 1.0,
            cascade_severity: 0.0,
        };
        let output = detector.detect(&inputs);
        assert_eq!(output.regime, MarketRegime::Volatile);
    }

    #[test]
    fn test_cascade_overrides_all() {
        let inputs = RegimeInputs {
            sigma: 0.00005,          // Low vol (would be calm)
            sigma_baseline: 0.0001,
            jump_ratio: 1.0,
            book_imbalance: 0.1,
            trade_rate: 0.5,
            trade_rate_baseline: 1.0,
            cascade_severity: 0.8,   // BUT cascade detected
        };
        let output = detector.detect(&inputs);
        assert_eq!(output.regime, MarketRegime::Cascade);
    }

    // Toxicity prediction
    #[test]
    fn test_toxicity_toxic_hour() {
        let inputs = ToxicityInputs {
            regime: MarketRegime::Calm,
            book_imbalance: 0.0,
            flow_imbalance: 0.0,
            hour_utc: 7,             // Toxic hour
            recent_fill_sizes: vec![],
            time_since_last_fill_ms: 1000,
        };
        let output = predictor.predict(&inputs);
        assert!(output.toxicity_score >= 0.2);
        assert_eq!(output.dominant_factor, ToxicityFactor::ToxicHour);
    }

    // Dynamic floor
    #[test]
    fn test_floor_respects_toxicity() {
        // Even in calm regime, high toxicity should widen floor
        let floor = calculate_dynamic_floor(
            MarketRegime::Calm,
            ToxicityOutput { toxicity_score: 0.4, .. },  // High toxicity
            0.1,  // Low inventory
            10,   // Safe hour
            &config,
        );
        assert_eq!(floor, config.volatile_floor_bps);
    }
}
```

### 7.2 Integration Tests

```rust
#[tokio::test]
async fn test_market_params_populated() {
    // Verify ParameterEstimator populates regime/toxicity fields
    let estimator = ParameterEstimator::new(config);
    // ... feed data ...
    let params = estimator.get_market_params();

    assert!(params.regime != MarketRegime::Cascade);  // Should have valid regime
    assert!(params.toxicity_score >= 0.0 && params.toxicity_score <= 1.0);
    assert!(params.dynamic_floor_bps > 0.0);
}

#[tokio::test]
async fn test_ladder_uses_dynamic_floor() {
    // Verify ladder generator respects dynamic floor
    let params = MarketParams {
        regime: MarketRegime::Calm,
        dynamic_floor_bps: 3.0,
        ..default_params()
    };

    let ladder = strategy.generate_ladder(&config, 0.0, 1.0, 0.1, &params);

    // Best bid/ask should be at least 3 bps from mid
    let best_bid_depth = (params.microprice - ladder.best_bid().unwrap()) / params.microprice * 10000.0;
    assert!(best_bid_depth >= 3.0);
}
```

### 7.3 Simulation Tests

```rust
#[test]
fn test_historical_replay() {
    // Replay historical data and compare static vs dynamic floor
    let historical_data = load_historical_data("trade_history.csv");

    let mut static_pnl = 0.0;
    let mut dynamic_pnl = 0.0;

    for tick in historical_data {
        // Simulate with static 8 bps floor
        static_pnl += simulate_fill(tick, 8.0);

        // Simulate with dynamic floor
        let regime = detector.detect(&tick);
        let floor = calculate_dynamic_floor(regime, ...);
        dynamic_pnl += simulate_fill(tick, floor);
    }

    println!("Static P&L: {}, Dynamic P&L: {}", static_pnl, dynamic_pnl);
    assert!(dynamic_pnl > static_pnl);  // Dynamic should outperform
}
```

---

## 8. Rollout Plan

### Phase 1: Metrics Only (1 week)
- Deploy regime detection and toxicity prediction
- Log outputs but DON'T use for spread adjustment
- Collect data for validation

### Phase 2: Shadow Mode (1 week)
- Calculate dynamic floor but log only
- Compare with actual spreads
- Measure what floor WOULD have been

### Phase 3: Gradual Activation (2 weeks)
- Enable with conservative thresholds
- `calm_floor_bps: 5.0` (not 3.0)
- Monitor adverse selection by regime

### Phase 4: Full Activation
- Lower `calm_floor_bps` to 3.0
- Enable online learning for toxicity
- Continuous monitoring and adjustment

---

## 9. Risk Mitigation

### 9.1 Fail-Safe Behaviors

| Failure Mode | Mitigation |
|--------------|------------|
| Low confidence in regime | Default to Normal (5 bps floor) |
| Regime detector crash | Use static 8 bps floor |
| Toxicity spike during calm | Widen immediately on next cycle |
| Inventory spike | Block tight quoting until normalized |

### 9.2 Kill Switch Integration

```rust
// In kill_switch.rs
if tight_spread_enabled && adverse_selection_by_regime["calm"] > 5.0 {
    warn!("[KillSwitch] High AS in calm regime - disabling tight spreads");
    config.tight_spread.enabled = false;
}
```

### 9.3 Alerting

```yaml
# Prometheus alerts
- alert: HighASInCalmRegime
  expr: mm_adverse_selection_by_regime{regime="calm"} > 3.0
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "High adverse selection during calm regime"

- alert: RegimeDetectionStale
  expr: time() - mm_last_regime_update_timestamp > 10
  for: 30s
  labels:
    severity: critical
  annotations:
    summary: "Regime detection not updating"
```

---

## 10. Dependencies

### 10.1 Internal Dependencies

| Module | Dependency | Type |
|--------|------------|------|
| MarketRegimeDetector | VolatilityEstimator | σ, jump_ratio |
| MarketRegimeDetector | LiquidationDetector | cascade_severity |
| ToxicityPredictor | MicropriceEstimator | book_imbalance, flow_imbalance |
| ToxicityPredictor | FillProcessor | recent_fill_sizes |
| LadderStrategy | DynamicSpreadFloor | min_floor calculation |

### 10.2 External Dependencies

None - uses only internal estimators.

---

## 11. Open Questions

1. **ML vs Threshold**: Should toxicity use ML model or stay with feature-based scoring?
   - Recommendation: Start with feature-based, add ML later with sufficient data

2. **Cross-Asset Signals**: Should BTC regime influence ETH spreads?
   - Recommendation: Phase 2 feature, not in initial design

3. **Latency Requirement**: Is 50ms update latency achievable?
   - Need to measure current quote cycle latency first

4. **Calibration Frequency**: How often should thresholds be recalibrated?
   - Recommendation: Weekly review, monthly updates

---

## Appendix A: Regime Detection Weights

| Component | Weight | Calm Threshold | Volatile Threshold |
|-----------|--------|----------------|-------------------|
| Volatility ratio | 0.40 | < 0.7 | > 2.0 |
| Jump ratio | 0.30 | < 1.2 | > 1.5 |
| Book imbalance | 0.15 | < 0.3 | > 0.6 |
| Trade rate ratio | 0.15 | < 0.8 | > 2.0 |

## Appendix B: Toxicity Scoring Weights

| Component | Max Score | Condition |
|-----------|-----------|-----------|
| Book imbalance | 0.25 | Linear with |imbalance| |
| Flow imbalance | 0.25 | Linear with |imbalance| |
| Toxic hour | 0.20 | Hour in [6, 7, 14] UTC |
| Large trades | 0.20 | Avg size > 2× median |
| High volatility | 0.10 | Regime == Volatile |
