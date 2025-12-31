# Design: Bayesian Time-Based Warmup Bypass

## Problem Statement

The market maker's parameter estimator requires 10 volume ticks and 5 L2 updates to warm up before quoting begins. In low-activity markets (e.g., testnet, off-hours), this warmup can take 10+ minutes, during which:

1. No orders are placed
2. Trading opportunities are missed
3. The system appears "stuck"

**Observed Issue (mm.log analysis):**
- Required: 10 volume ticks
- Achieved in 7 minutes: 5 volume ticks (tick=0 through tick=4)
- Result: `is_warmed_up() = false` → `update_quotes()` returns early → 0 orders placed

## Solution: Bayesian Time-Based Warmup Bypass

### Core Idea

Start quoting after a configurable timeout (default: 60 seconds) using Bayesian-blended parameters that combine:
- **Prior beliefs** (default_sigma, default_kappa from config)
- **Observations** (estimated values from volume buckets and L2 updates)

The blend is weighted by **confidence**, which grows naturally as data accumulates.

### Bayesian Framework

For volatility σ:
```
σ_posterior = (1 - confidence) × σ_prior + confidence × σ_observed

where:
  σ_prior = config.default_sigma = 0.0001 (0.01% per-second)
  σ_observed = sigma_clean() from bipower variation
  confidence = sigma_confidence() = 1 - exp(-tick_count / min_ticks)
```

Confidence curve:
| Tick Count | Confidence | Prior Weight | Observation Weight |
|------------|------------|--------------|-------------------|
| 0          | 0.00       | 100%         | 0%                |
| 1          | 0.10       | 90%          | 10%               |
| 5          | 0.39       | 61%          | 39%               |
| 10         | 0.63       | 37%          | 63%               |
| 20         | 0.86       | 14%          | 86%               |
| 30         | 0.95       | 5%           | 95%               |

### Risk Scaling During Low Confidence

When confidence is low, we increase γ (risk aversion) to widen spreads:

```
γ_effective = γ_base × uncertainty_gamma_mult

where:
  uncertainty_gamma_mult = 1 + (1 - confidence) × uncertainty_premium
  uncertainty_premium = 0.5 (default, configurable)
```

| Confidence | Gamma Multiplier (premium=0.5) |
|------------|-------------------------------|
| 0.0        | 1.50                          |
| 0.5        | 1.25                          |
| 1.0        | 1.00                          |

This ensures conservative quoting when parameters are uncertain.

---

## Implementation Design

### 1. Configuration Changes

**File: `src/market_maker/estimator/mod.rs`**

Add to `EstimatorConfig`:

```rust
pub struct EstimatorConfig {
    // ... existing fields ...

    // === Bayesian Warmup Bypass ===
    /// Time after which quoting starts with Bayesian priors (seconds).
    /// Set to 0.0 to disable time-based bypass and require full warmup.
    /// Default: 60.0 seconds
    pub warmup_timeout_secs: f64,

    /// Uncertainty premium for gamma scaling during low-confidence periods.
    /// gamma_mult = 1 + (1 - confidence) × uncertainty_gamma_premium
    /// Default: 0.5 (50% higher gamma when no observations)
    pub uncertainty_gamma_premium: f64,
}

impl Default for EstimatorConfig {
    fn default() -> Self {
        Self {
            // ... existing defaults ...

            warmup_timeout_secs: 60.0,
            uncertainty_gamma_premium: 0.5,
        }
    }
}
```

### 2. ParameterEstimator Changes

**File: `src/market_maker/estimator/parameter_estimator.rs`**

Add fields and methods:

```rust
use std::time::Instant;

pub struct ParameterEstimator {
    // ... existing fields ...

    /// Timestamp when estimator was created (for warmup timeout)
    created_at: Instant,
}

impl ParameterEstimator {
    pub fn new(config: EstimatorConfig) -> Self {
        Self {
            // ... existing initialization ...
            created_at: Instant::now(),
        }
    }

    // === NEW: Warmup Bypass Methods ===

    /// Check if quoting can begin (time-based bypass OR full warmup).
    ///
    /// Returns true if:
    /// 1. Full warmup complete (is_warmed_up()), OR
    /// 2. Warmup timeout elapsed AND we have at least 1 L2 update (for mid price)
    pub fn can_quote(&self) -> bool {
        if self.is_warmed_up() {
            return true;
        }

        // Time-based bypass: allow after timeout if we have basic data
        if self.config.warmup_timeout_secs > 0.0 {
            let elapsed = self.created_at.elapsed().as_secs_f64();
            let has_basic_data = self.market_kappa.update_count() >= 1;

            if elapsed >= self.config.warmup_timeout_secs && has_basic_data {
                return true;
            }
        }

        false
    }

    /// Get Bayesian-blended sigma (prior + observations).
    ///
    /// σ_posterior = (1 - conf) × σ_prior + conf × σ_observed
    pub fn bayesian_sigma(&self) -> f64 {
        let conf = self.sigma_confidence();
        let prior = self.config.default_sigma;
        let observed = self.sigma_clean();

        (1.0 - conf) * prior + conf * observed
    }

    /// Get Bayesian-blended kappa (prior + observations).
    ///
    /// κ_posterior = (1 - conf) × κ_prior + conf × κ_observed
    pub fn bayesian_kappa(&self) -> f64 {
        let conf = self.kappa_confidence();
        let prior = self.config.default_kappa;
        let observed = self.kappa();

        (1.0 - conf) * prior + conf * observed
    }

    /// Get kappa confidence (0.0 to 1.0) based on L2 update count.
    ///
    /// Similar to sigma_confidence but uses L2 updates instead of volume ticks.
    pub fn kappa_confidence(&self) -> f64 {
        let update_count = self.market_kappa.update_count();
        let min_updates = self.config.min_l2_updates.max(1);

        let ratio = update_count as f64 / min_updates as f64;
        1.0 - (-ratio).exp()
    }

    /// Get uncertainty gamma multiplier for conservative quoting.
    ///
    /// Returns a value >= 1.0 that scales gamma higher when confidence is low.
    /// gamma_mult = 1 + (1 - min_confidence) × uncertainty_premium
    pub fn uncertainty_gamma_mult(&self) -> f64 {
        let conf = self.sigma_confidence().min(self.kappa_confidence());
        1.0 + (1.0 - conf) * self.config.uncertainty_gamma_premium
    }

    /// Time elapsed since estimator creation.
    pub fn warmup_elapsed_secs(&self) -> f64 {
        self.created_at.elapsed().as_secs_f64()
    }

    /// Check if we're in bypass mode (quoting allowed but not fully warmed up).
    pub fn is_in_bypass_mode(&self) -> bool {
        self.can_quote() && !self.is_warmed_up()
    }
}
```

### 3. MarketMaker Changes

**File: `src/market_maker/mod.rs`**

Change `update_quotes()`:

```rust
async fn update_quotes(&mut self) -> Result<()> {
    // CHANGED: Use can_quote() instead of is_warmed_up()
    // Allows time-based bypass with Bayesian priors
    if !self.estimator.can_quote() {
        return Ok(());
    }

    // Log when entering bypass mode
    if self.estimator.is_in_bypass_mode() {
        debug!(
            elapsed_secs = %format!("{:.1}", self.estimator.warmup_elapsed_secs()),
            sigma_conf = %format!("{:.2}", self.estimator.sigma_confidence()),
            kappa_conf = %format!("{:.2}", self.estimator.kappa_confidence()),
            gamma_mult = %format!("{:.2}", self.estimator.uncertainty_gamma_mult()),
            "Quoting in bypass mode (warmup timeout reached)"
        );
    }

    // ... rest of method unchanged ...
}
```

### 4. ParameterAggregator Changes

**File: `src/market_maker/strategy/params.rs`**

Update `build()` to use Bayesian-blended values:

```rust
impl ParameterAggregator {
    pub fn build(sources: &ParameterSources) -> MarketParams {
        let est = sources.estimator;

        // Use Bayesian-blended values for robustness during low-confidence periods
        let sigma = est.bayesian_sigma();
        let kappa = est.bayesian_kappa();

        MarketParams {
            // === Volatility (use Bayesian blend) ===
            sigma,
            sigma_total: est.sigma_total(), // Keep raw for regime detection
            sigma_effective: est.bayesian_sigma(), // Use blended for quoting
            // ...

            // === Liquidity (use Bayesian blend) ===
            kappa,
            kappa_bid: est.kappa_bid(), // Keep directional for asymmetry
            kappa_ask: est.kappa_ask(),
            // ...

            // === NEW: Uncertainty fields ===
            uncertainty_gamma_mult: est.uncertainty_gamma_mult(),
            is_in_bypass_mode: est.is_in_bypass_mode(),
            // ...
        }
    }
}
```

### 5. RiskConfig Changes

**File: `src/market_maker/strategy/risk_config.rs`**

Apply uncertainty multiplier in `effective_gamma()`:

```rust
impl RiskConfig {
    pub fn effective_gamma(
        &self,
        market_params: &MarketParams,
        position: f64,
        max_position: f64,
    ) -> f64 {
        // ... existing gamma calculations ...

        let raw_gamma = cfg.gamma_base
            * vol_scalar
            * toxicity_scalar
            * inventory_scalar;

        // NEW: Apply uncertainty multiplier during low-confidence periods
        let uncertainty_adjusted = raw_gamma * market_params.uncertainty_gamma_mult;

        uncertainty_adjusted.clamp(cfg.gamma_min, cfg.gamma_max)
    }
}
```

---

## Logging and Observability

### New Log Messages

```
INFO  "Quoting enabled via warmup bypass after 60.0s (sigma_conf=0.39, kappa_conf=0.95)"
DEBUG "Bayesian blend: σ=0.000100 (prior=0.0001, obs=0.000098, conf=0.39)"
DEBUG "Uncertainty gamma multiplier: 1.30 (confidence=0.39, premium=0.50)"
```

### Prometheus Metrics

Add to `src/market_maker/infra/metrics.rs`:

```rust
mm_warmup_bypass_active{asset="BTC"} 1          // 1 = in bypass mode, 0 = fully warmed
mm_sigma_confidence{asset="BTC"} 0.39          // Confidence in sigma estimate
mm_kappa_confidence{asset="BTC"} 0.95          // Confidence in kappa estimate
mm_uncertainty_gamma_mult{asset="BTC"} 1.30    // Gamma multiplier from uncertainty
mm_warmup_elapsed_secs{asset="BTC"} 65.2       // Time since estimator creation
```

---

## Safety Considerations

### What This Changes

| Behavior | Before | After |
|----------|--------|-------|
| Quote start time | After 10 vol ticks (~10 min in low activity) | After 60s with any data |
| Initial sigma | Raw estimate (or nothing) | Bayesian blend with prior |
| Initial kappa | Raw estimate (or nothing) | Bayesian blend with prior |
| Risk during uncertainty | N/A (no quoting) | Gamma scaled 1.5x higher |

### What This Preserves

1. **is_warmed_up()** - Still exists for components that truly need full warmup
2. **Full parameter quality** - Converges to same behavior as observations accumulate
3. **Position limits** - Uses first-principles limits regardless of warmup state
4. **Kill switch** - Remains active and independent of warmup

### Edge Cases

1. **Zero L2 updates**: `can_quote()` requires at least 1 L2 update (for mid price)
2. **Timeout = 0**: Disables bypass, reverts to requiring full warmup
3. **Stale data**: Connection health monitor still triggers quote cancellation

---

## Testing Plan

### Unit Tests

1. `test_can_quote_full_warmup`: Returns true when fully warmed up
2. `test_can_quote_bypass_mode`: Returns true after timeout with basic data
3. `test_can_quote_no_data`: Returns false if no L2 updates even after timeout
4. `test_bayesian_sigma_prior_dominant`: Uses prior when confidence = 0
5. `test_bayesian_sigma_observation_dominant`: Uses observation when confidence = 1
6. `test_bayesian_sigma_blend`: Correctly blends at intermediate confidence
7. `test_uncertainty_gamma_mult`: Scales correctly with confidence
8. `test_is_in_bypass_mode`: Correctly identifies bypass vs full warmup

### Integration Tests

1. **Low-activity scenario**: Verify quoting starts within 60s on testnet
2. **High-activity scenario**: Verify normal warmup path still works
3. **Parameter convergence**: Verify Bayesian values converge to observations

---

## Configuration Recommendations

### Testnet / Low Activity
```rust
EstimatorConfig {
    warmup_timeout_secs: 30.0,        // Start faster
    uncertainty_gamma_premium: 0.5,   // 50% wider spreads initially
    min_volume_ticks: 10,             // Keep for "fully warmed" status
    min_l2_updates: 5,
    ..Default::default()
}
```

### Mainnet / High Activity
```rust
EstimatorConfig {
    warmup_timeout_secs: 60.0,        // Standard timeout
    uncertainty_gamma_premium: 0.3,   // 30% wider spreads initially
    min_volume_ticks: 10,
    min_l2_updates: 5,
    ..Default::default()
}
```

### Conservative (Full Warmup Required)
```rust
EstimatorConfig {
    warmup_timeout_secs: 0.0,         // Disable bypass
    min_volume_ticks: 10,
    min_l2_updates: 5,
    ..Default::default()
}
```

---

## Implementation Order

1. **Phase 1: Core bypass mechanism**
   - Add `created_at` field to ParameterEstimator
   - Add `can_quote()`, `is_in_bypass_mode()` methods
   - Add config fields `warmup_timeout_secs`
   - Change `update_quotes()` to use `can_quote()`

2. **Phase 2: Bayesian blending**
   - Add `kappa_confidence()` method
   - Add `bayesian_sigma()`, `bayesian_kappa()` methods
   - Update ParameterAggregator to use Bayesian values

3. **Phase 3: Uncertainty scaling**
   - Add `uncertainty_gamma_premium` to config
   - Add `uncertainty_gamma_mult()` method
   - Apply in RiskConfig.effective_gamma()

4. **Phase 4: Observability**
   - Add logging for bypass mode
   - Add Prometheus metrics
   - Update tests

---

## Summary

This design allows the market maker to start quoting quickly (within 60 seconds) in low-activity markets while maintaining safety through:

1. **Bayesian priors** - Sensible defaults when observations are sparse
2. **Confidence weighting** - Smooth transition from priors to observations
3. **Uncertainty scaling** - Wider spreads when parameters are uncertain
4. **Preserved safety** - Kill switch, position limits, and full warmup path unchanged

The implementation is backward-compatible: existing behavior is preserved for high-activity markets that warm up before the timeout.
