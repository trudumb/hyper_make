# Bayesian Estimator V2 - Comprehensive Design Document
**Date**: 2026-01-04
**Status**: Design Complete, Ready for Implementation

---

## Executive Summary

This document specifies a complete refactoring of the market maker's parameter estimation system to fix 14 identified statistical issues. The design includes:

1. **Corrected Bayesian Kappa** - Fix volume-weighting that breaks conjugacy
2. **Hierarchical Kappa Model** - Proper relationship between κ_market and κ_own
3. **Soft Jump Classification** - P(jump) ∈ [0,1] instead of binary
4. **Tick-Based EWMA** - Half-life in ticks, not seconds
5. **Parameter Covariance Tracking** - Joint (κ, σ) uncertainty

---

## Module 1: Corrected Bayesian Kappa

### The Bug (Current Code)
```rust
// kappa.rs:239-240 - WRONG
let posterior_alpha = self.prior_alpha + self.sum_volume;
let posterior_beta = self.prior_beta + self.sum_volume_weighted_distance;
```

### Why It's Wrong
For Gamma-Exponential conjugacy:
- Prior: κ ~ Gamma(α₀, β₀)
- Likelihood: X₁...Xₙ | κ ~ Exp(κ) → L(data|κ) = κⁿ exp(-κ Σxᵢ)
- Posterior: κ | data ~ Gamma(α₀ + n, β₀ + Σxᵢ)

The `n` must be COUNT of observations, not sum of volumes!

### Corrected Design
```rust
pub struct ConjugateBayesianKappa {
    prior_alpha: f64,  // Shape prior (α₀)
    prior_beta: f64,   // Rate prior (β₀)
    
    // Core sufficient statistics - UNWEIGHTED
    observation_count: usize,     // n = count of observations
    sum_distances: f64,           // Σδᵢ = unweighted sum
    sum_sq_distances: f64,        // Σδᵢ² for variance
    
    // Rolling window
    window: VecDeque<(u64, f64)>, // (timestamp, distance)
    window_ms: u64,
    
    // Cached posterior
    posterior_alpha: f64,
    posterior_beta: f64,
    posterior_mean: f64,
    posterior_var: f64,
    
    // Separate concern
    tail_detector: HeavyTailDetector,
}

impl ConjugateBayesianKappa {
    fn update_posterior(&mut self) {
        // CORRECT conjugate update
        self.posterior_alpha = self.prior_alpha + self.observation_count as f64;
        self.posterior_beta = self.prior_beta + self.sum_distances;
        self.posterior_mean = self.posterior_alpha / self.posterior_beta;
        self.posterior_var = self.posterior_alpha / (self.posterior_beta.powi(2));
    }
    
    fn credible_interval(&self, level: f64) -> (f64, f64) {
        // Use Gamma quantile function
        use statrs::distribution::{Gamma, ContinuousCDF};
        let gamma = Gamma::new(self.posterior_alpha, self.posterior_beta).unwrap();
        (gamma.inverse_cdf((1.0 - level) / 2.0),
         gamma.inverse_cdf((1.0 + level) / 2.0))
    }
}

pub struct HeavyTailDetector {
    cv_ewma: f64,
    cv_threshold: f64,  // 1.2 default
    is_heavy_tailed: bool,
}
```

---

## Module 2: Hierarchical Kappa (`hierarchical_kappa.rs`)

### The Bug (Current Code)
```rust
// parameter_estimator.rs:439 - WRONG
own_conf * own + (1.0 - own_conf) * market
```
Linear blending of different parameters!

### Corrected Design
```rust
pub struct HierarchicalKappa {
    // Market-level kappa from L2 book
    market_kappa: f64,
    market_kappa_conf: f64,
    
    // Own-fill kappa (hierarchical prior)
    prior_alpha: f64,  // Concentration parameter
    observation_count: usize,
    sum_distances: f64,
    
    // Adverse selection factor
    as_factor: f64,  // φ(AS) = exp(-c × AS)
    
    // Posterior
    posterior_alpha: f64,
    posterior_beta: f64,
}

impl HierarchicalKappa {
    fn update_from_fill(&mut self, distance: f64, timestamp_ms: u64) {
        self.observation_count += 1;
        self.sum_distances += distance;
        self.update_posterior();
    }
    
    fn update_market_kappa(&mut self, kappa: f64, conf: f64) {
        self.market_kappa = kappa;
        self.market_kappa_conf = conf;
        self.update_posterior();
    }
    
    fn update_posterior(&mut self) {
        // Hierarchical prior: scale β by market kappa
        let hierarchical_beta = self.prior_alpha / self.market_kappa;
        self.posterior_alpha = self.prior_alpha + self.observation_count as f64;
        self.posterior_beta = hierarchical_beta + self.sum_distances;
    }
    
    fn effective_kappa(&self) -> f64 {
        (self.posterior_alpha / self.posterior_beta) * self.as_factor
    }
}
```

---

## Module 3: Soft Jump Classification (`soft_jump.rs`)

### The Bug (Current Code)
```rust
// jump.rs:88-90 - WRONG
pub fn is_jump(&self, log_return: f64, sigma_clean: f64) -> bool {
    log_return.abs() > self.config.jump_threshold_sigmas * sigma_clean
}
```
Binary decision creates discontinuity!

### Corrected Design: Mixture Model
```rust
pub struct SoftJumpClassifier {
    // Prior jump probability (learned)
    pi: f64,
    
    // Diffusion component
    sigma_diffusion: f64,
    
    // Jump component
    sigma_jump: f64,  // 3-5× sigma_diffusion
    
    // EWMA for updating
    pi_alpha: f64,
    ewma_jump_var: f64,
}

impl SoftJumpClassifier {
    fn jump_probability(&self, log_return: f64) -> f64 {
        let ll_diff = gaussian_log_pdf(log_return, 0.0, self.sigma_diffusion);
        let ll_jump = gaussian_log_pdf(log_return, 0.0, self.sigma_jump);
        
        let log_post_jump = self.pi.ln() + ll_jump;
        let log_post_diff = (1.0 - self.pi).ln() + ll_diff;
        
        // Softmax
        let max_log = log_post_jump.max(log_post_diff);
        let exp_jump = (log_post_jump - max_log).exp();
        let exp_diff = (log_post_diff - max_log).exp();
        
        exp_jump / (exp_jump + exp_diff)
    }
    
    fn toxicity_score(&self) -> f64 {
        self.pi  // Rolling average of P(jump)
    }
    
    fn update(&mut self, log_return: f64) {
        let p_jump = self.jump_probability(log_return);
        self.pi = self.pi_alpha * p_jump + (1.0 - self.pi_alpha) * self.pi;
        
        if p_jump > 0.5 {
            let sq = log_return * log_return;
            self.ewma_jump_var = 0.1 * sq + 0.9 * self.ewma_jump_var;
            self.sigma_jump = self.ewma_jump_var.sqrt();
        }
    }
}
```

---

## Module 4: Tick-Based EWMA (`tick_ewma.rs`)

### The Bug (Current Code)
```rust
// volatility.rs:183
/// Assumes Δt ≈ 1 second between observations.
```
Volume clock produces irregular Δt!

### Corrected Design
```rust
pub struct TickEWMA {
    value: f64,
    half_life_ticks: f64,
    tick_count: usize,
    alpha: f64,
}

impl TickEWMA {
    fn new(half_life_ticks: f64, initial: f64) -> Self {
        let alpha = 1.0 - 2.0_f64.powf(-1.0 / half_life_ticks);
        Self { value: initial, half_life_ticks, tick_count: 0, alpha }
    }
    
    fn update(&mut self, observation: f64) {
        self.value = self.alpha * observation + (1.0 - self.alpha) * self.value;
        self.tick_count += 1;
    }
}

pub struct HybridEWMA {
    value: f64,
    tick_alpha: f64,
    time_half_life_ms: u64,
    last_update_ms: u64,
}

impl HybridEWMA {
    fn update(&mut self, observation: f64, timestamp_ms: u64) {
        let elapsed = timestamp_ms.saturating_sub(self.last_update_ms);
        let time_decay = 2.0_f64.powf(-(elapsed as f64) / (self.time_half_life_ms as f64));
        let decayed = self.value * time_decay;
        self.value = self.tick_alpha * observation + (1.0 - self.tick_alpha) * decayed;
        self.last_update_ms = timestamp_ms;
    }
}
```

---

## Module 5: Parameter Covariance (`covariance.rs`)

### New Capability
Track joint (κ, σ) uncertainty for proper spread uncertainty.

```rust
pub struct ParameterCovariance {
    mean_kappa: f64,
    mean_sigma: f64,
    mean_kappa_sq: f64,
    mean_sigma_sq: f64,
    mean_kappa_sigma: f64,
    alpha: f64,
    observation_count: usize,
}

impl ParameterCovariance {
    fn update(&mut self, kappa: f64, sigma: f64) {
        self.mean_kappa = self.alpha * kappa + (1.0 - self.alpha) * self.mean_kappa;
        self.mean_sigma = self.alpha * sigma + (1.0 - self.alpha) * self.mean_sigma;
        self.mean_kappa_sq = self.alpha * kappa * kappa + (1.0 - self.alpha) * self.mean_kappa_sq;
        self.mean_sigma_sq = self.alpha * sigma * sigma + (1.0 - self.alpha) * self.mean_sigma_sq;
        self.mean_kappa_sigma = self.alpha * kappa * sigma + (1.0 - self.alpha) * self.mean_kappa_sigma;
        self.observation_count += 1;
    }
    
    fn covariance(&self) -> f64 {
        self.mean_kappa_sigma - self.mean_kappa * self.mean_sigma
    }
    
    fn correlation(&self) -> f64 {
        let var_k = self.mean_kappa_sq - self.mean_kappa.powi(2);
        let var_s = self.mean_sigma_sq - self.mean_sigma.powi(2);
        self.covariance() / (var_k.sqrt() * var_s.sqrt()).max(1e-12)
    }
}
```

---

## Module 6: Unified Estimator (`unified_estimator.rs`)

Integrates all V2 components with feature flags.

```rust
pub struct UnifiedParameterEstimator {
    volume_clock: VolumeClock,
    bipower: MultiScaleBipowerEstimator,
    stochastic_vol: StochasticVolParams,
    market_kappa: MarketKappaFromL2,
    own_kappa: HierarchicalKappa,
    jump_classifier: SoftJumpClassifier,
    param_covariance: ParameterCovariance,
    microprice: MicropriceEstimator,
}
```

---

## Feature Flags (`config.rs`)

```rust
pub struct EstimatorVersion {
    pub use_v2_kappa: bool,
    pub use_hierarchical_kappa: bool,
    pub use_soft_jumps: bool,
    pub use_tick_ewma: bool,
    pub use_param_covariance: bool,
}
```

---

## Extended MarketParams

```rust
pub struct MarketParams {
    // Existing
    pub sigma: f64,
    pub sigma_effective: f64,
    pub kappa: f64,
    pub microprice: f64,
    pub jump_ratio: f64,
    
    // NEW: uncertainty quantification
    pub kappa_uncertainty: f64,
    pub toxicity_score: f64,
    pub param_correlation: f64,
    pub kappa_95_lower: f64,
    pub kappa_95_upper: f64,
}
```

---

## Implementation Order

### Week 1: Core Components
- Day 1-2: `kappa_v2.rs`, `tick_ewma.rs`
- Day 3-4: `soft_jump.rs`
- Day 5: `covariance.rs`

### Week 2: Integration
- Day 1-2: `hierarchical_kappa.rs`
- Day 3-4: `unified_estimator.rs`
- Day 5: `mod.rs` updates, feature flags

### Week 3: Wiring & Testing
- Day 1-2: `config.rs`, `market_params.rs`
- Day 3-4: `mod.rs` main loop
- Day 5: `metrics.rs`, shadow logging

### Week 4: Validation
- Unit tests, testnet, shadow mode, gradual rollout

---

## Implementation Status

**V2 Refactoring Complete (2026-01-04)**

All V2 components are now the permanent implementation:
- ✅ HierarchicalKappa - Always active, no feature flag
- ✅ SoftJumpClassifier - Always active, no feature flag  
- ✅ ParameterCovariance - Always active, no feature flag
- ✅ Prometheus metrics - Always exported

Feature flags removed from:
- `on_trade()` - soft_jump + param_covariance always update
- `on_own_fill()` - hierarchical_kappa always records fills
- `on_l2_book()` - hierarchical_kappa always gets prior
- `kappa_v2_aware()` - always uses hierarchical blending
- `new()` - default changed to `all_enabled()`

See `session_2026-01-04_v2_refactoring_complete` for details.

---

## Files Summary

### New Files (6)
- `src/market_maker/estimator/kappa_v2.rs`
- `src/market_maker/estimator/hierarchical_kappa.rs`
- `src/market_maker/estimator/soft_jump.rs`
- `src/market_maker/estimator/tick_ewma.rs`
- `src/market_maker/estimator/covariance.rs`
- `src/market_maker/estimator/unified_estimator.rs`

### Modified Files (6)
- `src/market_maker/estimator/mod.rs`
- `src/market_maker/estimator/volatility.rs`
- `src/market_maker/strategy/market_params.rs`
- `src/market_maker/config.rs`
- `src/market_maker/mod.rs`
- `src/market_maker/infra/metrics.rs`
