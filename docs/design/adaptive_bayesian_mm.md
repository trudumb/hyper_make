# Adaptive Bayesian Market Maker (ABMM) - Implementation Design

## Executive Summary

This document specifies the mathematical foundations and implementation plan for an Adaptive Bayesian Market Maker system that dynamically tunes quoting parameters for competitive spreads while preserving stochastic predictive power.

**Core Innovation**: Replace static/multiplicative parameter tuning with online Bayesian learning that adapts to observed fill outcomes, adverse selection, and market microstructure.

---

## Part 1: Mathematical Foundations

### 1.1 The Optimal Spread Problem

The market maker's objective is to maximize expected profit:

```
max_δ E[PnL] = E[spread_capture × P(fill)] - E[adverse_selection | fill] - fees
```

Where:
- `δ` = half-spread (distance from fair price to quote)
- `P(fill | δ)` = probability of execution at spread δ
- `AS(δ)` = expected adverse price movement given fill

**Key Insight**: The optimal spread balances:
1. **Too tight** → high fill rate but negative edge (AS > spread)
2. **Too wide** → positive edge but zero fills (no profit)

### 1.2 GLFT Foundation (Guéant-Lehalle-Fernandez-Tapia)

The GLFT optimal spread under exponential fill intensity:

```
δ* = (1/γ) × ln(1 + γ/κ)
```

Where:
- `γ` = risk aversion (inventory penalty coefficient)
- `κ` = fill intensity decay (λ(δ) = A × exp(-κδ))

**Derivation**: From the HJB equation for optimal market making:
```
0 = max_δ { -γσ²q² + λ(δ)(δ - γσ²q) }
∂/∂δ: 0 = λ'(δ)(δ - γσ²q) + λ(δ)
```

For exponential intensity λ(δ) = A×exp(-κδ):
```
0 = -κλ(δ)(δ - γσ²q) + λ(δ)
δ* = 1/κ + γσ²q
```

The inventory-independent part gives: `δ_base = (1/γ)ln(1 + γ/κ)`

### 1.3 Problem with Current Implementation

The current system compounds multiple scalers:

```
γ_eff = γ_base × vol_scalar × tox_scalar × inv_scalar × regime_scalar × hawkes_scalar × time_scalar
```

With typical values: `0.3 × 1.5 × 1.3 × 1.2 × 1.5 × 1.2 × 2.0 = 2.52`

This causes δ to explode because GLFT spread scales roughly as `1/γ × ln(γ)`:
- γ = 0.3 → δ ≈ 3.3 × ln(1.0001) ≈ 0.0003 (3 bps)
- γ = 2.5 → δ ≈ 0.4 × ln(1.001) ≈ 0.0004 (4 bps) BUT with much wider floor

The real damage is in `min_spread_floor` and the compounded floor effects.

---

## Part 2: Component Specifications

### 2.1 Component 1: Learned Spread Floor

**Mathematical Model**:

The break-even spread satisfies:
```
δ_BE = f_maker + E[AS | fill] + ε_execution
```

Where:
- `f_maker` = maker fee rate (known constant, e.g., 0.00015)
- `E[AS | fill]` = expected adverse selection (learned)
- `ε_execution` = execution cost buffer

**Bayesian Model for Adverse Selection**:

Prior:
```
AS ~ Normal(μ_AS, σ_AS²)
μ_AS ~ Normal(μ_0, σ_0²)        # Prior mean: 0.0003 (3 bps)
σ_AS ~ Half-Normal(σ_prior)     # Prior std: 0.0005 (5 bps)
```

Posterior Update (after fill at time t):
```
AS_realized = (mid_{t+Δt} - fill_price) × direction
             where Δt = 1 second (measurement horizon)
             direction = +1 for buy fill, -1 for sell fill

# Conjugate update for Normal-Normal:
μ_AS_posterior = (σ_0² × AS_realized + σ_AS² × μ_0) / (σ_0² + σ_AS²)
σ_AS_posterior² = (σ_0² × σ_AS²) / (σ_0² + σ_AS²)
```

**Dynamic Floor Calculation**:
```rust
fn learned_spread_floor(&self) -> f64 {
    let as_mean = self.as_estimator.posterior_mean();
    let as_std = self.as_estimator.posterior_std();

    // Risk-adjusted floor: mean + k standard deviations
    let k = 1.5; // 1.5σ covers ~87% of AS outcomes

    let floor = self.maker_fee + as_mean.max(0.0) + k * as_std;

    // Hard minimum at tick size
    floor.max(self.tick_size_as_fraction)
}
```

**Struct Definition**:
```rust
pub struct LearnedSpreadFloor {
    /// Maker fee rate (constant)
    maker_fee: f64,

    /// Prior mean for AS (initialized from config)
    prior_mean: f64,

    /// Prior variance for AS
    prior_variance: f64,

    /// Posterior mean (updated online)
    posterior_mean: f64,

    /// Posterior variance (updated online)
    posterior_variance: f64,

    /// Number of observations
    n_observations: usize,

    /// Risk multiplier k for floor calculation
    risk_k: f64,

    /// Minimum tick size as fraction
    tick_size_fraction: f64,

    /// EWMA decay for non-stationarity (0.99 = 100 obs half-life)
    ewma_decay: f64,
}
```

---

### 2.2 Component 2: Log-Additive Gamma with Shrinkage

**Mathematical Model**:

Replace multiplicative scaling with log-additive:
```
log(γ_eff) = log(γ_base) + Σᵢ wᵢ × zᵢ
```

Where:
- `zᵢ` = standardized signal (mean 0, variance 1)
- `wᵢ` = learned weight for signal i

**Shrinkage Prior (Horseshoe)**:

```
wᵢ | λᵢ, τ ~ Normal(0, λᵢ² × τ²)
λᵢ ~ Half-Cauchy(0, 1)           # Local shrinkage
τ ~ Half-Cauchy(0, τ_0)          # Global shrinkage
```

**Intuition**:
- τ controls TOTAL adjustment magnitude (global)
- λᵢ allows individual signals to escape shrinkage if strongly predictive
- If signal i is noise, posterior λᵢ → 0, so wᵢ → 0
- If signal i predicts profitability, posterior λᵢ >> 0

**Simplified Implementation (Empirical Bayes)**:

For computational tractability, use empirical Bayes approximation:

```rust
pub struct ShrinkageGamma {
    /// Base log-gamma
    log_gamma_base: f64,

    /// Signal weights (learned)
    weights: Vec<f64>,

    /// Signal names for logging
    signal_names: Vec<&'static str>,

    /// Global shrinkage τ² (learned)
    tau_squared: f64,

    /// Local shrinkage λᵢ² (learned per signal)
    lambda_squared: Vec<f64>,

    /// Accumulated gradient for online learning
    weight_gradients: Vec<f64>,

    /// Learning rate
    learning_rate: f64,

    /// Minimum/maximum gamma bounds
    gamma_min: f64,
    gamma_max: f64,
}

impl ShrinkageGamma {
    /// Compute effective gamma from standardized signals
    pub fn effective_gamma(&self, signals: &[f64]) -> f64 {
        assert_eq!(signals.len(), self.weights.len());

        let log_adjustment: f64 = self.weights.iter()
            .zip(signals.iter())
            .map(|(w, z)| w * z)
            .sum();

        let gamma = (self.log_gamma_base + log_adjustment).exp();
        gamma.clamp(self.gamma_min, self.gamma_max)
    }

    /// Update weights based on PnL outcome
    ///
    /// If wider spread (higher gamma) would have been better → positive gradient
    /// If tighter spread would have been better → negative gradient
    pub fn update(&mut self, signals: &[f64], pnl_gradient: f64) {
        for i in 0..self.weights.len() {
            // Gradient of PnL w.r.t. weight
            let grad = pnl_gradient * signals[i];

            // Ridge penalty from shrinkage prior
            let shrinkage_penalty = self.weights[i] / (self.lambda_squared[i] * self.tau_squared);

            // Update weight
            self.weights[i] += self.learning_rate * (grad - shrinkage_penalty);
        }

        // Update global shrinkage τ based on weight magnitudes
        let weight_variance: f64 = self.weights.iter().map(|w| w * w).sum::<f64>()
            / self.weights.len() as f64;
        self.tau_squared = 0.99 * self.tau_squared + 0.01 * weight_variance;
    }
}
```

**Signal Standardization**:

```rust
pub struct SignalStandardizer {
    /// Running mean
    mean: f64,
    /// Running variance
    variance: f64,
    /// Observation count
    n: usize,
}

impl SignalStandardizer {
    pub fn standardize(&mut self, raw: f64) -> f64 {
        // Welford's online algorithm
        self.n += 1;
        let delta = raw - self.mean;
        self.mean += delta / self.n as f64;
        let delta2 = raw - self.mean;
        self.variance += delta * delta2;

        let std = if self.n > 1 {
            (self.variance / (self.n - 1) as f64).sqrt()
        } else {
            1.0
        };

        (raw - self.mean) / std.max(1e-9)
    }
}
```

---

### 2.3 Component 3: Blended Kappa Estimation

**Mathematical Model**:

Two κ sources with different properties:
1. `κ_book` = from L2 order book structure (fast, approximate)
2. `κ_own` = from our fill distances (slow, accurate)

**Blending Formula**:
```
κ_eff = (1 - w(n)) × κ_book + w(n) × κ_own
```

Where w(n) is a sigmoid blend weight:
```
w(n) = sigmoid((n - n_min) / scale)
     = 1 / (1 + exp(-(n - n_min) / scale))
```

Parameters:
- `n` = number of own fills observed
- `n_min` = minimum fills before blending starts (e.g., 10)
- `scale` = steepness of transition (e.g., 5)

**Book-Based Kappa**:

From L2 order book, estimate κ as the decay rate of cumulative depth:
```
Depth(δ) = Σ size_i for all levels with distance_i ≤ δ
ln(Depth(δ)) ≈ α - κ × δ
```

Linear regression on (δᵢ, ln(Depth(δᵢ))) gives κ estimate.

**Own-Fill Kappa**:

From Bayesian conjugate model (existing implementation):
```
κ | data ~ Gamma(α_post, β_post)
α_post = α_prior + n
β_post = β_prior + Σδᵢ
E[κ] = α_post / β_post
```

**Implementation**:

```rust
pub struct BlendedKappaEstimator {
    /// Book-based kappa estimator
    book_kappa: BookStructureKappa,

    /// Own-fill Bayesian kappa
    own_kappa: BayesianKappaEstimator,

    /// Number of own fills
    own_fill_count: usize,

    /// Blend parameters
    blend_min_fills: usize,  // n_min
    blend_scale: f64,        // scale

    /// Warmup conservatism factor (< 1.0 widens spread during warmup)
    warmup_factor: f64,
}

impl BlendedKappaEstimator {
    pub fn kappa(&self) -> f64 {
        let n = self.own_fill_count as f64;
        let n_min = self.blend_min_fills as f64;

        // Sigmoid blend weight
        let w = 1.0 / (1.0 + (-(n - n_min) / self.blend_scale).exp());

        let kappa_book = self.book_kappa.kappa();
        let kappa_own = self.own_kappa.kappa();

        let blended = (1.0 - w) * kappa_book + w * kappa_own;

        // Apply warmup conservatism if still in warmup
        if self.own_fill_count < self.blend_min_fills {
            blended * self.warmup_factor
        } else {
            blended
        }
    }

    pub fn on_own_fill(&mut self, distance: f64, size: f64, mid: f64, timestamp_ms: u64) {
        self.own_fill_count += 1;
        self.own_kappa.on_trade(timestamp_ms, mid * (1.0 + distance), size, mid);
    }

    pub fn on_l2_update(&mut self, bids: &[(f64, f64)], asks: &[(f64, f64)], mid: f64) {
        self.book_kappa.update(bids, asks, mid);
    }
}
```

---

### 2.4 Component 4: Fill Rate Controller

**Mathematical Model**:

Target a minimum fill rate ρ_target (fills per second).

**Fill Rate Model**:
```
ρ(δ) = ρ_0 × exp(-κ × δ)
```

Given current observed fill rate ρ_obs and spread δ_current:
```
ρ_0 = ρ_obs × exp(κ × δ_current)  # Inferred base rate
```

**Target Spread for Fill Rate**:
```
δ_target = (1/κ) × ln(ρ_0 / ρ_target)
```

**Bayesian Fill Rate Estimation**:

Model fills as Poisson process:
```
fills_in_Δt ~ Poisson(ρ × Δt)
ρ ~ Gamma(α, β)  # Conjugate prior

# Update after observing k fills in time Δt:
α_post = α + k
β_post = β + Δt
E[ρ] = α_post / β_post
```

**Implementation**:

```rust
pub struct FillRateController {
    /// Target fill rate (fills per second)
    target_fill_rate: f64,

    /// Gamma posterior shape
    alpha: f64,

    /// Gamma posterior rate
    beta: f64,

    /// Current kappa estimate (from BlendedKappaEstimator)
    current_kappa: f64,

    /// Ceiling multiplier (allow GLFT to exceed fill target by this factor)
    ceiling_mult: f64,

    /// Minimum observation time before controller activates (seconds)
    min_observation_time: f64,

    /// Total observation time
    observation_time: f64,

    /// EWMA decay for non-stationarity
    decay: f64,
}

impl FillRateController {
    pub fn new(target_fill_rate: f64) -> Self {
        Self {
            target_fill_rate,
            alpha: 1.0,           // Prior: 1 fill
            beta: 60.0,           // Prior: in 60 seconds (= 1 fill/min)
            current_kappa: 500.0, // Default
            ceiling_mult: 1.5,
            min_observation_time: 120.0, // 2 minutes warmup
            observation_time: 0.0,
            decay: 0.995,
        }
    }

    /// Update after observing fills
    pub fn update(&mut self, fills: usize, elapsed_secs: f64, kappa: f64) {
        self.observation_time += elapsed_secs;
        self.current_kappa = kappa;

        // EWMA decay for non-stationarity
        self.alpha = self.decay * self.alpha + fills as f64;
        self.beta = self.decay * self.beta + elapsed_secs;
    }

    /// Get the spread ceiling based on fill rate target
    pub fn spread_ceiling(&self) -> Option<f64> {
        if self.observation_time < self.min_observation_time {
            return None; // Not enough data
        }

        // Posterior mean fill rate
        let rho_observed = self.alpha / self.beta;

        // If we're already above target, no ceiling needed
        if rho_observed >= self.target_fill_rate {
            return None;
        }

        // Infer base rate at zero spread
        // ρ_0 = ρ_obs × exp(κ × δ_current)
        // But we don't know δ_current here, so use ρ_0 ≈ κ × ρ_obs / target
        // This is approximate but directionally correct
        let rho_0 = self.current_kappa * rho_observed;

        // Target spread: δ = (1/κ) × ln(ρ_0 / ρ_target)
        let delta_target = (rho_0 / self.target_fill_rate).ln() / self.current_kappa;

        Some(delta_target.max(0.0) * self.ceiling_mult)
    }

    /// Get observed fill rate
    pub fn observed_fill_rate(&self) -> f64 {
        self.alpha / self.beta
    }
}
```

---

### 2.5 Component 5: Integrated Spread Calculator

**Mathematical Specification**:

The final spread combines all components:

```
δ_final = max(δ_floor, min(δ_GLFT, δ_ceiling))
```

Where:
- `δ_floor` = learned floor from Component 1
- `δ_GLFT` = GLFT optimal spread with shrinkage gamma
- `δ_ceiling` = fill rate ceiling from Component 4 (or ∞ if inactive)

**Full GLFT with Modifications**:

```
δ_GLFT = (1/γ_eff) × ln(1 + γ_eff/κ_eff) + f_maker
```

Where:
- `γ_eff` = exp(log(γ_base) + Σ wᵢzᵢ) from Component 2
- `κ_eff` = blended kappa from Component 3
- `f_maker` = maker fee rate

**Implementation**:

```rust
pub struct AdaptiveSpreadCalculator {
    /// Learned spread floor
    floor: LearnedSpreadFloor,

    /// Shrinkage gamma
    gamma: ShrinkageGamma,

    /// Blended kappa
    kappa: BlendedKappaEstimator,

    /// Fill rate controller
    fill_controller: FillRateController,

    /// Maker fee rate
    maker_fee: f64,

    /// Signal standardizers
    standardizers: HashMap<&'static str, SignalStandardizer>,
}

impl AdaptiveSpreadCalculator {
    /// Calculate optimal half-spread
    pub fn half_spread(&self, market_params: &MarketParams) -> f64 {
        // 1. Get effective parameters
        let gamma_eff = self.effective_gamma(market_params);
        let kappa_eff = self.kappa.kappa();

        // 2. GLFT optimal spread
        let delta_glft = if gamma_eff > 1e-9 && kappa_eff > 1e-9 {
            (1.0 / gamma_eff) * (1.0 + gamma_eff / kappa_eff).ln() + self.maker_fee
        } else {
            self.floor.learned_spread_floor() // Fallback
        };

        // 3. Floor from learned AS
        let delta_floor = self.floor.learned_spread_floor();

        // 4. Ceiling from fill rate controller
        let delta_ceiling = self.fill_controller.spread_ceiling()
            .unwrap_or(f64::INFINITY);

        // 5. Combine: max(floor, min(glft, ceiling))
        delta_floor.max(delta_glft.min(delta_ceiling))
    }

    fn effective_gamma(&self, market_params: &MarketParams) -> f64 {
        // Standardize signals
        let signals = vec![
            self.standardize("vol_ratio", market_params.sigma / 0.0002),
            self.standardize("jump_ratio", market_params.jump_ratio),
            self.standardize("inventory", market_params.inventory_ratio),
            self.standardize("hawkes", market_params.hawkes_intensity),
        ];

        self.gamma.effective_gamma(&signals)
    }

    fn standardize(&self, name: &'static str, raw: f64) -> f64 {
        // Would use mutable standardizers in real impl
        // Simplified here for clarity
        raw // Placeholder
    }

    /// Update after fill outcome
    pub fn on_fill(&mut self, fill: &FillEvent, mid_after_1s: f64) {
        // 1. Update AS estimator
        let as_realized = (mid_after_1s - fill.price) * fill.direction_sign();
        self.floor.update_as(as_realized);

        // 2. Update kappa with own fill
        let distance = (fill.price - fill.mid_at_fill).abs() / fill.mid_at_fill;
        self.kappa.on_own_fill(distance, fill.size, fill.mid_at_fill, fill.timestamp_ms);

        // 3. Update fill rate controller
        self.fill_controller.update(1, fill.elapsed_since_last_fill, self.kappa.kappa());

        // 4. Update gamma weights based on PnL
        let signals = self.current_signals();
        let pnl_gradient = if fill.pnl > 0.0 { -1.0 } else { 1.0 }; // If profitable, don't widen
        self.gamma.update(&signals, pnl_gradient * 0.1);
    }

    /// Update when no fill (quote expired)
    pub fn on_no_fill(&mut self, elapsed_secs: f64) {
        // Update fill rate controller (0 fills)
        self.fill_controller.update(0, elapsed_secs, self.kappa.kappa());

        // If fill rate is below target, signal to tighten
        if self.fill_controller.observed_fill_rate() < self.fill_controller.target_fill_rate {
            let signals = self.current_signals();
            self.gamma.update(&signals, -0.01); // Small push toward tighter
        }
    }
}
```

---

## Part 3: Stochastic Module Integration

### 3.1 Separation of Concerns

**Principle**: Spread width and inventory skew are SEPARATE optimization problems.

| Concern | Controls | Inputs |
|---------|----------|--------|
| Spread Width | δ_bid, δ_ask | κ, γ, AS, fill rate |
| Inventory Skew | bid_skew, ask_skew | σ, q, T, HJB |
| Position Sizing | order_size | margin, Kelly, σ |

### 3.2 Inventory Skew (Unchanged from Current)

The Avellaneda-Stoikov skew remains:
```
skew = q × γ × σ² × T
```

This shifts quotes asymmetrically to manage inventory, but does NOT affect spread width.

### 3.3 Jump Detection → Quote Pulling (Not Spread Widening)

Instead of adding jump premium to spread, use binary decision:

```rust
fn should_pull_quotes(&self, market_params: &MarketParams) -> bool {
    // Pull quotes entirely during detected jumps
    // Don't try to quote through crashes/spikes

    let jump_probability = self.jump_estimator.jump_probability();
    let cascade_severity = market_params.cascade_severity;

    jump_probability > 0.8 || cascade_severity > 0.7
}
```

**Rationale**: During true jumps, ANY spread is wrong. Better to not quote than quote badly.

### 3.4 Volatility → Position Sizing (Not Spread)

Use σ for Kelly-optimal position sizing:

```
size_kelly = (edge / variance) = (spread - AS - fees) / σ²
size_final = min(size_kelly, margin_limit, inventory_limit)
```

---

## Part 4: Configuration & Defaults

### 4.1 New Configuration Structure

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveBayesianConfig {
    // === Learned Floor ===
    /// Prior mean for adverse selection (fraction, e.g., 0.0003 = 3 bps)
    pub as_prior_mean: f64,
    /// Prior std for adverse selection
    pub as_prior_std: f64,
    /// Risk multiplier k for floor = fees + E[AS] + k×σ_AS
    pub floor_risk_k: f64,
    /// Hard minimum floor (tick size as fraction)
    pub floor_absolute_min: f64,

    // === Shrinkage Gamma ===
    /// Base gamma (log scale)
    pub gamma_base: f64,
    /// Initial global shrinkage τ
    pub tau_initial: f64,
    /// Learning rate for weight updates
    pub gamma_learning_rate: f64,
    /// Gamma bounds
    pub gamma_min: f64,
    pub gamma_max: f64,

    // === Blended Kappa ===
    /// Prior mean for Bayesian kappa
    pub kappa_prior_mean: f64,
    /// Prior strength (effective sample size)
    pub kappa_prior_strength: f64,
    /// Minimum fills before blending starts
    pub kappa_blend_min_fills: usize,
    /// Blend sigmoid steepness
    pub kappa_blend_scale: f64,
    /// Warmup conservatism factor
    pub kappa_warmup_factor: f64,

    // === Fill Rate Controller ===
    /// Target fill rate (fills per second)
    pub target_fill_rate: f64,
    /// Ceiling multiplier
    pub fill_ceiling_mult: f64,
    /// Minimum observation time before activation
    pub fill_min_observation_secs: f64,

    // === Maker Fee ===
    pub maker_fee_rate: f64,
}

impl Default for AdaptiveBayesianConfig {
    fn default() -> Self {
        Self {
            // Learned Floor - conservative start
            as_prior_mean: 0.0003,      // 3 bps prior AS
            as_prior_std: 0.0005,       // 5 bps uncertainty
            floor_risk_k: 1.5,          // 1.5σ safety margin
            floor_absolute_min: 0.0001, // 1 bp hard floor

            // Shrinkage Gamma - moderate base
            gamma_base: 0.3,
            tau_initial: 0.1,           // Small initial adjustments
            gamma_learning_rate: 0.001,
            gamma_min: 0.05,
            gamma_max: 2.0,

            // Blended Kappa - liquid market priors
            kappa_prior_mean: 2500.0,   // 4 bps avg fill distance
            kappa_prior_strength: 5.0,
            kappa_blend_min_fills: 10,
            kappa_blend_scale: 5.0,
            kappa_warmup_factor: 0.8,   // 20% conservative during warmup

            // Fill Rate Controller
            target_fill_rate: 0.02,     // 1 fill per 50 seconds
            fill_ceiling_mult: 1.5,
            fill_min_observation_secs: 120.0,

            // Fees
            maker_fee_rate: 0.0003,     // 3 bps (user's actual fee)
        }
    }
}
```

---

## Part 5: Implementation Plan

### Phase 1: Foundation (Estimated: 2-3 files)

**Files to Create**:
- `src/market_maker/adaptive/mod.rs` - Module definition
- `src/market_maker/adaptive/learned_floor.rs` - Component 1
- `src/market_maker/adaptive/config.rs` - Configuration

**Tasks**:
1. Create `LearnedSpreadFloor` struct with Bayesian AS tracking
2. Add unit tests for floor calculation
3. Wire into existing `AdverseSelectionEstimator` for data

### Phase 2: Gamma Rework (Estimated: 2 files)

**Files to Create/Modify**:
- `src/market_maker/adaptive/shrinkage_gamma.rs` - Component 2
- `src/market_maker/adaptive/standardizer.rs` - Signal standardization

**Tasks**:
1. Implement `ShrinkageGamma` with online learning
2. Add `SignalStandardizer` for each signal
3. Create mapping from `MarketParams` to standardized signals
4. Unit tests for gamma bounds and shrinkage behavior

### Phase 3: Kappa Enhancement (Estimated: 1-2 files)

**Files to Modify**:
- `src/market_maker/estimator/kappa.rs` - Add blending
- `src/market_maker/adaptive/blended_kappa.rs` - Wrapper

**Tasks**:
1. Add `BlendedKappaEstimator` wrapper
2. Implement sigmoid blending logic
3. Feed own-fill data separately from market trades
4. Unit tests for blend weight transitions

### Phase 4: Fill Controller (Estimated: 1 file)

**Files to Create**:
- `src/market_maker/adaptive/fill_controller.rs` - Component 4

**Tasks**:
1. Implement `FillRateController` with Poisson model
2. Add spread ceiling calculation
3. Unit tests for fill rate tracking

### Phase 5: Integration (Estimated: 2-3 files)

**Files to Modify**:
- `src/market_maker/mod.rs` - Wire new components
- `src/market_maker/strategy/glft.rs` - Use adaptive spread
- `src/market_maker/config.rs` - Add new config

**Tasks**:
1. Create `AdaptiveSpreadCalculator` orchestrator
2. Replace `min_spread_floor` usage with `learned_floor`
3. Replace multiplicative gamma with shrinkage gamma
4. Add fill/no-fill callbacks for learning
5. Integration tests

### Phase 6: Testing & Validation

**Tasks**:
1. Backtest on historical trade data
2. Verify spread tightens when AS is low
3. Verify spread widens when AS increases
4. Verify fill rate controller activates when too wide
5. Verify gamma doesn't explode under stress

---

## Part 6: Expected Outcomes

### Quantitative Targets

| Metric | Current | Target |
|--------|---------|--------|
| Minimum half-spread | 8+ bps | 3-5 bps |
| Warmup spread | 50-100 bps | 8-12 bps |
| Fill rate | ~0 | 1+ per minute |
| Gamma explosion (worst case) | ×4-8 | ×1.5-2 |
| Adverse selection coverage | Fixed | Learned ±1σ |

### Qualitative Benefits

1. **Self-Correcting**: If spreads too wide → fill controller tightens
2. **Market-Adaptive**: Floor learns actual AS, not arbitrary constant
3. **Risk-Preserving**: Shrinkage prevents gamma explosion while keeping signal sensitivity
4. **Stochastic Power**: σ, λ, HJB still control skew and sizing

---

## Appendix A: Mathematical Proofs

### A.1 Shrinkage Gamma Convergence

The horseshoe prior has the property that:
```
E[wᵢ | data] → 0  as signal i becomes noise
E[wᵢ | data] → MLE  as signal i strongly predicts
```

This is proven in Carvalho et al. (2010) "The Horseshoe Estimator for Sparse Signals".

### A.2 Fill Rate Controller Stability

The Gamma-Poisson conjugacy ensures:
```
E[ρ | k fills in t seconds] = (α + k) / (β + t)
```

As k/t → ρ_true, the posterior concentrates around the true rate.

### A.3 Blending Optimality

The sigmoid blend:
```
κ_eff = (1-w)×κ_book + w×κ_own
```

Minimizes expected squared error when:
- κ_book has variance σ²_book (high)
- κ_own has variance σ²_own / n (decreasing with fills)
- w = σ²_book / (σ²_book + σ²_own/n) ≈ sigmoid

---

## Appendix B: File Structure

```
src/market_maker/
├── adaptive/
│   ├── mod.rs                    # Module exports
│   ├── config.rs                 # AdaptiveBayesianConfig
│   ├── learned_floor.rs          # LearnedSpreadFloor
│   ├── shrinkage_gamma.rs        # ShrinkageGamma
│   ├── standardizer.rs           # SignalStandardizer
│   ├── blended_kappa.rs          # BlendedKappaEstimator
│   ├── fill_controller.rs        # FillRateController
│   └── calculator.rs             # AdaptiveSpreadCalculator
├── strategy/
│   ├── glft.rs                   # Modified to use adaptive
│   └── ...
└── mod.rs                        # Wire adaptive module
```
