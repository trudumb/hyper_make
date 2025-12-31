# Dynamic Ladder Depths with Bayesian Fill Probability

## Design Document

**Date**: 2024-12-30
**Status**: Proposed
**Author**: Claude (AI Assistant)

---

## 1. Problem Statement

### Current Behavior
The ladder strategy uses **static depth levels** from `LadderConfig`:
- `min_depth_bps: 2.0`
- `max_depth_bps: 50.0`
- `num_levels: 5`

With geometric spacing, this produces depths: 2, 6.3, 20, 50 bps regardless of market conditions.

### Observed Issues
1. **Wide spreads (~45 bps)** even in liquid markets with high κ
2. **Symmetric quoting** despite directional imbalances
3. **No adaptation** to volatility, order flow intensity, or regime changes
4. **Disconnected from GLFT theory** - depths should derive from optimal spread formula

### Root Cause
The GLFT optimal spread formula is:
```
δ* = (1/γ) × ln(1 + γ/κ)
```

With γ=0.5 and κ=255, this gives δ*≈40 bps. But:
1. The ladder doesn't compute depths from this formula
2. The depths are fixed at config time, not updated dynamically
3. Fill probability isn't incorporated into depth selection

---

## 2. Proposed Architecture

### 2.1 Dynamic Depth Computation

Replace static `LadderConfig` depths with a **DynamicDepthGenerator** that computes optimal depths from market params:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        MarketMaker<S, E>                            │
│                                                                      │
│  ┌─────────────────┐    ┌─────────────────┐    ┌──────────────────┐ │
│  │ ParameterEsti-  │───►│ DynamicDepth-   │───►│ LadderStrategy   │ │
│  │ mator           │    │ Generator       │    │                  │ │
│  │                 │    │                 │    │                  │ │
│  │ • sigma         │    │ • GLFT spread   │    │ • generate_ladder│ │
│  │ • kappa         │    │ • Fill prob     │    │ • constrained    │ │
│  │ • kappa_bid     │    │ • Kelly alloc   │    │   optimizer      │ │
│  │ • kappa_ask     │    │                 │    │                  │ │
│  └─────────────────┘    └─────────────────┘    └──────────────────┘ │
│           │                      │                      │           │
│           │         ┌────────────┴────────────┐        │           │
│           │         │                         │        │           │
│           │         ▼                         ▼        │           │
│           │  ┌─────────────────┐    ┌─────────────────┐│           │
│           │  │ BayesianFill-  │    │ OptimalDepth-   ││           │
│           │  │ Probability    │    │ Selector        ││           │
│           │  │                │    │                 ││           │
│           │  │ P(fill|δ,τ)   │    │ δ* = f(γ,κ,σ)  ││           │
│           │  │ posterior      │    │ depth spacing   ││           │
│           │  └─────────────────┘    └─────────────────┘│           │
│           │                                            │           │
│           └────────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Bayesian Fill Probability Model

Extend the existing first-passage model with Bayesian posterior:

```rust
/// Bayesian fill probability model with conjugate Beta prior
pub struct BayesianFillModel {
    /// Prior parameters (α₀, β₀) - Beta distribution
    prior_alpha: f64,
    prior_beta: f64,

    /// Observations by depth bucket: (fills, attempts)
    observations: HashMap<DepthBucket, (u64, u64)>,

    /// First-passage theoretical model for fallback
    theoretical: FirstPassageFillModel,
}

impl BayesianFillModel {
    /// Posterior mean fill probability at depth δ
    /// P(fill | δ) = (α₀ + fills) / (α₀ + β₀ + attempts)
    pub fn fill_probability(&self, depth_bps: f64, sigma: f64, tau: f64) -> f64 {
        let bucket = DepthBucket::from_bps(depth_bps);

        if let Some(&(fills, attempts)) = self.observations.get(&bucket) {
            // Posterior mean from conjugate update
            let posterior_alpha = self.prior_alpha + fills as f64;
            let posterior_beta = self.prior_beta + (attempts - fills) as f64;
            posterior_alpha / (posterior_alpha + posterior_beta)
        } else {
            // Fall back to theoretical first-passage
            self.theoretical.probability(depth_bps, sigma, tau)
        }
    }

    /// Posterior standard deviation (uncertainty)
    pub fn fill_uncertainty(&self, depth_bps: f64) -> f64 {
        let bucket = DepthBucket::from_bps(depth_bps);

        if let Some(&(fills, attempts)) = self.observations.get(&bucket) {
            let alpha = self.prior_alpha + fills as f64;
            let beta = self.prior_beta + (attempts - fills) as f64;
            // Beta distribution std: sqrt(α×β / ((α+β)²×(α+β+1)))
            (alpha * beta / ((alpha + beta).powi(2) * (alpha + beta + 1.0))).sqrt()
        } else {
            0.5 // High uncertainty for unobserved buckets
        }
    }
}
```

### 2.3 Optimal Depth Selection

```rust
/// Computes optimal depth levels from market parameters
pub struct OptimalDepthSelector {
    /// Number of levels per side
    num_levels: usize,

    /// Minimum practical depth (exchange tick constraints)
    min_depth_bps: f64,

    /// Maximum depth multiple of optimal (e.g., 3x)
    max_depth_multiple: f64,

    /// Spacing mode
    spacing: DepthSpacing,
}

#[derive(Debug, Clone, Copy)]
pub enum DepthSpacing {
    /// Geometric around optimal: [δ*/r², δ*/r, δ*, δ*×r, δ*×r²]
    Geometric { ratio: f64 },
    /// Linear around optimal: [δ*-2Δ, δ*-Δ, δ*, δ*+Δ, δ*+2Δ]
    Linear { step_bps: f64 },
    /// Quantile-based from fill probability: depths at 20%, 40%, 60%, 80% P(fill)
    Quantile { model: Arc<BayesianFillModel> },
}

impl OptimalDepthSelector {
    /// Compute GLFT optimal half-spread
    fn glft_optimal_spread(&self, gamma: f64, kappa: f64) -> f64 {
        let ratio = gamma / kappa;
        if ratio > 1e-9 && gamma > 1e-9 {
            (1.0 / gamma) * (1.0 + ratio).ln()
        } else {
            1.0 / kappa.max(1.0)
        }
    }

    /// Generate depth levels centered on optimal
    pub fn compute_depths(
        &self,
        gamma: f64,
        kappa: f64,
        sigma: f64,
        time_horizon: f64,
    ) -> Vec<f64> {
        let optimal_bps = self.glft_optimal_spread(gamma, kappa) * 10000.0;

        // Ensure optimal is at least min_depth
        let optimal_bps = optimal_bps.max(self.min_depth_bps);

        // Cap max depth at multiple of optimal
        let max_bps = optimal_bps * self.max_depth_multiple;

        match &self.spacing {
            DepthSpacing::Geometric { ratio } => {
                self.geometric_depths(optimal_bps, max_bps, *ratio)
            }
            DepthSpacing::Linear { step_bps } => {
                self.linear_depths(optimal_bps, max_bps, *step_bps)
            }
            DepthSpacing::Quantile { model } => {
                self.quantile_depths(model, sigma, time_horizon)
            }
        }
    }

    fn geometric_depths(&self, center: f64, max: f64, ratio: f64) -> Vec<f64> {
        let n = self.num_levels;
        let half = n / 2;

        (0..n)
            .map(|i| {
                let offset = i as i32 - half as i32;
                let depth = center * ratio.powi(offset);
                depth.clamp(self.min_depth_bps, max)
            })
            .collect()
    }
}
```

### 2.4 Integration with Orchestrator

The `MarketMaker` orchestrator wires everything together:

```rust
impl<S: QuotingStrategy, E: OrderExecutor> MarketMaker<S, E> {
    /// Quote cycle with dynamic depth computation
    async fn quote_cycle(&mut self) -> Result<()> {
        // 1. Get current market params from estimator
        let market_params = self.build_market_params();

        // 2. Compute effective gamma (includes all scaling)
        let gamma = self.compute_effective_gamma(&market_params);

        // 3. Get Bayesian kappa estimates (asymmetric bid/ask)
        let kappa_bid = self.estimator.kappa_bid();
        let kappa_ask = self.estimator.kappa_ask();

        // 4. **NEW**: Compute dynamic depths from GLFT
        let depth_generator = self.get_depth_generator();
        let bid_depths = depth_generator.compute_depths(
            gamma, kappa_bid,
            market_params.sigma,
            market_params.time_horizon
        );
        let ask_depths = depth_generator.compute_depths(
            gamma, kappa_ask,
            market_params.sigma,
            market_params.time_horizon
        );

        // 5. Update LadderConfig with dynamic depths
        let dynamic_config = LadderConfig {
            depths_bps: Some(DynamicDepths { bid: bid_depths, ask: ask_depths }),
            ..self.ladder_config.clone()
        };

        // 6. Generate ladder with dynamic depths
        let ladder = self.strategy.generate_ladder(
            &quote_config,
            position,
            max_position,
            target_liquidity,
            &market_params,
            &dynamic_config,  // Pass dynamic depths
        );

        // 7. Record fill outcomes for Bayesian updates
        // (handled in fill processing pipeline)

        Ok(())
    }
}
```

---

## 3. Data Flow

```
                                  Market Data
                                      │
                    ┌─────────────────┼─────────────────┐
                    │                 │                 │
                    ▼                 ▼                 ▼
            ┌───────────┐    ┌───────────┐    ┌───────────┐
            │  Trades   │    │  L2 Book  │    │  Fills    │
            └─────┬─────┘    └─────┬─────┘    └─────┬─────┘
                  │                │                │
                  ▼                ▼                ▼
       ┌──────────────────────────────────────────────────────┐
       │              Parameter Estimator                      │
       │                                                       │
       │  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐  │
       │  │ Volatility  │  │   Kappa     │  │ Fill Model   │  │
       │  │ (Bipower)   │  │ (Bayesian)  │  │ (Bayesian)   │  │
       │  │             │  │             │  │              │  │
       │  │ σ, regime   │  │ κ_bid, κ_ask│  │ P(fill|δ)   │  │
       │  └─────────────┘  └─────────────┘  └──────────────┘  │
       └───────────────────────────┬───────────────────────────┘
                                   │
                                   ▼
                    ┌───────────────────────────┐
                    │   Dynamic Depth Generator │
                    │                           │
                    │  δ*_bid = f(γ, κ_bid)    │
                    │  δ*_ask = f(γ, κ_ask)    │
                    │                           │
                    │  depths_bid = space(δ*_bid)
                    │  depths_ask = space(δ*_ask)
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │     Ladder Strategy       │
                    │                           │
                    │  • Kelly-weighted sizing  │
                    │  • Constrained optimizer  │
                    │  • Inventory skew         │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                          ┌─────────────┐
                          │   Orders    │
                          └─────────────┘
```

---

## 4. Key Components

### 4.1 New Files

| File | Purpose |
|------|---------|
| `src/market_maker/quoting/ladder/depth_generator.rs` | Dynamic depth computation |
| `src/market_maker/estimator/fill_probability.rs` | Bayesian fill model |

### 4.2 Modified Files

| File | Changes |
|------|---------|
| `src/market_maker/quoting/ladder/mod.rs` | Add `DynamicDepths` to `LadderConfig` |
| `src/market_maker/quoting/ladder/generator.rs` | Accept dynamic depths |
| `src/market_maker/strategy/ladder_strat.rs` | Wire depth generator |
| `src/market_maker/mod.rs` | Initialize depth generator in orchestrator |
| `src/market_maker/estimator/parameter_estimator.rs` | Expose fill probability model |

### 4.3 New Types

```rust
/// Dynamic depth configuration (per-side)
#[derive(Debug, Clone)]
pub struct DynamicDepths {
    /// Bid-side depths in bps (best to worst)
    pub bid: Vec<f64>,
    /// Ask-side depths in bps (best to worst)
    pub ask: Vec<f64>,
}

/// Depth bucket for Bayesian observations
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub struct DepthBucket(u8);

impl DepthBucket {
    pub fn from_bps(depth_bps: f64) -> Self {
        // 0-1 bps = bucket 0, 1-2 = bucket 1, etc., capped at 50
        let bucket = (depth_bps.floor() as u8).min(50);
        DepthBucket(bucket)
    }
}

/// Fill observation for Bayesian update
#[derive(Debug, Clone)]
pub struct FillObservation {
    /// Depth from mid where order was placed (bps)
    pub placement_depth_bps: f64,
    /// Whether the order was filled
    pub was_filled: bool,
    /// Time order was resting before fill/cancel (ms)
    pub resting_time_ms: u64,
    /// Side (bid or ask)
    pub is_bid: bool,
}
```

---

## 5. Mathematical Foundation

### 5.1 GLFT Optimal Spread

From Guéant-Lehalle-Fernandez-Tapia (2012):

```
δ* = (1/γ) × ln(1 + γ/κ)
```

Where:
- δ* = optimal half-spread (as fraction of price)
- γ = risk aversion parameter
- κ = order arrival intensity (fills per unit exposure)

### 5.2 First-Passage Fill Probability

For Brownian motion with volatility σ, probability of reaching depth δ in time τ:

```
P(fill | δ, τ) = 2 × Φ(-δ / (σ × √τ))
```

Where Φ is the standard normal CDF.

### 5.3 Bayesian Fill Probability

Using Beta-Binomial conjugate model:

```
Prior:     P(fill) ~ Beta(α₀, β₀)
Likelihood: fills ~ Binomial(n, p)
Posterior:  P(fill | data) ~ Beta(α₀ + fills, β₀ + (n - fills))

Posterior mean: (α₀ + fills) / (α₀ + β₀ + n)
```

### 5.4 Asymmetric Kappa for Bid/Ask

Informed traders may have directional preference. Track separate κ for each side:

```
κ_bid = E[fill_rate | bid orders]
κ_ask = E[fill_rate | ask orders]

δ*_bid = (1/γ) × ln(1 + γ/κ_bid)
δ*_ask = (1/γ) × ln(1 + γ/κ_ask)
```

When κ_bid < κ_ask (sell pressure), bids get filled more often → tighter bid spread.

---

## 6. Implementation Plan

### Phase 1: Dynamic Depth Generator (Priority: High)
1. Create `depth_generator.rs` with `OptimalDepthSelector`
2. Implement GLFT optimal spread computation
3. Add geometric/linear spacing modes
4. Wire into `LadderStrategy.generate_ladder()`

### Phase 2: Bayesian Fill Model (Priority: Medium)
1. Create `fill_probability.rs` with `BayesianFillModel`
2. Add `FillObservation` tracking in `FillProcessor`
3. Implement Beta-Binomial conjugate updates
4. Add quantile-based depth spacing

### Phase 3: Orchestrator Integration (Priority: Medium)
1. Add `DynamicDepthGenerator` to `core::InfraComponents`
2. Wire depth generator initialization in `MarketMaker::new()`
3. Update quote cycle to use dynamic depths
4. Add metrics for depth selection

### Phase 4: Configuration & Tuning (Priority: Low)
1. Add config options for depth spacing mode
2. Expose gamma base and multipliers to config
3. Add logging/metrics for depth adaptation
4. Performance testing with live data

---

## 7. Expected Outcomes

### Before (Current State)
- Spreads: ~45 bps (static)
- Depths: 2, 6.3, 20, 50 bps (fixed)
- Fill rate: Low (quotes too wide)

### After (Dynamic Depths)
- Spreads: Adapt to κ (5-20 bps in liquid conditions)
- Depths: Center on GLFT optimal, scale with regime
- Fill rate: Higher (competitive quotes)
- Asymmetric: Tighter on high-flow side

### Quantitative Targets
| Metric | Current | Target |
|--------|---------|--------|
| Best bid spread | 45 bps | 5-15 bps |
| Best ask spread | 45 bps | 5-15 bps |
| Fill rate (bids) | ~2% | ~15-25% |
| Fill rate (asks) | ~2% | ~15-25% |
| Spread adaptation | None | <1s latency |

---

## 8. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Spreads too tight | Adverse selection losses | Floor at 2× tick + fees |
| κ estimation noise | Unstable depths | Bayesian smoothing, min confidence |
| Regime change lag | Wrong depth during transition | Volatility regime trigger |
| Over-optimization | Complexity, bugs | Gradual rollout, fallback to static |

---

## 9. Testing Strategy

### Unit Tests
- `OptimalDepthSelector`: Verify GLFT formula, spacing modes
- `BayesianFillModel`: Conjugate update correctness, edge cases
- `DynamicDepths`: Integration with `LadderConfig`

### Integration Tests
- Quote cycle produces dynamic depths
- Fill observations update model
- Depths adapt to market regime changes

### Backtesting
- Replay historical tick data
- Compare static vs dynamic depth P&L
- Measure fill rate improvement

---

## 10. References

1. Guéant, O., Lehalle, C.-A., & Fernandez-Tapia, J. (2012). "Optimal Portfolio Liquidation with Limit Orders"
2. Avellaneda, M., & Stoikov, S. (2008). "High-frequency trading in a limit order book"
3. Cartea, Á., Jaimungal, S., & Penalva, J. (2015). "Algorithmic and High-Frequency Trading"
