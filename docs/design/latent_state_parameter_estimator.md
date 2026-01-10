# Latent State Parameter Estimator: Implementation Guide

> **Status:** Design Document
> **Author:** Claude
> **Date:** 2026-01-10
> **Branch:** `claude/latent-state-parameters-CxTNs`

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Theoretical Foundation](#theoretical-foundation)
3. [Architecture Overview](#architecture-overview)
4. [Implementation Phases](#implementation-phases)
5. [Component Specifications](#component-specifications)
6. [Integration with Existing Infrastructure](#integration-with-existing-infrastructure)
7. [Testing and Validation](#testing-and-validation)
8. [Operational Considerations](#operational-considerations)
9. [Appendix: Mathematical Details](#appendix-mathematical-details)

---

## Executive Summary

### The Core Insight

A market maker's edge isn't in predicting price. It's in predicting **who is about to trade and why**.

The observable order book and trade flow are the **output** of a latent process involving:

| Trader Type | Characteristics | Impact on MM |
|-------------|-----------------|--------------|
| **Informed traders** | Have alpha, will adversely select you | Negative edge |
| **Noise traders** | Random, no information | Positive edge |
| **Other market makers** | Competing for spread | Reduces fill rate |
| **Forced traders** | Liquidations, rebalances - predictable flow | Positive edge if detected |

### What We're Building

A diagnostic binary (`src/bin/parameter_estimator.rs`) that:

1. **Decomposes trade flow** into informed/noise/forced components
2. **Estimates latent state** via particle filtering
3. **Quantifies edge** as function of market conditions
4. **Produces actionable output**: when and how aggressively to quote

### Key Deliverables

```
src/bin/parameter_estimator.rs     # Main binary
src/market_maker/
├── estimator/
│   ├── informed_flow.rs           # Mixture model for flow decomposition
│   ├── volatility_filter.rs       # Particle filter for stochastic vol
│   ├── fill_rate_model.rs         # Conditional fill rate estimation
│   └── as_decomposition.rs        # Adverse selection attribution
├── latent/
│   ├── mod.rs                     # Latent state module
│   ├── joint_dynamics.rs          # Correlated parameter evolution
│   └── edge_surface.rs            # Edge quantification
└── process_models/
    └── regime_filter.rs           # Regime switching detection
```

---

## Theoretical Foundation

### The Latent State Space

Standard parameter estimation treats each parameter independently. Reality is different: **parameters are correlated through latent market state**.

```
┌─────────────────────────────────────────────────────────────────┐
│                    LATENT STATE SPACE                           │
│                                                                 │
│  θ(t) = [                                                       │
│    π_informed(t),      // P(next trade is informed)             │
│    π_forced(t),        // P(next trade is liquidation/rebalance)│
│    μ_arrival(t),       // Expected arrival rate                 │
│    σ_local(t),         // Local volatility                      │
│    ρ_flow(t),          // Flow autocorrelation                  │
│    regime(t),          // Discrete market state                 │
│  ]                                                              │
│                                                                 │
│  Observables: trades, L2 book, funding, open interest           │
│  Goal: P(θ | observables) via filtering                         │
└─────────────────────────────────────────────────────────────────┘
```

### Why This Matters

| Traditional Approach | Latent State Approach |
|---------------------|----------------------|
| Estimate σ, κ, AS independently | Model joint distribution P(σ, κ, AS \| state) |
| Point estimates only | Full posterior with uncertainty |
| Reactive (measures past) | Predictive (forecasts next trade) |
| Same parameters in all conditions | Condition-specific edge quantification |

### The Edge Equation

```
Edge(state) = E[spread_capture | state] - E[AS | state] - fees
            = ∫ P(fill | δ, state) × δ dδ - E[Δp | fill, state] - fees
```

The key insight: **Edge varies dramatically by market state**. Some states have reliable positive edge. Others never do.

---

## Architecture Overview

### Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    OBSERVATION LAYER                            │
│  Trades, L2Book, AllMids, UserFills, Funding, OI                │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                  DERIVED PROCESSES (Level 2)                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Volatility  │  │    Flow      │  │    Book      │          │
│  │   Filter     │  │ Decomposition│  │  Dynamics    │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│               LATENT STATE INFERENCE (Level 3)                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Informed   │  │    Regime    │  │    Joint     │          │
│  │ Probability  │  │    Filter    │  │   Dynamics   │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│             TRADING PARAMETERS (Level 4)                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Fill Rate   │  │   Adverse    │  │    Edge      │          │
│  │    Model     │  │  Selection   │  │   Surface    │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

### Component Hierarchy

| Level | Component | Input | Output | Update Frequency |
|-------|-----------|-------|--------|------------------|
| 1 | Raw Buffers | WebSocket | Buffered data | Per message |
| 2 | VolatilityFilter | Returns | σ(t), regime, uncertainty | Per volume tick |
| 2 | FlowDecomposition | Trades | Component probabilities | Per trade |
| 2 | BookDynamics | L2 snapshots | Shape features | Per L2 update |
| 3 | InformedProbability | Flow + Book | P(informed \| state) | Per trade |
| 3 | RegimeFilter | Vol + Flow | Discrete regime | Per trade |
| 3 | JointDynamics | All Level 2 | Correlated state | 100ms |
| 4 | FillRateModel | State + Fills | λ(δ, state) | Per fill |
| 4 | ASDecomposition | Fills + Prices | E[Δp \| fill, state] | Per fill (delayed) |
| 4 | EdgeSurface | All above | Edge by condition | 1 minute |

---

## Implementation Phases

### Phase 1: Infrastructure Setup (Week 1)

**Goal:** Binary skeleton with WebSocket connection and basic buffering.

#### Tasks

1. **Create binary scaffold**
   - File: `src/bin/parameter_estimator.rs`
   - Use existing `market_maker.rs` CLI patterns
   - Connect to exchange WebSocket
   - Subscribe to: AllMids, L2Book, Trades, UserFills

2. **Create latent module structure**
   - Directory: `src/market_maker/latent/`
   - Create `mod.rs` with public exports
   - Define core traits and types

3. **Implement observation buffers**
   - Trade buffer with timestamps
   - Book snapshot buffer
   - Configurable window sizes

#### Integration Points

Reference existing patterns in:
- CLI: `src/bin/market_maker.rs` lines 1-200 (argument parsing)
- WebSocket: `src/ws/ws_manager.rs` (connection handling)
- Buffers: `src/market_maker/estimator/volume.rs` (VecDeque patterns)

#### Deliverables

```rust
// src/bin/parameter_estimator.rs
#[derive(Parser)]
struct Cli {
    #[arg(short, long)]
    asset: String,

    #[arg(long, default_value = "24h")]
    duration: String,

    #[arg(long)]
    output: Option<PathBuf>,

    #[arg(long, default_value = "5m")]
    report_interval: String,

    #[arg(long)]
    network: Option<String>,  // testnet/mainnet
}
```

---

### Phase 2: Volatility Filter (Week 2)

**Goal:** Particle filter for stochastic volatility with regime switching.

#### Mathematical Foundation

State space model:
```
log(σ_t) = log(σ_{t-1}) + κ_r × (θ_r - log(σ_{t-1})) × Δt + ξ_r × √Δt × ε_t
r_t ~ N(0, σ_t²)
```

Where `r` indexes the regime (LOW, NORMAL, HIGH).

Regime transition:
```
P(regime_{t+1} | regime_t) = transition_matrix[regime_t][regime_{t+1}]
```

#### Implementation

```rust
// src/market_maker/estimator/volatility_filter.rs

pub struct VolatilityFilter {
    particles: Vec<VolParticle>,
    weights: Vec<f64>,
    transition: [[f64; 3]; 3],
    regime_params: [RegimeParams; 3],
    n_particles: usize,
    resampling_threshold: f64,  // ESS threshold
}

pub struct VolParticle {
    log_vol: f64,
    regime: Regime,
}

#[derive(Clone, Copy)]
pub struct RegimeParams {
    kappa: f64,     // Mean reversion speed
    theta: f64,     // Long-run mean (log scale)
    xi: f64,        // Vol of vol
}

#[derive(Clone, Copy, PartialEq)]
pub enum Regime {
    Low = 0,
    Normal = 1,
    High = 2,
}

impl VolatilityFilter {
    /// Initialize with default priors
    pub fn new(config: VolFilterConfig) -> Self;

    /// Update on new return observation
    pub fn on_return(&mut self, r: f64, dt_seconds: f64);

    /// Current volatility estimate (posterior mean)
    pub fn sigma(&self) -> f64;

    /// Current regime (MAP estimate)
    pub fn regime(&self) -> Regime;

    /// Regime confidence (posterior probability)
    pub fn regime_confidence(&self) -> f64;

    /// Predictive distribution over horizon
    pub fn predictive(&self, horizon_ms: u64) -> VolDistribution;

    /// Probability of regime change
    pub fn regime_change_probability(&self, horizon_ms: u64) -> f64;

    /// Uncertainty quantification
    pub fn sigma_credible_interval(&self, level: f64) -> (f64, f64);
}
```

#### Integration

- Integrate with existing `VolumeBucketAccumulator` for volume-time returns
- Use existing `VolatilityRegime` enum from `estimator/volatility.rs`
- Add to `MarketEstimator` trait if needed

---

### Phase 3: Flow Decomposition (Week 3)

**Goal:** Mixture model decomposing trades into informed/noise/forced components.

#### Mathematical Foundation

Each trade comes from one of K components:
```
P(trade | component_k) = P(size | k) × P(arrival | k) × P(impact | k)

P(component_k | trade) ∝ P(trade | k) × P(k)  [Bayes rule]
```

Component characteristics:

| Component | Size | Arrival | Impact | Signature |
|-----------|------|---------|--------|-----------|
| Informed | Large, clustered | Bursty (high α) | High, persistent | Before news/moves |
| Noise | Random | Poisson | Low, temporary | Background flow |
| Forced | Predictable | Time-clustered | One-sided | Near liquidation prices |

#### Implementation

```rust
// src/market_maker/estimator/informed_flow.rs

pub struct InformedFlowEstimator {
    components: [ComponentParams; 3],  // informed, noise, forced
    responsibilities: VecDeque<[f64; 3]>,  // Recent P(component | trade)
    em_buffer: VecDeque<TradeFeatures>,    // For online EM updates
    priors: [f64; 3],                      // P(component)
    config: InformedFlowConfig,
}

pub struct ComponentParams {
    // Size distribution (log-normal)
    mu_size: f64,
    sigma_size: f64,

    // Arrival process (Hawkes)
    lambda_0: f64,
    alpha: f64,
    beta: f64,

    // Price impact
    impact_per_unit: f64,
    impact_decay_ms: f64,
}

pub struct TradeFeatures {
    size: f64,
    inter_arrival_ms: u64,
    price_impact: f64,      // Measured after 1s
    book_imbalance: f64,    // At time of trade
    is_buy: bool,
}

pub struct FlowDecomposition {
    pub p_informed: f64,
    pub p_noise: f64,
    pub p_forced: f64,
    pub confidence: f64,  // Based on recent consistency
}

impl InformedFlowEstimator {
    /// Update on new trade
    pub fn on_trade(&mut self, trade: &Trade, features: &TradeFeatures);

    /// Current flow decomposition
    pub fn decomposition(&self) -> FlowDecomposition;

    /// Predictive: P(next trade is informed)
    pub fn p_next_informed(&self) -> f64;

    /// Update component parameters (online EM step)
    fn em_step(&mut self);
}
```

#### Reference Patterns

- Follow `HawkesEstimator` pattern in `process_models/hawkes.rs`
- Use existing `TradeFlowTracker` for imbalance features
- Integrate with `SoftJumpClassifier` for regime detection

---

### Phase 4: Fill Rate Model (Week 4)

**Goal:** Conditional fill rate as function of market state.

#### Mathematical Foundation

Fill rate model:
```
log(λ(δ)) = log(λ_0) - δ/δ_char + Σ β_i × x_i
```

Where `x_i` are state features:
- Volatility (higher vol → more fills)
- Spread (wider spread → fewer fills)
- Book imbalance (toward you → more fills)
- Time of day (hour effects)
- Regime (calm → fewer fills)

#### Implementation

```rust
// src/market_maker/estimator/fill_rate_model.rs

pub struct FillRateModel {
    // Base model: λ(δ) = λ_0 × exp(-δ/δ_char)
    lambda_0: BayesianEstimate,
    delta_char: BayesianEstimate,

    // Feature coefficients (log-linear)
    coefficients: FillRateCoefficients,

    // Online learning state
    observation_buffer: VecDeque<FillObservation>,
    feature_stats: OnlineRegression,

    config: FillRateConfig,
}

pub struct BayesianEstimate {
    pub mean: f64,
    pub variance: f64,
    pub n_observations: usize,
}

pub struct FillRateCoefficients {
    pub vol: f64,            // β_vol
    pub spread: f64,         // β_spread
    pub imbalance: f64,      // β_imbalance
    pub time_of_day: [f64; 24],  // β_hour
    pub regime: [f64; 3],    // β_regime

    // Uncertainty (for confidence intervals)
    pub covariance: Matrix6x6,  // Approximate
}

pub struct FillObservation {
    depth_bps: f64,
    filled: bool,
    state_at_quote: MarketState,
    time_to_fill_ms: Option<u64>,
}

impl FillRateModel {
    /// Record fill/no-fill observation
    pub fn observe(&mut self, obs: FillObservation);

    /// Expected fill rate at depth given current state
    pub fn fill_rate(&self, depth_bps: f64, state: &MarketState) -> f64;

    /// Optimal depth for given state and target fill rate
    pub fn optimal_depth(&self, target_fill_rate: f64, state: &MarketState) -> f64;

    /// Uncertainty in fill rate estimate
    pub fn fill_rate_ci(&self, depth_bps: f64, state: &MarketState, level: f64) -> (f64, f64);
}
```

#### Integration

- Use existing `BayesianFillModel` from `strategy/ladder_strat.rs` as reference
- Integrate with `PositionTracker` for fill observations
- Connect to `UserFills` message handler

---

### Phase 5: Adverse Selection Decomposition (Week 5)

**Goal:** Attribute AS to permanent/temporary/timing components.

#### Mathematical Foundation

Total adverse selection:
```
AS = E[Δp | fill]
   = permanent + temporary + timing
```

Where:
- **Permanent:** Price change that persists (information)
- **Temporary:** Price change that reverts (liquidity)
- **Timing:** Cost vs. TWAP execution (got picked off)

Measurement:
```
Δp(τ) = p(t+τ) - p(t)  for τ ∈ {1s, 5s, 30s, 5min}

permanent = lim_{τ→∞} E[Δp(τ) | fill]  ≈ E[Δp(5min) | fill]
temporary = E[Δp(1s) | fill] - permanent
```

#### Implementation

```rust
// src/market_maker/estimator/as_decomposition.rs

pub struct ASDecomposition {
    // Tracked fills awaiting measurement
    pending_measurements: VecDeque<PendingASMeasurement>,

    // Rolling statistics by horizon
    as_by_horizon: HashMap<u64, RollingMean>,  // τ_ms → mean

    // Conditional AS
    as_by_condition: HashMap<Condition, RollingMean>,

    // Component estimates
    permanent: RollingMean,
    temporary: RollingMean,
    timing: RollingMean,

    config: ASDecompositionConfig,
}

pub struct PendingASMeasurement {
    fill_time_ms: u64,
    fill_price: f64,
    fill_side: Side,
    state_at_fill: MarketState,
    measurement_times: Vec<u64>,  // When to measure
}

#[derive(Hash, Eq, PartialEq)]
pub enum Condition {
    FlowImbalance(i8),   // Discretized
    VolRegime(Regime),
    Hour(u8),
    Combined(Box<(Condition, Condition)>),
}

impl ASDecomposition {
    /// Schedule measurement after fill
    pub fn on_fill(&mut self, fill: &FillEvent, state: &MarketState);

    /// Process pending measurements
    pub fn on_price_update(&mut self, price: f64, timestamp_ms: u64);

    /// Current AS estimates
    pub fn total_as_bps(&self) -> f64;
    pub fn permanent_as_bps(&self) -> f64;
    pub fn temporary_as_bps(&self) -> f64;
    pub fn timing_cost_bps(&self) -> f64;

    /// Conditional AS
    pub fn as_given(&self, condition: &Condition) -> Option<f64>;

    /// AS prediction for current state
    pub fn predicted_as(&self, state: &MarketState) -> f64;
}
```

#### Integration

- Extend existing `AdverseSelectionEstimator` in `adverse_selection/`
- Use `FillConsumer` trait for fill observation
- Connect to `AllMids` for price measurements

---

### Phase 6: Edge Surface (Week 6)

**Goal:** Quantify edge as function of market state with uncertainty.

#### Mathematical Foundation

Edge surface:
```
Edge(state) = E[spread_capture | state] - E[AS | state] - fees

where:
  spread_capture = ∫ P(fill | δ, state) × δ × dδ

  for optimal δ:
  δ* = argmax_δ [P(fill | δ, state) × (δ - AS(state) - fees)]
```

Uncertainty propagation:
```
Var[Edge] ≈ (∂Edge/∂λ)² × Var[λ] + (∂Edge/∂AS)² × Var[AS] + 2×Cov[λ,AS]×...
```

#### Implementation

```rust
// src/market_maker/latent/edge_surface.rs

pub struct EdgeSurface {
    // Discretized edge estimates
    grid: EdgeGrid,

    // Uncertainty at each point
    uncertainty: EdgeGrid,

    // Component models (references)
    fill_rate: Arc<FillRateModel>,
    as_model: Arc<ASDecomposition>,

    // Fee structure
    fees_bps: f64,

    config: EdgeSurfaceConfig,
}

pub struct EdgeGrid {
    // Dimensions: [vol_bucket][regime][hour][flow_bucket]
    data: Vec<Vec<Vec<Vec<f64>>>>,
    vol_buckets: Vec<f64>,      // Bucket boundaries
    flow_buckets: Vec<f64>,     // Bucket boundaries
}

pub struct EdgeEstimate {
    pub edge_bps: f64,
    pub uncertainty_bps: f64,
    pub confidence: f64,           // P(edge > 0)
    pub optimal_spread_bps: f64,
    pub expected_fill_rate: f64,
    pub expected_as_bps: f64,
}

impl EdgeSurface {
    /// Estimate edge for current state
    pub fn estimate(&self, state: &MarketState) -> EdgeEstimate;

    /// Should we quote given current state?
    pub fn should_quote(&self, state: &MarketState) -> bool {
        let est = self.estimate(state);
        // Quote only if edge > 2σ (95% confidence)
        est.edge_bps > 2.0 * est.uncertainty_bps
    }

    /// Recalculate grid (expensive, do periodically)
    pub fn recalculate(&mut self);

    /// Export for analysis
    pub fn to_json(&self) -> serde_json::Value;
}
```

---

### Phase 7: Joint Dynamics (Week 7)

**Goal:** Model correlated evolution of parameters.

#### The Key Insight

Most people estimate parameters independently. But they're correlated:

- High informed flow → high AS → low edge
- High volatility → high fill rate → but also high AS
- Regime transitions → parameters become unreliable

The edge comes from modeling the **joint dynamics**.

#### Implementation

```rust
// src/market_maker/latent/joint_dynamics.rs

pub struct JointDynamics {
    // Current latent state
    state: LatentState,

    // State uncertainty (covariance)
    covariance: StateCovariance,

    // Transition model
    transition: TransitionModel,

    // Parameter correlation structure
    correlations: ParameterCorrelations,
}

pub struct LatentState {
    pub sigma: f64,
    pub regime: Regime,
    pub p_informed: f64,
    pub flow_momentum: f64,
    pub book_pressure: f64,
}

pub struct StateCovariance {
    // Lower triangular (Cholesky) for efficiency
    chol: Matrix5x5,
}

pub struct ParameterCorrelations {
    // Learned from data
    sigma_as: f64,          // σ ↔ AS correlation
    sigma_fill_rate: f64,   // σ ↔ λ correlation
    informed_as: f64,       // P(informed) ↔ AS correlation
}

impl JointDynamics {
    /// Update on new observations
    pub fn update(&mut self, obs: &Observations);

    /// Marginal for single parameter
    pub fn sigma(&self) -> (f64, f64);  // (mean, std)

    /// Joint prediction
    pub fn predict(&self, horizon_ms: u64) -> (LatentState, StateCovariance);

    /// Edge uncertainty accounting for correlations
    pub fn edge_uncertainty(&self, state: &MarketState) -> f64;
}
```

---

### Phase 8: Binary Integration (Week 8)

**Goal:** Complete diagnostic binary with reporting.

#### Implementation

```rust
// src/bin/parameter_estimator.rs

pub struct ParameterEstimator {
    // === LEVEL 1: Raw observations ===
    trade_buffer: VecDeque<Trade>,
    book_buffer: VecDeque<BookSnapshot>,

    // === LEVEL 2: Derived processes ===
    volatility: VolatilityFilter,
    flow: InformedFlowEstimator,
    book_dynamics: BookDynamicsTracker,

    // === LEVEL 3: Latent state inference ===
    informed_probability: f64,
    regime: Regime,
    joint: JointDynamics,

    // === LEVEL 4: Trading-relevant parameters ===
    fill_rate: FillRateModel,
    adverse_selection: ASDecomposition,

    // === LEVEL 5: Edge quantification ===
    edge_surface: EdgeSurface,

    // State
    config: EstimatorConfig,
    start_time: Instant,
}

impl ParameterEstimator {
    async fn run(&mut self, ws: WebSocketStream) {
        let mut report_interval = Instant::now();

        loop {
            let msg = ws.recv().await?;

            match msg {
                Trade(t) => self.on_trade(t),
                Book(b) => self.on_book(b),
                Fill(f) => self.on_fill(f),
                AllMids(m) => self.on_mid(m),
            }

            // Periodic AS measurements
            self.process_pending_as_measurements();

            // Periodic reporting
            if report_interval.elapsed() > self.config.report_interval {
                self.print_report();
                report_interval = Instant::now();
            }
        }
    }

    fn print_report(&self) {
        // Detailed report format - see Report Format section
    }
}
```

#### Report Format

```
============================================================
PARAMETER ESTIMATION REPORT
============================================================

VOLATILITY:
  Current: 12.34 bps/√s
  Regime: Normal (P=0.87)
  P(regime change next 5m): 8.2%
  95% CI: [10.2, 14.8] bps/√s

FLOW DECOMPOSITION:
  P(informed): 23.4%
  P(noise): 68.1%
  P(forced): 8.5%

FILL RATE MODEL:
  λ_0 = 2.3 ± 0.4 fills/min
  δ_char = 4.2 ± 0.8 bps
  Current fill rate at 8 bps: 0.8 fills/min

ADVERSE SELECTION:
  Total: 3.2 bps
  Permanent: 2.1 bps (66%)
  Temporary: 0.8 bps (25%)
  Timing: 0.3 bps (9%)

============================================================
EDGE ANALYSIS
============================================================

Current state:
  Vol regime: Normal
  Flow state: Mixed
  Hour (UTC): 14

Expected edge: 2.8 ± 1.4 bps
Confidence: 82%

✓ EDGE EXISTS at current state
  Optimal spread: 7.2 bps

EDGE BY CONDITION:
  Low × Asia:     4.2 bps ✓
  Low × London:   3.8 bps ✓
  Low × US:       3.1 bps ✓
  Normal × Asia:  2.9 bps ✓
  Normal × London: 2.4 bps ~
  Normal × US:    1.8 bps ~
  High × Asia:   -0.3 bps ✗
  High × London: -1.2 bps ✗
  High × US:     -2.1 bps ✗

============================================================
```

---

## Component Specifications

### VolatilityFilter

| Aspect | Specification |
|--------|---------------|
| Algorithm | Bootstrap particle filter with systematic resampling |
| Particles | 500 default, configurable |
| Regimes | LOW, NORMAL, HIGH |
| Resampling | When ESS < N/2 |
| Update | O(N) per observation |
| Memory | ~32KB for 500 particles |

#### Configuration

```rust
pub struct VolFilterConfig {
    pub n_particles: usize,          // Default: 500
    pub resampling_ess_threshold: f64,  // Default: 0.5
    pub regime_prior: [f64; 3],      // Default: [0.2, 0.6, 0.2]
    pub transition_prior: [[f64; 3]; 3],  // Sticky diagonal
    pub kappa_prior: [f64; 3],       // Mean reversion speeds
    pub theta_prior: [f64; 3],       // Long-run vol (log)
    pub xi_prior: [f64; 3],          // Vol of vol
}
```

### InformedFlowEstimator

| Aspect | Specification |
|--------|---------------|
| Algorithm | Online EM for Gaussian mixtures |
| Components | 3 (informed, noise, forced) |
| Features | size, inter-arrival, impact |
| EM Updates | Every 100 trades |
| Decay | Exponential with half-life 1000 trades |

#### Configuration

```rust
pub struct InformedFlowConfig {
    pub n_components: usize,         // Default: 3
    pub em_update_interval: usize,   // Default: 100
    pub observation_half_life: usize,  // Default: 1000
    pub min_observations: usize,     // Default: 200
    pub impact_horizon_ms: u64,      // Default: 1000
}
```

### FillRateModel

| Aspect | Specification |
|--------|---------------|
| Algorithm | Bayesian logistic regression |
| Features | vol, spread, imbalance, hour, regime |
| Prior | Weakly informative N(0, 1) on coefficients |
| Update | Online with forgetting factor |
| Uncertainty | Laplace approximation to posterior |

#### Configuration

```rust
pub struct FillRateConfig {
    pub prior_precision: f64,        // Default: 1.0
    pub forgetting_factor: f64,      // Default: 0.999
    pub min_observations: usize,     // Default: 100
    pub depth_buckets: Vec<f64>,     // Default: [2, 4, 6, 8, 10, 15, 20]
}
```

### ASDecomposition

| Aspect | Specification |
|--------|---------------|
| Horizons | 1s, 5s, 30s, 5min |
| Permanent | 5min horizon approximation |
| Temporary | 1s - permanent |
| Conditions | Flow × Vol × Hour |
| Window | Rolling 24 hours |

#### Configuration

```rust
pub struct ASDecompositionConfig {
    pub horizons_ms: Vec<u64>,       // Default: [1000, 5000, 30000, 300000]
    pub permanent_horizon_ms: u64,   // Default: 300000 (5min)
    pub window_hours: u64,           // Default: 24
    pub min_observations: usize,     // Default: 50 per condition
}
```

### EdgeSurface

| Aspect | Specification |
|--------|---------------|
| Dimensions | 5 × 3 × 3 × 5 = 225 cells |
| Vol buckets | 5 (very low → very high) |
| Regimes | 3 (LOW, NORMAL, HIGH) |
| Hour buckets | 3 (Asia, London, US) |
| Flow buckets | 5 (strong sell → strong buy) |
| Recalculation | Every 1 minute |

#### Configuration

```rust
pub struct EdgeSurfaceConfig {
    pub vol_percentiles: Vec<f64>,   // Default: [10, 30, 70, 90]
    pub flow_percentiles: Vec<f64>,  // Default: [10, 30, 70, 90]
    pub fees_bps: f64,               // Default: 1.5
    pub recalc_interval_secs: u64,   // Default: 60
    pub confidence_threshold: f64,   // Default: 0.95 (2σ)
}
```

---

## Integration with Existing Infrastructure

### Trait Implementations

The new components should implement existing traits where applicable:

```rust
// VolatilityFilter could implement MarketEstimator selectively
impl PartialMarketEstimator for VolatilityFilter {
    fn sigma_clean(&self) -> f64 { self.sigma() }
    fn volatility_regime(&self) -> VolatilityRegime {
        match self.regime() {
            Regime::Low => VolatilityRegime::Low,
            Regime::Normal => VolatilityRegime::Normal,
            Regime::High => VolatilityRegime::High,
        }
    }
}

// ASDecomposition should implement FillConsumer
impl FillConsumer for ASDecomposition {
    fn on_fill(&mut self, fill: &FillEvent) -> Option<String> {
        self.on_fill_internal(fill);
        None  // No blocking message
    }
    fn name(&self) -> &'static str { "ASDecomposition" }
    fn priority(&self) -> u32 { 30 }  // After AdverseSelectionEstimator
}
```

### Message Handler Integration

Follow patterns in `src/market_maker/messages/`:

```rust
// In the binary's event loop
match msg {
    Message::Trades(trades) => {
        for trade in trades {
            // Level 2 updates
            self.volatility.on_return(compute_return(&trade), dt);
            self.flow.on_trade(&trade, &features);

            // Level 3 updates
            self.joint.update(&observations);
        }
    }
    Message::L2Book(book) => {
        self.book_dynamics.update(&book);
        self.fill_rate.update_book_state(&book);
    }
    Message::UserFills(fills) => {
        for fill in fills {
            self.fill_rate.observe(&fill, &state);
            self.adverse_selection.on_fill(&fill, &state);
        }
    }
    Message::AllMids(mids) => {
        self.adverse_selection.on_price_update(mid, now_ms);
    }
}
```

### Configuration Integration

Extend `EstimatorConfig` or create parallel config:

```rust
pub struct LatentEstimatorConfig {
    // Volatility filter
    pub vol_filter: VolFilterConfig,

    // Flow decomposition
    pub informed_flow: InformedFlowConfig,

    // Fill rate model
    pub fill_rate: FillRateConfig,

    // AS decomposition
    pub as_decomposition: ASDecompositionConfig,

    // Edge surface
    pub edge_surface: EdgeSurfaceConfig,

    // Reporting
    pub report_interval_secs: u64,
}

impl Default for LatentEstimatorConfig {
    fn default() -> Self {
        Self {
            vol_filter: VolFilterConfig::default(),
            informed_flow: InformedFlowConfig::default(),
            fill_rate: FillRateConfig::default(),
            as_decomposition: ASDecompositionConfig::default(),
            edge_surface: EdgeSurfaceConfig::default(),
            report_interval_secs: 300,  // 5 minutes
        }
    }
}
```

### Metric Export

Extend `PrometheusMetrics` or create parallel exporter:

```rust
// New metrics for latent state estimation
mm_latent_vol_regime{regime="LOW|NORMAL|HIGH"}
mm_latent_vol_sigma_bps
mm_latent_vol_uncertainty_bps
mm_latent_p_informed
mm_latent_p_noise
mm_latent_p_forced
mm_latent_fill_rate_at_8bps
mm_latent_as_total_bps
mm_latent_as_permanent_bps
mm_latent_as_temporary_bps
mm_latent_edge_bps
mm_latent_edge_uncertainty_bps
mm_latent_should_quote
```

---

## Testing and Validation

### Unit Tests

Each component should have thorough unit tests:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_volatility_filter_regime_detection() {
        let mut filter = VolatilityFilter::new(VolFilterConfig::default());

        // Simulate calm market
        for _ in 0..100 {
            filter.on_return(0.0001, 0.1);
        }
        assert_eq!(filter.regime(), Regime::Low);

        // Simulate volatility spike
        for _ in 0..20 {
            filter.on_return(0.01, 0.1);  // 1% returns
        }
        assert_eq!(filter.regime(), Regime::High);
    }

    #[test]
    fn test_flow_decomposition_convergence() {
        let mut estimator = InformedFlowEstimator::new(config);

        // Feed synthetic informed flow (large, clustered)
        for _ in 0..100 {
            estimator.on_trade(&large_clustered_trade(), &features);
        }

        let decomp = estimator.decomposition();
        assert!(decomp.p_informed > 0.5);
    }

    #[test]
    fn test_edge_surface_negativity_in_high_vol() {
        let surface = EdgeSurface::new(/* mock models with high AS */);

        let high_vol_state = MarketState {
            vol_regime: Regime::High,
            ..Default::default()
        };

        let est = surface.estimate(&high_vol_state);
        assert!(est.edge_bps < 0.0, "Edge should be negative in high vol");
        assert!(!surface.should_quote(&high_vol_state));
    }
}
```

### Integration Tests

Test against recorded market data:

```rust
#[test]
fn test_against_recorded_data() {
    let trades = load_trades("testdata/btc_trades_24h.json");
    let books = load_books("testdata/btc_books_24h.json");
    let fills = load_fills("testdata/btc_fills_24h.json");

    let mut estimator = ParameterEstimator::new(config);

    // Replay data
    for event in merge_by_time(trades, books, fills) {
        match event {
            Event::Trade(t) => estimator.on_trade(t),
            Event::Book(b) => estimator.on_book(b),
            Event::Fill(f) => estimator.on_fill(f),
        }
    }

    // Verify edge surface has coverage
    assert!(estimator.edge_surface.coverage() > 0.8);

    // Verify AS decomposition sums correctly
    let as_total = estimator.adverse_selection.total_as_bps();
    let as_sum = estimator.adverse_selection.permanent_as_bps()
               + estimator.adverse_selection.temporary_as_bps();
    assert!((as_total - as_sum).abs() < 0.1);
}
```

### Live Validation

Run against testnet and verify:

```bash
# Run for 4 hours on testnet
cargo run --release --bin parameter_estimator -- \
    --asset BTC \
    --duration 4h \
    --output params_test.json \
    --report-interval 5m

# Analyze output
python scripts/analyze_parameters.py params_test.json
```

Validation checklist:
- [ ] Volatility tracks actual realized vol within 20%
- [ ] Regime switches correspond to visible market changes
- [ ] Flow decomposition is stable (not oscillating wildly)
- [ ] Fill rate predictions match actual fills (χ² test)
- [ ] AS decomposition sums to total AS
- [ ] Edge predictions are consistent across time

---

## Operational Considerations

### Resource Requirements

| Resource | Requirement |
|----------|-------------|
| CPU | Single core adequate (100ms cycles) |
| Memory | ~200MB (buffers + particle filter) |
| Network | WebSocket connection to exchange |
| Disk | Log file grows ~50MB/day |

### Failure Modes

| Failure | Detection | Recovery |
|---------|-----------|----------|
| WebSocket disconnect | No data for 5s | Reconnect with backoff |
| Estimator divergence | NaN/Inf values | Reset to priors |
| Memory growth | RSS > 1GB | Clear old buffers |
| Slow updates | Cycle > 1s | Reduce particle count |

### Monitoring

Key alerts:
- Estimator not warmed up after 5 minutes
- Edge uncertainty > 10 bps (model unreliable)
- AS decomposition doesn't sum (bug)
- Fill rate prediction error > 50%

---

## Appendix: Mathematical Details

### Particle Filter Algorithm

```
Algorithm: Bootstrap Particle Filter for Stochastic Volatility

Input: Observations y_{1:T}, particles {x^i_0}_{i=1}^N
Output: Filtered states {x^i_t, w^i_t}_{i=1}^N

For t = 1 to T:
  // Propagate
  For i = 1 to N:
    x^i_t ~ p(x_t | x^i_{t-1})  // State transition

  // Weight
  For i = 1 to N:
    w^i_t = p(y_t | x^i_t)      // Observation likelihood

  // Normalize
  w^i_t = w^i_t / Σ_j w^j_t

  // Resample (if ESS < N/2)
  ESS = 1 / Σ_i (w^i_t)²
  If ESS < N/2:
    Resample {x^i_t} with replacement according to {w^i_t}
    w^i_t = 1/N for all i
```

### Online EM for Mixture Models

```
Algorithm: Online EM for Flow Decomposition

For each trade t:
  // E-step: Compute responsibilities
  For k = 1 to K:
    γ_{t,k} = π_k × p(x_t | θ_k) / Σ_j π_j × p(x_t | θ_j)

  // M-step: Update parameters (exponential forgetting)
  For k = 1 to K:
    N_k = λ × N_k + γ_{t,k}
    S_k = λ × S_k + γ_{t,k} × x_t
    π_k = N_k / Σ_j N_j
    θ_k = update_component(S_k / N_k)

Where λ = exp(-1/half_life) is the forgetting factor.
```

### Edge Uncertainty Propagation

```
Edge = f(λ, AS, fees)
     = ∫ P(fill | δ) × δ dδ - AS - fees

Assuming λ and AS are estimated with uncertainty:
  λ ~ N(μ_λ, σ_λ²)
  AS ~ N(μ_AS, σ_AS²)
  Cov(λ, AS) = ρ × σ_λ × σ_AS

Then by delta method:
  Var[Edge] ≈ (∂f/∂λ)² σ_λ² + (∂f/∂AS)² σ_AS² + 2(∂f/∂λ)(∂f/∂AS) ρ σ_λ σ_AS

Note: The correlation ρ is typically positive (high vol → high λ AND high AS),
which can increase edge uncertainty beyond naive propagation.
```

---

## References

1. **Guéant, Lehalle, Fernandez-Tapia (2012)** - "Optimal Portfolio Liquidation with Limit Orders" - Foundation for GLFT model
2. **Cartea, Jaimungal, Penalva (2015)** - "Algorithmic and High-Frequency Trading" - Comprehensive MM theory
3. **Hawkes (1971)** - "Spectra of some self-exciting and mutually exciting point processes" - Order flow clustering
4. **Gordon, Salmond, Smith (1993)** - "Novel approach to nonlinear/non-Gaussian Bayesian state estimation" - Particle filtering

---

## Changelog

| Date | Author | Change |
|------|--------|--------|
| 2026-01-10 | Claude | Initial design document |
