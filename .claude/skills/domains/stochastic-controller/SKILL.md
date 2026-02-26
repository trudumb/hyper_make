---
name: stochastic-controller
description: Layer 3 optimal sequential decision-making with Bayesian belief tracking, HJB value functions, and changepoint detection. Use when working on control/, stochastic/, or process_models/ modules, debugging quote/wait/pull decisions, modifying the HJB solver, or adding action types. Covers conjugate updates, BOCD, and value of information.
user-invocable: false
---

# Stochastic Controller Skill

## Purpose

Understand and modify the Layer 3 stochastic control system that makes optimal sequential decisions. This layer sits between estimation (L1-L2) and execution (L4), answering: "Given what we believe about the market, what is the optimal action?"

No ad-hoc thresholds — all decisions emerge from Bayesian posteriors + HJB optimization.

## When to Use

- Working on `control/`, `stochastic/`, or `process_models/` modules
- Understanding how quotes are derived from beliefs
- Adding new action types or decision logic
- Debugging "why did it choose to quote/wait/pull?"
- Modifying the HJB solver or value function

## Module Map

All paths relative to `src/market_maker/`:

```
control/
  mod.rs                # StochasticController — main entry, Q-value maximization
  controller.rs         # OptimalController — core optimization via value function
  belief.rs             # ControllerBeliefState — fill rate, AS dist, ensemble weights
  types.rs              # Bayesian posteriors: GammaPosterior, NormalGammaPosterior, etc.
  actions.rs            # Action enum (Quote, Wait, Reduce, PullQuotes), NoQuoteReason
  state.rs              # ControlState — position, inventory, time, regime
  value.rs              # ValueFunction, ExpectedValueComputer — V(s) approximation
  theoretical_edge.rs   # TheoreticalEdge — expected value from stochastic model
  calibrated_edge.rs    # CalibratedEdge — blends calibration progress into edge
  changepoint.rs        # ChangepointDetector — BOCD for regime shifts
  information.rs        # InformationValue — when to wait vs act
  interface.rs          # ControlInterface trait — L2 to L3 communication
  traits.rs             # ControlSolver, ValueFunctionSolver traits
  position_pnl_tracker.rs # Position-level PnL for control decisions
  simulation.rs         # Path generation for EV computation
  quote_gate.rs         # QuoteGateDecision — whether to quote or skip

stochastic/
  mod.rs                # StochasticControlConfig, presets (blended, thin, liquid)
  beliefs.rs            # MarketBeliefs — NIG for mu/sigma, Gamma for kappa
  conjugate.rs          # Conjugate prior math — online Bayesian updates
  hjb_solver.rs         # HJBSolver — optimal quote derivation
  continuation.rs       # Continuation probability and value of waiting

process_models/
  spread.rs             # SpreadDynamics — bid-ask spread evolution
  funding.rs            # FundingRateModel — perpetual futures funding
  liquidation.rs        # LiquidationCascadeDetector — OI drop detection
  hjb/
    mod.rs              # HJBConfig, HJBInventoryController
    ou_drift.rs         # OUDriftEstimator — OU process drift for momentum
    skew.rs             # Inventory + predictive skew from value gradient
    summary.rs          # HJBSummary — diagnostics output
```

---

## Architecture: 4-Layer System

```
Layer 1: Estimation     Raw data -> kappa, sigma, regime
Layer 2: Learning       Estimates -> Bayesian beliefs, ensemble weights
Layer 3: Control        Beliefs -> optimal actions (THIS SKILL)
Layer 4: Execution      Actions -> order placement
```

The controller receives `LearningModuleOutput` via `ControlInterface` and produces `Action` decisions.

---

## Core Decision Loop

```
a* = argmax_a Q(s, a)
```

- `s` = `ControlState` (position, inventory, time, regime, beliefs)
- `a` = `Action` (Quote with spreads, Wait, Reduce, PullQuotes)
- `Q(s,a)` = expected value of taking action `a` in state `s`

The `OptimalController` evaluates Q for each candidate action and selects the best.

---

## Bayesian Beliefs

### MarketBeliefs (`stochastic/beliefs.rs`)

| Belief | Prior | Updated By | Posterior |
|--------|-------|------------|-----------|
| Drift (mu) | NormalInverseGamma | Price returns | N(mu_n, sigma_n^2/n) |
| Volatility (sigma^2) | InverseGamma | Squared returns | IG(alpha_n, beta_n) |
| Fill intensity (kappa) | Gamma | Fill/no-fill events | Gamma(alpha_n, beta_n) |
| Model weights | Dirichlet | Model performance | Dir(alpha_1..alpha_K) |

**Key insight**: The predictive bias `beta_t = E[mu|data]` is the NIG posterior mean — NOT a heuristic. It automatically goes negative in downturns without threshold logic.

### Conjugate Updates (`stochastic/conjugate.rs`)

Online Bayesian updates, O(1) per observation:
- NIG: observe return `r` -> update sufficient statistics
- Gamma: observe fill/no-fill -> update rate estimate
- Dirichlet: observe model outcomes -> update weights

---

## HJB Solver

### Optimal Quote Derivation (`stochastic/hjb_solver.rs`)

The Avellaneda-Stoikov framework with extensions:

```
Half-spread:     delta* = (1/gamma) * ln(1 + gamma/kappa)
Inventory skew:  skew = gamma * sigma^2 * q * T / 2
Predictive skew: + beta_t / 2
```

Where:
- `gamma` = risk aversion (regime-dependent, varies 5x)
- `kappa` = fill intensity (regime-dependent, varies 10x)
- `sigma` = volatility
- `q` = current inventory
- `T` = time horizon
- `beta_t` = predictive drift from NIG posterior

### HJB Inventory Controller (`process_models/hjb/`)

Value function approach with terminal condition:
```
V(T,x,q,S) = x + q*S - penalty*q^2
```

Backward induction gives optimal quotes at each time step. `ou_drift.rs` handles OU momentum, `skew.rs` computes the V-gradient for inventory skew.

---

## Changepoint Detection

`ChangepointDetector` in `control/changepoint.rs` — Bayesian Online Change Point Detection (BOCD):

- Maintains run-length distribution P(r_t | x_{1:t})
- Detects sudden shifts in market dynamics
- Triggers belief reset: posteriors revert toward priors
- Key for surviving regime transitions (quiet -> cascade)

Flow: Detection -> belief reset -> wider spreads until new regime characterized.

---

## Value of Information

`InformationValue` in `control/information.rs`:

Sometimes optimal action is to **wait** and gather more information:

```
VOI = E[V(act with more info)] - E[V(act now)]
```

If VOI > cost_of_waiting, controller chooses `Action::Wait`.

---

## Action Types

```rust
enum Action {
    Quote { bid_offset_bps: f64, ask_offset_bps: f64, size: f64 },
    Wait,           // Gather more information
    Reduce,         // Reduce existing position
    PullQuotes,     // Cancel all quotes (risk-driven)
}
```

`NoQuoteReason` provides diagnostics when quoting is suppressed.

---

## Quote Gate

`QuoteGateDecision` is the final check before quotes go out:

- Inputs: calibrated edge, risk state, position, regime
- Output: quote with adjusted params, or skip with reason
- Blends calibration progress — `CalibratedEdge` adjusts gamma to be fill-hungry during calibration (low `cal_gamma` = tighter spreads to get fills for learning)

---

## Process Models

### OU Drift (`process_models/hjb/ou_drift.rs`)
- Mean-reversion speed and level estimation
- Used for predictive skew in HJB formula

### Spread Dynamics (`process_models/spread.rs`)
- How bid-ask spread evolves over time
- Wider market spread -> lower fill intensity -> wider optimal quotes

### Funding Rate (`process_models/funding.rs`)
- Near settlement (every 8h), funding creates predictable flow
- Model accounts for this in directional bias

### Liquidation Cascade (`process_models/liquidation.rs`)
- OI drop detection for cascade events
- Feeds into circuit breaker system

---

## Common Debugging

### "Why did it wait instead of quote?"
1. Check `InformationValue` — VOI > cost of waiting?
2. Check `QuoteGateDecision` — calibration progress too low?
3. Check `CalibratedEdge` — theoretical edge negative?
4. Look at `NoQuoteReason` for specific diagnostic

### "Spreads too wide/tight"
1. Check `gamma` — regime-appropriate? (quiet: ~0.1, cascade: ~0.5)
2. Check `kappa` — stale or unreasonable? (must be > 0.0!)
3. Check `sigma` — volatility estimate reasonable?
4. Check inventory skew — large position shifting quotes?

### "Predictive bias seems wrong"
1. Check NIG posterior mean `beta_t`
2. Check changepoint detector — recent reset?
3. Check OU drift estimator — mean-reversion level calibrated?

---

## Key Invariants

1. `kappa > 0.0` — GLFT formula has `ln(1 + gamma/kappa)`, blows up at 0
2. `gamma > 0.0` — risk aversion must be positive
3. `sigma > 0.0` — zero volatility breaks value function
4. Posteriors must stay proper — alpha, beta > 0 for Gamma; alpha_i > 0 for Dirichlet
5. Value function must be concave in inventory

---

## Adding a New Action Type

1. Add variant to `Action` enum in `control/actions.rs`
2. Implement Q-value computation in `OptimalController`
3. Add transition dynamics in `simulation.rs`
4. Wire through `QuoteGateDecision`
5. Handle in orchestrator's action dispatch
6. Add diagnostic logging

---

## Supporting Files

| File | Description |
|------|-------------|
| [references/formula-reference.md](references/formula-reference.md) | Consolidated math reference: GLFT optimal spread, inventory skew, NIG conjugate updates, Gamma fill intensity updates, BOCD run-length equations, value function basis features, and OU drift process -- each with equation, variables, implementation location, and parameter ranges |
