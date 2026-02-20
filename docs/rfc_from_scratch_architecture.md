# RFC: Clean-Slate Foundational Architecture

**Status**: Draft  
**Context**: Designed as a ground-up rebuild assuming the unified spread extraction and adverse selection concepts were foundational assumptions from day one.  
**Team Structure**: Modeled for a dedicated team of 5-6 engineers.

---

## 1. The Core Philosophy

If we had started with the insights of the four recent RFCs (Unified Adverse Selection, Spread Extraction, Proprietary Algorithmic Edges, and the Options-Theoretic Pricing), we would never have built discrete components like `quote_gate`, `position_zones`, or statically-spaced geometric ladders.

We would have started from a single unifying principle: **Every quote is a short option written by the market maker. Its existence on the book is justified solely by a strictly positive Expected PnL (E[PnL]) derived from a continuous, coherent model of latent fair value and risk.**

A clean-slate architecture eliminates the distinction between "pricing" and "protection". Protection is just correct pricing under uncertainty and impact. 

---

## 2. Team Topology & Subsystem Ownership

To build this from scratch, the system is decomposed into 6 distinct subsystems, owned by a team of 6 specialized engineers. The architecture is a directed acyclic graph of state flowing from raw events to optimal control outputs.

### Engineer 1: Microstructure & Event Engine (The "Sensor")
**Responsibility**: Translate raw exchange data into high-resolution, low-latency microstructure features.
- **Micro-Price Estimator**: Computes fair value at sub-tick resolution using volume-weighted mid, book imbalance (BIM), and order flow imbalance (OFI).
- **Latency & Signal Alignment**: Handles cross-venue lead-lag modeling (e.g., BTC to HYPE) and aligns timestamps.
- **Participant Fingerprinting**: Maintains state on L2 diffs to cluster and fingerprint adversarial or informed participants on thin books.
- **Output**: Generates real-time features (`z_obs`, `R_noise`) for the Latent State Filter.

### Engineer 2: Latent State Filter (The "Belief System")
**Responsibility**: Maintain the continuous posterior belief of the market's true state.
- **UKF / IMM Filter**: Replaces the scalar Kalman drift estimator. Maintains a joint distribution over:
  - Latent Fair Value ($V$)
  - Instantaneous Drift ($\mu$)
  - Realized Volatility ($\sigma$)
  - Information State ($I(t)$)
- **Glosten-Milgrom Updates**: Treats our own fills as highly informative adversarial observations, shifting $V$ instantly based on the informed-flow fraction.
- **Output**: The probability distribution of the market state $(V, \mu, \sigma)$, not just point estimates.

### Engineer 3: Risk & Capital Allocation (The "Governor")
**Responsibility**: Determine the system's capacity for risk at any given microsecond.
- **Continuous Inventory Penalty**: Computes the dynamic convex risk aversion $\gamma(q)$, eliminating discrete position zones.
- **Funding Carry Cost**: integrates perpetual swap funding rates continuously into the holding cost of inventory.
- **Growth-Optimal Sizing (Kelly)**: Computes the target portfolio risk budget dynamically. In high-correlation (crisis) regimes, it inherently shrinks the portfolio size $S^*$.
- **Output**: The marginal cost of risk for an additional unit of inventory, and the allowed risk budget.

### Engineer 4: Economic Pricing Engine (The "Actuary")
**Responsibility**: Evaluate the profitability of hypothetical states.
- **Options-Theoretic Pricing**: Evaluates every potential quote as a short put or short call. Computes the required option premium constraint.
- **Endogenous Impact Modeler**: Adjusts expected edge based on the self-impact of our quotes on thin books (forward and round-trip impact).
- **E[PnL] Surface Computation**: Generates a continuous curve of expected profitability across all depths $\delta$, taking into account: `E[PnL] = P(fill) * (Edge - Adverse_Selection - Impact - Carry)`.
- **Output**: A payout surface mapping quote depth to expected marginal PnL.

### Engineer 5: Optimization & Control Orchestrator (The "Executor")
**Responsibility**: Translate economic surfaces into physical limit orders and manage their lifecycle.
- **Constrained Optimizer**: Solves for the subset of quotes that maximizes total $\Theta$ (spread income) subject to a strict $\Gamma$ (adverse selection exposure) limit and the Kelly risk budget. 
- **Adaptive Spacing**: Places quotes at local gaps in the book where marginal fill probability is highest, rather than naïve geometric log-spacing.
- **Staleness & Cycle Manager**: Uses an event-driven loop. Breaks out of the fixed 5s quote cycle if the "freshness" of our quotes drops below a regime-dependent threshold.
- **Output**: L2 order insertions, modifications, and cancellations sent to the exchange.

### Engineer 6: Online Learning & Feedback (The "Evaluator")
**Responsibility**: Ensure the system adapts continuously without manual intervention.
- **Markout Decomposition**: Breaks down historical fills into transient impact (reverting) vs permanent adverse selection.
- **Multiplicative Weights (MW) Learner**: Runs parallel counterfactual simulations of alternative parameter configurations (e.g., modifying base spread or size limits) and continuously redistributes weight to the most profitable configurations.
- **Output**: Slow loop updates to the hyper-parameters configuring Engineers 1-5.

---

## 3. The Data Flow (The "Quote Cycle")

The architecture operates on an interrupt-driven hybrid cycle, breaking away from the strict timer loop.

1. **Event Trigger**: A new L2 update, trade, cross-venue tick, or simply a timeout invokes the cycle.
2. **Feature Update**: Engineer 1's engine computes the instantaneous Micro-Price and the vector of observations $Z$.
3. **Belief Update**: Engineer 2's IMM filter updates the posterior $P(V, \mu, \sigma | Z)$.
4. **Risk Update**: Engineer 3 calculates $\gamma(q)$ and the carry adjusted holding cost.
5. **Pricing**: Engineer 4 evaluates the E[PnL] at a discrete grid of depths for both sides. Any depth where `E[PnL] <= 0` is discarded entirely (naturally replacing the Quote Gate).
6. **Allocation**: Engineer 5 runs the convex optimizer `argmax(Sum(E[PnL])) subject to Gamma <= Limit`, placing quotes at intelligently spaced gaps.
7. **Execution**: Diffs are sent to the exchange.

---

## 4. Why This Architecture Surpasses the Ad-Hoc Evolution

If built this way from the start, the system gains several structural properties:

### A. Smooth Degradation and Scaling
There are no binary cliffs. If inventory $q$ gets dangerously large, $\gamma(q)$ scales quadratically. The E[PnL] computation for adding more inventory simply sinks below zero, and the optimizer allocates zero size to those bids. If funding spikes, carry cost dominates the E[PnL], naturally skewing the book. 

### B. Natural Thin-Book Handling
By explicitly modeling Endogenous Impact in Engineer 4's pricing engine, the system natively handles venues where our flow is 40% of the book. As we become a larger fraction of the book, impact penalty rises, E[PnL] compresses, and the optimizer distributes our size further back from the touch natively. 

### C. True Agility vs Adversaries
Engineer 6's Multiplicative Weights algorithm removes the need for human parameter tuning during a volatile event. By running counterfactuals on every fill, the system shifts its own parameters to adapt to the present reality. Engineer 2's Glosten-Milgrom updates ensure that the instant we take a toxic fill, $V$ adjusts, ensuring the next quote is priced correctly.

---

## 5. Implementation Roadmap (Blank Slate -> Prod)

If a team of 6 were to build this from scratch today:

**Milestone 1: The Sensor & The Belief System (Weeks 1-3)**
- Build the feature pipeline, Micro-Price estimator, and the Latent State Filter.
- Run passively on historical data to tune observation noises and filter covariances.
- *Validation*: Predicted $V$ outperforms Exchange Mid predicting 10s forward mid.

**Milestone 2: The Actuary & The Governor (Weeks 4-6)**
- Implement the continuous risk functions, funding adjustments, and the Options-Theoretic E[PnL] surface.
- *Validation*: Offline E[PnL] models accurately reflect historical markouts of the previous system.

**Milestone 3: The Executor & The Evaluator (Weeks 7-9)**
- Connect the E[PnL] surface to the constrained optimizer. 
- Build the Markout classification and the initial MWA setup.
- Begin shadow trading to validate quote placements against the live book.
- *Validation*: Shadow PnL demonstrates strict adherence to Gamma limits and positive expected edge.

**Milestone 4: Full Loop & Proprietary Plugins (Weeks 10-12)**
- Add the proprietary extensions: Liquidation Frontier Mapping and Information Propagation modeling.
- Live testnet deployments.

---

## 6. Mapping to the Current `hyper_make` Codebase

While this RFC describes a ground-up rebuild, the reality is we must transition the existing `hyper_make` Rust codebase toward this ideal. Here is how the theoretical team topology maps to the physical layout of the current repository, and the gap analysis for each:

### Engineer 1: Microstructure (Sensor) -> `src/market_maker/process_models/` & `src/market_maker/estimator/`
**Current State**: We have `VolumeClock`, `BipowerVariation`, `MicropriceEstimator`, and `Hawkes`.
**The Gap**: We currently treat these as separate estimators that feed into a `ParameterEstimator` struct. To reach the clean-slate ideal, we need to unify these into a structured feature vector that emit standardized observations (`z_obs`, `R_noise`) for the Kalman/IMM filter.

### Engineer 2: Latent State Filter (Belief System) -> `src/market_maker/estimator/` & `src/market_maker/strategy/`
**Current State**: We have a scalar Kalman drift estimator tracking $\mu$ independently. We use exchange mid as ground truth.
**The Gap**: We need to abandon exchange mid as the anchor. The filter must become a joint UKF/IMM that emits $V$, $\mu$, and $\sigma$ simultaneously. We also need to implement the Glosten-Milgrom fill updates (treating our own fills as adversarial observations).

### Engineer 3: Risk & Capital Allocation (Governor) -> `src/market_maker/risk/` & `src/market_maker/strategy/risk_model.rs`
**Current State**: We just removed the discrete binary `quote_gate` and `position_zones` in favor of additive/intensity-based equivalents (SC/AS-ratio). We have a `ContinuousRisk` model warming up.
**The Gap**: The continuous risk penalty is implemented, but we still treat funding as a separate process model rather than explicitly discounting the holding cost of inventory in the primary pricing engine. The Kelly sizing framework also needs to be formally integrated to replace arbitrary size multipliers.

### Engineer 4: Economic Pricing Engine (Actuary) -> `src/market_maker/strategy/glft.rs` & `src/market_maker/adverse_selection/`
**Current State**: `GLFTStrategy` computes optimal half-spreads based on $\kappa$ and $\gamma$.
**The Gap**: The GLFT assumes we are infinitesimal (zero market impact). We need to implement the "Options-Theoretic Pricing" constraint (`min_half_spread >= Option Floor`). We also need to inject endogeous impact logic directly into the $E[PnL]$ evaluation for thin HIP-3 books natively, rather than as a secondary adjustment.

### Engineer 5: Optimization & Orchestrator (Executor) -> `src/market_maker/quoting/` & `src/market_maker/tracking/`
**Current State**: We use a `TickGrid` ladder strategy with utility-weighted softmax allocation (seen in the logs). The orchestrator handles reconciliation via `OrderManager`.
**The Gap**: The execution loop still largely runs on a fixed timer (`timer(100ms) => quote cycle`). It needs to move to a fully state/event-driven "freshness" model where quotes are reprieved exactly when staleness crosses a threshold, saving rate limits for high-volatility events.

### Engineer 6: Online Learning (Evaluator) -> `src/market_maker/control/` & Analytics
**Current State**: The system logs changepoints ("Changepoint confirmed - soft resetting beliefs" seen in logs) but does little structural adaptation beyond resetting memory.
**The Gap**: We need the Multiplicative Weights (MW) learner running counterfactual simulations over historical/recent fills to continuously tune the hyperparameters without manual intervention.
