# RFC Supplement: Integrating Proprietary Models into the Clean-Slate Architecture

**Status**: Draft  
**Context**: An in-depth mapping of the "Proprietary Algorithmic Edges" (from `rfc_proprietary_spread_extraction.md`) into the 6-Engineer Clean-Slate Architecture detailed in `rfc_from_scratch_architecture.md`.

---

## 1. The Goal of Proprietary Modeling

As defined in the proprietary spread extraction RFC, standard models like the GLFT or basic Kalman filters are highly public and commoditized. True alpha on thin, transparent venues (like HIP-3 assets on Hyperliquid) comes from modeling structure that competitors ignore:
- Observable liquidation cascades
- Information propagation delays across venues
- Adversarial participant fingerprinting
- Endogenous market impact

In the ad-hoc evolution of `hyper_make`, these were treated as loosely coupled "extensions" or overrides. In the **Clean-Slate Architecture**, they are not bolt-on features; they are foundational pipelines owned by specific engineering subsystems.

---

## 2. Mapping Proprietary Algorithms to the Clean-Slate Subsystems

### A. Adversarial Participant Fingerprinting & Information Propagation
**Owned by: Engineer 1 (Microstructure & Event Engine)**

Instead of just processing raw trades and book updates, Engineer 1 treats the market as a multiplayer game with identifiable actors.

1. **Information Propagation Model (`I(t)`)**: 
   When a Binance BTC move occurs, Engineer 1 does not just pass a price shift to the filter. It maintains the continuous function `I(t)`, broadcasting the percentage of information currently incorporated by the HIP-3 market. This allows the system to know exactly *where* it is in the propagation wave (e.g., "we are at `I=0.3`, 70% of the move is still coming").
   
2. **Participant Fingerprinting**: 
   Engineer 1 clusters L2 diffs in real-time to identify the 3-5 active MMs on the thin book. It tracks their quote cycle timing and withdrawal behavior. 
   
**Output to the system:** Instead of generic "volatility" or "imbalance" features, Engineer 1 outputs structured semantic events: `Participant_B_Withdrawing_Liquidity` or `Information_Wave_Early_Phase`.

### B. Liquidation Frontier Mapping
**Owned by: Engineer 2 (Latent State Filter) & Engineer 1**

The risk of a mass liquidation event is not a localized spread anomaly; it fundamentally alters the underlying state distribution of the market.

1. **Density Tracking (Engineer 1)**: 
   Engineer 1 tracks Open Interest (OI) accumulation and computes the `ρ_long(p)` and `ρ_short(p)` liquidation density functions.
   
2. **State Filter Integration (Engineer 2)**: 
   Engineer 2's IMM (Interacting Multiple Model) filter normally tracks steady-state drift and volatility. When the current price approaches a dense Liquidation Cluster computed by Engineer 1, Engineer 2 dynamically shifts probability mass into a "Crisis / Cascade Regime" model. 
   
**Output to the system:** Engineer 2 outputs a highly non-Gaussian probability distribution for the Latent Fair Value ($V$), heavily skewed toward the liquidation threshold.

### C. Endogenous Impact & Volatility Surface Extraction
**Owned by: Engineer 4 (Economic Pricing Engine)**

In the old system, GLFT computed an optimal spread, and we had to manually wide/tighten based on heuristics. In the clean slate, Engineer 4 treats every quote as a short option (put/call) and prices it economically.

1. **Endogenous Impact**: 
   Because books are thin, your own fills move the market. Engineer 4 calculates `SELF_IMPACT(size)` and `ROUND_TRIP_IMPACT(size)` natively. A massive bid at the touch generates so much self-impact that its `E[PnL]` instantly drops below zero, naturally preventing concentration.
   
2. **Volatility Edge**: 
   Engineer 4 computes the Black-Scholes implied volatility of our own resting quotes (`σ_implied = spread / (C × sqrt(τ))`). It strictly enforces `σ_implied > σ_realized`. If a proprietary algorithm from Engineer 1 (like the Information Propagation wave) causes `σ_realized` to spike, Engineer 4 immediately forces spreads wider to maintain positive vol-edge.

**Output to the system:** A continuous `E[PnL]` surface across all depths that natively incorporates the cost of our own market impact and strictly enforces positive option premium.

### D. Funding Settlement Microstructure Exploitation
**Owned by: Engineer 3 (Risk & Capital Allocation)**

Funding is not a random variable; it is a deterministic clock that forces participant behavior. 

1. **Continuous Carry Cost**: 
   Engineer 3 computes the instantaneous holding cost of inventory incorporating the continuous funding rate. 
   
2. **Directional Bias Ahead of Settlement**: 
   As the hourly settlement approaches, Engineer 3 shifts the system's target inventory (the Kelly allocation). If longs are paying shorts heavily, Engineer 3 aggressively shifts the target inventory to short, accepting tighter asks and wider bids to absorb the predictable pre-settlement selling pressure.

**Output to the system:** A dynamic risk aversion curve `γ(q)` and target portfolio skew that anticipates temporal market structure rather than just reacting to static positions.

---

## 3. The Quote Cycle in Action (Proprietary Flow)

Here is how a single "Proprietary Edge" event flows through the clean-slate DAG:

1. **The Event:** Bitcoin spikes $500 on Binance.
2. **Engineer 1 (Sensor):** Detects the cross-venue move. Initializes `I(t) = 0.05` (early propagation). Detects Participant C (a known fast MM) pulling their bids.
3. **Engineer 2 (Filter):** Receives the `I(t)` early signal and Participant C's withdrawal. Instantly shifts the posterior distribution of $V$ upward, increasing the uncertainty ($\sigma$).
4. **Engineer 3 (Risk):** Funding is currently neutral, no major adjustments to risk budget $S^*$, but $\gamma(q)$ remains strict.
5. **Engineer 4 (Pricing):** Evaluates the new E[PnL] surface. Because uncertainty went up, the required option premium increases. The bids (wrong side) instantly show deeply negative E[PnL]. The asks (correct side) show massive immediate profitability due to the delayed HYPE price.
6. **Engineer 5 (Executor):** Runs the convex optimizer. Allocates 0 size to bids. Concentrates available Kelly budget on aggressive asks to capture the incoming toxic flow from slower participants catching up to the Binance move.
7. **Engineer 6 (Evaluator):** Post-event, measures the markout of fills taken during this spike. Uses Multiplicative Weights to slightly tune the `latency_delay` assumptions for the Information Propagation model for the next event.

---

## 4. Summary

Proprietary modeling is the difference between a bot that survives and a bot that prints. 
By architecting the system as a clean-slate DAG of specialized engineers, these complex behaviors (like Liquidation Mapping and Participant Fingerprinting) are no longer hacky overrides applied *after* pricing. They are the fundamental data inputs that *drive* the pricing and execution constraints from the very first microsecond.
