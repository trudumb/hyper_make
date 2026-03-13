# Implementation Plan: Replacing Heuristics with Continuous Bayesian Posteriors

## Goal Description
The objective is to strip out the arbitrary "magic numbers" (hardcoded multipliers, rigid thresholds, static EWMA alphas, and manually tuned priors) that currently circumscribe the Bayesian posteriors in the market maker system. We will replace them with properly continuous, mathematically sound stochastic functions. This migration shifts the system from relying on human heuristics to dynamically characterizing the true structure of the market using Empirical/Hierarchical Bayes.

## User Review Required
> [!IMPORTANT]
> The transition away from step functions (like an immediate limit breach `3.0x` spread penalty) to smooth tracking equations means that circuit breakers and risk governors will react gradually. Please verify that this approach aligns with your expectations for absolute downside protection. 

## Proposed Changes

### Central Belief State (Dynamic Calibration)
We will remove the manually configured parameters for observation and process noise, replacing them with dynamic online estimates.

#### [MODIFY] [src/market_maker/belief/central.rs](file:///c:/Users/17808/Desktop/hyper_make/src/market_maker/belief/central.rs)
- **Replace Static Priors:** Remove `dir_noise_price`, `dir_noise_fill`, `dir_noise_as`, etc. Implement an Empirical Bayes updater to calibrate the observation noise matrices ($R$) from the trailing variance of the Kalman innovations.
- **Replace Static EWMAs:** Rather than using hardcoded $\alpha$ rates (e.g. `0.9`), implement a time-varying Kalman Gain update using $P_{t|t-1}$ for systems like the [kappa](file:///c:/Users/17808/Desktop/hyper_make/src/market_maker/belief/central.rs#1830-1863) estimator and latency trackers.
- **Learn Process Noise:** Track $Q$ (process noise) through autocorrelation timescales rather than presetting rigid decay factors.

---
### Quote Engine (Sizing & Spreads)
We will remove the step-functions applied to pricing logic.

#### [MODIFY] [src/market_maker/orchestrator/quote_engine.rs](file:///c:/Users/17808/Desktop/hyper_make/src/market_maker/orchestrator/quote_engine.rs)
- **Continuous Probabilistic Sizing:** Remove the hardcoded thresholding for directional bias: 
  ```rust
  // Old: if prob_bearish > 0.95 { risk_reduce_only = true; }
  ```
  Instead, utilize the continuous posterior for sizing reductions:
  ```rust
  // New: bid_size = max_bid_size * (1.0 - posterior_p_down)
  //      ask_size = max_ask_size * posterior_p_down
  ```
- **Continuous Risk Multipliers:** Eliminate layered multipliers like `spread_multiplier *= 3.0` and `size_reduction *= 0.1` inside safety governors. Modulate the risk-aversion penalty parameter ($\gamma$) natively using the portfolio HARA utility function relative to structural drawdown.

---
### GLFT Core Logic
Ensure the GLFT solver trusts the derived probabilities exclusively.

#### [MODIFY] [src/market_maker/strategy/glft.rs](file:///c:/Users/17808/Desktop/hyper_make/src/market_maker/strategy/glft.rs)
- **Refactor [expected_pnl_bps_enhanced](file:///c:/Users/17808/Desktop/hyper_make/src/market_maker/strategy/glft.rs#225-289):** Remove legacy continuation multipliers mapped by linear interpolations. Allow the `posterior_predictive_skew` logic to be the sole modulator of inventory penalties, governed entirely by $\mu$ and $\sigma^2$ variances.

---
### Ladder Allocation
Remove artificial clamps from the order utility distributions.

#### [MODIFY] [src/market_maker/strategy/ladder_strat.rs](file:///c:/Users/17808/Desktop/hyper_make/src/market_maker/strategy/ladder_strat.rs)
- Remove scalars such as `MAX_SINGLE_ORDER_FRACTION` which prevent the ladder allocator from placing meaningful quotes. Rely on the marginal utility of each depth level derived directly from `feature_store` posteriors to allocate the optimal liquidity dynamically through softmax gradients.

## Verification Plan

### Automated Tests
- Run `cargo test --all-features` to ensure removal of static arrays and variables does not break parameter aggregators.
- Run `cargo clippy` to ensure all unused `const` primitives have been safely removed.

### Mathematical Verification
- Implement unit tests around the new Empirical Bayes learning state to verify that the observation noise $R$ correctly converges to the sample variance of an injected synthetic signal.
- Verify through simulation that quote sizing scales linearly and correctly in response to arbitrary combinations of $P(\text{bull})$ and $P(\text{bear})$ outputs, avoiding any hard `0.0` or `1.0` walls until probability constraints are met.
