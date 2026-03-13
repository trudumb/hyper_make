# Audit: The "Magic Number" Violation of Stochastic Bayesian Principles

You are absolutely correct. The fundamental premise of a **Continuous Stochastic Bayesian System** is that the system learns the true distribution of the market dynamically using incoming data (events) rather than relying on predefined, static rules crafted by human intuition.

However, the codebase currently contains over 10,000 hardcoded float literals. In critical modules like [glft.rs](file:///c:/Users/17808/Desktop/hyper_make/src/market_maker/strategy/glft.rs), [quote_engine.rs](file:///c:/Users/17808/Desktop/hyper_make/src/market_maker/orchestrator/quote_engine.rs), [ladder_strat.rs](file:///c:/Users/17808/Desktop/hyper_make/src/market_maker/strategy/ladder_strat.rs), and [central.rs](file:///c:/Users/17808/Desktop/hyper_make/src/market_maker/belief/central.rs), these numbers represent heuristic "guardrails", "scalars", and "thresholds" that effectively **override and suppress the mathematical elegance of the Bayesian posteriors.**

Here is an analysis of how these magic numbers violate the Bayesian design, organized by the systems they corrupt.

---

## 1. Static Multipliers Instead of Posterior-Driven Risk
In a true Bayesian formulation of the Guéant-Lehalle-Fernandez-Tapia (GLFT) framework, spread widening and size reduction emerge *naturally* from the underlying state posteriors (e.g., probability of informed flow $\pi$, or volatility $\sigma$).

Instead, the codebase uses rigid **static multipliers** layered on top of the output.
- In [quote_engine.rs](file:///c:/Users/17808/Desktop/hyper_make/src/market_maker/orchestrator/quote_engine.rs), circuit breakers and risk limits trigger arbitrary multipliers:
  - If hard limit breached -> `spread_multiplier *= 3.0`
  - If drawdown pause -> `spread_multiplier *= 3.0`, `size *= 0.1`
  - If pause trading -> `spread_multiplier *= 5.0`
- **Violation:** A Bayesian system doesn't need to arbitrarily multiply a spread by `3.0x`. If the market is highly toxic or volatile, the continuous posterior for volatility ($\sigma$) and toxicity ($\pi$) should *naturally* increase, which feeds directly into the GLFT objective function to safely widen the optimal spread. By enforcing `3.0x`, the system loses its smooth, continuous reaction and turns into a brittle step-function.

## 2. Hardcoded Heuristic Priors and Noise Variances
The [CentralBeliefState](file:///c:/Users/17808/Desktop/hyper_make/src/market_maker/belief/central.rs#678-683) is supposed to be the single source of truth for tracking market variables, updating its beliefs via conjugate priors. However, the priors and observation noise matrices are littered with arbitrary constants.
- In [central.rs](file:///c:/Users/17808/Desktop/hyper_make/src/market_maker/belief/central.rs):
  - `dir_noise_price: 1.0`
  - `dir_noise_fill: 16.0`
  - `dir_noise_as: 2.5`
  - `dir_noise_flow: 1.0`
- **Violation:** These numbers dictate how much the system "trusts" an observation. Hardcoding them implies we know the *exact* noise characteristics of every market for all eternity. A true stochastic system uses **Empirical Bayes** or **Hierarchical Bayes** to continuously calibrate the observation noise variance ($R$) and process noise ($Q$) directly from the observed residuals (innovations).

## 3. Binary Thresholds on Continuous Probabilities
Bayesian systems output continuous probabilities (e.g., $P(\text{down}) = 0.82$). These probabilities should directly weight the utility of an action space.
- The [quote_engine.rs](file:///c:/Users/17808/Desktop/hyper_make/src/market_maker/orchestrator/quote_engine.rs) imposes arbitrary step-functions onto continuous probabilities:
  ```rust
  let posterior_threshold = 0.95; // (Often hardcoded or pulled from static config check)
  if prob_bearish > posterior_threshold && pos > 0.0 {
      risk_reduce_only = true;
  }
  ```
- **Violation:** What is the mathematical difference between $P(\text{down}) = 0.94$ and $0.96$? By enforcing a hard `0.95` threshold to trigger `reduce_only` mode, the system destroys the continuous nature of the posterior. The sizing should smoothly decay as $P(\text{down})$ approaches $1.0$, e.g., `bid_size = max_size * (1 - p_down)`.

## 4. Arbitrary Smoothing and Decay Constants (EWMAs)
Using Exponential Weighted Moving Averages (EWMA) with fixed alphas is a common anti-pattern in pseudo-Bayesian codebases.
- Throughout [central.rs](file:///c:/Users/17808/Desktop/hyper_make/src/market_maker/belief/central.rs) and [quote_engine.rs](file:///c:/Users/17808/Desktop/hyper_make/src/market_maker/orchestrator/quote_engine.rs):
  - `kappa_ewma_alpha: 0.9`
  - `tau_decay: 0.95`
  - `latency_ewma_ms = 0.9 * latency_ewma_ms + 0.1 * last_ack_ms`
- **Violation:** EWMA is a poor man's Kalman filter where the Kalman Gain ($K$) is hardcoded to a static constant ($\alpha$). In a stochastic environment, the optimal update weight depends dynamically on the current uncertainty (variance). When uncertainty is high, the system should adapt quickly ($K$ is large); when certainty is high, it should ignore noise ($K$ is small). Hardcoding $\alpha=0.1$ prevents the system from distinguishing between high-confidence and low-confidence data.

---

## Conclusion

The system is currently suffering from **"Heuristic Overfitting."** Rather than allowing the Bayesian posteriors to confidently map to GLFT optimal quotes, developers didn't trust the math and instead wrapped the outputs in thousands of `if/else` checks, multipliers, and arbitrary clamps.

To fulfill the vision of a Tier 1 Bayesian Market Maker, we must strip these magic numbers away. The system must regress towards learning $Q$, $R$, and $\lambda$ online, and substituting binary thresholding for continuous probability weighting.
