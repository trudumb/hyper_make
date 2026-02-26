# Architecture Audit: Principled Stochastic Bayesian Market Making

## Executive Summary

This audit evaluates the current system architecture against the stated goal of **principled stochastic Bayesian modeling** — where all quoting decisions emerge from measured quantities, Bayesian posteriors, and the Avellaneda-Stoikov (AS) framework, with zero arbitrary tuning constants. The system has made significant progress toward this ideal, but several structural tensions remain that compromise the principled foundation.

**Overall Assessment: 65% principled, 35% ad-hoc or structurally conflicted.**

The core GLFT pricer, Kalman drift estimator, and log-additive risk model represent genuine theoretical advances. However, the architecture suffers from *layering conflicts* where multiple subsystems compete to control the same degree of freedom, *measurement gaps* where calibrated quantities fall back to hardcoded constants, and one critical *correctness bug* (the inventory skew magnitude issue that caused the 2024-02-24 session loss).

---

## 1. The Three Estimators: μ, σ², γ

### 1.1 Drift Estimator (μ) — Grade: B+

**What's principled:**
The `KalmanDriftEstimator` uses proper Ornstein-Uhlenbeck dynamics with Kalman predict/update:

```
dμ = -θ·μ·dt + σ_μ·dW
Predict: μ̂⁻ = μ̂·exp(-θΔt), P⁻ = P·exp(-2θΔt) + Q(Δt)
Update:  K = P⁻/(P⁻+R), μ̂ = μ̂⁻ + K(z-μ̂⁻), P = (1-K)P⁻
```

This is textbook correct. OU mean-reversion replaces the old ad-hoc 0.9× decay. Posterior variance `P` naturally bounds the estimate without artificial caps. The online parameter adaptation (updating θ and process_noise from realized drift surprises) is a genuinely principled Bayesian approach.

**Structural concerns:**

1. **Feature variance constants are priors, not measurements.** `BASE_MOMENTUM_VAR = 100.0`, `BASE_TREND_VAR = 200.0`, `BASE_LL_VAR = 50.0` etc. are labeled as "CALIBRATION TARGET: replace with empirical MSE from paper trading." Until these are measured, the relative weighting of signals entering the Kalman filter is arbitrary. This is the single highest-impact calibration task for drift accuracy.

2. **P_MIN = 2.0 is a hardcoded floor.** The comment says "prevents overconfidence when many features feed the filter" — this is reasonable in principle but the value is not derived from any measurement. A principled floor would be the irreducible prediction variance from the markout engine.

3. **Autocorrelation warmup prior (0.3) washes out quickly** but during the first 20 fills it dampens fill impact by a fixed amount. A hierarchical prior on autocorrelation (learned from historical sessions) would be more principled.

4. **The 2024-02-24 session showed drift_adj_bps = -1.7 vs actual trend of +38 bps.** This 4% response is not a Kalman filter bug per se — it's a consequence of high posterior variance (uncertainty ~3.5 bps) relative to the signal. The fix is faster signal injection (more observations per unit time) or lower observation noise R, not a different filter.

### 1.2 Covariance Estimator (σ²) — Grade: B

**What's principled:**
- Bayesian posterior correction via `sigma_correction_factor` (realized vol tracker) — when realized > predicted, sigma inflates.
- Particle filter volatility (`sigma_particle`) provides regime-aware estimation with credible intervals.
- Vol term structure ratio (`σ_short / σ_long`) captures volatility expansion/contraction.
- Leverage-adjusted sigma incorporates asymmetric vol during down moves (ρ < 0).

**Structural concerns:**

1. **Three competing sigma sources.** The aggregator chooses between `sigma_particle`, `sigma_leverage_adjusted`, and `sigma_clean()` based on confidence thresholds (flow_decomp_confidence > 0.3). This priority chain is sensible but the thresholds are not derived from prediction accuracy. A proper model averaging approach (BMA or precision-weighted combination) would be more principled.

2. **sigma_for_skew has a hardcoded floor of 0.0001 (1 bp/sec).** The comment says "provides meaningful skew even with small positions" — this reveals an architectural problem. If sigma is genuinely near zero (ultra-calm market), the correct action is near-zero inventory skew because inventory risk is near zero. The floor was added to compensate for the skew magnitude bug, not because it's theoretically justified.

3. **The CovarianceTracker's correction factor replaces the old staleness/cascade defense addons.** This is a genuine improvement — σ feedback is the correct mechanism. But the correction factor itself needs monitoring: if it's persistently > 1.5 or < 0.7, the base sigma model is miscalibrated.

### 1.3 Risk Aversion Estimator (γ) — Grade: B-

**What's principled:**
The log-additive `CalibratedRiskModel` is a significant advance:

```
log(γ) = log(γ_base) + Σ βᵢ × xᵢ
γ = exp(log_gamma).clamp(γ_min, γ_max)
```

This prevents multiplicative explosion and the sigmoid "skeptic" regularization (`tanh(raw_sum / max_contrib)`) bounds gamma inflation regardless of how many features fire. The feature set (volatility, toxicity, inventory, Hawkes, book depth, uncertainty, confidence, cascade, tail risk) maps cleanly to observable market conditions.

**Structural concerns:**

1. **Beta coefficients are not calibrated from data.** The defaults (`beta_volatility = 1.0`, `beta_toxicity = 0.5`, `beta_cascade = 1.2`, etc.) and the conservative warmup variants (1.5×) are heuristic priors. The system has calibration infrastructure (markout engine, gamma_calibration.jsonl) but the regression hasn't been run. This is the second-highest priority calibration task.

2. **beta_inventory = 0.0 (DISABLED).** The comment says "inventory scaling handled by continuous γ(q) quadratic in glft.rs." This creates a *dual pathway* for inventory risk:
   - Path A: `CalibratedRiskModel` → log-additive gamma (beta_inventory disabled)
   - Path B: `effective_gamma()` → quadratic `γ(q) = γ_base × (1 + β × u²)`
   
   Path B is applied *after* the CalibratedRiskModel computes gamma_base. The quadratic sits in `effective_gamma()` in glft.rs. This means inventory risk enters gamma through a multiplicative post-process, not through the principled log-additive model. The theoretically correct approach is to re-enable beta_inventory in the risk model and remove the quadratic overlay — or prove they're equivalent (they're not, because log-additive and multiplicative scaling behave differently at extremes).

3. **`effective_gamma()` applies multiple post-processes:**
   - Quadratic inventory scaling: `1 + β × u²`
   - Drawdown multiplier: `1 + drawdown_frac × 2.0`
   - Regime gamma multiplier (from HMM blending)
   - Ghost liquidity multiplier
   
   These are all *multiplicative* on top of the log-additive model output. This violates the principled architecture: gamma should be computed in one place (the risk model) with all features entering log-additively. The post-processes are legacy remnants that should be refactored into additional betas.

4. **gamma_min = 0.05, gamma_max = 5.0 are hardcoded bounds.** In a fully principled system, gamma should be unbounded (or bounded only by numerical stability). If gamma hits the ceiling, it means the risk model is saying "conditions are extremely dangerous" and the system should respect that by quoting extremely wide or not quoting at all.

---

## 2. The GLFT Pricer

### 2.1 Half-Spread — Grade: A-

The core formula is correct:

```
δ = (1/γ) × ln(1 + γ/κ)
```

And the system correctly routes all risk factors through γ and κ rather than using arbitrary spread multipliers. The E[PnL] per-level filter is a genuine advance over binary quote gates — it drops levels where expected profit is negative, which is the economically correct decision.

**Concern:** The spread inflation cap at `4× GLFT optimal` (min 15 bps) is a safety guardrail, but it means the system cannot express extreme risk through spreads alone. If γ × σ² × T produces a 50 bps half-spread, the 4× cap at ~25 bps truncates the principled output.

### 2.2 Reservation Price / Inventory Skew — Grade: C

**This is the system's most critical weakness.**

The unified reservation price (WS3) correctly combines three components:

```
total_shift = drift - inv_penalty - funding_carry
inv_penalty = (q/Q_max) × γ × σ² × T
```

But the *magnitude* of inventory skew has been problematic:

1. **The 2024-02-24 session had skew_bps = 0.00 the entire time.** The post-mortem identifies this as "THE bug." With -8.65 short in a +38 bps uptrend, symmetric quotes are catastrophically wrong.

2. **Position amplifiers (5× and 10× for small positions) are unprincipled.** The code applies hardcoded multipliers when inventory_ratio is in (0.01, 0.1). This compensates for the fact that `q × γ × σ² × T` produces tiny skew for small positions — but the fix should be to ensure γ, σ, T are correctly calibrated, not to add arbitrary amplifiers.

3. **The 95% half-spread clamp bounds the reservation mid.** This is a safety measure to prevent BBO crossing, but it means that in extreme situations (large position + strong trend + high vol), the system cannot express the full urgency of its inventory problem.

4. **The Position Continuation Model (HOLD/ADD/REDUCE) overrides the AS inventory skew.** When HOLD fires, `effective_inventory_ratio = raw_ratio` (passes through), but when the old HOLD behavior was `0.0` (no skew), it created the exact problem seen in the session. The current code passes `raw_inventory_ratio` for HOLD, which is better but still conflates two separate concerns: (a) whether to mean-revert inventory, and (b) how much skew to apply. These should be independent: the Bayesian continuation probability should modulate γ (risk aversion about this position), not zero out the skew.

### 2.3 The Position Continuation Model Conflict — Grade: D

This subsystem represents the deepest architectural tension in the system.

**The problem it solves is real:** Pure AS mean-reversion exits positions prematurely. If you're long because you correctly predicted the trend, GLFT skew will aggressively try to flatten you.

**The solution is structurally wrong.** The `PositionDecisionEngine` decides HOLD/ADD/REDUCE and then *transforms the inventory_ratio* fed to GLFT:

```
HOLD:   q_eff = q_raw          (pass through — natural γσ²qτ skew)
ADD:    q_eff = (1-kelly) × q  (reduce skew toward zero)
REDUCE: q_eff = urgency × q   (amplify skew)
```

This is problematic because:

1. **It modifies a measured quantity (inventory) rather than a model parameter.** The principled approach is to keep q factual and modulate γ. High continuation confidence → lower γ → tighter spreads → more two-sided quoting. Low confidence → higher γ → wider spreads → aggressive mean-reversion. The system already has `beta_confidence` (negative) in the risk model for exactly this purpose, but Position Continuation operates on a *different* pathway that bypasses the risk model.

2. **The Beta-Binomial posterior for continuation probability is thin.** It resets on regime change, decays with a 20-fill half-life, and fuses with BOCD/momentum/trend/HMM signals. The fusion is pragmatic but the weights aren't derived from prediction accuracy — they're hardcoded in `ContinuationPosterior`.

3. **HOLD/ADD/REDUCE is a discrete action space imposed on a continuous control problem.** The AS framework already produces continuous skew from continuous inputs. Discretizing the action into three buckets and then applying different transformations creates discontinuities in the quoting surface that takers can exploit.

**Recommended fix:** Eliminate the HOLD/ADD/REDUCE action space. Instead, feed `p_continuation` and `continuation_confidence` as features into the `CalibratedRiskModel`, where they modulate γ continuously. High continuation + high confidence → lower γ → the GLFT formula naturally produces less aggressive mean-reversion without any special-casing.

---

## 3. Signal Architecture

### 3.1 Signal Integration — Grade: B

The `SignalIntegrator` correctly separates concerns:
- Lead-lag (cross-venue Binance signal)
- Informed flow decomposition (p_informed, p_noise, p_forced)
- Regime-conditioned kappa
- Model gating

The `UnifiedSkew` struct correctly prevents triple-counting of inventory skew (a historical bug). The CUSUM predictive lead-lag is a sound addition.

**Concern:** The `IntegratedSignals` struct has ~20 fields that get individually wired through `MarketParams` to the ladder strategy. This creates a wide coupling surface. A more principled architecture would have the signal integrator output exactly three quantities: μ_adjustment, σ²_adjustment, and κ_adjustment — the three AS inputs. Everything else is a derived quantity.

### 3.2 Belief System — Grade: B+

The centralized `BeliefSnapshot` (Normal-Inverse-Gamma posterior for drift/vol, Gamma posterior for kappa) is architecturally sound. The deprecation of the scattered `beliefs_builder` fallback in favor of a single source of truth is the right direction.

**Concern:** The "use centralized beliefs when available, else fall back" pattern appears ~10 times in the aggregator. This should be a single resolution function, not a repeated conditional. The risk is that one fallback path diverges from the primary path silently.

---

## 4. Protection and Safety

### 4.1 What's Been Correctly Eliminated

The codebase shows extensive cleanup:
- `zone_size_mult` REMOVED — inventory penalty captured by gamma
- `cascade_size_reduction` REMOVED — handled by beta_cascade and Kelly
- `pre-fill AS multipliers` REMOVED — E[PnL] filter handles AS
- `stochastic_spread_multiplier` REMOVED — uncertainty flows through gamma
- `staleness_addon_bps` REMOVED — latency-aware mid handles displacement
- `flow_toxicity_addon_bps` REMOVED — beta_toxicity in risk model

This is excellent architectural discipline. Each removal has a comment explaining the principled replacement.

### 4.2 What Remains as Ad-Hoc

1. **Governor bid/ask addon (bps)** — API rate limit protection. This is infrastructure, not market risk, so additive bps is defensible.

2. **Cascade bid/ask addon (bps)** — Fill cascade tracker. This overlaps with `beta_cascade` in the risk model. If beta_cascade is working correctly, the cascade addon is double-counting.

3. **Self-impact addon (bps)** — `coefficient × (our_fraction)²`. This is a principled model of market impact, correctly additive.

4. **Ghost liquidity gamma multiplier** — When book kappa >> robust kappa, inflate γ. This should be in the risk model as a feature, not a post-process multiplier.

5. **Kill-switch side clearing at 90% utilization** — This is a safety guardrail, not a model parameter. Defensible as-is.

6. **Spread inflation cap at 4× GLFT optimal** — Safety guardrail but truncates the principled output.

### 4.3 Cascade Defense Gap

The 2024-02-24 post-mortem identified that after 7 same-side fills in <1 second, the system widened for 15-30 seconds then returned to normal. This is because `beta_cascade` in the risk model decays on a timescale that doesn't account for the *elevated probability of a second sweep*. The principled fix is a Hawkes self-exciting process for cascade arrival intensity — after a cascade, the arrival rate of another cascade is elevated for minutes, not seconds. This should enter as a time-varying κ_cascade that reduces fill intensity assumptions.

---

## 5. Multi-Asset Extension Readiness

The system is currently single-asset. The multi-asset theory from the reference document (Guéant-Lehalle extensions, vector inventory, portfolio variance σ²_port = q^T Σ q) requires:

1. **Shared covariance matrix Σ** — Currently no cross-asset covariance estimation exists. This is a greenfield implementation.

2. **Vector reservation price** — The current `total_shift_bps` is a scalar. Extending to `r_i ≈ S_i - γ × ∂θ/∂q_i` requires solving coupled ODEs or using the closed-form approximations from the 2018 arXiv paper.

3. **Portfolio-level gamma** — Currently gamma is asset-specific. Multi-asset requires a single γ applied to portfolio risk q^T Σ q, which means cross-asset fills update gamma for all assets.

4. **Shared margin on Hyperliquid** — The `MarginAwareSizer` currently computes per-asset capacity. Extending to shared margin requires a portfolio-level margin calculator that accounts for cross-margining benefits.

**Assessment:** The single-asset architecture is *not* trivially extensible to multi-asset. The reservation price, gamma computation, and margin allocation would all need significant refactoring. However, the principled foundation (Bayesian estimators feeding AS pricer) is the right starting point — the theory generalizes cleanly even if the code doesn't yet.

---

## 6. Priority Recommendations

Ranked by impact on achieving principled stochastic Bayesian modeling:

### P0: Fix Inventory Skew Magnitude
The skew_bps = 0.00 bug is existential. Ensure `q × γ × σ² × T` produces correct magnitude by:
- Verifying γ, σ, T are all non-zero and correctly calibrated
- Removing the position amplifiers (5×, 10×) in favor of correct base quantities
- Adding a diagnostic alert when position > 10% of max but |skew| < 1 bps

### P1: Calibrate Risk Model Betas from Data
Run the regression on `gamma_calibration.jsonl` to replace heuristic betas with measured coefficients. This single change upgrades the entire risk model from "plausible priors" to "empirically calibrated."

### P2: Collapse Multiplicative Post-Processes into Log-Additive Model
Move drawdown, regime, ghost liquidity, and quadratic inventory scaling into the `CalibratedRiskModel` as additional beta features. Eliminate the multiplicative `effective_gamma()` overlay. Result: one gamma computation, fully log-additive, no post-processes.

### P3: Eliminate Position Continuation as Discrete Action Space
Feed `p_continuation` and `continuation_confidence` into the risk model as features (via beta_continuation). Remove HOLD/ADD/REDUCE entirely. The GLFT formula with properly calibrated γ already produces the correct behavior: high confidence in position → lower γ → less aggressive mean-reversion.

### P4: Calibrate Drift Signal Variances
Replace `BASE_MOMENTUM_VAR`, `BASE_TREND_VAR`, `BASE_LL_VAR`, `BASE_FLOW_VAR`, `BASE_BELIEF_VAR` with empirical MSE from the markout engine. This determines the relative weighting of all signals entering the Kalman filter.

### P5: Resolve Cascade Addon Double-Counting
If `beta_cascade` in the risk model is working, remove the additive `cascade_bid/ask_addon_bps`. If it's not working, fix it and then remove the addon. Don't maintain both.

### P6: Hawkes Cascade Arrival Model
Replace the fixed-window cascade detector with a proper Hawkes self-exciting process. This gives a time-varying cascade intensity that decays on the correct timescale (minutes, not seconds).

---

## 7. Scorecard

| Component | Principled | Ad-Hoc | Key Gap |
|---|---|---|---|
| Kalman Drift (μ) | ✅ OU dynamics, posterior bounds | ⚠️ Signal variances not calibrated | Feature MSE regression |
| Covariance (σ²) | ✅ Posterior correction, particle filter | ⚠️ Hardcoded sigma floor | Remove 0.0001 floor |
| Risk Aversion (γ) | ✅ Log-additive model | ❌ Multiplicative post-processes | Collapse into betas |
| GLFT Half-Spread | ✅ Correct formula | ⚠️ 4× inflation cap | Make cap adaptive |
| Reservation Price | ✅ Unified three-component | ❌ Skew magnitude bug | P0 fix |
| Position Continuation | ⚠️ Beta-Binomial posterior | ❌ Discrete action space | Eliminate, use γ |
| Signal Integration | ✅ Separated concerns | ⚠️ Wide coupling surface | Reduce to 3 AS inputs |
| Belief System | ✅ NIG/Gamma posteriors | ⚠️ Fallback sprawl | Single resolution fn |
| Protection/Safety | ✅ Most removed correctly | ⚠️ Cascade addon overlap | Remove redundancy |
| Multi-Asset | ❌ Not implemented | — | Greenfield work |

---

## 8. Theoretical Purity Test

For each quoting decision, ask: **"Can I trace this number back to a measured quantity or a Bayesian posterior?"**

| Parameter | Traceable? | Source |
|---|---|---|
| Half-spread δ | ✅ Yes | `(1/γ) × ln(1 + γ/κ)` |
| Reservation mid shift | ⚠️ Mostly | Drift + inv_penalty + funding, but drift signal variances are priors |
| Gamma (γ) | ⚠️ Partially | Log-additive betas are priors, not calibrated; plus multiplicative overlays |
| Kappa (κ) | ✅ Yes | Fill-rate estimation, coordinator chain |
| Sigma (σ) | ✅ Yes | Particle filter / realized vol tracker |
| Inventory skew | ❌ No | Position amplifiers (5×, 10×), continuation overrides |
| Level sizes | ✅ Yes | Entropy-constrained optimizer, Kelly criterion |
| E[PnL] filter | ✅ Yes | Per-level expected profit from measured quantities |

The system is close. Fixing the ❌ and ⚠️ items above would bring it to ~90% principled.
