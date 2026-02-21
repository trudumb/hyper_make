Plan: Bayesian Inference Chain for Fill Rate, Vol Debiasing, Spread Floor, and Stuck Inventory

 Context

 From a live HYPE session: 13 bps total spread against 16.23 bps realized AS = negative EV. Quote outcome model data-starved (60 fills / 8 bins). Vol estimate biased upward by fill-conditioned sampling.  
 Kill switch has no concept of un-unwindable position.

 Key insight: Q17-Q18-Q19 form a connected inference chain: fill rate posterior (Q17) → vol debiasing via importance weighting (Q18) → AS-derived spread floor (Q19). Implement sequentially with explicit  
 coupling.

 ---
 Q17: Beta-Posterior Fill Rate with Hierarchical Shrinkage

 Problem: Point estimation from counts with hard coarse/fine switching. Need proper uncertainty quantification.

 Fix: Replace raw counts with Beta posteriors per bin. Hierarchical model: coarse bins provide the prior for fine bins, which shrink toward coarse rate when data-sparse and converge to their own rate as  
 evidence accumulates.

 Design:

 Each SpreadBin stores Beta(α, β) instead of (fills, total):
 struct SpreadBin {
     lo_bps: f64,
     hi_bps: f64,
     alpha: f64,  // pseudo-count of fills (prior + observed)
     beta: f64,   // pseudo-count of misses (prior + observed)
 }

 Hierarchical structure:
 - Coarse layer (4 bins: [0,5), [5,10), [10,20), [20,∞)): aggregated Beta posteriors
 - Fine layer (8 bins): each bin's prior is the coarse parent's posterior
 - No hard switching threshold — shrinkage is continuous via the prior strength

 On each record(spread_bps, filled):
 1. Update fine bin: if filled { alpha += 1 } else { beta += 1 }
 2. Update coarse parent similarly
 3. Fine bin's effective posterior: Beta(α_fine + w*α_coarse, β_fine + w*β_coarse) where w = prior_weight / (1 + n_fine/τ) shrinks toward zero as fine data accumulates. τ controls shrinkage rate (~50     
 obs).

 Public API changes:
 - fill_rate_at(spread_bps) -> FillRateEstimate (not Option<f64>)
 struct FillRateEstimate {
     mean: f64,        // posterior mean = α/(α+β)
     variance: f64,    // posterior variance = αβ/((α+β)²(α+β+1))
     n_observations: u64,  // for diagnostics
 }
 - fill_rate_at() always returns a value (never None) — weak prior gives uncertain but usable estimate even with 0 observations
 - Consumers (spread bandit, E[PnL]) can use both mean and variance for Kelly-style uncertainty-penalized decisions

 Initial prior: Beta(1, 1) (uniform) for coarse bins. Fine bins inherit from coarse parent.

 Caveat — variance in the transition regime: The blended posterior Beta(α_fine + w·α_coarse, β_fine + w·β_coarse) is a heuristic, not a proper conjugate update. The variance formula αβ/((α+β)²(α+β+1))    
 will understate true uncertainty when w is significant because it treats blended pseudo-counts as real observations. Correct in both limits (pure coarse at n_fine=0, pure fine at n_fine>>τ), but
 slightly optimistic at ~20-80 fine observations. Mitigation: add a blend_active: bool flag to FillRateEstimate and a variance_adjusted(&self) -> f64 convenience method that applies 1.5x multiplier when  
 blend_active. Consumers use variance_adjusted() instead of raw variance — single point of logic, not reimplemented per consumer. Do NOT over-trust variance during this window.

 A proper hierarchical model would integrate over the coarse rate as a hyperprior, but the complexity jump is not justified for the marginal gain. The blend is defensible.

 Files:
 - src/market_maker/learning/quote_outcome.rs — replace SpreadBin internals, add FillRateEstimate, add hierarchical prior logic, modify fill_rate_at() and all_rates()
 - src/market_maker/checkpoint/types.rs — checkpoint stores (alpha, beta) per bin with #[serde(default)] for backward compat (migrate from old (fills, total) format: alpha = fills + 1.0, beta = total -   
 fills + 1.0)

 Reuse: The BayesianEstimate pattern in src/market_maker/estimator/covariance_tracker.rs uses conjugate normal priors — same philosophy, different family.

 Tests:
 - test_beta_posterior_with_zero_observations — returns prior mean (0.5) with high variance
 - test_beta_posterior_converges — 100 fills out of 200 → mean ≈ 0.5 with tight variance
 - test_hierarchical_shrinkage — fine bin with 3 obs shrinks toward coarse parent; fine bin with 300 obs ignores parent
 - test_fill_rate_always_returns_value — no more None returns
 - test_checkpoint_migration_from_counts — old (fills=10, total=20) → Beta(11, 11)

 Risk: LOW. Better estimation, never returns None, existing trading paths get a proper uncertainty measure.

 ---
 Q18: Importance-Weighted Vol Debiasing (Coupled to Q17)

 Problem: σ̂ is conditioned on fills existing → overweights volatile regimes. The observed vol samples are from p(σ | filled), but spread-setting needs unconditional p(σ)

 Fix (two phases):

 Phase 1 (this implementation): Diagnostics + infrastructure

 Add SamplingBiasTracker to VolatilityFilter:
 - Track EWMA of sigma during fill vs non-fill intervals
 - Log bias_ratio = σ̂_fillable / σ̂_nonfillable in ESTIMATOR_DIAGNOSTI
 - Warn when ratio > 1.5 (significant upward bias)
 - Track fillable_fraction for dead-time analysis

 Phase 2 (enabled once Q17 fill rate model is trusted): Importance weighting

 By Bayes' rule: p(σ) ∝ p(σ | filled) / p(filled | σ)

 The denominator p(filled | σ) is exactly what Q17's fill rate model provides — the Beta posterior mean at the current quoted spread. So we can importance-weight vol observations:

 // Conceptual: when a fill occurs, weight the vol observation
 let fill_prob = fill_rate_model.fill_rate_at(current_spread_bps).mean;
 let importance_weight = 1.0 / fill_prob.max(0.01); // inverse fill probability
 // Feed importance_weight into particle filter or EWMA

 Heavy tail problem: When fill_prob is small (e.g., 0.02 at wide spreads), weight = 50x. A handful of these observations dominate the estimate. Mitigations:
 1. Self-normalized importance weights: Divide each weight by sum of all weights in the window. This bounds the influence of any single observation.
 2. Effective Sample Size (ESS) monitoring: ESS = (Σw_i)² / Σ(w_i²). If ESS drops below 20% of nominal sample count, the importance-weighted estimate is unreliable → fall back to raw EWMA with bias       
 warning logged.
 3. Weight cap: max(0.05) floor on fill_prob (20x max weight) rather than max(0.01) (100x).

 struct ImportanceWeightedVol {
     weighted_sigma_sum: f64,
     weight_sum: f64,
     weight_sq_sum: f64,  // for ESS calculation
     n_samples: u64,
 }

 impl ImportanceWeightedVol {
     fn effective_sample_size(&self) -> f64 {
         if self.weight_sq_sum > 0.0 {
             self.weight_sum.powi(2) / self.weight_sq_sum
         } else {
             0.0
         }
     }

     fn is_reliable(&self) -> bool {
         self.n_samples > 20 && self.effective_sample_size() > 0.2 * self.n_samples as f64
     }
 }

 Phase 2 is gated on:
 1. Q17's fill rate model having sufficient data (total observations > 100 across all bins)
 2. ESS > 20% of nominal sample count
 3. Manual review of bias_ratio over multiple sessions

 Until all gates pass, raw sigma estimate is used with bias_ratio logged as warning.

 Files:
 - src/market_maker/estimator/volatility_filter.rs — add SamplingBiasTracker, record_fill_interval(), sampling_bias_ratio(), fillable_fraction(), and Phase 2 stub: set_fill_rate_model() /
 importance_weight_for()
 - src/market_maker/orchestrator/quote_engine.rs — wire record_fill_interval() each cycle, add vol_to_fill_ratio and sigma_fillable_ratio to ESTIMATOR_DIAGNOSTICS
 - src/market_maker/strategy/market_params.rs — add diagnostic fields to EstimatorDiagnostics
 - src/market_maker/checkpoint/types.rs — add fillable_intervals, total_intervals with #[serde(default)]

 Dependency: Q17 must land first. Phase 2 importance weighting consumes Q17's FillRateEstimate.mean.

 Tests:
 - test_bias_tracker_no_fills — ratio = 1.0
 - test_bias_tracker_volatile_fill_bias — fills during high-sigma periods, verify ratio > 1.0
 - test_importance_weight_computation — fill_prob=0.5 → weight=2.0, fill_prob=0.1 → weight=10.0
 - test_fillable_fraction

 Risk: LOW (Phase 1 is diagnostics-only). Phase 2 importance weighting could reduce sigma estimates, which tightens spreads — this is the UNSAFE direction. Gate Phase 2 behind both data sufficiency AND   
 manual review of bias_ratio over multiple sessions before enabling.

 ---
 Q19: AS-Posterior-Derived Spread Floor (with Static Safety Bound)

 Problem: HIP-3 profile's 15-25 bps target is decorative. The actual floor is fee + AS ≈ 3 bps. System quotes 13 bps total against 16.23 bps realized AS.

 Fix: Two-layer floor:

 Layer 1: Static safety bound (immediate)

 Add profile_min_half_spread_bps to SpreadProfile as a hard constraint below which quotes are never placed on this venue type. This is NOT the optimal policy — it's a constraint that prevents disaster    
 while the adaptive layer warms up.

 - Default: 0.0 (no profile floor)
 - Hip3: 7.5 bps/side (15 bps total minimum)
 - Aggressive: 5.0 bps/side

 Layer 2: AS-posterior-adaptive floor (principled)

 The spread floor should be the posterior mean of AS plus an uncertainty premium:
 floor = E[AS] + k × √Var[AS]
 where k controls risk aversion to AS uncertainty (higher k = wider when uncertain).

 Asymmetric loss argues for k > 1: Underestimating AS costs money on every fill. Overestimating AS only costs opportunity. At k=1 (one-sigma bound), you're quoting below true AS ~16% of the time your     
 posterior is well-calibrated. Default k=1.5 (1.5-sigma, ~7% below true AS) is more consistent with Kelly underbetting under parameter uncertainty. Configurable for tuning.

 Note: AS variance and fill rate variance are correlated — both improve with more fills. The uncertainty premium naturally shrinks as the system matures. This is the right behavior.

 The AS estimator already tracks a running estimate — extend it to maintain posterior variance. When the posterior is wide (few samples, high uncertainty), the floor is wider (Kelly-optimal underbet). As 
  data accumulates, the floor tightens toward the point estimate.

 The static bound (Layer 1) is max()'d with the adaptive floor — it can never make spreads tighter, only acts as a safety net when the AS posterior is poorly calibrated.

 Integration into glft.rs:574:
 // Current: physical_floor = (fee + as_floor).max(min_spread_floor)
 // New:     physical_floor = (fee + as_posterior_floor + profile_floor).max(min_spread_floor)
 // where as_posterior_floor = E[AS] + k * sqrt(Var[AS])
 // and profile_floor = static safety bound

 Files:
 - src/market_maker/config/spread_profile.rs — add profile_min_half_spread_bps() method
 - src/market_maker/strategy/risk_config.rs — add profile_spread_floor_frac: f64 with #[serde(default)], add as_uncertainty_premium_k: f64 (default 1.5) with #[serde(default)]
 - src/market_maker/strategy/glft.rs:574 — integrate both floors into physical_floor_frac
 - src/market_maker/strategy/market_params.rs — add as_floor_variance_bps2: f64 field for AS posterior variance, compute as_posterior_floor_bps = as_floor_bps + k * as_floor_variance_bps2.sqrt()
 - src/market_maker/adverse_selection/estimator.rs — expose posterior variance alongside point estimate (may already track this via EWMA variance)
 - src/bin/market_maker.rs — wire profile floor from SpreadProfile

 Tests:
 - test_profile_min_half_spread_per_type
 - test_physical_floor_with_hip3_profile — fee=1.5 + AS=5 + profile=7.5 = 14 bps floor
 - test_as_uncertainty_premium — wide AS posterior (var=25) with k=1 → floor += 5 bps
 - test_as_uncertainty_shrinks — after 500 samples, var is small, premium is negligible
 - test_static_bound_catches_poorly_calibrated_as — AS posterior says 2 bps but profile says 7.5 → floor = 7.5

 Risk: MEDIUM-LOW. Both layers widen spreads (safe direction). Static bound defaults to 0.0 for non-HIP-3 (backward compat). AS uncertainty premium is conservative by construction (wider when uncertain). 

 ---
 Q20: Posterior-Predictive Inventory Recovery + Graduated Escalation

 Problem: -3.26 position blocked by E[PnL] filter. Kill switch doesn't detect un-unwindable inventory.

 Fix: Replace cycle-counting with a principled stuck metric, but keep graduated escalation.

 Stuck metric: Unrealized Adverse Selection Cost

 Instead of |position| × time (dimensionally inventory-minutes), track:
 /// Dollar-denominated cost of being stuck: cumulative mid-move against the position.
 /// unrealized_as_cost = |position| × Σ(mid_move_against_us_per_cycle) × price
 /// Positive and growing = market moving against us while stuck.
 unrealized_as_cost_usd: f64,

 This connects directly to P&L: it's the paper loss from holding inventory while the market moves against you. Computed each cycle as:
 let mid_move_against = if position > 0.0 {
     (prev_mid - current_mid).max(0.0)  // long, market dropping
 } else {
     (current_mid - prev_mid).max(0.0)  // short, market rising
 };
 unrealized_as_cost_usd += position.abs() * mid_move_against;

 Design choice — cumulative adverse vs net mark-to-market: This accumulator only counts moves against the position, never crediting favorable moves. In a mean-reverting market, mid might drop, recover,   
 drop again — each drop adds cost but the recovery is ignored. This overstates actual paper loss compared to net MTM since stuck began. This is the conservative choice for a kill switch: better to        
 escalate slightly early than to delay because a favorable mean-reversion temporarily masked the stuck condition. A trending market (the truly dangerous case) sees no such masking. Log both
 cumulative-adverse and net-MTM for diagnostics so we can evaluate whether the conservative choice is too aggressive in practice.

 Escalation mechanism

 Still graduated (the concept is right even if the metric changes):
 1. Monitor: Track stuck_cycles (consecutive cycles with significant position + no reducing quotes) AND unrealized_as_cost_usd
 2. Warning at stuck_warning_cycles (10) OR unrealized_as_cost_usd > warn_threshold: ForceReducingQuotes at progressively wider spreads
 3. Kill at max_stuck_cycles (30) OR unrealized_as_cost_usd > kill_threshold: trigger kill switch

 The cost-based threshold triggers earlier when the market is moving against the position (more dangerous) and later when it's moving with the position (less dangerous). The cycle count is a fallback for 
  flat markets where cost doesn't accumulate but the position is still stuck.

 Distinguish stuck causes

 Add StuckCause enum:
 enum StuckCause {
     EpnlBlocking,     // E[PnL] filter says all reducing quotes are negative EV
     AdverseSelection,  // Reducing quotes exist but keep getting adversely filled
     NoLiquidity,       // Reducing side has no takers
 }

 Log the cause with each stuck cycle for diagnostics. The escalation response can differ:
 - EpnlBlocking → spread widening passes E[PnL] check at wider levels
 - AdverseSelection → need more aggressive pricing (accept loss)
 - NoLiquidity → wider spreads won't help, may need to cross the spread

 Files:
 - src/market_maker/risk/kill_switch.rs — add InventoryStuck to KillReason, add StuckEscalation enum, add unrealized_as_cost_usd/position_stuck_cycles/has_reducing_quotes to KillSwitchState, add config   
 fields with #[serde(default)], implement report_reducing_quote_status()
 - src/market_maker/orchestrator/quote_engine.rs — after quote generation, determine reducing-side status, call report_reducing_quote_status(), handle ForceReducingQuotes by injecting reducing quote at   
 widened spread
 - src/market_maker/checkpoint/types.rs — add position_stuck_cycles, unrealized_as_cost_usd with #[serde(default)]

 Config defaults (expressed as fractions of max_position_notional for scaling):
 max_stuck_cycles: 30,                    // ~5 min at 10s/cycle
 stuck_warning_cycles: 10,                // ~100s
 position_stuck_threshold_fraction: 0.10, // 10% of max position
 unrealized_as_warn_fraction: 0.01,       // 1% of max_position_notional (~$2 for $200 max)
 unrealized_as_kill_fraction: 0.05,       // 5% of max_position_notional (~$10 for $200 max)
 Dollar thresholds computed at runtime: warn_usd = max_position_value * unrealized_as_warn_fraction. Scales automatically with position limits and asset price.

 Tests:
 - test_stuck_counter_increments_and_resets
 - test_unrealized_as_cost_accumulates — position=-3, mid rises 10 bps → cost grows
 - test_unrealized_as_cost_resets_on_reducing_quotes
 - test_cost_threshold_triggers_before_cycle_threshold — fast-moving market triggers on cost
 - test_cycle_threshold_triggers_in_flat_market — flat market triggers on cycles
 - test_force_reducing_quotes_progressive_widening
 - test_stuck_kill_trigger
 - test_checkpoint_persistence

 Risk: MEDIUM. Same graduated escalation, but now with a dollar-denominated metric that connects to P&L. The ForceReducingQuotes override of E[PnL] is intentional — a stuck position's holding cost        
 exceeds the E[PnL] model's estimate of trade cost.

 ---
 Cross-Cutting: Spread Regime Transition When Q19 Activates

 Q19's profile floor shifts quoted spreads from ~6.5 bps/side to ~7.5+ bps/side. Q17's fill rate model was trained on the old spread regime and will be queried in a region where it has the least data     
 (the hierarchical prior does the most work here — this is exactly what it's designed for).

 Action: When Q19 activates, log a SPREAD_REGIME_CHANGE event with timestamp and old/new floor values. Tag pre/post data in Q17's FillRateEstimate so diagnostics can distinguish fill rate at old spreads  
 vs new spreads. The hierarchical prior will smoothly handle the transition, but having labeled data is essential for post-hoc analysis of whether the fill rate model adapted correctly.

 Implementation: Keep one (α, β) per bin for the live model (checkpoint hot path stays simple). Log epoch-tagged raw observations to a separate diagnostics file (not part of checkpoint). The checkpoint   
 stores only the current aggregate — post-hoc epoch analysis comes from the diagnostics log, not from reconstructing per-epoch posteriors at recovery time.

 Implementation Order (preserving inference chain)

 1. Q17 — Beta posteriors + hierarchical shrinkage. Foundation for everything else.
 2. Q18 — Diagnostics first (Phase 1). Phase 2 importance weighting enabled after Q17 data validation.
 3. Q19 — Static safety bound immediate; AS-posterior floor uses the same AS estimator infrastructure.
 4. Q20 — Stuck detection with cost metric. Independent of Q17-Q19 but benefits from Q19's wider floors reducing stuck frequency.

 Verification

 1. cargo clippy -- -D warnings after each question's changes (sequential, never parallel)
 2. cargo test after all changes
 3. For Q17: verify fill_rate_at() returns FillRateEstimate with non-zero variance on empty bins
 4. For Q18: inspect ESTIMATOR_DIAGNOSTICS in paper mode, verify bias_ratio is logged
 5. For Q19: verify GLFT output with Hip3 profile → half-spread >= 7.5 bps in unit tests
 6. For Q20: verify unrealized_as_cost_usd accumulates correctly in test, escalation triggers at thresholds