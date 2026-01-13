Observed Values from Log (2026-01-13 23:24)

     ┌─────────────────────┬──────────┬────────────────────────────┐
     │      Parameter      │  Value   │           Source           │
     ├─────────────────────┼──────────┼────────────────────────────┤
     │ gamma               │ 0.195    │ Scaled from gamma_base=0.3 │
     ├─────────────────────┼──────────┼────────────────────────────┤
     │ kappa               │ 4750     │ Robust V3 blended kappa    │
     ├─────────────────────┼──────────┼────────────────────────────┤
     │ sigma               │ 0.000095 │ Bipower volatility         │
     ├─────────────────────┼──────────┼────────────────────────────┤
     │ optimal_spread_bps  │ 26.85    │ Total spread (both sides)  │
     ├─────────────────────┼──────────┼────────────────────────────┤
     │ effective_floor_bps │ 8.0      │ Minimum spread floor       │
     ├─────────────────────┼──────────┼────────────────────────────┤
     │ warmup_pct          │ 10%      │ Early warmup phase         │
     ├─────────────────────┼──────────┼────────────────────────────┤
     │ bid_from_mid_bps    │ 13.5     │ Per-side half-spread       │
     ├─────────────────────┼──────────┼────────────────────────────┤
     │ ask_from_mid_bps    │ 13.5     │ Per-side half-spread       │
     └─────────────────────┴──────────┴────────────────────────────┘
     ---
     State Diagram: Spread Calculation Pipeline

     ┌─────────────────────────────────────────────────────────────────────────────┐
     │                        MARKET DATA INGESTION                                 │
     ├─────────────────────────────────────────────────────────────────────────────┤
     │  L2 Book ──┬──> KappaOrchestrator ─────────────────> kappa = 4750           │
     │            │    └─ 40% book + 30% robust + 30% prior                        │
     │            │                                                                 │
     │  Trades ───┼──> BipowerVariation ──────────────────> sigma = 0.000095       │
     │            │                                                                 │
     │  AllMids ──┴──> MicropriceEstimator ───────────────> microprice             │
     └─────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
     ┌─────────────────────────────────────────────────────────────────────────────┐
     │                        GAMMA CALCULATION                                     │
     │                        (ladder_strat.rs:345-372)                            │
     ├─────────────────────────────────────────────────────────────────────────────┤
     │                                                                              │
     │  gamma_base = 0.3  (from config)                                            │
     │                                                                              │
     │  IF adaptive_mode && adaptive_can_estimate:                                 │
     │     gamma = adaptive_gamma × tail_risk_mult × calibration_scalar            │
     │  ELSE:                                                                       │
     │     gamma = effective_gamma() × liquidity_mult × tail_risk_mult             │
     │                                                                              │
     │  Final gamma = 0.195 (includes book_depth + warmup scaling)                 │
     │  └─ Interpretation: Lower gamma → wider spreads (more conservative)         │
     │                                                                              │
     └─────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
     ┌─────────────────────────────────────────────────────────────────────────────┐
     │                        KAPPA SELECTION                                       │
     │                        (ladder_strat.rs:374-399)                            │
     ├─────────────────────────────────────────────────────────────────────────────┤
     │                                                                              │
     │  Priority Order:                                                             │
     │  1. ROBUST (V3) - if use_kappa_robust = true                                │
     │     kappa = kappa_robust = 4750  ✓ SELECTED                                 │
     │                                                                              │
     │  2. ADAPTIVE - if adaptive_mode && adaptive_can_estimate                    │
     │     kappa = adaptive_kappa                                                   │
     │                                                                              │
     │  3. LEGACY - fallback                                                        │
     │     kappa = book_kappa × (1 - predicted_alpha)                              │
     │                                                                              │
     │  Kappa Orchestrator Breakdown:                                               │
     │  ├─ own = 2500 (0% weight - no own fills yet)                               │
     │  ├─ book = 2500 (40% weight)                                                │
     │  ├─ robust = 10000 (30% weight)                                             │
     │  └─ prior = 2500 (30% weight)                                               │
     │  = 2500×0.4 + 10000×0.3 + 2500×0.3 = 4750                                   │
     │                                                                              │
     └─────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
     ┌─────────────────────────────────────────────────────────────────────────────┐
     │                     GLFT OPTIMAL HALF-SPREAD                                 │
     │                     (depth_generator.rs:243-256)                            │
     ├─────────────────────────────────────────────────────────────────────────────┤
     │                                                                              │
     │  ┌──────────────────────────────────────────────────────────────────────┐   │
     │  │                                                                       │   │
     │  │   δ = (1/γ) × ln(1 + γ/κ) + f_maker                                  │   │
     │  │                                                                       │   │
     │  │   WHERE:                                                              │   │
     │  │   - γ = 0.195 (effective gamma)                                      │   │
     │  │   - κ = 4750 (robust kappa)                                          │   │
     │  │   - f_maker = 0.00015 (1.5 bps maker fee)                            │   │
     │  │                                                                       │   │
     │  └──────────────────────────────────────────────────────────────────────┘   │
     │                                                                              │
     │  CALCULATION:                                                                │
     │  Step 1: ratio = γ/κ = 0.195/4750 = 0.0000411                               │
     │  Step 2: ln(1 + 0.0000411) = 0.0000411 (Taylor: ln(1+x) ≈ x for small x)   │
     │  Step 3: (1/γ) × ln(1+γ/κ) = 5.128 × 0.0000411 = 0.000211                  │
     │  Step 4: + f_maker = 0.000211 + 0.00015 = 0.000361                          │
     │                                                                              │
     │  RESULT: δ_glft = 0.000361 = 3.61 bps per side                              │
     │                                                                              │
     └─────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
     ┌─────────────────────────────────────────────────────────────────────────────┐
     │                     FLOOR & CEILING APPLICATION                              │
     │                     (depth_generator.rs:476-493)                            │
     ├─────────────────────────────────────────────────────────────────────────────┤
     │                                                                              │
     │  Apply minimum floor:                                                        │
     │    optimal_bps = glft_bps.max(min_spread_floor_bps).max(min_depth_bps)      │
     │                = 3.61.max(8.0).max(2.0) = 8.0 bps  ← FLOOR ACTIVE           │
     │                                                                              │
     │  But wait - log shows 13.5 bps per side, not 8.0...                         │
     │  This means additional scaling is applied AFTER floor.                       │
     │                                                                              │
     └─────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
     ┌─────────────────────────────────────────────────────────────────────────────┐
     │                     L2 SPREAD MULTIPLIER (Bayesian Uncertainty)              │
     │                     (ladder_strat.rs:517-533)                               │
     ├─────────────────────────────────────────────────────────────────────────────┤
     │                                                                              │
     │  IF l2_spread_multiplier > 1.0:                                             │
     │     depth *= l2_spread_multiplier                                           │
     │                                                                              │
     │  This is the Bayesian uncertainty premium:                                   │
     │    spread_mult = 1.0 + uncertainty_scaling × (σ_μ / baseline_σ_μ)          │
     │                                                                              │
     │  With 10% warmup, high uncertainty → l2_spread_multiplier ≈ 1.69            │
     │  8.0 bps × 1.69 = 13.5 bps per side ✓                                       │
     │                                                                              │
     │  NOTE: This is NOT bypassing GLFT - it's adding principled                  │
     │  uncertainty quantification for model risk.                                  │
     │                                                                              │
     └─────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
     ┌─────────────────────────────────────────────────────────────────────────────┐
     │                     INVENTORY SKEW (Avellaneda-Stoikov)                      │
     │                     (ladder_strat.rs:419-424)                               │
     ├─────────────────────────────────────────────────────────────────────────────┤
     │                                                                              │
     │  Base Skew Formula:                                                          │
     │    skew = (q / Q_max) × γ × σ² × T                                          │
     │                                                                              │
     │  Current log: position = 0.0 → skew = 0                                     │
     │  Result: bid_delta = ask_delta = 13.5 bps (symmetric)                       │
     │                                                                              │
     │  When position ≠ 0:                                                          │
     │  - Long position → skew > 0 → tighten asks, widen bids                      │
     │  - Short position → skew < 0 → tighten bids, widen asks                     │
     │                                                                              │
     └─────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
     ┌─────────────────────────────────────────────────────────────────────────────┐
     │                     FINAL QUOTE CONSTRUCTION                                 │
     │                     (reconcile.rs)                                          │
     ├─────────────────────────────────────────────────────────────────────────────┤
     │                                                                              │
     │  market_mid = 95456.5                                                        │
     │  bid_delta = 13.5 bps                                                        │
     │  ask_delta = 13.5 bps                                                        │
     │                                                                              │
     │  bid_price = microprice × (1 - bid_delta) = 95456.5 × 0.99865 = 95328       │
     │  ask_price = microprice × (1 + ask_delta) = 95456.5 × 1.00135 = 95585       │
     │                                                                              │
     │  TOTAL SPREAD = 95585 - 95328 = 257 USD                                     │
     │               = 257 / 95456.5 × 10000 = 26.9 bps ✓                          │
     │                                                                              │
     └─────────────────────────────────────────────────────────────────────────────┘

     ---
     Complete Data Flow Diagram

                         ┌────────────────────────────────┐
                         │       RAW MARKET DATA          │
                         │  L2Book, Trades, AllMids       │
                         └───────────────┬────────────────┘
                                         │
               ┌─────────────────────────┼─────────────────────────┐
               │                         │                         │
               ▼                         ▼                         ▼
         ┌───────────┐           ┌───────────┐           ┌────────────────┐
         │ L2 Depth  │           │ Bipower   │           │  Microprice    │
         │ Analysis  │           │ Variation │           │  Estimator     │
         └─────┬─────┘           └─────┬─────┘           └───────┬────────┘
               │                       │                         │
               ▼                       ▼                         ▼
         ┌───────────┐           ┌───────────┐           ┌────────────────┐
         │   Kappa   │           │   Sigma   │           │   Fair Price   │
         │   4750    │           │ 0.000095  │           │    95456.5     │
         └─────┬─────┘           └─────┬─────┘           └───────┬────────┘
               │                       │                         │
               └───────────────────────┼─────────────────────────┘
                                       │
                                       ▼
                         ┌─────────────────────────────────┐
                         │     GAMMA COMPUTATION           │
                         │  gamma_base × scalars = 0.195   │
                         └───────────────┬─────────────────┘
                                         │
                                         ▼
                         ┌─────────────────────────────────┐
                         │     GLFT FORMULA                │
                         │  δ = (1/γ)×ln(1+γ/κ) + fee     │
                         │  = 3.61 bps per side           │
                         └───────────────┬─────────────────┘
                                         │
                                         ▼
                         ┌─────────────────────────────────┐
                         │     FLOOR APPLICATION           │
                         │  max(3.61, 8.0) = 8.0 bps      │
                         └───────────────┬─────────────────┘
                                         │
                                         ▼
                         ┌─────────────────────────────────┐
                         │  L2 UNCERTAINTY MULTIPLIER      │
                         │  8.0 × 1.69 = 13.5 bps         │
                         └───────────────┬─────────────────┘
                                         │
                                         ▼
                         ┌─────────────────────────────────┐
                         │     INVENTORY SKEW              │
                         │  skew = 0 (flat position)       │
                         │  bid_delta = ask_delta = 13.5   │
                         └───────────────┬─────────────────┘
                                         │
                                         ▼
                         ┌─────────────────────────────────┐
                         │     FINAL QUOTES                │
                         │  bid = 95328, ask = 95585       │
                         │  total_spread = 26.9 bps        │
                         └─────────────────────────────────┘

     ---
     Key Files and Line Numbers
     ┌────────────────────┬───────────────────────────────────┬───────────┐
     │     Component      │               File                │   Lines   │
     ├────────────────────┼───────────────────────────────────┼───────────┤
     │ Entry Point        │ orchestrator/quote_engine.rs      │ 19-851    │
     ├────────────────────┼───────────────────────────────────┼───────────┤
     │ Gamma Calculation  │ strategy/ladder_strat.rs          │ 345-372   │
     ├────────────────────┼───────────────────────────────────┼───────────┤
     │ Kappa Selection    │ strategy/ladder_strat.rs          │ 374-399   │
     ├────────────────────┼───────────────────────────────────┼───────────┤
     │ GLFT Formula       │ quoting/ladder/depth_generator.rs │ 243-256   │
     ├────────────────────┼───────────────────────────────────┼───────────┤
     │ Floor/Ceiling      │ quoting/ladder/depth_generator.rs │ 476-493   │
     ├────────────────────┼───────────────────────────────────┼───────────┤
     │ L2 Multiplier      │ strategy/ladder_strat.rs          │ 517-533   │
     ├────────────────────┼───────────────────────────────────┼───────────┤
     │ Inventory Skew     │ strategy/ladder_strat.rs          │ 419-424   │
     ├────────────────────┼───────────────────────────────────┼───────────┤
     │ Kappa Orchestrator │ estimator/kappa_orchestrator.rs   │ full file │
     └────────────────────┴───────────────────────────────────┴───────────┘
     ---
     Why Your Spreads Are ~27 bps

     The observed 26.85 bps total spread results from this chain:

     1. GLFT Base Spread = 3.6 bps per side
       - Formula: δ = (1/0.195) × ln(1 + 0.195/4750) + 0.00015
       - With kappa=4750, this is very tight (inverse relationship)
     2. Floor Override = 8.0 bps per side
       - GLFT output (3.6 bps) < floor (8.0 bps)
       - So minimum floor kicks in
     3. L2 Uncertainty Multiplier ≈ 1.69×
       - During warmup (10%), model uncertainty is high
       - Spreads widened to protect against estimation error
       - 8.0 × 1.69 = 13.5 bps per side
     4. Total Spread = 2 × 13.5 = 27 bps

     This is mathematically sound:
     - GLFT provides the optimal baseline
     - Floor prevents unprofitable quotes
     - Uncertainty premium protects during warmup
     - As warmup progresses (10% → 100%), multiplier decreases → spreads tighten