# Alpha Improvement Roadmap

Comprehensive assessment of the Hyperliquid GLFT market making system's alpha extraction pipeline. Each section identifies what exists, what's missing, the quantitative impact, and a concrete implementation path.

**System baseline** (March 2026): +$4.29/30min (best session), -$1.24/10min (worst). 81% win rate on good session, 40% fill rate on short session. Edge is real but fragile — the system profits when conditions align but has no mechanism to learn which conditions those are.

---

## Table of Contents

1. [Experience Replay & Offline Learning](#1-experience-replay--offline-learning)
2. [Cascade Early Warning](#2-cascade-early-warning)
3. [Signal Decorrelation](#3-signal-decorrelation)
4. [Kappa Seasonality](#4-kappa-seasonality)
5. [Regime-Conditioned Adverse Selection](#5-regime-conditioned-adverse-selection)
6. [HJB Value Function Activation](#6-hjb-value-function-activation)
7. [Drift Model Improvements](#7-drift-model-improvements)
8. [Fill Quality & Markout Feedback](#8-fill-quality--markout-feedback)
9. [Dynamic Position Sizing](#9-dynamic-position-sizing)
10. [Cross-Asset Information Transfer](#10-cross-asset-information-transfer)
11. [Microstructure Feature Extraction](#11-microstructure-feature-extraction)
12. [Execution Optimization](#12-execution-optimization)
13. [Calibration Infrastructure](#13-calibration-infrastructure)
14. [Summary & Prioritization](#14-summary--prioritization)

---

## 1. Experience Replay & Offline Learning

### Current State

Full RL experience logging infrastructure exists. Every fill writes a `(state, action, reward, next_state)` tuple to `logs/experience/`:

```
{
  state_idx,              // Discretized [OI%, inventory%, vol_regime]
  action_idx,             // Spread tier × size tier (24 actions)
  reward_total,           // edge - inventory_penalty - volatility_penalty
  edge_component,         // realized_spread - AS - fee
  inventory_penalty,      // -gamma * q^2 * sigma^2 * tau
  next_state_idx,
  regime, mid_price, fill_price, ...
}
```

**Problem**: Nothing reads these tuples back. The system generates training data every session and discards it.

### What's Missing

1. **Offline batch Q-learning**: Train `Q(state, action)` on accumulated experience to learn optimal spread/size per market state
2. **Counterfactual regret minimization**: For each fill, estimate PnL of alternative actions (tighter spread → faster fill but more AS; wider → fewer fills but safer)
3. **Policy gradient update**: Use logged rewards to nudge Bayesian priors toward higher-reward parameter regions

### Implementation Path

```
src/market_maker/learning/
  replay_buffer.rs       -- Load JSONL experience files, dedup, time-weight
  batch_q_learner.rs     -- Fitted Q-iteration on discrete state-action space
  counterfactual.rs      -- Estimate E[reward | alternative_action] from logged fills
  policy_update.rs       -- Map Q-values back to BayesianParam priors
```

**Algorithm**: Fitted Q-Iteration (FQI) with experience replay

```
1. Load N sessions of experience tuples
2. Discretize states: (vol_regime × inventory_zone × flow_toxicity × hour_bucket)
   ~4 × 5 × 3 × 4 = 240 states
3. Actions: spread_tier (6 levels) × size_tier (4 levels) = 24 actions
4. Fit Q(s,a) = E[R + gamma * max_a' Q(s', a') | s, a]
   using random forest or linear model on features
5. Extract policy: for each state, recommended action = argmax_a Q(s,a)
6. Feed back as prior adjustments:
   - If Q says "tighter spread in low-vol + low-inventory" → nudge spread_floor prior down
   - If Q says "wider in high-toxicity" → nudge informed_flow weight up
```

**Validation**: Hold out 20% of sessions. Compare realized PnL of Q-policy vs. actual policy. Only deploy if Q-policy edge > actual + 1 standard error.

### Expected Impact

- **+3-8 bps/fill** from optimized spread selection per state
- **Compounds**: Each session generates more training data, improving the next session
- This is the single highest-leverage improvement

### Risk

- Overfitting to historical regimes (mitigate: rolling window, regime-stratified train/test)
- State discretization too coarse → misses microstructure (mitigate: start coarse, refine with more data)

---

## 2. Cascade Early Warning

### Current State

Regime detection uses a 4-state HMM with observation vector `[volatility, spread, flow_imbalance, OI_level, OI_velocity, liquidation_pressure]`. Detection latency is **10-30 seconds** — by then, liquidations are already running over the book.

OI velocity is a **lagging indicator**: it measures OI change after liquidations have already fired. The system loses 5-10 bps per cascade event from fills taken during the detection lag.

### What's Missing

1. **Funding rate extremes as leading indicator**: When funding > 0.05% (8h) on a long-heavy asset, liquidation probability spikes. This leads OI drops by 30-120 seconds.
2. **OI concentration at nearby price levels**: If 20% of OI is within 2% of current price, a 2% move triggers cascading liquidations. This is observable from the liquidation heatmap.
3. **Cross-exchange cascade detection**: Binance liquidations precede Hyperliquid by 1-5 seconds (cross-exchange lead-lag on liquidation events, not just price).
4. **Order book thinning velocity**: When L2 depth thins 50%+ in <5 seconds without price movement, market makers are pulling — cascade imminent.

### Implementation Path

```
src/market_maker/risk/cascade_detector.rs

struct CascadeDetector {
    funding_extremity: f64,       // |funding_rate| / historical_sigma
    oi_concentration_2pct: f64,   // fraction of OI within 2% of mark
    depth_thinning_velocity: f64, // L2 depth change rate (negative = thinning)
    binance_liquidation_flow: f64, // recent liquidation volume on Binance
    cascade_probability: f64,     // P(cascade in next 60s)
}

// Logistic regression on labeled cascade events:
// P(cascade) = sigma(w1*funding + w2*oi_conc + w3*depth_thin + w4*binance_liq)
// Train on historical cascade events (labeled by >3% move in <60s)
```

**Integration**: When `cascade_probability > 0.3`, inject into regime detector as synthetic observation favoring Extreme state. When `> 0.6`, trigger immediate spread widening (bypass HMM latency).

### Expected Impact

- **5-10 bps avoided loss per cascade event**
- Cascades occur ~2-5x per day on volatile alts
- Net: reduces tail-risk drawdowns by 30-50%

### Risk

- False positives cause unnecessary spread widening (mitigate: calibrate on 3+ months of cascade labels)
- Funding extremity varies by asset (mitigate: z-score relative to asset's own funding distribution)

---

## 3. Signal Decorrelation

### Current State

The system runs 6+ signals that feed into skew and spread decisions:

| Signal | What It Measures |
|--------|-----------------|
| Lead-lag (Binance) | Cross-venue price divergence |
| Cross-venue flow | Binance vs HL order flow direction |
| Informed flow (EM-GMM) | Trade arrival clustering → toxicity |
| VPIN | Volume-synchronized informed trading |
| BTC beta | Systematic risk correlation |
| Buy pressure | Fill-side imbalance z-score (DISABLED) |

**Problem**: No correlation check between signals. Lead-lag and cross-venue flow likely measure the **same information** (Binance → HL information flow). If they're 80% correlated, the system double-counts and over-skews, leading to:

- Inventory buildup on one side (over-skew attracts fills only on one side)
- Wider effective spreads than necessary (multiple signals all push the same direction)

### What's Missing

1. **Rolling correlation matrix**: Compute pairwise correlation of signal outputs over trailing 500 observations
2. **Redundancy detection**: If `corr(signal_i, signal_j) > 0.6`, reduce weight of the lower-IR signal
3. **Orthogonalization**: Gram-Schmidt or PCA on signal vector before combining
4. **Marginal contribution tracking**: For each signal, measure `PnL_with - PnL_without` (already partially logged in `signal_contributions.jsonl` but not acted on)

### Implementation Path

```
src/market_maker/strategy/signal_decorrelator.rs

struct SignalDecorrelator {
    correlation_matrix: [[f64; N_SIGNALS]; N_SIGNALS],  // Rolling EWMA
    eigenvalues: [f64; N_SIGNALS],                      // PCA decomposition
    effective_weights: [f64; N_SIGNALS],                 // Post-decorrelation
}

impl SignalDecorrelator {
    fn update(&mut self, signal_vector: &[f64; N_SIGNALS]) {
        // Update EWMA correlation matrix (decay = 0.995)
        // Recompute PCA every 100 observations
        // Adjust weights: if signal_i loads >50% on same PC as signal_j,
        //   reduce weight of lower-IR signal by (1 - unique_variance_fraction)
    }

    fn decorrelated_weights(&self, raw_weights: &[f64]) -> Vec<f64> {
        // Return weights adjusted for redundancy
    }
}
```

**Integration**: Call in `signal_integration.rs` before combining signals into skew/spread adjustments.

### Expected Impact

- **+1-3 bps/fill** from removing double-counted skew
- Reduces inventory accumulation from over-skewing
- Side benefit: identifies which signals are truly independent alpha sources

### Risk

- Low. Decorrelation is purely defensive — it reduces exposure to redundant signals.
- PCA can be unstable with few observations (mitigate: minimum 200 obs before adjusting weights)

---

## 4. Kappa Seasonality

### Current State

The kappa orchestrator blends three estimators (book, robust, own-fill) into a single `kappa_eff` value. This value has no concept of time-of-day. Fill intensity on crypto perps has strong intraday patterns:

- **Asian session open** (00:00-02:00 UTC): 2-3x baseline fill rate
- **US session open** (13:00-15:00 UTC): 3-5x baseline
- **Weekend lows** (Sat-Sun 06:00-12:00 UTC): 0.3-0.5x baseline
- **Funding settlement** (every 8h: 00:00, 08:00, 16:00 UTC): 1.5-2x spike for 5 minutes

### What's Missing

1. **Hour-of-day kappa multiplier**: Scale `kappa_eff` by empirically measured fill-rate ratio per hour
2. **Day-of-week adjustment**: Weekend kappa is materially lower
3. **Funding settlement window**: 5-minute kappa spike around settlement times
4. **Adaptive seasonality**: Learn the multipliers from own fills rather than hardcoding

### Implementation Path

```
src/market_maker/estimator/kappa_seasonality.rs

struct KappaSeasonality {
    hourly_multipliers: [f64; 24],    // Relative to daily mean
    dow_multipliers: [f64; 7],        // Mon=1 .. Sun=7
    funding_settlement_boost: f64,    // Multiplier within 5min of settlement
    ewma_decay: f64,                  // Learning rate for adaptive update
}

impl KappaSeasonality {
    fn multiplier(&self, timestamp: DateTime<Utc>) -> f64 {
        let hour = timestamp.hour() as usize;
        let dow = timestamp.weekday().num_days_from_monday() as usize;
        let near_settlement = is_near_funding_settlement(timestamp, 300); // 5min

        let mut mult = self.hourly_multipliers[hour] * self.dow_multipliers[dow];
        if near_settlement {
            mult *= self.funding_settlement_boost;
        }
        mult.clamp(0.3, 5.0)
    }

    fn update_from_fill(&mut self, timestamp: DateTime<Utc>, observed_kappa: f64, baseline_kappa: f64) {
        let hour = timestamp.hour() as usize;
        let ratio = observed_kappa / baseline_kappa;
        self.hourly_multipliers[hour] =
            self.ewma_decay * self.hourly_multipliers[hour] + (1.0 - self.ewma_decay) * ratio;
    }
}
```

**Integration**: In `kappa_orchestrator.rs`, after computing `kappa_eff`:

```rust
let seasonal_mult = self.seasonality.multiplier(now);
let kappa_final = kappa_eff * seasonal_mult;
```

Higher kappa → tighter spreads → more fills during peak hours. Lower kappa → wider spreads → fewer but safer fills during quiet hours.

### Expected Impact

- **+1-2 bps/fill** from better spread sizing per time period
- **+15-30% more fills** during peak hours (tighter spreads capture more flow)
- Passive: learns automatically from own fills after ~1 week of data

### Risk

- Overfitting to specific days (mitigate: EWMA with slow decay, 0.995)
- Crypto seasonality shifts with macro events (mitigate: 2-week half-life on multipliers)

---

## 5. Regime-Conditioned Adverse Selection

### Current State

Adverse selection is tracked as a single EWMA value (`recent_as_ewma_bps`). The Glosten-Milgrom kappa adjustment uses this flat AS estimate:

```
kappa_adj = kappa * (1 - alpha)
alpha = AS / (AS + spread + fee), capped at 0.5
```

**Problem**: AS during cascade is 5-10x AS during calm markets. A single EWMA blends these together, producing an AS estimate that is:

- **Too high** in calm markets (over-widening, losing fills)
- **Too low** in volatile markets (under-protecting, getting run over)

### What's Missing

1. **Per-regime AS tracking**: Maintain separate EWMA per regime state
2. **AS regime transition model**: When regime shifts, immediately load the regime-specific AS estimate instead of waiting for EWMA to catch up
3. **Conditional AS prediction**: `E[AS | regime, volatility, flow_toxicity]` instead of `E[AS]`

### Implementation Path

```
src/market_maker/adverse_selection/regime_as.rs

struct RegimeConditionedAS {
    per_regime_as: [EwmaEstimator; 4],  // Low, Normal, High, Extreme
    per_regime_count: [u64; 4],          // Fill count per regime (for confidence)
    blended_as_bps: f64,                 // Current output
}

impl RegimeConditionedAS {
    fn update(&mut self, regime: RegimeState, realized_as_bps: f64) {
        let idx = regime as usize;
        self.per_regime_as[idx].update(realized_as_bps);
        self.per_regime_count[idx] += 1;
    }

    fn estimate(&self, regime_probs: &[f64; 4]) -> f64 {
        // Bayesian mixture: weight per-regime AS by regime probability
        let mut as_est = 0.0;
        for i in 0..4 {
            let regime_as = if self.per_regime_count[i] > 10 {
                self.per_regime_as[i].value()
            } else {
                DEFAULT_AS_BY_REGIME[i] // Prior: [1.0, 2.5, 6.0, 15.0] bps
            };
            as_est += regime_probs[i] * regime_as;
        }
        as_est
    }
}
```

**Integration**: Replace `self.recent_as_ewma_bps` usage in `signal_integration.rs` with `regime_conditioned_as.estimate(regime_probs)`.

### Expected Impact

- **+1-2 bps/fill** in calm markets (tighter spreads when AS is truly low)
- **+2-5 bps avoided loss** in volatile markets (correct AS estimate immediately on regime shift)
- Most impactful during regime transitions where the old EWMA is stale

### Risk

- Per-regime counts may be low for Extreme regime (mitigate: use informative priors from historical data)
- Regime misclassification feeds wrong AS (mitigate: blend by probability, don't hard-switch)

---

## 6. HJB Value Function Activation

### Current State

`HJBSolver` infrastructure exists in `src/market_maker/stochastic/`. It computes optimal quote depth and size given beliefs via Hamilton-Jacobi-Bellman value iteration:

```
V(state) = max_action { E[profit] - gamma * risk }

State:  (inventory, drift_est, sigma_est, kappa_est)
Action: (spread_delta, size, skew)
```

**Currently disabled** in production. The system uses one-step GLFT optimal instead of multi-step HJB.

### What GLFT Misses (That HJB Captures)

1. **Trajectory optimization**: "If I fill this bid, what's my optimal ask 2s from now?" GLFT treats each quote cycle independently.
2. **Optimal stopping**: When to stop quoting entirely (pull all quotes) vs. widen. GLFT always quotes.
3. **Inventory trajectory planning**: "Given my current inventory of +5, what's the fastest safe path back to 0?" GLFT only skews.
4. **Time-to-close optimization**: Near session end or near funding settlement, optimal behavior changes.
5. **Risk budget allocation**: How much of the daily loss budget to "spend" now vs. save for later.

### Implementation Path

The solver exists — the work is validation and activation:

```
Phase 1: Shadow mode (2 weeks)
  - Run HJB solver alongside GLFT
  - Log HJB-recommended vs GLFT-actual spread/skew
  - Measure counterfactual PnL difference

Phase 2: Blended mode
  - weight = min(0.5, confidence_score)
  - spread = (1-w)*GLFT_spread + w*HJB_spread
  - Ramp weight up as confidence grows

Phase 3: Full activation
  - HJB as primary, GLFT as fallback
  - Fallback triggers: HJB solve time > 1ms, numerical instability
```

**Key validation metric**: HJB should produce **tighter** spreads in low-inventory, low-vol states and **wider** spreads in high-inventory, high-vol states compared to GLFT. If it doesn't, the value function is miscalibrated.

### Expected Impact

- **+1-3 bps/fill** from trajectory optimization
- **Reduced drawdown** from optimal stopping (don't quote during the worst 10% of conditions)
- Most impactful for larger position sizes where inventory trajectory matters

### Risk

- HJB solve time may be too slow for hot path (mitigate: precompute lookup table, interpolate)
- Numerical instability at grid boundaries (mitigate: GLFT fallback)
- High implementation complexity (mitigate: shadow mode first)

---

## 7. Drift Model Improvements

### Current State

Directional beliefs use a Kalman filter with Normal-Inverse-Gamma conjugate prior on drift (mu) and volatility (sigma^2). Per-source process noise is fixed:

| Source | Persistence | Q Rate |
|--------|------------|--------|
| Price returns | 0.2 | High Q (decays fast) |
| Order flow | 0.5 | Medium Q |
| Fill/AS | 0.8 | Low Q (persists) |
| Burst events | 0.9 | Very low Q |

**Problem**: The drift model predicted 0.0 bps in the last paper session while realized drift was -2.25 bps. James-Stein shrinkage zeros out sub-threshold drift, which is correct for preventing false signals but means the model adds zero directional alpha.

### What's Missing

1. **Multi-lag autoregressive structure**: Price returns have AR(2-5) structure on crypto perps (mean-reversion at 1-5s, momentum at 30-300s). The Kalman filter models drift as a random walk — it can't capture momentum.
2. **Volatility-scaled drift prior**: E[|drift|] scales with sigma. In high-vol regimes, drift is larger and more predictable (momentum during cascades). The prior should widen.
3. **Regime-dependent Q rates**: Price persistence should increase in trending markets (lower Q) and decrease in mean-reverting markets (higher Q). Currently Q is fixed.
4. **Funding rate as drift prior**: When funding is +0.05%, longs pay shorts. This creates predictable mean-reversion pressure toward settlement. Wire as Kalman prior.

### Implementation Path

```
src/market_maker/belief/adaptive_drift.rs

struct AdaptiveDriftModel {
    // Multi-lag AR structure
    ar_coefficients: [f64; 5],  // AR(1) through AR(5), learned online
    ar_residual_var: f64,

    // Regime-dependent process noise
    q_rate_by_regime: [f64; 4],  // Q multiplier per regime

    // Funding rate prior
    funding_drift_prior: f64,    // E[drift] from funding rate mechanics
    funding_prior_weight: f64,   // How much to trust funding signal
}

impl AdaptiveDriftModel {
    fn predict_drift(&self, recent_returns: &[f64], regime: RegimeState, funding_rate: f64) -> (f64, f64) {
        // AR prediction
        let ar_pred: f64 = self.ar_coefficients.iter()
            .zip(recent_returns.iter().rev())
            .map(|(c, r)| c * r)
            .sum();

        // Funding prior (mean-reversion pressure)
        let funding_pred = -funding_rate * self.funding_prior_weight;

        // Blend with confidence weighting
        let drift = ar_pred + funding_pred;
        let uncertainty = self.ar_residual_var * self.q_rate_by_regime[regime as usize];

        (drift, uncertainty)
    }
}
```

**Integration**: Replace flat `dir_mu` with `adaptive_drift.predict_drift()` output. Feed into PPIP inventory skew formula.

### Expected Impact

- **+1-3 bps/fill** from correctly predicting short-term drift direction
- **Reduced inventory cost** from better skew timing (skew into predicted mean-reversion)
- Funding rate prior alone could add 0.5-1 bps near settlement windows

### Risk

- AR coefficients are non-stationary (mitigate: EWMA learning with 1-hour half-life)
- Momentum signals can reverse sharply (mitigate: James-Stein shrinkage on AR coefficients)

---

## 8. Fill Quality & Markout Feedback

### Current State

Markout analysis runs at 6 horizons (500ms, 2s, 10s, 60s, 180s, 600s). The system selects the "best" horizon after 20+ fills and uses it for AS estimation. Per-fill metrics include predicted_edge, realized_edge, and gross_edge.

**Problem**: The markout feedback loop is one-directional. The system measures markout but doesn't condition future quoting on per-fill quality patterns. Not all fills are equal:

- Fills at the touch (0 bps depth) have 3-5x higher AS than fills at 5+ bps depth
- Fills during the first 100ms after a Binance price move are toxic
- Fills on the side opposite to inventory are always better

### What's Missing

1. **Per-depth markout tracking**: Separate AS estimate for each depth bucket, used in depth-specific spread decisions
2. **Temporal toxicity filter**: If Binance moved >2 bps in the last 100ms, fills in that direction are 5x more likely to be adverse. Temporarily pull that side.
3. **Fill quality scoring**: Score each fill by realized markout relative to predicted. Persistently toxic fill patterns (e.g., "fills in the first 5s after regime transition") should widen spreads.
4. **Adverse selection asymmetry**: Track AS_buy and AS_sell separately. If AS_buy >> AS_sell, widen bid more than ask.

### Implementation Path

```
src/market_maker/adverse_selection/fill_quality.rs

struct FillQualityTracker {
    // Per-depth-bucket markout
    depth_buckets: [MarkoutStats; 4],  // Touch, Near(2-5), Mid(5-10), Deep(10+)

    // Temporal toxicity
    recent_binance_moves: RingBuffer<(Instant, f64)>,  // (timestamp, bps_move)
    temporal_toxicity_decay_ms: u64,

    // Per-side AS
    as_buy_ewma: f64,
    as_sell_ewma: f64,

    // Fill quality score
    quality_ewma: f64,  // Mean(realized_edge / predicted_edge)
}

impl FillQualityTracker {
    fn is_temporally_toxic(&self, side: Side) -> bool {
        // Check if Binance moved >2 bps in the fill direction in last 100ms
        let recent = self.recent_binance_moves.iter()
            .filter(|(t, _)| t.elapsed() < Duration::from_millis(100))
            .map(|(_, bps)| *bps)
            .sum::<f64>();

        match side {
            Side::Buy => recent < -2.0,   // Binance dropped, buying is toxic
            Side::Sell => recent > 2.0,    // Binance spiked, selling is toxic
        }
    }

    fn per_side_spread_adjustment(&self) -> (f64, f64) {
        // (bid_extra_bps, ask_extra_bps)
        let as_diff = self.as_buy_ewma - self.as_sell_ewma;
        if as_diff > 1.0 {
            (as_diff * 0.5, 0.0)  // Widen bid
        } else if as_diff < -1.0 {
            (0.0, -as_diff * 0.5) // Widen ask
        } else {
            (0.0, 0.0)
        }
    }
}
```

### Expected Impact

- **+1-3 bps/fill** from depth-specific and side-specific spread adjustment
- **+2-5 bps avoided** from temporal toxicity filter (avoid the worst 5% of fills)
- Compounds with regime-conditioned AS for a more complete AS picture

### Risk

- Temporal filter may reduce fill count significantly (mitigate: only filter at touch, not deep levels)
- Per-side AS can be noisy with few fills (mitigate: 50+ fills per side before applying)

---

## 9. Dynamic Position Sizing

### Current State

Position sizing is static: `max_position` derived from capital tier, `target_liquidity` fixed per asset. Kelly sizing is infrastructure-ready but disabled until 50+ fills.

**Problem**: The optimal position size depends on current edge quality. When signals are strong and well-calibrated, the system should take larger positions. When signals are weak or degraded, it should reduce exposure.

### What's Missing

1. **Kelly criterion activation**: `f* = (p*b - q) / b` where p = win rate, b = avg win/avg loss. Already tracked but not applied.
2. **Edge-conditional sizing**: Scale `target_liquidity` by `realized_edge / expected_edge`. If realized edge is 2x expected, double the size.
3. **Drawdown-responsive sizing**: After drawdown, reduce size (not just via kill switch, but graduated). After recovery, ramp back up.
4. **Regime-conditional capacity**: In low-vol regime, max_position can be higher (lower risk per unit). In extreme, max_position should shrink automatically.

### Implementation Path

```
src/market_maker/strategy/dynamic_sizing.rs

struct DynamicSizer {
    kelly_fraction: f64,           // f* from Kelly tracker
    kelly_confidence: f64,         // ESS-based confidence in f*
    edge_ratio: f64,               // realized_edge / predicted_edge (EWMA)
    drawdown_scaling: f64,         // 1.0 at peak, decreases with drawdown
    regime_capacity_mult: [f64; 4], // Per-regime position limit multiplier
}

impl DynamicSizer {
    fn effective_max_position(&self, base_max: f64, regime_probs: &[f64; 4]) -> f64 {
        let kelly_scale = if self.kelly_confidence > 0.6 {
            self.kelly_fraction.clamp(0.1, 2.0)
        } else {
            1.0 // No adjustment until confident
        };

        let edge_scale = self.edge_ratio.clamp(0.5, 2.0);
        let regime_scale: f64 = regime_probs.iter()
            .zip(self.regime_capacity_mult.iter())
            .map(|(p, m)| p * m)
            .sum();

        base_max * kelly_scale * edge_scale * self.drawdown_scaling * regime_scale
    }
}
```

**Kelly activation gate**: Only apply when `n_wins + n_losses > 50` AND `kelly_confidence > 0.6`. Below that, use static sizing.

### Expected Impact

- **+20-50% PnL** when edge is strong (larger positions capture more)
- **-30-50% drawdown** when edge degrades (smaller positions limit damage)
- Kelly optimal sizing is the textbook answer to "how much to bet"

### Risk

- Kelly is aggressive at full fraction (mitigate: use half-Kelly as standard practice)
- Win rate estimate is noisy with few fills (mitigate: conservative ESS gate)

---

## 10. Cross-Asset Information Transfer

### Current State

Each asset is traded independently. BTC beta tracking exists (EWMA correlation, vol stress widening) but is used only for defensive spread widening, not for alpha extraction.

### What's Missing

1. **Cross-asset kappa transfer**: When starting a new asset, use kappa estimates from correlated assets as informative priors instead of uninformative Gamma(4, 0.002)
2. **Sector momentum**: If ETH + SOL + AVAX are all trending up, HYPE (DeFi/infra) likely follows. This is a 5-30s lead signal.
3. **Correlation regime switching**: BTC-HYPE correlation isn't constant. During risk-off, correlation spikes to 0.9+. During altcoin season, drops to 0.3. The system should detect this and adjust beta hedging.
4. **Liquidation contagion**: Liquidations on BTC/ETH cascade to altcoins with 2-10s delay. Track cascade propagation and pre-widen.

### Implementation Path

```
src/market_maker/signals/cross_asset.rs

struct CrossAssetSignals {
    // Sector momentum (rolling 5-min return of top-5 correlated assets)
    sector_momentum_bps: f64,
    sector_momentum_confidence: f64,

    // Correlation regime
    btc_correlation_regime: CorrelationRegime,  // Low/Normal/High/Crisis
    current_beta: f64,
    beta_stability: f64,  // CV of beta over last 100 observations

    // Liquidation contagion (Binance liq feed)
    btc_recent_liquidations_usd: f64,
    eth_recent_liquidations_usd: f64,
    contagion_probability: f64,
}
```

### Expected Impact

- **+0.5-2 bps/fill** from sector momentum alpha
- **Better cold-start** when adding new assets (kappa transfer)
- **Reduced cascade losses** from contagion early warning

### Risk

- Correlation estimates are noisy (mitigate: 200+ paired observations before trusting)
- Sector definitions are subjective (mitigate: use PCA-derived factors, not manual sectors)

---

## 11. Microstructure Feature Extraction

### Current State

The system uses L2 order book depth for kappa estimation and basic spread measurement. Book features beyond depth are not extracted.

### What's Missing

1. **Book imbalance signal**: `(bid_depth - ask_depth) / (bid_depth + ask_depth)` at levels 1-5. Predicts short-term price direction (50-500ms). Well-documented alpha source in equities, likely present on HL perps.
2. **Trade arrival clustering**: Inter-arrival time distribution encodes informed vs. noise flow. Currently used in EM-GMM but not as a standalone microstructure feature.
3. **Spread mean-reversion**: When the observed market spread is abnormally wide, it tends to narrow — this is a queue-position opportunity. When narrow, fills are harder.
4. **Queue position tracking**: For our own orders, track estimated queue position over time. Orders near the front of the queue are more valuable (higher fill probability per unit time).
5. **Minimum price variation effects**: HL perps have minimum tick sizes. When price is near a tick boundary, order flow dynamics change. Model the tick clustering.

### Implementation Path

```
src/market_maker/signals/microstructure.rs

struct MicrostructureFeatures {
    // L1-L5 book imbalance (EWMA smoothed)
    book_imbalance: [f64; 5],       // Per level [-1, 1]
    weighted_imbalance: f64,         // Depth-weighted composite

    // Trade arrival
    inter_arrival_cv: f64,           // CV of inter-arrival times (>1 = clustered)
    recent_trade_intensity: f64,     // Trades/second in last 10s

    // Spread dynamics
    spread_z_score: f64,             // Current spread vs. rolling mean
    spread_mean_bps: f64,
    spread_std_bps: f64,
}

impl MicrostructureFeatures {
    fn directional_signal(&self) -> f64 {
        // Weighted book imbalance predicts 50-500ms price direction
        // Weight by level: L1=0.4, L2=0.25, L3=0.15, L4=0.1, L5=0.1
        let weights = [0.4, 0.25, 0.15, 0.1, 0.1];
        self.book_imbalance.iter()
            .zip(weights.iter())
            .map(|(b, w)| b * w)
            .sum()
    }

    fn spread_opportunity(&self) -> f64 {
        // When spread is abnormally wide, tighten (good fills available)
        // When abnormally narrow, widen (crowded, low fill quality)
        -self.spread_z_score * 0.5  // Negative z = wide spread = opportunity
    }
}
```

### Expected Impact

- **+1-3 bps/fill** from book imbalance signal (well-documented in literature)
- **+0.5-1 bps** from spread mean-reversion timing
- Directional: book imbalance is the fastest signal (sub-second)

### Risk

- Book imbalance can be spoofed (mitigate: only trust levels 1-3, weight by trade flow confirmation)
- Feature computation adds latency (mitigate: incremental update on book changes, not full recomputation)

---

## 12. Execution Optimization

### Current State

Orders are placed via REST + WebSocket POST with bulk order support. Quote refresh happens on every cycle (~1-2s). Orders are placed at fixed depth levels from GLFT-computed half-spread.

### What's Missing

1. **Adaptive quote refresh rate**: In fast markets, refresh every 200ms (avoid stale quotes getting picked off). In slow markets, refresh every 5s (reduce API load, maintain queue priority).
2. **Cancel-and-replace vs. new order**: Amending an existing order preserves queue position. Creating a new order starts at the back. Track which orders can be amended vs. need replacement.
3. **Stale quote detection**: If our quote is >2x the market best bid/ask spread from mid, it's stale and attracting only toxic flow. Detect and cancel within 100ms.
4. **Depth optimization**: Instead of fixed levels from GLFT, optimize the depth ladder based on fill probability at each level. Place more orders where P(fill) * E[edge | fill] is maximized.

### Implementation Path

```
src/market_maker/orchestrator/execution_optimizer.rs

struct ExecutionOptimizer {
    // Adaptive refresh
    market_speed: f64,           // Realized vol per second (proxy for speed)
    optimal_refresh_ms: u64,     // Computed from market speed

    // Stale quote detection
    staleness_threshold_bps: f64, // Max distance from mid before considered stale
    time_since_last_refresh: Duration,

    // Depth ladder optimization
    depth_fill_rates: [f64; 10],  // Empirical fill rate per depth level
    depth_edge: [f64; 10],        // Empirical edge per depth level
    optimal_depths: Vec<f64>,     // Kelly-optimal depth allocation
}

impl ExecutionOptimizer {
    fn optimal_refresh_interval(&self) -> Duration {
        // In fast markets: refresh faster to avoid stale quotes
        // In slow markets: refresh slower to preserve queue position
        let base_ms = 1000.0;
        let speed_factor = (self.market_speed / 0.0003).clamp(0.2, 5.0);
        Duration::from_millis((base_ms / speed_factor) as u64)
    }

    fn optimize_depth_ladder(&self, half_spread_bps: f64, n_levels: usize) -> Vec<f64> {
        // Allocate order sizes across depths to maximize E[fill * edge]
        // More size at depths where fill_rate * edge is highest
        let expected_values: Vec<f64> = self.depth_fill_rates.iter()
            .zip(self.depth_edge.iter())
            .map(|(fr, e)| fr * e)
            .collect();
        // ... normalize and allocate
        todo!()
    }
}
```

### Expected Impact

- **+1-2 bps** from avoiding stale quote pickoff
- **+0.5-1 bps** from depth ladder optimization
- **Reduced API costs** from adaptive refresh rate (fewer unnecessary cancels)

### Risk

- Faster refresh = more API calls = potential rate limiting (mitigate: track rate limit headroom)
- Queue position loss from frequent cancels (mitigate: only cancel when price moved >1 bps)

---

## 13. Calibration Infrastructure

### Current State

Calibration tracking exists: Brier Score, IR, Conditional Calibration metrics are computed. Signal contributions are logged. But the calibration results don't automatically feed back into parameter adjustment.

### What's Missing

1. **Automated parameter nudging**: When Brier Score degrades >20% from baseline, automatically widen spreads by the degradation ratio. When it improves, tighten.
2. **Signal auto-disabling**: If a signal's marginal contribution is negative for 200+ fills (measured via leave-one-out), automatically set its weight to 0.
3. **Calibration drift alerting**: When model calibration drifts beyond 2 sigma of historical, flag for human review. Don't auto-adjust — just alert.
4. **A/B testing framework**: Run two parameter sets simultaneously (e.g., half the quote cycles use parameter set A, half use B). Compare edge after 500+ fills.
5. **Regime-specific calibration**: Track Brier Score per regime. A model can be well-calibrated overall but terrible in Extreme regime (where it matters most).

### Implementation Path

```
src/market_maker/calibration/auto_calibrator.rs

struct AutoCalibrator {
    // Per-signal health tracking
    signal_health: HashMap<String, SignalHealth>,

    // Global calibration quality
    brier_score_ewma: f64,
    brier_score_baseline: f64,
    calibration_degradation_ratio: f64,

    // A/B testing (future)
    active_experiment: Option<Experiment>,
}

struct SignalHealth {
    marginal_contribution_bps: f64,  // Rolling leave-one-out estimate
    consecutive_negative_fills: u32,
    is_auto_disabled: bool,
    last_evaluation_fills: u64,
}

impl AutoCalibrator {
    fn spread_adjustment_from_calibration(&self) -> f64 {
        // When calibration degrades, widen spreads proportionally
        // degradation_ratio > 1.0 means calibration is worse than baseline
        if self.calibration_degradation_ratio > 1.2 {
            1.0 + (self.calibration_degradation_ratio - 1.0) * 0.5
            // 20% degradation → 10% wider spreads
        } else {
            1.0
        }
    }

    fn evaluate_signal(&mut self, signal_name: &str, marginal_bps: f64) {
        let health = self.signal_health.entry(signal_name.to_string())
            .or_insert_with(|| SignalHealth::default());

        health.marginal_contribution_bps =
            0.99 * health.marginal_contribution_bps + 0.01 * marginal_bps;

        if health.marginal_contribution_bps < -0.5 {
            health.consecutive_negative_fills += 1;
            if health.consecutive_negative_fills > 200 {
                health.is_auto_disabled = true;
                tracing::warn!(signal = signal_name, "Auto-disabled signal: persistent negative marginal");
            }
        } else {
            health.consecutive_negative_fills = 0;
        }
    }
}
```

### Expected Impact

- **+1-2 bps** from auto-disabling harmful signals (buy pressure was left on too long)
- **Prevents slow-bleed** from model degradation (calibration drift is invisible until PnL shows it)
- **Systematic improvement** via A/B testing (data-driven parameter tuning instead of guesswork)

### Risk

- Auto-disabling can be premature (mitigate: 200-fill threshold, human review alert before disabling)
- A/B testing reduces power per variant (mitigate: use Thompson sampling instead of fixed 50/50 split)

---

## 14. Summary & Prioritization

### Impact vs. Effort Matrix

| # | Improvement | Expected bps | Effort | Priority |
|---|------------|-------------|--------|----------|
| 1 | Experience Replay & Offline Learning | +3-8 | Medium | **P0** |
| 2 | Cascade Early Warning | +5-10 (avoided) | Medium | **P0** |
| 3 | Signal Decorrelation | +1-3 | Low | **P1** |
| 4 | Kappa Seasonality | +1-2 | Low | **P1** |
| 5 | Regime-Conditioned AS | +1-2 | Low | **P1** |
| 6 | HJB Value Function | +1-3 | High | **P2** |
| 7 | Drift Model (AR + Funding) | +1-3 | Medium | **P2** |
| 8 | Fill Quality & Markout Feedback | +1-3 | Medium | **P2** |
| 9 | Dynamic Position Sizing (Kelly) | +20-50% PnL | Medium | **P1** |
| 10 | Cross-Asset Information | +0.5-2 | Medium | **P3** |
| 11 | Microstructure Features | +1-3 | Medium | **P2** |
| 12 | Execution Optimization | +1-2 | Medium | **P3** |
| 13 | Calibration Infrastructure | +1-2 | Low | **P1** |

### Recommended Sequence

**Phase 1 — Quick Wins (1-2 days each)**
1. Signal decorrelation (#3) — remove double-counting, immediate improvement
2. Kappa seasonality (#4) — hour-of-day multiplier, learns from own fills
3. Regime-conditioned AS (#5) — per-regime EWMA, straightforward
4. Calibration auto-nudging (#13) — defensive, prevents slow bleed
5. Kelly activation (#9) — infrastructure exists, just needs confidence gate

**Phase 2 — Core Alpha (3-5 days each)**
6. Cascade early warning (#2) — biggest loss reduction
7. Experience replay (#1) — biggest alpha addition, needs labeled data pipeline
8. Fill quality feedback (#8) — per-depth and temporal toxicity
9. Microstructure features (#11) — book imbalance signal

**Phase 3 — Advanced (1-2 weeks each)**
10. Drift model improvements (#7) — AR structure, funding prior
11. HJB value function (#6) — shadow mode validation
12. Cross-asset signals (#10) — sector momentum, contagion
13. Execution optimization (#12) — adaptive refresh, depth ladder

### Success Metrics

| Metric | Current | Phase 1 Target | Phase 2 Target | Phase 3 Target |
|--------|---------|---------------|---------------|---------------|
| Mean edge (bps/fill) | +5-11 | +8-15 | +12-20 | +15-25 |
| Win rate | 40-81% | 55-70% | 60-75% | 65-80% |
| Fills/30min | 20-92 | 50-120 | 80-150 | 100-200 |
| Sharpe (annualized) | ~1-2 | ~2-3 | ~3-5 | ~4-7 |
| Max drawdown/day | $15 cap | $10 cap | $8 cap | $5 cap |
| Cascade loss/event | 5-10 bps | 5-10 bps | 2-5 bps | 1-3 bps |

### The Core Insight

The system has strong **defensive infrastructure** (risk monitors, kill switch, reduce-only, position guards) and solid **parameter estimation** (Bayesian kappa, regime HMM, Kalman drift). What it lacks is:

1. **A closed learning loop** — data is generated but not fed back into policy
2. **Anticipation** — everything is reactive (detect regime, then adjust) instead of predictive (predict regime, pre-adjust)
3. **Granularity** — single estimates where per-condition estimates would capture alpha (AS by regime, kappa by hour, edge by depth)

Fix these three structural gaps and the system moves from "marginal positive" to "consistently profitable."
