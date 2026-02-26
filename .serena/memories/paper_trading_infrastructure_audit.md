# Paper Trading Infrastructure Audit

## Executive Summary

The paper trading infrastructure is **comprehensive and well-architected**, with clear separation between paper (simulation) and live modes. The system uses a modular design:
- **CLI activation**: `market_maker paper --duration SECS` subcommand
- **Execution**: `PaperEnvironment` (simulates order placement)
- **Fills**: `FillSimulator` (probabilistic fill model)
- **Analytics**: JSONL logging + calibration readiness assessment
- **Checkpoint**: Saved to `data/checkpoints/paper/{ASSET}/prior.json` for transfer to live
- **Transfer**: Formal φ (extract) and ψ (inject) protocol for paper→live graduation

---

## 1. CLI Activation & Mode Selection

### Entry Point: `src/bin/market_maker.rs`

**CLI Flag**: `Commands::Paper { duration }`
- Subcommand: `market_maker paper --asset BTC --duration 1800`
- Duration in seconds (default: 1800 = 30 min)
- Optional: `--duration 0` = run indefinitely (manual ^C)

**Paper Mode Activation Path** (lines 893-894):
```rust
Some(Commands::Paper { duration }) => {
    return run_paper_mode(&cli, *duration).await;
}
```

### Auto-Calibration Pipeline (lines 897-1015)

When running `market_maker run` (live mode) WITHOUT calibrated priors:
1. Check for prior at `data/checkpoints/paper/{ASSET}/prior.json`
2. If missing or stale (>4 hours), auto-run paper mode
3. Default duration: `cli.calibration_duration` (default: 1800s = 30 min)
4. Flag: `--no-auto-paper` to skip auto-calibration (must use `--force`)
5. Gate checks: `CalibrationGate::passes(readiness)` validates 5 estimators ready

**Paper Mode Can Be Skipped**:
- `--force`: Cold-start live without paper prior
- `--skip-calibration`: Bypass gate entirely

### Config Loading (Paper vs Live)

Both modes load the **same TOML config file** (`market_maker.toml`). The only difference is:
- Paper: Uses mock `PaperEnvironment` instead of `HyperliquidExecutor`
- Live: Uses real WebSocket connections to Hyperliquid

**Key Overrides in Paper Mode** (lines 2564-2591):
- `risk_aversion`: CLI > config > 0.3 (conservative default)
- `max_position`: CLI > config > 1.0 (small default for paper)
- `decimals`: Auto-derived from metadata
- Spread profile applied: `default`, `hip3`, or `aggressive`

---

## 2. What Paper Mode Actually Simulates vs Real

### Simulated (NOT Real)
| Component | Simulation | Why |
|-----------|-----------|-----|
| **Order Placement** | Instant, in-memory | No exchange latency |
| **Cancellations** | Instant | No race conditions |
| **Fills** | Probabilistic model | Based on trade flow, not actual queue |
| **Balances** | Fixed $1000 USD paper balance | No real account |
| **Position Tracking** | Simulated from fills | No exchange state sync |
| **Market Data** | Real WebSocket feed | Same as live |

### NOT Simulated (Uses Real Data)
| Component | Behavior | Source |
|-----------|----------|--------|
| **L2 Book** | Real-time updates | Hyperliquid WS (Trades channel) |
| **Trades** | Real market trades | Hyperliquid WS (Trades channel) |
| **Mids** | Real mid prices | Hyperliquid WS (AllMids channel) |
| **Quote Generation** | Same algorithm as live | LadderStrategy |
| **Signal Calculations** | Same as live | Estimators, HMM, etc. |
| **PnL Attribution** | Simulated against real fills | Paper fills aren't real |

### Architecture: PaperEnvironment ↔ SimulationExecutor

```
Market Data (Real)
  ↓ (WS: AllMids, L2Book, Trades)
  ↓
PaperEnvironment::observation_stream()
  ├─ 3 subscriptions: AllMids, L2Book, Trades
  ├─ Filter: Only market data (no UserFills, OrderUpdates, etc.)
  └─ Feed to: FillSimulator + MarketMaker event loop
    ↓
FillSimulator::on_trade() ← checks resting orders
  ├─ Computes P(fill | order, trade, queue_position)
  ├─ Updates queue estimators from L2 deltas
  └─ Returns: Vec<SimulatedFill> or empty
    ↓
SimulationExecutor::apply_fill() ← creates WsFillEvent
  └─ Injected into MarketMaker handlers as synthetic UserFills
```

**Key Difference from Live**: 3 WS channels vs 9 channels
- Paper: AllMids, L2Book, Trades (only market data)
- Live: +UserFills, OrderUpdates, OpenOrders, ActiveAssetData, WebData2, LedgerUpdates

---

## 3. Fill Simulation: Probabilistic Model

### Configuration: `FillSimulatorConfig`

```rust
pub struct FillSimulatorConfig {
    pub touch_fill_probability: f64,      // 0.3 (default: 30%)
    pub queue_position_factor: f64,       // 0.4 (size cap)
    pub max_order_age_s: f64,             // 300s
    pub min_triggering_trade_size: f64,   // 0.0
    pub placement_latency_ms: u64,        // 100ms
    pub cancel_latency_ms: u64,           // 50ms
    pub ignore_book_depth: bool,          // false (use L2 for queue est)
    pub queue_alpha: f64,                 // 1.5 (queue decay exponent)
}
```

### Fill Probability Computation (lines 426-448)

**Formula**: `P(fill) = base_prob × price_factor × size_factor × queue_factor`

1. **Base Probability**: `touch_fill_probability = 0.3`
   - Probability of fill when trade touches our level exactly

2. **Price Factor**: 
   - Trade through our level (better): 1.5×
   - Trade at/near our level: 1.0×

3. **Size Factor**: `clamp(trade_size / order_size, 0.5, 2.0)`
   - Larger trades → higher fill prob
   - Smaller trades → lower fill prob

4. **Queue Factor**: `(1 - queue_frac)^alpha`
   - `queue_frac` = estimated fraction of queue ahead (0 = front, 1 = back)
   - `alpha = 1.5` → back-of-queue gets ~15% vs front-of-queue ~100%
   - Uses L2 snapshots to estimate: `queue_ahead = current_depth - our_size`

### Queue Position Estimator (lines 106-169)

**Tracks Per-Order**:
```rust
pub struct QueuePositionEstimator {
    initial_size_at_level: f64,   // L2 depth when order placed
    current_size_at_level: f64,   // Updated on each L2 snapshot
    our_size: f64,                // Order size
    oid: u64,
}
```

**Queue Fraction Estimate**:
```
queue_fraction = max(current_size - our_size, 0) / max(current_size, our_size)
```

**Conditional Fill Probability**:
```
P(fill | depth_ahead) = touch_fill_prob / (1 + alpha × depth_ahead)
```

Uses absolute depth, not fractional position (more intuitive).

### Trade Matching Logic (lines 357-392)

**Order Gets Filled If**:
1. Order is `Resting` (not cancelled/filled)
2. Trade timestamp ≥ order creation + placement latency
3. Trade size ≥ `min_triggering_trade_size`
4. **Price condition**:
   - Buy order: `trade_price <= order_price`
   - Sell order: `trade_price >= order_price`
5. **Aggressor direction** (with fallback):
   - Buy order: Trade aggressor = Sell (normal)
   - Sell order: Trade aggressor = Buy (normal)
   - Wrong direction: 30% of touch_fill_probability (~9%)
6. **Probabilistic**: Roll uniform RNG against computed `P(fill)`

**Fill Size**: `min(order_size, trade_size × queue_position_factor)`
- Never fills more than order size
- Never fills more than ~40% of trade size (queue position factor)

### SimulatedFill Result

```rust
pub struct SimulatedFill {
    pub oid: u64,
    pub timestamp_ns: u64,
    pub fill_price: f64,           // Order limit price (not trade price)
    pub fill_size: f64,
    pub side: Side,
    pub triggering_trade_price: f64,
    pub triggering_trade_size: f64,
}
```

**Conversion to WsFillEvent** (executor.rs lines 203-249):
- Applied to order state: reduce `order.size`, update status to `Filled` if empty
- Generate unique trade ID: `tid = next_order_id()`
- Create synthetic `UserFills` observation for MarketMaker handlers
- Timestamp: Current system time (wall-clock, not trade time)

---

## 4. Analytics & Metrics Collection During Paper Trading

### Output Location: `data/analytics/` JSONL Files

**4 Parallel Streams** (persistence.rs lines 44-48):
```
sharpe_metrics.jsonl       — Rolling Sharpe ratio (1h, 24h, 7d, all-time)
signal_contributions.jsonl — Per-cycle signal PnL attribution
signal_pnl.jsonl          — Per-signal marginal value (active/inactive)
edge_validation.jsonl     — Predicted vs realized edge by phase
```

### AnalyticsLogger API (persistence.rs lines 51-85)

```rust
pub fn log_sharpe(&mut self, summary: &SharpeSummary) -> std::io::Result<()>
pub fn log_contributions(&mut self, cycle: &CycleContributions) -> std::io::Result<()>
pub fn log_signal_pnl(&mut self, attributor: &SignalPnLAttributor) -> std::io::Result<()>
pub fn log_edge(&mut self, snapshot: &EdgeSnapshot) -> std::io::Result<()>
```

**One JSON line per event**:
- Append mode (crash-safe)
- BufWriter for efficiency
- Timestamped, human-readable

### Metrics Collected

#### 1. Sharpe Ratio (SharpeSummary)
- `sharpe_1h`, `sharpe_24h`, `sharpe_7d`, `sharpe_all`
- `mean_return_bps`, `std_return_bps`
- Sample count, elapsed seconds

#### 2. Cycle Contributions (CycleContributions)
- Per-signal spread/skew adjustments
- Was each signal active?
- Total spread multiplier, combined skew

#### 3. Signal PnL Attribution (SignalPnLAttributor)
- Active PnL per signal: `signal_pnl_bps`
- Inactive (counterfactual): `signal_pnl_inactive_bps`
- Marginal value: `marginal_value_bps = active - inactive`

#### 4. Edge Validation (EdgeSnapshot)
- Phase: `Pending` (at placement), `Executed` (at fill), `Markout` (after)
- Predicted vs realized spread, AS, edge
- Gross edge (spread + AS)
- Mid price at placement (for post-analysis)
- Markout AS at 500ms, 2s, 10s horizons

### Experience Logging (For Offline RL Training)

**Enabled by** (run_paper_mode line 2723):
```rust
market_maker = market_maker.with_experience_logging("logs/experience");
```

Creates: `logs/experience/experiences.jsonl`
- (state, action, reward, next_state, done)
- Used by offline RL trainer binary for Q-table learning
- Persisted separately from analytics

---

## 5. Checkpoint Persistence & State Transfer

### Save Location

**Paper Mode**:
```
data/checkpoints/paper/{ASSET}/
├── prior.json                    ← Primary (latest)
└── {TIMESTAMP}/checkpoint.json   ← Timestamped backups
```

Example: `data/checkpoints/paper/BTC/prior.json`

**Live Mode**:
```
data/checkpoints/live/{ASSET}/
├── latest/checkpoint.json
└── {TIMESTAMP}/checkpoint.json
```

### CheckpointBundle Contents (types.rs lines 88-149)

**19 Components** across ~500 lines of structured state:

| Component | Purpose | Samples Used |
|-----------|---------|--------------|
| `metadata` | Version, timestamp, asset, duration | Diagnostic |
| `learned_params` | 20 Bayesian posteriors | All |
| `pre_fill` | Pre-fill AS classifier weights | Tautology detection |
| `enhanced` | Enhanced AS classifier | Updated learning |
| `vol_filter` | Volatility sufficient statistics | Vol-scaled kappa |
| `regime_hmm` | HMM belief state (4 regimes) | Regime gating |
| `informed_flow` | Mixture model params | Flow-informed spreads |
| `fill_rate` | Bayesian regression (P(fill\|spread)) | Fill-aware quoting |
| `kappa_own`, `kappa_bid`, `kappa_ask` | Order intensity (3 estimators) | Spread GLFT |
| `momentum` | Continuation probabilities | Momentum signals |
| `kelly_tracker` | Win/loss counts | Position sizing |
| `ensemble_weights` | Blending weights (signals) | Signal gating |
| `spread_bandit` | Contextual bandit (spread optimizer) | Spread deltas |
| `baseline_tracker` | EWMA reward centering | RL normalization |
| `quote_outcomes` | Fill rate bins (P(fill\|spread)) | Empirical fills |
| `kill_switch` | PnL peak tracking, daily loss | Safety persistence |
| `calibration_coordinator` | L2-derived kappa blending | Robust spreads |
| `readiness` | Verdict: Ready/Marginal/Insufficient | Go/no-go decision |

### Readiness Assessment

**Stamps into Checkpoint**:
```rust
pub struct PriorReadiness {
    pub verdict: PriorVerdict,  // Ready | Marginal | Insufficient
    pub vol_observations: usize,
    pub kappa_observations: usize,
    pub as_learning_samples: usize,
    pub regime_observations: usize,
    pub fill_rate_observations: usize,
    pub kelly_fills: usize,
    pub session_duration_s: f64,
    pub estimators_ready: u8,   // 0-5
}
```

**CalibrationGate** (used in main.rs lines 928-977):
- Thresholds: E.g., `vol_observations >= 100`, `kappa_observations >= 150`
- Checks: Prior age ≤ 4 hours
- Verdict: Passes if 4+ of 5 estimators ready

### Extract & Inject Protocol

**φ: Extract** (PriorExtract trait, transfer.rs lines 13-16)
```rust
fn extract_prior(&self) -> CheckpointBundle {
    // Collect learned state from MarketMaker into CheckpointBundle
    // Called at paper mode timeout: market_maker.extract_prior()
    // Saves to: data/checkpoints/paper/{ASSET}/prior.json
}
```

**ψ: Inject** (PriorInject trait, transfer.rs lines 47-52)
```rust
fn inject_prior(&mut self, prior: &CheckpointBundle, config: &InjectionConfig) -> usize {
    // Inject paper-learned state into live MarketMaker
    // Applied: when live mode boots with prior
    // Validation: asset match, age check, skip kill-switch
}
```

**Injection Configuration** (transfer.rs lines 18-44):
```rust
pub struct InjectionConfig {
    pub max_prior_age_s: f64,       // 4 hours default
    pub require_asset_match: bool,  // BTC paper → BTC live only
    pub skip_kill_switch: bool,     // true (never inherit paper kill state)
}
```

### Paper-to-Live Graduation

**Automatic** (when auto-calibration runs):
1. Paper runs for 1800s, saves prior
2. Live boots, detects prior, loads it
3. Prior injected into live MarketMaker state
4. Live uses all learned priors (kappa, AS, etc.)
5. Kill switch NOT inherited (safety)

**Manual** (via CLI):
```bash
./target/debug/market_maker paper --duration 1800  # Save prior
./target/debug/market_maker run                     # Load and use prior
```

---

## 6. Paper Mode Execution Flow

### run_paper_mode() (lines 2467-2770)

**Setup Phase** (lines 2474-2543):
1. Load TOML config
2. Setup logging (shared with live)
3. Resolve private key, wallet, network
4. Query metadata (asset, collateral, decimals)
5. Initialize paper balance: **$1000 USD** (line 2736)

**Why $1000?** (comment line 2732-2735)
> "without this, margin_available=0 and no orders are placed. Paper balance should be a realistic account size, NOT the max_position_usd (which is just the position limit). Default $1000 ensures orders pass the $10 minimum notional check even for low-priced assets."

**Configuration Phase** (lines 2564-2662):
1. Create `PaperEnvironment` with `FillSimulatorConfig::default()`
2. Build `MmConfig` (same as live)
3. Select strategy: `LadderStrategy` (same as live)
4. Initialize metrics, estimators, all modules

**Execution Phase** (lines 2698-2746):
1. Create `MarketMaker<LadderStrategy, PaperEnvironment>`
2. Enable experience logging: `with_experience_logging("logs/experience")`
3. Set checkpoint dir: `with_checkpoint_dir(format!("data/checkpoints/paper/{asset}"))`
4. Disable Binance signals: `disable_binance_signals()`
5. Set paper balance: `with_paper_balance(1000.0)`

**Duration Handling** (lines 2740-2770):
```rust
if duration > 0 {
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(duration),
        market_maker.run(),
    ).await;
    
    match result {
        Ok(Ok(())) => info!("Paper trading completed normally"),
        Ok(Err(e)) => return Err(e.into()),
        Err(_) => {
            // Timeout: extract prior and save
            let mut prior = market_maker.extract_prior();
            prior.readiness = gate.assess(&prior);
            // Save to data/checkpoints/paper/{asset}/prior.json
        }
    }
} else {
    // duration == 0: run indefinitely
    market_maker.run().await?;
}
```

**Exit Behavior**:
- Duration > 0: Runs for that many seconds, then auto-saves prior
- Duration == 0: Runs until ^C, then auto-saves prior on shutdown
- Both paths: Assess `PriorReadiness`, stamp into checkpoint

### Command-Line Script

**Usage**: `./scripts/paper_trading.sh [ASSET] [DURATION] [OPTIONS]`

**Features** (paper_trading.sh):
- Asset: BTC, ETH, HYPE, etc. (default: BTC)
- Duration: seconds (default: 60)
- Dashboard: Python HTTP server on port 3000
- Screenshot capture: Puppeteer automation, saves to `tools/dashboard-capture/screenshots/{YYYY-MM-DD}/`
- Analysis: Calls `scripts/analysis/analyze_session.sh` if `--report`
- Pre-flight: Builds `cargo build --bin market_maker`

**Example**:
```bash
./scripts/paper_trading.sh BTC 1800 --report  # 30 min, then calibration report
./scripts/paper_trading.sh HYPE 300 --dashboard --capture  # 5 min with vision
```

---

## 7. Key Implementation Details

### Missing/Stubbed Features

1. **Stale Data Circuit Breaker** — No automatic adjustment for old L2 snapshots
2. **Bootstrap from Book** — Config field exists but not implemented
3. **BaselineTracker Wiring** — Structure exists, not connected to RL reward
4. **QuoteOutcomeTracker Persistence** — Tracked in-memory, not checkpointed

### Design Decisions

1. **3 WS Channels (vs 9 for Live)**
   - Paper doesn't need user-specific channels
   - Saves quota, reduces noise
   - All fill synthesis is local

2. **Separate PaperEnvironment**
   - Clean interface: same `TradingEnvironment` trait as live
   - Swap at MarketMaker construction time
   - No #[cfg] hacks

3. **Latency Simulation**
   - Placement latency: 100ms (prevent front-running artifacts)
   - Cancel latency: 50ms (realistic)
   - Affects fill probability calculation

4. **Paper Balance = $1000**
   - Not configurable (by design)
   - Ensures all low-priced assets pass $10 notional minimum
   - Prevents position-limit confusion

### Testing Coverage

**Simulation executor tests** exist for:
- Order placement/cancellation
- Post-only validation
- Statistics tracking

**Fill simulator tests** exist for:
- Queue position estimation
- Fill probability computation
- Trade matching logic

**No regression tests** for:
- Paper vs live equivalence
- Prior injection specifics
- Auto-calibration gate precision

---

## 8. Paper-to-Live Workflow (Complete)

### Phase 1: Paper Training
```bash
./target/debug/market_maker paper --asset BTC --duration 1800
```
- Subscribes: AllMids, L2Book, Trades
- Quotes: Real algorithm against real data
- Fills: Simulated from trade flow
- Analytics: Logged to data/analytics/
- Checkpoint: Saved to data/checkpoints/paper/BTC/prior.json
- Readiness: Stamped into checkpoint

### Phase 2: Readiness Check
```bash
./target/debug/market_maker run --asset BTC
```
- On startup: Check for prior at data/checkpoints/paper/BTC/prior.json
- Gate: CalibrationGate::passes(readiness)?
  - Yes → Proceed to live
  - No → Auto-run phase 1 for 1800s, then retry
- Skip: Use `--force` or `--skip-calibration`

### Phase 3: Live with Prior
- Prior injected at MarketMaker construction
- All learned state (kappa, AS, regime, etc.) loaded
- Kill switch NOT inherited
- Executes against live Hyperliquid WS

### Phase 4: Session Start
- Uses paper-learned kappa as prior
- Updates live on every trade
- Accumulates new fills for continued learning
- Checkpoints saved to data/checkpoints/live/{ASSET}/latest/

---

## Summary of State

| Aspect | Status | Notes |
|--------|--------|-------|
| **CLI Activation** | ✅ Implemented | `paper` subcommand, auto-calibration |
| **Paper Environment** | ✅ Implemented | Wraps SimulationExecutor + FillSimulator |
| **Fill Simulation** | ✅ Implemented | Probabilistic, queue-aware, configurable |
| **Analytics** | ✅ Implemented | 4-stream JSONL, Sharpe/attribution/edge |
| **Checkpoints** | ✅ Implemented | Full transfer protocol (φ, ψ), readiness gate |
| **Paper-to-Live** | ✅ Implemented | Automatic injection, auto-calibration pipeline |
| **Experience Logging** | ✅ Implemented | For offline RL training |
| **Dashboard** | ✅ Partially | HTTP API works, Puppeteer capture available |
| **Stale Data Defense** | ❌ Stubbed | No automatic circuit breaker |
| **Bootstrap from Book** | ❌ Stubbed | Config exists, not used |
