# Market Maker Standard Operating Procedures (SOP)

**Version:** 2.0
**Last Updated:** 2025-01-02
**Document Type:** Standard Operating Procedure

This document provides comprehensive operational guidance for the Hyperliquid market maker, including setup, daily operations, monitoring, troubleshooting, and emergency procedures.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Pre-Flight Checklist](#2-pre-flight-checklist)
3. [Installation & Setup](#3-installation--setup)
4. [Configuration Reference](#4-configuration-reference)
5. [Operational Procedures](#5-operational-procedures)
6. [HIP-3 DEX Operations](#6-hip-3-dex-operations)
7. [Monitoring & Alerting](#7-monitoring--alerting)
8. [Risk Management](#8-risk-management)
9. [Troubleshooting Guide](#9-troubleshooting-guide)
10. [Emergency Procedures](#10-emergency-procedures)
11. [Maintenance Procedures](#11-maintenance-procedures)
12. [Architecture Reference](#12-architecture-reference)
13. [Module Reference](#13-module-reference)
14. [Quick Reference Card](#14-quick-reference-card)

---

## 1. System Overview

### 1.1 Purpose

This is a **production-grade automated market maker** for the Hyperliquid decentralized exchange. It provides continuous two-sided liquidity by placing bid and ask orders around a fair price, earning the spread while managing inventory risk.

### 1.2 Core Philosophy

| Principle | Description |
|-----------|-------------|
| **First-Principles Mathematics** | All decisions derive from stochastic control theory (GLFT model), not ad-hoc heuristics |
| **Data-Driven Adaptation** | Parameters (volatility, order flow, adverse selection) are estimated live from market data |
| **Defense in Depth** | Multiple layers of risk controls prevent catastrophic losses |
| **Modular Architecture** | Components are isolated, testable, and replaceable |

### 1.3 Key Capabilities

- **Optimal Quote Placement**: Guéant-Lehalle-Fernandez-Tapia (GLFT) model for spread and inventory skew
- **Multi-Level Ladder Quoting**: 5+ price levels per side for deeper liquidity provision
- **Live Parameter Estimation**: σ (volatility), κ (book depth), microprice from real-time data
- **Adverse Selection Measurement**: Ground truth E[Δp | fill] tracking
- **Liquidation Cascade Detection**: Hawkes process for tail risk identification
- **Kill Switch Protection**: Automatic shutdown on loss/position/staleness limits
- **HIP-3 DEX Support**: Trade on builder-deployed perpetuals (Hyena, Felix, etc.)

---

## 2. Pre-Flight Checklist

### 2.1 Before First Run

- [ ] Rust 1.70+ installed (`rustc --version`)
- [ ] Repository cloned and built (`cargo build --release`)
- [ ] All tests passing (`cargo test`)
- [ ] Private key configured (see [3.3 Environment Setup](#33-environment-setup))
- [ ] Configuration file created (`cargo run --bin market_maker -- generate-config`)
- [ ] Network selected (testnet for testing, mainnet for production)
- [ ] Account funded with sufficient collateral

### 2.2 Before Each Session

- [ ] Check account status: `cargo run --bin market_maker -- status`
- [ ] Verify no orphaned orders from previous session
- [ ] Confirm exchange connectivity (WebSocket, REST)
- [ ] Review current market conditions (volatility, spread regime)
- [ ] Validate configuration: `cargo run --bin market_maker -- validate-config`
- [ ] Run dry-run test: `cargo run --bin market_maker -- --dry-run`

### 2.3 Production Deployment Checklist

- [ ] Release build used (`cargo build --release`)
- [ ] Log file configured (`--log-file /var/log/market_maker.log`)
- [ ] Metrics endpoint enabled (`--metrics-port 9090`)
- [ ] Log rotation configured (external, e.g., logrotate)
- [ ] Alerting configured (Prometheus alertmanager)
- [ ] Backup private key secured (not on trading server)
- [ ] Kill switch thresholds reviewed and appropriate
- [ ] Process supervisor configured (systemd, supervisord)

---

## 3. Installation & Setup

### 3.1 Prerequisites

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Rust | 1.70+ | Latest stable |
| Memory | 512MB | 2GB |
| CPU | 1 core | 2+ cores |
| Network | Stable internet | Low-latency connection |
| Account | Funded Hyperliquid account | Sufficient margin for target position |

### 3.2 Installation

```bash
# Clone the repository
git clone <repo-url>
cd hyper_make

# Build release binary (required for production)
cargo build --release

# Verify build
./target/release/market_maker --version

# Run all tests
cargo test

# Run full CI pipeline (build, fmt, clippy, test)
./ci.sh
```

### 3.3 Environment Setup

**Option A: Environment Variable (Recommended)**
```bash
# Add to ~/.bashrc or ~/.zshrc
export HYPERLIQUID_PRIVATE_KEY="your_private_key_here"

# Reload shell
source ~/.bashrc
```

**Option B: .env File**
```bash
echo "HYPERLIQUID_PRIVATE_KEY=your_private_key_here" > .env
```

**Option C: Config File** (Not recommended for production)
```toml
# In market_maker.toml
[network]
private_key = "your_private_key_here"
```

**Security Notes:**
- Never commit private keys to version control
- Use environment variables for production deployments
- Consider using a secrets manager (Vault, AWS Secrets Manager)
- Restrict file permissions: `chmod 600 .env`

---

## 4. Configuration Reference

### 4.1 Generate Configuration

```bash
# Generate sample config with all options documented
cargo run --bin market_maker -- generate-config

# Output: market_maker.toml
```

### 4.2 Configuration File Structure

```toml
# market_maker.toml - Complete Reference

#══════════════════════════════════════════════════════════════════════════════
# NETWORK CONFIGURATION
#══════════════════════════════════════════════════════════════════════════════
[network]
base_url = "testnet"  # Options: mainnet, testnet, localhost
# private_key = "..."  # Prefer env var HYPERLIQUID_PRIVATE_KEY

#══════════════════════════════════════════════════════════════════════════════
# TRADING PARAMETERS
#══════════════════════════════════════════════════════════════════════════════
[trading]
asset = "BTC"                         # Trading pair (BTC, ETH, SOL, etc.)
target_liquidity = 0.01               # Order size per side in asset units
risk_aversion = 0.3                   # γ: 0.1 (aggressive) to 1.0 (conservative)
max_bps_diff = 5                      # Requote threshold in basis points
max_absolute_position_size = 0.05     # Maximum position in asset units
# leverage = 20                       # Optional: uses max available if not set

#══════════════════════════════════════════════════════════════════════════════
# STRATEGY CONFIGURATION
#══════════════════════════════════════════════════════════════════════════════
[strategy]
strategy_type = "ladder"         # Options: symmetric, inventory_aware, glft, ladder
estimation_window_secs = 300     # Rolling window for parameter estimation
min_trades = 50                  # Minimum trades before quoting begins
warmup_decay_secs = 300          # Adaptive warmup decay period

[strategy.ladder_config]
num_levels = 5                   # Number of price levels per side
min_depth_bps = 5                # Closest level distance from mid
max_depth_bps = 50               # Farthest level distance from mid
geometric_spacing = true         # Use geometric vs linear level spacing

#══════════════════════════════════════════════════════════════════════════════
# LOGGING CONFIGURATION
#══════════════════════════════════════════════════════════════════════════════
[logging]
level = "info"                   # Options: trace, debug, info, warn, error
format = "pretty"                # Options: pretty, json, compact

#══════════════════════════════════════════════════════════════════════════════
# MONITORING CONFIGURATION
#══════════════════════════════════════════════════════════════════════════════
[monitoring]
metrics_port = 9090              # Prometheus metrics port (0 to disable)
enable_http_metrics = true       # Enable HTTP metrics endpoint
```

### 4.3 CLI Options Reference

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `-c, --config` | PATH | `market_maker.toml` | Path to configuration file |
| `--asset` | STRING | `BTC` | Trading asset symbol |
| `--target-liquidity` | FLOAT | Config value | Order size per side |
| `--risk-aversion` | FLOAT | `0.3` | Gamma (γ) parameter |
| `--max-bps-diff` | FLOAT | `5` | Requote threshold |
| `--max-position` | FLOAT | Config value | Maximum position size |
| `--leverage` | INT | Max available | Leverage setting |
| `--decimals` | INT | Auto | Price decimals override |
| `--network` | STRING | `testnet` | Network selection |
| `--private-key` | STRING | Env var | Private key (prefer env var) |
| `--log-level` | STRING | `info` | Logging verbosity |
| `--log-format` | STRING | `pretty` | Log output format |
| `--log-file` | PATH | None | Log file path |
| `--multi-stream-logs` | FLAG | false | Enable multi-stream logging |
| `--log-dir` | PATH | `logs/` | Multi-stream log directory |
| `--metrics-port` | INT | `9090` | Prometheus metrics port |
| `--dry-run` | FLAG | false | Validate without trading |
| `--initial-isolated-margin` | FLOAT | `1000.0` | HIP-3 isolated margin (USD) |
| `--force-isolated` | FLAG | false | Force isolated margin mode |
| `--dex` | STRING | None | HIP-3 DEX name |
| `--list-dexs` | FLAG | false | List available DEXs |

### 4.4 Parameter Sizing Guidelines

**Position Sizing (Runtime Capped):**
```
max_position ≤ (account_value × leverage × 0.5) / price
target_liquidity ≤ max_position × 0.4
```

**Risk Aversion (γ) Guidelines:**

| γ Value | Profile | Spread | Inventory Sensitivity |
|---------|---------|--------|----------------------|
| 0.1 | Aggressive | Tight | Low |
| 0.3 | Moderate | Medium | Medium |
| 0.5 | Conservative | Wide | High |
| 1.0 | Very Conservative | Very Wide | Very High |

**Asset Selection:**

| Asset | Liquidity | Recommended For |
|-------|-----------|-----------------|
| BTC | Highest | Production, beginners |
| ETH | High | Production |
| SOL | Medium | Experienced operators |
| Altcoins | Lower | Advanced users only |

---

## 5. Operational Procedures

### 5.1 Starting the Market Maker

**Development/Testing:**
```bash
# Basic testnet run
cargo run --bin market_maker -- --asset BTC

# With debug logging
RUST_LOG=hyperliquid_rust_sdk::market_maker=debug cargo run --bin market_maker
RUST_LOG=hyperliquid_rust_sdk::market_maker=debug cargo run --bin market_maker -- --network mainnet --asset BTC --dex hyna
RUST_LOG=hyperliquid_rust_sdk::market_maker=debug cargo run --bin market_maker -- --network mainnet --asset BTC --dex hyna

# Dry run (validates everything without placing orders)
cargo run --bin market_maker -- --dry-run
```

**Production:**
```bash
# Full production command
RUST_LOG=hyperliquid_rust_sdk::market_maker=info \
./target/release/market_maker \
  --config /etc/market_maker/production.toml \
  --asset BTC \
  --network mainnet \
  --log-file /var/log/market_maker/mm.log \
  --metrics-port 9090
```

### 5.2 Stopping the Market Maker

**Graceful Shutdown (Preferred):**
```bash
# Send SIGINT (Ctrl+C)
# Market maker will:
# 1. Stop accepting new trades
# 2. Cancel all resting orders with retry
# 3. Log final position and P&L
# 4. Log adverse selection summary
# 5. Exit cleanly
```

**Emergency Shutdown:**
```bash
# Send SIGTERM
kill -TERM $(pgrep market_maker)

# Force kill (last resort - may leave orphaned orders!)
kill -9 $(pgrep market_maker)
```

**Post-Shutdown Verification:**
```bash
# Check for orphaned orders
cargo run --bin market_maker -- status

# Manually cancel if needed (use Hyperliquid UI)
```

### 5.3 Checking Status

```bash
# Full account status
cargo run --bin market_maker -- status

# Output includes:
# - Current position
# - Account balance
# - Open orders
# - Available margin
# - Leverage settings
```

### 5.4 Subcommands Reference

| Command | Purpose | When to Use |
|---------|---------|-------------|
| `generate-config` | Create sample config | Initial setup |
| `validate-config` | Validate config file | Before starting |
| `status` | Show account state | Before/after sessions |
| `run` | Run market maker | Normal operation (default) |

---

## 6. HIP-3 DEX Operations

### 6.1 Overview

HIP-3 allows third-party builders to deploy their own perpetual markets. When multiple DEXs offer the same asset (e.g., "BTC"), you must specify which DEX to trade on.

### 6.2 Listing Available DEXs

```bash
# List all HIP-3 DEXs
cargo run --bin market_maker -- --list-dexs

# Output example:
# Available HIP-3 DEXs:
#   hyena - Hyena Exchange (deployer: 0x...)
#   felix - Felix Protocol (deployer: 0x...)
```

### 6.3 Trading on HIP-3 DEXs

```bash
# Trade Hyena's BTC perp
cargo run --bin market_maker -- --asset BTC --dex hyena

# Trade Felix's BTC with custom margin
cargo run --bin market_maker -- --asset BTC --dex felix --initial-isolated-margin 2000

# Force isolated margin mode
cargo run --bin market_maker -- --asset BTC --dex hyena --force-isolated
```

### 6.4 HIP-3 Considerations

| Factor | Validator Perps | HIP-3 DEX Perps |
|--------|-----------------|-----------------|
| Liquidity | Highest | Varies by DEX |
| Margin Mode | Cross (default) | Isolated (required) |
| OI Limits | Exchange-wide | Per-DEX caps |
| Fee Structure | Standard | DEX-specific |

**HIP-3 CLI Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--dex <name>` | None | DEX name (e.g., "hyena", "felix") |
| `--list-dexs` | - | List available DEXs and exit |
| `--initial-isolated-margin` | $1,000 | Initial margin allocation |
| `--force-isolated` | false | Force isolated margin mode |

### 6.5 HIP-3 OI Cap Handling

The market maker automatically detects and respects OI caps:
- Queries DEX-specific limits at startup
- Reduces position sizes if approaching caps
- Logs warnings when near capacity

### 6.6 HIP-3 Testing Workflow

**Quick Start:**
```bash
# Use the test script (recommended)
./scripts/test_hip3.sh BTC hyna 60    # 1 minute test
./scripts/test_hip3.sh BTC hyna 300   # 5 minute test
./scripts/test_hip3.sh ETH flx 120    # 2 minute test on Felix
```

**Manual Command with Timestamped Logs:**
```bash
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S) && \
RUST_LOG=hyperliquid_rust_sdk::market_maker=debug \
cargo run --bin market_maker -- \
  --network mainnet \
  --asset BTC \
  --dex hyna \
  --log-file logs/mm_hyna_BTC_${TIMESTAMP}.log
```

**Log Naming Convention:**
```
logs/mm_{network|dex}_{asset}_{YYYY-MM-DD}_{HH-MM-SS}.log
```

| Identifier | Use Case |
|------------|----------|
| `testnet` | Testnet trading |
| `mainnet` | Validator perps production |
| `{dex}` | HIP-3 DEX (e.g., `hyna`, `flx`) |

### 6.7 Log Analysis Workflow

After running a test session:

1. **Quick Log Stats:**
   ```bash
   # Count key events
   grep -c "ERROR" logs/mm_hyna_BTC_*.log
   grep -c "WARN" logs/mm_hyna_BTC_*.log
   grep -c "Quote cycle" logs/mm_hyna_BTC_*.log
   grep -c "Trades processed" logs/mm_hyna_BTC_*.log
   ```

2. **Analyze with Claude:**
   Provide the log file to Claude and request `sc:analyze` for:
   - Behavior summary
   - Issues categorized by severity
   - Recommended fixes with code locations
   - Questions for implementation decisions

3. **Create Serena Checkpoint:**
   Claude will create a session memory for tracking:
   ```
   session_{YYYY-MM-DD}_{short_description}
   ```

4. **Track Fixes via Git PRs:**
   - Branch: `fix/{issue-description}`
   - Commit: Conventional format with Claude Code footer

### 6.8 Key Metrics to Monitor

During HIP-3 testing, watch for:

| Metric | Healthy Range | Warning Sign |
|--------|---------------|--------------|
| `mm_spread_bps` | 5-15 bps | >30 bps consistently |
| `mm_jump_ratio` | <1.5 | >3.0 (toxic flow) |
| `mm_inventory_utilization` | <50% | >80% |
| `mm_adverse_selection_bps` | <3 bps | >10 bps |
| Quote cycle latency | <100ms | >500ms |

---

## 7. Monitoring & Alerting

### 7.1 Prometheus Metrics

**Access Point:** `http://localhost:9090/metrics`

**Position Metrics:**
```
mm_position{asset="BTC"} 0.125           # Current position
mm_max_position{asset="BTC"} 0.5         # Position limit
mm_inventory_utilization{asset="BTC"} 0.25  # position / max_position
```

**P&L Metrics:**
```
mm_daily_pnl{asset="BTC"} 15.50          # Session P&L (realized + unrealized)
mm_realized_pnl{asset="BTC"} 12.30       # Closed trade P&L
mm_unrealized_pnl{asset="BTC"} 3.20      # Mark-to-market
mm_peak_pnl{asset="BTC"} 20.00           # Session high
mm_drawdown_pct{asset="BTC"} 0.02        # Current drawdown
```

**Market Metrics:**
```
mm_mid_price{asset="BTC"} 100500.50      # Current mid price
mm_spread_bps{asset="BTC"} 8.5           # Bid-ask spread
mm_sigma{asset="BTC"} 0.00015            # Volatility estimate
mm_kappa{asset="BTC"} 85.2               # Order flow intensity
mm_jump_ratio{asset="BTC"} 1.2           # RV/BV ratio (toxicity)
```

**Estimator Metrics:**
```
mm_microprice_deviation_bps{asset="BTC"} 0.5  # Microprice vs mid
mm_book_imbalance{asset="BTC"} 0.15           # L2 book imbalance (-1 to 1)
mm_flow_imbalance{asset="BTC"} -0.08          # Trade flow imbalance
mm_beta_book{asset="BTC"} 0.003               # Learned book coefficient
mm_beta_flow{asset="BTC"} 0.002               # Learned flow coefficient
```

**Risk Metrics:**
```
mm_kill_switch_triggered{asset="BTC"} 0       # 0=running, 1=triggered
mm_cascade_severity{asset="BTC"} 0.15         # Liquidation cascade (0-1)
mm_adverse_selection_bps{asset="BTC"} 0.8     # Running AS estimate
mm_tail_risk_multiplier{asset="BTC"} 1.0      # Gamma scaling factor
```

**Infrastructure Metrics:**
```
mm_data_staleness_secs{asset="BTC"} 0.5       # Data age
mm_quote_cycle_latency_ms{asset="BTC"} 15     # Quote cycle time
mm_volatility_regime{asset="BTC"} 0           # 0=Normal, 1=High, 2=Extreme
mm_orders_placed_total{asset="BTC"} 1250      # Total orders placed
mm_orders_filled_total{asset="BTC"} 180       # Total fills
```

### 7.2 Log Analysis

**Real-time Monitoring:**
```bash
# Watch all activity
tail -f mm.log

# Watch fills only
tail -f mm.log | grep "Fill processed"

# Watch quote updates
tail -f mm.log | grep "Calculated ladder"

# Watch risk events
tail -f mm.log | grep -E "(Kill switch|cascade|adverse selection|WARN|ERROR)"

# Watch warmup progress
tail -f mm.log | grep -E "(Warming up|warmed up)"
```

**JSON Log Parsing:**
```bash
# Parse JSON logs with jq
cat mm.log | jq 'select(.target == "hyperliquid_rust_sdk::market_maker")'

# Extract fills with position
cat mm.log | jq 'select(.message | contains("Fill")) | {time: .timestamp, msg: .message}'

# Count events by type
cat mm.log | jq -r '.message' | cut -d' ' -f1-3 | sort | uniq -c | sort -rn
```

### 7.3 Key Log Events

| Event | Level | Meaning | Action |
|-------|-------|---------|--------|
| `Market maker started` | INFO | Initialization complete | None |
| `Warming up parameter estimator` | INFO | Collecting initial data | Wait |
| `Parameter estimation warmed up` | INFO | Ready to quote | None |
| `Calculated ladder` | DEBUG | Quote calculation | None |
| `Fill processed` | INFO | Order filled | Monitor position |
| `Reduce-only mode` | WARN | Position at limit | Monitor |
| `Kill switch condition detected` | WARN | Risk limit approaching | Review |
| `Cascade detected` | WARN | Liquidations occurring | Monitor closely |
| `KILL SWITCH TRIGGERED` | ERROR | Emergency shutdown | Investigate |
| `Data staleness exceeded` | WARN | Market data delayed | Check connection |

### 7.4 Alerting Configuration

**Prometheus Alertmanager Rules (example):**
```yaml
groups:
  - name: market_maker
    rules:
      - alert: KillSwitchTriggered
        expr: mm_kill_switch_triggered == 1
        for: 0m
        labels:
          severity: critical
        annotations:
          summary: "Market maker kill switch triggered"

      - alert: HighDrawdown
        expr: mm_drawdown_pct > 0.03
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Drawdown exceeds 3%"

      - alert: DataStale
        expr: mm_data_staleness_secs > 3
        for: 30s
        labels:
          severity: warning
        annotations:
          summary: "Market data is stale"

      - alert: HighCascadeSeverity
        expr: mm_cascade_severity > 0.5
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "Liquidation cascade detected"
```

---

## 8. Risk Management

### 8.1 Kill Switch Conditions

| Condition | Default Threshold | Description |
|-----------|-------------------|-------------|
| Max Daily Loss | $500 | Cumulative session loss |
| Max Drawdown | 5% | Peak-to-trough P&L percentage |
| Max Position Value | $10,000 | \|position\| × mid_price |
| Stale Data | 5 seconds | No market data received |
| Cascade Severity | 0.95 | Liquidation cascade intensity |
| Rate Limit Errors | 10 | Exchange rate limiting count |

### 8.2 Graceful Shutdown Protocol

When kill switch triggers:
1. **Stop New Trades**: No new orders placed
2. **Cancel Orders**: Bulk cancel all resting orders
3. **Retry Cancels**: Retry failed cancellations up to 3x
4. **Log State**: Final position, P&L, adverse selection stats
5. **Exit**: Process terminates with non-zero exit code

### 8.3 Reduce-Only Mode

Automatically activated when:
- Position exceeds `max_position`
- Position value exceeds `max_position_value`

Behavior:
- Cancels orders that would increase position
- Only places orders that reduce exposure
- Logs WARN message when activated

### 8.4 Risk Parameter Tuning

**Dynamic Gamma (γ) Scaling:**

The effective gamma scales based on:
- Volatility regime (σ above baseline)
- Flow toxicity (RV/BV ratio)
- Inventory utilization
- Cascade severity

```
γ_effective = γ_base × volatility_mult × toxicity_mult × inventory_mult × tail_risk_mult
```

**Cascade Risk Management:**

| Cascade Severity | Action |
|-----------------|--------|
| 0.0 - 0.3 | Normal quoting |
| 0.3 - 0.5 | Widen spreads (γ × 1.5) |
| 0.5 - 0.8 | Reduce size (γ × 2.5) |
| 0.8 - 0.95 | Pull all quotes |
| > 0.95 | Kill switch |

---

## 9. Troubleshooting Guide

### 9.1 Startup Issues

**Problem: "Config file not found"**
```bash
# Solution: Generate config
cargo run --bin market_maker -- generate-config
```

**Problem: "Private key not set"**
```bash
# Solution: Set environment variable
export HYPERLIQUID_PRIVATE_KEY="0x..."
```

**Problem: "Failed to connect to exchange"**
```bash
# Check network setting
cargo run --bin market_maker -- --network testnet status

# Verify connectivity
curl https://api.hyperliquid.xyz/info
```

**Problem: "Asset not found"**
```bash
# Check asset exists
cargo run --bin market_maker -- status

# For HIP-3 assets, specify DEX
cargo run --bin market_maker -- --asset BTC --dex hyena
```

### 9.2 Runtime Issues

**Problem: "No quotes being placed"**
- Check warmup status (need 50+ trades, 20 volume ticks)
- Verify parameter estimator has warmed up
- Check for reduce-only mode (position at limit)
- Verify sufficient margin

**Problem: "All orders rejected"**
- Check account balance/margin
- Verify leverage settings
- Check minimum order size ($10 notional)
- Review rate limits

**Problem: "Orders placed but no fills"**
- Spread may be too wide (reduce γ)
- Check if market is active
- Review quote prices vs market

**Problem: "Excessive fills / inventory building"**
- Spread may be too tight (increase γ)
- Check for one-sided flow (toxicity)
- Review adverse selection metrics

**Problem: "WebSocket disconnections"**
```bash
# Check connection health in logs
tail -f mm.log | grep -E "(reconnect|disconnect|connection)"

# Monitor data staleness metric
curl -s localhost:9090/metrics | grep mm_data_staleness
```

### 9.3 Performance Issues

**Problem: "High quote cycle latency"**
- Check CPU usage
- Use release build (`cargo build --release`)
- Reduce logging level (`--log-level warn`)

**Problem: "Memory growing"**
- Check for fills backlog
- Monitor fill deduplication cache
- Review metrics accumulation

### 9.4 Common Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| `InsufficientMargin` | Not enough collateral | Add margin or reduce size |
| `OrderWouldBeLiquidated` | Order would cause liquidation | Reduce leverage/size |
| `RateLimited` | Too many requests | Reduce quoting frequency |
| `InvalidPrice` | Price precision error | Check decimals setting |
| `AssetNotFound` | Invalid asset | Check spelling, use --dex for HIP-3 |
| `DexNotFound` | Invalid HIP-3 DEX | Run --list-dexs |

---

## 10. Emergency Procedures

### 10.1 Kill Switch Triggered

**Immediate Actions:**
1. **Verify shutdown**: Check process is not running
2. **Check status**: `cargo run --bin market_maker -- status`
3. **Review cause**: Check last log entries
4. **Cancel orphans**: Manually cancel any remaining orders via UI

**Investigation:**
1. Review logs for kill switch trigger reason
2. Check metrics at time of trigger
3. Review market conditions
4. Document incident

**Recovery:**
1. Wait for market conditions to stabilize
2. Adjust risk parameters if needed
3. Run dry-run test
4. Restart with monitoring

### 10.2 Orphaned Orders

**Detection:**
```bash
# Check for open orders without running market maker
cargo run --bin market_maker -- status
```

**Resolution:**
1. Use Hyperliquid UI to cancel orders manually
2. Or restart market maker (will auto-cancel on startup)

### 10.3 Position Stuck

**Symptoms:**
- Large position that won't reduce
- Reduce-only mode active
- Low fill rate

**Resolution Options:**
1. **Wait**: Let reduce-only mode work naturally
2. **Widen spreads**: Restart with higher γ
3. **Manual trade**: Use Hyperliquid UI to market-close
4. **Stop market maker**: Cancel all orders, manage position manually

### 10.4 Exchange Issues

**Symptoms:**
- Repeated connection failures
- Orders not appearing on exchange
- Fill notifications delayed

**Actions:**
1. Stop market maker gracefully
2. Check Hyperliquid status (Discord, Twitter)
3. Wait for exchange issues to resolve
4. Verify account state via UI
5. Restart when stable

### 10.5 Emergency Contacts

| Issue Type | Resource |
|------------|----------|
| Exchange Issues | Hyperliquid Discord |
| Market Maker Bugs | GitHub Issues |
| Account Issues | Hyperliquid Support |

---

## 11. Maintenance Procedures

### 11.1 Daily Operations

| Task | Frequency | Procedure |
|------|-----------|-----------|
| Check account status | Before each session | `cargo run --bin market_maker -- status` |
| Review P&L | End of session | Check `mm_daily_pnl` metric |
| Review fills | Daily | Analyze fill quality, adverse selection |
| Check logs for errors | Daily | `grep -E "WARN|ERROR" mm.log` |

### 11.2 Weekly Operations

| Task | Procedure |
|------|-----------|
| Update repository | `git pull && cargo build --release` |
| Review performance | Analyze P&L, fill rate, adverse selection trends |
| Backup logs | Archive and rotate log files |
| Review risk parameters | Adjust based on performance |

### 11.3 Configuration Updates

```bash
# 1. Stop market maker gracefully (Ctrl+C)

# 2. Edit configuration
vim market_maker.toml

# 3. Validate changes
cargo run --bin market_maker -- validate-config

# 4. Test with dry-run
cargo run --bin market_maker -- --dry-run

# 5. Restart
cargo run --bin market_maker
```

### 11.4 Software Updates

```bash
# 1. Stop market maker gracefully

# 2. Pull latest changes
git pull

# 3. Rebuild
cargo build --release

# 4. Run tests
cargo test

# 5. Validate config (API changes may require updates)
cargo run --bin market_maker -- validate-config

# 6. Test with dry-run
cargo run --bin market_maker -- --dry-run

# 7. Restart
./target/release/market_maker --config production.toml
```

### 11.5 Log Management

**Log Rotation (logrotate example):**
```
/var/log/market_maker/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 0644 trader trader
}
```

---

## 12. Architecture Reference

### 12.1 System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         MarketMaker<S, E>                           │
│                     (Orchestrator / Event Loop)                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │   Strategy   │  │   Executor   │  │   Estimator  │               │
│  │  (GLFT/      │  │ (Hyperliquid │  │ (σ, κ, μ)    │               │
│  │   Ladder)    │  │   Exchange)  │  │              │               │
│  └──────────────┘  └──────────────┘  └──────────────┘               │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    Component Bundles                         │    │
│  ├─────────────────────────────────────────────────────────────┤    │
│  │  Tier1: AdverseSelection, QueueTracker, LiquidationDetector │    │
│  │  Tier2: Hawkes, Funding, Spread, PnL                        │    │
│  │  Safety: KillSwitch, RiskAggregator, FillProcessor          │    │
│  │  Infra: Margin, Prometheus, ConnectionHealth, DataQuality   │    │
│  │  Stochastic: HJBController, DynamicRisk                     │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Hyperliquid Exchange                             │
│  WebSocket: AllMids, L2Book, Trades, UserFills                      │
│  REST: Orders, Cancels, Account State                               │
└─────────────────────────────────────────────────────────────────────┘
```

### 12.2 Event Loop Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        Main Event Loop                           │
└─────────────────────────────────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌───────────────┐      ┌───────────────┐      ┌───────────────┐
│   AllMids     │      │    Trades     │      │  UserFills    │
│  (mid price)  │      │  (volatility) │      │   (fills)     │
└───────────────┘      └───────────────┘      └───────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌───────────────┐      ┌───────────────┐      ┌───────────────┐
│ Update        │      │ Update        │      │ Update        │
│ Microprice    │      │ σ, κ, regime  │      │ Position      │
└───────────────┘      └───────────────┘      └───────────────┘
        │                       │                       │
        └───────────────────────┼───────────────────────┘
                                ▼
                    ┌───────────────────────┐
                    │   Calculate Quotes    │
                    │   (Strategy + Params) │
                    └───────────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Reconcile Ladder    │
                    │   (Cancel/Place)      │
                    └───────────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Check Kill Switch   │
                    └───────────────────────┘
```

### 12.3 Quote Calculation

**1. Microprice Estimation:**
```
microprice = mid × (1 + β_book × book_imb + β_flow × flow_imb)
```

**2. GLFT Optimal Spread:**
```
δ = (1/γ) × ln(1 + γ/κ)
```

**3. Inventory Skew:**
```
skew = (q/Q_max) × γ × σ² × T
```

**4. Final Quotes:**
```
bid = microprice × (1 - δ - skew)
ask = microprice × (1 + δ - skew)
```

---

## 13. Module Reference

### 13.1 Module Organization (~24K lines, 74 files)

| Module | Purpose | Key Components |
|--------|---------|----------------|
| `strategy/` | Quote pricing logic | GLFTStrategy, LadderStrategy, RiskConfig |
| `estimator/` | Live parameter estimation | VolumeClock, BipowerVariation, Microprice |
| `tracking/` | Order & position state | OrderManager, PositionTracker, QueueTracker |
| `adverse_selection/` | Fill quality measurement | ASEstimator, DepthDecayAS |
| `process_models/` | Stochastic processes | Hawkes, Funding, Liquidation, Spread, HJB |
| `risk/` | Risk monitoring | KillSwitch, RiskAggregator, Monitors |
| `quoting/` | Quote generation | LadderGenerator, Optimizer, Filters |
| `infra/` | Infrastructure | Margin, Prometheus, Reconnection, DataQuality |
| `safety/` | Exchange reconciliation | SafetyAuditor |
| `core/` | Component bundles | Tier1, Tier2, Safety, Infra, Stochastic |

### 13.2 Strategy Module (`strategy/`)

| File | Purpose |
|------|---------|
| `glft.rs` | GLFT optimal market making |
| `ladder_strat.rs` | Multi-level ladder with GLFT |
| `simple.rs` | SymmetricStrategy, InventoryAwareStrategy |
| `risk_config.rs` | RiskConfig for dynamic gamma |
| `market_params.rs` | MarketParams aggregation |

### 13.3 Estimator Module (`estimator/`)

| File | Purpose |
|------|---------|
| `volume.rs` | Volume clock for time normalization |
| `volatility.rs` | Bipower variation σ estimation |
| `jump.rs` | Jump detection via RV/BV ratio |
| `kappa.rs` | Book depth decay estimation |
| `microprice.rs` | Fair price from signal regression |
| `kalman.rs` | Kalman filter price smoothing |

### 13.4 Risk Module (`risk/`)

| File | Purpose |
|------|---------|
| `kill_switch.rs` | Emergency shutdown |
| `aggregator.rs` | Multi-monitor risk eval |
| `monitors/` | Individual risk monitors |
| `state.rs` | Unified risk state |

### 13.5 Process Models (`process_models/`)

| File | Purpose |
|------|---------|
| `hawkes.rs` | Self-exciting order flow |
| `funding.rs` | Funding rate prediction |
| `liquidation.rs` | Cascade detection |
| `spread.rs` | Spread regime tracking |
| `hjb_control.rs` | HJB optimal inventory |

---

## 14. Quick Reference Card

### Build Commands

```bash
cargo build                     # Development build
cargo build --release           # Production build
cargo fmt && cargo clippy       # Format + lint
cargo test                      # Run all tests
./ci.sh                         # Full CI pipeline
```

### Operational Commands

```bash
# Config Management
cargo run --bin market_maker -- generate-config      # Create config
cargo run --bin market_maker -- validate-config      # Validate config

# Status & Testing
cargo run --bin market_maker -- status               # Account status
cargo run --bin market_maker -- --dry-run            # Dry run test

# Basic Run
cargo run --bin market_maker                         # Default (BTC/testnet)
cargo run --bin market_maker -- --asset ETH          # Different asset

# Production Run
RUST_LOG=info ./target/release/market_maker \
  --network mainnet \
  --log-file mm.log \
  --metrics-port 9090

# Debug Run
RUST_LOG=hyperliquid_rust_sdk::market_maker=debug \
cargo run --bin market_maker

# HIP-3 DEX
cargo run --bin market_maker -- --list-dexs          # List DEXs
cargo run --bin market_maker -- --asset BTC --dex hyena  # Trade on Hyena
```

### Monitoring Commands

```bash
# Metrics
curl localhost:9090/metrics
curl -s localhost:9090/metrics | grep mm_position

# Logs
tail -f mm.log                                        # All logs
tail -f mm.log | grep "Fill processed"               # Fills only
tail -f mm.log | grep -E "WARN|ERROR"                # Warnings/errors
```

### Key Metrics to Watch

| Metric | Normal Range | Alert Threshold |
|--------|--------------|-----------------|
| `mm_inventory_utilization` | 0 - 0.5 | > 0.8 |
| `mm_drawdown_pct` | 0 - 0.02 | > 0.03 |
| `mm_cascade_severity` | 0 - 0.3 | > 0.5 |
| `mm_data_staleness_secs` | 0 - 1 | > 3 |
| `mm_adverse_selection_bps` | 0 - 1.5 | > 2.5 |

### Emergency Contacts

| Resource | Purpose |
|----------|---------|
| Hyperliquid Discord | Exchange issues |
| GitHub Issues | Market maker bugs |
| Hyperliquid UI | Manual trading |

---

## Further Reading

- [CLAUDE.md](./CLAUDE.md) - Detailed architecture and implementation notes
- [Hyperliquid Docs](https://hyperliquid.gitbook.io/) - Exchange API reference
- [GLFT Paper](https://arxiv.org/abs/1105.3115) - Optimal market making theory

---

*Document maintained by the development team. Report issues via GitHub.*
