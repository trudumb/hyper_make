# Debug Logging Design Specification

**Date**: 2026-01-01
**Status**: Design Complete

---

## Executive Summary

The market maker logging system is redesigned to provide:
- **Component isolation** for focused debugging
- **Log rotation** to prevent disk exhaustion
- **Separate streams** for operational vs diagnostic logs
- **Performance-safe** async writes
- **Runtime configurability** via environment variables

---

## 1. Current State Analysis

### Problems Identified

| Issue | Impact |
|-------|--------|
| No log rotation | mm.log grew to 217MB (~6MB/hour at debug) |
| Flat structure | Hard to filter by component |
| No structured spans | Difficult to trace request flow |
| Mixed concerns | Metrics + operational logs interleaved |
| Static config | Can't change levels at runtime without restart |

### Current Implementation

- 249 log statements across 27 files
- Uses `tracing` + `tracing-subscriber`
- Supports JSON, compact, pretty formats
- Single log file + stdout
- Level control via `RUST_LOG` env var

---

## 2. Component Hierarchy

Organize logs by module using tracing targets:

```
hyperliquid_rust_sdk::market_maker
├── core                    # Main orchestrator lifecycle
├── strategy               # Strategy decisions (GLFT, ladder)
├── estimator              # Parameter estimation (σ, κ, microprice)
├── risk                   # Risk management & kill switch
├── execution              # Order placement/cancellation
├── fills                  # Fill processing & P&L
├── tracking               # Position & order tracking
├── infra                  # Infrastructure (rate limits, reconnection)
└── metrics                # Prometheus metrics emission
```

### Log Level Guidelines by Component

| Component | TRACE | DEBUG | INFO | WARN | ERROR |
|-----------|-------|-------|------|------|-------|
| core | msg routing | state changes | lifecycle | - | fatal |
| strategy | calculations | quote details | decisions | - | - |
| estimator | all updates | warmup | - | stale data | - |
| risk | - | checks | thresholds | breaches | kill switch |
| execution | API details | responses | orders | rejections | failures |
| fills | dedup | processing | fills | - | - |
| infra | - | health | sync | rate limits | disconnects |
| metrics | all emissions | - | - | - | - |

---

## 3. Structured Spans

Use `#[instrument]` for request tracing:

```rust
#[instrument(name = "quote_cycle", skip_all, fields(asset = %self.config.asset))]
async fn update_quotes(&mut self) -> Result<()> {
    // Spans appear as: quote_cycle{asset="BTC"}
}

#[instrument(name = "fill_processing", skip_all, fields(oid, tid))]
fn process_fill(&mut self, fill: &FillEvent) {
    // Child spans nested under parent
}
```

### Key Spans

| Span Name | Purpose | Fields |
|-----------|---------|--------|
| `quote_cycle` | Entire quote generation | asset |
| `order_placement` | Bulk order API call | count, side |
| `fill_processing` | Fill event handling | oid, tid, side |
| `risk_evaluation` | Risk aggregator check | severity |
| `safety_sync` | Periodic reconciliation | cycle_id |
| `parameter_update` | Estimator tick | sigma, kappa |

---

## 4. Log Rotation Strategy

### Size-Based Rotation (Primary)

| Parameter | Value |
|-----------|-------|
| Max file size | 50 MB |
| Rotated files kept | 5 |
| Total max disk usage | 250 MB per stream |
| Compression | gzip for rotated files |

### Time-Based Rotation (Secondary)

| Parameter | Value |
|-----------|-------|
| Rotation frequency | Daily at midnight UTC |
| Retention period | 7 days |

### Implementation

Use `tracing-appender` for daily rotation with external logrotate for size limits:

```rust
use tracing_appender::rolling::{RollingFileAppender, Rotation};

let file_appender = RollingFileAppender::new(
    Rotation::DAILY,
    "/var/log/market_maker",
    "mm.log",
);
let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);
```

---

## 5. Multi-Stream Output

### Log File Structure

```
logs/
├── mm-operational.log      # INFO+, trading decisions
├── mm-diagnostic.log       # DEBUG+, troubleshooting
└── mm-errors.log           # WARN+, issues only
```

### Stream Configuration

| Stream | Level Filter | Component Filter | Rotation |
|--------|--------------|------------------|----------|
| operational | INFO+ | all | daily, 7 days |
| diagnostic | DEBUG+ | estimator, tracking, fills | daily, 3 days |
| errors | WARN+ | all | daily, 30 days |
| stdout | configurable | all | none |

### Example Configuration

```rust
pub struct LogConfig {
    pub log_dir: PathBuf,
    pub operational_level: LevelFilter,    // Default: INFO
    pub diagnostic_level: LevelFilter,     // Default: DEBUG
    pub error_level: LevelFilter,          // Default: WARN
    pub rotation_size_mb: u64,             // Default: 50
    pub rotation_count: u32,               // Default: 5
    pub enable_stdout: bool,               // Default: true
    pub stdout_format: LogFormat,          // JSON, compact, pretty
}
```

---

## 6. Implementation Pattern

### Module-Specific Logging

```rust
// In estimator/volatility.rs
use tracing::{debug, info, warn};

pub fn update_volatility(&mut self, return_val: f64) {
    // Use module target for filtering
    debug!(
        target: "market_maker::estimator",
        sigma = %self.sigma,
        rv = %self.realized_variance,
        "Volatility updated"
    );
}
```

### Conditional Expensive Formatting

```rust
// Gate expensive debug computations
if tracing::enabled!(tracing::Level::DEBUG) {
    let debug_state = self.compute_debug_snapshot();
    debug!(state = ?debug_state, "Full state dump");
}
```

### Sampling for High-Frequency Events

```rust
// Sample 1 in 100 for ultra-high-frequency logs
static SAMPLE_COUNTER: AtomicU64 = AtomicU64::new(0);

if SAMPLE_COUNTER.fetch_add(1, Ordering::Relaxed) % 100 == 0 {
    trace!(target: "market_maker::metrics", value = %v, "Metric sample");
}
```

---

## 7. Initialization Code

```rust
use tracing_subscriber::{
    filter::EnvFilter,
    fmt,
    layer::SubscriberExt,
    util::SubscriberInitExt,
    Layer,
};
use tracing_appender::rolling::{RollingFileAppender, Rotation};

pub fn init_logging(config: &LogConfig) -> Vec<tracing_appender::non_blocking::WorkerGuard> {
    let mut guards = Vec::new();

    // Operational log (INFO+)
    let operational_appender = RollingFileAppender::new(
        Rotation::DAILY,
        &config.log_dir,
        "mm-operational.log",
    );
    let (operational_writer, guard1) = tracing_appender::non_blocking(operational_appender);
    guards.push(guard1);

    let operational_layer = fmt::layer()
        .with_writer(operational_writer)
        .with_ansi(false)
        .json()
        .with_filter(EnvFilter::new("info"));

    // Diagnostic log (DEBUG+, specific components)
    let diagnostic_appender = RollingFileAppender::new(
        Rotation::DAILY,
        &config.log_dir,
        "mm-diagnostic.log",
    );
    let (diagnostic_writer, guard2) = tracing_appender::non_blocking(diagnostic_appender);
    guards.push(guard2);

    let diagnostic_layer = fmt::layer()
        .with_writer(diagnostic_writer)
        .with_ansi(false)
        .json()
        .with_filter(EnvFilter::new(
            "market_maker::estimator=debug,market_maker::tracking=debug,market_maker::fills=debug"
        ));

    // Error log (WARN+)
    let error_appender = RollingFileAppender::new(
        Rotation::DAILY,
        &config.log_dir,
        "mm-errors.log",
    );
    let (error_writer, guard3) = tracing_appender::non_blocking(error_appender);
    guards.push(guard3);

    let error_layer = fmt::layer()
        .with_writer(error_writer)
        .with_ansi(false)
        .json()
        .with_filter(EnvFilter::new("warn"));

    // Stdout layer
    let stdout_layer = fmt::layer()
        .with_filter(EnvFilter::from_default_env());

    tracing_subscriber::registry()
        .with(operational_layer)
        .with(diagnostic_layer)
        .with(error_layer)
        .with(stdout_layer)
        .init();

    guards
}
```

---

## 8. Runtime Configuration

### Environment Variable Control

```bash
# Full debug for estimator only
RUST_LOG=hyperliquid_rust_sdk::market_maker::estimator=debug cargo run --bin market_maker

# Warn for all, debug for risk
RUST_LOG=warn,hyperliquid_rust_sdk::market_maker::risk=debug cargo run --bin market_maker

# Trace for execution (see all API calls)
RUST_LOG=hyperliquid_rust_sdk::market_maker::execution=trace cargo run --bin market_maker
```

### Future: SIGHUP Reload

```rust
// Planned: reload log config on SIGHUP
signal::ctrl_c().await?;
// or
unix::signal(SignalKind::hangup())?.recv().await;
reload_log_config(&new_config);
```

---

## 9. Performance Considerations

| Concern | Mitigation |
|---------|------------|
| String allocation | Use `%` for Display, `?` for Debug |
| Clone of large structs | `#[instrument(skip_all)]` |
| Blocking I/O | `non_blocking` writer from tracing-appender |
| High-frequency logs | Sampling + conditional gating |
| JSON serialization | Only for file output, compact for stdout |

### Benchmarks (Expected)

| Operation | Before | After |
|-----------|--------|-------|
| Log write (file) | 50μs sync | 1μs async |
| Debug format (skip) | 10μs | 0.1μs |
| Disabled level check | 0.5μs | 0.05μs |

---

## 10. Migration Steps

1. **Add log targets to existing statements** - prefix with component path
2. **Add `#[instrument]` to key functions** - quote_cycle, fill_processing, etc.
3. **Update init_logging in main.rs** - multi-layer setup
4. **Add LogConfig to MarketMakerConfig** - CLI flags for log dir/levels
5. **Deploy with logrotate** - system-level rotation config
6. **Monitor disk usage** - alert if logs exceed 200MB

---

## 11. Example Output

### Operational Log (JSON)

```json
{"timestamp":"2026-01-01T12:00:00.123Z","level":"INFO","target":"market_maker::core","span":{"name":"quote_cycle","asset":"BTC"},"message":"Quotes updated","bid_count":5,"ask_count":5}
```

### Diagnostic Log (JSON)

```json
{"timestamp":"2026-01-01T12:00:00.124Z","level":"DEBUG","target":"market_maker::estimator","message":"Volatility updated","sigma":0.00023,"rv":5.3e-8,"regime":"normal"}
```

### Error Log (JSON)

```json
{"timestamp":"2026-01-01T12:00:00.500Z","level":"WARN","target":"market_maker::infra","message":"IP rate limit warning","ip_weight_used":980,"budget":1200}
```

---

## 12. Summary

This design addresses all identified problems:

| Problem | Solution |
|---------|----------|
| No rotation | tracing-appender daily + logrotate |
| Flat structure | Component-based targets |
| No request tracing | #[instrument] spans |
| Mixed concerns | Separate log streams |
| Static config | EnvFilter + future SIGHUP |
| Performance | Non-blocking + conditional gating |

**Implementation effort**: ~4 hours for full migration
