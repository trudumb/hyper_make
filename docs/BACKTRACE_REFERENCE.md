# Backtrace Reference — Hyperliquid Market Maker

A diagnostic guide for interpreting Rust backtraces from the market maker binary.
Enable full backtraces with `RUST_BACKTRACE=1` (or `=full` for all frames).

---

## Table of Contents

- [Quick Triage](#quick-triage)
- [Startup Panic Paths](#startup-panic-paths)
- [Runtime Panic Paths](#runtime-panic-paths)
  - [GLFT Formula (kappa / gamma)](#glft-formula--kappa--gamma)
  - [Kill Switch Mutex Poisoning](#kill-switch-mutex-poisoning)
  - [Signal Handler Registration](#signal-handler-registration)
  - [Pending Fill Queue](#pending-fill-queue)
- [Safe-by-Design Paths (Won't Panic)](#safe-by-design-paths-wont-panic)
- [Common Backtrace Frames](#common-backtrace-frames)
- [Debugging Checklist](#debugging-checklist)

---

## Quick Triage

| Panic message | Root cause | Severity | Fix |
|---|---|---|---|
| `config validation failed` | Invalid TOML config or env vars | **Startup** | Check config values, run with `--dry-run` |
| `kappa must be > 0` | Zero/negative kappa fed to GLFT | **Debug-only** | Check estimator output; release builds clamp to `EPSILON` |
| `gamma must be > 0` | Zero/negative gamma | **Debug-only** | Check risk aversion config; release builds clamp to `EPSILON` |
| `Failed to register SIGTERM handler` | OS signal setup failure | **Startup** | Check platform signal support |
| `called unwrap() on a poisoned Mutex` | Prior panic poisoned a lock in kill_switch | **Critical** | Find the *original* panic that poisoned the mutex |
| `called unwrap() on None` (pending_fill_outcomes) | Fill queue drained unexpectedly | **Runtime** | Check fill deduplication logic |

---

## Startup Panic Paths

### Config Validation (`src/bin/market_maker.rs:2056`)

```
thread 'main' panicked at 'config validation failed: Invalid MarketMakerConfig: ...'
  → src/bin/market_maker.rs:2056
```

**Cause**: `mm_config.validate().expect(...)` — the loaded config fails validation.

**Common triggers**:
- `kappa` or `gamma` set to `0.0` in TOML
- Missing required fields after a config schema change
- Invalid spread profile values
- `max_inventory <= 0`

**Fix**: Check your TOML config against the schema. The error message after the colon tells you exactly which field failed.

---

### Signal Handler Registration (`src/market_maker/orchestrator/event_loop.rs:858`)

```
thread 'main' panicked at 'Failed to register SIGTERM handler'
  → src/market_maker/orchestrator/event_loop.rs:858
```

**Cause**: `signal(SignalKind::terminate()).expect(...)` — OS refused signal registration.

**Fix**: Unusual — only happens on non-standard runtimes or inside containers with restricted signal masks.

---

## Runtime Panic Paths

### GLFT Formula — kappa / gamma

**Files**: `src/market_maker/strategy/spread_oracle.rs:43-44, 88`

```
thread 'main' panicked at 'assertion failed: kappa must be > 0, got 0'
  → src/market_maker/strategy/spread_oracle.rs:43
```

**Key detail**: These are `debug_assert!` — they **only fire in debug builds**.

| Location | Assertion | Fires in release? |
|---|---|---|
| `spread_oracle.rs:43` | `kappa > 0.0` | No (debug only) |
| `spread_oracle.rs:44` | `gamma > 0.0` | No (debug only) |
| `spread_oracle.rs:88` | `gamma > 0.0` | No (debug only) |

In **release builds**, the code clamps to safe values:
```rust
let safe_kappa = kappa.max(f64::EPSILON);  // line 46
let safe_gamma = gamma.max(f64::EPSILON);  // line 47
```

**If you see this in a debug build**:
1. Check the kappa estimator — is it producing zeros?
2. Check regime detection — is a regime transition resetting kappa?
3. Check config — is `kappa` explicitly set to `0.0`?

**Call chain** (typical):
```
main
 → MarketMaker::start()
   → EventLoop::run()                    [event_loop.rs:823]
     → handle_observation()              [event_loop.rs:934]
       → QuoteEngine::update_quotes()    [quote_engine.rs:30]
         → spread_for_kappa()            [spread_oracle.rs:36]  ← PANIC
```

---

### Kill Switch Mutex Poisoning

**File**: `src/market_maker/risk/kill_switch.rs` (lines 492, 529, 543, 553, 561, 608, etc.)

```
thread 'main' panicked at 'called `Result::unwrap()` on an `Err` value: PoisonError(...)'
```

The kill switch uses `Mutex<T>` for state, config, trigger reasons, and last fill time. All access uses `.lock().unwrap()`.

**This is a secondary panic** — a previous panic on another thread poisoned the mutex. The backtrace you care about is the **first** one. Look for an earlier panic in the logs.

**Mutex locations** (all in `kill_switch.rs`):
- `self.config.lock().unwrap()` — lines 492, 553, 633, 732, 744
- `self.state.lock().unwrap()` — lines 561, 608, 625, 695, 701, 707, 715, 721, 727
- `self.trigger_reasons.lock().unwrap()` — lines 529, 543, 763
- `self.last_fill_time.lock().unwrap()` — lines 644, 669

---

### Pending Fill Queue (`src/market_maker/orchestrator/handlers.rs:242`)

```rust
let pending = self.infra.pending_fill_outcomes.pop_front().unwrap();
```

**Cause**: The code expected a pending fill outcome but the queue was empty.

**When this happens**: A fill confirmation arrived but was already consumed (double-delivery or deduplication race).

---

## Safe-by-Design Paths (Won't Panic)

These look scary in code but are protected:

| Path | Protection |
|---|---|
| Price parsing (`handlers.rs` throughout) | `.parse::<f64>().unwrap_or(0.0)` — defaults on bad data |
| Division by mid price (`quote_engine.rs:276-279`) | Guarded by `if mid > 0.0 && prev_mid > 0.0` |
| Division by max position (`quote_engine.rs:75-76`) | Guarded by `if effective_max_position > 0.0` |
| Drawdown calculation (`quote_engine.rs:904-908`) | Guarded by `if peak_pnl > 0.0` |
| GLFT logarithm (`spread_oracle.rs:52`) | Taylor fallback when `ratio ≤ 1e-12` |
| GLFT exponent overflow (`spread_oracle.rs:104`) | Returns `gamma * exp(-exponent)` when `> 700` |
| GLFT denominator underflow (`spread_oracle.rs:112`) | Taylor fallback when `denom < 1e-15` |
| Kappa inversion — spread ≤ fee (`spread_oracle.rs:95-98`) | Returns `f64::INFINITY` |
| Event loop observation errors (`event_loop.rs:934-936`) | Caught and logged, loop continues |

---

## Common Backtrace Frames

When reading a backtrace, these frames tell you where you are:

```
# Tokio async runtime (bottom of stack — ignore these)
tokio::runtime::scheduler::multi_thread::worker::*
tokio::runtime::task::harness::*

# Entry point
market_maker::main                          → src/bin/market_maker.rs

# Event loop
MarketMaker::start                          → orchestrator/event_loop.rs
EventLoop::run                              → orchestrator/event_loop.rs:823
EventLoop::handle_observation               → orchestrator/event_loop.rs:934

# Quote generation
QuoteEngine::update_quotes                  → orchestrator/quote_engine.rs:30

# Strategy / spread math
spread_for_kappa                            → strategy/spread_oracle.rs:36
kappa_for_spread                            → strategy/spread_oracle.rs:87
kappa_for_spread_with_vol                   → strategy/spread_oracle.rs:130+

# Risk
KillSwitch::check / evaluate / trigger      → risk/kill_switch.rs
CircuitBreaker::assess                      → risk/circuit_breaker.rs
PositionGuard::check_order                  → risk/position_guard.rs

# Handlers
EventLoop::handle_fill                      → orchestrator/handlers.rs
EventLoop::handle_l2_update                 → orchestrator/handlers.rs
EventLoop::handle_trade                     → orchestrator/handlers.rs
```

---

## Debugging Checklist

When you get a backtrace:

1. **Read the panic message first** — it's above the backtrace and usually tells you exactly what failed
2. **Find the highest frame in `src/`** — skip `std::`, `tokio::`, `core::` frames
3. **Check if it's debug-only** — `debug_assert!` won't fire in `--release` builds
4. **Check for mutex poisoning** — if you see `PoisonError`, find the **original** panic
5. **Check the config** — most startup panics are config validation failures
6. **Run with `RUST_BACKTRACE=full`** — shows inlined frames that `=1` hides
7. **Check the logs before the panic** — `warn!` / `error!` messages often precede the crash

### Reproducing

```bash
# Debug build (assertions enabled, slower)
RUST_BACKTRACE=1 cargo run --bin market_maker -- paper

# Release build (assertions disabled, production behavior)
RUST_BACKTRACE=1 cargo run --release --bin market_maker -- paper
```

---

## Numerical Stability Summary

The GLFT formula has **5 layers of protection** against bad inputs:

```
Layer 1: Config validation at startup         → market_maker.rs:2056
Layer 2: debug_assert! in debug builds        → spread_oracle.rs:43-44
Layer 3: Epsilon clamping in all builds       → spread_oracle.rs:46-47
Layer 4: Taylor expansion fallbacks           → spread_oracle.rs:54-56, 114-115
Layer 5: Overflow/underflow guards            → spread_oracle.rs:104-108, 112-116
```

If the formula still produces bad values, the quote engine has additional guards before placing orders.
