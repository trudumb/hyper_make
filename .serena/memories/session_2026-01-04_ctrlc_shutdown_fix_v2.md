# Session: Ctrl+C Shutdown Fix v2

**Date:** 2026-01-04
**Status:** Complete
**Issue:** Ctrl+C not triggering graceful shutdown - multiple ^C characters visible but process doesn't respond

## Problem Summary

When pressing Ctrl+C to stop the market maker, the process would not respond. Terminal showed `^C^C^C^C^C^C` characters, indicating the terminal was receiving the signal but the process wasn't handling it.

## Root Cause Analysis

**Two issues identified:**

### Issue 1: Signal Starvation in select! Loop

The original signal handling pattern:
```rust
tokio::select! {
    biased;
    result = tokio::signal::ctrl_c() => { break; }  // Never polled!
    message = receiver.recv() => {
        self.handle_message(msg).await;  // Runs 100-500ms
    }
}
```

Even with `biased`, the signal branch only gets priority when **multiple branches are ready simultaneously**. Since:
1. WebSocket messages arrive continuously (~1/second)
2. Message handling includes slow HTTP calls (order modifications)
3. By the time one message finishes, another is already waiting

The signal branch was **never actually polled**.

### Issue 2: Script Using timeout + cargo run

The test script `scripts/test_hip3.sh` used:
```bash
timeout "${DURATION}" cargo run --bin market_maker -- ...
```

Signal path: Terminal → timeout → cargo → market_maker

Both `timeout` and `cargo run` intercept signals, preventing them from reaching market_maker.

## Solutions Implemented

### Fix 1: Dedicated Signal Handler Task (mod.rs)

```rust
// Shared shutdown flag
let shutdown_flag = Arc::new(AtomicBool::new(false));
let shutdown_flag_clone = shutdown_flag.clone();

// Dedicated signal handler task (runs independently)
tokio::spawn(async move {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};
        let mut sigterm = signal(SignalKind::terminate())
            .expect("Failed to register SIGTERM handler");

        tokio::select! {
            _ = tokio::signal::ctrl_c() => {
                info!("Shutdown signal received (SIGINT/Ctrl+C)");
            }
            _ = sigterm.recv() => {
                info!("Shutdown signal received (SIGTERM)");
            }
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
        info!("Shutdown signal received (SIGINT/Ctrl+C)");
    }
    shutdown_flag_clone.store(true, Ordering::SeqCst);
});

loop {
    // Check flag at START of each iteration
    if shutdown_flag.load(Ordering::SeqCst) {
        info!("Shutdown flag detected, initiating graceful shutdown...");
        break;
    }
    // ... rest of loop
}
```

### Fix 2: Updated test_hip3.sh Script

Changed from:
```bash
timeout "${DURATION}" cargo run --bin market_maker -- ...
```

To:
```bash
# Use timeout with --foreground to forward signals
# Run binary directly (not via cargo run)
timeout --foreground "${DURATION}" ./target/debug/market_maker ...
```

## Files Modified

1. **src/market_maker/mod.rs**
   - Added `use std::sync::atomic::{AtomicBool, Ordering};`
   - Replaced signal handling with dedicated task pattern
   - Shutdown flag checked at start of each loop iteration

2. **scripts/test_hip3.sh**
   - Added `--foreground` to timeout command
   - Changed to run binary directly instead of via `cargo run`
   - Added user instruction about Ctrl+C for graceful shutdown

## Verification

Direct signal test passed:
```bash
./target/debug/market_maker --network mainnet --asset HYPE --dex hyna &
MM_PID=$!
sleep 4
kill -INT $MM_PID  # Process exits gracefully
```

## Key Learnings

1. `tokio::select!` with `biased` only prioritizes when multiple futures are ready simultaneously
2. Signal handlers in select! can be starved by continuously ready message channels
3. Dedicated signal handler tasks with atomic flags are more robust
4. `timeout` and `cargo run` both intercept signals - use `--foreground` and run binary directly
5. When debugging signal issues, test with direct `kill -INT` to isolate process vs script issues
