# Session: Ctrl+C Signal Handling Fix

**Date:** 2026-01-03
**Status:** Complete
**Issue:** Ctrl+C shutdown not working - process terminates abruptly without graceful shutdown

## Problem Summary

When pressing Ctrl+C to stop the market maker, the process would terminate immediately without:
- Logging "Shutdown signal received (SIGINT)"
- Running graceful shutdown sequence
- Cancelling resting orders
- Flushing logs

The log file just ended abruptly with no shutdown-related messages.

## Root Cause

The signal handling used `std::pin::pin!` with a pre-created `ctrl_c()` future:

```rust
// BROKEN PATTERN:
let mut shutdown_signal = std::pin::pin!(tokio::signal::ctrl_c());
loop {
    tokio::select! {
        _ = &mut shutdown_signal => {
            info!("Shutdown signal received (SIGINT)");
            break;
        }
        ...
    }
}
```

The issue is that using `&mut shutdown_signal` (a `&mut Pin<&mut CtrlC>`) in the select! 
macro doesn't reliably poll the signal handler. The signal registration may not be 
happening correctly when the future is pre-pinned outside the loop.

## Solution Implemented

Changed to call `tokio::signal::ctrl_c()` directly in each select! iteration:

```rust
// WORKING PATTERN:
loop {
    tokio::select! {
        biased;

        // SIGINT (Ctrl+C) - highest priority
        // Note: ctrl_c() called fresh each iteration - each call registers a new listener
        // and all pending listeners are notified when signal arrives
        result = tokio::signal::ctrl_c() => {
            match result {
                Ok(()) => info!("Shutdown signal received (SIGINT)"),
                Err(e) => error!("Failed to listen for SIGINT: {e}"),
            }
            break;
        }
        ...
    }
}
```

Per tokio documentation: "A new listener is registered each time the function is called,
meaning that ctrl_c().await can be called multiple times. Upon receiving a signal, all
listeners are notified."

This is the robust pattern for signal handling in select! loops.

## Files Modified

1. `src/market_maker/mod.rs` - Lines ~514-538
   - Removed pre-pinned `shutdown_signal`
   - Changed select! branch to call `ctrl_c()` directly
   - Added proper error handling for signal listener failures

## Testing

- Build: ✓
- Clippy: ✓

## Related Sessions

- `session_2026-01-03_hip3_open_orders_fix` - Fixed HIP-3 order state management in same session
