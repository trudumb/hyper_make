Fix Quote Gate Oscillation (Smoothed Spread + Quote State Machine)
Problem
The market maker is oscillating rapidly between quoting and not quoting, which:

Burns rate limits - hitting "Too many cumulative requests" error
Causes quote/cancel churn - orders get filled during cancel attempts
Wastes edge - we're paying fees to place/cancel without holding quotes long enough
Root Cause
The quote_gate_input.spread_bps uses instantaneous market spread from 

current_spread_bps()
:

spread_bps: 1.0  → edge: -0.86 → NO QUOTE  
spread_bps: 13.8 → edge: +2.41 → QUOTE
spread_bps: 1.0  → edge: -0.86 → NO QUOTE (cancel what we just placed)
The spread oscillates 1→14 bps tick-to-tick because of transient liquidity changes, causing the theoretical edge to flip-flop.

Solution
Two-part fix:

Use EWMA-smoothed spread instead of instantaneous spread for edge calculation
Add quote state hysteresis - require edge to drop significantly below threshold to cancel
Proposed Changes
Part 1: Smoothed Spread for Quote Gate
[MODIFY] 

quote_engine.rs
Change line 1057:

// BEFORE: Uses instantaneous spread (noisy)
spread_bps: market_params.market_spread_bps.max(1.0),
// AFTER: Use EWMA-smoothed spread from SpreadProcessEstimator
spread_bps: market_params.smoothed_spread_bps.max(1.0),
[MODIFY] 

market_params.rs
Add new field around line 225:

pub market_spread_bps: f64,
/// EWMA-smoothed spread for stable edge calculation
pub smoothed_spread_bps: f64,
Update Default impl to include smoothed_spread_bps: 0.0.

[MODIFY] 

aggregator.rs
Add smoothed spread computation after line 230:

market_spread_bps: sources.spread_tracker.current_spread_bps(),
// EWMA-smoothed spread (λ=0.95) for stable quote decisions
smoothed_spread_bps: sources.spread_tracker.ewma_spread() * 10000.0,
Part 2: Quote State Hysteresis
When we've committed to quoting, don't immediately cancel on a small edge drop. This prevents the rapid place→cancel→place cycle.

[MODIFY] 

quote_gate.rs
Add hysteresis to the quote decision. When already quoting, require edge to drop below a lower threshold to stop quoting:

// In QuoteGateConfig, add:
/// Hysteresis buffer to prevent quote/cancel oscillation (bps)
pub hysteresis_bps: f64,  // Default: 0.3
// In decide_with_theoretical_fallback, after computing edge:
let effective_min_edge = if is_currently_quoting {
    self.config.min_edge_bps - self.config.hysteresis_bps
} else {
    self.config.min_edge_bps
};
This means:

To start quoting: edge ≥ 0.0 bps (normal threshold)
To stop quoting: edge < -0.3 bps (requires meaningful negative edge)
Verification Plan
Automated Tests
cargo test theoretical_edge::tests
cargo test quote_gate::tests
cargo build --release
Manual Verification
Run the market maker and observe logs
Verify 

spread_bps
 in theoretical edge logs is now stable (not jumping 1→14 bps)
Confirm no rapid quote→cancel→quote cycles
Confirm rate limit errors stop