# Session: 2026-02-23 Fixed Stuck Short Phenomenon

## Summary
Fixed a critical bug where the market maker would get permanently stuck in a short position during a bearish trend. The fix involves relaxing the E[PnL] filter threshold for reducing remnant positions and removing a hardcoded "trend-momentum guard" that suppressed Bayesian inventory skewing.

## The Bug: "Stuck Short" Phenomenon
When the bot was holding a short position and the market began trending downwards (aligned with the position), it was supposed to gradually buy back (reduce the short) to lock in profit or normalize inventory. Instead, the bot failed to post covering bids and got "stuck short".

**Why it failed:**
1. **Trend-Momentum Guard Bypass:** `position_manager.rs` contained a hardcoded `TREND-MOMENTUM GUARD`. If the raw price momentum aligned with the position (e.g., trend is falling and position is short), the guard forced `PositionAction::Hold`. This set the `urgency` to 0.0, completely disabling the normal inventory skewing meant to reduce the position.
2. **E[PnL] Filter Starvation:** In `ladder_strat.rs`, the E[PnL] filter dynamically drops quotes if their expected value is < 0. For inventory reduction, there was a carved-out threshold `reducing_threshold_bps`. However, the formula was quadratic `-fee * (q_ratio)^1.5`. With a small remnant position (e.g., `q_ratio = 0.13`), the threshold was extremely tight (`-0.069` bps). The bearish drift penalty on bids was around `-2.02` bps. As a result, the algorithm viewed covering bids as "negative EV" and dropped them, preventing the bot from physically escaping the short position.

## What It *Should* Have Done (and what it will do now)
1. **Respect Bayesian Models:** Instead of a hardcoded guard forcing a `Hold` (which assumes we are directional traders riding a trend), the `PositionDecisionEngine` should calculate the Bayesian continuation probability and smoothly transition to `Reduce`. The bot is a market maker, not a momentum hedge fund—it should always be looking to exit directional risk.
2. **Provide Meaningful Spread Carve-Outs for Exit:** Rather than requiring virtually 0.0 EV to exit a small remnant position during an opposing trend, the bot must acknowledge square-root VaR scaling. The E[PnL] threshold for a reducing position is now `-2.0 * fee * sqrt(q_ratio)`. A 13% remaining position gets a generous `~ -1.6` bps carve-out.
3. **Continuous Bidding:** By combining the proper inventory skew (which adjusts prices upwards to get filled) and the relaxed E[PnL] filter, the bot will now post continuous covering bids just below the mid price. Even in a fast bearish market, it will successfully buy back the position and return to an inventory-neutral state.

## Files Modified
- `src/market_maker/strategy/position_manager.rs`: Removed the `TREND-MOMENTUM GUARD` and updated unit tests to ensure `Reduce` triggers reliably.
- `src/market_maker/strategy/glft.rs`: Updated `reducing_threshold_bps` formula from `q_ratio^1.5` to `2.0 * sqrt(q_ratio)` and updated corresponding integration tests.

## Verification
✅ Tested E[PnL] threshold calculations (`cargo test`)
✅ Tested Bayesian position action resolution correctly outputting `Reduce` without being overridden.
