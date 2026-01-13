# Session: 2026-01-13 Mainnet BTC Log Audit

## Summary
Audited mainnet BTC market maker session from 2026-01-13 (18-minute run). Identified critical microprice divergence issue and untracked fill synchronization bugs.

## Session Metrics
- **Duration**: ~18 minutes (22:14:01 → 22:32:00 UTC)
- **Asset**: BTC on Mainnet
- **Initial Account**: $235.14
- **Final Margin**: $209.10 (~$26 loss)
- **Final Position**: -0.0115 BTC (short)
- **Total Fills**: 27 (9 tracked, 18 untracked)
- **Warnings/Errors**: 1,698

## Critical Issues Identified

### 1. Microprice Divergence (CRITICAL - 10 errors)
Microprice estimator calculated ~$97,000-$97,400 while market mid was ~$95,400.
- **Deviation**: ~200 bps (2%)
- **Root Cause**: Warmup/calibration issue during startup
- **Safety**: System caught crossed-market and adjusted to 1bp below mid
- **Impact**: Large price drift warnings (103-213 bps)

### 2. Untracked Order Fills (18 occurrences)
Orders filled that weren't tracked by OrderManager:
```
[Fill] Untracked order filled: oid=293489830558 ... sold 0.00103 BTC
```
- **Root Cause**: Order state synchronization bug
- **Impact**: Incorrect position/P&L attribution

### 3. Account Value Loss (~$26)
- Tracked P&L on fills: ~+$1.95
- Unrealized loss on short position as BTC moved higher
- Not a bug, but market movement against position

## Persistent Warnings

### Rate Limit Pressure
- Headroom: 1.8% → 4.8%
- Used: 60,605 → 62,558 of 65,682 cap
- Result: Throttled reconciliation throughout

### Ladder Level Reduction
- Configured: 25 levels
- Effective: 17 bid / 13 ask
- Cause: Insufficient margin for full ladder

## What Worked Well
- Kill switch did NOT trigger
- Safety sync catching state mismatches
- Spread calculation reasonable (14.6 bps)
- Entropy optimizer functioning
- L1/L2/L3 pipeline operational

## Recommended Actions
1. **Investigate microprice warmup** - Why ~200 bps divergence?
2. **Fix untracked fills bug** - 18/27 fills untracked
3. **Reduce configured ladder levels** - Match actual capacity
4. **Review warmup bypass** - System quoted with only 9/10 volume ticks

## Files Analyzed
- `/mnt/c/Users/17808/Desktop/hyper_make/logs/mm_mainnet_BTC_2026-01-13_15-13-53.log`

## Related Memories
- `session_2026-01-13_l3_proper_fixes.md` - Recent L3 fixes
- `session_2026-01-13_ladder_concentration_fallback.md` - Ladder fixes
