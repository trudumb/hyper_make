# Test Analysis Template

Use this template when analyzing HIP-3 DEX test sessions.

## Session Info Template
```markdown
# Session: {YYYY-MM-DD} {Short Description}

## Test Configuration
- Asset: {ASSET}
- DEX: {DEX}
- Duration: {X} seconds
- Log File: `logs/mm_{dex}_{asset}_{timestamp}.log`
- Started: {datetime}

## Summary
{1-2 sentence overview of test behavior}

## Metrics Observed

| Metric | Value | Expected | Status |
|--------|-------|----------|--------|
| Spread (bps) | | 5-15 | |
| Jump ratio | | <1.5 | |
| Inventory util | | <50% | |
| Adverse selection | | <3 bps | |
| Quote cycles | | continuous | |

## Issues Found

### Issue 1: {Title}
- **Severity:** Critical/Warning/Info
- **Location:** `{file}:{line}`
- **Log Evidence:** `{relevant log line}`
- **Description:** ...
- **Fix:** ...

## Questions for Review
1. {Question about implementation choice}
2. {Question about expected behavior}

## Recommended PRs
1. `fix/{issue-1}` - {description}
2. `fix/{issue-2}` - {description}

## Next Steps
- [ ] Fix identified issues
- [ ] Re-run test
- [ ] Update this checkpoint with results
```

## Log Analysis Checklist

When analyzing logs, check for:

### Startup Phase
- [ ] WebSocket connections established
- [ ] Metadata loaded (asset info, DEX limits)
- [ ] Initial position synced
- [ ] Warmup progress logged

### Quoting Phase
- [ ] Quote cycles running regularly
- [ ] Spread within expected range
- [ ] No excessive cancellations
- [ ] Ladder levels populated

### Risk Events
- [ ] Kill switch triggered?
- [ ] Cascade severity spikes?
- [ ] Data staleness warnings?
- [ ] Rate limit hits?

### Fill Processing
- [ ] Fills deduplicated correctly
- [ ] Position updates accurate
- [ ] P&L tracking consistent
- [ ] Adverse selection measured

## Key Log Patterns

```bash
# Check for errors
grep "ERROR" logs/mm_*.log

# Check for warnings
grep "WARN" logs/mm_*.log

# Check quote activity
grep "Quote cycle" logs/mm_*.log | head -10

# Check trade processing
grep "Trades processed" logs/mm_*.log | head -10

# Check warmup
grep "Warming up" logs/mm_*.log

# Check fills
grep "fill" logs/mm_*.log | head -10

# Check subscriptions
grep -i "subscription" logs/mm_*.log | head -10
```
