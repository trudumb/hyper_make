# Session: 2026-01-04 HIP-3 Spread Profile Implementation (Complete)

## Summary
Implemented first-principles spread profile system to achieve 15-25 bps target spreads on HIP-3 DEX assets. Created comprehensive test scripts for different network/DEX configurations.

## Problem Statement
- Current HIP-3 HYPE spreads: ~47 bps
- Target: 15-25 bps using GLFT stochastic modeling

## Mathematical Foundation

### GLFT Optimal Half-Spread Formula
```
δ* = (1/γ) × ln(1 + γ/κ) + maker_fee
```

### Critical Insight: Kappa Dominance
When γ/κ << 1 (always true for reasonable params):
```
ln(1 + γ/κ) ≈ γ/κ  (Taylor expansion)
δ* ≈ 1/κ + maker_fee
```
**Spread converges to 1/κ regardless of gamma!**

### Target Calculations
| kappa | gamma | Spread (bps) |
|-------|-------|--------------|
| 500   | 0.30  | ~47          |
| 1000  | 0.15  | ~25          |
| 1500  | 0.15  | ~18          |
| 2000  | 0.10  | ~13          |

## Solution: SpreadProfile Enum

### Profile Configurations
| Profile    | kappa_prior | gamma_base | Target Spread |
|------------|-------------|------------|---------------|
| Default    | 500         | 0.30       | 40-50 bps     |
| Hip3       | 1500        | 0.15       | 15-25 bps     |
| Aggressive | 2000        | 0.10       | 10-20 bps     |

## Files Modified

| File | Changes |
|------|---------|
| `estimator/kappa_orchestrator.rs:102-118` | Added `KappaOrchestratorConfig::hip3()` |
| `strategy/risk_config.rs:238-278` | Added `RiskConfig::hip3()` |
| `config.rs` | Added `SpreadProfile` enum with `from_str()` |
| `lib.rs` | Exported `SpreadProfile` |
| `bin/market_maker.rs:125-131` | Added `--spread-profile` CLI argument |
| `bin/market_maker.rs:1199-1243` | Wire RiskConfig by profile |
| `bin/market_maker.rs:1255-1303` | Wire kappa_prior by profile |
| `scripts/test_hip3.sh` | Updated with `--spread-profile` support |
| `scripts/test_testnet.sh` | NEW: Testnet testing script |
| `scripts/test_mainnet.sh` | NEW: Mainnet validator perps script |

## Key Implementation Details

### RiskConfig::hip3()
```rust
gamma_base: 0.15,            // vs 0.3 default
gamma_min: 0.08,             // Allow tight quotes
gamma_max: 2.0,              // Lower ceiling
min_spread_floor: 0.0006,    // 6 bps floor (vs 8 bps)
enable_time_of_day_scaling: false,  // Disabled for HIP-3
enable_book_depth_scaling: false,   // Always thin, would cause widening
max_warmup_gamma_mult: 1.05, // Minimal warmup penalty (5% vs 10%)
```

### KappaOrchestratorConfig::hip3()
```rust
prior_kappa: 1500.0,         // Target ~18 bps (1/1500 + fees)
prior_strength: 15.0,        // Moderate confidence
use_book_kappa: false,       // Books too thin for reliable regression
use_robust_kappa: true,
```

## CLI Usage

```bash
# HIP-3 DEX with tight spreads (15-25 bps)
cargo run --bin market_maker -- \
  --network mainnet \
  --asset HYPE \
  --dex hyna \
  --spread-profile hip3

# Default behavior (unchanged)
cargo run --bin market_maker -- --asset BTC

# Aggressive (experimental)
cargo run --bin market_maker -- --spread-profile aggressive
```

## Test Scripts

```bash
# HIP-3 DEX testing
./scripts/test_hip3.sh HYPE hyna 600 hip3

# Testnet (safe, no real funds)
./scripts/test_testnet.sh BTC 300

# Mainnet validator perps
./scripts/test_mainnet.sh ETH 60
```

## Verification Results

Dry run with `--spread-profile hip3` confirmed:
- `spread_profile="hip3"`
- `gamma_base=0.15`
- `kappa_prior=1500.0`
- `target_spread_bps="15-25"`

## Safety Rails Preserved

**Kept active:**
- Volatility regime scaling (0.8-2.5x gamma)
- Inventory utilization scaling (quadratic near limits)
- Hawkes activity scaling (order flow detection)
- Warmup gamma scaling (reduced to 1.05x for HIP-3)

**Disabled for HIP-3:**
- Time-of-day scaling (different patterns than perps)
- Book depth scaling (always thin, would cause perpetual widening)

## Technical Decisions

1. **Kappa over Gamma**: GLFT math shows kappa dominates spread when γ/κ << 1
2. **Profile-based config**: Cleaner than per-parameter CLI args
3. **Disabled scaling factors**: Time-of-day and book depth irrelevant for HIP-3
4. **Conservative aggressive**: Even "aggressive" profile uses reasonable 2000 kappa

## Next Steps

1. Live test with `./scripts/test_hip3.sh HYPE hyna 3600 hip3`
2. Monitor actual spreads vs target 15-25 bps
3. Track fill rate improvement (tighter = more fills)
4. Monitor adverse selection (tighter = potentially worse selection)
5. Fine-tune kappa_prior based on observed results

## Session Metrics
- Duration: ~1 hour
- Files modified: 9
- New files: 2 (test scripts)
- Build status: Success (with unused code warnings)
- Tests: Dry run passed
