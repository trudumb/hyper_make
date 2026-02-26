---
name: checkpoint-management
description: State persistence, prior transfer, and warmup lifecycle. Read when working on checkpoint/, adding new checkpoint fields, debugging cold starts or stale priors, or understanding serde(default) requirements and backward compatibility rules.
user-invocable: false
---

# Checkpoint Management Skill

## Purpose

Understand and modify the checkpoint persistence system that saves learned model state across sessions. Without checkpoints, every restart is a cold start -- the MM must re-learn kappa, sigma, regime beliefs, AS classifier weights, and RL Q-values from scratch, losing hours of Bayesian posterior convergence.

## When to Use

- Working on `checkpoint/` module files
- Adding new persistent fields to any component (estimator, learning, risk)
- Debugging cold-start behavior or stale priors after restart
- Setting up paper-to-live prior transfer
- Investigating why a component has wrong parameters after restart
- Understanding `#[serde(default)]` requirements and backward compatibility

---

## 1. Module Map

All paths relative to `src/market_maker/checkpoint/`:

```
checkpoint/
  mod.rs              # CheckpointManager: save_all(), load_latest(), cleanup_old()
                      #   Atomic writes (tmp + rename), timestamped backups, 7-day retention
  types.rs            # CheckpointBundle + all component checkpoint structs
                      #   PriorReadiness, PriorVerdict, PreFillCheckpoint, EnhancedCheckpoint,
                      #   VolFilterCheckpoint, RegimeHMMCheckpoint, InformedFlowCheckpoint,
                      #   FillRateCheckpoint, KappaCheckpoint, MomentumCheckpoint,
                      #   KellyTrackerCheckpoint, EnsembleWeightsCheckpoint, RLCheckpoint,
                      #   QTableEntry, KillSwitchCheckpoint
  transfer.rs         # PriorExtract and PriorInject traits, InjectionConfig
                      #   Formal protocol: S_paper -[phi: extract_prior()]-> P -[psi: inject_prior()]-> S_live
  prediction_reader.rs # JSONL prediction log reader for batch retraining
                      #   read_resolved() and read_all() for PredictionRecord files
```

### Key Entry Points

- **Saving**: `MarketMaker::assemble_checkpoint_bundle()` in `mod.rs` (line ~674) assembles all state. Called by:
  - Periodic timer in `event_loop.rs` (every 300 seconds)
  - `shutdown()` in `recovery.rs` (graceful shutdown, final save)
- **Loading**: `CheckpointManager::load_latest()` in `checkpoint/mod.rs` returns `Option<CheckpointBundle>`
- **Restoring**: `MarketMaker::restore_from_bundle()` in `mod.rs` (line ~722) dispatches to each component's `restore_checkpoint()`
- **Transferring**: `PriorExtract` and `PriorInject` trait implementations on `MarketMaker` in `mod.rs` (lines ~750-823)
- **Gating**: `CalibrationGate::assess()` in `calibration/gate.rs` stamps readiness on each periodic save

### Directory Layout on Disk

```
data/checkpoints/
  latest/
    checkpoint.json          # Current checkpoint (atomically written)
  1700000000000/             # Timestamped backup (ms since epoch)
    checkpoint.json
  1700000300000/             # Another backup (5 min later)
    checkpoint.json
```

---

## 2. CheckpointBundle Structure

`CheckpointBundle` is the root serialization unit. It holds the minimum learned state needed to warm-start every component. Ephemeral state (cached values, timestamps, rolling VecDeque windows) is NOT persisted -- it rebuilds from live data within seconds.

### What Each Field Stores

| Category | Fields | What's Persisted |
|----------|--------|-----------------|
| Metadata | `metadata` | Schema version, save timestamp, asset, session duration |
| Bayesian params | `learned_params` | ~25 Bayesian posteriors (alpha_touch, gamma_base, kappa, etc.) |
| Volatility | `vol_filter` | Particle filter summary: sigma_mean, sigma_std, regime_probs |
| Regime | `regime_hmm` | HMM belief state + transition count matrix |
| Flow model | `informed_flow` | 3-component mixture: informed/noise/forced |
| Fill rate | `fill_rate` | Bayesian regression posteriors for lambda_0 and delta_char |
| Kappa | `kappa_own`, `kappa_bid`, `kappa_ask` | Per-side Bayesian kappa: alpha, beta, sum_distances |
| Momentum | `momentum` | Per-magnitude continuation probabilities (10 buckets) |
| AS classifier | `pre_fill`, `enhanced` | Signal weights, gradient momentum, EWMA normalizer state |
| Kelly sizing | `kelly_tracker` | EWMA win/loss sizes, counts, decay factor |
| Ensemble | `ensemble_weights` | Softmax model weights [GLFT, Empirical, Funding] |
| RL agent | `rl_q_table` | Bayesian Q-table entries (only non-default states), episodes, rewards |
| Kill switch | `kill_switch` | Triggered flag, reasons, daily/peak P&L, trigger timestamp |
| Readiness | `readiness` | CalibrationGate verdict + per-estimator observation counts |

See `references/checkpoint-schema.md` for the complete field-by-field listing with types and defaults.

### Serialization Format

- **Format**: JSON via serde (`serde_json::to_string_pretty`)
- **Size**: ~100KB per checkpoint (RL Q-table is the largest component)
- **Atomic writes**: Write to `.tmp`, then `fs::rename` to prevent corruption on crash
- **Backups**: Timestamped directory per save; `cleanup_old(7)` removes backups older than 7 days
- **Version field**: `metadata.version` = 1 (for future migration support)

### Save Triggers

1. **Periodic timer** (every 5 minutes): In `event_loop.rs`, the `last_checkpoint_save` instant is checked each monitoring tick. The save includes a `CalibrationGate::assess()` call to stamp readiness.
2. **Graceful shutdown**: `recovery.rs::shutdown()` saves a final checkpoint before cancelling orders and flushing analytics.

There is currently NO on-fill checkpoint save -- fills update in-memory state only, and the next periodic save captures the changes.

---

## 3. PriorReadiness Assessment

The `CalibrationGate` in `calibration/gate.rs` evaluates a `CheckpointBundle` and returns a `PriorReadiness` snapshot with a `PriorVerdict`: `Ready`, `Marginal`, or `Insufficient`.

### Verdict Criteria

| Verdict | Requirements |
|---------|-------------|
| **Ready** | Session duration >= 1800s (30 min) AND all 5 core estimators meet thresholds AND kelly_fills >= 20 |
| **Marginal** | Session duration >= 1800s AND at least 3/5 estimators meet thresholds |
| **Insufficient** | Anything else (default) |

### Per-Estimator Thresholds (from `CalibrationGateConfig::default()`)

| Estimator | Min Observations | What It Measures |
|-----------|-----------------|-----------------|
| Vol filter (`vol_observations`) | 200 | Price observations for sigma posterior convergence |
| Kappa (`kappa_observations`) | 50 | Own fills for fill intensity estimation |
| AS classifier (`as_learning_samples`) | 100 | Labeled fill examples for adverse selection classification |
| Regime HMM (`regime_observations`) | 200 | L2 book observations for regime belief stabilization |
| Fill rate (`fill_rate_observations`) | 200 | Fill events for Bayesian regression on lambda_0/delta_char |

Additionally:
- **Kelly fills**: `n_wins + n_losses >= 20` (required for Ready, not for Marginal)
- **Session duration**: >= 1800 seconds (required for both Ready and Marginal)

### How Readiness Affects Quoting Behavior

When a checkpoint is loaded, the readiness verdict (stamped at save time) determines warmup behavior:

- **Ready**: Full confidence quoting. No warmup discounts.
- **Marginal**: `allow_marginal = true` by default, so trading proceeds with defensive warmup. `warmup_spread_discount()` tightens gamma by 5-15% to attract fills; `warmup_size_multiplier()` limits size to 0.3-0.7x to cap learning cost.
- **Insufficient**: `CalibrationGate::passes()` returns false. The MM should cold-start from priors. Warmup discounts apply at maximum (0.85x gamma, 0.3x size) until fill count reaches thresholds.

The warmup functions in `gate.rs` are fill-count-based:
- 0-49 fills: 15% tighter spreads, 30% size
- 50-199 fills: 5% tighter spreads, 70% size
- 200+ fills: fully calibrated

---

## 4. Prior Transfer (Paper to Live)

### Protocol

The formal transfer protocol uses two traits:

```
S_paper --[phi: extract_prior()]--> CheckpointBundle --[psi: inject_prior()]--> S_live
```

- **phi** (`PriorExtract::extract_prior`): Calls `assemble_checkpoint_bundle()` on the paper trader
- **psi** (`PriorInject::inject_prior`): Validates age + asset match, restores components, special-handles RL

### InjectionConfig Defaults

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `rl_blend_weight` | 0.3 | Discount factor for paper RL Q-values |
| `max_prior_age_s` | 14400 (4h) | Reject priors older than this |
| `require_asset_match` | true | Prior must be for the same asset |
| `skip_rl` | false | Optionally skip RL injection entirely |
| `skip_kill_switch` | true | Never inherit kill switch state from paper |

### Q-Table Import Fix: Only Seed Cold States

The RL Q-table import (`rl_agent.rs::import_q_table_as_prior`) follows a critical rule:

**Only overwrite Q-values for cold-start states where `n == 0`.** States with live observations (`n > 0`) are preserved untouched.

Why: Paper trading has fundamentally different fill dynamics (simulated fills, no queue position, different latency). Live states that have accumulated real experience are strictly more valuable than paper priors. Overwriting them would destroy learned live behavior.

The import applies a discount via `BayesianQValue::with_discounted_prior(paper_qv, weight)` where `weight` is typically 0.3, reflecting lower confidence in paper-learned values.

### Transfer Procedure

1. Run paper trader long enough to reach `Ready` or `Marginal` verdict
2. Paper trader's checkpoint is saved to `data/checkpoints/latest/checkpoint.json`
3. On live startup, call `inject_prior(&bundle, &InjectionConfig::default())`
4. Injection validates:
   - Asset match (paper ETH checkpoint won't inject into HYPE live)
   - Age check (>4h = rejected as stale)
5. All non-RL components are restored via `restore_from_bundle()`
6. RL Q-table is imported via `load_paper_rl_prior()` with blend weight
7. Kill switch state is zeroed out (never inherit paper kill state)

---

## 5. Adding New Checkpoint Fields -- Step-by-Step

This is the most common checkpoint task. Follow every step exactly.

### Step 1: Add the field to the appropriate struct

Choose the right struct in `types.rs`:
- New estimator state? Add a new checkpoint struct + field on `CheckpointBundle`
- Extending an existing component? Add field to the existing checkpoint struct (e.g. `PreFillCheckpoint`)

### Step 2: Add `#[serde(default)]` -- NON-NEGOTIABLE

```rust
// On CheckpointBundle:
#[serde(default)]
pub my_new_field: MyNewCheckpoint,

// Or on an inner struct field:
#[serde(default)]
pub new_counter: usize,

// Or with a custom default function:
#[serde(default = "default_my_value")]
pub my_value: f64,
```

**Why this is non-negotiable**: Old checkpoint files on disk will not have the new field. Without `#[serde(default)]`, deserialization of any existing checkpoint will fail with a serde error, effectively breaking all restarts until the checkpoint file is manually deleted. This has happened and it causes real downtime.

### Step 3: Implement `Default` for the type

If you added a new struct, it must implement `Default`:

```rust
impl Default for MyNewCheckpoint {
    fn default() -> Self {
        Self {
            // Sensible priors, not zeros
            estimate: 0.5,
            observations: 0,
        }
    }
}
```

The default values must be sensible priors -- they are what the system uses when no prior data exists. Zeros are usually wrong for Bayesian parameters.

### Step 4: Wire saving

In `MarketMaker::assemble_checkpoint_bundle()` (in `mod.rs`, line ~674), add the new field to the bundle construction. If the component has a `to_checkpoint()` method, call it:

```rust
my_new_field: self.some_component.to_checkpoint(),
```

### Step 5: Wire loading

In `MarketMaker::restore_from_bundle()` (in `mod.rs`, line ~722), add the restore call:

```rust
self.some_component.restore_checkpoint(&bundle.my_new_field);
```

The component needs both `to_checkpoint() -> MyNewCheckpoint` and `restore_checkpoint(&MyNewCheckpoint)` methods.

### Step 6: Test backward compatibility

Write a test that:
1. Creates a valid checkpoint JSON
2. Removes the new field from the JSON (simulating an old checkpoint)
3. Deserializes and verifies the default is used

Pattern from `types.rs::test_checkpoint_bundle_backward_compat_no_kill_switch`:

```rust
#[test]
fn test_checkpoint_bundle_backward_compat_no_my_field() {
    let bundle = make_test_bundle(); // helper that creates a full bundle
    let json = serde_json::to_string(&bundle).expect("serialize");
    let mut map: serde_json::Value = serde_json::from_str(&json).expect("parse");
    map.as_object_mut().unwrap().remove("my_new_field");
    let old_json = serde_json::to_string(&map).expect("re-serialize");
    let restored: CheckpointBundle = serde_json::from_str(&old_json)
        .expect("deserialize old format");
    // Verify default is sensible
    assert_eq!(restored.my_new_field.observations, 0);
}
```

### Step 7: Document in checkpoint-schema.md

Add the field to `references/checkpoint-schema.md` in the appropriate category table.

### Checklist Summary

- [ ] Field added to struct in `types.rs`
- [ ] `#[serde(default)]` attribute present
- [ ] `Default` implemented with sensible priors
- [ ] Saving wired in `assemble_checkpoint_bundle()`
- [ ] Loading wired in `restore_from_bundle()`
- [ ] Backward compatibility test written
- [ ] Schema reference updated

---

## 6. Known Checkpoint Issues

### Kill Switch 24-Hour Expiry

`KillSwitchCheckpoint` is restored in `kill_switch.rs::restore_from_checkpoint()`. A triggered kill switch is only re-triggered if the trigger timestamp is within 24 hours. Beyond that, the trigger is silently dropped to avoid blocking trading after extended maintenance. Additionally, "Position runaway" reasons are filtered as transient -- only persistent reasons (daily loss, drawdown) survive restart.

**Implication**: If the MM was killed for a real risk reason but restarts >24h later, the kill switch will NOT re-trigger. The live monitoring checks will re-evaluate from scratch.

### QuoteOutcomeTracker Not Persisted

`QuoteOutcomeTracker` (in `learning/quote_outcome.rs`) tracks filled and unfilled quote outcomes for unbiased edge estimation. It uses in-memory `VecDeque` and `BinnedFillRate` structures. None of this state is checkpointed. On restart, all fill/unfill history is lost, and the tracker starts from empty bins.

**Impact**: After restart, the spread optimization that depends on empirical `P(fill | spread_bin)` has no data until enough quotes accumulate (~50+ per bin for statistical significance). During this period, fill rate estimates fall back to the Bayesian model.

### BaselineTracker Not Persisted

`BaselineTracker` (in `learning/baseline_tracker.rs`) maintains an EWMA baseline of RL rewards for counterfactual reward computation. It derives `Serialize`/`Deserialize` and has a `restore()` method, but is not wired into `CheckpointBundle`. On restart, the EWMA baseline resets to 0 and the RL agent's counterfactual rewards are biased by the fee drag until the tracker warms up again (`min_observations` samples, typically 50).

**Fix path**: Add a `BaselineTrackerCheckpoint` struct (or reuse the `Serialize`-derived fields) to `CheckpointBundle` with `#[serde(default)]`, wire into `assemble_checkpoint_bundle()` and `restore_from_bundle()`.

### Sigma Cap on Restore

When restoring from a checkpoint, very large sigma values from old or corrupted checkpoints could cause pathological spread widening. The `ParameterEstimator` applies a sigma cap (`10x * default_sigma`) on all five sigma accessors (`sigma()`, `sigma_clean()`, `sigma_total()`, `sigma_effective()`, `sigma_for_bps()`). This means even if a checkpoint contains `sigma_mean = 1.0` (absurd), the capped value used in production is bounded.

However, the cap is applied at the accessor level, not at restore time. The raw restored value remains in the particle filter state. This is by design -- the filter will converge back to reality as new observations arrive.

---

## 7. Debugging Guide

### "Wrong parameters on restart"

1. **Check if checkpoint loaded**: Look for `"Loaded checkpoint from ..."` in logs. If absent, no checkpoint was found.
2. **Check restore log**: After injection, look for `"Prior injected successfully"` with field counts.
3. **Verify defaults make sense**: If a field was added without `#[serde(default)]` or with a bad default (e.g. kappa=0 which blows up GLFT), the restored value may be pathological.
4. **Check LearnedParameters**: These are cloned directly from checkpoint (`bundle.learned_params.clone()`). If they contain stale calibration from a different market regime, all Bayesian priors will be off until live data corrects them.

### "Deserialization failed"

Almost always caused by a missing `#[serde(default)]` on a new field.

1. Check the serde error message -- it will name the missing field
2. Add `#[serde(default)]` to the field
3. Implement `Default` if needed
4. The old checkpoint on disk should now load correctly

If the checkpoint is deeply corrupted (manual edit, disk error), delete `data/checkpoints/latest/checkpoint.json` and let the MM cold-start.

### "Checkpoint not saving"

1. **Check `checkpoint_manager` is Some**: `init_checkpoint()` must have been called. The checkpoint path is based on the asset name.
2. **Check disk space**: `fs::write` will fail silently if disk is full. Check for `"Checkpoint save failed"` warnings.
3. **Check file permissions**: The `data/checkpoints/` directory must be writable.
4. **Check timer**: The periodic save runs every 300 seconds. If the session is shorter than that and no shutdown save occurs, no checkpoint is written.

### "Cold start despite checkpoint"

1. **Checkpoint too old**: `InjectionConfig::max_prior_age_s` defaults to 4 hours. If the checkpoint was saved >4h ago and you're using `inject_prior()`, it will be rejected. Direct `restore_from_bundle()` has no age check.
2. **Asset mismatch**: If `require_asset_match = true` and the checkpoint was for a different asset, injection is skipped.
3. **Readiness = Insufficient**: Check the `readiness` field in the checkpoint JSON. If `verdict = "Insufficient"`, the CalibrationGate will not pass, and the system enters full warmup mode.
4. **Checkpoint file missing**: Check if `data/checkpoints/latest/checkpoint.json` exists. Cleanup with aggressive retention could have removed it.

### "Kappa/sigma way off after restart"

1. **Stale checkpoint priors**: If the checkpoint is from a very different market regime (e.g. saved during a cascade, restored in calm), the posteriors will be biased. The Bayesian estimators will correct over time, but initial quotes may be too wide or too narrow.
2. **Sigma cap**: Check if `sigma_mean` in the checkpoint is unreasonably large. The 10x cap prevents the worst case, but a sigma_mean of 5x default still causes wide spreads until the filter converges.
3. **Kappa prior_alpha/prior_beta**: If these have accumulated many observations from a different regime, the posterior will be slow to adapt. Consider resetting to defaults for a regime change.

### "RL Q-table seems to have wrong values"

1. **Action space version mismatch**: `RLCheckpoint.action_space_version` must match the current agent. Version 0 = legacy, 1 = BPS-delta, 2 = parameter action. If the checkpoint has a different version, Q-values map to wrong actions.
2. **Reward config hash**: `reward_config_hash` detects incompatible reward function changes. If it's 0 (legacy), validation is skipped.
3. **Drift bucket mismatch**: `use_drift_bucket = true` means 2025-state space, `false` means 675-state. Mixing them maps states incorrectly.
4. **Paper prior overwrite**: After `import_q_table_as_prior()`, only cold states (n=0) get paper values. If all states are cold (fresh agent), the entire Q-table comes from paper. If paper was mis-calibrated, so is the live agent.

---

## Cross-References

- **CalibrationGate**: `src/market_maker/calibration/gate.rs` -- readiness assessment logic, warmup functions
- **MarketMaker save/restore**: `src/market_maker/mod.rs` -- `assemble_checkpoint_bundle()`, `restore_from_bundle()`, `PriorExtract`/`PriorInject` implementations
- **Event loop saves**: `src/market_maker/orchestrator/event_loop.rs` -- periodic 300s save with readiness stamping
- **Shutdown save**: `src/market_maker/orchestrator/recovery.rs` -- final checkpoint on graceful shutdown
- **RL Q-table**: `src/market_maker/learning/rl_agent.rs` -- `to_checkpoint()`, `restore_from_checkpoint()`, `import_q_table_as_prior()`
- **Kill switch restore**: `src/market_maker/risk/kill_switch.rs` -- 24h expiry, transient reason filtering
- **Sigma cap**: `src/market_maker/estimator/parameter_estimator.rs` -- `cap_sigma()` at 10x default
- **Field reference**: `references/checkpoint-schema.md` -- complete field listing with types and defaults
