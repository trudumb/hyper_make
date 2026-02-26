# CheckpointBundle Field Reference

Complete field listing for `CheckpointBundle` (`src/market_maker/checkpoint/types.rs`), grouped by category. All fields marked `#[serde(default)]` on the bundle itself are noted; inner struct fields with `#[serde(default)]` are listed where applicable.

---

## Metadata

| Field | Type | `#[serde(default)]` | Default | Description |
|-------|------|---------------------|---------|-------------|
| `metadata.version` | `u32` | No | — | Schema version for forward compatibility (currently `1`) |
| `metadata.timestamp_ms` | `u64` | No | — | Save time in ms since epoch |
| `metadata.asset` | `String` | No | — | Asset symbol (e.g. `"ETH"`, `"HYPE"`) |
| `metadata.session_duration_s` | `f64` | No | — | Session uptime at save time |

## Estimator State

| Field | Type | `#[serde(default)]` | Default | Description |
|-------|------|---------------------|---------|-------------|
| `learned_params` | `LearnedParameters` | No | ~25 Bayesian posteriors | Calibrated priors for all tiers (P&L, risk, calibration, microstructure) |
| `vol_filter` | `VolFilterCheckpoint` | No | sigma_mean=0.0005, sigma_std=0.0002, obs=0 | Particle filter posterior summary for sigma |
| `regime_hmm` | `RegimeHMMCheckpoint` | No | belief=[0.1,0.7,0.15,0.05], obs=0 | HMM belief + transition counts |
| `informed_flow` | `InformedFlowCheckpoint` | No | 3-component mixture, obs=0 | Informed/noise/forced flow mixture model |
| `fill_rate` | `FillRateCheckpoint` | No | lambda_0=0.1, delta_char=10.0 | Bayesian regression posteriors for fill rate |
| `kappa_own` | `KappaCheckpoint` | No | alpha=10, beta=0.02, kappa=500 | Bayesian kappa from own fills |
| `kappa_bid` | `KappaCheckpoint` | No | (same as kappa_own) | Kappa estimated from bid-side fills |
| `kappa_ask` | `KappaCheckpoint` | No | (same as kappa_own) | Kappa estimated from ask-side fills |
| `momentum` | `MomentumCheckpoint` | No | continuation=[0.5; 10], prior=0.5 | Per-magnitude continuation probabilities |

## Adverse Selection State

| Field | Type | `#[serde(default)]` | Default | Description |
|-------|------|---------------------|---------|-------------|
| `pre_fill` | `PreFillCheckpoint` | No | weights=[0.30,0.25,0.25,0.10,0.10], 5 EWMA normalizers | Pre-fill AS classifier weights + z-score normalizer EWMA state |
| `enhanced` | `EnhancedCheckpoint` | No | weights=[0.1; 10], samples=0 | Enhanced 10-feature AS classifier with gradient momentum |

### PreFillCheckpoint inner `#[serde(default)]` fields

All EWMA normalizer fields were added after v1 and carry `#[serde(default)]`:
`imbalance_ewma_mean`, `imbalance_ewma_var` (default 1.0), `flow_ewma_mean`, `flow_ewma_var` (default 1.0), `funding_ewma_mean`, `funding_ewma_var` (default 1.0), `regime_trust_prev` (default 1.0), `regime_ewma_mean`, `regime_ewma_var` (default 1.0), `changepoint_ewma_mean`, `changepoint_ewma_var` (default 1.0), `normalizer_obs_count`, `bias_correction_bps`, `bias_observation_count`.

## Learning State

| Field | Type | `#[serde(default)]` on bundle | Default | Description |
|-------|------|-------------------------------|---------|-------------|
| `kelly_tracker` | `KellyTrackerCheckpoint` | **Yes** | ewma_wins=5.0, ewma_losses=3.0, decay=0.99 | Kelly position sizing EWMA win/loss tracker |
| `ensemble_weights` | `EnsembleWeightsCheckpoint` | **Yes** | weights=[0.5,0.3,0.2], updates=0 | Softmax model ensemble weights [GLFT, Empirical, Funding] |
| `rl_q_table` | `RLCheckpoint` | **Yes** | empty q_entries, episodes=0 | Bayesian RL Q-table (only non-default entries stored) |

### RLCheckpoint inner `#[serde(default)]` fields

`action_space_version` (default 0), `use_compact_state` (default false), `reward_config_hash` (default 0), `use_drift_bucket` (default false).

## Risk State

| Field | Type | `#[serde(default)]` on bundle | Default | Description |
|-------|------|-------------------------------|---------|-------------|
| `kill_switch` | `KillSwitchCheckpoint` | **Yes** | triggered=false, daily_pnl=0.0, peak_pnl=0.0 | Kill switch triggered state + P&L for drawdown |

## Readiness Assessment

| Field | Type | `#[serde(default)]` on bundle | Default | Description |
|-------|------|-------------------------------|---------|-------------|
| `readiness` | `PriorReadiness` | **Yes** | verdict=Insufficient, all counts=0 | CalibrationGate assessment stamped at save time |
