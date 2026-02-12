//! Contextual Bandit SpreadOptimizer.
//!
//! Spread selection is a **contextual bandit** (i.i.d. rewards per quote cycle),
//! NOT an MDP. Each quote cycle independently selects a spread multiplier arm
//! conditioned on the current market context, receives a reward (realized edge),
//! and updates the posterior for that (context, arm) cell.
//!
//! ## Architecture
//!
//! ```text
//! Layer 1: THEORY (GLFT)
//!   solve_min_gamma(kappa, sigma, T, floor, fee) → base_half_spread_bps
//!
//! Layer 2: LEARNING (SpreadBandit)  ← this module
//!   81 contexts × 8 arms → spread_multiplier
//!   Thompson Sampling, Normal-Gamma posterior, exponential forgetting
//!
//! Layer 3: SAFETY (RiskOverlay — unchanged)
//!   risk_level × staleness × toxicity × model_gating multipliers
//!
//! Final spread = base_half_spread × bandit_mult × safety_mults
//! ```
//!
//! ## Why bandit, not MDP?
//!
//! - Rewards are i.i.d. given context (no state transitions to model)
//! - Normal-Gamma conjugate prior is statistically correct for i.i.d. observations
//! - 648 cells vs 1,125 → faster convergence (~2 hours vs ~50 days)
//! - Single influence channel (spread_multiplier) → clean reward attribution

use serde::{Deserialize, Serialize};

// Re-use the existing Normal-Gamma posterior from rl_agent
use super::rl_agent::BayesianQValue;

/// Number of regime buckets (Low, Normal, High).
const N_REGIME: usize = 3;
/// Number of position buckets (Short, Neutral, Long).
const N_POSITION: usize = 3;
/// Number of volatility buckets (Low, Normal, High).
const N_VOL: usize = 3;
/// Number of flow buckets (Sell, Neutral, Buy).
const N_FLOW: usize = 3;

/// Total contexts = 3×3×3×3 = 81.
const N_CONTEXTS: usize = N_REGIME * N_POSITION * N_VOL * N_FLOW;

/// Number of spread multiplier arms.
const N_ARMS: usize = 8;

/// Spread multiplier arms: 0.85, 0.90, 0.95, 1.00, 1.05, 1.15, 1.25, 1.40.
/// Arms are asymmetric — more aggressive tightening, cautious widening.
const ARM_MULTIPLIERS: [f64; N_ARMS] = [0.85, 0.90, 0.95, 1.00, 1.05, 1.15, 1.25, 1.40];

/// Default arm index (1.00 = pure GLFT, no modification).
const DEFAULT_ARM: usize = 3;

/// Minimum observations before exploiting (below this, default to arm 3).
const MIN_OBS_FOR_EXPLOIT: u64 = 3;

/// Exponential forgetting factor applied per update.
/// Half-life = ln(2) / (1 - 0.995) ≈ 138 observations.
const FORGETTING_FACTOR: f64 = 0.995;

/// Market context for bandit arm selection.
///
/// Discretized from continuous market features into 81 cells.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpreadContext {
    /// Regime bucket: 0=Low, 1=Normal, 2=High.
    pub regime: u8,
    /// Position bucket: 0=Short, 1=Neutral, 2=Long.
    pub position: u8,
    /// Volatility bucket: 0=Low, 1=Normal, 2=High.
    pub vol: u8,
    /// Flow bucket: 0=Sell, 1=Neutral, 2=Buy.
    pub flow: u8,
}

impl SpreadContext {
    /// Build context from continuous market features.
    ///
    /// # Arguments
    /// * `regime_idx` - Regime from HMM (0=Low, 1=Normal, 2=High/Extreme)
    /// * `position_frac` - position / max_position, in [-1, 1]
    /// * `vol_ratio` - sigma / sigma_baseline (1.0 = normal)
    /// * `flow_imbalance` - flow_imbalance in [-1, 1]
    pub fn from_continuous(
        regime_idx: usize,
        position_frac: f64,
        vol_ratio: f64,
        flow_imbalance: f64,
    ) -> Self {
        let regime = (regime_idx.min(2)) as u8;

        let position = if position_frac < -0.3 {
            0 // Short
        } else if position_frac > 0.3 {
            2 // Long
        } else {
            1 // Neutral
        };

        let vol = if vol_ratio < 0.7 {
            0 // Low
        } else if vol_ratio > 1.5 {
            2 // High
        } else {
            1 // Normal
        };

        let flow = if flow_imbalance < -0.2 {
            0 // Sell
        } else if flow_imbalance > 0.2 {
            2 // Buy
        } else {
            1 // Neutral
        };

        Self {
            regime,
            position,
            vol,
            flow,
        }
    }

    /// Convert context to flat index in [0, 81).
    pub fn to_index(self) -> usize {
        self.regime as usize * (N_POSITION * N_VOL * N_FLOW)
            + self.position as usize * (N_VOL * N_FLOW)
            + self.vol as usize * N_FLOW
            + self.flow as usize
    }

    /// Reconstruct context from flat index.
    pub fn from_index(idx: usize) -> Self {
        let flow = (idx % N_FLOW) as u8;
        let vol = ((idx / N_FLOW) % N_VOL) as u8;
        let position = ((idx / (N_FLOW * N_VOL)) % N_POSITION) as u8;
        let regime = ((idx / (N_FLOW * N_VOL * N_POSITION)) % N_REGIME) as u8;
        Self {
            regime,
            position,
            vol,
            flow,
        }
    }
}

/// Result of arm selection — carries context for later reward attribution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BanditSelection {
    /// Context index at selection time.
    pub context_idx: usize,
    /// Arm index selected.
    pub arm_idx: usize,
    /// Spread multiplier for this arm.
    pub multiplier: f64,
    /// Whether this was an exploration (Thompson) or exploitation (greedy).
    pub is_exploration: bool,
    /// Timestamp of selection (ms since epoch).
    pub timestamp_ms: u64,
}

/// Checkpoint data for a single (context, arm) cell.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BanditCellCheckpoint {
    /// Context index.
    #[serde(default)]
    pub context_idx: usize,
    /// Arm index.
    #[serde(default)]
    pub arm_idx: usize,
    /// Posterior mean.
    #[serde(default)]
    pub mu_n: f64,
    /// Posterior precision scale.
    #[serde(default)]
    pub kappa_n: f64,
    /// Gamma shape.
    #[serde(default)]
    pub alpha: f64,
    /// Gamma rate.
    #[serde(default)]
    pub beta: f64,
    /// Observation count.
    #[serde(default)]
    pub n: u64,
}

/// Checkpoint for the full SpreadBandit.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SpreadBanditCheckpoint {
    /// Non-default cells (only cells with observations > 0).
    #[serde(default)]
    pub cells: Vec<BanditCellCheckpoint>,
    /// Total updates across all cells.
    #[serde(default)]
    pub total_updates: u64,
}

/// Contextual bandit for spread multiplier optimization.
///
/// 81 contexts × 8 arms = 648 Normal-Gamma posteriors.
/// Uses Thompson Sampling with exponential forgetting.
pub struct SpreadBandit {
    /// Q-value posteriors: cells[context_idx][arm_idx].
    cells: Vec<Vec<BayesianQValue>>,
    /// Pending selection awaiting reward.
    pending: Option<BanditSelection>,
    /// Total reward updates performed.
    total_updates: u64,
    /// Exponential forgetting factor (applied to kappa_n/n before each update).
    forgetting_factor: f64,
}

impl SpreadBandit {
    /// Create a new bandit with uninformative priors on all cells.
    pub fn new() -> Self {
        let cells = (0..N_CONTEXTS)
            .map(|_| (0..N_ARMS).map(|_| BayesianQValue::new()).collect())
            .collect();

        Self {
            cells,
            pending: None,
            total_updates: 0,
            forgetting_factor: FORGETTING_FACTOR,
        }
    }

    /// Select an arm for the given context using Thompson Sampling.
    ///
    /// Returns the selected spread multiplier and stores the selection
    /// for later reward attribution via `update()`.
    pub fn select_arm(&mut self, context: SpreadContext) -> BanditSelection {
        let ctx_idx = context.to_index();
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        // Check if any arm in this context has enough observations
        let max_obs = self.cells[ctx_idx]
            .iter()
            .map(|q| q.count())
            .max()
            .unwrap_or(0);

        let (arm_idx, is_exploration) = if max_obs < MIN_OBS_FOR_EXPLOIT {
            // Cold start: use default arm (1.00 = pure GLFT)
            (DEFAULT_ARM, false)
        } else {
            // Thompson Sampling: sample from each arm's posterior, pick highest
            let mut best_arm = DEFAULT_ARM;
            let mut best_sample = f64::NEG_INFINITY;

            for (i, q) in self.cells[ctx_idx].iter().enumerate() {
                let sample = if q.count() == 0 {
                    // Unvisited arm: sample from prior (encourages exploration)
                    q.sample()
                } else {
                    q.sample()
                };

                if sample > best_sample {
                    best_sample = sample;
                    best_arm = i;
                }
            }

            // Exploration = chose a non-greedy arm
            let greedy_arm = self.cells[ctx_idx]
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.mean().partial_cmp(&b.1.mean()).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(DEFAULT_ARM);

            (best_arm, best_arm != greedy_arm)
        };

        let selection = BanditSelection {
            context_idx: ctx_idx,
            arm_idx,
            multiplier: ARM_MULTIPLIERS[arm_idx],
            is_exploration,
            timestamp_ms: now_ms,
        };

        self.pending = Some(selection.clone());
        selection
    }

    /// Update the bandit with the reward from the most recent selection.
    ///
    /// Call this on fill with `reward = baseline_adjusted_edge_bps`.
    /// The pending selection is consumed.
    pub fn update_from_pending(&mut self, reward: f64) -> bool {
        let Some(selection) = self.pending.take() else {
            return false;
        };

        self.update_cell(selection.context_idx, selection.arm_idx, reward);
        true
    }

    /// Update a specific (context, arm) cell with a reward observation.
    ///
    /// Applies exponential forgetting before the update to allow
    /// adaptation to regime changes.
    pub fn update_cell(&mut self, context_idx: usize, arm_idx: usize, reward: f64) {
        if context_idx >= N_CONTEXTS || arm_idx >= N_ARMS {
            return;
        }

        let cell = &mut self.cells[context_idx][arm_idx];

        // Exponential forgetting: scale down effective observations
        // This prevents the posterior from becoming too tight and unable to adapt
        apply_forgetting(cell, self.forgetting_factor);

        // Standard Normal-Gamma conjugate update (correct for i.i.d. bandit)
        cell.update(reward);
        self.total_updates += 1;
    }

    /// Get the current pending selection (if any).
    pub fn pending(&self) -> Option<&BanditSelection> {
        self.pending.as_ref()
    }

    /// Clear the pending selection without updating.
    /// Use when a quote cycle expires without a fill.
    pub fn clear_pending(&mut self) {
        self.pending = None;
    }

    /// Get the spread multiplier for the best arm in a given context.
    /// Returns (multiplier, mean_reward, n_observations).
    pub fn best_arm(&self, context: SpreadContext) -> (f64, f64, u64) {
        let ctx_idx = context.to_index();
        let (best_idx, best_q) = self.cells[ctx_idx]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.mean().partial_cmp(&b.1.mean()).unwrap())
            .unwrap();

        (ARM_MULTIPLIERS[best_idx], best_q.mean(), best_q.count())
    }

    /// Total updates across all cells.
    pub fn total_updates(&self) -> u64 {
        self.total_updates
    }

    /// Total observations across all cells.
    pub fn total_observations(&self) -> u64 {
        self.cells
            .iter()
            .flat_map(|ctx| ctx.iter())
            .map(|q| q.count())
            .sum()
    }

    /// Number of cells with at least one observation.
    pub fn observed_cells(&self) -> usize {
        self.cells
            .iter()
            .flat_map(|ctx| ctx.iter())
            .filter(|q| q.count() > 0)
            .count()
    }

    /// Get summary statistics for logging.
    pub fn summary(&self) -> BanditSummary {
        let total_obs = self.total_observations();
        let observed = self.observed_cells();

        // Find the globally best arm across all contexts
        let mut best_mult = 1.0;
        let mut best_mean = f64::NEG_INFINITY;
        let mut best_ctx = 0;
        for (ctx_idx, arms) in self.cells.iter().enumerate() {
            for (arm_idx, q) in arms.iter().enumerate() {
                if q.count() >= MIN_OBS_FOR_EXPLOIT && q.mean() > best_mean {
                    best_mean = q.mean();
                    best_mult = ARM_MULTIPLIERS[arm_idx];
                    best_ctx = ctx_idx;
                }
            }
        }

        if best_mean == f64::NEG_INFINITY {
            best_mean = 0.0;
            best_mult = 1.0;
        }

        BanditSummary {
            total_observations: total_obs,
            observed_cells: observed,
            total_cells: N_CONTEXTS * N_ARMS,
            best_multiplier: best_mult,
            best_mean_reward_bps: best_mean,
            best_context_idx: best_ctx,
            total_updates: self.total_updates,
        }
    }

    /// Create checkpoint for persistence.
    pub fn to_checkpoint(&self) -> SpreadBanditCheckpoint {
        let mut cells = Vec::new();
        for (ctx_idx, arms) in self.cells.iter().enumerate() {
            for (arm_idx, q) in arms.iter().enumerate() {
                if q.count() > 0 {
                    cells.push(BanditCellCheckpoint {
                        context_idx: ctx_idx,
                        arm_idx,
                        mu_n: q.mean(),
                        kappa_n: q.kappa_n(),
                        alpha: q.alpha(),
                        beta: q.beta(),
                        n: q.count(),
                    });
                }
            }
        }

        SpreadBanditCheckpoint {
            cells,
            total_updates: self.total_updates,
        }
    }

    /// Restore from checkpoint.
    pub fn restore_from_checkpoint(&mut self, checkpoint: &SpreadBanditCheckpoint) {
        for cell in &checkpoint.cells {
            if cell.context_idx < N_CONTEXTS && cell.arm_idx < N_ARMS {
                self.cells[cell.context_idx][cell.arm_idx] =
                    BayesianQValue::from_checkpoint(
                        cell.mu_n,
                        cell.kappa_n,
                        cell.alpha,
                        cell.beta,
                        cell.n,
                    );
            }
        }
        self.total_updates = checkpoint.total_updates;
    }

    /// Get the arm multiplier values.
    pub fn arm_multipliers() -> &'static [f64; N_ARMS] {
        &ARM_MULTIPLIERS
    }

    /// Get the number of contexts.
    pub fn n_contexts() -> usize {
        N_CONTEXTS
    }

    /// Get the number of arms.
    pub fn n_arms() -> usize {
        N_ARMS
    }
}

impl Default for SpreadBandit {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for SpreadBandit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpreadBandit")
            .field("total_updates", &self.total_updates)
            .field("observed_cells", &self.observed_cells())
            .field("has_pending", &self.pending.is_some())
            .finish()
    }
}

/// Apply exponential forgetting to a BayesianQValue cell.
///
/// Scales kappa_n toward the prior strength and alpha/beta toward their priors.
/// This effectively reduces the "effective sample size" so the posterior
/// remains responsive to recent observations.
/// Minimum observations before forgetting kicks in.
/// Below this, cells are still accumulating initial data and forgetting
/// would prevent them from ever reaching MIN_OBS_FOR_EXPLOIT.
const MIN_OBS_FOR_FORGETTING: u64 = 10;

fn apply_forgetting(cell: &mut BayesianQValue, factor: f64) {
    let n = cell.count();
    if n < MIN_OBS_FOR_FORGETTING {
        // Don't apply forgetting to young cells — let them accumulate
        // initial observations so Thompson Sampling can activate.
        return;
    }

    // We can't directly modify the private fields of BayesianQValue,
    // so we reconstruct with discounted parameters.
    // The forgetting is: treat the cell as if it had `factor * n` observations.
    let effective_n = ((n as f64) * factor).max(1.0) as u64;

    // Only apply forgetting when it actually reduces observations
    if effective_n < n {
        // Reconstruct with reduced effective count but same posterior statistics
        *cell = BayesianQValue::from_checkpoint(
            cell.mean(),
            cell.kappa_n() * factor,
            1.0 + (cell.alpha() - 1.0) * factor,
            cell.beta() * factor,
            effective_n,
        );
    }
}

/// Summary statistics for logging.
#[derive(Debug, Clone)]
pub struct BanditSummary {
    /// Total observations across all cells.
    pub total_observations: u64,
    /// Number of cells with at least one observation.
    pub observed_cells: usize,
    /// Total cells (81 × 8 = 648).
    pub total_cells: usize,
    /// Best multiplier (highest mean reward across all observed cells).
    pub best_multiplier: f64,
    /// Best mean reward in bps.
    pub best_mean_reward_bps: f64,
    /// Context index of the best arm.
    pub best_context_idx: usize,
    /// Total updates performed.
    pub total_updates: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_roundtrip() {
        for regime in 0..N_REGIME {
            for pos in 0..N_POSITION {
                for vol in 0..N_VOL {
                    for flow in 0..N_FLOW {
                        let ctx = SpreadContext {
                            regime: regime as u8,
                            position: pos as u8,
                            vol: vol as u8,
                            flow: flow as u8,
                        };
                        let idx = ctx.to_index();
                        assert!(idx < N_CONTEXTS, "index {idx} >= {N_CONTEXTS}");
                        let ctx2 = SpreadContext::from_index(idx);
                        assert_eq!(ctx, ctx2, "roundtrip failed for index {idx}");
                    }
                }
            }
        }
    }

    #[test]
    fn test_context_from_continuous() {
        // Neutral everything
        let ctx = SpreadContext::from_continuous(1, 0.0, 1.0, 0.0);
        assert_eq!(ctx.regime, 1);
        assert_eq!(ctx.position, 1); // Neutral
        assert_eq!(ctx.vol, 1); // Normal
        assert_eq!(ctx.flow, 1); // Neutral

        // Extreme: short position, high vol, sell flow, high regime
        let ctx = SpreadContext::from_continuous(2, -0.8, 3.0, -0.5);
        assert_eq!(ctx.regime, 2);
        assert_eq!(ctx.position, 0); // Short
        assert_eq!(ctx.vol, 2); // High
        assert_eq!(ctx.flow, 0); // Sell
    }

    #[test]
    fn test_bandit_cold_start_defaults_to_pure_glft() {
        let mut bandit = SpreadBandit::new();
        let ctx = SpreadContext::from_continuous(1, 0.0, 1.0, 0.0);
        let selection = bandit.select_arm(ctx);

        // Cold start should select arm 3 (multiplier 1.00)
        assert_eq!(selection.arm_idx, DEFAULT_ARM);
        assert!((selection.multiplier - 1.0).abs() < 1e-10);
        assert!(!selection.is_exploration);
    }

    #[test]
    fn test_bandit_explores_after_warmup() {
        let mut bandit = SpreadBandit::new();
        let ctx = SpreadContext::from_continuous(1, 0.0, 1.0, 0.0);
        let ctx_idx = ctx.to_index();

        // Warm up ALL arms with some observations so Thompson Sampling activates
        // (same reward for all arms → TS should explore broadly)
        for arm in 0..N_ARMS {
            for _ in 0..5 {
                bandit.update_cell(ctx_idx, arm, 1.0);
            }
        }

        // After warmup, Thompson Sampling should explore different arms
        let mut arms_seen = std::collections::HashSet::new();
        for _ in 0..200 {
            let selection = bandit.select_arm(ctx);
            arms_seen.insert(selection.arm_idx);
            bandit.update_from_pending(1.0); // Same reward for all arms
        }

        // Should have explored at least 3 different arms within 200 cycles
        assert!(
            arms_seen.len() >= 3,
            "expected exploration of ≥3 arms, got {} ({:?})",
            arms_seen.len(),
            arms_seen
        );
    }

    #[test]
    fn test_bandit_converges_to_best_arm() {
        let mut bandit = SpreadBandit::new();
        let ctx = SpreadContext::from_continuous(1, 0.0, 1.0, 0.0);
        let ctx_idx = ctx.to_index();

        // Feed arm 5 (multiplier 1.15) much higher rewards
        let target_arm = 5;
        for arm in 0..N_ARMS {
            for _ in 0..20 {
                let reward = if arm == target_arm { 5.0 } else { -1.0 };
                bandit.update_cell(ctx_idx, arm, reward);
            }
        }

        // Best arm should be the one with highest rewards
        let (mult, mean, _n) = bandit.best_arm(ctx);
        assert!(
            (mult - ARM_MULTIPLIERS[target_arm]).abs() < 1e-10,
            "expected best mult {}, got {}",
            ARM_MULTIPLIERS[target_arm],
            mult
        );
        assert!(mean > 0.0);
    }

    #[test]
    fn test_update_from_pending() {
        let mut bandit = SpreadBandit::new();
        let ctx = SpreadContext::from_continuous(1, 0.0, 1.0, 0.0);

        // Warm up so Thompson Sampling activates
        let ctx_idx = ctx.to_index();
        for arm in 0..N_ARMS {
            for _ in 0..5 {
                bandit.update_cell(ctx_idx, arm, 0.0);
            }
        }

        let _ = bandit.select_arm(ctx);
        assert!(bandit.pending().is_some());

        let updated = bandit.update_from_pending(2.5);
        assert!(updated);
        assert!(bandit.pending().is_none());

        // Second call should return false (no pending)
        let updated2 = bandit.update_from_pending(1.0);
        assert!(!updated2);
    }

    #[test]
    fn test_clear_pending() {
        let mut bandit = SpreadBandit::new();
        let ctx = SpreadContext::from_continuous(1, 0.0, 1.0, 0.0);

        let _ = bandit.select_arm(ctx);
        assert!(bandit.pending().is_some());

        bandit.clear_pending();
        assert!(bandit.pending().is_none());
    }

    #[test]
    fn test_checkpoint_roundtrip() {
        let mut bandit = SpreadBandit::new();
        let ctx = SpreadContext::from_continuous(1, 0.0, 1.0, 0.0);
        let ctx_idx = ctx.to_index();

        // Feed some observations
        for arm in 0..3 {
            for i in 0..10 {
                bandit.update_cell(ctx_idx, arm, i as f64 * 0.5);
            }
        }

        let checkpoint = bandit.to_checkpoint();
        assert!(!checkpoint.cells.is_empty());

        // Restore into a fresh bandit
        let mut bandit2 = SpreadBandit::new();
        bandit2.restore_from_checkpoint(&checkpoint);

        // Verify posteriors match
        for arm in 0..3 {
            let obs_original = bandit.cells[ctx_idx][arm].count();
            let obs_restored = bandit2.cells[ctx_idx][arm].count();
            assert_eq!(obs_original, obs_restored, "arm {arm} count mismatch");

            let mean_original = bandit.cells[ctx_idx][arm].mean();
            let mean_restored = bandit2.cells[ctx_idx][arm].mean();
            assert!(
                (mean_original - mean_restored).abs() < 1e-6,
                "arm {arm} mean mismatch: {mean_original} vs {mean_restored}"
            );
        }
    }

    #[test]
    fn test_forgetting_reduces_effective_count() {
        let mut q = BayesianQValue::new();
        for _ in 0..100 {
            q.update(1.0);
        }
        assert_eq!(q.count(), 100);

        let original_kappa = q.kappa_n();
        apply_forgetting(&mut q, 0.995);

        // After forgetting, effective count should be reduced
        assert!(q.count() < 100);
        // Kappa should be reduced
        assert!(q.kappa_n() < original_kappa);
    }

    #[test]
    fn test_summary() {
        let mut bandit = SpreadBandit::new();
        let summary = bandit.summary();
        assert_eq!(summary.total_observations, 0);
        assert_eq!(summary.observed_cells, 0);
        assert_eq!(summary.total_cells, N_CONTEXTS * N_ARMS);

        // Add some data
        let ctx = SpreadContext::from_continuous(1, 0.0, 1.0, 0.0);
        let ctx_idx = ctx.to_index();
        for _ in 0..5 {
            bandit.update_cell(ctx_idx, 3, 2.0);
        }

        let summary = bandit.summary();
        assert!(summary.total_observations > 0);
        assert!(summary.observed_cells > 0);
        assert!((summary.best_multiplier - 1.0).abs() < 1e-10);
        assert!(summary.best_mean_reward_bps > 0.0);
    }

    #[test]
    fn test_arm_multipliers_are_sorted() {
        for i in 1..N_ARMS {
            assert!(
                ARM_MULTIPLIERS[i] > ARM_MULTIPLIERS[i - 1],
                "arms not sorted at index {i}"
            );
        }
    }

    #[test]
    fn test_all_contexts_have_valid_indices() {
        for i in 0..N_CONTEXTS {
            let ctx = SpreadContext::from_index(i);
            assert!(ctx.regime < N_REGIME as u8);
            assert!(ctx.position < N_POSITION as u8);
            assert!(ctx.vol < N_VOL as u8);
            assert!(ctx.flow < N_FLOW as u8);
        }
    }
}
