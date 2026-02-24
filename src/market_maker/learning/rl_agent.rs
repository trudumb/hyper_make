//! Phase 8: Reinforcement Learning Agent for Competitive Quoting
//!
//! **DEPRECATED**: This MDP-based RL agent is superseded by `SpreadBandit`
//! (contextual bandit with Thompson Sampling). The MDP framing is incorrect
//! for spread selection — quote cycles are i.i.d., not state-transitions.
//! Kept for checkpoint backward compatibility only.
//!
//! This module implements an MDP-based approach to quoting decisions with:
//! - Discretized state space (inventory, OBI, regime, posteriors)
//! - Action space (spread adjustments, skew modifications)
//! - Q-learning with Thompson sampling for Bayesian exploration
//! - Competitor modeling for game-theoretic adaptation
//!
//! Theory: Frame quoting as an MDP where the agent learns optimal policies
//! by exploring via Thompson sampling on Bayesian Q-value posteriors.

use std::collections::{HashMap, VecDeque};
use tracing::debug;

use super::baseline_tracker::BaselineTracker;
use serde::{Deserialize, Serialize};

/// RL agent Q-table entry for checkpoint persistence (deprecated — local to rl_agent).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QTableEntry {
    pub state_index: usize,
    pub action_index: usize,
    pub mu_n: f64,
    pub kappa_n: f64,
    pub alpha: f64,
    pub beta: f64,
    pub n: u64,
}

/// RL agent checkpoint (deprecated — local to rl_agent).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RLCheckpoint {
    pub q_entries: Vec<QTableEntry>,
    pub episodes: u64,
    pub total_reward: f64,
    pub total_observations: u64,
    #[serde(default)]
    pub action_space_version: u32,
    #[serde(default)]
    pub use_compact_state: bool,
    #[serde(default)]
    pub reward_config_hash: u64,
    #[serde(default)]
    pub use_drift_bucket: bool,
}

// ============================================================================
// MDP State Space
// ============================================================================

/// Discretized inventory bucket for state representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InventoryBucket {
    /// Short position (< -20% of max)
    Short,
    /// Small short position (-20% to -5%)
    SmallShort,
    /// Neutral position (-5% to +5%)
    Neutral,
    /// Small long position (+5% to +20%)
    SmallLong,
    /// Long position (> +20%)
    Long,
}

impl InventoryBucket {
    /// Convert continuous position to bucket.
    pub fn from_position(position: f64, max_position: f64) -> Self {
        if max_position <= 0.0 {
            return Self::Neutral;
        }
        let ratio = position / max_position;
        match ratio {
            r if r < -0.2 => Self::Short,
            r if r < -0.05 => Self::SmallShort,
            r if r < 0.05 => Self::Neutral,
            r if r < 0.2 => Self::SmallLong,
            _ => Self::Long,
        }
    }

    /// Get bucket index (0-4).
    pub fn index(&self) -> usize {
        match self {
            Self::Short => 0,
            Self::SmallShort => 1,
            Self::Neutral => 2,
            Self::SmallLong => 3,
            Self::Long => 4,
        }
    }

    /// Number of buckets.
    pub const COUNT: usize = 5;

    /// Reconstruct from bucket index (0-4).
    pub fn from_index(idx: usize) -> Self {
        match idx {
            0 => Self::Short,
            1 => Self::SmallShort,
            2 => Self::Neutral,
            3 => Self::SmallLong,
            4 => Self::Long,
            _ => Self::Neutral,
        }
    }
}

/// Discretized order book imbalance bucket.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ImbalanceBucket {
    /// Sell pressure (< -0.15)
    Sell,
    /// Weak sell pressure (-0.15 to -0.05)
    WeakSell,
    /// Neutral (-0.05 to +0.05)
    Neutral,
    /// Weak buy pressure (+0.05 to +0.15)
    WeakBuy,
    /// Buy pressure (> +0.15)
    Buy,
}

impl ImbalanceBucket {
    /// Convert continuous imbalance to bucket.
    pub fn from_imbalance(imbalance: f64) -> Self {
        match imbalance {
            i if i < -0.15 => Self::Sell,
            i if i < -0.05 => Self::WeakSell,
            i if i < 0.05 => Self::Neutral,
            i if i < 0.15 => Self::WeakBuy,
            _ => Self::Buy,
        }
    }

    /// Get bucket index (0-4).
    pub fn index(&self) -> usize {
        match self {
            Self::Sell => 0,
            Self::WeakSell => 1,
            Self::Neutral => 2,
            Self::WeakBuy => 3,
            Self::Buy => 4,
        }
    }

    /// Number of buckets.
    pub const COUNT: usize = 5;

    /// Reconstruct from bucket index (0-4).
    pub fn from_index(idx: usize) -> Self {
        match idx {
            0 => Self::Sell,
            1 => Self::WeakSell,
            2 => Self::Neutral,
            3 => Self::WeakBuy,
            4 => Self::Buy,
            _ => Self::Neutral,
        }
    }
}

/// Discretized volatility regime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VolatilityBucket {
    /// Low volatility (< 0.8x baseline)
    Low,
    /// Normal volatility (0.8x to 1.2x baseline)
    Normal,
    /// High volatility (> 1.2x baseline)
    High,
}

impl VolatilityBucket {
    /// Convert volatility ratio to bucket.
    pub fn from_vol_ratio(vol_ratio: f64) -> Self {
        match vol_ratio {
            r if r < 0.8 => Self::Low,
            r if r < 1.2 => Self::Normal,
            _ => Self::High,
        }
    }

    /// Get bucket index (0-2).
    pub fn index(&self) -> usize {
        match self {
            Self::Low => 0,
            Self::Normal => 1,
            Self::High => 2,
        }
    }

    /// Number of buckets.
    pub const COUNT: usize = 3;

    /// Reconstruct from bucket index (0-2).
    pub fn from_index(idx: usize) -> Self {
        match idx {
            0 => Self::Low,
            1 => Self::Normal,
            2 => Self::High,
            _ => Self::Normal,
        }
    }
}

/// Discretized adverse selection posterior belief.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AdverseBucket {
    /// Low adverse risk (< 0.2)
    Low,
    /// Moderate adverse risk (0.2 to 0.35)
    Moderate,
    /// High adverse risk (> 0.35)
    High,
}

impl AdverseBucket {
    /// Convert posterior mean to bucket.
    pub fn from_posterior_mean(mean: f64) -> Self {
        match mean {
            m if m < 0.2 => Self::Low,
            m if m < 0.35 => Self::Moderate,
            _ => Self::High,
        }
    }

    /// Get bucket index (0-2).
    pub fn index(&self) -> usize {
        match self {
            Self::Low => 0,
            Self::Moderate => 1,
            Self::High => 2,
        }
    }

    /// Number of buckets.
    pub const COUNT: usize = 3;

    /// Reconstruct from bucket index (0-2).
    pub fn from_index(idx: usize) -> Self {
        match idx {
            0 => Self::Low,
            1 => Self::Moderate,
            2 => Self::High,
            _ => Self::Moderate,
        }
    }
}

/// Discretized Hawkes excitation level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExcitationBucket {
    /// Normal excitation (branching ratio < 0.6)
    Normal,
    /// Elevated excitation (0.6 to 0.8)
    Elevated,
    /// High excitation (> 0.8)
    High,
}

/// Discretized drift/momentum bucket.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DriftBucket {
    /// Bearish drift (momentum < -5 bps)
    Bearish,
    /// Neutral drift (-5 to +5 bps)
    Neutral,
    /// Bullish drift (momentum > +5 bps)
    Bullish,
}

impl DriftBucket {
    /// Convert continuous momentum (in bps) to a drift bucket.
    pub fn from_momentum_bps(momentum_bps: f64) -> Self {
        if momentum_bps < -5.0 {
            Self::Bearish
        } else if momentum_bps > 5.0 {
            Self::Bullish
        } else {
            Self::Neutral
        }
    }

    /// Get bucket index (0-2).
    pub fn index(&self) -> usize {
        match self {
            Self::Bearish => 0,
            Self::Neutral => 1,
            Self::Bullish => 2,
        }
    }

    /// Number of buckets.
    pub const COUNT: usize = 3;

    /// Reconstruct from bucket index (0-2).
    pub fn from_index(idx: usize) -> Self {
        match idx {
            0 => Self::Bearish,
            2 => Self::Bullish,
            _ => Self::Neutral,
        }
    }
}

impl ExcitationBucket {
    /// Convert branching ratio to bucket.
    pub fn from_branching_ratio(ratio: f64) -> Self {
        match ratio {
            r if r < 0.6 => Self::Normal,
            r if r < 0.8 => Self::Elevated,
            _ => Self::High,
        }
    }

    /// Get bucket index (0-2).
    pub fn index(&self) -> usize {
        match self {
            Self::Normal => 0,
            Self::Elevated => 1,
            Self::High => 2,
        }
    }

    /// Number of buckets.
    pub const COUNT: usize = 3;

    /// Reconstruct from bucket index (0-2).
    pub fn from_index(idx: usize) -> Self {
        match idx {
            0 => Self::Normal,
            1 => Self::Elevated,
            2 => Self::High,
            _ => Self::Normal,
        }
    }
}

/// Complete discretized MDP state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MDPState {
    /// Inventory bucket
    pub inventory: InventoryBucket,
    /// Order book imbalance bucket
    pub imbalance: ImbalanceBucket,
    /// Volatility bucket
    pub volatility: VolatilityBucket,
    /// Adverse selection posterior bucket
    pub adverse: AdverseBucket,
    /// Hawkes excitation bucket
    pub excitation: ExcitationBucket,
    /// Drift/momentum bucket
    pub drift: DriftBucket,
}

impl MDPState {
    /// Create from continuous state values.
    ///
    /// `momentum_bps` is used for drift bucketing when `use_drift_bucket` is enabled.
    /// Pass 0.0 when drift bucketing is disabled.
    pub fn from_continuous(
        position: f64,
        max_position: f64,
        book_imbalance: f64,
        vol_ratio: f64,
        adverse_posterior: f64,
        hawkes_branching: f64,
        momentum_bps: f64,
    ) -> Self {
        Self {
            inventory: InventoryBucket::from_position(position, max_position),
            imbalance: ImbalanceBucket::from_imbalance(book_imbalance),
            volatility: VolatilityBucket::from_vol_ratio(vol_ratio),
            adverse: AdverseBucket::from_posterior_mean(adverse_posterior),
            excitation: ExcitationBucket::from_branching_ratio(hawkes_branching),
            drift: DriftBucket::from_momentum_bps(momentum_bps),
        }
    }

    /// Convert to flat state index for Q-table lookup.
    /// Total states = 5 * 5 * 3 * 3 * 3 * 3 = 2025
    pub fn to_index(&self) -> usize {
        let mut idx = self.inventory.index();
        idx = idx * ImbalanceBucket::COUNT + self.imbalance.index();
        idx = idx * VolatilityBucket::COUNT + self.volatility.index();
        idx = idx * AdverseBucket::COUNT + self.adverse.index();
        idx = idx * ExcitationBucket::COUNT + self.excitation.index();
        idx = idx * DriftBucket::COUNT + self.drift.index();
        idx
    }

    /// Total number of discrete states.
    pub const STATE_COUNT: usize = InventoryBucket::COUNT
        * ImbalanceBucket::COUNT
        * VolatilityBucket::COUNT
        * AdverseBucket::COUNT
        * ExcitationBucket::COUNT
        * DriftBucket::COUNT;

    /// Reconstruct an MDPState from a flat index (inverse of `to_index()`).
    ///
    /// Uses modular arithmetic to extract each dimension in reverse order
    /// of how `to_index()` encodes them.
    pub fn from_index(idx: usize) -> Self {
        let mut remaining = idx;
        let drift_idx = remaining % DriftBucket::COUNT;
        remaining /= DriftBucket::COUNT;
        let excitation_idx = remaining % ExcitationBucket::COUNT;
        remaining /= ExcitationBucket::COUNT;
        let adverse_idx = remaining % AdverseBucket::COUNT;
        remaining /= AdverseBucket::COUNT;
        let volatility_idx = remaining % VolatilityBucket::COUNT;
        remaining /= VolatilityBucket::COUNT;
        let imbalance_idx = remaining % ImbalanceBucket::COUNT;
        remaining /= ImbalanceBucket::COUNT;
        let inventory_idx = remaining % InventoryBucket::COUNT;

        Self {
            inventory: InventoryBucket::from_index(inventory_idx),
            imbalance: ImbalanceBucket::from_index(imbalance_idx),
            volatility: VolatilityBucket::from_index(volatility_idx),
            adverse: AdverseBucket::from_index(adverse_idx),
            excitation: ExcitationBucket::from_index(excitation_idx),
            drift: DriftBucket::from_index(drift_idx),
        }
    }
}

impl Default for MDPState {
    fn default() -> Self {
        Self {
            inventory: InventoryBucket::Neutral,
            imbalance: ImbalanceBucket::Neutral,
            volatility: VolatilityBucket::Normal,
            adverse: AdverseBucket::Moderate,
            excitation: ExcitationBucket::Normal,
            drift: DriftBucket::Neutral,
        }
    }
}

// ============================================================================
// Compact 3D State Space (Phase 2: P1-2)
// ============================================================================

/// Compact 3-bucket imbalance for reduced state space.
/// Merges WeakSell→Sell, WeakBuy→Buy from the 5-bucket version.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ImbalanceBucketCompact {
    /// Sell pressure (< -0.05)
    Sell,
    /// Neutral (-0.05 to +0.05)
    Neutral,
    /// Buy pressure (> +0.05)
    Buy,
}

impl ImbalanceBucketCompact {
    /// Convert continuous imbalance to compact bucket.
    pub fn from_imbalance(imbalance: f64) -> Self {
        match imbalance {
            i if i < -0.05 => Self::Sell,
            i if i < 0.05 => Self::Neutral,
            _ => Self::Buy,
        }
    }

    /// Get bucket index (0-2).
    pub fn index(&self) -> usize {
        match self {
            Self::Sell => 0,
            Self::Neutral => 1,
            Self::Buy => 2,
        }
    }

    /// Number of buckets.
    pub const COUNT: usize = 3;

    /// Reconstruct from bucket index (0-2).
    pub fn from_index(idx: usize) -> Self {
        match idx {
            0 => Self::Sell,
            1 => Self::Neutral,
            _ => Self::Buy,
        }
    }
}

/// Compact 3D MDP state: Inventory(5) x Volatility(3) x Imbalance(3) = 45 states.
///
/// Drops AdverseBucket (AS captured in reward after P0-2) and ExcitationBucket
/// (correlated with volatility). Merges imbalance from 5 to 3 buckets.
///
/// Literature (ISAC): even 1D state (|inventory|) with 1D action (gamma)
/// achieves 36% inventory reduction. State space efficiency > expressiveness
/// when data is scarce.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MDPStateCompact {
    /// Inventory bucket (5 levels)
    pub inventory: InventoryBucket,
    /// Volatility bucket (3 levels)
    pub volatility: VolatilityBucket,
    /// Order book imbalance bucket (3 levels, compact)
    pub imbalance: ImbalanceBucketCompact,
}

impl MDPStateCompact {
    /// Create from continuous state values.
    pub fn from_continuous(
        position: f64,
        max_position: f64,
        book_imbalance: f64,
        vol_ratio: f64,
    ) -> Self {
        Self {
            inventory: InventoryBucket::from_position(position, max_position),
            volatility: VolatilityBucket::from_vol_ratio(vol_ratio),
            imbalance: ImbalanceBucketCompact::from_imbalance(book_imbalance),
        }
    }

    /// Convert to flat state index for Q-table lookup.
    /// Total states = 5 * 3 * 3 = 45
    pub fn to_index(&self) -> usize {
        let mut idx = self.inventory.index();
        idx = idx * VolatilityBucket::COUNT + self.volatility.index();
        idx = idx * ImbalanceBucketCompact::COUNT + self.imbalance.index();
        idx
    }

    /// Total number of discrete states.
    pub const STATE_COUNT: usize =
        InventoryBucket::COUNT * VolatilityBucket::COUNT * ImbalanceBucketCompact::COUNT;

    /// Reconstruct from flat index.
    pub fn from_index(idx: usize) -> Self {
        let mut remaining = idx;
        let imbalance_idx = remaining % ImbalanceBucketCompact::COUNT;
        remaining /= ImbalanceBucketCompact::COUNT;
        let volatility_idx = remaining % VolatilityBucket::COUNT;
        remaining /= VolatilityBucket::COUNT;
        let inventory_idx = remaining % InventoryBucket::COUNT;

        Self {
            inventory: InventoryBucket::from_index(inventory_idx),
            volatility: VolatilityBucket::from_index(volatility_idx),
            imbalance: ImbalanceBucketCompact::from_index(imbalance_idx),
        }
    }
}

impl Default for MDPStateCompact {
    fn default() -> Self {
        Self {
            inventory: InventoryBucket::Neutral,
            volatility: VolatilityBucket::Normal,
            imbalance: ImbalanceBucketCompact::Neutral,
        }
    }
}

/// Maximum number of pending state-action pairs in the FIFO queue.
const STATE_ACTION_QUEUE_CAPACITY: usize = 8;

// ============================================================================
// Action Space
// ============================================================================

/// Spread adjustment action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpreadAction {
    /// Tighten spread significantly (-3 bps)
    TightenLarge,
    /// Tighten spread moderately (-1.5 bps)
    TightenSmall,
    /// Keep current spread (0 bps)
    Maintain,
    /// Widen spread moderately (+1.5 bps)
    WidenSmall,
    /// Widen spread significantly (+3 bps)
    WidenLarge,
}

impl SpreadAction {
    /// Get the spread delta in basis points.
    pub fn delta_bps(&self) -> f64 {
        match self {
            Self::TightenLarge => -3.0,
            Self::TightenSmall => -1.5,
            Self::Maintain => 0.0,
            Self::WidenSmall => 1.5,
            Self::WidenLarge => 3.0,
        }
    }

    /// Get action index (0-4).
    pub fn index(&self) -> usize {
        match self {
            Self::TightenLarge => 0,
            Self::TightenSmall => 1,
            Self::Maintain => 2,
            Self::WidenSmall => 3,
            Self::WidenLarge => 4,
        }
    }

    /// Create from index.
    pub fn from_index(idx: usize) -> Self {
        match idx {
            0 => Self::TightenLarge,
            1 => Self::TightenSmall,
            2 => Self::Maintain,
            3 => Self::WidenSmall,
            _ => Self::WidenLarge,
        }
    }

    /// Number of spread actions.
    pub const COUNT: usize = 5;
}

/// Skew adjustment action (asymmetric bid/ask).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SkewAction {
    /// Strongly favor asks (bid wider by 2 bps)
    StrongAskBias,
    /// Moderately favor asks (bid wider by 1 bps)
    ModerateAskBias,
    /// Symmetric quotes
    Symmetric,
    /// Moderately favor bids (ask wider by 1 bps)
    ModerateBidBias,
    /// Strongly favor bids (ask wider by 2 bps)
    StrongBidBias,
}

impl SkewAction {
    /// Get bid skew in basis points (positive = wider bid).
    pub fn bid_skew_bps(&self) -> f64 {
        match self {
            Self::StrongAskBias => 2.0,
            Self::ModerateAskBias => 1.0,
            Self::Symmetric => 0.0,
            Self::ModerateBidBias => -1.0,
            Self::StrongBidBias => -2.0,
        }
    }

    /// Get ask skew in basis points (positive = wider ask).
    pub fn ask_skew_bps(&self) -> f64 {
        -self.bid_skew_bps()
    }

    /// Get action index (0-4).
    pub fn index(&self) -> usize {
        match self {
            Self::StrongAskBias => 0,
            Self::ModerateAskBias => 1,
            Self::Symmetric => 2,
            Self::ModerateBidBias => 3,
            Self::StrongBidBias => 4,
        }
    }

    /// Create from index.
    pub fn from_index(idx: usize) -> Self {
        match idx {
            0 => Self::StrongAskBias,
            1 => Self::ModerateAskBias,
            2 => Self::Symmetric,
            3 => Self::ModerateBidBias,
            _ => Self::StrongBidBias,
        }
    }

    /// Number of skew actions.
    pub const COUNT: usize = 5;
}

// ============================================================================
// Phase 6A.2: Parameter-Based RL Actions
// ============================================================================
// Instead of directly modifying spreads/skews in bps, tune multipliers
// on the HJB-optimal parameters. This keeps RL within safe stochastic
// control bounds and is more theoretically sound.

/// Risk aversion (γ) multiplier action.
/// Applied to base γ from HJB solver: effective_γ = base_γ × multiplier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GammaAction {
    /// Very defensive (2.0× base γ = wider spreads)
    VeryDefensive,
    /// Moderately defensive (1.5× base γ)
    Defensive,
    /// Use base γ as-is (1.0×)
    Neutral,
    /// Moderately aggressive (0.75× base γ = tighter spreads)
    Aggressive,
    /// Very aggressive (0.5× base γ)
    VeryAggressive,
}

impl GammaAction {
    /// Get multiplier for γ.
    pub fn multiplier(&self) -> f64 {
        match self {
            Self::VeryDefensive => 2.0,
            Self::Defensive => 1.5,
            Self::Neutral => 1.0,
            Self::Aggressive => 0.75,
            Self::VeryAggressive => 0.5,
        }
    }

    /// Get action index (0-4).
    pub fn index(&self) -> usize {
        match self {
            Self::VeryDefensive => 0,
            Self::Defensive => 1,
            Self::Neutral => 2,
            Self::Aggressive => 3,
            Self::VeryAggressive => 4,
        }
    }

    /// Create from index.
    pub fn from_index(idx: usize) -> Self {
        match idx {
            0 => Self::VeryDefensive,
            1 => Self::Defensive,
            2 => Self::Neutral,
            3 => Self::Aggressive,
            _ => Self::VeryAggressive,
        }
    }

    /// Number of γ actions.
    pub const COUNT: usize = 5;
}

/// Inventory skew (ω) multiplier action.
/// Applied to base skew from position manager: effective_ω = base_ω × multiplier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OmegaAction {
    /// Strong skew (2.0× = aggressively clear inventory)
    StrongSkew,
    /// Moderate skew (1.5×)
    ModerateSkew,
    /// Use base skew as-is (1.0×)
    Neutral,
    /// Reduced skew (0.5× = slower inventory clearing)
    ReducedSkew,
    /// Minimal skew (0.25× = almost ignore inventory)
    MinimalSkew,
}

impl OmegaAction {
    /// Get multiplier for ω.
    pub fn multiplier(&self) -> f64 {
        match self {
            Self::StrongSkew => 2.0,
            Self::ModerateSkew => 1.5,
            Self::Neutral => 1.0,
            Self::ReducedSkew => 0.5,
            Self::MinimalSkew => 0.25,
        }
    }

    /// Get action index (0-4).
    pub fn index(&self) -> usize {
        match self {
            Self::StrongSkew => 0,
            Self::ModerateSkew => 1,
            Self::Neutral => 2,
            Self::ReducedSkew => 3,
            Self::MinimalSkew => 4,
        }
    }

    /// Create from index.
    pub fn from_index(idx: usize) -> Self {
        match idx {
            0 => Self::StrongSkew,
            1 => Self::ModerateSkew,
            2 => Self::Neutral,
            3 => Self::ReducedSkew,
            _ => Self::MinimalSkew,
        }
    }

    /// Number of ω actions.
    pub const COUNT: usize = 5;
}

/// Quote intensity action.
/// Controls what fraction of maximum size to quote.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IntensityAction {
    /// Don't quote at all (wait for better conditions)
    NoQuote,
    /// Quote at 25% of max size
    Light,
    /// Quote at 50% of max size
    Moderate,
    /// Quote at 75% of max size
    Heavy,
    /// Quote at full size
    Full,
}

impl IntensityAction {
    /// Get intensity as fraction [0, 1].
    pub fn intensity(&self) -> f64 {
        match self {
            Self::NoQuote => 0.0,
            Self::Light => 0.25,
            Self::Moderate => 0.5,
            Self::Heavy => 0.75,
            Self::Full => 1.0,
        }
    }

    /// Get action index (0-4).
    pub fn index(&self) -> usize {
        match self {
            Self::NoQuote => 0,
            Self::Light => 1,
            Self::Moderate => 2,
            Self::Heavy => 3,
            Self::Full => 4,
        }
    }

    /// Create from index.
    pub fn from_index(idx: usize) -> Self {
        match idx {
            0 => Self::NoQuote,
            1 => Self::Light,
            2 => Self::Moderate,
            3 => Self::Heavy,
            _ => Self::Full,
        }
    }

    /// Number of intensity actions.
    pub const COUNT: usize = 5;
}

/// Complete parameter-based action for the MDP.
/// Tunes γ, ω multipliers instead of raw bps adjustments.
/// Total actions: 5 × 5 = 25 (same count as MDPAction, but operating on GLFT parameters).
///
/// Literature (Falces Marin 2022): "RL agent controls gamma and skew, NOT raw bid/ask prices.
/// Won 24/30 days vs pure AS on Sharpe."
/// ISAC: "1D action (gamma) with 36% inventory reduction."
///
/// IntensityAction dropped — sizing is the risk manager's job, not RL's.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ParameterAction {
    /// Risk aversion multiplier
    pub gamma: GammaAction,
    /// Inventory skew multiplier
    pub omega: OmegaAction,
}

impl ParameterAction {
    /// Create a new parameter action.
    pub fn new(gamma: GammaAction, omega: OmegaAction) -> Self {
        Self { gamma, omega }
    }

    /// Create from flat index.
    pub fn from_index(idx: usize) -> Self {
        let omega_idx = idx % OmegaAction::COUNT;
        let gamma_idx = idx / OmegaAction::COUNT;

        Self {
            gamma: GammaAction::from_index(gamma_idx),
            omega: OmegaAction::from_index(omega_idx),
        }
    }

    /// Convert to flat index.
    pub fn to_index(&self) -> usize {
        self.gamma.index() * OmegaAction::COUNT + self.omega.index()
    }

    /// Get γ multiplier.
    pub fn gamma_multiplier(&self) -> f64 {
        self.gamma.multiplier()
    }

    /// Get ω multiplier.
    pub fn omega_multiplier(&self) -> f64 {
        self.omega.multiplier()
    }

    /// Total number of parameter actions (5 × 5 = 25).
    pub const ACTION_COUNT: usize = GammaAction::COUNT * OmegaAction::COUNT;

    /// Default neutral action (no changes to base parameters).
    pub fn neutral() -> Self {
        Self {
            gamma: GammaAction::Neutral,
            omega: OmegaAction::Neutral,
        }
    }

    /// Defensive action (wider spreads, strong skew).
    pub fn defensive() -> Self {
        Self {
            gamma: GammaAction::Defensive,
            omega: OmegaAction::StrongSkew,
        }
    }
}

impl Default for ParameterAction {
    fn default() -> Self {
        Self::neutral()
    }
}

/// Complete action for the MDP.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MDPAction {
    /// Spread adjustment
    pub spread: SpreadAction,
    /// Skew adjustment
    pub skew: SkewAction,
}

impl MDPAction {
    /// Create a new action.
    pub fn new(spread: SpreadAction, skew: SkewAction) -> Self {
        Self { spread, skew }
    }

    /// Convert to flat action index.
    /// Total actions = 5 * 5 = 25
    pub fn to_index(&self) -> usize {
        self.spread.index() * SkewAction::COUNT + self.skew.index()
    }

    /// Create from flat index.
    pub fn from_index(idx: usize) -> Self {
        let spread_idx = idx / SkewAction::COUNT;
        let skew_idx = idx % SkewAction::COUNT;
        Self {
            spread: SpreadAction::from_index(spread_idx),
            skew: SkewAction::from_index(skew_idx),
        }
    }

    /// Total number of actions.
    pub const ACTION_COUNT: usize = SpreadAction::COUNT * SkewAction::COUNT;

    /// Get the bid spread delta in bps.
    pub fn bid_delta_bps(&self) -> f64 {
        self.spread.delta_bps() + self.skew.bid_skew_bps()
    }

    /// Get the ask spread delta in bps.
    pub fn ask_delta_bps(&self) -> f64 {
        self.spread.delta_bps() + self.skew.ask_skew_bps()
    }
}

impl Default for MDPAction {
    fn default() -> Self {
        Self {
            spread: SpreadAction::Maintain,
            skew: SkewAction::Symmetric,
        }
    }
}

// ============================================================================
// Reward Function
// ============================================================================

/// Configuration for reward computation.
#[derive(Debug, Clone)]
pub struct RewardConfig {
    /// Weight for realized edge component
    pub edge_weight: f64,
    /// Weight for inventory risk penalty
    pub inventory_penalty_weight: f64,
    /// Weight for volatility penalty
    pub volatility_penalty_weight: f64,
    /// Weight for inventory change penalty (penalizes accumulation, not just level)
    pub inventory_change_weight: f64,
    /// Discount factor for future rewards
    pub gamma: f64,
}

impl Default for RewardConfig {
    fn default() -> Self {
        Self {
            edge_weight: 1.0,
            inventory_penalty_weight: 0.1,
            volatility_penalty_weight: 0.05,
            inventory_change_weight: 0.05,
            gamma: 0.95,
        }
    }
}

impl RewardConfig {
    /// Compute a deterministic hash of all reward config fields.
    /// Used to detect incompatible reward config changes across checkpoint save/restore.
    pub fn config_hash(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        self.edge_weight.to_bits().hash(&mut hasher);
        self.inventory_penalty_weight.to_bits().hash(&mut hasher);
        self.volatility_penalty_weight.to_bits().hash(&mut hasher);
        self.inventory_change_weight.to_bits().hash(&mut hasher);
        self.gamma.to_bits().hash(&mut hasher);
        hasher.finish()
    }
}

/// Reward signal from a transition.
///
/// Literature (Falces Marin 2022): reward = spread_capture - realized_AS - fees.
/// No separate adverse penalty — AS cost is already embedded in realized_edge_bps
/// when computed as `depth_bps - as_realized_bps - fee_bps`.
#[derive(Debug, Clone, Copy)]
pub struct Reward {
    /// Total reward
    pub total: f64,
    /// Edge component (can be negative for adverse fills)
    pub edge_component: f64,
    /// Inventory risk penalty (always non-positive)
    pub inventory_penalty: f64,
    /// Volatility penalty (always non-positive)
    pub volatility_penalty: f64,
    /// Inventory change penalty (penalizes accumulation, always non-positive)
    pub inventory_change_penalty: f64,
    /// Drift opposition penalty (penalizes holding against momentum, always non-positive)
    pub drift_penalty: f64,
}

impl Reward {
    /// Compute reward from transition.
    ///
    /// `realized_edge_bps` should be `spread_capture - AS_cost - fees` (P0-2).
    /// `prev_inventory_risk` is `|prev_position| / max_position` from the state
    /// at the time the action was chosen (available from pending state-action queue).
    /// `momentum_bps` and `drift_penalty_weight` control the drift opposition penalty.
    /// `position` is the signed inventory (needed for drift alignment check).
    pub fn compute(
        config: &RewardConfig,
        realized_edge_bps: f64,
        inventory_risk: f64, // |position| / max_position (current)
        vol_ratio: f64,
        prev_inventory_risk: f64, // |prev_position| / max_position
    ) -> Self {
        Self::compute_with_drift(
            config,
            realized_edge_bps,
            inventory_risk,
            vol_ratio,
            prev_inventory_risk,
            0.0,
            0.0,
            0.0,
        )
    }

    /// Compute reward with drift opposition penalty.
    ///
    /// `position` is the signed inventory (for drift alignment check).
    /// `momentum_bps` is the current price drift in bps.
    /// `drift_penalty_weight` controls the magnitude of the penalty.
    #[allow(clippy::too_many_arguments)]
    pub fn compute_with_drift(
        config: &RewardConfig,
        realized_edge_bps: f64,
        inventory_risk: f64, // |position| / max_position (current)
        vol_ratio: f64,
        prev_inventory_risk: f64, // |prev_position| / max_position
        position: f64,            // signed inventory
        momentum_bps: f64,
        drift_penalty_weight: f64,
    ) -> Self {
        let edge_component = config.edge_weight * realized_edge_bps;

        // Quadratic inventory penalty (penalizes level)
        let inventory_penalty = -config.inventory_penalty_weight * inventory_risk.powi(2) * 10.0;

        // Volatility penalty (penalize holding in high vol)
        let vol_penalty_factor = (vol_ratio - 1.0).max(0.0);
        let volatility_penalty =
            -config.volatility_penalty_weight * vol_penalty_factor * inventory_risk * 5.0;

        // Inventory change penalty (penalizes accumulation, not just level)
        // An agent that increases inventory from 50% to 60% is penalized more
        // than one maintaining 60%.
        let inventory_change_penalty =
            -config.inventory_change_weight * (inventory_risk - prev_inventory_risk).abs() * 10.0;

        // Drift opposition penalty: penalize holding inventory against price drift
        let drift_penalty = if drift_penalty_weight > 0.0 && position * momentum_bps < 0.0 {
            // Position opposes drift direction
            let alignment = (position * momentum_bps).abs().min(1.0);
            -drift_penalty_weight * alignment * inventory_risk * 5.0
        } else {
            0.0
        };

        let total = edge_component
            + inventory_penalty
            + volatility_penalty
            + inventory_change_penalty
            + drift_penalty;

        Self {
            total,
            edge_component,
            inventory_penalty,
            volatility_penalty,
            inventory_change_penalty,
            drift_penalty,
        }
    }
}

// ============================================================================
// Bayesian Q-Value with Thompson Sampling
// ============================================================================

/// Bayesian posterior for a Q-value (Normal-Gamma conjugate prior).
///
/// Models Q(s, a) ~ Normal(μ, 1/τ) where:
/// - μ | τ ~ Normal(μ₀, 1/(κ₀τ))
/// - τ ~ Gamma(α, β)
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
pub struct BayesianQValue {
    /// Prior mean
    mu_0: f64,
    /// Prior precision scale
    kappa_0: f64,
    /// Posterior mean (updated)
    mu_n: f64,
    /// Posterior precision scale
    kappa_n: f64,
    /// Gamma shape parameter
    alpha: f64,
    /// Gamma rate parameter
    beta: f64,
    /// Number of observations
    n: u64,
}

impl BayesianQValue {
    /// Create with prior centered at zero.
    pub fn new() -> Self {
        Self {
            mu_0: 0.0,
            kappa_0: 0.01, // Weak prior on mean
            mu_n: 0.0,
            kappa_n: 0.01,
            alpha: 1.0, // Weak prior on precision
            beta: 1.0,
            n: 0,
        }
    }

    /// Create with specified prior mean (for warm start).
    pub fn with_prior(prior_mean: f64, prior_strength: f64) -> Self {
        Self {
            mu_0: prior_mean,
            kappa_0: prior_strength,
            mu_n: prior_mean,
            kappa_n: prior_strength,
            alpha: 1.0,
            beta: 1.0,
            n: 0,
        }
    }

    /// Update posterior with observed reward.
    pub fn update(&mut self, reward: f64) {
        self.n += 1;

        // Update sufficient statistics
        let kappa_new = self.kappa_n + 1.0;
        let mu_new = (self.kappa_n * self.mu_n + reward) / kappa_new;

        // Update Gamma parameters
        let alpha_new = self.alpha + 0.5;
        let beta_new = self.beta + 0.5 * (reward - self.mu_n).powi(2) * self.kappa_n / kappa_new;

        self.mu_n = mu_new;
        self.kappa_n = kappa_new;
        self.alpha = alpha_new;
        self.beta = beta_new;
    }

    /// Get posterior mean of Q-value.
    pub fn mean(&self) -> f64 {
        self.mu_n
    }

    /// Get posterior variance of Q-value.
    pub fn variance(&self) -> f64 {
        if self.alpha <= 1.0 {
            return 100.0; // High variance for weak prior
        }
        self.beta / ((self.alpha - 1.0) * self.kappa_n)
    }

    /// Get posterior standard deviation.
    pub fn std(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Sample from posterior (Thompson sampling).
    pub fn sample(&self) -> f64 {
        // Sample precision from Gamma
        let tau = sample_gamma(self.alpha, self.beta);
        // Sample mean from Normal
        let sigma = (1.0 / (self.kappa_n * tau)).sqrt();
        self.mu_n + sigma * sample_standard_normal()
    }

    /// Get upper confidence bound (UCB).
    pub fn ucb(&self, c: f64) -> f64 {
        self.mean() + c * self.std()
    }

    /// Number of observations.
    pub fn count(&self) -> u64 {
        self.n
    }

    /// Create a new Q-value using `other`'s posterior as a discounted prior.
    ///
    /// Used for sim-to-real transfer: paper trading Q-values become informative
    /// but down-weighted priors for live trading. `weight` in [0, 1] controls
    /// how much to trust the paper posterior (0.3 = paper counts as 30% of real).
    pub fn with_discounted_prior(other: &BayesianQValue, weight: f64) -> BayesianQValue {
        BayesianQValue {
            mu_0: other.mu_n,
            kappa_0: weight * other.kappa_n,
            mu_n: other.mu_n,
            kappa_n: weight * other.kappa_n,
            alpha: 1.0 + weight * (other.alpha - 1.0),
            beta: other.beta * weight,
            n: 0, // No real observations yet
        }
    }

    /// Posterior precision scale (for checkpoint persistence).
    pub fn kappa_n(&self) -> f64 {
        self.kappa_n
    }

    /// Gamma shape parameter (for checkpoint persistence).
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Gamma rate parameter (for checkpoint persistence).
    pub fn beta(&self) -> f64 {
        self.beta
    }

    /// Restore a Q-value from checkpoint data.
    ///
    /// Faithfully restores the full posterior state (mu_n, kappa_n, alpha, beta, n)
    /// so that learning continues from where it left off.
    pub fn from_checkpoint(mu_n: f64, kappa_n: f64, alpha: f64, beta: f64, n: u64) -> Self {
        Self {
            mu_0: mu_n, // Use posterior as prior for continued learning
            kappa_0: kappa_n,
            mu_n,
            kappa_n,
            alpha,
            beta,
            n,
        }
    }
}

impl Default for BayesianQValue {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Q-Learning Agent
// ============================================================================

/// Configuration for the Q-learning agent.
#[derive(Debug, Clone)]
pub struct QLearningConfig {
    /// Learning rate for TD updates
    pub learning_rate: f64,
    /// Discount factor
    pub gamma: f64,
    /// Exploration strategy
    pub exploration: ExplorationStrategy,
    /// Minimum observations before exploitation
    pub min_observations: u64,
    /// UCB exploration constant (if using UCB)
    pub ucb_c: f64,
    /// Reward configuration
    pub reward_config: RewardConfig,
    /// Minimum exploration rate for anti-ossification (default 0.05).
    /// Even with Thompson sampling, mix in uniform random actions at this rate.
    /// Zheng & Ding (2024): "epsilon should decay to small nonzero baseline, not zero."
    pub min_exploration_rate: f64,
    /// Use compact 3D state space (45 states) instead of 5D (675 states).
    /// Inventory(5) x Volatility(3) x Imbalance(3) = 45.
    /// ISAC shows even 1D state achieves 36% inventory reduction.
    pub use_compact_state: bool,
    /// Use parameter-based actions (GammaAction x OmegaAction = 25) instead of
    /// BPS delta actions (SpreadAction x SkewAction = 25).
    /// Falces Marin (2022): "RL controls gamma and skew, NOT raw bid/ask prices."
    pub use_parameter_actions: bool,
    /// Enable drift/momentum dimension in MDP state space.
    /// When false, drift is always DriftBucket::Neutral (no state expansion).
    /// Default false for safe rollout.
    pub use_drift_bucket: bool,
    /// Weight for drift opposition penalty in reward computation.
    /// Penalizes holding inventory against price drift direction.
    /// Only active when > 0.0. Default 0.3.
    pub drift_penalty_weight: f64,
}

impl Default for QLearningConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.1,
            gamma: 0.95,
            exploration: ExplorationStrategy::ThompsonSampling,
            min_observations: 10,
            ucb_c: 2.0,
            reward_config: RewardConfig::default(),
            min_exploration_rate: 0.05,
            use_compact_state: true,
            use_parameter_actions: true,
            use_drift_bucket: false,
            drift_penalty_weight: 0.3,
        }
    }
}

/// Exploration strategy for action selection.
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(dead_code)]
pub enum ExplorationStrategy {
    /// Thompson sampling on Bayesian Q-values
    ThompsonSampling,
    /// Upper Confidence Bound
    UCB,
    /// Epsilon-greedy with decay
    EpsilonGreedy { epsilon: f64, decay: f64 },
}

/// Configuration for sim-to-real transfer of RL Q-tables.
///
/// Controls how paper trading experience is incorporated as a prior
/// when transitioning to live trading.
#[derive(Debug, Clone)]
pub struct SimToRealConfig {
    /// Paper fill counts as this fraction of a real fill (0.3 = paper is 30% weight)
    pub paper_prior_weight: f64,
    /// Minimum real fills before RL controls actions (observation-only before this)
    pub min_real_fills: usize,
    /// Clip actions within this many sigma of paper mean
    pub action_bound_sigma: f64,
    /// Auto-disable RL if mean reward < 0 after this many fills
    pub auto_disable_after_fills: usize,
    /// Reward threshold (bps) below which RL auto-disables.
    /// Default -1.5 bps (maker fee) so fee drag alone does not trigger disable.
    pub auto_disable_threshold_bps: f64,
}

impl Default for SimToRealConfig {
    fn default() -> Self {
        Self {
            paper_prior_weight: 0.3,
            min_real_fills: 20,
            action_bound_sigma: 1.5,
            auto_disable_after_fills: 100,
            auto_disable_threshold_bps: -1.5,
        }
    }
}

/// Unified state index that can come from either MDPState or MDPStateCompact.
/// Stored as a flat index for Q-table lookup.
pub type StateIndex = usize;

/// Unified action index that can come from either MDPAction or ParameterAction.
/// Both have 25 actions, so the index range is 0..24.
pub type ActionIndex = usize;

/// Number of actions (same for both MDPAction and ParameterAction = 25).
pub const UNIFIED_ACTION_COUNT: usize = 25;

/// Q-learning agent with Bayesian Q-values.
///
/// The agent operates on state/action indices internally. The caller is responsible
/// for converting between concrete types (MDPState vs MDPStateCompact, MDPAction vs
/// ParameterAction) and indices using the config flags.
#[derive(Debug)]
pub struct QLearningAgent {
    /// Configuration
    config: QLearningConfig,
    /// Sim-to-real transfer configuration
    sim_to_real_config: SimToRealConfig,
    /// Q-table: state_index -> [action Q-values]
    q_table: HashMap<usize, Vec<BayesianQValue>>,
    /// Episode count
    episodes: u64,
    /// Total reward accumulated
    total_reward: f64,
    /// Recent rewards for monitoring (VecDeque for O(1) pop_front)
    recent_rewards: VecDeque<f64>,
    /// FIFO queue of pending (state_index, action_index) pairs awaiting reward updates.
    /// Supports clustered fills where multiple quotes are outstanding.
    pending_state_actions: VecDeque<(StateIndex, ActionIndex)>,
    /// Parallel queue storing inventory_risk at the time each action was chosen.
    /// Used by `Reward::compute()` to correctly compute the inventory change penalty
    /// instead of approximating prev_inventory_risk with the current value.
    pending_inventory_risks: VecDeque<f64>,
    /// EWMA baseline tracker for counterfactual reward centering.
    /// QLearningAgent is deprecated (replaced by SpreadBandit in stochastic module).
    /// Production baseline tracker lives at stochastic.baseline_tracker, wired in
    /// handlers.rs:1050-1055. This field is dead code since QL agent is unused.
    #[allow(dead_code)]
    baseline: BaselineTracker,
}

impl QLearningAgent {
    /// Create a new Q-learning agent.
    pub fn new(config: QLearningConfig) -> Self {
        Self {
            config,
            sim_to_real_config: SimToRealConfig::default(),
            q_table: HashMap::new(),
            episodes: 0,
            total_reward: 0.0,
            recent_rewards: VecDeque::with_capacity(1000),
            pending_state_actions: VecDeque::with_capacity(STATE_ACTION_QUEUE_CAPACITY),
            pending_inventory_risks: VecDeque::with_capacity(STATE_ACTION_QUEUE_CAPACITY),
            baseline: BaselineTracker::default(),
        }
    }

    /// Create a new Q-learning agent with sim-to-real configuration.
    pub fn with_sim_to_real(config: QLearningConfig, sim_to_real_config: SimToRealConfig) -> Self {
        Self {
            config,
            sim_to_real_config,
            q_table: HashMap::new(),
            episodes: 0,
            total_reward: 0.0,
            recent_rewards: VecDeque::with_capacity(1000),
            pending_state_actions: VecDeque::with_capacity(STATE_ACTION_QUEUE_CAPACITY),
            pending_inventory_risks: VecDeque::with_capacity(STATE_ACTION_QUEUE_CAPACITY),
            baseline: BaselineTracker::default(),
        }
    }

    /// Whether to use compact 3D state space.
    pub fn use_compact_state(&self) -> bool {
        self.config.use_compact_state
    }

    /// Whether to use parameter-based actions.
    pub fn use_parameter_actions(&self) -> bool {
        self.config.use_parameter_actions
    }

    /// Get Q-values for a state index (initialize if needed).
    fn get_q_values_by_idx(&mut self, state_idx: usize) -> &mut Vec<BayesianQValue> {
        self.q_table
            .entry(state_idx)
            .or_insert_with(|| vec![BayesianQValue::new(); UNIFIED_ACTION_COUNT])
    }

    /// Select action index using the configured exploration strategy.
    ///
    /// Returns a flat action index (0..24). Caller converts to MDPAction or ParameterAction.
    /// Enforces SimToRealConfig safety guards:
    /// - Returns neutral action if insufficient real fills
    /// - Clips actions within action_bound_sigma of paper mean
    /// - Auto-disables RL on persistent negative reward
    pub fn select_action_idx(&mut self, state_idx: StateIndex) -> ActionIndex {
        let neutral_idx = ParameterAction::neutral().to_index();

        // Guard 1: Not enough real fills — observation only, return neutral
        let total_updates = self.total_updates();
        if total_updates < self.sim_to_real_config.min_real_fills as u64 {
            return neutral_idx;
        }

        // Guard 2: Auto-disable on persistent negative reward (below threshold, not just < 0)
        let threshold = self.sim_to_real_config.auto_disable_threshold_bps;
        if total_updates > self.sim_to_real_config.auto_disable_after_fills as u64
            && self.mean_recent_reward() < threshold
        {
            debug!(
                mean_reward = %format!("{:.3}", self.mean_recent_reward()),
                threshold_bps = %format!("{:.1}", threshold),
                "RL auto-disabled: mean reward below threshold"
            );
            return neutral_idx;
        }

        // Copy config values to avoid borrow issues
        let min_observations = self.config.min_observations;
        let exploration = self.config.exploration;
        let ucb_c = self.config.ucb_c;
        let episodes = self.episodes;
        let min_exploration_rate = self.config.min_exploration_rate;
        let action_bound_sigma = self.sim_to_real_config.action_bound_sigma;
        let use_parameter_actions = self.config.use_parameter_actions;

        let q_values = self.get_q_values_by_idx(state_idx);

        // Check if we have enough observations for exploitation
        let total_obs: u64 = q_values.iter().map(|q| q.count()).sum();
        if total_obs < min_observations {
            // Pure exploration: uniform random
            let action_idx = (sample_uniform() * UNIFIED_ACTION_COUNT as f64) as usize;
            return action_idx.min(UNIFIED_ACTION_COUNT - 1);
        }

        // Anti-ossification: mix in uniform random action at min_exploration_rate
        if min_exploration_rate > 0.0 && sample_uniform() < min_exploration_rate {
            let action_idx = (sample_uniform() * UNIFIED_ACTION_COUNT as f64) as usize;
            return action_idx.min(UNIFIED_ACTION_COUNT - 1);
        }

        let action_idx = match exploration {
            ExplorationStrategy::ThompsonSampling => q_values
                .iter()
                .enumerate()
                .map(|(i, q)| (i, q.sample()))
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0),
            ExplorationStrategy::UCB => q_values
                .iter()
                .enumerate()
                .map(|(i, q)| (i, q.ucb(ucb_c)))
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0),
            ExplorationStrategy::EpsilonGreedy { epsilon, decay } => {
                let effective_epsilon = epsilon * decay.powf(episodes as f64);
                if sample_uniform() < effective_epsilon {
                    (sample_uniform() * UNIFIED_ACTION_COUNT as f64) as usize
                } else {
                    q_values
                        .iter()
                        .enumerate()
                        .map(|(i, q)| (i, q.mean()))
                        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                        .map(|(i, _)| i)
                        .unwrap_or(0)
                }
            }
        };

        let action_idx = action_idx.min(UNIFIED_ACTION_COUNT - 1);

        // Guard 3: Clip parameter actions within action_bound_sigma of neutral
        // For parameter actions, ensure gamma/omega indices stay within bounds
        if use_parameter_actions && action_bound_sigma > 0.0 {
            let selected = ParameterAction::from_index(action_idx);
            let neutral = ParameterAction::neutral();

            let gamma_dist = (selected.gamma.index() as f64 - neutral.gamma.index() as f64).abs();
            let omega_dist = (selected.omega.index() as f64 - neutral.omega.index() as f64).abs();

            if gamma_dist > action_bound_sigma || omega_dist > action_bound_sigma {
                // Clip to nearest action within bounds
                let clamp = |idx: usize, center: usize| -> usize {
                    let low = (center as f64 - action_bound_sigma).round().max(0.0) as usize;
                    let high = (center as f64 + action_bound_sigma).round().min(4.0) as usize;
                    idx.clamp(low, high)
                };
                let clamped_gamma =
                    GammaAction::from_index(clamp(selected.gamma.index(), neutral.gamma.index()));
                let clamped_omega =
                    OmegaAction::from_index(clamp(selected.omega.index(), neutral.omega.index()));
                return ParameterAction::new(clamped_gamma, clamped_omega).to_index();
            }
        }

        action_idx
    }

    /// Legacy: Select action using the configured exploration strategy.
    /// Returns an MDPAction (for backward compatibility).
    pub fn select_action(&mut self, state: &MDPState) -> MDPAction {
        let action_idx = self.select_action_idx(state.to_index());
        MDPAction::from_index(action_idx)
    }

    /// Update Q-values with observed transition (index-based).
    pub fn update_idx(
        &mut self,
        state_idx: StateIndex,
        action_idx: ActionIndex,
        reward: Reward,
        next_state_idx: StateIndex,
        done: bool,
    ) {
        let gamma = self.config.gamma;

        // Get max Q-value for next state
        let max_next_q = if done {
            0.0
        } else {
            let next_q_values = self.get_q_values_by_idx(next_state_idx);
            next_q_values
                .iter()
                .map(|q| q.mean())
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(0.0)
        };

        // TD target
        let td_target = reward.total + gamma * max_next_q;

        // Update Q-value with Bayesian posterior update
        let q_values = self.get_q_values_by_idx(state_idx);
        if action_idx < q_values.len() {
            q_values[action_idx].update(td_target);
        }
        let q_mean = q_values.get(action_idx).map(|q| q.mean()).unwrap_or(0.0);
        let q_std = q_values.get(action_idx).map(|q| q.std()).unwrap_or(0.0);

        // Track rewards
        self.total_reward += reward.total;
        self.recent_rewards.push_back(reward.total);
        if self.recent_rewards.len() > 1000 {
            self.recent_rewards.pop_front();
        }

        debug!(
            state_idx = state_idx,
            action_idx = action_idx,
            reward = %format!("{:.3}", reward.total),
            td_target = %format!("{:.3}", td_target),
            q_mean = %format!("{:.3}", q_mean),
            q_std = %format!("{:.3}", q_std),
            "Q-learning update"
        );
    }

    /// Legacy: Update Q-values with observed transition using concrete types.
    pub fn update(
        &mut self,
        state: MDPState,
        action: MDPAction,
        reward: Reward,
        next_state: MDPState,
        done: bool,
    ) {
        self.update_idx(
            state.to_index(),
            action.to_index(),
            reward,
            next_state.to_index(),
            done,
        );
    }

    /// Record start of a new episode.
    pub fn start_episode(&mut self) {
        self.episodes += 1;
        self.pending_state_actions.clear();
        self.pending_inventory_risks.clear();
    }

    /// Get the greedy action index (exploitation only).
    pub fn get_greedy_action_idx(&mut self, state_idx: StateIndex) -> ActionIndex {
        let q_values = self.get_q_values_by_idx(state_idx);
        q_values
            .iter()
            .enumerate()
            .map(|(i, q)| (i, q.mean()))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Legacy: Get the greedy action (exploitation only).
    pub fn get_greedy_action(&mut self, state: &MDPState) -> MDPAction {
        let action_idx = self.get_greedy_action_idx(state.to_index());
        MDPAction::from_index(action_idx)
    }

    /// Get Q-value statistics for a state index.
    pub fn get_q_stats_idx(&mut self, state_idx: StateIndex) -> QValueStats {
        let q_values = self.get_q_values_by_idx(state_idx);
        let best_idx = q_values
            .iter()
            .enumerate()
            .map(|(i, q)| (i, q.mean()))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        let best_q = &q_values[best_idx];

        QValueStats {
            best_action_idx: best_idx,
            best_q_mean: best_q.mean(),
            best_q_std: best_q.std(),
            best_q_count: best_q.count(),
            total_observations: q_values.iter().map(|q| q.count()).sum(),
        }
    }

    /// Legacy: Get Q-value statistics for a state.
    pub fn get_q_stats(&mut self, state: &MDPState) -> QValueStats {
        self.get_q_stats_idx(state.to_index())
    }

    /// Get summary statistics.
    pub fn summary(&self) -> AgentSummary {
        let recent_reward = if self.recent_rewards.is_empty() {
            0.0
        } else {
            self.recent_rewards.iter().sum::<f64>() / self.recent_rewards.len() as f64
        };

        AgentSummary {
            episodes: self.episodes,
            total_reward: self.total_reward,
            recent_avg_reward: recent_reward,
            states_visited: self.q_table.len(),
        }
    }

    /// Get the reward configuration.
    pub fn reward_config(&self) -> &RewardConfig {
        &self.config.reward_config
    }

    /// Get the full agent configuration.
    pub fn config(&self) -> &QLearningConfig {
        &self.config
    }

    /// Push a state-action pair onto the pending queue (FIFO) using indices.
    pub fn push_state_action_idx(&mut self, state_idx: StateIndex, action_idx: ActionIndex) {
        if self.pending_state_actions.len() >= STATE_ACTION_QUEUE_CAPACITY {
            self.pending_state_actions.pop_front();
            self.pending_inventory_risks.pop_front();
        }
        self.pending_state_actions
            .push_back((state_idx, action_idx));
        // Default to 0.0; callers should use push_state_action_with_risk for accurate prev_inventory_risk
        self.pending_inventory_risks.push_back(0.0);
    }

    /// Push a state-action pair with the associated inventory risk at action time.
    ///
    /// `inventory_risk` is `|position| / max_position` at the time the action was chosen.
    /// Stored so `Reward::compute()` can use the actual previous inventory risk
    /// instead of approximating with the current value.
    pub fn push_state_action_with_risk(
        &mut self,
        state_idx: StateIndex,
        action_idx: ActionIndex,
        inventory_risk: f64,
    ) {
        if self.pending_state_actions.len() >= STATE_ACTION_QUEUE_CAPACITY {
            self.pending_state_actions.pop_front();
            self.pending_inventory_risks.pop_front();
        }
        self.pending_state_actions
            .push_back((state_idx, action_idx));
        self.pending_inventory_risks.push_back(inventory_risk);
    }

    /// Legacy: Push a state-action pair onto the pending queue (FIFO).
    pub fn push_state_action(&mut self, state: MDPState, action: MDPAction) {
        self.push_state_action_idx(state.to_index(), action.to_index());
    }

    /// Pop the next (oldest) pending state-action pair for reward update.
    ///
    /// Returns `(state_idx, action_idx)`.
    pub fn take_next_state_action(&mut self) -> Option<(StateIndex, ActionIndex)> {
        self.pending_state_actions.pop_front()
    }

    /// Pop the inventory risk associated with the next pending state-action.
    ///
    /// Must be called immediately after `take_next_state_action()` to stay in sync.
    /// Returns the inventory risk at the time the action was chosen, for use
    /// as `prev_inventory_risk` in `Reward::compute()`.
    pub fn take_next_inventory_risk(&mut self) -> f64 {
        self.pending_inventory_risks.pop_front().unwrap_or(0.0)
    }

    /// Number of pending state-action pairs awaiting reward updates.
    pub fn pending_action_count(&self) -> usize {
        self.pending_state_actions.len()
    }

    /// Record the last state-action pair (for delayed reward updates).
    /// Prefer `push_state_action()` for new code — this delegates to it.
    pub fn set_last_state_action(&mut self, state: MDPState, action: MDPAction) {
        self.push_state_action(state, action);
    }

    /// Get and clear the oldest pending state-action pair.
    /// Prefer `take_next_state_action()` for new code — this delegates to it.
    pub fn take_last_state_action(&mut self) -> Option<(StateIndex, ActionIndex)> {
        self.take_next_state_action()
    }

    /// Export the Q-table for checkpoint persistence or sim-to-real transfer.
    pub fn export_q_table(&self) -> HashMap<usize, Vec<BayesianQValue>> {
        self.q_table.clone()
    }

    /// Import a paper trading Q-table as a discounted prior for live trading.
    ///
    /// Only overwrites Q-values for cold-start states (n == 0). States with
    /// live observations (n > 0) are preserved — real experience is more
    /// valuable than simulated experience.
    pub fn import_q_table_as_prior(
        &mut self,
        paper_q: &HashMap<usize, Vec<BayesianQValue>>,
        weight: f64,
    ) {
        for (&state_idx, paper_values) in paper_q {
            let live_values = self
                .q_table
                .entry(state_idx)
                .or_insert_with(|| vec![BayesianQValue::new(); UNIFIED_ACTION_COUNT]);
            for (i, paper_qv) in paper_values.iter().enumerate() {
                if i < live_values.len() && live_values[i].count() == 0 {
                    // Only import paper prior for cold-start states (no live data)
                    live_values[i] = BayesianQValue::with_discounted_prior(paper_qv, weight);
                }
                // States with live data (n > 0) are preserved untouched
            }
        }
    }

    /// Total number of Bayesian updates across all states and actions.
    pub fn total_updates(&self) -> u64 {
        self.q_table
            .values()
            .flat_map(|actions| actions.iter())
            .map(|q| q.count())
            .sum()
    }

    /// Average of recent rewards, or 0.0 if no rewards recorded.
    pub fn mean_recent_reward(&self) -> f64 {
        if self.recent_rewards.is_empty() {
            0.0
        } else {
            self.recent_rewards.iter().sum::<f64>() / self.recent_rewards.len() as f64
        }
    }

    /// Serialize Q-table and agent state to checkpoint.
    pub fn to_checkpoint(&self) -> RLCheckpoint {
        let mut entries = Vec::new();
        let mut total_observations: u64 = 0;

        for (&state_idx, actions) in &self.q_table {
            for (action_idx, q_val) in actions.iter().enumerate() {
                if q_val.count() > 0 {
                    entries.push(QTableEntry {
                        state_index: state_idx,
                        action_index: action_idx,
                        mu_n: q_val.mean(),
                        kappa_n: q_val.kappa_n(),
                        alpha: q_val.alpha(),
                        beta: q_val.beta(),
                        n: q_val.count(),
                    });
                    total_observations += q_val.count();
                }
            }
        }

        RLCheckpoint {
            q_entries: entries,
            episodes: self.episodes,
            total_reward: self.total_reward,
            total_observations,
            action_space_version: if self.config.use_parameter_actions {
                2
            } else {
                1
            },
            use_compact_state: self.config.use_compact_state,
            reward_config_hash: self.config.reward_config.config_hash(),
            use_drift_bucket: self.config.use_drift_bucket,
        }
    }

    /// Restore Q-table and agent state from checkpoint.
    pub fn restore_from_checkpoint(&mut self, ckpt: &RLCheckpoint) {
        // Check action space version compatibility
        let expected_version = if self.config.use_parameter_actions {
            2
        } else {
            1
        };
        if ckpt.action_space_version != 0 && ckpt.action_space_version != expected_version {
            debug!(
                checkpoint_version = ckpt.action_space_version,
                expected_version = expected_version,
                "Action space version mismatch — starting fresh Q-table"
            );
            // Don't restore incompatible Q-table, but keep episode count
            self.episodes = ckpt.episodes;
            self.total_reward = ckpt.total_reward;
            return;
        }

        // Check state space compatibility
        if ckpt.use_compact_state != self.config.use_compact_state {
            debug!(
                checkpoint_compact = ckpt.use_compact_state,
                config_compact = self.config.use_compact_state,
                "State space mismatch — starting fresh Q-table"
            );
            self.episodes = ckpt.episodes;
            self.total_reward = ckpt.total_reward;
            return;
        }

        // Check drift bucket compatibility (675-state vs 2025-state)
        if ckpt.use_drift_bucket != self.config.use_drift_bucket {
            debug!(
                checkpoint_drift = ckpt.use_drift_bucket,
                config_drift = self.config.use_drift_bucket,
                "Drift bucket mismatch — starting fresh Q-table"
            );
            self.episodes = ckpt.episodes;
            self.total_reward = ckpt.total_reward;
            return;
        }

        // Check reward config compatibility (warn only, don't hard fail)
        let current_hash = self.config.reward_config.config_hash();
        if ckpt.reward_config_hash != 0 && ckpt.reward_config_hash != current_hash {
            eprintln!(
                "[RL] WARNING: RewardConfig changed since checkpoint was saved \
                 (checkpoint hash={}, current hash={}). \
                 Q-values may be stale — consider resetting.",
                ckpt.reward_config_hash, current_hash
            );
        }

        self.episodes = ckpt.episodes;
        self.total_reward = ckpt.total_reward;

        // Clear stale reward history so previous session doesn't poison auto-disable
        self.recent_rewards.clear();

        // Clear and rebuild q_table from checkpoint entries
        self.q_table.clear();
        for entry in &ckpt.q_entries {
            let actions = self
                .q_table
                .entry(entry.state_index)
                .or_insert_with(|| vec![BayesianQValue::new(); UNIFIED_ACTION_COUNT]);
            if entry.action_index < actions.len() {
                actions[entry.action_index] = BayesianQValue::from_checkpoint(
                    entry.mu_n,
                    entry.kappa_n,
                    entry.alpha,
                    entry.beta,
                    entry.n,
                );
            }
        }

        debug!(
            episodes = ckpt.episodes,
            total_reward = ckpt.total_reward,
            entries = ckpt.q_entries.len(),
            total_observations = ckpt.total_observations,
            "RL agent restored from checkpoint"
        );
    }
}

impl Default for QLearningAgent {
    fn default() -> Self {
        Self::new(QLearningConfig::default())
    }
}

impl QLearningAgent {
    /// Get the sim-to-real configuration.
    pub fn sim_to_real_config(&self) -> &SimToRealConfig {
        &self.sim_to_real_config
    }
}

/// Q-value statistics for a state.
#[derive(Debug, Clone)]
pub struct QValueStats {
    /// Best action index according to posterior mean
    pub best_action_idx: ActionIndex,
    /// Posterior mean of best Q-value
    pub best_q_mean: f64,
    /// Posterior std of best Q-value
    pub best_q_std: f64,
    /// Observations for best action
    pub best_q_count: u64,
    /// Total observations for this state
    pub total_observations: u64,
}

/// Agent summary statistics.
#[derive(Debug, Clone)]
pub struct AgentSummary {
    /// Total episodes
    pub episodes: u64,
    /// Total cumulative reward
    pub total_reward: f64,
    /// Recent average reward
    pub recent_avg_reward: f64,
    /// Number of unique states visited
    pub states_visited: usize,
}

// ============================================================================
// RL Policy Recommendation
// ============================================================================

/// Policy recommendation from the RL agent.
///
/// Supports both legacy BPS-delta mode and new parameter-multiplier mode.
/// When `use_parameter_actions` is true, `gamma_multiplier` and `omega_multiplier`
/// are populated instead of `spread_delta_bps`/skew fields.
#[derive(Debug, Clone)]
pub struct RLPolicyRecommendation {
    /// Recommended spread delta (bps) — legacy BPS-delta mode only
    pub spread_delta_bps: f64,
    /// Recommended bid skew (bps) — legacy BPS-delta mode only
    pub bid_skew_bps: f64,
    /// Recommended ask skew (bps) — legacy BPS-delta mode only
    pub ask_skew_bps: f64,
    /// Gamma (risk aversion) multiplier — parameter mode only
    pub gamma_multiplier: f64,
    /// Omega (inventory skew) multiplier — parameter mode only
    pub omega_multiplier: f64,
    /// Confidence in recommendation [0, 1]
    pub confidence: f64,
    /// Whether this is exploration or exploitation
    pub is_exploration: bool,
    /// Action index (0..24)
    pub action_idx: ActionIndex,
    /// State index used for this recommendation
    pub state_idx: StateIndex,
    /// Expected Q-value
    pub expected_q: f64,
    /// Q-value uncertainty
    pub q_uncertainty: f64,
}

impl RLPolicyRecommendation {
    /// Create from agent action selection using index-based API.
    ///
    /// Always uses Thompson sampling (explore=true) per P1-1. The `explore`
    /// parameter is kept for API compat but ignored — Thompson sampling
    /// self-regulates exploration via posterior width.
    pub fn from_agent(agent: &mut QLearningAgent, state_idx: StateIndex, _explore: bool) -> Self {
        // Always explore via Thompson sampling (P1-1)
        let action_idx = agent.select_action_idx(state_idx);
        let stats = agent.get_q_stats_idx(state_idx);

        // Confidence based on observations and uncertainty
        let obs_factor = (stats.total_observations as f64 / 100.0).min(1.0);
        let uncertainty_factor = 1.0 / (1.0 + stats.best_q_std);
        let confidence = obs_factor * uncertainty_factor;

        let is_exploration = action_idx != stats.best_action_idx;

        if agent.use_parameter_actions() {
            let param_action = ParameterAction::from_index(action_idx);
            Self {
                spread_delta_bps: 0.0,
                bid_skew_bps: 0.0,
                ask_skew_bps: 0.0,
                gamma_multiplier: param_action.gamma_multiplier(),
                omega_multiplier: param_action.omega_multiplier(),
                confidence,
                is_exploration,
                action_idx,
                state_idx,
                expected_q: stats.best_q_mean,
                q_uncertainty: stats.best_q_std,
            }
        } else {
            let mdp_action = MDPAction::from_index(action_idx);
            Self {
                spread_delta_bps: mdp_action.spread.delta_bps(),
                bid_skew_bps: mdp_action.skew.bid_skew_bps(),
                ask_skew_bps: mdp_action.skew.ask_skew_bps(),
                gamma_multiplier: 1.0,
                omega_multiplier: 1.0,
                confidence,
                is_exploration,
                action_idx,
                state_idx,
                expected_q: stats.best_q_mean,
                q_uncertainty: stats.best_q_std,
            }
        }
    }
}

// ============================================================================
// Helper Functions for Sampling
// ============================================================================

/// Sample from standard normal distribution (Box-Muller transform).
fn sample_standard_normal() -> f64 {
    let u1 = sample_uniform();
    let u2 = sample_uniform();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Sample from Gamma distribution (Marsaglia and Tsang's method).
fn sample_gamma(alpha: f64, beta: f64) -> f64 {
    if alpha < 1.0 {
        // Use Ahrens-Dieter method for alpha < 1
        let u = sample_uniform();
        return sample_gamma(alpha + 1.0, beta) * u.powf(1.0 / alpha);
    }

    let d = alpha - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();

    loop {
        let x = sample_standard_normal();
        let v = (1.0 + c * x).powi(3);
        if v > 0.0 {
            let u = sample_uniform();
            if u < 1.0 - 0.0331 * x.powi(4) {
                return d * v / beta;
            }
            if u.ln() < 0.5 * x.powi(2) + d * (1.0 - v + v.ln()) {
                return d * v / beta;
            }
        }
    }
}

/// Sample from uniform distribution [0, 1).
fn sample_uniform() -> f64 {
    use rand::rngs::SmallRng;
    use rand::Rng;
    use rand::SeedableRng;

    thread_local! {
        static RNG: std::cell::RefCell<SmallRng> = std::cell::RefCell::new(
            SmallRng::from_entropy()
        );
    }

    RNG.with(|rng| rng.borrow_mut().gen::<f64>())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inventory_bucket_from_position() {
        assert_eq!(
            InventoryBucket::from_position(-60.0, 100.0),
            InventoryBucket::Short
        );
        assert_eq!(
            InventoryBucket::from_position(-30.0, 100.0),
            InventoryBucket::Short
        );
        assert_eq!(
            InventoryBucket::from_position(-10.0, 100.0),
            InventoryBucket::SmallShort
        );
        assert_eq!(
            InventoryBucket::from_position(0.0, 100.0),
            InventoryBucket::Neutral
        );
        assert_eq!(
            InventoryBucket::from_position(10.0, 100.0),
            InventoryBucket::SmallLong
        );
        assert_eq!(
            InventoryBucket::from_position(30.0, 100.0),
            InventoryBucket::Long
        );
        assert_eq!(
            InventoryBucket::from_position(60.0, 100.0),
            InventoryBucket::Long
        );
    }

    #[test]
    fn test_imbalance_bucket_from_imbalance() {
        assert_eq!(ImbalanceBucket::from_imbalance(-0.5), ImbalanceBucket::Sell);
        assert_eq!(
            ImbalanceBucket::from_imbalance(-0.25),
            ImbalanceBucket::Sell
        );
        assert_eq!(
            ImbalanceBucket::from_imbalance(0.0),
            ImbalanceBucket::Neutral
        );
        assert_eq!(ImbalanceBucket::from_imbalance(0.25), ImbalanceBucket::Buy);
        assert_eq!(ImbalanceBucket::from_imbalance(0.5), ImbalanceBucket::Buy);
    }

    #[test]
    fn test_mdp_state_to_index() {
        let state1 = MDPState::default();
        let state2 = MDPState {
            inventory: InventoryBucket::Long,
            imbalance: ImbalanceBucket::Buy,
            volatility: VolatilityBucket::High,
            adverse: AdverseBucket::High,
            excitation: ExcitationBucket::High,
            drift: DriftBucket::Bullish,
        };

        let idx1 = state1.to_index();
        let idx2 = state2.to_index();

        assert!(idx1 < MDPState::STATE_COUNT);
        assert!(idx2 < MDPState::STATE_COUNT);
        assert_eq!(MDPState::STATE_COUNT, 2025);
        assert_ne!(idx1, idx2);
    }

    #[test]
    fn test_mdp_state_from_index_round_trip() {
        // Test every possible state round-trips through to_index → from_index
        for idx in 0..MDPState::STATE_COUNT {
            let state = MDPState::from_index(idx);
            assert_eq!(
                state.to_index(),
                idx,
                "Round-trip failed for index {idx}: got {:?} which maps to {}",
                state,
                state.to_index()
            );
        }
    }

    #[test]
    fn test_mdp_state_from_index_specific() {
        // Default state (Neutral everything) should be in the middle
        let default_state = MDPState::default();
        let default_idx = default_state.to_index();
        let reconstructed = MDPState::from_index(default_idx);
        assert_eq!(reconstructed, default_state);

        // All-max state
        let max_state = MDPState {
            inventory: InventoryBucket::Long,
            imbalance: ImbalanceBucket::Buy,
            volatility: VolatilityBucket::High,
            adverse: AdverseBucket::High,
            excitation: ExcitationBucket::High,
            drift: DriftBucket::Bullish,
        };
        let max_idx = max_state.to_index();
        assert_eq!(MDPState::from_index(max_idx), max_state);
        assert_eq!(max_idx, MDPState::STATE_COUNT - 1);

        // All-min state
        let min_state = MDPState {
            inventory: InventoryBucket::Short,
            imbalance: ImbalanceBucket::Sell,
            volatility: VolatilityBucket::Low,
            adverse: AdverseBucket::Low,
            excitation: ExcitationBucket::Normal,
            drift: DriftBucket::Bearish,
        };
        assert_eq!(MDPState::from_index(0), min_state);
    }

    #[test]
    fn test_mdp_action_deltas() {
        let action = MDPAction::new(SpreadAction::TightenSmall, SkewAction::ModerateBidBias);
        assert_eq!(action.spread.delta_bps(), -1.5);
        assert_eq!(action.skew.bid_skew_bps(), -1.0);
        assert_eq!(action.skew.ask_skew_bps(), 1.0);
        assert_eq!(action.bid_delta_bps(), -2.5);
        assert_eq!(action.ask_delta_bps(), -0.5);
    }

    #[test]
    fn test_mdp_action_round_trip() {
        for i in 0..MDPAction::ACTION_COUNT {
            let action = MDPAction::from_index(i);
            assert_eq!(action.to_index(), i);
        }
    }

    #[test]
    fn test_reward_computation() {
        let config = RewardConfig::default();
        let reward = Reward::compute(&config, 2.0, 0.3, 1.0, 0.2);

        assert!(reward.edge_component > 0.0);
        assert!(reward.inventory_penalty <= 0.0);
        assert_eq!(reward.volatility_penalty, 0.0); // vol_ratio = 1.0
    }

    #[test]
    fn test_reward_no_double_count_on_adverse() {
        let config = RewardConfig::default();
        // Adverse fill: negative edge. Should NOT have extra penalty.
        let reward = Reward::compute(&config, -3.0, 0.2, 1.0, 0.2);

        assert!(reward.edge_component < 0.0);
        // No separate adverse penalty — AS cost is embedded in realized_edge_bps
        // Total should be edge + inventory penalty + vol penalty + inv change
        let expected = reward.edge_component
            + reward.inventory_penalty
            + reward.volatility_penalty
            + reward.inventory_change_penalty;
        assert!((reward.total - expected).abs() < 1e-10);
    }

    #[test]
    fn test_reward_with_adverse_selection_subtracted() {
        // Validates that the RL reward uses realized_edge = depth - AS - fee,
        // matching the edge tracker formula in handlers.rs.
        let config = RewardConfig::default();

        // Scenario: fill at 5 bps depth, 2 bps AS realized, 1.5 bps fee
        let depth_bps = 5.0;
        let as_realized_bps = 2.0;
        let fee_bps = 1.5;
        let realized_edge_with_as = depth_bps - as_realized_bps - fee_bps; // 1.5

        // Without AS subtraction (the old incorrect formula)
        let realized_edge_without_as = depth_bps - fee_bps; // 3.5

        let reward_correct = Reward::compute(&config, realized_edge_with_as, 0.3, 1.0, 0.3);
        let reward_inflated = Reward::compute(&config, realized_edge_without_as, 0.3, 1.0, 0.3);

        // The correct reward (with AS) should be lower than the inflated one
        assert!(
            reward_correct.total < reward_inflated.total,
            "Reward with AS subtracted ({:.4}) should be less than without ({:.4})",
            reward_correct.total,
            reward_inflated.total
        );

        // With AS, the edge is still positive (1.5 bps)
        assert!(
            reward_correct.edge_component > 0.0,
            "Edge should be positive when depth > AS + fee"
        );
    }

    #[test]
    fn test_reward_adverse_fill_has_negative_edge() {
        // When AS exceeds depth, the edge is negative (bad fill)
        let config = RewardConfig::default();

        // Scenario: fill at 3 bps depth, 6 bps AS (price moved against us), 1.5 bps fee
        let depth_bps = 3.0;
        let as_realized_bps = 6.0;
        let fee_bps = 1.5;
        let realized_edge = depth_bps - as_realized_bps - fee_bps; // -4.5

        let reward = Reward::compute(&config, realized_edge, 0.3, 1.0, 0.3);
        assert!(
            reward.edge_component < 0.0,
            "Edge should be negative when AS > depth + fee"
        );
    }

    #[test]
    fn test_reward_inventory_change_penalty() {
        let config = RewardConfig::default();
        // Inventory increased from 20% to 50% → should be penalized
        let reward_increase = Reward::compute(&config, 1.0, 0.5, 1.0, 0.2);
        // Inventory stayed at 50% → no change penalty
        let reward_stable = Reward::compute(&config, 1.0, 0.5, 1.0, 0.5);

        assert!(
            reward_increase.inventory_change_penalty < reward_stable.inventory_change_penalty,
            "Increasing inventory should be penalized more than stable inventory"
        );
    }

    #[test]
    fn test_bayesian_q_value_update() {
        let mut q = BayesianQValue::new();
        assert_eq!(q.count(), 0);
        assert_eq!(q.mean(), 0.0);

        q.update(1.0);
        assert_eq!(q.count(), 1);
        assert!(q.mean() > 0.0);

        q.update(1.0);
        q.update(1.0);
        assert_eq!(q.count(), 3);
        assert!(q.mean() > 0.5); // Should converge toward 1.0
    }

    #[test]
    fn test_bayesian_q_value_variance_decreases() {
        let mut q = BayesianQValue::new();
        let initial_var = q.variance();

        for _ in 0..10 {
            q.update(0.5);
        }

        assert!(q.variance() < initial_var);
    }

    #[test]
    fn test_q_learning_agent_action_selection() {
        let mut agent = QLearningAgent::default();
        let state = MDPState::default();

        // Should be able to select actions
        let action = agent.select_action(&state);
        assert!(action.to_index() < MDPAction::ACTION_COUNT);
    }

    #[test]
    fn test_q_learning_agent_update() {
        let mut agent = QLearningAgent::new(QLearningConfig {
            use_compact_state: false,
            use_parameter_actions: false,
            ..Default::default()
        });
        let state = MDPState::default();
        let action = MDPAction::default();
        let reward = Reward {
            total: 1.0,
            edge_component: 1.0,
            inventory_penalty: 0.0,
            volatility_penalty: 0.0,
            inventory_change_penalty: 0.0,
            drift_penalty: 0.0,
        };
        let next_state = MDPState::default();

        agent.update(state, action, reward, next_state, false);

        let summary = agent.summary();
        assert!(summary.total_reward > 0.0);
    }

    #[test]
    fn test_rl_policy_recommendation_parameter_mode() {
        let mut agent = QLearningAgent::default(); // use_parameter_actions = true
        let state_idx = MDPStateCompact::default().to_index();

        let rec = RLPolicyRecommendation::from_agent(&mut agent, state_idx, true);

        // In parameter mode, gamma/omega multipliers should be set
        assert!(rec.gamma_multiplier >= 0.5 && rec.gamma_multiplier <= 2.0);
        assert!(rec.omega_multiplier >= 0.25 && rec.omega_multiplier <= 2.0);
        assert!(rec.confidence >= 0.0 && rec.confidence <= 1.0);
    }

    #[test]
    fn test_rl_policy_recommendation_legacy_mode() {
        let mut agent = QLearningAgent::new(QLearningConfig {
            use_parameter_actions: false,
            use_compact_state: false,
            ..Default::default()
        });
        let state_idx = MDPState::default().to_index();

        let rec = RLPolicyRecommendation::from_agent(&mut agent, state_idx, false);

        assert!(rec.spread_delta_bps >= -3.0 && rec.spread_delta_bps <= 3.0);
        assert!(rec.confidence >= 0.0 && rec.confidence <= 1.0);
        // In legacy mode, multipliers should be 1.0
        assert_eq!(rec.gamma_multiplier, 1.0);
        assert_eq!(rec.omega_multiplier, 1.0);
    }

    #[test]
    fn test_thompson_sampling_varies() {
        let mut q = BayesianQValue::new();
        // Add some observations to get non-trivial posterior
        for _ in 0..10 {
            q.update(1.0);
        }

        // Samples should vary (stochastic)
        let samples: Vec<f64> = (0..100).map(|_| q.sample()).collect();
        let mean = samples.iter().sum::<f64>() / 100.0;
        let variance = samples.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / 100.0;

        // Variance should be positive (samples differ)
        assert!(variance > 0.0);
    }

    // ========================================================================
    // Phase 6A.2: Parameter Action Tests
    // ========================================================================

    #[test]
    fn test_gamma_action_multipliers() {
        assert_eq!(GammaAction::VeryDefensive.multiplier(), 2.0);
        assert_eq!(GammaAction::Defensive.multiplier(), 1.5);
        assert_eq!(GammaAction::Neutral.multiplier(), 1.0);
        assert_eq!(GammaAction::Aggressive.multiplier(), 0.75);
        assert_eq!(GammaAction::VeryAggressive.multiplier(), 0.5);
    }

    #[test]
    fn test_omega_action_multipliers() {
        assert_eq!(OmegaAction::StrongSkew.multiplier(), 2.0);
        assert_eq!(OmegaAction::ModerateSkew.multiplier(), 1.5);
        assert_eq!(OmegaAction::Neutral.multiplier(), 1.0);
        assert_eq!(OmegaAction::ReducedSkew.multiplier(), 0.5);
        assert_eq!(OmegaAction::MinimalSkew.multiplier(), 0.25);
    }

    #[test]
    fn test_intensity_action_values() {
        assert_eq!(IntensityAction::NoQuote.intensity(), 0.0);
        assert_eq!(IntensityAction::Light.intensity(), 0.25);
        assert_eq!(IntensityAction::Moderate.intensity(), 0.5);
        assert_eq!(IntensityAction::Heavy.intensity(), 0.75);
        assert_eq!(IntensityAction::Full.intensity(), 1.0);
    }

    #[test]
    fn test_parameter_action_round_trip() {
        // Test all 25 parameter actions round-trip through index
        for i in 0..ParameterAction::ACTION_COUNT {
            let action = ParameterAction::from_index(i);
            let recovered = action.to_index();
            assert_eq!(i, recovered, "Index {} didn't round-trip", i);
        }
    }

    #[test]
    fn test_parameter_action_count() {
        // 5 × 5 = 25 actions (IntensityAction dropped — sizing is risk manager's job)
        assert_eq!(ParameterAction::ACTION_COUNT, 25);
        assert_eq!(ParameterAction::ACTION_COUNT, UNIFIED_ACTION_COUNT);
        assert_eq!(MDPAction::ACTION_COUNT, UNIFIED_ACTION_COUNT);
    }

    #[test]
    fn test_parameter_action_neutral() {
        let neutral = ParameterAction::neutral();
        assert_eq!(neutral.gamma_multiplier(), 1.0);
        assert_eq!(neutral.omega_multiplier(), 1.0);
    }

    #[test]
    fn test_parameter_action_defensive() {
        let defensive = ParameterAction::defensive();
        assert!(
            defensive.gamma_multiplier() > 1.0,
            "Defensive should have higher γ"
        );
        assert!(
            defensive.omega_multiplier() > 1.0,
            "Defensive should have higher skew"
        );
    }

    #[test]
    fn test_compact_state_round_trip() {
        for idx in 0..MDPStateCompact::STATE_COUNT {
            let state = MDPStateCompact::from_index(idx);
            assert_eq!(
                state.to_index(),
                idx,
                "Compact state round-trip failed for index {idx}"
            );
        }
    }

    #[test]
    fn test_compact_state_count() {
        // 5 × 3 × 3 = 45 states
        assert_eq!(MDPStateCompact::STATE_COUNT, 45);
    }

    #[test]
    fn test_compact_state_from_continuous() {
        let state = MDPStateCompact::from_continuous(0.0, 100.0, 0.0, 1.0);
        assert_eq!(state.inventory, InventoryBucket::Neutral);
        assert_eq!(state.volatility, VolatilityBucket::Normal);
        assert_eq!(state.imbalance, ImbalanceBucketCompact::Neutral);
    }

    // ================================================================
    // Task 9: import_q_table_as_prior blend tests
    // ================================================================

    #[test]
    fn test_import_q_table_cold_state_gets_paper_prior() {
        let mut agent = QLearningAgent::default();
        let state_idx = 0usize;

        // Create a paper Q-table with known values
        let mut paper_q = HashMap::new();
        let mut paper_values = vec![BayesianQValue::new(); UNIFIED_ACTION_COUNT];
        // Give paper action 5 some updates so it has a meaningful posterior
        for _ in 0..10 {
            paper_values[5].update(2.0);
        }
        paper_q.insert(state_idx, paper_values);

        // Agent has no data for this state (cold start)
        agent.import_q_table_as_prior(&paper_q, 0.3);

        // Cold state should receive the paper prior
        let q_vals = agent.get_q_values_by_idx(state_idx);
        // The paper prior should be applied — mu_n should reflect paper's posterior mean
        assert!(
            q_vals[5].mean() > 0.5,
            "Cold state should get paper prior mean"
        );
        // n should be 0 (discounted prior, no real observations)
        assert_eq!(q_vals[5].count(), 0, "Discounted prior should have n=0");
    }

    #[test]
    fn test_import_q_table_warm_state_preserved() {
        let mut agent = QLearningAgent::default();
        let state_idx = 0usize;

        // Give the live agent real observations at state 0, action 3
        let reward = Reward {
            total: 5.0,
            edge_component: 5.0,
            inventory_penalty: 0.0,
            volatility_penalty: 0.0,
            inventory_change_penalty: 0.0,
            drift_penalty: 0.0,
        };
        agent.update_idx(state_idx, 3, reward, 1, false);
        let live_mean_before = agent.get_q_values_by_idx(state_idx)[3].mean();
        let live_count_before = agent.get_q_values_by_idx(state_idx)[3].count();
        assert!(live_count_before > 0, "Should have live data");

        // Create paper Q-table with different values
        let mut paper_q = HashMap::new();
        let mut paper_values = vec![BayesianQValue::new(); UNIFIED_ACTION_COUNT];
        for _ in 0..10 {
            paper_values[3].update(-10.0); // very different from live
        }
        paper_q.insert(state_idx, paper_values);

        // Import — should NOT overwrite warm states
        agent.import_q_table_as_prior(&paper_q, 0.3);

        let q_vals = agent.get_q_values_by_idx(state_idx);
        assert_eq!(
            q_vals[3].count(),
            live_count_before,
            "Live count must be preserved"
        );
        assert!(
            (q_vals[3].mean() - live_mean_before).abs() < 1e-10,
            "Live mean must be preserved"
        );
    }

    #[test]
    fn test_import_q_table_observation_count_preserved() {
        let mut agent = QLearningAgent::default();
        let state_idx = 2usize;

        // Give live agent 5 observations at action 0
        for _ in 0..5 {
            let reward = Reward {
                total: 1.0,
                edge_component: 1.0,
                inventory_penalty: 0.0,
                volatility_penalty: 0.0,
                inventory_change_penalty: 0.0,
                drift_penalty: 0.0,
            };
            agent.update_idx(state_idx, 0, reward, 3, false);
        }
        assert_eq!(agent.get_q_values_by_idx(state_idx)[0].count(), 5);

        let mut paper_q = HashMap::new();
        let mut paper_values = vec![BayesianQValue::new(); UNIFIED_ACTION_COUNT];
        for _ in 0..20 {
            paper_values[0].update(0.5);
        }
        paper_q.insert(state_idx, paper_values);

        agent.import_q_table_as_prior(&paper_q, 0.3);

        // n must still be 5, not reset to 0
        assert_eq!(
            agent.get_q_values_by_idx(state_idx)[0].count(),
            5,
            "Observation count must not be reset by import"
        );
    }

    #[test]
    fn test_import_q_table_mixed_cold_warm() {
        let mut agent = QLearningAgent::default();
        let state_idx = 0usize;

        // Warm up action 2 only
        let reward = Reward {
            total: 3.0,
            edge_component: 3.0,
            inventory_penalty: 0.0,
            volatility_penalty: 0.0,
            inventory_change_penalty: 0.0,
            drift_penalty: 0.0,
        };
        agent.update_idx(state_idx, 2, reward, 1, false);
        let warm_count = agent.get_q_values_by_idx(state_idx)[2].count();
        assert!(warm_count > 0);

        // Paper Q-table has data for actions 2 and 7
        let mut paper_q = HashMap::new();
        let mut paper_values = vec![BayesianQValue::new(); UNIFIED_ACTION_COUNT];
        for _ in 0..10 {
            paper_values[2].update(10.0);
            paper_values[7].update(5.0);
        }
        paper_q.insert(state_idx, paper_values);

        agent.import_q_table_as_prior(&paper_q, 0.3);

        // Action 2 (warm) should be untouched
        assert_eq!(
            agent.get_q_values_by_idx(state_idx)[2].count(),
            warm_count,
            "Warm action should be preserved"
        );
        // Action 7 (cold) should get paper prior
        let q7 = &agent.get_q_values_by_idx(state_idx)[7];
        assert_eq!(
            q7.count(),
            0,
            "Cold action should get discounted prior (n=0)"
        );
        assert!(q7.mean() > 1.0, "Cold action should have paper mean");
    }

    // ================================================================
    // Task 11: SimToRealConfig safety guard tests
    // ================================================================

    #[test]
    fn test_sim_to_real_guards_block_early_actions() {
        // Agent with min_real_fills=20 and no updates should return neutral
        let sim_config = SimToRealConfig {
            min_real_fills: 20,
            action_bound_sigma: 2.0,
            auto_disable_after_fills: 100,
            ..Default::default()
        };
        let mut agent = QLearningAgent::with_sim_to_real(QLearningConfig::default(), sim_config);
        let state_idx = MDPStateCompact::default().to_index();
        let neutral_idx = ParameterAction::neutral().to_index();

        // With 0 total updates, should always return neutral
        for _ in 0..20 {
            let action = agent.select_action_idx(state_idx);
            assert_eq!(
                action, neutral_idx,
                "Must return neutral when total_updates < min_real_fills"
            );
        }
    }

    #[test]
    fn test_sim_to_real_actions_clipped() {
        // With action_bound_sigma=1.0, only actions within 1 step of neutral
        // should be allowed (gamma index 1..3, omega index 1..3)
        let sim_config = SimToRealConfig {
            min_real_fills: 0, // disable min fills guard
            action_bound_sigma: 1.0,
            auto_disable_after_fills: 100_000, // disable auto-disable
            ..Default::default()
        };
        let config = QLearningConfig {
            min_observations: 0,       // allow immediate exploitation
            min_exploration_rate: 0.0, // no random exploration
            exploration: ExplorationStrategy::EpsilonGreedy {
                epsilon: 0.0,
                decay: 1.0,
            },
            use_parameter_actions: true,
            ..Default::default()
        };
        let mut agent = QLearningAgent::with_sim_to_real(config, sim_config);
        let state_idx = 0usize;

        // Make action at index for VeryAggressive gamma (idx=4) + MinimalSkew omega (idx=4)
        // extremely attractive so the agent wants to pick it
        let extreme_action_idx =
            ParameterAction::new(GammaAction::VeryAggressive, OmegaAction::MinimalSkew).to_index();
        // Need some observations first
        for _ in 0..50 {
            let reward = Reward {
                total: 100.0,
                edge_component: 100.0,
                inventory_penalty: 0.0,
                volatility_penalty: 0.0,
                inventory_change_penalty: 0.0,
                drift_penalty: 0.0,
            };
            agent.update_idx(state_idx, extreme_action_idx, reward, state_idx, false);
        }

        // Select action — should be clipped
        let action_idx = agent.select_action_idx(state_idx);
        let action = ParameterAction::from_index(action_idx);
        let neutral = ParameterAction::neutral();

        let gamma_dist = (action.gamma.index() as f64 - neutral.gamma.index() as f64).abs();
        let omega_dist = (action.omega.index() as f64 - neutral.omega.index() as f64).abs();

        assert!(
            gamma_dist <= 1.0,
            "Gamma must be clipped within 1 sigma of neutral, got dist={}",
            gamma_dist
        );
        assert!(
            omega_dist <= 1.0,
            "Omega must be clipped within 1 sigma of neutral, got dist={}",
            omega_dist
        );
    }

    #[test]
    fn test_sim_to_real_auto_disable_on_negative_reward() {
        let sim_config = SimToRealConfig {
            min_real_fills: 0,
            action_bound_sigma: 10.0, // wide bounds, don't clip
            auto_disable_after_fills: 5,
            ..Default::default()
        };
        let mut agent = QLearningAgent::with_sim_to_real(QLearningConfig::default(), sim_config);
        let state_idx = 0usize;
        let neutral_idx = ParameterAction::neutral().to_index();

        // Give agent negative rewards to trigger auto-disable
        for i in 0..10 {
            let reward = Reward {
                total: -2.0,
                edge_component: -2.0,
                inventory_penalty: 0.0,
                volatility_penalty: 0.0,
                inventory_change_penalty: 0.0,
                drift_penalty: 0.0,
            };
            agent.update_idx(
                state_idx,
                i % UNIFIED_ACTION_COUNT,
                reward,
                state_idx,
                false,
            );
        }

        // total_updates > 5 and mean_reward < 0 => should return neutral
        assert!(agent.total_updates() > 5);
        assert!(agent.mean_recent_reward() < 0.0);
        let action = agent.select_action_idx(state_idx);
        assert_eq!(
            action, neutral_idx,
            "Must return neutral when auto-disable triggers"
        );
    }

    #[test]
    fn test_sim_to_real_guards_pass_after_sufficient_positive_fills() {
        let sim_config = SimToRealConfig {
            min_real_fills: 5,
            action_bound_sigma: 10.0, // wide bounds
            auto_disable_after_fills: 1000,
            ..Default::default()
        };
        let config = QLearningConfig {
            min_observations: 0,
            min_exploration_rate: 0.0,
            exploration: ExplorationStrategy::EpsilonGreedy {
                epsilon: 0.0,
                decay: 1.0,
            },
            ..Default::default()
        };
        let mut agent = QLearningAgent::with_sim_to_real(config, sim_config);
        let state_idx = 0usize;

        // Give 10 positive fills — should exceed min_real_fills threshold
        for _ in 0..10 {
            let reward = Reward {
                total: 5.0,
                edge_component: 5.0,
                inventory_penalty: 0.0,
                volatility_penalty: 0.0,
                inventory_change_penalty: 0.0,
                drift_penalty: 0.0,
            };
            agent.update_idx(state_idx, 0, reward, state_idx, false);
        }

        assert!(agent.total_updates() >= 5);
        assert!(agent.mean_recent_reward() > 0.0);

        // Should NOT be forced to neutral — the agent can now pick freely
        // With epsilon=0.0 and all updates at action 0, greedy should pick action 0
        let action = agent.select_action_idx(state_idx);
        let neutral_action = ParameterAction::neutral().to_index();
        // The agent should be able to pick action 0 (not forced to neutral=12)
        assert_ne!(
            action, neutral_action,
            "Agent should NOT be forced to neutral after sufficient positive fills"
        );
        assert_eq!(
            action, 0,
            "Agent should exploit best action after sufficient fills"
        );
    }

    #[test]
    fn test_auto_disable_threshold_not_triggered_by_fee_drag() {
        // Fee drag of -1.0 bps should NOT trigger auto-disable with threshold at -1.5 bps
        let sim_config = SimToRealConfig {
            min_real_fills: 0,
            action_bound_sigma: 10.0,
            auto_disable_after_fills: 5,
            auto_disable_threshold_bps: -1.5,
            ..Default::default()
        };
        let config = QLearningConfig {
            min_observations: 0,
            min_exploration_rate: 0.0,
            exploration: ExplorationStrategy::EpsilonGreedy {
                epsilon: 0.0,
                decay: 1.0,
            },
            ..Default::default()
        };
        let mut agent = QLearningAgent::with_sim_to_real(config, sim_config);
        let state_idx = 0usize;
        let neutral_idx = ParameterAction::neutral().to_index();

        // Simulate fee-drag-only rewards: mean = -1.0 bps (above -1.5 threshold)
        for _ in 0..10 {
            let reward = Reward {
                total: -1.0,
                edge_component: -1.0,
                inventory_penalty: 0.0,
                volatility_penalty: 0.0,
                inventory_change_penalty: 0.0,
                drift_penalty: 0.0,
            };
            agent.update_idx(state_idx, 0, reward, state_idx, false);
        }

        assert!(agent.total_updates() > 5);
        // Mean reward is -1.0, which is above the -1.5 threshold
        assert!(agent.mean_recent_reward() > -1.5);

        let action = agent.select_action_idx(state_idx);
        // Agent should NOT be forced to neutral — fee drag alone doesn't disable RL
        assert_ne!(
            action, neutral_idx,
            "Fee-drag-only reward (-1.0 bps) should not trigger auto-disable at -1.5 threshold"
        );
    }

    #[test]
    fn test_auto_disable_triggers_below_threshold() {
        // Reward of -2.0 bps should trigger auto-disable with threshold at -1.5 bps
        let sim_config = SimToRealConfig {
            min_real_fills: 0,
            action_bound_sigma: 10.0,
            auto_disable_after_fills: 5,
            auto_disable_threshold_bps: -1.5,
            ..Default::default()
        };
        let mut agent = QLearningAgent::with_sim_to_real(QLearningConfig::default(), sim_config);
        let state_idx = 0usize;
        let neutral_idx = ParameterAction::neutral().to_index();

        // Rewards well below threshold
        for i in 0..10 {
            let reward = Reward {
                total: -2.0,
                edge_component: -2.0,
                inventory_penalty: 0.0,
                volatility_penalty: 0.0,
                inventory_change_penalty: 0.0,
                drift_penalty: 0.0,
            };
            agent.update_idx(
                state_idx,
                i % UNIFIED_ACTION_COUNT,
                reward,
                state_idx,
                false,
            );
        }

        assert!(agent.total_updates() > 5);
        assert!(agent.mean_recent_reward() < -1.5);

        let action = agent.select_action_idx(state_idx);
        assert_eq!(
            action, neutral_idx,
            "Reward below threshold (-2.0 < -1.5) must trigger auto-disable"
        );
    }

    #[test]
    fn test_recent_rewards_cleared_on_checkpoint_restore() {
        let mut agent = QLearningAgent::new(QLearningConfig::default());
        let state_idx = 0usize;

        // Accumulate negative rewards from a previous session
        for _ in 0..20 {
            let reward = Reward {
                total: -5.0,
                edge_component: -5.0,
                inventory_penalty: 0.0,
                volatility_penalty: 0.0,
                inventory_change_penalty: 0.0,
                drift_penalty: 0.0,
            };
            agent.update_idx(state_idx, 0, reward, state_idx, false);
        }

        assert!(agent.mean_recent_reward() < -1.5);
        assert!(!agent.recent_rewards.is_empty());

        // Save and restore from checkpoint
        let ckpt = agent.to_checkpoint();
        agent.restore_from_checkpoint(&ckpt);

        // Recent rewards must be cleared — fresh session starts clean
        assert!(
            agent.recent_rewards.is_empty(),
            "recent_rewards must be cleared on checkpoint restore"
        );
        // mean_recent_reward returns 0.0 for empty rewards, which is above any negative threshold
        assert_eq!(
            agent.mean_recent_reward(),
            0.0,
            "mean_recent_reward must be 0.0 after restore (empty history)"
        );
    }

    // ==========================================================================
    // DriftBucket tests
    // ==========================================================================

    #[test]
    fn test_drift_bucket_classification() {
        // Bearish: momentum < -5 bps
        assert_eq!(DriftBucket::from_momentum_bps(-10.0), DriftBucket::Bearish);
        assert_eq!(DriftBucket::from_momentum_bps(-5.1), DriftBucket::Bearish);
        // Neutral: -5 to +5 bps
        assert_eq!(DriftBucket::from_momentum_bps(-5.0), DriftBucket::Neutral);
        assert_eq!(DriftBucket::from_momentum_bps(0.0), DriftBucket::Neutral);
        assert_eq!(DriftBucket::from_momentum_bps(5.0), DriftBucket::Neutral);
        // Bullish: momentum > +5 bps
        assert_eq!(DriftBucket::from_momentum_bps(5.1), DriftBucket::Bullish);
        assert_eq!(DriftBucket::from_momentum_bps(20.0), DriftBucket::Bullish);
        // Round-trip index
        for i in 0..DriftBucket::COUNT {
            assert_eq!(DriftBucket::from_index(i).index(), i);
        }
    }

    #[test]
    fn test_drift_state_count_2025() {
        // 5 inventory * 5 imbalance * 3 volatility * 3 adverse * 3 excitation * 3 drift = 2025
        assert_eq!(MDPState::STATE_COUNT, 2025);
        assert_eq!(
            InventoryBucket::COUNT
                * ImbalanceBucket::COUNT
                * VolatilityBucket::COUNT
                * AdverseBucket::COUNT
                * ExcitationBucket::COUNT
                * DriftBucket::COUNT,
            2025,
        );
    }

    #[test]
    fn test_drift_state_unique_indices() {
        // Verify no collisions: states differing only in drift produce different indices
        let base = MDPState {
            inventory: InventoryBucket::Neutral,
            imbalance: ImbalanceBucket::Neutral,
            volatility: VolatilityBucket::Normal,
            adverse: AdverseBucket::Moderate,
            excitation: ExcitationBucket::Normal,
            drift: DriftBucket::Bearish,
        };
        let neutral = MDPState {
            drift: DriftBucket::Neutral,
            ..base
        };
        let bullish = MDPState {
            drift: DriftBucket::Bullish,
            ..base
        };

        let idx_bear = base.to_index();
        let idx_neut = neutral.to_index();
        let idx_bull = bullish.to_index();

        assert_ne!(idx_bear, idx_neut);
        assert_ne!(idx_bear, idx_bull);
        assert_ne!(idx_neut, idx_bull);
        // All within range
        assert!(idx_bear < MDPState::STATE_COUNT);
        assert!(idx_neut < MDPState::STATE_COUNT);
        assert!(idx_bull < MDPState::STATE_COUNT);
        // Round-trip
        assert_eq!(MDPState::from_index(idx_bear), base);
        assert_eq!(MDPState::from_index(idx_neut), neutral);
        assert_eq!(MDPState::from_index(idx_bull), bullish);
    }

    #[test]
    fn test_drift_penalty_applied_when_opposing() {
        let config = RewardConfig::default();
        // Position is long (+5), drift is bearish (-10 bps): opposing
        let reward = Reward::compute_with_drift(
            &config, 2.0,   // realized_edge_bps
            0.5,   // inventory_risk
            1.0,   // vol_ratio
            0.5,   // prev_inventory_risk
            5.0,   // position (long)
            -10.0, // momentum_bps (bearish)
            0.3,   // drift_penalty_weight
        );
        assert!(
            reward.drift_penalty < 0.0,
            "Drift penalty should be negative when position opposes drift"
        );
        // Also verify total includes the penalty
        let reward_no_drift =
            Reward::compute_with_drift(&config, 2.0, 0.5, 1.0, 0.5, 5.0, -10.0, 0.0);
        assert!(
            reward.total < reward_no_drift.total,
            "Total reward with drift penalty ({:.4}) should be less than without ({:.4})",
            reward.total,
            reward_no_drift.total
        );
    }

    #[test]
    fn test_drift_no_penalty_when_aligned() {
        let config = RewardConfig::default();
        // Position is long (+5), drift is bullish (+10 bps): aligned
        let reward = Reward::compute_with_drift(
            &config, 2.0,  // realized_edge_bps
            0.5,  // inventory_risk
            1.0,  // vol_ratio
            0.5,  // prev_inventory_risk
            5.0,  // position (long)
            10.0, // momentum_bps (bullish)
            0.3,  // drift_penalty_weight
        );
        assert_eq!(
            reward.drift_penalty, 0.0,
            "No drift penalty when position aligned with drift"
        );

        // Also test: zero momentum means no penalty
        let reward_zero = Reward::compute_with_drift(&config, 2.0, 0.5, 1.0, 0.5, 5.0, 0.0, 0.3);
        assert_eq!(
            reward_zero.drift_penalty, 0.0,
            "No drift penalty when momentum is zero"
        );

        // Also test: zero position means no penalty (position * momentum_bps = 0)
        let reward_flat = Reward::compute_with_drift(&config, 2.0, 0.0, 1.0, 0.0, 0.0, -10.0, 0.3);
        assert_eq!(
            reward_flat.drift_penalty, 0.0,
            "No drift penalty when flat position"
        );
    }
}
