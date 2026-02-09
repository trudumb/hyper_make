//! Phase 8: Reinforcement Learning Agent for Competitive Quoting
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

use crate::market_maker::checkpoint::types::{QTableEntry, RLCheckpoint};

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
}

impl MDPState {
    /// Create from continuous state values.
    pub fn from_continuous(
        position: f64,
        max_position: f64,
        book_imbalance: f64,
        vol_ratio: f64,
        adverse_posterior: f64,
        hawkes_branching: f64,
    ) -> Self {
        Self {
            inventory: InventoryBucket::from_position(position, max_position),
            imbalance: ImbalanceBucket::from_imbalance(book_imbalance),
            volatility: VolatilityBucket::from_vol_ratio(vol_ratio),
            adverse: AdverseBucket::from_posterior_mean(adverse_posterior),
            excitation: ExcitationBucket::from_branching_ratio(hawkes_branching),
        }
    }

    /// Convert to flat state index for Q-table lookup.
    /// Total states = 5 * 5 * 3 * 3 * 3 = 675
    pub fn to_index(&self) -> usize {
        let mut idx = self.inventory.index();
        idx = idx * ImbalanceBucket::COUNT + self.imbalance.index();
        idx = idx * VolatilityBucket::COUNT + self.volatility.index();
        idx = idx * AdverseBucket::COUNT + self.adverse.index();
        idx = idx * ExcitationBucket::COUNT + self.excitation.index();
        idx
    }

    /// Total number of discrete states.
    pub const STATE_COUNT: usize = InventoryBucket::COUNT
        * ImbalanceBucket::COUNT
        * VolatilityBucket::COUNT
        * AdverseBucket::COUNT
        * ExcitationBucket::COUNT;

    /// Reconstruct an MDPState from a flat index (inverse of `to_index()`).
    ///
    /// Uses modular arithmetic to extract each dimension in reverse order
    /// of how `to_index()` encodes them.
    pub fn from_index(idx: usize) -> Self {
        let mut remaining = idx;
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
/// Tunes γ, ω multipliers and quote intensity instead of raw bps adjustments.
/// Total actions: 5 × 5 × 5 = 125 (more than 27 but still tractable)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ParameterAction {
    /// Risk aversion multiplier
    pub gamma: GammaAction,
    /// Inventory skew multiplier
    pub omega: OmegaAction,
    /// Quote intensity
    pub intensity: IntensityAction,
}

impl ParameterAction {
    /// Create a new parameter action.
    pub fn new(gamma: GammaAction, omega: OmegaAction, intensity: IntensityAction) -> Self {
        Self { gamma, omega, intensity }
    }

    /// Create from flat index.
    pub fn from_index(idx: usize) -> Self {
        let intensity_idx = idx % IntensityAction::COUNT;
        let remaining = idx / IntensityAction::COUNT;
        let omega_idx = remaining % OmegaAction::COUNT;
        let gamma_idx = remaining / OmegaAction::COUNT;

        Self {
            gamma: GammaAction::from_index(gamma_idx),
            omega: OmegaAction::from_index(omega_idx),
            intensity: IntensityAction::from_index(intensity_idx),
        }
    }

    /// Convert to flat index.
    pub fn to_index(&self) -> usize {
        let mut idx = self.gamma.index();
        idx = idx * OmegaAction::COUNT + self.omega.index();
        idx = idx * IntensityAction::COUNT + self.intensity.index();
        idx
    }

    /// Get γ multiplier.
    pub fn gamma_multiplier(&self) -> f64 {
        self.gamma.multiplier()
    }

    /// Get ω multiplier.
    pub fn omega_multiplier(&self) -> f64 {
        self.omega.multiplier()
    }

    /// Get quote intensity [0, 1].
    pub fn quote_intensity(&self) -> f64 {
        self.intensity.intensity()
    }

    /// Total number of parameter actions.
    pub const ACTION_COUNT: usize = GammaAction::COUNT * OmegaAction::COUNT * IntensityAction::COUNT;

    /// Default neutral action (no changes to base parameters).
    pub fn neutral() -> Self {
        Self {
            gamma: GammaAction::Neutral,
            omega: OmegaAction::Neutral,
            intensity: IntensityAction::Full,
        }
    }

    /// Defensive action (wider spreads, strong skew, full quote).
    pub fn defensive() -> Self {
        Self {
            gamma: GammaAction::Defensive,
            omega: OmegaAction::StrongSkew,
            intensity: IntensityAction::Full,
        }
    }

    /// Cautious action (don't quote).
    pub fn cautious() -> Self {
        Self {
            gamma: GammaAction::VeryDefensive,
            omega: OmegaAction::Neutral,
            intensity: IntensityAction::NoQuote,
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
    /// Weight for adverse selection penalty
    pub adverse_penalty_weight: f64,
    /// Discount factor for future rewards
    pub gamma: f64,
}

impl Default for RewardConfig {
    fn default() -> Self {
        Self {
            edge_weight: 1.0,
            inventory_penalty_weight: 0.1,
            volatility_penalty_weight: 0.05,
            adverse_penalty_weight: 0.2,
            gamma: 0.95,
        }
    }
}

/// Reward signal from a transition.
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
    /// Adverse selection penalty (always non-positive)
    pub adverse_penalty: f64,
}

impl Reward {
    /// Compute reward from transition.
    pub fn compute(
        config: &RewardConfig,
        realized_edge_bps: f64,
        inventory_risk: f64,  // |position| / max_position
        vol_ratio: f64,
        was_adverse: bool,
    ) -> Self {
        let edge_component = config.edge_weight * realized_edge_bps;

        // Quadratic inventory penalty
        let inventory_penalty =
            -config.inventory_penalty_weight * inventory_risk.powi(2) * 10.0;

        // Volatility penalty (penalize holding in high vol)
        let vol_penalty_factor = (vol_ratio - 1.0).max(0.0);
        let volatility_penalty =
            -config.volatility_penalty_weight * vol_penalty_factor * inventory_risk * 5.0;

        // Adverse selection penalty
        let adverse_penalty = if was_adverse {
            -config.adverse_penalty_weight * realized_edge_bps.abs()
        } else {
            0.0
        };

        let total = edge_component + inventory_penalty + volatility_penalty + adverse_penalty;

        Self {
            total,
            edge_component,
            inventory_penalty,
            volatility_penalty,
            adverse_penalty,
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
            kappa_0: 0.01,  // Weak prior on mean
            mu_n: 0.0,
            kappa_n: 0.01,
            alpha: 1.0,     // Weak prior on precision
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
        let beta_new = self.beta
            + 0.5 * (reward - self.mu_n).powi(2) * self.kappa_n / kappa_new;

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
            mu_0: mu_n,    // Use posterior as prior for continued learning
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
}

impl Default for SimToRealConfig {
    fn default() -> Self {
        Self {
            paper_prior_weight: 0.3,
            min_real_fills: 20,
            action_bound_sigma: 1.5,
            auto_disable_after_fills: 100,
        }
    }
}

/// Q-learning agent with Bayesian Q-values.
#[derive(Debug)]
pub struct QLearningAgent {
    /// Configuration
    config: QLearningConfig,
    /// Q-table: state -> action -> BayesianQValue
    q_table: HashMap<usize, Vec<BayesianQValue>>,
    /// Episode count
    episodes: u64,
    /// Total reward accumulated
    total_reward: f64,
    /// Recent rewards for monitoring
    recent_rewards: Vec<f64>,
    /// FIFO queue of pending state-action pairs awaiting reward updates.
    /// Supports clustered fills where multiple quotes are outstanding.
    pending_state_actions: VecDeque<(MDPState, MDPAction)>,
}

impl QLearningAgent {
    /// Create a new Q-learning agent.
    pub fn new(config: QLearningConfig) -> Self {
        Self {
            config,
            q_table: HashMap::new(),
            episodes: 0,
            total_reward: 0.0,
            recent_rewards: Vec::with_capacity(1000),
            pending_state_actions: VecDeque::with_capacity(STATE_ACTION_QUEUE_CAPACITY),
        }
    }

    /// Get Q-values for a state (initialize if needed).
    fn get_q_values(&mut self, state: &MDPState) -> &mut Vec<BayesianQValue> {
        let idx = state.to_index();
        self.q_table.entry(idx).or_insert_with(|| {
            vec![BayesianQValue::new(); MDPAction::ACTION_COUNT]
        })
    }

    /// Select action using the configured exploration strategy.
    pub fn select_action(&mut self, state: &MDPState) -> MDPAction {
        // Copy config values to avoid borrow issues
        let min_observations = self.config.min_observations;
        let exploration = self.config.exploration;
        let ucb_c = self.config.ucb_c;
        let episodes = self.episodes;

        let q_values = self.get_q_values(state);

        // Check if we have enough observations for exploitation
        let total_obs: u64 = q_values.iter().map(|q| q.count()).sum();
        if total_obs < min_observations {
            // Pure exploration: uniform random
            let action_idx = (sample_uniform() * MDPAction::ACTION_COUNT as f64) as usize;
            return MDPAction::from_index(action_idx.min(MDPAction::ACTION_COUNT - 1));
        }

        let action_idx = match exploration {
            ExplorationStrategy::ThompsonSampling => {
                // Sample from each Q-value posterior and select max
                q_values
                    .iter()
                    .enumerate()
                    .map(|(i, q)| (i, q.sample()))
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            }
            ExplorationStrategy::UCB => {
                // Select action with highest UCB
                q_values
                    .iter()
                    .enumerate()
                    .map(|(i, q)| (i, q.ucb(ucb_c)))
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            }
            ExplorationStrategy::EpsilonGreedy { epsilon, decay } => {
                let effective_epsilon = epsilon * decay.powf(episodes as f64);
                if sample_uniform() < effective_epsilon {
                    // Random action
                    (sample_uniform() * MDPAction::ACTION_COUNT as f64) as usize
                } else {
                    // Greedy action
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

        MDPAction::from_index(action_idx.min(MDPAction::ACTION_COUNT - 1))
    }

    /// Update Q-values with observed transition.
    pub fn update(
        &mut self,
        state: MDPState,
        action: MDPAction,
        reward: Reward,
        next_state: MDPState,
        done: bool,
    ) {
        // Copy gamma to avoid borrow issues
        let gamma = self.config.gamma;

        // Get max Q-value for next state
        let max_next_q = if done {
            0.0
        } else {
            let next_q_values = self.get_q_values(&next_state);
            next_q_values
                .iter()
                .map(|q| q.mean())
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(0.0)
        };

        // TD target
        let td_target = reward.total + gamma * max_next_q;

        // Update Q-value with Bayesian posterior update
        let action_idx = action.to_index();
        let state_idx = state.to_index();
        let q_values = self.get_q_values(&state);
        q_values[action_idx].update(td_target);
        let q_mean = q_values[action_idx].mean();
        let q_std = q_values[action_idx].std();

        // Track rewards
        self.total_reward += reward.total;
        self.recent_rewards.push(reward.total);
        if self.recent_rewards.len() > 1000 {
            self.recent_rewards.remove(0);
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

    /// Record start of a new episode.
    pub fn start_episode(&mut self, _initial_state: MDPState) {
        self.episodes += 1;
        self.pending_state_actions.clear();
    }

    /// Get the greedy action (exploitation only).
    pub fn get_greedy_action(&mut self, state: &MDPState) -> MDPAction {
        let q_values = self.get_q_values(state);
        let action_idx = q_values
            .iter()
            .enumerate()
            .map(|(i, q)| (i, q.mean()))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        MDPAction::from_index(action_idx)
    }

    /// Get Q-value statistics for a state.
    pub fn get_q_stats(&mut self, state: &MDPState) -> QValueStats {
        let q_values = self.get_q_values(state);
        let best_idx = q_values
            .iter()
            .enumerate()
            .map(|(i, q)| (i, q.mean()))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        let best_q = &q_values[best_idx];

        QValueStats {
            best_action: MDPAction::from_index(best_idx),
            best_q_mean: best_q.mean(),
            best_q_std: best_q.std(),
            best_q_count: best_q.count(),
            total_observations: q_values.iter().map(|q| q.count()).sum(),
        }
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

    /// Push a state-action pair onto the pending queue (FIFO).
    ///
    /// If the queue exceeds capacity, the oldest entry is dropped.
    pub fn push_state_action(&mut self, state: MDPState, action: MDPAction) {
        if self.pending_state_actions.len() >= STATE_ACTION_QUEUE_CAPACITY {
            self.pending_state_actions.pop_front();
        }
        self.pending_state_actions.push_back((state, action));
    }

    /// Pop the next (oldest) pending state-action pair for reward update.
    pub fn take_next_state_action(&mut self) -> Option<(MDPState, MDPAction)> {
        self.pending_state_actions.pop_front()
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
    pub fn take_last_state_action(&mut self) -> Option<(MDPState, MDPAction)> {
        self.take_next_state_action()
    }

    /// Export the Q-table for checkpoint persistence or sim-to-real transfer.
    pub fn export_q_table(&self) -> HashMap<usize, Vec<BayesianQValue>> {
        self.q_table.clone()
    }

    /// Import a paper trading Q-table as a discounted prior for live trading.
    ///
    /// For each state-action pair in `paper_q`, creates a `BayesianQValue` with
    /// the paper posterior as a down-weighted prior. States not present in `paper_q`
    /// retain their current (default) Q-values.
    pub fn import_q_table_as_prior(
        &mut self,
        paper_q: &HashMap<usize, Vec<BayesianQValue>>,
        weight: f64,
    ) {
        for (&state_idx, paper_values) in paper_q {
            let live_values = self
                .q_table
                .entry(state_idx)
                .or_insert_with(|| vec![BayesianQValue::new(); MDPAction::ACTION_COUNT]);
            for (i, paper_qv) in paper_values.iter().enumerate() {
                if i < live_values.len() {
                    live_values[i] = BayesianQValue::with_discounted_prior(paper_qv, weight);
                }
            }
        }
    }

    /// Total number of Bayesian updates across all states and actions.
    ///
    /// Useful for checking `SimToRealConfig::min_real_fills` threshold.
    pub fn total_updates(&self) -> u64 {
        self.q_table
            .values()
            .flat_map(|actions| actions.iter())
            .map(|q| q.count())
            .sum()
    }

    /// Average of recent rewards, or 0.0 if no rewards recorded.
    ///
    /// Used by `SimToRealConfig::auto_disable_after_fills` to detect
    /// negative-EV RL policies that should be turned off.
    pub fn mean_recent_reward(&self) -> f64 {
        if self.recent_rewards.is_empty() {
            0.0
        } else {
            self.recent_rewards.iter().sum::<f64>() / self.recent_rewards.len() as f64
        }
    }

    /// Serialize Q-table and agent state to checkpoint.
    ///
    /// Only stores (state, action) pairs that have been observed (n > 0)
    /// to keep the checkpoint compact. Default Q-values are reconstructed
    /// on restore for missing entries.
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
        }
    }

    /// Restore Q-table and agent state from checkpoint.
    ///
    /// Recreates `BayesianQValue` entries using `from_checkpoint()` for each
    /// stored entry. States/actions not in the checkpoint get default Q-values.
    pub fn restore_from_checkpoint(&mut self, ckpt: &RLCheckpoint) {
        self.episodes = ckpt.episodes;
        self.total_reward = ckpt.total_reward;

        // Clear and rebuild q_table from checkpoint entries
        self.q_table.clear();
        for entry in &ckpt.q_entries {
            let actions = self
                .q_table
                .entry(entry.state_index)
                .or_insert_with(|| vec![BayesianQValue::new(); MDPAction::ACTION_COUNT]);
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

/// Q-value statistics for a state.
#[derive(Debug, Clone)]
pub struct QValueStats {
    /// Best action according to posterior mean
    pub best_action: MDPAction,
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
#[derive(Debug, Clone)]
pub struct RLPolicyRecommendation {
    /// Recommended spread delta (bps)
    pub spread_delta_bps: f64,
    /// Recommended bid skew (bps)
    pub bid_skew_bps: f64,
    /// Recommended ask skew (bps)
    pub ask_skew_bps: f64,
    /// Confidence in recommendation [0, 1]
    pub confidence: f64,
    /// Whether this is exploration or exploitation
    pub is_exploration: bool,
    /// Underlying MDP action
    pub action: MDPAction,
    /// Expected Q-value
    pub expected_q: f64,
    /// Q-value uncertainty
    pub q_uncertainty: f64,
}

impl RLPolicyRecommendation {
    /// Create from agent action selection.
    pub fn from_agent(
        agent: &mut QLearningAgent,
        state: &MDPState,
        explore: bool,
    ) -> Self {
        let action = if explore {
            agent.select_action(state)
        } else {
            agent.get_greedy_action(state)
        };

        let stats = agent.get_q_stats(state);

        // Confidence based on observations and uncertainty
        let obs_factor = (stats.total_observations as f64 / 100.0).min(1.0);
        let uncertainty_factor = 1.0 / (1.0 + stats.best_q_std);
        let confidence = obs_factor * uncertainty_factor;

        Self {
            spread_delta_bps: action.spread.delta_bps(),
            bid_skew_bps: action.skew.bid_skew_bps(),
            ask_skew_bps: action.skew.ask_skew_bps(),
            confidence,
            is_exploration: explore && action != stats.best_action,
            action,
            expected_q: stats.best_q_mean,
            q_uncertainty: stats.best_q_std,
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
    use std::time::{SystemTime, UNIX_EPOCH};
    // Simple LCG for sampling (replace with proper RNG in production)
    static mut SEED: u64 = 0;
    unsafe {
        if SEED == 0 {
            SEED = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(12345);
        }
        SEED = SEED.wrapping_mul(6364136223846793005).wrapping_add(1);
        (SEED >> 33) as f64 / (1u64 << 31) as f64
    }
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
        assert_eq!(
            ImbalanceBucket::from_imbalance(-0.5),
            ImbalanceBucket::Sell
        );
        assert_eq!(
            ImbalanceBucket::from_imbalance(-0.25),
            ImbalanceBucket::Sell
        );
        assert_eq!(
            ImbalanceBucket::from_imbalance(0.0),
            ImbalanceBucket::Neutral
        );
        assert_eq!(
            ImbalanceBucket::from_imbalance(0.25),
            ImbalanceBucket::Buy
        );
        assert_eq!(
            ImbalanceBucket::from_imbalance(0.5),
            ImbalanceBucket::Buy
        );
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
        };

        let idx1 = state1.to_index();
        let idx2 = state2.to_index();

        assert!(idx1 < MDPState::STATE_COUNT);
        assert!(idx2 < MDPState::STATE_COUNT);
        assert_eq!(MDPState::STATE_COUNT, 675);
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
        let reward = Reward::compute(&config, 2.0, 0.3, 1.0, false);

        assert!(reward.edge_component > 0.0);
        assert!(reward.inventory_penalty <= 0.0);
        assert_eq!(reward.volatility_penalty, 0.0); // vol_ratio = 1.0
        assert_eq!(reward.adverse_penalty, 0.0);    // not adverse
    }

    #[test]
    fn test_reward_adverse_penalty() {
        let config = RewardConfig::default();
        let reward = Reward::compute(&config, -3.0, 0.2, 1.0, true);

        assert!(reward.edge_component < 0.0);
        assert!(reward.adverse_penalty < 0.0);
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
        let mut agent = QLearningAgent::default();
        let state = MDPState::default();
        let action = MDPAction::default();
        let reward = Reward {
            total: 1.0,
            edge_component: 1.0,
            inventory_penalty: 0.0,
            volatility_penalty: 0.0,
            adverse_penalty: 0.0,
        };
        let next_state = MDPState::default();

        agent.update(state, action, reward, next_state, false);

        let summary = agent.summary();
        assert!(summary.total_reward > 0.0);
    }

    #[test]
    fn test_rl_policy_recommendation() {
        let mut agent = QLearningAgent::default();
        let state = MDPState::default();

        let rec = RLPolicyRecommendation::from_agent(&mut agent, &state, false);

        assert!(rec.spread_delta_bps >= -3.0 && rec.spread_delta_bps <= 3.0);
        assert!(rec.confidence >= 0.0 && rec.confidence <= 1.0);
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
        // Test all 125 parameter actions round-trip through index
        for i in 0..ParameterAction::ACTION_COUNT {
            let action = ParameterAction::from_index(i);
            let recovered = action.to_index();
            assert_eq!(i, recovered, "Index {} didn't round-trip", i);
        }
    }

    #[test]
    fn test_parameter_action_count() {
        // 5 × 5 × 5 = 125 actions
        assert_eq!(ParameterAction::ACTION_COUNT, 125);
    }

    #[test]
    fn test_parameter_action_neutral() {
        let neutral = ParameterAction::neutral();
        assert_eq!(neutral.gamma_multiplier(), 1.0);
        assert_eq!(neutral.omega_multiplier(), 1.0);
        assert_eq!(neutral.quote_intensity(), 1.0);
    }

    #[test]
    fn test_parameter_action_defensive() {
        let defensive = ParameterAction::defensive();
        assert!(defensive.gamma_multiplier() > 1.0, "Defensive should have higher γ");
        assert!(defensive.omega_multiplier() > 1.0, "Defensive should have higher skew");
        assert_eq!(defensive.quote_intensity(), 1.0, "Defensive should still quote");
    }

    #[test]
    fn test_parameter_action_cautious() {
        let cautious = ParameterAction::cautious();
        assert_eq!(cautious.quote_intensity(), 0.0, "Cautious should not quote");
    }
}
