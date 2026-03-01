//! Multi-channel directional conviction fusion.
//!
//! Fuses 5 independent directional channels into a single conviction score
//! using log-odds aggregation with independence weighting. Channels:
//!
//! 1. Kalman drift posterior (μ/σ → p_bearish)
//! 2. Multi-TF trend agreement (persistence across timescales)
//! 3. Hawkes sell/buy imbalance (self-exciting intensity asymmetry)
//! 4. Fill toxicity (fraction of recent fills that are adverse)
//! 5. OI/funding pressure (OI drops + extreme funding)
//!
//! Output feeds graduated defense (gamma escalation, margin shift, reduce-only),
//! proactive ask/bid tightening, and conviction-weighted position targeting.

/// Number of directional channels in the fusion.
pub const NUM_CHANNELS: usize = 5;

/// Maximum absolute log-odds from any single channel.
/// Prevents a single high-confidence channel from dominating the fusion.
const DEFAULT_LOG_ODDS_CAP: f64 = 3.0;

/// Minimum |log-odds| for a channel to count as corroborating.
const DEFAULT_CORROBORATION_THRESHOLD: f64 = 0.3;

/// Minimum number of agreeing channels before conviction is actionable.
const DEFAULT_MIN_CHANNELS_FOR_CONVICTION: usize = 2;

/// Default info coefficients τ per channel:
/// [kalman_drift, multi_tf_trend, hawkes_imbalance, fill_toxicity, oi_funding]
const DEFAULT_CHANNEL_WEIGHTS: [f64; NUM_CHANNELS] = [1.0, 0.8, 0.6, 0.5, 0.4];

/// Configuration for the directional conviction system.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ConvictionConfig {
    /// Maximum absolute log-odds from any single channel.
    /// Prevents single-channel domination. Default: 3.0.
    #[serde(default = "default_log_odds_cap")]
    pub log_odds_cap: f64,

    /// Minimum |L_k| for a channel to count as corroborating. Default: 0.3.
    #[serde(default = "default_corroboration_threshold")]
    pub corroboration_threshold: f64,

    /// Minimum agreeing channels for conviction to be actionable. Default: 2.
    #[serde(default = "default_min_channels")]
    pub min_channels_for_conviction: usize,

    /// Per-channel info coefficients τ_k. Higher = more weight in fusion.
    /// Order: [kalman_drift, multi_tf_trend, hawkes_imbalance, fill_toxicity, oi_funding]
    #[serde(default = "default_channel_weights")]
    pub channel_weights: [f64; NUM_CHANNELS],

    // === Graduated Defense Thresholds ===
    /// Conviction score threshold for gamma escalation (Tier 1). Default: 0.55.
    #[serde(default = "default_gamma_escalation_threshold")]
    pub gamma_escalation_threshold: f64,

    /// Conviction score threshold for margin shift (Tier 2). Default: 0.70.
    #[serde(default = "default_margin_shift_threshold")]
    pub margin_shift_threshold: f64,

    /// Conviction score threshold for reduce-only (Tier 3). Default: 0.85.
    #[serde(default = "default_reduce_only_threshold")]
    pub reduce_only_threshold: f64,

    /// Maximum gamma multiplier at extreme conviction. Default: 1.5.
    #[serde(default = "default_max_gamma_mult")]
    pub max_gamma_mult: f64,

    /// Maximum margin shift magnitude. Default: 0.15.
    #[serde(default = "default_max_margin_shift")]
    pub max_margin_shift: f64,

    /// Maximum ask tightening fraction. Default: 0.40.
    #[serde(default = "default_max_ask_tightening")]
    pub max_ask_tightening: f64,

    /// Ask tightening activation threshold. Default: 0.60.
    #[serde(default = "default_ask_tightening_threshold")]
    pub ask_tightening_threshold: f64,

    /// Conviction blending weight for q_target. Default: 0.30.
    #[serde(default = "default_conviction_q_blend")]
    pub conviction_q_blend: f64,

    /// Max fraction of max_inventory for conviction position. Default: 0.50.
    #[serde(default = "default_max_conviction_position_frac")]
    pub max_conviction_position_frac: f64,

    /// Corroboration relaxation strength for adaptive shrinkage. Default: 0.50.
    #[serde(default = "default_corroboration_kappa")]
    pub corroboration_kappa: f64,
}

fn default_log_odds_cap() -> f64 {
    DEFAULT_LOG_ODDS_CAP
}
fn default_corroboration_threshold() -> f64 {
    DEFAULT_CORROBORATION_THRESHOLD
}
fn default_min_channels() -> usize {
    DEFAULT_MIN_CHANNELS_FOR_CONVICTION
}
fn default_channel_weights() -> [f64; NUM_CHANNELS] {
    DEFAULT_CHANNEL_WEIGHTS
}
fn default_gamma_escalation_threshold() -> f64 {
    0.55
}
fn default_margin_shift_threshold() -> f64 {
    0.70
}
fn default_reduce_only_threshold() -> f64 {
    0.85
}
fn default_max_gamma_mult() -> f64 {
    1.5
}
fn default_max_margin_shift() -> f64 {
    0.15
}
fn default_max_ask_tightening() -> f64 {
    0.40
}
fn default_ask_tightening_threshold() -> f64 {
    0.60
}
fn default_conviction_q_blend() -> f64 {
    0.30
}
fn default_max_conviction_position_frac() -> f64 {
    0.50
}
fn default_corroboration_kappa() -> f64 {
    0.50
}

impl Default for ConvictionConfig {
    fn default() -> Self {
        Self {
            log_odds_cap: DEFAULT_LOG_ODDS_CAP,
            corroboration_threshold: DEFAULT_CORROBORATION_THRESHOLD,
            min_channels_for_conviction: DEFAULT_MIN_CHANNELS_FOR_CONVICTION,
            channel_weights: DEFAULT_CHANNEL_WEIGHTS,
            gamma_escalation_threshold: 0.55,
            margin_shift_threshold: 0.70,
            reduce_only_threshold: 0.85,
            max_gamma_mult: 1.5,
            max_margin_shift: 0.15,
            max_ask_tightening: 0.40,
            ask_tightening_threshold: 0.60,
            conviction_q_blend: 0.30,
            max_conviction_position_frac: 0.50,
            corroboration_kappa: 0.50,
        }
    }
}

/// Point-in-time conviction snapshot from multi-channel fusion.
#[derive(Debug, Clone, Default)]
pub struct ConvictionSnapshot {
    /// Conviction strength [0, 1]. 0 = no directional conviction, 1 = maximum.
    pub score: f64,
    /// Conviction direction: -1.0 (bearish) to +1.0 (bullish).
    pub direction: f64,
    /// Number of channels agreeing with the dominant direction above threshold.
    pub corroboration_count: usize,
    /// Raw unified log-odds before sigmoid transform.
    pub unified_log_odds: f64,
    /// Per-channel weighted log-odds contributions.
    /// Order: [kalman_drift, multi_tf_trend, hawkes_imbalance, fill_toxicity, oi_funding]
    pub channel_contributions: [f64; NUM_CHANNELS],
}

/// Input probabilities for each directional channel.
///
/// Each value is P(bearish) ∈ [0, 1]. p = 0.5 means no directional signal.
/// p > 0.5 = bearish, p < 0.5 = bullish.
#[derive(Debug, Clone, Default)]
pub struct ChannelInputs {
    /// P(bearish) from Kalman drift posterior: Φ(μ/σ).
    pub p_bearish_kalman: f64,
    /// P(bearish) from multi-TF trend: agreement across 3 timescales.
    pub p_bearish_trend: f64,
    /// P(bearish) from Hawkes imbalance: sell intensity > buy intensity.
    pub p_bearish_hawkes: f64,
    /// P(bearish) from fill toxicity: fraction of recent fills adverse on bid side.
    pub p_bearish_fill_toxicity: f64,
    /// P(bearish) from OI/funding: OI drops + extreme positive funding.
    pub p_bearish_oi_funding: f64,
}

impl ChannelInputs {
    /// Return channel probabilities as an array for iteration.
    fn as_array(&self) -> [f64; NUM_CHANNELS] {
        [
            self.p_bearish_kalman,
            self.p_bearish_trend,
            self.p_bearish_hawkes,
            self.p_bearish_fill_toxicity,
            self.p_bearish_oi_funding,
        ]
    }
}

/// Compute multi-channel directional conviction via log-odds fusion.
///
/// Algorithm:
/// 1. Transform each channel's P(bearish) to log-odds: L_k = ln(p / (1-p))
/// 2. Cap each |L_k| at log_odds_cap to prevent single-channel domination
/// 3. Weight by normalized info coefficients: w_k = τ_k / Σ τ_j
/// 4. Sum: L_unified = Σ w_k × L_k
/// 5. Back to probability: p_unified = sigmoid(L_unified)
/// 6. Score = |2 × p_unified - 1|, direction = sign(L_unified)
/// 7. Corroboration = count of channels agreeing with direction above threshold
pub fn compute_conviction(inputs: &ChannelInputs, config: &ConvictionConfig) -> ConvictionSnapshot {
    let probs = inputs.as_array();
    let cap = config.log_odds_cap;

    // Normalize weights
    let weight_sum: f64 = config.channel_weights.iter().sum();
    if weight_sum < 1e-12 {
        return ConvictionSnapshot::default();
    }

    let mut channel_log_odds = [0.0f64; NUM_CHANNELS];
    let mut channel_contributions = [0.0f64; NUM_CHANNELS];

    for i in 0..NUM_CHANNELS {
        let p = probs[i].clamp(0.001, 0.999); // Prevent log(0) and log(inf)
        let log_odds = (p / (1.0 - p)).ln();
        let capped = log_odds.clamp(-cap, cap);
        channel_log_odds[i] = capped;

        let w = config.channel_weights[i] / weight_sum;
        channel_contributions[i] = w * capped;
    }

    let unified_log_odds: f64 = channel_contributions.iter().sum();

    // Sigmoid: p_unified = 1 / (1 + exp(-L))
    let p_unified = 1.0 / (1.0 + (-unified_log_odds).exp());

    // Conviction score: how far from 0.5 (undecided)
    let score = (2.0 * p_unified - 1.0).abs();

    // Direction: negative log-odds = bullish (p_bearish < 0.5)
    let direction = if unified_log_odds > 0.0 {
        -1.0 // Bearish (p_bearish > 0.5 → L > 0)
    } else if unified_log_odds < 0.0 {
        1.0 // Bullish
    } else {
        0.0
    };

    // Count corroborating channels: same direction as unified AND above threshold
    let corroboration_count = (0..NUM_CHANNELS)
        .filter(|&i| {
            let agrees = (channel_log_odds[i] > 0.0 && direction < 0.0)
                || (channel_log_odds[i] < 0.0 && direction > 0.0);
            agrees && channel_log_odds[i].abs() > config.corroboration_threshold
        })
        .count();

    ConvictionSnapshot {
        score,
        direction,
        corroboration_count,
        unified_log_odds,
        channel_contributions,
    }
}

/// Compute graduated defense parameters from conviction snapshot.
///
/// Returns (gamma_mult, margin_shift, ask_tightening, is_reduce_only).
pub fn graduated_defense(
    conviction: &ConvictionSnapshot,
    config: &ConvictionConfig,
) -> (f64, f64, f64, bool) {
    let score = conviction.score;
    let dir = conviction.direction;
    let corr = conviction.corroboration_count;

    // Require minimum corroboration for any action
    if corr < config.min_channels_for_conviction {
        return (1.0, 0.0, 0.0, false);
    }

    // Tier 1: Gamma escalation (wider spreads)
    let gamma_mult = if score > config.gamma_escalation_threshold {
        let t = ((score - config.gamma_escalation_threshold)
            / (1.0 - config.gamma_escalation_threshold))
            .min(1.0);
        1.0 + (config.max_gamma_mult - 1.0) * t
    } else {
        1.0
    };

    // Tier 2: Margin shift (shift allocation toward reducing side)
    let margin_shift = if score > config.margin_shift_threshold {
        let t = ((score - config.margin_shift_threshold) / (1.0 - config.margin_shift_threshold))
            .min(1.0);
        // dir < 0 (bearish): shift away from bids → negative shift
        // dir > 0 (bullish): shift away from asks → positive shift
        -dir * config.max_margin_shift * t
    } else {
        0.0
    };

    // Tier 3: Reduce-only on accumulating side
    let is_reduce_only = score > config.reduce_only_threshold;

    // Ask/bid tightening on informed side
    let ask_tightening = if score > config.ask_tightening_threshold && corr >= 2 {
        let t = ((score - config.ask_tightening_threshold)
            / (1.0 - config.ask_tightening_threshold))
            .min(1.0);
        // Bearish: tighten asks (we want to sell). Bullish: tighten bids (we want to buy).
        // Return magnitude; caller uses direction to decide which side.
        config.max_ask_tightening * t
    } else {
        0.0
    };

    (gamma_mult, margin_shift, ask_tightening, is_reduce_only)
}

/// Compute conviction-weighted position target.
///
/// Blends base CJ q_target with conviction-derived directional target.
/// q_conviction = direction × score × max_fraction_of_inventory
/// q_blended = (1 - α) × q_base + α × q_conviction
pub fn conviction_q_target(
    q_base: f64,
    conviction: &ConvictionSnapshot,
    config: &ConvictionConfig,
    max_inventory: f64,
) -> f64 {
    if conviction.corroboration_count < config.min_channels_for_conviction {
        return q_base;
    }

    let q_conviction =
        conviction.direction * conviction.score * config.max_conviction_position_frac;
    let alpha = config.conviction_q_blend;
    let blended = (1.0 - alpha) * q_base + alpha * q_conviction;

    // Clamp to [-1, 1] (normalized by max_inventory)
    let _ = max_inventory; // Used conceptually; clamping handles the bound
    blended.clamp(-1.0, 1.0)
}

/// Compute corroboration score for adaptive shrinkage.
/// Returns corroboration_count / NUM_CHANNELS ∈ [0, 1].
pub fn corroboration_score(conviction: &ConvictionSnapshot) -> f64 {
    conviction.corroboration_count as f64 / NUM_CHANNELS as f64
}

// ============================================================================
// Channel Names (for logging)
// ============================================================================

/// Human-readable channel names for diagnostics.
pub const CHANNEL_NAMES: [&str; NUM_CHANNELS] = [
    "kalman_drift",
    "multi_tf_trend",
    "hawkes_imbalance",
    "fill_toxicity",
    "oi_funding",
];

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> ConvictionConfig {
        ConvictionConfig::default()
    }

    #[test]
    fn test_neutral_inputs_produce_zero_conviction() {
        let inputs = ChannelInputs {
            p_bearish_kalman: 0.5,
            p_bearish_trend: 0.5,
            p_bearish_hawkes: 0.5,
            p_bearish_fill_toxicity: 0.5,
            p_bearish_oi_funding: 0.5,
        };
        let result = compute_conviction(&inputs, &default_config());
        assert!(result.score < 0.01, "Neutral inputs should give ~0 score");
        assert_eq!(result.corroboration_count, 0);
    }

    #[test]
    fn test_all_bearish_channels_high_conviction() {
        // During a real 50bps selloff, channels produce strong signals
        let inputs = ChannelInputs {
            p_bearish_kalman: 0.95,
            p_bearish_trend: 0.90,
            p_bearish_hawkes: 0.85,
            p_bearish_fill_toxicity: 0.80,
            p_bearish_oi_funding: 0.75,
        };
        let result = compute_conviction(&inputs, &default_config());
        assert!(
            result.score > 0.70,
            "All-bearish should give high score: {}",
            result.score
        );
        assert!(result.direction < 0.0, "Should be bearish");
        assert!(
            result.corroboration_count >= 3,
            "Multiple channels should corroborate"
        );
    }

    #[test]
    fn test_all_bullish_channels_high_conviction() {
        // Mirror of bearish: strong bullish signals across all channels
        let inputs = ChannelInputs {
            p_bearish_kalman: 0.05,
            p_bearish_trend: 0.10,
            p_bearish_hawkes: 0.15,
            p_bearish_fill_toxicity: 0.20,
            p_bearish_oi_funding: 0.25,
        };
        let result = compute_conviction(&inputs, &default_config());
        assert!(
            result.score > 0.70,
            "All-bullish should give high score: {}",
            result.score
        );
        assert!(result.direction > 0.0, "Should be bullish");
        assert!(result.corroboration_count >= 3);
    }

    #[test]
    fn test_mixed_channels_low_conviction() {
        let inputs = ChannelInputs {
            p_bearish_kalman: 0.80,        // bearish
            p_bearish_trend: 0.20,         // bullish
            p_bearish_hawkes: 0.50,        // neutral
            p_bearish_fill_toxicity: 0.60, // slight bearish
            p_bearish_oi_funding: 0.40,    // slight bullish
        };
        let result = compute_conviction(&inputs, &default_config());
        assert!(
            result.score < 0.50,
            "Mixed signals should give moderate score: {}",
            result.score
        );
    }

    #[test]
    fn test_log_odds_cap_prevents_domination() {
        // One extreme channel + 4 neutral should not create extreme conviction
        let inputs = ChannelInputs {
            p_bearish_kalman: 0.999,
            p_bearish_trend: 0.5,
            p_bearish_hawkes: 0.5,
            p_bearish_fill_toxicity: 0.5,
            p_bearish_oi_funding: 0.5,
        };
        let result = compute_conviction(&inputs, &default_config());
        // With cap=3.0, one channel at 0.999 gives L=6.9 → capped to 3.0
        // Weighted: 1.0/3.3 * 3.0 = 0.91 log-odds → p = 0.71 → score = 0.42
        assert!(
            result.score < 0.60,
            "Cap should prevent single-channel domination: {}",
            result.score
        );
        assert_eq!(
            result.corroboration_count, 1,
            "Only one channel should corroborate"
        );
    }

    #[test]
    fn test_graduated_defense_tiers() {
        let config = default_config();

        // Below threshold: no defense
        let low = ConvictionSnapshot {
            score: 0.40,
            direction: -1.0,
            corroboration_count: 3,
            ..Default::default()
        };
        let (gamma, margin, tightening, reduce) = graduated_defense(&low, &config);
        assert!((gamma - 1.0).abs() < 0.01);
        assert!(margin.abs() < 0.01);
        assert!(tightening.abs() < 0.01);
        assert!(!reduce);

        // Moderate: gamma escalation + ask tightening (both active above 0.55/0.60)
        let moderate = ConvictionSnapshot {
            score: 0.65,
            direction: -1.0,
            corroboration_count: 3,
            ..Default::default()
        };
        let (gamma, margin, tightening, reduce) = graduated_defense(&moderate, &config);
        assert!(gamma > 1.0, "Should escalate gamma: {}", gamma);
        assert!(margin.abs() < 0.01, "Below margin threshold");
        assert!(tightening > 0.0, "Should tighten asks");
        assert!(!reduce);

        // High: gamma + margin shift
        let high = ConvictionSnapshot {
            score: 0.80,
            direction: -1.0,
            corroboration_count: 4,
            ..Default::default()
        };
        let (gamma, margin, _tightening, reduce) = graduated_defense(&high, &config);
        assert!(gamma > 1.1, "Strong gamma escalation: {}", gamma);
        assert!(
            margin > 0.0,
            "Bearish → positive margin shift (away from bids): {}",
            margin
        );
        assert!(!reduce);

        // Extreme: all tiers including reduce-only
        let extreme = ConvictionSnapshot {
            score: 0.90,
            direction: -1.0,
            corroboration_count: 5,
            ..Default::default()
        };
        let (gamma, margin, tightening, reduce) = graduated_defense(&extreme, &config);
        assert!(gamma > 1.3);
        assert!(margin > 0.05);
        assert!(tightening > 0.2);
        assert!(reduce, "Should be reduce-only at extreme conviction");
    }

    #[test]
    fn test_corroboration_gate() {
        let config = default_config();

        // High conviction but only 1 channel → no defense
        let ungated = ConvictionSnapshot {
            score: 0.90,
            direction: -1.0,
            corroboration_count: 1,
            ..Default::default()
        };
        let (gamma, margin, _, reduce) = graduated_defense(&ungated, &config);
        assert!((gamma - 1.0).abs() < 0.01, "Should be gated: {}", gamma);
        assert!(margin.abs() < 0.01);
        assert!(!reduce);
    }

    #[test]
    fn test_conviction_q_target_blending() {
        let config = default_config();

        // No conviction: q_base passes through
        let neutral = ConvictionSnapshot::default();
        let q = conviction_q_target(0.3, &neutral, &config, 50.0);
        assert!((q - 0.3).abs() < 0.01);

        // Strong bearish conviction: pulls q_target negative
        let bearish = ConvictionSnapshot {
            score: 0.80,
            direction: -1.0,
            corroboration_count: 3,
            ..Default::default()
        };
        let q = conviction_q_target(0.0, &bearish, &config, 50.0);
        // q_conviction = -1.0 * 0.8 * 0.5 = -0.4
        // blended = 0.7 * 0.0 + 0.3 * (-0.4) = -0.12
        assert!(
            q < -0.05,
            "Bearish conviction should pull q negative: {}",
            q
        );
    }

    #[test]
    fn test_corroboration_score_fraction() {
        let snap = ConvictionSnapshot {
            corroboration_count: 3,
            ..Default::default()
        };
        let cs = corroboration_score(&snap);
        assert!((cs - 0.6).abs() < 0.01);
    }
}
