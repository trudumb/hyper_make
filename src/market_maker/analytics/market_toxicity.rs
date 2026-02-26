//! Market toxicity composite scorer.
//!
//! Combines multiple public-data signals (VPIN, informed flow, trend, book imbalance,
//! price velocity) into a single [0,1] toxicity score with derived spread multiplier
//! and directional skew.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// All tunables for the toxicity composite. Every field has `#[serde(default)]`
/// so that old checkpoints deserialize cleanly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketToxicityConfig {
    // --- component weights (must sum to 1.0) ---
    #[serde(default = "default_w_vpin")]
    pub weight_vpin: f64,
    #[serde(default = "default_w_informed")]
    pub weight_informed: f64,
    #[serde(default = "default_w_trend")]
    pub weight_trend: f64,
    #[serde(default = "default_w_book")]
    pub weight_book: f64,
    #[serde(default = "default_w_velocity")]
    pub weight_velocity: f64,

    // --- per-signal mapping thresholds ---
    /// VPIN value at which sigmoid midpoint sits.
    #[serde(default = "default_vpin_mid")]
    pub vpin_sigmoid_mid: f64,
    /// Steepness of the VPIN sigmoid.
    #[serde(default = "default_vpin_k")]
    pub vpin_sigmoid_k: f64,

    /// P(informed) sigmoid midpoint.
    #[serde(default = "default_informed_mid")]
    pub informed_sigmoid_mid: f64,
    #[serde(default = "default_informed_k")]
    pub informed_sigmoid_k: f64,

    /// Trend magnitude (absolute bps) at which ramp saturates.
    #[serde(default = "default_trend_sat_bps")]
    pub trend_saturation_bps: f64,

    /// Book-imbalance absolute value at which ramp saturates.
    #[serde(default = "default_book_sat")]
    pub book_imbalance_saturation: f64,

    /// Price velocity (bps/s) at which ramp saturates.
    #[serde(default = "default_vel_sat")]
    pub velocity_saturation_bps_per_s: f64,

    // --- composite → spread mapping ---
    /// Toxicity below this produces multiplier = 1.0.
    #[serde(default = "default_min_tox")]
    pub min_toxicity: f64,
    /// Maximum spread multiplier at toxicity = 1.0.
    #[serde(default = "default_max_mult")]
    pub max_spread_multiplier: f64,
    /// Hard cap on composed spread multiplier (safety).
    #[serde(default = "default_max_composed")]
    pub max_composed_spread_mult: f64,

    // --- cold-start ---
    /// Fraction of invalid signals that triggers cold-start baseline.
    #[serde(default = "default_cold_frac")]
    pub cold_start_invalid_fraction: f64,
    /// Baseline multiplier during cold-start.
    #[serde(default = "default_cold_mult")]
    pub cold_start_multiplier: f64,

    // --- skew ---
    /// Maximum absolute skew in bps.
    #[serde(default = "default_max_skew")]
    pub max_skew_bps: f64,
    /// Weight of trend component in skew direction (rest is book).
    #[serde(default = "default_skew_trend_w")]
    pub skew_trend_weight: f64,
}

// --- serde default helpers ---
fn default_w_vpin() -> f64 {
    0.25
}
fn default_w_informed() -> f64 {
    0.25
}
fn default_w_trend() -> f64 {
    0.20
}
fn default_w_book() -> f64 {
    0.15
}
fn default_w_velocity() -> f64 {
    0.15
}
fn default_vpin_mid() -> f64 {
    0.55
}
fn default_vpin_k() -> f64 {
    15.0
}
fn default_informed_mid() -> f64 {
    0.40
}
fn default_informed_k() -> f64 {
    12.0
}
fn default_trend_sat_bps() -> f64 {
    30.0
}
fn default_book_sat() -> f64 {
    0.70
}
fn default_vel_sat() -> f64 {
    10.0
}
fn default_min_tox() -> f64 {
    0.3
}
fn default_max_mult() -> f64 {
    5.0
}
fn default_max_composed() -> f64 {
    10.0
}
fn default_cold_frac() -> f64 {
    0.35
}
fn default_cold_mult() -> f64 {
    1.5
}
fn default_max_skew() -> f64 {
    2.0
}
fn default_skew_trend_w() -> f64 {
    0.60
}

impl Default for MarketToxicityConfig {
    fn default() -> Self {
        Self {
            weight_vpin: default_w_vpin(),
            weight_informed: default_w_informed(),
            weight_trend: default_w_trend(),
            weight_book: default_w_book(),
            weight_velocity: default_w_velocity(),
            vpin_sigmoid_mid: default_vpin_mid(),
            vpin_sigmoid_k: default_vpin_k(),
            informed_sigmoid_mid: default_informed_mid(),
            informed_sigmoid_k: default_informed_k(),
            trend_saturation_bps: default_trend_sat_bps(),
            book_imbalance_saturation: default_book_sat(),
            velocity_saturation_bps_per_s: default_vel_sat(),
            min_toxicity: default_min_tox(),
            max_spread_multiplier: default_max_mult(),
            max_composed_spread_mult: default_max_composed(),
            cold_start_invalid_fraction: default_cold_frac(),
            cold_start_multiplier: default_cold_mult(),
            max_skew_bps: default_max_skew(),
            skew_trend_weight: default_skew_trend_w(),
        }
    }
}

// ---------------------------------------------------------------------------
// Input / Output types
// ---------------------------------------------------------------------------

/// All signal values needed for one toxicity evaluation cycle.
/// `None` means the signal is invalid or not yet warmed up.
#[derive(Debug, Clone)]
pub struct ToxicityInput {
    /// VPIN [0,1]. None = invalid/unwarmed.
    pub vpin: Option<f64>,
    /// Rate of change of VPIN (units/s). None = invalid.
    pub vpin_velocity: Option<f64>,
    /// Probability of informed flow [0,1].
    pub p_informed: f64,
    /// Long-window trend magnitude in bps. None = unwarmed.
    pub trend_long_bps: Option<f64>,
    /// Fraction of trend timeframes agreeing [0,1]. None = unwarmed.
    pub trend_agreement: Option<f64>,
    /// L2 book imbalance [-1, 1].
    pub book_imbalance: f64,
    /// Recent price velocity in bps/s.
    pub price_velocity_1s: f64,
}

/// Per-signal [0,1] toxicity scores.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToxicityComponents {
    pub vpin: f64,
    pub informed: f64,
    pub trend: f64,
    pub book_imbalance: f64,
    pub velocity: f64,
}

/// Result of a single toxicity evaluation.
#[derive(Debug, Clone)]
pub struct ToxicityAssessment {
    /// Weighted composite score in [0, 1].
    pub composite_score: f64,
    /// Spread multiplier (>= 1.0).
    pub spread_multiplier: f64,
    /// Additive directional skew in bps.
    pub skew_bps: f64,
    /// Breakdown per signal.
    pub components: ToxicityComponents,
    /// True when too many signals are invalid.
    pub cold_start: bool,
}

// ---------------------------------------------------------------------------
// Core scorer
// ---------------------------------------------------------------------------

/// Continuous market toxicity composite.
///
/// Replaces the binary `negative_edge_alarm()` with a smooth, weighted score
/// derived entirely from public-data signals.
#[derive(Debug, Clone)]
pub struct MarketToxicityComposite {
    config: MarketToxicityConfig,
}

impl MarketToxicityComposite {
    pub fn new(config: MarketToxicityConfig) -> Self {
        Self { config }
    }

    /// Get the current configuration.
    pub fn config(&self) -> &MarketToxicityConfig {
        &self.config
    }

    /// Evaluate current toxicity from input signals.
    pub fn evaluate(&self, input: &ToxicityInput) -> ToxicityAssessment {
        let cfg = &self.config;

        // ---- per-signal component mapping ----
        let (vpin_score, vpin_valid) = match input.vpin {
            Some(v) => (sigmoid(v, cfg.vpin_sigmoid_mid, cfg.vpin_sigmoid_k), true),
            None => (0.0, false),
        };

        let informed_score = sigmoid(
            input.p_informed,
            cfg.informed_sigmoid_mid,
            cfg.informed_sigmoid_k,
        );

        let (trend_score, trend_valid) = match input.trend_long_bps {
            Some(bps) => {
                let base = ramp(bps.abs(), cfg.trend_saturation_bps);
                let agreement_boost = input.trend_agreement.unwrap_or(0.0);
                // Amplify when multiple timeframes agree.
                let score = (base * (0.5 + 0.5 * agreement_boost)).clamp(0.0, 1.0);
                (score, true)
            }
            None => (0.0, false),
        };

        let book_score = ramp(input.book_imbalance.abs(), cfg.book_imbalance_saturation);

        let vel_score = ramp(
            input.price_velocity_1s.abs(),
            cfg.velocity_saturation_bps_per_s,
        );

        // ---- cold-start detection ----
        let total_signals = 5u32;
        let invalid_count = [!vpin_valid, false, !trend_valid, false, false]
            .iter()
            .filter(|&&b| b)
            .count() as u32;
        let cold_start =
            (invalid_count as f64 / total_signals as f64) > cfg.cold_start_invalid_fraction;

        let components = ToxicityComponents {
            vpin: vpin_score,
            informed: informed_score,
            trend: trend_score,
            book_imbalance: book_score,
            velocity: vel_score,
        };

        // ---- weighted composite ----
        let raw_composite = cfg.weight_vpin * vpin_score
            + cfg.weight_informed * informed_score
            + cfg.weight_trend * trend_score
            + cfg.weight_book * book_score
            + cfg.weight_velocity * vel_score;
        let composite_score = raw_composite.clamp(0.0, 1.0);

        // ---- spread multiplier ----
        let spread_multiplier = if cold_start {
            cfg.cold_start_multiplier
        } else if composite_score <= cfg.min_toxicity {
            1.0
        } else {
            let t = (composite_score - cfg.min_toxicity) / (1.0 - cfg.min_toxicity);
            let mult = 1.0 + t * (cfg.max_spread_multiplier - 1.0);
            mult.min(cfg.max_composed_spread_mult)
        };

        // ---- directional skew ----
        let trend_dir = input.trend_long_bps.unwrap_or(0.0).signum();
        let book_dir = input.book_imbalance.signum();
        let raw_skew = cfg.skew_trend_weight * trend_dir * trend_score
            + (1.0 - cfg.skew_trend_weight) * book_dir * book_score;
        let skew_bps = (raw_skew * cfg.max_skew_bps).clamp(-cfg.max_skew_bps, cfg.max_skew_bps);

        log::debug!(
            "toxicity: composite={composite_score:.3} mult={spread_multiplier:.2} skew={skew_bps:.2}bps cold={cold_start} | \
             vpin={vpin_score:.2} informed={informed_score:.2} trend={trend_score:.2} book={book_score:.2} vel={vel_score:.2}"
        );

        ToxicityAssessment {
            composite_score,
            spread_multiplier,
            skew_bps,
            components,
            cold_start,
        }
    }
}

// ---------------------------------------------------------------------------
// Mapping helpers
// ---------------------------------------------------------------------------

/// Standard sigmoid mapping: `1 / (1 + exp(-k * (x - mid)))`.
fn sigmoid(x: f64, mid: f64, k: f64) -> f64 {
    let z = -k * (x - mid);
    // Guard against overflow in exp.
    if z > 500.0 {
        return 0.0;
    }
    if z < -500.0 {
        return 1.0;
    }
    1.0 / (1.0 + z.exp())
}

/// Linear ramp from 0 to 1 as `x` goes from 0 to `saturation`, clamped.
fn ramp(x: f64, saturation: f64) -> f64 {
    if saturation <= 0.0 {
        return 1.0;
    }
    (x / saturation).clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_input() -> ToxicityInput {
        ToxicityInput {
            vpin: Some(0.3),
            vpin_velocity: Some(0.0),
            p_informed: 0.1,
            trend_long_bps: Some(0.0),
            trend_agreement: Some(0.0),
            book_imbalance: 0.0,
            price_velocity_1s: 0.0,
        }
    }

    fn scorer() -> MarketToxicityComposite {
        MarketToxicityComposite::new(MarketToxicityConfig::default())
    }

    #[test]
    fn test_benign_market_below_threshold() {
        let s = scorer();
        let result = s.evaluate(&default_input());
        assert!(
            result.composite_score < 0.3,
            "benign market should be below min_toxicity"
        );
        assert!(
            (result.spread_multiplier - 1.0).abs() < 1e-9,
            "multiplier should be 1.0 when below threshold"
        );
        assert!(!result.cold_start);
    }

    #[test]
    fn test_high_vpin_raises_score() {
        let s = scorer();
        let mut input = default_input();
        input.vpin = Some(0.90);
        let result = s.evaluate(&input);
        assert!(
            result.components.vpin > 0.9,
            "vpin component should be high for VPIN=0.9"
        );
        assert!(
            result.composite_score > 0.2,
            "composite should rise with high VPIN"
        );
    }

    #[test]
    fn test_high_informed_raises_score() {
        let s = scorer();
        let mut input = default_input();
        input.p_informed = 0.85;
        let result = s.evaluate(&input);
        assert!(
            result.components.informed > 0.9,
            "informed component should be high"
        );
    }

    #[test]
    fn test_trend_component_with_agreement() {
        let s = scorer();
        let mut input = default_input();
        input.trend_long_bps = Some(25.0);
        input.trend_agreement = Some(1.0);
        let result = s.evaluate(&input);
        assert!(
            result.components.trend > 0.5,
            "strong trend with agreement should produce high component"
        );
    }

    #[test]
    fn test_book_imbalance_component() {
        let s = scorer();
        let mut input = default_input();
        input.book_imbalance = 0.7;
        let result = s.evaluate(&input);
        assert!(
            (result.components.book_imbalance - 1.0).abs() < 1e-9,
            "book at saturation should score 1.0"
        );
    }

    #[test]
    fn test_velocity_component() {
        let s = scorer();
        let mut input = default_input();
        input.price_velocity_1s = 5.0;
        let result = s.evaluate(&input);
        assert!(
            (result.components.velocity - 0.5).abs() < 0.01,
            "5 bps/s with 10 bps/s saturation should be ~0.5"
        );
    }

    #[test]
    fn test_cold_start_with_many_invalid() {
        let s = scorer();
        let input = ToxicityInput {
            vpin: None,
            vpin_velocity: None,
            p_informed: 0.1,
            trend_long_bps: None,
            trend_agreement: None,
            book_imbalance: 0.0,
            price_velocity_1s: 0.0,
        };
        let result = s.evaluate(&input);
        assert!(
            result.cold_start,
            "should be cold-start with VPIN and trend invalid"
        );
        assert!(
            (result.spread_multiplier - 1.5).abs() < 1e-9,
            "cold-start multiplier should be 1.5"
        );
    }

    #[test]
    fn test_spread_multiplier_cap() {
        let cfg = MarketToxicityConfig {
            max_spread_multiplier: 20.0,
            max_composed_spread_mult: 10.0,
            ..Default::default()
        };
        let s = MarketToxicityComposite::new(cfg);
        let input = ToxicityInput {
            vpin: Some(1.0),
            vpin_velocity: Some(10.0),
            p_informed: 1.0,
            trend_long_bps: Some(100.0),
            trend_agreement: Some(1.0),
            book_imbalance: 1.0,
            price_velocity_1s: 50.0,
        };
        let result = s.evaluate(&input);
        assert!(
            result.spread_multiplier <= 10.0,
            "multiplier must respect max_composed_spread_mult cap: got {}",
            result.spread_multiplier
        );
    }

    #[test]
    fn test_composite_clamped_at_1() {
        let s = scorer();
        let input = ToxicityInput {
            vpin: Some(1.0),
            vpin_velocity: Some(10.0),
            p_informed: 1.0,
            trend_long_bps: Some(100.0),
            trend_agreement: Some(1.0),
            book_imbalance: 1.0,
            price_velocity_1s: 50.0,
        };
        let result = s.evaluate(&input);
        assert!(
            result.composite_score <= 1.0,
            "composite must be clamped to 1.0: got {}",
            result.composite_score
        );
    }

    #[test]
    fn test_skew_sign_follows_trend_and_book() {
        let s = scorer();
        // Positive trend + positive book imbalance → positive skew.
        let mut input = default_input();
        input.trend_long_bps = Some(20.0);
        input.trend_agreement = Some(0.8);
        input.book_imbalance = 0.5;
        let result = s.evaluate(&input);
        assert!(
            result.skew_bps > 0.0,
            "skew should be positive: {}",
            result.skew_bps
        );

        // Negative trend + negative book → negative skew.
        input.trend_long_bps = Some(-20.0);
        input.book_imbalance = -0.5;
        let neg = s.evaluate(&input);
        assert!(
            neg.skew_bps < 0.0,
            "skew should be negative: {}",
            neg.skew_bps
        );
    }

    #[test]
    fn test_skew_capped_at_max() {
        let s = scorer();
        let input = ToxicityInput {
            vpin: Some(0.9),
            vpin_velocity: Some(5.0),
            p_informed: 0.9,
            trend_long_bps: Some(100.0),
            trend_agreement: Some(1.0),
            book_imbalance: 1.0,
            price_velocity_1s: 20.0,
        };
        let result = s.evaluate(&input);
        assert!(
            result.skew_bps.abs() <= 2.0 + 1e-9,
            "skew must respect ±2 bps cap: got {}",
            result.skew_bps
        );
    }

    #[test]
    fn test_unwarmed_vpin_zeroes_component() {
        let s = scorer();
        let mut input = default_input();
        input.vpin = None;
        let result = s.evaluate(&input);
        assert!(
            result.components.vpin.abs() < 1e-9,
            "unwarmed VPIN should contribute 0"
        );
    }

    #[test]
    fn test_sigmoid_extremes() {
        // Very high x → near 1.0.
        assert!((sigmoid(100.0, 0.5, 15.0) - 1.0).abs() < 1e-6);
        // Very low x → near 0.0.
        assert!(sigmoid(-100.0, 0.5, 15.0) < 1e-6);
        // At midpoint → 0.5.
        assert!((sigmoid(0.5, 0.5, 15.0) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_ramp_edge_cases() {
        assert!((ramp(0.0, 10.0)).abs() < 1e-9);
        assert!((ramp(5.0, 10.0) - 0.5).abs() < 1e-9);
        assert!((ramp(10.0, 10.0) - 1.0).abs() < 1e-9);
        assert!((ramp(20.0, 10.0) - 1.0).abs() < 1e-9);
        // Zero saturation → 1.0.
        assert!((ramp(5.0, 0.0) - 1.0).abs() < 1e-9);
    }
}
