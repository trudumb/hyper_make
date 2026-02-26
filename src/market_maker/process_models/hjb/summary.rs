//! Output and summary types for HJB controller.

/// Output from drift-adjusted skew calculation.
///
/// Contains both the total skew and its components for analysis.
#[derive(Debug, Clone, Copy, Default)]
pub struct DriftAdjustedSkew {
    /// Total optimal skew (base + drift urgency + predictive bias).
    pub total_skew: f64,

    /// Base HJB skew (γσ²qT + terminal + funding).
    pub base_skew: f64,

    /// Additional urgency from momentum-position opposition.
    /// Same sign as base_skew when opposed (amplifies reduction).
    pub drift_urgency: f64,

    /// Predictive bias from changepoint detection.
    /// β_t = -sensitivity × prob_excess × σ
    /// Negative when regime change is imminent → widen bids, tighten asks.
    pub predictive_bias: f64,

    /// Variance multiplier for inventory risk.
    /// > 1.0 when opposed, used for σ²_effective calculation.
    pub variance_multiplier: f64,

    /// Whether position is opposed to momentum.
    pub is_opposed: bool,

    /// Urgency score [0, 5] for diagnostics.
    pub urgency_score: f64,
}

/// Summary of HJB controller state for diagnostics.
#[derive(Debug, Clone)]
pub struct HJBSummary {
    pub time_remaining_secs: f64,
    pub terminal_urgency: f64,
    pub is_terminal_zone: bool,
    pub gamma_multiplier: f64,
    pub effective_gamma: f64,
    pub funding_rate_ewma: f64,
    pub optimal_inventory_target: f64,
    pub sigma: f64,
    /// Whether using funding-cycle horizon (true) or session-based fallback (false).
    pub funding_horizon_active: bool,
    /// Effective terminal penalty (calibrated or configured).
    pub effective_terminal_penalty: f64,
}

/// Momentum statistics for diagnostics and confidence estimation.
#[derive(Debug, Clone, Default)]
pub struct MomentumStats {
    /// Mean momentum over window (bps).
    pub mean_bps: f64,
    /// Standard deviation of momentum (bps).
    pub std_dev_bps: f64,
    /// Number of direction changes in window.
    pub direction_changes: usize,
    /// Number of samples in window.
    pub sample_count: usize,
    /// Average continuation probability.
    pub avg_continuation: f64,
}

impl MomentumStats {
    /// Check if momentum signal is noisy (many direction changes).
    pub fn is_noisy(&self) -> bool {
        if self.sample_count < 10 {
            return true;
        }
        // More than 40% direction changes = noisy
        (self.direction_changes as f64 / self.sample_count as f64) > 0.4
    }

    /// Get signal quality score [0, 1].
    /// Higher = more reliable momentum signal.
    pub fn signal_quality(&self) -> f64 {
        if self.sample_count < 10 {
            return 0.0;
        }

        // Factors that increase quality:
        // 1. Mean momentum magnitude (stronger signals)
        let magnitude_score = (self.mean_bps.abs() / 50.0).min(1.0);

        // 2. Low direction changes (consistent trend)
        let consistency =
            1.0 - (self.direction_changes as f64 / self.sample_count as f64).min(0.5) * 2.0;

        // 3. High continuation probability
        let continuation_score = self.avg_continuation;

        // 4. Low volatility relative to mean (high signal-to-noise)
        let snr = if self.std_dev_bps > 0.0 {
            (self.mean_bps.abs() / self.std_dev_bps).min(2.0) / 2.0
        } else {
            0.0
        };

        // Combine factors
        (magnitude_score * 0.2 + consistency * 0.3 + continuation_score * 0.3 + snr * 0.2)
            .clamp(0.0, 1.0)
    }
}
