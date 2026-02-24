//! Position-Based P&L Tracker.
//!
//! Derives position thresholds from actual P&L data rather than arbitrary
//! numbers. The key insight: the optimal position threshold is where
//! expected P&L crosses zero.
//!
//! # Architecture
//!
//! Tracks P&L by position quantile and regime:
//!
//! ```ignore
//! let mut tracker = PositionPnLTracker::new(config);
//!
//! // On each fill, record P&L with position context
//! tracker.record(position_ratio, regime, pnl_bps);
//!
//! // Get derived thresholds
//! let threshold = tracker.derived_position_threshold();
//! let reduce_only = tracker.reduce_only_threshold(regime);
//! ```
//!
//! # Principled Thresholds
//!
//! Position threshold = where E[PnL | position ratio] crosses zero
//! Reduce-only threshold = regime-specific, derived from P&L data
//!
//! No arbitrary 0.05 or 0.70 - derived from actual performance.

/// Number of position quantiles (10 = deciles).
pub const NUM_QUANTILES: usize = 10;

/// Number of regimes tracked (calm=0, volatile=1, cascade=2).
pub const NUM_REGIMES: usize = 3;

/// Configuration for position P&L tracker.
#[derive(Debug, Clone)]
pub struct PositionPnLConfig {
    /// Minimum samples per quantile-regime cell before trusting.
    /// Default: 10
    pub min_samples_per_cell: usize,

    /// Cold-start position threshold (conservative).
    /// Used before calibration data available.
    /// Default: 0.03 (lower than old 0.05 = more conservative)
    pub cold_start_position_threshold: f64,

    /// Cold-start reduce-only threshold (conservative).
    /// Used before calibration data available.
    /// Default: 0.50 (lower than old 0.70 = more conservative)
    pub cold_start_reduce_only_threshold: f64,

    /// Regime-specific conservative adjustments for reduce-only.
    /// [calm_mult, volatile_mult, cascade_mult]
    /// Cascade should be more conservative (lower threshold).
    /// Default: [1.0, 0.8, 0.6]
    pub regime_reduce_only_multipliers: [f64; NUM_REGIMES],

    /// Minimum warmup samples before using calibrated thresholds.
    /// Default: 50
    pub min_warmup_samples: usize,
}

impl Default for PositionPnLConfig {
    fn default() -> Self {
        Self {
            min_samples_per_cell: 10,
            cold_start_position_threshold: 0.03,
            cold_start_reduce_only_threshold: 0.50,
            regime_reduce_only_multipliers: [1.0, 0.8, 0.6],
            min_warmup_samples: 50,
        }
    }
}

/// Online rolling statistics (mean and variance).
#[derive(Debug, Clone, Default)]
pub struct RollingStats {
    /// Number of samples.
    n: usize,
    /// Running mean.
    mean: f64,
    /// Running M2 for variance calculation (Welford's algorithm).
    m2: f64,
}

impl RollingStats {
    /// Create new rolling stats.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a sample using Welford's online algorithm.
    pub fn push(&mut self, value: f64) {
        self.n += 1;
        let delta = value - self.mean;
        self.mean += delta / self.n as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
    }

    /// Get sample count.
    pub fn count(&self) -> usize {
        self.n
    }

    /// Get mean.
    pub fn mean(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            self.mean
        }
    }

    /// Get sample variance.
    pub fn variance(&self) -> f64 {
        if self.n < 2 {
            0.0
        } else {
            self.m2 / (self.n - 1) as f64
        }
    }

    /// Get standard deviation.
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Get standard error of the mean.
    pub fn std_error(&self) -> f64 {
        if self.n < 2 {
            f64::INFINITY
        } else {
            self.std_dev() / (self.n as f64).sqrt()
        }
    }

    /// Check if mean is significantly different from zero.
    /// Uses t-test with alpha=0.05 (t > 2 for n > 30).
    pub fn is_significantly_nonzero(&self) -> bool {
        if self.n < 10 {
            return false;
        }
        let t = self.mean.abs() / self.std_error();
        t > 2.0
    }

    /// Clear all data.
    pub fn clear(&mut self) {
        self.n = 0;
        self.mean = 0.0;
        self.m2 = 0.0;
    }
}

/// Position-based P&L tracker.
///
/// Tracks P&L by position quantile and regime to derive optimal
/// position thresholds from actual data.
#[derive(Debug)]
pub struct PositionPnLTracker {
    /// P&L stats by position quantile [0-9] and regime [0-2].
    /// Indexed as pnl_by_quantile[quantile][regime].
    pnl_by_quantile: [[RollingStats; NUM_REGIMES]; NUM_QUANTILES],

    /// Overall P&L stats (not segmented).
    overall_pnl: RollingStats,

    /// Total samples recorded.
    total_samples: usize,

    /// Configuration.
    config: PositionPnLConfig,
}

impl PositionPnLTracker {
    /// Create a new position P&L tracker.
    pub fn new(config: PositionPnLConfig) -> Self {
        Self {
            pnl_by_quantile: Default::default(),
            overall_pnl: RollingStats::new(),
            total_samples: 0,
            config,
        }
    }

    /// Record fill P&L with position context.
    ///
    /// # Arguments
    /// * `position_ratio` - Current position as fraction of max (0.0 to 1.0)
    /// * `regime` - Current regime index (0=calm, 1=volatile, 2=cascade)
    /// * `pnl_bps` - Realized P&L in basis points
    pub fn record(&mut self, position_ratio: f64, regime: usize, pnl_bps: f64) {
        // Clamp inputs
        let position_ratio = position_ratio.abs().clamp(0.0, 1.0);
        let regime = regime.min(NUM_REGIMES - 1);

        // Map position ratio to quantile (0-9)
        let quantile =
            ((position_ratio * NUM_QUANTILES as f64).floor() as usize).min(NUM_QUANTILES - 1);

        // Record to specific cell
        self.pnl_by_quantile[quantile][regime].push(pnl_bps);

        // Record to overall
        self.overall_pnl.push(pnl_bps);
        self.total_samples += 1;
    }

    /// Derive the position threshold where P&L crosses zero.
    ///
    /// Finds the position ratio where expected P&L changes from positive
    /// to negative. Below this threshold, we have edge; above, we're
    /// taking too much inventory risk.
    ///
    /// Returns a value between 0.0 and 1.0.
    pub fn derived_position_threshold(&self) -> f64 {
        if !self.is_warmed_up() {
            return self.cold_start_position_threshold();
        }

        // Find the quantile where mean P&L crosses from positive to negative
        let mut last_positive_quantile = None;

        for q in 0..NUM_QUANTILES {
            let mean_pnl = self.mean_pnl_for_quantile(q);
            let samples = self.samples_for_quantile(q);

            // Need sufficient samples
            if samples < self.config.min_samples_per_cell {
                continue;
            }

            if mean_pnl > 0.0 {
                last_positive_quantile = Some(q);
            } else if last_positive_quantile.is_some() {
                // Found the crossing point
                break;
            }
        }

        // Convert quantile to position ratio
        match last_positive_quantile {
            Some(q) => {
                // Position threshold is the upper bound of the last profitable quantile
                let threshold = (q + 1) as f64 / NUM_QUANTILES as f64;
                // Blend with cold-start based on confidence
                self.blend_with_cold_start(threshold, self.config.cold_start_position_threshold)
            }
            None => {
                // No clear crossing - either all positive or all negative
                // Be conservative
                self.cold_start_position_threshold()
            }
        }
    }

    /// Derive the reduce-only threshold for a specific regime.
    ///
    /// This is the position ratio above which we should only quote
    /// to reduce position (too risky to add).
    pub fn reduce_only_threshold(&self, regime: usize) -> f64 {
        let regime = regime.min(NUM_REGIMES - 1);

        if !self.is_warmed_up() {
            return self.cold_start_reduce_only_threshold(regime);
        }

        // Start from the opposite end: find where P&L becomes significantly negative
        let mut reduce_only_quantile = NUM_QUANTILES - 1;

        for q in (0..NUM_QUANTILES).rev() {
            let stats = &self.pnl_by_quantile[q][regime];

            if stats.count() < self.config.min_samples_per_cell {
                continue;
            }

            // If P&L is significantly negative, this is the reduce-only zone
            if stats.is_significantly_nonzero() && stats.mean() < 0.0 {
                reduce_only_quantile = q;
            } else {
                // Found a quantile where P&L is not significantly negative
                break;
            }
        }

        // Convert to position ratio
        let threshold = reduce_only_quantile as f64 / NUM_QUANTILES as f64;

        // Blend with cold-start
        let cold_start = self.cold_start_reduce_only_threshold(regime);
        self.blend_with_cold_start(threshold, cold_start)
    }

    /// Get mean P&L for a quantile (across all regimes).
    pub fn mean_pnl_for_quantile(&self, quantile: usize) -> f64 {
        let quantile = quantile.min(NUM_QUANTILES - 1);

        let mut total_n = 0usize;
        let mut weighted_mean = 0.0;

        for regime in 0..NUM_REGIMES {
            let stats = &self.pnl_by_quantile[quantile][regime];
            let n = stats.count();
            if n > 0 {
                total_n += n;
                weighted_mean += stats.mean() * n as f64;
            }
        }

        if total_n > 0 {
            weighted_mean / total_n as f64
        } else {
            0.0
        }
    }

    /// Get total samples for a quantile (across all regimes).
    pub fn samples_for_quantile(&self, quantile: usize) -> usize {
        let quantile = quantile.min(NUM_QUANTILES - 1);
        self.pnl_by_quantile[quantile]
            .iter()
            .map(|s| s.count())
            .sum()
    }

    /// Get mean P&L for a specific quantile-regime cell.
    pub fn mean_pnl_for_cell(&self, quantile: usize, regime: usize) -> f64 {
        let quantile = quantile.min(NUM_QUANTILES - 1);
        let regime = regime.min(NUM_REGIMES - 1);
        self.pnl_by_quantile[quantile][regime].mean()
    }

    /// Get samples for a specific quantile-regime cell.
    pub fn samples_for_cell(&self, quantile: usize, regime: usize) -> usize {
        let quantile = quantile.min(NUM_QUANTILES - 1);
        let regime = regime.min(NUM_REGIMES - 1);
        self.pnl_by_quantile[quantile][regime].count()
    }

    /// Check if tracker is warmed up.
    pub fn is_warmed_up(&self) -> bool {
        self.total_samples >= self.config.min_warmup_samples
    }

    /// Get warmup progress (0.0 to 1.0).
    pub fn warmup_progress(&self) -> f64 {
        (self.total_samples as f64 / self.config.min_warmup_samples as f64).min(1.0)
    }

    /// Get total sample count.
    pub fn total_samples(&self) -> usize {
        self.total_samples
    }

    /// Get overall mean P&L.
    pub fn overall_mean_pnl(&self) -> f64 {
        self.overall_pnl.mean()
    }

    /// Get overall P&L std dev.
    pub fn overall_pnl_std(&self) -> f64 {
        self.overall_pnl.std_dev()
    }

    /// Get cold-start position threshold.
    pub fn cold_start_position_threshold(&self) -> f64 {
        self.config.cold_start_position_threshold
    }

    /// Get cold-start reduce-only threshold for a regime.
    pub fn cold_start_reduce_only_threshold(&self, regime: usize) -> f64 {
        let regime = regime.min(NUM_REGIMES - 1);
        self.config.cold_start_reduce_only_threshold
            * self.config.regime_reduce_only_multipliers[regime]
    }

    /// Blend calibrated threshold with cold-start based on warmup.
    fn blend_with_cold_start(&self, calibrated: f64, cold_start: f64) -> f64 {
        let blend = self.warmup_progress();
        cold_start * (1.0 - blend) + calibrated * blend
    }

    /// Get configuration.
    pub fn config(&self) -> &PositionPnLConfig {
        &self.config
    }

    /// Generate diagnostic summary.
    pub fn diagnostic_summary(&self) -> PositionPnLDiagnostics {
        PositionPnLDiagnostics {
            derived_position_threshold: self.derived_position_threshold(),
            reduce_only_calm: self.reduce_only_threshold(0),
            reduce_only_volatile: self.reduce_only_threshold(1),
            reduce_only_cascade: self.reduce_only_threshold(2),
            overall_mean_pnl: self.overall_mean_pnl(),
            overall_pnl_std: self.overall_pnl_std(),
            total_samples: self.total_samples,
            is_warmed_up: self.is_warmed_up(),
            warmup_progress: self.warmup_progress(),
        }
    }

    /// Clear all data.
    pub fn clear(&mut self) {
        for row in &mut self.pnl_by_quantile {
            for cell in row {
                cell.clear();
            }
        }
        self.overall_pnl.clear();
        self.total_samples = 0;
    }
}

impl Default for PositionPnLTracker {
    fn default() -> Self {
        Self::new(PositionPnLConfig::default())
    }
}

/// Diagnostic summary for position P&L tracker.
#[derive(Debug, Clone)]
pub struct PositionPnLDiagnostics {
    /// Derived position threshold.
    pub derived_position_threshold: f64,
    /// Reduce-only threshold for calm regime.
    pub reduce_only_calm: f64,
    /// Reduce-only threshold for volatile regime.
    pub reduce_only_volatile: f64,
    /// Reduce-only threshold for cascade regime.
    pub reduce_only_cascade: f64,
    /// Overall mean P&L.
    pub overall_mean_pnl: f64,
    /// Overall P&L standard deviation.
    pub overall_pnl_std: f64,
    /// Total samples.
    pub total_samples: usize,
    /// Whether warmed up.
    pub is_warmed_up: bool,
    /// Warmup progress.
    pub warmup_progress: f64,
}

impl std::fmt::Display for PositionPnLDiagnostics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== Position Thresholds ===")?;
        writeln!(
            f,
            "Derived position threshold: {:.2}",
            self.derived_position_threshold
        )?;
        writeln!(f, "Reduce-only thresholds:")?;
        writeln!(f, "  - Calm:     {:.2}", self.reduce_only_calm)?;
        writeln!(f, "  - Volatile: {:.2}", self.reduce_only_volatile)?;
        writeln!(
            f,
            "  - Cascade:  {:.2} (more conservative)",
            self.reduce_only_cascade
        )?;
        writeln!(f)?;
        writeln!(
            f,
            "Overall P&L: mean={:.2} bps, std={:.2} bps",
            self.overall_mean_pnl, self.overall_pnl_std
        )?;
        writeln!(
            f,
            "Samples: {} (warmup: {:.0}%)",
            self.total_samples,
            self.warmup_progress * 100.0
        )?;

        Ok(())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config() -> PositionPnLConfig {
        PositionPnLConfig {
            min_samples_per_cell: 3,
            cold_start_position_threshold: 0.03,
            cold_start_reduce_only_threshold: 0.50,
            regime_reduce_only_multipliers: [1.0, 0.8, 0.6],
            min_warmup_samples: 10,
        }
    }

    #[test]
    fn test_rolling_stats() {
        let mut stats = RollingStats::new();

        stats.push(1.0);
        stats.push(2.0);
        stats.push(3.0);

        assert_eq!(stats.count(), 3);
        assert!((stats.mean() - 2.0).abs() < 1e-10);
        assert!(stats.variance() > 0.0);
    }

    #[test]
    fn test_recording() {
        let mut tracker = PositionPnLTracker::new(make_config());

        tracker.record(0.15, 0, 5.0); // 15% position, calm, +5 bps

        assert_eq!(tracker.total_samples(), 1);
        assert!(tracker.samples_for_quantile(1) > 0); // 15% → quantile 1
    }

    #[test]
    fn test_cold_start_thresholds() {
        let tracker = PositionPnLTracker::new(make_config());

        // Not warmed up - should use cold start
        assert!(!tracker.is_warmed_up());
        assert_eq!(tracker.derived_position_threshold(), 0.03);
        assert_eq!(tracker.cold_start_reduce_only_threshold(0), 0.50);
        assert_eq!(tracker.cold_start_reduce_only_threshold(1), 0.40);
        assert_eq!(tracker.cold_start_reduce_only_threshold(2), 0.30);
    }

    #[test]
    fn test_regime_reduce_only_ordering() {
        let tracker = PositionPnLTracker::new(make_config());

        // Cascade should have lowest (most conservative) threshold
        let calm = tracker.cold_start_reduce_only_threshold(0);
        let volatile = tracker.cold_start_reduce_only_threshold(1);
        let cascade = tracker.cold_start_reduce_only_threshold(2);

        assert!(calm > volatile);
        assert!(volatile > cascade);
    }

    #[test]
    fn test_warmup_progress() {
        let mut tracker = PositionPnLTracker::new(make_config());

        for i in 0..5 {
            tracker.record(0.1 * i as f64, 0, 1.0);
        }

        assert_eq!(tracker.warmup_progress(), 0.5);
        assert!(!tracker.is_warmed_up());

        for i in 0..5 {
            tracker.record(0.1 * i as f64, 0, 1.0);
        }

        assert_eq!(tracker.warmup_progress(), 1.0);
        assert!(tracker.is_warmed_up());
    }

    #[test]
    fn test_derived_threshold_with_data() {
        let mut tracker = PositionPnLTracker::new(make_config());

        // Simulate: low position = positive PnL, high position = negative PnL
        for _ in 0..10 {
            tracker.record(0.05, 0, 5.0); // Low position, positive
            tracker.record(0.15, 0, 3.0); // Medium-low, positive
            tracker.record(0.25, 0, 1.0); // Medium, small positive
            tracker.record(0.35, 0, -1.0); // Medium-high, small negative
            tracker.record(0.45, 0, -3.0); // High, negative
            tracker.record(0.55, 0, -5.0); // Very high, very negative
        }

        assert!(tracker.is_warmed_up());

        let threshold = tracker.derived_position_threshold();
        // Should be somewhere between 0.2 and 0.4 where PnL crosses zero
        assert!(threshold > 0.1, "Threshold should be > 0.1: {}", threshold);
        assert!(threshold < 0.5, "Threshold should be < 0.5: {}", threshold);
    }

    #[test]
    fn test_diagnostics() {
        let mut tracker = PositionPnLTracker::new(make_config());

        for i in 0..15 {
            let regime = i % 3;
            tracker.record(0.1 * (i % 5) as f64, regime, (i as f64) - 7.0);
        }

        let diag = tracker.diagnostic_summary();

        assert!(diag.is_warmed_up);
        assert!(diag.total_samples == 15);
        assert!(diag.warmup_progress == 1.0);
    }

    #[test]
    fn test_clear() {
        let mut tracker = PositionPnLTracker::new(make_config());

        for i in 0..20 {
            tracker.record(0.1, 0, i as f64);
        }

        assert!(tracker.is_warmed_up());

        tracker.clear();

        assert!(!tracker.is_warmed_up());
        assert_eq!(tracker.total_samples(), 0);
    }

    #[test]
    fn test_quantile_mapping() {
        let mut tracker = PositionPnLTracker::new(make_config());

        // Position 0.0 → quantile 0
        tracker.record(0.0, 0, 1.0);
        assert!(tracker.samples_for_quantile(0) > 0);

        // Position 0.5 → quantile 5
        tracker.record(0.5, 0, 1.0);
        assert!(tracker.samples_for_quantile(5) > 0);

        // Position 1.0 → quantile 9 (clamped)
        tracker.record(1.0, 0, 1.0);
        assert!(tracker.samples_for_quantile(9) > 0);
    }
}
