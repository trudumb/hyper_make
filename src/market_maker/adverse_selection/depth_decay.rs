//! Depth-dependent adverse selection model.
//!
//! Implements the first-principles formula: AS(δ) = AS₀ × exp(-δ/δ_char)
//! Calibrated from fill history by depth bucket.

use std::collections::VecDeque;
use std::time::Instant;
use tracing::debug;

/// Depth buckets for AS calibration (in basis points)
pub(crate) const DEPTH_BUCKETS: [(f64, f64); 4] = [
    (0.0, 2.0),    // Touch: 0-2 bps
    (2.0, 5.0),    // Near: 2-5 bps
    (5.0, 10.0),   // Mid: 5-10 bps
    (10.0, 100.0), // Deep: 10+ bps
];

/// Pending fill awaiting AS measurement (for depth-aware calibration).
#[derive(Debug, Clone)]
struct PendingDepthFill {
    /// Trade ID for deduplication
    tid: u64,
    /// Fill time for horizon tracking
    fill_time: Instant,
    /// Mid price at fill time (for AS calculation)
    fill_mid: f64,
    /// Fill size
    size: f64,
    /// True if this was a buy fill
    is_buy: bool,
    /// Depth from mid at fill time in basis points
    depth_bps: f64,
}

/// A fill with associated depth information for calibration.
#[derive(Debug, Clone)]
pub struct FillWithDepth {
    /// Trade ID for deduplication
    pub tid: u64,
    /// Fill size
    pub size: f64,
    /// True if this was a buy fill
    pub is_buy: bool,
    /// Depth from mid at fill time in basis points
    pub depth_bps: f64,
    /// Realized adverse selection (measured after horizon)
    pub realized_as: f64,
}

/// Depth-dependent adverse selection model.
///
/// Implements the first-principles formula: AS(δ) = AS₀ × exp(-δ/δ_char)
///
/// Where:
/// - AS₀ = adverse selection at the touch (calibrated from fills at depth 0-2bp)
/// - δ_char = characteristic depth where AS decays to 37% (e^-1)
/// - Deep levels have exponentially less informed flow
///
/// This model captures the empirical fact that informed traders prioritize
/// execution speed over price improvement, so they hit the touch more than
/// deep levels.
///
/// Additionally calibrates `alpha_touch` for Kelly-Stochastic allocation:
/// - alpha_touch = P(informed) at the touch (0-1)
/// - Measured as fraction of fills with significant adverse movement
#[derive(Debug, Clone)]
pub struct DepthDecayAS {
    /// AS at touch in basis points (calibrated from fills at depth 0-2bp)
    pub as_touch_bps: f64,
    /// Characteristic depth for exponential decay in basis points
    /// At δ = δ_char, AS decays to ~37% of touch AS
    pub delta_char_bps: f64,
    /// Calibration confidence (0-1), based on fill count and R²
    pub confidence: f64,
    /// Fill history by bucket for ongoing calibration
    bucket_fills: [Vec<f64>; 4], // AS measurements per bucket
    /// Total fills used for calibration
    total_calibration_fills: usize,
    /// Minimum fills per bucket before calibration starts
    min_fills_per_bucket: usize,
    /// EWMA decay for bucket AS estimates
    ewma_alpha: f64,
    /// EWMA AS estimate per bucket
    bucket_as_ewma: [f64; 4],
    /// Pending fills awaiting AS measurement (stochastic module integration)
    pending_fills: VecDeque<PendingDepthFill>,
    /// Measurement horizon in milliseconds (default 1000ms)
    measurement_horizon_ms: u64,
    /// Maximum pending fills to track
    max_pending_fills: usize,

    // === Kelly-Stochastic Alpha Calibration ===
    /// Calibrated alpha (informed probability) at touch (0-1)
    /// Measured as fraction of fills showing significant adverse movement
    pub alpha_touch: f64,
    /// EWMA of alpha per bucket (for alpha decay estimation)
    bucket_alpha_ewma: [f64; 4],
    /// Count of "informed" fills per bucket (AS > threshold)
    bucket_informed_count: [usize; 4],
    /// Total fills per bucket for alpha calculation
    bucket_total_count: [usize; 4],
    /// Threshold for classifying a fill as "informed" (bps)
    /// Fills with |AS| > threshold are counted as informed
    informed_threshold_bps: f64,
}

impl Default for DepthDecayAS {
    fn default() -> Self {
        Self {
            // Conservative defaults before calibration
            as_touch_bps: 3.0,   // 3 bps AS at touch (typical)
            delta_char_bps: 8.0, // AS decays to 37% by 8 bps depth
            confidence: 0.0,     // No confidence until calibrated
            bucket_fills: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            total_calibration_fills: 0,
            min_fills_per_bucket: 10, // Need 10 fills per bucket
            ewma_alpha: 0.1,
            bucket_as_ewma: [0.0; 4],
            // Stochastic module integration - pending fill tracking
            pending_fills: VecDeque::new(),
            measurement_horizon_ms: 1000, // 1 second AS measurement horizon
            max_pending_fills: 100,       // Track up to 100 pending fills
            // Kelly-Stochastic alpha calibration
            alpha_touch: 0.15, // Conservative default: 15% informed at touch
            bucket_alpha_ewma: [0.15, 0.10, 0.05, 0.02], // Decay with depth
            bucket_informed_count: [0; 4],
            bucket_total_count: [0; 4],
            informed_threshold_bps: 1.0, // 1 bp threshold for "informed" classification
        }
    }
}

impl DepthDecayAS {
    /// Create with custom initial parameters.
    pub fn new(as_touch_bps: f64, delta_char_bps: f64) -> Self {
        Self {
            as_touch_bps,
            delta_char_bps,
            ..Default::default()
        }
    }

    /// Adverse selection at arbitrary depth: AS(δ) = AS₀ × exp(-δ/δ_char)
    ///
    /// This is the core first-principles formula from the manual.
    pub fn as_at_depth(&self, depth_bps: f64) -> f64 {
        self.as_touch_bps * (-depth_bps / self.delta_char_bps).exp()
    }

    /// Expected spread capture at depth: SC(δ) = δ - AS(δ) - fees
    ///
    /// Positive spread capture means profitable, negative means losing money.
    pub fn spread_capture(&self, depth_bps: f64, fees_bps: f64) -> f64 {
        depth_bps - self.as_at_depth(depth_bps) - fees_bps
    }

    /// Find depth where spread capture becomes positive (break-even depth)
    ///
    /// Solves: δ - AS₀ × exp(-δ/δ_char) - fees = 0
    /// Using Newton-Raphson iteration.
    pub fn break_even_depth(&self, fees_bps: f64) -> f64 {
        // Initial guess: AS₀ + fees (would be break-even with no decay)
        let mut delta = self.as_touch_bps + fees_bps;

        for _ in 0..10 {
            let f = delta - self.as_touch_bps * (-delta / self.delta_char_bps).exp() - fees_bps;
            let f_prime = 1.0
                + (self.as_touch_bps / self.delta_char_bps) * (-delta / self.delta_char_bps).exp();

            if f_prime.abs() < 1e-10 {
                break;
            }

            let delta_new = delta - f / f_prime;
            if (delta_new - delta).abs() < 0.01 {
                return delta_new.max(0.0);
            }
            delta = delta_new;
        }

        delta.max(0.0)
    }

    /// Record a fill for calibration.
    ///
    /// Call this after measuring realized AS for a fill at a known depth.
    pub fn record_fill(&mut self, fill: &FillWithDepth) {
        // Find the bucket for this depth
        let bucket_idx = self.get_bucket_index(fill.depth_bps);

        // Update EWMA for this bucket
        let alpha = self.ewma_alpha;
        self.bucket_as_ewma[bucket_idx] =
            alpha * fill.realized_as.abs() + (1.0 - alpha) * self.bucket_as_ewma[bucket_idx];

        // Store for batch calibration (keep last 100 per bucket)
        if self.bucket_fills[bucket_idx].len() >= 100 {
            self.bucket_fills[bucket_idx].remove(0);
        }
        self.bucket_fills[bucket_idx].push(fill.realized_as.abs());

        // === Kelly-Stochastic Alpha Tracking ===
        // Track whether this fill was "informed" (AS > threshold)
        self.bucket_total_count[bucket_idx] += 1;
        if fill.realized_as.abs() > self.informed_threshold_bps {
            self.bucket_informed_count[bucket_idx] += 1;
        }

        // Update alpha EWMA for this bucket
        let is_informed = if fill.realized_as.abs() > self.informed_threshold_bps {
            1.0
        } else {
            0.0
        };
        self.bucket_alpha_ewma[bucket_idx] =
            alpha * is_informed + (1.0 - alpha) * self.bucket_alpha_ewma[bucket_idx];

        self.total_calibration_fills += 1;

        // Attempt recalibration periodically
        if self.total_calibration_fills.is_multiple_of(20) {
            self.calibrate();
        }
    }

    /// Get bucket index for a given depth.
    pub(crate) fn get_bucket_index(&self, depth_bps: f64) -> usize {
        for (i, (min, max)) in DEPTH_BUCKETS.iter().enumerate() {
            if depth_bps >= *min && depth_bps < *max {
                return i;
            }
        }
        3 // Deep bucket for anything beyond 10bp
    }

    /// Calibrate AS₀ and δ_char from fill history.
    ///
    /// Uses weighted least squares on log-transformed AS values:
    /// ln(AS) = ln(AS₀) - δ/δ_char
    ///
    /// This is a linear regression: y = a + bx where
    /// - y = ln(AS)
    /// - x = δ (bucket midpoint)
    /// - a = ln(AS₀)
    /// - b = -1/δ_char
    pub fn calibrate(&mut self) {
        // Check if we have enough fills in each bucket
        let has_enough = self
            .bucket_fills
            .iter()
            .all(|b| b.len() >= self.min_fills_per_bucket);

        if !has_enough {
            return;
        }

        // Compute mean AS per bucket
        let bucket_means: Vec<f64> = self
            .bucket_fills
            .iter()
            .map(|fills| {
                if fills.is_empty() {
                    0.0
                } else {
                    fills.iter().sum::<f64>() / fills.len() as f64
                }
            })
            .collect();

        // Bucket midpoints for regression
        let bucket_midpoints: Vec<f64> = DEPTH_BUCKETS
            .iter()
            .map(|(min, max)| (min + max) / 2.0)
            .collect();

        // Weighted least squares on log(AS) = log(AS₀) - δ/δ_char
        // Weight by number of fills in each bucket
        let weights: Vec<f64> = self.bucket_fills.iter().map(|b| b.len() as f64).collect();

        // Filter out buckets with zero or very small AS (can't take log)
        let valid_data: Vec<(f64, f64, f64)> = bucket_midpoints
            .iter()
            .zip(bucket_means.iter())
            .zip(weights.iter())
            .filter(|((_, &as_val), _)| as_val > 0.001) // 0.1 bps minimum
            .map(|((&depth, &as_val), &w)| (depth, as_val.ln(), w))
            .collect();

        if valid_data.len() < 2 {
            return; // Need at least 2 points for regression
        }

        // Weighted linear regression: y = a + bx
        let sum_w: f64 = valid_data.iter().map(|(_, _, w)| w).sum();
        let sum_wx: f64 = valid_data.iter().map(|(x, _, w)| w * x).sum();
        let sum_wy: f64 = valid_data.iter().map(|(_, y, w)| w * y).sum();
        let sum_wxx: f64 = valid_data.iter().map(|(x, _, w)| w * x * x).sum();
        let sum_wxy: f64 = valid_data.iter().map(|(x, y, w)| w * x * y).sum();

        let denom = sum_w * sum_wxx - sum_wx * sum_wx;
        if denom.abs() < 1e-10 {
            return;
        }

        let b = (sum_w * sum_wxy - sum_wx * sum_wy) / denom;
        let a = (sum_wy - b * sum_wx) / sum_w;

        // Extract parameters
        let new_as_touch = a.exp(); // AS₀ = exp(a)
        let new_delta_char = if b.abs() > 1e-10 {
            -1.0 / b
        } else {
            self.delta_char_bps
        };

        // Compute R² for confidence
        let y_mean = sum_wy / sum_w;
        let ss_tot: f64 = valid_data
            .iter()
            .map(|(_, y, w)| w * (y - y_mean).powi(2))
            .sum();
        let ss_res: f64 = valid_data
            .iter()
            .map(|(x, y, w)| {
                let y_pred = a + b * x;
                w * (y - y_pred).powi(2)
            })
            .sum();

        let r_squared = if ss_tot > 1e-10 {
            1.0 - ss_res / ss_tot
        } else {
            0.0
        };

        // Only update if results are sensible
        if new_as_touch > 0.1
            && new_as_touch < 50.0
            && new_delta_char > 1.0
            && new_delta_char < 100.0
            && r_squared > 0.3
        {
            // Smooth update to avoid jumps
            let smooth = 0.2;
            self.as_touch_bps = smooth * new_as_touch + (1.0 - smooth) * self.as_touch_bps;
            self.delta_char_bps = smooth * new_delta_char + (1.0 - smooth) * self.delta_char_bps;
            self.confidence = r_squared.sqrt(); // sqrt(R²) as confidence

            // === Kelly-Stochastic: Update alpha_touch from calibrated data ===
            // Alpha at touch = fraction of fills at touch that are informed
            let touch_total = self.bucket_total_count[0];
            if touch_total >= self.min_fills_per_bucket {
                let touch_informed = self.bucket_informed_count[0] as f64;
                let new_alpha_touch = touch_informed / touch_total as f64;
                // Smooth update and clamp to [0.01, 0.5] range
                self.alpha_touch =
                    (smooth * new_alpha_touch + (1.0 - smooth) * self.alpha_touch).clamp(0.01, 0.5);
            }

            debug!(
                as_touch = self.as_touch_bps,
                delta_char = self.delta_char_bps,
                alpha_touch = self.alpha_touch,
                confidence = self.confidence,
                r_squared = r_squared,
                fills = self.total_calibration_fills,
                "DepthDecayAS: Calibrated from fill history"
            );
        }
    }

    /// Check if model is calibrated with sufficient confidence.
    pub fn is_calibrated(&self) -> bool {
        self.confidence > 0.5 && self.total_calibration_fills >= 40
    }

    /// Get calibrated alpha at touch for Kelly-Stochastic allocation.
    /// Returns the fraction of fills at touch that show significant adverse selection.
    pub fn calibrated_alpha_touch(&self) -> f64 {
        self.alpha_touch
    }

    /// Get alpha at arbitrary depth using exponential decay.
    /// α(δ) = α_touch × exp(-δ/δ_char)
    /// Uses the same characteristic depth as AS decay for consistency.
    pub fn alpha_at_depth(&self, depth_bps: f64) -> f64 {
        self.alpha_touch * (-depth_bps / self.delta_char_bps).exp()
    }

    /// Get summary for diagnostics.
    pub fn summary(&self) -> DepthDecayASSummary {
        DepthDecayASSummary {
            as_touch_bps: self.as_touch_bps,
            delta_char_bps: self.delta_char_bps,
            alpha_touch: self.alpha_touch,
            confidence: self.confidence,
            total_fills: self.total_calibration_fills,
            bucket_counts: self.bucket_fills.iter().map(|b| b.len()).collect(),
            bucket_as_bps: self.bucket_as_ewma.to_vec(),
            bucket_alpha: self.bucket_alpha_ewma.to_vec(),
        }
    }

    // ========================================================================
    // Stochastic Module Integration: Pending Fill Tracking
    // ========================================================================

    /// Record a pending fill for depth-aware AS calibration.
    ///
    /// Call this immediately when a fill is received, with fill price and current mid.
    /// The fill will be resolved later in `resolve_pending_fills` after the
    /// measurement horizon elapses.
    ///
    /// # Arguments
    /// * `tid` - Trade ID for deduplication
    /// * `fill_price` - Price at which the fill executed
    /// * `size` - Fill size
    /// * `is_buy` - True if this was a buy fill
    /// * `current_mid` - Mid price at fill time
    pub fn record_pending_fill(
        &mut self,
        tid: u64,
        fill_price: f64,
        size: f64,
        is_buy: bool,
        current_mid: f64,
    ) {
        // Compute depth from mid in basis points
        let depth_bps = if current_mid > 0.0 {
            ((fill_price - current_mid) / current_mid).abs() * 10000.0
        } else {
            0.0
        };

        // Enforce max pending fills (FIFO eviction)
        while self.pending_fills.len() >= self.max_pending_fills {
            self.pending_fills.pop_front();
        }

        self.pending_fills.push_back(PendingDepthFill {
            tid,
            fill_time: Instant::now(),
            fill_mid: current_mid,
            size,
            is_buy,
            depth_bps,
        });

        tracing::debug!(
            tid = tid,
            fill_price = fill_price,
            mid = current_mid,
            depth_bps = depth_bps,
            is_buy = is_buy,
            pending = self.pending_fills.len(),
            "DepthDecayAS: Recorded pending fill for calibration"
        );
    }

    /// Resolve pending fills whose measurement horizon has elapsed.
    ///
    /// Call this on each mid price update to resolve fills and feed them
    /// into the depth-dependent AS calibration.
    ///
    /// # Arguments
    /// * `current_mid` - Current mid price for AS calculation
    pub fn resolve_pending_fills(&mut self, current_mid: f64) {
        let now = Instant::now();
        let horizon = std::time::Duration::from_millis(self.measurement_horizon_ms);

        // Process all pending fills whose horizon has elapsed
        while let Some(front) = self.pending_fills.front() {
            if now.duration_since(front.fill_time) < horizon {
                break; // Not ready yet
            }

            let fill = self.pending_fills.pop_front().unwrap();

            // Calculate realized adverse selection (price change as fraction)
            let price_change = (current_mid - fill.fill_mid) / fill.fill_mid;

            // AS from MM's perspective:
            // - Buy fill: price going DOWN is adverse (we bought, price dropped)
            // - Sell fill: price going UP is adverse (we sold, price rose)
            let realized_as = if fill.is_buy {
                -price_change // Positive AS if price dropped after buying
            } else {
                price_change // Positive AS if price rose after selling
            };

            // Convert to basis points
            let realized_as_bps = realized_as * 10000.0;

            // Create FillWithDepth and feed to calibration
            let fill_with_depth = FillWithDepth {
                tid: fill.tid,
                size: fill.size,
                is_buy: fill.is_buy,
                depth_bps: fill.depth_bps,
                realized_as: realized_as_bps,
            };

            self.record_fill(&fill_with_depth);

            tracing::debug!(
                tid = fill.tid,
                depth_bps = fill.depth_bps,
                realized_as_bps = realized_as_bps,
                is_buy = fill.is_buy,
                "DepthDecayAS: Resolved fill for calibration"
            );
        }
    }

    /// Get number of pending fills awaiting resolution.
    pub fn pending_count(&self) -> usize {
        self.pending_fills.len()
    }
}

/// Summary of depth-decay AS model for logging.
#[derive(Debug, Clone)]
pub struct DepthDecayASSummary {
    pub as_touch_bps: f64,
    pub delta_char_bps: f64,
    /// Calibrated alpha (informed probability) at touch for Kelly-Stochastic
    pub alpha_touch: f64,
    pub confidence: f64,
    pub total_fills: usize,
    pub bucket_counts: Vec<usize>,
    pub bucket_as_bps: Vec<f64>,
    /// EWMA alpha (informed probability) per bucket
    pub bucket_alpha: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_depth_decay_as_formula() {
        let model = DepthDecayAS::new(5.0, 10.0); // AS₀=5bp, δ_char=10bp

        // At touch (0 bps): AS = 5.0 * exp(0) = 5.0
        assert!((model.as_at_depth(0.0) - 5.0).abs() < 0.01);

        // At characteristic depth (10 bps): AS = 5.0 * exp(-1) ≈ 1.84
        assert!((model.as_at_depth(10.0) - 5.0 * (-1.0_f64).exp()).abs() < 0.01);

        // At 2x characteristic depth (20 bps): AS = 5.0 * exp(-2) ≈ 0.68
        assert!((model.as_at_depth(20.0) - 5.0 * (-2.0_f64).exp()).abs() < 0.01);

        // Deep (50 bps): AS ≈ 0 (effectively)
        assert!(model.as_at_depth(50.0) < 0.1);
    }

    #[test]
    fn test_depth_decay_spread_capture() {
        let model = DepthDecayAS::new(5.0, 10.0);
        let fees = 0.5; // 0.5 bps fees

        // At touch: SC = 0 - 5.0 - 0.5 = -5.5 (unprofitable)
        assert!(model.spread_capture(0.0, fees) < 0.0);

        // At 5 bps: SC = 5 - 5*exp(-0.5) - 0.5 ≈ 5 - 3.03 - 0.5 = 1.47
        let sc_5 = model.spread_capture(5.0, fees);
        assert!(sc_5 > 0.0);

        // At 20 bps: SC ≈ 20 - 0.68 - 0.5 = 18.82 (very profitable)
        let sc_20 = model.spread_capture(20.0, fees);
        assert!(sc_20 > 15.0);
    }

    #[test]
    fn test_depth_decay_break_even() {
        let model = DepthDecayAS::new(5.0, 10.0);
        let fees = 0.5;

        let be = model.break_even_depth(fees);

        // At break-even, spread capture should be ~0
        let sc_at_be = model.spread_capture(be, fees);
        assert!(sc_at_be.abs() < 0.1); // Within 0.1 bps of break-even
    }

    #[test]
    fn test_depth_decay_calibration() {
        let mut model = DepthDecayAS::default();

        // Simulate fills with known AS pattern (higher at touch, decaying)
        // Touch bucket: 0-2bp → high AS (~4bp)
        for i in 0..15 {
            model.record_fill(&FillWithDepth {
                tid: i,
                size: 1.0,
                is_buy: true,
                depth_bps: 1.0,
                realized_as: 0.0004, // 4 bps
            });
        }

        // Near bucket: 2-5bp → medium AS (~2bp)
        for i in 15..30 {
            model.record_fill(&FillWithDepth {
                tid: i,
                size: 1.0,
                is_buy: true,
                depth_bps: 3.5,
                realized_as: 0.0002, // 2 bps
            });
        }

        // Mid bucket: 5-10bp → lower AS (~1bp)
        for i in 30..45 {
            model.record_fill(&FillWithDepth {
                tid: i,
                size: 1.0,
                is_buy: true,
                depth_bps: 7.5,
                realized_as: 0.0001, // 1 bp
            });
        }

        // Deep bucket: 10+bp → very low AS (~0.5bp)
        for i in 45..60 {
            model.record_fill(&FillWithDepth {
                tid: i,
                size: 1.0,
                is_buy: true,
                depth_bps: 15.0,
                realized_as: 0.00005, // 0.5 bps
            });
        }

        // Model should have calibrated
        assert!(model.total_calibration_fills >= 40);

        // AS at touch should be positive
        assert!(model.as_touch_bps > 0.0);

        // Delta char should be reasonable (between 1 and 100)
        assert!(model.delta_char_bps > 1.0 && model.delta_char_bps < 100.0);
    }

    #[test]
    fn test_depth_decay_bucket_assignment() {
        let model = DepthDecayAS::default();

        // Test bucket boundaries
        assert_eq!(model.get_bucket_index(0.5), 0); // Touch
        assert_eq!(model.get_bucket_index(1.5), 0); // Touch
        assert_eq!(model.get_bucket_index(2.5), 1); // Near
        assert_eq!(model.get_bucket_index(4.0), 1); // Near
        assert_eq!(model.get_bucket_index(6.0), 2); // Mid
        assert_eq!(model.get_bucket_index(9.0), 2); // Mid
        assert_eq!(model.get_bucket_index(15.0), 3); // Deep
        assert_eq!(model.get_bucket_index(100.0), 3); // Deep
    }

    #[test]
    fn test_depth_decay_summary() {
        let model = DepthDecayAS::new(4.0, 12.0);
        let summary = model.summary();

        assert!((summary.as_touch_bps - 4.0).abs() < 0.01);
        assert!((summary.delta_char_bps - 12.0).abs() < 0.01);
        assert_eq!(summary.total_fills, 0);
        assert_eq!(summary.bucket_counts.len(), 4);
        assert_eq!(summary.bucket_alpha.len(), 4);
        // Default alpha_touch is 0.15
        assert!((summary.alpha_touch - 0.15).abs() < 0.01);
    }

    #[test]
    fn test_alpha_at_depth() {
        let model = DepthDecayAS::new(5.0, 10.0);
        // Default alpha_touch is 0.15

        // At touch: alpha = 0.15
        assert!((model.alpha_at_depth(0.0) - 0.15).abs() < 0.01);

        // At characteristic depth: alpha = 0.15 * exp(-1) ≈ 0.055
        assert!((model.alpha_at_depth(10.0) - 0.15 * (-1.0_f64).exp()).abs() < 0.01);

        // Deep: alpha approaches 0
        assert!(model.alpha_at_depth(50.0) < 0.01);
    }
}
