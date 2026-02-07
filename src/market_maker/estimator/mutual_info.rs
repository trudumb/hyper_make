//! Mutual Information estimation for feature quality assessment.
//!
//! Implements the Kraskov-Stögbauer-Grassberger (KSG) algorithm for estimating
//! mutual information between continuous random variables using k-nearest neighbors.
//!
//! # Key Concepts
//!
//! Mutual Information I(X;Y) measures the amount of information (in bits or nats)
//! that one variable contains about another. For market making:
//!
//! - **I(book_imbalance; price_direction)** ≈ 0.02-0.05 bits: How much does order
//!   book asymmetry tell us about where price will go?
//! - **I(momentum; fill)** ≈ 0.03-0.08 bits: Does recent price movement predict fills?
//! - **I(signal; adverse_selection)** ≈ 0.01-0.03 bits: Can we predict toxic flow?
//!
//! If MI < 0.01 bits, the signal is likely noise. Remove it.
//! If MI drops by 50% week-over-week, the alpha is decaying.
//!
//! # Algorithm
//!
//! The KSG estimator avoids binning (which loses information) by using k-NN distances:
//!
//! ```text
//! I(X;Y) ≈ ψ(k) - ⟨ψ(nₓ) + ψ(nᵧ)⟩ + ψ(N)
//! ```
//!
//! Where ψ is the digamma function, k is the neighbor count, nₓ and nᵧ are the number
//! of points within the k-th neighbor distance in marginal spaces, and N is total samples.
//!
//! # References
//!
//! - Kraskov, A., Stögbauer, H., & Grassberger, P. (2004). Estimating mutual information.
//!   Physical Review E, 69(6), 066138.

use std::collections::BinaryHeap;

/// Euler-Mascheroni constant γ ≈ 0.5772156649 (used in tests)
#[allow(dead_code)]
const EULER_GAMMA: f64 = 0.5772156649015329;

// ============================================================================
// Core MI Estimator
// ============================================================================

/// k-NN Mutual Information estimator using the Kraskov (KSG) algorithm.
///
/// # Usage
///
/// ```ignore
/// let mi = MutualInfoEstimator::new(3);
///
/// // Compute MI between book imbalance and price direction
/// let book_imbalance: Vec<f64> = /* ... */;
/// let price_direction: Vec<f64> = /* ... */;
///
/// let mi_bits = mi.estimate(&book_imbalance, &price_direction);
/// println!("MI(book, direction) = {:.4} bits", mi_bits);
///
/// // MI > 0.01 bits suggests signal has predictive value
/// if mi_bits < 0.01 {
///     println!("Signal is likely noise - consider removing");
/// }
/// ```
#[derive(Debug, Clone)]
pub struct MutualInfoEstimator {
    /// Number of nearest neighbors (typically 3-5)
    k: usize,
    /// Minimum samples required for valid estimation
    min_samples: usize,
}

impl MutualInfoEstimator {
    /// Create a new MI estimator with specified k.
    ///
    /// # Arguments
    /// * `k` - Number of nearest neighbors. Higher k reduces variance but increases bias.
    ///   Recommended: 3 for small samples (<500), 5 for larger samples.
    pub fn new(k: usize) -> Self {
        Self {
            k: k.max(1),
            min_samples: 50,
        }
    }

    /// Set minimum samples required for estimation.
    pub fn with_min_samples(mut self, min_samples: usize) -> Self {
        self.min_samples = min_samples;
        self
    }

    /// Estimate mutual information I(X;Y) in nats.
    ///
    /// # Arguments
    /// * `x` - First variable samples
    /// * `y` - Second variable samples (must be same length as x)
    ///
    /// # Returns
    /// MI estimate in nats. Multiply by log2(e) ≈ 1.4427 for bits.
    /// Returns 0.0 if insufficient samples.
    pub fn estimate(&self, x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.len() < self.min_samples {
            return 0.0;
        }

        self.ksg_estimator_1(x, y)
    }

    /// Estimate MI in bits (more interpretable for feature selection).
    pub fn estimate_bits(&self, x: &[f64], y: &[f64]) -> f64 {
        self.estimate(x, y) * std::f64::consts::LOG2_E
    }

    /// KSG Algorithm 1: Uses max-norm distance in joint space.
    ///
    /// This is the more common variant and works well for most distributions.
    fn ksg_estimator_1(&self, x: &[f64], y: &[f64]) -> f64 {
        let n = x.len();
        let k = self.k.min(n - 1);

        // For each point, find k-th neighbor distance in joint space
        // then count points within that distance in marginal spaces
        let mut sum_psi_nx_ny = 0.0;

        for i in 0..n {
            // Find k-th neighbor distance in joint space (max-norm)
            let eps = self.kth_neighbor_distance_joint(x, y, i, k);

            // Count points within eps in each marginal space
            let nx = self.count_within_distance_1d(x, x[i], eps);
            let ny = self.count_within_distance_1d(y, y[i], eps);

            // Accumulate digamma terms
            sum_psi_nx_ny += digamma(nx as f64) + digamma(ny as f64);
        }

        // KSG formula: I(X;Y) = ψ(k) - ⟨ψ(nₓ) + ψ(nᵧ)⟩ + ψ(N)
        let mi = digamma(k as f64) - sum_psi_nx_ny / n as f64 + digamma(n as f64);

        // MI is non-negative (can be slightly negative due to estimation error)
        mi.max(0.0)
    }

    /// Find distance to k-th nearest neighbor in joint (x,y) space using max-norm.
    fn kth_neighbor_distance_joint(&self, x: &[f64], y: &[f64], idx: usize, k: usize) -> f64 {
        let n = x.len();

        // Use binary heap to find k smallest distances efficiently
        let mut heap: BinaryHeap<OrderedFloat> = BinaryHeap::new();

        for j in 0..n {
            if j == idx {
                continue;
            }

            // Max-norm distance: max(|xᵢ - xⱼ|, |yᵢ - yⱼ|)
            let dx = (x[idx] - x[j]).abs();
            let dy = (y[idx] - y[j]).abs();
            let dist = dx.max(dy);

            if heap.len() < k {
                heap.push(OrderedFloat(dist));
            } else if let Some(&OrderedFloat(max)) = heap.peek() {
                if dist < max {
                    heap.pop();
                    heap.push(OrderedFloat(dist));
                }
            }
        }

        // Return k-th smallest distance
        heap.peek().map(|f| f.0).unwrap_or(f64::INFINITY)
    }

    /// Count points within distance eps of target in 1D (strictly less than eps).
    fn count_within_distance_1d(&self, data: &[f64], target: f64, eps: f64) -> usize {
        data.iter()
            .filter(|&&v| (v - target).abs() < eps)
            .count()
            .max(1) // Avoid digamma(0)
    }
}

// ============================================================================
// Signal Catalog for Feature Tracking
// ============================================================================

/// Tracked signal types for MI-based feature selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SignalType {
    // Order book signals
    BookImbalance,
    NearTouchSize,
    BookSlope,
    SpreadPercentile,

    // Trade flow signals
    TradeImbalance,
    AgressorRatio,
    TradeIntensity,
    VolumeSpike,

    // Price signals
    Momentum1s,
    Momentum10s,
    Momentum60s,
    TrendAgreement,

    // Volatility signals
    SigmaClean,
    SigmaTotal,
    JumpRatio,
    RegimeScore,

    // Cross-asset signals
    BinanceLead,
    FundingRate,
    OpenInterestDelta,
}

impl SignalType {
    /// Human-readable name for reporting.
    pub fn name(&self) -> &'static str {
        match self {
            Self::BookImbalance => "book_imbalance",
            Self::NearTouchSize => "near_touch_size",
            Self::BookSlope => "book_slope",
            Self::SpreadPercentile => "spread_percentile",
            Self::TradeImbalance => "trade_imbalance",
            Self::AgressorRatio => "agressor_ratio",
            Self::TradeIntensity => "trade_intensity",
            Self::VolumeSpike => "volume_spike",
            Self::Momentum1s => "momentum_1s",
            Self::Momentum10s => "momentum_10s",
            Self::Momentum60s => "momentum_60s",
            Self::TrendAgreement => "trend_agreement",
            Self::SigmaClean => "sigma_clean",
            Self::SigmaTotal => "sigma_total",
            Self::JumpRatio => "jump_ratio",
            Self::RegimeScore => "regime_score",
            Self::BinanceLead => "binance_lead",
            Self::FundingRate => "funding_rate",
            Self::OpenInterestDelta => "oi_delta",
        }
    }
}

/// Target prediction variables.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TargetType {
    /// Price direction (sign of return over horizon)
    PriceDirection,
    /// Whether our order got filled (binary)
    Fill,
    /// Adverse selection (post-fill price movement)
    AdverseSelection,
    /// Whether this is a regime shift
    RegimeChange,
}

impl TargetType {
    pub fn name(&self) -> &'static str {
        match self {
            Self::PriceDirection => "price_direction",
            Self::Fill => "fill",
            Self::AdverseSelection => "adverse_selection",
            Self::RegimeChange => "regime_change",
        }
    }
}

// ============================================================================
// Signal Quality Tracker
// ============================================================================

/// Tracks MI history for a single signal-target pair.
#[derive(Debug, Clone)]
pub struct SignalQualityTracker {
    signal: SignalType,
    target: TargetType,
    estimator: MutualInfoEstimator,
    /// Ring buffer of (timestamp_ms, signal_value)
    signal_buffer: Vec<(i64, f64)>,
    /// Ring buffer of (timestamp_ms, target_value)
    target_buffer: Vec<(i64, f64)>,
    /// Maximum buffer size
    max_samples: usize,
    /// Historical MI estimates (for decay tracking)
    mi_history: Vec<(i64, f64)>, // (timestamp_ms, mi_bits)
    /// Last computed MI
    last_mi_bits: f64,
}

impl SignalQualityTracker {
    pub fn new(signal: SignalType, target: TargetType) -> Self {
        Self {
            signal,
            target,
            estimator: MutualInfoEstimator::new(3),
            signal_buffer: Vec::with_capacity(1000),
            target_buffer: Vec::with_capacity(1000),
            max_samples: 1000,
            mi_history: Vec::new(),
            last_mi_bits: 0.0,
        }
    }

    pub fn with_max_samples(mut self, max_samples: usize) -> Self {
        self.max_samples = max_samples;
        self
    }

    /// Add a signal observation.
    pub fn add_signal(&mut self, timestamp_ms: i64, value: f64) {
        if self.signal_buffer.len() >= self.max_samples {
            self.signal_buffer.remove(0);
        }
        self.signal_buffer.push((timestamp_ms, value));
    }

    /// Add a target observation (outcome).
    pub fn add_target(&mut self, timestamp_ms: i64, value: f64) {
        if self.target_buffer.len() >= self.max_samples {
            self.target_buffer.remove(0);
        }
        self.target_buffer.push((timestamp_ms, value));
    }

    /// Compute current MI estimate.
    ///
    /// Aligns signal and target by timestamp before computing.
    pub fn compute_mi(&mut self, current_timestamp_ms: i64) -> f64 {
        let aligned = self.align_samples();
        if aligned.is_empty() {
            return 0.0;
        }

        let (signals, targets): (Vec<f64>, Vec<f64>) = aligned.into_iter().unzip();
        let mi_bits = self.estimator.estimate_bits(&signals, &targets);

        // Update history
        self.last_mi_bits = mi_bits;
        self.mi_history.push((current_timestamp_ms, mi_bits));

        // Keep last 30 days of history
        if self.mi_history.len() > 30 {
            self.mi_history.remove(0);
        }

        mi_bits
    }

    /// Align signal and target samples by closest timestamp.
    fn align_samples(&self) -> Vec<(f64, f64)> {
        let mut aligned = Vec::new();

        for &(sig_ts, sig_val) in &self.signal_buffer {
            // Find closest target within 100ms
            if let Some(target_val) = self.find_closest_target(sig_ts, 100) {
                aligned.push((sig_val, target_val));
            }
        }

        aligned
    }

    fn find_closest_target(&self, timestamp_ms: i64, max_delta_ms: i64) -> Option<f64> {
        self.target_buffer
            .iter()
            .filter(|(ts, _)| (ts - timestamp_ms).abs() <= max_delta_ms)
            .min_by_key(|(ts, _)| (ts - timestamp_ms).abs())
            .map(|(_, val)| *val)
    }

    /// Get the most recent MI estimate.
    pub fn current_mi_bits(&self) -> f64 {
        self.last_mi_bits
    }

    /// Estimate MI decay rate (bits per day).
    ///
    /// Returns None if insufficient history.
    pub fn decay_rate(&self) -> Option<f64> {
        if self.mi_history.len() < 7 {
            return None;
        }

        // Simple linear regression
        let n = self.mi_history.len() as f64;
        let (sum_t, sum_mi, sum_t2, sum_t_mi) =
            self.mi_history
                .iter()
                .fold((0.0, 0.0, 0.0, 0.0), |(st, sm, st2, stm), &(ts, mi)| {
                    let t = ts as f64 / (24.0 * 3600.0 * 1000.0); // Convert to days
                    (st + t, sm + mi, st2 + t * t, stm + t * mi)
                });

        let denom = n * sum_t2 - sum_t * sum_t;
        if denom.abs() < 1e-10 {
            return None;
        }

        let slope = (n * sum_t_mi - sum_t * sum_mi) / denom;
        Some(slope)
    }

    /// Estimate half-life of MI decay in days.
    ///
    /// Returns None if MI is not decaying or insufficient history.
    pub fn half_life_days(&self) -> Option<f64> {
        let rate = self.decay_rate()?;

        // Only meaningful if decaying (negative rate)
        if rate >= 0.0 || self.last_mi_bits < 1e-6 {
            return None;
        }

        // Half-life = -MI / (2 * rate)
        Some(-self.last_mi_bits / (2.0 * rate))
    }

    /// Check if signal is stale (MI dropped significantly or below threshold).
    pub fn is_stale(&self, threshold_bits: f64) -> bool {
        if self.last_mi_bits < threshold_bits {
            return true;
        }

        // Check for rapid decay
        if let Some(half_life) = self.half_life_days() {
            if half_life < 7.0 {
                return true; // Will be useless within a week
            }
        }

        false
    }

    pub fn signal_type(&self) -> SignalType {
        self.signal
    }

    pub fn target_type(&self) -> TargetType {
        self.target
    }
}

// ============================================================================
// Aggregate Signal Audit
// ============================================================================

/// Manages MI tracking across all signal-target pairs.
#[derive(Debug, Clone)]
pub struct SignalAuditManager {
    trackers: Vec<SignalQualityTracker>,
    /// Minimum MI threshold for inclusion (bits)
    mi_threshold_bits: f64,
}

impl SignalAuditManager {
    pub fn new() -> Self {
        Self {
            trackers: Vec::new(),
            mi_threshold_bits: 0.01, // Default: 0.01 bits
        }
    }

    /// Set MI threshold below which signals are considered noise.
    pub fn with_threshold(mut self, threshold_bits: f64) -> Self {
        self.mi_threshold_bits = threshold_bits;
        self
    }

    /// Add a signal-target tracker.
    pub fn add_tracker(&mut self, tracker: SignalQualityTracker) {
        self.trackers.push(tracker);
    }

    /// Create default trackers for common signal-target pairs.
    pub fn with_default_trackers(mut self) -> Self {
        // Book imbalance → price direction
        self.add_tracker(SignalQualityTracker::new(
            SignalType::BookImbalance,
            TargetType::PriceDirection,
        ));

        // Trade imbalance → fill
        self.add_tracker(SignalQualityTracker::new(
            SignalType::TradeImbalance,
            TargetType::Fill,
        ));

        // Momentum → adverse selection
        self.add_tracker(SignalQualityTracker::new(
            SignalType::Momentum10s,
            TargetType::AdverseSelection,
        ));

        // Jump ratio → regime change
        self.add_tracker(SignalQualityTracker::new(
            SignalType::JumpRatio,
            TargetType::RegimeChange,
        ));

        self
    }

    /// Compute MI for all trackers.
    pub fn compute_all(&mut self, current_timestamp_ms: i64) {
        for tracker in &mut self.trackers {
            tracker.compute_mi(current_timestamp_ms);
        }
    }

    /// Get ranked signals by MI (descending).
    pub fn rank_signals(&self) -> Vec<SignalRankEntry> {
        let mut entries: Vec<_> = self
            .trackers
            .iter()
            .map(|t| SignalRankEntry {
                signal: t.signal_type(),
                target: t.target_type(),
                mi_bits: t.current_mi_bits(),
                half_life_days: t.half_life_days(),
                is_stale: t.is_stale(self.mi_threshold_bits),
            })
            .collect();

        entries.sort_by(|a, b| {
            b.mi_bits
                .partial_cmp(&a.mi_bits)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        entries
    }

    /// Get signals that should be removed (MI below threshold or rapidly decaying).
    pub fn stale_signals(&self) -> Vec<SignalType> {
        self.trackers
            .iter()
            .filter(|t| t.is_stale(self.mi_threshold_bits))
            .map(|t| t.signal_type())
            .collect()
    }

    /// Generate a text report of signal quality.
    pub fn generate_report(&self) -> String {
        let ranked = self.rank_signals();
        let mut report = String::new();

        report.push_str("=== Signal Audit Report ===\n\n");
        report.push_str(&format!(
            "MI Threshold: {:.4} bits\n\n",
            self.mi_threshold_bits
        ));

        report.push_str("| Signal | Target | MI (bits) | Half-Life | Status |\n");
        report.push_str("|--------|--------|-----------|-----------|--------|\n");

        for entry in &ranked {
            let half_life_str = match entry.half_life_days {
                Some(hl) => format!("{:.1}d", hl),
                None => "N/A".to_string(),
            };
            let status = if entry.is_stale { "STALE" } else { "OK" };

            report.push_str(&format!(
                "| {} | {} | {:.4} | {} | {} |\n",
                entry.signal.name(),
                entry.target.name(),
                entry.mi_bits,
                half_life_str,
                status
            ));
        }

        let stale_count = ranked.iter().filter(|e| e.is_stale).count();
        if stale_count > 0 {
            report.push_str(&format!(
                "\nWarning: {} signal(s) are stale and should be reviewed.\n",
                stale_count
            ));
        }

        report
    }
}

impl Default for SignalAuditManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Entry in signal ranking.
#[derive(Debug, Clone)]
pub struct SignalRankEntry {
    pub signal: SignalType,
    pub target: TargetType,
    pub mi_bits: f64,
    pub half_life_days: Option<f64>,
    pub is_stale: bool,
}

// ============================================================================
// Helper: Digamma Function
// ============================================================================

/// Digamma function ψ(x) = d/dx ln(Γ(x)) = Γ'(x)/Γ(x).
///
/// Uses asymptotic expansion for large x and recurrence for small x.
fn digamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NEG_INFINITY;
    }

    // Use recurrence to move to larger x
    let mut result = 0.0;
    let mut x = x;

    while x < 6.0 {
        result -= 1.0 / x;
        x += 1.0;
    }

    // Asymptotic expansion: ψ(x) ≈ ln(x) - 1/(2x) - 1/(12x²) + 1/(120x⁴) - ...
    result += x.ln() - 0.5 / x;
    let x2 = x * x;
    result -= 1.0 / (12.0 * x2);
    result += 1.0 / (120.0 * x2 * x2);
    result -= 1.0 / (252.0 * x2 * x2 * x2);

    result
}

// ============================================================================
// Helper: OrderedFloat for BinaryHeap
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq)]
struct OrderedFloat(f64);

impl Eq for OrderedFloat {}

impl PartialOrd for OrderedFloat {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(std::cmp::Ordering::Equal)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_digamma_basic() {
        // ψ(1) = -γ ≈ -0.5772
        let psi_1 = digamma(1.0);
        assert!((psi_1 - (-EULER_GAMMA)).abs() < 0.01);

        // ψ(2) = 1 - γ ≈ 0.4228
        let psi_2 = digamma(2.0);
        assert!((psi_2 - (1.0 - EULER_GAMMA)).abs() < 0.01);
    }

    #[test]
    fn test_mi_independent_variables() {
        // For independent variables, MI should be close to 0
        let estimator = MutualInfoEstimator::new(3).with_min_samples(30);

        let x: Vec<f64> = (0..100).map(|i| (i as f64).sin()).collect();
        let y: Vec<f64> = (0..100).map(|i| ((i as f64) * 1.618).cos()).collect(); // Different phase

        let mi = estimator.estimate_bits(&x, &y);
        assert!(
            mi < 0.1,
            "MI for independent vars should be near 0, got {}",
            mi
        );
    }

    #[test]
    fn test_mi_identical_variables() {
        // For identical variables, MI should be high
        let estimator = MutualInfoEstimator::new(3).with_min_samples(30);

        let x: Vec<f64> = (0..100).map(|i| i as f64 / 100.0).collect();
        let y = x.clone();

        let mi = estimator.estimate_bits(&x, &y);
        // MI(X;X) = H(X), which depends on the distribution
        // For uniform [0,1], H ≈ log₂(100) ≈ 6.6 bits
        assert!(mi > 1.0, "MI for identical vars should be high, got {}", mi);
    }

    #[test]
    fn test_mi_linear_relationship() {
        // For Y = aX + b, MI should equal H(X) for noiseless case
        let estimator = MutualInfoEstimator::new(3).with_min_samples(30);

        let x: Vec<f64> = (0..100).map(|i| i as f64 / 100.0).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();

        let mi = estimator.estimate_bits(&x, &y);
        assert!(
            mi > 1.0,
            "MI for linear relationship should be high, got {}",
            mi
        );
    }

    #[test]
    fn test_mi_noisy_relationship() {
        // Add noise to reduce MI
        let estimator = MutualInfoEstimator::new(3).with_min_samples(30);

        let x: Vec<f64> = (0..100).map(|i| i as f64 / 100.0).collect();
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, &xi)| xi + 0.5 * ((i as f64 * 7.0).sin())) // Add noise
            .collect();

        let mi_noisy = estimator.estimate_bits(&x, &y);

        // Pure linear for comparison
        let y_pure: Vec<f64> = x.iter().map(|&xi| xi).collect();
        let mi_pure = estimator.estimate_bits(&x, &y_pure);

        assert!(
            mi_noisy < mi_pure,
            "Noisy MI {} should be less than pure MI {}",
            mi_noisy,
            mi_pure
        );
    }

    #[test]
    fn test_signal_quality_tracker() {
        let mut tracker =
            SignalQualityTracker::new(SignalType::BookImbalance, TargetType::PriceDirection)
                .with_max_samples(100);

        // Add some correlated data
        for i in 0..100 {
            let ts = i as i64 * 1000;
            let signal = (i as f64).sin();
            let target = signal + 0.1 * ((i as f64 * 2.0).cos()); // Noisy correlation

            tracker.add_signal(ts, signal);
            tracker.add_target(ts, target);
        }

        let mi = tracker.compute_mi(100_000);
        assert!(mi > 0.0, "Should have non-zero MI for correlated data");
    }

    #[test]
    fn test_insufficient_samples() {
        let estimator = MutualInfoEstimator::new(3);

        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0, 3.0];

        let mi = estimator.estimate(&x, &y);
        assert_eq!(mi, 0.0, "Should return 0 for insufficient samples");
    }
}
