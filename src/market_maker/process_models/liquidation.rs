//! Liquidation cascade detection for tail risk management.
//!
//! Uses a size-weighted Hawkes process to model liquidation clustering.
//! Large liquidations excite more future liquidations than small ones,
//! creating cascade dynamics.
//!
//! Key components:
//! - Size-weighted Hawkes: λ(t) = μ + ∫ α × size(s) × e^(-β(t-s)) dN(s)
//! - Cascade detection: intensity > 3× baseline
//! - Risk response: gamma multipliers and quote pulling

use std::collections::VecDeque;
use std::time::Instant;
use tracing::{debug, warn};

// ============================================================================
// O(1) Incremental Hawkes Intensity Tracker
// ============================================================================

/// O(1) incremental Hawkes process intensity tracker.
///
/// Instead of iterating over all events for each intensity computation,
/// we maintain a weighted sum that decays uniformly. The key insight is:
///
/// λ(t) = μ + Σᵢ α × sᵢ × exp(-β(t - tᵢ))
///
/// At time t₀, if we have:
///   S = Σᵢ α × sᵢ × exp(-β(t₀ - tᵢ))
///
/// Then at time t₁ > t₀:
///   S' = S × exp(-β(t₁ - t₀))  [all past events decay uniformly]
///
/// When new event arrives with size s:
///   S' = S' + α × s
///
/// This gives us O(1) updates and queries instead of O(n).
#[derive(Debug, Clone)]
pub struct IncrementalHawkes {
    /// Baseline intensity (μ)
    baseline: f64,
    /// Self-excitation parameter (α)
    alpha: f64,
    /// Decay parameter (β, per second)
    beta: f64,
    /// Size scaling factor
    size_scale: f64,
    /// Cumulative weighted intensity from past events (excludes baseline)
    /// This is Σᵢ α × sᵢ × exp(-β(t_now - tᵢ)) at last update time
    weighted_sum: f64,
    /// Last update timestamp for decay calculation
    last_update: Instant,
}

impl IncrementalHawkes {
    /// Create a new incremental Hawkes tracker.
    pub fn new(baseline: f64, alpha: f64, beta: f64, size_scale: f64) -> Self {
        Self {
            baseline,
            alpha,
            beta,
            size_scale,
            weighted_sum: 0.0,
            last_update: Instant::now(),
        }
    }

    /// Decay the weighted sum to current time.
    #[inline]
    fn decay_to_now(&mut self, now: Instant) {
        let dt = now.duration_since(self.last_update).as_secs_f64();
        if dt > 0.0 {
            self.weighted_sum *= (-self.beta * dt).exp();
            self.last_update = now;
        }
    }

    /// Record a new event and return updated intensity.
    ///
    /// This is O(1) - just decay existing sum and add new contribution.
    #[inline]
    pub fn record_event(&mut self, size: f64) -> f64 {
        let now = Instant::now();
        self.decay_to_now(now);

        // Add new event contribution
        let normalized_size = (size / self.size_scale).max(0.01);
        self.weighted_sum += self.alpha * normalized_size;

        self.baseline + self.weighted_sum
    }

    /// Get current intensity without recording an event.
    ///
    /// This is O(1) - just decay and return.
    #[inline]
    pub fn intensity(&mut self) -> f64 {
        let now = Instant::now();
        self.decay_to_now(now);
        self.baseline + self.weighted_sum
    }

    /// Get current intensity (const version using last known state).
    ///
    /// Note: This doesn't decay to current time, so may be slightly stale.
    /// Use for queries where exact freshness isn't critical.
    #[inline]
    pub fn intensity_snapshot(&self) -> f64 {
        let dt = self.last_update.elapsed().as_secs_f64();
        let decayed_sum = self.weighted_sum * (-self.beta * dt).exp();
        self.baseline + decayed_sum
    }

    /// Get intensity ratio (current / baseline).
    #[inline]
    pub fn intensity_ratio(&mut self) -> f64 {
        let intensity = self.intensity();
        if self.baseline > 0.0 {
            intensity / self.baseline
        } else {
            1.0
        }
    }

    /// Get intensity ratio from snapshot.
    #[inline]
    pub fn intensity_ratio_snapshot(&self) -> f64 {
        let intensity = self.intensity_snapshot();
        if self.baseline > 0.0 {
            intensity / self.baseline
        } else {
            1.0
        }
    }

    /// Reset to baseline (clears all event history).
    pub fn reset(&mut self) {
        self.weighted_sum = 0.0;
        self.last_update = Instant::now();
    }
}

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for liquidation cascade detection.
#[derive(Debug, Clone)]
pub struct LiquidationConfig {
    /// Baseline liquidation intensity (μ) - expected liquidations per second
    /// in normal market conditions
    pub baseline_intensity: f64,

    /// Self-excitation parameter (α) - how much each liquidation increases intensity
    /// Higher values = stronger cascade dynamics
    pub alpha: f64,

    /// Decay parameter (β) - how quickly excitation decays (per second)
    /// Higher values = faster decay back to baseline
    pub beta: f64,

    /// Size scaling factor - normalizes liquidation sizes
    /// Intensity contribution = α × (size / size_scale)
    pub size_scale: f64,

    /// Cascade threshold multiplier - cascade detected when intensity > threshold × baseline
    pub cascade_threshold: f64,

    /// Quote pull threshold multiplier - pull all quotes when intensity > threshold × baseline
    pub quote_pull_threshold: f64,

    /// Minimum gamma multiplier (floor)
    pub gamma_min: f64,

    /// Maximum gamma multiplier (ceiling)
    pub gamma_max: f64,

    /// Maximum events to track (memory bound)
    pub max_events: usize,

    /// Warmup period (seconds) before detection is active
    pub warmup_seconds: f64,
}

impl Default for LiquidationConfig {
    fn default() -> Self {
        Self {
            baseline_intensity: 0.01,  // 1 liquidation per 100 seconds baseline
            alpha: 0.5,                // Moderate self-excitation
            beta: 0.1,                 // 10-second decay half-life
            size_scale: 10000.0,       // Normalize by $10k
            cascade_threshold: 3.0,    // 3× baseline = cascade
            quote_pull_threshold: 5.0, // 5× baseline = pull quotes
            gamma_min: 1.0,            // No reduction below 1.0
            gamma_max: 5.0,            // Max 5× gamma increase
            max_events: 500,           // Track last 500 events
            warmup_seconds: 60.0,      // 1 minute warmup
        }
    }
}

// ============================================================================
// Liquidation Event
// ============================================================================

/// A liquidation event for Hawkes process tracking.
#[derive(Debug, Clone)]
struct LiquidationEvent {
    /// Event timestamp
    time: Instant,
    /// Normalized size (size / size_scale)
    /// NOTE: This field is used by compute_intensity_bruteforce() in tests
    /// for numerical validation against the O(1) incremental tracker.
    #[allow(dead_code)]
    normalized_size: f64,
    /// Direction: true = long liquidated (price dropping), false = short liquidated (price rising)
    is_long_liquidation: bool,
}

/// Direction of a detected cascade.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CascadeDirection {
    /// Longs being liquidated (price dropping)
    Long,
    /// Shorts being liquidated (price rising)
    Short,
    /// Both sides being liquidated (high volatility)
    Both,
}

// ============================================================================
// Liquidation Cascade Detector
// ============================================================================

/// Detects liquidation cascades using a size-weighted Hawkes process.
///
/// Uses O(1) incremental intensity tracking for performance.
#[derive(Debug)]
pub struct LiquidationCascadeDetector {
    config: LiquidationConfig,

    /// O(1) incremental Hawkes intensity tracker
    hawkes: IncrementalHawkes,

    /// Recent liquidation events (kept for direction detection only, not intensity)
    events: VecDeque<LiquidationEvent>,

    /// Start time for warmup tracking
    start_time: Instant,

    /// Total liquidations observed
    total_liquidations: usize,

    /// Recent long liquidation count (for direction detection)
    recent_long_count: usize,

    /// Recent short liquidation count (for direction detection)
    recent_short_count: usize,

    /// Cached intensity (updated on each observation)
    cached_intensity: f64,

    /// Cached cascade state
    cached_cascade_active: bool,
}

impl LiquidationCascadeDetector {
    /// Create a new liquidation cascade detector.
    pub fn new(config: LiquidationConfig) -> Self {
        let baseline = config.baseline_intensity;
        let hawkes = IncrementalHawkes::new(
            config.baseline_intensity,
            config.alpha,
            config.beta,
            config.size_scale,
        );
        Self {
            config,
            hawkes,
            events: VecDeque::new(),
            start_time: Instant::now(),
            total_liquidations: 0,
            recent_long_count: 0,
            recent_short_count: 0,
            cached_intensity: baseline, // Start at baseline
            cached_cascade_active: false,
        }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(LiquidationConfig::default())
    }

    /// Check if detector is warmed up.
    pub fn is_warmed_up(&self) -> bool {
        self.start_time.elapsed().as_secs_f64() >= self.config.warmup_seconds
    }

    /// Record a liquidation event.
    ///
    /// # Parameters
    /// - `size`: Notional value of the liquidation (in USD or base currency)
    /// - `is_long_liquidation`: true if a long position was liquidated
    ///
    /// Complexity: O(1) using incremental Hawkes tracker.
    pub fn record_liquidation(&mut self, size: f64, is_long_liquidation: bool) {
        // Update O(1) incremental Hawkes intensity
        self.cached_intensity = self.hawkes.record_event(size);
        self.cached_cascade_active =
            self.cached_intensity > self.config.baseline_intensity * self.config.cascade_threshold;

        // Maintain event list for direction detection only (bounded by max_events)
        while self.events.len() >= self.config.max_events {
            let removed = self.events.pop_front();
            if let Some(ev) = removed {
                if ev.is_long_liquidation {
                    self.recent_long_count = self.recent_long_count.saturating_sub(1);
                } else {
                    self.recent_short_count = self.recent_short_count.saturating_sub(1);
                }
            }
        }

        let normalized_size = (size / self.config.size_scale).max(0.01);

        let event = LiquidationEvent {
            time: Instant::now(),
            normalized_size,
            is_long_liquidation,
        };

        self.events.push_back(event);
        self.total_liquidations += 1;

        if is_long_liquidation {
            self.recent_long_count += 1;
        } else {
            self.recent_short_count += 1;
        }

        debug!(
            size = size,
            normalized_size = normalized_size,
            is_long = is_long_liquidation,
            intensity = self.cached_intensity,
            cascade = self.cached_cascade_active,
            "Liquidation: Recorded event"
        );

        if self.cached_cascade_active {
            warn!(
                intensity = self.cached_intensity,
                threshold = self.config.baseline_intensity * self.config.cascade_threshold,
                "Liquidation: CASCADE DETECTED"
            );
        }
    }

    /// Update cached intensity and cascade state.
    ///
    /// Complexity: O(1) using incremental Hawkes tracker.
    fn update_intensity(&mut self) {
        self.cached_intensity = self.hawkes.intensity();
        self.cached_cascade_active =
            self.cached_intensity > self.config.baseline_intensity * self.config.cascade_threshold;
    }

    /// Compute current Hawkes intensity using O(n) iteration (for testing/validation only).
    ///
    /// λ(t) = μ + Σ α × size_i × e^(-β × (t - t_i))
    ///
    /// NOTE: This is the original O(n) implementation kept for numerical validation.
    /// Production code uses the O(1) incremental tracker.
    #[cfg(test)]
    fn compute_intensity_bruteforce(&self, now: Instant) -> f64 {
        let mut intensity = self.config.baseline_intensity;

        for event in &self.events {
            let dt = now.duration_since(event.time).as_secs_f64();
            let contribution =
                self.config.alpha * event.normalized_size * (-self.config.beta * dt).exp();
            intensity += contribution;
        }

        intensity
    }

    /// Periodic update - call this regularly to decay old events.
    pub fn update(&mut self) {
        let now = Instant::now();

        // Remove very old events (contribution negligible)
        let decay_threshold = 10.0 / self.config.beta; // ~10 half-lives
        while let Some(front) = self.events.front() {
            if now.duration_since(front.time).as_secs_f64() > decay_threshold {
                let removed = self.events.pop_front().unwrap();
                if removed.is_long_liquidation {
                    self.recent_long_count = self.recent_long_count.saturating_sub(1);
                } else {
                    self.recent_short_count = self.recent_short_count.saturating_sub(1);
                }
            } else {
                break;
            }
        }

        // Update cached values
        self.update_intensity();
    }

    // ========================================================================
    // Public Query Methods
    // ========================================================================

    /// Get current intensity.
    pub fn intensity(&self) -> f64 {
        self.cached_intensity
    }

    /// Get intensity ratio (current / baseline).
    pub fn intensity_ratio(&self) -> f64 {
        if self.config.baseline_intensity > 0.0 {
            self.cached_intensity / self.config.baseline_intensity
        } else {
            1.0
        }
    }

    /// Check if a cascade is currently active.
    pub fn cascade_active(&self) -> bool {
        self.is_warmed_up() && self.cached_cascade_active
    }

    /// Get cascade severity as a normalized score [0, 1].
    /// 0 = at baseline, 1 = at or above quote pull threshold.
    pub fn cascade_severity(&self) -> f64 {
        if !self.is_warmed_up() {
            return 0.0;
        }

        let ratio = self.intensity_ratio();
        let threshold_range = self.config.quote_pull_threshold - 1.0;

        if threshold_range <= 0.0 {
            return 0.0;
        }

        ((ratio - 1.0) / threshold_range).clamp(0.0, 1.0)
    }

    /// Detect cascade direction based on recent liquidation mix.
    pub fn cascade_direction(&self) -> Option<CascadeDirection> {
        if !self.cascade_active() {
            return None;
        }

        let total = self.recent_long_count + self.recent_short_count;
        if total == 0 {
            return None;
        }

        let long_ratio = self.recent_long_count as f64 / total as f64;

        if long_ratio > 0.7 {
            Some(CascadeDirection::Long)
        } else if long_ratio < 0.3 {
            Some(CascadeDirection::Short)
        } else {
            Some(CascadeDirection::Both)
        }
    }

    /// Calculate tail risk multiplier for gamma scaling.
    ///
    /// Returns a value in [gamma_min, gamma_max] based on cascade severity.
    /// Higher values = more risk averse (wider spreads).
    pub fn tail_risk_multiplier(&self) -> f64 {
        if !self.is_warmed_up() {
            return 1.0;
        }

        let severity = self.cascade_severity();

        // Linear interpolation from gamma_min to gamma_max
        let multiplier =
            self.config.gamma_min + severity * (self.config.gamma_max - self.config.gamma_min);

        multiplier.clamp(self.config.gamma_min, self.config.gamma_max)
    }

    /// Check if quotes should be pulled (extreme cascade).
    pub fn should_pull_quotes(&self) -> bool {
        if !self.is_warmed_up() {
            return false;
        }

        self.intensity_ratio() >= self.config.quote_pull_threshold
    }

    /// Calculate size reduction factor for graceful degradation.
    ///
    /// Returns a value in [0, 1] to multiply order sizes by.
    /// 1.0 = full size, 0.0 = zero size (effectively pulled).
    pub fn size_reduction_factor(&self) -> f64 {
        if !self.is_warmed_up() {
            return 1.0;
        }

        // Start reducing at cascade threshold, reach 0 at pull threshold
        let ratio = self.intensity_ratio();

        if ratio <= 1.0 {
            return 1.0;
        }

        let cascade_start = self.config.cascade_threshold;
        let pull_point = self.config.quote_pull_threshold;

        if ratio >= pull_point {
            return 0.0;
        }

        if ratio <= cascade_start {
            return 1.0;
        }

        // Linear interpolation
        let progress = (ratio - cascade_start) / (pull_point - cascade_start);
        (1.0 - progress).clamp(0.0, 1.0)
    }

    /// Get expected number of liquidations in the next N seconds.
    ///
    /// For a Hawkes process, E[N(t, t+T)] ≈ ∫ λ(s) ds
    /// With our parametrization, this is approximately λ(t) × T for short T.
    pub fn expected_liquidations(&self, horizon_seconds: f64) -> f64 {
        // Simple approximation: intensity × horizon
        // More accurate would integrate the decaying intensity
        self.cached_intensity * horizon_seconds
    }

    // === Cascade Hedging Cost Feedback (First Principles Gap 7) ===

    /// Calculate expected hedging cost during a cascade.
    ///
    /// During liquidation cascades, slippage increases due to:
    /// 1. Reduced liquidity (MMs pull quotes)
    /// 2. Increased volatility
    /// 3. Order book imbalance from forced liquidations
    ///
    /// Returns the expected additional slippage cost in USD.
    pub fn cascade_hedging_cost(&self, size: f64, mid_price: f64) -> f64 {
        if !self.is_warmed_up() || !self.cascade_active() {
            return 0.0;
        }

        let severity = self.cascade_severity();
        let notional = size.abs() * mid_price;

        // Base slippage assumption: 5 bps in normal conditions
        // Cascade increases slippage by up to 10x at max severity
        let base_slippage_bps = 5.0;
        let cascade_multiplier = 1.0 + 9.0 * severity; // 1x to 10x

        let slippage_bps = base_slippage_bps * cascade_multiplier;
        notional * slippage_bps / 10000.0
    }

    /// Get cascade-adjusted maximum position.
    ///
    /// Reduces max position based on cascade severity to limit
    /// exposure during volatile periods.
    pub fn cascade_adjusted_max_position(&self, base_max: f64) -> f64 {
        base_max * self.size_reduction_factor()
    }

    /// Get spread adjustment factor during cascade.
    ///
    /// Returns a multiplier (>= 1.0) to widen spreads by.
    /// Accounts for increased adverse selection during cascades.
    pub fn cascade_spread_multiplier(&self) -> f64 {
        if !self.is_warmed_up() || !self.cascade_active() {
            return 1.0;
        }

        let severity = self.cascade_severity();
        // Widen spreads by up to 3x during severe cascades
        1.0 + 2.0 * severity
    }

    /// Get diagnostic summary.
    pub fn summary(&self) -> LiquidationSummary {
        LiquidationSummary {
            is_warmed_up: self.is_warmed_up(),
            total_liquidations: self.total_liquidations,
            recent_events: self.events.len(),
            intensity: self.cached_intensity,
            intensity_ratio: self.intensity_ratio(),
            cascade_active: self.cascade_active(),
            cascade_severity: self.cascade_severity(),
            cascade_direction: self.cascade_direction(),
            tail_risk_multiplier: self.tail_risk_multiplier(),
            should_pull_quotes: self.should_pull_quotes(),
            size_reduction_factor: self.size_reduction_factor(),
        }
    }
}

/// Summary of liquidation state for logging.
#[derive(Debug, Clone)]
pub struct LiquidationSummary {
    pub is_warmed_up: bool,
    pub total_liquidations: usize,
    pub recent_events: usize,
    pub intensity: f64,
    pub intensity_ratio: f64,
    pub cascade_active: bool,
    pub cascade_severity: f64,
    pub cascade_direction: Option<CascadeDirection>,
    pub tail_risk_multiplier: f64,
    pub should_pull_quotes: bool,
    pub size_reduction_factor: f64,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_detector() -> LiquidationCascadeDetector {
        LiquidationCascadeDetector::new(LiquidationConfig {
            baseline_intensity: 0.1,
            alpha: 1.0,
            beta: 1.0,
            size_scale: 1000.0,
            cascade_threshold: 3.0,
            quote_pull_threshold: 5.0,
            gamma_min: 1.0,
            gamma_max: 5.0,
            max_events: 100,
            warmup_seconds: 0.0, // No warmup for tests
        })
    }

    #[test]
    fn test_baseline_intensity() {
        let detector = make_detector();

        // With no events, intensity should be at baseline
        assert!((detector.intensity() - 0.1).abs() < 0.01);
        assert!((detector.intensity_ratio() - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_liquidation_increases_intensity() {
        let mut detector = make_detector();
        let initial = detector.intensity();

        // Record a liquidation
        detector.record_liquidation(10000.0, true);

        // Intensity should increase
        assert!(detector.intensity() > initial);
    }

    #[test]
    fn test_cascade_detection() {
        let mut detector = make_detector();

        // Initially no cascade
        assert!(!detector.cascade_active());

        // Simulate cascade - many liquidations in short time
        for _ in 0..20 {
            detector.record_liquidation(5000.0, true);
        }

        // Should detect cascade
        assert!(detector.cascade_active());
        assert!(detector.intensity_ratio() > 3.0);
    }

    #[test]
    fn test_cascade_direction() {
        let mut detector = make_detector();

        // Mostly long liquidations
        for _ in 0..10 {
            detector.record_liquidation(5000.0, true); // long
        }
        detector.record_liquidation(5000.0, false); // short

        if detector.cascade_active() {
            assert_eq!(detector.cascade_direction(), Some(CascadeDirection::Long));
        }
    }

    #[test]
    fn test_tail_risk_multiplier() {
        let mut detector = make_detector();

        // At baseline, multiplier should be 1.0
        let baseline_mult = detector.tail_risk_multiplier();
        assert!((baseline_mult - 1.0).abs() < 0.1);

        // During cascade, multiplier should increase
        for _ in 0..20 {
            detector.record_liquidation(5000.0, true);
        }

        let cascade_mult = detector.tail_risk_multiplier();
        assert!(cascade_mult > baseline_mult);
    }

    #[test]
    fn test_should_pull_quotes() {
        let mut detector = make_detector();

        assert!(!detector.should_pull_quotes());

        // Extreme cascade
        for _ in 0..50 {
            detector.record_liquidation(10000.0, true);
        }

        // Should recommend pulling quotes
        assert!(detector.should_pull_quotes());
    }

    #[test]
    fn test_size_reduction_factor() {
        let mut detector = make_detector();

        // At baseline, full size
        assert!((detector.size_reduction_factor() - 1.0).abs() < 0.01);

        // During cascade, reduced size
        for _ in 0..15 {
            detector.record_liquidation(5000.0, true);
        }

        let factor = detector.size_reduction_factor();
        assert!(factor < 1.0);
        assert!(factor >= 0.0);
    }

    #[test]
    fn test_intensity_decay() {
        let mut detector = make_detector();

        detector.record_liquidation(10000.0, true);
        let peak = detector.intensity();

        // Wait for decay
        std::thread::sleep(std::time::Duration::from_millis(1100));
        detector.update();

        // Intensity should have decayed
        assert!(detector.intensity() < peak);
    }

    #[test]
    fn test_max_events_limit() {
        let mut detector = LiquidationCascadeDetector::new(LiquidationConfig {
            max_events: 5,
            warmup_seconds: 0.0,
            ..Default::default()
        });

        for i in 0..10 {
            detector.record_liquidation(1000.0, i % 2 == 0);
        }

        // Should only track max_events
        assert_eq!(detector.events.len(), 5);
    }

    #[test]
    fn test_expected_liquidations() {
        let mut detector = make_detector();

        // Add some liquidations to increase intensity
        for _ in 0..5 {
            detector.record_liquidation(5000.0, true);
        }

        let expected = detector.expected_liquidations(10.0);
        assert!(expected > 0.0);

        // Expected should scale with horizon
        let expected_longer = detector.expected_liquidations(20.0);
        assert!((expected_longer - 2.0 * expected).abs() < 0.1);
    }

    // ========================================================================
    // O(1) Incremental Hawkes Tests
    // ========================================================================

    #[test]
    fn test_incremental_hawkes_baseline() {
        let mut hawkes = IncrementalHawkes::new(0.1, 1.0, 1.0, 1000.0);

        // With no events, intensity should be baseline
        assert!((hawkes.intensity() - 0.1).abs() < 1e-9);
        assert!((hawkes.intensity_ratio() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_incremental_hawkes_event_increases_intensity() {
        let mut hawkes = IncrementalHawkes::new(0.1, 1.0, 1.0, 1000.0);
        let initial = hawkes.intensity();

        // Record an event
        let new_intensity = hawkes.record_event(5000.0);

        // Intensity should increase
        assert!(new_intensity > initial);

        // Contribution should be α × (size / scale) = 1.0 × 5.0 = 5.0
        // So intensity = 0.1 + 5.0 = 5.1
        assert!((new_intensity - 5.1).abs() < 0.01);
    }

    #[test]
    fn test_incremental_hawkes_decay() {
        let mut hawkes = IncrementalHawkes::new(0.1, 1.0, 1.0, 1000.0);

        // Record event
        let peak = hawkes.record_event(5000.0);

        // Wait for decay (with β=1.0, after 1 second, e^-1 ≈ 0.368)
        std::thread::sleep(std::time::Duration::from_millis(1000));

        let after_decay = hawkes.intensity();

        // Should have decayed significantly
        assert!(after_decay < peak);
        // Expected: 0.1 + 5.0 * e^-1 ≈ 0.1 + 1.84 = 1.94
        assert!((after_decay - 1.94).abs() < 0.2);
    }

    #[test]
    fn test_incremental_hawkes_multiple_events() {
        let mut hawkes = IncrementalHawkes::new(0.1, 1.0, 1.0, 1000.0);

        // Add multiple events quickly
        for _ in 0..10 {
            hawkes.record_event(1000.0);
        }

        // Each event contributes α × 1.0 = 1.0
        // Total should be approximately 0.1 + 10.0 = 10.1 (with minimal decay)
        let intensity = hawkes.intensity();
        assert!(intensity > 9.0); // At least 9 due to minimal decay
        assert!(intensity < 11.0);
    }

    #[test]
    fn test_incremental_hawkes_snapshot() {
        let hawkes = IncrementalHawkes::new(0.1, 1.0, 1.0, 1000.0);

        // Snapshot should return baseline for new tracker
        assert!((hawkes.intensity_snapshot() - 0.1).abs() < 1e-9);
    }

    #[test]
    fn test_incremental_vs_bruteforce_numerical_equivalence() {
        // This test validates that the O(1) incremental implementation
        // produces the same results as the O(n) bruteforce implementation
        let mut detector = make_detector();

        // Record some events
        for i in 0..20 {
            detector.record_liquidation(1000.0 + i as f64 * 100.0, i % 2 == 0);
        }

        // Get both intensities
        let incremental = detector.cached_intensity;
        let bruteforce = detector.compute_intensity_bruteforce(Instant::now());

        // They should be very close (within numerical precision)
        let relative_error = (incremental - bruteforce).abs() / bruteforce.max(1e-9);
        assert!(
            relative_error < 0.01, // 1% tolerance for timing differences
            "Incremental ({}) and bruteforce ({}) differ by {:.2}%",
            incremental,
            bruteforce,
            relative_error * 100.0
        );
    }

    #[test]
    fn test_incremental_vs_bruteforce_after_decay() {
        let mut detector = make_detector();

        // Record events
        for i in 0..10 {
            detector.record_liquidation(2000.0, i % 2 == 0);
        }

        // Wait for some decay
        std::thread::sleep(std::time::Duration::from_millis(500));

        // Update via periodic update (which uses incremental)
        detector.update();

        let now = Instant::now();
        let incremental = detector.cached_intensity;
        let bruteforce = detector.compute_intensity_bruteforce(now);

        // Should still be close after decay
        let relative_error = (incremental - bruteforce).abs() / bruteforce.max(1e-9);
        assert!(
            relative_error < 0.05, // 5% tolerance for timing variations
            "After decay: incremental ({}) vs bruteforce ({}) differ by {:.2}%",
            incremental,
            bruteforce,
            relative_error * 100.0
        );
    }

    #[test]
    fn test_incremental_reset() {
        let mut hawkes = IncrementalHawkes::new(0.1, 1.0, 1.0, 1000.0);

        // Add events
        for _ in 0..10 {
            hawkes.record_event(5000.0);
        }

        assert!(hawkes.intensity() > 10.0);

        // Reset
        hawkes.reset();

        // Should be back to baseline
        assert!((hawkes.intensity() - 0.1).abs() < 1e-9);
    }
}
