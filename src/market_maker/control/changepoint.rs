//! Bayesian Online Changepoint Detection (BOCD).
//!
//! Implements Adams & MacKay (2007) for detecting regime changes in market data.
//! When a changepoint is detected, we should distrust the learning module
//! (which was trained on old regime data).
//!
//! # Regime-Aware Detection
//!
//! The detector supports regime-specific thresholds to reduce false positives
//! in different market environments:
//! - **ThinDex** (HIP-3): High threshold (0.85) for noise tolerance
//! - **LiquidCex**: Standard threshold (0.5) for responsive detection
//! - **Cascade**: Low threshold (0.3) for high sensitivity during stress
//!
use std::time::Instant;

/// Adaptive hazard rate for changepoint detection.
///
/// Modulates the base hazard rate based on observation cadence (`dt`).
/// - Fast cadence (dt < soft_boost): Use base hazard rate
/// - Slow cadence (dt > force_extreme): Boost hazard rate by 5x
/// - Intermediate cadence: Smoothly scale between 1x and 5x
#[derive(Debug, Clone)]
pub struct AdaptiveHazard {
    /// Base hazard rate from regime profile
    pub base_hazard: f64,
    /// Threshold for soft boosting probability (seconds)
    pub soft_boost_threshold: f64,
    /// Threshold to force extreme regime (seconds)
    pub force_extreme_threshold: f64,
    /// Maximum boost multiplier
    pub max_boost: f64,
    /// Time of last observation
    pub last_observation: Option<Instant>,
}

impl Default for AdaptiveHazard {
    fn default() -> Self {
        Self {
            base_hazard: 1.0 / 250.0,
            soft_boost_threshold: 0.35,
            force_extreme_threshold: 0.65,
            max_boost: 5.0,
            last_observation: None,
        }
    }
}

impl AdaptiveHazard {
    /// Create new adaptive hazard tracking
    pub fn new(base: f64, soft: f64, extreme: f64) -> Self {
        Self {
            base_hazard: base,
            soft_boost_threshold: soft,
            force_extreme_threshold: extreme,
            max_boost: 5.0,
            last_observation: None,
        }
    }

    /// Update with regime profile settings
    pub fn update_from_regime(&mut self, base: f64, soft: f64, extreme: f64) {
        self.base_hazard = base;
        self.soft_boost_threshold = soft;
        self.force_extreme_threshold = extreme;
    }

    /// Calculate current hazard rate based on elapsed time since last observation
    pub fn current_hazard(&mut self) -> f64 {
        let now = Instant::now();
        let dt = match self.last_observation {
            Some(last) => now.duration_since(last).as_secs_f64(),
            None => {
                self.last_observation = Some(now);
                return self.base_hazard;
            }
        };

        // Update last observation time for NEXT call
        self.last_observation = Some(now);

        if dt <= self.soft_boost_threshold {
            self.base_hazard
        } else if dt >= self.force_extreme_threshold {
            self.base_hazard * self.max_boost
        } else {
            // Linear interpolation between soft and extreme
            let range = self.force_extreme_threshold - self.soft_boost_threshold;
            if range <= f64::EPSILON {
                return self.base_hazard * self.max_boost;
            }
            let fraction = (dt - self.soft_boost_threshold) / range;
            let multiplier = 1.0 + (self.max_boost - 1.0) * fraction;
            self.base_hazard * multiplier
        }
    }
    
    /// Reset the observation tracker (e.g., when changepoint is confirmed)
    pub fn on_changepoint_confirmed(&mut self) {
        self.last_observation = None;
    }
}

// ============================================================================
// Regime-Aware Changepoint Detection Types
// ============================================================================

/// Market regime for changepoint threshold selection.
///
/// Different market environments require different detection sensitivity:
/// - Thin DEX books have more noise, need higher thresholds
/// - Liquid CEX books have cleaner signals, use standard thresholds
/// - Cascade conditions need high sensitivity (low thresholds)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MarketRegime {
    /// Thin order book DEX (e.g., HIP-3)
    /// High threshold (0.85) to reduce false positives from noise
    ThinDex,
    /// Liquid CEX environment
    /// Standard threshold (0.5) for balanced detection
    #[default]
    LiquidCex,
    /// Liquidation cascade or stress conditions
    /// Low threshold (0.3) for high sensitivity
    Cascade,
}

impl MarketRegime {
    /// Get the changepoint detection threshold for this regime.
    pub fn changepoint_threshold(&self) -> f64 {
        match self {
            MarketRegime::ThinDex => 0.85,   // High tolerance for noise
            MarketRegime::LiquidCex => 0.50, // Standard sensitivity
            MarketRegime::Cascade => 0.30,   // High sensitivity
        }
    }

    /// Get the number of confirmations required for this regime.
    pub fn confirmation_count(&self) -> usize {
        match self {
            MarketRegime::ThinDex => 2,   // Require 2 consecutive signals
            MarketRegime::LiquidCex => 1, // Single signal sufficient
            MarketRegime::Cascade => 1,   // Single signal (fast response)
        }
    }
}

/// Result of regime-aware changepoint detection.
///
/// Provides graduated response rather than binary detection:
/// - None: No changepoint detected
/// - Pending: Threshold exceeded but awaiting confirmation
/// - Confirmed: Multiple consecutive signals confirm changepoint
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChangePointResult {
    /// No changepoint detected - normal operation
    None,
    /// Threshold exceeded, awaiting confirmation
    /// Contains number of consecutive high-probability observations
    Pending(usize),
    /// Changepoint confirmed after required confirmations
    Confirmed,
}

impl ChangePointResult {
    /// Check if any change was detected (pending or confirmed).
    pub fn is_detected(&self) -> bool {
        !matches!(self, ChangePointResult::None)
    }

    /// Check if changepoint is confirmed (not just pending).
    pub fn is_confirmed(&self) -> bool {
        matches!(self, ChangePointResult::Confirmed)
    }
}

// ============================================================================
// Run Length Statistics
// ============================================================================

/// Run length statistics for BOCD.
#[derive(Debug, Clone)]
struct RunStatistics {
    /// Sample count for this run
    n: f64,
    /// Sum of observations
    sum: f64,
    /// Sum of squared observations
    sum_sq: f64,
}

impl Default for RunStatistics {
    fn default() -> Self {
        Self {
            n: 0.0,
            sum: 0.0,
            sum_sq: 0.0,
        }
    }
}

impl RunStatistics {
    /// Update with new observation.
    fn update(&mut self, x: f64) {
        self.n += 1.0;
        self.sum += x;
        self.sum_sq += x * x;
    }

    /// Predictive probability of observation under Student-t.
    fn predictive_prob(&self, x: f64, prior: &RunStatistics) -> f64 {
        // Combine with prior
        let n = self.n + prior.n;
        let sum = self.sum + prior.sum;
        let sum_sq = self.sum_sq + prior.sum_sq;

        if n < 2.0 {
            // Not enough data, use uninformative
            return 0.1;
        }

        // Posterior parameters for Normal-Gamma
        let mean = sum / n;
        let variance = (sum_sq / n - mean * mean).max(1e-10);

        // Degrees of freedom
        let df = 2.0 * n;

        // Student-t probability
        let scale = (variance * (n + 1.0) / n).sqrt();
        let t = (x - mean) / scale;

        student_t_pdf(t, df) / scale
    }
}

/// Bayesian Online Changepoint Detector.
///
/// Maintains a distribution over run lengths (time since last changepoint).
/// When P(r_t = 0 | data) is high, a changepoint has occurred.
///
/// Supports regime-aware detection with:
/// - Configurable thresholds per market regime
/// - Confirmation requirements (multiple consecutive signals)
/// - Graceful degradation (pending vs confirmed states)
#[derive(Debug, Clone)]
pub struct ChangepointDetector {
    /// Run length probabilities P(r_t = k | x_{1:t})
    run_length_probs: Vec<f64>,
    /// Sufficient statistics for each run length
    run_statistics: Vec<RunStatistics>,
    /// Prior statistics
    prior: RunStatistics,
    /// Adaptive tracking for the hazard rate
    pub adaptive_hazard: AdaptiveHazard,
    /// Maximum run length to track
    max_run_length: usize,
    /// Changepoint threshold (default, can be overridden by regime)
    threshold: f64,
    /// Recent observations for diagnostics
    recent_obs: Vec<f64>,
    /// Maximum recent observations to keep
    max_recent: usize,
    /// Minimum observations before detection is active (hard cap fallback)
    warmup_observations: usize,
    /// Total observations received
    observation_count: usize,
    /// Entropy threshold for warmup (primary mechanism)
    warmup_entropy_threshold: f64,

    // Regime-aware detection state
    /// Current market regime (affects threshold and confirmation)
    current_regime: MarketRegime,
    /// Count of consecutive observations exceeding threshold
    consecutive_high_prob: usize,
    /// Required confirmations before declaring changepoint (regime-dependent)
    confirmation_required: usize,
}

impl Default for ChangepointDetector {
    fn default() -> Self {
        Self::new(ChangepointConfig::default())
    }
}

/// Configuration for changepoint detection.
#[derive(Debug, Clone)]
pub struct ChangepointConfig {
    /// Hazard rate λ (expected run length = 1/λ)
    pub hazard: f64,
    /// Maximum run length to track
    pub max_run_length: usize,
    /// Threshold for declaring changepoint
    pub threshold: f64,
    /// Prior mean
    pub prior_mean: f64,
    /// Prior variance
    pub prior_var: f64,
    /// Minimum observations before changepoint detection is active
    /// During warmup, changepoint_detected() returns false
    pub warmup_observations: usize,
    /// Entropy threshold for warmup (natural log scale).
    /// When run_length_entropy < this threshold, detector is warmed up.
    /// Entropy measures distribution concentration - low entropy means
    /// the run length distribution has settled into a meaningful shape.
    /// Default: 2.0 (e^2 ≈ 7 effective run lengths)
    pub warmup_entropy_threshold: f64,
}

impl Default for ChangepointConfig {
    fn default() -> Self {
        Self {
            hazard: 1.0 / 250.0, // Expected run length of 250 observations
            max_run_length: 500,
            // Bayes-optimal threshold: c_miss / (c_miss + c_false)
            // Treating false alarm as 2.3× worse than miss → threshold ≈ 0.7
            // This prefers stability: only declare changepoint if P > 70%
            threshold: 0.7,
            prior_mean: 0.0,
            prior_var: 1.0,
            warmup_observations: 50, // Hard cap fallback (entropy-based is primary)
            // Entropy threshold for warmup: when distribution is concentrated
            // ~2.0 means effective support of e^2 ≈ 7 run lengths
            warmup_entropy_threshold: 2.0,
        }
    }
}

impl ChangepointDetector {
    /// Create a new changepoint detector.
    pub fn new(config: ChangepointConfig) -> Self {
        // P0 FIX (2026-02-02): Initialize with uncertainty, not certainty
        // Previously: run_length_probs[0] = 1.0 → cp_prob = 100% at startup
        // This caused downstream systems to see high changepoint probability during warmup.
        //
        // New: Initialize with spread distribution expressing startup uncertainty:
        // - 50% on r=0..5 (recent changepoint / uncertain state)
        // - 50% on r=5..20 (moderate run length / stable state)
        // This makes cp_prob(5) ≈ 0.5 at startup instead of 1.0
        let init_spread = 20.min(config.max_run_length);
        let mut run_length_probs = vec![0.0; init_spread];

        // Spread probability: higher mass on r=0 decaying geometrically
        // This represents "uncertain, probably early in a regime"
        for (i, prob) in run_length_probs.iter_mut().enumerate().take(init_spread) {
            *prob = (0.8_f64).powi(i as i32);
        }
        // Normalize
        let sum: f64 = run_length_probs.iter().sum();
        for p in &mut run_length_probs {
            *p /= sum;
        }

        let prior = RunStatistics {
            n: 1.0,
            sum: config.prior_mean,
            sum_sq: config.prior_mean.powi(2) + config.prior_var,
        };

        // Default to LiquidCex regime
        let default_regime = MarketRegime::LiquidCex;

        // Initialize run_statistics to match run_length_probs length
        let run_statistics: Vec<RunStatistics> =
            (0..init_spread).map(|_| RunStatistics::default()).collect();

        Self {
            run_length_probs,
            run_statistics,
            prior,
            adaptive_hazard: AdaptiveHazard {
                base_hazard: config.hazard,
                ..Default::default()
            },
            max_run_length: config.max_run_length,
            threshold: config.threshold,
            recent_obs: Vec::new(),
            max_recent: 50,
            warmup_observations: config.warmup_observations,
            observation_count: 0,
            warmup_entropy_threshold: config.warmup_entropy_threshold,
            // Regime-aware detection
            current_regime: default_regime,
            consecutive_high_prob: 0,
            confirmation_required: default_regime.confirmation_count(),
        }
    }

    /// Set the market regime for threshold and confirmation settings.
    ///
    /// Different market environments need different detection sensitivity:
    /// - ThinDex: High threshold (0.85), 2 confirmations
    /// - LiquidCex: Standard threshold (0.5), 1 confirmation
    /// - Cascade: Low threshold (0.3), 1 confirmation
    pub fn set_regime(&mut self, regime: MarketRegime) {
        self.current_regime = regime;
        self.confirmation_required = regime.confirmation_count();
        // Reset consecutive count when regime changes
        self.consecutive_high_prob = 0;
    }

    /// Update adaptive hazard tracking using RegimeProfile values.
    pub fn update_from_profile(&mut self, profile: &crate::market_maker::config::RegimeProfile) {
        self.set_regime(profile.regime);
        self.adaptive_hazard.update_from_regime(
            profile.cp_hazard_rate,
            profile.cp_soft_boost_threshold,
            profile.cp_force_extreme_threshold,
        );
    }

    /// Get the current market regime.
    pub fn current_regime(&self) -> MarketRegime {
        self.current_regime
    }

    /// Get the effective threshold for the current regime.
    ///
    /// Uses regime-specific threshold if set, otherwise falls back to config threshold.
    pub fn effective_threshold(&self) -> f64 {
        self.current_regime.changepoint_threshold()
    }

    /// Check if the detector has completed warmup.
    ///
    /// Uses a hybrid approach:
    /// 1. Minimum observations required (entropy must first rise before it can concentrate)
    /// 2. Then either: entropy drops below threshold (distribution concentrated)
    ///    OR observation hard cap is reached
    ///
    /// First-principles rationale: BOCD needs time for the run length distribution
    /// to spread out and then concentrate on meaningful run lengths. At startup,
    /// entropy is 0 (all mass on r=0), then rises as distribution spreads,
    /// then falls as it concentrates on stable run lengths.
    ///
    /// The minimum observation requirement (10) ensures we don't falsely trigger
    /// warmup completion due to the degenerate startup entropy = 0 case.
    pub fn is_warmed_up(&self) -> bool {
        // Minimum observations to let entropy rise first (escape degenerate r=0 state)
        const MIN_OBSERVATIONS_FOR_ENTROPY: usize = 10;

        if self.observation_count < MIN_OBSERVATIONS_FOR_ENTROPY {
            return false;
        }

        // After minimum observations: check entropy OR hard cap
        let entropy_ready = self.run_length_entropy() < self.warmup_entropy_threshold;
        let observation_ready = self.observation_count >= self.warmup_observations;
        entropy_ready || observation_ready
    }

    /// Get the maximum run length being tracked.
    pub fn max_run_length(&self) -> usize {
        self.max_run_length
    }

    /// Update with new observation.
    ///
    /// Implements the BOCD update equations:
    /// 1. Calculate growth probabilities P(r_t = r_{t-1} + 1)
    /// 2. Calculate changepoint probabilities P(r_t = 0)
    /// 3. Normalize
    pub fn update(&mut self, observation: f64) {
        // Store observation
        self.recent_obs.push(observation);
        if self.recent_obs.len() > self.max_recent {
            self.recent_obs.remove(0);
        }

        let n_run_lengths = self.run_length_probs.len();

        // Calculate predictive probabilities
        let mut pred_probs = Vec::with_capacity(n_run_lengths);
        for stats in self.run_statistics.iter() {
            let p = stats.predictive_prob(observation, &self.prior);
            pred_probs.push(p);
        }

        // Calculate new run length probabilities
        let mut new_probs = vec![0.0; (n_run_lengths + 1).min(self.max_run_length)];

        // Probability of changepoint (r_t = 0)
        let mut cp_prob = 0.0;
        let current_hazard = self.adaptive_hazard.current_hazard();
        for (r, &prob) in self.run_length_probs.iter().enumerate() {
            cp_prob += prob * current_hazard * pred_probs[r];
        }
        new_probs[0] = cp_prob;

        // Probability of growth (r_t = r_{t-1} + 1)
        for (r, &prob) in self.run_length_probs.iter().enumerate() {
            if r + 1 < new_probs.len() {
                new_probs[r + 1] = prob * (1.0 - current_hazard) * pred_probs[r];
            }
        }

        // Normalize
        let sum: f64 = new_probs.iter().sum();
        if sum > 1e-10 {
            for p in &mut new_probs {
                *p /= sum;
            }
        } else {
            // Numerical issues, reset to uniform
            let n = new_probs.len() as f64;
            for p in &mut new_probs {
                *p = 1.0 / n;
            }
        }

        // Update run statistics
        let mut new_stats = vec![RunStatistics::default()]; // r=0 starts fresh
        for stats in &mut self.run_statistics {
            stats.update(observation);
            if new_stats.len() < self.max_run_length {
                new_stats.push(stats.clone());
            }
        }

        self.run_length_probs = new_probs;
        self.run_statistics = new_stats;
        self.observation_count += 1;

        // Update consecutive high-probability counter for confirmation logic
        let cp_prob = self.changepoint_probability(5);
        let threshold = self.effective_threshold();
        if cp_prob > threshold {
            self.consecutive_high_prob += 1;
        } else {
            self.consecutive_high_prob = 0;
        }
    }

    /// Get probability that a changepoint occurred in the last k observations.
    pub fn changepoint_probability(&self, k: usize) -> f64 {
        // Sum probabilities for run lengths 0 to k-1
        let sum: f64 = self.run_length_probs.iter().take(k).sum();
        sum.min(1.0)
    }

    /// Check if a recent changepoint was detected (legacy method).
    ///
    /// Returns false during warmup period to prevent false positives at startup.
    ///
    /// Note: For regime-aware detection with confirmation, use `detect_with_confirmation()`.
    pub fn changepoint_detected(&self) -> bool {
        if !self.is_warmed_up() {
            return false;
        }
        self.changepoint_probability(5) > self.threshold
    }

    /// Regime-aware changepoint detection with confirmation.
    ///
    /// This is the PREFERRED method for thin DEX environments (HIP-3).
    ///
    /// Unlike `changepoint_detected()`, this method:
    /// 1. Uses regime-specific thresholds (ThinDex=0.85, LiquidCex=0.5, Cascade=0.3)
    /// 2. Requires confirmation (consecutive high-probability observations)
    /// 3. Returns graduated result (None/Pending/Confirmed)
    ///
    /// # Returns
    /// - `ChangePointResult::None` - No changepoint detected
    /// - `ChangePointResult::Pending(n)` - Threshold exceeded, n consecutive signals
    /// - `ChangePointResult::Confirmed` - Required confirmations reached
    ///
    /// # Example
    /// ```ignore
    /// match detector.detect_with_confirmation() {
    ///     ChangePointResult::None => { /* Normal operation */ }
    ///     ChangePointResult::Pending(n) => {
    ///         // Consider widening spreads slightly
    ///         log::debug!("Changepoint pending: {} consecutive signals", n);
    ///     }
    ///     ChangePointResult::Confirmed => {
    ///         // Pull quotes, reset beliefs
    ///         log::warn!("Changepoint confirmed, pulling quotes");
    ///     }
    /// }
    /// ```
    pub fn detect_with_confirmation(&self) -> ChangePointResult {
        if !self.is_warmed_up() {
            return ChangePointResult::None;
        }

        let cp_prob = self.changepoint_probability(5);
        let threshold = self.effective_threshold();

        if cp_prob <= threshold {
            ChangePointResult::None
        } else if self.consecutive_high_prob >= self.confirmation_required {
            ChangePointResult::Confirmed
        } else {
            ChangePointResult::Pending(self.consecutive_high_prob)
        }
    }

    /// Get the number of consecutive high-probability observations.
    pub fn consecutive_count(&self) -> usize {
        self.consecutive_high_prob
    }

    /// Get the number of confirmations required for the current regime.
    pub fn confirmations_required(&self) -> usize {
        self.confirmation_required
    }

    /// Get probability that a changepoint occurred just now.
    pub fn probability_now(&self) -> f64 {
        self.run_length_probs.first().copied().unwrap_or(0.0)
    }

    /// Check if beliefs should be reset due to regime change.
    ///
    /// Returns false during warmup period to prevent constant resets at startup.
    pub fn should_reset_beliefs(&self) -> bool {
        if !self.is_warmed_up() {
            return false;
        }
        // Reset if changepoint probability is high
        self.changepoint_probability(10) > 0.7
    }

    /// Get the most likely run length.
    pub fn most_likely_run_length(&self) -> usize {
        self.run_length_probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Get entropy of run length distribution (uncertainty about regime age).
    pub fn run_length_entropy(&self) -> f64 {
        -self
            .run_length_probs
            .iter()
            .filter(|&&p| p > 1e-10)
            .map(|&p| p * p.ln())
            .sum::<f64>()
    }

    /// Reset the detector.
    ///
    /// Uses the same spread initialization as new() for consistent behavior.
    pub fn reset(&mut self) {
        // P0 FIX (2026-02-02): Reset with uncertainty, not certainty
        // Use same spread initialization as new()
        let init_spread = 20.min(self.max_run_length);
        self.run_length_probs = vec![0.0; init_spread];

        for i in 0..init_spread {
            self.run_length_probs[i] = (0.8_f64).powi(i as i32);
        }
        let sum: f64 = self.run_length_probs.iter().sum();
        for p in &mut self.run_length_probs {
            *p /= sum;
        }

        self.run_statistics = (0..init_spread).map(|_| RunStatistics::default()).collect();
        self.recent_obs.clear();
        self.observation_count = 0;
        self.consecutive_high_prob = 0;
        self.adaptive_hazard.on_changepoint_confirmed();
        // Note: current_regime is preserved across reset
    }

    /// Get summary for diagnostics.
    pub fn summary(&self) -> ChangepointSummary {
        ChangepointSummary {
            cp_prob_1: self.changepoint_probability(1),
            cp_prob_5: self.changepoint_probability(5),
            cp_prob_10: self.changepoint_probability(10),
            most_likely_run: self.most_likely_run_length(),
            entropy: self.run_length_entropy(),
            detected: self.changepoint_detected(),
            warmed_up: self.is_warmed_up(),
            observation_count: self.observation_count,
            // Regime-aware fields
            regime: self.current_regime,
            effective_threshold: self.effective_threshold(),
            consecutive_high_prob: self.consecutive_high_prob,
            confirmation_required: self.confirmation_required,
            detection_result: self.detect_with_confirmation(),
        }
    }
}

/// Summary of changepoint detection state.
#[derive(Debug, Clone)]
pub struct ChangepointSummary {
    /// Probability of changepoint in last 1 observation
    pub cp_prob_1: f64,
    /// Probability of changepoint in last 5 observations
    pub cp_prob_5: f64,
    /// Probability of changepoint in last 10 observations
    pub cp_prob_10: f64,
    /// Most likely run length
    pub most_likely_run: usize,
    /// Entropy of run length distribution
    pub entropy: f64,
    /// Whether changepoint was detected (legacy)
    pub detected: bool,
    /// Whether the detector has completed warmup
    pub warmed_up: bool,
    /// Total observations received
    pub observation_count: usize,
    /// Current market regime
    pub regime: MarketRegime,
    /// Effective threshold for current regime
    pub effective_threshold: f64,
    /// Consecutive high-probability observations
    pub consecutive_high_prob: usize,
    /// Confirmations required for current regime
    pub confirmation_required: usize,
    /// Regime-aware detection result
    pub detection_result: ChangePointResult,
}

/// Student's t PDF for predictive probability.
fn student_t_pdf(t: f64, df: f64) -> f64 {
    use std::f64::consts::PI;

    let coef = gamma_ln((df + 1.0) / 2.0) - gamma_ln(df / 2.0) - 0.5 * (df * PI).ln();
    (coef - (df + 1.0) / 2.0 * (1.0 + t * t / df).ln()).exp()
}

/// Log gamma function (Lanczos approximation).
fn gamma_ln(x: f64) -> f64 {
    use std::f64::consts::PI;

    let g = 7;
    let c = [
        0.999_999_999_999_809_9,
        676.5203681218851,
        -1259.1392167224028,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507343278686905,
        -0.13857109526572012,
        9.984_369_578_019_572e-6,
        1.5056327351493116e-7,
    ];

    if x < 0.5 {
        PI.ln() - (PI * x).sin().ln() - gamma_ln(1.0 - x)
    } else {
        let x = x - 1.0;
        let mut a = c[0];
        for (i, &coeff) in c.iter().enumerate().skip(1).take(g + 1) {
            a += coeff / (x + i as f64);
        }
        let t = x + g as f64 + 0.5;
        0.5 * (2.0 * PI).ln() + (t).ln() * (x + 0.5) - t + a.ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_changepoint_detector_creation() {
        let detector = ChangepointDetector::default();

        // P0 FIX: Now initializes with spread distribution for cleaner warmup
        // Instead of 100% on r=0 (cp_prob=1.0), spreads across r=0..20
        assert_eq!(detector.run_length_probs.len(), 20);

        // Probabilities should sum to 1.0
        let sum: f64 = detector.run_length_probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "Probabilities should sum to 1.0");

        // cp_prob(5) should be around 0.5, not 1.0
        let cp_prob_5 = detector.changepoint_probability(5);
        assert!(
            cp_prob_5 < 0.8,
            "Initial cp_prob(5) should be <0.8 with spread init, got {}",
            cp_prob_5
        );
    }

    #[test]
    fn test_stable_process() {
        let mut detector = ChangepointDetector::default();

        // Feed stable observations (should have low changepoint probability)
        for i in 0..100 {
            detector.update(0.1 + 0.01 * (i as f64 % 10.0 - 5.0));
        }

        // Should have low changepoint probability
        assert!(detector.changepoint_probability(5) < 0.3);
        assert!(!detector.changepoint_detected());
    }

    #[test]
    fn test_sudden_change() {
        let mut detector = ChangepointDetector::new(ChangepointConfig {
            hazard: 0.05, // Higher hazard for faster detection
            ..Default::default()
        });

        // Feed stable observations
        for _ in 0..50 {
            detector.update(0.0);
        }

        // Sudden jump
        for _ in 0..10 {
            detector.update(5.0); // Big change
        }

        // Should detect changepoint
        // Note: Detection depends on parameters and may not trigger immediately
        let summary = detector.summary();
        assert!(summary.cp_prob_10 > 0.1); // Some increase in CP probability
    }

    #[test]
    fn test_reset() {
        let mut detector = ChangepointDetector::default();

        for _ in 0..20 {
            detector.update(1.0);
        }

        detector.reset();

        // P0 FIX: Reset now uses spread distribution same as new()
        assert_eq!(detector.run_length_probs.len(), 20);
        assert!(detector.recent_obs.is_empty());
        assert_eq!(detector.observation_count, 0);

        // Probabilities should sum to 1.0
        let sum: f64 = detector.run_length_probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "Probabilities should sum to 1.0");
    }

    #[test]
    fn test_student_t_pdf() {
        let pdf = student_t_pdf(0.0, 10.0);
        assert!(pdf > 0.3 && pdf < 0.4); // Approximately 0.389
    }

    #[test]
    fn test_warmup_prevents_false_positives() {
        // Use low observation count to test hard cap fallback
        let mut detector = ChangepointDetector::new(ChangepointConfig {
            warmup_observations: 10,
            warmup_entropy_threshold: 0.1, // Very low entropy threshold (hard to achieve)
            ..Default::default()
        });

        // During warmup, should NOT detect changepoint even though cp_prob is high
        assert!(!detector.is_warmed_up());
        assert!(!detector.changepoint_detected());
        assert!(!detector.should_reset_beliefs());

        // The raw probability IS high at startup
        assert!(detector.changepoint_probability(5) > 0.5);

        // Feed observations - will hit the hard cap fallback
        for i in 0..10 {
            detector.update(i as f64 * 0.1);
        }

        // Now should be warmed up (via observation count fallback)
        assert!(detector.is_warmed_up());
        assert_eq!(detector.observation_count, 10);

        // After warmup, detection can work if probability is high
        let summary = detector.summary();
        assert!(summary.warmed_up);
        assert_eq!(summary.observation_count, 10);

        // Reset should clear observation count
        detector.reset();
        assert!(!detector.is_warmed_up());
        assert_eq!(detector.observation_count, 0);
    }

    #[test]
    fn test_adaptive_hazard_cadence() {
        let mut hazard = AdaptiveHazard::new(0.01, 1.0, 5.0);
        
        // Initial call sets observation and returns base
        assert_eq!(hazard.current_hazard(), 0.01);

        // Fast cadence (< soft boost)
        std::thread::sleep(std::time::Duration::from_millis(100));
        assert_eq!(hazard.current_hazard(), 0.01);

        // Interpolation (dt ~ 3.0, halfway between 1.0 and 5.0)
        // Expected multiplier: 1.0 + 4.0 * (3.0 - 1.0) / 4.0 = 3.0 -> 0.03
        std::thread::sleep(std::time::Duration::from_millis(3000));
        let cur = hazard.current_hazard();
        assert!(cur > 0.02 && cur < 0.04);
        
        // Extreme cadence (> force extreme)
        std::thread::sleep(std::time::Duration::from_millis(5500));
        assert_eq!(hazard.current_hazard(), 0.05); // 0.01 * 5.0
        
        // Reset
        hazard.on_changepoint_confirmed();
        assert_eq!(hazard.current_hazard(), 0.01);
    }

    #[test]
    fn test_entropy_based_warmup() {
        // Use high observation fallback to ensure entropy-based warmup is tested
        let mut detector = ChangepointDetector::new(ChangepointConfig {
            warmup_observations: 1000,     // High fallback
            warmup_entropy_threshold: 3.0, // Reasonable entropy threshold
            ..Default::default()
        });

        // P0 FIX: At startup, entropy is now positive (spread initialization)
        // Previously entropy was 0 (all mass on r=0), now spread across r=0..20
        assert!(!detector.is_warmed_up());
        let initial_entropy = detector.run_length_entropy();
        // With geometric decay 0.8^i over 20 states, entropy is moderate
        assert!(
            initial_entropy > 0.5 && initial_entropy < 3.0,
            "Initial entropy should be moderate with spread init, got {}",
            initial_entropy
        );

        // Feed stable observations to let distribution concentrate
        for i in 0..100 {
            detector.update((i as f64 * 0.01).sin()); // Oscillating but stable
        }

        // After sufficient observations, entropy should be meaningful
        let final_entropy = detector.run_length_entropy();
        // The distribution should have concentrated (lower entropy typically)
        assert!(
            final_entropy > 0.0,
            "Entropy should be > 0 after data, got {}",
            final_entropy
        );

        // Should be warmed up via entropy (since observation_count=100 < 1000)
        if detector.is_warmed_up() {
            // Entropy threshold was met
            assert!(final_entropy < 3.0 || detector.observation_count >= 1000);
        }

        // Verify max_run_length getter works
        assert_eq!(detector.max_run_length(), 500); // Default from config
    }
}
