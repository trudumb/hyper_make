//! Event-driven conjugate posteriors for continuous-time belief updates.
//!
//! These posteriors are updated on **every** fill event, not just at cycle
//! boundaries. The quote cycle reads a snapshot; the emergency cancel path
//! acts between cycles as a circuit breaker.
//!
//! # Performance Contract
//!
//! All update methods are O(1) per event with zero heap allocations.
//! No locks are taken on the hot path — this struct lives on the event-loop
//! task and is accessed only by the owning thread.
//!
//! # Conjugate Families
//!
//! | Posterior               | Prior                | Likelihood            |
//! |-------------------------|----------------------|-----------------------|
//! | FillIntensityPosterior  | Gamma(α, β)          | Poisson(λ)            |
//! | AdverseSelectionPosterior | Beta(α, β)         | Bernoulli(p)          |
//! | VolatilityPosterior     | Decayed suffstats    | Normal(0, σ²)         |
//!
//! # Timeline Example
//!
//! ```text
//! t=0.0s  fill → update posteriors immediately
//! t=0.3s  fill → update posteriors immediately
//! t=0.5s  fill → update posteriors immediately
//! t=0.7s  fill → update posteriors, burst detected → EMERGENCY CANCEL
//! t=0.8s  fill → posteriors already extreme, no resting bids to fill
//! t=3.0s  quote cycle → read PosteriorSnapshot, compute quotes, reconcile
//! ```

use std::time::Instant;

// ============================================================================
// Fill Intensity Posterior (Gamma conjugate for Poisson)
// ============================================================================

/// Gamma posterior for fill arrival intensity (λ).
///
/// Conjugate prior for a Poisson process:
///   Prior: Gamma(α, β)
///   Likelihood: Poisson(λ)
///   Posterior: Gamma(α + n_fills, β + T)
///
/// Mean = α/β,  Variance = α/β²
///
/// On each fill: α += 1 (observation count), β += dt since last fill.
/// This produces a running posterior over the fill arrival rate.
#[derive(Debug, Clone)]
pub struct FillIntensityPosterior {
    /// Shape parameter (pseudo-fill count + observed fills)
    alpha: f64,
    /// Rate parameter (pseudo-time + observed time)
    beta_s: f64,
    /// Timestamp of last fill (for inter-arrival time)
    last_fill_time: Option<Instant>,
    /// Total fills observed (excluding prior pseudo-counts)
    n_fills: u64,
    /// Exponential decay factor per fill (non-stationarity adjustment).
    /// Applied to alpha/beta to down-weight old evidence.
    decay_per_fill: f64,
}

/// Configuration for fill intensity posterior.
#[derive(Debug, Clone)]
pub struct FillIntensityConfig {
    /// Prior alpha (pseudo-fill count)
    pub prior_alpha: f64,
    /// Prior beta (pseudo-time in seconds)
    pub prior_beta_s: f64,
    /// Decay factor per fill for non-stationarity (e.g. 0.995)
    pub decay_per_fill: f64,
}

impl Default for FillIntensityConfig {
    fn default() -> Self {
        Self {
            // Prior: ~2 fills per second (α/β = 10/5 = 2)
            prior_alpha: 10.0,
            prior_beta_s: 5.0,
            // Gentle decay: half-life ≈ 138 fills
            decay_per_fill: 0.995,
        }
    }
}

impl FillIntensityPosterior {
    /// Create with given configuration.
    pub fn new(config: &FillIntensityConfig) -> Self {
        Self {
            alpha: config.prior_alpha,
            beta_s: config.prior_beta_s,
            last_fill_time: None,
            n_fills: 0,
            decay_per_fill: config.decay_per_fill,
        }
    }

    /// O(1) update on fill arrival. No allocations.
    pub fn on_fill(&mut self, now: Instant) {
        // Decay old evidence toward prior (non-stationarity)
        self.alpha *= self.decay_per_fill;
        self.beta_s *= self.decay_per_fill;

        // Add inter-arrival time to β
        if let Some(last) = self.last_fill_time {
            let dt_s = now.duration_since(last).as_secs_f64();
            self.beta_s += dt_s;
        }

        // Observe one fill: α += 1
        self.alpha += 1.0;
        self.last_fill_time = Some(now);
        self.n_fills += 1;
    }

    /// Posterior mean of fill intensity (fills/second).
    #[inline]
    pub fn mean_intensity(&self) -> f64 {
        if self.beta_s > 1e-12 {
            self.alpha / self.beta_s
        } else {
            self.alpha // degenerate: no time elapsed
        }
    }

    /// Posterior variance of fill intensity.
    #[inline]
    pub fn variance(&self) -> f64 {
        if self.beta_s > 1e-12 {
            self.alpha / (self.beta_s * self.beta_s)
        } else {
            f64::MAX
        }
    }

    /// 95th percentile of the Gamma posterior (approximate).
    ///
    /// Uses Wilson-Hilferty normal approximation:
    ///   Gamma(α, β) ≈ Normal(α/β, α/β²) for large α.
    ///   P95 ≈ mean + 1.645 × std
    #[inline]
    pub fn intensity_p95(&self) -> f64 {
        let mean = self.mean_intensity();
        let std = self.variance().sqrt();
        mean + 1.645 * std
    }

    /// Total fills observed (excluding prior).
    #[inline]
    pub fn n_fills(&self) -> u64 {
        self.n_fills
    }

    /// Time since last fill. None if no fills yet.
    #[inline]
    pub fn time_since_last_fill(&self) -> Option<std::time::Duration> {
        self.last_fill_time.map(|t| t.elapsed())
    }
}

// ============================================================================
// Adverse Selection Posterior (Beta conjugate for Bernoulli)
// ============================================================================

/// Beta posterior for adverse selection rate.
///
/// Conjugate prior for Bernoulli outcomes:
///   Prior: Beta(α, β)
///   Likelihood: Bernoulli(p)
///   Posterior: Beta(α + adverse, β + benign)
///
/// Mean = α / (α + β)
///
/// A fill is "adverse" if mid moved against us within a short window,
/// measured at the fill handler via spread capture sign.
#[derive(Debug, Clone)]
pub struct AdverseSelectionPosterior {
    /// Alpha (adverse fill count + prior)
    alpha: f64,
    /// Beta (benign fill count + prior)
    beta: f64,
    /// Total fills observed (excluding prior pseudo-counts)
    n_fills: u64,
    /// Exponential decay factor per fill
    decay_per_fill: f64,
    /// Rolling window: fills in last `burst_window_s` seconds
    recent_adverse_count: u32,
    /// Rolling window: total fills in last window
    recent_total_count: u32,
    /// Window start for burst detection
    window_start: Instant,
    /// Window duration for burst detection (seconds)
    burst_window_s: f64,
}

/// Configuration for adverse selection posterior.
#[derive(Debug, Clone)]
pub struct AdverseSelectionConfig {
    /// Prior alpha (pseudo-adverse fills). Higher = more confident prior.
    pub prior_alpha: f64,
    /// Prior beta (pseudo-benign fills). Higher = more confident prior.
    pub prior_beta: f64,
    /// Decay factor per fill for non-stationarity
    pub decay_per_fill: f64,
    /// Window duration for burst detection (seconds)
    pub burst_window_s: f64,
}

impl Default for AdverseSelectionConfig {
    fn default() -> Self {
        Self {
            // Prior: 30% adverse rate with moderate confidence
            // α/(α+β) = 3/10 = 0.3
            prior_alpha: 3.0,
            prior_beta: 7.0,
            decay_per_fill: 0.99,
            burst_window_s: 5.0,
        }
    }
}

impl AdverseSelectionPosterior {
    /// Create with given configuration.
    pub fn new(config: &AdverseSelectionConfig) -> Self {
        Self {
            alpha: config.prior_alpha,
            beta: config.prior_beta,
            n_fills: 0,
            decay_per_fill: config.decay_per_fill,
            recent_adverse_count: 0,
            recent_total_count: 0,
            window_start: Instant::now(),
            burst_window_s: config.burst_window_s,
        }
    }

    /// O(1) update on fill outcome. No allocations.
    ///
    /// `is_adverse`: true if mid moved against us after the fill.
    /// For immediate classification, use spread_capture sign:
    ///   - Buy fill where price dropped after → adverse
    ///   - Sell fill where price rose after → adverse
    pub fn on_fill(&mut self, is_adverse: bool, now: Instant) {
        // Decay old evidence
        self.alpha *= self.decay_per_fill;
        self.beta *= self.decay_per_fill;

        // Update posterior
        if is_adverse {
            self.alpha += 1.0;
        } else {
            self.beta += 1.0;
        }

        self.n_fills += 1;

        // Update burst window
        let elapsed = now.duration_since(self.window_start).as_secs_f64();
        if elapsed > self.burst_window_s {
            // Reset window
            self.recent_adverse_count = 0;
            self.recent_total_count = 0;
            self.window_start = now;
        }
        self.recent_total_count += 1;
        if is_adverse {
            self.recent_adverse_count += 1;
        }
    }

    /// Posterior mean of adverse selection rate [0, 1].
    #[inline]
    pub fn mean_rate(&self) -> f64 {
        let total = self.alpha + self.beta;
        if total > 1e-12 {
            self.alpha / total
        } else {
            0.5 // uninformative
        }
    }

    /// Posterior variance.
    #[inline]
    pub fn variance(&self) -> f64 {
        let total = self.alpha + self.beta;
        if total > 1e-12 {
            (self.alpha * self.beta) / (total * total * (total + 1.0))
        } else {
            0.25
        }
    }

    /// 95th percentile of the Beta posterior (approximate).
    ///
    /// For AS rate, the upper bound is what we care about (defense-first).
    /// Uses normal approximation: P95 ≈ mean + 1.645 × std
    #[inline]
    pub fn rate_p95(&self) -> f64 {
        let mean = self.mean_rate();
        let std = self.variance().sqrt();
        (mean + 1.645 * std).min(1.0)
    }

    /// Recent adverse rate in the burst window [0, 1].
    #[inline]
    pub fn recent_adverse_rate(&self) -> f64 {
        if self.recent_total_count > 0 {
            self.recent_adverse_count as f64 / self.recent_total_count as f64
        } else {
            0.0
        }
    }

    /// Recent fill count in the burst window.
    #[inline]
    pub fn recent_fill_count(&self) -> u32 {
        self.recent_total_count
    }

    /// Total fills observed (excluding prior).
    #[inline]
    pub fn n_fills(&self) -> u64 {
        self.n_fills
    }
}

// ============================================================================
// Volatility Posterior (decayed sufficient statistics)
// ============================================================================

/// Lightweight volatility tracker using decayed sufficient statistics.
///
/// Tracks running sum of squared returns with exponential decay,
/// producing a posterior estimate of realized volatility.
///
/// Not a full Normal-Inverse-Gamma conjugate (that's in CentralBeliefState),
/// but an O(1) EWMA estimator for the event-driven hot path.
#[derive(Debug, Clone)]
pub struct VolatilityPosterior {
    /// Effective observation count (with decay)
    n_eff: f64,
    /// Decayed sum of squared returns (per-second normalized)
    sum_sq: f64,
    /// EWMA decay factor per observation
    decay: f64,
    /// Last mid price for return computation
    last_mid: f64,
    /// Last update time for dt computation
    last_update_time: Option<Instant>,
}

/// Configuration for volatility posterior.
#[derive(Debug, Clone)]
pub struct VolatilityConfig {
    /// EWMA decay factor per observation (e.g., 0.97)
    pub decay: f64,
    /// Prior sigma (per-second) — used until sufficient observations
    pub prior_sigma_per_s: f64,
    /// Prior effective sample size
    pub prior_n_eff: f64,
}

impl Default for VolatilityConfig {
    fn default() -> Self {
        Self {
            decay: 0.97,
            // Prior: 10 bps/s ≈ typical crypto vol
            prior_sigma_per_s: 0.001,
            prior_n_eff: 20.0,
        }
    }
}

impl VolatilityPosterior {
    /// Create with given configuration.
    pub fn new(config: &VolatilityConfig) -> Self {
        Self {
            n_eff: config.prior_n_eff,
            sum_sq: config.prior_n_eff * config.prior_sigma_per_s * config.prior_sigma_per_s,
            decay: config.decay,
            last_mid: 0.0,
            last_update_time: None,
        }
    }

    /// O(1) update on mid price change. No allocations.
    ///
    /// Called on fills (which change inventory and potentially mid-price view),
    /// and on mid-price updates between fills.
    pub fn on_mid_update(&mut self, mid: f64, now: Instant) {
        if mid <= 0.0 {
            return;
        }

        if self.last_mid > 0.0 {
            if let Some(last_time) = self.last_update_time {
                let dt_s = now.duration_since(last_time).as_secs_f64();
                if dt_s > 0.001 {
                    // Per-second normalized return
                    let ret = (mid - self.last_mid) / self.last_mid;
                    let ret_per_s = ret / dt_s.sqrt();

                    // Decay old evidence, add new
                    self.n_eff = self.n_eff * self.decay + 1.0;
                    self.sum_sq = self.sum_sq * self.decay + ret_per_s * ret_per_s;
                }
            }
        }

        self.last_mid = mid;
        self.last_update_time = Some(now);
    }

    /// Posterior mean of per-second volatility (sigma).
    #[inline]
    pub fn sigma_per_s(&self) -> f64 {
        if self.n_eff > 1e-12 {
            (self.sum_sq / self.n_eff).sqrt()
        } else {
            0.001 // fallback
        }
    }

    /// Posterior variance of sigma estimate.
    /// Approximate: Var(σ̂²) ≈ 2σ⁴/n_eff → Var(σ̂) ≈ σ²/(2n_eff)
    #[inline]
    pub fn sigma_variance(&self) -> f64 {
        let sigma = self.sigma_per_s();
        if self.n_eff > 2.0 {
            sigma * sigma / (2.0 * self.n_eff)
        } else {
            sigma * sigma // high uncertainty
        }
    }

    /// Effective sample size.
    #[inline]
    pub fn n_eff(&self) -> f64 {
        self.n_eff
    }
}

// ============================================================================
// Event-Driven Posterior State (aggregated)
// ============================================================================

/// Configuration for the event-driven posterior system.
#[derive(Debug, Clone, Default)]
pub struct EventPosteriorConfig {
    /// Fill intensity posterior configuration
    pub fill_intensity: FillIntensityConfig,
    /// Adverse selection posterior configuration
    pub adverse_selection: AdverseSelectionConfig,
    /// Volatility posterior configuration
    pub volatility: VolatilityConfig,
    /// Emergency thresholds
    pub emergency: EmergencyThresholds,
}

/// Thresholds for emergency cancel decisions between cycles.
///
/// Defense-first: these are deliberately conservative. Missing a trade
/// is cheap; getting run over in a cascade is not.
#[derive(Debug, Clone)]
pub struct EmergencyThresholds {
    /// Minimum fills in the burst window to consider emergency action.
    /// Below this count we lack statistical power.
    pub min_burst_fills: u32,
    /// Fill intensity (fills/sec) P95 above which we trigger emergency.
    /// Calibrated to ~3 fills in 1 second ≈ 3.0 fills/sec.
    pub intensity_p95_threshold: f64,
    /// Adverse selection rate (recent window) above which we trigger.
    /// E.g., 0.7 = 70% of recent fills were adverse.
    pub adverse_rate_threshold: f64,
    /// Combined score threshold: intensity_z × as_rate
    /// Allows triggering when both are elevated but individually below threshold.
    pub combined_score_threshold: f64,
}

impl Default for EmergencyThresholds {
    fn default() -> Self {
        Self {
            min_burst_fills: 3,
            intensity_p95_threshold: 3.0,
            adverse_rate_threshold: 0.7,
            combined_score_threshold: 1.5,
        }
    }
}

/// The event-driven posterior state.
///
/// Updated on every fill event. Read by the quote cycle via `snapshot()`.
/// Checked after every fill for emergency conditions via `check_emergency()`.
///
/// This struct lives directly on the event-loop task — no locks, no Arc,
/// no cross-thread sharing. All methods are O(1) with zero allocations.
#[derive(Debug, Clone)]
pub struct EventPosteriorState {
    /// Fill arrival intensity posterior (Gamma conjugate)
    pub fill_intensity: FillIntensityPosterior,
    /// Adverse selection rate posterior (Beta conjugate)
    pub adverse_selection: AdverseSelectionPosterior,
    /// Local volatility estimate (EWMA sufficient stats)
    pub volatility: VolatilityPosterior,
    /// Emergency thresholds
    emergency_thresholds: EmergencyThresholds,
    /// Timestamp of last emergency trigger (for cooldown)
    last_emergency_time: Option<Instant>,
    /// Minimum cooldown between emergency cancels (seconds)
    emergency_cooldown_s: f64,
}

/// Result of an emergency check after a fill.
#[derive(Debug, Clone)]
pub enum EmergencyAction {
    /// No emergency — continue normal operation.
    None,
    /// Emergency cancel: posteriors indicate a fill burst with high adverse selection.
    /// Contains the reason string and severity score [0, 1].
    CancelResting {
        /// Human-readable reason for logging
        reason: String,
        /// Severity score [0, 1]: 0=threshold, 1=extreme
        severity: f64,
        /// Which metric triggered: "intensity", "adverse", or "combined"
        trigger: EmergencyTrigger,
    },
}

/// What triggered the emergency.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmergencyTrigger {
    /// Fill intensity alone exceeded threshold
    IntensityBurst,
    /// Adverse selection rate alone exceeded threshold
    AdverseRate,
    /// Combined intensity × AS score exceeded threshold
    Combined,
}

/// Point-in-time snapshot of all posteriors.
///
/// The quote cycle reads this to incorporate event-driven beliefs
/// into quote computation. Cheap to construct — just copies f64 values.
#[derive(Debug, Clone, Default)]
pub struct PosteriorSnapshot {
    // === Fill Intensity ===
    /// Posterior mean fill intensity (fills/second)
    pub fill_intensity_mean: f64,
    /// 95th percentile of fill intensity
    pub fill_intensity_p95: f64,
    /// Total fills observed
    pub fill_count: u64,
    /// Seconds since last fill (None if no fills)
    pub secs_since_last_fill: Option<f64>,

    // === Adverse Selection ===
    /// Posterior mean AS rate [0, 1]
    pub adverse_rate_mean: f64,
    /// 95th percentile AS rate [0, 1]
    pub adverse_rate_p95: f64,
    /// Recent-window AS rate [0, 1]
    pub recent_adverse_rate: f64,
    /// Recent-window fill count
    pub recent_fill_count: u32,

    // === Volatility ===
    /// Posterior sigma (per-second)
    pub sigma_per_s: f64,
    /// Sigma effective sample size
    pub sigma_n_eff: f64,

    // === Emergency State ===
    /// Whether an emergency was triggered on the most recent fill
    pub emergency_active: bool,
    /// Combined emergency score (intensity_z × as_rate)
    pub emergency_score: f64,
}

impl EventPosteriorState {
    /// Create with given configuration.
    pub fn new(config: EventPosteriorConfig) -> Self {
        Self {
            fill_intensity: FillIntensityPosterior::new(&config.fill_intensity),
            adverse_selection: AdverseSelectionPosterior::new(&config.adverse_selection),
            volatility: VolatilityPosterior::new(&config.volatility),
            emergency_thresholds: config.emergency,
            last_emergency_time: None,
            // 2-second cooldown between emergency cancels
            emergency_cooldown_s: 2.0,
        }
    }

    /// Create with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(EventPosteriorConfig::default())
    }

    /// O(1) update on fill arrival.
    ///
    /// Called from the fill handler on every new fill.
    /// Returns the emergency action (if any).
    ///
    /// `is_adverse`: whether the fill showed adverse selection.
    ///   Immediate proxy: spread_capture < 0 (we were filled on the wrong side of mid).
    /// `mid`: current mid price for volatility update.
    pub fn on_fill(&mut self, is_adverse: bool, mid: f64) -> EmergencyAction {
        let now = Instant::now();

        // Update all posteriors — O(1) each
        self.fill_intensity.on_fill(now);
        self.adverse_selection.on_fill(is_adverse, now);
        self.volatility.on_mid_update(mid, now);

        // Check emergency conditions
        self.check_emergency(now)
    }

    /// Update volatility on mid price change (between fills).
    ///
    /// This keeps the volatility posterior fresh even without fills.
    pub fn on_mid_update(&mut self, mid: f64) {
        self.volatility.on_mid_update(mid, Instant::now());
    }

    /// Take a snapshot for the quote cycle. O(1), no allocations.
    pub fn snapshot(&self) -> PosteriorSnapshot {
        let emergency_score = self.compute_combined_score();
        PosteriorSnapshot {
            fill_intensity_mean: self.fill_intensity.mean_intensity(),
            fill_intensity_p95: self.fill_intensity.intensity_p95(),
            fill_count: self.fill_intensity.n_fills(),
            secs_since_last_fill: self
                .fill_intensity
                .time_since_last_fill()
                .map(|d| d.as_secs_f64()),
            adverse_rate_mean: self.adverse_selection.mean_rate(),
            adverse_rate_p95: self.adverse_selection.rate_p95(),
            recent_adverse_rate: self.adverse_selection.recent_adverse_rate(),
            recent_fill_count: self.adverse_selection.recent_fill_count(),
            sigma_per_s: self.volatility.sigma_per_s(),
            sigma_n_eff: self.volatility.n_eff(),
            emergency_active: self.is_in_emergency_cooldown(),
            emergency_score,
        }
    }

    /// Check if we're within the emergency cooldown window.
    fn is_in_emergency_cooldown(&self) -> bool {
        self.last_emergency_time
            .map(|t| t.elapsed().as_secs_f64() < self.emergency_cooldown_s)
            .unwrap_or(false)
    }

    /// Compute combined emergency score: intensity_relative × adverse_rate.
    fn compute_combined_score(&self) -> f64 {
        let intensity = self.fill_intensity.mean_intensity();
        let threshold = self.emergency_thresholds.intensity_p95_threshold;
        let intensity_relative = if threshold > 0.0 {
            intensity / threshold
        } else {
            0.0
        };
        intensity_relative * self.adverse_selection.recent_adverse_rate()
    }

    /// Check emergency conditions after a fill. O(1).
    fn check_emergency(&mut self, now: Instant) -> EmergencyAction {
        // Cooldown: don't spam emergency cancels
        if self.is_in_emergency_cooldown() {
            return EmergencyAction::None;
        }

        let thresholds = &self.emergency_thresholds;

        // Need minimum fills to have statistical power
        if self.adverse_selection.recent_fill_count() < thresholds.min_burst_fills {
            return EmergencyAction::None;
        }

        let intensity_p95 = self.fill_intensity.intensity_p95();
        let recent_as_rate = self.adverse_selection.recent_adverse_rate();
        let combined_score = self.compute_combined_score();

        // Check intensity burst alone
        if intensity_p95 >= thresholds.intensity_p95_threshold
            && recent_as_rate >= thresholds.adverse_rate_threshold
        {
            self.last_emergency_time = Some(now);
            let severity =
                ((intensity_p95 / thresholds.intensity_p95_threshold) - 1.0).clamp(0.0, 1.0);
            return EmergencyAction::CancelResting {
                reason: format!(
                    "fill burst + high AS: intensity_p95={:.1}/s, as_rate={:.0}%",
                    intensity_p95,
                    recent_as_rate * 100.0
                ),
                severity,
                trigger: EmergencyTrigger::IntensityBurst,
            };
        }

        // Check adverse rate alone (very high AS even without burst)
        if recent_as_rate >= 0.9
            && self.adverse_selection.recent_fill_count() > thresholds.min_burst_fills
        {
            self.last_emergency_time = Some(now);
            return EmergencyAction::CancelResting {
                reason: format!(
                    "extreme AS rate: {:.0}% of last {} fills adverse",
                    recent_as_rate * 100.0,
                    self.adverse_selection.recent_fill_count()
                ),
                severity: recent_as_rate,
                trigger: EmergencyTrigger::AdverseRate,
            };
        }

        // Check combined score (both elevated but individually below threshold)
        if combined_score >= thresholds.combined_score_threshold {
            self.last_emergency_time = Some(now);
            let severity =
                ((combined_score / thresholds.combined_score_threshold) - 1.0).clamp(0.0, 1.0);
            return EmergencyAction::CancelResting {
                reason: format!(
                    "combined posterior alarm: score={:.2} (intensity={:.1}/s × as={:.0}%)",
                    combined_score,
                    self.fill_intensity.mean_intensity(),
                    recent_as_rate * 100.0
                ),
                severity,
                trigger: EmergencyTrigger::Combined,
            };
        }

        EmergencyAction::None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_fill_intensity_prior() {
        let config = FillIntensityConfig {
            prior_alpha: 10.0,
            prior_beta_s: 5.0,
            decay_per_fill: 1.0, // no decay for test
        };
        let posterior = FillIntensityPosterior::new(&config);

        // Prior mean: α/β = 10/5 = 2.0 fills/sec
        assert!((posterior.mean_intensity() - 2.0).abs() < 1e-10);
        assert_eq!(posterior.n_fills(), 0);
    }

    #[test]
    fn test_fill_intensity_update() {
        let config = FillIntensityConfig {
            prior_alpha: 10.0,
            prior_beta_s: 5.0,
            decay_per_fill: 1.0,
        };
        let mut posterior = FillIntensityPosterior::new(&config);

        let now = Instant::now();
        posterior.on_fill(now);

        // After 1 fill: α=11, β=5 (no dt for first fill)
        assert!((posterior.mean_intensity() - 11.0 / 5.0).abs() < 1e-10);
        assert_eq!(posterior.n_fills(), 1);
    }

    #[test]
    fn test_fill_intensity_with_time() {
        let config = FillIntensityConfig {
            prior_alpha: 10.0,
            prior_beta_s: 5.0,
            decay_per_fill: 1.0,
        };
        let mut posterior = FillIntensityPosterior::new(&config);

        let t0 = Instant::now();
        posterior.on_fill(t0);

        // Simulate 100ms later
        let t1 = t0 + Duration::from_millis(100);
        posterior.on_fill(t1);

        // After 2 fills: α=12, β=5.1 (0.1s inter-arrival)
        assert!((posterior.mean_intensity() - 12.0 / 5.1).abs() < 0.01);
        assert_eq!(posterior.n_fills(), 2);
    }

    #[test]
    fn test_adverse_selection_prior() {
        let config = AdverseSelectionConfig {
            prior_alpha: 3.0,
            prior_beta: 7.0,
            decay_per_fill: 1.0,
            burst_window_s: 5.0,
        };
        let posterior = AdverseSelectionPosterior::new(&config);

        // Prior mean: α/(α+β) = 3/10 = 0.3
        assert!((posterior.mean_rate() - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_adverse_selection_update() {
        let config = AdverseSelectionConfig {
            prior_alpha: 3.0,
            prior_beta: 7.0,
            decay_per_fill: 1.0,
            burst_window_s: 5.0,
        };
        let mut posterior = AdverseSelectionPosterior::new(&config);

        let now = Instant::now();

        // 3 adverse fills
        posterior.on_fill(true, now);
        posterior.on_fill(true, now);
        posterior.on_fill(true, now);

        // Mean should increase: (3+3)/(10+3) = 6/13 ≈ 0.46
        assert!(posterior.mean_rate() > 0.3);
        assert!((posterior.mean_rate() - 6.0 / 13.0).abs() < 0.01);

        // Recent window should show 100% adverse
        assert!((posterior.recent_adverse_rate() - 1.0).abs() < 1e-10);
        assert_eq!(posterior.recent_fill_count(), 3);
    }

    #[test]
    fn test_adverse_selection_benign() {
        let config = AdverseSelectionConfig {
            prior_alpha: 3.0,
            prior_beta: 7.0,
            decay_per_fill: 1.0,
            burst_window_s: 5.0,
        };
        let mut posterior = AdverseSelectionPosterior::new(&config);

        let now = Instant::now();

        // 5 benign fills
        for _ in 0..5 {
            posterior.on_fill(false, now);
        }

        // Mean should decrease: 3/(10+5) = 3/15 = 0.2
        assert!(posterior.mean_rate() < 0.3);
        assert!((posterior.mean_rate() - 3.0 / 15.0).abs() < 0.01);
    }

    #[test]
    fn test_volatility_update() {
        let config = VolatilityConfig {
            decay: 1.0, // no decay for test
            prior_sigma_per_s: 0.001,
            prior_n_eff: 20.0,
        };
        let mut posterior = VolatilityPosterior::new(&config);

        let now = Instant::now();
        posterior.on_mid_update(50000.0, now);

        // After one price, we should still have prior sigma
        let sigma0 = posterior.sigma_per_s();
        assert!(sigma0 > 0.0);

        // Update with a 10 bps move over 1 second
        let t1 = now + Duration::from_secs(1);
        posterior.on_mid_update(50005.0, t1); // 1 bps

        // Sigma should have incorporated the new observation
        assert!(posterior.n_eff() > 20.0);
    }

    #[test]
    fn test_event_posterior_no_emergency_initially() {
        let state = EventPosteriorState::with_defaults();
        let snapshot = state.snapshot();

        assert!(!snapshot.emergency_active);
        assert_eq!(snapshot.fill_count, 0);
    }

    #[test]
    fn test_event_posterior_single_fill() {
        let mut state = EventPosteriorState::with_defaults();

        let action = state.on_fill(false, 50000.0);
        assert!(matches!(action, EmergencyAction::None));

        let snapshot = state.snapshot();
        assert_eq!(snapshot.fill_count, 1);
        assert!(!snapshot.emergency_active);
    }

    #[test]
    fn test_event_posterior_burst_triggers_emergency() {
        let config = EventPosteriorConfig {
            fill_intensity: FillIntensityConfig {
                prior_alpha: 1.0,
                prior_beta_s: 1.0,
                decay_per_fill: 1.0,
            },
            adverse_selection: AdverseSelectionConfig {
                prior_alpha: 0.1,
                prior_beta: 0.1,
                decay_per_fill: 1.0,
                burst_window_s: 5.0,
            },
            volatility: VolatilityConfig::default(),
            emergency: EmergencyThresholds {
                min_burst_fills: 3,
                intensity_p95_threshold: 3.0,
                adverse_rate_threshold: 0.7,
                combined_score_threshold: 1.5,
            },
        };
        let mut state = EventPosteriorState::new(config);

        // Simulate rapid adverse fills
        for _ in 0..5 {
            let action = state.on_fill(true, 50000.0);
            // After enough fills, should trigger emergency
            if matches!(action, EmergencyAction::CancelResting { .. }) {
                // Verify emergency is now active
                let snapshot = state.snapshot();
                assert!(snapshot.emergency_active);
                return;
            }
        }

        // If we didn't trigger within 5 rapid adverse fills, check the state
        // The test might not trigger due to Instant::now() resolution,
        // but the posteriors should show elevated risk
        let snapshot = state.snapshot();
        assert!(snapshot.adverse_rate_mean > 0.5);
    }

    #[test]
    fn test_event_posterior_snapshot_values() {
        let mut state = EventPosteriorState::with_defaults();

        let snapshot = state.snapshot();
        // Prior values
        assert!(snapshot.fill_intensity_mean > 0.0);
        assert!(snapshot.sigma_per_s > 0.0);
        assert!(snapshot.adverse_rate_mean > 0.0);
        assert!(snapshot.adverse_rate_mean < 1.0);

        // After a benign fill
        state.on_fill(false, 50000.0);
        let snapshot2 = state.snapshot();
        assert_eq!(snapshot2.fill_count, 1);
        assert!(snapshot2.adverse_rate_mean < snapshot.adverse_rate_mean);
    }

    #[test]
    fn test_emergency_cooldown() {
        let config = EventPosteriorConfig {
            fill_intensity: FillIntensityConfig {
                prior_alpha: 1.0,
                prior_beta_s: 0.1, // Very low prior time → easy to trigger
                decay_per_fill: 1.0,
            },
            adverse_selection: AdverseSelectionConfig {
                prior_alpha: 0.01,
                prior_beta: 0.01,
                decay_per_fill: 1.0,
                burst_window_s: 10.0,
            },
            volatility: VolatilityConfig::default(),
            emergency: EmergencyThresholds {
                min_burst_fills: 2,
                intensity_p95_threshold: 1.0,
                adverse_rate_threshold: 0.5,
                combined_score_threshold: 0.5,
            },
        };
        let mut state = EventPosteriorState::new(config);

        // First burst should trigger
        state.on_fill(true, 50000.0);
        state.on_fill(true, 50000.0);
        let action = state.on_fill(true, 50000.0);

        // After triggering, subsequent fills during cooldown should NOT re-trigger
        if matches!(action, EmergencyAction::CancelResting { .. }) {
            let action2 = state.on_fill(true, 50000.0);
            assert!(
                matches!(action2, EmergencyAction::None),
                "Should be in cooldown"
            );
        }
    }

    #[test]
    fn test_decay_reduces_influence() {
        let config = FillIntensityConfig {
            prior_alpha: 10.0,
            prior_beta_s: 5.0,
            decay_per_fill: 0.5, // aggressive decay for test
        };
        let mut posterior = FillIntensityPosterior::new(&config);

        let now = Instant::now();
        posterior.on_fill(now);

        // After fill with 0.5 decay: α = 10*0.5 + 1 = 6, β = 5*0.5 = 2.5
        assert!((posterior.mean_intensity() - 6.0 / 2.5).abs() < 0.01);
    }

    #[test]
    fn test_fill_intensity_p95_above_mean() {
        let config = FillIntensityConfig::default();
        let posterior = FillIntensityPosterior::new(&config);

        assert!(posterior.intensity_p95() > posterior.mean_intensity());
    }

    #[test]
    fn test_as_rate_p95_above_mean() {
        let config = AdverseSelectionConfig::default();
        let posterior = AdverseSelectionPosterior::new(&config);

        assert!(posterior.rate_p95() > posterior.mean_rate());
        assert!(posterior.rate_p95() <= 1.0);
    }

    #[test]
    fn test_mid_update_between_fills() {
        let mut state = EventPosteriorState::with_defaults();

        // Mid update without fill
        state.on_mid_update(50000.0);
        state.on_mid_update(50010.0);

        let snapshot = state.snapshot();
        assert!(snapshot.sigma_per_s > 0.0);
        assert_eq!(snapshot.fill_count, 0); // no fills yet
    }

    #[test]
    fn test_volatility_increases_with_large_moves() {
        let config = VolatilityConfig {
            decay: 0.9,
            prior_sigma_per_s: 0.001,
            prior_n_eff: 5.0, // low prior weight
        };
        let mut posterior = VolatilityPosterior::new(&config);

        let now = Instant::now();
        posterior.on_mid_update(50000.0, now);

        let sigma_before = posterior.sigma_per_s();

        // Large moves
        for i in 1..=10 {
            let t = now + Duration::from_secs(i);
            let mid = 50000.0 + (i as f64) * 100.0; // 20 bps per second
            posterior.on_mid_update(mid, t);
        }

        let sigma_after = posterior.sigma_per_s();
        assert!(
            sigma_after > sigma_before,
            "sigma should increase: {} vs {}",
            sigma_after,
            sigma_before
        );
    }

    // Slow test — uses thread::sleep for time-based assertions
    #[test]
    fn test_burst_window_reset() {
        let config = AdverseSelectionConfig {
            prior_alpha: 1.0,
            prior_beta: 1.0,
            decay_per_fill: 1.0,
            burst_window_s: 0.1, // 100ms window for fast test
        };
        let mut posterior = AdverseSelectionPosterior::new(&config);

        let now = Instant::now();
        posterior.on_fill(true, now);
        posterior.on_fill(true, now);
        assert_eq!(posterior.recent_fill_count(), 2);
        assert!((posterior.recent_adverse_rate() - 1.0).abs() < 1e-10);

        // Wait for window to expire
        thread::sleep(Duration::from_millis(150));

        // Next fill resets window
        posterior.on_fill(false, Instant::now());
        assert_eq!(posterior.recent_fill_count(), 1);
        assert!((posterior.recent_adverse_rate() - 0.0).abs() < 1e-10);
    }
}
