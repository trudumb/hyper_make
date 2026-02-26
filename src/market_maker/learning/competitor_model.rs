//! Phase 8: Competitor Modeling for Game-Theoretic Quoting
//!
//! This module implements Bayesian inference of rival market maker behavior:
//! - Competitor arrival rate estimation (λ_competitor)
//! - Snipe probability computation
//! - Queue competition modeling
//! - Spread adjustment for competitive equilibrium
//!
//! Theory: Model competitors as latent agents with arrival rates inferred
//! from market data (fills, cancels, depth changes). Use posteriors to
//! compute P(sniped) and adjust spreads for game-theoretic equilibrium.

use std::collections::VecDeque;
use tracing::debug;

// ============================================================================
// Bayesian Gamma Distribution for Arrival Rate
// ============================================================================

/// Bayesian posterior for arrival rate using Gamma-Poisson conjugacy.
///
/// Prior: λ ~ Gamma(α, β)
/// Likelihood: N ~ Poisson(λT) for N events in time T
/// Posterior: λ | N ~ Gamma(α + N, β + T)
#[derive(Debug, Clone, Copy)]
pub struct BayesianGamma {
    /// Shape parameter (α)
    alpha: f64,
    /// Rate parameter (β)
    beta: f64,
    /// Prior shape (for reset)
    prior_alpha: f64,
    /// Prior rate (for reset)
    prior_beta: f64,
}

impl BayesianGamma {
    /// Create with specified prior.
    pub fn new(prior_alpha: f64, prior_beta: f64) -> Self {
        Self {
            alpha: prior_alpha,
            beta: prior_beta,
            prior_alpha,
            prior_beta,
        }
    }

    /// Create with prior E[λ] = mean, Var[λ] = variance.
    pub fn from_mean_variance(mean: f64, variance: f64) -> Self {
        // Gamma: E[X] = α/β, Var[X] = α/β²
        // α = mean² / variance, β = mean / variance
        let alpha = mean * mean / variance;
        let beta = mean / variance;
        Self::new(alpha, beta)
    }

    /// Update posterior with observed count in time window.
    pub fn update(&mut self, count: u64, time_window_secs: f64) {
        self.alpha += count as f64;
        self.beta += time_window_secs;
    }

    /// Update in direction (increase or decrease expected rate).
    pub fn update_upward(&mut self, strength: f64) {
        // Increase rate by adding pseudo-observations
        self.alpha += strength;
    }

    /// Posterior mean E[λ].
    pub fn mean(&self) -> f64 {
        self.alpha / self.beta
    }

    /// Posterior variance Var[λ].
    pub fn variance(&self) -> f64 {
        self.alpha / (self.beta * self.beta)
    }

    /// Posterior standard deviation.
    pub fn std(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Posterior mode (most likely λ).
    pub fn mode(&self) -> f64 {
        if self.alpha >= 1.0 {
            (self.alpha - 1.0) / self.beta
        } else {
            0.0
        }
    }

    /// Reset to prior.
    pub fn reset(&mut self) {
        self.alpha = self.prior_alpha;
        self.beta = self.prior_beta;
    }

    /// Sample from posterior (for Thompson sampling).
    pub fn sample(&self) -> f64 {
        sample_gamma(self.alpha, self.beta)
    }

    /// Get 95% credible interval.
    pub fn credible_interval_95(&self) -> (f64, f64) {
        // Approximate using normal approximation for large α
        let mean = self.mean();
        let std = self.std();
        (mean - 1.96 * std, mean + 1.96 * std)
    }

    /// Effective observations (pseudo-count).
    pub fn effective_observations(&self) -> f64 {
        self.alpha
    }
}

impl Default for BayesianGamma {
    fn default() -> Self {
        // Prior: E[λ] = 10 events/sec, moderate uncertainty
        Self::from_mean_variance(10.0, 25.0)
    }
}

// ============================================================================
// Snipe Tracking
// ============================================================================

/// Tracker for snipe events (getting picked off).
#[derive(Debug, Clone)]
pub struct SnipeTracker {
    /// Recent snipe events (price move in bps)
    recent_snipes: VecDeque<SnipeEvent>,
    /// Maximum events to track
    max_events: usize,
    /// Total snipes observed
    total_snipes: u64,
    /// Total fills observed
    total_fills: u64,
    /// EWMA of snipe rate
    ewma_snipe_rate: f64,
    /// EWMA of snipe loss
    ewma_snipe_loss: f64,
    /// EWMA decay
    ewma_alpha: f64,
}

/// A single snipe event.
#[derive(Debug, Clone, Copy)]
pub struct SnipeEvent {
    /// Timestamp in milliseconds
    pub timestamp_ms: u64,
    /// Price move against us (bps)
    pub price_move_bps: f64,
    /// Time from quote to snipe (ms)
    pub time_to_snipe_ms: u64,
    /// Our latency when sniped (ms)
    pub our_latency_ms: u64,
}

impl SnipeTracker {
    /// Create a new tracker.
    pub fn new() -> Self {
        Self {
            recent_snipes: VecDeque::with_capacity(100),
            max_events: 100,
            total_snipes: 0,
            total_fills: 0,
            ewma_snipe_rate: 0.1, // Start with 10% baseline
            ewma_snipe_loss: 5.0, // Start with 5 bps baseline
            ewma_alpha: 0.1,
        }
    }

    /// Record a snipe event.
    pub fn record_snipe(&mut self, event: SnipeEvent) {
        self.total_snipes += 1;

        // Update EWMA
        self.ewma_snipe_rate =
            self.ewma_alpha * 1.0 + (1.0 - self.ewma_alpha) * self.ewma_snipe_rate;
        self.ewma_snipe_loss =
            self.ewma_alpha * event.price_move_bps + (1.0 - self.ewma_alpha) * self.ewma_snipe_loss;

        // Store event
        self.recent_snipes.push_back(event);
        if self.recent_snipes.len() > self.max_events {
            self.recent_snipes.pop_front();
        }

        debug!(
            price_move_bps = %format!("{:.2}", event.price_move_bps),
            time_to_snipe_ms = event.time_to_snipe_ms,
            total_snipes = self.total_snipes,
            ewma_rate = %format!("{:.3}", self.ewma_snipe_rate),
            "Snipe event recorded"
        );
    }

    /// Record a non-snipe fill.
    pub fn record_fill(&mut self) {
        self.total_fills += 1;
        // Decay snipe rate toward zero
        self.ewma_snipe_rate =
            self.ewma_alpha * 0.0 + (1.0 - self.ewma_alpha) * self.ewma_snipe_rate;
    }

    /// Get current snipe rate estimate.
    pub fn snipe_rate(&self) -> f64 {
        self.ewma_snipe_rate
    }

    /// Get expected snipe loss in bps.
    pub fn expected_snipe_loss(&self) -> f64 {
        self.ewma_snipe_loss
    }

    /// Get total snipe count.
    pub fn total_snipes(&self) -> u64 {
        self.total_snipes
    }

    /// Get historical snipe rate.
    pub fn historical_snipe_rate(&self) -> f64 {
        if self.total_fills == 0 {
            return self.ewma_snipe_rate;
        }
        self.total_snipes as f64 / (self.total_snipes + self.total_fills) as f64
    }

    /// Get average snipe latency (ms).
    pub fn avg_snipe_latency_ms(&self) -> f64 {
        if self.recent_snipes.is_empty() {
            return 100.0; // Default assumption
        }
        self.recent_snipes
            .iter()
            .map(|e| e.time_to_snipe_ms as f64)
            .sum::<f64>()
            / self.recent_snipes.len() as f64
    }
}

impl Default for SnipeTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Queue Competition Model
// ============================================================================

/// Model for queue competition dynamics.
#[derive(Debug, Clone)]
pub struct QueueCompetitionModel {
    /// Estimated competitor queue refresh rate (orders/sec)
    competitor_refresh_rate: BayesianGamma,
    /// Average queue position when filled
    avg_queue_position: f64,
    /// EWMA decay
    ewma_alpha: f64,
    /// Queue position observations
    queue_observations: u64,
}

impl QueueCompetitionModel {
    /// Create a new model.
    pub fn new() -> Self {
        Self {
            // Prior: competitors refresh ~5 orders/sec
            competitor_refresh_rate: BayesianGamma::from_mean_variance(5.0, 10.0),
            avg_queue_position: 0.5, // Start at middle
            ewma_alpha: 0.1,
            queue_observations: 0,
        }
    }

    /// Update with observed fill at queue position.
    pub fn observe_fill(&mut self, our_queue_position: f64, time_in_queue_ms: u64) {
        self.queue_observations += 1;

        // Update average queue position
        self.avg_queue_position = self.ewma_alpha * our_queue_position
            + (1.0 - self.ewma_alpha) * self.avg_queue_position;

        // Infer competitor arrival rate from queue position
        // If we were near front (position ~0), fewer competitors ahead
        // If we were near back (position ~1), more competitors ahead
        let implied_arrivals = our_queue_position * 10.0; // Rough estimate
        let time_window_secs = time_in_queue_ms as f64 / 1000.0;

        if time_window_secs > 0.01 {
            let implied_rate = implied_arrivals / time_window_secs;
            // Soft update toward implied rate
            self.competitor_refresh_rate
                .update_upward((implied_rate - self.competitor_refresh_rate.mean()).signum() * 0.1);
        }
    }

    /// Probability we'll be ahead in queue given our latency.
    pub fn p_ahead_in_queue(&self, our_latency_ms: f64) -> f64 {
        let lambda = self.competitor_refresh_rate.mean();
        let our_delay_secs = our_latency_ms / 1000.0;

        // Probability no competitor arrived before us
        // P(ahead) ≈ exp(-λ × our_delay)
        (-lambda * our_delay_secs).exp()
    }

    /// Expected queue position given latency.
    pub fn expected_queue_position(&self, our_latency_ms: f64) -> f64 {
        let p_ahead = self.p_ahead_in_queue(our_latency_ms);
        // Rough: if P(ahead) = 1, position = 0; if P(ahead) = 0, position = 1
        1.0 - p_ahead
    }

    /// Get competitor refresh rate estimate.
    pub fn competitor_rate(&self) -> f64 {
        self.competitor_refresh_rate.mean()
    }
}

impl Default for QueueCompetitionModel {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Complete Competitor Model
// ============================================================================

/// Configuration for competitor modeling.
#[derive(Debug, Clone)]
pub struct CompetitorModelConfig {
    /// Prior mean for competitor arrival rate
    pub prior_lambda_mean: f64,
    /// Prior variance for competitor arrival rate
    pub prior_lambda_variance: f64,
    /// Our typical latency (ms)
    pub our_latency_ms: f64,
    /// Snipe sensitivity: how much depth affects snipe probability
    pub snipe_sensitivity: f64,
    /// Large order threshold for detecting competitor activity
    pub large_order_threshold: f64,
    /// Maximum spread adjustment from competition (bps)
    pub max_spread_adjustment_bps: f64,
}

impl Default for CompetitorModelConfig {
    fn default() -> Self {
        Self {
            prior_lambda_mean: 10.0, // 10 competitor orders/sec baseline
            prior_lambda_variance: 25.0,
            our_latency_ms: 50.0, // 50ms typical latency
            snipe_sensitivity: 0.5,
            large_order_threshold: 10.0, // 10 contracts = large order
            max_spread_adjustment_bps: 5.0,
        }
    }
}

/// Market event for competitor inference.
#[derive(Debug, Clone)]
pub enum MarketEvent {
    /// Our order was filled at a queue position
    OurFill {
        queue_position: f64,
        time_in_queue_ms: u64,
    },
    /// Our order was cancelled due to adverse move (sniped)
    OurCancelledBySweep {
        price_move_bps: f64,
        time_to_sweep_ms: u64,
    },
    /// Depth change observed
    DepthChange { side: Side, delta: f64 },
    /// Competitor order detected (large order at edge)
    CompetitorOrder { size: f64, is_aggressive: bool },
}

/// Order side.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Bid,
    Ask,
}

/// Complete competitor model with all inference components.
#[derive(Debug, Clone)]
pub struct CompetitorModel {
    /// Configuration
    config: CompetitorModelConfig,
    /// Estimated total MM arrival rate (orders per second, market-wide)
    lambda_competitor: BayesianGamma,
    /// Estimated number of active MMs
    n_competitors_estimate: f64,
    /// Snipe tracking
    snipe_tracker: SnipeTracker,
    /// Queue competition model
    queue_model: QueueCompetitionModel,
    /// Recent depth changes for activity inference
    recent_depth_changes: VecDeque<f64>,
    /// Max depth changes to track
    max_depth_changes: usize,
}

impl CompetitorModel {
    /// Create a new competitor model.
    pub fn new(config: CompetitorModelConfig) -> Self {
        Self {
            lambda_competitor: BayesianGamma::from_mean_variance(
                config.prior_lambda_mean,
                config.prior_lambda_variance,
            ),
            n_competitors_estimate: 3.0, // Start with assumption of 3 active MMs
            snipe_tracker: SnipeTracker::new(),
            queue_model: QueueCompetitionModel::new(),
            recent_depth_changes: VecDeque::with_capacity(50),
            max_depth_changes: 50,
            config,
        }
    }

    /// Update model with observed market event.
    pub fn observe(&mut self, event: &MarketEvent) {
        match event {
            MarketEvent::OurFill {
                queue_position,
                time_in_queue_ms,
            } => {
                // Fill at good queue position = fewer competitors ahead
                self.queue_model
                    .observe_fill(*queue_position, *time_in_queue_ms);
                self.snipe_tracker.record_fill();

                // Infer competitor rate from fill success
                let implied_lambda =
                    self.infer_lambda_from_fill(*queue_position, *time_in_queue_ms);
                // Soft update toward implied rate
                if implied_lambda > 0.0 {
                    let diff = implied_lambda - self.lambda_competitor.mean();
                    self.lambda_competitor.update_upward(diff.signum() * 0.1);
                }
            }
            MarketEvent::OurCancelledBySweep {
                price_move_bps,
                time_to_sweep_ms,
            } => {
                // Got sniped = competitor was faster
                self.snipe_tracker.record_snipe(SnipeEvent {
                    timestamp_ms: now_ms(),
                    price_move_bps: *price_move_bps,
                    time_to_snipe_ms: *time_to_sweep_ms,
                    our_latency_ms: self.config.our_latency_ms as u64,
                });
                // Competitors are active, increase rate estimate
                self.lambda_competitor.update_upward(0.5);
            }
            MarketEvent::DepthChange { side: _, delta } => {
                // Large depth changes = competitor activity
                self.recent_depth_changes.push_back(delta.abs());
                if self.recent_depth_changes.len() > self.max_depth_changes {
                    self.recent_depth_changes.pop_front();
                }

                if delta.abs() > self.config.large_order_threshold {
                    // Large order = competitor activity
                    self.n_competitors_estimate = self.estimate_n_competitors(*delta);
                }
            }
            MarketEvent::CompetitorOrder {
                size,
                is_aggressive,
            } => {
                // Direct competitor detection
                if *is_aggressive && *size > self.config.large_order_threshold {
                    self.lambda_competitor.update_upward(0.3);
                    self.n_competitors_estimate = (self.n_competitors_estimate + 0.1).min(10.0);
                }
            }
        }
    }

    /// Infer competitor arrival rate from fill success.
    fn infer_lambda_from_fill(&self, queue_position: f64, time_in_queue_ms: u64) -> f64 {
        // If we filled near front of queue quickly, fewer competitors
        // If we filled near back after long wait, more competitors
        let time_secs = time_in_queue_ms as f64 / 1000.0;
        if time_secs < 0.01 {
            return self.lambda_competitor.mean();
        }

        // Rough inference: queue_position indicates how many others were ahead
        // λ ≈ (queue_position × estimated_queue_size) / time
        let estimated_queue_size = 10.0; // Assume ~10 orders in queue
        queue_position * estimated_queue_size / time_secs
    }

    /// Estimate number of competitors from depth change.
    fn estimate_n_competitors(&self, delta: f64) -> f64 {
        // Heuristic: large delta implies active competitor
        // More competitors = more frequent large deltas
        let avg_delta = if self.recent_depth_changes.is_empty() {
            1.0
        } else {
            self.recent_depth_changes.iter().sum::<f64>() / self.recent_depth_changes.len() as f64
        };

        // Scale: if this delta is 2x average, estimate +1 competitor
        let delta_ratio = delta.abs() / avg_delta.max(0.1);
        (self.n_competitors_estimate + (delta_ratio - 1.0) * 0.5).clamp(1.0, 10.0)
    }

    /// Probability we'll be sniped if we quote at this depth.
    pub fn p_snipe(&self, depth_bps: f64, hold_time_ms: f64) -> f64 {
        // P(price moves against us faster than we can cancel)
        let lambda = self.lambda_competitor.mean();
        let sensitivity = self.config.snipe_sensitivity * (10.0 / depth_bps.max(1.0)); // Tighter = more exposed
        let lambda_snipe = lambda * sensitivity;
        let hold_time_secs = hold_time_ms / 1000.0;

        // P(snipe) = 1 - exp(-λ_snipe × hold_time)
        1.0 - (-lambda_snipe * hold_time_secs).exp()
    }

    /// Spread adjustment to compensate for snipe risk.
    pub fn snipe_adjusted_spread(&self, base_spread_bps: f64) -> f64 {
        let p_snipe = self.p_snipe(base_spread_bps, self.config.our_latency_ms);
        let snipe_cost_bps = p_snipe * self.snipe_tracker.expected_snipe_loss();
        (base_spread_bps + snipe_cost_bps)
            .min(base_spread_bps + self.config.max_spread_adjustment_bps)
    }

    /// Get spread widening factor for competition.
    pub fn competition_spread_factor(&self) -> f64 {
        // More competitors = wider spreads needed
        // Baseline at 3 competitors = 1.0
        let n_factor = self.n_competitors_estimate / 3.0;

        // Higher snipe rate = wider spreads
        let snipe_factor = 1.0 + self.snipe_tracker.snipe_rate();

        // Combined factor (capped)
        (n_factor * snipe_factor).clamp(0.8, 1.5)
    }

    /// Get summary of competitor state.
    pub fn summary(&self) -> CompetitorSummary {
        CompetitorSummary {
            lambda_competitor: self.lambda_competitor.mean(),
            lambda_std: self.lambda_competitor.std(),
            n_competitors: self.n_competitors_estimate,
            snipe_rate: self.snipe_tracker.snipe_rate(),
            expected_snipe_loss_bps: self.snipe_tracker.expected_snipe_loss(),
            queue_ahead_prob: self
                .queue_model
                .p_ahead_in_queue(self.config.our_latency_ms),
            competition_spread_factor: self.competition_spread_factor(),
        }
    }

    /// Get the Bayesian lambda posterior for Thompson sampling.
    pub fn lambda_posterior(&self) -> &BayesianGamma {
        &self.lambda_competitor
    }

    /// Sample competitor rate for Thompson sampling.
    pub fn sample_lambda(&self) -> f64 {
        self.lambda_competitor.sample()
    }
}

impl Default for CompetitorModel {
    fn default() -> Self {
        Self::new(CompetitorModelConfig::default())
    }
}

/// Summary of competitor model state.
#[derive(Debug, Clone, Copy)]
pub struct CompetitorSummary {
    /// Posterior mean of competitor arrival rate
    pub lambda_competitor: f64,
    /// Posterior std of competitor rate
    pub lambda_std: f64,
    /// Estimated number of active competitors
    pub n_competitors: f64,
    /// Current snipe rate estimate
    pub snipe_rate: f64,
    /// Expected loss from snipe (bps)
    pub expected_snipe_loss_bps: f64,
    /// Probability of being ahead in queue
    pub queue_ahead_prob: f64,
    /// Spread widening factor for competition
    pub competition_spread_factor: f64,
}

// ============================================================================
// Game-Theoretic Equilibrium
// ============================================================================

/// Compute Nash equilibrium spread adjustment.
///
/// In a symmetric MM game, the equilibrium spread depends on:
/// - Number of competitors n
/// - Competitor arrival rates λ
/// - Adverse selection probability
///
/// Returns spread multiplier for equilibrium.
pub fn compute_equilibrium_spread_factor(
    n_competitors: f64,
    competitor_lambda: f64,
    our_lambda: f64,
    adverse_prob: f64,
) -> f64 {
    // Simplified equilibrium: spread ∝ (1 + adverse_prob) × (1 + competition_intensity)
    // where competition_intensity = n × competitor_lambda / (our_lambda + n × competitor_lambda)

    let total_lambda = our_lambda + n_competitors * competitor_lambda;
    let competition_intensity = if total_lambda > 0.0 {
        (n_competitors * competitor_lambda) / total_lambda
    } else {
        n_competitors / (1.0 + n_competitors)
    };

    // Adverse selection increases spread linearly
    let adverse_factor = 1.0 + adverse_prob * 0.5; // 0.5 sensitivity

    // Competition intensity increases spread
    let competition_factor = 1.0 + competition_intensity * 0.3; // 0.3 sensitivity

    // Combine multiplicatively (no harsh clamp to preserve gradients)
    (adverse_factor * competition_factor).clamp(0.9, 1.8)
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Get current timestamp in milliseconds.
fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Sample from Gamma distribution.
fn sample_gamma(alpha: f64, beta: f64) -> f64 {
    if alpha < 1.0 {
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

/// Sample from standard normal.
fn sample_standard_normal() -> f64 {
    let u1 = sample_uniform();
    let u2 = sample_uniform();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Sample from uniform [0, 1).
fn sample_uniform() -> f64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    static mut SEED: u64 = 0;
    unsafe {
        if SEED == 0 {
            SEED = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(12345);
        }
        SEED = SEED
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
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
    fn test_bayesian_gamma_update() {
        let mut gamma = BayesianGamma::from_mean_variance(10.0, 25.0);
        let initial_mean = gamma.mean();

        // Observe 20 events in 1 second (high rate)
        gamma.update(20, 1.0);

        // Mean should increase toward 20
        assert!(gamma.mean() > initial_mean);
    }

    #[test]
    fn test_bayesian_gamma_variance_decreases() {
        let mut gamma = BayesianGamma::from_mean_variance(10.0, 25.0);
        let initial_var = gamma.variance();

        // More observations = less uncertainty
        for _ in 0..10 {
            gamma.update(10, 1.0);
        }

        assert!(gamma.variance() < initial_var);
    }

    #[test]
    fn test_snipe_tracker() {
        let mut tracker = SnipeTracker::new();

        // Record some fills (non-snipes)
        for _ in 0..5 {
            tracker.record_fill();
        }

        // Snipe rate should be low
        assert!(tracker.snipe_rate() < 0.5);

        // Record a snipe
        tracker.record_snipe(SnipeEvent {
            timestamp_ms: now_ms(),
            price_move_bps: 10.0,
            time_to_snipe_ms: 50,
            our_latency_ms: 50,
        });

        // Snipe rate should increase
        assert!(tracker.snipe_rate() > 0.0);
        assert!(tracker.total_snipes() == 1);
    }

    #[test]
    fn test_queue_competition_model() {
        let mut model = QueueCompetitionModel::new();

        // Fill at front of queue quickly
        model.observe_fill(0.1, 100); // 10% back, 100ms

        // Should indicate fewer competitors
        let p_ahead = model.p_ahead_in_queue(50.0);
        assert!(p_ahead > 0.5); // Should have decent chance of being ahead
    }

    #[test]
    fn test_competitor_model_fill_event() {
        let mut model = CompetitorModel::default();

        model.observe(&MarketEvent::OurFill {
            queue_position: 0.2,
            time_in_queue_ms: 200,
        });

        let summary = model.summary();
        assert!(summary.lambda_competitor > 0.0);
        assert!(summary.queue_ahead_prob > 0.0);
    }

    #[test]
    fn test_competitor_model_snipe_event() {
        let mut model = CompetitorModel::default();
        let initial_lambda = model.lambda_competitor.mean();

        model.observe(&MarketEvent::OurCancelledBySweep {
            price_move_bps: 15.0,
            time_to_sweep_ms: 30,
        });

        // Lambda should increase after snipe
        assert!(model.lambda_competitor.mean() > initial_lambda);
        assert!(model.snipe_tracker.total_snipes() == 1);
    }

    #[test]
    fn test_p_snipe_increases_with_tighter_spread() {
        let model = CompetitorModel::default();

        let p_snipe_tight = model.p_snipe(2.0, 100.0); // 2 bps spread
        let p_snipe_wide = model.p_snipe(10.0, 100.0); // 10 bps spread

        // Tighter spread = higher snipe risk
        assert!(p_snipe_tight > p_snipe_wide);
    }

    #[test]
    fn test_snipe_adjusted_spread() {
        let model = CompetitorModel::default();

        let base_spread = 5.0;
        let adjusted = model.snipe_adjusted_spread(base_spread);

        // Adjusted should be >= base
        assert!(adjusted >= base_spread);
    }

    #[test]
    fn test_competition_spread_factor() {
        let mut model = CompetitorModel::default();

        // After many snipes, factor should increase
        for _ in 0..5 {
            model.observe(&MarketEvent::OurCancelledBySweep {
                price_move_bps: 10.0,
                time_to_sweep_ms: 50,
            });
        }

        assert!(model.competition_spread_factor() > 1.0);
    }

    #[test]
    fn test_equilibrium_spread_factor() {
        // More competitors = wider spreads
        let factor_few = compute_equilibrium_spread_factor(2.0, 10.0, 10.0, 0.1);
        let factor_many = compute_equilibrium_spread_factor(5.0, 10.0, 10.0, 0.1);

        assert!(factor_many > factor_few);
    }

    #[test]
    fn test_equilibrium_spread_adverse() {
        // Higher adverse selection = wider spreads
        let factor_low = compute_equilibrium_spread_factor(3.0, 10.0, 10.0, 0.1);
        let factor_high = compute_equilibrium_spread_factor(3.0, 10.0, 10.0, 0.4);

        assert!(factor_high > factor_low);
    }
}
