//! Funding Rate Estimator - tracks and predicts perpetual funding costs.
//!
//! On perpetual exchanges like Hyperliquid, funding rates represent the cost
//! of holding a position. Long positions pay short positions when funding is
//! positive, and vice versa.
//!
//! # Key Concepts
//! - **Funding Rate**: Periodic payment (usually every 8h or 1h) between longs/shorts
//! - **Mark-Index Spread**: Premium/discount of perpetual vs spot price
//! - **Carry Cost**: Expected funding cost for holding a position over time
//!
//! # Mean Reversion Model
//! Funding rates tend to mean-revert to a long-run average:
//! ```text
//! dF = κ(θ - F)dt + σ_f × dW_f
//! ```
//! where:
//! - F: current funding rate
//! - κ: mean-reversion speed
//! - θ: long-run average funding rate
//! - σ_f: volatility of funding rate changes

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Configuration for the funding rate estimator.
#[derive(Debug, Clone)]
pub struct FundingConfig {
    /// Funding interval (Hyperliquid uses 1 hour)
    pub funding_interval: Duration,
    /// Half-life for EWMA of funding rate history (in funding periods)
    /// Default: 24 (one day of history at 1h intervals)
    pub ewma_half_life_periods: f64,
    /// Mean-reversion speed (κ) for predictions
    /// Default: 0.1 (slow mean reversion)
    pub mean_reversion_speed: f64,
    /// Long-run average funding rate (θ) - typically near zero
    /// Default: 0.0
    pub long_run_rate: f64,
    /// Maximum funding rate to consider (for sanity check)
    /// Default: 0.01 (1% per period)
    pub max_rate: f64,
    /// Minimum observations before warmed up
    pub min_observations: usize,
}

impl Default for FundingConfig {
    fn default() -> Self {
        Self {
            funding_interval: Duration::from_secs(3600), // 1 hour
            ewma_half_life_periods: 24.0,                // 1 day of history
            mean_reversion_speed: 0.1,
            long_run_rate: 0.0,
            max_rate: 0.01,
            min_observations: 3,
        }
    }
}

/// A single funding rate observation.
#[derive(Debug, Clone, Copy)]
pub struct FundingObservation {
    /// The funding rate (as a fraction, e.g., 0.0001 = 0.01%)
    pub rate: f64,
    /// When this rate was observed
    pub timestamp: Instant,
    /// The funding period this applies to
    pub period: u64,
}

/// Funding rate estimator for perpetual contracts.
pub struct FundingRateEstimator {
    config: FundingConfig,

    /// Current funding rate (most recent observation)
    current_rate: f64,

    /// Historical funding rates for EWMA calculation
    rate_history: VecDeque<FundingObservation>,

    /// EWMA of funding rate
    ewma_rate: f64,

    /// EWMA decay factor (λ = exp(-ln(2) / half_life))
    ewma_lambda: f64,

    /// Time of last funding update
    last_update: Instant,

    /// Next expected funding time
    next_funding_time: Option<Instant>,

    /// Total observations received
    observation_count: usize,

    /// Start time for warmup tracking
    #[allow(dead_code)]
    start_time: Instant,

    // === Premium Tracking (First Principles Gap 6) ===
    /// Current mark-index premium (as a fraction)
    current_premium: f64,
    /// EWMA of premium
    ewma_premium: f64,
    /// Premium history for analysis
    premium_history: VecDeque<(Instant, f64)>,

    // === Basis Velocity Tracking ===
    /// Previous basis (mark-index premium) for velocity computation
    prev_basis: f64,
    /// Instant of previous basis snapshot
    prev_basis_time: Option<Instant>,
    /// Instant of current basis snapshot (when current_premium was last set via update_from_prices)
    current_basis_time: Option<Instant>,
}

impl FundingRateEstimator {
    /// Create a new funding rate estimator.
    pub fn new(config: FundingConfig) -> Self {
        let ewma_lambda = (-2.0_f64.ln() / config.ewma_half_life_periods).exp();

        Self {
            config,
            current_rate: 0.0,
            rate_history: VecDeque::with_capacity(100),
            ewma_rate: 0.0,
            ewma_lambda,
            last_update: Instant::now(),
            next_funding_time: None,
            observation_count: 0,
            start_time: Instant::now(),
            // Premium tracking
            current_premium: 0.0,
            ewma_premium: 0.0,
            premium_history: VecDeque::with_capacity(500),
            // Basis velocity tracking
            prev_basis: 0.0,
            prev_basis_time: None,
            current_basis_time: None,
        }
    }

    /// Record a new funding rate observation.
    ///
    /// # Arguments
    /// - `rate`: The funding rate as a fraction (e.g., 0.0001 for 0.01%)
    /// - `period`: The funding period number (for deduplication)
    pub fn record_funding(&mut self, rate: f64, period: u64) {
        let now = Instant::now();

        // Clamp rate to reasonable bounds
        let clamped_rate = rate.clamp(-self.config.max_rate, self.config.max_rate);

        // Check for duplicate period
        if let Some(last) = self.rate_history.back() {
            if last.period == period {
                return; // Already recorded this period
            }
        }

        // Record observation
        let obs = FundingObservation {
            rate: clamped_rate,
            timestamp: now,
            period,
        };
        self.rate_history.push_back(obs);

        // Trim old history (keep ~1 week of hourly data)
        while self.rate_history.len() > 168 {
            self.rate_history.pop_front();
        }

        // Update EWMA
        if self.observation_count == 0 {
            self.ewma_rate = clamped_rate;
        } else {
            self.ewma_rate =
                self.ewma_lambda * self.ewma_rate + (1.0 - self.ewma_lambda) * clamped_rate;
        }

        self.current_rate = clamped_rate;
        self.observation_count += 1;
        self.last_update = now;

        // Estimate next funding time
        self.next_funding_time = Some(now + self.config.funding_interval);
    }

    /// Update from mark-index spread (alternative to direct funding rate).
    ///
    /// Some systems estimate funding from the mark-index premium.
    /// Typical formula: funding ≈ premium / 8 (for 8h funding)
    pub fn update_from_premium(&mut self, premium: f64, period: u64) {
        // Convert premium to hourly funding rate
        // Assuming premium is annualized, convert to hourly
        let hourly_rate = premium / (365.25 * 24.0);
        self.record_funding(hourly_rate, period);
    }

    /// Check if the estimator has enough data.
    pub fn is_warmed_up(&self) -> bool {
        self.observation_count >= self.config.min_observations
    }

    /// Get the current funding rate.
    pub fn current_rate(&self) -> f64 {
        self.current_rate
    }

    /// Get the EWMA (smoothed) funding rate.
    pub fn ewma_rate(&self) -> f64 {
        self.ewma_rate
    }

    /// Predict the funding rate at a future time using mean-reversion.
    ///
    /// Uses Ornstein-Uhlenbeck mean-reversion:
    /// E[F(t+T)] = F(t) × exp(-κT) + θ × (1 - exp(-κT))
    ///
    /// # Arguments
    /// - `horizon_secs`: Time horizon in seconds
    pub fn predicted_rate(&self, horizon_secs: f64) -> f64 {
        let kappa = self.config.mean_reversion_speed;
        let theta = self.config.long_run_rate;

        // Convert horizon to funding periods
        let t = horizon_secs / self.config.funding_interval.as_secs_f64();

        // Mean-reversion prediction
        let decay = (-kappa * t).exp();
        self.current_rate * decay + theta * (1.0 - decay)
    }

    /// Calculate the expected funding cost for a position over a holding period.
    ///
    /// Positive return = cost (long pays), Negative return = revenue (short receives)
    ///
    /// # Arguments
    /// - `position`: Position size (positive = long, negative = short)
    /// - `mid_price`: Current mid price
    /// - `holding_secs`: Expected holding time in seconds
    ///
    /// # Returns
    /// Expected funding cost in USD (positive = cost to position holder)
    pub fn funding_cost(&self, position: f64, mid_price: f64, holding_secs: f64) -> f64 {
        let notional = position * mid_price;

        // Number of funding periods in holding time
        let periods = holding_secs / self.config.funding_interval.as_secs_f64();

        // Use EWMA rate as best estimate for average rate over holding period
        // More sophisticated: integrate predicted_rate over horizon
        let avg_rate = if self.is_warmed_up() {
            self.ewma_rate
        } else {
            self.current_rate
        };

        // Funding cost = notional × rate × periods
        // Positive rate + long position = cost
        // Negative rate + long position = revenue
        notional * avg_rate * periods
    }

    /// Get drift adjustment for price process.
    ///
    /// For perpetuals, the fair price drifts by -funding_rate:
    /// dp = (μ - funding_rate) dt + σ dW
    ///
    /// This returns the funding-induced drift adjustment.
    pub fn drift_adjustment(&self) -> f64 {
        // Annualize the hourly rate for drift calculation
        let hours_per_year = 365.25 * 24.0;
        -self.current_rate * hours_per_year
    }

    /// Get time until next funding.
    pub fn time_to_next_funding(&self) -> Option<Duration> {
        self.next_funding_time.map(|t| {
            let now = Instant::now();
            if t > now {
                t - now
            } else {
                Duration::ZERO
            }
        })
    }

    /// Check if funding is extreme (high absolute value).
    ///
    /// Extreme funding creates carry opportunities and risks.
    pub fn is_funding_extreme(&self) -> bool {
        // Consider extreme if > 0.1% per period (roughly 8.76% annualized)
        self.current_rate.abs() > 0.001
    }

    /// Get funding direction bias.
    ///
    /// Returns:
    /// - Positive: Longs paying (bearish pressure)
    /// - Negative: Shorts paying (bullish pressure)
    /// - Near zero: Neutral
    pub fn funding_bias(&self) -> f64 {
        self.ewma_rate
    }

    /// Get summary statistics.
    pub fn summary(&self) -> FundingSummary {
        let annualized_rate = self.current_rate * 365.25 * 24.0;

        FundingSummary {
            is_warmed_up: self.is_warmed_up(),
            current_rate: self.current_rate,
            ewma_rate: self.ewma_rate,
            annualized_rate,
            observation_count: self.observation_count,
            is_extreme: self.is_funding_extreme(),
            time_to_next_funding: self.time_to_next_funding(),
        }
    }

    /// Calculate realized funding over a period.
    ///
    /// Sums actual funding payments from history.
    pub fn realized_funding(&self, lookback: Duration) -> f64 {
        let cutoff = Instant::now() - lookback;

        self.rate_history
            .iter()
            .filter(|obs| obs.timestamp > cutoff)
            .map(|obs| obs.rate)
            .sum()
    }

    /// Get the volatility of funding rate changes.
    pub fn funding_volatility(&self) -> f64 {
        if self.rate_history.len() < 2 {
            return 0.0;
        }

        let rates: Vec<f64> = self.rate_history.iter().map(|obs| obs.rate).collect();
        let mean = rates.iter().sum::<f64>() / rates.len() as f64;

        let variance = rates.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / rates.len() as f64;

        variance.sqrt()
    }

    // === Premium Tracking Methods (First Principles Gap 6) ===

    /// Update premium from mark and index prices.
    ///
    /// Premium = (mark - index) / index
    ///
    /// A positive premium indicates perpetual trading above spot (bullish bias),
    /// which typically leads to positive funding (longs pay shorts).
    pub fn update_from_prices(&mut self, mark: f64, index: f64) {
        if index <= 0.0 {
            return;
        }

        let now = Instant::now();
        let premium = (mark - index) / index;

        // Update basis velocity tracking: shift current → prev, then set new current
        self.prev_basis = self.current_premium;
        self.prev_basis_time = self.current_basis_time;
        self.current_basis_time = Some(now);

        self.current_premium = premium;

        // Update EWMA
        if self.premium_history.is_empty() {
            self.ewma_premium = premium;
        } else {
            self.ewma_premium =
                self.ewma_lambda * self.ewma_premium + (1.0 - self.ewma_lambda) * premium;
        }

        // Store history
        self.premium_history.push_back((now, premium));

        // Trim old entries (keep ~1 hour at 1-second intervals)
        while self.premium_history.len() > 3600 {
            self.premium_history.pop_front();
        }
    }

    /// Get current premium (mark vs index).
    pub fn current_premium(&self) -> f64 {
        self.current_premium
    }

    /// Get EWMA smoothed premium.
    pub fn ewma_premium(&self) -> f64 {
        self.ewma_premium
    }

    /// Detect if there's an arbitrage opportunity.
    ///
    /// Returns Some(ArbitrageSignal) if premium is extreme enough to warrant action.
    pub fn arbitrage_opportunity(&self) -> Option<ArbitrageSignal> {
        // Threshold for considering arbitrage (50 bps premium)
        let arb_threshold = 0.005;

        if self.current_premium > arb_threshold {
            // Perp trading above spot: go short perp, long spot
            Some(ArbitrageSignal {
                direction: ArbitrageDirection::ShortPerp,
                premium: self.current_premium,
                expected_convergence: self.ewma_premium,
            })
        } else if self.current_premium < -arb_threshold {
            // Perp trading below spot: go long perp, short spot
            Some(ArbitrageSignal {
                direction: ArbitrageDirection::LongPerp,
                premium: self.current_premium,
                expected_convergence: self.ewma_premium,
            })
        } else {
            None
        }
    }

    /// Get alpha signal from premium.
    ///
    /// The premium often predicts short-term price direction:
    /// - High premium → expect funding to push prices down
    /// - Low premium → expect funding to push prices up
    ///
    /// Returns adjustment to apply to microprice (in same units as premium).
    pub fn premium_alpha(&self) -> f64 {
        // Premium mean-reverts, so current premium predicts negative return
        // Scale by a factor based on typical premium-to-return relationship
        let alpha_scale = 0.1; // 10% of premium predicts return
        -self.current_premium * alpha_scale
    }

    /// Check if premium is elevated (indicating market stress).
    pub fn is_premium_elevated(&self) -> bool {
        self.current_premium.abs() > 0.002 // 20 bps
    }

    /// Get premium direction (-1 to 1).
    pub fn premium_direction(&self) -> f64 {
        // Normalize to -1 to 1 range
        (self.current_premium / 0.01).clamp(-1.0, 1.0)
    }

    /// Basis velocity: rate of change of mark-index premium, scaled by settlement proximity.
    ///
    /// Captures whether the basis is converging or diverging as funding settlement
    /// approaches. Fast-moving basis near settlement is a stronger signal.
    ///
    /// Returns a value clamped to [-1, 1].
    pub fn basis_velocity(&self) -> f64 {
        let (prev_time, cur_time) = match (self.prev_basis_time, self.current_basis_time) {
            (Some(pt), Some(ct)) => (pt, ct),
            _ => return 0.0, // Need at least 2 price updates
        };

        let elapsed_s = cur_time.duration_since(prev_time).as_secs_f64();
        if elapsed_s < 0.01 {
            return 0.0; // Avoid division by near-zero
        }

        // Raw velocity: change in basis per second
        let raw_velocity = (self.current_premium - self.prev_basis) / elapsed_s;

        // Scale by settlement proximity: velocity matters more near settlement
        // settlement_fraction = time remaining / 8h period, in [0, 1]
        let funding_interval_s = self.config.funding_interval.as_secs_f64();
        let settlement_fraction = self
            .time_to_next_funding()
            .map(|d| d.as_secs_f64() / funding_interval_s)
            .unwrap_or(0.5); // default to mid-period if unknown

        // 1 / (fraction + 0.1) amplifies near settlement: ranges from ~1 (far) to ~10 (near)
        let proximity_scale = 1.0 / (settlement_fraction + 0.1);

        // Scale raw velocity to a reasonable range:
        // Typical basis changes are ~0.001 per second at most.
        // Multiply by 100 to get into [-1, 1] range before clamping.
        let scaled = raw_velocity * 100.0 * proximity_scale;

        scaled.clamp(-1.0, 1.0)
    }
}

/// Signal from arbitrage opportunity detection.
#[derive(Debug, Clone)]
pub struct ArbitrageSignal {
    /// Direction to take on perpetual
    pub direction: ArbitrageDirection,
    /// Current premium level
    pub premium: f64,
    /// Expected convergence (EWMA)
    pub expected_convergence: f64,
}

/// Arbitrage direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArbitrageDirection {
    /// Short perpetual (premium too high)
    ShortPerp,
    /// Long perpetual (premium too low)
    LongPerp,
}

/// Summary of funding rate status.
#[derive(Debug, Clone)]
pub struct FundingSummary {
    pub is_warmed_up: bool,
    pub current_rate: f64,
    pub ewma_rate: f64,
    pub annualized_rate: f64,
    pub observation_count: usize,
    pub is_extreme: bool,
    pub time_to_next_funding: Option<Duration>,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = FundingConfig::default();
        assert_eq!(config.funding_interval, Duration::from_secs(3600));
        assert_eq!(config.min_observations, 3);
    }

    #[test]
    fn test_new_estimator() {
        let estimator = FundingRateEstimator::new(FundingConfig::default());
        assert!(!estimator.is_warmed_up());
        assert_eq!(estimator.current_rate(), 0.0);
    }

    #[test]
    fn test_record_funding() {
        let mut estimator = FundingRateEstimator::new(FundingConfig::default());

        estimator.record_funding(0.0001, 1);
        estimator.record_funding(0.0002, 2);
        estimator.record_funding(0.0003, 3);

        assert!(estimator.is_warmed_up());
        assert!((estimator.current_rate() - 0.0003).abs() < 1e-10);
    }

    #[test]
    fn test_duplicate_period_ignored() {
        let mut estimator = FundingRateEstimator::new(FundingConfig::default());

        estimator.record_funding(0.0001, 1);
        estimator.record_funding(0.0002, 1); // Same period, should be ignored
        estimator.record_funding(0.0003, 2);

        assert_eq!(estimator.observation_count, 2); // Only 2 unique periods
    }

    #[test]
    fn test_rate_clamping() {
        let config = FundingConfig {
            max_rate: 0.001,
            ..Default::default()
        };
        let mut estimator = FundingRateEstimator::new(config);

        estimator.record_funding(0.1, 1); // Way over max
        assert!((estimator.current_rate() - 0.001).abs() < 1e-10); // Clamped to max
    }

    #[test]
    fn test_predicted_rate_mean_reversion() {
        let config = FundingConfig {
            mean_reversion_speed: 0.5, // Fast mean reversion
            long_run_rate: 0.0,
            ..Default::default()
        };
        let mut estimator = FundingRateEstimator::new(config);

        estimator.record_funding(0.001, 1); // High positive rate

        // After many periods, should converge to long-run rate (0)
        let far_future = estimator.predicted_rate(100.0 * 3600.0); // 100 hours
        assert!(far_future.abs() < 0.0001); // Should be near 0
    }

    #[test]
    fn test_funding_cost_long() {
        let mut estimator = FundingRateEstimator::new(FundingConfig::default());

        // Record positive funding (longs pay shorts)
        estimator.record_funding(0.0001, 1); // 0.01% per hour

        let position = 1.0; // 1 BTC long
        let mid_price = 50000.0; // $50k
        let holding_secs = 3600.0; // 1 hour

        let cost = estimator.funding_cost(position, mid_price, holding_secs);

        // Expected: $50k × 0.0001 × 1 = $5
        assert!((cost - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_funding_cost_short() {
        let mut estimator = FundingRateEstimator::new(FundingConfig::default());

        // Record positive funding (shorts receive)
        estimator.record_funding(0.0001, 1);

        let position = -1.0; // 1 BTC short
        let mid_price = 50000.0;
        let holding_secs = 3600.0;

        let cost = estimator.funding_cost(position, mid_price, holding_secs);

        // Expected: -$50k × 0.0001 × 1 = -$5 (revenue for short)
        assert!((cost + 5.0).abs() < 0.01);
    }

    #[test]
    fn test_drift_adjustment() {
        let mut estimator = FundingRateEstimator::new(FundingConfig::default());

        estimator.record_funding(0.0001, 1); // 0.01% per hour

        let drift = estimator.drift_adjustment();

        // Expected: -0.0001 × 8766 ≈ -0.8766 (negative because positive funding)
        assert!(drift < 0.0);
        assert!((drift + 0.8766).abs() < 0.01);
    }

    #[test]
    fn test_extreme_funding_detection() {
        let mut estimator = FundingRateEstimator::new(FundingConfig::default());

        estimator.record_funding(0.0005, 1); // 0.05% - not extreme
        assert!(!estimator.is_funding_extreme());

        estimator.record_funding(0.002, 2); // 0.2% - extreme
        assert!(estimator.is_funding_extreme());
    }

    #[test]
    fn test_funding_volatility() {
        let mut estimator = FundingRateEstimator::new(FundingConfig::default());

        // Record varying funding rates
        estimator.record_funding(0.0001, 1);
        estimator.record_funding(-0.0001, 2);
        estimator.record_funding(0.0002, 3);
        estimator.record_funding(-0.0002, 4);

        let vol = estimator.funding_volatility();
        assert!(vol > 0.0); // Should have some volatility
    }

    #[test]
    fn test_summary() {
        let mut estimator = FundingRateEstimator::new(FundingConfig::default());

        estimator.record_funding(0.0001, 1);
        estimator.record_funding(0.0001, 2);
        estimator.record_funding(0.0001, 3);

        let summary = estimator.summary();
        assert!(summary.is_warmed_up);
        assert!((summary.current_rate - 0.0001).abs() < 1e-10);
        assert_eq!(summary.observation_count, 3);
    }

    #[test]
    fn test_basis_velocity_no_data() {
        let estimator = FundingRateEstimator::new(FundingConfig::default());
        assert_eq!(estimator.basis_velocity(), 0.0);
    }

    #[test]
    fn test_basis_velocity_single_update() {
        let mut estimator = FundingRateEstimator::new(FundingConfig::default());
        estimator.update_from_prices(100.01, 100.0); // 0.01% premium
        // Only one update, prev_basis_time is None
        assert_eq!(estimator.basis_velocity(), 0.0);
    }

    #[test]
    fn test_basis_velocity_increasing_premium() {
        let mut estimator = FundingRateEstimator::new(FundingConfig::default());
        // Record a funding so next_funding_time is set
        estimator.record_funding(0.0001, 1);

        // First update: premium = 0.01%
        estimator.update_from_prices(100.01, 100.0);
        // Need time to elapse for non-zero velocity
        std::thread::sleep(std::time::Duration::from_millis(50));
        // Second update: premium = 0.05% (increasing)
        estimator.update_from_prices(100.05, 100.0);

        let vel = estimator.basis_velocity();
        assert!(vel > 0.0, "Increasing premium should give positive velocity, got {vel}");
    }

    #[test]
    fn test_basis_velocity_decreasing_premium() {
        let mut estimator = FundingRateEstimator::new(FundingConfig::default());
        estimator.record_funding(0.0001, 1);

        // First update: premium = 0.05%
        estimator.update_from_prices(100.05, 100.0);
        std::thread::sleep(std::time::Duration::from_millis(50));
        // Second update: premium = 0.01% (decreasing)
        estimator.update_from_prices(100.01, 100.0);

        let vel = estimator.basis_velocity();
        assert!(vel < 0.0, "Decreasing premium should give negative velocity, got {vel}");
    }

    #[test]
    fn test_basis_velocity_clamped() {
        let mut estimator = FundingRateEstimator::new(FundingConfig::default());
        estimator.record_funding(0.0001, 1);

        // First update: very low premium
        estimator.update_from_prices(100.0, 100.0);
        std::thread::sleep(std::time::Duration::from_millis(20));
        // Second update: huge premium jump (should clamp to 1.0)
        estimator.update_from_prices(110.0, 100.0); // 10% premium jump

        let vel = estimator.basis_velocity();
        assert!(vel <= 1.0 && vel >= -1.0, "Velocity should be clamped, got {vel}");
    }
}
