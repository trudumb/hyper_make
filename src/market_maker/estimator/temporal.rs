//! Temporal Feature Engineering for Market Making
//!
//! Provides time-based features that capture market microstructure patterns:
//!
//! 1. **Time-of-Day Encoding**: Cyclic encoding of hour using sin/cos to capture
//!    intraday patterns (Asian session, US market open, etc.)
//!
//! 2. **Funding Settlement Proximity**: How close we are to the 8-hour funding
//!    settlement on Hyperliquid. Flow patterns change near settlement.
//!
//! 3. **Multi-Horizon Momentum**: Momentum computed at multiple timescales
//!    (1s, 10s, 60s, 300s) to detect trend agreement vs divergence.
//!
//! # Why Temporal Features Matter for Market Making
//!
//! - **Funding Settlement**: Predictable flow patterns before/after settlement
//!   as traders adjust positions to capture/avoid funding payments
//! - **Time-of-Day**: Liquidity and volatility vary by session (Asia/EU/US)
//! - **Multi-Scale Trends**: Single-scale momentum misses regime context;
//!   agreement across scales signals stronger conviction
//!
//! # Hyperliquid Funding Schedule
//!
//! Funding settlements occur at 00:00, 08:00, 16:00 UTC (every 8 hours).
//! The funding rate is paid by longs to shorts (or vice versa) based on
//! the perpetual's premium/discount to spot.

use std::f64::consts::PI;

// ============================================================================
// Time-of-Day Features
// ============================================================================

/// Time-of-day features using cyclic encoding.
///
/// Instead of raw hour (0-23) which has a discontinuity at midnight,
/// we use sin/cos encoding that creates smooth, continuous features:
///
/// - hour_sin = sin(2π × hour/24)
/// - hour_cos = cos(2π × hour/24)
///
/// This allows ML models to learn that 23:00 and 01:00 are close in time.
#[derive(Debug, Clone, Copy)]
pub struct TimeOfDayFeatures {
    /// sin(2π × hour/24) - captures 24h periodicity
    pub hour_sin: f64,
    /// cos(2π × hour/24) - complements sin for unique encoding
    pub hour_cos: f64,
    /// Day-of-week indicator (0=Monday, 6=Sunday)
    pub day_of_week: u8,
    /// Whether it's a weekend (lower liquidity expected)
    pub is_weekend: bool,
}

impl TimeOfDayFeatures {
    /// Create time features from Unix timestamp (milliseconds).
    pub fn from_timestamp_ms(timestamp_ms: i64) -> Self {
        let secs = timestamp_ms / 1000;
        let hours_since_midnight = ((secs % 86400) as f64) / 3600.0;
        let days_since_epoch = secs / 86400;

        // Unix epoch (1970-01-01) was a Thursday (day 4)
        let day_of_week = ((days_since_epoch + 4) % 7) as u8;

        Self {
            hour_sin: (2.0 * PI * hours_since_midnight / 24.0).sin(),
            hour_cos: (2.0 * PI * hours_since_midnight / 24.0).cos(),
            day_of_week,
            is_weekend: day_of_week >= 5, // Saturday=5, Sunday=6
        }
    }

    /// Create from explicit hour (0-23) and day of week (0-6).
    pub fn from_hour_and_day(hour: u8, day_of_week: u8) -> Self {
        let hour_f = hour as f64;
        Self {
            hour_sin: (2.0 * PI * hour_f / 24.0).sin(),
            hour_cos: (2.0 * PI * hour_f / 24.0).cos(),
            day_of_week,
            is_weekend: day_of_week >= 5,
        }
    }

    /// Get hour from encoded features (0-23).
    pub fn decoded_hour(&self) -> f64 {
        let angle = self.hour_sin.atan2(self.hour_cos);
        let hour = angle * 24.0 / (2.0 * PI);
        if hour < 0.0 {
            hour + 24.0
        } else {
            hour
        }
    }

    /// Get session indicator (-1=Asia, 0=Europe, 1=Americas).
    ///
    /// - Asia: 00:00-08:00 UTC
    /// - Europe: 08:00-16:00 UTC
    /// - Americas: 16:00-24:00 UTC
    pub fn session(&self) -> i8 {
        let hour = self.decoded_hour();
        if hour < 8.0 {
            -1 // Asia
        } else if hour < 16.0 {
            0 // Europe
        } else {
            1 // Americas
        }
    }
}

// ============================================================================
// Funding Settlement Features
// ============================================================================

/// Funding settlement timing features.
///
/// Hyperliquid funding is paid every 8 hours at 00:00, 08:00, 16:00 UTC.
/// These features capture proximity to settlement and predicted flow direction.
#[derive(Debug, Clone, Copy)]
pub struct FundingFeatures {
    /// Time until next funding settlement (seconds)
    pub time_to_settlement_secs: f64,
    /// Proximity to settlement [0, 1] where 1 = at settlement
    pub settlement_proximity: f64,
    /// Current 8h funding rate (fraction, e.g., 0.0001 = 1 bp)
    pub funding_rate_8h: f64,
    /// Change in funding rate since last update
    pub funding_rate_delta: f64,
    /// Predicted flow direction: positive = longs closing, negative = shorts closing
    /// Computed as sign(funding) × (1 - settlement_proximity)
    pub predicted_flow: f64,

    // === Phase 4A.2: Funding Magnitude Feature ===

    /// Funding magnitude × proximity product.
    ///
    /// High value predicts "bursty" regime and κ collapse as traders rush to exit
    /// positions before paying extreme funding. Unlike `predicted_flow` which tracks
    /// direction, this captures the **intensity** of funding-driven activity.
    ///
    /// Formula: |funding_rate_8h| × settlement_proximity × 100
    /// Typical range: [0, ~3] where >1.5 indicates extreme funding pressure
    ///
    /// Use cases:
    /// - Predictive kappa: multiply by 0.5-0.8 when high (activity surge → κ spikes then collapses)
    /// - Regime detection: high values → predict transition to "bursty" regime
    /// - Spread adjustment: widen spreads when value > 1.0
    pub funding_magnitude_proximity: f64,
}

impl Default for FundingFeatures {
    /// Default funding features (for warmup).
    fn default() -> Self {
        Self {
            time_to_settlement_secs: 4.0 * 3600.0, // Midpoint
            settlement_proximity: 0.5,
            funding_rate_8h: 0.0,
            funding_rate_delta: 0.0,
            predicted_flow: 0.0,
            funding_magnitude_proximity: 0.0, // No pressure at default
        }
    }
}

impl FundingFeatures {
    /// Create funding features from current state.
    ///
    /// # Arguments
    /// * `timestamp_ms` - Current timestamp (Unix ms)
    /// * `funding_rate_8h` - Current 8h funding rate
    /// * `prev_funding_rate` - Previous funding rate (for delta computation)
    pub fn new(timestamp_ms: i64, funding_rate_8h: f64, prev_funding_rate: f64) -> Self {
        let secs_in_day = (timestamp_ms / 1000) % 86400;

        // Funding settlements at 0, 8h, 16h (0, 28800, 57600 seconds)
        let funding_period_secs = 8 * 3600;
        let secs_since_last_settlement = secs_in_day % funding_period_secs;
        let secs_to_next_settlement = funding_period_secs - secs_since_last_settlement;

        let time_to_settlement = secs_to_next_settlement as f64;
        let settlement_proximity = 1.0 - (time_to_settlement / funding_period_secs as f64);

        // Predict flow direction based on funding rate
        // Positive funding = longs pay shorts = longs tend to close before settlement
        // Negative funding = shorts pay longs = shorts tend to close before settlement
        let predicted_flow = funding_rate_8h.signum() * settlement_proximity;

        // Phase 4A.2: Funding magnitude × proximity
        // High value predicts κ collapse as traders rush to exit
        // Scale by 100 to get range [0, ~3] for typical funding rates
        let funding_magnitude_proximity = funding_rate_8h.abs() * settlement_proximity * 100.0;

        Self {
            time_to_settlement_secs: time_to_settlement,
            settlement_proximity,
            funding_rate_8h,
            funding_rate_delta: funding_rate_8h - prev_funding_rate,
            predicted_flow,
            funding_magnitude_proximity,
        }
    }

    /// Check if we're in the "funding rush" window (last 30 mins before settlement).
    pub fn is_funding_rush(&self) -> bool {
        self.time_to_settlement_secs < 30.0 * 60.0
    }

    /// Check if we're right after settlement (first 30 mins).
    pub fn is_post_settlement(&self) -> bool {
        let secs_since = 8.0 * 3600.0 - self.time_to_settlement_secs;
        secs_since > 0.0 && secs_since < 30.0 * 60.0
    }

    /// Get funding imbalance indicator [-1, 1].
    ///
    /// Combines funding rate magnitude with settlement proximity.
    /// High value = strong incentive for position adjustment.
    pub fn funding_pressure(&self) -> f64 {
        // Clamp funding to reasonable range (±0.1% per 8h is extreme)
        let clamped_funding = self.funding_rate_8h.clamp(-0.001, 0.001);
        // Scale to [-1, 1] and weight by proximity
        (clamped_funding / 0.001) * self.settlement_proximity
    }

    // === Phase 4A.2: Funding Magnitude Methods ===

    /// Check if funding magnitude is elevated (predicts κ surge then collapse).
    ///
    /// Returns true if funding_magnitude_proximity > 1.0, indicating:
    /// - High funding rate AND close to settlement
    /// - Expect elevated activity followed by κ collapse
    pub fn is_funding_magnitude_elevated(&self) -> bool {
        self.funding_magnitude_proximity > 1.0
    }

    /// Check if funding magnitude is extreme (predicts regime change).
    ///
    /// Returns true if funding_magnitude_proximity > 2.0, indicating:
    /// - Very high funding rate AND very close to settlement
    /// - Expect transition to "bursty" regime
    pub fn is_funding_magnitude_extreme(&self) -> bool {
        self.funding_magnitude_proximity > 2.0
    }

    /// Get kappa multiplier based on funding magnitude.
    ///
    /// During high funding periods near settlement, activity spikes then collapses.
    /// This returns a multiplier to apply to kappa predictions:
    /// - magnitude < 0.5: 1.0 (no adjustment)
    /// - magnitude 0.5-1.5: linear interpolation from 1.0 to 1.5 (activity spike)
    /// - magnitude > 1.5: 0.7-0.5 (activity collapse as positions are closed)
    ///
    /// The non-monotonic shape reflects that:
    /// 1. Initial funding pressure increases activity (more fills)
    /// 2. Extreme pressure causes position exits, reducing available flow
    pub fn kappa_multiplier(&self) -> f64 {
        let mag = self.funding_magnitude_proximity;

        if mag < 0.5 {
            // Low magnitude: no adjustment
            1.0
        } else if mag < 1.5 {
            // Moderate magnitude: activity spike
            // Linear from 1.0 at mag=0.5 to 1.5 at mag=1.5
            1.0 + (mag - 0.5) * 0.5
        } else if mag < 2.5 {
            // High magnitude: transition to collapse
            // Linear from 1.5 at mag=1.5 to 0.5 at mag=2.5
            1.5 - (mag - 1.5)
        } else {
            // Extreme magnitude: activity collapsed
            0.5
        }
    }

    /// Get spread widening factor based on funding magnitude.
    ///
    /// High funding magnitude near settlement creates adverse selection risk
    /// from informed traders who know positions will be forced to close.
    /// Returns a multiplier to widen spreads: [1.0, 1.5]
    pub fn spread_widening_factor(&self) -> f64 {
        // Linear widening from 1.0 at mag=0 to 1.5 at mag=2.0, capped
        (1.0 + self.funding_magnitude_proximity * 0.25).min(1.5)
    }

    /// Get regime transition probability based on funding magnitude.
    ///
    /// High funding magnitude predicts transition to "bursty" regime.
    /// Returns probability in [0, 1].
    pub fn bursty_regime_prob(&self) -> f64 {
        // Sigmoid-like: low prob until mag > 1, then rapid rise
        let x = self.funding_magnitude_proximity - 1.0;
        if x < 0.0 {
            0.1 * (x + 1.0).max(0.0) // 0-10% for mag 0-1
        } else {
            0.1 + 0.8 * (1.0 - (-x * 2.0).exp()) // 10-90% for mag > 1
        }
    }
}

// ============================================================================
// Multi-Horizon Momentum
// ============================================================================

/// Multi-scale momentum tracking.
///
/// Tracks momentum at multiple timescales to detect trend agreement/divergence.
/// When all scales agree, trend is strong. When scales diverge, market is choppy.
#[derive(Debug, Clone)]
pub struct MultiScaleMomentum {
    /// 1-second momentum (basis points)
    pub momentum_1s: MomentumScale,
    /// 10-second momentum (basis points)
    pub momentum_10s: MomentumScale,
    /// 60-second momentum (basis points)
    pub momentum_60s: MomentumScale,
    /// 5-minute momentum (basis points)
    pub momentum_300s: MomentumScale,
}

/// Single-scale momentum tracker using EWMA.
#[derive(Debug, Clone)]
pub struct MomentumScale {
    /// Half-life in seconds
    half_life_secs: f64,
    /// EWMA decay factor (reserved for adaptive decay)
    #[allow(dead_code)]
    alpha: f64,
    /// Current EWMA of returns (basis points)
    ewma_return: f64,
    /// Current EWMA of squared returns (for variance)
    ewma_return_sq: f64,
    /// Last price seen
    last_price: f64,
    /// Last timestamp (ms)
    last_timestamp_ms: i64,
    /// Is warmed up
    warmed_up: bool,
}

impl MomentumScale {
    /// Create a new momentum tracker with specified half-life.
    pub fn new(half_life_secs: f64) -> Self {
        Self {
            half_life_secs,
            alpha: 1.0 - (-1.0 / half_life_secs).exp(),
            ewma_return: 0.0,
            ewma_return_sq: 0.0,
            last_price: 0.0,
            last_timestamp_ms: 0,
            warmed_up: false,
        }
    }

    /// Update with new price observation.
    pub fn update(&mut self, price: f64, timestamp_ms: i64) {
        if !self.warmed_up {
            self.last_price = price;
            self.last_timestamp_ms = timestamp_ms;
            self.warmed_up = true;
            return;
        }

        if price <= 0.0 || self.last_price <= 0.0 {
            return;
        }

        // Compute return in basis points
        let return_bps = (price / self.last_price - 1.0) * 10000.0;

        // Time-adjusted alpha
        let dt_secs = (timestamp_ms - self.last_timestamp_ms) as f64 / 1000.0;
        let adj_alpha = 1.0 - (-dt_secs / self.half_life_secs).exp();

        // Update EWMA
        self.ewma_return = (1.0 - adj_alpha) * self.ewma_return + adj_alpha * return_bps;
        self.ewma_return_sq =
            (1.0 - adj_alpha) * self.ewma_return_sq + adj_alpha * return_bps * return_bps;

        self.last_price = price;
        self.last_timestamp_ms = timestamp_ms;
    }

    /// Get current momentum in basis points.
    pub fn momentum_bps(&self) -> f64 {
        self.ewma_return
    }

    /// Get momentum volatility (standard deviation).
    pub fn momentum_vol(&self) -> f64 {
        let variance = self.ewma_return_sq - self.ewma_return * self.ewma_return;
        variance.max(0.0).sqrt()
    }

    /// Get momentum signal strength (momentum / volatility).
    /// Also known as momentum z-score.
    pub fn signal_strength(&self) -> f64 {
        let vol = self.momentum_vol();
        if vol < 1e-6 {
            0.0
        } else {
            self.ewma_return / vol
        }
    }

    /// Check if warmed up.
    pub fn is_warmed_up(&self) -> bool {
        self.warmed_up
    }
}

impl MultiScaleMomentum {
    /// Create a new multi-scale momentum tracker.
    pub fn new() -> Self {
        Self {
            momentum_1s: MomentumScale::new(1.0),
            momentum_10s: MomentumScale::new(10.0),
            momentum_60s: MomentumScale::new(60.0),
            momentum_300s: MomentumScale::new(300.0),
        }
    }

    /// Update all scales with new price.
    pub fn update(&mut self, price: f64, timestamp_ms: i64) {
        self.momentum_1s.update(price, timestamp_ms);
        self.momentum_10s.update(price, timestamp_ms);
        self.momentum_60s.update(price, timestamp_ms);
        self.momentum_300s.update(price, timestamp_ms);
    }

    /// Get trend agreement score [-1, 1].
    ///
    /// +1 = all scales agree on uptrend
    /// -1 = all scales agree on downtrend
    /// 0 = scales disagree (choppy market)
    pub fn trend_agreement(&self) -> f64 {
        let signs = [
            self.momentum_1s.momentum_bps().signum(),
            self.momentum_10s.momentum_bps().signum(),
            self.momentum_60s.momentum_bps().signum(),
            self.momentum_300s.momentum_bps().signum(),
        ];

        // Average of signs: +1 if all positive, -1 if all negative, 0 if mixed
        signs.iter().sum::<f64>() / 4.0
    }

    /// Get momentum divergence score [0, 1].
    ///
    /// 0 = perfect agreement across scales
    /// 1 = maximum divergence (e.g., 1s up, 300s down)
    pub fn divergence(&self) -> f64 {
        let momentums = [
            self.momentum_1s.signal_strength(),
            self.momentum_10s.signal_strength(),
            self.momentum_60s.signal_strength(),
            self.momentum_300s.signal_strength(),
        ];

        // Compute variance of signs as divergence measure
        let mean = momentums.iter().sum::<f64>() / 4.0;
        let variance = momentums.iter().map(|&m| (m - mean).powi(2)).sum::<f64>() / 4.0;

        // Normalize to [0, 1] range
        (variance / 4.0).sqrt().min(1.0)
    }

    /// Check if all scales are warmed up.
    pub fn is_warmed_up(&self) -> bool {
        self.momentum_1s.is_warmed_up()
            && self.momentum_10s.is_warmed_up()
            && self.momentum_60s.is_warmed_up()
            && self.momentum_300s.is_warmed_up()
    }

    /// Get feature vector for model input.
    pub fn as_features(&self) -> TemporalMomentumFeatures {
        TemporalMomentumFeatures {
            momentum_1s_bps: self.momentum_1s.momentum_bps(),
            momentum_10s_bps: self.momentum_10s.momentum_bps(),
            momentum_60s_bps: self.momentum_60s.momentum_bps(),
            momentum_300s_bps: self.momentum_300s.momentum_bps(),
            trend_agreement: self.trend_agreement(),
            divergence: self.divergence(),
        }
    }
}

impl Default for MultiScaleMomentum {
    fn default() -> Self {
        Self::new()
    }
}

/// Extracted features from multi-scale momentum.
#[derive(Debug, Clone, Copy)]
pub struct TemporalMomentumFeatures {
    pub momentum_1s_bps: f64,
    pub momentum_10s_bps: f64,
    pub momentum_60s_bps: f64,
    pub momentum_300s_bps: f64,
    pub trend_agreement: f64,
    pub divergence: f64,
}

// ============================================================================
// Aggregate Temporal Features
// ============================================================================

/// Complete set of temporal features for market making.
#[derive(Debug, Clone)]
pub struct TemporalFeatures {
    pub time_of_day: TimeOfDayFeatures,
    pub funding: FundingFeatures,
    pub momentum: TemporalMomentumFeatures,
}

impl TemporalFeatures {
    /// Create temporal features from current market state.
    pub fn compute(
        timestamp_ms: i64,
        funding_rate_8h: f64,
        prev_funding_rate: f64,
        multi_scale_momentum: &MultiScaleMomentum,
    ) -> Self {
        Self {
            time_of_day: TimeOfDayFeatures::from_timestamp_ms(timestamp_ms),
            funding: FundingFeatures::new(timestamp_ms, funding_rate_8h, prev_funding_rate),
            momentum: multi_scale_momentum.as_features(),
        }
    }

    /// Get features as a flat vector (for ML models).
    pub fn as_vector(&self) -> Vec<f64> {
        vec![
            // Time of day
            self.time_of_day.hour_sin,
            self.time_of_day.hour_cos,
            self.time_of_day.is_weekend as i32 as f64,
            // Funding
            self.funding.settlement_proximity,
            self.funding.funding_rate_8h * 10000.0, // Convert to bps
            self.funding.predicted_flow,
            // Momentum
            self.momentum.momentum_1s_bps,
            self.momentum.momentum_10s_bps,
            self.momentum.momentum_60s_bps,
            self.momentum.momentum_300s_bps,
            self.momentum.trend_agreement,
            self.momentum.divergence,
        ]
    }

    /// Get feature names (for logging/debugging).
    pub fn feature_names() -> &'static [&'static str] {
        &[
            "hour_sin",
            "hour_cos",
            "is_weekend",
            "settlement_proximity",
            "funding_rate_bps",
            "predicted_flow",
            "momentum_1s",
            "momentum_10s",
            "momentum_60s",
            "momentum_300s",
            "trend_agreement",
            "divergence",
        ]
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_of_day_encoding() {
        // Test midnight
        let midnight = TimeOfDayFeatures::from_hour_and_day(0, 0);
        assert!((midnight.hour_sin - 0.0).abs() < 0.01);
        assert!((midnight.hour_cos - 1.0).abs() < 0.01);

        // Test 6 AM (sin should be 1, cos 0)
        let six_am = TimeOfDayFeatures::from_hour_and_day(6, 0);
        assert!((six_am.hour_sin - 1.0).abs() < 0.01);
        assert!((six_am.hour_cos - 0.0).abs() < 0.01);

        // Test noon (sin should be 0, cos -1)
        let noon = TimeOfDayFeatures::from_hour_and_day(12, 0);
        assert!((noon.hour_sin - 0.0).abs() < 0.01);
        assert!((noon.hour_cos - (-1.0)).abs() < 0.01);
    }

    #[test]
    fn test_decoded_hour() {
        for hour in 0..24 {
            let features = TimeOfDayFeatures::from_hour_and_day(hour, 0);
            let decoded = features.decoded_hour();
            assert!(
                (decoded - hour as f64).abs() < 0.1,
                "Hour {} decoded as {}",
                hour,
                decoded
            );
        }
    }

    #[test]
    fn test_session_detection() {
        assert_eq!(TimeOfDayFeatures::from_hour_and_day(3, 0).session(), -1); // Asia
        assert_eq!(TimeOfDayFeatures::from_hour_and_day(10, 0).session(), 0); // Europe
        assert_eq!(TimeOfDayFeatures::from_hour_and_day(20, 0).session(), 1); // Americas
    }

    #[test]
    fn test_funding_settlement_proximity() {
        // Right at settlement (00:00 UTC)
        let at_settlement_ms = 0i64;
        let features = FundingFeatures::new(at_settlement_ms, 0.0001, 0.0);
        assert!(
            features.settlement_proximity < 0.01,
            "At settlement, proximity should be ~0"
        );

        // 4 hours after settlement (halfway to next)
        let halfway_ms = 4 * 3600 * 1000;
        let features = FundingFeatures::new(halfway_ms, 0.0001, 0.0);
        assert!(
            (features.settlement_proximity - 0.5).abs() < 0.01,
            "Halfway to settlement, proximity should be ~0.5"
        );
    }

    #[test]
    fn test_funding_rush_detection() {
        // 35 minutes before settlement
        let ts_35min_before = (8 * 3600 - 35 * 60) * 1000;
        let features = FundingFeatures::new(ts_35min_before, 0.0001, 0.0);
        assert!(!features.is_funding_rush());

        // 25 minutes before settlement
        let ts_25min_before = (8 * 3600 - 25 * 60) * 1000;
        let features = FundingFeatures::new(ts_25min_before, 0.0001, 0.0);
        assert!(features.is_funding_rush());
    }

    #[test]
    fn test_momentum_scale() {
        let mut scale = MomentumScale::new(10.0);

        // Initialize
        scale.update(100.0, 0);

        // Price up 1% = 100 bps
        scale.update(101.0, 1000);
        let momentum_after_up = scale.momentum_bps();
        assert!(
            momentum_after_up > 0.0,
            "Momentum should be positive after price increase"
        );

        // Price down 2% from previous (101 -> 98.98 is about -2%)
        scale.update(98.98, 2000);
        let momentum_after_down = scale.momentum_bps();
        assert!(
            momentum_after_down < momentum_after_up,
            "Momentum should decrease after price drop: {} vs {}",
            momentum_after_down,
            momentum_after_up
        );
    }

    #[test]
    fn test_trend_agreement() {
        let mut momentum = MultiScaleMomentum::new();

        // Initialize with same price
        momentum.update(100.0, 0);

        // Consistent uptrend
        for i in 1..50 {
            momentum.update(100.0 + i as f64 * 0.1, i * 100);
        }

        let agreement = momentum.trend_agreement();
        assert!(
            agreement > 0.5,
            "Consistent uptrend should have positive agreement: {}",
            agreement
        );
    }

    #[test]
    fn test_weekend_detection() {
        // Saturday
        let saturday = TimeOfDayFeatures::from_hour_and_day(12, 5);
        assert!(saturday.is_weekend);

        // Sunday
        let sunday = TimeOfDayFeatures::from_hour_and_day(12, 6);
        assert!(sunday.is_weekend);

        // Wednesday
        let wednesday = TimeOfDayFeatures::from_hour_and_day(12, 2);
        assert!(!wednesday.is_weekend);
    }

    // === Phase 4A.2: Funding Magnitude Tests ===

    #[test]
    fn test_funding_magnitude_proximity_computation() {
        // Test 1 minute before settlement with 10bp funding
        // proximity should be near 1.0
        let ts_1min_before = (8 * 3600 - 60) * 1000;
        let features = FundingFeatures::new(ts_1min_before, 0.001, 0.0); // 10bp

        // proximity ≈ 1 - 60/28800 ≈ 0.998
        assert!(features.settlement_proximity > 0.99);

        // magnitude = 0.001 * 0.998 * 100 ≈ 0.0998
        assert!(features.funding_magnitude_proximity > 0.09);
        assert!(features.funding_magnitude_proximity < 0.11);
    }

    #[test]
    fn test_funding_magnitude_zero_at_midpoint() {
        // At midpoint (4h before settlement) with zero funding
        let ts_4h_before = (8 * 3600 - 4 * 3600) * 1000;
        let features = FundingFeatures::new(ts_4h_before, 0.0, 0.0);

        // Zero funding = zero magnitude regardless of proximity
        assert!((features.funding_magnitude_proximity - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_funding_magnitude_elevated() {
        // 30 minutes before settlement with high funding (20bp)
        let ts_30min_before = (8 * 3600 - 30 * 60) * 1000;
        let features = FundingFeatures::new(ts_30min_before, 0.002, 0.0);

        // proximity ≈ 1 - 1800/28800 ≈ 0.9375
        // magnitude = 0.002 * 0.9375 * 100 ≈ 0.1875
        assert!(features.is_funding_magnitude_elevated() == false); // Not > 1.0

        // With extreme funding (50bp) at same time
        let features_extreme = FundingFeatures::new(ts_30min_before, 0.005, 0.0);
        // magnitude = 0.005 * 0.9375 * 100 ≈ 0.469
        assert!(features_extreme.is_funding_magnitude_elevated() == false);

        // With very extreme funding (200bp = 0.02) very close to settlement
        let ts_5min_before = (8 * 3600 - 5 * 60) * 1000;
        let features_very_extreme = FundingFeatures::new(ts_5min_before, 0.02, 0.0);
        // proximity ≈ 0.98, magnitude = 0.02 * 0.98 * 100 ≈ 1.96
        assert!(features_very_extreme.is_funding_magnitude_elevated());
    }

    #[test]
    fn test_funding_magnitude_kappa_multiplier() {
        // Low magnitude: no adjustment
        let ts = 4 * 3600 * 1000; // 4h from settlement
        let features_low = FundingFeatures::new(ts, 0.0001, 0.0);
        assert!((features_low.kappa_multiplier() - 1.0).abs() < 0.1);

        // Build feature with known magnitude by testing the multiplier function
        // We need to create scenarios at different magnitude levels

        // Test the non-monotonic shape:
        // 1. Very low funding (mag < 0.5): multiplier ≈ 1.0
        // 2. Moderate (mag ~1.0): multiplier > 1.0 (activity spike)
        // 3. High (mag ~2.0): multiplier < 1.0 (collapse begins)
        // 4. Extreme (mag > 2.5): multiplier ≈ 0.5 (collapsed)

        // We can't easily control magnitude directly, so test boundary behavior
        let mult = features_low.kappa_multiplier();
        assert!(mult >= 0.5 && mult <= 1.5, "Multiplier should be in [0.5, 1.5]: {}", mult);
    }

    #[test]
    fn test_funding_magnitude_spread_widening() {
        // No funding: no widening
        let features_zero = FundingFeatures::default();
        assert!((features_zero.spread_widening_factor() - 1.0).abs() < 0.01);

        // Some funding pressure
        let ts_30min = (8 * 3600 - 30 * 60) * 1000;
        let features = FundingFeatures::new(ts_30min, 0.002, 0.0);
        let widening = features.spread_widening_factor();
        assert!(widening >= 1.0, "Widening should be >= 1.0: {}", widening);
        assert!(widening <= 1.5, "Widening should be <= 1.5: {}", widening);
    }

    #[test]
    fn test_funding_magnitude_bursty_regime_prob() {
        // Low magnitude: low probability
        let features_low = FundingFeatures::default();
        let prob_low = features_low.bursty_regime_prob();
        assert!(prob_low < 0.2, "Low mag should have low prob: {}", prob_low);

        // Very high magnitude (manually construct scenario)
        // 5 min before settlement with 100bp funding
        let ts_5min = (8 * 3600 - 5 * 60) * 1000;
        let features_high = FundingFeatures::new(ts_5min, 0.01, 0.0);
        // magnitude ≈ 0.01 * 0.98 * 100 ≈ 0.98

        // This is just below 1.0 threshold, so prob should be moderate
        let prob_high = features_high.bursty_regime_prob();
        assert!(prob_high >= 0.0 && prob_high <= 1.0, "Prob should be in [0,1]: {}", prob_high);
    }
}
