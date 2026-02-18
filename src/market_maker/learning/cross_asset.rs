//! Level 5: Cross-Asset Information
//!
//! Edge might not be in the asset you're trading. This module provides:
//! - BTC → Altcoin lead-lag signals
//! - Funding rate divergence
//! - Open interest → volatility predictions

use super::types::{CrossSignal, RingBuffer};

/// Bayesian estimate with mean and variance.
#[derive(Debug, Clone, Copy, Default)]
pub struct BayesianEstimate {
    /// Mean of the posterior
    pub mean: f64,
    /// Variance of the posterior
    pub variance: f64,
    /// Number of observations
    pub n_observations: usize,
}

impl BayesianEstimate {
    /// Create a new estimate with prior.
    pub fn new(prior_mean: f64, prior_variance: f64) -> Self {
        Self {
            mean: prior_mean,
            variance: prior_variance,
            n_observations: 0,
        }
    }

    /// Update with new observation (conjugate normal update).
    pub fn update(&mut self, observation: f64, observation_variance: f64) {
        let prior_precision = 1.0 / self.variance;
        let obs_precision = 1.0 / observation_variance;

        let new_precision = prior_precision + obs_precision;
        let new_mean = (prior_precision * self.mean + obs_precision * observation) / new_precision;

        self.mean = new_mean;
        // Floor prevents division-by-zero in downstream confidence calculations
        // after many convergent updates collapse variance toward zero
        self.variance = (1.0 / new_precision).max(1e-10);
        self.n_observations += 1;
    }

    /// Get 95% confidence interval.
    pub fn confidence_interval(&self) -> (f64, f64) {
        let std = self.variance.sqrt();
        (self.mean - 1.96 * std, self.mean + 1.96 * std)
    }
}

/// Lead-lag model between two assets.
///
/// Models how movements in the leader asset predict movements in the follower.
#[derive(Debug, Clone)]
pub struct LeadLagModel {
    /// Leader asset name
    pub leader: String,
    /// Follower asset name
    pub follower: String,
    /// Estimated lead time in milliseconds
    pub lead_ms: BayesianEstimate,
    /// Transfer coefficient (how much of leader move transfers)
    pub transfer_coef: BayesianEstimate,
    /// Rolling correlation for confidence
    pub correlation: f64,
    /// Recent leader returns for signal generation
    leader_returns: RingBuffer<(u64, f64)>, // (timestamp_ms, return)
}

impl LeadLagModel {
    /// Create a new lead-lag model.
    pub fn new(leader: &str, follower: &str) -> Self {
        Self {
            leader: leader.to_string(),
            follower: follower.to_string(),
            lead_ms: BayesianEstimate::new(500.0, 200.0 * 200.0), // Prior: 500ms ± 200ms
            transfer_coef: BayesianEstimate::new(0.7, 0.2 * 0.2), // Prior: 70% ± 20%
            correlation: 0.8,
            leader_returns: RingBuffer::new(100),
        }
    }

    /// Record a leader return.
    pub fn record_leader_return(&mut self, timestamp_ms: u64, return_bps: f64) {
        self.leader_returns.push((timestamp_ms, return_bps));
    }

    /// Get signal from leader move.
    ///
    /// Returns expected follower return based on recent leader moves.
    pub fn signal(&self, now_ms: u64) -> f64 {
        let lead_time = self.lead_ms.mean as u64;

        // Find leader returns within the lead window
        let mut signal = 0.0;
        let mut weight_sum = 0.0;

        for &(ts, ret) in self.leader_returns.iter() {
            let age_ms = now_ms.saturating_sub(ts);

            // Only consider returns within 3× lead time
            if age_ms > lead_time * 3 {
                continue;
            }

            // Exponential decay of signal
            let decay = (-(age_ms as f64) / self.lead_ms.mean).exp();
            signal += ret * self.transfer_coef.mean * decay;
            weight_sum += decay;
        }

        if weight_sum > 0.0 {
            signal / weight_sum
        } else {
            0.0
        }
    }

    /// Get confidence in the signal.
    pub fn signal_confidence(&self) -> f64 {
        // Confidence based on correlation and observations
        let obs_factor = (self.lead_ms.n_observations as f64 / 100.0).min(1.0);
        self.correlation * obs_factor
    }

    /// Update model with observed lead-lag relationship.
    pub fn update(&mut self, observed_lead_ms: f64, observed_transfer: f64) {
        self.lead_ms.update(observed_lead_ms, 100.0 * 100.0);
        self.transfer_coef.update(observed_transfer, 0.1 * 0.1);
    }
}

/// Funding rate divergence model.
///
/// Predicts mean reversion when funding diverges from historical norm.
#[derive(Debug, Clone)]
pub struct FundingDivergenceModel {
    /// Long-term mean funding rate (8h rate)
    pub long_term_mean: f64,
    /// Mean reversion speed (fraction per 8h period)
    pub mean_reversion_speed: f64,
    /// Recent funding observations
    funding_history: RingBuffer<f64>,
}

impl Default for FundingDivergenceModel {
    fn default() -> Self {
        Self {
            long_term_mean: 0.0001, // 0.01% per 8h (slightly positive bias)
            mean_reversion_speed: 0.3, // 30% mean reversion per period
            funding_history: RingBuffer::new(100),
        }
    }
}

impl FundingDivergenceModel {
    /// Record a funding observation.
    pub fn record_funding(&mut self, rate: f64) {
        self.funding_history.push(rate);
        self.update_long_term_mean();
    }

    /// Update long-term mean from history.
    fn update_long_term_mean(&mut self) {
        if self.funding_history.len() < 10 {
            return;
        }

        let sum: f64 = self.funding_history.iter().sum();
        let n = self.funding_history.len() as f64;

        // Exponential smoothing toward historical mean
        let historical_mean = sum / n;
        self.long_term_mean = 0.9 * self.long_term_mean + 0.1 * historical_mean;
    }

    /// Get signal from funding divergence.
    ///
    /// Positive signal = expect price to rise (funding too high, shorts get paid)
    /// Negative signal = expect price to fall (funding too low, longs get paid)
    pub fn signal(&self, current_funding: f64) -> f64 {
        let divergence = current_funding - self.long_term_mean;

        // Expected convergence
        divergence * self.mean_reversion_speed * 100.0 // Convert to bps
    }

    /// Get confidence in the signal.
    pub fn confidence(&self) -> f64 {
        (self.funding_history.len() as f64 / 50.0).min(1.0)
    }
}

/// Open interest → volatility predictor.
///
/// Large OI changes often precede volatility changes.
#[derive(Debug, Clone)]
pub struct OIVolModel {
    /// Recent OI observations (timestamp, OI)
    oi_history: RingBuffer<(u64, f64)>,
    /// Threshold for significant OI change (fraction)
    significance_threshold: f64,
}

impl Default for OIVolModel {
    fn default() -> Self {
        Self {
            oi_history: RingBuffer::new(100),
            significance_threshold: 0.05, // 5% change is significant
        }
    }
}

impl OIVolModel {
    /// Record an OI observation.
    pub fn record_oi(&mut self, timestamp_ms: u64, oi: f64) {
        self.oi_history.push((timestamp_ms, oi));
    }

    /// Get volatility regime prediction.
    ///
    /// Returns multiplier for expected volatility (1.0 = normal).
    pub fn vol_multiplier(&self) -> f64 {
        if self.oi_history.len() < 10 {
            return 1.0;
        }

        // Calculate recent OI change
        let recent: Vec<_> = self.oi_history.iter().collect();
        let latest_oi = recent.last().map(|(_, oi)| *oi).unwrap_or(1.0);
        let oldest_oi = recent.first().map(|(_, oi)| *oi).unwrap_or(1.0);

        if oldest_oi < 0.001 {
            return 1.0;
        }

        let oi_change = (latest_oi - oldest_oi) / oldest_oi;

        // Large OI increase → expect higher vol
        // Large OI decrease → also expect higher vol (liquidations)
        if oi_change.abs() > self.significance_threshold {
            1.0 + oi_change.abs() * 2.0 // Up to 1.1× vol multiplier
        } else {
            1.0
        }
    }
}

/// Aggregated cross-asset signals.
#[derive(Debug, Clone, Default)]
pub struct CrossAssetSignals {
    /// BTC → Altcoin lead-lag model
    pub btc_alt_lead: Option<LeadLagModel>,
    /// Funding rate divergence model
    pub funding_divergence: FundingDivergenceModel,
    /// OI → vol predictor
    pub oi_vol_predictor: OIVolModel,
}

impl CrossAssetSignals {
    /// Create for an altcoin (with BTC lead-lag).
    pub fn for_altcoin(asset: &str) -> Self {
        Self {
            btc_alt_lead: Some(LeadLagModel::new("BTC", asset)),
            funding_divergence: FundingDivergenceModel::default(),
            oi_vol_predictor: OIVolModel::default(),
        }
    }

    /// Create for BTC (no lead-lag).
    pub fn for_btc() -> Self {
        Self {
            btc_alt_lead: None,
            funding_divergence: FundingDivergenceModel::default(),
            oi_vol_predictor: OIVolModel::default(),
        }
    }

    /// Aggregate all signals into a single CrossSignal.
    pub fn aggregate_signal(&self, current_funding: f64) -> CrossSignal {
        let mut expected_move = 0.0;
        let mut confidence = 0.0;
        let mut age_ms = 0;

        // Lead-lag signal
        if let Some(ref lead_lag) = self.btc_alt_lead {
            let now_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0);

            let ll_signal = lead_lag.signal(now_ms);
            let ll_confidence = lead_lag.signal_confidence();

            expected_move += ll_signal * ll_confidence;
            confidence = ll_confidence;
            age_ms = lead_lag.lead_ms.mean as u64;
        }

        // Funding signal (smaller weight)
        let funding_signal = self.funding_divergence.signal(current_funding);
        let funding_conf = self.funding_divergence.confidence();
        expected_move += funding_signal * funding_conf * 0.3; // 30% weight

        CrossSignal {
            expected_move_bps: expected_move,
            confidence: confidence.max(funding_conf * 0.3),
            leader: self
                .btc_alt_lead
                .as_ref()
                .map(|m| m.leader.clone())
                .unwrap_or_default(),
            age_ms,
        }
    }

    /// Get volatility multiplier from OI.
    pub fn vol_multiplier(&self) -> f64 {
        self.oi_vol_predictor.vol_multiplier()
    }

    /// Update BTC return (for altcoin lead-lag).
    pub fn update_btc_return(&mut self, timestamp_ms: u64, return_bps: f64) {
        if let Some(ref mut lead_lag) = self.btc_alt_lead {
            lead_lag.record_leader_return(timestamp_ms, return_bps);
        }
    }

    /// Update funding observation.
    pub fn update_funding(&mut self, rate: f64) {
        self.funding_divergence.record_funding(rate);
    }

    /// Update OI observation.
    pub fn update_oi(&mut self, timestamp_ms: u64, oi: f64) {
        self.oi_vol_predictor.record_oi(timestamp_ms, oi);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bayesian_estimate() {
        let mut est = BayesianEstimate::new(100.0, 25.0);

        // Update with observations near 90
        for _ in 0..10 {
            est.update(90.0, 10.0);
        }

        // Mean should shift toward 90
        assert!(est.mean < 100.0);
        assert!(est.mean > 85.0);

        // Variance should decrease
        assert!(est.variance < 25.0);
    }

    #[test]
    fn test_lead_lag_signal() {
        let mut model = LeadLagModel::new("BTC", "SOL");

        let now_ms = 1000000;

        // Record BTC move up
        model.record_leader_return(now_ms - 100, 10.0); // 10bp up, 100ms ago

        // Signal should be positive (expect SOL to follow)
        let signal = model.signal(now_ms);
        assert!(signal > 0.0, "Expected positive signal, got {signal}");
    }

    #[test]
    fn test_funding_divergence() {
        let mut model = FundingDivergenceModel::default();

        // Record some funding observations around 0.01%
        for _ in 0..20 {
            model.record_funding(0.0001);
        }

        // Signal when funding is high (0.05%)
        let signal_high = model.signal(0.0005);
        assert!(signal_high > 0.0, "Expected positive signal for high funding");

        // Signal when funding is low (-0.05%)
        let signal_low = model.signal(-0.0005);
        assert!(signal_low < 0.0, "Expected negative signal for low funding");
    }

    #[test]
    fn test_oi_vol_multiplier() {
        let mut model = OIVolModel::default();

        // Record stable OI
        for i in 0..20 {
            model.record_oi(i * 1000, 1000000.0);
        }

        // Should be normal vol
        let mult_stable = model.vol_multiplier();
        assert!((mult_stable - 1.0).abs() < 0.01);

        // Record big OI increase
        for i in 20..30 {
            model.record_oi(i * 1000, 1000000.0 + (i as f64 - 20.0) * 20000.0);
        }

        // Should predict higher vol
        let mult_increase = model.vol_multiplier();
        assert!(mult_increase > 1.0, "Expected higher vol multiplier after OI increase");
    }

    #[test]
    fn test_cross_asset_aggregate() {
        let mut signals = CrossAssetSignals::for_altcoin("SOL");

        // Record some BTC returns
        let now_ms = 1000000;
        signals.update_btc_return(now_ms - 100, 5.0);
        signals.update_btc_return(now_ms - 200, 3.0);

        // Update funding
        for _ in 0..20 {
            signals.update_funding(0.0001);
        }

        let signal = signals.aggregate_signal(0.0001);

        // Should have some signal
        println!(
            "Aggregate signal: move={:.2}bp, confidence={:.2}",
            signal.expected_move_bps, signal.confidence
        );
    }
}
