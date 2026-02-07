//! Historical Calibrator - Batch calibration from logged data.
//!
//! Provides tools to calibrate parameters from historical fill data, trade logs,
//! and market snapshots. This complements the online learning in `parameter_learner.rs`
//! by allowing initial parameter estimation from historical data.
//!
//! # Usage
//!
//! ```rust,ignore
//! let calibrator = HistoricalCalibrator::new();
//!
//! // Load historical fills
//! calibrator.load_fills_from_log("logs/mm_hip3_2024-01-15.log")?;
//!
//! // Calibrate parameters
//! let learned = calibrator.calibrate()?;
//!
//! // Use calibrated parameters
//! let alpha = learned.alpha_touch.estimate();
//! ```

use super::parameter_learner::{BayesianParam, LearnedParameters};
use std::collections::HashMap;

/// Fill record for calibration.
#[derive(Debug, Clone)]
pub struct FillRecord {
    /// Timestamp in milliseconds
    pub timestamp_ms: u64,
    /// Fill price
    pub price: f64,
    /// Fill size
    pub size: f64,
    /// True if buy side
    pub is_buy: bool,
    /// Distance from mid-price in bps
    pub depth_bps: f64,
    /// Mid-price at time of fill
    pub mid_price: f64,
    /// Price 1 second after fill (for AS measurement)
    pub price_1s_later: Option<f64>,
    /// Price 5 seconds after fill
    pub price_5s_later: Option<f64>,
    /// Volatility regime at time of fill
    pub regime: Option<String>,
    /// Hour of day (UTC)
    pub hour_utc: u8,
}

/// Market snapshot for volatility/kappa calibration.
#[derive(Debug, Clone)]
pub struct MarketSnapshot {
    /// Timestamp in milliseconds
    pub timestamp_ms: u64,
    /// Mid-price
    pub mid_price: f64,
    /// Best bid
    pub bid: f64,
    /// Best ask
    pub ask: f64,
    /// Total bid size within 5 bps
    pub bid_depth_5bps: f64,
    /// Total ask size within 5 bps
    pub ask_depth_5bps: f64,
    /// Number of trades in last 1 second
    pub trade_count_1s: usize,
    /// Total trade volume in last 1 second
    pub trade_volume_1s: f64,
    /// Open interest
    pub open_interest: Option<f64>,
    /// Funding rate
    pub funding_rate: Option<f64>,
}

/// Trade record for Hawkes process calibration.
#[derive(Debug, Clone)]
pub struct TradeRecord {
    /// Timestamp in milliseconds
    pub timestamp_ms: u64,
    /// Trade price
    pub price: f64,
    /// Trade size
    pub size: f64,
    /// True if aggressor was buyer
    pub is_buy: bool,
}

/// Historical calibrator for batch parameter estimation.
#[derive(Debug, Default)]
pub struct HistoricalCalibrator {
    /// Fill records
    fills: Vec<FillRecord>,
    /// Market snapshots
    snapshots: Vec<MarketSnapshot>,
    /// Trade records
    trades: Vec<TradeRecord>,
    /// Regime-specific fill counts
    regime_fills: HashMap<String, (usize, usize)>, // (informed, uninformed)
    /// Hour-specific adverse selection
    hourly_as: HashMap<u8, Vec<f64>>,
}

impl HistoricalCalibrator {
    /// Create a new historical calibrator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a fill record.
    pub fn add_fill(&mut self, fill: FillRecord) {
        // Track regime-specific fills
        if let Some(regime) = &fill.regime {
            let entry = self.regime_fills.entry(regime.clone()).or_insert((0, 0));

            // Classify as informed if price moved adversely by >5 bps in 1 second
            if let Some(price_1s) = fill.price_1s_later {
                let adverse_move = if fill.is_buy {
                    (price_1s - fill.price) / fill.price * 10_000.0 // bps
                } else {
                    (fill.price - price_1s) / fill.price * 10_000.0
                };

                if adverse_move < -5.0 {
                    entry.0 += 1; // Informed
                } else {
                    entry.1 += 1; // Uninformed
                }
            }
        }

        // Track hourly adverse selection
        if let Some(price_1s) = fill.price_1s_later {
            let adverse_move = if fill.is_buy {
                (price_1s - fill.price) / fill.price * 10_000.0
            } else {
                (fill.price - price_1s) / fill.price * 10_000.0
            };
            self.hourly_as
                .entry(fill.hour_utc)
                .or_default()
                .push(adverse_move);
        }

        self.fills.push(fill);
    }

    /// Add a market snapshot.
    pub fn add_snapshot(&mut self, snapshot: MarketSnapshot) {
        self.snapshots.push(snapshot);
    }

    /// Add a trade record.
    pub fn add_trade(&mut self, trade: TradeRecord) {
        self.trades.push(trade);
    }

    /// Calibrate alpha_touch from fill data.
    ///
    /// Classifies fills as "informed" if price moved adversely by >5 bps within 1 second.
    pub fn calibrate_alpha_touch(&self) -> BayesianParam {
        let mut param = BayesianParam::beta("alpha_touch", 0.25, 8.0);

        let mut informed = 0;
        let mut uninformed = 0;

        for fill in &self.fills {
            if let Some(price_1s) = fill.price_1s_later {
                let adverse_move = if fill.is_buy {
                    (price_1s - fill.price) / fill.price * 10_000.0
                } else {
                    (fill.price - price_1s) / fill.price * 10_000.0
                };

                // Informed if adverse move > 5 bps
                if adverse_move < -5.0 {
                    informed += 1;
                } else {
                    uninformed += 1;
                }
            }
        }

        if informed + uninformed > 0 {
            param.observe_beta(informed, uninformed);
        }

        param
    }

    /// Calibrate kappa from market snapshots.
    ///
    /// Estimates fill intensity from observed trade rates and book depth.
    pub fn calibrate_kappa(&self) -> BayesianParam {
        let mut param = BayesianParam::gamma("kappa", 2000.0, 4.0);

        // Sort snapshots by time
        let mut snapshots = self.snapshots.clone();
        snapshots.sort_by_key(|s| s.timestamp_ms);

        // Calculate kappa = fills / spread for each snapshot
        for window in snapshots.windows(2) {
            let dt_s = (window[1].timestamp_ms - window[0].timestamp_ms) as f64 / 1000.0;
            if !(0.1..=10.0).contains(&dt_s) {
                continue; // Skip invalid windows
            }

            let spread_bps =
                (window[0].ask - window[0].bid) / window[0].mid_price * 10_000.0;
            if !(0.5..=100.0).contains(&spread_bps) {
                continue; // Skip invalid spreads
            }

            let fill_rate = window[0].trade_count_1s as f64 / dt_s;
            let kappa_obs = fill_rate / (spread_bps / 10_000.0);

            if kappa_obs > 100.0 && kappa_obs < 100_000.0 {
                param.observe_gamma_exponential(1.0 / kappa_obs);
            }
        }

        param
    }

    /// Calibrate Hawkes process parameters from trade data.
    ///
    /// Uses method of moments estimation for baseline μ, excitation α, and decay β.
    pub fn calibrate_hawkes(&self) -> (BayesianParam, BayesianParam, BayesianParam) {
        let mut mu = BayesianParam::gamma("hawkes_mu", 0.5, 5.0);
        let mut alpha = BayesianParam::beta("hawkes_alpha", 0.3, 10.0);
        let mut beta = BayesianParam::gamma("hawkes_beta", 0.1, 1.0);

        if self.trades.len() < 100 {
            return (mu, alpha, beta);
        }

        // Sort trades by time
        let mut trades = self.trades.clone();
        trades.sort_by_key(|t| t.timestamp_ms);

        // Calculate inter-arrival times
        let mut inter_arrivals: Vec<f64> = Vec::new();
        for i in 1..trades.len() {
            let dt = (trades[i].timestamp_ms - trades[i - 1].timestamp_ms) as f64 / 1000.0;
            if dt > 0.001 && dt < 60.0 {
                inter_arrivals.push(dt);
            }
        }

        if inter_arrivals.is_empty() {
            return (mu, alpha, beta);
        }

        // Method of moments: mean inter-arrival = 1 / (μ / (1 - α/β))
        let mean_ia = inter_arrivals.iter().sum::<f64>() / inter_arrivals.len() as f64;
        let var_ia = inter_arrivals
            .iter()
            .map(|x| (x - mean_ia).powi(2))
            .sum::<f64>()
            / inter_arrivals.len() as f64;

        // Estimate parameters from moments
        // λ_∞ = μ / (1 - α/β) = 1 / mean_ia
        let lambda_inf = 1.0 / mean_ia;

        // For Hawkes, variance of inter-arrivals depends on α, β
        // Simplified: if var >> mean², there's clustering (high α)
        let cv = var_ia.sqrt() / mean_ia;
        let alpha_est = (cv - 1.0).clamp(0.1, 0.8) / 2.0; // Rough heuristic

        // β should be > α for stability
        let beta_est = (alpha_est * 2.0).max(0.2);

        // μ = λ_∞ × (1 - α/β)
        let mu_est = lambda_inf * (1.0 - alpha_est / beta_est);

        // Update parameters with observations
        if mu_est > 0.01 && mu_est < 10.0 {
            mu.observe_gamma_exponential(1.0 / mu_est);
        }
        if alpha_est > 0.0 && alpha_est < 1.0 {
            alpha.observe_beta(
                (alpha_est * 100.0) as usize,
                ((1.0 - alpha_est) * 100.0) as usize,
            );
        }
        if beta_est > 0.01 && beta_est < 1.0 {
            beta.observe_gamma_exponential(1.0 / beta_est);
        }

        (mu, alpha, beta)
    }

    /// Calibrate toxic hour gamma multiplier.
    ///
    /// Identifies hours with significantly worse adverse selection and calculates
    /// the required gamma multiplier to maintain profitability.
    pub fn calibrate_toxic_hours(&self) -> (BayesianParam, Vec<u32>) {
        let mut gamma_mult = BayesianParam::log_normal("toxic_hour_gamma_mult", 2.0, 0.3);

        // Calculate average AS per hour
        let mut hourly_avg: HashMap<u8, f64> = HashMap::new();
        for (hour, as_values) in &self.hourly_as {
            if as_values.len() >= 10 {
                let avg = as_values.iter().sum::<f64>() / as_values.len() as f64;
                hourly_avg.insert(*hour, avg);
            }
        }

        if hourly_avg.is_empty() {
            return (gamma_mult, vec![6, 7, 14]); // Default toxic hours
        }

        // Find overall average AS
        let total_as: f64 = hourly_avg.values().sum();
        let overall_avg = total_as / hourly_avg.len() as f64;

        // Identify toxic hours (AS worse than overall by >50%)
        let mut toxic_hours: Vec<u32> = Vec::new();
        let mut toxic_as_sum = 0.0;
        let mut toxic_count = 0;

        for (hour, avg_as) in &hourly_avg {
            // AS is negative (adverse), so worse means more negative
            if *avg_as < overall_avg * 1.5 {
                toxic_hours.push(*hour as u32);
                toxic_as_sum += avg_as.abs();
                toxic_count += 1;
            }
        }

        // Calculate required gamma multiplier
        // gamma_mult = |toxic_AS| / |normal_AS|
        if toxic_count > 0 && overall_avg.abs() > 0.1 {
            let toxic_avg = toxic_as_sum / toxic_count as f64;
            let mult = (toxic_avg / overall_avg.abs()).clamp(1.5, 4.0);
            gamma_mult.observe_log_normal(mult);
        }

        toxic_hours.sort();
        (gamma_mult, toxic_hours)
    }

    /// Calibrate spread floor from historical fill data.
    ///
    /// Analyzes fill profitability at different spread levels to find optimal floor.
    pub fn calibrate_spread_floor(&self) -> BayesianParam {
        let mut param = BayesianParam::normal("spread_floor_bps", 5.0, 4.0);

        // Group fills by spread level
        let mut spread_pnl: HashMap<i32, Vec<f64>> = HashMap::new();

        for fill in &self.fills {
            if let Some(price_1s) = fill.price_1s_later {
                // P&L = spread captured - adverse move
                let spread_earned = fill.depth_bps / 2.0; // Half-spread earned
                let adverse_move = if fill.is_buy {
                    (price_1s - fill.price) / fill.price * 10_000.0
                } else {
                    (fill.price - price_1s) / fill.price * 10_000.0
                };

                let pnl = spread_earned + adverse_move; // adverse_move is usually negative

                // Bucket by spread level (0-5, 5-10, 10-15, etc.)
                let bucket = (fill.depth_bps / 5.0).floor() as i32 * 5;
                spread_pnl.entry(bucket).or_default().push(pnl);
            }
        }

        // Find minimum spread where average P&L is positive
        let mut sorted_buckets: Vec<_> = spread_pnl.keys().copied().collect();
        sorted_buckets.sort();

        let mut breakeven_spread = 5.0; // Default
        for bucket in sorted_buckets {
            if let Some(pnls) = spread_pnl.get(&bucket) {
                if pnls.len() >= 10 {
                    let avg_pnl = pnls.iter().sum::<f64>() / pnls.len() as f64;
                    if avg_pnl > 0.0 {
                        breakeven_spread = bucket as f64;
                        break;
                    }
                }
            }
        }

        // Add buffer of 1 bps above breakeven
        param.observe_normal(breakeven_spread + 1.0, 2.0);

        param
    }

    /// Calibrate cascade detection threshold from OI data.
    pub fn calibrate_cascade_threshold(&self) -> BayesianParam {
        let mut param = BayesianParam::beta("cascade_oi_threshold", 0.02, 100.0);

        // Look for large OI drops that preceded bad fills
        let mut oi_drops_before_bad: Vec<f64> = Vec::new();

        // Sort snapshots and fills by time
        let mut snapshots = self.snapshots.clone();
        snapshots.sort_by_key(|s| s.timestamp_ms);

        for i in 1..snapshots.len() {
            if let (Some(oi_prev), Some(oi_curr)) =
                (snapshots[i - 1].open_interest, snapshots[i].open_interest)
            {
                if oi_prev > 0.0 {
                    let oi_change = (oi_curr - oi_prev) / oi_prev;

                    // Check if fills in this period had bad outcomes
                    let ts_start = snapshots[i - 1].timestamp_ms;
                    let ts_end = snapshots[i].timestamp_ms;

                    let fills_in_period: Vec<_> = self
                        .fills
                        .iter()
                        .filter(|f| f.timestamp_ms >= ts_start && f.timestamp_ms <= ts_end)
                        .collect();

                    if !fills_in_period.is_empty() {
                        let avg_as: f64 = fills_in_period
                            .iter()
                            .filter_map(|f| f.price_1s_later.map(|p| {
                                if f.is_buy {
                                    (p - f.price) / f.price * 10_000.0
                                } else {
                                    (f.price - p) / f.price * 10_000.0
                                }
                            }))
                            .sum::<f64>()
                            / fills_in_period.len() as f64;

                        // Bad period if average AS < -10 bps
                        if avg_as < -10.0 && oi_change < 0.0 {
                            oi_drops_before_bad.push(oi_change.abs());
                        }
                    }
                }
            }
        }

        // Cascade threshold = 30th percentile of OI drops before bad periods
        if oi_drops_before_bad.len() >= 5 {
            oi_drops_before_bad.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let p30_idx = (oi_drops_before_bad.len() as f64 * 0.3) as usize;
            let threshold = oi_drops_before_bad[p30_idx].clamp(0.005, 0.10);

            // Convert to Beta observations
            let threshold_scaled = (threshold / 0.10).clamp(0.01, 0.99);
            param.observe_beta(
                (threshold_scaled * 100.0) as usize,
                ((1.0 - threshold_scaled) * 100.0) as usize,
            );
        }

        param
    }

    /// Calibrate regime transition stickiness from regime history.
    pub fn calibrate_regime_stickiness(&self, regime_history: &[String]) -> BayesianParam {
        let mut param = BayesianParam::beta("regime_sticky_diagonal", 0.95, 100.0);

        if regime_history.len() < 10 {
            return param;
        }

        // Count regime transitions
        let mut stays = 0;
        let mut transitions = 0;

        for i in 1..regime_history.len() {
            if regime_history[i] == regime_history[i - 1] {
                stays += 1;
            } else {
                transitions += 1;
            }
        }

        if stays + transitions > 0 {
            param.observe_beta(stays, transitions);
        }

        param
    }

    /// Run full calibration and return learned parameters.
    pub fn calibrate(&self) -> LearnedParameters {
        let mut params = LearnedParameters {
            alpha_touch: self.calibrate_alpha_touch(),
            spread_floor_bps: self.calibrate_spread_floor(),
            ..Default::default()
        };

        let (gamma_mult, _toxic_hours) = self.calibrate_toxic_hours();
        params.toxic_hour_gamma_mult = gamma_mult;

        // Tier 2: Risk
        params.cascade_oi_threshold = self.calibrate_cascade_threshold();

        // Tier 3: Calibration
        params.kappa = self.calibrate_kappa();
        let (hawkes_mu, hawkes_alpha, hawkes_beta) = self.calibrate_hawkes();
        params.hawkes_mu = hawkes_mu;
        params.hawkes_alpha = hawkes_alpha;
        params.hawkes_beta = hawkes_beta;

        params.total_fills_observed = self.fills.len();
        params.last_calibration = Some(std::time::Instant::now());

        params
    }

    /// Get calibration summary.
    pub fn summary(&self) -> CalibrationSummary {
        CalibrationSummary {
            n_fills: self.fills.len(),
            n_snapshots: self.snapshots.len(),
            n_trades: self.trades.len(),
            n_regimes: self.regime_fills.len(),
            hours_with_data: self.hourly_as.len(),
        }
    }
}

/// Summary of data available for calibration.
#[derive(Debug, Clone)]
pub struct CalibrationSummary {
    pub n_fills: usize,
    pub n_snapshots: usize,
    pub n_trades: usize,
    pub n_regimes: usize,
    pub hours_with_data: usize,
}

/// Power analysis for sample size requirements.
///
/// Calculates minimum samples needed for a given precision target.
pub struct PowerAnalysis;

impl PowerAnalysis {
    /// Minimum samples for a proportion estimate with given margin of error.
    ///
    /// For 95% CI width of 2×margin:
    /// n = (z² × p × (1-p)) / margin²
    ///
    /// # Arguments
    /// * `margin` - Desired margin of error (e.g., 0.1 for ±10%)
    /// * `prior_p` - Prior estimate of proportion (default 0.5 for max variance)
    pub fn samples_for_proportion(margin: f64, prior_p: f64) -> usize {
        let z = 1.96; // 95% confidence
        let p = prior_p.clamp(0.01, 0.99);
        let n = (z * z * p * (1.0 - p)) / (margin * margin);
        (n.ceil() as usize).max(10)
    }

    /// Minimum samples for mean estimate with given relative error.
    ///
    /// For 95% CI width of 2×margin×mean:
    /// n = (z × CV / margin)²
    ///
    /// # Arguments
    /// * `margin` - Desired relative margin of error (e.g., 0.2 for ±20%)
    /// * `cv` - Coefficient of variation (std/mean)
    pub fn samples_for_mean(margin: f64, cv: f64) -> usize {
        let z = 1.96;
        let n = (z * cv / margin).powi(2);
        (n.ceil() as usize).max(10)
    }

    /// Minimum samples for IR estimate to be reliable.
    ///
    /// IR needs enough samples to distinguish signal from noise.
    /// Rule of thumb: n >= 100 for IR CI width < 0.2
    pub fn samples_for_ir(target_ci_width: f64) -> usize {
        // IR standard error ≈ 1/√n (empirical approximation)
        // For CI width < target: 2 × 1.96 / √n < target
        // n > (2 × 1.96 / target)²
        let n = (2.0 * 1.96 / target_ci_width).powi(2);
        (n.ceil() as usize).max(50)
    }

    /// Check if sample size is sufficient for a given parameter type.
    pub fn is_sufficient(param_type: &str, n_samples: usize) -> bool {
        match param_type {
            "alpha_touch" => n_samples >= Self::samples_for_proportion(0.1, 0.25),
            "kappa" => n_samples >= Self::samples_for_mean(0.2, 0.5),
            "ir" => n_samples >= Self::samples_for_ir(0.2),
            "hawkes" => n_samples >= 100, // Method of moments needs ~100 samples
            "regime" => n_samples >= 50,  // HMM needs ~50 transitions
            _ => n_samples >= 30,         // Generic rule of thumb
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fill_classification() {
        let mut calibrator = HistoricalCalibrator::new();

        // Add an "informed" fill (price dropped after buy)
        calibrator.add_fill(FillRecord {
            timestamp_ms: 1000,
            price: 100.0,
            size: 1.0,
            is_buy: true,
            depth_bps: 5.0,
            mid_price: 100.0,
            price_1s_later: Some(99.9), // -10 bps adverse
            price_5s_later: None,
            regime: Some("calm".to_string()),
            hour_utc: 10,
        });

        // Add an "uninformed" fill (price stayed)
        calibrator.add_fill(FillRecord {
            timestamp_ms: 2000,
            price: 100.0,
            size: 1.0,
            is_buy: true,
            depth_bps: 5.0,
            mid_price: 100.0,
            price_1s_later: Some(100.01), // +1 bp favorable
            price_5s_later: None,
            regime: Some("calm".to_string()),
            hour_utc: 10,
        });

        let alpha = calibrator.calibrate_alpha_touch();
        // Should be between prior (0.25) and 50% (1 informed, 1 uninformed)
        assert!(alpha.estimate() > 0.25 && alpha.estimate() < 0.6);
    }

    #[test]
    fn test_power_analysis() {
        // For proportion with ±10% margin
        let n = PowerAnalysis::samples_for_proportion(0.1, 0.5);
        assert!(n >= 90 && n <= 100); // Should be ~96

        // For mean with ±20% margin and CV=0.5
        let n = PowerAnalysis::samples_for_mean(0.2, 0.5);
        assert!(n >= 20 && n <= 30); // Should be ~24

        // For IR
        let n = PowerAnalysis::samples_for_ir(0.2);
        assert!(n >= 380); // Should be ~384
    }

    #[test]
    fn test_sufficiency_checks() {
        assert!(!PowerAnalysis::is_sufficient("alpha_touch", 50));
        assert!(PowerAnalysis::is_sufficient("alpha_touch", 100));
        assert!(!PowerAnalysis::is_sufficient("ir", 50));
        assert!(PowerAnalysis::is_sufficient("ir", 400));
    }
}
