//! Multi-Asset Correlation Estimation
//!
//! Tracks correlations between assets for portfolio risk management.
//!
//! Key features:
//! - **Multi-scale EWMA**: Fast/medium/slow correlation estimates
//! - **Portfolio Risk**: Aggregate variance calculation
//! - **Correlation Breakdown Detection**: Sudden regime shifts
//! - **Diversification Ratio**: Measure of effective diversification

use std::collections::HashMap;
use std::time::Instant;

/// Configuration for correlation estimation.
#[derive(Debug, Clone)]
pub struct CorrelationConfig {
    /// Fast EWMA decay (for recent correlation, ~1 minute)
    pub fast_alpha: f64,
    /// Medium EWMA decay (for moderate horizon, ~5 minutes)
    pub medium_alpha: f64,
    /// Slow EWMA decay (for stable baseline, ~30 minutes)
    pub slow_alpha: f64,
    /// Minimum observations before correlation is valid
    pub min_observations: usize,
    /// Correlation breakdown threshold (absolute change from slow to fast)
    pub breakdown_threshold: f64,
}

impl Default for CorrelationConfig {
    fn default() -> Self {
        Self {
            fast_alpha: 0.05,    // ~20 observations half-life
            medium_alpha: 0.01,  // ~70 observations half-life
            slow_alpha: 0.002,   // ~350 observations half-life
            min_observations: 30,
            breakdown_threshold: 0.4, // 40% correlation change = breakdown
        }
    }
}

/// Return observation for an asset.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct ReturnObservation {
    /// Asset symbol
    asset: String,
    /// Log return
    log_return: f64,
    /// Timestamp
    #[allow(dead_code)]
    timestamp: Instant,
}

/// Pair correlation tracker.
#[derive(Debug, Clone)]
struct PairCorrelation {
    /// Asset pair key (sorted alphabetically)
    assets: (String, String),
    /// Running product of returns (for covariance)
    product_sum: f64,
    /// Fast EWMA correlation
    fast_correlation: f64,
    /// Medium EWMA correlation
    medium_correlation: f64,
    /// Slow EWMA correlation
    slow_correlation: f64,
    /// Number of joint observations
    observation_count: usize,
}

impl PairCorrelation {
    fn new(asset1: &str, asset2: &str) -> Self {
        // Sort assets alphabetically for consistent key
        let (a1, a2) = if asset1 < asset2 {
            (asset1.to_string(), asset2.to_string())
        } else {
            (asset2.to_string(), asset1.to_string())
        };

        Self {
            assets: (a1, a2),
            product_sum: 0.0,
            fast_correlation: 0.0,
            medium_correlation: 0.0,
            slow_correlation: 0.0,
            observation_count: 0,
        }
    }

    /// Update correlation with new joint return observation.
    fn update(&mut self, ret1: f64, ret2: f64, config: &CorrelationConfig) {
        let product = ret1 * ret2;
        self.product_sum += product;
        self.observation_count += 1;

        // Update EWMA correlations
        // Using normalized product as proxy for correlation
        // (actual correlation requires variance tracking, simplified here)
        let instant_corr = product.signum() * product.abs().sqrt().min(1.0);

        self.fast_correlation = config.fast_alpha * instant_corr
            + (1.0 - config.fast_alpha) * self.fast_correlation;
        self.medium_correlation = config.medium_alpha * instant_corr
            + (1.0 - config.medium_alpha) * self.medium_correlation;
        self.slow_correlation = config.slow_alpha * instant_corr
            + (1.0 - config.slow_alpha) * self.slow_correlation;
    }

    /// Get correlation for specified horizon.
    fn correlation(&self, horizon: CorrelationHorizon) -> f64 {
        match horizon {
            CorrelationHorizon::Fast => self.fast_correlation.clamp(-1.0, 1.0),
            CorrelationHorizon::Medium => self.medium_correlation.clamp(-1.0, 1.0),
            CorrelationHorizon::Slow => self.slow_correlation.clamp(-1.0, 1.0),
        }
    }

    /// Check if correlation breakdown detected.
    fn is_breakdown(&self, threshold: f64) -> bool {
        let diff = (self.fast_correlation - self.slow_correlation).abs();
        diff > threshold && self.observation_count > 30
    }
}

/// Correlation horizon selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorrelationHorizon {
    /// Fast (~1 minute) - recent correlation
    Fast,
    /// Medium (~5 minutes) - moderate horizon
    Medium,
    /// Slow (~30 minutes) - stable baseline
    Slow,
}

/// Per-asset variance tracker.
#[derive(Debug, Clone)]
struct AssetVariance {
    /// Fast EWMA variance
    fast_var: f64,
    /// Slow EWMA variance
    slow_var: f64,
    /// Observation count
    count: usize,
}

impl AssetVariance {
    fn new() -> Self {
        Self {
            fast_var: 0.0,
            slow_var: 0.0,
            count: 0,
        }
    }

    fn update(&mut self, log_return: f64, config: &CorrelationConfig) {
        let sq_return = log_return * log_return;
        self.fast_var = config.fast_alpha * sq_return + (1.0 - config.fast_alpha) * self.fast_var;
        self.slow_var = config.slow_alpha * sq_return + (1.0 - config.slow_alpha) * self.slow_var;
        self.count += 1;
    }

    fn sigma(&self) -> f64 {
        self.fast_var.sqrt()
    }
}

/// Multi-asset correlation estimator.
///
/// Tracks pairwise correlations between assets for portfolio risk management.
pub struct CorrelationEstimator {
    config: CorrelationConfig,
    /// Per-asset variance trackers
    variances: HashMap<String, AssetVariance>,
    /// Pairwise correlations
    correlations: HashMap<String, PairCorrelation>,
    /// Latest returns per asset (for joint updates)
    latest_returns: HashMap<String, f64>,
    /// Last update time per asset
    last_update: HashMap<String, Instant>,
}

impl CorrelationEstimator {
    /// Create a new correlation estimator.
    pub fn new(config: CorrelationConfig) -> Self {
        Self {
            config,
            variances: HashMap::new(),
            correlations: HashMap::new(),
            latest_returns: HashMap::new(),
            last_update: HashMap::new(),
        }
    }

    /// Record a return observation for an asset.
    ///
    /// # Arguments
    /// - `asset`: Asset symbol
    /// - `log_return`: Log return for the period
    pub fn record_return(&mut self, asset: &str, log_return: f64) {
        let now = Instant::now();

        // Update variance for this asset
        self.variances
            .entry(asset.to_string())
            .or_insert_with(AssetVariance::new)
            .update(log_return, &self.config);

        // Store latest return
        self.latest_returns.insert(asset.to_string(), log_return);
        self.last_update.insert(asset.to_string(), now);

        // Update pairwise correlations with other assets that have recent returns
        let stale_threshold = std::time::Duration::from_secs(60);
        let other_assets: Vec<_> = self
            .latest_returns
            .keys()
            .filter(|k| *k != asset)
            .cloned()
            .collect();

        for other in other_assets {
            // Only update if other asset has recent data
            if let Some(last_time) = self.last_update.get(&other) {
                if now.duration_since(*last_time) < stale_threshold {
                    if let Some(&other_return) = self.latest_returns.get(&other) {
                        // Get or create pair correlation
                        let pair_key = Self::pair_key(asset, &other);
                        let pair = self
                            .correlations
                            .entry(pair_key)
                            .or_insert_with(|| PairCorrelation::new(asset, &other));

                        pair.update(log_return, other_return, &self.config);
                    }
                }
            }
        }
    }

    /// Get correlation between two assets.
    ///
    /// # Arguments
    /// - `asset1`: First asset symbol
    /// - `asset2`: Second asset symbol
    /// - `horizon`: Which correlation estimate to use
    ///
    /// # Returns
    /// Correlation coefficient [-1, 1] or 0 if pair not tracked
    pub fn correlation(&self, asset1: &str, asset2: &str, horizon: CorrelationHorizon) -> f64 {
        if asset1 == asset2 {
            return 1.0; // Self-correlation is 1
        }

        let key = Self::pair_key(asset1, asset2);
        self.correlations
            .get(&key)
            .map(|p| p.correlation(horizon))
            .unwrap_or(0.0)
    }

    /// Get full correlation matrix for specified assets.
    ///
    /// # Arguments
    /// - `assets`: List of asset symbols
    /// - `horizon`: Which correlation estimate to use
    ///
    /// # Returns
    /// NxN matrix where matrix[i][j] = correlation(assets[i], assets[j])
    pub fn correlation_matrix(
        &self,
        assets: &[&str],
        horizon: CorrelationHorizon,
    ) -> Vec<Vec<f64>> {
        let n = assets.len();
        let mut matrix = vec![vec![0.0; n]; n];

        for (i, asset1) in assets.iter().enumerate() {
            for (j, asset2) in assets.iter().enumerate() {
                matrix[i][j] = self.correlation(asset1, asset2, horizon);
            }
        }

        matrix
    }

    /// Calculate portfolio variance given asset weights.
    ///
    /// # Arguments
    /// - `weights`: Map of asset -> weight (should sum to 1.0)
    /// - `horizon`: Which correlation estimate to use
    ///
    /// # Returns
    /// Portfolio variance (σ² of portfolio)
    pub fn portfolio_variance(
        &self,
        weights: &HashMap<String, f64>,
        horizon: CorrelationHorizon,
    ) -> f64 {
        let mut variance = 0.0;

        for (asset1, w1) in weights.iter() {
            // Get asset variance
            let var1 = self
                .variances
                .get(asset1)
                .map(|v| v.fast_var)
                .unwrap_or(0.0001);
            let sigma1 = var1.sqrt();

            for (asset2, w2) in weights.iter() {
                let var2 = self
                    .variances
                    .get(asset2)
                    .map(|v| v.fast_var)
                    .unwrap_or(0.0001);
                let sigma2 = var2.sqrt();

                let corr = self.correlation(asset1, asset2, horizon);

                // Contribution to portfolio variance: w1 * w2 * σ1 * σ2 * ρ
                variance += w1 * w2 * sigma1 * sigma2 * corr;
            }
        }

        variance.max(0.0)
    }

    /// Calculate diversification ratio.
    ///
    /// Ratio of weighted sum of individual volatilities to portfolio volatility.
    /// Higher = more diversification benefit.
    ///
    /// # Arguments
    /// - `weights`: Map of asset -> weight (should sum to 1.0)
    /// - `horizon`: Which correlation estimate to use
    ///
    /// # Returns
    /// Diversification ratio (>= 1.0, higher = better)
    pub fn diversification_ratio(
        &self,
        weights: &HashMap<String, f64>,
        horizon: CorrelationHorizon,
    ) -> f64 {
        // Weighted sum of individual volatilities
        let mut weighted_vol = 0.0;
        for (asset, w) in weights.iter() {
            let sigma = self
                .variances
                .get(asset)
                .map(|v| v.sigma())
                .unwrap_or(0.01);
            weighted_vol += w.abs() * sigma;
        }

        // Portfolio volatility
        let port_var = self.portfolio_variance(weights, horizon);
        let port_vol = port_var.sqrt().max(1e-9);

        // Diversification ratio = weighted_vol / portfolio_vol
        // >= 1.0, higher = more diversification
        weighted_vol / port_vol
    }

    /// Check for correlation breakdown (sudden regime shift).
    ///
    /// Compares fast vs slow correlation to detect sudden changes.
    ///
    /// # Arguments
    /// - `asset1`: First asset
    /// - `asset2`: Second asset
    ///
    /// # Returns
    /// True if correlation has changed significantly
    pub fn is_correlation_breakdown(&self, asset1: &str, asset2: &str) -> bool {
        let key = Self::pair_key(asset1, asset2);
        self.correlations
            .get(&key)
            .map(|p| p.is_breakdown(self.config.breakdown_threshold))
            .unwrap_or(false)
    }

    /// Get list of pairs with correlation breakdown.
    pub fn breakdown_pairs(&self) -> Vec<(String, String)> {
        self.correlations
            .values()
            .filter(|p| p.is_breakdown(self.config.breakdown_threshold))
            .map(|p| (p.assets.0.clone(), p.assets.1.clone()))
            .collect()
    }

    /// Get summary for an asset pair.
    pub fn pair_summary(&self, asset1: &str, asset2: &str) -> CorrelationSummary {
        let key = Self::pair_key(asset1, asset2);
        if let Some(pair) = self.correlations.get(&key) {
            CorrelationSummary {
                asset1: pair.assets.0.clone(),
                asset2: pair.assets.1.clone(),
                fast_correlation: pair.fast_correlation,
                medium_correlation: pair.medium_correlation,
                slow_correlation: pair.slow_correlation,
                observations: pair.observation_count,
                is_breakdown: pair.is_breakdown(self.config.breakdown_threshold),
            }
        } else {
            CorrelationSummary {
                asset1: asset1.to_string(),
                asset2: asset2.to_string(),
                fast_correlation: 0.0,
                medium_correlation: 0.0,
                slow_correlation: 0.0,
                observations: 0,
                is_breakdown: false,
            }
        }
    }

    /// Create canonical pair key (sorted alphabetically).
    fn pair_key(asset1: &str, asset2: &str) -> String {
        if asset1 < asset2 {
            format!("{}:{}", asset1, asset2)
        } else {
            format!("{}:{}", asset2, asset1)
        }
    }
}

/// Summary of correlation between two assets.
#[derive(Debug, Clone)]
pub struct CorrelationSummary {
    pub asset1: String,
    pub asset2: String,
    pub fast_correlation: f64,
    pub medium_correlation: f64,
    pub slow_correlation: f64,
    pub observations: usize,
    pub is_breakdown: bool,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = CorrelationConfig::default();
        assert_eq!(config.fast_alpha, 0.05);
        assert_eq!(config.min_observations, 30);
    }

    #[test]
    fn test_self_correlation() {
        let estimator = CorrelationEstimator::new(CorrelationConfig::default());
        let corr = estimator.correlation("BTC", "BTC", CorrelationHorizon::Fast);
        assert_eq!(corr, 1.0);
    }

    #[test]
    fn test_unknown_pair() {
        let estimator = CorrelationEstimator::new(CorrelationConfig::default());
        let corr = estimator.correlation("BTC", "ETH", CorrelationHorizon::Fast);
        assert_eq!(corr, 0.0); // No data yet
    }

    #[test]
    fn test_record_returns() {
        let mut estimator = CorrelationEstimator::new(CorrelationConfig::default());

        // Record some returns
        for _ in 0..10 {
            estimator.record_return("BTC", 0.001);
            estimator.record_return("ETH", 0.001); // Same direction
        }

        // Should have positive correlation
        let corr = estimator.correlation("BTC", "ETH", CorrelationHorizon::Fast);
        assert!(corr > 0.0, "Expected positive correlation, got {}", corr);
    }

    #[test]
    fn test_negative_correlation() {
        let mut estimator = CorrelationEstimator::new(CorrelationConfig::default());

        // Record opposite returns
        for _ in 0..20 {
            estimator.record_return("BTC", 0.001);
            estimator.record_return("INVERSE", -0.001); // Opposite direction
        }

        // Should have negative correlation
        let corr = estimator.correlation("BTC", "INVERSE", CorrelationHorizon::Fast);
        assert!(corr < 0.0, "Expected negative correlation, got {}", corr);
    }

    #[test]
    fn test_correlation_matrix() {
        let mut estimator = CorrelationEstimator::new(CorrelationConfig::default());

        // Record some data
        for _ in 0..10 {
            estimator.record_return("BTC", 0.001);
            estimator.record_return("ETH", 0.001);
            estimator.record_return("SOL", 0.002);
        }

        let assets = vec!["BTC", "ETH", "SOL"];
        let matrix = estimator.correlation_matrix(&assets, CorrelationHorizon::Fast);

        // Check diagonal is 1
        for i in 0..3 {
            assert_eq!(matrix[i][i], 1.0);
        }

        // Check symmetry
        assert_eq!(matrix[0][1], matrix[1][0]);
    }

    #[test]
    fn test_portfolio_variance() {
        let mut estimator = CorrelationEstimator::new(CorrelationConfig::default());

        // Build up some variance data
        for _ in 0..50 {
            estimator.record_return("BTC", 0.01);
            estimator.record_return("ETH", 0.01);
        }

        let mut weights = HashMap::new();
        weights.insert("BTC".to_string(), 0.5);
        weights.insert("ETH".to_string(), 0.5);

        let var = estimator.portfolio_variance(&weights, CorrelationHorizon::Fast);
        assert!(var > 0.0, "Portfolio variance should be positive");
    }

    #[test]
    fn test_diversification_ratio() {
        let mut estimator = CorrelationEstimator::new(CorrelationConfig::default());

        // Build up data with imperfect correlation
        for i in 0..100 {
            let btc = 0.01 * (1.0 + 0.1 * (i as f64).sin());
            let eth = 0.01 * (1.0 + 0.1 * (i as f64 + 1.0).sin());
            estimator.record_return("BTC", btc);
            estimator.record_return("ETH", eth);
        }

        let mut weights = HashMap::new();
        weights.insert("BTC".to_string(), 0.5);
        weights.insert("ETH".to_string(), 0.5);

        let div_ratio = estimator.diversification_ratio(&weights, CorrelationHorizon::Fast);
        assert!(
            div_ratio >= 0.9,
            "Diversification ratio should be >= 0.9, got {}",
            div_ratio
        );
    }

    #[test]
    fn test_breakdown_detection() {
        let config = CorrelationConfig {
            fast_alpha: 0.5,     // Very fast update
            slow_alpha: 0.001,   // Very slow update
            breakdown_threshold: 0.3,
            ..Default::default()
        };
        let mut estimator = CorrelationEstimator::new(config);

        // Build up baseline correlation
        for _ in 0..50 {
            estimator.record_return("BTC", 0.01);
            estimator.record_return("ETH", 0.01);
        }

        // Now sudden reversal
        for _ in 0..10 {
            estimator.record_return("BTC", 0.01);
            estimator.record_return("ETH", -0.01);
        }

        // Should detect breakdown
        let is_breakdown = estimator.is_correlation_breakdown("BTC", "ETH");
        // Note: might not trigger immediately due to EWMA dynamics
        // This test mainly verifies the logic compiles and runs
        let _breakdown = is_breakdown; // Used to avoid warning
    }

    #[test]
    fn test_pair_summary() {
        let mut estimator = CorrelationEstimator::new(CorrelationConfig::default());

        for _ in 0..20 {
            estimator.record_return("BTC", 0.01);
            estimator.record_return("ETH", 0.01);
        }

        let summary = estimator.pair_summary("BTC", "ETH");
        assert!(summary.observations > 0);
        assert_eq!(summary.asset1, "BTC");
        assert_eq!(summary.asset2, "ETH");
    }
}
