//! A/B testing framework for solver comparison.
//!
//! Enables rigorous comparison of control solvers (e.g., GLFT vs custom)
//! using statistical tests to determine which performs better.
//!
//! ## Key Features
//!
//! - **Random allocation**: Deterministic PRNG for reproducible tests
//! - **Metrics tracking**: PnL, fill rate, adverse selection
//! - **Statistical significance**: Z-test for proportions
//! - **Auto-promotion**: Optionally promote winner automatically
//!
//! ## Usage
//!
//! ```ignore
//! let config = ABTestConfig {
//!     name: "glft_vs_custom".to_string(),
//!     treatment_allocation: 0.1,  // 10% to treatment
//!     min_samples: 100,
//!     auto_promote: true,
//!     auto_promote_threshold: 0.95,
//! };
//!
//! let test = ABTest::new(config, now_ms);
//!
//! // For each quote cycle
//! match test.allocate() {
//!     ABVariant::Control => { /* use control solver */ }
//!     ABVariant::Treatment => { /* use treatment solver */ }
//! }
//!
//! // After trade
//! test.record_control_trade(pnl, filled, adverse);
//! ```

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Aggregated metrics for A/B test comparison.
#[derive(Debug, Clone, Default)]
pub struct ABMetrics {
    /// Number of trades in control group
    pub control_trades: usize,
    /// Number of trades in treatment group
    pub treatment_trades: usize,
    /// Total PnL for control group (bps)
    pub control_pnl: f64,
    /// Total PnL for treatment group (bps)
    pub treatment_pnl: f64,
    /// Fill rate for control group
    pub control_fill_rate: f64,
    /// Fill rate for treatment group
    pub treatment_fill_rate: f64,
    /// Adverse selection rate for control group
    pub control_adverse_rate: f64,
    /// Adverse selection rate for treatment group
    pub treatment_adverse_rate: f64,

    // Internal tracking
    control_fills: usize,
    treatment_fills: usize,
    control_quotes: usize,
    treatment_quotes: usize,
    control_adverse: usize,
    treatment_adverse: usize,
}

impl ABMetrics {
    /// Calculate PnL difference (treatment - control).
    pub fn pnl_difference(&self) -> f64 {
        let control_avg = if self.control_trades > 0 {
            self.control_pnl / self.control_trades as f64
        } else {
            0.0
        };

        let treatment_avg = if self.treatment_trades > 0 {
            self.treatment_pnl / self.treatment_trades as f64
        } else {
            0.0
        };

        treatment_avg - control_avg
    }

    /// Calculate PnL difference as percentage.
    pub fn pnl_difference_pct(&self) -> f64 {
        let control_avg = if self.control_trades > 0 {
            self.control_pnl / self.control_trades as f64
        } else {
            return 0.0;
        };

        if control_avg.abs() < 1e-10 {
            return 0.0;
        }

        let treatment_avg = if self.treatment_trades > 0 {
            self.treatment_pnl / self.treatment_trades as f64
        } else {
            0.0
        };

        (treatment_avg - control_avg) / control_avg.abs() * 100.0
    }

    /// Calculate fill rate difference.
    pub fn fill_rate_difference(&self) -> f64 {
        self.treatment_fill_rate - self.control_fill_rate
    }

    /// Check if treatment is performing better.
    pub fn is_treatment_better(&self) -> bool {
        // Treatment is better if:
        // 1. Higher PnL per trade, OR
        // 2. Same PnL but lower adverse selection
        let pnl_diff = self.pnl_difference();

        if pnl_diff > 1e-6 {
            return true;
        }

        if pnl_diff.abs() < 1e-6 {
            // PnL is approximately equal, prefer lower adverse selection
            return self.treatment_adverse_rate < self.control_adverse_rate;
        }

        false
    }

    /// Calculate statistical significance using two-proportion z-test.
    ///
    /// Returns p-value for the hypothesis that treatment PnL is greater than control.
    pub fn statistical_significance(&self) -> f64 {
        let n1 = self.control_trades as f64;
        let n2 = self.treatment_trades as f64;

        if n1 < 10.0 || n2 < 10.0 {
            return 0.0; // Not enough samples
        }

        // Use average PnL per trade as the test statistic
        let p1 = if n1 > 0.0 { self.control_pnl / n1 } else { 0.0 };
        let p2 = if n2 > 0.0 {
            self.treatment_pnl / n2
        } else {
            0.0
        };

        // Estimate pooled variance
        // For PnL, we use a simple approximation assuming similar variance
        let var1 = self.estimate_variance(true);
        let var2 = self.estimate_variance(false);

        let se = ((var1 / n1) + (var2 / n2)).sqrt();

        if se < 1e-10 {
            return 0.0;
        }

        // Z-score for one-sided test (treatment > control)
        let z = (p2 - p1) / se;

        // Convert to p-value using normal CDF approximation
        // 1 - CDF(z) for one-sided test
        1.0 - normal_cdf(z)
    }

    /// Estimate variance for a group.
    fn estimate_variance(&self, is_control: bool) -> f64 {
        // Simple variance estimate based on fill rate variability
        // In practice, would track sum of squares for proper calculation
        let fill_rate = if is_control {
            self.control_fill_rate
        } else {
            self.treatment_fill_rate
        };

        // Variance of Bernoulli for fill rate
        // Use fill rate as proxy for overall volatility
        fill_rate * (1.0 - fill_rate) + 1.0 // Add 1.0 for baseline variance in bps
    }

    /// Calculate minimum samples needed for desired statistical power.
    ///
    /// # Arguments
    /// * `desired_power` - Statistical power (e.g., 0.8 for 80%)
    ///
    /// # Returns
    /// Minimum number of samples per group
    pub fn min_samples_for_significance(&self, desired_power: f64) -> usize {
        // Using simplified formula for two-sample test
        // n = 2 * ((z_alpha + z_beta) / effect_size)^2
        //
        // Where:
        // - z_alpha = 1.96 for 95% confidence
        // - z_beta = 0.84 for 80% power
        // - effect_size = expected difference / pooled std

        let z_alpha = 1.96;
        let z_beta = z_score_from_power(desired_power);

        // Estimate effect size from current data
        let effect_size = if self.control_trades > 0 && self.treatment_trades > 0 {
            let diff = self.pnl_difference().abs();
            let pooled_std = ((self.estimate_variance(true) + self.estimate_variance(false)) / 2.0)
                .sqrt()
                .max(0.1);
            diff / pooled_std
        } else {
            0.2 // Default small effect size
        };

        let effect_size = effect_size.max(0.1); // Minimum effect size

        let n = 2.0 * ((z_alpha + z_beta) / effect_size).powi(2);
        n.ceil() as usize
    }

    /// Record a trade for internal bookkeeping.
    fn record_trade(&mut self, is_control: bool, pnl: f64, filled: bool, adverse: bool) {
        if is_control {
            self.control_quotes += 1;
            if filled {
                self.control_trades += 1;
                self.control_pnl += pnl;
                self.control_fills += 1;
                if adverse {
                    self.control_adverse += 1;
                }
            }
            self.control_fill_rate = if self.control_quotes > 0 {
                self.control_fills as f64 / self.control_quotes as f64
            } else {
                0.0
            };
            self.control_adverse_rate = if self.control_fills > 0 {
                self.control_adverse as f64 / self.control_fills as f64
            } else {
                0.0
            };
        } else {
            self.treatment_quotes += 1;
            if filled {
                self.treatment_trades += 1;
                self.treatment_pnl += pnl;
                self.treatment_fills += 1;
                if adverse {
                    self.treatment_adverse += 1;
                }
            }
            self.treatment_fill_rate = if self.treatment_quotes > 0 {
                self.treatment_fills as f64 / self.treatment_quotes as f64
            } else {
                0.0
            };
            self.treatment_adverse_rate = if self.treatment_fills > 0 {
                self.treatment_adverse as f64 / self.treatment_fills as f64
            } else {
                0.0
            };
        }
    }
}

/// A/B test variant assignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ABVariant {
    /// Control group (baseline solver)
    Control,
    /// Treatment group (new solver)
    Treatment,
}

/// Configuration for an A/B test.
#[derive(Debug, Clone)]
pub struct ABTestConfig {
    /// Test name for identification
    pub name: String,
    /// Fraction of traffic allocated to treatment (0.0 to 1.0)
    pub treatment_allocation: f64,
    /// Minimum samples before drawing conclusions
    pub min_samples: usize,
    /// Whether to auto-promote treatment if it wins
    pub auto_promote: bool,
    /// Required significance level for auto-promotion
    pub auto_promote_threshold: f64,
}

impl Default for ABTestConfig {
    fn default() -> Self {
        Self {
            name: "unnamed".to_string(),
            treatment_allocation: 0.1,
            min_samples: 100,
            auto_promote: false,
            auto_promote_threshold: 0.95,
        }
    }
}

/// An active A/B test.
///
/// Thread-safe (Send + Sync) for use in async contexts.
pub struct ABTest {
    /// Test configuration
    config: ABTestConfig,
    /// Aggregated metrics
    metrics: Arc<RwLock<ABMetrics>>,
    /// PRNG state for deterministic allocation
    rng_state: Arc<RwLock<u64>>,
    /// Test start time (epoch ms)
    started_at: u64,
    /// Whether test has concluded
    concluded: Arc<RwLock<bool>>,
    /// Winner if concluded
    winner: Arc<RwLock<Option<ABVariant>>>,
}

// Safety: ABTest only contains Arc<RwLock<_>> which are Send + Sync
unsafe impl Send for ABTest {}
unsafe impl Sync for ABTest {}

impl ABTest {
    /// Create a new A/B test.
    ///
    /// # Arguments
    /// * `config` - Test configuration
    /// * `start_time` - Epoch milliseconds
    pub fn new(config: ABTestConfig, start_time: u64) -> Self {
        Self {
            config,
            metrics: Arc::new(RwLock::new(ABMetrics::default())),
            rng_state: Arc::new(RwLock::new(start_time)),
            started_at: start_time,
            concluded: Arc::new(RwLock::new(false)),
            winner: Arc::new(RwLock::new(None)),
        }
    }

    /// Allocate to a variant using deterministic PRNG.
    ///
    /// Uses a simple LCG (Linear Congruential Generator) for reproducibility.
    pub fn allocate(&self) -> ABVariant {
        if *self.concluded.read().unwrap() {
            // If concluded, always return winner
            return self.winner.read().unwrap().unwrap_or(ABVariant::Control);
        }

        // Simple LCG: state = (a * state + c) mod m
        // Using parameters from Numerical Recipes
        const A: u64 = 1664525;
        const C: u64 = 1013904223;

        let mut state = self.rng_state.write().unwrap();
        *state = state.wrapping_mul(A).wrapping_add(C);

        // Convert to [0, 1)
        let random = (*state as f64) / (u64::MAX as f64);

        if random < self.config.treatment_allocation {
            ABVariant::Treatment
        } else {
            ABVariant::Control
        }
    }

    /// Record a control group trade.
    ///
    /// # Arguments
    /// * `pnl` - PnL in basis points
    /// * `filled` - Whether the quote was filled
    /// * `adverse` - Whether adverse selection occurred
    pub fn record_control_trade(&self, pnl: f64, filled: bool, adverse: bool) {
        if !*self.concluded.read().unwrap() {
            self.metrics
                .write()
                .unwrap()
                .record_trade(true, pnl, filled, adverse);
            self.check_auto_promote();
        }
    }

    /// Record a treatment group trade.
    ///
    /// # Arguments
    /// * `pnl` - PnL in basis points
    /// * `filled` - Whether the quote was filled
    /// * `adverse` - Whether adverse selection occurred
    pub fn record_treatment_trade(&self, pnl: f64, filled: bool, adverse: bool) {
        if !*self.concluded.read().unwrap() {
            self.metrics
                .write()
                .unwrap()
                .record_trade(false, pnl, filled, adverse);
            self.check_auto_promote();
        }
    }

    /// Get current metrics snapshot.
    pub fn metrics(&self) -> ABMetrics {
        self.metrics.read().unwrap().clone()
    }

    /// Check if test has reached a conclusive result.
    pub fn is_conclusive(&self) -> bool {
        let metrics = self.metrics.read().unwrap();

        // Need minimum samples in both groups
        if metrics.control_trades < self.config.min_samples
            || metrics.treatment_trades < self.config.min_samples
        {
            return false;
        }

        // Check statistical significance
        let p_value = metrics.statistical_significance();
        p_value < 0.05 || p_value > 0.95
    }

    /// Get the winner if test is conclusive.
    pub fn winner(&self) -> Option<ABVariant> {
        *self.winner.read().unwrap()
    }

    /// Check if treatment should be promoted.
    pub fn should_promote_treatment(&self) -> bool {
        if !self.config.auto_promote {
            return false;
        }

        let metrics = self.metrics.read().unwrap();

        // Need minimum samples
        if metrics.control_trades < self.config.min_samples
            || metrics.treatment_trades < self.config.min_samples
        {
            return false;
        }

        // Check if treatment is better with sufficient significance
        // Note: p-value is 1 - CDF(z), so small p-value means treatment > control
        let p_value = metrics.statistical_significance();
        metrics.is_treatment_better() && p_value < (1.0 - self.config.auto_promote_threshold)
    }

    /// Conclude the test with a winner.
    pub fn conclude(&self, winner: ABVariant) {
        *self.concluded.write().unwrap() = true;
        *self.winner.write().unwrap() = Some(winner);
    }

    /// Check if test has concluded.
    pub fn is_concluded(&self) -> bool {
        *self.concluded.read().unwrap()
    }

    /// Get test duration in seconds.
    pub fn duration_s(&self, current_time: u64) -> u64 {
        (current_time.saturating_sub(self.started_at)) / 1000
    }

    /// Check for auto-promotion.
    fn check_auto_promote(&self) {
        if self.should_promote_treatment() {
            self.conclude(ABVariant::Treatment);
        }
    }

    /// Get test name.
    pub fn name(&self) -> &str {
        &self.config.name
    }
}

impl Clone for ABTest {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            metrics: Arc::new(RwLock::new(self.metrics.read().unwrap().clone())),
            rng_state: Arc::new(RwLock::new(*self.rng_state.read().unwrap())),
            started_at: self.started_at,
            concluded: Arc::new(RwLock::new(*self.concluded.read().unwrap())),
            winner: Arc::new(RwLock::new(*self.winner.read().unwrap())),
        }
    }
}

/// Manager for multiple concurrent A/B tests.
pub struct ABTestManager {
    /// Active tests by name
    active_tests: Arc<RwLock<HashMap<String, ABTest>>>,
    /// History of completed tests
    completed_tests: Arc<RwLock<Vec<(String, ABMetrics, ABVariant)>>>,
}

// Safety: ABTestManager only contains Arc<RwLock<_>> which are Send + Sync
unsafe impl Send for ABTestManager {}
unsafe impl Sync for ABTestManager {}

impl ABTestManager {
    /// Create a new test manager.
    pub fn new() -> Self {
        Self {
            active_tests: Arc::new(RwLock::new(HashMap::new())),
            completed_tests: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Create a new A/B test.
    ///
    /// # Returns
    /// Error if test with same name already exists.
    pub fn create_test(&self, config: ABTestConfig, start_time: u64) -> Result<(), String> {
        let mut tests = self.active_tests.write().unwrap();

        if tests.contains_key(&config.name) {
            return Err(format!("Test '{}' already exists", config.name));
        }

        let name = config.name.clone();
        let test = ABTest::new(config, start_time);
        tests.insert(name, test);
        Ok(())
    }

    /// Get a test by name.
    ///
    /// Returns a clone of the test for reading.
    pub fn get_test(&self, name: &str) -> Option<ABTest> {
        self.active_tests.read().unwrap().get(name).cloned()
    }

    /// Get names of all active tests.
    pub fn active_test_names(&self) -> Vec<String> {
        self.active_tests.read().unwrap().keys().cloned().collect()
    }

    /// Conclude a test and move to history.
    pub fn conclude_test(&self, name: &str, winner: ABVariant) -> Result<(), String> {
        let mut tests = self.active_tests.write().unwrap();

        if let Some(test) = tests.remove(name) {
            test.conclude(winner);
            let metrics = test.metrics();
            self.completed_tests
                .write()
                .unwrap()
                .push((name.to_string(), metrics, winner));
            Ok(())
        } else {
            Err(format!("Test '{}' not found", name))
        }
    }

    /// Get history of completed tests.
    pub fn test_history(&self) -> Vec<(String, ABMetrics, ABVariant)> {
        self.completed_tests.read().unwrap().clone()
    }
}

impl Default for ABTestManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Normal CDF approximation using error function.
fn normal_cdf(x: f64) -> f64 {
    // Approximation using erf
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Error function approximation.
fn erf(x: f64) -> f64 {
    // Abramowitz and Stegun approximation
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0
        - (a1 * t + a2 * t.powi(2) + a3 * t.powi(3) + a4 * t.powi(4) + a5 * t.powi(5))
            * (-x * x).exp();

    sign * y
}

/// Get z-score from desired power.
fn z_score_from_power(power: f64) -> f64 {
    // Approximate inverse normal CDF for common power values
    match power {
        p if p <= 0.5 => 0.0,
        p if p <= 0.6 => 0.25,
        p if p <= 0.7 => 0.52,
        p if p <= 0.8 => 0.84,
        p if p <= 0.9 => 1.28,
        p if p <= 0.95 => 1.64,
        p if p <= 0.99 => 2.33,
        _ => 2.58,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ab_metrics_default() {
        let metrics = ABMetrics::default();

        assert_eq!(metrics.control_trades, 0);
        assert_eq!(metrics.treatment_trades, 0);
        assert_eq!(metrics.pnl_difference(), 0.0);
    }

    #[test]
    fn test_ab_metrics_recording() {
        let mut metrics = ABMetrics::default();

        // Record control trades
        metrics.record_trade(true, 2.0, true, false);
        metrics.record_trade(true, 3.0, true, false);
        metrics.record_trade(true, -1.0, true, true);

        assert_eq!(metrics.control_trades, 3);
        assert_eq!(metrics.control_pnl, 4.0);
        assert_eq!(metrics.control_adverse, 1);

        // Record treatment trades
        metrics.record_trade(false, 5.0, true, false);
        metrics.record_trade(false, 4.0, true, false);

        assert_eq!(metrics.treatment_trades, 2);
        assert_eq!(metrics.treatment_pnl, 9.0);
    }

    #[test]
    fn test_pnl_difference() {
        let mut metrics = ABMetrics::default();

        // Control: avg 2.0 bps
        metrics.record_trade(true, 2.0, true, false);
        metrics.record_trade(true, 2.0, true, false);

        // Treatment: avg 3.0 bps
        metrics.record_trade(false, 3.0, true, false);
        metrics.record_trade(false, 3.0, true, false);

        let diff = metrics.pnl_difference();
        assert!((diff - 1.0).abs() < 1e-10); // 3.0 - 2.0 = 1.0
    }

    #[test]
    fn test_is_treatment_better() {
        let mut metrics = ABMetrics::default();

        // Treatment has higher PnL
        metrics.record_trade(true, 1.0, true, false);
        metrics.record_trade(false, 2.0, true, false);

        assert!(metrics.is_treatment_better());
    }

    #[test]
    fn test_ab_test_allocation() {
        let config = ABTestConfig {
            name: "test".to_string(),
            treatment_allocation: 0.5,
            min_samples: 10,
            ..Default::default()
        };

        let test = ABTest::new(config, 12345);

        // Run 1000 allocations
        let mut control_count = 0;
        let mut treatment_count = 0;

        for _ in 0..1000 {
            match test.allocate() {
                ABVariant::Control => control_count += 1,
                ABVariant::Treatment => treatment_count += 1,
            }
        }

        // With 50% allocation, should be roughly balanced
        // Allow 10% tolerance
        let ratio = treatment_count as f64 / (control_count + treatment_count) as f64;
        assert!(
            ratio > 0.4 && ratio < 0.6,
            "Allocation ratio {:.2} outside expected range",
            ratio
        );
    }

    #[test]
    fn test_ab_test_deterministic() {
        let config = ABTestConfig {
            name: "test".to_string(),
            treatment_allocation: 0.3,
            min_samples: 10,
            ..Default::default()
        };

        // Same seed should produce same sequence
        let test1 = ABTest::new(config.clone(), 42);
        let test2 = ABTest::new(config, 42);

        for _ in 0..100 {
            assert_eq!(test1.allocate(), test2.allocate());
        }
    }

    #[test]
    fn test_ab_test_conclude() {
        let config = ABTestConfig {
            name: "test".to_string(),
            ..Default::default()
        };

        let test = ABTest::new(config, 0);

        assert!(!test.is_concluded());
        assert_eq!(test.winner(), None);

        test.conclude(ABVariant::Treatment);

        assert!(test.is_concluded());
        assert_eq!(test.winner(), Some(ABVariant::Treatment));
    }

    #[test]
    fn test_ab_test_duration() {
        let test = ABTest::new(ABTestConfig::default(), 1000);

        assert_eq!(test.duration_s(5000), 4); // 4 seconds
        assert_eq!(test.duration_s(1000), 0); // 0 seconds
    }

    #[test]
    fn test_ab_manager_create_and_get() {
        let manager = ABTestManager::new();

        let config = ABTestConfig {
            name: "solver_comparison".to_string(),
            treatment_allocation: 0.1,
            min_samples: 50,
            ..Default::default()
        };

        // Create test
        assert!(manager.create_test(config.clone(), 0).is_ok());

        // Duplicate should fail
        assert!(manager.create_test(config, 0).is_err());

        // Get test
        let test = manager.get_test("solver_comparison");
        assert!(test.is_some());
        assert_eq!(test.unwrap().name(), "solver_comparison");
    }

    #[test]
    fn test_ab_manager_conclude() {
        let manager = ABTestManager::new();

        let config = ABTestConfig {
            name: "test1".to_string(),
            ..Default::default()
        };

        manager.create_test(config, 0).unwrap();

        // Conclude test
        assert!(manager.conclude_test("test1", ABVariant::Treatment).is_ok());

        // Should be in history now
        let history = manager.test_history();
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].0, "test1");
        assert_eq!(history[0].2, ABVariant::Treatment);

        // Should not be active
        assert!(manager.get_test("test1").is_none());
    }

    #[test]
    fn test_normal_cdf() {
        // Test known values
        assert!((normal_cdf(0.0) - 0.5).abs() < 0.01);
        assert!(normal_cdf(3.0) > 0.99);
        assert!(normal_cdf(-3.0) < 0.01);
    }

    #[test]
    fn test_statistical_significance() {
        let mut metrics = ABMetrics::default();

        // Not enough samples
        assert_eq!(metrics.statistical_significance(), 0.0);

        // Add enough samples
        for _ in 0..50 {
            metrics.record_trade(true, 1.0, true, false);
            metrics.record_trade(false, 2.0, true, false); // Treatment is better
        }

        let p_value = metrics.statistical_significance();
        // Treatment is clearly better, p-value should be small
        assert!(
            p_value < 0.5,
            "p-value {} should indicate treatment better",
            p_value
        );
    }

    #[test]
    fn test_min_samples_for_significance() {
        let metrics = ABMetrics::default();

        // Default effect size should require reasonable samples
        let min_samples = metrics.min_samples_for_significance(0.8);
        assert!(min_samples > 10);
        assert!(min_samples < 10000);
    }

    #[test]
    fn test_ab_test_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ABTest>();
        assert_send_sync::<ABTestManager>();
    }

    #[test]
    fn test_concluded_test_returns_winner() {
        let config = ABTestConfig {
            name: "test".to_string(),
            treatment_allocation: 0.5, // 50% split normally
            ..Default::default()
        };

        let test = ABTest::new(config, 0);
        test.conclude(ABVariant::Treatment);

        // After conclusion, should always return winner
        for _ in 0..100 {
            assert_eq!(test.allocate(), ABVariant::Treatment);
        }
    }
}
