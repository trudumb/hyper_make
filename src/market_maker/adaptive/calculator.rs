//! Adaptive Spread Calculator - Component 5
//!
//! Orchestrates all adaptive components to produce the final optimal spread.
//!
//! # Final Spread Formula
//!
//! ```text
//! δ_final = max(δ_floor, min(δ_GLFT, δ_ceiling))
//! ```
//!
//! Where:
//! - δ_floor = learned floor from LearnedSpreadFloor (Component 1)
//! - δ_GLFT = (1/γ_eff) × ln(1 + γ_eff/κ_eff) + fees
//! - γ_eff = shrinkage gamma from ShrinkageGamma (Component 2)
//! - κ_eff = blended kappa from BlendedKappaEstimator (Component 3)
//! - δ_ceiling = fill rate ceiling from FillRateController (Component 4)

use tracing::{debug, info};

use super::blended_kappa::BlendedKappaEstimator;
use super::config::{AdaptiveBayesianConfig, GammaSignal};
use super::fill_controller::FillRateController;
use super::learned_floor::LearnedSpreadFloor;
use super::shrinkage_gamma::ShrinkageGamma;

/// Fill outcome for learning updates.
#[derive(Debug, Clone)]
pub struct FillOutcome {
    /// Fill price
    pub fill_price: f64,
    /// Mid price at time of fill
    pub mid_at_fill: f64,
    /// Mid price 1 second after fill (for AS measurement)
    pub mid_after_horizon: f64,
    /// Fill size
    pub size: f64,
    /// Fill direction: +1 for buy, -1 for sell
    pub direction: f64,
    /// Timestamp in milliseconds
    pub timestamp_ms: u64,
    /// PnL from this fill (if known)
    pub pnl: Option<f64>,
}

impl FillOutcome {
    /// Calculate realized adverse selection.
    pub fn realized_as(&self) -> f64 {
        // AS = (mid_after - fill_price) × direction
        // Positive = adverse (price moved against us)
        (self.mid_after_horizon - self.fill_price) * self.direction / self.mid_at_fill
    }

    /// Calculate fill distance from mid.
    pub fn fill_distance(&self) -> f64 {
        ((self.fill_price - self.mid_at_fill) / self.mid_at_fill).abs()
    }
}

/// Market signals for gamma calculation.
#[derive(Debug, Clone, Default)]
pub struct MarketSignals {
    /// Volatility ratio (σ / σ_baseline)
    pub vol_ratio: f64,
    /// Jump ratio (RV / BV)
    pub jump_ratio: f64,
    /// Inventory utilization (|q| / q_max)
    pub inventory_util: f64,
    /// Hawkes intensity percentile
    pub hawkes_percentile: f64,
    /// Spread regime indicator (-1 to 1)
    pub spread_regime: f64,
    /// Cascade severity (0 to 1)
    pub cascade_severity: f64,
}

impl MarketSignals {
    /// Convert to signal-value pairs for ShrinkageGamma.
    pub fn to_pairs(&self) -> Vec<(GammaSignal, f64)> {
        vec![
            (GammaSignal::VolatilityRatio, self.vol_ratio),
            (GammaSignal::JumpRatio, self.jump_ratio),
            (GammaSignal::InventoryUtilization, self.inventory_util),
            (GammaSignal::HawkesIntensity, self.hawkes_percentile),
            (GammaSignal::SpreadRegime, self.spread_regime),
            (GammaSignal::CascadeSeverity, self.cascade_severity),
        ]
    }
}

/// Adaptive spread calculator orchestrating all components.
#[derive(Debug, Clone)]
pub struct AdaptiveSpreadCalculator {
    /// Configuration
    config: AdaptiveBayesianConfig,

    /// Learned spread floor (Component 1)
    floor: LearnedSpreadFloor,

    /// Shrinkage gamma (Component 2)
    gamma: ShrinkageGamma,

    /// Blended kappa (Component 3)
    kappa: BlendedKappaEstimator,

    /// Fill rate controller (Component 4)
    fill_controller: FillRateController,

    /// Last calculated spread (for logging)
    last_spread: f64,

    /// Last gamma used (for learning feedback)
    last_gamma: f64,

    /// Last standardized signals (for learning feedback)
    last_signals: Vec<f64>,

    /// Quote cycle count (for logging)
    quote_count: u64,

    /// Time since last fill (seconds)
    time_since_last_fill: f64,
}

impl AdaptiveSpreadCalculator {
    /// Create a new adaptive spread calculator.
    pub fn new(config: AdaptiveBayesianConfig) -> Self {
        let floor = LearnedSpreadFloor::from_config(&config);
        let gamma = ShrinkageGamma::from_config(&config);
        let kappa = BlendedKappaEstimator::from_config(&config);
        let fill_controller = FillRateController::from_config(&config);

        Self {
            config,
            floor,
            gamma,
            kappa,
            fill_controller,
            last_spread: 0.0,
            last_gamma: 0.3,
            last_signals: Vec::new(),
            quote_count: 0,
            time_since_last_fill: 0.0,
        }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(AdaptiveBayesianConfig::default())
    }

    /// Calculate the optimal half-spread.
    ///
    /// # Arguments
    /// * `signals` - Current market signals
    ///
    /// # Returns
    /// Optimal half-spread as a fraction of price
    pub fn half_spread(&mut self, signals: &MarketSignals) -> f64 {
        self.quote_count += 1;

        // 1. Get effective gamma (with shrinkage)
        let gamma_eff = if self.config.enable_shrinkage_gamma {
            self.gamma.effective_gamma(&signals.to_pairs())
        } else {
            self.config.gamma_base
        };
        self.last_gamma = gamma_eff;

        // Store signals for learning feedback
        self.last_signals = self.gamma.get_standardized_signals(&signals.to_pairs());

        // 2. Get effective kappa (blended)
        let kappa_eff = if self.config.enable_blended_kappa {
            self.kappa.kappa()
        } else {
            self.config.kappa_prior_mean
        };

        // 3. Calculate GLFT optimal spread
        let delta_glft = if gamma_eff > 1e-9 && kappa_eff > 1e-9 {
            (1.0 / gamma_eff) * (1.0 + gamma_eff / kappa_eff).ln() + self.config.maker_fee_rate
        } else {
            self.config.maker_fee_rate + 0.001 // Fallback: fees + 10 bps
        };

        // 4. Get learned floor
        let delta_floor = if self.config.enable_adaptive_floor {
            self.floor.learned_spread_floor()
        } else {
            self.config.maker_fee_rate + self.config.as_prior_mean
        };

        // 5. Get fill rate ceiling
        let delta_ceiling = if self.config.enable_fill_controller {
            self.fill_controller.spread_ceiling()
        } else {
            None
        };

        // 6. Combine: max(floor, min(glft, ceiling))
        let delta_pre_ceiling = delta_glft.max(delta_floor);
        let delta_final = match delta_ceiling {
            Some(ceiling) => delta_pre_ceiling.min(ceiling),
            None => delta_pre_ceiling,
        };

        self.last_spread = delta_final;

        // Log periodically
        if self.quote_count.is_multiple_of(100) {
            debug!(
                quote_count = self.quote_count,
                gamma_eff = %format!("{:.4}", gamma_eff),
                kappa_eff = %format!("{:.1}", kappa_eff),
                delta_floor_bps = %format!("{:.2}", delta_floor * 10000.0),
                delta_glft_bps = %format!("{:.2}", delta_glft * 10000.0),
                delta_ceiling_bps = delta_ceiling.map(|c| format!("{:.2}", c * 10000.0)),
                delta_final_bps = %format!("{:.2}", delta_final * 10000.0),
                "Adaptive spread calculated"
            );
        }

        delta_final
    }

    /// Calculate asymmetric half-spreads (bid and ask separately).
    ///
    /// # Arguments
    /// * `signals` - Current market signals
    ///
    /// # Returns
    /// (half_spread_bid, half_spread_ask) as fractions of price
    pub fn asymmetric_half_spreads(&mut self, signals: &MarketSignals) -> (f64, f64) {
        let base = self.half_spread(signals);

        // Get directional kappa if available
        let (kappa_bid, kappa_ask) = self.kappa.directional_kappa();

        // Asymmetric adjustment based on kappa ratio
        let kappa_ratio = kappa_ask / kappa_bid.max(1.0);

        if (kappa_ratio - 1.0).abs() < 0.1 {
            // Approximately symmetric
            (base, base)
        } else {
            // Asymmetric: lower kappa side needs wider spread
            let bid_mult = (1.0 / kappa_ratio).sqrt();
            let ask_mult = kappa_ratio.sqrt();

            (base * bid_mult, base * ask_mult)
        }
    }

    /// Process a fill outcome and update all learning components.
    ///
    /// # Arguments
    /// * `outcome` - The fill outcome with AS measurement
    pub fn on_fill(&mut self, outcome: &FillOutcome) {
        // 1. Update learned floor with realized AS
        let as_realized = outcome.realized_as();
        self.floor.update(as_realized);

        // 2. Update blended kappa with own fill
        self.kappa.on_own_fill(
            outcome.fill_price,
            outcome.mid_at_fill,
            outcome.size,
            outcome.timestamp_ms,
        );

        // 3. Update fill rate controller
        self.fill_controller
            .update(1, self.time_since_last_fill, self.kappa.kappa());
        self.time_since_last_fill = 0.0;

        // 4. Update gamma weights based on PnL
        if !self.last_signals.is_empty() {
            // Determine if wider spread would have been better
            let pnl_gradient = if as_realized > self.config.as_prior_mean {
                // AS was higher than expected → wider would be better
                1.0
            } else {
                // AS was lower than expected → tighter could work
                -0.5
            };

            let magnitude = (as_realized / self.config.as_prior_mean).abs().min(2.0);
            self.gamma
                .update(&self.last_signals, pnl_gradient, magnitude);
        }

        // Log fill
        info!(
            fill_price = %format!("{:.2}", outcome.fill_price),
            mid = %format!("{:.2}", outcome.mid_at_fill),
            as_bps = %format!("{:.2}", as_realized * 10000.0),
            floor_bps = %format!("{:.2}", self.floor.learned_spread_floor() * 10000.0),
            fill_count = self.kappa.own_fill_count(),
            "Fill processed by adaptive calculator"
        );
    }

    /// Update when no fill occurred (quote expired or cancelled).
    ///
    /// # Arguments
    /// * `elapsed_secs` - Time the quote was active
    pub fn on_no_fill(&mut self, elapsed_secs: f64) {
        self.time_since_last_fill += elapsed_secs;

        // Update fill rate controller (0 fills)
        self.fill_controller
            .update(0, elapsed_secs, self.kappa.kappa());

        // If we're consistently not getting fills, nudge gamma down (tighter)
        if self.fill_controller.is_active()
            && !self.fill_controller.is_meeting_target()
            && !self.last_signals.is_empty()
        {
            // Small push toward tighter spreads
            self.gamma.update(&self.last_signals, -0.1, 0.1);
        }
    }

    /// Update book-based kappa from L2 data.
    pub fn on_l2_update(&mut self, bids: &[(f64, f64)], asks: &[(f64, f64)], mid: f64) {
        self.kappa.on_l2_update(bids, asks, mid);
    }

    /// Get current floor estimate.
    pub fn current_floor(&self) -> f64 {
        self.floor.learned_spread_floor()
    }

    /// Get current kappa estimate.
    pub fn current_kappa(&self) -> f64 {
        self.kappa.kappa()
    }

    /// Get current effective gamma.
    pub fn current_gamma(&self) -> f64 {
        self.last_gamma
    }

    /// Get observed fill rate.
    pub fn observed_fill_rate(&self) -> f64 {
        self.fill_controller.observed_fill_rate()
    }

    /// Check if all components are warmed up.
    ///
    /// NOTE: This is for CONFIDENCE reporting only. The adaptive system
    /// provides usable values IMMEDIATELY via reasonable priors.
    /// GLFT should use adaptive values even when not "warmed up".
    pub fn is_warmed_up(&self) -> bool {
        self.floor.is_warmed_up() && self.kappa.is_warmed_up() && self.gamma.is_warmed_up()
    }

    /// Check if the system can provide reasonable estimates.
    ///
    /// This is TRUE immediately - the adaptive system uses Bayesian priors
    /// that give sensible starting values. Use this instead of `is_warmed_up()`
    /// to determine if adaptive values should be used.
    ///
    /// The difference:
    /// - `can_provide_estimates()` = TRUE immediately (use adaptive values)
    /// - `is_warmed_up()` = TRUE after 20+ fills (fully calibrated)
    pub fn can_provide_estimates(&self) -> bool {
        // Always true - our priors give reasonable starting points
        true
    }

    /// Get warmup progress as a fraction (0.0 to 1.0).
    ///
    /// This can be used to scale uncertainty margins during warmup.
    /// Higher = more confident in learned values.
    pub fn warmup_progress(&self) -> f64 {
        // Components contribute to overall progress
        let floor_progress = (self.floor.observation_count().min(20) as f64) / 20.0;
        let kappa_progress = (self.kappa.own_fill_count().min(10) as f64) / 10.0;
        // Gamma uses standardizers which need ~20 observations
        let gamma_progress = if self.gamma.is_warmed_up() { 1.0 } else { 0.5 };

        // Weighted average (floor and kappa most important)
        floor_progress * 0.4 + kappa_progress * 0.4 + gamma_progress * 0.2
    }

    /// Get an uncertainty scaling factor for warmup period.
    ///
    /// During warmup, we should be more conservative. This returns a factor
    /// to multiply spreads by (> 1.0 during warmup, approaches 1.0 when warmed up).
    ///
    /// Formula: 1 + (1 - warmup_progress) * warmup_conservatism
    pub fn warmup_uncertainty_factor(&self) -> f64 {
        let progress = self.warmup_progress();
        // Start with 10% wider spreads, decay to 0% as we warm up
        // Reduced from 20% since priors are now well-calibrated
        1.0 + (1.0 - progress) * 0.1
    }

    /// Get component status for diagnostics.
    pub fn status(&self) -> AdaptiveStatus {
        AdaptiveStatus {
            floor_bps: self.floor.learned_spread_floor() * 10000.0,
            floor_as_mean_bps: self.floor.posterior_mean() * 10000.0,
            floor_observations: self.floor.observation_count(),
            gamma_effective: self.last_gamma,
            gamma_weights: self.gamma.weights().to_vec(),
            gamma_tau_squared: self.gamma.tau_squared(),
            kappa_blended: self.kappa.kappa(),
            kappa_book: self.kappa.book_kappa(),
            kappa_own: self.kappa.own_kappa(),
            kappa_own_fills: self.kappa.own_fill_count(),
            fill_rate_observed: self.fill_controller.observed_fill_rate(),
            fill_rate_target: self.fill_controller.target_fill_rate(),
            fill_ceiling: self.fill_controller.spread_ceiling(),
            is_meeting_fill_target: self.fill_controller.is_meeting_target(),
        }
    }

    /// Get the configuration.
    pub fn config(&self) -> &AdaptiveBayesianConfig {
        &self.config
    }

    /// Access the learned floor (for external AS updates).
    pub fn floor_mut(&mut self) -> &mut LearnedSpreadFloor {
        &mut self.floor
    }

    /// Access the blended kappa (for external updates).
    pub fn kappa_mut(&mut self) -> &mut BlendedKappaEstimator {
        &mut self.kappa
    }

    /// Access the fill controller (for external updates).
    pub fn fill_controller_mut(&mut self) -> &mut FillRateController {
        &mut self.fill_controller
    }

    /// Access the fill controller (read-only).
    pub fn fill_rate_controller(&self) -> &FillRateController {
        &self.fill_controller
    }

    // === Convenience Getters for ParameterAggregator ===

    /// Get the learned spread floor.
    pub fn spread_floor(&self) -> f64 {
        self.floor.learned_spread_floor()
    }

    /// Get blended kappa from Bayesian estimator.
    ///
    /// The BlendedKappaEstimator has its own well-calibrated prior (κ=2500 for BTC).
    /// During warmup with no fills, it returns prior_mean * warmup_factor = 2500 * 0.8 = 2000.
    /// This ensures tight spreads from startup rather than falling back to potentially
    /// unreliable book-based estimates from thin order books (like on testnet).
    ///
    /// The `_book_kappa_fallback` parameter is kept for API compatibility but not used.
    pub fn kappa(&self, _book_kappa_fallback: f64) -> f64 {
        // Always use BlendedKappaEstimator - it has proper prior-based warmup logic.
        // DO NOT fall back to external kappa estimates which can be unreliable on thin books.
        self.kappa.kappa()
    }

    /// Get adaptive gamma given market signals.
    ///
    /// This is a convenience wrapper around ShrinkageGamma that takes individual
    /// signal values rather than a MarketSignals struct.
    ///
    /// Note: This is an immutable version that doesn't update the standardizers.
    /// For real-time quoting, prefer calling `half_spread()` which updates all state.
    pub fn gamma(
        &self,
        gamma_base: f64,
        _vol_ratio: f64,
        _jump_ratio: f64,
        _inventory_util: f64,
        _hawkes_percentile: f64,
        _cascade_severity: f64,
    ) -> f64 {
        // For immutable access, use the last computed gamma if shrinkage is enabled
        // The standardizers are updated during half_spread() calls
        if self.config.enable_shrinkage_gamma && self.gamma.is_warmed_up() {
            // Use cached gamma since we can't mutate standardizers here
            self.last_gamma
        } else {
            gamma_base
        }
    }

    /// Get spread ceiling from fill rate controller.
    pub fn spread_ceiling(&self) -> f64 {
        self.fill_controller.spread_ceiling().unwrap_or(f64::MAX)
    }

    /// Compute dynamic spread ceiling from Bayesian models.
    ///
    /// Replaces hardcoded `--max-spread-bps` CLI argument with principled model-driven bounds:
    /// - Fill rate controller ceiling: Ensures we meet fill rate targets
    /// - Market spread p80: Ensures competitiveness vs other market makers
    ///
    /// # Arguments
    /// - `market_spread_p80_bps`: 80th percentile of observed market spreads (in bps)
    ///
    /// # Returns
    /// - `Some(ceiling)` in basis points if warmed up
    /// - `None` during warmup (trust GLFT, no ceiling)
    ///
    /// # Why `max()` not `min()`
    ///
    /// Taking the more permissive ceiling ensures we satisfy EITHER:
    /// - Fill rate targets (tight enough to get fills)
    /// - Market competitiveness (not wider than market)
    ///
    /// If fill controller wants 15 bps but market is at 20 bps, we use 20 bps
    /// (being wider than market is never good, but tighter than needed wastes edge).
    pub fn dynamic_spread_ceiling(&self, market_spread_p80_bps: Option<f64>) -> Option<f64> {
        // Don't apply ceiling during warmup - trust GLFT
        if !self.is_warmed_up() {
            return None;
        }

        // Get fill controller ceiling (returns None if meeting target)
        let fill_ceiling_bps = self
            .fill_controller
            .spread_ceiling()
            .map(|c| c * 10000.0); // Convert fraction to bps

        // Combine with market p80 using max (more permissive wins)
        match (fill_ceiling_bps, market_spread_p80_bps) {
            (Some(fc), Some(mc)) => Some(fc.max(mc)),
            (Some(fc), None) => Some(fc),
            (None, Some(mc)) => Some(mc),
            (None, None) => None, // No ceiling needed
        }
    }

    /// Compute inventory utilization.
    pub fn inventory_utilization(&self, position: f64, max_position: f64) -> f64 {
        if max_position > 0.0 {
            (position / max_position).abs()
        } else {
            0.0
        }
    }

    // === Simplified Fill Processing ===

    /// Simplified on_fill for direct integration.
    ///
    /// This wrapper handles fill processing without requiring the full FillOutcome struct.
    ///
    /// # Arguments
    /// * `as_realized` - Realized adverse selection (signed, positive = adverse)
    /// * `fill_distance` - Distance from mid when filled (fraction)
    /// * `pnl` - PnL from this fill
    /// * `book_kappa` - Current book-based kappa estimate
    pub fn on_fill_simple(
        &mut self,
        as_realized: f64,
        fill_distance: f64,
        pnl: f64,
        book_kappa: f64,
    ) {
        // 1. Update learned floor with realized AS
        self.floor.update(as_realized);

        // 2. Update kappa blend weight (fill indicates our kappa is accurate)
        // Since we don't have the full fill info, just update the blend weight
        self.kappa.increment_fill_count();

        // 3. Update fill rate controller
        self.fill_controller
            .update(1, self.time_since_last_fill, book_kappa);
        self.time_since_last_fill = 0.0;

        // 4. Update gamma weights based on PnL
        if !self.last_signals.is_empty() {
            let pnl_gradient = if pnl < 0.0 { 1.0 } else { -0.5 };
            let magnitude = (pnl.abs() / 0.0003).min(2.0); // Normalize by ~3 bps
            self.gamma
                .update(&self.last_signals, pnl_gradient, magnitude);
        }

        debug!(
            as_bps = %format!("{:.2}", as_realized * 10000.0),
            fill_dist_bps = %format!("{:.2}", fill_distance * 10000.0),
            pnl_bps = %format!("{:.2}", pnl * 10000.0),
            floor_bps = %format!("{:.2}", self.floor.learned_spread_floor() * 10000.0),
            kappa = %format!("{:.0}", self.kappa.kappa()),
            "Adaptive calculator: fill processed"
        );
    }

    /// Simplified on_no_fill for quote cycle updates.
    ///
    /// Call this each quote cycle when no fill occurred.
    /// Uses a default elapsed time assumption.
    pub fn on_no_fill_simple(&mut self) {
        const DEFAULT_ELAPSED_SECS: f64 = 1.0; // Approximate quote cycle interval
        self.on_no_fill(DEFAULT_ELAPSED_SECS);
    }
}

/// Diagnostic status of the adaptive calculator.
#[derive(Debug, Clone)]
pub struct AdaptiveStatus {
    // Floor
    pub floor_bps: f64,
    pub floor_as_mean_bps: f64,
    pub floor_observations: usize,

    // Gamma
    pub gamma_effective: f64,
    pub gamma_weights: Vec<f64>,
    pub gamma_tau_squared: f64,

    // Kappa
    pub kappa_blended: f64,
    pub kappa_book: f64,
    pub kappa_own: f64,
    pub kappa_own_fills: usize,

    // Fill rate
    pub fill_rate_observed: f64,
    pub fill_rate_target: f64,
    pub fill_ceiling: Option<f64>,
    pub is_meeting_fill_target: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_signals() -> MarketSignals {
        MarketSignals {
            vol_ratio: 1.0,
            jump_ratio: 1.5,
            inventory_util: 0.3,
            hawkes_percentile: 0.5,
            spread_regime: 0.0,
            cascade_severity: 0.0,
        }
    }

    #[test]
    fn test_initial_spread() {
        let mut calc = AdaptiveSpreadCalculator::default_config();
        let signals = default_signals();

        let spread = calc.half_spread(&signals);

        // Should be reasonable (between 2 and 20 bps)
        assert!(
            spread > 0.0002 && spread < 0.002,
            "Initial spread should be reasonable: {:.4} bps",
            spread * 10000.0
        );
    }

    #[test]
    fn test_spread_respects_floor() {
        let mut config = AdaptiveBayesianConfig::default();
        config.as_prior_mean = 0.001; // High AS prior = high floor

        let mut calc = AdaptiveSpreadCalculator::new(config);
        let signals = default_signals();

        let spread = calc.half_spread(&signals);

        // Should be at least the floor
        let floor = calc.current_floor();
        assert!(
            spread >= floor * 0.99, // Allow small floating point error
            "Spread {:.4} should respect floor {:.4}",
            spread * 10000.0,
            floor * 10000.0
        );
    }

    #[test]
    fn test_fill_updates_learning() {
        let mut calc = AdaptiveSpreadCalculator::default_config();
        let signals = default_signals();

        // Calculate initial spread
        let _ = calc.half_spread(&signals);
        let initial_floor = calc.current_floor();

        // Simulate low-AS fills
        for i in 0..50 {
            let outcome = FillOutcome {
                fill_price: 100.0,
                mid_at_fill: 100.0,
                mid_after_horizon: 100.001, // Very low AS
                size: 1.0,
                direction: 1.0,
                timestamp_ms: i * 1000,
                pnl: Some(0.01),
            };
            calc.on_fill(&outcome);
        }

        // Floor should decrease with low AS
        let new_floor = calc.current_floor();
        assert!(
            new_floor < initial_floor,
            "Floor should decrease with low AS: {:.4} -> {:.4}",
            initial_floor * 10000.0,
            new_floor * 10000.0
        );
    }

    #[test]
    fn test_no_fill_tightens() {
        let mut calc = AdaptiveSpreadCalculator::default_config();
        let signals = default_signals();

        // Calculate initial spread
        let _ = calc.half_spread(&signals);

        // Simulate long period with no fills
        for _ in 0..60 {
            calc.on_no_fill(5.0); // 5 seconds per iteration = 5 minutes total
        }

        // Fill controller should be active and suggesting tighter
        assert!(calc.fill_controller.is_active());

        // If not meeting target, adjustment factor should be < 1
        if !calc.fill_controller.is_meeting_target() {
            let factor = calc.fill_controller.spread_adjustment_factor();
            assert!(
                factor < 1.0,
                "Should suggest tighter when not filling: {}",
                factor
            );
        }
    }

    #[test]
    fn test_status_diagnostics() {
        let calc = AdaptiveSpreadCalculator::default_config();
        let status = calc.status();

        assert!(status.floor_bps > 0.0);
        assert!(status.kappa_blended > 0.0);
        assert!(status.fill_rate_target > 0.0);
    }

    #[test]
    fn test_asymmetric_spreads() {
        let mut calc = AdaptiveSpreadCalculator::default_config();
        let signals = default_signals();

        let (bid, ask) = calc.asymmetric_half_spreads(&signals);

        // Both should be positive
        assert!(bid > 0.0 && ask > 0.0);

        // Initially should be approximately symmetric
        let diff = ((bid - ask) / bid).abs();
        assert!(diff < 0.2, "Initial spreads should be roughly symmetric");
    }
}
