//! Main parameter estimation orchestrator.
//!
//! ParameterEstimator coordinates all sub-estimators to provide
//! HFT-grade market parameter estimates for the GLFT strategy.

use tracing::debug;

use super::covariance::ParameterCovariance;
use super::hierarchical_kappa::HierarchicalKappa;
use super::jump::JumpEstimator;
use super::kalman::KalmanPriceFilter;
use super::kappa::{BayesianKappaEstimator, BookStructureEstimator};
use super::kappa_orchestrator::{KappaOrchestrator, KappaOrchestratorConfig};
use super::microprice::MicropriceEstimator;
use super::momentum::{MomentumDetector, MomentumModel, TradeFlowTracker};
use super::soft_jump::SoftJumpClassifier;
use super::volatility::{MultiScaleBipowerEstimator, StochasticVolParams, VolatilityRegimeTracker};
use super::volume::{VolumeBucketAccumulator, VolumeTickArrivalEstimator};
use super::{EstimatorConfig, EstimatorV2Flags, MarketEstimator, VolatilityRegime};

// ============================================================================
// ParameterEstimator
// ============================================================================

/// Central parameter estimation pipeline for GLFT market making.
///
/// Pipeline:
/// 1. Raw trades → Volume Clock → Normalized volume buckets with VWAP
/// 2. VWAP returns → Multi-Scale Bipower → sigma_clean, sigma_total, sigma_effective
/// 3. VWAP returns → Momentum Detector → falling/rising knife scores
/// 4. Trade tape → Flow Tracker → buy/sell imbalance
/// 5. Trade tape → Fill Rate Kappa → trade distance distribution (CORRECT κ for GLFT)
/// 6. L2 Book → Book analysis for auxiliary adjustments (imbalance, etc.)
/// 7. Regime detection: fast jump_ratio > threshold = toxic
#[derive(Debug)]
pub struct ParameterEstimator {
    config: EstimatorConfig,
    /// Volume bucket accumulator (volume clock)
    bucket_accumulator: VolumeBucketAccumulator,
    /// Multi-timescale bipower estimator (replaces single-scale)
    multi_scale: MultiScaleBipowerEstimator,
    /// Momentum detector for falling/rising knife patterns
    momentum: MomentumDetector,
    /// Trade flow tracker for buy/sell imbalance
    flow: TradeFlowTracker,
    /// Bayesian kappa from OUR order fills (PRIMARY - correct GLFT semantics)
    own_kappa: BayesianKappaEstimator,
    /// Bayesian kappa from OUR BID fills (buy fills = bid got hit)
    /// Theory: Separate bid/ask kappas capture asymmetric informed flow
    own_kappa_bid: BayesianKappaEstimator,
    /// Bayesian kappa from OUR ASK fills (sell fills = ask got lifted)
    own_kappa_ask: BayesianKappaEstimator,
    /// Bayesian kappa from market-wide trades (FALLBACK during warmup)
    market_kappa: BayesianKappaEstimator,
    /// Volume tick arrival estimator
    arrival: VolumeTickArrivalEstimator,
    /// Book structure estimator (imbalance, near-touch liquidity)
    book_structure: BookStructureEstimator,
    /// Microprice estimator (data-driven fair price)
    microprice_estimator: MicropriceEstimator,
    /// Current mid price
    current_mid: f64,
    /// Current timestamp for momentum queries
    current_time_ms: u64,
    /// 4-state volatility regime tracker with hysteresis
    volatility_regime: VolatilityRegimeTracker,
    /// Jump process estimator (λ, μ_j, σ_j)
    jump_estimator: JumpEstimator,
    /// Stochastic volatility parameters (κ_vol, θ_vol, ξ, ρ)
    stoch_vol: StochasticVolParams,
    /// Kalman filter for denoising mid price (stochastic module integration)
    kalman_filter: KalmanPriceFilter,
    /// Probabilistic momentum model for learned continuation probabilities
    momentum_model: MomentumModel,
    /// Last momentum value (for detecting continuation vs reversal)
    last_momentum_bps: f64,

    // === V2 Bayesian Estimator Components ===
    /// Feature flags for V2 components
    v2_flags: EstimatorV2Flags,
    /// Hierarchical kappa estimator (market kappa as prior for own fills)
    hierarchical_kappa: HierarchicalKappa,
    /// Soft jump classifier (P(jump) ∈ [0,1] mixture model)
    soft_jump: SoftJumpClassifier,
    /// (κ, σ) parameter covariance tracker
    param_covariance: ParameterCovariance,

    // === V3 Robust Kappa Orchestrator ===
    /// Robust kappa orchestrator: confidence-weighted blending of
    /// book-structure κ, robust (Student-t) κ, and own-fill κ.
    /// Resists outlier poisoning from liquidation cascades.
    kappa_orchestrator: KappaOrchestrator,

    // === Competitive Spread Control ===
    /// Optional minimum floor for kappa output.
    /// Applied after Bayesian estimation to force competitive spreads on illiquid assets.
    /// Higher kappa = tighter GLFT spreads.
    kappa_floor: Option<f64>,

    /// Optional maximum spread ceiling from CLI override.
    /// When set, static ceiling is used instead of dynamic model-driven ceiling.
    max_spread_ceiling_bps: Option<f64>,
}

impl ParameterEstimator {
    /// Create a new parameter estimator with the given config.
    pub fn new(config: EstimatorConfig) -> Self {
        let bucket_accumulator = VolumeBucketAccumulator::new(&config);
        let multi_scale = MultiScaleBipowerEstimator::new(&config);
        let momentum = MomentumDetector::new(config.momentum_window_ms);
        let flow = TradeFlowTracker::new(config.trade_flow_window_ms, config.trade_flow_alpha);

        // Bayesian kappa estimators with Gamma conjugate prior:
        // 1. own_kappa: Fed by ALL our order fills (PRIMARY - correct GLFT semantics)
        // 2. own_kappa_bid/ask: Fed by our fills split by side (for asymmetric spreads)
        // 3. market_kappa: Fed by market-wide trades (FALLBACK during warmup)
        // All use same prior and window configuration.
        let own_kappa = BayesianKappaEstimator::new(
            config.kappa_prior_mean,
            config.kappa_prior_strength,
            config.kappa_window_ms,
        );
        let own_kappa_bid = BayesianKappaEstimator::new(
            config.kappa_prior_mean,
            config.kappa_prior_strength,
            config.kappa_window_ms,
        );
        let own_kappa_ask = BayesianKappaEstimator::new(
            config.kappa_prior_mean,
            config.kappa_prior_strength,
            config.kappa_window_ms,
        );
        let market_kappa = BayesianKappaEstimator::new(
            config.kappa_prior_mean,
            config.kappa_prior_strength,
            config.kappa_window_ms,
        );

        let arrival = VolumeTickArrivalEstimator::new(
            config.medium_half_life_ticks, // Use medium timescale
            config.default_arrival_intensity,
        );

        let book_structure = BookStructureEstimator::new();

        // Microprice estimator: 60s window, 300ms forward horizon, 50 min observations
        let microprice_estimator = MicropriceEstimator::new(60_000, 300, 50);

        // Volatility regime tracker with baseline from config
        let volatility_regime = VolatilityRegimeTracker::new(config.default_sigma);

        // Jump process estimator (First Principles Gap 1)
        let jump_estimator = JumpEstimator::new();

        // Stochastic volatility params (First Principles Gap 2)
        let stoch_vol = StochasticVolParams::new(config.default_sigma);

        // Kalman filter for denoising mid price (Stochastic Module Integration)
        // Uses sensible defaults for crypto markets (Q=1bp², R=0.5bp²)
        let kalman_filter = KalmanPriceFilter::default_crypto();

        // Momentum model for learned continuation probabilities (First Principles Gap 10)
        // Uses 5-minute window and 0.1 EWMA alpha for smooth learning
        let momentum_model = MomentumModel::default_config();

        // V2 Bayesian Estimator Components
        // Hierarchical kappa uses market kappa as prior, updated on our fills
        let hierarchical_kappa = HierarchicalKappa::new(
            config.kappa_prior_strength,
            config.kappa_prior_mean,
            config.kappa_window_ms,
        );
        // Soft jump classifier for probabilistic toxicity detection
        let soft_jump = SoftJumpClassifier::default_params();
        // Parameter covariance tracker (50-tick half-life for (κ, σ) correlation)
        let param_covariance = ParameterCovariance::with_half_life(50.0);

        // V3 Robust Kappa Orchestrator: confidence-weighted blending of multiple estimators
        // Uses book-structure κ (L2 depth decay), robust κ (Student-t), and own-fill κ
        // Resists outlier poisoning from liquidation cascades
        let kappa_orchestrator = KappaOrchestrator::new(KappaOrchestratorConfig {
            prior_kappa: config.kappa_prior_mean,
            prior_strength: config.kappa_prior_strength,
            robust_nu: 4.0,                          // Moderate heavy tails
            robust_window_ms: config.kappa_window_ms,
            own_fill_window_ms: config.kappa_window_ms,
            use_book_kappa: true,
            use_robust_kappa: true,
        });

        // Kappa floor for competitive spread control on illiquid assets
        let kappa_floor = config.kappa_floor;
        // Max spread ceiling for spread control (CLI override)
        let max_spread_ceiling_bps = config.max_spread_ceiling_bps;

        Self {
            config,
            bucket_accumulator,
            multi_scale,
            momentum,
            flow,
            own_kappa,
            own_kappa_bid,
            own_kappa_ask,
            market_kappa,
            arrival,
            book_structure,
            microprice_estimator,
            current_mid: 0.0,
            current_time_ms: 0,
            volatility_regime,
            jump_estimator,
            stoch_vol,
            kalman_filter,
            momentum_model,
            last_momentum_bps: 0.0,
            // V2 Bayesian components - always enabled
            v2_flags: EstimatorV2Flags::all_enabled(),
            hierarchical_kappa,
            soft_jump,
            param_covariance,
            // V3 Robust kappa orchestrator
            kappa_orchestrator,
            // Competitive spread control
            kappa_floor,
            max_spread_ceiling_bps,
        }
    }

    /// Update current mid price.
    pub fn on_mid_update(&mut self, mid_price: f64) {
        self.current_mid = mid_price;
        // Feed Kalman filter (stochastic module integration)
        self.kalman_filter.filter(mid_price);
    }

    /// Process a new trade (feeds into volume clock, flow tracker, AND market kappa).
    ///
    /// # Arguments
    /// * `timestamp_ms` - Trade timestamp
    /// * `price` - Trade price
    /// * `size` - Trade size
    /// * `is_buy_aggressor` - Whether buyer was the taker (if available from exchange)
    pub fn on_trade(
        &mut self,
        timestamp_ms: u64,
        price: f64,
        size: f64,
        is_buy_aggressor: Option<bool>,
    ) {
        self.current_time_ms = timestamp_ms;

        // Track trade flow if we know aggressor side
        if let Some(is_buy) = is_buy_aggressor {
            self.flow.on_trade(timestamp_ms, size, is_buy);
        }

        // Initialize current_mid from trade price if not yet set.
        // This prevents warmup deadlock where trades arrive before L2Book/AllMids.
        // The trade price is close enough to mid for initial kappa estimates.
        if self.current_mid <= 0.0 && price > 0.0 {
            self.current_mid = price;
        }

        // Feed into MARKET kappa estimator (LEGACY - kept for backwards compatibility)
        // WARNING: This is vulnerable to outlier poisoning from liquidations!
        // The V3 KappaOrchestrator uses a robust Student-t estimator instead.
        if self.current_mid > 0.0 {
            self.market_kappa
                .on_trade(timestamp_ms, price, size, self.current_mid);

            // V3: Feed into robust kappa orchestrator (outlier-resistant)
            // Uses Student-t likelihood to resist liquidation cascade poisoning
            self.kappa_orchestrator
                .on_market_trade(timestamp_ms, price, self.current_mid);
        }

        // Feed into volume bucket accumulator
        if let Some(bucket) = self.bucket_accumulator.on_trade(timestamp_ms, price, size) {
            // Get log return BEFORE updating multi_scale (it will update last_vwap)
            let log_return = self.multi_scale.last_log_return(&bucket);

            // Bucket completed - update estimators
            self.multi_scale.on_bucket(&bucket);
            self.arrival.on_bucket(&bucket);

            // Update momentum detector with signed return
            if let Some(ret) = log_return {
                self.momentum.on_bucket(bucket.end_time_ms, ret);

                // Update probabilistic momentum model with continuation observation
                // Continuation = return has same sign as previous momentum
                let ret_bps = ret * 10_000.0;
                if self.last_momentum_bps.abs() > 1.0 {
                    // Only record if there was meaningful prior momentum
                    let continued = (ret_bps > 0.0) == (self.last_momentum_bps > 0.0);
                    self.momentum_model.record_observation(
                        bucket.end_time_ms,
                        self.last_momentum_bps,
                        continued,
                    );
                }
                // Update last momentum for next iteration
                self.last_momentum_bps = self.momentum.momentum_bps(bucket.end_time_ms);

                // Update jump estimator with return (detects jumps > 3σ)
                let sigma_clean = self.multi_scale.sigma_clean();
                self.jump_estimator
                    .on_return(bucket.end_time_ms, ret, sigma_clean);

                // Update stochastic vol with current variance observation
                let variance = self.multi_scale.sigma_total().powi(2);
                self.stoch_vol
                    .on_variance(bucket.end_time_ms, variance, ret);
            }

            // Update volatility regime with current sigma and jump ratio
            let sigma = self.multi_scale.sigma_clean();
            let jump_ratio = self.multi_scale.jump_ratio_fast();
            self.volatility_regime.update(sigma, jump_ratio);

            // Update Kalman filter process noise Q from current volatility estimate
            // Adaptive Q = σ² × dt, where dt is approximate bucket duration in seconds
            // This makes the Kalman filter more responsive during high volatility
            let bucket_duration_secs =
                bucket.end_time_ms.saturating_sub(bucket.start_time_ms) as f64 / 1000.0;
            // Use a minimum dt to avoid numerical instability
            let dt = bucket_duration_secs.max(0.1);
            self.kalman_filter
                .set_process_noise_from_volatility(sigma, dt);

            // Slowly update baseline from slow sigma (long-term anchor)
            // Using slow sigma as the stable reference for regime thresholds
            self.volatility_regime
                .update_baseline(self.multi_scale.sigma_clean());

            // === Bayesian Component Updates (always active) ===
            // Update soft jump classifier with current sigma and return
            self.soft_jump.update_sigma(sigma);
            if let Some(ret) = log_return {
                self.soft_jump.on_return(ret);
            }

            // Update parameter covariance tracker
            let kappa = self.kappa();
            self.param_covariance.update(kappa, sigma);

            debug!(
                vwap = %format!("{:.4}", bucket.vwap),
                volume = %format!("{:.4}", bucket.volume),
                duration_ms = bucket.end_time_ms.saturating_sub(bucket.start_time_ms),
                tick = self.multi_scale.tick_count(),
                sigma_clean = %format!("{:.6}", self.multi_scale.sigma_clean()),
                sigma_total = %format!("{:.6}", self.multi_scale.sigma_total()),
                jump_ratio = %format!("{:.2}", self.multi_scale.jump_ratio_fast()),
                kappa_blended = %format!("{:.0}", self.kappa()),
                own_kappa_conf = %format!("{:.2}", self.own_kappa.confidence()),
                regime = ?self.volatility_regime.regime(),
                "Volume bucket completed"
            );
        }
    }

    /// Process a fill from our own order for kappa estimation.
    ///
    /// This provides the TRUE fill rate decay for our orders (correct GLFT semantics),
    /// not market-wide proxy data. Call this when we receive a fill notification.
    ///
    /// # Theory (First Principles Fix 2):
    /// Order book depth is asymmetric during flow imbalance. When informed traders
    /// are selling, our bids get hit more (lower κ_bid). When they're buying, our
    /// asks get lifted (lower κ_ask). By tracking bid/ask fills separately, we can
    /// compute asymmetric GLFT spreads:
    ///   δ_bid = (1/γ) × ln(1 + γ/κ_bid)
    ///   δ_ask = (1/γ) × ln(1 + γ/κ_ask)
    ///
    /// # Arguments
    /// * `timestamp_ms` - Fill timestamp
    /// * `placement_price` - Where we originally placed the order
    /// * `fill_price` - Where the order actually filled
    /// * `fill_size` - Size of the fill
    /// * `is_buy` - True if this was a buy order (our bid got hit)
    pub fn on_own_fill(
        &mut self,
        timestamp_ms: u64,
        placement_price: f64,
        fill_price: f64,
        fill_size: f64,
        is_buy: bool,
    ) {
        // Feed ALL fills into aggregate kappa (for backward compatibility)
        self.own_kappa
            .record_fill_distance(timestamp_ms, placement_price, fill_price, fill_size);

        // Feed into directional kappa estimator:
        // - is_buy=true means our BID was filled (we bought)
        // - is_buy=false means our ASK was filled (we sold)
        if is_buy {
            self.own_kappa_bid.record_fill_distance(
                timestamp_ms,
                placement_price,
                fill_price,
                fill_size,
            );
        } else {
            self.own_kappa_ask.record_fill_distance(
                timestamp_ms,
                placement_price,
                fill_price,
                fill_size,
            );
        }

        // === Hierarchical Kappa Update (always active) ===
        // Calculate fill distance in basis points
        let distance_bps = if placement_price > 0.0 {
            ((fill_price - placement_price).abs() / placement_price) * 10_000.0
        } else {
            0.0
        };
        // Record fill in hierarchical kappa (adverse selection not yet available here)
        self.hierarchical_kappa
            .record_fill(distance_bps, timestamp_ms, None);

        // V3: Feed into kappa orchestrator's own-fill estimator (gold standard)
        // Own fills are the most accurate source - they dominate as fills accumulate
        if self.current_mid > 0.0 {
            self.kappa_orchestrator
                .on_own_fill(timestamp_ms, fill_price, fill_size, self.current_mid);
        }
    }

    /// Legacy on_trade without aggressor info (backward compatibility).
    pub fn on_trade_legacy(&mut self, timestamp_ms: u64, price: f64, size: f64) {
        self.on_trade(timestamp_ms, price, size, None);
    }

    /// Process L2 order book update for book structure analysis.
    /// bids and asks are slices of (price, size) tuples, best first.
    /// V3: Also feeds into book-structure kappa estimator (L2 depth decay regression).
    pub fn on_l2_book(&mut self, bids: &[(f64, f64)], asks: &[(f64, f64)], mid: f64) {
        self.current_mid = mid;
        // Book structure for imbalance and liquidity signals (still valid uses)
        self.book_structure.update(bids, asks, mid);

        // V3: Feed into kappa orchestrator for book-structure κ estimation
        // Fits exponential decay model: Depth(δ) = D₀ × exp(-κδ)
        self.kappa_orchestrator.on_l2_book(bids, asks, mid);

        // Adapt microprice forward horizon to current arrival intensity
        // Fast markets → shorter horizon, slow markets → longer horizon
        let arrival = self.arrival.ticks_per_second();
        self.microprice_estimator.update_horizon(arrival);

        // Feed microprice estimator with current signals
        self.microprice_estimator.on_book_update(
            self.current_time_ms,
            mid,
            self.book_structure.imbalance(),
            self.flow.imbalance(),
        );

        // === Hierarchical Kappa: Update market kappa as prior (always active) ===
        let market_kappa = self.market_kappa.posterior_mean();
        let market_conf = self.market_kappa.confidence();
        self.hierarchical_kappa
            .update_market_kappa(market_kappa, market_conf);
    }

    // === Volatility Accessors ===

    /// Get clean volatility (σ_clean) - per-second, NOT annualized.
    /// Based on Bipower Variation, robust to jumps.
    /// Use for base spread pricing (continuous risk).
    pub fn sigma(&self) -> f64 {
        self.multi_scale.sigma_clean()
    }

    /// Get clean volatility - alias for sigma()
    pub fn sigma_clean(&self) -> f64 {
        self.multi_scale.sigma_clean()
    }

    /// Get total volatility (σ_total) - includes jumps.
    /// Based on Realized Variance, captures full price risk.
    pub fn sigma_total(&self) -> f64 {
        self.multi_scale.sigma_total()
    }

    /// Get effective volatility (σ_effective) - blended.
    /// Blends clean and total based on jump regime.
    /// Use for inventory skew (reacts appropriately to jumps).
    pub fn sigma_effective(&self) -> f64 {
        self.multi_scale.sigma_effective()
    }

    /// Get leverage-adjusted effective volatility.
    ///
    /// Applies the leverage effect: volatility increases during down moves
    /// when ρ (price-vol correlation) is negative.
    ///
    /// Formula: sigma_adjusted = sigma_effective × (1 + |ρ| × 0.2 × down_move_indicator)
    ///
    /// This provides wider spreads during falling markets when volatility
    /// typically spikes, reducing adverse selection.
    pub fn sigma_leverage_adjusted(&self) -> f64 {
        let base_sigma = self.multi_scale.sigma_effective();
        let rho = self.stoch_vol.rho();
        let momentum_bps = self.momentum.momentum_bps(self.current_time_ms);

        // Only apply leverage effect when:
        // 1. Momentum is negative (falling market)
        // 2. ρ is negative (leverage effect exists)
        if momentum_bps < 0.0 && rho < 0.0 {
            // Scale adjustment by momentum magnitude (capped at 50bps)
            let momentum_factor = (momentum_bps.abs() / 50.0).clamp(0.0, 1.0);
            // Adjustment = |ρ| × 0.2 × momentum_factor
            // Max adjustment: 0.95 × 0.2 × 1.0 = 19% vol increase
            let adjustment = rho.abs() * 0.2 * momentum_factor;
            base_sigma * (1.0 + adjustment)
        } else {
            base_sigma
        }
    }

    // === Order Book Accessors ===

    /// Apply kappa floor if configured.
    /// Returns the kappa value clamped to at least the floor value.
    #[inline]
    fn apply_kappa_floor(&self, kappa: f64) -> f64 {
        match self.kappa_floor {
            Some(floor) => kappa.max(floor),
            None => kappa,
        }
    }

    /// Compute dynamic kappa floor based on Bayesian confidence.
    ///
    /// This replaces hardcoded CLI arguments with principled model-driven bounds:
    /// - Low confidence (< 0.3): Use prior_mean × 0.8 (conservative during warmup)
    /// - Medium confidence: Blend between prior and 95% credible interval lower bound
    /// - High confidence (> 0.7): Use ci_95_lower (data-driven, principled)
    ///
    /// # Mathematical Justification
    ///
    /// - Prior provides regularization during warmup (Bayesian shrinkage)
    /// - 95% CI lower bound is principled: P(true κ ≥ ci_lower) = 0.975
    /// - Smooth blending prevents discontinuities in spread during transition
    /// - Floor never exceeds posterior mean (would invert the logic)
    ///
    /// # Returns
    ///
    /// Dynamic kappa floor value. Higher floor = tighter GLFT spreads.
    pub fn dynamic_kappa_floor(&self) -> f64 {
        let conf = self.own_kappa.confidence().clamp(0.0, 1.0);
        let prior = self.config.kappa_prior_mean;

        // During warmup (low confidence): use prior mean with slight discount
        // The 0.8 factor provides conservatism - we'd rather quote slightly wider
        // than too tight when we don't have confidence in the estimate
        if conf < 0.3 {
            return prior * 0.8;
        }

        // Get 95% credible interval lower bound from own kappa
        // This is data-driven: 95% posterior probability that true κ ≥ ci_lower
        let ci_lower = self.own_kappa.ci_95_lower().max(100.0); // Minimum sensible kappa

        // Transition zone: blend prior and CI lower bound based on confidence
        // At conf=0.3: blend_weight=0, floor=prior
        // At conf=1.0: blend_weight=1, floor=ci_lower
        let blend_weight = (conf - 0.3) / 0.7;
        let blended = (1.0 - blend_weight) * prior + blend_weight * ci_lower;

        // Never allow floor above posterior mean (would invert GLFT logic)
        // If ci_lower > posterior_mean, something is wrong with the posterior
        let posterior_mean = self.own_kappa.posterior_mean();
        blended.min(posterior_mean)
    }

    /// Check if a static kappa floor is configured (CLI override).
    ///
    /// Returns true if a static floor was set via CLI, false if dynamic bounds should be used.
    pub fn has_static_kappa_floor(&self) -> bool {
        self.kappa_floor.is_some()
    }

    /// Check if a static max spread ceiling is configured (CLI override).
    ///
    /// Returns true if a static ceiling was set via CLI, false if dynamic bounds should be used.
    pub fn has_static_max_spread_ceiling(&self) -> bool {
        self.max_spread_ceiling_bps.is_some()
    }

    /// Get current kappa estimate (blended from own fills and market data).
    ///
    /// Blending formula:
    /// - At startup (0% own confidence): 100% market data
    /// - After some fills (50% own confidence): 50/50 blend
    /// - After many fills (100% own confidence): 100% own data
    ///
    /// This gives fast warmup from market data, but converges to
    /// the theoretically correct own-fill based estimate.
    pub fn kappa(&self) -> f64 {
        let own_conf = self.own_kappa.confidence();

        let own = self.own_kappa.posterior_mean();
        let market = self.market_kappa.posterior_mean();

        // Linear blending based on confidence:
        // - 0% confidence → 100% market kappa (prior-based)
        // - 100% confidence → 100% own kappa (fill-based)
        //
        // The prior is now calibrated for liquid markets (κ=2500 for BTC/ETH),
        // so we no longer need the conservative 50% discount that was causing
        // excessively wide spreads during warmup.
        //
        // With proper prior calibration:
        // - κ=2500 → δ* ≈ 4-6bp (competitive)
        // - As fills accumulate, κ adapts to true fill rate
        let blended = own_conf * own + (1.0 - own_conf) * market;

        // Apply floor for competitive spread control on illiquid assets
        self.apply_kappa_floor(blended)
    }

    /// Get kappa from our own order fills only (no blending).
    ///
    /// This is the theoretically correct κ for GLFT - our actual fill rate decay.
    /// May have high uncertainty if we haven't received many fills yet.
    pub fn kappa_own(&self) -> f64 {
        self.own_kappa.posterior_mean()
    }

    /// Get kappa from market-wide trades (fallback source).
    ///
    /// This is a proxy estimate based on where all trades execute vs mid.
    /// Used during warmup when we don't have enough own-fill data.
    pub fn kappa_market(&self) -> f64 {
        self.market_kappa.posterior_mean()
    }

    /// Get kappa posterior standard deviation (uncertainty estimate).
    ///
    /// Returns weighted combination of both estimator uncertainties.
    pub fn kappa_std(&self) -> f64 {
        let own_conf = self.own_kappa.confidence();
        let own_std = self.own_kappa.posterior_std();
        let market_std = self.market_kappa.posterior_std();

        // Weighted combination of uncertainties
        own_conf * own_std + (1.0 - own_conf) * market_std
    }

    /// Get own-fill kappa confidence [0, 1].
    ///
    /// Based on effective sample size of our own fills.
    /// Low confidence means we're relying more on market data.
    pub fn kappa_confidence(&self) -> f64 {
        self.own_kappa.confidence()
    }

    /// Get market kappa confidence [0, 1].
    pub fn kappa_market_confidence(&self) -> f64 {
        self.market_kappa.confidence()
    }

    // === V3 Robust Kappa Orchestrator Accessors ===

    /// Get robust kappa from the V3 orchestrator (outlier-resistant).
    ///
    /// This is the PRIMARY κ estimate for illiquid/HIP-3 markets.
    /// It combines three sources with confidence-weighted blending:
    /// 1. Book-structure κ: From L2 depth decay (exponential fit)
    /// 2. Robust κ: From market trades with Student-t likelihood (resists outliers)
    /// 3. Own-fill κ: From our actual order fills (gold standard when available)
    ///
    /// The orchestrator automatically selects the best source based on confidence.
    /// During warmup, book-structure dominates. As fills accumulate, own-fill takes over.
    pub fn kappa_robust(&self) -> f64 {
        let kappa = self.kappa_orchestrator.kappa_effective();
        let final_kappa = self.apply_kappa_floor(kappa);
        // Debug: Log every call to trace source of drop
        debug!(
            orchestrator_kappa = %format!("{:.0}", kappa),
            floor_applied = self.kappa_floor.is_some(),
            final_kappa = %format!("{:.0}", final_kappa),
            own_fills = self.kappa_orchestrator.own_kappa_observation_count(),
            "kappa_robust computed"
        );
        final_kappa
    }

    /// Get robust kappa confidence from orchestrator [0, 1].
    pub fn kappa_robust_confidence(&self) -> f64 {
        self.kappa_orchestrator.confidence()
    }

    /// Check if robust kappa orchestrator is warmed up.
    pub fn kappa_robust_warmed_up(&self) -> bool {
        self.kappa_orchestrator.is_warmed_up()
    }

    /// Get component breakdown from kappa orchestrator for diagnostics.
    ///
    /// Returns ((own_kappa, own_weight), (book_kappa, book_weight),
    ///          (robust_kappa, robust_weight), (prior_kappa, prior_weight), is_warmup)
    pub fn kappa_robust_breakdown(
        &self,
    ) -> ((f64, f64), (f64, f64), (f64, f64), (f64, f64), bool) {
        self.kappa_orchestrator.component_breakdown()
    }

    /// Get book-structure kappa from L2 depth regression.
    pub fn kappa_book(&self) -> f64 {
        self.kappa_orchestrator.book_kappa()
    }

    /// Get robust kappa from Student-t estimator.
    pub fn kappa_studentt(&self) -> f64 {
        self.kappa_orchestrator.robust_kappa()
    }

    /// Get own-fill kappa from Bayesian estimator (in orchestrator).
    pub fn kappa_own_robust(&self) -> f64 {
        self.kappa_orchestrator.own_kappa()
    }

    /// Get number of outliers detected by robust estimator.
    pub fn kappa_outlier_count(&self) -> u64 {
        self.kappa_orchestrator.outlier_count()
    }

    /// Get robust kappa effective sample size.
    pub fn robust_kappa_ess(&self) -> f64 {
        self.kappa_orchestrator.robust_kappa_ess()
    }

    /// Get robust kappa nu (degrees of freedom).
    pub fn robust_kappa_nu(&self) -> f64 {
        self.kappa_orchestrator.robust_kappa_nu()
    }

    /// Get robust kappa observation count.
    pub fn robust_kappa_obs_count(&self) -> u64 {
        self.kappa_orchestrator.robust_kappa_obs_count()
    }

    /// Get coefficient of variation for fill distance distribution.
    ///
    /// For exponential: CV ≈ 1.0
    /// CV > 1.0: Heavy tail (power-law like) - common in crypto
    /// CV < 1.0: Light tail
    ///
    /// Uses blended CV from both sources based on confidence.
    pub fn kappa_cv(&self) -> f64 {
        let own_conf = self.own_kappa.confidence();
        let own_cv = self.own_kappa.cv();
        let market_cv = self.market_kappa.cv();

        own_conf * own_cv + (1.0 - own_conf) * market_cv
    }

    /// Get directional kappa for bid side (our buy fills).
    ///
    /// Theory: When informed flow is selling, our bids get hit more often
    /// and at worse prices → lower κ_bid → wider bid spread.
    ///
    /// Uses smooth confidence-weighted blending to avoid spread blow-up during warmup:
    /// - Low confidence: Blend toward market kappa with conservative scaling
    /// - High confidence: Use own fill data directly
    pub fn kappa_bid(&self) -> f64 {
        let own_conf = self.own_kappa_bid.confidence();
        let own = self.own_kappa_bid.posterior_mean();
        let market = self.market_kappa.posterior_mean();
        let market_conf = self.market_kappa.confidence();

        // Smooth blending formula:
        // kappa_bid = own_conf * own_kappa + (1-own_conf) * market_kappa * market_scale
        // market_scale smoothly transitions from 0.5 to 1.0 based on market confidence
        // This avoids the hard 0.3 threshold that caused spread discontinuities
        let market_scale = 0.5 + 0.5 * market_conf;
        let blended = own_conf * own + (1.0 - own_conf) * market * market_scale;

        // Apply floor for competitive spread control on illiquid assets
        self.apply_kappa_floor(blended)
    }

    /// Get directional kappa for ask side (our sell fills).
    ///
    /// Theory: When informed flow is buying, our asks get lifted more often
    /// and at worse prices → lower κ_ask → wider ask spread.
    ///
    /// Uses smooth confidence-weighted blending to avoid spread blow-up during warmup:
    /// - Low confidence: Blend toward market kappa with conservative scaling
    /// - High confidence: Use own fill data directly
    pub fn kappa_ask(&self) -> f64 {
        let own_conf = self.own_kappa_ask.confidence();
        let own = self.own_kappa_ask.posterior_mean();
        let market = self.market_kappa.posterior_mean();
        let market_conf = self.market_kappa.confidence();

        // Smooth blending formula:
        // kappa_ask = own_conf * own_kappa + (1-own_conf) * market_kappa * market_scale
        // market_scale smoothly transitions from 0.5 to 1.0 based on market confidence
        // This avoids the hard 0.3 threshold that caused spread discontinuities
        let market_scale = 0.5 + 0.5 * market_conf;
        let blended = own_conf * own + (1.0 - own_conf) * market * market_scale;

        // Apply floor for competitive spread control on illiquid assets
        self.apply_kappa_floor(blended)
    }

    /// Get confidence for directional kappa estimates.
    pub fn kappa_bid_confidence(&self) -> f64 {
        self.own_kappa_bid.confidence()
    }

    /// Get confidence for directional kappa estimates.
    pub fn kappa_ask_confidence(&self) -> f64 {
        self.own_kappa_ask.confidence()
    }

    /// Check if fill distance distribution is heavy-tailed (CV > 1.2).
    /// Heavy-tailed distributions mean occasional large fills are more likely.
    pub fn is_heavy_tailed(&self) -> bool {
        self.market_kappa.is_heavy_tailed() || self.own_kappa.is_heavy_tailed()
    }

    /// Get current order arrival intensity (volume ticks per second).
    pub fn arrival_intensity(&self) -> f64 {
        self.arrival.ticks_per_second()
    }

    /// Get L2 book imbalance [-1, 1].
    /// Positive = more bids (buying pressure), Negative = more asks (selling pressure).
    /// Use for directional quote skew.
    pub fn book_imbalance(&self) -> f64 {
        self.book_structure.imbalance()
    }

    /// Get liquidity-based gamma multiplier [1.0, 2.0].
    /// Returns > 1.0 when near-touch liquidity is below average (thin book).
    /// Use to scale gamma up for wider spreads in thin conditions.
    pub fn liquidity_gamma_multiplier(&self) -> f64 {
        self.book_structure.gamma_multiplier()
    }

    /// Get near-touch book depth in USD (within 10 bps of mid).
    /// Used for stochastic constraint on tight quoting.
    pub fn near_touch_depth_usd(&self) -> f64 {
        // near_touch_depth is in contracts, multiply by mid to get USD
        self.book_structure.near_touch_depth() * self.current_mid
    }

    // === Microprice Accessors ===

    /// Get microprice (data-driven fair price).
    /// Incorporates book imbalance and flow imbalance predictions.
    /// Falls back to raw mid if not warmed up.
    pub fn microprice(&self) -> f64 {
        self.microprice_estimator.microprice(
            self.current_mid,
            self.book_structure.imbalance(),
            self.flow.imbalance(),
        )
    }

    /// Get β_book coefficient (return prediction per unit book imbalance).
    pub fn beta_book(&self) -> f64 {
        self.microprice_estimator.beta_book()
    }

    /// Get β_flow coefficient (return prediction per unit flow imbalance).
    pub fn beta_flow(&self) -> f64 {
        self.microprice_estimator.beta_flow()
    }

    /// Get R² of microprice regression.
    pub fn microprice_r_squared(&self) -> f64 {
        self.microprice_estimator.r_squared()
    }

    /// Check if microprice estimator is warmed up.
    pub fn microprice_warmed_up(&self) -> bool {
        self.microprice_estimator.is_warmed_up()
    }

    // === Regime Detection ===

    /// Get fast RV/BV jump ratio.
    /// - ≈ 1.0: Normal diffusion (safe to market make)
    /// - > 1.5: Jumps present (toxic environment)
    pub fn jump_ratio(&self) -> f64 {
        self.multi_scale.jump_ratio_fast()
    }

    /// Check if currently in toxic (jump) regime.
    pub fn is_toxic_regime(&self) -> bool {
        self.multi_scale.jump_ratio_fast() > self.config.jump_ratio_threshold
    }

    /// Get current 4-state volatility regime.
    ///
    /// Regime classification with hysteresis:
    /// - Low: Quiet market (σ < 0.5 × baseline) - can tighten spreads
    /// - Normal: Standard conditions
    /// - High: Elevated volatility (σ > 1.5 × baseline) - widen spreads
    /// - Extreme: Crisis/toxic (σ > 3 × baseline OR high jump ratio) - consider pulling quotes
    pub fn volatility_regime(&self) -> VolatilityRegime {
        self.volatility_regime.regime()
    }

    /// Get spread multiplier based on current volatility regime.
    ///
    /// Ranges from 0.8 (Low) to 2.5 (Extreme).
    pub fn regime_spread_multiplier(&self) -> f64 {
        self.volatility_regime.regime().spread_multiplier()
    }

    /// Get gamma multiplier based on current volatility regime.
    ///
    /// Ranges from 0.8 (Low) to 3.0 (Extreme).
    pub fn regime_gamma_multiplier(&self) -> f64 {
        self.volatility_regime.regime().gamma_multiplier()
    }

    // === Directional Flow Accessors ===

    /// Get signed momentum in bps over momentum window.
    /// Negative = market falling, Positive = market rising.
    pub fn momentum_bps(&self) -> f64 {
        self.momentum.momentum_bps(self.current_time_ms)
    }

    /// Get falling knife score [0, 3].
    /// > 0.5 = some downward momentum
    /// > 1.0 = severe downward momentum (protect bids!)
    pub fn falling_knife_score(&self) -> f64 {
        self.momentum.falling_knife_score(self.current_time_ms)
    }

    /// Get rising knife score [0, 3].
    /// > 0.5 = some upward momentum
    /// > 1.0 = severe upward momentum (protect asks!)
    pub fn rising_knife_score(&self) -> f64 {
        self.momentum.rising_knife_score(self.current_time_ms)
    }

    /// Get trade flow imbalance [-1, 1].
    /// Negative = sell pressure, Positive = buy pressure.
    pub fn flow_imbalance(&self) -> f64 {
        self.flow.imbalance()
    }

    // === Probabilistic Momentum Model Accessors ===

    /// Get learned probability of momentum continuation.
    ///
    /// Returns P(next return has same sign as current momentum).
    /// Uses learned probabilities if calibrated, falls back to 0.5 prior otherwise.
    pub fn momentum_continuation_probability(&self) -> f64 {
        self.momentum_model
            .continuation_probability(self.last_momentum_bps)
    }

    /// Get bid protection factor based on learned momentum model.
    ///
    /// Returns multiplier > 1 if we should protect bids (falling market).
    /// Based on learned continuation probability and momentum magnitude.
    pub fn bid_protection_factor(&self) -> f64 {
        self.momentum_model
            .bid_protection_factor(self.last_momentum_bps)
    }

    /// Get ask protection factor based on learned momentum model.
    ///
    /// Returns multiplier > 1 if we should protect asks (rising market).
    /// Based on learned continuation probability and momentum magnitude.
    pub fn ask_protection_factor(&self) -> f64 {
        self.momentum_model
            .ask_protection_factor(self.last_momentum_bps)
    }

    /// Get overall momentum strength [0, 1] from learned model.
    ///
    /// Combines continuation probability with magnitude for a single
    /// "how strong is the directional signal" metric.
    pub fn momentum_strength(&self) -> f64 {
        self.momentum_model
            .momentum_strength(self.last_momentum_bps)
    }

    /// Check if momentum model is calibrated (enough observations).
    pub fn momentum_model_calibrated(&self) -> bool {
        self.momentum_model.is_calibrated()
    }

    // === Jump Process Accessors (First Principles Gap 1) ===

    /// Get jump intensity (λ) - expected jumps per second.
    pub fn lambda_jump(&self) -> f64 {
        self.jump_estimator.lambda()
    }

    /// Get mean jump size (μ_j) in log-return units.
    pub fn mu_jump(&self) -> f64 {
        self.jump_estimator.mu()
    }

    /// Get jump size standard deviation (σ_j).
    pub fn sigma_jump(&self) -> f64 {
        self.jump_estimator.sigma()
    }

    /// Get total variance including jumps over horizon.
    ///
    /// Var[P(t+h) - P(t)] = σ²h + λh×E[J²]
    pub fn total_variance(&self, horizon_secs: f64) -> f64 {
        let sigma_diffusion = self.multi_scale.sigma_clean();
        self.jump_estimator
            .total_variance(sigma_diffusion, horizon_secs)
    }

    /// Get total volatility including jumps (sqrt of total variance).
    pub fn total_sigma(&self, horizon_secs: f64) -> f64 {
        let sigma_diffusion = self.multi_scale.sigma_clean();
        self.jump_estimator
            .total_sigma(sigma_diffusion, horizon_secs)
    }

    /// Check if jump estimator has enough data.
    pub fn jump_estimator_warmed_up(&self) -> bool {
        self.jump_estimator.is_warmed_up()
    }

    // === Stochastic Volatility Accessors (First Principles Gap 2) ===

    /// Get current instantaneous volatility from stochastic vol model.
    pub fn sigma_stoch_vol(&self) -> f64 {
        self.stoch_vol.sigma_t()
    }

    /// Get volatility mean-reversion speed (κ_vol).
    pub fn kappa_vol(&self) -> f64 {
        self.stoch_vol.kappa()
    }

    /// Get long-run volatility (√θ_vol).
    pub fn theta_vol_sigma(&self) -> f64 {
        self.stoch_vol.theta_sigma()
    }

    /// Get vol-of-vol (ξ).
    pub fn xi_vol(&self) -> f64 {
        self.stoch_vol.xi()
    }

    /// Get price-vol correlation (ρ, typically negative - leverage effect).
    pub fn rho_price_vol(&self) -> f64 {
        self.stoch_vol.rho()
    }

    /// Get expected average volatility over horizon using OU dynamics.
    ///
    /// Accounts for mean-reversion: if σ > θ, vol will decrease toward θ.
    pub fn expected_avg_sigma(&self, horizon_secs: f64) -> f64 {
        self.stoch_vol.expected_avg_sigma(horizon_secs)
    }

    /// Get leverage-adjusted volatility based on recent return.
    ///
    /// When returns are negative and ρ < 0, volatility increases.
    pub fn leverage_adjusted_vol(&self, recent_return: f64) -> f64 {
        self.stoch_vol.leverage_adjusted_vol(recent_return)
    }

    /// Check if stochastic vol is calibrated.
    pub fn stoch_vol_calibrated(&self) -> bool {
        self.stoch_vol.is_calibrated()
    }

    // === Warmup ===

    /// Check if estimator has collected enough data.
    ///
    /// Uses market_kappa for warmup since it receives trade tape data
    /// immediately, while own_kappa needs actual fills to accumulate.
    pub fn is_warmed_up(&self) -> bool {
        self.multi_scale.tick_count() >= self.config.min_volume_ticks
            && self.market_kappa.update_count() >= self.config.min_trade_observations
    }

    /// Get confidence in sigma estimate (0.0 to 1.0).
    ///
    /// Confidence is based on how much data we've collected relative to
    /// minimum warmup requirements. Uses a smooth transition:
    /// - 0.0 when no data
    /// - 0.5 at minimum warmup threshold
    /// - Approaches 1.0 as data accumulates (3x warmup ≈ 0.95)
    ///
    /// This is used for Bayesian blending with the prior - low confidence
    /// means the prior dominates, high confidence means observations dominate.
    pub fn sigma_confidence(&self) -> f64 {
        let tick_count = self.multi_scale.tick_count();
        let min_ticks = self.config.min_volume_ticks.max(1);

        // Use a sigmoid-like function: confidence = 1 - exp(-ratio / scale)
        // At ratio = 1 (min warmup): confidence ≈ 0.63
        // At ratio = 2: confidence ≈ 0.86
        // At ratio = 3: confidence ≈ 0.95
        let ratio = tick_count as f64 / min_ticks as f64;
        1.0 - (-ratio).exp()
    }

    /// Get current warmup progress.
    /// Returns (volume_ticks, min_volume_ticks, trade_observations, min_trade_observations)
    ///
    /// Uses market_kappa for progress since own_kappa needs fills.
    pub fn warmup_progress(&self) -> (usize, usize, usize, usize) {
        (
            self.multi_scale.tick_count(),
            self.config.min_volume_ticks,
            self.market_kappa.update_count(),
            self.config.min_trade_observations,
        )
    }

    /// Get simplified warmup progress for legacy compatibility.
    /// Returns (current_samples, min_samples) based on volume ticks.
    pub fn warmup_progress_simple(&self) -> (usize, usize) {
        (self.multi_scale.tick_count(), self.config.min_volume_ticks)
    }

    // === Stochastic Module: Kalman Filter ===

    /// Get Kalman-filtered fair price (posterior mean).
    ///
    /// The Kalman filter denoises the mid price by separating true price
    /// movements from bid-ask bounce noise. Use this for:
    /// - Fair price base in microprice calculation
    /// - Position valuation (more stable than raw mid)
    pub fn kalman_fair_price(&self) -> f64 {
        self.kalman_filter.fair_price()
    }

    /// Get Kalman filter uncertainty (posterior standard deviation).
    ///
    /// Higher uncertainty means less confidence in the fair price estimate.
    /// Use this for spread widening when uncertain.
    pub fn kalman_uncertainty(&self) -> f64 {
        self.kalman_filter.uncertainty()
    }

    /// Get Kalman filter uncertainty in basis points.
    pub fn kalman_uncertainty_bps(&self) -> f64 {
        self.kalman_filter.uncertainty_bps()
    }

    /// Compute Kalman-based spread widening.
    ///
    /// Formula: spread_add = γ × σ_kalman × √T
    /// where σ_kalman is the Kalman filter uncertainty.
    ///
    /// Higher uncertainty → wider spreads (protecting against fair price misestimation).
    pub fn kalman_spread_widening(&self, gamma: f64, time_horizon: f64) -> f64 {
        self.kalman_filter.uncertainty_spread(gamma, time_horizon)
    }

    /// Check if Kalman filter is warmed up (enough observations).
    pub fn kalman_warmed_up(&self) -> bool {
        self.kalman_filter.is_warmed_up()
    }

    // === V2 Configuration ===

    /// Enable V2 feature flags.
    ///
    /// This activates the improved Bayesian estimation components:
    /// - Hierarchical kappa (market kappa as prior for own fills)
    /// - Soft jump classification (probabilistic P(jump))
    /// - Parameter covariance tracking ((κ, σ) correlation)
    pub fn set_v2_flags(&mut self, flags: EstimatorV2Flags) {
        self.v2_flags = flags;
    }

    /// Get current V2 feature flags.
    pub fn v2_flags(&self) -> EstimatorV2Flags {
        self.v2_flags
    }

    /// Configure microprice EMA smoothing.
    ///
    /// - `alpha`: Smoothing factor (0.0-1.0). 0.0 disables smoothing.
    /// - `min_change_bps`: Minimum change in bps to update EMA (noise filter).
    pub fn set_microprice_ema(&mut self, alpha: f64, min_change_bps: f64) {
        self.microprice_estimator.set_ema_config(alpha, min_change_bps);
    }

    // === V2 Hierarchical Kappa Accessors ===

    /// Get hierarchical kappa effective value (AS-adjusted posterior mean).
    ///
    /// This is the primary V2 kappa output. Uses market kappa as prior,
    /// then updates based on our actual fills with proper Bayesian conjugacy.
    pub fn hierarchical_kappa(&self) -> f64 {
        self.hierarchical_kappa.effective_kappa()
    }

    /// Get hierarchical kappa posterior standard deviation.
    pub fn hierarchical_kappa_std(&self) -> f64 {
        self.hierarchical_kappa.posterior_std()
    }

    /// Get hierarchical kappa 95% credible interval.
    pub fn hierarchical_kappa_ci_95(&self) -> (f64, f64) {
        self.hierarchical_kappa.credible_interval_95()
    }

    /// Get hierarchical kappa confidence [0, 1].
    pub fn hierarchical_kappa_confidence(&self) -> f64 {
        self.hierarchical_kappa.confidence()
    }

    /// Get adverse selection factor φ(AS) ∈ [0.5, 1.0].
    ///
    /// This is the multiplier applied to posterior kappa based on observed AS.
    /// Lower values mean more adverse selection detected → lower effective kappa.
    pub fn hierarchical_as_factor(&self) -> f64 {
        self.hierarchical_kappa.as_factor()
    }

    // === V2 Soft Jump Accessors ===

    /// Get soft toxicity score [0, 1] from mixture model.
    ///
    /// This is a probabilistic alternative to the binary is_toxic_regime().
    /// Higher values indicate more jump-like returns.
    pub fn soft_toxicity_score(&self) -> f64 {
        self.soft_jump.toxicity_score()
    }

    /// Get learned prior probability of jump.
    pub fn soft_jump_pi(&self) -> f64 {
        self.soft_jump.pi()
    }

    /// Get effective jump ratio from soft classification.
    ///
    /// Maps soft toxicity to a ratio for backwards compatibility.
    /// toxicity=0 → ratio=1.0, toxicity=0.5 → ratio=3.0
    pub fn soft_jump_ratio(&self) -> f64 {
        self.soft_jump.effective_jump_ratio()
    }

    /// Check if soft jump classifier is calibrated.
    pub fn soft_jump_calibrated(&self) -> bool {
        self.soft_jump.is_calibrated()
    }

    // === V2 Parameter Covariance Accessors ===

    /// Get (κ, σ) correlation coefficient.
    ///
    /// Positive correlation means κ and σ tend to move together.
    /// Negative correlation (common during crises) means high vol = low fill rate.
    pub fn kappa_sigma_correlation(&self) -> f64 {
        self.param_covariance.correlation()
    }

    /// Get spread uncertainty from parameter covariance.
    ///
    /// Uses delta method to propagate kappa uncertainty to spread uncertainty.
    pub fn spread_uncertainty(&self, gamma: f64) -> f64 {
        self.param_covariance.spread_uncertainty(gamma)
    }

    /// Check if parameter covariance is valid (enough observations).
    pub fn param_covariance_valid(&self) -> bool {
        self.param_covariance.is_valid()
    }

    // === V2 Blended Kappa (Optional Use) ===

    /// Get blended kappa using hierarchical model.
    ///
    /// Returns hierarchical kappa blended with standard kappa based on
    /// model confidence. As fills accumulate, the hierarchical estimate
    /// dominates.
    pub fn kappa_v2_aware(&self) -> f64 {
        let hier_conf = self.hierarchical_kappa.confidence();
        let hier_kappa = self.hierarchical_kappa.effective_kappa();
        let std_kappa = self.kappa(); // Already has floor applied

        // Blend based on hierarchical confidence
        // At 0 confidence: use standard kappa
        // At 1 confidence: use hierarchical kappa
        let blended = hier_conf * hier_kappa + (1.0 - hier_conf) * std_kappa;

        // Apply floor for competitive spread control on illiquid assets
        // (redundant when std_kappa dominates, but ensures hier_kappa respects floor too)
        self.apply_kappa_floor(blended)
    }
}

// ============================================================================
// MarketEstimator Implementation for ParameterEstimator
// ============================================================================

impl MarketEstimator for ParameterEstimator {
    fn sigma_clean(&self) -> f64 {
        self.sigma_clean()
    }
    fn sigma_total(&self) -> f64 {
        self.sigma_total()
    }
    fn sigma_effective(&self) -> f64 {
        self.sigma_effective()
    }
    fn sigma_leverage_adjusted(&self) -> f64 {
        self.sigma_leverage_adjusted()
    }
    fn volatility_regime(&self) -> VolatilityRegime {
        self.volatility_regime()
    }
    fn kappa(&self) -> f64 {
        self.kappa()
    }
    fn kappa_bid(&self) -> f64 {
        self.kappa_bid()
    }
    fn kappa_ask(&self) -> f64 {
        self.kappa_ask()
    }
    fn is_heavy_tailed(&self) -> bool {
        // Check if any kappa estimator shows heavy-tailed behavior
        // Use market kappa since it has more observations during warmup
        self.market_kappa.is_heavy_tailed() || self.own_kappa.is_heavy_tailed()
    }
    fn kappa_cv(&self) -> f64 {
        self.kappa_cv()
    }
    fn arrival_intensity(&self) -> f64 {
        self.arrival_intensity()
    }
    fn liquidity_gamma_multiplier(&self) -> f64 {
        self.liquidity_gamma_multiplier()
    }
    fn is_toxic_regime(&self) -> bool {
        self.is_toxic_regime()
    }
    fn jump_ratio(&self) -> f64 {
        self.jump_ratio()
    }
    fn momentum_bps(&self) -> f64 {
        self.momentum_bps()
    }
    fn flow_imbalance(&self) -> f64 {
        self.flow_imbalance()
    }
    fn falling_knife_score(&self) -> f64 {
        self.falling_knife_score()
    }
    fn rising_knife_score(&self) -> f64 {
        self.rising_knife_score()
    }
    fn momentum_continuation_probability(&self) -> f64 {
        self.momentum_continuation_probability()
    }
    fn bid_protection_factor(&self) -> f64 {
        self.bid_protection_factor()
    }
    fn ask_protection_factor(&self) -> f64 {
        self.ask_protection_factor()
    }
    fn momentum_strength(&self) -> f64 {
        self.momentum_strength()
    }
    fn momentum_model_calibrated(&self) -> bool {
        self.momentum_model_calibrated()
    }
    fn book_imbalance(&self) -> f64 {
        self.book_imbalance()
    }
    fn microprice(&self) -> f64 {
        self.microprice()
    }
    fn beta_book(&self) -> f64 {
        self.beta_book()
    }
    fn beta_flow(&self) -> f64 {
        self.beta_flow()
    }
    fn lambda_jump(&self) -> f64 {
        self.lambda_jump()
    }
    fn mu_jump(&self) -> f64 {
        self.mu_jump()
    }
    fn sigma_jump(&self) -> f64 {
        self.sigma_jump()
    }
    fn kappa_vol(&self) -> f64 {
        self.kappa_vol()
    }
    fn theta_vol_sigma(&self) -> f64 {
        self.theta_vol_sigma()
    }
    fn xi_vol(&self) -> f64 {
        self.xi_vol()
    }
    fn rho_price_vol(&self) -> f64 {
        self.rho_price_vol()
    }
    fn is_warmed_up(&self) -> bool {
        self.is_warmed_up()
    }
    fn sigma_confidence(&self) -> f64 {
        self.sigma_confidence()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::market_maker::estimator::kalman::KalmanPriceFilter;
    use crate::market_maker::estimator::kappa::BayesianKappaEstimator;
    use crate::market_maker::estimator::volatility::SingleScaleBipower;
    use crate::market_maker::estimator::volume::VolumeBucketAccumulator;

    fn make_config() -> EstimatorConfig {
        EstimatorConfig {
            initial_bucket_volume: 1.0,
            min_volume_ticks: 5,
            min_trade_observations: 3,
            fast_half_life_ticks: 5.0,
            medium_half_life_ticks: 10.0,
            slow_half_life_ticks: 50.0,
            kappa_half_life_updates: 10.0,
            ..Default::default()
        }
    }

    #[test]
    fn test_volume_bucket_accumulation() {
        let config = EstimatorConfig {
            initial_bucket_volume: 10.0,
            ..Default::default()
        };
        let mut acc = VolumeBucketAccumulator::new(&config);

        // Should not complete bucket yet
        assert!(acc.on_trade(1000, 100.0, 3.0).is_none());
        assert!(acc.on_trade(2000, 101.0, 3.0).is_none());

        // Should complete bucket at 10+ volume
        let bucket = acc.on_trade(3000, 102.0, 5.0);
        assert!(bucket.is_some());

        let b = bucket.unwrap();
        assert!((b.volume - 11.0).abs() < 0.01);
        // VWAP = (100*3 + 101*3 + 102*5) / 11 = 1113/11 ≈ 101.18
        assert!((b.vwap - 101.18).abs() < 0.1);
    }

    #[test]
    fn test_single_scale_bipower_no_jumps() {
        let mut bv = SingleScaleBipower::new(10.0, 0.001_f64.powi(2));

        // Feed stable returns (no jumps) - small oscillations
        let vwaps: [f64; 8] = [100.0, 100.1, 100.0, 100.1, 100.0, 100.1, 100.0, 100.1];
        let mut last_vwap: f64 = vwaps[0];
        for vwap in vwaps.iter().skip(1) {
            let log_return = (vwap / last_vwap).ln();
            bv.update(log_return);
            last_vwap = *vwap;
        }

        // Jump ratio should be close to 1.0 (no jumps)
        let ratio = bv.jump_ratio();
        assert!(
            ratio > 0.5 && ratio < 2.0,
            "Expected ratio ~1.0 for no jumps, got {}",
            ratio
        );
    }

    #[test]
    fn test_bayesian_kappa_prior_dominates_with_no_data() {
        // With no data, posterior mean should equal prior mean
        let prior_mean = 500.0;
        let prior_strength = 10.0;
        let kappa = BayesianKappaEstimator::new(prior_mean, prior_strength, 300_000);

        assert!(
            (kappa.posterior_mean() - prior_mean).abs() < 1e-6,
            "Posterior should equal prior with no data, got {}",
            kappa.posterior_mean()
        );
    }

    #[test]
    fn test_regime_detection() {
        let mut config = make_config();
        config.jump_ratio_threshold = 2.0;
        let estimator = ParameterEstimator::new(config);

        // Initially not toxic (default ratio = 1.0)
        assert!(!estimator.is_toxic_regime());
    }

    #[test]
    fn test_full_pipeline_warmup() {
        let config = make_config();
        let mut estimator = ParameterEstimator::new(config);

        assert!(!estimator.is_warmed_up());

        // Feed initial L2 book to set current_mid (needed for fill-rate kappa)
        let bids = vec![(99.9, 5.0), (99.8, 10.0), (99.7, 15.0)];
        let asks = vec![(100.1, 5.0), (100.2, 10.0), (100.3, 15.0)];
        estimator.on_l2_book(&bids, &asks, 100.0);

        // Feed trades to fill buckets (need 5 volume ticks)
        let mut time = 1000u64;
        for i in 0..100 {
            let price = 100.0 + (i as f64 * 0.1).sin() * 0.5;
            let is_buy = i % 2 == 0;
            estimator.on_trade(time, price, 0.5, Some(is_buy));
            time += 100;
        }

        // Feed more L2 books (for book structure analysis)
        for _ in 0..5 {
            estimator.on_l2_book(&bids, &asks, 100.0);
        }

        // Should be warmed up
        assert!(estimator.is_warmed_up());

        // Params should be in reasonable ranges
        let sigma = estimator.sigma();
        let kappa = estimator.kappa();
        let ratio = estimator.jump_ratio();

        assert!(sigma > 0.0, "sigma should be positive");
        assert!(kappa > 1.0, "kappa should be > 1");
        assert!(ratio > 0.0, "jump_ratio should be positive");
    }

    #[test]
    fn test_kalman_filter_basic() {
        let mut filter = KalmanPriceFilter::new(100.0, 1e-6, 1e-8, 2.5e-9);

        // Filter should initialize properly
        assert_eq!(filter.update_count(), 0);
        assert!(!filter.is_warmed_up());

        // First observation sets the mean
        filter.update(100.0);
        assert!((filter.fair_price() - 100.0).abs() < 0.001);
        assert_eq!(filter.update_count(), 1);

        // Second observation updates the mean
        filter.filter(100.1);
        assert!(filter.fair_price() > 100.0);
        assert!(filter.fair_price() < 100.1); // Should be smoothed
    }
}
