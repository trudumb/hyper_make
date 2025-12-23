// ============================================================================
// PROPOSED CHANGES TO src/market_maker/estimator.rs
// ============================================================================

// Add to EstimatorConfig:

impl Default for EstimatorConfig {
    fn default() -> Self {
        Self {
            // ... existing fields ...
            
            // NEW: Multi-timescale half-lives
            fast_half_life_ticks: 5.0,    // ~2 seconds - reacts to crashes
            medium_half_life_ticks: 20.0, // ~10 seconds
            slow_half_life_ticks: 100.0,  // ~60 seconds - baseline
            
            // NEW: Momentum detection
            momentum_window_ms: 500,       // Track signed returns over 500ms
            
            // NEW: Trade flow tracking
            trade_flow_window_ms: 1000,    // Track buy/sell imbalance over 1s
            trade_flow_alpha: 0.1,         // EWMA smoothing for flow
        }
    }
}

// ============================================================================
// NEW: Multi-Timescale Bipower Estimator
// ============================================================================

/// Tracks RV and BV at a single timescale
#[derive(Debug)]
struct SingleScaleBipower {
    alpha: f64,
    rv: f64,  // Realized variance (EWMA of r²)
    bv: f64,  // Bipower variation (EWMA of π/2 × |r_t| × |r_{t-1}|)
    last_abs_return: Option<f64>,
}

impl SingleScaleBipower {
    fn new(half_life_ticks: f64, default_var: f64) -> Self {
        Self {
            alpha: (2.0_f64.ln() / half_life_ticks).clamp(0.001, 1.0),
            rv: default_var,
            bv: default_var,
            last_abs_return: None,
        }
    }
    
    fn update(&mut self, log_return: f64) {
        let abs_return = log_return.abs();
        
        // RV: EWMA of r²
        let rv_obs = log_return.powi(2);
        self.rv = self.alpha * rv_obs + (1.0 - self.alpha) * self.rv;
        
        // BV: EWMA of π/2 × |r_t| × |r_{t-1}|
        if let Some(last_abs) = self.last_abs_return {
            let bv_obs = std::f64::consts::FRAC_PI_2 * abs_return * last_abs;
            self.bv = self.alpha * bv_obs + (1.0 - self.alpha) * self.bv;
        }
        
        self.last_abs_return = Some(abs_return);
    }
    
    fn sigma_total(&self) -> f64 {
        self.rv.sqrt().clamp(1e-7, 0.05)
    }
    
    fn sigma_clean(&self) -> f64 {
        self.bv.sqrt().clamp(1e-7, 0.05)
    }
    
    fn jump_ratio(&self) -> f64 {
        if self.bv > 1e-12 {
            (self.rv / self.bv).clamp(0.1, 100.0)
        } else {
            1.0
        }
    }
}

/// Multi-timescale volatility with fast/medium/slow components
#[derive(Debug)]
pub struct MultiScaleBipowerEstimator {
    fast: SingleScaleBipower,   // ~5 ticks / 2 seconds
    medium: SingleScaleBipower, // ~20 ticks / 10 seconds  
    slow: SingleScaleBipower,   // ~100 ticks / 60 seconds
    tick_count: usize,
}

impl MultiScaleBipowerEstimator {
    pub fn new(config: &EstimatorConfig) -> Self {
        let default_var = config.default_sigma.powi(2);
        Self {
            fast: SingleScaleBipower::new(config.fast_half_life_ticks, default_var),
            medium: SingleScaleBipower::new(config.medium_half_life_ticks, default_var),
            slow: SingleScaleBipower::new(config.slow_half_life_ticks, default_var),
            tick_count: 0,
        }
    }
    
    pub fn on_bucket(&mut self, bucket: &VolumeBucket, last_vwap: Option<f64>) {
        if let Some(prev) = last_vwap {
            if bucket.vwap > 0.0 && prev > 0.0 {
                let log_return = (bucket.vwap / prev).ln();
                self.fast.update(log_return);
                self.medium.update(log_return);
                self.slow.update(log_return);
                self.tick_count += 1;
            }
        }
    }
    
    /// Clean sigma (BV-based) for spread pricing
    /// Uses slow timescale for stability
    pub fn sigma_clean(&self) -> f64 {
        self.slow.sigma_clean()
    }
    
    /// Total sigma (RV-based) for risk assessment
    /// Blends fast + slow: uses fast when market is accelerating
    pub fn sigma_total(&self) -> f64 {
        let fast = self.fast.sigma_total();
        let slow = self.slow.sigma_total();
        
        // If fast >> slow, market is accelerating - trust fast more
        let ratio = fast / slow.max(1e-9);
        
        if ratio > 1.5 {
            // Acceleration: blend toward fast
            let weight = ((ratio - 1.0) / 3.0).clamp(0.0, 0.7);
            weight * fast + (1.0 - weight) * slow
        } else {
            // Stable: prefer slow for less noise
            0.2 * fast + 0.8 * slow
        }
    }
    
    /// Effective sigma for inventory skew
    /// Blends clean and total based on jump regime
    pub fn sigma_effective(&self) -> f64 {
        let clean = self.sigma_clean();
        let total = self.sigma_total();
        let jump_ratio = self.jump_ratio_fast();
        
        // At ratio=1: pure clean (no jumps)
        // At ratio=3: 67% total (jumps dominant)
        // At ratio=5: 80% total
        let jump_weight = 1.0 - (1.0 / jump_ratio.max(1.0));
        let jump_weight = jump_weight.clamp(0.0, 0.85);
        
        (1.0 - jump_weight) * clean + jump_weight * total
    }
    
    /// Fast jump ratio (detects recent jumps quickly)
    pub fn jump_ratio_fast(&self) -> f64 {
        self.fast.jump_ratio()
    }
    
    /// Medium jump ratio (more stable signal)
    pub fn jump_ratio_medium(&self) -> f64 {
        self.medium.jump_ratio()
    }
    
    pub fn tick_count(&self) -> usize {
        self.tick_count
    }
}

// ============================================================================
// NEW: Momentum Detector (Signed Directional Flow)
// ============================================================================

use std::collections::VecDeque;

/// Detects directional momentum from signed returns
#[derive(Debug)]
pub struct MomentumDetector {
    /// Recent (timestamp_ms, log_return) pairs
    returns: VecDeque<(u64, f64)>,
    /// Window for momentum calculation
    window_ms: u64,
}

impl MomentumDetector {
    pub fn new(window_ms: u64) -> Self {
        Self {
            returns: VecDeque::with_capacity(100),
            window_ms,
        }
    }
    
    /// Add a new VWAP-based return
    pub fn on_bucket(&mut self, end_time_ms: u64, log_return: f64) {
        self.returns.push_back((end_time_ms, log_return));
        
        // Expire old returns
        let cutoff = end_time_ms.saturating_sub(self.window_ms * 2); // Keep 2x window
        while self.returns.front().map(|(t, _)| *t < cutoff).unwrap_or(false) {
            self.returns.pop_front();
        }
    }
    
    /// Signed momentum in bps over the configured window
    pub fn momentum_bps(&self, now_ms: u64) -> f64 {
        let cutoff = now_ms.saturating_sub(self.window_ms);
        let sum: f64 = self.returns.iter()
            .filter(|(t, _)| *t >= cutoff)
            .map(|(_, r)| r)
            .sum();
        sum * 10_000.0  // Convert to bps
    }
    
    /// Falling knife score: 0 = normal, 1+ = severe downward momentum
    pub fn falling_knife_score(&self, now_ms: u64) -> f64 {
        let momentum = self.momentum_bps(now_ms);
        
        // Only trigger on negative momentum
        if momentum >= 0.0 {
            return 0.0;
        }
        
        // Score: -20 bps = 1.0, -40 bps = 2.0, etc.
        (momentum.abs() / 20.0).clamp(0.0, 3.0)
    }
    
    /// Rising knife score (for protecting asks during pumps)
    pub fn rising_knife_score(&self, now_ms: u64) -> f64 {
        let momentum = self.momentum_bps(now_ms);
        
        if momentum <= 0.0 {
            return 0.0;
        }
        
        (momentum / 20.0).clamp(0.0, 3.0)
    }
}

// ============================================================================
// NEW: Trade Flow Imbalance Tracker
// ============================================================================

/// Tracks buy vs sell aggressor imbalance from trade tape
#[derive(Debug)]
pub struct TradeFlowTracker {
    /// (timestamp_ms, signed_volume): positive = buy aggressor
    trades: VecDeque<(u64, f64)>,
    /// Rolling window
    window_ms: u64,
    /// EWMA smoothed imbalance
    ewma_imbalance: f64,
    /// EWMA alpha
    alpha: f64,
}

impl TradeFlowTracker {
    pub fn new(window_ms: u64, alpha: f64) -> Self {
        Self {
            trades: VecDeque::with_capacity(500),
            window_ms,
            ewma_imbalance: 0.0,
            alpha,
        }
    }
    
    /// Add a trade from the tape
    /// is_buy_aggressor: true if buyer was taker (lifted the ask)
    pub fn on_trade(&mut self, timestamp_ms: u64, size: f64, is_buy_aggressor: bool) {
        let signed = if is_buy_aggressor { size } else { -size };
        self.trades.push_back((timestamp_ms, signed));
        
        // Expire old trades
        let cutoff = timestamp_ms.saturating_sub(self.window_ms);
        while self.trades.front().map(|(t, _)| *t < cutoff).unwrap_or(false) {
            self.trades.pop_front();
        }
        
        // Update EWMA
        let instant = self.compute_instant_imbalance();
        self.ewma_imbalance = self.alpha * instant + (1.0 - self.alpha) * self.ewma_imbalance;
    }
    
    /// Compute instantaneous imbalance: (buy - sell) / total
    fn compute_instant_imbalance(&self) -> f64 {
        let (buy_vol, sell_vol) = self.trades.iter().fold((0.0, 0.0), |(b, s), (_, v)| {
            if *v > 0.0 { (b + v, s) } else { (b, s - v) }
        });
        let total = buy_vol + sell_vol;
        if total < 1e-12 { 0.0 } else { (buy_vol - sell_vol) / total }
    }
    
    /// Smoothed flow imbalance [-1, 1]
    /// Negative = sell pressure, Positive = buy pressure
    pub fn imbalance(&self) -> f64 {
        self.ewma_imbalance.clamp(-1.0, 1.0)
    }
    
    /// Is there dominant selling (for bid protection)?
    pub fn is_sell_pressure(&self) -> bool {
        self.ewma_imbalance < -0.25
    }
    
    /// Is there dominant buying (for ask protection)?
    pub fn is_buy_pressure(&self) -> bool {
        self.ewma_imbalance > 0.25
    }
}

// ============================================================================
// REVISED: MarketParams with new fields
// ============================================================================

/// Parameters estimated from live market data.
#[derive(Debug, Clone, Copy)]
pub struct MarketParams {
    // === Volatility (per-second, NOT annualized) ===
    
    /// Clean volatility (√BV) - for base spread pricing
    /// Robust to jumps, measures continuous diffusion
    pub sigma_clean: f64,
    
    /// Total volatility (√RV) - includes jumps
    pub sigma_total: f64,
    
    /// Effective sigma for inventory skew
    /// Blends clean and total based on regime
    pub sigma_effective: f64,
    
    // === Order Book ===
    
    /// Order book depth decay (κ)
    pub kappa: f64,
    
    /// Volume tick arrival intensity
    pub arrival_intensity: f64,
    
    // === Regime Detection ===
    
    /// RV/BV jump ratio (1.0 = normal, >2.0 = jumps)
    pub jump_ratio: f64,
    
    /// Is market in toxic regime?
    pub is_toxic_regime: bool,
    
    // === NEW: Directional Flow ===
    
    /// Signed momentum over 500ms (in bps)
    /// Negative = market falling
    pub momentum_bps: f64,
    
    /// Order flow imbalance [-1, 1]
    /// Negative = sell pressure dominant
    pub flow_imbalance: f64,
    
    /// Falling knife score [0, 3]
    /// >1.0 = dangerous downward momentum
    pub falling_knife_score: f64,
    
    /// Rising knife score [0, 3]
    /// >1.0 = dangerous upward momentum
    pub rising_knife_score: f64,
}

impl Default for MarketParams {
    fn default() -> Self {
        Self {
            sigma_clean: 0.0001,
            sigma_total: 0.0001,
            sigma_effective: 0.0001,
            kappa: 100.0,
            arrival_intensity: 0.5,
            jump_ratio: 1.0,
            is_toxic_regime: false,
            momentum_bps: 0.0,
            flow_imbalance: 0.0,
            falling_knife_score: 0.0,
            rising_knife_score: 0.0,
        }
    }
}

// ============================================================================
// REVISED: ParameterEstimator with new components
// ============================================================================

pub struct ParameterEstimator {
    config: EstimatorConfig,
    
    /// Volume bucket accumulator (volume clock)
    bucket_accumulator: VolumeBucketAccumulator,
    
    /// Multi-timescale bipower estimator (replaces single BipowerVariationEstimator)
    multi_scale: MultiScaleBipowerEstimator,
    
    /// Momentum detector (NEW)
    momentum: MomentumDetector,
    
    /// Trade flow tracker (NEW)
    flow: TradeFlowTracker,
    
    /// Weighted kappa estimator
    kappa: WeightedKappaEstimator,
    
    /// Volume tick arrival estimator
    arrival: VolumeTickArrivalEstimator,
    
    /// Last VWAP for return calculation
    last_vwap: Option<f64>,
    
    /// Current timestamp for momentum queries
    current_time_ms: u64,
}

impl ParameterEstimator {
    pub fn new(config: EstimatorConfig) -> Self {
        let bucket_accumulator = VolumeBucketAccumulator::new(&config);
        let multi_scale = MultiScaleBipowerEstimator::new(&config);
        let momentum = MomentumDetector::new(config.momentum_window_ms);
        let flow = TradeFlowTracker::new(config.trade_flow_window_ms, config.trade_flow_alpha);
        let kappa = WeightedKappaEstimator::new(
            config.kappa_half_life_updates,
            config.default_kappa,
            config.kappa_max_distance,
            config.kappa_max_levels,
        );
        let arrival = VolumeTickArrivalEstimator::new(
            config.medium_half_life_ticks, // Use medium timescale
            config.default_arrival_intensity,
        );

        Self {
            config,
            bucket_accumulator,
            multi_scale,
            momentum,
            flow,
            kappa,
            arrival,
            last_vwap: None,
            current_time_ms: 0,
        }
    }

    /// Process a new trade (feeds into volume clock AND flow tracker)
    /// 
    /// # Arguments
    /// * `timestamp_ms` - Trade timestamp
    /// * `price` - Trade price
    /// * `size` - Trade size
    /// * `is_buy_aggressor` - Whether buyer was the taker (if available from exchange)
    pub fn on_trade(&mut self, timestamp_ms: u64, price: f64, size: f64, is_buy_aggressor: Option<bool>) {
        self.current_time_ms = timestamp_ms;
        
        // Track trade flow if we know aggressor side
        if let Some(is_buy) = is_buy_aggressor {
            self.flow.on_trade(timestamp_ms, size, is_buy);
        }
        
        // Feed into volume bucket accumulator
        if let Some(bucket) = self.bucket_accumulator.on_trade(timestamp_ms, price, size) {
            // Bucket completed - compute return
            if let Some(prev_vwap) = self.last_vwap {
                if bucket.vwap > 0.0 && prev_vwap > 0.0 {
                    let log_return = (bucket.vwap / prev_vwap).ln();
                    
                    // Update multi-scale volatility
                    self.multi_scale.on_bucket(&bucket, Some(prev_vwap));
                    
                    // Update momentum detector
                    self.momentum.on_bucket(bucket.end_time_ms, log_return);
                }
            }
            
            self.last_vwap = Some(bucket.vwap);
            self.arrival.on_bucket(&bucket);
        }
    }
    
    /// Get current market parameters (call after processing recent data)
    pub fn market_params(&self) -> MarketParams {
        let jump_ratio = self.multi_scale.jump_ratio_fast();
        
        MarketParams {
            sigma_clean: self.multi_scale.sigma_clean(),
            sigma_total: self.multi_scale.sigma_total(),
            sigma_effective: self.multi_scale.sigma_effective(),
            kappa: self.kappa.kappa(),
            arrival_intensity: self.arrival.ticks_per_second(),
            jump_ratio,
            is_toxic_regime: jump_ratio > self.config.jump_ratio_threshold,
            momentum_bps: self.momentum.momentum_bps(self.current_time_ms),
            flow_imbalance: self.flow.imbalance(),
            falling_knife_score: self.momentum.falling_knife_score(self.current_time_ms),
            rising_knife_score: self.momentum.rising_knife_score(self.current_time_ms),
        }
    }
    
    // ... existing L2 and accessor methods ...
}

// ============================================================================
// REVISED: GLFTStrategy with directional protection
// ============================================================================

impl QuotingStrategy for GLFTStrategy {
    fn calculate_quotes(
        &self,
        config: &QuoteConfig,
        position: f64,
        max_position: f64,
        target_liquidity: f64,
        market_params: &MarketParams,
    ) -> (Option<Quote>, Option<Quote>) {
        // 1. Use sigma_clean for base half-spread (continuous pricing)
        let sigma_for_spread = market_params.sigma_clean;
        let kappa = market_params.kappa;
        
        let target_half_spread = config.half_spread_bps as f64 / 10000.0;
        
        // Derive gamma for spread calculation
        let gamma_spread = self.derive_gamma(target_half_spread, kappa, sigma_for_spread, config.max_position);
        let half_spread = self.half_spread(gamma_spread, kappa);
        
        // 2. Use sigma_effective for skew (reacts to jumps properly)
        let sigma_for_skew = market_params.sigma_effective;
        let gamma_skew = self.derive_gamma(target_half_spread, kappa, sigma_for_skew, config.max_position);
        
        let inventory_ratio = if max_position > EPSILON {
            (position / max_position).clamp(-1.0, 1.0)
        } else {
            0.0
        };
        
        let base_skew = self.inventory_skew(inventory_ratio, sigma_for_skew, gamma_skew, kappa);
        
        // 3. Toxic regime spread widening
        let toxicity_multiplier = if market_params.is_toxic_regime {
            (market_params.jump_ratio / 2.0).clamp(1.0, 2.5)
        } else {
            1.0
        };
        
        // 4. NEW: Directional protection (falling knife / rising knife)
        let mut bid_protection = 0.0;
        let mut ask_protection = 0.0;
        
        // Falling knife: protect bids (don't get run over by sellers)
        if market_params.falling_knife_score > 0.5 {
            bid_protection = market_params.falling_knife_score * half_spread * 0.5;
            
            // Extra protection if we're already long
            if inventory_ratio > 0.0 {
                bid_protection *= 1.0 + inventory_ratio;
            }
        }
        
        // Rising knife: protect asks (don't get run over by buyers)
        if market_params.rising_knife_score > 0.5 {
            ask_protection = market_params.rising_knife_score * half_spread * 0.5;
            
            // Extra protection if we're already short
            if inventory_ratio < 0.0 {
                ask_protection *= 1.0 + inventory_ratio.abs();
            }
        }
        
        // 5. NEW: Flow imbalance adjustment (anticipatory)
        // If heavy selling, shift quotes down slightly (anticipate continued decline)
        let flow_adjustment = market_params.flow_imbalance * half_spread * 0.2;
        
        // 6. Combine all adjustments
        let bid_delta = (half_spread + base_skew + bid_protection - flow_adjustment) * toxicity_multiplier;
        let ask_delta = ((half_spread - base_skew + ask_protection - flow_adjustment).max(0.0)) * toxicity_multiplier;
        
        debug!(
            sigma_clean = %format!("{:.6}", market_params.sigma_clean),
            sigma_effective = %format!("{:.6}", market_params.sigma_effective),
            momentum_bps = %format!("{:.1}", market_params.momentum_bps),
            flow = %format!("{:.2}", market_params.flow_imbalance),
            falling_knife = %format!("{:.2}", market_params.falling_knife_score),
            bid_protection = %format!("{:.6}", bid_protection),
            "GLFT with directional protection"
        );
        
        // ... rest of price calculation unchanged ...
    }
}
