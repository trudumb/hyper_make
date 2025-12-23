//! Configuration for econometric parameter estimation.

/// Configuration for econometric parameter estimation pipeline.
#[derive(Debug, Clone)]
pub struct EstimatorConfig {
    // === Volume Clock ===
    /// Initial volume bucket threshold (in asset units, e.g., 1.0 BTC)
    pub initial_bucket_volume: f64,
    /// Rolling window for adaptive bucket calculation (seconds)
    pub volume_window_secs: f64,
    /// Percentile of rolling volume for adaptive bucket (e.g., 0.01 = 1%)
    pub volume_percentile: f64,
    /// Minimum bucket volume floor
    pub min_bucket_volume: f64,
    /// Maximum bucket volume ceiling
    pub max_bucket_volume: f64,

    // === Multi-Timescale EWMA ===
    /// Fast half-life for variance EWMA (~2 seconds, reacts to crashes)
    pub fast_half_life_ticks: f64,
    /// Medium half-life for variance EWMA (~10 seconds)
    pub medium_half_life_ticks: f64,
    /// Slow half-life for variance EWMA (~60 seconds, baseline)
    pub slow_half_life_ticks: f64,
    /// Half-life for kappa EWMA (in L2 updates)
    pub kappa_half_life_updates: f64,

    // === Momentum Detection ===
    /// Window for momentum calculation (milliseconds)
    pub momentum_window_ms: u64,

    // === Trade Flow Tracking ===
    /// Window for trade flow imbalance calculation (milliseconds)
    pub trade_flow_window_ms: u64,
    /// EWMA alpha for smoothing flow imbalance
    pub trade_flow_alpha: f64,

    // === Regime Detection ===
    /// Jump detection threshold: RV/BV ratio > threshold = toxic regime
    pub jump_ratio_threshold: f64,

    // === Kappa Estimation ===
    /// Maximum distance from mid for kappa regression (as fraction, e.g., 0.01 = 1%)
    pub kappa_max_distance: f64,
    /// Maximum number of L2 levels to use per side
    pub kappa_max_levels: usize,

    // === Warmup ===
    /// Minimum volume ticks before volatility estimates are valid
    pub min_volume_ticks: usize,
    /// Minimum L2 updates before kappa is valid
    pub min_l2_updates: usize,

    // === Defaults ===
    /// Default sigma during warmup (per-second volatility)
    pub default_sigma: f64,
    /// Default kappa during warmup (order book decay)
    pub default_kappa: f64,
    /// Default arrival intensity during warmup (ticks per second)
    pub default_arrival_intensity: f64,
}

impl Default for EstimatorConfig {
    fn default() -> Self {
        Self {
            // Volume Clock - tuned for testnet/low-activity markets
            initial_bucket_volume: 0.01, // Start very small (0.01 BTC)
            volume_window_secs: 300.0,   // 5 minutes
            volume_percentile: 0.01,     // 1% of 5-min volume
            min_bucket_volume: 0.001,    // Floor at 0.001 BTC
            max_bucket_volume: 10.0,     // Cap at 10 BTC

            // Multi-Timescale EWMA
            fast_half_life_ticks: 5.0,     // ~2 seconds - reacts to crashes
            medium_half_life_ticks: 20.0,  // ~10 seconds
            slow_half_life_ticks: 100.0,   // ~60 seconds - baseline
            kappa_half_life_updates: 30.0, // 30 L2 updates

            // Momentum Detection
            momentum_window_ms: 500, // Track signed returns over 500ms

            // Trade Flow Tracking
            trade_flow_window_ms: 1000, // Track buy/sell imbalance over 1s
            trade_flow_alpha: 0.1,      // EWMA smoothing for flow

            // Regime Detection - LOWERED from 3.0 for earlier detection
            jump_ratio_threshold: 1.5, // RV/BV > 1.5 = toxic

            // Kappa
            kappa_max_distance: 0.01, // 1% from mid
            kappa_max_levels: 15,

            // Warmup - reasonable for testnet/low-activity
            min_volume_ticks: 10,
            min_l2_updates: 5,

            // Defaults
            default_sigma: 0.0001,          // 0.01% per-second
            default_kappa: 100.0,           // Moderate depth decay
            default_arrival_intensity: 0.5, // 0.5 ticks per second
        }
    }
}

impl EstimatorConfig {
    /// Create config from legacy fields (for backward compatibility with bin/market_maker.rs)
    pub fn from_legacy(
        _window_ms: u64,
        _min_trades: usize,
        default_sigma: f64,
        default_kappa: f64,
        default_arrival_intensity: f64,
        _decay_secs: u64,
        min_warmup_trades: usize,
    ) -> Self {
        Self {
            default_sigma,
            default_kappa,
            default_arrival_intensity,
            min_volume_ticks: min_warmup_trades.max(10),
            ..Default::default()
        }
    }

    /// Create config with custom toxic threshold
    pub fn with_toxic_threshold(mut self, threshold: f64) -> Self {
        self.jump_ratio_threshold = threshold;
        self
    }
}
