//! Microstructure-Based Feature Extraction for Adverse Selection Prediction
//!
//! This module implements statistically grounded features based on market microstructure
//! theory. Each feature is designed to detect informed trading activity.
//!
//! # Theoretical Foundations
//!
//! 1. **Kyle (1985)**: Price impact (λ) measures information content per trade
//! 2. **Easley-O'Hara (1992)**: Trade arrival patterns reveal information events
//! 3. **Hasbrouck (1991)**: Cross-market information shares
//! 4. **Glosten-Milgrom (1985)**: Spread widening signals adverse selection
//!
//! # Feature Design Principles
//!
//! - All features normalized to [-1, 1] or z-scores for comparability
//! - Online computation with O(1) update complexity
//! - Statistical significance built-in (compare to null distribution)
//! - Uncorrelated features for diverse signal

use std::collections::VecDeque;

/// Configuration for microstructure feature extraction
#[derive(Debug, Clone)]
pub struct MicrostructureConfig {
    /// Window for intensity estimation (trades)
    pub intensity_window: usize,
    /// Window for price impact estimation (trades)
    pub impact_window: usize,
    /// Window for run length statistics (trades)
    pub run_window: usize,
    /// Half-life for EWMA calculations (seconds)
    pub ewma_half_life_s: f64,
    /// Minimum trades before features are valid
    pub min_warmup_trades: usize,
    /// Z-score cap for outlier handling
    pub zscore_cap: f64,
}

impl Default for MicrostructureConfig {
    fn default() -> Self {
        Self {
            intensity_window: 100,
            impact_window: 50,
            run_window: 30,
            ewma_half_life_s: 30.0,
            min_warmup_trades: 50,
            zscore_cap: 3.0,
        }
    }
}

/// A single trade observation for feature computation
#[derive(Debug, Clone, Copy)]
pub struct TradeObservation {
    /// Trade timestamp in milliseconds
    pub timestamp_ms: u64,
    /// Trade price
    pub price: f64,
    /// Trade size
    pub size: f64,
    /// True if buyer-initiated (taker bought)
    pub is_buy: bool,
}

/// Rolling statistics with Welford's online algorithm
#[derive(Debug, Clone)]
struct RollingStats {
    window: VecDeque<f64>,
    max_size: usize,
    sum: f64,
    sum_sq: f64,
}

impl RollingStats {
    fn new(max_size: usize) -> Self {
        Self {
            window: VecDeque::with_capacity(max_size),
            max_size,
            sum: 0.0,
            sum_sq: 0.0,
        }
    }

    fn push(&mut self, value: f64) {
        if self.window.len() >= self.max_size {
            if let Some(old) = self.window.pop_front() {
                self.sum -= old;
                self.sum_sq -= old * old;
            }
        }
        self.window.push_back(value);
        self.sum += value;
        self.sum_sq += value * value;
    }

    fn mean(&self) -> f64 {
        if self.window.is_empty() {
            0.0
        } else {
            self.sum / self.window.len() as f64
        }
    }

    fn variance(&self) -> f64 {
        let n = self.window.len() as f64;
        if n < 2.0 {
            return 0.0;
        }
        let mean = self.mean();
        (self.sum_sq / n) - (mean * mean)
    }

    fn std(&self) -> f64 {
        self.variance().sqrt().max(1e-10)
    }

    fn zscore(&self, value: f64) -> f64 {
        (value - self.mean()) / self.std()
    }

    fn len(&self) -> usize {
        self.window.len()
    }
}

/// EWMA (Exponentially Weighted Moving Average) calculator
#[derive(Debug, Clone)]
struct Ewma {
    value: f64,
    variance: f64,
    alpha: f64,
    initialized: bool,
}

#[allow(dead_code)]
impl Ewma {
    fn new(half_life_samples: f64) -> Self {
        let alpha = 1.0 - (-2.0_f64.ln() / half_life_samples).exp();
        Self {
            value: 0.0,
            variance: 0.0,
            alpha,
            initialized: false,
        }
    }

    fn update(&mut self, x: f64) {
        if !self.initialized {
            self.value = x;
            self.variance = 0.0;
            self.initialized = true;
        } else {
            let diff = x - self.value;
            self.value += self.alpha * diff;
            self.variance = (1.0 - self.alpha) * (self.variance + self.alpha * diff * diff);
        }
    }

    fn mean(&self) -> f64 {
        self.value
    }

    fn std(&self) -> f64 {
        self.variance.sqrt().max(1e-10)
    }

    fn zscore(&self, x: f64) -> f64 {
        if !self.initialized {
            0.0
        } else {
            (x - self.value) / self.std()
        }
    }
}

/// Extracted microstructure features (all normalized)
#[derive(Debug, Clone, Copy, Default)]
pub struct MicrostructureFeatures {
    /// Trade arrival intensity vs baseline (z-score)
    /// High positive = burst of trades (information event)
    pub intensity_zscore: f64,

    /// Price impact per unit volume vs historical (z-score)
    /// High positive = high information content trades
    pub price_impact_zscore: f64,

    /// Consecutive same-side trades vs expected (z-score)
    /// High positive = informed trader on one side
    pub run_length_zscore: f64,

    /// Volume-weighted order imbalance [-1, 1]
    /// Extreme values = directional pressure
    pub volume_imbalance: f64,

    /// Current spread / average spread ratio - 1
    /// Positive = spread widening (MM detecting info)
    pub spread_widening: f64,

    /// Rate of change of book imbalance (z-score)
    /// High absolute = rapid book change
    pub book_velocity_zscore: f64,

    /// Inter-arrival time vs expected (z-score, inverted)
    /// High positive = faster than normal (information)
    pub arrival_speed_zscore: f64,

    /// Trade size vs average (z-score)
    /// High positive = unusually large trades
    pub size_zscore: f64,

    /// Combined toxicity score [0, 1]
    /// Weighted combination of above features
    pub toxicity_score: f64,

    /// Confidence in the features (0 = no data, 1 = fully warmed up)
    pub confidence: f64,

    // === NEW ENTROPY FEATURES ===

    /// Trade size entropy [0, 1]
    /// Low entropy = concentrated sizes (potential informed trading)
    /// High entropy = diverse sizes (noise trading)
    /// We invert this: higher value = more toxic (less entropy = more informed)
    pub size_concentration: f64,

    /// Direction entropy [0, 1]
    /// Low entropy = one-sided flow (informed)
    /// High entropy = balanced flow (noise)
    /// We invert this: higher value = more toxic (less entropy = more informed)
    pub direction_concentration: f64,
}

impl MicrostructureFeatures {
    /// Get feature vector for ML/learning
    pub fn as_vector(&self) -> [f64; 10] {
        [
            self.intensity_zscore,
            self.price_impact_zscore,
            self.run_length_zscore,
            self.volume_imbalance,
            self.spread_widening,
            self.book_velocity_zscore,
            self.arrival_speed_zscore,
            self.size_zscore,
            self.size_concentration,
            self.direction_concentration,
        ]
    }

    /// Feature names for logging/display
    pub fn feature_names() -> [&'static str; 10] {
        [
            "intensity",
            "impact",
            "run_length",
            "vol_imbal",
            "spread_widen",
            "book_vel",
            "arrival_spd",
            "size",
            "size_conc",
            "dir_conc",
        ]
    }
}

/// Microstructure feature extractor
///
/// Computes statistically meaningful features from trade and book data.
/// All features are designed to detect informed trading.
#[derive(Debug, Clone)]
pub struct MicrostructureExtractor {
    config: MicrostructureConfig,

    // Trade tracking
    trades: VecDeque<TradeObservation>,
    last_trade_time_ms: u64,

    // Intensity estimation (trades per second)
    intensity_stats: RollingStats,
    current_intensity: f64,

    // Price impact estimation (Kyle's lambda)
    // λ = Cov(ΔP, SignedVolume) / Var(SignedVolume)
    impact_stats: RollingStats,
    signed_volume_stats: RollingStats,
    last_mid: f64,

    // Run length tracking
    current_run_length: usize,
    current_run_side: Option<bool>, // true = buy run
    run_length_stats: RollingStats,

    // Volume imbalance (VPIN-style)
    buy_volume_ewma: Ewma,
    sell_volume_ewma: Ewma,

    // Spread tracking
    spread_ewma: Ewma,
    current_spread_bps: f64,

    // Book imbalance velocity
    book_imbalance_history: VecDeque<(u64, f64)>,
    book_velocity_stats: RollingStats,

    // Inter-arrival time
    arrival_time_stats: RollingStats,

    // Trade size
    size_stats: RollingStats,

    // === Entropy tracking ===
    // Size buckets for entropy calculation (log-spaced)
    size_bucket_counts: [usize; 8],
    direction_counts: [usize; 2], // [sell, buy]
    entropy_window_count: usize,

    // Warmup tracking
    trade_count: usize,
}

impl MicrostructureExtractor {
    pub fn new(config: MicrostructureConfig) -> Self {
        let ewma_samples = config.ewma_half_life_s * 10.0; // Assume ~10 trades/sec baseline

        Self {
            trades: VecDeque::with_capacity(config.intensity_window),
            last_trade_time_ms: 0,

            intensity_stats: RollingStats::new(config.intensity_window),
            current_intensity: 0.0,

            impact_stats: RollingStats::new(config.impact_window),
            signed_volume_stats: RollingStats::new(config.impact_window),
            last_mid: 0.0,

            current_run_length: 0,
            current_run_side: None,
            run_length_stats: RollingStats::new(config.run_window),

            buy_volume_ewma: Ewma::new(ewma_samples),
            sell_volume_ewma: Ewma::new(ewma_samples),

            spread_ewma: Ewma::new(ewma_samples),
            current_spread_bps: 0.0,

            book_imbalance_history: VecDeque::with_capacity(100),
            book_velocity_stats: RollingStats::new(50),

            arrival_time_stats: RollingStats::new(config.intensity_window),
            size_stats: RollingStats::new(config.intensity_window),

            // Entropy tracking
            size_bucket_counts: [0; 8],
            direction_counts: [0; 2],
            entropy_window_count: 0,

            trade_count: 0,
            config,
        }
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        Self::new(MicrostructureConfig::default())
    }

    /// Process a new trade observation
    pub fn on_trade(&mut self, trade: TradeObservation) {
        self.trade_count += 1;

        // === Inter-arrival time ===
        if self.last_trade_time_ms > 0 {
            let arrival_ms = trade.timestamp_ms.saturating_sub(self.last_trade_time_ms) as f64;
            self.arrival_time_stats.push(arrival_ms);
        }
        self.last_trade_time_ms = trade.timestamp_ms;

        // === Trade intensity ===
        // Count trades in last second
        let one_sec_ago = trade.timestamp_ms.saturating_sub(1000);
        while let Some(old) = self.trades.front() {
            if old.timestamp_ms < one_sec_ago {
                self.trades.pop_front();
            } else {
                break;
            }
        }
        self.trades.push_back(trade);
        self.current_intensity = self.trades.len() as f64;
        self.intensity_stats.push(self.current_intensity);

        // === Price impact (Kyle's lambda) ===
        if self.last_mid > 0.0 {
            let price_change_bps = (trade.price - self.last_mid) / self.last_mid * 10_000.0;
            let signed_volume = if trade.is_buy { trade.size } else { -trade.size };

            // Estimate impact as price_change / signed_volume
            if signed_volume.abs() > 1e-10 {
                let impact = price_change_bps / signed_volume;
                self.impact_stats.push(impact.abs()); // Use absolute impact
            }
            self.signed_volume_stats.push(signed_volume);
        }

        // === Run length ===
        match self.current_run_side {
            Some(side) if side == trade.is_buy => {
                self.current_run_length += 1;
            }
            _ => {
                // End of run, record it
                if self.current_run_length > 0 {
                    self.run_length_stats.push(self.current_run_length as f64);
                }
                self.current_run_length = 1;
                self.current_run_side = Some(trade.is_buy);
            }
        }

        // === Volume tracking ===
        if trade.is_buy {
            self.buy_volume_ewma.update(trade.size);
            self.sell_volume_ewma.update(0.0);
        } else {
            self.buy_volume_ewma.update(0.0);
            self.sell_volume_ewma.update(trade.size);
        }

        // === Trade size ===
        self.size_stats.push(trade.size);

        // === Entropy tracking ===
        // Size bucket: log-spaced buckets for trade sizes
        // Bucket i covers sizes from 10^(i/2) to 10^((i+1)/2) roughly
        let size_bucket = if trade.size < 0.1 {
            0
        } else {
            let log_size = trade.size.log10();
            ((log_size + 1.0) * 2.0).floor().clamp(0.0, 7.0) as usize
        };
        self.size_bucket_counts[size_bucket] += 1;

        // Direction count
        let dir_idx = if trade.is_buy { 1 } else { 0 };
        self.direction_counts[dir_idx] += 1;

        self.entropy_window_count += 1;

        // Decay entropy counts to keep them recent (every 200 trades)
        if self.entropy_window_count >= 200 {
            for count in &mut self.size_bucket_counts {
                *count = *count / 2;
            }
            for count in &mut self.direction_counts {
                *count = *count / 2;
            }
            self.entropy_window_count = self.entropy_window_count / 2;
        }
    }

    /// Update with new book state
    pub fn on_book_update(&mut self, bid: f64, ask: f64, bid_size: f64, ask_size: f64, timestamp_ms: u64) {
        // === Spread tracking ===
        if bid > 0.0 && ask > 0.0 {
            let mid = (bid + ask) / 2.0;
            self.current_spread_bps = (ask - bid) / mid * 10_000.0;
            self.spread_ewma.update(self.current_spread_bps);
            self.last_mid = mid;
        }

        // === Book imbalance velocity ===
        let total_size = bid_size + ask_size;
        if total_size > 0.0 {
            let imbalance = (bid_size - ask_size) / total_size;

            // Compute velocity if we have history
            if let Some((old_ts, old_imb)) = self.book_imbalance_history.front() {
                let dt_ms = timestamp_ms.saturating_sub(*old_ts) as f64;
                if dt_ms > 0.0 {
                    let velocity = (imbalance - old_imb) / (dt_ms / 1000.0); // per second
                    self.book_velocity_stats.push(velocity);
                }
            }

            // Keep last 100ms of history
            let cutoff = timestamp_ms.saturating_sub(100);
            while let Some((ts, _)) = self.book_imbalance_history.front() {
                if *ts < cutoff {
                    self.book_imbalance_history.pop_front();
                } else {
                    break;
                }
            }
            self.book_imbalance_history.push_back((timestamp_ms, imbalance));
        }
    }

    /// Compute normalized entropy of count distribution, returns [0, 1]
    /// 0 = concentrated (one bucket dominates), 1 = uniform
    fn compute_entropy(&self, counts: &[usize]) -> f64 {
        let total: usize = counts.iter().sum();
        if total == 0 {
            return 0.5; // No data, neutral
        }

        let total_f = total as f64;
        let mut entropy = 0.0;
        let mut non_zero_buckets = 0;

        for &count in counts {
            if count > 0 {
                let p = count as f64 / total_f;
                entropy -= p * p.ln();
                non_zero_buckets += 1;
            }
        }

        // Normalize by max possible entropy (uniform distribution)
        let max_entropy = (non_zero_buckets.max(1) as f64).ln();
        if max_entropy > 0.0 {
            (entropy / max_entropy).clamp(0.0, 1.0)
        } else {
            0.5
        }
    }

    /// Extract current features
    pub fn extract(&self) -> MicrostructureFeatures {
        let warmup_ratio = (self.trade_count as f64 / self.config.min_warmup_trades as f64).min(1.0);

        if warmup_ratio < 0.5 {
            return MicrostructureFeatures {
                confidence: warmup_ratio,
                ..Default::default()
            };
        }

        let cap = self.config.zscore_cap;

        // === Intensity z-score ===
        let intensity_zscore = self.intensity_stats
            .zscore(self.current_intensity)
            .clamp(-cap, cap);

        // === Price impact z-score ===
        let recent_impact = self.impact_stats.mean();
        let impact_zscore = self.impact_stats
            .zscore(recent_impact)
            .clamp(-cap, cap);

        // === Run length z-score ===
        let run_length_zscore = self.run_length_stats
            .zscore(self.current_run_length as f64)
            .clamp(-cap, cap);

        // === Volume imbalance [-1, 1] ===
        let buy_vol = self.buy_volume_ewma.mean();
        let sell_vol = self.sell_volume_ewma.mean();
        let total_vol = buy_vol + sell_vol;
        let volume_imbalance = if total_vol > 1e-10 {
            ((buy_vol - sell_vol) / total_vol).clamp(-1.0, 1.0)
        } else {
            0.0
        };

        // === Spread widening ===
        let avg_spread = self.spread_ewma.mean();
        let spread_widening = if avg_spread > 0.0 {
            (self.current_spread_bps / avg_spread - 1.0).clamp(-1.0, 2.0)
        } else {
            0.0
        };

        // === Book velocity z-score ===
        let book_velocity_zscore = if self.book_velocity_stats.len() > 10 {
            self.book_velocity_stats
                .zscore(self.book_velocity_stats.mean())
                .clamp(-cap, cap)
        } else {
            0.0
        };

        // === Arrival speed z-score (inverted - fast arrivals = positive) ===
        let arrival_speed_zscore = if self.arrival_time_stats.len() > 10 {
            let mean_arrival = self.arrival_time_stats.mean();
            // Invert: short arrival time = positive z-score
            (-self.arrival_time_stats.zscore(mean_arrival)).clamp(-cap, cap)
        } else {
            0.0
        };

        // === Size z-score ===
        let avg_size = self.size_stats.mean();
        let size_zscore = self.size_stats.zscore(avg_size).clamp(-cap, cap);

        // === Entropy features ===
        // Size concentration: low entropy = concentrated = more informed
        // We invert so higher value = more toxic (less entropy)
        let size_entropy = self.compute_entropy(&self.size_bucket_counts);
        let size_concentration = (1.0 - size_entropy).clamp(0.0, 1.0);

        // Direction concentration: low entropy = one-sided = more informed
        // We invert so higher value = more toxic (less entropy)
        let direction_entropy = self.compute_entropy(&self.direction_counts);
        let direction_concentration = (1.0 - direction_entropy).clamp(0.0, 1.0);

        // === Combined toxicity score ===
        // Weighted combination emphasizing strong signals
        let toxicity_score = self.compute_toxicity_score(
            intensity_zscore,
            impact_zscore,
            run_length_zscore,
            volume_imbalance,
            spread_widening,
            arrival_speed_zscore,
            size_zscore,
            size_concentration,
            direction_concentration,
        );

        MicrostructureFeatures {
            intensity_zscore,
            price_impact_zscore: impact_zscore,
            run_length_zscore,
            volume_imbalance,
            spread_widening,
            book_velocity_zscore,
            arrival_speed_zscore,
            size_zscore,
            toxicity_score,
            confidence: warmup_ratio,
            size_concentration,
            direction_concentration,
        }
    }

    /// Compute combined toxicity score using theory-driven weights
    fn compute_toxicity_score(
        &self,
        intensity: f64,
        impact: f64,
        run_length: f64,
        vol_imbalance: f64,
        spread_widen: f64,
        arrival_speed: f64,
        size: f64,
        size_concentration: f64,
        direction_concentration: f64,
    ) -> f64 {
        // Weights based on microstructure theory importance:
        // - Price impact (Kyle's λ) is the gold standard for information
        // - Run length is strong evidence of informed trading
        // - Intensity and arrival speed indicate information events
        // - Spread widening is MM's response to detected info
        // - Volume imbalance and size are supporting signals
        // - Entropy features indicate informed vs noise trading

        const W_IMPACT: f64 = 0.22;      // Kyle's lambda - most theoretically grounded
        const W_RUN: f64 = 0.18;         // Run length - strong clustering signal
        const W_INTENSITY: f64 = 0.12;   // Trade bursts
        const W_ARRIVAL: f64 = 0.12;     // Fast arrivals
        const W_SPREAD: f64 = 0.08;      // MM response
        const W_IMBALANCE: f64 = 0.08;   // Directional pressure
        const W_SIZE: f64 = 0.05;        // Large trades
        const W_SIZE_CONC: f64 = 0.08;   // Size concentration (informed = concentrated)
        const W_DIR_CONC: f64 = 0.07;    // Direction concentration (informed = one-sided)

        // Convert z-scores to [0, 1] probabilities using sigmoid
        let sigmoid = |z: f64| 1.0 / (1.0 + (-z).exp());

        let score = W_IMPACT * sigmoid(impact)
            + W_RUN * sigmoid(run_length)
            + W_INTENSITY * sigmoid(intensity)
            + W_ARRIVAL * sigmoid(arrival_speed)
            + W_SPREAD * sigmoid(spread_widen * 2.0) // Scale spread signal
            + W_IMBALANCE * sigmoid(vol_imbalance.abs() * 2.0)
            + W_SIZE * sigmoid(size)
            + W_SIZE_CONC * size_concentration  // Already [0, 1]
            + W_DIR_CONC * direction_concentration; // Already [0, 1]

        score.clamp(0.0, 1.0)
    }

    /// Get diagnostic information
    pub fn diagnostics(&self) -> MicrostructureDiagnostics {
        MicrostructureDiagnostics {
            trade_count: self.trade_count,
            intensity_mean: self.intensity_stats.mean(),
            intensity_std: self.intensity_stats.std(),
            impact_mean: self.impact_stats.mean(),
            impact_std: self.impact_stats.std(),
            avg_run_length: self.run_length_stats.mean(),
            avg_spread_bps: self.spread_ewma.mean(),
            avg_arrival_ms: self.arrival_time_stats.mean(),
            avg_size: self.size_stats.mean(),
            is_warmed_up: self.trade_count >= self.config.min_warmup_trades,
        }
    }

    /// Reset all state
    pub fn reset(&mut self) {
        *self = Self::new(self.config.clone());
    }
}

/// Diagnostic information for the extractor
#[derive(Debug, Clone)]
pub struct MicrostructureDiagnostics {
    pub trade_count: usize,
    pub intensity_mean: f64,
    pub intensity_std: f64,
    pub impact_mean: f64,
    pub impact_std: f64,
    pub avg_run_length: f64,
    pub avg_spread_bps: f64,
    pub avg_arrival_ms: f64,
    pub avg_size: f64,
    pub is_warmed_up: bool,
}

impl MicrostructureDiagnostics {
    pub fn summary(&self) -> String {
        format!(
            "trades={} intensity={:.1}±{:.1} impact={:.2}±{:.2} run={:.1} spread={:.1}bps arr={:.0}ms size={:.2}{}",
            self.trade_count,
            self.intensity_mean,
            self.intensity_std,
            self.impact_mean,
            self.impact_std,
            self.avg_run_length,
            self.avg_spread_bps,
            self.avg_arrival_ms,
            self.avg_size,
            if self.is_warmed_up { " [READY]" } else { " [WARMING]" }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_trade(ts: u64, price: f64, size: f64, is_buy: bool) -> TradeObservation {
        TradeObservation {
            timestamp_ms: ts,
            price,
            size,
            is_buy,
        }
    }

    #[test]
    fn test_warmup() {
        let mut extractor = MicrostructureExtractor::default_config();

        // Before warmup, confidence should be low
        let features = extractor.extract();
        assert!(features.confidence < 0.5);

        // Add trades
        for i in 0..60 {
            extractor.on_trade(make_trade(i * 100, 100.0, 1.0, i % 2 == 0));
        }

        let features = extractor.extract();
        assert!(features.confidence >= 0.5);
    }

    #[test]
    fn test_run_length_detection() {
        let mut extractor = MicrostructureExtractor::default_config();

        // Warmup with mixed trades
        for i in 0..50 {
            extractor.on_trade(make_trade(i * 100, 100.0, 1.0, i % 2 == 0));
        }

        // Now create a long buy run
        for i in 50..60 {
            extractor.on_trade(make_trade(i * 100, 100.0 + (i as f64) * 0.01, 1.0, true));
        }

        let features = extractor.extract();
        // Run length z-score should be positive (longer than average)
        assert!(features.run_length_zscore > 0.0, "Run length z-score should be positive for long run");
    }

    #[test]
    fn test_intensity_spike() {
        let mut extractor = MicrostructureExtractor::default_config();

        // Normal intensity: 1 trade per 100ms
        for i in 0..50 {
            extractor.on_trade(make_trade(i * 100, 100.0, 1.0, i % 2 == 0));
        }

        // Spike: 20 trades in 100ms
        let base_ts = 5000;
        for i in 0..20 {
            extractor.on_trade(make_trade(base_ts + i * 5, 100.0, 1.0, true));
        }

        let features = extractor.extract();
        assert!(features.intensity_zscore > 1.0, "Intensity should spike: {}", features.intensity_zscore);
    }

    #[test]
    fn test_volume_imbalance() {
        let mut extractor = MicrostructureExtractor::default_config();

        // All buy volume
        for i in 0..50 {
            extractor.on_trade(make_trade(i * 100, 100.0, 1.0, true));
        }

        let features = extractor.extract();
        assert!(features.volume_imbalance > 0.5, "Should show buy imbalance");

        // Now sell pressure
        for i in 50..100 {
            extractor.on_trade(make_trade(i * 100, 100.0, 2.0, false)); // Larger sells
        }

        let features = extractor.extract();
        assert!(features.volume_imbalance < 0.0, "Should show sell imbalance");
    }

    #[test]
    fn test_spread_widening() {
        let mut extractor = MicrostructureExtractor::default_config();

        // Normal spread: 5 bps
        for i in 0..50 {
            extractor.on_book_update(99.975, 100.025, 10.0, 10.0, i * 100);
            extractor.on_trade(make_trade(i * 100, 100.0, 1.0, i % 2 == 0));
        }

        let features = extractor.extract();
        assert!(features.spread_widening.abs() < 0.5, "Normal spread should have low widening signal");

        // Widen spread to 20 bps
        extractor.on_book_update(99.9, 100.1, 10.0, 10.0, 5000);

        let features = extractor.extract();
        assert!(features.spread_widening > 0.5, "Widened spread should show positive signal");
    }

    #[test]
    fn test_toxicity_score_bounds() {
        let mut extractor = MicrostructureExtractor::default_config();

        for i in 0..100 {
            extractor.on_trade(make_trade(i * 100, 100.0, 1.0, i % 2 == 0));
            extractor.on_book_update(99.99, 100.01, 10.0, 10.0, i * 100);
        }

        let features = extractor.extract();
        assert!(features.toxicity_score >= 0.0 && features.toxicity_score <= 1.0);
    }
}
