//! Volume clock and arrival estimation.
//!
//! - VolumeBucket: Completed volume bucket with VWAP
//! - VolumeBucketAccumulator: Adaptive volume-based sampling
//! - VolumeTickArrivalEstimator: Trade arrival intensity estimation

use std::collections::VecDeque;

use super::EstimatorConfig;

// ============================================================================
// Volume Clock (Data Normalization)
// ============================================================================

/// A completed volume bucket with VWAP.
#[derive(Debug, Clone)]
pub(crate) struct VolumeBucket {
    /// Start timestamp of bucket (ms)
    pub start_time_ms: u64,
    /// End timestamp of bucket (ms)
    pub end_time_ms: u64,
    /// VWAP: sum(price * size) / sum(size)
    pub vwap: f64,
    /// Total volume in bucket
    pub volume: f64,
}

/// Accumulates trades until volume threshold is reached.
/// Implements adaptive volume clock for normalized economic sampling.
#[derive(Debug)]
pub(crate) struct VolumeBucketAccumulator {
    /// Current bucket start time
    start_time_ms: Option<u64>,
    /// Sum of (price * size) for current bucket
    price_volume_sum: f64,
    /// Sum of sizes for current bucket
    volume_sum: f64,
    /// Current adaptive threshold
    threshold: f64,
    /// Rolling volume tracker for adaptive threshold: (timestamp_ms, volume)
    rolling_volumes: VecDeque<(u64, f64)>,
    /// Config reference
    initial_bucket_volume: f64,
    volume_window_secs: f64,
    volume_percentile: f64,
    min_bucket_volume: f64,
    max_bucket_volume: f64,
}

impl VolumeBucketAccumulator {
    pub(crate) fn new(config: &EstimatorConfig) -> Self {
        Self {
            start_time_ms: None,
            price_volume_sum: 0.0,
            volume_sum: 0.0,
            threshold: config.initial_bucket_volume,
            rolling_volumes: VecDeque::new(),
            initial_bucket_volume: config.initial_bucket_volume,
            volume_window_secs: config.volume_window_secs,
            volume_percentile: config.volume_percentile,
            min_bucket_volume: config.min_bucket_volume,
            max_bucket_volume: config.max_bucket_volume,
        }
    }

    /// Add a trade. Returns Some(VolumeBucket) if bucket completed.
    pub(crate) fn on_trade(&mut self, time_ms: u64, price: f64, size: f64) -> Option<VolumeBucket> {
        if price <= 0.0 || size <= 0.0 {
            return None;
        }

        // Initialize bucket start if needed
        if self.start_time_ms.is_none() {
            self.start_time_ms = Some(time_ms);
        }

        // Accumulate
        self.price_volume_sum += price * size;
        self.volume_sum += size;

        // Update rolling volumes for adaptive threshold
        self.rolling_volumes.push_back((time_ms, size));
        let cutoff = time_ms.saturating_sub((self.volume_window_secs * 1000.0) as u64);
        while self
            .rolling_volumes
            .front()
            .map(|(t, _)| *t < cutoff)
            .unwrap_or(false)
        {
            self.rolling_volumes.pop_front();
        }

        // Check if bucket is complete
        if self.volume_sum >= self.threshold {
            let vwap = self.price_volume_sum / self.volume_sum;
            let bucket = VolumeBucket {
                start_time_ms: self.start_time_ms.unwrap(),
                end_time_ms: time_ms,
                vwap,
                volume: self.volume_sum,
            };

            // Reset for next bucket
            self.price_volume_sum = 0.0;
            self.volume_sum = 0.0;
            self.start_time_ms = None;

            // Update adaptive threshold based on rolling volume
            self.update_adaptive_threshold();

            Some(bucket)
        } else {
            None
        }
    }

    /// Calculate adaptive threshold as percentile of recent volume.
    fn update_adaptive_threshold(&mut self) {
        let total_rolling: f64 = self.rolling_volumes.iter().map(|(_, v)| v).sum();

        if total_rolling > 0.0 {
            let target = total_rolling * self.volume_percentile;
            self.threshold = target.clamp(self.min_bucket_volume, self.max_bucket_volume);
        } else {
            self.threshold = self.initial_bucket_volume;
        }
    }

    /// Get current bucket fill progress (0.0 to 1.0+)
    #[allow(dead_code)]
    pub(crate) fn bucket_progress(&self) -> f64 {
        if self.threshold > 0.0 {
            self.volume_sum / self.threshold
        } else {
            0.0
        }
    }
}

// ============================================================================
// Volume Tick Arrival Estimator
// ============================================================================

/// Arrival intensity estimator based on volume clock ticks.
/// Measures volume ticks per second (not raw trades per second).
#[derive(Debug)]
pub(crate) struct VolumeTickArrivalEstimator {
    alpha: f64,
    intensity: f64, // Volume ticks per second
    last_tick_ms: Option<u64>,
    tick_count: usize,
}

impl VolumeTickArrivalEstimator {
    pub(crate) fn new(half_life_ticks: f64, default_intensity: f64) -> Self {
        Self {
            alpha: (2.0_f64.ln() / half_life_ticks).clamp(0.001, 1.0),
            intensity: default_intensity,
            last_tick_ms: None,
            tick_count: 0,
        }
    }

    pub(crate) fn on_bucket(&mut self, bucket: &VolumeBucket) {
        if let Some(last_ms) = self.last_tick_ms {
            let interval_secs = (bucket.end_time_ms.saturating_sub(last_ms)) as f64 / 1000.0;
            if interval_secs > 0.001 {
                let rate = 1.0 / interval_secs;
                let rate_clamped = rate.clamp(0.001, 100.0);
                self.intensity = self.alpha * rate_clamped + (1.0 - self.alpha) * self.intensity;
                self.tick_count += 1;
            }
        }
        self.last_tick_ms = Some(bucket.end_time_ms);
    }

    pub(crate) fn ticks_per_second(&self) -> f64 {
        self.intensity.clamp(0.001, 100.0)
    }
}
