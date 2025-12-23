//! Volume tick arrival intensity estimator.
//!
//! Measures the rate of volume-clock ticks (not raw trades) per second.

use super::volume_clock::VolumeBucket;

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
    pub(super) fn new(half_life_ticks: f64, default_intensity: f64) -> Self {
        Self {
            alpha: (2.0_f64.ln() / half_life_ticks).clamp(0.001, 1.0),
            intensity: default_intensity,
            last_tick_ms: None,
            tick_count: 0,
        }
    }

    pub(super) fn on_bucket(&mut self, bucket: &VolumeBucket) {
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

    pub(super) fn ticks_per_second(&self) -> f64 {
        self.intensity.clamp(0.001, 100.0)
    }
}
