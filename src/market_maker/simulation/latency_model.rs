//! Empirical latency model for order book replay simulation.
//!
//! Models placement and cancellation latency using empirical percentile distributions.
//! Used by `ReplayEngine` to simulate realistic order lifecycle timing.

use serde::{Deserialize, Serialize};

/// Empirical latency distribution modeled as percentile buckets.
///
/// Percentile-based (not parametric) to capture real-world heavy tails
/// where p99 can be 10-50x the median.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyDistribution {
    /// p10 latency in microseconds
    pub p10_us: u64,
    /// p50 (median) latency in microseconds
    pub p50_us: u64,
    /// p90 latency in microseconds
    pub p90_us: u64,
    /// p99 latency in microseconds
    pub p99_us: u64,
}

impl Default for LatencyDistribution {
    fn default() -> Self {
        // Typical HL API latency profile (observed Feb 2026)
        Self {
            p10_us: 30_000,  // 30ms
            p50_us: 80_000,  // 80ms
            p90_us: 200_000, // 200ms
            p99_us: 500_000, // 500ms
        }
    }
}

impl LatencyDistribution {
    /// Sample a latency in microseconds from the empirical distribution.
    ///
    /// Uses a simple linear interpolation between percentile buckets,
    /// driven by a uniform random value in [0, 1].
    pub fn sample_us(&self, uniform_01: f64) -> u64 {
        let u = uniform_01.clamp(0.0, 1.0);
        if u < 0.1 {
            // 0-10th percentile: linear interp from 0 to p10
            let frac = u / 0.1;
            (self.p10_us as f64 * frac) as u64
        } else if u < 0.5 {
            // 10th-50th percentile
            let frac = (u - 0.1) / 0.4;
            self.p10_us + ((self.p50_us - self.p10_us) as f64 * frac) as u64
        } else if u < 0.9 {
            // 50th-90th percentile
            let frac = (u - 0.5) / 0.4;
            self.p50_us + ((self.p90_us - self.p50_us) as f64 * frac) as u64
        } else {
            // 90th-99th+ percentile
            let frac = (u - 0.9) / 0.1;
            self.p90_us + ((self.p99_us - self.p90_us) as f64 * frac) as u64
        }
    }
}

/// Latency model combining placement, cancellation, and market data latencies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyModel {
    /// Latency for placing new orders
    pub placement: LatencyDistribution,
    /// Latency for cancelling orders
    pub cancellation: LatencyDistribution,
    /// Latency for receiving market data updates (L2 + trades)
    pub market_data: LatencyDistribution,
}

impl Default for LatencyModel {
    fn default() -> Self {
        Self {
            placement: LatencyDistribution {
                p10_us: 30_000,
                p50_us: 80_000,
                p90_us: 200_000,
                p99_us: 500_000,
            },
            cancellation: LatencyDistribution {
                p10_us: 20_000, // Cancels slightly faster
                p50_us: 60_000,
                p90_us: 150_000,
                p99_us: 400_000,
            },
            market_data: LatencyDistribution {
                p10_us: 5_000, // WebSocket data is faster
                p50_us: 15_000,
                p90_us: 50_000,
                p99_us: 150_000,
            },
        }
    }
}

impl LatencyModel {
    /// Update from observed latency measurements.
    ///
    /// Takes a sorted slice of latency observations (in microseconds)
    /// and extracts empirical percentiles.
    pub fn update_from_observations(sorted_latencies_us: &[u64]) -> Option<LatencyDistribution> {
        if sorted_latencies_us.len() < 10 {
            return None;
        }
        let n = sorted_latencies_us.len();
        Some(LatencyDistribution {
            p10_us: sorted_latencies_us[n / 10],
            p50_us: sorted_latencies_us[n / 2],
            p90_us: sorted_latencies_us[n * 9 / 10],
            p99_us: sorted_latencies_us[n * 99 / 100],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latency_sample_monotonic() {
        let dist = LatencyDistribution::default();
        let mut prev = 0;
        for i in 0..=100 {
            let u = i as f64 / 100.0;
            let sample = dist.sample_us(u);
            assert!(sample >= prev, "sample should be monotonically increasing");
            prev = sample;
        }
    }

    #[test]
    fn test_latency_sample_boundaries() {
        let dist = LatencyDistribution::default();
        assert_eq!(dist.sample_us(0.0), 0);
        assert_eq!(dist.sample_us(0.5), dist.p50_us);
    }

    #[test]
    fn test_latency_sample_clamping() {
        let dist = LatencyDistribution::default();
        // Values outside [0, 1] should be clamped
        let _ = dist.sample_us(-1.0);
        let _ = dist.sample_us(2.0);
    }

    #[test]
    fn test_latency_model_default() {
        let model = LatencyModel::default();
        // Cancel should be faster than placement at median
        assert!(model.cancellation.p50_us < model.placement.p50_us);
        // Market data should be fastest
        assert!(model.market_data.p50_us < model.cancellation.p50_us);
    }

    #[test]
    fn test_update_from_observations() {
        let mut obs: Vec<u64> = (0..100).map(|i| i * 1000).collect();
        obs.sort();
        let dist = LatencyModel::update_from_observations(&obs).unwrap();
        assert_eq!(dist.p10_us, 10_000);
        assert_eq!(dist.p50_us, 50_000);
        assert_eq!(dist.p90_us, 90_000);
        assert_eq!(dist.p99_us, 99_000);
    }

    #[test]
    fn test_update_from_observations_too_few() {
        let obs: Vec<u64> = vec![1, 2, 3];
        assert!(LatencyModel::update_from_observations(&obs).is_none());
    }
}
