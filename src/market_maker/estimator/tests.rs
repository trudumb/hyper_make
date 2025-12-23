//! Tests for the estimator module.

use super::bipower::SingleScaleBipower;
use super::config::EstimatorConfig;
use super::kappa::WeightedKappaEstimator;
use super::volume_clock::VolumeBucketAccumulator;
use super::ParameterEstimator;

fn make_config() -> EstimatorConfig {
    EstimatorConfig {
        initial_bucket_volume: 1.0,
        min_volume_ticks: 5,
        min_l2_updates: 3,
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
fn test_vwap_calculation() {
    let config = EstimatorConfig {
        initial_bucket_volume: 5.0,
        ..Default::default()
    };
    let mut acc = VolumeBucketAccumulator::new(&config);

    // Trades at different prices
    acc.on_trade(1000, 100.0, 2.0); // 200, vol = 2
    acc.on_trade(2000, 110.0, 2.0); // 220, vol = 4

    // Total: 420 / 4 = 105.0, but need 5 volume
    let bucket = acc.on_trade(2000, 110.0, 0.0); // Force check with zero size
    assert!(bucket.is_none()); // Not enough volume yet

    // Add more to complete (need 1 more)
    let bucket = acc.on_trade(3000, 120.0, 1.0); // 120, vol = 5
    assert!(bucket.is_some());

    let b = bucket.unwrap();
    // VWAP = (200 + 220 + 120) / 5 = 540/5 = 108.0
    assert!((b.vwap - 108.0).abs() < 0.1);
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
fn test_single_scale_bipower_with_jump() {
    let mut bv = SingleScaleBipower::new(5.0, 0.001_f64.powi(2));

    // Feed returns with a sudden jump
    // Normal, normal, JUMP, normal, normal
    let vwaps: [f64; 7] = [100.0, 100.1, 100.0, 105.0, 105.1, 105.0, 105.1];
    let mut last_vwap: f64 = vwaps[0];
    for vwap in vwaps.iter().skip(1) {
        let log_return = (vwap / last_vwap).ln();
        bv.update(log_return);
        last_vwap = *vwap;
    }

    // Jump ratio should be elevated (RV > BV due to jump)
    let ratio = bv.jump_ratio();
    assert!(
        ratio > 1.5,
        "Expected elevated ratio due to jump, got {}",
        ratio
    );
}

#[test]
fn test_weighted_kappa_estimator() {
    let mut kappa = WeightedKappaEstimator::new(10.0, 100.0, 0.01, 15);
    let mid = 100.0;

    // Synthetic book where depth increases at each level
    // This is typical: more liquidity accumulates further from mid
    // Cumulative depth at level i: sum of sizes from 0 to i
    let bids: Vec<(f64, f64)> = (1..=10)
        .map(|i| {
            let price = mid - i as f64 * 0.05; // 99.95, 99.90, ...
            let size = 1.0; // Constant size at each level
            (price, size)
        })
        .collect();

    let asks: Vec<(f64, f64)> = (1..=10)
        .map(|i| {
            let price = mid + i as f64 * 0.05; // 100.05, 100.10, ...
            let size = 1.0;
            (price, size)
        })
        .collect();

    // Run multiple updates to converge
    for _ in 0..30 {
        kappa.update(&bids, &asks, mid);
    }

    // Kappa should be in a reasonable range (changed from default 100)
    let k = kappa.kappa();
    assert!(
        k > 1.0 && k < 10000.0,
        "Kappa should be in valid range, got {}",
        k
    );
    // Verify it's updating (not stuck at default)
    assert!(
        k != 100.0,
        "Kappa should have changed from default 100, got {}",
        k
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

    // Feed trades to fill buckets (need 5 volume ticks)
    let mut time = 1000u64;
    for i in 0..100 {
        let price = 100.0 + (i as f64 * 0.1).sin() * 0.5;
        // Alternate buy/sell to simulate balanced flow
        let is_buy = i % 2 == 0;
        estimator.on_trade(time, price, 0.5, Some(is_buy)); // 0.5 per trade, 2 trades per bucket
        time += 100;
    }

    // Feed L2 books (need 3 updates)
    let bids = vec![(99.9, 5.0), (99.8, 10.0), (99.7, 15.0)];
    let asks = vec![(100.1, 5.0), (100.2, 10.0), (100.3, 15.0)];
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
fn test_adaptive_bucket_threshold() {
    let config = EstimatorConfig {
        initial_bucket_volume: 1.0,
        volume_window_secs: 10.0,
        volume_percentile: 0.1, // 10% of rolling volume
        min_bucket_volume: 0.5,
        max_bucket_volume: 10.0,
        ..Default::default()
    };
    let mut acc = VolumeBucketAccumulator::new(&config);

    // Fill several buckets to build up rolling volume
    let mut time = 0u64;
    for _ in 0..10 {
        while acc.on_trade(time, 100.0, 0.5).is_none() {
            time += 100;
        }
    }

    // Threshold should have adapted based on rolling volume
    // With 10 buckets of ~1.0 volume each in 10 seconds,
    // and 10% percentile, threshold should be around 1.0
    assert!(acc.threshold >= 0.5);
    assert!(acc.threshold <= 10.0);
}

#[test]
fn test_arrival_intensity() {
    let config = make_config();
    let mut estimator = ParameterEstimator::new(config);

    // Feed trades at consistent rate to fill buckets
    let mut time = 0u64;
    for i in 0..50 {
        let is_buy = i % 2 == 0;
        estimator.on_trade(time, 100.0, 1.0, Some(is_buy)); // Each trade = 1 bucket
        time += 500; // 500ms between buckets = 2 ticks/sec
    }

    let intensity = estimator.arrival_intensity();
    // Should be close to 2 ticks/sec
    assert!(
        intensity > 1.0 && intensity < 5.0,
        "Expected ~2 ticks/sec, got {}",
        intensity
    );
}
