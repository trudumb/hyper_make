//! Test binary to verify entropy optimizer produces multiple active levels.
//!
//! This validates the rank-based utility fix that ensures bounded utilities
//! regardless of EV range, preventing the optimizer from collapsing to 1 level.
//!
//! Tests both simple config (5 levels) and production config (25 levels, 2-200 bps).
//!
//! Run with: cargo run --bin test_entropy

use hyperliquid_rust_sdk::market_maker::quoting::{
    create_entropy_optimizer, LevelOptimizationParams, MarketRegime,
};

fn main() {
    println!("=== Entropy Optimizer Test ===\n");

    // Test 1: Simple config (original test)
    println!("--- Test 1: Simple Config (5 levels, 5-25 bps) ---");
    test_with_levels(create_simple_levels(), 3);

    // Test 2: Production config (25 levels, 2-200 bps geometric spacing)
    println!("\n--- Test 2: Production Config (25 levels, 2-200 bps) ---");
    test_with_levels(create_production_levels(), 5);

    println!("\n=== All tests passed! ===");
}

fn test_with_levels(levels: Vec<LevelOptimizationParams>, min_active_levels: usize) {
    println!("Input levels ({} total):", levels.len());
    for (i, level) in levels.iter().take(5).enumerate() {
        let fill_prob = level.fill_intensity / (level.fill_intensity + 1.0);
        println!(
            "  Level {}: depth={:.1}bps, spread_capture={:.2}, fill_prob={:.3}",
            i, level.depth_bps, level.spread_capture, fill_prob
        );
    }
    if levels.len() > 5 {
        println!("  ... ({} more levels)", levels.len() - 5);
        let last = levels.last().unwrap();
        let fill_prob = last.fill_intensity / (last.fill_intensity + 1.0);
        println!(
            "  Level {}: depth={:.1}bps, spread_capture={:.2}, fill_prob={:.3}",
            levels.len() - 1,
            last.depth_bps,
            last.spread_capture,
            fill_prob
        );
    }
    println!();

    // Create optimizer with production-like parameters
    let price = 90000.0;
    let margin_available = 1000.0;
    let max_position = 0.05; // More capacity for production test
    let leverage = 40.0;

    let mut optimizer = create_entropy_optimizer(price, margin_available, max_position, leverage);

    // Test with calm regime
    let calm_regime = MarketRegime::default();
    let allocation = optimizer.optimize(&levels, &calm_regime);

    println!("Results (calm regime):");
    println!("  Entropy: {:.3}", allocation.distribution.entropy);
    println!(
        "  Effective levels: {:.1}",
        allocation.distribution.effective_levels
    );
    println!("  Active levels: {}", allocation.active_levels);
    println!(
        "  Entropy floor active: {}",
        allocation.entropy_floor_active
    );
    println!();

    // Show allocation distribution
    let active_sizes: Vec<_> = allocation
        .sizes
        .iter()
        .enumerate()
        .filter(|(_, &s)| s > 0.0001)
        .collect();
    println!("Size allocation ({} active):", active_sizes.len());
    for (i, &size) in active_sizes.iter().take(5) {
        let notional = size * price;
        let depth = levels[*i].depth_bps;
        println!(
            "  Level {} ({}bps): size={:.6} BTC (${:.2})",
            i, depth, size, notional
        );
    }
    if active_sizes.len() > 5 {
        println!("  ... ({} more active levels)", active_sizes.len() - 5);
    }
    println!();

    // Verify the fix
    println!("Verification:");
    let pass = allocation.active_levels >= min_active_levels;
    if pass {
        println!(
            "  ✓ PASS: active_levels ({}) >= {}",
            allocation.active_levels, min_active_levels
        );
    } else {
        println!(
            "  ✗ FAIL: active_levels ({}) < {}",
            allocation.active_levels, min_active_levels
        );
    }

    let entropy_pass = allocation.distribution.entropy > 0.5;
    if entropy_pass {
        println!(
            "  ✓ PASS: entropy ({:.3}) > 0.5",
            allocation.distribution.entropy
        );
    } else {
        println!(
            "  ✗ FAIL: entropy ({:.3}) <= 0.5",
            allocation.distribution.entropy
        );
    }

    if !pass || !entropy_pass {
        println!("\n=== Test failed! ===");
        std::process::exit(1);
    }
}

/// Create simple levels (original test, 5 levels, 5-25 bps)
fn create_simple_levels() -> Vec<LevelOptimizationParams> {
    let depths: [f64; 5] = [5.0, 10.0, 15.0, 20.0, 25.0];
    let fill_intensities: [f64; 5] = [0.8, 0.5, 0.3, 0.15, 0.08];
    let as_at_touch: f64 = 3.0;
    let fees: f64 = 1.5;

    depths
        .iter()
        .zip(fill_intensities.iter())
        .map(|(&depth, &intensity)| {
            let as_at_depth = as_at_touch * (-depth / 10.0_f64).exp();
            let spread_capture = (depth - as_at_depth - fees).max(0.0);

            LevelOptimizationParams {
                depth_bps: depth,
                fill_intensity: intensity,
                spread_capture,
                margin_per_unit: 90000.0 / 40.0,
                adverse_selection: as_at_depth,
            }
        })
        .collect()
}

/// Create production-like levels (25 levels, 2-200 bps geometric spacing)
///
/// This mimics the actual production config which was causing active_levels: 1
/// due to extreme EV range (176:1) before the rank-based utility fix.
fn create_production_levels() -> Vec<LevelOptimizationParams> {
    let num_levels = 25;
    let min_depth = 2.0_f64;
    let max_depth = 200.0_f64;

    // Geometric spacing: depth[i] = min * ratio^i where ratio = (max/min)^(1/(n-1))
    let ratio = (max_depth / min_depth).powf(1.0 / (num_levels - 1) as f64);

    // Production parameters (from logs)
    let sigma = 0.000113_f64; // 1.13 bps/sec
    let kappa = 2500.0_f64;
    let time_horizon = 10.0_f64;
    let fees = 1.5_f64;
    let as_at_touch = 0.0_f64; // Before AS warmup

    (0..num_levels)
        .map(|i| {
            let depth_bps = min_depth * ratio.powi(i);
            let depth = depth_bps / 10000.0; // Convert to fraction

            // Fill intensity: λ(δ) = κ × min(1, (σ×√τ / δ)²)
            let expected_move = sigma * time_horizon.sqrt();
            let fill_prob_raw = (expected_move / depth).powi(2).min(1.0);
            let fill_intensity = fill_prob_raw * kappa;

            // Spread capture: SC(δ) = δ - AS(δ) - fees
            let as_at_depth = as_at_touch * (-depth_bps / 10.0).exp();
            let spread_capture = (depth_bps - as_at_depth - fees).max(0.0);

            LevelOptimizationParams {
                depth_bps,
                fill_intensity,
                spread_capture,
                margin_per_unit: 90000.0 / 40.0,
                adverse_selection: as_at_depth,
            }
        })
        .collect()
}
