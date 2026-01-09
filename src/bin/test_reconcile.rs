use hyperliquid_rust_sdk::market_maker::quoting::LadderLevel;
use hyperliquid_rust_sdk::market_maker::tracking::{
    priority_based_matching, DynamicReconcileConfig, LadderAction, Side, TrackedOrder,
};
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

fn main() {
    // Setup logging
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber).expect("setting default subscriber failed");

    info!("=== Testing Stochastic Reconciliation Logic (Revised) ===");

    test_low_volatility_behavior();
    test_high_volatility_behavior();
}

fn test_low_volatility_behavior() {
    info!("\n--- Test: Low Volatility (Calm Market) ---");

    // Low vol (1bps/sec), High Liquidity (kappa=5000)
    let gamma = 0.1;
    let kappa = 5000.0;
    let sigma = 0.0001; // 1 bps/sec
    let horizon = 10.0; // 10s horizon

    let config = DynamicReconcileConfig::from_market_params(gamma, kappa, sigma, horizon);

    info!(
        "Low Vol Params: Gamma={}, Kappa={}, Sigma={:.6}",
        gamma, kappa, sigma
    );
    info!(
        " Calculated optimal_spread_bps: {:.2}",
        config.optimal_spread_bps
    );
    info!(
        " Calculated max_modify_price_bps: {:.2}",
        config.max_modify_price_bps
    );

    // Current order at 100.00
    let order = TrackedOrder::new(1, Side::Buy, 100.00, 1.0);
    let current = vec![&order];

    // Target moved to 100.04 (4bps move)
    // Optimal spread is ~2bps. max_modify might be small (clamped to 10bps).
    // So 4bps might be within the 10bps clamp?
    // Let's see if we can trigger a Cancel by moving FURTHER.
    // Move to 100.20 (20bps move).
    let target_level = LadderLevel {
        price: 100.20,
        size: 1.0,
        depth_bps: 10.0,
    };
    let targets = vec![target_level];

    let actions: Vec<LadderAction> =
        priority_based_matching(&current, &targets, Side::Buy, &config, None);

    info!("Actions for 20bps price move (Low Vol):");
    for action in &actions {
        info!("  {:?}", action);
    }
}

fn test_high_volatility_behavior() {
    info!("\n--- Test: High Volatility (Stormy Market) ---");

    // High vol (20bps/sec), High Liquidity
    let gamma = 0.1;
    let kappa = 5000.0;
    let sigma = 0.0020; // 20 bps/sec
    let horizon = 10.0; // 10s horizon

    let config = DynamicReconcileConfig::from_market_params(gamma, kappa, sigma, horizon);

    info!(
        "High Vol Params: Gamma={}, Kappa={}, Sigma={:.6}",
        gamma, kappa, sigma
    );
    info!(
        " Calculated optimal_spread_bps: {:.2}",
        config.optimal_spread_bps
    );
    info!(
        " Calculated max_modify_price_bps: {:.2}",
        config.max_modify_price_bps
    );

    // Current order at 100.00
    let order = TrackedOrder::new(1, Side::Buy, 100.00, 1.0);
    let current = vec![&order];

    // Target moved to 100.20 (20bps move) - SAME AS ABOVE
    // But here sigma=20bps, horizon=10 -> vol_bps = 20 * sqrt(10) ~ 63bps.
    // max_modify should be ~100bps (clamped) or ~126bps?
    // So 20bps should be well within modify tolerance.

    // Force size change to trigger modify
    let target_level = LadderLevel {
        price: 100.20, // 20bps away
        size: 0.5,     // Size change
        depth_bps: 10.0,
    };
    let targets = vec![target_level];

    let actions: Vec<LadderAction> =
        priority_based_matching(&current, &targets, Side::Buy, &config, None);

    info!("Actions for 20bps price move (High Vol):");
    for action in &actions {
        info!("  {:?}", action);
    }
}
