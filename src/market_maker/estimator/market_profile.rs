//! Observable-based parameter seeding from L2 market data.
//!
//! `MarketProfile` derives initial kappa, sigma, and regime parameters from the
//! first L2 snapshot and early trades — no fills required. This eliminates the
//! cold-start problem where BTC-calibrated priors produce wrong spreads for
//! assets like HYPE (30 bps BBO vs BTC's 2 bps).
//!
//! # Architecture
//!
//! ```text
//! L2 Book ──→ MarketProfile ──→ implied_kappa, implied_sigma, liquidity_class
//!                  │                    ↓
//!             on_trade()        CalibrationCoordinator (seeds from profile)
//!                  │                    ↓
//!             sigma, rate       RegimeHMM (asset-adaptive emissions)
//! ```

use serde::{Deserialize, Serialize};

// Re-implement the kappa_for_spread inversion locally to avoid cross-module dependency.
// This matches the GLFT formula: kappa = gamma / (exp(gamma * (target_bps - fee_bps) / 10000) - 1)
#[inline]
fn kappa_for_spread(target_half_spread_bps: f64, gamma: f64, fee_bps: f64) -> f64 {
    let safe_gamma = gamma.max(f64::EPSILON);
    let glft_bps = target_half_spread_bps - fee_bps;
    if glft_bps <= 0.0 {
        return f64::INFINITY;
    }
    let exponent = safe_gamma * glft_bps / 10_000.0;
    if exponent > 700.0 {
        return safe_gamma * (-exponent).exp();
    }
    let denom = exponent.exp() - 1.0;
    if denom < 1e-15 {
        return 10_000.0 / glft_bps;
    }
    safe_gamma / denom
}

/// Liquidity classification based on BBO half-spread.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LiquidityClass {
    /// < 3 bps half-spread (e.g. BTC, ETH on major venues)
    VeryLiquid,
    /// 3-10 bps (e.g. SOL, large-cap alts)
    Liquid,
    /// 10-25 bps (e.g. mid-cap perpetuals)
    Moderate,
    /// 25-50 bps (e.g. HYPE, small-cap perps)
    Illiquid,
    /// >= 50 bps (e.g. micro-cap, newly listed)
    VeryIlliquid,
}

impl LiquidityClass {
    /// Classify from BBO half-spread in bps.
    pub fn from_half_spread_bps(half_spread_bps: f64) -> Self {
        if half_spread_bps < 3.0 {
            Self::VeryLiquid
        } else if half_spread_bps < 10.0 {
            Self::Liquid
        } else if half_spread_bps < 25.0 {
            Self::Moderate
        } else if half_spread_bps < 50.0 {
            Self::Illiquid
        } else {
            Self::VeryIlliquid
        }
    }
}

/// HMM emission parameters derived from asset observables.
///
/// Used by `RegimeHMM::seed_emissions()` to center emission distributions
/// on THIS asset's normal conditions rather than BTC defaults.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetEmissionParams {
    /// Normal-regime volatility center (per-second sigma)
    pub normal_vol: f64,
    /// Normal-regime BBO half-spread in bps
    pub normal_spread_bps: f64,
    /// Observed trade rate (trades/second)
    pub trade_rate: f64,
}

/// Safety factor applied to implied kappa for conservative initialization.
const SAFETY_FACTOR: f64 = 0.7;

/// Default gamma for kappa inversion (moderate risk aversion).
const DEFAULT_GAMMA: f64 = 0.15;

/// Default maker fee in bps (Hyperliquid).
const DEFAULT_FEE_BPS: f64 = 1.5;

/// Minimum BBO half-spread before we consider the book valid (bps).
const MIN_BBO_HALF_SPREAD_BPS: f64 = 0.5;

/// Default sigma when no trades observed yet.
const DEFAULT_SIGMA: f64 = 0.00025;

/// EWMA alpha for BBO spread (moderate speed).
const SPREAD_ALPHA: f64 = 0.05;

/// EWMA alpha for depth (slower, more stable).
const DEPTH_ALPHA: f64 = 0.02;

/// EWMA alpha for trade rate (faster, responsive).
const TRADE_RATE_ALPHA: f64 = 0.1;

/// EWMA alpha for sigma from returns.
const SIGMA_ALPHA: f64 = 0.05;

/// Regime kappa multipliers: [Quiet, Normal, Volatile, Extreme]
const REGIME_KAPPA_MULTS: [f64; 4] = [1.5, 1.0, 0.5, 0.25];

/// Observable-based market profile for parameter seeding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketProfile {
    /// EWMA of BBO half-spread in bps.
    bbo_half_spread_bps: f64,
    /// EWMA of total bid depth within 50 bps (in asset units).
    bid_depth_50bps: f64,
    /// EWMA of total ask depth within 50 bps (in asset units).
    ask_depth_50bps: f64,
    /// EWMA of trade rate (trades per second).
    trade_rate: f64,
    /// EWMA of realized sigma (per-second vol from trade returns).
    sigma_realized: f64,
    /// Number of L2 snapshots processed.
    l2_count: usize,
    /// Number of trades processed.
    trade_count: usize,
    /// Last trade price (for return calculation).
    last_trade_price: f64,
    /// Last trade timestamp (ms).
    last_trade_ts_ms: u64,
    /// Whether we've received at least one valid L2 snapshot.
    initialized: bool,
}

impl Default for MarketProfile {
    fn default() -> Self {
        Self::new()
    }
}

impl MarketProfile {
    /// Create an empty, uninitialized profile.
    pub fn new() -> Self {
        Self {
            bbo_half_spread_bps: 0.0,
            bid_depth_50bps: 0.0,
            ask_depth_50bps: 0.0,
            trade_rate: 0.0,
            sigma_realized: DEFAULT_SIGMA,
            l2_count: 0,
            trade_count: 0,
            last_trade_price: 0.0,
            last_trade_ts_ms: 0,
            initialized: false,
        }
    }

    /// Process an L2 book update.
    ///
    /// # Arguments
    /// * `best_bid` - Best bid price
    /// * `best_ask` - Best ask price
    /// * `bid_depth_within_50bps` - Total bid depth within 50 bps of mid
    /// * `ask_depth_within_50bps` - Total ask depth within 50 bps of mid
    pub fn on_l2_book(
        &mut self,
        best_bid: f64,
        best_ask: f64,
        bid_depth_within_50bps: f64,
        ask_depth_within_50bps: f64,
    ) {
        if best_bid <= 0.0 || best_ask <= 0.0 || best_ask <= best_bid {
            return;
        }

        let mid = (best_bid + best_ask) / 2.0;
        let spread = best_ask - best_bid;
        let half_spread_bps = (spread / mid) * 10_000.0 / 2.0;

        if half_spread_bps < MIN_BBO_HALF_SPREAD_BPS {
            return; // Suspicious data
        }

        if self.l2_count == 0 {
            // First snapshot: initialize directly
            self.bbo_half_spread_bps = half_spread_bps;
            self.bid_depth_50bps = bid_depth_within_50bps;
            self.ask_depth_50bps = ask_depth_within_50bps;
        } else {
            // EWMA update
            self.bbo_half_spread_bps =
                ewma(self.bbo_half_spread_bps, half_spread_bps, SPREAD_ALPHA);
            self.bid_depth_50bps = ewma(self.bid_depth_50bps, bid_depth_within_50bps, DEPTH_ALPHA);
            self.ask_depth_50bps = ewma(self.ask_depth_50bps, ask_depth_within_50bps, DEPTH_ALPHA);
        }

        self.l2_count += 1;
        self.initialized = true;
    }

    /// Process a trade observation.
    ///
    /// # Arguments
    /// * `price` - Trade price
    /// * `_size` - Trade size (reserved for future depth-weighted sigma)
    /// * `timestamp_ms` - Trade timestamp in milliseconds
    pub fn on_trade(&mut self, price: f64, _size: f64, timestamp_ms: u64) {
        if price <= 0.0 {
            return;
        }

        if self.trade_count > 0 && self.last_trade_price > 0.0 {
            // Compute return for sigma estimation
            let ret = (price / self.last_trade_price).ln();
            let dt_s = (timestamp_ms.saturating_sub(self.last_trade_ts_ms)) as f64 / 1000.0;

            if dt_s > 0.001 {
                // Per-second sigma from this return
                let sigma_instant = ret.abs() / dt_s.sqrt();
                self.sigma_realized = ewma(self.sigma_realized, sigma_instant, SIGMA_ALPHA);

                // Trade rate: 1/dt smoothed
                let rate_instant = 1.0 / dt_s;
                self.trade_rate = ewma(self.trade_rate, rate_instant, TRADE_RATE_ALPHA);
            }
        }

        self.last_trade_price = price;
        self.last_trade_ts_ms = timestamp_ms;
        self.trade_count += 1;
    }

    /// Whether the profile has been initialized with at least one L2 snapshot.
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Number of L2 snapshots processed.
    pub fn l2_count(&self) -> usize {
        self.l2_count
    }

    /// Number of trades processed.
    pub fn trade_count(&self) -> usize {
        self.trade_count
    }

    /// Current EWMA BBO half-spread in bps.
    pub fn bbo_half_spread_bps(&self) -> f64 {
        self.bbo_half_spread_bps
    }

    /// Implied kappa by inverting the BBO half-spread through the GLFT formula.
    ///
    /// Uses SpreadOracle with default gamma and fee. This represents the kappa
    /// that would produce the observed BBO spread under GLFT.
    ///
    /// For very liquid markets where BBO half-spread <= fee, returns a high
    /// kappa (50000) since the book is extremely deep.
    pub fn implied_kappa(&self) -> f64 {
        if !self.initialized {
            return 1000.0; // Fallback: moderate assumption
        }
        if self.bbo_half_spread_bps <= DEFAULT_FEE_BPS + 0.5 {
            // BBO tighter than fee → extremely liquid market
            return 50000.0;
        }
        let k = kappa_for_spread(self.bbo_half_spread_bps, DEFAULT_GAMMA, DEFAULT_FEE_BPS);
        if k.is_infinite() || k.is_nan() {
            50000.0
        } else {
            k.clamp(10.0, 100000.0)
        }
    }

    /// Conservative kappa: implied × safety factor.
    ///
    /// The safety factor (0.7) ensures we start with wider spreads than the
    /// observed BBO. This is safe because:
    /// 1. The BBO is set by more sophisticated/faster participants
    /// 2. We face adverse selection they don't (slower cancellation)
    /// 3. Tightening from wider is always safe; starting too tight gets run over
    pub fn conservative_kappa(&self) -> f64 {
        let implied = self.implied_kappa();
        let conservative = implied * SAFETY_FACTOR;
        // Floor at 10 (very illiquid) to prevent GLFT blowup
        conservative.max(10.0)
    }

    /// Implied sigma from trade returns, or default if insufficient trades.
    pub fn implied_sigma(&self) -> f64 {
        if self.trade_count < 5 {
            DEFAULT_SIGMA
        } else {
            self.sigma_realized.max(1e-6)
        }
    }

    /// Liquidity classification based on current BBO half-spread.
    pub fn liquidity_class(&self) -> LiquidityClass {
        LiquidityClass::from_half_spread_bps(self.bbo_half_spread_bps)
    }

    /// HMM emission parameters centered on this asset's observed conditions.
    pub fn asset_emission_params(&self) -> AssetEmissionParams {
        AssetEmissionParams {
            normal_vol: self.implied_sigma(),
            normal_spread_bps: self.bbo_half_spread_bps.max(1.0),
            trade_rate: self.trade_rate.max(0.01),
        }
    }

    /// Regime kappa priors as multipliers of conservative_kappa.
    ///
    /// Returns `[Quiet, Normal, Volatile, Extreme]` kappa values:
    /// - Quiet: 1.5× base (tighter spreads in calm markets)
    /// - Normal: 1.0× base
    /// - Volatile: 0.5× base (wider spreads)
    /// - Extreme: 0.25× base (very wide spreads)
    pub fn asset_regime_kappa_priors(&self) -> [f64; 4] {
        let base = self.conservative_kappa();
        [
            base * REGIME_KAPPA_MULTS[0],
            base * REGIME_KAPPA_MULTS[1],
            base * REGIME_KAPPA_MULTS[2],
            base * REGIME_KAPPA_MULTS[3],
        ]
    }

    /// Diagnostic log string for startup.
    pub fn log_summary(&self) -> String {
        format!(
            "MarketProfile initialized: bbo_spread_bps={:.1}, implied_kappa={:.0}, \
             conservative_kappa={:.0}, liquidity_class={:?}, sigma={:.6}, trades={}",
            self.bbo_half_spread_bps * 2.0,
            self.implied_kappa(),
            self.conservative_kappa(),
            self.liquidity_class(),
            self.implied_sigma(),
            self.trade_count,
        )
    }
}

/// Simple EWMA update.
#[inline]
fn ewma(prev: f64, new: f64, alpha: f64) -> f64 {
    (1.0 - alpha) * prev + alpha * new
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a HYPE-like L2 book (30 bps BBO spread).
    fn hype_l2() -> (f64, f64, f64, f64) {
        // BBO spread = 30 bps => half_spread = 15 bps
        // mid = 25.0, spread = 25.0 * 30/10000 = 0.075
        let mid = 25.0;
        let half_spread = mid * 15.0 / 10_000.0; // 0.0375
        (mid - half_spread, mid + half_spread, 100.0, 100.0)
    }

    /// Create a BTC-like L2 book (2 bps BBO spread).
    fn btc_l2() -> (f64, f64, f64, f64) {
        let mid = 50000.0;
        let half_spread = mid * 1.0 / 10_000.0; // 1 bps each side = 2 bps total
        (mid - half_spread, mid + half_spread, 50.0, 50.0)
    }

    /// Create a SOL-like L2 book (8 bps BBO spread).
    fn sol_l2() -> (f64, f64, f64, f64) {
        let mid = 100.0;
        let half_spread = mid * 4.0 / 10_000.0; // 4 bps each side = 8 bps total
        (mid - half_spread, mid + half_spread, 200.0, 200.0)
    }

    #[test]
    fn test_hype_kappa_range() {
        let mut profile = MarketProfile::new();
        let (bid, ask, bd, ad) = hype_l2();
        profile.on_l2_book(bid, ask, bd, ad);

        assert!(profile.is_initialized());
        let ck = profile.conservative_kappa();
        assert!(
            ck > 200.0 && ck < 2000.0,
            "HYPE conservative_kappa should be 200-2000, got {ck:.0}"
        );
        // 15 bps half-spread → Moderate (10-25 bps range)
        assert_eq!(profile.liquidity_class(), LiquidityClass::Moderate);
    }

    #[test]
    fn test_btc_kappa_range() {
        let mut profile = MarketProfile::new();
        let (bid, ask, bd, ad) = btc_l2();
        profile.on_l2_book(bid, ask, bd, ad);

        let ck = profile.conservative_kappa();
        assert!(
            ck > 3000.0,
            "BTC conservative_kappa should be >3000, got {ck:.0}"
        );
        assert_eq!(profile.liquidity_class(), LiquidityClass::VeryLiquid);
    }

    #[test]
    fn test_sol_kappa_range() {
        let mut profile = MarketProfile::new();
        let (bid, ask, bd, ad) = sol_l2();
        profile.on_l2_book(bid, ask, bd, ad);

        let ck = profile.conservative_kappa();
        assert!(
            ck > 1000.0 && ck < 5000.0,
            "SOL conservative_kappa should be 1000-5000, got {ck:.0}"
        );
        assert_eq!(profile.liquidity_class(), LiquidityClass::Liquid);
    }

    #[test]
    fn test_ewma_convergence() {
        let mut profile = MarketProfile::new();
        let (bid, ask, bd, ad) = hype_l2();

        // First update sets directly
        profile.on_l2_book(bid, ask, bd, ad);
        let first = profile.bbo_half_spread_bps;

        // Subsequent updates EWMA-smooth
        for _ in 0..50 {
            profile.on_l2_book(bid, ask, bd, ad);
        }

        // Should converge to same value (constant input)
        assert!(
            (profile.bbo_half_spread_bps - first).abs() < 0.01,
            "EWMA should converge with constant input"
        );
    }

    #[test]
    fn test_sigma_from_trades() {
        let mut profile = MarketProfile::new();
        let (bid, ask, bd, ad) = hype_l2();
        profile.on_l2_book(bid, ask, bd, ad);

        // No trades yet => default sigma
        assert_eq!(profile.implied_sigma(), DEFAULT_SIGMA);

        // Feed synthetic trades (1s apart, ~0.1% moves)
        let base_price = 25.0;
        let mut ts = 1000u64;
        for i in 0..10 {
            let price = base_price * (1.0 + 0.001 * (i as f64 % 3.0 - 1.0));
            profile.on_trade(price, 1.0, ts);
            ts += 1000; // 1 second apart
        }

        let sigma = profile.implied_sigma();
        assert!(
            sigma > 1e-5 && sigma < 0.01,
            "sigma from trades should be reasonable, got {sigma:.6}"
        );
    }

    #[test]
    fn test_asset_emission_params() {
        let mut profile = MarketProfile::new();
        let (bid, ask, bd, ad) = hype_l2();
        profile.on_l2_book(bid, ask, bd, ad);

        let params = profile.asset_emission_params();
        assert!(
            params.normal_spread_bps > 10.0,
            "HYPE normal spread should be >10 bps, got {:.1}",
            params.normal_spread_bps
        );
        assert!(params.normal_vol > 0.0);
    }

    #[test]
    fn test_regime_kappa_priors() {
        let mut profile = MarketProfile::new();
        let (bid, ask, bd, ad) = hype_l2();
        profile.on_l2_book(bid, ask, bd, ad);

        let priors = profile.asset_regime_kappa_priors();
        // Quiet > Normal > Volatile > Extreme
        assert!(priors[0] > priors[1], "Quiet > Normal");
        assert!(priors[1] > priors[2], "Normal > Volatile");
        assert!(priors[2] > priors[3], "Volatile > Extreme");

        // All should be positive
        for (i, &p) in priors.iter().enumerate() {
            assert!(p > 0.0, "Prior[{i}] should be positive, got {p}");
        }
    }

    #[test]
    fn test_uninitialized_defaults() {
        let profile = MarketProfile::new();
        assert!(!profile.is_initialized());
        assert_eq!(profile.l2_count(), 0);
        assert_eq!(profile.trade_count(), 0);
        // Should return reasonable defaults even uninitialized
        assert!(profile.conservative_kappa() > 0.0);
        assert!(profile.implied_sigma() > 0.0);
    }

    #[test]
    fn test_invalid_l2_ignored() {
        let mut profile = MarketProfile::new();
        // Invalid: ask <= bid
        profile.on_l2_book(100.0, 99.0, 50.0, 50.0);
        assert!(!profile.is_initialized());

        // Invalid: zero prices
        profile.on_l2_book(0.0, 100.0, 50.0, 50.0);
        assert!(!profile.is_initialized());
    }

    #[test]
    fn test_serialization_round_trip() {
        let mut profile = MarketProfile::new();
        let (bid, ask, bd, ad) = hype_l2();
        profile.on_l2_book(bid, ask, bd, ad);

        let json = serde_json::to_string(&profile).unwrap();
        let restored: MarketProfile = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.l2_count(), profile.l2_count());
        assert!((restored.conservative_kappa() - profile.conservative_kappa()).abs() < 0.01);
    }

    #[test]
    fn test_log_summary_format() {
        let mut profile = MarketProfile::new();
        let (bid, ask, bd, ad) = hype_l2();
        profile.on_l2_book(bid, ask, bd, ad);

        let summary = profile.log_summary();
        assert!(summary.contains("MarketProfile initialized"));
        assert!(summary.contains("bbo_spread_bps"));
        assert!(summary.contains("implied_kappa"));
        assert!(summary.contains("conservative_kappa"));
        assert!(summary.contains("liquidity_class"));
    }

    #[test]
    fn test_very_illiquid_asset() {
        let mut profile = MarketProfile::new();
        // 100 bps BBO spread (50 bps half)
        let mid = 10.0;
        let half_spread = mid * 50.0 / 10_000.0;
        profile.on_l2_book(mid - half_spread, mid + half_spread, 10.0, 10.0);

        assert_eq!(profile.liquidity_class(), LiquidityClass::VeryIlliquid);
        let ck = profile.conservative_kappa();
        assert!(
            ck < 300.0,
            "Very illiquid should have low kappa, got {ck:.0}"
        );
    }
}
