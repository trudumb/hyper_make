//! Compact feature snapshot for downstream models.

use crate::market_maker::belief::BeliefSnapshot;
use crate::market_maker::estimator::TradeFlowTracker;
use crate::market_maker::strategy::MarketParams;

/// Compact projection of market state for downstream models.
///
/// Computed once per quote cycle from [`MarketParams`], [`BeliefSnapshot`],
/// and [`TradeFlowTracker`]. Provides a clean input contract for:
/// - Toxicity classification (Phase 2)
/// - Queue-value estimation (Phase 3)
/// - Execution mode selection (Phase 4)
#[derive(Debug, Clone)]
pub struct FeatureSnapshot {
    // === Price / Spread ===
    /// Best bid price
    pub best_bid: f64,
    /// Best ask price
    pub best_ask: f64,
    /// Current spread in basis points
    pub spread_bps: f64,

    // === Order Flow ===
    /// L2 book imbalance [-1, 1]
    pub book_imbalance: f64,
    /// Order flow imbalance, 1-second EWMA
    pub ofi_1s: f64,
    /// Order flow imbalance, 5-second EWMA
    pub ofi_5s: f64,
    /// Order flow imbalance, 30-second EWMA
    pub ofi_30s: f64,

    // === Trade Intensity ===
    /// Hawkes excess intensity ratio (self-exciting / baseline)
    pub trade_intensity_ratio: f64,

    // === Volatility ===
    /// Clean realized volatility (σ_clean)
    pub sigma_clean: f64,
    /// Effective volatility including jumps
    pub sigma_effective: f64,
    /// Jump-diffusion ratio (jump vol / total vol)
    pub jump_ratio: f64,

    // === Regime ===
    /// Regime probabilities [quiet, normal, bursty, cascade]
    pub regime_probs: [f64; 4],

    // === Toxicity ===
    /// Pre-fill toxicity score [0, 1]
    pub toxicity_score: f64,
    /// Probability of informed flow
    pub p_informed: f64,

    // === Inventory ===
    /// Current position in units
    pub inventory: f64,
    /// Position / max_position ratio [-1, 1]
    pub inventory_ratio: f64,

    // === Fill Intensity ===
    /// Estimated fill arrival rate
    pub kappa: f64,

    // === Queue Position ===
    /// Queue rank for bid side [0, 1] (0 = front)
    pub queue_rank_bid: f64,
    /// Queue rank for ask side [0, 1] (0 = front)
    pub queue_rank_ask: f64,
}

impl FeatureSnapshot {
    /// Construct a feature snapshot from existing market state components.
    ///
    /// All fields are extracted from already-computed values — no new computation.
    pub fn from_market_state(
        params: &MarketParams,
        beliefs: &BeliefSnapshot,
        flow_tracker: &TradeFlowTracker,
        inventory: f64,
        max_position: f64,
    ) -> Self {
        let spread_bps = if params.cached_best_bid > 0.0 && params.cached_best_ask > 0.0 {
            let mid = (params.cached_best_bid + params.cached_best_ask) * 0.5;
            if mid > 0.0 {
                (params.cached_best_ask - params.cached_best_bid) / mid * 10_000.0
            } else {
                0.0
            }
        } else {
            params.market_spread_bps
        };

        let inventory_ratio = if max_position > 0.0 {
            (inventory / max_position).clamp(-1.0, 1.0)
        } else {
            0.0
        };

        // Extract regime probs from beliefs if available, else from params
        let regime_probs = if beliefs.regime.confidence > 0.0 {
            beliefs.regime.probs
        } else {
            params.regime_probs
        };

        Self {
            best_bid: params.cached_best_bid,
            best_ask: params.cached_best_ask,
            spread_bps,
            book_imbalance: params.book_imbalance,
            ofi_1s: flow_tracker.imbalance_at_1s(),
            ofi_5s: flow_tracker.imbalance_at_5s(),
            ofi_30s: flow_tracker.imbalance_at_30s(),
            trade_intensity_ratio: params.hawkes_excess_intensity_ratio,
            sigma_clean: params.sigma,
            sigma_effective: params.sigma_effective,
            jump_ratio: params.jump_ratio,
            regime_probs,
            toxicity_score: params.toxicity_score,
            p_informed: params.p_informed,
            inventory,
            inventory_ratio,
            kappa: params.kappa,
            queue_rank_bid: 0.0, // Wired in Phase 3
            queue_rank_ask: 0.0, // Wired in Phase 3
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::market_maker::belief::BeliefSnapshot;
    use crate::market_maker::estimator::TradeFlowTracker;
    use crate::market_maker::strategy::MarketParams;

    fn default_params() -> MarketParams {
        let mut p = MarketParams::default();
        p.cached_best_bid = 100.0;
        p.cached_best_ask = 100.10;
        p.market_spread_bps = 10.0;
        p.book_imbalance = 0.3;
        p.hawkes_excess_intensity_ratio = 1.5;
        p.sigma = 0.02;
        p.sigma_effective = 0.025;
        p.jump_ratio = 0.1;
        p.regime_probs = [0.6, 0.2, 0.15, 0.05];
        p.toxicity_score = 0.35;
        p.p_informed = 0.2;
        p.kappa = 5000.0;
        p
    }

    #[test]
    fn test_snapshot_from_market_state() {
        let params = default_params();
        let beliefs = BeliefSnapshot::default();
        let tracker = TradeFlowTracker::new();
        let snap = FeatureSnapshot::from_market_state(&params, &beliefs, &tracker, 0.5, 2.0);

        assert_eq!(snap.best_bid, 100.0);
        assert_eq!(snap.best_ask, 100.10);
        assert!(snap.spread_bps > 9.9 && snap.spread_bps < 10.1);
        assert!((snap.book_imbalance - 0.3).abs() < 1e-10);
        assert!((snap.inventory_ratio - 0.25).abs() < 1e-10);
        assert_eq!(snap.sigma_clean, 0.02);
        assert_eq!(snap.kappa, 5000.0);
    }

    #[test]
    fn test_snapshot_zero_max_position() {
        let params = default_params();
        let beliefs = BeliefSnapshot::default();
        let tracker = TradeFlowTracker::new();
        let snap = FeatureSnapshot::from_market_state(&params, &beliefs, &tracker, 1.0, 0.0);

        assert_eq!(snap.inventory_ratio, 0.0);
    }

    #[test]
    fn test_snapshot_inventory_ratio_clamped() {
        let params = default_params();
        let beliefs = BeliefSnapshot::default();
        let tracker = TradeFlowTracker::new();
        let snap = FeatureSnapshot::from_market_state(&params, &beliefs, &tracker, 5.0, 2.0);

        assert_eq!(snap.inventory_ratio, 1.0);
    }

    #[test]
    fn test_snapshot_ofi_from_flow_tracker() {
        let params = default_params();
        let beliefs = BeliefSnapshot::default();
        let mut tracker = TradeFlowTracker::new();
        // Feed some trades to get non-zero OFI
        for _ in 0..10 {
            tracker.on_trade(1.0, true);
        }
        let snap = FeatureSnapshot::from_market_state(&params, &beliefs, &tracker, 0.0, 2.0);

        // After all buys, OFI should be positive
        assert!(snap.ofi_1s > 0.0);
        assert!(snap.ofi_5s > 0.0);
        assert!(snap.ofi_30s > 0.0);
    }

    #[test]
    fn test_snapshot_regime_probs_from_beliefs() {
        let params = default_params();
        let mut beliefs = BeliefSnapshot::default();
        beliefs.regime.probs = [0.1, 0.2, 0.3, 0.4];
        beliefs.regime.confidence = 0.8;
        let tracker = TradeFlowTracker::new();
        let snap = FeatureSnapshot::from_market_state(&params, &beliefs, &tracker, 0.0, 2.0);

        assert_eq!(snap.regime_probs, [0.1, 0.2, 0.3, 0.4]);
    }

    #[test]
    fn test_snapshot_regime_probs_falls_back_to_params() {
        let params = default_params();
        let mut beliefs = BeliefSnapshot::default();
        beliefs.regime.confidence = 0.0;
        let tracker = TradeFlowTracker::new();
        let snap = FeatureSnapshot::from_market_state(&params, &beliefs, &tracker, 0.0, 2.0);

        assert_eq!(snap.regime_probs, params.regime_probs);
    }
}
