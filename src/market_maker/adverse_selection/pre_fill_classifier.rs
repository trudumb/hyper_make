//! Pre-fill adverse selection classifier.
//!
//! Predicts toxicity BEFORE a fill occurs based on market microstructure signals.
//! This enables proactive spread widening rather than reactive adjustment after losses.
//!
//! # Key Insight
//!
//! Traditional AS measurement tracks price movement AFTER fills (markout analysis).
//! This classifier predicts toxicity BEFORE fills using:
//! - Orderbook imbalance (directional pressure)
//! - Recent trade flow (momentum vs mean-reversion)
//! - Regime trust (changepoint probability)
//! - Funding rate (crowd positioning)
//!
//! # Integration
//!
//! The `spread_multiplier()` output should be fed into GLFT's `tox` variable
//! to widen spreads when toxic flow is predicted.

/// Configuration for pre-fill AS classifier.
#[derive(Debug, Clone)]
pub struct PreFillClassifierConfig {
    /// Weight for orderbook imbalance signal (default: 0.30)
    pub imbalance_weight: f64,
    /// Weight for trade flow momentum signal (default: 0.25)
    pub flow_weight: f64,
    /// Weight for regime distrust signal (default: 0.25)
    pub regime_weight: f64,
    /// Weight for funding rate signal (default: 0.10)
    pub funding_weight: f64,
    /// Weight for BOCD changepoint probability (default: 0.10)
    pub changepoint_weight: f64,

    /// Maximum spread multiplier (caps toxicity impact)
    pub max_spread_multiplier: f64,

    /// Orderbook imbalance threshold for full toxicity (bid/ask ratio)
    /// E.g., 2.0 means 2:1 imbalance = full toxicity signal
    pub imbalance_threshold: f64,

    /// Trade flow threshold for full toxicity (fraction of volume)
    pub flow_threshold: f64,

    /// Funding rate threshold for full toxicity (annualized rate)
    pub funding_threshold: f64,
}

impl Default for PreFillClassifierConfig {
    fn default() -> Self {
        Self {
            imbalance_weight: 0.30,
            flow_weight: 0.25,
            regime_weight: 0.25,
            funding_weight: 0.10,
            changepoint_weight: 0.10,

            max_spread_multiplier: 3.0, // Cap at 3x base spread

            imbalance_threshold: 2.0,   // 2:1 bid/ask ratio
            flow_threshold: 0.7,        // 70% one-sided flow
            funding_threshold: 0.001,   // 0.1% per 8h (annualized ~110%)
        }
    }
}

/// Pre-fill adverse selection classifier.
///
/// Predicts fill toxicity using real-time microstructure signals.
/// Higher toxicity → widen spreads to avoid adverse fills.
#[derive(Debug, Clone)]
pub struct PreFillASClassifier {
    config: PreFillClassifierConfig,

    // === Cached Signal Values ===
    /// Orderbook imbalance: bid_size / ask_size (>1 = bid pressure)
    orderbook_imbalance: f64,

    /// Recent trade direction: net_buy_volume / total_volume [-1, 1]
    recent_trade_direction: f64,

    /// Regime trust from HMM/BOCD (1.0 = high trust, 0.0 = regime uncertain)
    regime_trust: f64,

    /// BOCD changepoint probability [0, 1]
    changepoint_prob: f64,

    /// Current funding rate (8h rate as fraction)
    funding_rate: f64,

    /// Cached toxicity prediction [0, 1]
    cached_toxicity: f64,

    /// Update count for statistics
    update_count: u64,

    /// Timestamp of last orderbook update (ms since epoch)
    orderbook_updated_at_ms: u64,

    /// Timestamp of last trade flow update (ms since epoch)
    trade_flow_updated_at_ms: u64,

    /// Timestamp of last regime update (ms since epoch)
    regime_updated_at_ms: u64,
}

impl PreFillASClassifier {
    /// Create a new pre-fill classifier with default config.
    pub fn new() -> Self {
        Self::with_config(PreFillClassifierConfig::default())
    }

    /// Create with custom configuration.
    pub fn with_config(config: PreFillClassifierConfig) -> Self {
        Self {
            config,
            orderbook_imbalance: 1.0, // Neutral
            recent_trade_direction: 0.0, // Neutral
            regime_trust: 1.0, // High trust initially
            changepoint_prob: 0.0, // No changepoint
            funding_rate: 0.0, // Neutral funding
            cached_toxicity: 0.0,
            update_count: 0,
            orderbook_updated_at_ms: 0,
            trade_flow_updated_at_ms: 0,
            regime_updated_at_ms: 0,
        }
    }

    /// Update orderbook imbalance signal.
    ///
    /// # Arguments
    /// * `bid_depth` - Total bid depth (top N levels)
    /// * `ask_depth` - Total ask depth (top N levels)
    pub fn update_orderbook(&mut self, bid_depth: f64, ask_depth: f64) {
        if ask_depth > 0.0 {
            self.orderbook_imbalance = bid_depth / ask_depth;
        } else {
            self.orderbook_imbalance = if bid_depth > 0.0 { 10.0 } else { 1.0 };
        }
        self.orderbook_updated_at_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        self.update_cached_toxicity();
    }

    /// Update trade flow signal.
    ///
    /// # Arguments
    /// * `buy_volume` - Recent buy volume
    /// * `sell_volume` - Recent sell volume
    pub fn update_trade_flow(&mut self, buy_volume: f64, sell_volume: f64) {
        let total = buy_volume + sell_volume;
        if total > 0.0 {
            self.recent_trade_direction = (buy_volume - sell_volume) / total;
        } else {
            self.recent_trade_direction = 0.0;
        }
        self.trade_flow_updated_at_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        self.update_cached_toxicity();
    }

    /// Update regime trust signal.
    ///
    /// # Arguments
    /// * `hmm_confidence` - HMM regime probability (max across regimes)
    /// * `changepoint_prob` - BOCD changepoint probability
    pub fn update_regime(&mut self, hmm_confidence: f64, changepoint_prob: f64) {
        self.regime_trust = hmm_confidence.clamp(0.0, 1.0);
        self.changepoint_prob = changepoint_prob.clamp(0.0, 1.0);
        self.regime_updated_at_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        self.update_cached_toxicity();
    }

    /// Update funding rate signal.
    ///
    /// # Arguments
    /// * `funding_rate_8h` - 8-hour funding rate as fraction (e.g., 0.0001 = 1 bp)
    pub fn update_funding(&mut self, funding_rate_8h: f64) {
        self.funding_rate = funding_rate_8h;
        self.update_cached_toxicity();
    }

    /// Predict fill toxicity for a given side.
    ///
    /// Returns toxicity probability [0, 1] where:
    /// - 0 = no expected adverse selection
    /// - 1 = maximum expected adverse selection
    ///
    /// # Arguments
    /// * `is_bid` - True if predicting toxicity for bid (buy) side
    pub fn predict_toxicity(&self, is_bid: bool) -> f64 {
        let mut toxicity = 0.0;

        // === Orderbook Imbalance ===
        // If we're bidding, heavy bid pressure = others know something = toxic
        // If we're asking, heavy ask pressure = others know something = toxic
        let imbalance_signal = if is_bid {
            // Bid side: imbalance > 1 means bid pressure, toxic for buyers
            ((self.orderbook_imbalance - 1.0) / (self.config.imbalance_threshold - 1.0))
                .clamp(0.0, 1.0)
        } else {
            // Ask side: imbalance < 1 means ask pressure, toxic for sellers
            ((1.0 / self.orderbook_imbalance - 1.0) / (self.config.imbalance_threshold - 1.0))
                .clamp(0.0, 1.0)
        };
        toxicity += self.config.imbalance_weight * imbalance_signal;

        // === Trade Flow Momentum ===
        // If we're bidding and flow is buying, we're with the crowd = toxic
        // If we're asking and flow is selling, we're with the crowd = toxic
        let flow_signal = if is_bid {
            (self.recent_trade_direction / self.config.flow_threshold).clamp(0.0, 1.0)
        } else {
            (-self.recent_trade_direction / self.config.flow_threshold).clamp(0.0, 1.0)
        };
        toxicity += self.config.flow_weight * flow_signal;

        // === Regime Distrust ===
        // Low regime trust = uncertain market = higher toxicity
        let regime_signal = 1.0 - self.regime_trust;
        toxicity += self.config.regime_weight * regime_signal;

        // === Changepoint Probability ===
        // High changepoint prob = regime shift in progress = higher toxicity
        toxicity += self.config.changepoint_weight * self.changepoint_prob;

        // === Funding Rate ===
        // Extreme funding = crowd positioning = potential reversal = toxic
        let funding_signal =
            (self.funding_rate.abs() / self.config.funding_threshold).clamp(0.0, 1.0);
        toxicity += self.config.funding_weight * funding_signal;

        toxicity.clamp(0.0, 1.0)
    }

    /// Get spread multiplier for a given side.
    ///
    /// Returns a multiplier >= 1.0 to apply to base spread.
    /// Higher toxicity → higher multiplier → wider spreads.
    ///
    /// # Arguments
    /// * `is_bid` - True for bid side, false for ask side
    pub fn spread_multiplier(&self, is_bid: bool) -> f64 {
        let toxicity = self.predict_toxicity(is_bid);
        // Linear scaling: 0 toxicity = 1x, max toxicity = max_multiplier
        let multiplier = 1.0 + toxicity * (self.config.max_spread_multiplier - 1.0);
        multiplier.clamp(1.0, self.config.max_spread_multiplier)
    }

    /// Get symmetric spread multiplier (max of bid/ask).
    ///
    /// Use this when applying the same multiplier to both sides.
    pub fn spread_multiplier_symmetric(&self) -> f64 {
        let bid_mult = self.spread_multiplier(true);
        let ask_mult = self.spread_multiplier(false);
        bid_mult.max(ask_mult)
    }

    /// Get cached toxicity estimate (average of bid/ask).
    pub fn cached_toxicity(&self) -> f64 {
        self.cached_toxicity
    }

    /// Update cached toxicity (average of bid and ask toxicity).
    fn update_cached_toxicity(&mut self) {
        self.cached_toxicity = (self.predict_toxicity(true) + self.predict_toxicity(false)) / 2.0;
        self.update_count += 1;
    }

    /// Check if orderbook signal is stale.
    ///
    /// # Arguments
    /// * `max_age_ms` - Maximum age in milliseconds before considered stale
    pub fn is_orderbook_stale(&self, max_age_ms: u64) -> bool {
        if self.orderbook_updated_at_ms == 0 {
            return true; // Never updated
        }
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        now_ms.saturating_sub(self.orderbook_updated_at_ms) > max_age_ms
    }

    /// Check if trade flow signal is stale.
    pub fn is_trade_flow_stale(&self, max_age_ms: u64) -> bool {
        if self.trade_flow_updated_at_ms == 0 {
            return true;
        }
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        now_ms.saturating_sub(self.trade_flow_updated_at_ms) > max_age_ms
    }

    /// Check if regime signal is stale.
    pub fn is_regime_stale(&self, max_age_ms: u64) -> bool {
        if self.regime_updated_at_ms == 0 {
            return true;
        }
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        now_ms.saturating_sub(self.regime_updated_at_ms) > max_age_ms
    }

    /// Check if any critical signal is stale.
    ///
    /// Returns true if orderbook or trade flow signals are stale.
    /// These are the highest-weight signals in toxicity prediction.
    pub fn has_stale_signals(&self, max_age_ms: u64) -> bool {
        self.is_orderbook_stale(max_age_ms) || self.is_trade_flow_stale(max_age_ms)
    }

    /// Get staleness summary for all signals.
    pub fn staleness_summary(&self, max_age_ms: u64) -> SignalStaleness {
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        SignalStaleness {
            orderbook_age_ms: now_ms.saturating_sub(self.orderbook_updated_at_ms),
            trade_flow_age_ms: now_ms.saturating_sub(self.trade_flow_updated_at_ms),
            regime_age_ms: now_ms.saturating_sub(self.regime_updated_at_ms),
            orderbook_stale: self.is_orderbook_stale(max_age_ms),
            trade_flow_stale: self.is_trade_flow_stale(max_age_ms),
            regime_stale: self.is_regime_stale(max_age_ms),
        }
    }

    /// Get summary for diagnostics.
    pub fn summary(&self) -> PreFillSummary {
        PreFillSummary {
            orderbook_imbalance: self.orderbook_imbalance,
            trade_direction: self.recent_trade_direction,
            regime_trust: self.regime_trust,
            changepoint_prob: self.changepoint_prob,
            funding_rate: self.funding_rate,
            bid_toxicity: self.predict_toxicity(true),
            ask_toxicity: self.predict_toxicity(false),
            bid_spread_mult: self.spread_multiplier(true),
            ask_spread_mult: self.spread_multiplier(false),
            update_count: self.update_count,
            orderbook_updated_at_ms: self.orderbook_updated_at_ms,
            trade_flow_updated_at_ms: self.trade_flow_updated_at_ms,
            regime_updated_at_ms: self.regime_updated_at_ms,
        }
    }
}

impl Default for PreFillASClassifier {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary of pre-fill classifier state.
#[derive(Debug, Clone)]
pub struct PreFillSummary {
    pub orderbook_imbalance: f64,
    pub trade_direction: f64,
    pub regime_trust: f64,
    pub changepoint_prob: f64,
    pub funding_rate: f64,
    pub bid_toxicity: f64,
    pub ask_toxicity: f64,
    pub bid_spread_mult: f64,
    pub ask_spread_mult: f64,
    pub update_count: u64,
    /// Timestamp of last orderbook update (ms since epoch)
    pub orderbook_updated_at_ms: u64,
    /// Timestamp of last trade flow update (ms since epoch)
    pub trade_flow_updated_at_ms: u64,
    /// Timestamp of last regime update (ms since epoch)
    pub regime_updated_at_ms: u64,
}

/// Signal staleness information.
#[derive(Debug, Clone)]
pub struct SignalStaleness {
    pub orderbook_age_ms: u64,
    pub trade_flow_age_ms: u64,
    pub regime_age_ms: u64,
    pub orderbook_stale: bool,
    pub trade_flow_stale: bool,
    pub regime_stale: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_classifier() {
        let classifier = PreFillASClassifier::new();
        // Neutral state should have low toxicity
        assert!(classifier.predict_toxicity(true) < 0.5);
        assert!(classifier.predict_toxicity(false) < 0.5);
        assert!((classifier.spread_multiplier(true) - 1.0).abs() < 1.0);
    }

    #[test]
    fn test_orderbook_imbalance() {
        let mut classifier = PreFillASClassifier::new();

        // Heavy bid pressure → toxic for bids
        classifier.update_orderbook(2.0, 1.0); // 2:1 bid/ask
        assert!(classifier.predict_toxicity(true) > classifier.predict_toxicity(false));

        // Heavy ask pressure → toxic for asks
        classifier.update_orderbook(1.0, 2.0); // 1:2 bid/ask
        assert!(classifier.predict_toxicity(false) > classifier.predict_toxicity(true));
    }

    #[test]
    fn test_trade_flow() {
        let mut classifier = PreFillASClassifier::new();

        // Heavy buy flow → toxic for bids
        classifier.update_trade_flow(80.0, 20.0);
        let bid_tox = classifier.predict_toxicity(true);

        // Reset and test sell flow
        classifier.update_trade_flow(20.0, 80.0);
        let ask_tox = classifier.predict_toxicity(false);

        // Both should be elevated
        assert!(bid_tox > 0.1);
        assert!(ask_tox > 0.1);
    }

    #[test]
    fn test_regime_distrust() {
        let mut classifier = PreFillASClassifier::new();

        // Low HMM confidence + high changepoint = high toxicity
        classifier.update_regime(0.3, 0.8);
        assert!(classifier.predict_toxicity(true) > 0.2);
        assert!(classifier.predict_toxicity(false) > 0.2);

        // High confidence + low changepoint = low toxicity
        classifier.update_regime(0.95, 0.05);
        let low_tox = (classifier.predict_toxicity(true) + classifier.predict_toxicity(false)) / 2.0;
        assert!(low_tox < 0.3);
    }

    #[test]
    fn test_spread_multiplier_capped() {
        let mut classifier = PreFillASClassifier::new();

        // Max out all signals
        classifier.update_orderbook(10.0, 1.0);
        classifier.update_trade_flow(100.0, 0.0);
        classifier.update_regime(0.1, 0.9);
        classifier.update_funding(0.01);

        // Should be capped at max_spread_multiplier
        let mult = classifier.spread_multiplier(true);
        assert!(mult <= classifier.config.max_spread_multiplier + 0.01);
        assert!(mult >= 1.0);
    }

    #[test]
    fn test_summary() {
        let classifier = PreFillASClassifier::new();
        let summary = classifier.summary();

        assert_eq!(summary.orderbook_imbalance, 1.0);
        assert_eq!(summary.trade_direction, 0.0);
        assert_eq!(summary.regime_trust, 1.0);
    }
}
