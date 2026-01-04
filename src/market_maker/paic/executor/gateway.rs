//! Order Gateway - Unified entry point for order execution.
//!
//! Combines batching and rate limit shadow pricing to optimize API usage.
//!
//! # Key Principles
//!
//! 1. **Prefer Modify over Cancel+Place**: Atomic replacement is safer
//! 2. **Batch Updates**: Multiple orders in one API call
//! 3. **Shadow Pricing**: High-importance actions get priority when rate limited

use super::batcher::{ActionBatch, BatcherConfig, OrderBatcher};
use super::rate_limit::RateLimitShadowPrice;
use super::super::config::{PAICConfig, RateLimitConfig};
use super::super::controller::{ImpulseAction, ImpulseDecision, OrderAction};

/// Result of submitting an action to the gateway.
#[derive(Debug)]
pub enum GatewayResult {
    /// Action was batched, waiting for flush
    Batched,
    /// Action was flushed immediately
    Flushed(ActionBatch),
    /// Action was rejected due to rate limiting
    RateLimited { importance: f64, cost: f64 },
    /// Action was skipped (no-op)
    Skipped,
}

/// Gateway statistics.
#[derive(Debug, Clone, Default)]
pub struct GatewayStats {
    /// Total actions submitted
    pub total_submitted: u64,
    /// Actions batched
    pub batched: u64,
    /// Batches flushed
    pub batches_flushed: u64,
    /// Actions rejected due to rate limiting
    pub rate_limited: u64,
    /// Total tokens consumed
    pub tokens_consumed: u64,
}

/// Order Gateway - manages batching and rate limiting.
#[derive(Debug)]
pub struct OrderGateway {
    /// Order batcher
    batcher: OrderBatcher,

    /// Rate limit shadow price
    rate_limiter: RateLimitShadowPrice,

    /// Statistics
    stats: GatewayStats,
}

impl OrderGateway {
    /// Create a new order gateway.
    pub fn new(config: PAICConfig) -> Self {
        let batcher_config = BatcherConfig {
            max_wait_us: config.batch_window_us,
            max_batch_size: config.max_batch_size,
            ..Default::default()
        };

        Self {
            batcher: OrderBatcher::new(batcher_config),
            rate_limiter: RateLimitShadowPrice::new(config.rate_limit_config),
            stats: GatewayStats::default(),
        }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(PAICConfig::default())
    }

    /// Create with custom rate limit config.
    pub fn with_rate_limit(rate_config: RateLimitConfig) -> Self {
        Self {
            batcher: OrderBatcher::default_config(),
            rate_limiter: RateLimitShadowPrice::new(rate_config),
            stats: GatewayStats::default(),
        }
    }

    /// Submit an impulse decision to the gateway.
    pub fn submit(&mut self, decision: &ImpulseDecision) -> GatewayResult {
        self.stats.total_submitted += 1;

        // Skip HOLDs
        if decision.action == ImpulseAction::Hold {
            return GatewayResult::Skipped;
        }

        // Check rate limit
        let importance = decision.importance;
        let cost = self.rate_limiter.shadow_cost();

        if !self.rate_limiter.check_budget(importance) {
            self.stats.rate_limited += 1;
            return GatewayResult::RateLimited { importance, cost };
        }

        // Add to batcher
        let action = decision.order_action.clone();
        self.submit_action(action, importance)
    }

    /// Submit a raw order action.
    pub fn submit_action(&mut self, action: OrderAction, importance: f64) -> GatewayResult {
        if action.is_none() {
            return GatewayResult::Skipped;
        }

        // Add to batcher
        if let Some(batch) = self.batcher.add(action, importance) {
            self.on_batch_ready(batch)
        } else {
            self.stats.batched += 1;
            GatewayResult::Batched
        }
    }

    /// Called when a batch is ready to send.
    fn on_batch_ready(&mut self, batch: ActionBatch) -> GatewayResult {
        // Consume rate limit tokens
        if self.rate_limiter.consume(batch.tokens_needed) {
            self.stats.batches_flushed += 1;
            self.stats.tokens_consumed += batch.tokens_needed;
            GatewayResult::Flushed(batch)
        } else {
            // This shouldn't happen often since we check budget before adding
            self.stats.rate_limited += 1;
            GatewayResult::RateLimited {
                importance: batch.total_importance,
                cost: self.rate_limiter.shadow_cost(),
            }
        }
    }

    /// Try to flush the current batch if window expired.
    pub fn try_flush(&mut self) -> Option<ActionBatch> {
        if let Some(batch) = self.batcher.try_flush() {
            if batch.is_empty() {
                return None;
            }

            // Consume rate limit tokens
            if self.rate_limiter.consume(batch.tokens_needed) {
                self.stats.batches_flushed += 1;
                self.stats.tokens_consumed += batch.tokens_needed;
                return Some(batch);
            }
        }
        None
    }

    /// Force flush the current batch.
    pub fn force_flush(&mut self) -> Option<ActionBatch> {
        if !self.batcher.has_pending() {
            return None;
        }

        let batch = self.batcher.flush();
        if batch.is_empty() {
            return None;
        }

        // Try to consume tokens
        if self.rate_limiter.consume(batch.tokens_needed) {
            self.stats.batches_flushed += 1;
            self.stats.tokens_consumed += batch.tokens_needed;
            Some(batch)
        } else {
            // Put actions back? For now, just drop them
            self.stats.rate_limited += 1;
            None
        }
    }

    /// Check if gateway is rate limited.
    pub fn is_rate_limited(&mut self) -> bool {
        self.rate_limiter.is_rate_limited()
    }

    /// Get current shadow cost.
    pub fn shadow_cost(&mut self) -> f64 {
        self.rate_limiter.shadow_cost()
    }

    /// Get available rate limit tokens.
    pub fn available_tokens(&mut self) -> f64 {
        self.rate_limiter.current_tokens()
    }

    /// Get rate limit utilization.
    pub fn utilization(&mut self) -> f64 {
        self.rate_limiter.utilization()
    }

    /// Get number of pending actions in batcher.
    pub fn pending_count(&self) -> usize {
        self.batcher.pending_count()
    }

    /// Get gateway statistics.
    pub fn stats(&self) -> &GatewayStats {
        &self.stats
    }

    /// Reset statistics.
    pub fn reset_stats(&mut self) {
        self.stats = GatewayStats::default();
    }

    /// Cancel pending actions for an order.
    pub fn cancel_pending(&mut self, oid: u64) {
        self.batcher.cancel_order(oid);
    }

    /// Clear all pending actions.
    pub fn clear(&mut self) {
        self.batcher.clear();
    }

    /// Get reference to batcher.
    pub fn batcher(&self) -> &OrderBatcher {
        &self.batcher
    }

    /// Get mutable reference to batcher.
    pub fn batcher_mut(&mut self) -> &mut OrderBatcher {
        &mut self.batcher
    }

    /// Get reference to rate limiter.
    pub fn rate_limiter(&self) -> &RateLimitShadowPrice {
        &self.rate_limiter
    }

    /// Get mutable reference to rate limiter.
    pub fn rate_limiter_mut(&mut self) -> &mut RateLimitShadowPrice {
        &mut self.rate_limiter
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gateway_submit_hold() {
        let mut gateway = OrderGateway::default_config();
        let decision = ImpulseDecision::hold(1, 0.1, 1.0, 5.0, 10.0);

        match gateway.submit(&decision) {
            GatewayResult::Skipped => (),
            _ => panic!("Expected Skipped for HOLD"),
        }
    }

    #[test]
    fn test_gateway_submit_action() {
        let mut gateway = OrderGateway::default_config();

        let action = OrderAction::ModifyPrice {
            oid: 1,
            new_price: 100.0,
        };

        match gateway.submit_action(action, 1.0) {
            GatewayResult::Batched => (),
            _ => panic!("Expected Batched"),
        }

        assert_eq!(gateway.pending_count(), 1);
    }

    #[test]
    fn test_gateway_force_flush() {
        let mut gateway = OrderGateway::default_config();

        // Add some actions
        gateway.submit_action(
            OrderAction::ModifyPrice {
                oid: 1,
                new_price: 100.0,
            },
            1.0,
        );
        gateway.submit_action(
            OrderAction::ModifyPrice {
                oid: 2,
                new_price: 101.0,
            },
            1.0,
        );

        let batch = gateway.force_flush();
        assert!(batch.is_some());
        assert_eq!(batch.unwrap().len(), 2);
        assert_eq!(gateway.pending_count(), 0);
    }

    #[test]
    fn test_gateway_rate_limiting() {
        let mut gateway = OrderGateway::with_rate_limit(RateLimitConfig {
            max_tokens: 10,
            reserve_ratio: 0.0,
            min_importance: 0.5,
            ..Default::default()
        });

        // Exhaust tokens
        for _ in 0..10 {
            gateway.rate_limiter_mut().consume(1);
        }

        // Force flush to trigger rate limiting (batch consume fails)
        gateway.submit_action(
            OrderAction::ModifyPrice { oid: 1, new_price: 100.0 },
            1.0,
        );

        // Force flush should fail due to no tokens
        let batch = gateway.force_flush();
        // No batch returned because tokens exhausted
        assert!(batch.is_none(), "Expected batch to be None when rate limited");
    }

    #[test]
    fn test_gateway_stats() {
        let mut gateway = OrderGateway::default_config();

        // submit_action increments batched stat
        gateway.submit_action(
            OrderAction::ModifyPrice {
                oid: 1,
                new_price: 100.0,
            },
            1.0,
        );

        let stats = gateway.stats();
        // submit_action increments batched, not total_submitted
        assert_eq!(stats.batched, 1, "Expected batched = 1");
    }
}
