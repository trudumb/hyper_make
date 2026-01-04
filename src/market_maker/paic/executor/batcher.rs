//! Order Batcher - Compresses multiple updates into single API calls.
//!
//! Instead of sending orders instantly, the batcher holds them in a
//! micro-buffer (e.g., 5-10ms or until N orders accumulate) before
//! sending as a batch.
//!
//! # Why Batch?
//!
//! Hyperliquid supports batchModify. Updating 4 quotes in 1 message
//! reduces rate limit consumption by 75%.

use std::time::{Duration, Instant};

use super::super::controller::OrderAction;

/// Configuration for order batching.
#[derive(Debug, Clone)]
pub struct BatcherConfig {
    /// Maximum time to wait for batch (microseconds)
    pub max_wait_us: u64,

    /// Maximum orders per batch
    pub max_batch_size: usize,

    /// Minimum importance to trigger immediate flush
    pub immediate_flush_importance: f64,
}

impl Default for BatcherConfig {
    fn default() -> Self {
        Self {
            max_wait_us: 5_000,          // 5ms
            max_batch_size: 4,            // 4 orders per batch
            immediate_flush_importance: 3.0, // Very urgent actions flush immediately
        }
    }
}

/// Entry in the batch queue.
#[derive(Debug, Clone)]
pub(crate) struct BatchEntry {
    /// Order action
    pub action: OrderAction,
    /// Importance score
    #[allow(dead_code)]
    pub importance: f64,
    /// When this entry was added (for latency tracking)
    #[allow(dead_code)]
    pub added_at: Instant,
}

/// Batch of actions ready to send.
#[derive(Debug, Clone)]
pub struct ActionBatch {
    /// Actions in the batch
    pub actions: Vec<OrderAction>,
    /// Total importance
    pub total_importance: f64,
    /// Rate limit tokens needed
    pub tokens_needed: u64,
}

impl ActionBatch {
    /// Check if batch is empty.
    pub fn is_empty(&self) -> bool {
        self.actions.is_empty()
    }

    /// Get number of actions.
    pub fn len(&self) -> usize {
        self.actions.len()
    }

    /// Split into modify and cancel batches.
    pub fn split_by_type(self) -> (Vec<OrderAction>, Vec<OrderAction>) {
        let mut modifies = Vec::new();
        let mut cancels = Vec::new();

        for action in self.actions {
            match action {
                OrderAction::Cancel { .. } => cancels.push(action),
                _ => modifies.push(action),
            }
        }

        (modifies, cancels)
    }
}

/// Order batcher for compressing API calls.
#[derive(Debug)]
pub struct OrderBatcher {
    /// Configuration
    config: BatcherConfig,

    /// Pending entries
    pending: Vec<BatchEntry>,

    /// When the batch window started
    window_start: Option<Instant>,
}

impl OrderBatcher {
    /// Create a new order batcher.
    pub fn new(config: BatcherConfig) -> Self {
        Self {
            config,
            pending: Vec::with_capacity(8),
            window_start: None,
        }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(BatcherConfig::default())
    }

    /// Add an action to the batch.
    ///
    /// Returns Some(batch) if the batch is ready to send.
    pub fn add(&mut self, action: OrderAction, importance: f64) -> Option<ActionBatch> {
        // Skip no-ops
        if action.is_none() {
            return None;
        }

        let now = Instant::now();

        // Start window if first entry
        if self.window_start.is_none() {
            self.window_start = Some(now);
        }

        // Add entry
        self.pending.push(BatchEntry {
            action,
            importance,
            added_at: now,
        });

        // Check if should flush immediately
        if importance >= self.config.immediate_flush_importance {
            return Some(self.flush());
        }

        // Check if batch is full
        if self.pending.len() >= self.config.max_batch_size {
            return Some(self.flush());
        }

        None
    }

    /// Check if batch window has expired.
    pub fn should_flush(&self) -> bool {
        if self.pending.is_empty() {
            return false;
        }

        if let Some(start) = self.window_start {
            let elapsed = start.elapsed().as_micros() as u64;
            elapsed >= self.config.max_wait_us
        } else {
            false
        }
    }

    /// Flush the current batch.
    pub fn flush(&mut self) -> ActionBatch {
        let actions: Vec<OrderAction> = self.pending.drain(..).map(|e| e.action).collect();
        let total_importance: f64 = self.pending.iter().map(|e| e.importance).sum();

        // Calculate tokens needed (1 per order for modifies, batch counts as 1)
        // Batching saves tokens: 4 modifies in 1 batch = 1 token instead of 4
        let tokens_needed = if actions.is_empty() { 0 } else { 1 };

        self.window_start = None;

        ActionBatch {
            actions,
            total_importance,
            tokens_needed,
        }
    }

    /// Try to flush if window has expired.
    pub fn try_flush(&mut self) -> Option<ActionBatch> {
        if self.should_flush() {
            Some(self.flush())
        } else {
            None
        }
    }

    /// Get number of pending actions.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Check if there are pending actions.
    pub fn has_pending(&self) -> bool {
        !self.pending.is_empty()
    }

    /// Get time until window expires.
    pub fn time_until_flush(&self) -> Option<Duration> {
        self.window_start.map(|start| {
            let elapsed = start.elapsed().as_micros() as u64;
            let remaining = self.config.max_wait_us.saturating_sub(elapsed);
            Duration::from_micros(remaining)
        })
    }

    /// Cancel all pending actions for an order.
    pub fn cancel_order(&mut self, oid: u64) {
        self.pending.retain(|e| e.action.oid() != Some(oid));
    }

    /// Clear all pending actions.
    pub fn clear(&mut self) {
        self.pending.clear();
        self.window_start = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batcher_add() {
        let mut batcher = OrderBatcher::default_config();

        let action = OrderAction::ModifyPrice {
            oid: 1,
            new_price: 100.0,
        };

        // First action shouldn't flush
        let result = batcher.add(action, 1.0);
        assert!(result.is_none());
        assert_eq!(batcher.pending_count(), 1);
    }

    #[test]
    fn test_batcher_flush_on_full() {
        let mut batcher = OrderBatcher::new(BatcherConfig {
            max_batch_size: 2,
            ..Default::default()
        });

        let action1 = OrderAction::ModifyPrice {
            oid: 1,
            new_price: 100.0,
        };
        let action2 = OrderAction::ModifyPrice {
            oid: 2,
            new_price: 101.0,
        };

        batcher.add(action1, 1.0);
        let result = batcher.add(action2, 1.0);

        assert!(result.is_some());
        assert_eq!(result.unwrap().len(), 2);
        assert_eq!(batcher.pending_count(), 0);
    }

    #[test]
    fn test_batcher_immediate_flush() {
        let mut batcher = OrderBatcher::new(BatcherConfig {
            immediate_flush_importance: 2.0,
            ..Default::default()
        });

        let action = OrderAction::ModifyPrice {
            oid: 1,
            new_price: 100.0,
        };

        // High importance should flush immediately
        let result = batcher.add(action, 3.0);
        assert!(result.is_some());
    }

    #[test]
    fn test_batcher_skip_none() {
        let mut batcher = OrderBatcher::default_config();

        let result = batcher.add(OrderAction::None, 1.0);
        assert!(result.is_none());
        assert_eq!(batcher.pending_count(), 0);
    }

    #[test]
    fn test_batch_split_by_type() {
        let batch = ActionBatch {
            actions: vec![
                OrderAction::ModifyPrice {
                    oid: 1,
                    new_price: 100.0,
                },
                OrderAction::Cancel { oid: 2 },
                OrderAction::ModifySize {
                    oid: 3,
                    new_size: 0.5,
                },
            ],
            total_importance: 3.0,
            tokens_needed: 1,
        };

        let (modifies, cancels) = batch.split_by_type();
        assert_eq!(modifies.len(), 2);
        assert_eq!(cancels.len(), 1);
    }
}
