//! Fill processing pipeline.
//!
//! Orchestrates fill deduplication and consumer notification.

use super::consumer::FillConsumerBox;
use super::dedup::FillDeduplicator;
use super::{FillEvent, FillResult};
use tracing::{debug, warn};

/// Fill processing pipeline.
///
/// Provides single-point deduplication and broadcasts fills to all consumers.
/// This is the central coordination point for fill handling, replacing
/// scattered deduplication logic across multiple modules.
///
/// # Usage
///
/// ```ignore
/// let mut pipeline = FillPipeline::new();
/// pipeline.add_consumer(Box::new(PositionConsumer::new()));
/// pipeline.add_consumer(Box::new(PnLConsumer::new()));
///
/// // Process a fill
/// let result = pipeline.process(fill_event);
/// if result.is_new {
///     info!("New fill processed by {} consumers", result.consumers_notified);
/// }
/// ```
pub struct FillPipeline {
    /// Single source of deduplication
    deduplicator: FillDeduplicator,
    /// Registered consumers (sorted by priority)
    consumers: Vec<FillConsumerBox>,
    /// Whether consumers are sorted
    sorted: bool,
}

impl FillPipeline {
    /// Create a new fill pipeline with default deduplication capacity.
    pub fn new() -> Self {
        Self {
            deduplicator: FillDeduplicator::new(),
            consumers: Vec::new(),
            sorted: true,
        }
    }

    /// Create a new fill pipeline with custom deduplication capacity.
    pub fn with_capacity(dedup_capacity: usize) -> Self {
        Self {
            deduplicator: FillDeduplicator::with_capacity(dedup_capacity),
            consumers: Vec::new(),
            sorted: true,
        }
    }

    /// Add a consumer to the pipeline.
    ///
    /// Consumers are sorted by priority before processing.
    pub fn add_consumer(&mut self, consumer: FillConsumerBox) {
        self.consumers.push(consumer);
        self.sorted = false;
    }

    /// Add a consumer using builder pattern.
    pub fn with_consumer(mut self, consumer: FillConsumerBox) -> Self {
        self.add_consumer(consumer);
        self
    }

    /// Process a fill through the pipeline.
    ///
    /// 1. Deduplicates by trade ID
    /// 2. If new, broadcasts to all enabled consumers in priority order
    /// 3. Collects any errors from consumers
    ///
    /// Returns a FillResult indicating success/duplicate status.
    pub fn process(&mut self, fill: FillEvent) -> FillResult {
        // Check for duplicate
        if !self.deduplicator.check_and_mark(fill.tid) {
            debug!(
                tid = fill.tid,
                "Fill is duplicate, skipping"
            );
            return FillResult::duplicate();
        }

        // Ensure consumers are sorted by priority
        if !self.sorted {
            self.consumers.sort_by_key(|c| c.priority());
            self.sorted = true;
        }

        // Broadcast to all enabled consumers
        let mut result = FillResult::new_fill(0);
        for consumer in &mut self.consumers {
            if !consumer.is_enabled() {
                continue;
            }

            if let Some(error) = consumer.on_fill(&fill) {
                warn!(
                    consumer = consumer.name(),
                    error = %error,
                    tid = fill.tid,
                    "Consumer error processing fill"
                );
                result.add_error(consumer.name(), error);
            }
            result.consumers_notified += 1;
        }

        debug!(
            tid = fill.tid,
            oid = fill.oid,
            size = fill.size,
            is_buy = fill.is_buy,
            consumers = result.consumers_notified,
            "Fill processed"
        );

        result
    }

    /// Check if a trade ID has already been processed.
    pub fn is_duplicate(&self, tid: u64) -> bool {
        self.deduplicator.is_duplicate(tid)
    }

    /// Get the number of registered consumers.
    pub fn consumer_count(&self) -> usize {
        self.consumers.len()
    }

    /// Get the number of tracked fills (for deduplication).
    pub fn tracked_fill_count(&self) -> usize {
        self.deduplicator.len()
    }

    /// Clear the deduplication cache.
    ///
    /// Use with caution - this could cause fills to be double-processed.
    pub fn clear_dedup_cache(&mut self) {
        self.deduplicator.clear();
    }

    /// Get consumer names for debugging.
    pub fn consumer_names(&self) -> Vec<&'static str> {
        self.consumers.iter().map(|c| c.name()).collect()
    }
}

impl Default for FillPipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for FillPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FillPipeline")
            .field("consumers", &self.consumer_names())
            .field("tracked_fills", &self.tracked_fill_count())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::market_maker::fills::consumer::{ErrorConsumer, NoOpConsumer};

    fn make_test_fill(tid: u64) -> FillEvent {
        FillEvent::new(
            tid,
            100,
            1.0,
            50000.0,
            true,
            50000.0,
            None,
            "BTC".to_string(),
        )
    }

    #[test]
    fn test_pipeline_new_fill() {
        let mut pipeline = FillPipeline::new();

        let fill = make_test_fill(1);
        let result = pipeline.process(fill);

        assert!(result.is_new);
        assert_eq!(result.consumers_notified, 0); // No consumers
        assert!(result.errors.is_empty());
    }

    #[test]
    fn test_pipeline_duplicate_rejected() {
        let mut pipeline = FillPipeline::new();

        let fill1 = make_test_fill(1);
        let fill2 = make_test_fill(1); // Same TID

        let result1 = pipeline.process(fill1);
        let result2 = pipeline.process(fill2);

        assert!(result1.is_new);
        assert!(!result2.is_new); // Duplicate
    }

    #[test]
    fn test_pipeline_with_consumers() {
        let mut pipeline = FillPipeline::new()
            .with_consumer(Box::new(NoOpConsumer::new("Consumer1")))
            .with_consumer(Box::new(NoOpConsumer::new("Consumer2")));

        let fill = make_test_fill(1);
        let result = pipeline.process(fill);

        assert!(result.is_new);
        assert_eq!(result.consumers_notified, 2);
        assert!(result.errors.is_empty());
    }

    #[test]
    fn test_pipeline_consumer_error() {
        let mut pipeline = FillPipeline::new()
            .with_consumer(Box::new(NoOpConsumer::new("Good")))
            .with_consumer(Box::new(ErrorConsumer::new("test error")));

        let fill = make_test_fill(1);
        let result = pipeline.process(fill);

        assert!(result.is_new);
        assert_eq!(result.consumers_notified, 2);
        assert_eq!(result.errors.len(), 1);
        assert_eq!(result.errors[0].0, "ErrorConsumer");
        assert_eq!(result.errors[0].1, "test error");
    }

    #[test]
    fn test_pipeline_is_duplicate() {
        let mut pipeline = FillPipeline::new();

        assert!(!pipeline.is_duplicate(1));

        pipeline.process(make_test_fill(1));

        assert!(pipeline.is_duplicate(1));
        assert!(!pipeline.is_duplicate(2));
    }

    #[test]
    fn test_pipeline_clear_cache() {
        let mut pipeline = FillPipeline::new();
        pipeline.process(make_test_fill(1));
        assert!(pipeline.is_duplicate(1));

        pipeline.clear_dedup_cache();
        assert!(!pipeline.is_duplicate(1));
    }

    #[test]
    fn test_pipeline_debug() {
        let pipeline = FillPipeline::new()
            .with_consumer(Box::new(NoOpConsumer::new("Test")));

        let debug_str = format!("{:?}", pipeline);
        assert!(debug_str.contains("FillPipeline"));
        assert!(debug_str.contains("Test"));
    }
}
