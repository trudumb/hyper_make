//! Fill consumer trait and implementations.
//!
//! Each consumer handles fills for a specific purpose (position tracking,
//! P&L calculation, adverse selection measurement, etc.).

use super::FillEvent;

/// Type alias for boxed consumer.
pub type FillConsumerBox = Box<dyn FillConsumer>;

/// Trait for fill consumers.
///
/// Each consumer receives deduplicated fill events and processes them
/// for a specific purpose. Consumers should not do their own deduplication
/// as that is handled by the FillPipeline.
///
/// # Thread Safety
///
/// Consumers are `Send + Sync` to allow for potential parallelization.
pub trait FillConsumer: Send + Sync {
    /// Called when a new (deduplicated) fill arrives.
    ///
    /// # Arguments
    ///
    /// * `fill` - The fill event containing all fill data
    ///
    /// # Returns
    ///
    /// An optional error message if processing failed.
    fn on_fill(&mut self, fill: &FillEvent) -> Option<String>;

    /// Consumer name for logging and debugging.
    fn name(&self) -> &'static str;

    /// Priority for ordering (lower = processed first).
    ///
    /// Default is 100. Position tracking should be 0 (first),
    /// P&L should be 10 (after position), metrics can be 200 (last).
    fn priority(&self) -> u32 {
        100
    }

    /// Whether this consumer is enabled.
    ///
    /// Disabled consumers are skipped during fill processing.
    fn is_enabled(&self) -> bool {
        true
    }
}

// Blanket implementation for Box<dyn FillConsumer>
impl FillConsumer for Box<dyn FillConsumer> {
    fn on_fill(&mut self, fill: &FillEvent) -> Option<String> {
        (**self).on_fill(fill)
    }

    fn name(&self) -> &'static str {
        (**self).name()
    }

    fn priority(&self) -> u32 {
        (**self).priority()
    }

    fn is_enabled(&self) -> bool {
        (**self).is_enabled()
    }
}

/// A no-op consumer for testing.
#[cfg(test)]
pub(crate) struct NoOpConsumer {
    pub(crate) name: &'static str,
    pub(crate) fills_received: Vec<u64>,
}

#[cfg(test)]
impl NoOpConsumer {
    pub(crate) fn new(name: &'static str) -> Self {
        Self {
            name,
            fills_received: Vec::new(),
        }
    }
}

// Implement Send and Sync for testing (single-threaded tests only)
#[cfg(test)]
unsafe impl Send for NoOpConsumer {}
#[cfg(test)]
unsafe impl Sync for NoOpConsumer {}

#[cfg(test)]
impl FillConsumer for NoOpConsumer {
    fn on_fill(&mut self, fill: &FillEvent) -> Option<String> {
        self.fills_received.push(fill.tid);
        None
    }

    fn name(&self) -> &'static str {
        self.name
    }
}

/// A consumer that always errors (for testing).
#[cfg(test)]
pub(crate) struct ErrorConsumer {
    pub(crate) error_message: String,
}

#[cfg(test)]
impl ErrorConsumer {
    pub(crate) fn new(message: &str) -> Self {
        Self {
            error_message: message.to_string(),
        }
    }
}

#[cfg(test)]
impl FillConsumer for ErrorConsumer {
    fn on_fill(&mut self, _fill: &FillEvent) -> Option<String> {
        Some(self.error_message.clone())
    }

    fn name(&self) -> &'static str {
        "ErrorConsumer"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_noop_consumer() {
        let mut consumer = NoOpConsumer::new("test");
        let fill = make_test_fill(1);

        let result = consumer.on_fill(&fill);
        assert!(result.is_none());
        assert_eq!(consumer.fills_received.len(), 1);
        assert_eq!(consumer.fills_received[0], 1);
    }

    #[test]
    fn test_error_consumer() {
        let mut consumer = ErrorConsumer::new("test error");
        let fill = make_test_fill(1);

        let result = consumer.on_fill(&fill);
        assert!(result.is_some());
        assert_eq!(result.unwrap(), "test error");
    }

    #[test]
    fn test_default_priority() {
        let consumer = NoOpConsumer::new("test");
        assert_eq!(consumer.priority(), 100);
    }

    #[test]
    fn test_default_enabled() {
        let consumer = NoOpConsumer::new("test");
        assert!(consumer.is_enabled());
    }
}
