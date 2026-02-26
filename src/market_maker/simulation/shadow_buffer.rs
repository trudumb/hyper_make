//! Lock-free market data buffer for the Shadow Tuner system.
//!
//! Uses a [`flume`] bounded MPSC channel so the live tokio event loop never
//! blocks. The producer side is held by the event loop (zero-lock sends),
//! and the consumer side is held by the ShadowTuner background thread.
//!
//! # Architecture
//!
//! ```text
//! Live Event Loop                     Shadow Tuner Thread
//! ───────────────                     ───────────────────
//! ShadowBufferProducer::push()  ─►  flume channel  ─►  ShadowBufferConsumer::drain()
//!   (try_send, never blocks)                              (try_iter + VecDeque)
//! ```
//!
//! Events are evicted from the consumer-side buffer when either:
//! - The buffer exceeds `max_events` in length, or
//! - The oldest event's timestamp falls outside `max_duration_ns` of the newest.

use std::collections::VecDeque;

use super::replay::ReplayEvent;

/// Producer handle for pushing replay events into the shadow buffer.
///
/// Wraps a [`flume::Sender`] and provides a non-blocking `push()` method.
/// Designed to live on the hot path of the tokio event loop -- `push()` will
/// silently drop events when the channel is full rather than blocking.
#[derive(Clone)]
pub struct ShadowBufferProducer {
    tx: flume::Sender<ReplayEvent>,
}

impl ShadowBufferProducer {
    /// Push an event into the buffer without blocking.
    ///
    /// Returns `true` if the event was enqueued, `false` if the channel was
    /// full and the event was dropped. This is intentional -- the live event
    /// loop must never stall waiting for the shadow tuner.
    pub fn push(&self, event: ReplayEvent) -> bool {
        self.tx.try_send(event).is_ok()
    }

    /// Returns `true` if the consumer side has been dropped.
    pub fn is_disconnected(&self) -> bool {
        self.tx.is_disconnected()
    }
}

/// Consumer handle for draining replay events from the shadow buffer.
///
/// Owns a private [`VecDeque`] that accumulates events pulled from the
/// [`flume::Receiver`]. The shadow tuner thread calls [`drain()`](Self::drain)
/// periodically to ingest all pending events, then [`snapshot()`](Self::snapshot)
/// to obtain an immutable view for replay.
pub struct ShadowBufferConsumer {
    rx: flume::Receiver<ReplayEvent>,
    buffer: VecDeque<ReplayEvent>,
    max_events: usize,
    max_duration_ns: u64,
}

impl ShadowBufferConsumer {
    /// Drain all pending events from the channel into the internal buffer.
    ///
    /// After draining, events are evicted from the front of the buffer if:
    /// - `len > max_events`, or
    /// - `front.timestamp_ns < newest.timestamp_ns - max_duration_ns`
    ///
    /// Returns the number of new events ingested from the channel.
    pub fn drain(&mut self) -> usize {
        let mut ingested = 0usize;
        for event in self.rx.try_iter() {
            self.buffer.push_back(event);
            ingested += 1;
        }

        self.evict();
        ingested
    }

    /// Return a cloned snapshot of the current buffer contents.
    ///
    /// The returned `Vec` is independent of the internal buffer -- mutations
    /// to it do not affect the consumer's state.
    pub fn snapshot(&self) -> Vec<ReplayEvent> {
        self.buffer.iter().cloned().collect()
    }

    /// Number of events currently in the internal buffer.
    ///
    /// This does NOT include events still sitting in the flume channel that
    /// have not yet been [`drain()`](Self::drain)ed.
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Returns `true` if the internal buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Evict stale or excess events from the front of the buffer.
    fn evict(&mut self) {
        // Evict by count limit.
        while self.buffer.len() > self.max_events {
            self.buffer.pop_front();
        }

        // Evict by timestamp duration.
        if let Some(newest_ts) = self.buffer.back().map(|e| e.timestamp_ns()) {
            let cutoff = newest_ts.saturating_sub(self.max_duration_ns);
            while let Some(front) = self.buffer.front() {
                if front.timestamp_ns() < cutoff {
                    self.buffer.pop_front();
                } else {
                    break;
                }
            }
        }
    }
}

/// Create a paired shadow buffer producer and consumer.
///
/// # Arguments
///
/// * `max_events` -- Maximum number of events retained in the consumer buffer.
///   Older events are evicted from the front when this limit is exceeded.
/// * `max_duration_ns` -- Maximum time span (in nanoseconds) between the
///   newest and oldest event in the buffer. Events older than
///   `newest_ts - max_duration_ns` are evicted.
///
/// The underlying flume channel is bounded to 200,000 entries. If the producer
/// outpaces the consumer by more than that, events are silently dropped.
pub fn create_shadow_buffer(
    max_events: usize,
    max_duration_ns: u64,
) -> (ShadowBufferProducer, ShadowBufferConsumer) {
    const CHANNEL_CAPACITY: usize = 200_000;
    let (tx, rx) = flume::bounded(CHANNEL_CAPACITY);

    let producer = ShadowBufferProducer { tx };
    let consumer = ShadowBufferConsumer {
        rx,
        buffer: VecDeque::with_capacity(max_events.min(CHANNEL_CAPACITY)),
        max_events,
        max_duration_ns,
    };

    (producer, consumer)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create an L2Update event with the given timestamp.
    fn l2(ts: u64) -> ReplayEvent {
        ReplayEvent::L2Update {
            timestamp_ns: ts,
            best_bid: 100.0,
            best_ask: 100.10,
            bid_depth: 5.0,
            ask_depth: 5.0,
        }
    }

    /// Helper: create a Trade event with the given timestamp.
    fn trade(ts: u64) -> ReplayEvent {
        ReplayEvent::Trade {
            timestamp_ns: ts,
            price: 100.05,
            size: 1.0,
            is_buy: true,
        }
    }

    #[test]
    fn test_basic_send_drain_snapshot() {
        let (producer, mut consumer) = create_shadow_buffer(1000, u64::MAX);

        // Push a mix of event types.
        assert!(producer.push(l2(1)));
        assert!(producer.push(trade(2)));
        assert!(producer.push(l2(3)));

        // Nothing in buffer until we drain.
        assert_eq!(consumer.len(), 0);

        let ingested = consumer.drain();
        assert_eq!(ingested, 3);
        assert_eq!(consumer.len(), 3);

        // Snapshot should match.
        let snap = consumer.snapshot();
        assert_eq!(snap.len(), 3);
        assert_eq!(snap[0].timestamp_ns(), 1);
        assert_eq!(snap[1].timestamp_ns(), 2);
        assert_eq!(snap[2].timestamp_ns(), 3);
    }

    #[test]
    fn test_eviction_by_count_limit() {
        let max_events = 5;
        let (producer, mut consumer) = create_shadow_buffer(max_events, u64::MAX);

        // Push more events than the count limit.
        for i in 0..10 {
            assert!(producer.push(l2(i)));
        }

        consumer.drain();

        // Only the last `max_events` should remain.
        assert_eq!(consumer.len(), max_events);
        let snap = consumer.snapshot();
        assert_eq!(snap[0].timestamp_ns(), 5);
        assert_eq!(snap[4].timestamp_ns(), 9);
    }

    #[test]
    fn test_eviction_by_timestamp_limit() {
        let max_duration_ns = 100;
        let (producer, mut consumer) = create_shadow_buffer(1000, max_duration_ns);

        // Events spanning 0..250 ns.
        for i in 0..=250 {
            assert!(producer.push(l2(i)));
        }

        consumer.drain();

        // Newest is 250. Cutoff = 250 - 100 = 150.
        // Events with ts < 150 should be evicted.
        let snap = consumer.snapshot();
        assert!(!snap.is_empty());
        assert!(
            snap[0].timestamp_ns() >= 150,
            "oldest event ts {} should be >= cutoff 150",
            snap[0].timestamp_ns()
        );
        assert_eq!(
            snap.last().unwrap().timestamp_ns(),
            250,
            "newest event should be 250"
        );
        // Expected count: timestamps 150..=250 = 101 events.
        assert_eq!(snap.len(), 101);
    }

    #[test]
    fn test_producer_drops_when_channel_full() {
        // Use a very small channel to force drops.
        let (tx, rx) = flume::bounded(3);
        let producer = ShadowBufferProducer { tx };
        let mut consumer = ShadowBufferConsumer {
            rx,
            buffer: VecDeque::new(),
            max_events: 1000,
            max_duration_ns: u64::MAX,
        };

        // Fill the channel exactly.
        assert!(producer.push(l2(1)));
        assert!(producer.push(l2(2)));
        assert!(producer.push(l2(3)));

        // Channel is now full -- next push should fail silently.
        assert!(!producer.push(l2(4)));
        assert!(!producer.push(l2(5)));

        // Only the first 3 made it through.
        consumer.drain();
        assert_eq!(consumer.len(), 3);
        let snap = consumer.snapshot();
        assert_eq!(snap[0].timestamp_ns(), 1);
        assert_eq!(snap[2].timestamp_ns(), 3);
    }

    #[test]
    fn test_snapshot_returns_independent_clone() {
        let (producer, mut consumer) = create_shadow_buffer(1000, u64::MAX);

        producer.push(l2(10));
        producer.push(l2(20));
        consumer.drain();

        let snap1 = consumer.snapshot();
        assert_eq!(snap1.len(), 2);

        // Push more events and drain again.
        producer.push(l2(30));
        consumer.drain();

        // snap1 should be unaffected by the new drain.
        assert_eq!(snap1.len(), 2);
        assert_eq!(snap1[0].timestamp_ns(), 10);
        assert_eq!(snap1[1].timestamp_ns(), 20);

        // But a fresh snapshot reflects the new state.
        let snap2 = consumer.snapshot();
        assert_eq!(snap2.len(), 3);
    }

    #[test]
    fn test_drain_returns_zero_when_empty() {
        let (_producer, mut consumer) = create_shadow_buffer(1000, u64::MAX);
        assert_eq!(consumer.drain(), 0);
        assert!(consumer.is_empty());
    }

    #[test]
    fn test_incremental_drain() {
        let (producer, mut consumer) = create_shadow_buffer(1000, u64::MAX);

        // First batch.
        producer.push(l2(1));
        producer.push(l2(2));
        assert_eq!(consumer.drain(), 2);
        assert_eq!(consumer.len(), 2);

        // Second batch accumulates on top.
        producer.push(l2(3));
        assert_eq!(consumer.drain(), 1);
        assert_eq!(consumer.len(), 3);

        let snap = consumer.snapshot();
        assert_eq!(snap[0].timestamp_ns(), 1);
        assert_eq!(snap[2].timestamp_ns(), 3);
    }

    #[test]
    fn test_eviction_combined_count_and_timestamp() {
        // Count limit = 10, duration limit = 50 ns.
        // Duration limit should be the binding constraint here.
        let (producer, mut consumer) = create_shadow_buffer(10, 50);

        // Push 20 events spanning 0..19 ns (all within 50 ns window).
        for i in 0..20 {
            producer.push(l2(i));
        }
        consumer.drain();
        // Count limit (10) kicks in first since time span (19 ns) < 50 ns.
        assert_eq!(consumer.len(), 10);
        assert_eq!(consumer.snapshot()[0].timestamp_ns(), 10);

        // Now push events with a large time gap so duration limit dominates.
        // Clear by pushing far-future events.
        for i in 0..5 {
            producer.push(l2(1000 + i));
        }
        consumer.drain();
        // Buffer: timestamps [10..19] + [1000..1004] = 15 items before eviction.
        // Count limit evicts to 10, then duration evicts anything < 1004 - 50 = 954.
        // So only [1000..1004] survive.
        assert_eq!(consumer.len(), 5);
        assert_eq!(consumer.snapshot()[0].timestamp_ns(), 1000);
    }

    #[test]
    fn test_producer_clone() {
        let (producer, mut consumer) = create_shadow_buffer(1000, u64::MAX);
        let producer2 = producer.clone();

        producer.push(l2(1));
        producer2.push(l2(2));

        consumer.drain();
        assert_eq!(consumer.len(), 2);
    }

    #[test]
    fn test_disconnection_detection() {
        let (producer, consumer) = create_shadow_buffer(1000, u64::MAX);
        assert!(!producer.is_disconnected());

        drop(consumer);
        assert!(producer.is_disconnected());
        // push should return false after consumer is dropped.
        assert!(!producer.push(l2(1)));
    }

    #[test]
    fn test_mixed_event_types_preserved() {
        let (producer, mut consumer) = create_shadow_buffer(1000, u64::MAX);

        producer.push(l2(1));
        producer.push(trade(2));
        producer.push(l2(3));
        producer.push(trade(4));

        consumer.drain();
        let snap = consumer.snapshot();
        assert_eq!(snap.len(), 4);

        // Verify variant types are preserved by checking timestamps and
        // pattern-matching the variants.
        assert!(matches!(snap[0], ReplayEvent::L2Update { .. }));
        assert!(matches!(snap[1], ReplayEvent::Trade { .. }));
        assert!(matches!(snap[2], ReplayEvent::L2Update { .. }));
        assert!(matches!(snap[3], ReplayEvent::Trade { .. }));
    }
}
