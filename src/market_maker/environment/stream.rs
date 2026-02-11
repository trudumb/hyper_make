//! Observation stream implementations.
//!
//! Each environment produces observations through a typed stream.
//! The `LiveObservationStream` translates `Message` → `Observation`,
//! filtering out non-observation messages (NoData, Pong, etc.).

use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use futures_util::Stream;
use tokio::sync::mpsc;

use crate::Message;

use super::Observation;

/// Observation stream for the live environment.
///
/// Receives [`Message`]s from the WS subscription channel and translates
/// them to [`Observation`]s, filtering out non-observation messages.
pub struct LiveObservationStream {
    receiver: mpsc::UnboundedReceiver<Arc<Message>>,
}

impl LiveObservationStream {
    /// Create a new live observation stream from a WS message receiver.
    pub fn new(receiver: mpsc::UnboundedReceiver<Arc<Message>>) -> Self {
        Self { receiver }
    }
}

impl Stream for LiveObservationStream {
    type Item = Observation;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            match self.receiver.poll_recv(cx) {
                Poll::Ready(Some(arc_msg)) => {
                    // Zero-copy unwrap: Arc::try_unwrap succeeds when we're the only owner.
                    // Falls back to clone only if Arc is still shared.
                    let msg = Arc::try_unwrap(arc_msg).unwrap_or_else(|arc| (*arc).clone());

                    // Try to convert Message → Observation.
                    // Skip non-observation messages (NoData, Pong, SubscriptionResponse, etc.)
                    match Observation::try_from(msg) {
                        Ok(obs) => return Poll::Ready(Some(obs)),
                        Err(()) => continue, // Skip and poll next message
                    }
                }
                Poll::Ready(None) => return Poll::Ready(None),
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}
