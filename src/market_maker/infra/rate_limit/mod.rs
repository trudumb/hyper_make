//! Rate limiting infrastructure.
//!
//! This module provides rate limiting for order operations:
//!
//! - `error_type`: Error classification for different rejection types
//! - `rejection`: Rejection-aware rate limiting with exponential backoff
//! - `proactive`: Proactive rate limit tracking to avoid hitting limits
//!
//! # Design
//!
//! - Separate tracking for bid and ask sides
//! - Exponential backoff with configurable thresholds
//! - Automatic reset on successful order placement
//! - Metrics for monitoring backoff state
//! - Error classification for different rejection types

mod error_type;
mod proactive;
mod rejection;

// Re-export everything for backward compatibility
pub use error_type::*;
pub use proactive::*;
pub use rejection::*;
