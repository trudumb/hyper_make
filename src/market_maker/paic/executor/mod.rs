//! Layer 3: Executor - Order Gateway.
//!
//! This layer optimizes API mechanics:
//! - Order batching (compress multiple updates into single API call)
//! - Rate limit shadow pricing (treat rate limit tokens as currency)

mod batcher;
mod gateway;
mod rate_limit;

pub use batcher::OrderBatcher;
pub use gateway::OrderGateway;
pub use rate_limit::RateLimitShadowPrice;
