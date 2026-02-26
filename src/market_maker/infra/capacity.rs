//! Pre-allocation capacity constants for collections.
//!
//! These values are tuned for typical market maker workloads to avoid
//! heap reallocation during hot paths.
//!
//! Multi-asset mode supports up to 1000 orders across multiple assets,
//! following Hyperliquid's default order limit.

/// OrderManager: supports up to 1000 orders for multi-asset mode
/// Single-asset mode (5 bid + 5 ask) uses a small fraction of this.
pub const ORDER_MANAGER_CAPACITY: usize = 1000;

/// Queue tracker: per-asset capacity (25 levels × 2 sides per asset)
pub const QUEUE_TRACKER_CAPACITY: usize = 50;

/// Data quality monitor: AllMids, L2Book, Trades, UserFills channels
pub const DATA_QUALITY_CHANNELS: usize = 4;

/// Fill tracking: typical fills per order before full execution (1-4)
pub const FILL_TID_INLINE_CAPACITY: usize = 4;

/// Bulk order operations: supports larger batches for multi-asset mode
pub const BULK_ORDER_CAPACITY: usize = 200;

/// Ladder level inline capacity: supports up to 25 levels per side
/// Using SmallVec with this capacity keeps ladder data on the stack.
pub const LADDER_LEVEL_INLINE_CAPACITY: usize = 25;

/// Depth computation inline capacity: same as ladder levels
pub const DEPTH_INLINE_CAPACITY: usize = 25;

/// Maximum assets in multi-asset mode (for pre-allocation)
pub const MAX_MULTI_ASSETS: usize = 20;

/// Default order limit per Hyperliquid docs
pub const DEFAULT_ORDER_LIMIT: usize = 1000;
