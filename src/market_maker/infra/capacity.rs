//! Pre-allocation capacity constants for collections.
//!
//! These values are tuned for typical market maker workloads to avoid
//! heap reallocation during hot paths.

/// OrderManager: 5 bid levels + 5 ask levels + margin for partial fills
pub const ORDER_MANAGER_CAPACITY: usize = 20;

/// Queue tracker: same as order manager
pub const QUEUE_TRACKER_CAPACITY: usize = 20;

/// Data quality monitor: AllMids, L2Book, Trades, UserFills channels
pub const DATA_QUALITY_CHANNELS: usize = 4;

/// Fill tracking: typical fills per order before full execution (1-4)
pub const FILL_TID_INLINE_CAPACITY: usize = 4;

/// Bulk order operations: typical ladder size
pub const BULK_ORDER_CAPACITY: usize = 10;

/// Ladder level inline capacity: typical 5 levels per side, 8 for safety margin
/// Using SmallVec with this capacity keeps ladder data on the stack.
pub const LADDER_LEVEL_INLINE_CAPACITY: usize = 8;

/// Depth computation inline capacity: same as ladder levels
pub const DEPTH_INLINE_CAPACITY: usize = 8;
