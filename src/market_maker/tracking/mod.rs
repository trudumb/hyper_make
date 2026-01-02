//! State tracking modules.
//!
//! This module contains components for tracking trading state:
//! - **OrderManager**: Tracks resting orders and their lifecycle
//! - **Position**: Position state tracking
//! - **Queue**: Queue position tracking for fill probability
//! - **PnL**: Profit and loss attribution
//! - **WsOrderState**: WebSocket-based order state management (new)

mod order_manager;
mod pnl;
mod position;
mod queue;
pub mod ws_order_state;

pub use order_manager::*;
pub use pnl::*;
pub use position::*;
pub use queue::*;
pub use ws_order_state::{WsOrderSpec, WsOrderStateConfig, WsOrderStateManager};
