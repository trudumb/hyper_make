//! Message processing module.
//!
//! Breaks down the monolithic handle_message into focused processors.
//!
//! # Architecture
//!
//! ```text
//! WebSocket Message
//!        |
//!        v
//! +------------------+
//! |  handle_message  |  <- Dispatcher (thin, in mod.rs)
//! +------------------+
//!     |    |    |    |
//!     v    v    v    v
//! +------+ +------+ +------+ +------+
//! | Mids | |Trades| |L2Book| |Fills |  <- Focused handlers
//! +------+ +------+ +------+ +------+
//! ```
//!
//! Each handler module provides:
//! - A state struct containing mutable references to needed components
//! - A process function that performs the actual work
//! - Type-safe results for each message type

mod all_mids;
mod context;
mod l2_book;
mod processors;
mod trades;
mod user_fills;

// Re-export all handler components
pub use all_mids::{process_all_mids, AllMidsResult, AllMidsState};
pub use context::MessageContext;
pub use l2_book::{process_l2_book, L2BookState};
pub use processors::*;
pub use trades::{process_trades, TradesState};
pub use user_fills::{cleanup_orders, process_user_fills, FillObservation, UserFillsResult};
