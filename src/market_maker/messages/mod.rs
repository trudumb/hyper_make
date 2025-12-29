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
//! |  handle_message  |  <- Dispatcher (thin)
//! +------------------+
//!     |    |    |
//!     v    v    v
//! +------+ +------+ +------+
//! | Mids | |Trades| |Fills |  <- Focused processors
//! +------+ +------+ +------+
//! ```

mod context;
mod processors;

pub use context::MessageContext;
pub use processors::*;
