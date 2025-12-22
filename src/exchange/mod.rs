//! Exchange module for interacting with the Hyperliquid exchange.
//!
//! This module provides the `ExchangeClient` and related types for:
//! - Order placement and management
//! - Order cancellation
//! - Asset transfers
//! - Account settings
//!
//! # Submodules
//! - `accounts` - Account settings (leverage, margin, referrer)
//! - `approvals` - Agent and builder fee approvals
//! - `cancels` - Order cancellation methods
//! - `modifies` - Order modification methods
//! - `orders` - Order placement and market orders
//! - `transfers` - Asset transfers and withdrawals

// Type definition modules
mod actions;
mod builder;
mod cancel;
mod exchange_client;
mod exchange_responses;
mod modify;
mod order;
mod signing;

// Method implementation modules (impl ExchangeClient)
mod accounts;
mod approvals;
mod cancels;
mod modifies;
mod orders;
mod transfers;

pub use actions::*;
pub use builder::*;
pub use cancel::{ClientCancelRequest, ClientCancelRequestCloid};
pub use exchange_client::*;
pub use exchange_responses::*;
pub use modify::{ClientModifyRequest, ModifyRequest};
pub use order::{
    ClientLimit, ClientOrder, ClientOrderRequest, ClientTrigger, MarketCloseParams,
    MarketOrderParams, Order,
};
pub use signing::SigningContext;
