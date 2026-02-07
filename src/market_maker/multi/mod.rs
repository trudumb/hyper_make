//! Multi-asset market making module.
//!
//! This module provides infrastructure for quoting multiple assets from a single
//! capital pool, utilizing the 1000-order limit for improved capital efficiency.
//!
//! # Architecture
//!
//! The multi-asset system follows a **Hub-and-Spoke** pattern:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │              MultiAssetCoordinator (Hub)                    │
//! │  - Shared margin pool    - Asset allocator                  │
//! │  - Batch accumulator     - Cross-asset risk                 │
//! │  - Unified kill switch   - Event routing                    │
//! ├─────────────────────────────────────────────────────────────┤
//! │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
//! │  │ Worker   │  │ Worker   │  │ Worker   │  │ Worker   │    │
//! │  │ BTC      │  │ ETH      │  │ SOL      │  │ HYPE     │    │
//! │  │ 10 lvls  │  │ 8 lvls   │  │ 6 lvls   │  │ 5 lvls   │    │
//! │  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Components
//!
//! - **AssetAllocator**: Volatility-weighted order budget allocation
//! - **BatchAccumulator**: Accumulates order operations for batched execution
//! - **SharedMarginPool**: Cross-asset margin tracking and capacity
//! - **AssetWorker**: Per-asset state and quoting logic (future)
//! - **MultiAssetCoordinator**: Central orchestrator (future)
//!
//! # Usage
//!
//! ```ignore
//! use hyper_make::market_maker::multi::{AssetAllocator, AllocationConfig};
//!
//! let config = AllocationConfig::default();
//! let mut allocator = AssetAllocator::new(config);
//!
//! // Add assets
//! allocator.add_asset(AssetId::new("BTC", None));
//! allocator.add_asset(AssetId::new("ETH", None));
//!
//! // Rebalance based on current volatilities
//! let vols = hashmap! {
//!     btc_id => 0.0015,  // 15 bps/tick
//!     eth_id => 0.0025,  // 25 bps/tick
//! };
//! allocator.rebalance(&vols);
//! ```

mod allocator;
mod batch;
mod margin_pool;

pub use allocator::{AllocationConfig, AssetAllocator, AssetBudget};
pub use batch::{BatchAccumulator, BatchEntry, BatchResults};
pub use margin_pool::SharedMarginPool;
