//! Priority-Aware Impulse Control (PAIC) Framework for HFT.
//!
//! This framework moves beyond standard "tick-based" updates to a state-dependent
//! system that treats queue priority as an asset and rate limits as a currency.
//!
//! # Architecture Overview
//!
//! The system is divided into three distinct layers:
//!
//! | Layer | Component | Responsibility | Stochastic Inputs |
//! |-------|-----------|----------------|-------------------|
//! | 1. Observer | StateEstimator | Derives hidden states from market data | σ_t, π_t, λ_t |
//! | 2. Controller | ImpulseEngine | Determines optimal intervention timing | V_hold vs V_move, μ_t |
//! | 3. Executor | OrderGateway | Optimizes API mechanics (Batching, Rate Limits) | Token Bucket K_t |
//!
//! # Key Concepts
//!
//! ## Queue Priority Index (π_t)
//!
//! Since Hyperliquid doesn't expose exact queue position via API, we estimate it:
//! ```text
//! π_t = volume_traded_at_level_since_placement / total_volume_at_level
//! ```
//!
//! π ranges from 0.0 (front of queue) to 1.0 (back of queue).
//!
//! ## The Strategy Matrix
//!
//! |                    | Small Drift | Large Drift |
//! |--------------------|-------------|-------------|
//! | High Priority (π≈0) | HOLD        | LEAK        |
//! | Low Priority (π≈1)  | SHADOW      | RESET       |
//!
//! - **HOLD**: Order is an "option" on passive fill, keep it
//! - **LEAK**: Market moving against us, reduce size without losing spot
//! - **SHADOW**: Update price frequently, nothing to lose
//! - **RESET**: Aggressive modify, move price immediately
//!
//! # Module Structure
//!
//! - `config`: PAIC configuration parameters
//! - `observer/`: Layer 1 - State estimation (σ, π, α)
//! - `controller/`: Layer 2 - Impulse control engine
//! - `executor/`: Layer 3 - Order gateway with batching

pub mod config;
pub mod controller;
pub mod executor;
pub mod observer;

pub use config::{PAICConfig, RateLimitConfig};
pub use controller::{ImpulseAction, ImpulseEngine};
pub use executor::{OrderBatcher, OrderGateway, RateLimitShadowPrice};
pub use observer::{MarketState, StateEstimator, VirtualQueueTracker};
