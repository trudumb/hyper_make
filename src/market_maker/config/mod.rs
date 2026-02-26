//! Configuration types for the market maker.
//!
//! This module is organized into focused submodules:
//!
//! - `spread_profile`: Spread profile for different market types
//! - `runtime`: Pre-computed asset runtime configuration
//! - `core`: Core market maker config, quote config, monitoring
//! - `risk`: Dynamic risk configuration
//! - `stochastic`: Stochastic module integration settings
//! - `impulse`: Statistical impulse control configuration
//! - `multi_asset`: Multi-asset market making configuration
//! - `regime_profile`: Thin DEX vs liquid CEX regime profiles

pub mod auto_derive;
mod capacity;
mod core;
mod impulse;
mod multi_asset;
mod regime_profile;
mod risk;
mod runtime;
mod shadow_tuner;
mod spread_profile;
mod stochastic;

// Re-export everything for backward compatibility
pub use auto_derive::{auto_derive, CapitalProfile, CapitalTier, DerivedParams, ExchangeContext};
pub use capacity::{CapacityBudget, CapitalAwarePolicy, SizeQuantum, Viability};
pub use core::*;
pub use impulse::*;
pub use multi_asset::*;
pub use regime_profile::*;
pub use risk::*;
pub use runtime::*;
pub use shadow_tuner::ShadowTunerConfig;
pub use spread_profile::*;
pub use stochastic::*;
