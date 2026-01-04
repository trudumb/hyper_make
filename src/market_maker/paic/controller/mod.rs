//! Layer 2: Controller - Impulse Control Engine.
//!
//! The controller determines the optimal time and type of intervention
//! for each order based on the market state.
//!
//! # Strategy Matrix
//!
//! |                    | Small Drift | Large Drift |
//! |--------------------|-------------|-------------|
//! | High Priority (π≈0) | HOLD        | LEAK        |
//! | Low Priority (π≈1)  | SHADOW      | RESET       |

mod actions;
mod impulse_engine;
mod priority_value;

pub use actions::{ImpulseAction, ImpulseDecision, OrderAction};
pub use impulse_engine::{ImpulseEngine, OrderState};
pub use priority_value::PriorityValueCalculator;
