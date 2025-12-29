//! Fill consumer implementations.
//!
//! Each consumer handles fills for a specific purpose.

mod position;
mod pnl;
mod metrics;
mod adverse_selection;
mod estimator;

pub use position::PositionConsumer;
pub use pnl::PnLConsumer;
pub use metrics::MetricsConsumer;
pub use adverse_selection::AdverseSelectionConsumer;
pub use estimator::EstimatorConsumer;
