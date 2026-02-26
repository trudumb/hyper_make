//! Edge Surface Quantification
//!
//! Quantifies expected market making edge as a function of market conditions.
//!
//! # Grid Structure
//!
//! The edge surface is discretized over 4 dimensions:
//!
//! - **Volatility**: 5 buckets (very low → very high)
//! - **Regime**: 3 states (LOW, NORMAL, HIGH)
//! - **Hour**: 3 buckets (Asia, London, US)
//! - **Flow**: 5 buckets (strong sell → strong buy)
//!
//! Total: 5 × 3 × 3 × 5 = 225 cells
//!
//! # Edge Calculation
//!
//! ```text
//! Edge = E[Spread Captured] - E[Adverse Selection] - Fees
//!      = (spread/2) × fill_rate - AS - fees
//! ```
//!
//! # Usage
//!
//! ```ignore
//! let mut surface = EdgeSurface::new(EdgeSurfaceConfig::default());
//!
//! // Update with observations
//! surface.observe(&observation);
//!
//! // Get edge estimate for current conditions
//! let condition = MarketCondition::from_state(sigma, regime, hour, flow);
//! let estimate = surface.edge_estimate(&condition);
//!
//! if estimate.should_quote() {
//!     // Expected positive edge with confidence
//! }
//! ```

use super::{EdgeEstimate, MarketCondition};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

// ============================================================================
// Constants
// ============================================================================

/// Total grid cells (5 vol × 3 regime × 3 hour × 5 flow)
pub const GRID_SIZE: usize = 225;

/// Default fees in basis points
pub const DEFAULT_FEES_BPS: f64 = 1.5;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for edge surface
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EdgeSurfaceConfig {
    /// Trading fees (bps, one side)
    pub fees_bps: f64,

    /// Minimum observations per cell for reliable estimate
    pub min_cell_observations: usize,

    /// EWMA half-life for cell updates
    pub ewma_half_life: usize,

    /// Minimum edge for quoting (bps)
    pub min_edge_bps: f64,

    /// Minimum confidence for quoting
    pub min_confidence: f64,

    /// Maximum observations per cell
    pub max_cell_observations: usize,

    /// Prior edge estimate (bps)
    pub prior_edge_bps: f64,

    /// Prior uncertainty (bps)
    pub prior_uncertainty_bps: f64,

    /// Recalculation interval (observations)
    pub recalc_interval: usize,
}

impl Default for EdgeSurfaceConfig {
    fn default() -> Self {
        Self {
            fees_bps: DEFAULT_FEES_BPS,
            min_cell_observations: 10,
            ewma_half_life: 100,
            min_edge_bps: 1.0,
            min_confidence: 0.6,
            max_cell_observations: 500,
            prior_edge_bps: 2.0,        // Assume 2 bps baseline edge
            prior_uncertainty_bps: 5.0, // High uncertainty initially
            recalc_interval: 50,
        }
    }
}

// ============================================================================
// Cell Statistics
// ============================================================================

/// Statistics for a single grid cell
#[derive(Debug, Clone)]
struct CellStats {
    /// EWMA of edge observations
    mean_edge: f64,

    /// EWMA of squared edge (for variance)
    mean_edge_sq: f64,

    /// EWMA of spread captured
    mean_spread: f64,

    /// EWMA of adverse selection
    mean_as: f64,

    /// EWMA of fill rate
    mean_fill_rate: f64,

    /// Number of observations
    n_obs: usize,
}

impl CellStats {
    fn new(prior_edge: f64, prior_var: f64) -> Self {
        Self {
            mean_edge: prior_edge,
            mean_edge_sq: prior_edge * prior_edge + prior_var,
            mean_spread: 8.0,    // Default 8 bps spread
            mean_as: 2.0,        // Default 2 bps AS
            mean_fill_rate: 0.1, // Default 10% fill rate
            n_obs: 0,
        }
    }

    fn variance(&self) -> f64 {
        (self.mean_edge_sq - self.mean_edge * self.mean_edge).max(0.0)
    }

    fn std_error(&self) -> f64 {
        if self.n_obs < 2 {
            f64::INFINITY
        } else {
            (self.variance() / self.n_obs as f64).sqrt()
        }
    }

    fn update(&mut self, edge: f64, spread: f64, as_bps: f64, fill_rate: f64, alpha: f64) {
        if self.n_obs == 0 {
            self.mean_edge = edge;
            self.mean_edge_sq = edge * edge;
            self.mean_spread = spread;
            self.mean_as = as_bps;
            self.mean_fill_rate = fill_rate;
        } else {
            self.mean_edge = alpha * edge + (1.0 - alpha) * self.mean_edge;
            self.mean_edge_sq = alpha * edge * edge + (1.0 - alpha) * self.mean_edge_sq;
            self.mean_spread = alpha * spread + (1.0 - alpha) * self.mean_spread;
            self.mean_as = alpha * as_bps + (1.0 - alpha) * self.mean_as;
            self.mean_fill_rate = alpha * fill_rate + (1.0 - alpha) * self.mean_fill_rate;
        }
        self.n_obs += 1;
    }
}

// ============================================================================
// Edge Observation
// ============================================================================

/// A single edge observation for updating the surface
#[derive(Debug, Clone)]
pub struct EdgeObservation {
    /// Market condition at observation
    pub condition: MarketCondition,

    /// Spread at which order was placed (bps)
    pub spread_bps: f64,

    /// Whether fill occurred
    pub filled: bool,

    /// Realized adverse selection if filled (bps)
    pub realized_as_bps: f64,

    /// Timestamp (milliseconds)
    pub timestamp_ms: u64,
}

// ============================================================================
// Edge Surface
// ============================================================================

/// Edge surface for quantifying expected market making edge by condition
#[derive(Debug)]
pub struct EdgeSurface {
    /// Configuration
    config: EdgeSurfaceConfig,

    /// Grid of cell statistics (225 cells)
    grid: Vec<CellStats>,

    /// Recent observations (for batch updates)
    recent_observations: VecDeque<EdgeObservation>,

    /// EWMA alpha
    ewma_alpha: f64,

    /// Total observations
    total_observations: usize,

    /// Observations since last recalculation
    obs_since_recalc: usize,

    /// Global edge statistics (for cells with insufficient data)
    global_stats: CellStats,
}

impl EdgeSurface {
    /// Create new edge surface
    pub fn new(config: EdgeSurfaceConfig) -> Self {
        let ewma_alpha = 1.0 - 0.5f64.powf(1.0 / config.ewma_half_life as f64);
        let prior_var = config.prior_uncertainty_bps.powi(2);

        let grid = (0..GRID_SIZE)
            .map(|_| CellStats::new(config.prior_edge_bps, prior_var))
            .collect();

        Self {
            config: config.clone(),
            grid,
            recent_observations: VecDeque::with_capacity(1000),
            ewma_alpha,
            total_observations: 0,
            obs_since_recalc: 0,
            global_stats: CellStats::new(config.prior_edge_bps, prior_var),
        }
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        Self::new(EdgeSurfaceConfig::default())
    }

    /// Record an edge observation
    pub fn observe(&mut self, obs: &EdgeObservation) {
        // Calculate realized edge
        let realized_edge = if obs.filled {
            obs.spread_bps / 2.0 - obs.realized_as_bps - self.config.fees_bps
        } else {
            0.0 // No edge if no fill
        };

        let fill_rate = if obs.filled { 1.0 } else { 0.0 };

        // Update specific cell
        let idx = obs.condition.grid_index();
        if idx < GRID_SIZE {
            self.grid[idx].update(
                realized_edge,
                obs.spread_bps,
                obs.realized_as_bps,
                fill_rate,
                self.ewma_alpha,
            );
        }

        // Update global stats
        self.global_stats.update(
            realized_edge,
            obs.spread_bps,
            obs.realized_as_bps,
            fill_rate,
            self.ewma_alpha,
        );

        // Store recent observation
        if self.recent_observations.len() >= 1000 {
            self.recent_observations.pop_front();
        }
        self.recent_observations.push_back(obs.clone());

        self.total_observations += 1;
        self.obs_since_recalc += 1;
    }

    /// Get edge estimate for specific condition
    pub fn edge_estimate(&self, condition: &MarketCondition) -> EdgeEstimate {
        let idx = condition.grid_index();

        let cell = if idx < GRID_SIZE && self.grid[idx].n_obs >= self.config.min_cell_observations {
            &self.grid[idx]
        } else {
            // Use global stats if cell has insufficient data
            &self.global_stats
        };

        let edge_bps = cell.mean_edge;
        let uncertainty_bps = cell.std_error().min(self.config.prior_uncertainty_bps);

        // Confidence based on observations and consistency
        let n_factor = (cell.n_obs as f64 / 50.0).tanh();
        let var_factor = if cell.variance() > 1e-10 {
            // Higher mean/std ratio = higher consistency = higher confidence
            (cell.mean_edge.abs() / cell.variance().sqrt()).tanh()
        } else {
            // Zero variance means perfectly consistent observations
            // If there are enough observations, this should give high confidence
            if cell.n_obs >= 2 {
                1.0
            } else {
                0.0
            }
        };
        let confidence = n_factor * 0.6 + var_factor * 0.4;

        EdgeEstimate {
            edge_bps,
            uncertainty_bps,
            confidence,
            optimal_spread_bps: cell.mean_spread,
            expected_fill_rate: cell.mean_fill_rate,
            expected_as_bps: cell.mean_as,
        }
    }

    /// Check if should quote at given condition
    pub fn should_quote(&self, condition: &MarketCondition) -> bool {
        let estimate = self.edge_estimate(condition);

        estimate.edge_bps > self.config.min_edge_bps
            && estimate.confidence > self.config.min_confidence
            && estimate.edge_bps > 2.0 * estimate.uncertainty_bps
    }

    /// Get expected edge for given spread and state
    pub fn expected_edge(
        &self,
        spread_bps: f64,
        condition: &MarketCondition,
        fill_prob: f64,
    ) -> f64 {
        let estimate = self.edge_estimate(condition);

        // Edge = fill_prob × (spread/2 - AS - fees)
        fill_prob * (spread_bps / 2.0 - estimate.expected_as_bps - self.config.fees_bps)
    }

    /// Find optimal spread for given condition
    pub fn optimal_spread(&self, condition: &MarketCondition) -> f64 {
        let estimate = self.edge_estimate(condition);

        // Simple heuristic: spread = 2 × (AS + fees + target_edge)
        let target_edge = self.config.min_edge_bps.max(estimate.edge_bps);
        2.0 * (estimate.expected_as_bps + self.config.fees_bps + target_edge)
    }

    // ========================================================================
    // Surface Analysis
    // ========================================================================

    /// Get statistics across all cells
    pub fn surface_statistics(&self) -> SurfaceStatistics {
        let mut total_edge = 0.0;
        let mut total_obs = 0;
        let mut cells_with_data = 0;
        let mut best_edge = f64::NEG_INFINITY;
        let mut worst_edge = f64::INFINITY;
        let mut best_condition: Option<(usize, f64)> = None;
        let mut worst_condition: Option<(usize, f64)> = None;

        for (idx, cell) in self.grid.iter().enumerate() {
            if cell.n_obs >= self.config.min_cell_observations {
                total_edge += cell.mean_edge * cell.n_obs as f64;
                total_obs += cell.n_obs;
                cells_with_data += 1;

                if cell.mean_edge > best_edge {
                    best_edge = cell.mean_edge;
                    best_condition = Some((idx, cell.mean_edge));
                }
                if cell.mean_edge < worst_edge {
                    worst_edge = cell.mean_edge;
                    worst_condition = Some((idx, cell.mean_edge));
                }
            }
        }

        let mean_edge = if total_obs > 0 {
            total_edge / total_obs as f64
        } else {
            self.config.prior_edge_bps
        };

        SurfaceStatistics {
            mean_edge_bps: mean_edge,
            cells_with_data,
            total_observations: self.total_observations,
            best_condition,
            worst_condition,
            global_fill_rate: self.global_stats.mean_fill_rate,
            global_as_bps: self.global_stats.mean_as,
        }
    }

    /// Get edge at specific grid indices
    pub fn edge_at_indices(&self, vol: usize, regime: usize, hour: usize, flow: usize) -> f64 {
        let idx = vol * 45 + regime * 15 + hour * 5 + flow;
        if idx < GRID_SIZE {
            self.grid[idx].mean_edge
        } else {
            self.config.prior_edge_bps
        }
    }

    // ========================================================================
    // Status Methods
    // ========================================================================

    /// Total observations
    pub fn observation_count(&self) -> usize {
        self.total_observations
    }

    /// Check if surface has enough data
    pub fn is_warmed_up(&self) -> bool {
        self.total_observations >= 100
    }

    /// Reset surface to priors
    pub fn reset(&mut self) {
        *self = Self::new(self.config.clone());
    }
}

/// Statistics about the edge surface
#[derive(Debug, Clone)]
pub struct SurfaceStatistics {
    /// Mean edge across all observed cells
    pub mean_edge_bps: f64,

    /// Number of cells with sufficient data
    pub cells_with_data: usize,

    /// Total observations across all cells
    pub total_observations: usize,

    /// Best condition (index, edge)
    pub best_condition: Option<(usize, f64)>,

    /// Worst condition (index, edge)
    pub worst_condition: Option<(usize, f64)>,

    /// Global fill rate
    pub global_fill_rate: f64,

    /// Global AS
    pub global_as_bps: f64,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_surface() -> EdgeSurface {
        EdgeSurface::default_config()
    }

    #[test]
    fn test_initialization() {
        let surface = create_surface();
        assert_eq!(surface.observation_count(), 0);
        assert!(!surface.is_warmed_up());
    }

    #[test]
    fn test_observation() {
        let mut surface = create_surface();

        let obs = EdgeObservation {
            condition: MarketCondition {
                vol_bucket: 2,
                regime: 1,
                hour_bucket: 1,
                flow_bucket: 2,
            },
            spread_bps: 8.0,
            filled: true,
            realized_as_bps: 2.0,
            timestamp_ms: 1000,
        };

        surface.observe(&obs);
        assert_eq!(surface.observation_count(), 1);
    }

    #[test]
    fn test_edge_estimate_with_data() {
        let mut surface = create_surface();

        let condition = MarketCondition {
            vol_bucket: 2,
            regime: 1,
            hour_bucket: 1,
            flow_bucket: 2,
        };

        // Add observations
        for i in 0..50 {
            let obs = EdgeObservation {
                condition: condition.clone(),
                spread_bps: 10.0,
                filled: true,
                realized_as_bps: 2.0, // Edge = 10/2 - 2 - 1.5 = 1.5 bps
                timestamp_ms: i * 1000,
            };
            surface.observe(&obs);
        }

        let estimate = surface.edge_estimate(&condition);
        assert!(estimate.edge_bps > 0.0, "Edge should be positive");
        assert!(estimate.confidence > 0.0, "Should have some confidence");
    }

    #[test]
    fn test_should_quote_logic() {
        let mut surface = create_surface();

        let good_condition = MarketCondition {
            vol_bucket: 1, // Low vol
            regime: 1,
            hour_bucket: 1,
            flow_bucket: 2,
        };

        // Add profitable observations
        for i in 0..100 {
            let obs = EdgeObservation {
                condition: good_condition.clone(),
                spread_bps: 12.0,
                filled: true,
                realized_as_bps: 1.0, // Edge = 12/2 - 1 - 1.5 = 3.5 bps
                timestamp_ms: i * 1000,
            };
            surface.observe(&obs);
        }

        assert!(
            surface.should_quote(&good_condition),
            "Should quote with positive edge"
        );
    }

    #[test]
    fn test_global_fallback() {
        let mut surface = create_surface();

        // Add observations to one condition
        let observed_condition = MarketCondition {
            vol_bucket: 2,
            regime: 1,
            hour_bucket: 1,
            flow_bucket: 2,
        };

        for i in 0..50 {
            let obs = EdgeObservation {
                condition: observed_condition.clone(),
                spread_bps: 10.0,
                filled: true,
                realized_as_bps: 2.0,
                timestamp_ms: i * 1000,
            };
            surface.observe(&obs);
        }

        // Query unobserved condition
        let unobserved_condition = MarketCondition {
            vol_bucket: 4,
            regime: 2,
            hour_bucket: 0,
            flow_bucket: 0,
        };

        let estimate = surface.edge_estimate(&unobserved_condition);
        // Should use global stats or prior
        assert!(estimate.edge_bps.is_finite());
    }

    #[test]
    fn test_surface_statistics() {
        let mut surface = create_surface();

        for i in 0..200 {
            let obs = EdgeObservation {
                condition: MarketCondition {
                    vol_bucket: (i % 5) as u8,
                    regime: (i % 3) as u8,
                    hour_bucket: (i % 3) as u8,
                    flow_bucket: (i % 5) as u8,
                },
                spread_bps: 10.0,
                filled: i % 2 == 0,
                realized_as_bps: 2.0,
                timestamp_ms: i * 1000,
            };
            surface.observe(&obs);
        }

        let stats = surface.surface_statistics();
        assert!(stats.total_observations == 200);
        assert!(stats.cells_with_data > 0);
    }

    #[test]
    fn test_optimal_spread() {
        let surface = create_surface();

        let condition = MarketCondition {
            vol_bucket: 2,
            regime: 1,
            hour_bucket: 1,
            flow_bucket: 2,
        };

        let spread = surface.optimal_spread(&condition);
        assert!(spread > 0.0, "Optimal spread should be positive");
        assert!(spread < 100.0, "Optimal spread should be reasonable");
    }

    #[test]
    fn test_reset() {
        let mut surface = create_surface();

        for i in 0..50 {
            let obs = EdgeObservation {
                condition: MarketCondition {
                    vol_bucket: 2,
                    regime: 1,
                    hour_bucket: 1,
                    flow_bucket: 2,
                },
                spread_bps: 10.0,
                filled: true,
                realized_as_bps: 2.0,
                timestamp_ms: i * 1000,
            };
            surface.observe(&obs);
        }

        assert!(surface.observation_count() > 0);

        surface.reset();

        assert_eq!(surface.observation_count(), 0);
    }

    #[test]
    fn test_expected_edge() {
        let surface = create_surface();

        let condition = MarketCondition {
            vol_bucket: 2,
            regime: 1,
            hour_bucket: 1,
            flow_bucket: 2,
        };

        // With 50% fill probability
        let edge = surface.expected_edge(10.0, &condition, 0.5);
        // edge = 0.5 × (10/2 - 2 - 1.5) = 0.5 × 1.5 = 0.75
        assert!(edge.is_finite());
    }
}
