//! Outcome Tracking and PnL Attribution
//!
//! Tracks the outcomes of simulated fills and decomposes PnL into components:
//! - Spread capture (bid-ask revenue)
//! - Adverse selection (informed flow losses)
//! - Inventory cost (mark-to-market on held positions)
//! - Fee cost (exchange fees)

use super::fill_sim::SimulatedFill;
use super::prediction::{PredictionRecord, Regime};
use crate::Side;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// PnL decomposition for a single fill
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PnLDecomposition {
    /// Gross PnL
    pub gross_pnl: f64,
    /// Spread capture component
    pub spread_capture: f64,
    /// Adverse selection component (negative is loss)
    pub adverse_selection: f64,
    /// Inventory cost component
    pub inventory_cost: f64,
    /// Fee cost component
    pub fee_cost: f64,
}

impl PnLDecomposition {
    /// Sum multiple decompositions
    pub fn sum(decomps: &[PnLDecomposition]) -> Self {
        let mut total = Self::default();
        for d in decomps {
            total.gross_pnl += d.gross_pnl;
            total.spread_capture += d.spread_capture;
            total.adverse_selection += d.adverse_selection;
            total.inventory_cost += d.inventory_cost;
            total.fee_cost += d.fee_cost;
        }
        total
    }
}

impl Default for PnLDecomposition {
    fn default() -> Self {
        Self {
            gross_pnl: 0.0,
            spread_capture: 0.0,
            adverse_selection: 0.0,
            inventory_cost: 0.0,
            fee_cost: 0.0,
        }
    }
}

/// Attribution for a single quote cycle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CycleAttribution {
    /// Cycle ID
    pub cycle_id: u64,
    /// Timestamp
    pub timestamp_ns: u64,
    /// Gross PnL
    pub gross_pnl: f64,
    /// PnL decomposition
    pub decomposition: PnLDecomposition,

    // Model accuracy metrics
    /// Fill prediction error (predicted - realized fills)
    pub fill_prediction_error: f64,
    /// Adverse selection prediction error
    pub as_prediction_error: f64,
    /// Volatility prediction error
    pub vol_prediction_error: f64,

    /// Regime at this cycle
    pub regime: Regime,
}

/// Fill record for tracking price evolution
#[derive(Debug, Clone)]
struct TrackedFill {
    /// Fill details
    fill: SimulatedFill,
    /// Mid price at fill time
    mid_at_fill: f64,
    /// Mid price 100ms after fill
    mid_100ms_later: Option<f64>,
    /// Mid price 1s after fill
    mid_1s_later: Option<f64>,
    /// Mid price 10s after fill
    mid_10s_later: Option<f64>,
}

/// Outcome tracker for paper trading
pub struct OutcomeTracker {
    /// Recent fills awaiting price evolution data
    pending_fills: VecDeque<TrackedFill>,
    /// Completed cycle attributions
    completed_attributions: VecDeque<CycleAttribution>,
    /// Maximum pending fills
    max_pending: usize,
    /// Maximum completed attributions to keep (reserved for future trimming)
    #[allow(dead_code)]
    max_completed: usize,

    /// Running totals
    total_spread_capture: f64,
    total_adverse_selection: f64,
    total_inventory_cost: f64,
    total_fee_cost: f64,
    total_fills: u64,

    /// Current inventory (simulated)
    simulated_inventory: f64,
    /// Average entry price
    avg_entry_price: f64,

    /// Maker fee rate
    maker_fee_rate: f64,

    /// Last known mid price (for inventory cost computation)
    last_mid: f64,
}

impl OutcomeTracker {
    /// Create a new outcome tracker
    pub fn new(maker_fee_rate: f64) -> Self {
        Self {
            pending_fills: VecDeque::new(),
            completed_attributions: VecDeque::new(),
            max_pending: 1000,
            max_completed: 10000,
            total_spread_capture: 0.0,
            total_adverse_selection: 0.0,
            total_inventory_cost: 0.0,
            total_fee_cost: 0.0,
            total_fills: 0,
            simulated_inventory: 0.0,
            avg_entry_price: 0.0,
            maker_fee_rate,
            last_mid: 0.0,
        }
    }

    /// Record a simulated fill
    pub fn on_fill(&mut self, fill: SimulatedFill, mid_at_fill: f64) {
        // Update inventory
        let fill_direction = if fill.side == Side::Buy { 1.0 } else { -1.0 };
        let old_inventory = self.simulated_inventory;
        self.simulated_inventory += fill_direction * fill.fill_size;

        // Update average entry
        if (old_inventory * self.simulated_inventory) >= 0.0 {
            // Same direction or crossing through flat
            if self.simulated_inventory.abs() > 0.0001 {
                self.avg_entry_price = (self.avg_entry_price * old_inventory.abs()
                    + fill.fill_price * fill.fill_size)
                    / self.simulated_inventory.abs();
            }
        } else {
            // Crossed through - reset average
            self.avg_entry_price = fill.fill_price;
        }

        // Record for tracking
        self.pending_fills.push_back(TrackedFill {
            fill,
            mid_at_fill,
            mid_100ms_later: None,
            mid_1s_later: None,
            mid_10s_later: None,
        });

        // Keep bounded
        while self.pending_fills.len() > self.max_pending {
            self.pending_fills.pop_front();
        }

        self.total_fills += 1;
    }

    /// Update with new price data
    pub fn on_price_update(&mut self, mid: f64, timestamp_ns: u64) {
        self.last_mid = mid;

        // Update price tracking for pending fills
        for tracked in self.pending_fills.iter_mut() {
            let age_ns = timestamp_ns.saturating_sub(tracked.fill.timestamp_ns);
            let age_ms = age_ns / 1_000_000;

            // Update price at various horizons
            if age_ms >= 100 && tracked.mid_100ms_later.is_none() {
                tracked.mid_100ms_later = Some(mid);
            }
            if age_ms >= 1000 && tracked.mid_1s_later.is_none() {
                tracked.mid_1s_later = Some(mid);
            }
            if age_ms >= 10000 && tracked.mid_10s_later.is_none() {
                tracked.mid_10s_later = Some(mid);
                // Can now compute full attribution
            }
        }

        // Process completed fills
        self.process_completed_fills();
    }

    /// Process fills that have all price data
    fn process_completed_fills(&mut self) {
        let current_mid = self.last_mid;
        while let Some(tracked) = self.pending_fills.front() {
            if tracked.mid_10s_later.is_none() {
                break;
            }

            let tracked = self.pending_fills.pop_front().unwrap();
            let decomp = self.compute_fill_decomposition(&tracked, current_mid);

            self.total_spread_capture += decomp.spread_capture;
            self.total_adverse_selection += decomp.adverse_selection;
            self.total_inventory_cost += decomp.inventory_cost;
            self.total_fee_cost += decomp.fee_cost;
        }
    }

    /// Compute PnL decomposition for a single fill
    fn compute_fill_decomposition(&self, tracked: &TrackedFill, current_mid: f64) -> PnLDecomposition {
        let fill = &tracked.fill;
        let notional = fill.fill_price * fill.fill_size;

        // Spread capture: distance from mid at fill time
        let spread_capture = match fill.side {
            Side::Buy => (tracked.mid_at_fill - fill.fill_price) * fill.fill_size,
            Side::Sell => (fill.fill_price - tracked.mid_at_fill) * fill.fill_size,
        };

        // Adverse selection: price move against us after fill
        let mid_1s = tracked.mid_1s_later.unwrap_or(tracked.mid_at_fill);
        let price_move = mid_1s - tracked.mid_at_fill;

        let adverse_selection = match fill.side {
            Side::Buy => {
                // Bought, price dropped = adverse (price_move negative → adverse_selection negative)
                price_move.min(0.0) * fill.fill_size
            }
            Side::Sell => {
                // Sold, price rose = adverse (price_move positive → -price_move negative)
                (-price_move).min(0.0) * fill.fill_size
            }
        };

        // Fee cost
        let fee_cost = notional * self.maker_fee_rate;

        // Inventory cost: mark-to-market on current position
        let inventory_cost = self.compute_inventory_mtm(current_mid);

        let gross_pnl = spread_capture + adverse_selection + inventory_cost - fee_cost;

        PnLDecomposition {
            gross_pnl,
            spread_capture,
            adverse_selection,
            inventory_cost,
            fee_cost,
        }
    }

    /// Compute inventory cost from mark-to-market
    pub fn compute_inventory_mtm(&self, current_mid: f64) -> f64 {
        if self.simulated_inventory.abs() < 0.0001 {
            return 0.0;
        }

        let position_value = self.simulated_inventory * current_mid;
        let entry_value = self.simulated_inventory * self.avg_entry_price;

        position_value - entry_value
    }

    /// Create attribution for a cycle
    pub fn create_cycle_attribution(
        &self,
        record: &PredictionRecord,
        fills_this_cycle: &[SimulatedFill],
    ) -> CycleAttribution {
        let mut decomp = PnLDecomposition::default();

        for fill in fills_this_cycle {
            // Simple decomposition without full tracking
            let notional = fill.fill_price * fill.fill_size;
            decomp.fee_cost += notional * self.maker_fee_rate;
            decomp.spread_capture += fill.fill_price * fill.fill_size * 0.0005; // Estimate
        }

        decomp.gross_pnl = decomp.spread_capture - decomp.fee_cost;

        // Compute prediction errors
        let predicted_fills = record.predictions.expected_fill_rate_1s;
        let actual_fills = fills_this_cycle.len() as f64;
        let fill_prediction_error = predicted_fills - actual_fills;

        CycleAttribution {
            cycle_id: record.quote_cycle_id,
            timestamp_ns: record.timestamp_ns,
            gross_pnl: decomp.gross_pnl,
            decomposition: decomp,
            fill_prediction_error,
            as_prediction_error: 0.0, // Would need outcome data
            vol_prediction_error: 0.0,
            regime: record.market_state.regime,
        }
    }

    /// Get summary statistics
    pub fn get_summary(&self) -> OutcomeSummary {
        OutcomeSummary {
            total_pnl: self.total_spread_capture + self.total_adverse_selection
                - self.total_fee_cost
                + self.total_inventory_cost,
            spread_capture: self.total_spread_capture,
            adverse_selection: self.total_adverse_selection,
            inventory_cost: self.total_inventory_cost,
            fee_cost: self.total_fee_cost,
            total_fills: self.total_fills,
            current_inventory: self.simulated_inventory,
            avg_entry_price: self.avg_entry_price,
        }
    }

    /// Reset all state
    pub fn reset(&mut self) {
        self.pending_fills.clear();
        self.completed_attributions.clear();
        self.total_spread_capture = 0.0;
        self.total_adverse_selection = 0.0;
        self.total_inventory_cost = 0.0;
        self.total_fee_cost = 0.0;
        self.total_fills = 0;
        self.simulated_inventory = 0.0;
        self.avg_entry_price = 0.0;
        self.last_mid = 0.0;
    }

    /// Get completed attributions for analysis
    pub fn get_attributions(&self) -> &VecDeque<CycleAttribution> {
        &self.completed_attributions
    }

    /// Get pending fill count
    pub fn pending_fill_count(&self) -> usize {
        self.pending_fills.len()
    }
}

/// Summary of outcome tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutcomeSummary {
    /// Total PnL
    pub total_pnl: f64,
    /// Spread capture component
    pub spread_capture: f64,
    /// Adverse selection component
    pub adverse_selection: f64,
    /// Inventory cost component
    pub inventory_cost: f64,
    /// Fee cost component
    pub fee_cost: f64,
    /// Total number of fills
    pub total_fills: u64,
    /// Current inventory
    pub current_inventory: f64,
    /// Average entry price
    pub avg_entry_price: f64,
}

impl OutcomeSummary {
    /// Format as human-readable string
    pub fn format(&self) -> String {
        let mut output = String::new();

        output.push_str("=== PnL Attribution ===\n\n");

        output.push_str(&format!("Total PnL:          ${:.2}\n", self.total_pnl));
        output.push_str(&format!(
            "├── Spread Capture: ${:.2}\n",
            self.spread_capture
        ));
        output.push_str(&format!(
            "├── Adverse Select: ${:.2}\n",
            self.adverse_selection
        ));
        output.push_str(&format!(
            "├── Inventory Cost: ${:.2}\n",
            self.inventory_cost
        ));
        output.push_str(&format!("└── Fee Cost:       ${:.2}\n", self.fee_cost));

        output.push_str("\nPosition State:\n");
        output.push_str(&format!("  Fills:     {}\n", self.total_fills));
        output.push_str(&format!("  Inventory: {:.4}\n", self.current_inventory));
        output.push_str(&format!("  Avg Entry: ${:.2}\n", self.avg_entry_price));

        if self.total_fills > 0 {
            let pnl_per_fill = self.total_pnl / self.total_fills as f64;
            output.push_str(&format!("\nPnL per Fill: ${pnl_per_fill:.4}\n"));
        }

        output
    }
}

/// Attribution by regime
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RegimeAttribution {
    /// Quiet regime attribution
    pub quiet: PnLDecomposition,
    /// Active regime attribution
    pub active: PnLDecomposition,
    /// Volatile regime attribution
    pub volatile: PnLDecomposition,
    /// Cascade regime attribution
    pub cascade: PnLDecomposition,

    /// Time in each regime (seconds)
    pub time_in_regime: [f64; 4],
}

impl RegimeAttribution {
    /// Add a cycle attribution
    pub fn add_cycle(&mut self, attr: &CycleAttribution) {
        let decomp = match attr.regime {
            Regime::Quiet => &mut self.quiet,
            Regime::Active => &mut self.active,
            Regime::Volatile => &mut self.volatile,
            Regime::Cascade => &mut self.cascade,
        };

        decomp.gross_pnl += attr.decomposition.gross_pnl;
        decomp.spread_capture += attr.decomposition.spread_capture;
        decomp.adverse_selection += attr.decomposition.adverse_selection;
        decomp.inventory_cost += attr.decomposition.inventory_cost;
        decomp.fee_cost += attr.decomposition.fee_cost;
    }

    /// Format as human-readable string
    pub fn format(&self) -> String {
        let mut output = String::new();

        output.push_str("=== Regime Attribution ===\n\n");

        let regimes = [
            ("Quiet", &self.quiet),
            ("Active", &self.active),
            ("Volatile", &self.volatile),
            ("Cascade", &self.cascade),
        ];

        for (name, decomp) in regimes {
            output.push_str(&format!(
                "{:10}: PnL=${:.2} (spread=${:.2}, AS=${:.2})\n",
                name, decomp.gross_pnl, decomp.spread_capture, decomp.adverse_selection
            ));
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inventory_tracking() {
        let mut tracker = OutcomeTracker::new(0.00015);

        // Simulate a buy fill
        let fill = SimulatedFill {
            oid: 1,
            timestamp_ns: 1_000_000_000,
            fill_price: 100.0,
            fill_size: 1.0,
            side: Side::Buy,
            triggering_trade_price: 99.9,
            triggering_trade_size: 2.0,
        };

        tracker.on_fill(fill, 100.1);

        let summary = tracker.get_summary();
        assert_eq!(summary.current_inventory, 1.0);
        assert_eq!(summary.total_fills, 1);
    }

    #[test]
    fn test_pnl_decomposition() {
        let decomp1 = PnLDecomposition {
            gross_pnl: 10.0,
            spread_capture: 15.0,
            adverse_selection: -3.0,
            inventory_cost: -1.0,
            fee_cost: 1.0,
        };

        let decomp2 = PnLDecomposition {
            gross_pnl: 5.0,
            spread_capture: 8.0,
            adverse_selection: -2.0,
            inventory_cost: 0.0,
            fee_cost: 1.0,
        };

        let total = PnLDecomposition::sum(&[decomp1, decomp2]);

        assert_eq!(total.gross_pnl, 15.0);
        assert_eq!(total.spread_capture, 23.0);
        assert_eq!(total.adverse_selection, -5.0);
    }
}
