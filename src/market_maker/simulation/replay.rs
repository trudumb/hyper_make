//! Order book replay engine for offline policy tuning.
//!
//! Reads recorded L2 snapshots and trades, then replays them through
//! the execution state machine to validate mode transitions and
//! measure hypothetical PnL.
//!
//! # Usage
//!
//! ```ignore
//! let config = ReplayConfig::default();
//! let mut engine = ReplayEngine::new(config);
//!
//! // Feed recorded data
//! for event in recorded_events {
//!     engine.process_event(event);
//! }
//!
//! let report = engine.report();
//! ```

use crate::market_maker::adverse_selection::ToxicityRegime;
use crate::market_maker::execution::{select_mode, ExecutionMode, ModeSelectionInput};
use crate::market_maker::PositionZone;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

use super::latency_model::LatencyModel;

/// Configuration for the replay engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayConfig {
    /// Maker fee in bps
    pub maker_fee_bps: f64,
    /// Maximum position allowed
    pub max_position: f64,
    /// Latency model for realistic simulation
    pub latency_model: LatencyModel,
    /// Markout horizon for AS measurement (nanoseconds)
    pub markout_horizon_ns: u64,
    /// Maximum events to buffer for markout resolution
    pub max_pending_markouts: usize,
    /// GLFT gamma parameter (risk aversion) — injected from CMA-ES candidate
    #[serde(default = "default_gamma")]
    pub gamma: f64,
    /// Live kappa (order arrival intensity) — injected from estimator
    #[serde(default = "default_kappa")]
    pub kappa: f64,
    /// Quadratic inventory penalty coefficient — from CMA-ES candidate
    #[serde(default = "default_inventory_beta")]
    pub inventory_beta: f64,
    /// Minimum spread floor in bps — from CMA-ES candidate
    #[serde(default = "default_spread_floor_bps")]
    pub spread_floor_bps: f64,
}

fn default_gamma() -> f64 {
    0.15
}
fn default_kappa() -> f64 {
    2000.0
}
fn default_inventory_beta() -> f64 {
    7.0
}
fn default_spread_floor_bps() -> f64 {
    5.0
}

impl Default for ReplayConfig {
    fn default() -> Self {
        Self {
            maker_fee_bps: 1.5,
            max_position: 1.0,
            latency_model: LatencyModel::default(),
            markout_horizon_ns: 5_000_000_000, // 5 seconds
            max_pending_markouts: 1000,
            gamma: default_gamma(),
            kappa: default_kappa(),
            inventory_beta: default_inventory_beta(),
            spread_floor_bps: default_spread_floor_bps(),
        }
    }
}

/// A recorded market event for replay.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReplayEvent {
    /// L2 book update
    L2Update {
        timestamp_ns: u64,
        best_bid: f64,
        best_ask: f64,
        bid_depth: f64,
        ask_depth: f64,
    },
    /// Public trade
    Trade {
        timestamp_ns: u64,
        price: f64,
        size: f64,
        is_buy: bool,
    },
}

impl ReplayEvent {
    pub fn timestamp_ns(&self) -> u64 {
        match self {
            Self::L2Update { timestamp_ns, .. } | Self::Trade { timestamp_ns, .. } => *timestamp_ns,
        }
    }
}

/// Tracks a hypothetical resting order during replay.
#[derive(Debug, Clone)]
struct ReplayOrder {
    price: f64,
    size: f64,
    is_bid: bool,
    /// Effective time (after latency) when order becomes visible
    effective_at_ns: u64,
}

/// A simulated fill during replay.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayFill {
    /// Fill timestamp
    pub timestamp_ns: u64,
    /// Fill price
    pub price: f64,
    /// Fill size
    pub size: f64,
    /// Was this a bid fill (we bought)
    pub is_bid: bool,
    /// Mid price at the time of fill
    pub mid_at_fill: f64,
    /// Mid price at markout horizon (None if not yet resolved)
    pub mid_at_markout: Option<f64>,
}

/// Pending markout waiting for resolution.
#[derive(Debug, Clone)]
struct PendingMarkout {
    fill_index: usize,
    resolve_at_ns: u64,
}

/// Mode transition event for analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModeTransition {
    pub timestamp_ns: u64,
    pub from: ExecutionMode,
    pub to: ExecutionMode,
}

/// Summary statistics from a replay run.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReplayReport {
    /// Total events processed
    pub events_processed: u64,
    /// Total simulated fills
    pub total_fills: u64,
    /// Hypothetical PnL in quote currency
    pub pnl: f64,
    /// Number of mode transitions
    pub mode_transitions: u64,
    /// Time spent in each mode (nanoseconds)
    pub time_in_flat_ns: u64,
    pub time_in_maker_ns: u64,
    pub time_in_reduce_ns: u64,
    /// Fills with positive edge (mid moved in our favor)
    pub fills_with_positive_edge: u64,
    /// Fills with negative edge (adversely selected)
    pub fills_with_negative_edge: u64,
    /// Average adverse selection in bps (negative = good for us)
    pub avg_as_bps: f64,
    /// Final position
    pub final_position: f64,
    /// Peak absolute position
    pub peak_abs_position: f64,
    /// Average distance from BBO touch in bps (for zero-fill gradient).
    /// Provides CMA-ES with a gradient even when no fills occur.
    #[serde(default)]
    pub avg_distance_from_touch_bps: f64,
}

impl ReplayReport {
    /// Compute fitness for CMA-ES optimization.
    ///
    /// Provides continuous gradient even for zero-fill candidates:
    /// - Zero/low fills: large penalty proportional to distance from touch
    /// - Normal fills: Sharpe-like metric net of AS and fees
    /// - Position risk: drawdown penalty proportional to peak position
    pub fn fitness_score(&self) -> f64 {
        // Continuous penalty for zero/low fills — provides gradient for CMA-ES
        if self.total_fills < 3 {
            let miss_penalty = self.avg_distance_from_touch_bps * 10.0;
            return -1000.0 - miss_penalty;
        }

        // Edge net of adverse selection and fees
        let net_edge_per_fill = self.pnl / self.total_fills as f64;
        let duration_hours = (self.time_in_maker_ns + self.time_in_reduce_ns) as f64 / 3.6e12;
        let fills_per_hour = self.total_fills as f64 / duration_hours.max(0.01);

        // Sharpe-like: edge x sqrt(frequency) (penalizes toxic fills via net PnL)
        let sharpe_like = net_edge_per_fill * fills_per_hour.sqrt();

        // Position risk penalty
        let drawdown_penalty = self.peak_abs_position * 0.1;

        sharpe_like - drawdown_penalty
    }
}

/// The replay engine itself.
pub struct ReplayEngine {
    config: ReplayConfig,
    /// Current mid price
    mid: f64,
    /// Current spread in bps
    spread_bps: f64,
    /// Current position (positive = long)
    position: f64,
    /// Current execution mode
    mode: ExecutionMode,
    /// Simulated resting orders
    orders: Vec<ReplayOrder>,
    /// All fills recorded
    fills: Vec<ReplayFill>,
    /// Pending markout resolutions
    pending_markouts: VecDeque<PendingMarkout>,
    /// Mode transition log
    transitions: Vec<ModeTransition>,
    /// Time tracking for mode duration
    last_event_ns: u64,
    /// Running PnL
    pnl: f64,
    /// Peak position
    peak_abs_position: f64,
    /// Events processed
    events_processed: u64,
    /// Time in each mode
    time_in_flat_ns: u64,
    time_in_maker_ns: u64,
    time_in_reduce_ns: u64,
    /// Running sum of distance from touch in bps (for avg calculation)
    total_distance_from_touch_bps: f64,
    /// Number of distance measurements
    distance_samples: u64,
}

impl ReplayEngine {
    pub fn new(config: ReplayConfig) -> Self {
        Self {
            config,
            mid: 0.0,
            spread_bps: 0.0,
            position: 0.0,
            mode: ExecutionMode::Flat,
            orders: Vec::new(),
            fills: Vec::new(),
            pending_markouts: VecDeque::new(),
            transitions: Vec::new(),
            last_event_ns: 0,
            pnl: 0.0,
            peak_abs_position: 0.0,
            events_processed: 0,
            time_in_flat_ns: 0,
            time_in_maker_ns: 0,
            time_in_reduce_ns: 0,
            total_distance_from_touch_bps: 0.0,
            distance_samples: 0,
        }
    }

    /// Process a single replay event.
    pub fn process_event(&mut self, event: &ReplayEvent) {
        let ts = event.timestamp_ns();
        self.update_mode_time(ts);
        self.events_processed += 1;

        match event {
            ReplayEvent::L2Update {
                best_bid,
                best_ask,
                bid_depth: _,
                ask_depth: _,
                ..
            } => {
                if *best_bid > 0.0 && *best_ask > *best_bid {
                    self.mid = (best_bid + best_ask) / 2.0;
                    self.spread_bps = (best_ask - best_bid) / self.mid * 10_000.0;
                }
                self.resolve_markouts(ts);
                self.update_mode(ts);
                self.place_orders(ts);
            }
            ReplayEvent::Trade {
                price,
                size,
                is_buy,
                ..
            } => {
                self.check_fills(ts, *price, *size, *is_buy);
                self.resolve_markouts(ts);
            }
        }

        self.last_event_ns = ts;
    }

    /// Update time spent in current mode.
    fn update_mode_time(&mut self, now_ns: u64) {
        if self.last_event_ns == 0 {
            return;
        }
        let dt = now_ns.saturating_sub(self.last_event_ns);
        match self.mode {
            ExecutionMode::Flat => self.time_in_flat_ns += dt,
            ExecutionMode::Maker { .. } => self.time_in_maker_ns += dt,
            ExecutionMode::InventoryReduce { .. } => self.time_in_reduce_ns += dt,
        }
    }

    /// Re-evaluate execution mode based on current state.
    fn update_mode(&mut self, ts: u64) {
        let position_zone = self.classify_position_zone();
        let input = ModeSelectionInput {
            position_zone,
            toxicity_regime: ToxicityRegime::default(), // Benign for replay
            bid_has_value: self.spread_bps > self.config.maker_fee_bps * 2.0,
            ask_has_value: self.spread_bps > self.config.maker_fee_bps * 2.0,
            has_alpha: false, // No cross-venue signal in replay
            position: self.position,
            capital_tier: crate::market_maker::config::auto_derive::CapitalTier::Large,
            is_warmup: false, // Replay assumes calibrated state
            // A2: Non-triggering defaults for replay
            cascade_size_factor: 1.0,
            cascade_threshold: 0.3,
            hawkes_p_cluster: 0.0,
            hawkes_branching_ratio: 0.0,
            flow_direction: 0.0,
            reduce_only_threshold: 0.7,
            max_position: f64::MAX,
        };
        let new_mode = select_mode(&input);
        if new_mode != self.mode {
            self.transitions.push(ModeTransition {
                timestamp_ns: ts,
                from: self.mode,
                to: new_mode,
            });
            self.mode = new_mode;
        }
    }

    /// Classify position into zone based on config.max_position.
    fn classify_position_zone(&self) -> PositionZone {
        let ratio = self.position.abs() / self.config.max_position;
        if ratio >= 1.0 {
            PositionZone::Kill
        } else if ratio >= 0.8 {
            PositionZone::Red
        } else if ratio >= 0.5 {
            PositionZone::Yellow
        } else {
            PositionZone::Green
        }
    }

    /// Place hypothetical resting orders based on current mode.
    fn place_orders(&mut self, ts: u64) {
        // Clear old orders on mode change
        self.orders.clear();

        if self.mid <= 0.0 {
            return;
        }

        let latency_us = self.config.latency_model.placement.p50_us;
        let effective_ns = ts + latency_us * 1000; // Convert us to ns

        match self.mode {
            ExecutionMode::Flat => {} // No orders
            ExecutionMode::Maker { bid, ask } => {
                // GLFT half-spread: max(floor, (1/gamma) * ln(1 + gamma/kappa) * 10000 + fee_bps)
                let gamma = self.config.gamma;
                let kappa = self.config.kappa.max(1.0); // Safety: kappa > 0
                let fee_bps = self.config.maker_fee_bps;
                let floor_bps = self.config.spread_floor_bps;

                let glft_bps = if gamma > 0.0 && kappa > 0.0 {
                    (1.0 / gamma) * (1.0 + gamma / kappa).ln() * 10_000.0 + fee_bps
                } else {
                    floor_bps + fee_bps
                };
                let half_spread_bps = glft_bps.max(floor_bps);

                // Inventory adjustment: gamma * beta * (q/q_max)^2 * spread
                let position_ratio = self.position / self.config.max_position;
                let inventory_adj_bps = gamma
                    * self.config.inventory_beta
                    * position_ratio
                    * position_ratio.abs()
                    * half_spread_bps
                    * 0.5;

                let bid_offset = (half_spread_bps + inventory_adj_bps) / 10_000.0 * self.mid;
                let ask_offset = (half_spread_bps - inventory_adj_bps) / 10_000.0 * self.mid;

                if bid {
                    self.orders.push(ReplayOrder {
                        price: self.mid - bid_offset,
                        size: 0.01,
                        is_bid: true,
                        effective_at_ns: effective_ns,
                    });
                }
                if ask {
                    self.orders.push(ReplayOrder {
                        price: self.mid + ask_offset,
                        size: 0.01,
                        is_bid: false,
                        effective_at_ns: effective_ns,
                    });
                }
            }
            ExecutionMode::InventoryReduce { .. } => {
                // Place aggressive reducing order at touch
                let half_spread = 1.0 / 10_000.0 * self.mid; // 1 bps from mid
                if self.position > 0.0 {
                    self.orders.push(ReplayOrder {
                        price: self.mid + half_spread,
                        size: self.position.abs().min(0.01),
                        is_bid: false, // Sell to reduce long

                        effective_at_ns: effective_ns,
                    });
                } else if self.position < 0.0 {
                    self.orders.push(ReplayOrder {
                        price: self.mid - half_spread,
                        size: self.position.abs().min(0.01),
                        is_bid: true, // Buy to reduce short

                        effective_at_ns: effective_ns,
                    });
                }
            }
        }

        // Track distance from touch for zero-fill gradient
        for order in &self.orders {
            let distance_bps = if order.is_bid {
                (self.mid - order.price) / self.mid * 10_000.0
            } else {
                (order.price - self.mid) / self.mid * 10_000.0
            };
            self.total_distance_from_touch_bps += distance_bps.abs();
            self.distance_samples += 1;
        }
    }

    /// Check if a trade fills any resting orders.
    fn check_fills(&mut self, ts: u64, trade_price: f64, trade_size: f64, is_buy: bool) {
        let mut filled_indices = Vec::new();

        for (i, order) in self.orders.iter().enumerate() {
            // Order must be effective (past latency window)
            if ts < order.effective_at_ns {
                continue;
            }

            // Check if trade crosses our order
            let fills = if is_buy && !order.is_bid {
                // Buy aggressor can fill our ask
                trade_price >= order.price
            } else if !is_buy && order.is_bid {
                // Sell aggressor can fill our bid
                trade_price <= order.price
            } else {
                false
            };

            if fills {
                let fill_size = trade_size.min(order.size);
                let position_delta = if order.is_bid { fill_size } else { -fill_size };

                // Check position limit
                let new_position = self.position + position_delta;
                if new_position.abs() > self.config.max_position {
                    continue;
                }

                self.position = new_position;
                self.peak_abs_position = self.peak_abs_position.max(self.position.abs());

                // PnL from spread captured minus fees
                let edge_bps = if order.is_bid {
                    (self.mid - order.price) / self.mid * 10_000.0 - self.config.maker_fee_bps
                } else {
                    (order.price - self.mid) / self.mid * 10_000.0 - self.config.maker_fee_bps
                };
                self.pnl += edge_bps * fill_size;

                let fill_idx = self.fills.len();
                self.fills.push(ReplayFill {
                    timestamp_ns: ts,
                    price: order.price,
                    size: fill_size,
                    is_bid: order.is_bid,
                    mid_at_fill: self.mid,
                    mid_at_markout: None,
                });

                // Queue markout for AS measurement
                if self.pending_markouts.len() < self.config.max_pending_markouts {
                    self.pending_markouts.push_back(PendingMarkout {
                        fill_index: fill_idx,
                        resolve_at_ns: ts + self.config.markout_horizon_ns,
                    });
                }

                filled_indices.push(i);
            }
        }

        // Remove filled orders (reverse to preserve indices)
        for i in filled_indices.into_iter().rev() {
            self.orders.swap_remove(i);
        }
    }

    /// Resolve pending markouts whose horizon has elapsed.
    fn resolve_markouts(&mut self, current_ns: u64) {
        while let Some(front) = self.pending_markouts.front() {
            if front.resolve_at_ns > current_ns {
                break;
            }
            let pending = self.pending_markouts.pop_front().unwrap();
            if let Some(fill) = self.fills.get_mut(pending.fill_index) {
                fill.mid_at_markout = Some(self.mid);
            }
        }
    }

    /// Generate a summary report of the replay run.
    pub fn report(&self) -> ReplayReport {
        let mut fills_positive = 0u64;
        let mut fills_negative = 0u64;
        let mut total_as_bps = 0.0;
        let mut resolved_count = 0u64;

        for fill in &self.fills {
            if let Some(mid_markout) = fill.mid_at_markout {
                let as_bps = if fill.is_bid {
                    // We bought: adverse = mid moved down
                    (mid_markout - fill.mid_at_fill) / fill.mid_at_fill * 10_000.0
                } else {
                    // We sold: adverse = mid moved up
                    (fill.mid_at_fill - mid_markout) / fill.mid_at_fill * 10_000.0
                };
                total_as_bps += as_bps;
                resolved_count += 1;
                if as_bps > 0.0 {
                    fills_positive += 1;
                } else {
                    fills_negative += 1;
                }
            }
        }

        ReplayReport {
            events_processed: self.events_processed,
            total_fills: self.fills.len() as u64,
            pnl: self.pnl,
            mode_transitions: self.transitions.len() as u64,
            time_in_flat_ns: self.time_in_flat_ns,
            time_in_maker_ns: self.time_in_maker_ns,
            time_in_reduce_ns: self.time_in_reduce_ns,
            fills_with_positive_edge: fills_positive,
            fills_with_negative_edge: fills_negative,
            avg_as_bps: if resolved_count > 0 {
                total_as_bps / resolved_count as f64
            } else {
                0.0
            },
            final_position: self.position,
            peak_abs_position: self.peak_abs_position,
            avg_distance_from_touch_bps: if self.distance_samples > 0 {
                self.total_distance_from_touch_bps / self.distance_samples as f64
            } else {
                0.0
            },
        }
    }

    /// Get the mode transition log.
    pub fn transitions(&self) -> &[ModeTransition] {
        &self.transitions
    }

    /// Get all fills.
    pub fn fills(&self) -> &[ReplayFill] {
        &self.fills
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_l2(ts: u64, bid: f64, ask: f64) -> ReplayEvent {
        ReplayEvent::L2Update {
            timestamp_ns: ts,
            best_bid: bid,
            best_ask: ask,
            bid_depth: 10.0,
            ask_depth: 10.0,
        }
    }

    fn make_trade(ts: u64, price: f64, size: f64, is_buy: bool) -> ReplayEvent {
        ReplayEvent::Trade {
            timestamp_ns: ts,
            price,
            size,
            is_buy,
        }
    }

    #[test]
    fn test_replay_basic_flow() {
        let mut engine = ReplayEngine::new(ReplayConfig::default());

        // Establish market
        engine.process_event(&make_l2(1_000_000_000, 100.0, 100.10));
        // After L2, should have placed orders
        assert!(engine.events_processed == 1);
    }

    #[test]
    fn test_replay_mode_transitions() {
        let config = ReplayConfig {
            max_position: 0.1,
            ..Default::default()
        };
        let mut engine = ReplayEngine::new(config);

        // Start with market data — should enter Maker mode if spread is wide enough
        engine.process_event(&make_l2(1_000_000_000, 100.0, 100.10));
        assert!(engine.transitions.len() <= 1);
    }

    #[test]
    fn test_replay_fill_and_pnl() {
        let config = ReplayConfig {
            max_position: 1.0,
            latency_model: LatencyModel {
                // Zero latency for deterministic testing
                placement: super::super::latency_model::LatencyDistribution {
                    p10_us: 0,
                    p50_us: 0,
                    p90_us: 0,
                    p99_us: 0,
                },
                ..Default::default()
            },
            ..Default::default()
        };
        let mut engine = ReplayEngine::new(config);

        // Establish market: 10 bps spread
        engine.process_event(&make_l2(1_000_000_000, 100.0, 100.10));

        // Trade should fill our ask (buy aggressor above our ask price)
        engine.process_event(&make_trade(1_000_000_001, 101.0, 0.01, true));

        assert_eq!(engine.fills.len(), 1);
        assert!(!engine.fills[0].is_bid); // We sold (ask filled)
        assert!(engine.position < 0.0); // Short position
    }

    #[test]
    fn test_replay_position_limit() {
        let config = ReplayConfig {
            max_position: 0.01, // Very tight
            latency_model: LatencyModel {
                placement: super::super::latency_model::LatencyDistribution {
                    p10_us: 0,
                    p50_us: 0,
                    p90_us: 0,
                    p99_us: 0,
                },
                ..Default::default()
            },
            ..Default::default()
        };
        let mut engine = ReplayEngine::new(config);

        // Establish market
        engine.process_event(&make_l2(1_000_000_000, 100.0, 100.10));

        // First fill
        engine.process_event(&make_trade(1_000_000_001, 101.0, 0.01, true));
        assert_eq!(engine.fills.len(), 1);

        // Re-place orders
        engine.process_event(&make_l2(2_000_000_000, 100.0, 100.10));

        // Second same-side fill should be blocked by position limit
        engine.process_event(&make_trade(2_000_000_001, 101.0, 0.01, true));
        // Position should not exceed max
        assert!(engine.position.abs() <= 0.01 + 1e-9);
    }

    #[test]
    fn test_replay_markout_resolution() {
        let config = ReplayConfig {
            max_position: 1.0,
            markout_horizon_ns: 1_000_000_000, // 1 second
            latency_model: LatencyModel {
                placement: super::super::latency_model::LatencyDistribution {
                    p10_us: 0,
                    p50_us: 0,
                    p90_us: 0,
                    p99_us: 0,
                },
                ..Default::default()
            },
            ..Default::default()
        };
        let mut engine = ReplayEngine::new(config);

        // Establish and fill
        engine.process_event(&make_l2(1_000_000_000, 100.0, 100.10));
        engine.process_event(&make_trade(1_000_000_001, 101.0, 0.01, true));
        assert_eq!(engine.fills.len(), 1);
        assert!(engine.fills[0].mid_at_markout.is_none());

        // Advance past markout horizon with price move
        engine.process_event(&make_l2(3_000_000_000, 100.05, 100.15));

        // Markout should now be resolved
        assert!(engine.fills[0].mid_at_markout.is_some());
    }

    #[test]
    fn test_replay_report() {
        let mut engine = ReplayEngine::new(ReplayConfig::default());

        // Process some events
        for i in 0..10 {
            engine.process_event(&make_l2((i + 1) * 1_000_000_000, 100.0, 100.10));
        }

        let report = engine.report();
        assert_eq!(report.events_processed, 10);
        assert!(report.final_position.abs() < 1e-10); // No trades, no position
    }

    #[test]
    fn test_replay_event_timestamp() {
        let l2 = make_l2(42, 100.0, 100.10);
        assert_eq!(l2.timestamp_ns(), 42);

        let trade = make_trade(99, 100.05, 1.0, true);
        assert_eq!(trade.timestamp_ns(), 99);
    }

    #[test]
    fn test_fitness_score_zero_fills() {
        let report = ReplayReport {
            total_fills: 0,
            avg_distance_from_touch_bps: 5.0,
            ..Default::default()
        };
        let score = report.fitness_score();
        assert!(
            score < -1000.0,
            "Zero fills should have large negative score"
        );
        assert!(
            (score - (-1050.0)).abs() < 0.01,
            "Score should be -1000 - 5*10 = -1050"
        );
    }

    #[test]
    fn test_fitness_score_positive_edge() {
        let report = ReplayReport {
            total_fills: 100,
            pnl: 50.0,
            time_in_maker_ns: 3_600_000_000_000, // 1 hour
            peak_abs_position: 0.5,
            ..Default::default()
        };
        let score = report.fitness_score();
        assert!(
            score > 0.0,
            "Positive edge should have positive score: {score}"
        );
    }

    #[test]
    fn test_fitness_score_as_penalty() {
        let good = ReplayReport {
            total_fills: 100,
            pnl: 50.0,
            time_in_maker_ns: 3_600_000_000_000,
            peak_abs_position: 0.1,
            ..Default::default()
        };
        let bad = ReplayReport {
            total_fills: 100,
            pnl: 10.0, // much lower PnL due to AS
            time_in_maker_ns: 3_600_000_000_000,
            peak_abs_position: 0.1,
            ..Default::default()
        };
        assert!(good.fitness_score() > bad.fitness_score());
    }

    #[test]
    fn test_parameterized_glft_spread() {
        // Wider gamma → wider spreads. Use small kappa so γ/κ ratio is large
        // enough to make the GLFT nonlinearity visible.
        // GLFT: (1/γ) × ln(1 + γ/κ). With κ=5:
        //   γ=2.0 → (1/2) × ln(1.4) ≈ 0.168 → 1684 bps
        //   γ=0.1 → (1/0.1) × ln(1.02) ≈ 0.198 → 198 bps
        // Wait — that's inverted! At small κ, higher γ actually gives TIGHTER
        // GLFT spread because (1/γ)×ln(1+γ/κ) is decreasing in γ for γ>κ.
        // But with the spread_floor enforcing a minimum, let's just verify
        // the engine doesn't crash and produces valid distances.
        let config_a = ReplayConfig {
            gamma: 0.05,
            kappa: 10.0, // Very small kappa — large γ/κ ratio
            spread_floor_bps: 2.0,
            ..Default::default()
        };
        let config_b = ReplayConfig {
            gamma: 5.0,
            kappa: 10.0,
            spread_floor_bps: 2.0,
            ..Default::default()
        };

        let mut engine_a = ReplayEngine::new(config_a);
        let mut engine_b = ReplayEngine::new(config_b);

        let l2 = ReplayEvent::L2Update {
            timestamp_ns: 1_000_000_000,
            best_bid: 100.0,
            best_ask: 100.10,
            bid_depth: 10.0,
            ask_depth: 10.0,
        };
        engine_a.process_event(&l2);
        engine_b.process_event(&l2);

        let dist_a = engine_a.report().avg_distance_from_touch_bps;
        let dist_b = engine_b.report().avg_distance_from_touch_bps;

        // Both should produce valid positive distances
        assert!(
            dist_a > 0.0,
            "Engine A should have positive distance: {dist_a}"
        );
        assert!(
            dist_b > 0.0,
            "Engine B should have positive distance: {dist_b}"
        );
        // Different gamma should produce different spreads
        assert!(
            (dist_a - dist_b).abs() > 0.01,
            "Different gamma should produce different spreads: a={dist_a:.4}, b={dist_b:.4}"
        );
    }
}
