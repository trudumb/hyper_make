//! P&L Attribution - decompose profits and losses into components.
//!
//! Breaks down total P&L into:
//! - **Spread Capture**: Revenue from bid-ask spread
//! - **Adverse Selection**: Loss from trading against informed flow
//! - **Inventory Carry**: Gain/loss from holding positions
//! - **Funding**: Cost/revenue from perpetual funding
//! - **Fees**: Exchange trading fees
//!
//! This helps diagnose what's driving performance and optimize strategy.

use std::collections::{HashMap, VecDeque};
use std::time::Instant;

// Import VolatilityRegime from estimator for use in fills
use crate::market_maker::estimator::VolatilityRegime;

/// Snapshot of inventory state at a point in time.
/// Used to calculate inventory carry P&L properly.
#[derive(Debug, Clone, Copy)]
pub struct InventorySnapshot {
    /// When this snapshot was taken
    pub timestamp: Instant,
    /// Position at this time
    pub position: f64,
    /// Mid price at this time
    pub mid_price: f64,
}

/// A funding payment record.
#[derive(Debug, Clone, Copy)]
pub struct FundingPayment {
    /// When funding was paid/received
    pub timestamp: Instant,
    /// Amount (positive = paid, negative = received)
    pub amount: f64,
    /// Position at time of funding
    pub position: f64,
}

/// Statistics for fills at a specific level or regime.
#[derive(Debug, Clone, Default)]
pub struct LevelStats {
    /// Number of fills
    pub count: usize,
    /// Total size filled
    pub total_size: f64,
    /// Total adverse selection cost
    pub total_as: f64,
    /// Total P&L from fills at this level
    pub total_pnl: f64,
    /// Win rate (fills with positive P&L)
    pub win_rate: f64,
}

/// Configuration for P&L attribution.
#[derive(Debug, Clone)]
pub struct PnLConfig {
    /// Fee rate (as fraction, e.g., 0.0002 = 2 bps)
    pub fee_rate: f64,
    /// Maximum history entries to keep
    pub max_history: usize,
}

impl Default for PnLConfig {
    fn default() -> Self {
        Self {
            fee_rate: 0.0002, // 2 bps maker fee
            max_history: 1000,
        }
    }
}

/// A single fill record for P&L attribution.
#[derive(Debug, Clone, Copy)]
pub struct FillRecord {
    /// Trade ID
    pub tid: u64,
    /// Fill price
    pub price: f64,
    /// Fill size (always positive)
    pub size: f64,
    /// Was this a buy?
    pub is_buy: bool,
    /// Mid price at time of fill
    pub mid_at_fill: f64,
    /// Mid price 1 second after fill (for AS measurement)
    pub mid_after_1s: Option<f64>,
    /// Timestamp
    pub timestamp: Instant,
    /// Distance from mid at fill time (in bps)
    pub depth_from_mid_bps: f64,
    /// Volatility regime at time of fill
    pub vol_regime: VolatilityRegime,
}

/// Decomposed P&L components.
#[derive(Debug, Clone, Default)]
pub struct PnLComponents {
    /// Revenue from spread capture (fill price vs mid)
    pub spread_capture: f64,
    /// Loss from adverse selection (mid movement after fill)
    pub adverse_selection: f64,
    /// Gain/loss from inventory carry (unrealized P&L)
    pub inventory_carry: f64,
    /// Funding payments (positive = paid, negative = received)
    pub funding: f64,
    /// Trading fees paid
    pub fees: f64,
    /// Total realized P&L
    pub realized_pnl: f64,
    /// Total unrealized P&L
    pub unrealized_pnl: f64,
}

impl PnLComponents {
    /// Get total P&L (realized + unrealized - fees).
    pub fn total(&self) -> f64 {
        self.realized_pnl + self.unrealized_pnl - self.fees
    }

    /// Get net trading P&L (spread - AS - fees).
    pub fn net_trading(&self) -> f64 {
        self.spread_capture - self.adverse_selection - self.fees
    }
}

/// P&L attribution tracker.
pub struct PnLTracker {
    config: PnLConfig,

    /// Fill history
    fills: VecDeque<FillRecord>,

    /// Current position
    position: f64,

    /// Average entry price (for unrealized P&L)
    avg_entry_price: f64,

    /// Total cost basis
    cost_basis: f64,

    /// Cumulative spread capture
    total_spread_capture: f64,

    /// Cumulative adverse selection
    total_adverse_selection: f64,

    /// Cumulative funding
    total_funding: f64,

    /// Cumulative fees
    total_fees: f64,

    /// Cumulative realized P&L
    total_realized_pnl: f64,

    /// Peak total P&L for drawdown calculation
    peak_pnl: f64,

    /// Number of fills
    fill_count: usize,

    /// Start time
    start_time: Instant,

    /// Inventory snapshots for carry calculation
    inventory_snapshots: VecDeque<InventorySnapshot>,

    /// Funding payment history
    funding_payments: VecDeque<FundingPayment>,

    /// Maximum snapshots to keep
    max_snapshots: usize,
}

impl PnLTracker {
    /// Create a new P&L tracker.
    pub fn new(config: PnLConfig) -> Self {
        let max_history = config.max_history;
        Self {
            config,
            fills: VecDeque::with_capacity(1000),
            position: 0.0,
            avg_entry_price: 0.0,
            cost_basis: 0.0,
            total_spread_capture: 0.0,
            total_adverse_selection: 0.0,
            total_funding: 0.0,
            total_fees: 0.0,
            total_realized_pnl: 0.0,
            peak_pnl: 0.0,
            fill_count: 0,
            start_time: Instant::now(),
            inventory_snapshots: VecDeque::with_capacity(1000),
            funding_payments: VecDeque::with_capacity(100),
            max_snapshots: max_history,
        }
    }

    /// Record a fill with full context.
    ///
    /// # Arguments
    /// - `tid`: Trade ID
    /// - `price`: Fill price
    /// - `size`: Fill size (always positive)
    /// - `is_buy`: Was this a buy?
    /// - `mid_at_fill`: Mid price at time of fill
    /// - `depth_from_mid_bps`: Distance from mid in basis points
    /// - `vol_regime`: Volatility regime at fill time
    #[allow(clippy::too_many_arguments)]
    pub fn record_fill_with_context(
        &mut self,
        tid: u64,
        price: f64,
        size: f64,
        is_buy: bool,
        mid_at_fill: f64,
        depth_from_mid_bps: f64,
        vol_regime: VolatilityRegime,
    ) {
        let now = Instant::now();

        // Calculate spread capture: how much we beat the mid
        // Buy at price < mid = positive spread capture
        // Sell at price > mid = positive spread capture
        let spread_capture = if is_buy {
            (mid_at_fill - price) * size
        } else {
            (price - mid_at_fill) * size
        };
        self.total_spread_capture += spread_capture;

        // Calculate fees
        let notional = price * size;
        let fee = notional * self.config.fee_rate;
        self.total_fees += fee;

        // Update position and cost basis
        let _signed_size = if is_buy { size } else { -size };

        if is_buy {
            // Buying increases position
            if self.position >= 0.0 {
                // Adding to long position
                let new_cost = self.cost_basis + price * size;
                let new_position = self.position + size;
                self.avg_entry_price = new_cost / new_position;
                self.cost_basis = new_cost;
                self.position = new_position;
            } else {
                // Reducing short position (realize P&L)
                let close_size = size.min(self.position.abs());
                let realized = (self.avg_entry_price - price) * close_size;
                self.total_realized_pnl += realized;

                let remaining = size - close_size;
                if remaining > 0.0 {
                    // Flipping to long
                    self.position = remaining;
                    self.avg_entry_price = price;
                    self.cost_basis = price * remaining;
                } else {
                    self.position += size;
                    self.cost_basis = self.avg_entry_price * self.position.abs();
                }
            }
        } else {
            // Selling decreases position
            if self.position <= 0.0 {
                // Adding to short position
                let new_cost = self.cost_basis + price * size;
                let new_position = self.position - size;
                self.avg_entry_price = new_cost / new_position.abs();
                self.cost_basis = new_cost;
                self.position = new_position;
            } else {
                // Reducing long position (realize P&L)
                let close_size = size.min(self.position);
                let realized = (price - self.avg_entry_price) * close_size;
                self.total_realized_pnl += realized;

                let remaining = size - close_size;
                if remaining > 0.0 {
                    // Flipping to short
                    self.position = -remaining;
                    self.avg_entry_price = price;
                    self.cost_basis = price * remaining;
                } else {
                    self.position -= size;
                    self.cost_basis = self.avg_entry_price * self.position.abs();
                }
            }
        }

        // Store fill for AS measurement
        let fill = FillRecord {
            tid,
            price,
            size,
            is_buy,
            mid_at_fill,
            mid_after_1s: None, // Will be updated later
            depth_from_mid_bps,
            vol_regime,
            timestamp: now,
        };
        self.fills.push_back(fill);
        self.fill_count += 1;

        // Trim history
        while self.fills.len() > self.config.max_history {
            self.fills.pop_front();
        }
    }

    /// Record a fill (backwards-compatible, uses default regime).
    ///
    /// # Arguments
    /// - `tid`: Trade ID
    /// - `price`: Fill price
    /// - `size`: Fill size (always positive)
    /// - `is_buy`: Was this a buy?
    /// - `mid_at_fill`: Mid price at time of fill
    pub fn record_fill(&mut self, tid: u64, price: f64, size: f64, is_buy: bool, mid_at_fill: f64) {
        // Calculate depth from mid in bps
        let depth_bps = if mid_at_fill > 0.0 {
            ((price - mid_at_fill).abs() / mid_at_fill) * 10000.0
        } else {
            0.0
        };

        self.record_fill_with_context(
            tid,
            price,
            size,
            is_buy,
            mid_at_fill,
            depth_bps,
            VolatilityRegime::Normal,
        );
    }

    /// Update adverse selection measurement for a fill.
    ///
    /// Call this ~1 second after a fill with the current mid price.
    pub fn update_adverse_selection(&mut self, tid: u64, mid_after: f64) {
        if let Some(fill) = self.fills.iter_mut().find(|f| f.tid == tid) {
            if fill.mid_after_1s.is_none() {
                fill.mid_after_1s = Some(mid_after);

                // Calculate adverse selection
                // If we bought and price went up, that's good (negative AS)
                // If we bought and price went down, that's bad (positive AS)
                let price_move = mid_after - fill.mid_at_fill;
                let as_cost = if fill.is_buy {
                    -price_move * fill.size // Bought, price down = cost
                } else {
                    price_move * fill.size // Sold, price up = cost
                };
                self.total_adverse_selection += as_cost.max(0.0);
            }
        }
    }

    /// Record a funding payment.
    ///
    /// Positive = paid (cost), Negative = received (revenue)
    pub fn record_funding(&mut self, amount: f64) {
        self.total_funding += amount;
    }

    /// Record a funding payment with full context.
    pub fn record_funding_payment(&mut self, payment: FundingPayment) {
        self.total_funding += payment.amount;
        self.funding_payments.push_back(payment);

        // Trim history
        while self.funding_payments.len() > 100 {
            self.funding_payments.pop_front();
        }
    }

    /// Record an inventory snapshot for carry calculation.
    ///
    /// Should be called periodically (e.g., every position change or every N seconds).
    pub fn record_inventory_snapshot(&mut self, mid_price: f64) {
        let snapshot = InventorySnapshot {
            timestamp: Instant::now(),
            position: self.position,
            mid_price,
        };
        self.inventory_snapshots.push_back(snapshot);

        // Trim history
        while self.inventory_snapshots.len() > self.max_snapshots {
            self.inventory_snapshots.pop_front();
        }
    }

    /// Calculate inventory carry P&L from snapshots.
    ///
    /// Uses the formula: sum of (avg_position * price_change) between consecutive snapshots.
    /// This measures how much P&L came from holding inventory as prices moved.
    pub fn calculate_inventory_carry(&self) -> f64 {
        if self.inventory_snapshots.len() < 2 {
            return 0.0;
        }

        let mut carry = 0.0;
        let snapshots: Vec<_> = self.inventory_snapshots.iter().collect();

        for i in 1..snapshots.len() {
            let prev = snapshots[i - 1];
            let curr = snapshots[i];

            let price_change = curr.mid_price - prev.mid_price;
            let avg_inv = (prev.position + curr.position) / 2.0;
            carry += price_change * avg_inv;
        }

        carry
    }

    /// Analyze fills grouped by depth level (in bps buckets).
    ///
    /// Returns statistics for each depth bucket (0-5bps, 5-10bps, etc.).
    pub fn analyze_fills_by_level(&self) -> HashMap<usize, LevelStats> {
        let mut by_level: HashMap<usize, Vec<&FillRecord>> = HashMap::new();

        for fill in &self.fills {
            // Bucket by 5 bps increments
            let bucket = (fill.depth_from_mid_bps / 5.0).floor() as usize;
            by_level.entry(bucket).or_default().push(fill);
        }

        let mut result = HashMap::new();
        for (level, fills) in by_level {
            let count = fills.len();
            let total_size: f64 = fills.iter().map(|f| f.size).sum();

            // Calculate AS and PnL for this level
            let mut total_as = 0.0;
            let mut total_pnl = 0.0;
            let mut wins = 0;

            for fill in &fills {
                // Calculate spread capture for this fill
                let spread_capture = if fill.is_buy {
                    (fill.mid_at_fill - fill.price) * fill.size
                } else {
                    (fill.price - fill.mid_at_fill) * fill.size
                };

                // Calculate AS if we have the post-fill mid
                if let Some(mid_after) = fill.mid_after_1s {
                    let price_move = mid_after - fill.mid_at_fill;
                    let as_cost = if fill.is_buy {
                        -price_move * fill.size
                    } else {
                        price_move * fill.size
                    };
                    total_as += as_cost.max(0.0);

                    let fill_pnl = spread_capture - as_cost.max(0.0);
                    total_pnl += fill_pnl;
                    if fill_pnl > 0.0 {
                        wins += 1;
                    }
                } else {
                    total_pnl += spread_capture;
                    if spread_capture > 0.0 {
                        wins += 1;
                    }
                }
            }

            result.insert(
                level,
                LevelStats {
                    count,
                    total_size,
                    total_as,
                    total_pnl,
                    win_rate: if count > 0 {
                        wins as f64 / count as f64
                    } else {
                        0.0
                    },
                },
            );
        }

        result
    }

    /// Analyze fills grouped by volatility regime.
    pub fn analyze_fills_by_regime(&self) -> HashMap<VolatilityRegime, LevelStats> {
        let mut by_regime: HashMap<VolatilityRegime, Vec<&FillRecord>> = HashMap::new();

        for fill in &self.fills {
            by_regime.entry(fill.vol_regime).or_default().push(fill);
        }

        let mut result = HashMap::new();
        for (regime, fills) in by_regime {
            let count = fills.len();
            let total_size: f64 = fills.iter().map(|f| f.size).sum();

            let mut total_as = 0.0;
            let mut total_pnl = 0.0;
            let mut wins = 0;

            for fill in &fills {
                let spread_capture = if fill.is_buy {
                    (fill.mid_at_fill - fill.price) * fill.size
                } else {
                    (fill.price - fill.mid_at_fill) * fill.size
                };

                if let Some(mid_after) = fill.mid_after_1s {
                    let price_move = mid_after - fill.mid_at_fill;
                    let as_cost = if fill.is_buy {
                        -price_move * fill.size
                    } else {
                        price_move * fill.size
                    };
                    total_as += as_cost.max(0.0);

                    let fill_pnl = spread_capture - as_cost.max(0.0);
                    total_pnl += fill_pnl;
                    if fill_pnl > 0.0 {
                        wins += 1;
                    }
                } else {
                    total_pnl += spread_capture;
                    if spread_capture > 0.0 {
                        wins += 1;
                    }
                }
            }

            result.insert(
                regime,
                LevelStats {
                    count,
                    total_size,
                    total_as,
                    total_pnl,
                    win_rate: if count > 0 {
                        wins as f64 / count as f64
                    } else {
                        0.0
                    },
                },
            );
        }

        result
    }

    /// Get funding payments history.
    pub fn funding_payments(&self) -> &VecDeque<FundingPayment> {
        &self.funding_payments
    }

    /// Get inventory snapshots history.
    pub fn inventory_snapshots(&self) -> &VecDeque<InventorySnapshot> {
        &self.inventory_snapshots
    }

    /// Get recent fills (most recent first).
    ///
    /// Returns an iterator over the last `n` fills, useful for
    /// analyzing position direction and fill alignment.
    pub fn recent_fills(&self, n: usize) -> impl Iterator<Item = &FillRecord> {
        self.fills.iter().rev().take(n)
    }

    /// Get all fills.
    pub fn fills(&self) -> &VecDeque<FillRecord> {
        &self.fills
    }

    /// Get unrealized P&L at current mid price.
    pub fn unrealized_pnl(&self, current_mid: f64) -> f64 {
        if self.position.abs() < 1e-9 {
            return 0.0;
        }

        if self.position > 0.0 {
            (current_mid - self.avg_entry_price) * self.position
        } else {
            (self.avg_entry_price - current_mid) * self.position.abs()
        }
    }

    /// Get full P&L attribution.
    pub fn attribution(&self, current_mid: f64) -> PnLComponents {
        let unrealized = self.unrealized_pnl(current_mid);

        PnLComponents {
            spread_capture: self.total_spread_capture,
            adverse_selection: self.total_adverse_selection,
            inventory_carry: unrealized, // Inventory carry = unrealized P&L
            funding: self.total_funding,
            fees: self.total_fees,
            realized_pnl: self.total_realized_pnl,
            unrealized_pnl: unrealized,
        }
    }

    /// Get current position.
    pub fn position(&self) -> f64 {
        self.position
    }

    /// Get average entry price.
    pub fn avg_entry_price(&self) -> f64 {
        self.avg_entry_price
    }

    /// Get fill count.
    pub fn fill_count(&self) -> usize {
        self.fill_count
    }

    /// Get summary (read-only, uses current peak_pnl without updating).
    pub fn summary(&self, current_mid: f64) -> PnLSummary {
        let attr = self.attribution(current_mid);
        let total_pnl = attr.total();

        PnLSummary {
            position: self.position,
            avg_entry_price: self.avg_entry_price,
            fill_count: self.fill_count,
            spread_capture: attr.spread_capture,
            adverse_selection: attr.adverse_selection,
            funding: attr.funding,
            fees: attr.fees,
            realized_pnl: attr.realized_pnl,
            unrealized_pnl: attr.unrealized_pnl,
            total_pnl,
            peak_pnl: self.peak_pnl,
            uptime_secs: self.start_time.elapsed().as_secs_f64(),
        }
    }

    /// Get summary and update peak P&L for accurate drawdown tracking.
    /// Call this version after fills or when you want to record new highs.
    pub fn summary_update_peak(&mut self, current_mid: f64) -> PnLSummary {
        let attr = self.attribution(current_mid);
        let total_pnl = attr.total();

        // Update peak for drawdown tracking
        self.peak_pnl = self.peak_pnl.max(total_pnl);

        PnLSummary {
            position: self.position,
            avg_entry_price: self.avg_entry_price,
            fill_count: self.fill_count,
            spread_capture: attr.spread_capture,
            adverse_selection: attr.adverse_selection,
            funding: attr.funding,
            fees: attr.fees,
            realized_pnl: attr.realized_pnl,
            unrealized_pnl: attr.unrealized_pnl,
            total_pnl,
            peak_pnl: self.peak_pnl,
            uptime_secs: self.start_time.elapsed().as_secs_f64(),
        }
    }

    /// Reset the tracker.
    pub fn reset(&mut self) {
        self.fills.clear();
        self.position = 0.0;
        self.avg_entry_price = 0.0;
        self.cost_basis = 0.0;
        self.total_spread_capture = 0.0;
        self.total_adverse_selection = 0.0;
        self.total_funding = 0.0;
        self.total_fees = 0.0;
        self.total_realized_pnl = 0.0;
        self.peak_pnl = 0.0;
        self.fill_count = 0;
        self.start_time = Instant::now();
        self.inventory_snapshots.clear();
        self.funding_payments.clear();
    }

    /// Get full attribution using calculated inventory carry.
    ///
    /// This is the correct formula from the manual:
    /// total = spread_capture - adverse_selection + inventory_carry + funding - fees
    pub fn full_attribution(&self, current_mid: f64) -> PnLComponents {
        let inventory_carry = self.calculate_inventory_carry();
        let unrealized = self.unrealized_pnl(current_mid);

        PnLComponents {
            spread_capture: self.total_spread_capture,
            adverse_selection: self.total_adverse_selection,
            inventory_carry,
            funding: self.total_funding,
            fees: self.total_fees,
            realized_pnl: self.total_realized_pnl,
            unrealized_pnl: unrealized,
        }
    }
}

/// P&L summary for reporting.
#[derive(Debug, Clone)]
pub struct PnLSummary {
    pub position: f64,
    pub avg_entry_price: f64,
    pub fill_count: usize,
    pub spread_capture: f64,
    pub adverse_selection: f64,
    pub funding: f64,
    pub fees: f64,
    pub realized_pnl: f64,
    pub unrealized_pnl: f64,
    pub total_pnl: f64,
    pub peak_pnl: f64,
    pub uptime_secs: f64,
}

impl PnLSummary {
    /// Calculate current drawdown from peak.
    pub fn drawdown(&self) -> f64 {
        (self.peak_pnl - self.total_pnl).max(0.0)
    }
}

// =============================================================================
// Performance-Gated Capacity
// =============================================================================

/// Performance-based position capacity gating.
///
/// Implements "earn the right to size up" by adjusting max position based on
/// realized P&L. When losing money, capacity is reduced to limit further losses.
/// When profitable, full capacity is allowed.
///
/// # Philosophy
/// - Making money → earned trust → full capacity
/// - Losing money → reduced trust → reduced capacity
/// - This creates a natural feedback loop that prevents blowing up
///
/// # Configuration
/// - `loss_reduction_multiplier`: How aggressively to reduce (2.0 = 2x loss ratio)
/// - `min_capacity_fraction`: Floor on capacity (0.3 = never below 30%)
#[derive(Debug, Clone)]
pub struct PerformanceGatedCapacity {
    /// Base max position from configuration (units, e.g., BTC).
    base_max_position: f64,

    /// Reference price for notional calculations.
    /// Updated periodically with current mid price.
    reference_price: f64,

    /// Loss reduction multiplier.
    /// 2.0 means capacity_reduction = 2 × loss_ratio.
    /// Higher = more aggressive reduction when losing.
    loss_reduction_mult: f64,

    /// Minimum capacity fraction (floor).
    /// 0.3 means never go below 30% of base_max_position.
    min_capacity_fraction: f64,

    /// Whether performance gating is enabled.
    enabled: bool,
}

impl Default for PerformanceGatedCapacity {
    fn default() -> Self {
        Self {
            base_max_position: 0.0,
            reference_price: 0.0,
            loss_reduction_mult: 2.0,
            min_capacity_fraction: 0.3,
            enabled: true,
        }
    }
}

impl PerformanceGatedCapacity {
    /// Create a new performance-gated capacity tracker.
    pub fn new(
        base_max_position: f64,
        reference_price: f64,
        loss_reduction_mult: f64,
        min_capacity_fraction: f64,
    ) -> Self {
        Self {
            base_max_position: base_max_position.max(0.0),
            reference_price: reference_price.max(0.0),
            loss_reduction_mult: loss_reduction_mult.max(0.0),
            min_capacity_fraction: min_capacity_fraction.clamp(0.1, 1.0),
            enabled: true,
        }
    }

    /// Create disabled (pass-through) gating.
    pub fn disabled(base_max_position: f64) -> Self {
        Self {
            base_max_position,
            enabled: false,
            ..Default::default()
        }
    }

    /// Update reference price (call periodically).
    pub fn update_reference_price(&mut self, price: f64) {
        if price > 0.0 {
            self.reference_price = price;
        }
    }

    /// Update base max position (if config changes).
    pub fn update_base_max_position(&mut self, max_position: f64) {
        self.base_max_position = max_position.max(0.0);
    }

    /// Calculate allowed max position based on current P&L.
    ///
    /// # Arguments
    /// * `realized_pnl` - Session realized P&L (USD)
    ///
    /// # Returns
    /// Allowed max position (units)
    pub fn allowed_max_position(&self, realized_pnl: f64) -> f64 {
        if !self.enabled || self.base_max_position <= 0.0 {
            return self.base_max_position;
        }

        // Calculate base position notional for reference
        let base_notional = self.base_max_position * self.reference_price.max(1.0);

        // P&L ratio relative to base notional
        // Positive = making money, Negative = losing money
        let pnl_ratio = realized_pnl / base_notional.max(1.0);

        let capacity_fraction = if pnl_ratio >= 0.0 {
            // Making money → full capacity
            1.0
        } else {
            // Losing money → reduce capacity proportionally
            // reduction = loss_ratio × multiplier, clamped to [0, 1 - min_fraction]
            let loss_ratio = pnl_ratio.abs();
            let reduction = (loss_ratio * self.loss_reduction_mult).min(1.0 - self.min_capacity_fraction);
            1.0 - reduction
        };

        // Apply capacity fraction with floor
        self.base_max_position * capacity_fraction.max(self.min_capacity_fraction)
    }

    /// Get the current capacity fraction [min_capacity_fraction, 1.0].
    pub fn capacity_fraction(&self, realized_pnl: f64) -> f64 {
        if !self.enabled || self.base_max_position <= 0.0 {
            return 1.0;
        }

        let allowed = self.allowed_max_position(realized_pnl);
        (allowed / self.base_max_position).clamp(self.min_capacity_fraction, 1.0)
    }

    /// Check if currently at reduced capacity.
    pub fn is_reduced(&self, realized_pnl: f64) -> bool {
        self.enabled && self.capacity_fraction(realized_pnl) < 0.99
    }

    /// Get the base max position.
    pub fn base_max_position(&self) -> f64 {
        self.base_max_position
    }

    /// Check if enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Enable/disable gating.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = PnLConfig::default();
        assert_eq!(config.fee_rate, 0.0002);
    }

    #[test]
    fn test_new_tracker() {
        let tracker = PnLTracker::new(PnLConfig::default());
        assert_eq!(tracker.position(), 0.0);
        assert_eq!(tracker.fill_count(), 0);
    }

    #[test]
    fn test_buy_fill() {
        let mut tracker = PnLTracker::new(PnLConfig::default());

        // Buy 1 BTC at $50000, mid was $50010 (10 bps better)
        tracker.record_fill(1, 50000.0, 1.0, true, 50010.0);

        assert_eq!(tracker.position(), 1.0);
        assert_eq!(tracker.avg_entry_price(), 50000.0);
        assert!((tracker.total_spread_capture - 10.0).abs() < 0.01); // $10 spread capture
    }

    #[test]
    fn test_sell_fill() {
        let mut tracker = PnLTracker::new(PnLConfig::default());

        // Sell 1 BTC at $50010, mid was $50000 (10 bps better)
        tracker.record_fill(1, 50010.0, 1.0, false, 50000.0);

        assert_eq!(tracker.position(), -1.0);
        assert!((tracker.total_spread_capture - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_close_position_realized_pnl() {
        let mut tracker = PnLTracker::new(PnLConfig::default());

        // Buy 1 BTC at $50000
        tracker.record_fill(1, 50000.0, 1.0, true, 50000.0);

        // Sell 1 BTC at $51000 (profit)
        tracker.record_fill(2, 51000.0, 1.0, false, 51000.0);

        assert_eq!(tracker.position(), 0.0);
        assert!((tracker.total_realized_pnl - 1000.0).abs() < 0.01); // $1000 profit
    }

    #[test]
    fn test_unrealized_pnl() {
        let mut tracker = PnLTracker::new(PnLConfig::default());

        // Buy 1 BTC at $50000
        tracker.record_fill(1, 50000.0, 1.0, true, 50000.0);

        // Current mid is $51000
        let unrealized = tracker.unrealized_pnl(51000.0);
        assert!((unrealized - 1000.0).abs() < 0.01);
    }

    #[test]
    fn test_fees() {
        let config = PnLConfig {
            fee_rate: 0.001, // 10 bps
            ..Default::default()
        };
        let mut tracker = PnLTracker::new(config);

        // Trade $50000 notional
        tracker.record_fill(1, 50000.0, 1.0, true, 50000.0);

        assert!((tracker.total_fees - 50.0).abs() < 0.01); // $50 fee
    }

    #[test]
    fn test_attribution() {
        let mut tracker = PnLTracker::new(PnLConfig::default());

        tracker.record_fill(1, 50000.0, 1.0, true, 50010.0);
        tracker.record_funding(5.0); // Paid $5 funding

        let attr = tracker.attribution(50100.0);

        assert!(attr.spread_capture > 0.0);
        assert!(attr.fees > 0.0);
        assert_eq!(attr.funding, 5.0);
        assert!(attr.unrealized_pnl > 0.0); // Price went up
    }

    #[test]
    fn test_adverse_selection_measurement() {
        let mut tracker = PnLTracker::new(PnLConfig::default());

        // Buy at mid $50000
        tracker.record_fill(1, 50000.0, 1.0, true, 50000.0);

        // Price dropped to $49900 after 1s (adverse selection!)
        tracker.update_adverse_selection(1, 49900.0);

        assert!((tracker.total_adverse_selection - 100.0).abs() < 0.01); // $100 AS cost
    }

    #[test]
    fn test_summary() {
        let mut tracker = PnLTracker::new(PnLConfig::default());

        tracker.record_fill(1, 50000.0, 1.0, true, 50010.0);

        let summary = tracker.summary(50100.0);
        assert_eq!(summary.fill_count, 1);
        assert!(summary.total_pnl > 0.0);
    }

    // =============================================================================
    // PerformanceGatedCapacity Tests
    // =============================================================================

    #[test]
    fn test_performance_gated_capacity_profitable() {
        let capacity = PerformanceGatedCapacity::new(
            10.0,     // 10 BTC max
            50000.0,  // $50k reference price
            2.0,      // 2x loss multiplier
            0.3,      // 30% minimum
        );

        // Making money → full capacity
        let allowed = capacity.allowed_max_position(5000.0); // +$5k profit
        assert!((allowed - 10.0).abs() < 0.01);
        assert!(!capacity.is_reduced(5000.0));
    }

    #[test]
    fn test_performance_gated_capacity_losing() {
        let capacity = PerformanceGatedCapacity::new(
            10.0,     // 10 BTC max
            50000.0,  // $50k reference price ($500k notional)
            2.0,      // 2x loss multiplier
            0.3,      // 30% minimum
        );

        // Losing 5% of notional ($25k) → reduction = 5% × 2 = 10%
        let allowed = capacity.allowed_max_position(-25000.0);
        assert!((allowed - 9.0).abs() < 0.1, "Expected ~9.0, got {}", allowed);
        assert!(capacity.is_reduced(-25000.0));

        // Losing 50% of notional ($250k) → reduction capped at 70% (min 30%)
        let allowed_max_loss = capacity.allowed_max_position(-250000.0);
        assert!((allowed_max_loss - 3.0).abs() < 0.1, "Expected ~3.0, got {}", allowed_max_loss);
    }

    #[test]
    fn test_performance_gated_capacity_floor() {
        let capacity = PerformanceGatedCapacity::new(
            10.0,
            50000.0,
            2.0,
            0.3, // 30% floor
        );

        // Even with massive loss, should never go below 30%
        let allowed = capacity.allowed_max_position(-1_000_000.0);
        assert!(allowed >= 3.0, "Should be at least 30% = 3.0, got {}", allowed);
    }

    #[test]
    fn test_performance_gated_capacity_disabled() {
        let capacity = PerformanceGatedCapacity::disabled(10.0);

        // When disabled, always returns base
        assert_eq!(capacity.allowed_max_position(-100000.0), 10.0);
        assert!(!capacity.is_reduced(-100000.0));
    }

    #[test]
    fn test_performance_gated_capacity_fraction() {
        let capacity = PerformanceGatedCapacity::new(10.0, 50000.0, 2.0, 0.3);

        // Profitable → 100%
        assert!((capacity.capacity_fraction(1000.0) - 1.0).abs() < 0.01);

        // Break-even → 100%
        assert!((capacity.capacity_fraction(0.0) - 1.0).abs() < 0.01);

        // Losing → reduced
        assert!(capacity.capacity_fraction(-50000.0) < 1.0);
        assert!(capacity.capacity_fraction(-50000.0) >= 0.3);
    }
}
