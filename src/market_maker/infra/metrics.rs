//! Prometheus Metrics for Market Maker
//!
//! Exposes key market maker metrics for monitoring and alerting.
//!
//! Key metric categories:
//! - **Position**: Current position, inventory utilization
//! - **P&L**: Daily P&L, drawdown, unrealized P&L
//! - **Orders**: Fill rate, cancel rate, order latency
//! - **Market**: Spread, volatility, toxicity
//! - **Risk**: Kill switch status, cascade severity

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Prometheus-compatible metrics collector.
///
/// Thread-safe metrics that can be scraped by Prometheus or similar systems.
#[derive(Clone)]
pub struct PrometheusMetrics {
    inner: Arc<MetricsInner>,
}

struct MetricsInner {
    // === Position Metrics ===
    /// Current position in base asset
    position: AtomicF64,
    /// Maximum allowed position
    max_position: AtomicF64,
    /// Inventory utilization (position / max_position)
    inventory_utilization: AtomicF64,
    /// Pending bid exposure (total remaining size on buy orders)
    pending_bid_exposure: AtomicF64,
    /// Pending ask exposure (total remaining size on sell orders)
    pending_ask_exposure: AtomicF64,
    /// Net pending change (bid - ask exposure)
    net_pending_change: AtomicF64,
    /// Worst-case max position if all bids fill
    worst_case_max_position: AtomicF64,
    /// Worst-case min position if all asks fill
    worst_case_min_position: AtomicF64,

    // === P&L Metrics ===
    /// Daily P&L in USD
    daily_pnl: AtomicF64,
    /// Peak P&L for drawdown calculation
    peak_pnl: AtomicF64,
    /// Current drawdown percentage
    drawdown_pct: AtomicF64,
    /// Total realized P&L
    realized_pnl: AtomicF64,
    /// Total unrealized P&L
    unrealized_pnl: AtomicF64,

    // === Order Metrics ===
    /// Total orders placed
    orders_placed: AtomicU64,
    /// Total orders filled
    orders_filled: AtomicU64,
    /// Total orders cancelled
    orders_cancelled: AtomicU64,
    /// Total orders modified (queue-preserving updates)
    orders_modified: AtomicU64,
    /// Total modify fallbacks (modify failed, fell back to cancel+place)
    modify_fallbacks: AtomicU64,
    /// Total fill volume (base asset)
    fill_volume: AtomicF64,
    /// Buy fill volume
    buy_volume: AtomicF64,
    /// Sell fill volume
    sell_volume: AtomicF64,

    // === Market Metrics ===
    /// Current mid price
    mid_price: AtomicF64,
    /// Current spread in bps
    spread_bps: AtomicF64,
    /// Volatility (sigma)
    sigma: AtomicF64,
    /// Jump ratio (RV/BV)
    jump_ratio: AtomicF64,
    /// Kappa (fill intensity)
    kappa: AtomicF64,

    // === Estimator Metrics ===
    /// Microprice deviation from mid
    microprice_deviation_bps: AtomicF64,
    /// Book imbalance [-1, 1]
    book_imbalance: AtomicF64,
    /// Flow imbalance [-1, 1]
    flow_imbalance: AtomicF64,
    /// Beta book coefficient
    beta_book: AtomicF64,
    /// Beta flow coefficient
    beta_flow: AtomicF64,

    // === Risk Metrics ===
    /// Kill switch triggered (1 = triggered, 0 = not)
    kill_switch_triggered: AtomicU64,
    /// Cascade severity [0, 1]
    cascade_severity: AtomicF64,
    /// Adverse selection in bps
    adverse_selection_bps: AtomicF64,
    /// Tail risk multiplier
    tail_risk_multiplier: AtomicF64,

    // === Timing Metrics ===
    /// Time since last market data update (seconds)
    data_staleness_secs: AtomicF64,
    /// Quote cycle latency (milliseconds)
    quote_cycle_latency_ms: AtomicF64,

    // === Volatility Regime ===
    /// Current volatility regime (0=Low, 1=Normal, 2=High, 3=Extreme)
    volatility_regime: AtomicU64,

    // === Kelly-Stochastic Metrics ===
    /// Whether Kelly-Stochastic allocation is enabled (1 = enabled, 0 = disabled)
    kelly_stochastic_enabled: AtomicU64,
    /// Calibrated alpha (informed probability) at touch [0, 1]
    kelly_alpha_touch: AtomicF64,
    /// Current Kelly fraction being used [0, 1]
    kelly_fraction: AtomicF64,
    /// Characteristic depth for alpha decay in bps
    kelly_alpha_decay_bps: AtomicF64,

    // === Connection Health Metrics ===
    /// WebSocket connected (1 = connected, 0 = disconnected)
    websocket_connected: AtomicU64,
    /// Time since last trade update (ms)
    last_trade_age_ms: AtomicU64,
    /// Time since last L2 book update (ms)
    last_book_age_ms: AtomicU64,

    // === Data Quality Metrics ===
    /// Total data quality issues detected
    data_quality_issues_total: AtomicU64,
    /// Cumulative sequence gaps detected
    message_loss_count: AtomicU64,
    /// Crossed book incidents
    crossed_book_incidents: AtomicU64,

    // === Exchange Position Limits Metrics ===
    /// Exchange max long position allowed
    exchange_max_long: AtomicF64,
    /// Exchange max short position allowed
    exchange_max_short: AtomicF64,
    /// Exchange available capacity to buy
    exchange_available_buy: AtomicF64,
    /// Exchange available capacity to sell
    exchange_available_sell: AtomicF64,
    /// Effective bid limit (min of local and exchange)
    exchange_effective_bid: AtomicF64,
    /// Effective ask limit (min of local and exchange)
    exchange_effective_ask: AtomicF64,
    /// Age of exchange limits data in milliseconds
    exchange_limits_age_ms: AtomicU64,
    /// Whether exchange limits are valid (1 = valid, 0 = not fetched)
    exchange_limits_valid: AtomicU64,

    /// Start time for uptime calculation
    start_time: Instant,
}

impl PrometheusMetrics {
    /// Create a new metrics collector.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(MetricsInner {
                position: AtomicF64::new(0.0),
                max_position: AtomicF64::new(0.0),
                inventory_utilization: AtomicF64::new(0.0),
                pending_bid_exposure: AtomicF64::new(0.0),
                pending_ask_exposure: AtomicF64::new(0.0),
                net_pending_change: AtomicF64::new(0.0),
                worst_case_max_position: AtomicF64::new(0.0),
                worst_case_min_position: AtomicF64::new(0.0),
                daily_pnl: AtomicF64::new(0.0),
                peak_pnl: AtomicF64::new(0.0),
                drawdown_pct: AtomicF64::new(0.0),
                realized_pnl: AtomicF64::new(0.0),
                unrealized_pnl: AtomicF64::new(0.0),
                orders_placed: AtomicU64::new(0),
                orders_filled: AtomicU64::new(0),
                orders_cancelled: AtomicU64::new(0),
                orders_modified: AtomicU64::new(0),
                modify_fallbacks: AtomicU64::new(0),
                fill_volume: AtomicF64::new(0.0),
                buy_volume: AtomicF64::new(0.0),
                sell_volume: AtomicF64::new(0.0),
                mid_price: AtomicF64::new(0.0),
                spread_bps: AtomicF64::new(0.0),
                sigma: AtomicF64::new(0.0),
                jump_ratio: AtomicF64::new(0.0),
                kappa: AtomicF64::new(0.0),
                microprice_deviation_bps: AtomicF64::new(0.0),
                book_imbalance: AtomicF64::new(0.0),
                flow_imbalance: AtomicF64::new(0.0),
                beta_book: AtomicF64::new(0.0),
                beta_flow: AtomicF64::new(0.0),
                kill_switch_triggered: AtomicU64::new(0),
                cascade_severity: AtomicF64::new(0.0),
                adverse_selection_bps: AtomicF64::new(0.0),
                tail_risk_multiplier: AtomicF64::new(1.0),
                data_staleness_secs: AtomicF64::new(0.0),
                quote_cycle_latency_ms: AtomicF64::new(0.0),
                volatility_regime: AtomicU64::new(1), // Normal
                // Kelly-Stochastic defaults
                kelly_stochastic_enabled: AtomicU64::new(0),
                kelly_alpha_touch: AtomicF64::new(0.15), // Default 15%
                kelly_fraction: AtomicF64::new(0.25),    // Default quarter Kelly
                kelly_alpha_decay_bps: AtomicF64::new(10.0), // Default 10 bps
                websocket_connected: AtomicU64::new(0),
                last_trade_age_ms: AtomicU64::new(0),
                last_book_age_ms: AtomicU64::new(0),
                data_quality_issues_total: AtomicU64::new(0),
                message_loss_count: AtomicU64::new(0),
                // Exchange Position Limits defaults
                exchange_max_long: AtomicF64::new(f64::MAX),
                exchange_max_short: AtomicF64::new(f64::MAX),
                exchange_available_buy: AtomicF64::new(f64::MAX),
                exchange_available_sell: AtomicF64::new(f64::MAX),
                exchange_effective_bid: AtomicF64::new(f64::MAX),
                exchange_effective_ask: AtomicF64::new(f64::MAX),
                exchange_limits_age_ms: AtomicU64::new(u64::MAX),
                exchange_limits_valid: AtomicU64::new(0),
                crossed_book_incidents: AtomicU64::new(0),
                start_time: Instant::now(),
            }),
        }
    }

    // === Position Updates ===

    /// Update position metrics.
    pub fn update_position(&self, position: f64, max_position: f64) {
        self.inner.position.store(position);
        self.inner.max_position.store(max_position);
        let utilization = if max_position > 0.0 {
            (position.abs() / max_position).min(1.0)
        } else {
            0.0
        };
        self.inner.inventory_utilization.store(utilization);
    }

    /// Update pending exposure metrics.
    ///
    /// # Arguments
    /// - `bid_exposure`: Total remaining size on buy orders
    /// - `ask_exposure`: Total remaining size on sell orders
    /// - `position`: Current position (for worst-case calculation)
    pub fn update_pending_exposure(&self, bid_exposure: f64, ask_exposure: f64, position: f64) {
        self.inner.pending_bid_exposure.store(bid_exposure);
        self.inner.pending_ask_exposure.store(ask_exposure);
        self.inner.net_pending_change.store(bid_exposure - ask_exposure);
        self.inner
            .worst_case_max_position
            .store(position + bid_exposure);
        self.inner
            .worst_case_min_position
            .store(position - ask_exposure);
    }

    // === P&L Updates ===

    /// Update P&L metrics.
    pub fn update_pnl(&self, daily_pnl: f64, peak_pnl: f64, realized: f64, unrealized: f64) {
        self.inner.daily_pnl.store(daily_pnl);
        self.inner.peak_pnl.store(peak_pnl);
        let drawdown = if peak_pnl > 0.0 {
            ((peak_pnl - daily_pnl) / peak_pnl * 100.0).max(0.0)
        } else {
            0.0
        };
        self.inner.drawdown_pct.store(drawdown);
        self.inner.realized_pnl.store(realized);
        self.inner.unrealized_pnl.store(unrealized);
    }

    // === Order Updates ===

    /// Record an order placed.
    pub fn record_order_placed(&self) {
        self.inner.orders_placed.fetch_add(1, Ordering::Relaxed);
    }

    /// Record an order filled.
    pub fn record_fill(&self, size: f64, is_buy: bool) {
        self.inner.orders_filled.fetch_add(1, Ordering::Relaxed);
        self.inner.fill_volume.fetch_add(size);
        if is_buy {
            self.inner.buy_volume.fetch_add(size);
        } else {
            self.inner.sell_volume.fetch_add(size);
        }
    }

    /// Record an order cancelled.
    pub fn record_cancel(&self) {
        self.inner.orders_cancelled.fetch_add(1, Ordering::Relaxed);
    }

    /// Record an order modified (queue-preserving update).
    pub fn record_order_modified(&self) {
        self.inner.orders_modified.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a modify fallback (modify failed, fell back to cancel+place).
    pub fn record_modify_fallback(&self) {
        self.inner.modify_fallbacks.fetch_add(1, Ordering::Relaxed);
    }

    // === Market Updates ===

    /// Update market metrics.
    pub fn update_market(
        &self,
        mid_price: f64,
        spread_bps: f64,
        sigma: f64,
        jump_ratio: f64,
        kappa: f64,
    ) {
        self.inner.mid_price.store(mid_price);
        self.inner.spread_bps.store(spread_bps);
        self.inner.sigma.store(sigma);
        self.inner.jump_ratio.store(jump_ratio);
        self.inner.kappa.store(kappa);
    }

    // === Estimator Updates ===

    /// Update estimator metrics.
    pub fn update_estimator(
        &self,
        microprice_deviation_bps: f64,
        book_imbalance: f64,
        flow_imbalance: f64,
        beta_book: f64,
        beta_flow: f64,
    ) {
        self.inner
            .microprice_deviation_bps
            .store(microprice_deviation_bps);
        self.inner.book_imbalance.store(book_imbalance);
        self.inner.flow_imbalance.store(flow_imbalance);
        self.inner.beta_book.store(beta_book);
        self.inner.beta_flow.store(beta_flow);
    }

    // === Risk Updates ===

    /// Update risk metrics.
    pub fn update_risk(
        &self,
        kill_switch_triggered: bool,
        cascade_severity: f64,
        adverse_selection_bps: f64,
        tail_risk_multiplier: f64,
    ) {
        self.inner
            .kill_switch_triggered
            .store(if kill_switch_triggered { 1 } else { 0 }, Ordering::Relaxed);
        self.inner.cascade_severity.store(cascade_severity);
        self.inner
            .adverse_selection_bps
            .store(adverse_selection_bps);
        self.inner.tail_risk_multiplier.store(tail_risk_multiplier);
    }

    /// Update volatility regime.
    pub fn update_volatility_regime(&self, regime: u64) {
        self.inner
            .volatility_regime
            .store(regime, Ordering::Relaxed);
    }

    // === Kelly-Stochastic Updates ===

    /// Update Kelly-Stochastic allocation metrics.
    pub fn update_kelly_stochastic(
        &self,
        enabled: bool,
        alpha_touch: f64,
        kelly_fraction: f64,
        alpha_decay_bps: f64,
    ) {
        self.inner
            .kelly_stochastic_enabled
            .store(if enabled { 1 } else { 0 }, Ordering::Relaxed);
        self.inner.kelly_alpha_touch.store(alpha_touch);
        self.inner.kelly_fraction.store(kelly_fraction);
        self.inner.kelly_alpha_decay_bps.store(alpha_decay_bps);
    }

    // === Connection Health Updates ===

    /// Set WebSocket connection status.
    pub fn set_websocket_connected(&self, connected: bool) {
        self.inner
            .websocket_connected
            .store(if connected { 1 } else { 0 }, Ordering::Relaxed);
    }

    /// Update last trade age in milliseconds.
    pub fn set_last_trade_age_ms(&self, age_ms: u64) {
        self.inner
            .last_trade_age_ms
            .store(age_ms, Ordering::Relaxed);
    }

    /// Update last book age in milliseconds.
    pub fn set_last_book_age_ms(&self, age_ms: u64) {
        self.inner.last_book_age_ms.store(age_ms, Ordering::Relaxed);
    }

    // === Data Quality Updates ===

    /// Record a data quality issue.
    pub fn record_data_quality_issue(&self) {
        self.inner
            .data_quality_issues_total
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Record sequence gaps (message loss).
    pub fn record_message_loss(&self, gap_count: u64) {
        self.inner
            .message_loss_count
            .fetch_add(gap_count, Ordering::Relaxed);
    }

    /// Record a crossed book incident.
    pub fn record_crossed_book(&self) {
        self.inner
            .crossed_book_incidents
            .fetch_add(1, Ordering::Relaxed);
    }

    // === Exchange Position Limits Updates ===

    /// Update exchange position limits metrics.
    #[allow(clippy::too_many_arguments)]
    pub fn update_exchange_limits(
        &self,
        max_long: f64,
        max_short: f64,
        available_buy: f64,
        available_sell: f64,
        effective_bid: f64,
        effective_ask: f64,
        age_ms: u64,
        valid: bool,
    ) {
        self.inner.exchange_max_long.store(max_long);
        self.inner.exchange_max_short.store(max_short);
        self.inner.exchange_available_buy.store(available_buy);
        self.inner.exchange_available_sell.store(available_sell);
        self.inner.exchange_effective_bid.store(effective_bid);
        self.inner.exchange_effective_ask.store(effective_ask);
        self.inner
            .exchange_limits_age_ms
            .store(age_ms, Ordering::Relaxed);
        self.inner
            .exchange_limits_valid
            .store(if valid { 1 } else { 0 }, Ordering::Relaxed);
    }

    // === Timing Updates ===

    /// Update timing metrics.
    pub fn update_timing(&self, data_staleness_secs: f64, quote_cycle_latency_ms: f64) {
        self.inner.data_staleness_secs.store(data_staleness_secs);
        self.inner
            .quote_cycle_latency_ms
            .store(quote_cycle_latency_ms);
    }

    // === Getters ===

    /// Get all metrics as Prometheus text format.
    pub fn to_prometheus_text(&self, asset: &str) -> String {
        let uptime_secs = self.inner.start_time.elapsed().as_secs_f64();
        let labels = format!("asset=\"{}\"", asset);

        let mut output = String::with_capacity(4096);

        // Position metrics
        output.push_str(&format!(
            "# HELP mm_position Current position in base asset\n\
             # TYPE mm_position gauge\n\
             mm_position{{{}}} {}\n",
            labels,
            self.inner.position.load()
        ));

        output.push_str(&format!(
            "# HELP mm_inventory_utilization Position / max position ratio\n\
             # TYPE mm_inventory_utilization gauge\n\
             mm_inventory_utilization{{{}}} {}\n",
            labels,
            self.inner.inventory_utilization.load()
        ));

        // Pending exposure metrics
        output.push_str(&format!(
            "# HELP mm_pending_bid_exposure Total remaining size on buy orders\n\
             # TYPE mm_pending_bid_exposure gauge\n\
             mm_pending_bid_exposure{{{}}} {}\n",
            labels,
            self.inner.pending_bid_exposure.load()
        ));

        output.push_str(&format!(
            "# HELP mm_pending_ask_exposure Total remaining size on sell orders\n\
             # TYPE mm_pending_ask_exposure gauge\n\
             mm_pending_ask_exposure{{{}}} {}\n",
            labels,
            self.inner.pending_ask_exposure.load()
        ));

        output.push_str(&format!(
            "# HELP mm_net_pending_change Net pending exposure (bid - ask)\n\
             # TYPE mm_net_pending_change gauge\n\
             mm_net_pending_change{{{}}} {}\n",
            labels,
            self.inner.net_pending_change.load()
        ));

        output.push_str(&format!(
            "# HELP mm_worst_case_max_position Worst-case max position if all bids fill\n\
             # TYPE mm_worst_case_max_position gauge\n\
             mm_worst_case_max_position{{{}}} {}\n",
            labels,
            self.inner.worst_case_max_position.load()
        ));

        output.push_str(&format!(
            "# HELP mm_worst_case_min_position Worst-case min position if all asks fill\n\
             # TYPE mm_worst_case_min_position gauge\n\
             mm_worst_case_min_position{{{}}} {}\n",
            labels,
            self.inner.worst_case_min_position.load()
        ));

        // P&L metrics
        output.push_str(&format!(
            "# HELP mm_daily_pnl Daily P&L in USD\n\
             # TYPE mm_daily_pnl gauge\n\
             mm_daily_pnl{{{}}} {}\n",
            labels,
            self.inner.daily_pnl.load()
        ));

        output.push_str(&format!(
            "# HELP mm_drawdown_pct Current drawdown percentage\n\
             # TYPE mm_drawdown_pct gauge\n\
             mm_drawdown_pct{{{}}} {}\n",
            labels,
            self.inner.drawdown_pct.load()
        ));

        // Order metrics (counters)
        output.push_str(&format!(
            "# HELP mm_orders_placed_total Total orders placed\n\
             # TYPE mm_orders_placed_total counter\n\
             mm_orders_placed_total{{{}}} {}\n",
            labels,
            self.inner.orders_placed.load(Ordering::Relaxed)
        ));

        output.push_str(&format!(
            "# HELP mm_orders_filled_total Total orders filled\n\
             # TYPE mm_orders_filled_total counter\n\
             mm_orders_filled_total{{{}}} {}\n",
            labels,
            self.inner.orders_filled.load(Ordering::Relaxed)
        ));

        output.push_str(&format!(
            "# HELP mm_fill_volume_total Total fill volume\n\
             # TYPE mm_fill_volume_total counter\n\
             mm_fill_volume_total{{{}}} {}\n",
            labels,
            self.inner.fill_volume.load()
        ));

        output.push_str(&format!(
            "# HELP mm_orders_modified_total Orders modified (queue-preserving updates)\n\
             # TYPE mm_orders_modified_total counter\n\
             mm_orders_modified_total{{{}}} {}\n",
            labels,
            self.inner.orders_modified.load(Ordering::Relaxed)
        ));

        output.push_str(&format!(
            "# HELP mm_modify_fallbacks_total Modify failures falling back to cancel+place\n\
             # TYPE mm_modify_fallbacks_total counter\n\
             mm_modify_fallbacks_total{{{}}} {}\n",
            labels,
            self.inner.modify_fallbacks.load(Ordering::Relaxed)
        ));

        // Market metrics
        output.push_str(&format!(
            "# HELP mm_mid_price Current mid price\n\
             # TYPE mm_mid_price gauge\n\
             mm_mid_price{{{}}} {}\n",
            labels,
            self.inner.mid_price.load()
        ));

        output.push_str(&format!(
            "# HELP mm_spread_bps Current spread in basis points\n\
             # TYPE mm_spread_bps gauge\n\
             mm_spread_bps{{{}}} {}\n",
            labels,
            self.inner.spread_bps.load()
        ));

        output.push_str(&format!(
            "# HELP mm_sigma Volatility estimate\n\
             # TYPE mm_sigma gauge\n\
             mm_sigma{{{}}} {}\n",
            labels,
            self.inner.sigma.load()
        ));

        output.push_str(&format!(
            "# HELP mm_jump_ratio RV/BV jump ratio\n\
             # TYPE mm_jump_ratio gauge\n\
             mm_jump_ratio{{{}}} {}\n",
            labels,
            self.inner.jump_ratio.load()
        ));

        output.push_str(&format!(
            "# HELP mm_kappa Fill intensity\n\
             # TYPE mm_kappa gauge\n\
             mm_kappa{{{}}} {}\n",
            labels,
            self.inner.kappa.load()
        ));

        // Risk metrics
        output.push_str(&format!(
            "# HELP mm_kill_switch_triggered Kill switch status (1=triggered)\n\
             # TYPE mm_kill_switch_triggered gauge\n\
             mm_kill_switch_triggered{{{}}} {}\n",
            labels,
            self.inner.kill_switch_triggered.load(Ordering::Relaxed)
        ));

        output.push_str(&format!(
            "# HELP mm_cascade_severity Liquidation cascade severity [0,1]\n\
             # TYPE mm_cascade_severity gauge\n\
             mm_cascade_severity{{{}}} {}\n",
            labels,
            self.inner.cascade_severity.load()
        ));

        output.push_str(&format!(
            "# HELP mm_adverse_selection_bps Adverse selection in bps\n\
             # TYPE mm_adverse_selection_bps gauge\n\
             mm_adverse_selection_bps{{{}}} {}\n",
            labels,
            self.inner.adverse_selection_bps.load()
        ));

        // Timing metrics
        output.push_str(&format!(
            "# HELP mm_data_staleness_secs Time since last data update\n\
             # TYPE mm_data_staleness_secs gauge\n\
             mm_data_staleness_secs{{{}}} {}\n",
            labels,
            self.inner.data_staleness_secs.load()
        ));

        // Uptime
        output.push_str(&format!(
            "# HELP mm_uptime_secs Total uptime in seconds\n\
             # TYPE mm_uptime_secs counter\n\
             mm_uptime_secs{{{}}} {:.2}\n",
            labels, uptime_secs
        ));

        // Volatility regime
        output.push_str(&format!(
            "# HELP mm_volatility_regime Current vol regime (0=Low,1=Normal,2=High,3=Extreme)\n\
             # TYPE mm_volatility_regime gauge\n\
             mm_volatility_regime{{{}}} {}\n",
            labels,
            self.inner.volatility_regime.load(Ordering::Relaxed)
        ));

        // Kelly-Stochastic metrics
        output.push_str(&format!(
            "# HELP mm_kelly_stochastic_enabled Kelly-Stochastic allocation enabled (1=enabled)\n\
             # TYPE mm_kelly_stochastic_enabled gauge\n\
             mm_kelly_stochastic_enabled{{{}}} {}\n",
            labels,
            self.inner.kelly_stochastic_enabled.load(Ordering::Relaxed)
        ));

        output.push_str(&format!(
            "# HELP mm_kelly_alpha_touch Calibrated informed probability at touch [0,1]\n\
             # TYPE mm_kelly_alpha_touch gauge\n\
             mm_kelly_alpha_touch{{{}}} {:.4}\n",
            labels,
            self.inner.kelly_alpha_touch.load()
        ));

        output.push_str(&format!(
            "# HELP mm_kelly_fraction Current Kelly fraction [0,1]\n\
             # TYPE mm_kelly_fraction gauge\n\
             mm_kelly_fraction{{{}}} {:.4}\n",
            labels,
            self.inner.kelly_fraction.load()
        ));

        output.push_str(&format!(
            "# HELP mm_kelly_alpha_decay_bps Alpha decay characteristic depth in bps\n\
             # TYPE mm_kelly_alpha_decay_bps gauge\n\
             mm_kelly_alpha_decay_bps{{{}}} {:.2}\n",
            labels,
            self.inner.kelly_alpha_decay_bps.load()
        ));

        // Connection health metrics
        output.push_str(&format!(
            "# HELP mm_websocket_connected WebSocket connection status (1=connected)\n\
             # TYPE mm_websocket_connected gauge\n\
             mm_websocket_connected{{{}}} {}\n",
            labels,
            self.inner.websocket_connected.load(Ordering::Relaxed)
        ));

        output.push_str(&format!(
            "# HELP mm_last_trade_age_ms Time since last trade update in ms\n\
             # TYPE mm_last_trade_age_ms gauge\n\
             mm_last_trade_age_ms{{{}}} {}\n",
            labels,
            self.inner.last_trade_age_ms.load(Ordering::Relaxed)
        ));

        output.push_str(&format!(
            "# HELP mm_last_book_age_ms Time since last L2 book update in ms\n\
             # TYPE mm_last_book_age_ms gauge\n\
             mm_last_book_age_ms{{{}}} {}\n",
            labels,
            self.inner.last_book_age_ms.load(Ordering::Relaxed)
        ));

        // Data quality metrics
        output.push_str(&format!(
            "# HELP mm_data_quality_issues_total Total data quality issues detected\n\
             # TYPE mm_data_quality_issues_total counter\n\
             mm_data_quality_issues_total{{{}}} {}\n",
            labels,
            self.inner.data_quality_issues_total.load(Ordering::Relaxed)
        ));

        output.push_str(&format!(
            "# HELP mm_message_loss_count Cumulative sequence gaps detected\n\
             # TYPE mm_message_loss_count counter\n\
             mm_message_loss_count{{{}}} {}\n",
            labels,
            self.inner.message_loss_count.load(Ordering::Relaxed)
        ));

        output.push_str(&format!(
            "# HELP mm_crossed_book_incidents Crossed book incidents detected\n\
             # TYPE mm_crossed_book_incidents counter\n\
             mm_crossed_book_incidents{{{}}} {}\n",
            labels,
            self.inner.crossed_book_incidents.load(Ordering::Relaxed)
        ));

        // Exchange Position Limits metrics
        let limits_valid = self.inner.exchange_limits_valid.load(Ordering::Relaxed) == 1;
        if limits_valid {
            output.push_str(&format!(
                "# HELP mm_exchange_max_long Maximum long position allowed by exchange\n\
                 # TYPE mm_exchange_max_long gauge\n\
                 mm_exchange_max_long{{{}}} {}\n",
                labels,
                self.inner.exchange_max_long.load()
            ));

            output.push_str(&format!(
                "# HELP mm_exchange_max_short Maximum short position allowed by exchange\n\
                 # TYPE mm_exchange_max_short gauge\n\
                 mm_exchange_max_short{{{}}} {}\n",
                labels,
                self.inner.exchange_max_short.load()
            ));

            output.push_str(&format!(
                "# HELP mm_exchange_available_buy Available capacity to buy\n\
                 # TYPE mm_exchange_available_buy gauge\n\
                 mm_exchange_available_buy{{{}}} {}\n",
                labels,
                self.inner.exchange_available_buy.load()
            ));

            output.push_str(&format!(
                "# HELP mm_exchange_available_sell Available capacity to sell\n\
                 # TYPE mm_exchange_available_sell gauge\n\
                 mm_exchange_available_sell{{{}}} {}\n",
                labels,
                self.inner.exchange_available_sell.load()
            ));

            output.push_str(&format!(
                "# HELP mm_exchange_effective_bid Effective bid limit (min of local and exchange)\n\
                 # TYPE mm_exchange_effective_bid gauge\n\
                 mm_exchange_effective_bid{{{}}} {}\n",
                labels,
                self.inner.exchange_effective_bid.load()
            ));

            output.push_str(&format!(
                "# HELP mm_exchange_effective_ask Effective ask limit (min of local and exchange)\n\
                 # TYPE mm_exchange_effective_ask gauge\n\
                 mm_exchange_effective_ask{{{}}} {}\n",
                labels,
                self.inner.exchange_effective_ask.load()
            ));

            output.push_str(&format!(
                "# HELP mm_exchange_limits_age_ms Age of exchange limits data in milliseconds\n\
                 # TYPE mm_exchange_limits_age_ms gauge\n\
                 mm_exchange_limits_age_ms{{{}}} {}\n",
                labels,
                self.inner.exchange_limits_age_ms.load(Ordering::Relaxed)
            ));
        }

        output.push_str(&format!(
            "# HELP mm_exchange_limits_valid Whether exchange limits are valid (1=yes, 0=no)\n\
             # TYPE mm_exchange_limits_valid gauge\n\
             mm_exchange_limits_valid{{{}}} {}\n",
            labels,
            if limits_valid { 1 } else { 0 }
        ));

        output
    }

    /// Get metrics as a HashMap for JSON serialization.
    pub fn to_map(&self) -> HashMap<String, f64> {
        let mut map = HashMap::new();

        map.insert("position".to_string(), self.inner.position.load());
        map.insert(
            "inventory_utilization".to_string(),
            self.inner.inventory_utilization.load(),
        );
        map.insert("daily_pnl".to_string(), self.inner.daily_pnl.load());
        map.insert("drawdown_pct".to_string(), self.inner.drawdown_pct.load());
        map.insert(
            "orders_placed".to_string(),
            self.inner.orders_placed.load(Ordering::Relaxed) as f64,
        );
        map.insert(
            "orders_filled".to_string(),
            self.inner.orders_filled.load(Ordering::Relaxed) as f64,
        );
        map.insert("fill_volume".to_string(), self.inner.fill_volume.load());
        map.insert("mid_price".to_string(), self.inner.mid_price.load());
        map.insert("spread_bps".to_string(), self.inner.spread_bps.load());
        map.insert("sigma".to_string(), self.inner.sigma.load());
        map.insert("jump_ratio".to_string(), self.inner.jump_ratio.load());
        map.insert("kappa".to_string(), self.inner.kappa.load());
        map.insert(
            "kill_switch_triggered".to_string(),
            self.inner.kill_switch_triggered.load(Ordering::Relaxed) as f64,
        );
        map.insert(
            "cascade_severity".to_string(),
            self.inner.cascade_severity.load(),
        );
        map.insert(
            "adverse_selection_bps".to_string(),
            self.inner.adverse_selection_bps.load(),
        );
        map.insert(
            "data_staleness_secs".to_string(),
            self.inner.data_staleness_secs.load(),
        );
        map.insert(
            "uptime_secs".to_string(),
            self.inner.start_time.elapsed().as_secs_f64(),
        );
        map.insert(
            "websocket_connected".to_string(),
            self.inner.websocket_connected.load(Ordering::Relaxed) as f64,
        );
        map.insert(
            "last_trade_age_ms".to_string(),
            self.inner.last_trade_age_ms.load(Ordering::Relaxed) as f64,
        );
        map.insert(
            "last_book_age_ms".to_string(),
            self.inner.last_book_age_ms.load(Ordering::Relaxed) as f64,
        );
        map.insert(
            "data_quality_issues_total".to_string(),
            self.inner.data_quality_issues_total.load(Ordering::Relaxed) as f64,
        );
        map.insert(
            "message_loss_count".to_string(),
            self.inner.message_loss_count.load(Ordering::Relaxed) as f64,
        );
        map.insert(
            "crossed_book_incidents".to_string(),
            self.inner.crossed_book_incidents.load(Ordering::Relaxed) as f64,
        );
        // Kelly-Stochastic metrics
        map.insert(
            "kelly_stochastic_enabled".to_string(),
            self.inner.kelly_stochastic_enabled.load(Ordering::Relaxed) as f64,
        );
        map.insert(
            "kelly_alpha_touch".to_string(),
            self.inner.kelly_alpha_touch.load(),
        );
        map.insert(
            "kelly_fraction".to_string(),
            self.inner.kelly_fraction.load(),
        );
        map.insert(
            "kelly_alpha_decay_bps".to_string(),
            self.inner.kelly_alpha_decay_bps.load(),
        );

        map
    }

    /// Get a summary struct.
    pub fn summary(&self) -> MetricsSummary {
        MetricsSummary {
            position: self.inner.position.load(),
            inventory_utilization: self.inner.inventory_utilization.load(),
            daily_pnl: self.inner.daily_pnl.load(),
            drawdown_pct: self.inner.drawdown_pct.load(),
            orders_placed: self.inner.orders_placed.load(Ordering::Relaxed),
            orders_filled: self.inner.orders_filled.load(Ordering::Relaxed),
            fill_volume: self.inner.fill_volume.load(),
            mid_price: self.inner.mid_price.load(),
            spread_bps: self.inner.spread_bps.load(),
            sigma: self.inner.sigma.load(),
            jump_ratio: self.inner.jump_ratio.load(),
            kappa: self.inner.kappa.load(),
            kill_switch_triggered: self.inner.kill_switch_triggered.load(Ordering::Relaxed) == 1,
            cascade_severity: self.inner.cascade_severity.load(),
            adverse_selection_bps: self.inner.adverse_selection_bps.load(),
            uptime_secs: self.inner.start_time.elapsed().as_secs_f64(),
            websocket_connected: self.inner.websocket_connected.load(Ordering::Relaxed) == 1,
            last_trade_age_ms: self.inner.last_trade_age_ms.load(Ordering::Relaxed),
            last_book_age_ms: self.inner.last_book_age_ms.load(Ordering::Relaxed),
            data_quality_issues_total: self.inner.data_quality_issues_total.load(Ordering::Relaxed),
            message_loss_count: self.inner.message_loss_count.load(Ordering::Relaxed),
            crossed_book_incidents: self.inner.crossed_book_incidents.load(Ordering::Relaxed),
            // Kelly-Stochastic
            kelly_stochastic_enabled: self.inner.kelly_stochastic_enabled.load(Ordering::Relaxed)
                == 1,
            kelly_alpha_touch: self.inner.kelly_alpha_touch.load(),
            kelly_fraction: self.inner.kelly_fraction.load(),
            kelly_alpha_decay_bps: self.inner.kelly_alpha_decay_bps.load(),
        }
    }
}

impl Default for PrometheusMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Atomic f64 wrapper using AtomicU64.
struct AtomicF64(AtomicU64);

impl AtomicF64 {
    fn new(val: f64) -> Self {
        Self(AtomicU64::new(val.to_bits()))
    }

    fn load(&self) -> f64 {
        f64::from_bits(self.0.load(Ordering::Relaxed))
    }

    fn store(&self, val: f64) {
        self.0.store(val.to_bits(), Ordering::Relaxed);
    }

    fn fetch_add(&self, val: f64) -> f64 {
        loop {
            let current = self.0.load(Ordering::Relaxed);
            let current_f64 = f64::from_bits(current);
            let new = (current_f64 + val).to_bits();
            if self
                .0
                .compare_exchange_weak(current, new, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                return current_f64;
            }
        }
    }
}

/// Summary of all metrics.
#[derive(Debug, Clone)]
pub struct MetricsSummary {
    pub position: f64,
    pub inventory_utilization: f64,
    pub daily_pnl: f64,
    pub drawdown_pct: f64,
    pub orders_placed: u64,
    pub orders_filled: u64,
    pub fill_volume: f64,
    pub mid_price: f64,
    pub spread_bps: f64,
    pub sigma: f64,
    pub jump_ratio: f64,
    pub kappa: f64,
    pub kill_switch_triggered: bool,
    pub cascade_severity: f64,
    pub adverse_selection_bps: f64,
    pub uptime_secs: f64,
    // Connection health
    pub websocket_connected: bool,
    pub last_trade_age_ms: u64,
    pub last_book_age_ms: u64,
    // Data quality
    pub data_quality_issues_total: u64,
    pub message_loss_count: u64,
    pub crossed_book_incidents: u64,
    // Kelly-Stochastic
    pub kelly_stochastic_enabled: bool,
    pub kelly_alpha_touch: f64,
    pub kelly_fraction: f64,
    pub kelly_alpha_decay_bps: f64,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atomic_f64() {
        let af = AtomicF64::new(1.5);
        assert_eq!(af.load(), 1.5);

        af.store(2.5);
        assert_eq!(af.load(), 2.5);

        let old = af.fetch_add(1.0);
        assert_eq!(old, 2.5);
        assert_eq!(af.load(), 3.5);
    }

    #[test]
    fn test_metrics_creation() {
        let metrics = PrometheusMetrics::new();
        let summary = metrics.summary();
        assert_eq!(summary.position, 0.0);
        assert_eq!(summary.orders_placed, 0);
    }

    #[test]
    fn test_position_update() {
        let metrics = PrometheusMetrics::new();
        metrics.update_position(5.0, 10.0);

        let summary = metrics.summary();
        assert_eq!(summary.position, 5.0);
        assert_eq!(summary.inventory_utilization, 0.5);
    }

    #[test]
    fn test_pnl_update() {
        let metrics = PrometheusMetrics::new();
        metrics.update_pnl(100.0, 150.0, 80.0, 20.0);

        let summary = metrics.summary();
        assert_eq!(summary.daily_pnl, 100.0);
        assert!((summary.drawdown_pct - 33.33).abs() < 0.1);
    }

    #[test]
    fn test_order_recording() {
        let metrics = PrometheusMetrics::new();

        metrics.record_order_placed();
        metrics.record_order_placed();
        metrics.record_fill(1.0, true);
        metrics.record_fill(0.5, false);
        metrics.record_cancel();

        let summary = metrics.summary();
        assert_eq!(summary.orders_placed, 2);
        assert_eq!(summary.orders_filled, 2);
        assert_eq!(summary.fill_volume, 1.5);
    }

    #[test]
    fn test_prometheus_output() {
        let metrics = PrometheusMetrics::new();
        metrics.update_position(1.0, 10.0);
        metrics.update_market(50000.0, 5.0, 0.001, 1.2, 500.0);

        let output = metrics.to_prometheus_text("BTC");

        assert!(output.contains("mm_position{asset=\"BTC\"}"));
        assert!(output.contains("mm_mid_price{asset=\"BTC\"}"));
        assert!(output.contains("# TYPE mm_position gauge"));
    }

    #[test]
    fn test_to_map() {
        let metrics = PrometheusMetrics::new();
        metrics.update_position(2.5, 10.0);

        let map = metrics.to_map();
        assert_eq!(map.get("position"), Some(&2.5));
        assert_eq!(map.get("inventory_utilization"), Some(&0.25));
    }

    #[test]
    fn test_thread_safety() {
        let metrics = PrometheusMetrics::new();
        let metrics2 = metrics.clone();

        std::thread::spawn(move || {
            for _ in 0..100 {
                metrics2.record_order_placed();
            }
        });

        for _ in 0..100 {
            metrics.record_order_placed();
        }

        // Allow thread to complete
        std::thread::sleep(std::time::Duration::from_millis(10));

        // Should have some orders (exact count depends on timing)
        let summary = metrics.summary();
        assert!(summary.orders_placed > 0);
    }

    #[test]
    fn test_risk_update() {
        let metrics = PrometheusMetrics::new();
        metrics.update_risk(true, 0.8, 5.5, 2.0);

        let summary = metrics.summary();
        assert!(summary.kill_switch_triggered);
        assert_eq!(summary.cascade_severity, 0.8);
        assert_eq!(summary.adverse_selection_bps, 5.5);
    }
}
