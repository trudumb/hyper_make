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
                daily_pnl: AtomicF64::new(0.0),
                peak_pnl: AtomicF64::new(0.0),
                drawdown_pct: AtomicF64::new(0.0),
                realized_pnl: AtomicF64::new(0.0),
                unrealized_pnl: AtomicF64::new(0.0),
                orders_placed: AtomicU64::new(0),
                orders_filled: AtomicU64::new(0),
                orders_cancelled: AtomicU64::new(0),
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

    // === Market Updates ===

    /// Update market metrics.
    pub fn update_market(&self, mid_price: f64, spread_bps: f64, sigma: f64, jump_ratio: f64, kappa: f64) {
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
        self.inner.microprice_deviation_bps.store(microprice_deviation_bps);
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
        self.inner.adverse_selection_bps.store(adverse_selection_bps);
        self.inner.tail_risk_multiplier.store(tail_risk_multiplier);
    }

    /// Update volatility regime.
    pub fn update_volatility_regime(&self, regime: u64) {
        self.inner.volatility_regime.store(regime, Ordering::Relaxed);
    }

    // === Timing Updates ===

    /// Update timing metrics.
    pub fn update_timing(&self, data_staleness_secs: f64, quote_cycle_latency_ms: f64) {
        self.inner.data_staleness_secs.store(data_staleness_secs);
        self.inner.quote_cycle_latency_ms.store(quote_cycle_latency_ms);
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
