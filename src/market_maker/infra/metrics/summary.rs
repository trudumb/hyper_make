//! Metrics summary struct for easy access.

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
