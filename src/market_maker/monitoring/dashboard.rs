//! Dashboard state for terminal-friendly real-time monitoring.
//!
//! Provides a simplified, terminal-friendly view of market maker state
//! with ASCII formatting for quick status checks.

use std::time::{SystemTime, UNIX_EPOCH};

/// Side of the current position.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PositionSide {
    /// No position
    #[default]
    Flat,
    /// Long position (positive size)
    Long,
    /// Short position (negative size)
    Short,
}

impl PositionSide {
    /// Create from position size.
    pub fn from_size(size: f64) -> Self {
        if size.abs() < 1e-10 {
            PositionSide::Flat
        } else if size > 0.0 {
            PositionSide::Long
        } else {
            PositionSide::Short
        }
    }

    /// Get display string.
    pub fn as_str(&self) -> &'static str {
        match self {
            PositionSide::Flat => "FLAT",
            PositionSide::Long => "LONG",
            PositionSide::Short => "SHORT",
        }
    }
}

/// Real-time dashboard state for terminal display.
///
/// Tracks all key metrics for a market making session in a simple,
/// serializable format suitable for terminal dashboards.
#[derive(Debug, Clone)]
pub struct DashboardState {
    // === PnL ===
    /// Total unrealized + realized PnL in USD
    pub pnl_usd: f64,
    /// PnL as percentage of account
    pub pnl_pct: f64,
    /// Session-specific PnL (resets on restart)
    pub session_pnl_usd: f64,

    // === Position ===
    /// Current position size (signed, positive = long)
    pub position_size: f64,
    /// Position notional value in USD
    pub position_notional: f64,
    /// Position side (Long/Short/Flat)
    pub position_side: PositionSide,

    // === Market ===
    /// Current regime: 0=calm, 1=volatile, 2=cascade
    pub current_regime: usize,
    /// Confidence in regime classification [0, 1]
    pub regime_confidence: f64,
    /// Current spread in basis points
    pub current_spread_bps: f64,
    /// Current mid price
    pub mid_price: f64,

    // === Execution ===
    /// Fill rate (fills per hour)
    pub fill_rate: f64,
    /// Average queue position [0, 1] (0 = front)
    pub avg_queue_position: f64,
    /// Median latency in milliseconds
    pub latency_p50_ms: f64,

    // === Calibration ===
    /// Information ratio for fill probability model
    pub ir_fill_prob: f64,
    /// Information ratio for adverse selection model
    pub ir_adverse_sel: f64,
    /// Information ratio for regime detection model
    pub ir_regime: f64,

    // === Activity ===
    /// Number of trades in last 24 hours
    pub trades_24h: usize,
    /// Volume traded in last 24 hours (USD)
    pub volume_24h_usd: f64,
    /// Fees paid in last 24 hours (USD)
    pub fees_24h_usd: f64,

    // === Timestamps ===
    /// Unix timestamp of last update (millis)
    pub last_update: u64,
    /// Unix timestamp of last trade (millis)
    pub last_trade: u64,
}

impl Default for DashboardState {
    fn default() -> Self {
        Self::new()
    }
}

impl DashboardState {
    /// Create a new dashboard state with default values.
    pub fn new() -> Self {
        let now = current_timestamp_millis();
        Self {
            // PnL
            pnl_usd: 0.0,
            pnl_pct: 0.0,
            session_pnl_usd: 0.0,
            // Position
            position_size: 0.0,
            position_notional: 0.0,
            position_side: PositionSide::Flat,
            // Market
            current_regime: 0,
            regime_confidence: 1.0,
            current_spread_bps: 0.0,
            mid_price: 0.0,
            // Execution
            fill_rate: 0.0,
            avg_queue_position: 0.5,
            latency_p50_ms: 0.0,
            // Calibration
            ir_fill_prob: 1.0,
            ir_adverse_sel: 1.0,
            ir_regime: 1.0,
            // Activity
            trades_24h: 0,
            volume_24h_usd: 0.0,
            fees_24h_usd: 0.0,
            // Timestamps
            last_update: now,
            last_trade: 0,
        }
    }

    /// Update PnL metrics.
    pub fn update_pnl(&mut self, pnl_usd: f64, pnl_pct: f64) {
        self.pnl_usd = pnl_usd;
        self.pnl_pct = pnl_pct;
        self.last_update = current_timestamp_millis();
    }

    /// Update session PnL.
    pub fn update_session_pnl(&mut self, session_pnl_usd: f64) {
        self.session_pnl_usd = session_pnl_usd;
        self.last_update = current_timestamp_millis();
    }

    /// Update position metrics.
    pub fn update_position(&mut self, size: f64, notional: f64) {
        self.position_size = size;
        self.position_notional = notional;
        self.position_side = PositionSide::from_size(size);
        self.last_update = current_timestamp_millis();
    }

    /// Update market metrics.
    pub fn update_market(&mut self, regime: usize, confidence: f64, spread_bps: f64, mid: f64) {
        self.current_regime = regime;
        self.regime_confidence = confidence;
        self.current_spread_bps = spread_bps;
        self.mid_price = mid;
        self.last_update = current_timestamp_millis();
    }

    /// Update execution metrics.
    pub fn update_execution(&mut self, fill_rate: f64, queue_pos: f64, latency: f64) {
        self.fill_rate = fill_rate;
        self.avg_queue_position = queue_pos;
        self.latency_p50_ms = latency;
        self.last_update = current_timestamp_millis();
    }

    /// Update calibration metrics (information ratios).
    pub fn update_calibration(&mut self, ir_fill: f64, ir_adverse: f64, ir_regime: f64) {
        self.ir_fill_prob = ir_fill;
        self.ir_adverse_sel = ir_adverse;
        self.ir_regime = ir_regime;
        self.last_update = current_timestamp_millis();
    }

    /// Update activity metrics.
    pub fn update_activity(&mut self, trades: usize, volume: f64, fees: f64) {
        self.trades_24h = trades;
        self.volume_24h_usd = volume;
        self.fees_24h_usd = fees;
        self.last_update = current_timestamp_millis();
    }

    /// Record a trade occurred.
    pub fn record_trade(&mut self) {
        self.last_trade = current_timestamp_millis();
    }

    /// Get human-readable regime name.
    pub fn regime_name(&self) -> &'static str {
        match self.current_regime {
            0 => "Calm",
            1 => "Volatile",
            2 => "Cascade",
            _ => "Unknown",
        }
    }

    /// Check if all calibration metrics are healthy.
    ///
    /// Returns true if all information ratios are above the minimum threshold.
    /// IR > 1.0 means the model adds value over the base rate.
    pub fn all_calibration_healthy(&self, min_ir: f64) -> bool {
        self.ir_fill_prob >= min_ir && self.ir_adverse_sel >= min_ir && self.ir_regime >= min_ir
    }

    /// Format a compact ASCII summary for terminal display.
    ///
    /// Returns a multi-line string with all key metrics formatted
    /// for easy reading in a terminal.
    pub fn format_summary(&self) -> String {
        let pnl_sign = if self.pnl_usd >= 0.0 { "+" } else { "-" };
        let pnl_abs = self.pnl_usd.abs();
        let session_sign = if self.session_pnl_usd >= 0.0 {
            "+"
        } else {
            "-"
        };
        let session_abs = self.session_pnl_usd.abs();

        let calib_status = if self.all_calibration_healthy(1.0) {
            "OK"
        } else {
            "WARN"
        };

        let regime_indicator = match self.current_regime {
            0 => "[=]", // Calm
            1 => "[~]", // Volatile
            2 => "[!]", // Cascade
            _ => "[?]",
        };

        format!(
            r#"
+==============================================================================+
|                         MARKET MAKER DASHBOARD                               |
+==============================================================================+
|  PnL                                                                         |
|    Total:     {pnl_sign}${pnl_abs:>12.2} ({pnl_sign}{pnl_pct:>6.2}%)                                       |
|    Session:   {session_sign}${session_abs:>12.2}                                               |
+------------------------------------------------------------------------------+
|  Position                                                                    |
|    Size:      {pos_size:>12.4} {side:<6}                                         |
|    Notional:  ${pos_notional:>12.2}                                               |
+------------------------------------------------------------------------------+
|  Market {regime_indicator}                                                                 |
|    Regime:    {regime:<12} (conf: {regime_conf:>5.1}%)                              |
|    Spread:    {spread:>12.2} bps                                                |
|    Mid:       ${mid:>12.2}                                                 |
+------------------------------------------------------------------------------+
|  Execution                                                                   |
|    Fill Rate: {fill_rate:>12.1}/hr                                                |
|    Queue Pos: {queue:>12.2}                                                      |
|    Latency:   {latency:>12.1} ms (p50)                                           |
+------------------------------------------------------------------------------+
|  Calibration [{calib_status}]                                                           |
|    Fill IR:   {ir_fill:>12.3}    AS IR: {ir_as:>8.3}    Regime IR: {ir_regime:>6.3}    |
+------------------------------------------------------------------------------+
|  Activity (24h)                                                              |
|    Trades:    {trades:>12}                                                       |
|    Volume:    ${volume:>12.2}                                               |
|    Fees:      ${fees:>12.2}                                                  |
+==============================================================================+
|  Last Update: {last_update}    Last Trade: {last_trade}                      |
+==============================================================================+
"#,
            pnl_sign = pnl_sign,
            pnl_abs = pnl_abs,
            pnl_pct = self.pnl_pct.abs() * 100.0,
            session_sign = session_sign,
            session_abs = session_abs,
            pos_size = self.position_size,
            side = self.position_side.as_str(),
            pos_notional = self.position_notional,
            regime_indicator = regime_indicator,
            regime = self.regime_name(),
            regime_conf = self.regime_confidence * 100.0,
            spread = self.current_spread_bps,
            mid = self.mid_price,
            fill_rate = self.fill_rate,
            queue = self.avg_queue_position,
            latency = self.latency_p50_ms,
            calib_status = calib_status,
            ir_fill = self.ir_fill_prob,
            ir_as = self.ir_adverse_sel,
            ir_regime = self.ir_regime,
            trades = self.trades_24h,
            volume = self.volume_24h_usd,
            fees = self.fees_24h_usd,
            last_update = format_timestamp(self.last_update),
            last_trade = if self.last_trade > 0 {
                format_timestamp(self.last_trade)
            } else {
                "Never".to_string()
            },
        )
    }

    /// Format a single-line status for compact display.
    pub fn format_one_line(&self) -> String {
        let pnl_sign = if self.pnl_usd >= 0.0 { "+" } else { "" };
        format!(
            "PnL: {}{:.2} | Pos: {:.4} {} | Spread: {:.1}bps | Regime: {} | IR: {:.2}/{:.2}/{:.2}",
            pnl_sign,
            self.pnl_usd,
            self.position_size,
            self.position_side.as_str(),
            self.current_spread_bps,
            self.regime_name(),
            self.ir_fill_prob,
            self.ir_adverse_sel,
            self.ir_regime,
        )
    }
}

/// Get current timestamp in milliseconds.
fn current_timestamp_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Format timestamp for display (HH:MM:SS).
fn format_timestamp(millis: u64) -> String {
    // Simple formatting without external deps
    let secs = millis / 1000;
    let hours = (secs / 3600) % 24;
    let mins = (secs / 60) % 60;
    let s = secs % 60;
    format!("{:02}:{:02}:{:02}", hours, mins, s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_position_side_from_size() {
        assert_eq!(PositionSide::from_size(0.0), PositionSide::Flat);
        assert_eq!(PositionSide::from_size(1e-15), PositionSide::Flat);
        assert_eq!(PositionSide::from_size(1.0), PositionSide::Long);
        assert_eq!(PositionSide::from_size(-1.0), PositionSide::Short);
    }

    #[test]
    fn test_dashboard_new() {
        let dashboard = DashboardState::new();
        assert_eq!(dashboard.pnl_usd, 0.0);
        assert_eq!(dashboard.position_side, PositionSide::Flat);
        assert_eq!(dashboard.current_regime, 0);
        assert!(dashboard.last_update > 0);
    }

    #[test]
    fn test_update_pnl() {
        let mut dashboard = DashboardState::new();
        dashboard.update_pnl(100.0, 0.05);

        assert_eq!(dashboard.pnl_usd, 100.0);
        assert_eq!(dashboard.pnl_pct, 0.05);
    }

    #[test]
    fn test_update_position() {
        let mut dashboard = DashboardState::new();
        dashboard.update_position(2.5, 125000.0);

        assert_eq!(dashboard.position_size, 2.5);
        assert_eq!(dashboard.position_notional, 125000.0);
        assert_eq!(dashboard.position_side, PositionSide::Long);

        dashboard.update_position(-1.0, 50000.0);
        assert_eq!(dashboard.position_side, PositionSide::Short);
    }

    #[test]
    fn test_update_market() {
        let mut dashboard = DashboardState::new();
        dashboard.update_market(1, 0.85, 5.5, 50000.0);

        assert_eq!(dashboard.current_regime, 1);
        assert_eq!(dashboard.regime_confidence, 0.85);
        assert_eq!(dashboard.current_spread_bps, 5.5);
        assert_eq!(dashboard.mid_price, 50000.0);
    }

    #[test]
    fn test_update_execution() {
        let mut dashboard = DashboardState::new();
        dashboard.update_execution(25.0, 0.3, 15.5);

        assert_eq!(dashboard.fill_rate, 25.0);
        assert_eq!(dashboard.avg_queue_position, 0.3);
        assert_eq!(dashboard.latency_p50_ms, 15.5);
    }

    #[test]
    fn test_update_calibration() {
        let mut dashboard = DashboardState::new();
        dashboard.update_calibration(1.2, 0.9, 1.5);

        assert_eq!(dashboard.ir_fill_prob, 1.2);
        assert_eq!(dashboard.ir_adverse_sel, 0.9);
        assert_eq!(dashboard.ir_regime, 1.5);
    }

    #[test]
    fn test_all_calibration_healthy() {
        let mut dashboard = DashboardState::new();

        // All default to 1.0, should be healthy at threshold 1.0
        assert!(dashboard.all_calibration_healthy(1.0));

        // Update one below threshold
        dashboard.update_calibration(1.2, 0.8, 1.5);
        assert!(!dashboard.all_calibration_healthy(1.0));

        // All above threshold
        dashboard.update_calibration(1.2, 1.1, 1.5);
        assert!(dashboard.all_calibration_healthy(1.0));
    }

    #[test]
    fn test_regime_name() {
        let mut dashboard = DashboardState::new();

        dashboard.current_regime = 0;
        assert_eq!(dashboard.regime_name(), "Calm");

        dashboard.current_regime = 1;
        assert_eq!(dashboard.regime_name(), "Volatile");

        dashboard.current_regime = 2;
        assert_eq!(dashboard.regime_name(), "Cascade");

        dashboard.current_regime = 99;
        assert_eq!(dashboard.regime_name(), "Unknown");
    }

    #[test]
    fn test_format_summary() {
        let mut dashboard = DashboardState::new();
        dashboard.update_pnl(150.50, 0.025);
        dashboard.update_position(1.5, 75000.0);
        dashboard.update_market(0, 0.95, 4.5, 50000.0);
        dashboard.update_execution(30.0, 0.2, 12.0);
        dashboard.update_calibration(1.2, 1.1, 1.3);
        dashboard.update_activity(45, 225000.0, 33.75);

        let summary = dashboard.format_summary();

        // Check key elements are present
        assert!(summary.contains("MARKET MAKER DASHBOARD"));
        assert!(summary.contains("+$"));
        assert!(summary.contains("150.50"));
        assert!(summary.contains("LONG"));
        assert!(summary.contains("Calm"));
        assert!(summary.contains("4.5"));
        assert!(summary.contains("[OK]"));
    }

    #[test]
    fn test_format_summary_negative_pnl() {
        let mut dashboard = DashboardState::new();
        dashboard.update_pnl(-50.25, -0.01);

        let summary = dashboard.format_summary();
        assert!(summary.contains("-$"));
    }

    #[test]
    fn test_format_one_line() {
        let mut dashboard = DashboardState::new();
        dashboard.update_pnl(100.0, 0.02);
        dashboard.update_position(0.5, 25000.0);
        dashboard.update_market(0, 0.9, 3.0, 50000.0);
        dashboard.update_calibration(1.1, 1.0, 1.2);

        let line = dashboard.format_one_line();

        assert!(line.contains("PnL:"));
        assert!(line.contains("100.00"));
        assert!(line.contains("LONG"));
        assert!(line.contains("3.0bps"));
        assert!(line.contains("Calm"));
    }

    #[test]
    fn test_record_trade() {
        let mut dashboard = DashboardState::new();
        assert_eq!(dashboard.last_trade, 0);

        dashboard.record_trade();
        assert!(dashboard.last_trade > 0);
    }

    #[test]
    fn test_update_activity() {
        let mut dashboard = DashboardState::new();
        dashboard.update_activity(100, 500000.0, 75.0);

        assert_eq!(dashboard.trades_24h, 100);
        assert_eq!(dashboard.volume_24h_usd, 500000.0);
        assert_eq!(dashboard.fees_24h_usd, 75.0);
    }

    #[test]
    fn test_default() {
        let dashboard = DashboardState::default();
        assert_eq!(dashboard.pnl_usd, 0.0);
        assert_eq!(dashboard.position_side, PositionSide::Flat);
    }

    #[test]
    fn test_format_timestamp() {
        // Test known timestamp: 3661000ms = 1h 1m 1s
        let ts = 3661000;
        let formatted = format_timestamp(ts);
        assert_eq!(formatted, "01:01:01");
    }
}
