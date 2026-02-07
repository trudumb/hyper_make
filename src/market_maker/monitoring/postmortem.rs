//! Post-mortem state dump on kill switch trigger.
//!
//! Writes a detailed JSON file capturing the full system state at the moment
//! the kill switch fires. Essential for incident analysis and debugging.

use serde::Serialize;
use std::path::{Path, PathBuf};

/// Complete state dump for post-mortem analysis.
#[derive(Debug, Clone, Serialize)]
pub struct PostMortemDump {
    /// ISO 8601 timestamp
    pub timestamp: String,
    /// What triggered the kill switch
    pub trigger: String,
    /// The metric value that breached the threshold
    pub trigger_value: f64,
    /// The threshold that was breached
    pub threshold: f64,
    /// Current position in contracts
    pub position: f64,
    /// Current unrealized P&L in USD
    pub unrealized_pnl: f64,
    /// Current realized P&L in USD
    pub realized_pnl: f64,
    /// Total daily P&L
    pub daily_pnl: f64,
    /// Current regime (0=calm, 1=volatile, 2=cascade)
    pub regime: u8,
    /// Cascade severity at time of kill
    pub cascade_severity: f64,
    /// Current mid price
    pub mid_price: f64,
    /// Current spread in bps
    pub spread_bps: f64,
    /// Recent fills (last 10)
    pub recent_fills: Vec<FillRecord>,
    /// Signal state at time of kill
    pub signal_state: SignalSnapshot,
    /// Market microstructure state
    pub market_state: MarketSnapshot,
    /// Risk monitor states
    pub risk_state: RiskSnapshot,
}

/// Record of a recent fill for post-mortem.
#[derive(Debug, Clone, Serialize)]
pub struct FillRecord {
    pub timestamp_ms: u64,
    pub side: String,
    pub price: f64,
    pub size: f64,
    pub fee: f64,
}

/// Signal state snapshot.
#[derive(Debug, Clone, Serialize)]
pub struct SignalSnapshot {
    pub lead_lag_alpha_bps: f64,
    pub lead_lag_mi: f64,
    pub informed_flow: f64,
    pub buy_pressure: f64,
    pub regime_volatility: f64,
    pub vpin: f64,
    pub pre_fill_toxicity: f64,
}

impl Default for SignalSnapshot {
    fn default() -> Self {
        Self {
            lead_lag_alpha_bps: 0.0,
            lead_lag_mi: 0.0,
            informed_flow: 0.0,
            buy_pressure: 0.0,
            regime_volatility: 0.0,
            vpin: 0.0,
            pre_fill_toxicity: 0.0,
        }
    }
}

/// Market microstructure snapshot.
#[derive(Debug, Clone, Serialize)]
pub struct MarketSnapshot {
    pub best_bid: f64,
    pub best_ask: f64,
    pub bid_depth_usd: f64,
    pub ask_depth_usd: f64,
    pub last_trade_price: f64,
    pub volume_24h_usd: f64,
    pub data_age_ms: u64,
}

impl Default for MarketSnapshot {
    fn default() -> Self {
        Self {
            best_bid: 0.0,
            best_ask: 0.0,
            bid_depth_usd: 0.0,
            ask_depth_usd: 0.0,
            last_trade_price: 0.0,
            volume_24h_usd: 0.0,
            data_age_ms: 0,
        }
    }
}

/// Risk monitor states at time of kill.
#[derive(Debug, Clone, Serialize)]
pub struct RiskSnapshot {
    pub drawdown_pct: f64,
    pub margin_utilization_pct: f64,
    pub position_utilization_pct: f64,
    pub rate_limit_errors: u32,
    pub kill_switch_reasons: Vec<String>,
}

impl Default for RiskSnapshot {
    fn default() -> Self {
        Self {
            drawdown_pct: 0.0,
            margin_utilization_pct: 0.0,
            position_utilization_pct: 0.0,
            rate_limit_errors: 0,
            kill_switch_reasons: Vec::new(),
        }
    }
}

impl PostMortemDump {
    /// Write the dump to a JSON file.
    ///
    /// File is written to `{output_dir}/postmortem_{timestamp}.json`.
    /// Creates the output directory if it doesn't exist.
    ///
    /// Returns the path of the written file, or an error description.
    pub fn write_to_dir(&self, output_dir: &Path) -> Result<PathBuf, String> {
        // Create directory if needed
        std::fs::create_dir_all(output_dir)
            .map_err(|e| format!("Failed to create postmortem dir: {e}"))?;

        // Generate filename from timestamp
        let safe_ts = self.timestamp.replace([':', '.'], "-");
        let filename = format!("postmortem_{safe_ts}.json");
        let path = output_dir.join(filename);

        // Serialize and write
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| format!("Failed to serialize postmortem: {e}"))?;

        std::fs::write(&path, json)
            .map_err(|e| format!("Failed to write postmortem to {path:?}: {e}"))?;

        Ok(path)
    }

    /// Create a minimal dump with just trigger info.
    ///
    /// Use this as a starting point and fill in additional fields.
    pub fn new(trigger: String, trigger_value: f64, threshold: f64) -> Self {
        let timestamp = chrono_timestamp();
        Self {
            timestamp,
            trigger,
            trigger_value,
            threshold,
            position: 0.0,
            unrealized_pnl: 0.0,
            realized_pnl: 0.0,
            daily_pnl: 0.0,
            regime: 0,
            cascade_severity: 0.0,
            mid_price: 0.0,
            spread_bps: 0.0,
            recent_fills: Vec::new(),
            signal_state: SignalSnapshot::default(),
            market_state: MarketSnapshot::default(),
            risk_state: RiskSnapshot::default(),
        }
    }
}

/// ISO 8601 timestamp string.
fn chrono_timestamp() -> String {
    use std::time::SystemTime;
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs();
    let millis = now.subsec_millis();

    // Manual ISO 8601 formatting (avoids chrono dependency)
    // Days since epoch
    let days = secs / 86400;
    let rem = secs % 86400;
    let hours = rem / 3600;
    let mins = (rem % 3600) / 60;
    let s = rem % 60;

    // Approximate date from days since 1970-01-01
    // Using a simplified algorithm
    let (year, month, day) = days_to_date(days);

    format!("{year:04}-{month:02}-{day:02}T{hours:02}:{mins:02}:{s:02}.{millis:03}Z")
}

/// Convert days since Unix epoch to (year, month, day).
fn days_to_date(days: u64) -> (u64, u64, u64) {
    // Rata Die algorithm adapted for Unix epoch
    let z = days + 719468;
    let era = z / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if m <= 2 { y + 1 } else { y };
    (year, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_new_dump() {
        let dump = PostMortemDump::new("MaxLoss".to_string(), 150.0, 50.0);
        assert_eq!(dump.trigger, "MaxLoss");
        assert_eq!(dump.trigger_value, 150.0);
        assert_eq!(dump.threshold, 50.0);
        assert!(!dump.timestamp.is_empty());
    }

    #[test]
    fn test_serialize_dump() {
        let dump = PostMortemDump::new("StaleData".to_string(), 60.0, 10.0);
        let json = serde_json::to_string_pretty(&dump).unwrap();
        assert!(json.contains("StaleData"));
        assert!(json.contains("60"));
    }

    #[test]
    fn test_write_to_dir() {
        let dir = PathBuf::from("/tmp/claude-test-postmortem");
        let _ = std::fs::remove_dir_all(&dir);

        let mut dump = PostMortemDump::new("MaxDrawdown".to_string(), 0.025, 0.02);
        dump.position = 0.005;
        dump.daily_pnl = -45.0;
        dump.mid_price = 100_000.0;
        dump.recent_fills.push(FillRecord {
            timestamp_ms: 1000,
            side: "buy".to_string(),
            price: 99_950.0,
            size: 0.001,
            fee: 0.15,
        });

        let path = dump.write_to_dir(&dir).unwrap();
        assert!(path.exists());

        // Read back and verify
        let contents = std::fs::read_to_string(&path).unwrap();
        assert!(contents.contains("MaxDrawdown"));
        assert!(contents.contains("99950"));

        // Cleanup
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_chrono_timestamp_format() {
        let ts = chrono_timestamp();
        // Should be ISO 8601 format: YYYY-MM-DDTHH:MM:SS.mmmZ
        assert!(ts.contains('T'));
        assert!(ts.ends_with('Z'));
        assert_eq!(ts.len(), 24); // "2026-02-07T12:34:56.789Z"
    }

    #[test]
    fn test_days_to_date() {
        // 2026-02-07 = day 20,491 since 1970-01-01
        // Let's test a known date
        let (y, m, d) = days_to_date(0); // 1970-01-01
        assert_eq!((y, m, d), (1970, 1, 1));

        let (y, m, d) = days_to_date(365); // 1971-01-01
        assert_eq!((y, m, d), (1971, 1, 1));
    }
}
