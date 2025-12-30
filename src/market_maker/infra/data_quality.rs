//! Data Quality Monitoring
//!
//! Validates incoming market data to detect and handle:
//! - Sequence gaps (message loss)
//! - Timestamp regression
//! - Stale data
//! - Crossed books
//! - Invalid prices/sizes

use std::collections::HashMap;
use std::fmt;
use std::time::Instant;

/// Configuration for data quality checks.
#[derive(Debug, Clone)]
pub struct DataQualityConfig {
    /// Maximum age of data before considered stale (ms)
    pub max_data_age_ms: u64,
    /// Maximum allowed sequence gap
    pub max_sequence_gap: u64,
    /// Maximum price deviation from mid (fractional, e.g., 0.20 = 20%)
    pub max_price_deviation_pct: f64,
    /// Whether to check for crossed books
    pub check_crossed_books: bool,
}

impl Default for DataQualityConfig {
    fn default() -> Self {
        Self {
            max_data_age_ms: 30_000,
            max_sequence_gap: 10,
            max_price_deviation_pct: 0.20,
            check_crossed_books: true,
        }
    }
}

/// Types of data quality anomalies.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AnomalyType {
    /// Gap in sequence numbers (detected gap size)
    SequenceGap(u64),
    /// Timestamp went backwards
    TimestampRegression,
    /// Data is too old
    StaleData,
    /// Best bid >= best ask
    CrossedBook,
    /// Price deviates too far from reference
    InvalidPrice,
    /// Size is invalid (negative, zero, or too large)
    InvalidSize,
}

impl fmt::Display for AnomalyType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AnomalyType::SequenceGap(gap) => write!(f, "sequence_gap_{}", gap),
            AnomalyType::TimestampRegression => write!(f, "timestamp_regression"),
            AnomalyType::StaleData => write!(f, "stale_data"),
            AnomalyType::CrossedBook => write!(f, "crossed_book"),
            AnomalyType::InvalidPrice => write!(f, "invalid_price"),
            AnomalyType::InvalidSize => write!(f, "invalid_size"),
        }
    }
}

/// Data quality monitoring and validation.
#[derive(Debug)]
pub struct DataQualityMonitor {
    /// Last seen sequence number per symbol
    last_sequences: HashMap<String, u64>,
    /// Last seen timestamp per symbol (ms)
    last_timestamps: HashMap<String, u64>,
    /// Count of each anomaly type
    anomaly_counts: HashMap<AnomalyType, u64>,
    /// Last update time per symbol
    last_update_times: HashMap<String, Instant>,
    /// Total sequence gaps detected
    total_sequence_gaps: u64,
    /// Total crossed book incidents
    total_crossed_books: u64,
    /// Configuration
    config: DataQualityConfig,
}

impl DataQualityMonitor {
    /// Create a new data quality monitor with given config.
    pub fn new(config: DataQualityConfig) -> Self {
        Self {
            last_sequences: HashMap::new(),
            last_timestamps: HashMap::new(),
            anomaly_counts: HashMap::new(),
            last_update_times: HashMap::new(),
            total_sequence_gaps: 0,
            total_crossed_books: 0,
            config,
        }
    }

    /// Check a trade for data quality issues.
    ///
    /// # Arguments
    /// * `symbol` - Trading symbol
    /// * `seq` - Sequence number (if available, 0 to skip check)
    /// * `ts_ms` - Timestamp in milliseconds
    /// * `price` - Trade price
    /// * `size` - Trade size
    /// * `mid` - Current mid price for deviation check
    ///
    /// # Returns
    /// Ok(()) if valid, Err(AnomalyType) for the first detected issue
    pub fn check_trade(
        &mut self,
        symbol: &str,
        seq: u64,
        ts_ms: u64,
        price: f64,
        size: f64,
        mid: f64,
    ) -> Result<(), AnomalyType> {
        // Check sequence gap (skip if seq is 0)
        if seq > 0 {
            self.check_sequence(symbol, seq)?;
        }

        // Check timestamp
        self.check_timestamp(symbol, ts_ms)?;

        // Check price validity
        self.validate_price(price, mid)?;

        // Check size validity
        self.validate_size(size)?;

        // Update last update time
        self.last_update_times
            .insert(symbol.to_string(), Instant::now());

        Ok(())
    }

    /// Check an L2 book update for data quality issues.
    ///
    /// # Arguments
    /// * `symbol` - Trading symbol
    /// * `seq` - Sequence number (if available, 0 to skip check)
    /// * `best_bid` - Best bid price
    /// * `best_ask` - Best ask price
    ///
    /// # Returns
    /// Ok(()) if valid, Err(AnomalyType) for the first detected issue
    pub fn check_l2_book(
        &mut self,
        symbol: &str,
        seq: u64,
        best_bid: f64,
        best_ask: f64,
    ) -> Result<(), AnomalyType> {
        // Check sequence gap (skip if seq is 0)
        if seq > 0 {
            self.check_sequence(symbol, seq)?;
        }

        // Check for crossed book
        if self.config.check_crossed_books && Self::is_crossed(best_bid, best_ask) {
            self.record_anomaly(AnomalyType::CrossedBook);
            self.total_crossed_books += 1;
            return Err(AnomalyType::CrossedBook);
        }

        // Update last update time
        self.last_update_times
            .insert(symbol.to_string(), Instant::now());

        Ok(())
    }

    /// Check if a book is crossed (best bid >= best ask).
    #[inline]
    pub fn is_crossed(best_bid: f64, best_ask: f64) -> bool {
        best_bid >= best_ask
    }

    /// Validate a price against the reference mid price.
    pub fn validate_price(&mut self, price: f64, mid: f64) -> Result<f64, AnomalyType> {
        if price <= 0.0 || !price.is_finite() {
            self.record_anomaly(AnomalyType::InvalidPrice);
            return Err(AnomalyType::InvalidPrice);
        }

        if mid > 0.0 {
            let deviation = (price - mid).abs() / mid;
            if deviation > self.config.max_price_deviation_pct {
                self.record_anomaly(AnomalyType::InvalidPrice);
                return Err(AnomalyType::InvalidPrice);
            }
        }

        Ok(price)
    }

    /// Validate a size value.
    pub fn validate_size(&mut self, size: f64) -> Result<f64, AnomalyType> {
        if size <= 0.0 || !size.is_finite() {
            self.record_anomaly(AnomalyType::InvalidSize);
            return Err(AnomalyType::InvalidSize);
        }
        Ok(size)
    }

    /// Check if data for a symbol is stale.
    pub fn is_stale(&self, symbol: &str) -> bool {
        if let Some(last_time) = self.last_update_times.get(symbol) {
            let age_ms = last_time.elapsed().as_millis() as u64;
            age_ms > self.config.max_data_age_ms
        } else {
            // Never received data = stale
            true
        }
    }

    /// Check for staleness and record anomaly if stale.
    pub fn check_staleness(&mut self, symbol: &str) -> Result<(), AnomalyType> {
        if self.is_stale(symbol) {
            self.record_anomaly(AnomalyType::StaleData);
            Err(AnomalyType::StaleData)
        } else {
            Ok(())
        }
    }

    /// Get the count of a specific anomaly type.
    pub fn anomaly_count(&self, anomaly: &AnomalyType) -> u64 {
        *self.anomaly_counts.get(anomaly).unwrap_or(&0)
    }

    /// Get all anomaly counts.
    pub fn anomaly_summary(&self) -> &HashMap<AnomalyType, u64> {
        &self.anomaly_counts
    }

    /// Get total number of sequence gaps detected.
    pub fn total_sequence_gaps(&self) -> u64 {
        self.total_sequence_gaps
    }

    /// Get total number of crossed book incidents.
    pub fn total_crossed_books(&self) -> u64 {
        self.total_crossed_books
    }

    /// Get total anomaly count across all types.
    pub fn total_anomalies(&self) -> u64 {
        self.anomaly_counts.values().sum()
    }

    /// Get time since last update for a symbol.
    pub fn time_since_last_update(&self, symbol: &str) -> Option<u64> {
        self.last_update_times
            .get(symbol)
            .map(|t| t.elapsed().as_millis() as u64)
    }

    /// Reset all counters.
    pub fn reset(&mut self) {
        self.last_sequences.clear();
        self.last_timestamps.clear();
        self.anomaly_counts.clear();
        self.last_update_times.clear();
        self.total_sequence_gaps = 0;
        self.total_crossed_books = 0;
    }

    // === Private Methods ===

    fn check_sequence(&mut self, symbol: &str, seq: u64) -> Result<(), AnomalyType> {
        if let Some(&last_seq) = self.last_sequences.get(symbol) {
            if seq <= last_seq {
                // Sequence not advancing - could be duplicate or out of order
                // We don't flag this as error, just skip update
                return Ok(());
            }

            let gap = seq - last_seq - 1;
            if gap > 0 && gap > self.config.max_sequence_gap {
                let anomaly = AnomalyType::SequenceGap(gap);
                self.record_anomaly(anomaly.clone());
                self.total_sequence_gaps += gap;
                self.last_sequences.insert(symbol.to_string(), seq);
                return Err(anomaly);
            }
        }

        self.last_sequences.insert(symbol.to_string(), seq);
        Ok(())
    }

    fn check_timestamp(&mut self, symbol: &str, ts_ms: u64) -> Result<(), AnomalyType> {
        if let Some(&last_ts) = self.last_timestamps.get(symbol) {
            if ts_ms < last_ts {
                self.record_anomaly(AnomalyType::TimestampRegression);
                // Still update timestamp to recover
                self.last_timestamps.insert(symbol.to_string(), ts_ms);
                return Err(AnomalyType::TimestampRegression);
            }
        }

        self.last_timestamps.insert(symbol.to_string(), ts_ms);
        Ok(())
    }

    fn record_anomaly(&mut self, anomaly: AnomalyType) {
        *self.anomaly_counts.entry(anomaly).or_insert(0) += 1;
    }
}

impl Default for DataQualityMonitor {
    fn default() -> Self {
        Self::new(DataQualityConfig::default())
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
        let config = DataQualityConfig::default();
        assert_eq!(config.max_data_age_ms, 30_000);
        assert_eq!(config.max_sequence_gap, 10);
        assert_eq!(config.max_price_deviation_pct, 0.20);
        assert!(config.check_crossed_books);
    }

    #[test]
    fn test_crossed_book_detection() {
        assert!(DataQualityMonitor::is_crossed(100.0, 99.0));
        assert!(DataQualityMonitor::is_crossed(100.0, 100.0));
        assert!(!DataQualityMonitor::is_crossed(99.0, 100.0));
    }

    #[test]
    fn test_l2_book_crossed() {
        let mut monitor = DataQualityMonitor::default();

        // Valid book
        assert!(monitor.check_l2_book("BTC", 0, 99.0, 100.0).is_ok());

        // Crossed book
        let result = monitor.check_l2_book("BTC", 0, 100.0, 99.0);
        assert!(matches!(result, Err(AnomalyType::CrossedBook)));
        assert_eq!(monitor.total_crossed_books(), 1);
    }

    #[test]
    fn test_sequence_gap_detection() {
        let mut monitor = DataQualityMonitor::default();

        // First message
        assert!(monitor.check_l2_book("BTC", 1, 99.0, 100.0).is_ok());

        // Normal increment
        assert!(monitor.check_l2_book("BTC", 2, 99.0, 100.0).is_ok());

        // Small gap (within tolerance)
        assert!(monitor.check_l2_book("BTC", 5, 99.0, 100.0).is_ok());

        // Large gap (exceeds max_sequence_gap of 10)
        let result = monitor.check_l2_book("BTC", 100, 99.0, 100.0);
        assert!(matches!(result, Err(AnomalyType::SequenceGap(_))));
    }

    #[test]
    fn test_price_validation() {
        let mut monitor = DataQualityMonitor::default();

        // Valid price
        assert!(monitor.validate_price(100.0, 100.0).is_ok());

        // Price within 20% deviation
        assert!(monitor.validate_price(115.0, 100.0).is_ok());

        // Price exceeds 20% deviation
        assert!(monitor.validate_price(150.0, 100.0).is_err());

        // Invalid prices
        assert!(monitor.validate_price(-1.0, 100.0).is_err());
        assert!(monitor.validate_price(0.0, 100.0).is_err());
        assert!(monitor.validate_price(f64::NAN, 100.0).is_err());
    }

    #[test]
    fn test_size_validation() {
        let mut monitor = DataQualityMonitor::default();

        assert!(monitor.validate_size(1.0).is_ok());
        assert!(monitor.validate_size(0.001).is_ok());
        assert!(monitor.validate_size(0.0).is_err());
        assert!(monitor.validate_size(-1.0).is_err());
    }

    #[test]
    fn test_trade_check() {
        let mut monitor = DataQualityMonitor::default();

        // Valid trade
        assert!(monitor
            .check_trade("BTC", 1, 1000, 50000.0, 1.0, 50000.0)
            .is_ok());

        // Invalid price
        assert!(monitor
            .check_trade("BTC", 2, 1001, -1.0, 1.0, 50000.0)
            .is_err());

        // Invalid size
        assert!(monitor
            .check_trade("BTC", 3, 1002, 50000.0, 0.0, 50000.0)
            .is_err());
    }

    #[test]
    fn test_timestamp_regression() {
        let mut monitor = DataQualityMonitor::default();

        // First timestamp
        assert!(monitor
            .check_trade("BTC", 0, 1000, 100.0, 1.0, 100.0)
            .is_ok());

        // Later timestamp
        assert!(monitor
            .check_trade("BTC", 0, 2000, 100.0, 1.0, 100.0)
            .is_ok());

        // Earlier timestamp (regression)
        let result = monitor.check_trade("BTC", 0, 1500, 100.0, 1.0, 100.0);
        assert!(matches!(result, Err(AnomalyType::TimestampRegression)));
    }

    #[test]
    fn test_anomaly_counting() {
        let mut monitor = DataQualityMonitor::default();

        // Trigger some anomalies
        let _ = monitor.validate_price(-1.0, 100.0);
        let _ = monitor.validate_price(-2.0, 100.0);
        let _ = monitor.validate_size(0.0);

        assert_eq!(monitor.anomaly_count(&AnomalyType::InvalidPrice), 2);
        assert_eq!(monitor.anomaly_count(&AnomalyType::InvalidSize), 1);
        assert_eq!(monitor.total_anomalies(), 3);
    }

    #[test]
    fn test_staleness_check() {
        let mut monitor = DataQualityMonitor::new(DataQualityConfig {
            max_data_age_ms: 100, // 100ms for testing
            ..Default::default()
        });

        // No data yet = stale
        assert!(monitor.is_stale("BTC"));

        // Receive data
        let _ = monitor.check_l2_book("BTC", 0, 99.0, 100.0);
        assert!(!monitor.is_stale("BTC"));

        // Wait for staleness (in real tests this would use mock time)
        std::thread::sleep(std::time::Duration::from_millis(150));
        assert!(monitor.is_stale("BTC"));
    }

    #[test]
    fn test_reset() {
        let mut monitor = DataQualityMonitor::default();

        // Add some data
        let _ = monitor.check_l2_book("BTC", 1, 100.0, 99.0); // crossed
        let _ = monitor.validate_price(-1.0, 100.0);

        assert!(monitor.total_anomalies() > 0);

        // Reset
        monitor.reset();

        assert_eq!(monitor.total_anomalies(), 0);
        assert_eq!(monitor.total_crossed_books(), 0);
        assert_eq!(monitor.total_sequence_gaps(), 0);
    }

    #[test]
    fn test_anomaly_display() {
        assert_eq!(format!("{}", AnomalyType::SequenceGap(5)), "sequence_gap_5");
        assert_eq!(format!("{}", AnomalyType::CrossedBook), "crossed_book");
        assert_eq!(format!("{}", AnomalyType::StaleData), "stale_data");
    }
}
