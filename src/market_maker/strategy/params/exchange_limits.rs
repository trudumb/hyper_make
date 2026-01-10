//! Exchange limits parameter set.

/// Exchange position limit parameters (from active_asset_data API).
///
/// Used to prevent order rejections due to exchange-enforced position limits.
#[derive(Debug, Clone, Copy)]
pub struct ExchangeLimitsParams {
    /// Whether exchange limits have been fetched and are valid.
    pub valid: bool,

    /// Effective bid limit: min(local_max, exchange_available_buy).
    /// Use this to cap bid ladder sizes.
    pub effective_bid_limit: f64,

    /// Effective ask limit: min(local_max, exchange_available_sell).
    /// Use this to cap ask ladder sizes.
    pub effective_ask_limit: f64,

    /// Age of exchange limits in milliseconds.
    /// > 120,000 (2 min) = stale, consider reducing sizes.
    /// > 300,000 (5 min) = critically stale, pause quoting.
    pub age_ms: u64,
}

impl Default for ExchangeLimitsParams {
    fn default() -> Self {
        Self {
            valid: false,
            effective_bid_limit: f64::MAX,
            effective_ask_limit: f64::MAX,
            age_ms: u64::MAX,
        }
    }
}

impl ExchangeLimitsParams {
    /// Check if limits are stale (> 2 minutes old).
    pub fn is_stale(&self) -> bool {
        self.age_ms > 120_000
    }

    /// Check if limits are critically stale (> 5 minutes old).
    pub fn is_critically_stale(&self) -> bool {
        self.age_ms > 300_000
    }

    /// Get effective bid limit with staleness factor applied.
    pub fn safe_bid_limit(&self) -> f64 {
        if !self.valid {
            return f64::MAX;
        }
        match self.age_ms {
            0..=120_000 => self.effective_bid_limit,
            120_001..=300_000 => self.effective_bid_limit * 0.5,
            _ => 0.0,
        }
    }

    /// Get effective ask limit with staleness factor applied.
    pub fn safe_ask_limit(&self) -> f64 {
        if !self.valid {
            return f64::MAX;
        }
        match self.age_ms {
            0..=120_000 => self.effective_ask_limit,
            120_001..=300_000 => self.effective_ask_limit * 0.5,
            _ => 0.0,
        }
    }
}
