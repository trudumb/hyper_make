//! Error classification for rate limiting.

/// Classification of order rejection error types.
///
/// Different error types require different handling strategies:
/// - `PositionLimit`: Skip side entirely until position changes (do NOT retry)
/// - `Margin`: Transient, use exponential backoff
/// - `PriceError`: May retry with adjusted price
/// - `RateLimit`: Exchange rate limit hit - triggers kill switch counter
/// - `Other`: Unknown error, use default backoff
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RejectionErrorType {
    /// Position limit exceeded (PerpMaxPosition, position exceeded, etc.)
    /// Should NOT retry - skip side until position changes.
    PositionLimit,
    /// Margin or leverage related (PerpMargin, leverage limit, etc.)
    /// Transient, use exponential backoff.
    Margin,
    /// Price-related error (BadAloPx - price outside spread)
    /// May retry immediately with adjusted price.
    PriceError,
    /// Rate limit exceeded (429, too many requests, rate limit, etc.)
    /// Triggers kill switch counter - must back off aggressively.
    RateLimit,
    /// Unknown error type - use default handling.
    Other,
}

impl RejectionErrorType {
    /// Classify an error message into a rejection type.
    ///
    /// # Arguments
    /// - `error`: The error message from the exchange
    ///
    /// # Returns
    /// The classified error type
    pub fn classify(error: &str) -> Self {
        let error_lower = error.to_lowercase();

        // Rate limit errors - must detect FIRST (before other patterns)
        // These trigger the kill switch counter and require aggressive backoff.
        if error_lower.contains("rate limit")
            || error_lower.contains("rate_limit")
            || error_lower.contains("too many requests")
            || error_lower.contains("429")
            || error_lower.contains("throttl")
        {
            return RejectionErrorType::RateLimit;
        }

        // Position limit errors - these should skip the side, not backoff
        if error_lower.contains("perpmaxposition")
            || error_lower.contains("max position")
            || error_lower.contains("position size")
            || error_lower.contains("exceed")
                && (error_lower.contains("position") || error_lower.contains("size"))
        {
            return RejectionErrorType::PositionLimit;
        }

        // Margin/leverage errors - transient, use backoff
        if error_lower.contains("perpmargin")
            || error_lower.contains("margin")
            || error_lower.contains("leverage")
            || error_lower.contains("insufficient")
        {
            return RejectionErrorType::Margin;
        }

        // Price errors - can retry with adjusted price
        if error_lower.contains("badalopx")
            || error_lower.contains("price")
            || error_lower.contains("px")
        {
            return RejectionErrorType::PriceError;
        }

        RejectionErrorType::Other
    }

    /// Returns true if this error type should trigger position/side skipping.
    pub fn should_skip_side(&self) -> bool {
        matches!(self, RejectionErrorType::PositionLimit)
    }

    /// Returns true if this error type should use exponential backoff.
    pub fn should_backoff(&self) -> bool {
        matches!(
            self,
            RejectionErrorType::PositionLimit
                | RejectionErrorType::Margin
                | RejectionErrorType::RateLimit
                | RejectionErrorType::Other
        )
    }

    /// Returns true if this error type is transient and may resolve on its own.
    pub fn is_transient(&self) -> bool {
        matches!(
            self,
            RejectionErrorType::Margin | RejectionErrorType::PriceError
        )
    }

    /// Returns true if this is a rate limit error that should trigger
    /// the kill switch counter.
    pub fn is_rate_limit(&self) -> bool {
        matches!(self, RejectionErrorType::RateLimit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_classification_position_limit() {
        // PerpMaxPosition error
        let err_type = RejectionErrorType::classify("PerpMaxPosition: Order would exceed maximum");
        assert_eq!(err_type, RejectionErrorType::PositionLimit);
        assert!(err_type.should_skip_side());
        assert!(err_type.should_backoff());

        // Position exceeded
        let err_type = RejectionErrorType::classify("position exceeded for current leverage");
        assert_eq!(err_type, RejectionErrorType::PositionLimit);

        // Max position
        let err_type = RejectionErrorType::classify("Order would exceed max position size");
        assert_eq!(err_type, RejectionErrorType::PositionLimit);
    }

    #[test]
    fn test_error_classification_margin() {
        // PerpMargin error
        let err_type = RejectionErrorType::classify("PerpMargin: Insufficient margin");
        assert_eq!(err_type, RejectionErrorType::Margin);
        assert!(!err_type.should_skip_side());
        assert!(err_type.should_backoff());
        assert!(err_type.is_transient());

        // Leverage error
        let err_type = RejectionErrorType::classify("Max leverage at current position is 25x");
        assert_eq!(err_type, RejectionErrorType::Margin);

        // Insufficient funds
        let err_type = RejectionErrorType::classify("Insufficient balance for order");
        assert_eq!(err_type, RejectionErrorType::Margin);
    }

    #[test]
    fn test_error_classification_price_error() {
        // BadAloPx error
        let err_type = RejectionErrorType::classify("BadAloPx: Price too aggressive");
        assert_eq!(err_type, RejectionErrorType::PriceError);
        assert!(!err_type.should_skip_side());
        assert!(!err_type.should_backoff());
        assert!(err_type.is_transient());

        // Generic price error
        let err_type = RejectionErrorType::classify("Invalid price specified");
        assert_eq!(err_type, RejectionErrorType::PriceError);
    }

    #[test]
    fn test_error_classification_rate_limit() {
        // Explicit rate limit
        let err_type = RejectionErrorType::classify("Rate limit exceeded");
        assert_eq!(err_type, RejectionErrorType::RateLimit);
        assert!(err_type.is_rate_limit());
        assert!(err_type.should_backoff());
        assert!(!err_type.should_skip_side());

        // HTTP 429
        let err_type = RejectionErrorType::classify("HTTP 429: Too Many Requests");
        assert_eq!(err_type, RejectionErrorType::RateLimit);

        // rate_limit with underscore
        let err_type = RejectionErrorType::classify("rate_limit: slow down");
        assert_eq!(err_type, RejectionErrorType::RateLimit);

        // Throttle variant
        let err_type = RejectionErrorType::classify("Request throttled, try again later");
        assert_eq!(err_type, RejectionErrorType::RateLimit);

        // "too many requests" lowercase
        let err_type = RejectionErrorType::classify("too many requests from this IP");
        assert_eq!(err_type, RejectionErrorType::RateLimit);
    }

    #[test]
    fn test_error_classification_other() {
        // Unknown error
        let err_type = RejectionErrorType::classify("Some unknown error occurred");
        assert_eq!(err_type, RejectionErrorType::Other);
        assert!(!err_type.should_skip_side());
        assert!(err_type.should_backoff());
        assert!(!err_type.is_transient());
        assert!(!err_type.is_rate_limit());
    }
}
