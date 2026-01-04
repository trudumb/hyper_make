//! Rate Limit Shadow Pricing.
//!
//! Treats rate limit tokens as a currency with a shadow price. As tokens
//! deplete, the "cost" of using them rises exponentially, naturally
//! prioritizing high-importance actions.
//!
//! # Model
//!
//! ```text
//! shadow_cost = (max_tokens / current_tokens)^exponent
//! ```
//!
//! - At full capacity: cost = 1.0
//! - At 50% capacity: cost = 2^exponent = 4.0 (with exponent=2)
//! - At 25% capacity: cost = 4^exponent = 16.0
//!
//! Actions are only executed if their importance exceeds the shadow cost.

use std::time::Instant;

use super::super::config::RateLimitConfig;

/// Rate limit shadow pricing calculator.
#[derive(Debug)]
pub struct RateLimitShadowPrice {
    /// Configuration
    config: RateLimitConfig,

    /// Current token count
    tokens: f64,

    /// Last update time
    last_update: Instant,

    /// Tokens used in current period (for metrics)
    tokens_used_period: u64,

    /// Period start (for metrics)
    period_start: Instant,
}

impl RateLimitShadowPrice {
    /// Create a new rate limit tracker.
    pub fn new(config: RateLimitConfig) -> Self {
        let now = Instant::now();
        Self {
            tokens: config.max_tokens as f64,
            config,
            last_update: now,
            tokens_used_period: 0,
            period_start: now,
        }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(RateLimitConfig::default())
    }

    /// Refill tokens based on elapsed time.
    fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_update).as_secs_f64();

        // Refill at configured rate
        let refill = elapsed * self.config.refill_rate;
        self.tokens = (self.tokens + refill).min(self.config.max_tokens as f64);

        self.last_update = now;

        // Reset period metrics every minute
        if now.duration_since(self.period_start).as_secs() >= 60 {
            self.tokens_used_period = 0;
            self.period_start = now;
        }
    }

    /// Calculate the current shadow cost of using a token.
    ///
    /// ```text
    /// cost = (max_tokens / current_tokens)^exponent
    /// ```
    pub fn shadow_cost(&mut self) -> f64 {
        self.refill();

        let available = self.available_tokens();
        if available <= 0.0 {
            return f64::MAX;
        }

        let ratio = self.config.max_tokens as f64 / available;
        ratio.powf(self.config.shadow_price_exponent)
    }

    /// Available tokens (after reserve).
    pub fn available_tokens(&self) -> f64 {
        let reserve = self.config.max_tokens as f64 * self.config.reserve_ratio;
        (self.tokens - reserve).max(0.0)
    }

    /// Check if an action with given importance should be executed.
    ///
    /// Returns true if importance > shadow_cost.
    pub fn check_budget(&mut self, importance: f64) -> bool {
        if importance < self.config.min_importance {
            return false;
        }

        let cost = self.shadow_cost();
        importance > cost
    }

    /// Consume tokens for an action.
    ///
    /// Returns true if tokens were successfully consumed.
    pub fn consume(&mut self, tokens: u64) -> bool {
        self.refill();

        if self.tokens >= tokens as f64 {
            self.tokens -= tokens as f64;
            self.tokens_used_period += tokens;
            true
        } else {
            false
        }
    }

    /// Try to consume tokens for an action if importance justifies it.
    ///
    /// Returns true if action was approved and tokens consumed.
    pub fn try_consume(&mut self, importance: f64, tokens: u64) -> bool {
        if self.check_budget(importance) {
            self.consume(tokens)
        } else {
            false
        }
    }

    /// Get current token count.
    pub fn current_tokens(&mut self) -> f64 {
        self.refill();
        self.tokens
    }

    /// Get utilization ratio [0, 1].
    pub fn utilization(&mut self) -> f64 {
        self.refill();
        1.0 - (self.tokens / self.config.max_tokens as f64)
    }

    /// Get tokens used in current period.
    pub fn tokens_used_period(&self) -> u64 {
        self.tokens_used_period
    }

    /// Check if rate limited (tokens exhausted).
    pub fn is_rate_limited(&mut self) -> bool {
        self.refill();
        self.available_tokens() <= 0.0
    }

    /// Reset to full capacity.
    pub fn reset(&mut self) {
        self.tokens = self.config.max_tokens as f64;
        self.last_update = Instant::now();
    }

    /// Get time until tokens refill to a threshold.
    pub fn time_until_refill(&self, target_tokens: f64) -> f64 {
        if self.tokens >= target_tokens {
            return 0.0;
        }

        let needed = target_tokens - self.tokens;
        needed / self.config.refill_rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;
    use std::time::Duration;

    #[test]
    fn test_shadow_cost_at_full_capacity() {
        // Use config with no reserve for simple calculation
        let mut limiter = RateLimitShadowPrice::new(RateLimitConfig {
            max_tokens: 100,
            reserve_ratio: 0.0, // No reserve
            ..Default::default()
        });
        let cost = limiter.shadow_cost();
        // At full capacity with no reserve, cost should be 1.0
        // cost = (max / available)^exponent = (100/100)^2 = 1.0
        assert!((cost - 1.0).abs() < 0.1, "Expected cost ≈ 1.0, got {}", cost);
    }

    #[test]
    fn test_shadow_cost_increases_with_usage() {
        let mut limiter = RateLimitShadowPrice::new(RateLimitConfig {
            max_tokens: 100,
            reserve_ratio: 0.0,
            ..Default::default()
        });

        let cost_before = limiter.shadow_cost();
        limiter.consume(50);
        let cost_after = limiter.shadow_cost();

        assert!(cost_after > cost_before);
    }

    #[test]
    fn test_check_budget() {
        let mut limiter = RateLimitShadowPrice::new(RateLimitConfig {
            max_tokens: 100,
            reserve_ratio: 0.0,
            min_importance: 0.1,
            shadow_price_exponent: 2.0,
            ..Default::default()
        });

        // High importance should pass
        assert!(limiter.check_budget(2.0));

        // Below min importance should fail
        assert!(!limiter.check_budget(0.05));

        // Deplete tokens
        limiter.consume(90);

        // Now even moderate importance may fail
        let cost = limiter.shadow_cost();
        assert!(cost > 5.0); // Cost should be high when 90% depleted
    }

    #[test]
    fn test_consume() {
        let mut limiter = RateLimitShadowPrice::new(RateLimitConfig {
            max_tokens: 100,
            reserve_ratio: 0.0,
            ..Default::default()
        });

        assert!(limiter.consume(50));
        assert!((limiter.current_tokens() - 50.0).abs() < 0.1);

        // Can't consume more than available
        assert!(!limiter.consume(100));
    }

    #[test]
    fn test_refill() {
        let mut limiter = RateLimitShadowPrice::new(RateLimitConfig {
            max_tokens: 100,
            refill_rate: 1000.0, // 1000/s for fast test
            reserve_ratio: 0.0,
            ..Default::default()
        });

        limiter.consume(50);
        let before = limiter.current_tokens();

        // Wait a bit
        sleep(Duration::from_millis(10));

        let after = limiter.current_tokens();
        assert!(after > before);
    }

    #[test]
    fn test_reserve() {
        let mut limiter = RateLimitShadowPrice::new(RateLimitConfig {
            max_tokens: 100,
            reserve_ratio: 0.2, // 20% reserve
            ..Default::default()
        });

        // Available should be 80% of max
        let available = limiter.available_tokens();
        assert!((available - 80.0).abs() < 0.1);
    }
}
