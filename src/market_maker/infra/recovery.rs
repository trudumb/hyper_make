//! Recovery State Machine for Stuck Reduce-Only Mode (Phase 3 Fix)
//!
//! When the market maker enters reduce-only mode but cannot place orders
//! to reduce position (exchange has no capacity), this module manages
//! the escalation process:
//!
//! 1. Normal: Regular quoting
//! 2. ReduceOnlyStuck: Limit orders rejected 3+ times
//! 3. IocRecovery: Using IOC orders to reduce position
//! 4. Cooldown: Waiting between recovery attempts
//!
//! # Design
//!
//! - Tracks consecutive rejections per side
//! - Escalates to IOC orders after threshold
//! - Implements backoff to avoid rate limiting
//! - Emits metrics for monitoring

use std::time::{Duration, Instant};

/// Configuration for the recovery state machine.
#[derive(Debug, Clone)]
pub struct RecoveryConfig {
    /// Number of consecutive rejections before escalating to IOC
    pub rejection_threshold: u32,
    /// Slippage allowed for IOC orders (basis points)
    pub ioc_slippage_bps: u32,
    /// Maximum IOC attempts before giving up
    pub max_ioc_attempts: u32,
    /// Cooldown duration between IOC attempts
    pub ioc_cooldown: Duration,
    /// Cooldown duration after max IOC attempts exhausted
    pub stuck_cooldown: Duration,
    /// Minimum position size to attempt IOC recovery
    pub min_ioc_size: f64,
}

impl Default for RecoveryConfig {
    fn default() -> Self {
        Self {
            rejection_threshold: 3,
            ioc_slippage_bps: 50, // 0.5%
            max_ioc_attempts: 3,
            ioc_cooldown: Duration::from_secs(5),
            stuck_cooldown: Duration::from_secs(30),
            min_ioc_size: 0.001,
        }
    }
}

/// Current state of the recovery process.
#[derive(Debug, Clone)]
pub enum RecoveryState {
    /// Normal operation - no recovery needed
    Normal,
    /// Reduce-only mode, experiencing rejections
    ReduceOnlyStuck {
        /// Consecutive rejections on reducing side
        consecutive_rejections: u32,
        /// Which side is trying to reduce (true = buy to reduce short)
        is_buy_side: bool,
        /// When the first rejection occurred
        first_rejection: Instant,
    },
    /// Attempting IOC recovery
    IocRecovery {
        /// Number of IOC attempts made
        attempts: u32,
        /// Which side we're reducing (true = buy to reduce short)
        is_buy_side: bool,
        /// When the last IOC was sent
        last_attempt: Instant,
        /// Size filled so far through IOC
        filled_so_far: f64,
    },
    /// Cooling down after failed recovery
    Cooldown {
        /// When cooldown ends
        until: Instant,
        /// Why we're in cooldown
        reason: CooldownReason,
    },
}

/// Reason for entering cooldown state.
#[derive(Debug, Clone, Copy)]
pub enum CooldownReason {
    /// IOC attempts exhausted
    IocExhausted,
    /// Waiting between IOC attempts
    BetweenAttempts,
    /// Position recovered, cool down before normal ops
    RecoverySucceeded,
}

impl RecoveryState {
    /// Check if we're in a non-normal state.
    pub fn is_active(&self) -> bool {
        !matches!(self, RecoveryState::Normal)
    }

    /// Get the current state name for logging.
    pub fn state_name(&self) -> &'static str {
        match self {
            RecoveryState::Normal => "Normal",
            RecoveryState::ReduceOnlyStuck { .. } => "ReduceOnlyStuck",
            RecoveryState::IocRecovery { .. } => "IocRecovery",
            RecoveryState::Cooldown { .. } => "Cooldown",
        }
    }
}

/// Recovery state machine manager.
#[derive(Debug)]
pub struct RecoveryManager {
    state: RecoveryState,
    config: RecoveryConfig,
}

impl Default for RecoveryManager {
    fn default() -> Self {
        Self::new()
    }
}

impl RecoveryManager {
    /// Create a new recovery manager with default configuration.
    pub fn new() -> Self {
        Self {
            state: RecoveryState::Normal,
            config: RecoveryConfig::default(),
        }
    }

    /// Create a new recovery manager with custom configuration.
    pub fn with_config(config: RecoveryConfig) -> Self {
        Self {
            state: RecoveryState::Normal,
            config,
        }
    }

    /// Get current state.
    pub fn state(&self) -> &RecoveryState {
        &self.state
    }

    /// Get configuration.
    pub fn config(&self) -> &RecoveryConfig {
        &self.config
    }

    /// Reset to normal state.
    pub fn reset(&mut self) {
        self.state = RecoveryState::Normal;
    }

    /// Record a successful order placement (resets rejection counter).
    pub fn record_success(&mut self) {
        let transition = match &self.state {
            RecoveryState::ReduceOnlyStuck { .. } => {
                // Successful order while stuck - likely position reducing naturally
                Some((RecoveryState::Normal, None))
            }
            RecoveryState::IocRecovery { is_buy_side, .. } => {
                // IOC succeeded - cool down briefly then resume
                let is_buy = *is_buy_side;
                Some((
                    RecoveryState::Cooldown {
                        until: Instant::now() + Duration::from_secs(2),
                        reason: CooldownReason::RecoverySucceeded,
                    },
                    Some(is_buy),
                ))
            }
            _ => None,
        };

        if let Some((new_state, log_info)) = transition {
            self.state = new_state;
            if let Some(is_buy_side) = log_info {
                tracing::info!(
                    is_buy_side = is_buy_side,
                    "IOC recovery succeeded, entering brief cooldown"
                );
            }
        }
    }

    /// Record an IOC fill (partial or full).
    pub fn record_ioc_fill(&mut self, size: f64) {
        if let RecoveryState::IocRecovery {
            filled_so_far,
            is_buy_side,
            attempts,
            ..
        } = &mut self.state
        {
            *filled_so_far += size;
            tracing::info!(
                size = %format!("{:.6}", size),
                total_filled = %format!("{:.6}", *filled_so_far),
                is_buy_side = *is_buy_side,
                attempts = *attempts,
                "IOC recovery fill recorded"
            );
        }
    }

    /// Record an order rejection.
    ///
    /// # Arguments
    /// - `is_buy`: Whether the rejected order was a buy
    /// - `error`: Error message from exchange
    ///
    /// # Returns
    /// `true` if this rejection triggers escalation to IOC recovery
    pub fn record_rejection(&mut self, is_buy: bool, error: &str) -> bool {
        // Only track position-related rejections
        if !error.contains("position") && !error.contains("exceed") {
            return false;
        }

        match &mut self.state {
            RecoveryState::Normal => {
                // First rejection - enter stuck state
                self.state = RecoveryState::ReduceOnlyStuck {
                    consecutive_rejections: 1,
                    is_buy_side: is_buy,
                    first_rejection: Instant::now(),
                };
                false
            }
            RecoveryState::ReduceOnlyStuck {
                consecutive_rejections,
                is_buy_side,
                ..
            } => {
                // Same side rejection
                if *is_buy_side == is_buy {
                    *consecutive_rejections += 1;

                    if *consecutive_rejections >= self.config.rejection_threshold {
                        // Escalate to IOC recovery
                        tracing::warn!(
                            consecutive_rejections = *consecutive_rejections,
                            is_buy_side = *is_buy_side,
                            "Escalating to IOC recovery after {} rejections",
                            *consecutive_rejections
                        );

                        self.state = RecoveryState::IocRecovery {
                            attempts: 0,
                            is_buy_side: is_buy,
                            last_attempt: Instant::now() - self.config.ioc_cooldown,
                            filled_so_far: 0.0,
                        };
                        return true;
                    }
                } else {
                    // Different side - reset
                    self.state = RecoveryState::ReduceOnlyStuck {
                        consecutive_rejections: 1,
                        is_buy_side: is_buy,
                        first_rejection: Instant::now(),
                    };
                }
                false
            }
            RecoveryState::IocRecovery { attempts, .. } => {
                // IOC also rejected - increment attempts
                *attempts += 1;

                if *attempts >= self.config.max_ioc_attempts {
                    // Give up for now
                    tracing::error!(
                        attempts = *attempts,
                        "IOC recovery exhausted, entering cooldown"
                    );
                    self.state = RecoveryState::Cooldown {
                        until: Instant::now() + self.config.stuck_cooldown,
                        reason: CooldownReason::IocExhausted,
                    };
                }
                false
            }
            RecoveryState::Cooldown { .. } => false,
        }
    }

    /// Check if IOC recovery should be attempted now.
    ///
    /// # Returns
    /// `Some((is_buy, slippage_bps))` if IOC should be sent, `None` otherwise
    pub fn should_send_ioc(&mut self) -> Option<(bool, u32)> {
        match &self.state {
            RecoveryState::IocRecovery {
                last_attempt,
                is_buy_side,
                attempts,
                ..
            } => {
                // Check cooldown between attempts
                if last_attempt.elapsed() < self.config.ioc_cooldown {
                    return None;
                }

                // Don't exceed max attempts
                if *attempts >= self.config.max_ioc_attempts {
                    return None;
                }

                Some((*is_buy_side, self.config.ioc_slippage_bps))
            }
            _ => None,
        }
    }

    /// Record that an IOC was sent.
    pub fn record_ioc_sent(&mut self) {
        if let RecoveryState::IocRecovery {
            last_attempt,
            attempts,
            ..
        } = &mut self.state
        {
            *last_attempt = Instant::now();
            *attempts += 1;

            tracing::info!(
                attempts = *attempts,
                max_attempts = self.config.max_ioc_attempts,
                "IOC recovery attempt sent"
            );
        }
    }

    /// Check and potentially exit cooldown.
    ///
    /// Should be called periodically (e.g., every quote cycle).
    pub fn check_cooldown(&mut self) {
        if let RecoveryState::Cooldown { until, reason } = &self.state {
            if Instant::now() >= *until {
                tracing::info!(
                    reason = ?reason,
                    "Exiting recovery cooldown"
                );
                self.state = RecoveryState::Normal;
            }
        }
    }

    /// Get metrics for observability.
    pub fn get_metrics(&self) -> RecoveryMetrics {
        match &self.state {
            RecoveryState::Normal => RecoveryMetrics {
                is_active: false,
                state_name: "Normal",
                consecutive_rejections: 0,
                ioc_attempts: 0,
                ioc_filled: 0.0,
                in_cooldown: false,
            },
            RecoveryState::ReduceOnlyStuck {
                consecutive_rejections,
                ..
            } => RecoveryMetrics {
                is_active: true,
                state_name: "ReduceOnlyStuck",
                consecutive_rejections: *consecutive_rejections,
                ioc_attempts: 0,
                ioc_filled: 0.0,
                in_cooldown: false,
            },
            RecoveryState::IocRecovery {
                attempts,
                filled_so_far,
                ..
            } => RecoveryMetrics {
                is_active: true,
                state_name: "IocRecovery",
                consecutive_rejections: 0,
                ioc_attempts: *attempts,
                ioc_filled: *filled_so_far,
                in_cooldown: false,
            },
            RecoveryState::Cooldown { .. } => RecoveryMetrics {
                is_active: true,
                state_name: "Cooldown",
                consecutive_rejections: 0,
                ioc_attempts: 0,
                ioc_filled: 0.0,
                in_cooldown: true,
            },
        }
    }
}

/// Metrics for recovery state observability.
#[derive(Debug, Clone)]
pub struct RecoveryMetrics {
    pub is_active: bool,
    pub state_name: &'static str,
    pub consecutive_rejections: u32,
    pub ioc_attempts: u32,
    pub ioc_filled: f64,
    pub in_cooldown: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normal_to_stuck() {
        let mut mgr = RecoveryManager::new();
        assert!(!mgr.state().is_active());

        // First rejection enters stuck state
        assert!(!mgr.record_rejection(true, "position exceeded"));
        assert!(mgr.state().is_active());
        assert_eq!(mgr.state().state_name(), "ReduceOnlyStuck");
    }

    #[test]
    fn test_escalation_to_ioc() {
        let mut mgr = RecoveryManager::with_config(RecoveryConfig {
            rejection_threshold: 2,
            ..Default::default()
        });

        // First rejection
        assert!(!mgr.record_rejection(true, "position exceeded"));

        // Second rejection triggers escalation
        assert!(mgr.record_rejection(true, "position exceeded"));
        assert_eq!(mgr.state().state_name(), "IocRecovery");
    }

    #[test]
    fn test_success_resets() {
        let mut mgr = RecoveryManager::new();

        mgr.record_rejection(true, "position exceeded");
        assert!(mgr.state().is_active());

        mgr.record_success();
        assert!(!mgr.state().is_active());
    }

    #[test]
    fn test_different_side_resets() {
        let mut mgr = RecoveryManager::new();

        // Rejection on buy side
        mgr.record_rejection(true, "position exceeded");

        // Rejection on sell side resets counter
        mgr.record_rejection(false, "position exceeded");

        if let RecoveryState::ReduceOnlyStuck {
            consecutive_rejections,
            is_buy_side,
            ..
        } = mgr.state()
        {
            assert_eq!(*consecutive_rejections, 1);
            assert!(!*is_buy_side);
        } else {
            panic!("Expected ReduceOnlyStuck state");
        }
    }
}
