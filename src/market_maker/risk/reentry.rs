//! Gradual re-entry after kill switch trigger.
//!
//! Implements a state machine for safe recovery after the kill switch fires:
//! 1. **Cooling**: Mandatory wait period after trigger (default 5 min)
//! 2. **Recovery**: Re-enter with reduced position limits (50%), ramp up over time
//! 3. **Normal**: Full position limits restored
//!
//! Multi-kill protection:
//! - Killed again within 1 hour of re-entry → wait until next day
//! - 3+ kills in a single day → manual review required (no auto re-entry)

use std::time::{Duration, Instant};

use super::KillReason;

/// Configuration for gradual re-entry after kill switch.
#[derive(Debug, Clone)]
pub struct ReentryConfig {
    /// Mandatory cooling period after kill switch trigger (default 5 min)
    pub cooling_period: Duration,
    /// Position limit multiplier during recovery phase (default 0.5 = 50%)
    pub recovery_position_fraction: f64,
    /// Duration of recovery phase before returning to full limits (default 30 min)
    pub recovery_duration: Duration,
    /// If killed again within this window after re-entry, escalate (default 1 hour)
    pub rapid_kill_window: Duration,
    /// Maximum kills per day before requiring manual review (default 3)
    pub max_daily_kills: u32,
    /// Whether auto re-entry is enabled at all (default true)
    pub auto_reentry_enabled: bool,
}

impl Default for ReentryConfig {
    fn default() -> Self {
        Self {
            cooling_period: Duration::from_secs(300),     // 5 minutes
            recovery_position_fraction: 0.5,              // 50% position limits
            recovery_duration: Duration::from_secs(1800), // 30 minutes
            rapid_kill_window: Duration::from_secs(3600), // 1 hour
            max_daily_kills: 3,
            auto_reentry_enabled: true,
        }
    }
}

impl ReentryConfig {
    /// Production preset per Phase 3 plan.
    pub fn production() -> Self {
        Self {
            cooling_period: Duration::from_secs(300),
            recovery_position_fraction: 0.5,
            recovery_duration: Duration::from_secs(1800),
            rapid_kill_window: Duration::from_secs(3600),
            max_daily_kills: 3,
            auto_reentry_enabled: true,
        }
    }
}

/// Current phase in the re-entry state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReentryPhase {
    /// Normal operation — no kill switch active
    Normal,
    /// Kill switch triggered, in mandatory cooling period
    Cooling,
    /// Cooling complete, operating with reduced position limits
    Recovery,
    /// Requires manual intervention (too many kills today, or rapid re-kill)
    ManualReviewRequired,
}

impl ReentryPhase {
    /// Display name for logging.
    pub fn as_str(&self) -> &'static str {
        match self {
            ReentryPhase::Normal => "Normal",
            ReentryPhase::Cooling => "Cooling",
            ReentryPhase::Recovery => "Recovery",
            ReentryPhase::ManualReviewRequired => "ManualReviewRequired",
        }
    }
}

/// A record of a kill switch trigger event.
#[derive(Debug, Clone)]
struct KillEvent {
    #[allow(dead_code)]
    time: Instant,
    reason: KillReason,
}

/// Manages gradual re-entry after kill switch triggers.
///
/// Thread safety: This struct is NOT internally synchronized. The caller
/// (typically the event loop) must ensure single-threaded access.
pub struct ReentryManager {
    config: ReentryConfig,
    /// Current phase
    phase: ReentryPhase,
    /// When the current kill switch was triggered
    last_kill_time: Option<Instant>,
    /// When recovery phase started
    recovery_start: Option<Instant>,
    /// All kill events today
    daily_kills: Vec<KillEvent>,
    /// Last time we re-entered (transitioned from Cooling → Recovery)
    last_reentry_time: Option<Instant>,
}

impl ReentryManager {
    /// Create a new re-entry manager.
    pub fn new(config: ReentryConfig) -> Self {
        Self {
            config,
            phase: ReentryPhase::Normal,
            last_kill_time: None,
            recovery_start: None,
            daily_kills: Vec::new(),
            last_reentry_time: None,
        }
    }

    /// Current phase of the re-entry state machine.
    pub fn phase(&self) -> ReentryPhase {
        self.phase
    }

    /// Position limit multiplier (0.0 to 1.0).
    ///
    /// Returns:
    /// - 1.0 during Normal
    /// - 0.0 during Cooling and ManualReviewRequired
    /// - Linear ramp from `recovery_position_fraction` to 1.0 during Recovery
    pub fn position_multiplier(&self) -> f64 {
        match self.phase {
            ReentryPhase::Normal => 1.0,
            ReentryPhase::Cooling | ReentryPhase::ManualReviewRequired => 0.0,
            ReentryPhase::Recovery => {
                if let Some(start) = self.recovery_start {
                    let elapsed = start.elapsed();
                    let progress = (elapsed.as_secs_f64()
                        / self.config.recovery_duration.as_secs_f64())
                    .clamp(0.0, 1.0);
                    // Linear ramp: fraction → 1.0
                    let frac = self.config.recovery_position_fraction;
                    frac + (1.0 - frac) * progress
                } else {
                    self.config.recovery_position_fraction
                }
            }
        }
    }

    /// Whether trading is allowed right now.
    pub fn can_trade(&self) -> bool {
        matches!(self.phase, ReentryPhase::Normal | ReentryPhase::Recovery)
    }

    /// Number of kills today.
    pub fn daily_kill_count(&self) -> u32 {
        self.daily_kills.len() as u32
    }

    /// Record a kill switch trigger.
    ///
    /// Transitions to Cooling or ManualReviewRequired depending on history.
    pub fn on_kill(&mut self, reason: KillReason) {
        let now = Instant::now();

        // Check if this is a rapid re-kill (within window of last re-entry)
        let is_rapid_rekill = self
            .last_reentry_time
            .is_some_and(|t| now.duration_since(t) < self.config.rapid_kill_window);

        // Record this kill
        self.daily_kills.push(KillEvent { time: now, reason });
        self.last_kill_time = Some(now);

        // Decide next phase
        if !self.config.auto_reentry_enabled {
            self.phase = ReentryPhase::ManualReviewRequired;
        } else if self.daily_kills.len() as u32 >= self.config.max_daily_kills {
            // Too many kills today — require manual review
            self.phase = ReentryPhase::ManualReviewRequired;
        } else if is_rapid_rekill {
            // Killed again shortly after re-entry — require manual review
            self.phase = ReentryPhase::ManualReviewRequired;
        } else {
            self.phase = ReentryPhase::Cooling;
        }

        self.recovery_start = None;
    }

    /// Tick the state machine. Call this periodically (e.g., every second).
    ///
    /// Handles transitions:
    /// - Cooling → Recovery (after cooling period)
    /// - Recovery → Normal (after recovery duration)
    ///
    /// Returns `true` if a phase transition occurred.
    pub fn tick(&mut self) -> bool {
        match self.phase {
            ReentryPhase::Cooling => {
                if let Some(kill_time) = self.last_kill_time {
                    if kill_time.elapsed() >= self.config.cooling_period {
                        // Cooling complete — enter recovery
                        self.phase = ReentryPhase::Recovery;
                        self.recovery_start = Some(Instant::now());
                        self.last_reentry_time = Some(Instant::now());
                        return true;
                    }
                }
                false
            }
            ReentryPhase::Recovery => {
                if let Some(start) = self.recovery_start {
                    if start.elapsed() >= self.config.recovery_duration {
                        // Recovery complete — back to normal
                        self.phase = ReentryPhase::Normal;
                        self.recovery_start = None;
                        return true;
                    }
                }
                false
            }
            _ => false,
        }
    }

    /// Manually approve re-entry (for ManualReviewRequired state).
    ///
    /// Transitions to Cooling, which will then transition to Recovery after the cooling period.
    pub fn approve_reentry(&mut self) {
        if self.phase == ReentryPhase::ManualReviewRequired {
            self.phase = ReentryPhase::Cooling;
            self.last_kill_time = Some(Instant::now());
            self.recovery_start = None;
        }
    }

    /// Reset daily kill counter. Call at the start of each trading day.
    pub fn reset_daily(&mut self) {
        self.daily_kills.clear();
        // If stuck in ManualReviewRequired due to daily limit, go back to Normal
        if self.phase == ReentryPhase::ManualReviewRequired {
            self.phase = ReentryPhase::Normal;
        }
    }

    /// Force return to Normal phase (for testing / manual override).
    pub fn force_normal(&mut self) {
        self.phase = ReentryPhase::Normal;
        self.recovery_start = None;
    }

    /// Summary for logging/monitoring.
    pub fn summary(&self) -> ReentrySummary {
        let cooling_remaining = match (self.phase, self.last_kill_time) {
            (ReentryPhase::Cooling, Some(kill_time)) => self
                .config
                .cooling_period
                .checked_sub(kill_time.elapsed())
                .map(|d| d.as_secs_f64()),
            _ => None,
        };

        let recovery_remaining = match (self.phase, self.recovery_start) {
            (ReentryPhase::Recovery, Some(start)) => self
                .config
                .recovery_duration
                .checked_sub(start.elapsed())
                .map(|d| d.as_secs_f64()),
            _ => None,
        };

        ReentrySummary {
            phase: self.phase,
            position_multiplier: self.position_multiplier(),
            can_trade: self.can_trade(),
            daily_kill_count: self.daily_kill_count(),
            cooling_remaining_s: cooling_remaining,
            recovery_remaining_s: recovery_remaining,
            last_kill_reason: self.daily_kills.last().map(|e| e.reason.to_string()),
        }
    }
}

/// Summary of re-entry state for logging.
#[derive(Debug, Clone)]
pub struct ReentrySummary {
    pub phase: ReentryPhase,
    pub position_multiplier: f64,
    pub can_trade: bool,
    pub daily_kill_count: u32,
    pub cooling_remaining_s: Option<f64>,
    pub recovery_remaining_s: Option<f64>,
    pub last_kill_reason: Option<String>,
}

impl std::fmt::Display for ReentrySummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Reentry[{}] pos_mult={:.0}% kills_today={} trade={}",
            self.phase.as_str(),
            self.position_multiplier * 100.0,
            self.daily_kill_count,
            if self.can_trade { "YES" } else { "NO" },
        )?;
        if let Some(cool) = self.cooling_remaining_s {
            write!(f, " cooling={cool:.0}s")?;
        }
        if let Some(recov) = self.recovery_remaining_s {
            write!(f, " recovery={recov:.0}s")?;
        }
        if let Some(reason) = &self.last_kill_reason {
            write!(f, " last_kill={reason}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_state() {
        let mgr = ReentryManager::new(ReentryConfig::default());
        assert_eq!(mgr.phase(), ReentryPhase::Normal);
        assert_eq!(mgr.position_multiplier(), 1.0);
        assert!(mgr.can_trade());
        assert_eq!(mgr.daily_kill_count(), 0);
    }

    #[test]
    fn test_kill_transitions_to_cooling() {
        let mut mgr = ReentryManager::new(ReentryConfig::default());
        mgr.on_kill(KillReason::MaxLoss {
            loss: 100.0,
            limit: 50.0,
        });
        assert_eq!(mgr.phase(), ReentryPhase::Cooling);
        assert_eq!(mgr.position_multiplier(), 0.0);
        assert!(!mgr.can_trade());
        assert_eq!(mgr.daily_kill_count(), 1);
    }

    #[test]
    fn test_cooling_to_recovery_transition() {
        let config = ReentryConfig {
            cooling_period: Duration::from_millis(10),
            ..Default::default()
        };
        let mut mgr = ReentryManager::new(config);
        mgr.on_kill(KillReason::MaxLoss {
            loss: 100.0,
            limit: 50.0,
        });

        // Before cooling expires
        assert!(!mgr.tick());
        assert_eq!(mgr.phase(), ReentryPhase::Cooling);

        // Wait for cooling
        std::thread::sleep(Duration::from_millis(15));
        assert!(mgr.tick());
        assert_eq!(mgr.phase(), ReentryPhase::Recovery);
        assert!(mgr.can_trade());

        // Position multiplier should be at recovery fraction initially
        let mult = mgr.position_multiplier();
        assert!(mult >= 0.49 && mult <= 0.55, "Expected ~0.5, got {}", mult);
    }

    #[test]
    fn test_recovery_to_normal_transition() {
        let config = ReentryConfig {
            cooling_period: Duration::from_millis(5),
            recovery_duration: Duration::from_millis(10),
            ..Default::default()
        };
        let mut mgr = ReentryManager::new(config);
        mgr.on_kill(KillReason::MaxLoss {
            loss: 100.0,
            limit: 50.0,
        });

        // Wait for cooling
        std::thread::sleep(Duration::from_millis(10));
        mgr.tick();
        assert_eq!(mgr.phase(), ReentryPhase::Recovery);

        // Wait for recovery
        std::thread::sleep(Duration::from_millis(15));
        assert!(mgr.tick());
        assert_eq!(mgr.phase(), ReentryPhase::Normal);
        assert_eq!(mgr.position_multiplier(), 1.0);
    }

    #[test]
    fn test_max_daily_kills_triggers_manual_review() {
        let config = ReentryConfig {
            max_daily_kills: 2,
            ..Default::default()
        };
        let mut mgr = ReentryManager::new(config);

        mgr.on_kill(KillReason::MaxLoss {
            loss: 100.0,
            limit: 50.0,
        });
        assert_eq!(mgr.phase(), ReentryPhase::Cooling);

        mgr.force_normal();
        mgr.on_kill(KillReason::MaxLoss {
            loss: 100.0,
            limit: 50.0,
        });
        // 2nd kill hits max_daily_kills=2 → manual review
        assert_eq!(mgr.phase(), ReentryPhase::ManualReviewRequired);
        assert!(!mgr.can_trade());
    }

    #[test]
    fn test_rapid_rekill_triggers_manual_review() {
        let config = ReentryConfig {
            cooling_period: Duration::from_millis(5),
            rapid_kill_window: Duration::from_secs(3600),
            max_daily_kills: 10, // High limit so we test rapid-rekill specifically
            ..Default::default()
        };
        let mut mgr = ReentryManager::new(config);

        // First kill → cooling
        mgr.on_kill(KillReason::MaxLoss {
            loss: 100.0,
            limit: 50.0,
        });
        std::thread::sleep(Duration::from_millis(10));
        mgr.tick(); // → Recovery
        assert_eq!(mgr.phase(), ReentryPhase::Recovery);

        // Kill again while within rapid_kill_window → manual review
        mgr.on_kill(KillReason::MaxLoss {
            loss: 100.0,
            limit: 50.0,
        });
        assert_eq!(mgr.phase(), ReentryPhase::ManualReviewRequired);
    }

    #[test]
    fn test_approve_reentry() {
        let config = ReentryConfig {
            max_daily_kills: 1,
            ..Default::default()
        };
        let mut mgr = ReentryManager::new(config);
        mgr.on_kill(KillReason::Manual {
            reason: "test".into(),
        });
        assert_eq!(mgr.phase(), ReentryPhase::ManualReviewRequired);

        mgr.approve_reentry();
        assert_eq!(mgr.phase(), ReentryPhase::Cooling);
    }

    #[test]
    fn test_reset_daily() {
        let config = ReentryConfig {
            max_daily_kills: 1,
            ..Default::default()
        };
        let mut mgr = ReentryManager::new(config);
        mgr.on_kill(KillReason::Manual {
            reason: "test".into(),
        });
        assert_eq!(mgr.phase(), ReentryPhase::ManualReviewRequired);
        assert_eq!(mgr.daily_kill_count(), 1);

        mgr.reset_daily();
        assert_eq!(mgr.daily_kill_count(), 0);
        assert_eq!(mgr.phase(), ReentryPhase::Normal);
    }

    #[test]
    fn test_auto_reentry_disabled() {
        let config = ReentryConfig {
            auto_reentry_enabled: false,
            ..Default::default()
        };
        let mut mgr = ReentryManager::new(config);
        mgr.on_kill(KillReason::MaxLoss {
            loss: 100.0,
            limit: 50.0,
        });
        assert_eq!(mgr.phase(), ReentryPhase::ManualReviewRequired);
    }

    #[test]
    fn test_summary_display() {
        let mut mgr = ReentryManager::new(ReentryConfig::default());
        let summary = mgr.summary();
        assert_eq!(summary.phase, ReentryPhase::Normal);
        assert_eq!(summary.position_multiplier, 1.0);
        assert!(summary.can_trade);

        // Format should not panic
        let s = format!("{}", summary);
        assert!(s.contains("Normal"));

        mgr.on_kill(KillReason::StaleData {
            elapsed: Duration::from_secs(60),
            threshold: Duration::from_secs(10),
        });
        let summary = mgr.summary();
        assert_eq!(summary.phase, ReentryPhase::Cooling);
        assert!(summary.cooling_remaining_s.is_some());
        assert!(summary.last_kill_reason.is_some());
        let s = format!("{}", summary);
        assert!(s.contains("Cooling"));
    }

    #[test]
    fn test_position_multiplier_ramps() {
        let config = ReentryConfig {
            cooling_period: Duration::from_millis(1),
            recovery_duration: Duration::from_millis(100),
            recovery_position_fraction: 0.5,
            ..Default::default()
        };
        let mut mgr = ReentryManager::new(config);
        mgr.on_kill(KillReason::MaxLoss {
            loss: 100.0,
            limit: 50.0,
        });

        std::thread::sleep(Duration::from_millis(5));
        mgr.tick(); // → Recovery

        // Should start at ~0.5
        let m1 = mgr.position_multiplier();
        assert!(m1 >= 0.4 && m1 <= 0.65, "Expected ~0.5, got {}", m1);

        // Wait a bit, multiplier should increase
        std::thread::sleep(Duration::from_millis(50));
        let m2 = mgr.position_multiplier();
        assert!(m2 > m1, "Multiplier should ramp up: {} > {}", m2, m1);
    }
}
