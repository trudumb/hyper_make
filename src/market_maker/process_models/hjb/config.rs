//! HJB Controller Configuration.

/// Configuration for HJB inventory controller.
#[derive(Debug, Clone)]
pub struct HJBConfig {
    /// Session duration for terminal penalty (seconds)
    /// For 24/7 markets, use daily session (86400) or shorter sub-sessions
    pub session_duration_secs: f64,

    /// Terminal inventory penalty ($/unit²)
    /// Higher values force more aggressive position reduction near session end
    /// Typical: 0.0001 - 0.001 (0.01% - 0.1% per unit²)
    pub terminal_penalty: f64,

    /// Base risk aversion (γ)
    /// Used in diffusion skew calculation: γσ²qT
    pub gamma_base: f64,

    /// Funding rate half-life for EWMA (seconds)
    pub funding_ewma_half_life: f64,

    /// Minimum time remaining before terminal urgency kicks in (seconds)
    /// Avoids division by zero and extreme values near session end
    pub min_time_remaining: f64,

    /// Maximum terminal skew multiplier
    /// Caps the terminal penalty contribution to prevent extreme skews
    pub max_terminal_multiplier: f64,

    // === Drift-Adjusted Skew (First Principles Extension) ===
    /// Enable drift-adjusted skew from momentum signals.
    /// When true, incorporates predicted price drift into optimal skew.
    pub use_drift_adjusted_skew: bool,

    /// Sensitivity to momentum-position opposition [0.5, 3.0].
    /// Higher values increase skew urgency when position opposes momentum.
    pub opposition_sensitivity: f64,

    /// Maximum drift urgency multiplier [1.5, 5.0].
    /// Caps the drift contribution to prevent extreme skews.
    pub max_drift_urgency: f64,

    /// Minimum continuation probability to apply drift adjustment [0.3, 0.7].
    /// Below this, momentum is considered noise.
    pub min_continuation_prob: f64,

    // === EWMA Smoothing for Stability (Legacy) ===
    /// EWMA half-life for smoothing drift estimates (seconds).
    /// Longer values give more stable but slower-reacting signals.
    /// Range: [5.0, 60.0], Default: 15.0
    /// NOTE: Deprecated in favor of OU drift model when use_ou_drift is true.
    pub drift_ewma_half_life_secs: f64,

    /// Window size for momentum statistics (data points).
    /// Range: [20, 200], Default: 50
    pub momentum_stats_window: usize,

    /// Minimum observations before using smoothed signals.
    /// Range: [10, 50], Default: 20
    pub min_warmup_observations: usize,

    // === Ornstein-Uhlenbeck Drift Model ===
    /// Enable OU drift model (replaces EWMA for drift smoothing).
    /// When true, uses mean-reverting OU process with threshold gating
    /// to filter noise-induced reconciliations.
    pub use_ou_drift: bool,

    /// OU mean reversion rate (θ).
    /// Higher values = faster mean reversion.
    /// At θ=0.5, half-life is ~1.4 seconds.
    /// Range: [0.1, 2.0], Default: 0.5
    pub ou_theta: f64,

    /// OU reconciliation threshold multiplier (k).
    /// Only reconcile if |innovation| > k × σ × √dt.
    /// Higher values filter more noise but may miss regime changes.
    /// Range: [1.0, 4.0], Default: 2.0
    pub ou_reconcile_k: f64,

    // === Funding Horizon ===
    /// Use funding-cycle time horizon instead of fixed session duration.
    /// When true, `time_remaining()` uses time-to-next-funding-settlement
    /// (8h cycle) when available, falling back to `session_duration_secs`.
    /// This creates natural urgency near funding settlement since inventory
    /// matters for funding P&L.
    pub use_funding_horizon: bool,

    /// Calibrate terminal penalty from observed market spread.
    /// When true, terminal_penalty = expected_spread_to_cross / 10000 + overnight_vol_cost.
    /// When false, uses the static `terminal_penalty` value.
    pub calibrate_terminal_penalty: bool,

    // === Queue Value Parameters ===
    /// Enable HJB queue value preservation.
    /// When true, considers queue position value before cancelling orders.
    pub use_queue_value: bool,

    /// Queue value decay rate (α).
    /// Higher values = queue value decays faster with depth.
    /// Range: [0.05, 0.3], Default: 0.1
    pub queue_alpha: f64,

    /// Queue value linear cost (β).
    /// Penalizes orders deep in queue.
    /// Range: [0.01, 0.05], Default: 0.02
    pub queue_beta: f64,

    /// Cost of modifying an order (in bps).
    /// Only cancel+replace if queue value < this cost.
    /// Range: [1.0, 10.0], Default: 3.0
    pub queue_modify_cost_bps: f64,
}

impl Default for HJBConfig {
    fn default() -> Self {
        Self {
            session_duration_secs: 86400.0, // 24 hour session
            terminal_penalty: 0.0005,       // 0.05% per unit²
            gamma_base: 0.3,                // Moderate risk aversion
            funding_ewma_half_life: 3600.0, // 1 hour
            min_time_remaining: 60.0,       // 1 minute minimum
            max_terminal_multiplier: 5.0,   // Cap at 5x normal skew
            // Drift-adjusted skew (enabled by default for first-principles trading)
            use_drift_adjusted_skew: true,
            opposition_sensitivity: 1.5,
            max_drift_urgency: 3.0,
            min_continuation_prob: 0.5,
            // EWMA smoothing for stability (legacy, used when use_ou_drift is false)
            drift_ewma_half_life_secs: 15.0,
            momentum_stats_window: 50,
            min_warmup_observations: 20,
            // OU drift model (preferred over EWMA)
            use_ou_drift: true,
            ou_theta: 0.5,           // ~1.4s half-life
            ou_reconcile_k: 2.0,     // 2σ threshold
            // Funding horizon (enabled by default — perpetuals settle every 8h)
            use_funding_horizon: true,
            calibrate_terminal_penalty: true,
            // Queue value parameters
            use_queue_value: true,
            queue_alpha: 0.1,
            queue_beta: 0.02,
            queue_modify_cost_bps: 3.0,
        }
    }
}
