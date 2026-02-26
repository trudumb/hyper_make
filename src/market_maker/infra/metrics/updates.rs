//! Update methods for PrometheusMetrics.
//!
//! Contains all methods that modify metric values.

use std::sync::atomic::Ordering;

use super::PrometheusMetrics;

impl PrometheusMetrics {
    // === Position Updates ===

    /// Update position metrics.
    pub fn update_position(&self, position: f64, max_position: f64) {
        self.inner.position.store(position);
        self.inner.max_position.store(max_position);
        let utilization = if max_position > 0.0 {
            (position.abs() / max_position).min(1.0)
        } else {
            0.0
        };
        self.inner.inventory_utilization.store(utilization);
    }

    /// Update pending exposure metrics.
    ///
    /// # Arguments
    /// - `bid_exposure`: Total remaining size on buy orders
    /// - `ask_exposure`: Total remaining size on sell orders
    /// - `position`: Current position (for worst-case calculation)
    pub fn update_pending_exposure(&self, bid_exposure: f64, ask_exposure: f64, position: f64) {
        self.inner.pending_bid_exposure.store(bid_exposure);
        self.inner.pending_ask_exposure.store(ask_exposure);
        self.inner
            .net_pending_change
            .store(bid_exposure - ask_exposure);
        self.inner
            .worst_case_max_position
            .store(position + bid_exposure);
        self.inner
            .worst_case_min_position
            .store(position - ask_exposure);
    }

    // === P&L Updates ===

    /// Update P&L metrics.
    pub fn update_pnl(&self, daily_pnl: f64, peak_pnl: f64, realized: f64, unrealized: f64) {
        self.inner.daily_pnl.store(daily_pnl);
        self.inner.peak_pnl.store(peak_pnl);
        let drawdown = if peak_pnl > 0.0 {
            ((peak_pnl - daily_pnl) / peak_pnl * 100.0).max(0.0)
        } else {
            0.0
        };
        self.inner.drawdown_pct.store(drawdown);
        self.inner.realized_pnl.store(realized);
        self.inner.unrealized_pnl.store(unrealized);
    }

    // === Order Updates ===

    /// Record an order placed.
    pub fn record_order_placed(&self) {
        self.inner.orders_placed.fetch_add(1, Ordering::Relaxed);
    }

    /// Record an order filled.
    pub fn record_fill(&self, size: f64, is_buy: bool) {
        self.inner.orders_filled.fetch_add(1, Ordering::Relaxed);
        self.inner.fill_volume.fetch_add(size);
        if is_buy {
            self.inner.buy_volume.fetch_add(size);
        } else {
            self.inner.sell_volume.fetch_add(size);
        }
    }

    /// Record an order cancelled.
    pub fn record_cancel(&self) {
        self.inner.orders_cancelled.fetch_add(1, Ordering::Relaxed);
    }

    /// Record an order modified (queue-preserving update).
    pub fn record_order_modified(&self) {
        self.inner.orders_modified.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a modify fallback (modify failed, fell back to cancel+place).
    pub fn record_modify_fallback(&self) {
        self.inner.modify_fallbacks.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a quote cycle skipped due to BBO crossing detection.
    pub fn record_bbo_crossing_skip(&self) {
        self.inner
            .bbo_crossing_skips
            .fetch_add(1, Ordering::Relaxed);
    }

    // === Market Updates ===

    /// Update market metrics.
    pub fn update_market(
        &self,
        mid_price: f64,
        spread_bps: f64,
        sigma: f64,
        jump_ratio: f64,
        kappa: f64,
    ) {
        self.inner.mid_price.store(mid_price);
        self.inner.spread_bps.store(spread_bps);
        self.inner.sigma.store(sigma);
        self.inner.jump_ratio.store(jump_ratio);
        self.inner.kappa.store(kappa);
    }

    // === Estimator Updates ===

    /// Update estimator metrics.
    pub fn update_estimator(
        &self,
        microprice_deviation_bps: f64,
        book_imbalance: f64,
        flow_imbalance: f64,
        beta_book: f64,
        beta_flow: f64,
    ) {
        self.inner
            .microprice_deviation_bps
            .store(microprice_deviation_bps);
        self.inner.book_imbalance.store(book_imbalance);
        self.inner.flow_imbalance.store(flow_imbalance);
        self.inner.beta_book.store(beta_book);
        self.inner.beta_flow.store(beta_flow);
    }

    /// Update V2 Bayesian estimator metrics.
    ///
    /// These metrics provide uncertainty quantification for the kappa parameter
    /// and improved toxicity detection from the soft jump classifier.
    pub fn update_v2_estimator(
        &self,
        kappa_uncertainty: f64,
        kappa_95_lower: f64,
        kappa_95_upper: f64,
        toxicity_score: f64,
        param_correlation: f64,
        as_factor: f64,
    ) {
        self.inner.kappa_uncertainty.store(kappa_uncertainty);
        self.inner.kappa_95_lower.store(kappa_95_lower);
        self.inner.kappa_95_upper.store(kappa_95_upper);
        self.inner.toxicity_score.store(toxicity_score);
        self.inner.param_correlation.store(param_correlation);
        self.inner.as_factor.store(as_factor);
    }

    /// Update robust kappa diagnostic metrics.
    ///
    /// These metrics track the outlier-resistant Student-t kappa estimator.
    pub fn update_robust_kappa(&self, ess: f64, outliers: u64, nu: f64, obs_count: u64) {
        self.inner.robust_kappa_ess.store(ess);
        self.inner
            .robust_kappa_outliers
            .store(outliers, Ordering::Relaxed);
        self.inner.robust_kappa_nu.store(nu);
        self.inner
            .robust_kappa_obs_count
            .store(obs_count, Ordering::Relaxed);
    }

    // === Risk Updates ===

    /// Update risk metrics.
    pub fn update_risk(
        &self,
        kill_switch_triggered: bool,
        cascade_severity: f64,
        adverse_selection_bps: f64,
        tail_risk_multiplier: f64,
    ) {
        self.inner
            .kill_switch_triggered
            .store(if kill_switch_triggered { 1 } else { 0 }, Ordering::Relaxed);
        self.inner.cascade_severity.store(cascade_severity);
        self.inner
            .adverse_selection_bps
            .store(adverse_selection_bps);
        self.inner.tail_risk_multiplier.store(tail_risk_multiplier);
    }

    /// Update volatility regime.
    pub fn update_volatility_regime(&self, regime: u64) {
        self.inner
            .volatility_regime
            .store(regime, Ordering::Relaxed);
    }

    /// Update multi-horizon adverse selection metrics.
    ///
    /// # Arguments
    /// - `as_500ms_bps`: AS measured at 500ms horizon
    /// - `as_1000ms_bps`: AS measured at 1000ms horizon
    /// - `as_2000ms_bps`: AS measured at 2000ms horizon
    /// - `best_horizon_ms`: Current best horizon in use
    pub fn update_multi_horizon_as(
        &self,
        as_500ms_bps: f64,
        as_1000ms_bps: f64,
        as_2000ms_bps: f64,
        best_horizon_ms: u64,
    ) {
        self.inner.as_500ms_bps.store(as_500ms_bps);
        self.inner.as_1000ms_bps.store(as_1000ms_bps);
        self.inner.as_2000ms_bps.store(as_2000ms_bps);
        self.inner
            .as_best_horizon_ms
            .store(best_horizon_ms, Ordering::Relaxed);
    }

    // === Kelly-Stochastic Updates ===

    /// Update Kelly-Stochastic allocation metrics.
    pub fn update_kelly_stochastic(
        &self,
        enabled: bool,
        alpha_touch: f64,
        kelly_fraction: f64,
        alpha_decay_bps: f64,
    ) {
        self.inner
            .kelly_stochastic_enabled
            .store(if enabled { 1 } else { 0 }, Ordering::Relaxed);
        self.inner.kelly_alpha_touch.store(alpha_touch);
        self.inner.kelly_fraction.store(kelly_fraction);
        self.inner.kelly_alpha_decay_bps.store(alpha_decay_bps);
    }

    // === Connection Health Updates ===

    /// Set WebSocket connection status.
    pub fn set_websocket_connected(&self, connected: bool) {
        self.inner
            .websocket_connected
            .store(if connected { 1 } else { 0 }, Ordering::Relaxed);
    }

    /// Update last trade age in milliseconds.
    pub fn set_last_trade_age_ms(&self, age_ms: u64) {
        self.inner
            .last_trade_age_ms
            .store(age_ms, Ordering::Relaxed);
    }

    /// Update last book age in milliseconds.
    pub fn set_last_book_age_ms(&self, age_ms: u64) {
        self.inner.last_book_age_ms.store(age_ms, Ordering::Relaxed);
    }

    /// Update WebSocket health statistics.
    ///
    /// # Arguments
    /// - `reconnection_count`: Total reconnections since start
    /// - `pong_timeout_count`: Number of pong timeouts detected
    /// - `avg_ping_latency_ms`: Average ping RTT in milliseconds
    /// - `time_since_pong_ms`: Time since last pong received (ms)
    pub fn update_ws_health(
        &self,
        reconnection_count: u64,
        pong_timeout_count: u64,
        avg_ping_latency_ms: f64,
        time_since_pong_ms: u64,
    ) {
        self.inner
            .ws_reconnection_count
            .store(reconnection_count, Ordering::Relaxed);
        self.inner
            .ws_pong_timeout_count
            .store(pong_timeout_count, Ordering::Relaxed);
        self.inner.ws_ping_latency_ms.store(avg_ping_latency_ms);
        self.inner
            .ws_time_since_pong_ms
            .store(time_since_pong_ms, Ordering::Relaxed);
    }

    /// Update connection supervisor statistics.
    ///
    /// # Arguments
    /// - `stale_count`: Consecutive stale readings
    /// - `reconnect_signal_count`: Total reconnection signals issued
    pub fn update_supervisor_stats(&self, stale_count: u32, reconnect_signal_count: u64) {
        self.inner
            .supervisor_stale_count
            .store(stale_count as u64, Ordering::Relaxed);
        self.inner
            .supervisor_reconnect_signals
            .store(reconnect_signal_count, Ordering::Relaxed);
    }

    /// Get measured WebSocket ping latency in milliseconds.
    /// Returns the EWMA-smoothed round-trip time from ping/pong measurements.
    /// Used for latency-adjusted spread floor and position sizing.
    pub fn ws_ping_latency_ms(&self) -> f64 {
        self.inner.ws_ping_latency_ms.load()
    }

    // === Data Quality Updates ===

    /// Record a data quality issue.
    pub fn record_data_quality_issue(&self) {
        self.inner
            .data_quality_issues_total
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Record sequence gaps (message loss).
    pub fn record_message_loss(&self, gap_count: u64) {
        self.inner
            .message_loss_count
            .fetch_add(gap_count, Ordering::Relaxed);
    }

    /// Record a crossed book incident.
    pub fn record_crossed_book(&self) {
        self.inner
            .crossed_book_incidents
            .fetch_add(1, Ordering::Relaxed);
    }

    // === Exchange Position Limits Updates ===

    /// Update exchange position limits metrics.
    #[allow(clippy::too_many_arguments)]
    pub fn update_exchange_limits(
        &self,
        max_long: f64,
        max_short: f64,
        available_buy: f64,
        available_sell: f64,
        effective_bid: f64,
        effective_ask: f64,
        age_ms: u64,
        valid: bool,
    ) {
        self.inner.exchange_max_long.store(max_long);
        self.inner.exchange_max_short.store(max_short);
        self.inner.exchange_available_buy.store(available_buy);
        self.inner.exchange_available_sell.store(available_sell);
        self.inner.exchange_effective_bid.store(effective_bid);
        self.inner.exchange_effective_ask.store(effective_ask);
        self.inner
            .exchange_limits_age_ms
            .store(age_ms, Ordering::Relaxed);
        self.inner
            .exchange_limits_valid
            .store(if valid { 1 } else { 0 }, Ordering::Relaxed);
    }

    // === Timing Updates ===

    /// Update timing metrics.
    pub fn update_timing(&self, data_staleness_secs: f64, quote_cycle_latency_ms: f64) {
        self.inner.data_staleness_secs.store(data_staleness_secs);
        self.inner
            .quote_cycle_latency_ms
            .store(quote_cycle_latency_ms);
    }

    // === Calibration Fill Rate Controller Updates ===

    /// Update calibration fill rate controller metrics.
    ///
    /// # Arguments
    /// - `gamma_mult`: Current gamma multiplier [0.3, 1.0]
    /// - `progress`: Calibration progress [0.0, 1.0]
    /// - `fill_count`: Number of fills in lookback window
    /// - `complete`: Whether calibration is complete
    pub fn update_calibration(
        &self,
        gamma_mult: f64,
        progress: f64,
        fill_count: usize,
        complete: bool,
    ) {
        self.inner.calibration_gamma_mult.store(gamma_mult);
        self.inner.calibration_progress.store(progress);
        self.inner
            .calibration_fill_count
            .store(fill_count as u64, Ordering::Relaxed);
        self.inner
            .calibration_complete
            .store(if complete { 1 } else { 0 }, Ordering::Relaxed);
    }

    /// Update impulse control metrics.
    ///
    /// # Arguments
    /// * `enabled` - Whether impulse control is active
    /// * `available` - Current available tokens
    /// * `utilization` - Budget utilization ratio
    /// * `earned` - Total tokens earned from fills
    /// * `spent` - Total tokens spent on actions
    /// * `filter_blocked` - Actions blocked by Δλ filter
    /// * `queue_locked` - Actions blocked by queue lock
    /// * `budget_skipped` - Cycles skipped due to budget
    #[allow(clippy::too_many_arguments)]
    pub fn update_impulse_control(
        &self,
        enabled: bool,
        available: f64,
        utilization: f64,
        earned: f64,
        spent: f64,
        filter_blocked: u64,
        queue_locked: u64,
        budget_skipped: u64,
    ) {
        self.inner
            .impulse_control_enabled
            .store(if enabled { 1 } else { 0 }, Ordering::Relaxed);
        self.inner.impulse_budget_available.store(available);
        self.inner.impulse_budget_utilization.store(utilization);
        self.inner.impulse_budget_earned.store(earned);
        self.inner.impulse_budget_spent.store(spent);
        self.inner
            .impulse_filter_blocked
            .store(filter_blocked, Ordering::Relaxed);
        self.inner
            .impulse_queue_locked
            .store(queue_locked, Ordering::Relaxed);
        self.inner
            .impulse_budget_skipped
            .store(budget_skipped, Ordering::Relaxed);
    }

    /// Increment impulse filter blocked counter.
    pub fn incr_impulse_filter_blocked(&self, count: u64) {
        self.inner
            .impulse_filter_blocked
            .fetch_add(count, Ordering::Relaxed);
    }

    /// Increment impulse queue locked counter.
    pub fn incr_impulse_queue_locked(&self, count: u64) {
        self.inner
            .impulse_queue_locked
            .fetch_add(count, Ordering::Relaxed);
    }

    /// Increment impulse budget skipped counter.
    pub fn incr_impulse_budget_skipped(&self) {
        self.inner
            .impulse_budget_skipped
            .fetch_add(1, Ordering::Relaxed);
    }

    // === Learned Parameters Updates ===

    /// Update learned parameters metrics.
    ///
    /// # Arguments
    /// * `alpha_touch` - Learned informed trader probability at touch [0, 1]
    /// * `kappa` - Learned fill intensity
    /// * `spread_floor_bps` - Learned spread floor in bps
    /// * `observations` - Total observations for learned parameters
    /// * `calibrated` - Whether tier1 parameters are calibrated
    pub fn update_learned_params(
        &self,
        alpha_touch: f64,
        kappa: f64,
        spread_floor_bps: f64,
        observations: usize,
        calibrated: bool,
    ) {
        self.inner.learned_alpha_touch.store(alpha_touch);
        self.inner.learned_kappa.store(kappa);
        self.inner.learned_spread_floor_bps.store(spread_floor_bps);
        self.inner
            .learned_params_observations
            .store(observations as u64, Ordering::Relaxed);
        self.inner
            .learned_params_calibrated
            .store(if calibrated { 1 } else { 0 }, Ordering::Relaxed);
    }
}
