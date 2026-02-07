//! Output methods for PrometheusMetrics.
//!
//! Contains to_prometheus_text, to_map, and summary methods.

use std::collections::HashMap;
use std::sync::atomic::Ordering;

use super::dashboard::{DashboardSnapshotParams, DashboardState};
use super::summary::MetricsSummary;
use super::PrometheusMetrics;

impl PrometheusMetrics {
    /// Get all metrics as Prometheus text format.
    ///
    /// # Arguments
    /// * `asset` - The trading asset (e.g., "BTC", "hyna:BTC")
    /// * `quote_asset` - The collateral/quote asset (e.g., "USDC", "USDE", "USDH")
    pub fn to_prometheus_text(&self, asset: &str, quote_asset: &str) -> String {
        let uptime_secs = self.inner.start_time.elapsed().as_secs_f64();
        let labels = format!("asset=\"{}\",quote=\"{}\"", asset, quote_asset);

        let mut output = String::with_capacity(4096);

        // Position metrics
        output.push_str(&format!(
            "# HELP mm_position Current position in base asset\n\
             # TYPE mm_position gauge\n\
             mm_position{{{}}} {}\n",
            labels,
            self.inner.position.load()
        ));

        output.push_str(&format!(
            "# HELP mm_inventory_utilization Position / max position ratio\n\
             # TYPE mm_inventory_utilization gauge\n\
             mm_inventory_utilization{{{}}} {}\n",
            labels,
            self.inner.inventory_utilization.load()
        ));

        // Pending exposure metrics
        output.push_str(&format!(
            "# HELP mm_pending_bid_exposure Total remaining size on buy orders\n\
             # TYPE mm_pending_bid_exposure gauge\n\
             mm_pending_bid_exposure{{{}}} {}\n",
            labels,
            self.inner.pending_bid_exposure.load()
        ));

        output.push_str(&format!(
            "# HELP mm_pending_ask_exposure Total remaining size on sell orders\n\
             # TYPE mm_pending_ask_exposure gauge\n\
             mm_pending_ask_exposure{{{}}} {}\n",
            labels,
            self.inner.pending_ask_exposure.load()
        ));

        output.push_str(&format!(
            "# HELP mm_net_pending_change Net pending exposure (bid - ask)\n\
             # TYPE mm_net_pending_change gauge\n\
             mm_net_pending_change{{{}}} {}\n",
            labels,
            self.inner.net_pending_change.load()
        ));

        output.push_str(&format!(
            "# HELP mm_worst_case_max_position Worst-case max position if all bids fill\n\
             # TYPE mm_worst_case_max_position gauge\n\
             mm_worst_case_max_position{{{}}} {}\n",
            labels,
            self.inner.worst_case_max_position.load()
        ));

        output.push_str(&format!(
            "# HELP mm_worst_case_min_position Worst-case min position if all asks fill\n\
             # TYPE mm_worst_case_min_position gauge\n\
             mm_worst_case_min_position{{{}}} {}\n",
            labels,
            self.inner.worst_case_min_position.load()
        ));

        // P&L metrics
        output.push_str(&format!(
            "# HELP mm_daily_pnl Daily P&L in USD\n\
             # TYPE mm_daily_pnl gauge\n\
             mm_daily_pnl{{{}}} {}\n",
            labels,
            self.inner.daily_pnl.load()
        ));

        output.push_str(&format!(
            "# HELP mm_drawdown_pct Current drawdown percentage\n\
             # TYPE mm_drawdown_pct gauge\n\
             mm_drawdown_pct{{{}}} {}\n",
            labels,
            self.inner.drawdown_pct.load()
        ));

        // Order metrics (counters)
        output.push_str(&format!(
            "# HELP mm_orders_placed_total Total orders placed\n\
             # TYPE mm_orders_placed_total counter\n\
             mm_orders_placed_total{{{}}} {}\n",
            labels,
            self.inner.orders_placed.load(Ordering::Relaxed)
        ));

        output.push_str(&format!(
            "# HELP mm_orders_filled_total Total orders filled\n\
             # TYPE mm_orders_filled_total counter\n\
             mm_orders_filled_total{{{}}} {}\n",
            labels,
            self.inner.orders_filled.load(Ordering::Relaxed)
        ));

        output.push_str(&format!(
            "# HELP mm_fill_volume_total Total fill volume\n\
             # TYPE mm_fill_volume_total counter\n\
             mm_fill_volume_total{{{}}} {}\n",
            labels,
            self.inner.fill_volume.load()
        ));

        output.push_str(&format!(
            "# HELP mm_orders_modified_total Orders modified (queue-preserving updates)\n\
             # TYPE mm_orders_modified_total counter\n\
             mm_orders_modified_total{{{}}} {}\n",
            labels,
            self.inner.orders_modified.load(Ordering::Relaxed)
        ));

        output.push_str(&format!(
            "# HELP mm_modify_fallbacks_total Modify failures falling back to cancel+place\n\
             # TYPE mm_modify_fallbacks_total counter\n\
             mm_modify_fallbacks_total{{{}}} {}\n",
            labels,
            self.inner.modify_fallbacks.load(Ordering::Relaxed)
        ));

        // Market metrics
        output.push_str(&format!(
            "# HELP mm_mid_price Current mid price\n\
             # TYPE mm_mid_price gauge\n\
             mm_mid_price{{{}}} {}\n",
            labels,
            self.inner.mid_price.load()
        ));

        output.push_str(&format!(
            "# HELP mm_spread_bps Current spread in basis points\n\
             # TYPE mm_spread_bps gauge\n\
             mm_spread_bps{{{}}} {}\n",
            labels,
            self.inner.spread_bps.load()
        ));

        output.push_str(&format!(
            "# HELP mm_sigma Volatility estimate\n\
             # TYPE mm_sigma gauge\n\
             mm_sigma{{{}}} {}\n",
            labels,
            self.inner.sigma.load()
        ));

        output.push_str(&format!(
            "# HELP mm_jump_ratio RV/BV jump ratio\n\
             # TYPE mm_jump_ratio gauge\n\
             mm_jump_ratio{{{}}} {}\n",
            labels,
            self.inner.jump_ratio.load()
        ));

        output.push_str(&format!(
            "# HELP mm_kappa Fill intensity\n\
             # TYPE mm_kappa gauge\n\
             mm_kappa{{{}}} {}\n",
            labels,
            self.inner.kappa.load()
        ));

        // V2 Bayesian Estimator metrics
        output.push_str(&format!(
            "# HELP mm_kappa_uncertainty Kappa posterior standard deviation\n\
             # TYPE mm_kappa_uncertainty gauge\n\
             mm_kappa_uncertainty{{{}}} {}\n",
            labels,
            self.inner.kappa_uncertainty.load()
        ));

        output.push_str(&format!(
            "# HELP mm_kappa_95_lower Kappa 95% credible interval lower bound\n\
             # TYPE mm_kappa_95_lower gauge\n\
             mm_kappa_95_lower{{{}}} {}\n",
            labels,
            self.inner.kappa_95_lower.load()
        ));

        output.push_str(&format!(
            "# HELP mm_kappa_95_upper Kappa 95% credible interval upper bound\n\
             # TYPE mm_kappa_95_upper gauge\n\
             mm_kappa_95_upper{{{}}} {}\n",
            labels,
            self.inner.kappa_95_upper.load()
        ));

        output.push_str(&format!(
            "# HELP mm_toxicity_score Soft toxicity score from mixture model [0,1]\n\
             # TYPE mm_toxicity_score gauge\n\
             mm_toxicity_score{{{}}} {}\n",
            labels,
            self.inner.toxicity_score.load()
        ));

        output.push_str(&format!(
            "# HELP mm_param_correlation Kappa-sigma correlation coefficient [-1,1]\n\
             # TYPE mm_param_correlation gauge\n\
             mm_param_correlation{{{}}} {}\n",
            labels,
            self.inner.param_correlation.load()
        ));

        output.push_str(&format!(
            "# HELP mm_as_factor Adverse selection adjustment factor [0.5,1.0]\n\
             # TYPE mm_as_factor gauge\n\
             mm_as_factor{{{}}} {}\n",
            labels,
            self.inner.as_factor.load()
        ));

        // Robust kappa diagnostics
        output.push_str(&format!(
            "# HELP mm_robust_kappa_ess Robust kappa effective sample size\n\
             # TYPE mm_robust_kappa_ess gauge\n\
             mm_robust_kappa_ess{{{}}} {:.2}\n",
            labels,
            self.inner.robust_kappa_ess.load()
        ));

        output.push_str(&format!(
            "# HELP mm_robust_kappa_outliers Outliers detected by robust kappa\n\
             # TYPE mm_robust_kappa_outliers gauge\n\
             mm_robust_kappa_outliers{{{}}} {}\n",
            labels,
            self.inner.robust_kappa_outliers.load(Ordering::Relaxed)
        ));

        output.push_str(&format!(
            "# HELP mm_robust_kappa_nu Student-t degrees of freedom\n\
             # TYPE mm_robust_kappa_nu gauge\n\
             mm_robust_kappa_nu{{{}}} {:.2}\n",
            labels,
            self.inner.robust_kappa_nu.load()
        ));

        output.push_str(&format!(
            "# HELP mm_robust_kappa_obs_count Robust kappa observation count\n\
             # TYPE mm_robust_kappa_obs_count gauge\n\
             mm_robust_kappa_obs_count{{{}}} {}\n",
            labels,
            self.inner.robust_kappa_obs_count.load(Ordering::Relaxed)
        ));

        // Risk metrics
        output.push_str(&format!(
            "# HELP mm_kill_switch_triggered Kill switch status (1=triggered)\n\
             # TYPE mm_kill_switch_triggered gauge\n\
             mm_kill_switch_triggered{{{}}} {}\n",
            labels,
            self.inner.kill_switch_triggered.load(Ordering::Relaxed)
        ));

        output.push_str(&format!(
            "# HELP mm_cascade_severity Liquidation cascade severity [0,1]\n\
             # TYPE mm_cascade_severity gauge\n\
             mm_cascade_severity{{{}}} {}\n",
            labels,
            self.inner.cascade_severity.load()
        ));

        output.push_str(&format!(
            "# HELP mm_adverse_selection_bps Adverse selection in bps\n\
             # TYPE mm_adverse_selection_bps gauge\n\
             mm_adverse_selection_bps{{{}}} {}\n",
            labels,
            self.inner.adverse_selection_bps.load()
        ));

        // Multi-horizon AS metrics
        output.push_str(&format!(
            "# HELP mm_as_500ms_bps Adverse selection at 500ms horizon\n\
             # TYPE mm_as_500ms_bps gauge\n\
             mm_as_500ms_bps{{{}}} {:.2}\n",
            labels,
            self.inner.as_500ms_bps.load()
        ));

        output.push_str(&format!(
            "# HELP mm_as_1000ms_bps Adverse selection at 1000ms horizon\n\
             # TYPE mm_as_1000ms_bps gauge\n\
             mm_as_1000ms_bps{{{}}} {:.2}\n",
            labels,
            self.inner.as_1000ms_bps.load()
        ));

        output.push_str(&format!(
            "# HELP mm_as_2000ms_bps Adverse selection at 2000ms horizon\n\
             # TYPE mm_as_2000ms_bps gauge\n\
             mm_as_2000ms_bps{{{}}} {:.2}\n",
            labels,
            self.inner.as_2000ms_bps.load()
        ));

        output.push_str(&format!(
            "# HELP mm_as_best_horizon_ms Best AS measurement horizon in ms\n\
             # TYPE mm_as_best_horizon_ms gauge\n\
             mm_as_best_horizon_ms{{{}}} {}\n",
            labels,
            self.inner.as_best_horizon_ms.load(Ordering::Relaxed)
        ));

        // Timing metrics
        output.push_str(&format!(
            "# HELP mm_data_staleness_secs Time since last data update\n\
             # TYPE mm_data_staleness_secs gauge\n\
             mm_data_staleness_secs{{{}}} {}\n",
            labels,
            self.inner.data_staleness_secs.load()
        ));

        // Uptime
        output.push_str(&format!(
            "# HELP mm_uptime_secs Total uptime in seconds\n\
             # TYPE mm_uptime_secs counter\n\
             mm_uptime_secs{{{}}} {:.2}\n",
            labels, uptime_secs
        ));

        // Volatility regime
        output.push_str(&format!(
            "# HELP mm_volatility_regime Current vol regime (0=Low,1=Normal,2=High,3=Extreme)\n\
             # TYPE mm_volatility_regime gauge\n\
             mm_volatility_regime{{{}}} {}\n",
            labels,
            self.inner.volatility_regime.load(Ordering::Relaxed)
        ));

        // Kelly-Stochastic metrics
        output.push_str(&format!(
            "# HELP mm_kelly_stochastic_enabled Kelly-Stochastic allocation enabled (1=enabled)\n\
             # TYPE mm_kelly_stochastic_enabled gauge\n\
             mm_kelly_stochastic_enabled{{{}}} {}\n",
            labels,
            self.inner.kelly_stochastic_enabled.load(Ordering::Relaxed)
        ));

        output.push_str(&format!(
            "# HELP mm_kelly_alpha_touch Calibrated informed probability at touch [0,1]\n\
             # TYPE mm_kelly_alpha_touch gauge\n\
             mm_kelly_alpha_touch{{{}}} {:.4}\n",
            labels,
            self.inner.kelly_alpha_touch.load()
        ));

        output.push_str(&format!(
            "# HELP mm_kelly_fraction Current Kelly fraction [0,1]\n\
             # TYPE mm_kelly_fraction gauge\n\
             mm_kelly_fraction{{{}}} {:.4}\n",
            labels,
            self.inner.kelly_fraction.load()
        ));

        output.push_str(&format!(
            "# HELP mm_kelly_alpha_decay_bps Alpha decay characteristic depth in bps\n\
             # TYPE mm_kelly_alpha_decay_bps gauge\n\
             mm_kelly_alpha_decay_bps{{{}}} {:.2}\n",
            labels,
            self.inner.kelly_alpha_decay_bps.load()
        ));

        // Connection health metrics
        output.push_str(&format!(
            "# HELP mm_websocket_connected WebSocket connection status (1=connected)\n\
             # TYPE mm_websocket_connected gauge\n\
             mm_websocket_connected{{{}}} {}\n",
            labels,
            self.inner.websocket_connected.load(Ordering::Relaxed)
        ));

        output.push_str(&format!(
            "# HELP mm_last_trade_age_ms Time since last trade update in ms\n\
             # TYPE mm_last_trade_age_ms gauge\n\
             mm_last_trade_age_ms{{{}}} {}\n",
            labels,
            self.inner.last_trade_age_ms.load(Ordering::Relaxed)
        ));

        output.push_str(&format!(
            "# HELP mm_last_book_age_ms Time since last L2 book update in ms\n\
             # TYPE mm_last_book_age_ms gauge\n\
             mm_last_book_age_ms{{{}}} {}\n",
            labels,
            self.inner.last_book_age_ms.load(Ordering::Relaxed)
        ));

        // Enhanced WebSocket health metrics
        output.push_str(&format!(
            "# HELP mm_ws_reconnection_total Total WebSocket reconnections since start\n\
             # TYPE mm_ws_reconnection_total counter\n\
             mm_ws_reconnection_total{{{}}} {}\n",
            labels,
            self.inner.ws_reconnection_count.load(Ordering::Relaxed)
        ));

        output.push_str(&format!(
            "# HELP mm_ws_pong_timeout_total Number of pong timeouts detected\n\
             # TYPE mm_ws_pong_timeout_total counter\n\
             mm_ws_pong_timeout_total{{{}}} {}\n",
            labels,
            self.inner.ws_pong_timeout_count.load(Ordering::Relaxed)
        ));

        output.push_str(&format!(
            "# HELP mm_ws_ping_latency_ms Average WebSocket ping RTT in ms\n\
             # TYPE mm_ws_ping_latency_ms gauge\n\
             mm_ws_ping_latency_ms{{{}}} {:.1}\n",
            labels,
            self.inner.ws_ping_latency_ms.load()
        ));

        output.push_str(&format!(
            "# HELP mm_ws_time_since_pong_ms Time since last pong received (ms)\n\
             # TYPE mm_ws_time_since_pong_ms gauge\n\
             mm_ws_time_since_pong_ms{{{}}} {}\n",
            labels,
            self.inner.ws_time_since_pong_ms.load(Ordering::Relaxed)
        ));

        output.push_str(&format!(
            "# HELP mm_supervisor_stale_count Consecutive stale data readings\n\
             # TYPE mm_supervisor_stale_count gauge\n\
             mm_supervisor_stale_count{{{}}} {}\n",
            labels,
            self.inner.supervisor_stale_count.load(Ordering::Relaxed)
        ));

        output.push_str(&format!(
            "# HELP mm_supervisor_reconnect_signals_total Total reconnect signals from supervisor\n\
             # TYPE mm_supervisor_reconnect_signals_total counter\n\
             mm_supervisor_reconnect_signals_total{{{}}} {}\n",
            labels,
            self.inner.supervisor_reconnect_signals.load(Ordering::Relaxed)
        ));

        // Data quality metrics
        output.push_str(&format!(
            "# HELP mm_data_quality_issues_total Total data quality issues detected\n\
             # TYPE mm_data_quality_issues_total counter\n\
             mm_data_quality_issues_total{{{}}} {}\n",
            labels,
            self.inner.data_quality_issues_total.load(Ordering::Relaxed)
        ));

        output.push_str(&format!(
            "# HELP mm_message_loss_count Cumulative sequence gaps detected\n\
             # TYPE mm_message_loss_count counter\n\
             mm_message_loss_count{{{}}} {}\n",
            labels,
            self.inner.message_loss_count.load(Ordering::Relaxed)
        ));

        output.push_str(&format!(
            "# HELP mm_crossed_book_incidents Crossed book incidents detected\n\
             # TYPE mm_crossed_book_incidents counter\n\
             mm_crossed_book_incidents{{{}}} {}\n",
            labels,
            self.inner.crossed_book_incidents.load(Ordering::Relaxed)
        ));

        // Exchange Position Limits metrics
        let limits_valid = self.inner.exchange_limits_valid.load(Ordering::Relaxed) == 1;
        if limits_valid {
            output.push_str(&format!(
                "# HELP mm_exchange_max_long Maximum long position allowed by exchange\n\
                 # TYPE mm_exchange_max_long gauge\n\
                 mm_exchange_max_long{{{}}} {}\n",
                labels,
                self.inner.exchange_max_long.load()
            ));

            output.push_str(&format!(
                "# HELP mm_exchange_max_short Maximum short position allowed by exchange\n\
                 # TYPE mm_exchange_max_short gauge\n\
                 mm_exchange_max_short{{{}}} {}\n",
                labels,
                self.inner.exchange_max_short.load()
            ));

            output.push_str(&format!(
                "# HELP mm_exchange_available_buy Available capacity to buy\n\
                 # TYPE mm_exchange_available_buy gauge\n\
                 mm_exchange_available_buy{{{}}} {}\n",
                labels,
                self.inner.exchange_available_buy.load()
            ));

            output.push_str(&format!(
                "# HELP mm_exchange_available_sell Available capacity to sell\n\
                 # TYPE mm_exchange_available_sell gauge\n\
                 mm_exchange_available_sell{{{}}} {}\n",
                labels,
                self.inner.exchange_available_sell.load()
            ));

            output.push_str(&format!(
                "# HELP mm_exchange_effective_bid Effective bid limit (min of local and exchange)\n\
                 # TYPE mm_exchange_effective_bid gauge\n\
                 mm_exchange_effective_bid{{{}}} {}\n",
                labels,
                self.inner.exchange_effective_bid.load()
            ));

            output.push_str(&format!(
                "# HELP mm_exchange_effective_ask Effective ask limit (min of local and exchange)\n\
                 # TYPE mm_exchange_effective_ask gauge\n\
                 mm_exchange_effective_ask{{{}}} {}\n",
                labels,
                self.inner.exchange_effective_ask.load()
            ));

            output.push_str(&format!(
                "# HELP mm_exchange_limits_age_ms Age of exchange limits data in milliseconds\n\
                 # TYPE mm_exchange_limits_age_ms gauge\n\
                 mm_exchange_limits_age_ms{{{}}} {}\n",
                labels,
                self.inner.exchange_limits_age_ms.load(Ordering::Relaxed)
            ));
        }

        output.push_str(&format!(
            "# HELP mm_exchange_limits_valid Whether exchange limits are valid (1=yes, 0=no)\n\
             # TYPE mm_exchange_limits_valid gauge\n\
             mm_exchange_limits_valid{{{}}} {}\n",
            labels,
            if limits_valid { 1 } else { 0 }
        ));

        // Calibration fill rate controller metrics
        output.push_str(&format!(
            "# HELP mm_calibration_gamma_mult Fill-hungry gamma multiplier [0.3=max hunger, 1.0=calibrated]\n\
             # TYPE mm_calibration_gamma_mult gauge\n\
             mm_calibration_gamma_mult{{{}}} {}\n",
            labels,
            self.inner.calibration_gamma_mult.load()
        ));

        output.push_str(&format!(
            "# HELP mm_calibration_progress Calibration progress [0.0=cold start, 1.0=complete]\n\
             # TYPE mm_calibration_progress gauge\n\
             mm_calibration_progress{{{}}} {}\n",
            labels,
            self.inner.calibration_progress.load()
        ));

        output.push_str(&format!(
            "# HELP mm_calibration_fill_count Fills in lookback window for rate calculation\n\
             # TYPE mm_calibration_fill_count gauge\n\
             mm_calibration_fill_count{{{}}} {}\n",
            labels,
            self.inner.calibration_fill_count.load(Ordering::Relaxed)
        ));

        let calibration_complete = self.inner.calibration_complete.load(Ordering::Relaxed) == 1;
        output.push_str(&format!(
            "# HELP mm_calibration_complete Whether calibration is complete (1=yes, 0=no)\n\
             # TYPE mm_calibration_complete gauge\n\
             mm_calibration_complete{{{}}} {}\n",
            labels,
            if calibration_complete { 1 } else { 0 }
        ));

        // Impulse Control metrics
        let impulse_enabled = self.inner.impulse_control_enabled.load(Ordering::Relaxed) == 1;
        output.push_str(&format!(
            "# HELP mm_impulse_control_enabled Whether impulse control is active (1=yes, 0=no)\n\
             # TYPE mm_impulse_control_enabled gauge\n\
             mm_impulse_control_enabled{{{}}} {}\n",
            labels,
            if impulse_enabled { 1 } else { 0 }
        ));

        output.push_str(&format!(
            "# HELP mm_impulse_budget_available Available execution budget tokens\n\
             # TYPE mm_impulse_budget_available gauge\n\
             mm_impulse_budget_available{{{}}} {}\n",
            labels,
            self.inner.impulse_budget_available.load()
        ));

        output.push_str(&format!(
            "# HELP mm_impulse_budget_utilization Budget utilization ratio (spent / total)\n\
             # TYPE mm_impulse_budget_utilization gauge\n\
             mm_impulse_budget_utilization{{{}}} {}\n",
            labels,
            self.inner.impulse_budget_utilization.load()
        ));

        output.push_str(&format!(
            "# HELP mm_impulse_budget_earned Total tokens earned from fills\n\
             # TYPE mm_impulse_budget_earned counter\n\
             mm_impulse_budget_earned{{{}}} {}\n",
            labels,
            self.inner.impulse_budget_earned.load()
        ));

        output.push_str(&format!(
            "# HELP mm_impulse_budget_spent Total tokens spent on actions\n\
             # TYPE mm_impulse_budget_spent counter\n\
             mm_impulse_budget_spent{{{}}} {}\n",
            labels,
            self.inner.impulse_budget_spent.load()
        ));

        output.push_str(&format!(
            "# HELP mm_impulse_filter_blocked Actions blocked by delta-lambda filter\n\
             # TYPE mm_impulse_filter_blocked counter\n\
             mm_impulse_filter_blocked{{{}}} {}\n",
            labels,
            self.inner.impulse_filter_blocked.load(Ordering::Relaxed)
        ));

        output.push_str(&format!(
            "# HELP mm_impulse_queue_locked Actions blocked by queue lock (high P fill)\n\
             # TYPE mm_impulse_queue_locked counter\n\
             mm_impulse_queue_locked{{{}}} {}\n",
            labels,
            self.inner.impulse_queue_locked.load(Ordering::Relaxed)
        ));

        output.push_str(&format!(
            "# HELP mm_impulse_budget_skipped Cycles skipped due to insufficient budget\n\
             # TYPE mm_impulse_budget_skipped counter\n\
             mm_impulse_budget_skipped{{{}}} {}\n",
            labels,
            self.inner.impulse_budget_skipped.load(Ordering::Relaxed)
        ));

        // Learned Parameters metrics
        output.push_str(&format!(
            "# HELP mm_learned_alpha_touch Learned informed trader probability at touch [0,1]\n\
             # TYPE mm_learned_alpha_touch gauge\n\
             mm_learned_alpha_touch{{{}}} {:.4}\n",
            labels,
            self.inner.learned_alpha_touch.load()
        ));

        output.push_str(&format!(
            "# HELP mm_learned_kappa Learned fill intensity\n\
             # TYPE mm_learned_kappa gauge\n\
             mm_learned_kappa{{{}}} {:.2}\n",
            labels,
            self.inner.learned_kappa.load()
        ));

        output.push_str(&format!(
            "# HELP mm_learned_spread_floor_bps Learned spread floor in bps\n\
             # TYPE mm_learned_spread_floor_bps gauge\n\
             mm_learned_spread_floor_bps{{{}}} {:.2}\n",
            labels,
            self.inner.learned_spread_floor_bps.load()
        ));

        output.push_str(&format!(
            "# HELP mm_learned_params_observations Total observations for learned parameters\n\
             # TYPE mm_learned_params_observations counter\n\
             mm_learned_params_observations{{{}}} {}\n",
            labels,
            self.inner.learned_params_observations.load(Ordering::Relaxed)
        ));

        let learned_calibrated = self.inner.learned_params_calibrated.load(Ordering::Relaxed) == 1;
        output.push_str(&format!(
            "# HELP mm_learned_params_calibrated Whether learned params tier1 is calibrated (1=yes, 0=no)\n\
             # TYPE mm_learned_params_calibrated gauge\n\
             mm_learned_params_calibrated{{{}}} {}\n",
            labels,
            if learned_calibrated { 1 } else { 0 }
        ));

        output
    }

    /// Get metrics as a HashMap for JSON serialization.
    pub fn to_map(&self) -> HashMap<String, f64> {
        let mut map = HashMap::new();

        map.insert("position".to_string(), self.inner.position.load());
        map.insert(
            "inventory_utilization".to_string(),
            self.inner.inventory_utilization.load(),
        );
        map.insert("daily_pnl".to_string(), self.inner.daily_pnl.load());
        map.insert("drawdown_pct".to_string(), self.inner.drawdown_pct.load());
        map.insert(
            "orders_placed".to_string(),
            self.inner.orders_placed.load(Ordering::Relaxed) as f64,
        );
        map.insert(
            "orders_filled".to_string(),
            self.inner.orders_filled.load(Ordering::Relaxed) as f64,
        );
        map.insert("fill_volume".to_string(), self.inner.fill_volume.load());
        map.insert("mid_price".to_string(), self.inner.mid_price.load());
        map.insert("spread_bps".to_string(), self.inner.spread_bps.load());
        map.insert("sigma".to_string(), self.inner.sigma.load());
        map.insert("jump_ratio".to_string(), self.inner.jump_ratio.load());
        map.insert("kappa".to_string(), self.inner.kappa.load());
        map.insert(
            "kill_switch_triggered".to_string(),
            self.inner.kill_switch_triggered.load(Ordering::Relaxed) as f64,
        );
        map.insert(
            "cascade_severity".to_string(),
            self.inner.cascade_severity.load(),
        );
        map.insert(
            "adverse_selection_bps".to_string(),
            self.inner.adverse_selection_bps.load(),
        );
        map.insert(
            "data_staleness_secs".to_string(),
            self.inner.data_staleness_secs.load(),
        );
        map.insert(
            "uptime_secs".to_string(),
            self.inner.start_time.elapsed().as_secs_f64(),
        );
        map.insert(
            "websocket_connected".to_string(),
            self.inner.websocket_connected.load(Ordering::Relaxed) as f64,
        );
        map.insert(
            "last_trade_age_ms".to_string(),
            self.inner.last_trade_age_ms.load(Ordering::Relaxed) as f64,
        );
        map.insert(
            "last_book_age_ms".to_string(),
            self.inner.last_book_age_ms.load(Ordering::Relaxed) as f64,
        );
        map.insert(
            "data_quality_issues_total".to_string(),
            self.inner.data_quality_issues_total.load(Ordering::Relaxed) as f64,
        );
        map.insert(
            "message_loss_count".to_string(),
            self.inner.message_loss_count.load(Ordering::Relaxed) as f64,
        );
        map.insert(
            "crossed_book_incidents".to_string(),
            self.inner.crossed_book_incidents.load(Ordering::Relaxed) as f64,
        );
        // Kelly-Stochastic metrics
        map.insert(
            "kelly_stochastic_enabled".to_string(),
            self.inner.kelly_stochastic_enabled.load(Ordering::Relaxed) as f64,
        );
        map.insert(
            "kelly_alpha_touch".to_string(),
            self.inner.kelly_alpha_touch.load(),
        );
        map.insert(
            "kelly_fraction".to_string(),
            self.inner.kelly_fraction.load(),
        );
        map.insert(
            "kelly_alpha_decay_bps".to_string(),
            self.inner.kelly_alpha_decay_bps.load(),
        );

        // Impulse Control metrics
        map.insert(
            "impulse_control_enabled".to_string(),
            self.inner.impulse_control_enabled.load(Ordering::Relaxed) as f64,
        );
        map.insert(
            "impulse_budget_available".to_string(),
            self.inner.impulse_budget_available.load(),
        );
        map.insert(
            "impulse_budget_utilization".to_string(),
            self.inner.impulse_budget_utilization.load(),
        );
        map.insert(
            "impulse_filter_blocked".to_string(),
            self.inner.impulse_filter_blocked.load(Ordering::Relaxed) as f64,
        );
        map.insert(
            "impulse_queue_locked".to_string(),
            self.inner.impulse_queue_locked.load(Ordering::Relaxed) as f64,
        );

        // Learned Parameters metrics
        map.insert(
            "learned_alpha_touch".to_string(),
            self.inner.learned_alpha_touch.load(),
        );
        map.insert(
            "learned_kappa".to_string(),
            self.inner.learned_kappa.load(),
        );
        map.insert(
            "learned_spread_floor_bps".to_string(),
            self.inner.learned_spread_floor_bps.load(),
        );
        map.insert(
            "learned_params_observations".to_string(),
            self.inner.learned_params_observations.load(Ordering::Relaxed) as f64,
        );
        map.insert(
            "learned_params_calibrated".to_string(),
            self.inner.learned_params_calibrated.load(Ordering::Relaxed) as f64,
        );

        map
    }

    /// Get a summary struct.
    pub fn summary(&self) -> MetricsSummary {
        MetricsSummary {
            position: self.inner.position.load(),
            inventory_utilization: self.inner.inventory_utilization.load(),
            daily_pnl: self.inner.daily_pnl.load(),
            drawdown_pct: self.inner.drawdown_pct.load(),
            orders_placed: self.inner.orders_placed.load(Ordering::Relaxed),
            orders_filled: self.inner.orders_filled.load(Ordering::Relaxed),
            fill_volume: self.inner.fill_volume.load(),
            mid_price: self.inner.mid_price.load(),
            spread_bps: self.inner.spread_bps.load(),
            sigma: self.inner.sigma.load(),
            jump_ratio: self.inner.jump_ratio.load(),
            kappa: self.inner.kappa.load(),
            kill_switch_triggered: self.inner.kill_switch_triggered.load(Ordering::Relaxed) == 1,
            cascade_severity: self.inner.cascade_severity.load(),
            adverse_selection_bps: self.inner.adverse_selection_bps.load(),
            uptime_secs: self.inner.start_time.elapsed().as_secs_f64(),
            websocket_connected: self.inner.websocket_connected.load(Ordering::Relaxed) == 1,
            last_trade_age_ms: self.inner.last_trade_age_ms.load(Ordering::Relaxed),
            last_book_age_ms: self.inner.last_book_age_ms.load(Ordering::Relaxed),
            data_quality_issues_total: self.inner.data_quality_issues_total.load(Ordering::Relaxed),
            message_loss_count: self.inner.message_loss_count.load(Ordering::Relaxed),
            crossed_book_incidents: self.inner.crossed_book_incidents.load(Ordering::Relaxed),
            // Kelly-Stochastic
            kelly_stochastic_enabled: self.inner.kelly_stochastic_enabled.load(Ordering::Relaxed)
                == 1,
            kelly_alpha_touch: self.inner.kelly_alpha_touch.load(),
            kelly_fraction: self.inner.kelly_fraction.load(),
            kelly_alpha_decay_bps: self.inner.kelly_alpha_decay_bps.load(),
        }
    }

    /// Get dashboard state for web UI.
    ///
    /// Creates a JSON-serializable dashboard state from current metrics.
    /// This provides a snapshot suitable for the real-time dashboard.
    /// Uses the dashboard aggregator to include fill history and calibration data.
    pub fn to_dashboard_state(&self) -> DashboardState {
        let mid_price = self.inner.mid_price.load();
        let spread_bps = self.inner.spread_bps.load();
        let position = self.inner.position.load();
        let kappa = self.inner.kappa.load();
        let sigma = self.inner.sigma.load();
        let jump_ratio = self.inner.jump_ratio.load();
        let cascade_severity = self.inner.cascade_severity.load();
        let adverse_selection_bps = self.inner.adverse_selection_bps.load();
        let _daily_pnl = self.inner.daily_pnl.load();

        // Estimate fill probability and adverse probability from available data
        // For fill probability, use a simple heuristic based on spread
        let fill_prob = (0.3 / (1.0 + spread_bps / 10.0)).clamp(0.05, 0.5);
        // For adverse probability, use the AS estimator's output
        let adverse_prob = (adverse_selection_bps / 10.0).clamp(0.0, 1.0);

        // Current gamma (use kelly fraction as proxy, or default)
        let gamma = self.inner.kelly_fraction.load().max(0.1);

        // P&L attribution from accumulated fill data
        // These are tracked per-fill in FillProcessor and accumulated in DashboardAggregator
        let spread_capture = self.dashboard.total_spread_capture();
        let adverse_selection = self.dashboard.total_adverse_selection();
        let inventory_cost = self.dashboard.total_inventory_cost();
        let fees = self.dashboard.total_fees();

        // Use dashboard aggregator snapshot to get fills, calibration, and regime history
        let params = DashboardSnapshotParams {
            mid_price,
            spread_bps,
            position,
            kappa,
            gamma,
            cascade_severity,
            jump_ratio,
            sigma,
            fill_prob: fill_prob * 100.0, // As percentage
            adverse_prob: adverse_prob * 100.0,
            spread_capture,
            adverse_selection,
            inventory_cost,
            fees,
        };
        self.dashboard.snapshot(&params)
    }
}
