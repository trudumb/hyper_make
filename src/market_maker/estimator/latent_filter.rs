//! Latent State Filter (UKF / IMM Approximation)
//!
//! Replaces the scalar mid-price and isolated drift estimators with a joint filter
//! tracking Latent Fair Value (V), Instantaneous Drift (μ), and Realized Volatility (σ).
//!
//! Includes Glosten-Milgrom updates to incorporate our own fills as 
//! adversarial information.

use tracing::debug;

/// Represents the joint belief of the market state.
#[derive(Debug, Clone, Default)]
pub struct LatentState {
    /// Latent Fair Value (V)
    pub v: f64,
    /// Instantaneous Drift (μ) in bps/s
    pub mu: f64,
    /// Realized Volatility (σ) in bps/√s
    pub sigma: f64,
}

#[derive(Debug, Clone)]
pub struct LatentStateFilter {
    // State Means
    pub v: f64,
    pub mu: f64,
    pub sigma: f64,

    // State Covariance (approx 3x3 diagonal for simplicity initially)
    pub p_v: f64,
    pub p_mu: f64,
    pub p_sigma: f64,

    // Process Noise
    q_v: f64,
    q_mu: f64,
    q_sigma: f64,

    last_update_ms: u64,
    is_warmed_up: bool,
    update_count: u64,
}

impl Default for LatentStateFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl LatentStateFilter {
    pub fn new() -> Self {
        Self {
            v: 0.0,
            mu: 0.0,
            sigma: 0.025, // default assumption (e.g. 0.025 bps/sqrt(s))
            p_v: 1e6, // High initial uncertainty
            p_mu: 1e3,
            p_sigma: 1e1,
            q_v: 1e-4, // base process noise for V
            q_mu: 1e-6,
            q_sigma: 1e-6,
            last_update_ms: 0,
            is_warmed_up: false,
            update_count: 0,
        }
    }

    /// Predict the future state based on dt (seconds).
    fn predict(&mut self, dt: f64) {
        if dt <= 0.0 {
            return;
        }

        // V(t) = V(t-1) * (1 + μ*dt) (approx for small dt)
        if self.v > 0.0 {
            // Apply drift to latent value
            let drift_return = (self.mu / 10000.0) * dt; 
            self.v *= 1.0 + drift_return;
        }

        // Covariance inflation (Predict)
        // V variance scales with dt and current vol
        let dynamic_q_v = self.q_v + (self.sigma / 10000.0).powi(2) * dt * self.v.powi(2);
        self.p_v += dynamic_q_v;
        self.p_mu += self.q_mu * dt;
        self.p_sigma += self.q_sigma * dt;
    }

    /// Update with an observation of Microprice (z_micro) with associated variance R.
    pub fn update_microprice(&mut self, timestamp_ms: u64, z_micro: f64, r_noise: f64) {
        if self.v == 0.0 {
            self.v = z_micro;
            self.last_update_ms = timestamp_ms;
            return;
        }

        let dt = (timestamp_ms.saturating_sub(self.last_update_ms)) as f64 / 1000.0;
        self.predict(dt);

        // Kalman Update for V
        // Measurement residual
        let y = z_micro - self.v;
        // Innovation covariance
        let s = self.p_v + r_noise;
        if s > 0.0 {
            // Kalman gain
            let k = self.p_v / s;
            // Update state
            self.v += k * y;
            // Update covariance
            self.p_v *= 1.0 - k;
            
            // Also update mu based on the innovation if dt > 0
            if dt > 0.1 {
                let implied_mu = (y / self.v) * 10000.0 / dt;
                let k_mu = 0.05; // fixed weak gain for drift
                self.mu += k_mu * (implied_mu - self.mu);
                self.p_mu *= 1.0 - k_mu;
            }
        }

        self.last_update_ms = timestamp_ms;
        self.update_count += 1;
        if self.update_count > 100 {
            self.is_warmed_up = true;
        }
    }

    /// Treat our own fill as an adversarial Glosten-Milgrom observation.
    /// If an ask is filled, the true value is likely higher (informed buy).
    /// If a bid is filled, the true value is likely lower (informed sell).
    pub fn glosten_milgrom_update(&mut self, timestamp_ms: u64, is_buy: bool, p_informed: f64, spread_bps: f64) {
        if self.v <= 0.0 {
            return;
        }
        
        // P(informed) is the fraction of flow that knows the true value V.
        // E[V | Ask Lifted] = V + (P_inf * Spread/2)
        // E[V | Bid Hit] = V - (P_inf * Spread/2)
        
        let shift_bps = p_informed * (spread_bps / 2.0);
        let shift_factor = 1.0 + (if is_buy { -shift_bps } else { shift_bps } / 10000.0);
        
        self.v *= shift_factor;
        
        // Decrease confidence in V because adversarial selection just occurred
        self.p_v *= 1.1; 
        
        debug!(
            is_bid_filled = is_buy,
            p_informed = %format!("{:.2}", p_informed),
            shift_bps = %format!("{:.2}", if is_buy { -shift_bps} else { shift_bps }),
            new_v = %format!("{:.2}", self.v),
            "Glosten-Milgrom Latent State Shift"
        );
        
        self.last_update_ms = timestamp_ms;
    }

    pub fn update_volatility(&mut self, observed_sigma: f64) {
        // Simple EWMA update for sigma inside the filter
        let alpha = 0.1;
        self.sigma = alpha * observed_sigma + (1.0 - alpha) * self.sigma;
    }

    pub fn state(&self) -> LatentState {
        LatentState {
            v: self.v,
            mu: self.mu,
            sigma: self.sigma,
        }
    }

    pub fn is_warmed_up(&self) -> bool {
        self.is_warmed_up
    }
}
