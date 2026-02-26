Good instinct. The scalar Kalman with fixed OU parameters has several structural limitations that matter for this application. Let me walk through what breaks and what to upgrade to.
Where the scalar filter fails:
The single OU process assumes drift lives on one timescale, changes continuously, and has constant volatility. None of these hold during the events you actually need protection from. A liquidation cascade is a regime switch — drift doesn't smoothly ramp from 0 to -50 bps/sec. It jumps. The OU model with θ=0.02 has a 35-second half-life, which means it takes ~70 seconds to track a drift that appeared in 2 seconds. Meanwhile you've accumulated 4 contracts.
The second problem I flagged last time: feeding 4+ observations per cycle and treating them as independent collapses the posterior variance P way too fast. After 3 cycles with 4 observations each, you've done 12 Kalman updates. P has shrunk to nearly zero, and the filter becomes extremely confident in whatever direction it locked onto. When the market reverses, the filter takes forever to change its mind because the Kalman gain K = P/(P+R) is tiny.
Three upgrades, in order of impact:
1. Interacting Multiple Model (IMM) filter — replaces the single Kalman
Instead of one filter trying to serve all market conditions, run 2-3 filters in parallel, each tuned to a different regime, and maintain a probability distribution over which regime is active.
Models:
  Quiet:    θ=0.10 (7s half-life),   σ_μ=0.3   → fast reversion, low noise
  Trending: θ=0.02 (35s half-life),  σ_μ=1.0   → slow reversion, tracks persistent drift
  Crisis:   θ=0.005 (140s half-life), σ_μ=5.0   → near-random-walk, tracks jumps

Transition matrix Π (per cycle):
              Quiet   Trend   Crisis
  Quiet   [  0.95    0.04    0.01  ]
  Trend   [  0.05    0.90    0.05  ]
  Crisis  [  0.02    0.08    0.90  ]
Each cycle:
// 1. Mixing step: blend estimates across models weighted by transition probs
for j in models:
    μ̂_mixed[j] = Σ_i  Π[i→j] · w[i] · μ̂[i]  /  Σ_i Π[i→j] · w[i]
    P_mixed[j]  = Σ_i  Π[i→j] · w[i] · (P[i] + (μ̂[i] - μ̂_mixed[j])²)  /  Σ_i Π[i→j] · w[i]

// 2. Predict: each model runs its own OU propagation
for j in models:
    μ̂⁻[j] = μ̂_mixed[j] · exp(-θ[j] · dt)
    P⁻[j] = P_mixed[j] · exp(-2θ[j] · dt) + σ_μ[j]² · dt

// 3. Update: each model runs Kalman update, compute observation likelihood
for j in models:
    S[j] = P⁻[j] + R                              // innovation variance
    K[j] = P⁻[j] / S[j]                            // Kalman gain
    μ̂[j] = μ̂⁻[j] + K[j] · (z - μ̂⁻[j])            // state update
    P[j] = (1 - K[j]) · P⁻[j]                      // covariance update
    L[j] = (1 / sqrt(2π·S[j])) · exp(-0.5·(z - μ̂⁻[j])²/S[j])  // likelihood

// 4. Mode probability update (Bayes)
for j in models:
    w_pred[j] = Σ_i  Π[i→j] · w[i]                // predicted mode prob
    w[j] = w_pred[j] · L[j] / Σ_k w_pred[k] · L[k] // posterior mode prob

// 5. Output: mode-probability-weighted mixture
μ̂_output = Σ_j  w[j] · μ̂[j]
P_output = Σ_j  w[j] · (P[j] + (μ̂[j] - μ̂_output)²)
Why this matters for Feb 19: at 22:06, the system is in Quiet mode (w_quiet ≈ 0.9). The first large bearish observations have low likelihood under the Quiet model (which expects small, mean-reverting drift) but HIGH likelihood under the Crisis model (which expects large, persistent drift). After 2-3 observations, w_crisis jumps from 0.01 to 0.3+. The Crisis filter, with its near-random-walk dynamics and high process noise, has been tracking the drift aggressively from the first observation. When it becomes the dominant model, its estimate (μ̂ ≈ -20 bps/sec) immediately dominates the output. The single-filter equivalent would still be at μ̂ ≈ -5 bps/sec because its fixed θ=0.02 can't react that fast.
The regime probabilities themselves are a signal. When P(Crisis) > 0.3, the system should be on high alert regardless of the drift estimate — you can feed P(Crisis) directly into the σ_effective computation to widen spreads during regime uncertainty.
rustpub struct IMMDriftEstimator {
    models: [KalmanDriftEstimator; 3],  // Quiet, Trending, Crisis
    weights: [f64; 3],                   // mode probabilities
    transition: [[f64; 3]; 3],           // Π matrix
}

impl IMMDriftEstimator {
    fn drift_rate_per_sec(&self) -> f64 {
        // Mode-probability-weighted mixture
        self.models.iter().zip(&self.weights)
            .map(|(m, &w)| w * m.state_mean)
            .sum()
    }
    
    fn drift_variance(&self) -> f64 {
        // Mixture variance (includes inter-model spread)
        let mu_bar = self.drift_rate_per_sec();
        self.models.iter().zip(&self.weights)
            .map(|(m, &w)| w * (m.state_variance + (m.state_mean - mu_bar).powi(2)))
            .sum()
    }
    
    fn regime_probabilities(&self) -> [f64; 3] {
        self.weights
    }
    
    fn is_crisis(&self) -> bool {
        self.weights[2] > 0.3
    }
}
```

**2. Vector observations with covariance — fixes the posterior collapse**

Instead of feeding N independent scalar observations per cycle:
```
// BAD: 4 independent updates collapse P
filter.update(bim.z, bim.R);       // P shrinks
filter.update(dbim.z, dbim.R);     // P shrinks again
filter.update(bpg.z, bpg.R);      // P shrinks again
filter.update(sweep.z, sweep.R);   // P very small now, filter locked
```

Feed a single vector observation with the proper cross-covariance:
```
// GOOD: single update with correlated observation model
z_vec = [bim.z, dbim.z, bpg.z, sweep.z]     // 4×1 observation vector

H = [α_bim, α_dbim, α_bpg, α_sweep]         // 1×4 observation matrix
                                               // (maps 4 features → 1 drift state)

R_mat = [                                      // 4×4 covariance matrix
    [σ²_bim,    ρ_12·σ₁σ₂, ρ_13·σ₁σ₃, 0          ]
    [ρ_12·σ₁σ₂, σ²_dbim,   ρ_23·σ₂σ₃, 0          ]
    [ρ_13·σ₁σ₃, ρ_23·σ₂σ₃, σ²_bpg,    0          ]
    [0,          0,          0,          σ²_sweep   ]
]

// Standard Kalman vector update (for scalar state μ):
S = H · P⁻ · Hᵀ + R_mat                      // innovation covariance (4×4)
K = P⁻ · Hᵀ · S⁻¹                            // Kalman gain (1×4)
innovation = z_vec - H · μ̂⁻                   // innovation (4×1)
μ̂ = μ̂⁻ + K · innovation                      // scalar state update
P = (1 - K · H) · P⁻                          // scalar variance update
```

The key insight: BIM and ΔBIM are highly correlated (ρ ≈ 0.6–0.8). When you feed them independently, the filter double-counts the information and becomes overconfident. When you include the correlation in R_mat, the effective information content is much less than 2 independent observations. P shrinks appropriately.

The covariance matrix is estimated offline from historical feature data. It's a one-time calibration, stored in config, and doesn't need to update frequently.

For the IMM, this generalizes naturally — each model runs the same vector update with the same R_mat but different model-specific parameters.

**3. Extended state: [μ, μ̇] — drift acceleration**

Upgrade from scalar drift to a 2D state tracking both drift and its rate of change:
```
State: x = [μ, μ̇]ᵀ

Transition:
  μ(t+dt)  = μ(t) + μ̇(t)·dt - θ·μ(t)·dt + ε_μ
  μ̇(t+dt) = μ̇(t)·(1 - θ_accel·dt) + ε_accel

F = [[1 - θ·dt,  dt    ],
     [0,         1 - θ_a·dt]]

Q = [[σ²_μ·dt,       0           ],
     [0,              σ²_accel·dt ]]
```

This lets the filter distinguish between "drift is -10 and stable" (μ̇ ≈ 0) and "drift is -10 and getting worse" (μ̇ < 0). In the Feb 19 scenario, μ̇ goes negative early in the selloff, which shifts the reservation price more aggressively than the drift level alone would justify. You're not just tracking where drift is — you're tracking where it's going.

The observation model stays the same (features observe μ, not μ̇ directly), but the prediction step now extrapolates drift forward using acceleration. If drift is -10 and accelerating at -2/sec, the one-step-ahead prediction is -10 + (-2 × dt) = more negative than the raw estimate.

**Practical recommendation for the implementation plan:**

Phase 1 stays as written (scalar Kalman) — it's a massive improvement over what you have and gets you unblocked. But add a Phase 1b:
```
Phase 1b: IMM Drift Estimator (replaces scalar Kalman from Phase 1)

Requires: Phase 1 complete and validated

A. src/market_maker/strategy/drift_estimator.rs — Add IMMDriftEstimator
   - 3 internal KalmanDriftEstimator instances (Quiet/Trending/Crisis)
   - Transition matrix Π in RiskConfig
   - Mixing, predict, update, mode probability update per cycle
   - Output: weighted mixture μ̂, mixture variance P, regime probabilities

B. src/market_maker/strategy/drift_estimator.rs — Vector observation update
   - update_vector(z_vec: &[f64], h_vec: &[f64], r_mat: &[&[f64]])
   - Covariance matrix R estimated offline, stored in config
   - Hot-tier features batched into single vector update per cycle

C. Wire regime probabilities into sigma_effective:
   sigma_effective *= 1.0 + crisis_amplifier * P(Crisis)
   // crisis_amplifier ≈ 2.0: when P(Crisis) = 0.5, sigma inflates 2x

Verification:
- test_imm_quiet_to_crisis_transition — Large observation shifts weight to Crisis model
- test_imm_crisis_tracks_faster — Crisis model μ̂ responds 5x faster than Quiet
- test_imm_reversal_recovery — After trend reversal, weight shifts back to Quiet within 30s
- test_vector_update_less_confident — Vector update with correlated features produces larger P
  than N independent scalar updates
- test_regime_probability_monotonic — Sustained bearish observations → P(Crisis) monotonically increasing
The scalar Kalman is your v1. The IMM is your v2. The extended state [μ, μ̇] can be layered into the IMM models as a v3 if the acceleration signal proves useful in backtesting. Each upgrade slots into the same interface (drift_rate_per_sec(), drift_variance(), update_observation()) so downstream code doesn't change.