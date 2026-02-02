# First-Principles Stochastic Control Grounding

**Goal**: Move from heuristic-based to mathematically-grounded market making
**Status**: Planning
**Theoretical Foundation**: HJB Optimal Control + Bayesian Inference

---

## Mathematical Framework

### 1. Price Process (Jump-Diffusion)
```
dS_t = κ(θ - S_t)dt + σ(R_t)dW_t + J_t dN_t
```

| Component | Current Implementation | First-Principles Requirement |
|-----------|----------------------|------------------------------|
| κ (mean-reversion) | Linear, constant | Threshold-dependent: κ(δ) |
| σ(R_t) | HMM regime-dependent | ✅ Implemented |
| J_t dN_t (jumps) | soft_jump.rs detection | Needs Poisson intensity λ_J |

### 2. Order Arrival Process (Intensity)
```
λ(δ) = A × e^(-k×δ)
```

| Component | Current Implementation | First-Principles Requirement |
|-----------|----------------------|------------------------------|
| A (activity) | Hawkes μ + excitation | ✅ Implemented (process_models/hawkes.rs) |
| k (liquidity) | Implicit in kappa | Needs explicit estimation |
| Conditional λ | Independent fills | **GAP**: Needs Hawkes for fill clustering |

### 3. Utility Maximization (HJB)
```
r(s, q, t) = s - q×γ×σ²×(T-t)
```

| Component | Current Implementation | First-Principles Requirement |
|-----------|----------------------|------------------------------|
| Reservation price | GLFT formula | ✅ Implemented |
| γ (risk aversion) | Config + RL tuning | ✅ Being learned |
| T-t (horizon) | Implicit in decay | Needs explicit terminal time |

---

## Gap Analysis & Fixes

### Gap 1: Stationary Hazard Rate in BOCPD

**Current**: `hazard_rate: 0.005` (constant)

**First-Principles**: Hazard rate should be a function of market stress:
```
h(t) = h_base × (1 + α×VPIN + β×hawkes_intensity + γ×size_anomaly)
```

**File**: `src/market_maker/estimator/bocpd_kappa.rs`

**Implementation**:
```rust
pub struct AdaptiveHazard {
    base_rate: f64,
    vpin_sensitivity: f64,      // α
    intensity_sensitivity: f64,  // β
    anomaly_sensitivity: f64,    // γ
}

impl AdaptiveHazard {
    /// Compute time-varying hazard rate.
    /// Higher market stress = higher probability of regime change.
    pub fn hazard(&self, vpin: f64, hawkes_intensity: f64, size_anomaly: f64) -> f64 {
        let stress = 1.0
            + self.vpin_sensitivity * vpin
            + self.intensity_sensitivity * (hawkes_intensity - 1.0).max(0.0)
            + self.anomaly_sensitivity * size_anomaly;

        (self.base_rate * stress).min(0.1) // Cap at 10%
    }
}
```

**Integration**: Pass VPIN and Hawkes intensity to BOCPD each observation.

---

### Gap 2: Threshold-Dependent Kappa (TAR Model)

**Current**: κ is treated as a linear coefficient for fill intensity.

**First-Principles**: Crypto markets exhibit threshold behavior:
- Small deviations (|δ| < threshold): Mean reversion dominates (high κ)
- Large deviations (|δ| > threshold): Momentum dominates (κ → 0)

**Mathematical Model** (Threshold AutoRegressive):
```
κ_eff(δ) = κ_base × exp(-decay × max(0, |δ| - threshold))
```

**File**: Create `src/market_maker/estimator/threshold_kappa.rs`

**Implementation**:
```rust
/// Threshold-dependent kappa following TAR model.
/// Captures the transition from mean-reversion to momentum regimes.
pub struct ThresholdKappa {
    /// Base kappa for small deviations
    kappa_base: f64,
    /// Deviation threshold (in bps) where momentum begins
    threshold_bps: f64,
    /// Decay rate beyond threshold
    decay_rate: f64,
    /// EMA of recent returns for deviation tracking
    return_ema: f64,
    /// EMA alpha
    alpha: f64,
}

impl ThresholdKappa {
    /// Update with new return observation.
    pub fn update(&mut self, return_bps: f64) {
        self.return_ema = self.alpha * return_bps + (1.0 - self.alpha) * self.return_ema;
    }

    /// Compute effective kappa given current deviation.
    pub fn kappa_effective(&self) -> f64 {
        let deviation = self.return_ema.abs();

        if deviation < self.threshold_bps {
            // Mean-reversion regime
            self.kappa_base
        } else {
            // Momentum regime: kappa decays exponentially
            let excess = deviation - self.threshold_bps;
            self.kappa_base * (-self.decay_rate * excess / 100.0).exp()
        }
    }

    /// Returns regime classification for logging.
    pub fn regime(&self) -> KappaRegime {
        let deviation = self.return_ema.abs();
        if deviation < self.threshold_bps * 0.5 {
            KappaRegime::StrongMeanReversion
        } else if deviation < self.threshold_bps {
            KappaRegime::WeakMeanReversion
        } else if deviation < self.threshold_bps * 2.0 {
            KappaRegime::WeakMomentum
        } else {
            KappaRegime::StrongMomentum
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KappaRegime {
    StrongMeanReversion,
    WeakMeanReversion,
    WeakMomentum,
    StrongMomentum,
}
```

**Integration**: Add to KappaOrchestrator as another source with regime-dependent weighting.

---

### Gap 3: Conditional Fill Intensity (Hawkes for Fills)

**Current**: Each fill is treated as independent. BOCPD observes (features, kappa) pairs.

**First-Principles**: Fills are autocorrelated. Toxic flow comes in waves.
```
λ_fill(t) = μ + Σ α × exp(-β × (t - t_i)) for all past fills t_i
```

**File**: Modify `src/market_maker/process_models/hawkes.rs` or create `fill_hawkes.rs`

**Implementation**:
```rust
/// Hawkes process specifically for own fill arrivals.
/// Captures clustering of fills (momentum traders hit us repeatedly).
pub struct FillHawkesProcess {
    /// Baseline fill rate (fills per minute in quiet markets)
    mu: f64,
    /// Self-excitation: each fill increases probability of next
    alpha: f64,
    /// Decay rate (how fast excitation fades)
    beta: f64,
    /// Recent fill timestamps
    fill_times_ms: VecDeque<i64>,
    /// Max history (ms)
    max_history_ms: i64,
}

impl FillHawkesProcess {
    /// Record a fill event.
    pub fn on_fill(&mut self, timestamp_ms: i64, is_adverse: bool) {
        // Adverse fills have higher excitation (toxic flow clusters)
        let excitation = if is_adverse { self.alpha * 1.5 } else { self.alpha };
        self.fill_times_ms.push_back(timestamp_ms);
        self.prune_old(timestamp_ms);
    }

    /// Current fill intensity (expected fills per minute).
    pub fn intensity(&self, now_ms: i64) -> f64 {
        let mut lambda = self.mu;

        for &t in &self.fill_times_ms {
            let dt_minutes = (now_ms - t) as f64 / 60_000.0;
            lambda += self.alpha * (-self.beta * dt_minutes).exp();
        }

        lambda
    }

    /// Probability of at least one fill in next dt minutes.
    pub fn p_fill_in_window(&self, now_ms: i64, window_ms: i64) -> f64 {
        let lambda = self.intensity(now_ms);
        let dt_minutes = window_ms as f64 / 60_000.0;

        // P(N > 0) = 1 - P(N = 0) = 1 - exp(-λ×dt)
        1.0 - (-lambda * dt_minutes).exp()
    }

    /// Intensity percentile (0-1) relative to typical range.
    /// Used for regime detection and spread adjustment.
    pub fn intensity_percentile(&self, now_ms: i64) -> f64 {
        let lambda = self.intensity(now_ms);
        // Sigmoid normalization: median at μ, saturates at 3μ
        let x = (lambda - self.mu) / self.mu;
        1.0 / (1.0 + (-2.0 * x).exp())
    }
}
```

**Integration**:
- Use `intensity_percentile()` to inform BOCPD hazard rate
- Use `p_fill_in_window()` for fill rate predictions in BeliefSnapshot

---

### Gap 4: Bayesian Evidence Update (P(Trade|Regime))

**Current**: Regime updates use HMM emissions but not Bayesian likelihood of specific trade characteristics.

**First-Principles**: When a 3σ trade occurs:
```
P(Cascade | Trade) ∝ P(Trade | Cascade) × P(Cascade)
                   ≫ P(Trade | Quiet) × P(Quiet)
```

**File**: Modify `src/market_maker/belief/central.rs`

**Implementation**:
```rust
/// Compute likelihood of trade given regime.
/// Large trades are much more likely in cascade/extreme regimes.
fn trade_likelihood(size_sigma: f64, regime: RegimeState) -> f64 {
    // Log-likelihood of observing this trade size given regime
    let expected_sigma = match regime {
        RegimeState::Low => 0.5,      // Expect small trades
        RegimeState::Normal => 1.0,   // Unit variance
        RegimeState::High => 2.0,     // Expect larger trades
        RegimeState::Extreme => 3.0,  // Expect very large trades
    };

    // Gaussian likelihood (simplified)
    let z = (size_sigma - expected_sigma) / 0.5;
    (-0.5 * z * z).exp()
}

/// Bayesian update of regime probabilities given trade observation.
pub fn update_regime_from_trade(&mut self, trade_size_sigma: f64) {
    let regimes = [RegimeState::Low, RegimeState::Normal, RegimeState::High, RegimeState::Extreme];
    let current_probs = self.regime_probs();

    // Compute posterior ∝ likelihood × prior
    let mut posteriors = [0.0; 4];
    let mut total = 0.0;

    for (i, &regime) in regimes.iter().enumerate() {
        let likelihood = trade_likelihood(trade_size_sigma, regime);
        posteriors[i] = likelihood * current_probs[i];
        total += posteriors[i];
    }

    // Normalize
    for p in &mut posteriors {
        *p /= total;
    }

    // Update internal state (blend with HMM for stability)
    const BAYESIAN_WEIGHT: f64 = 0.3;
    for i in 0..4 {
        self.regime_probs[i] = BAYESIAN_WEIGHT * posteriors[i]
                             + (1.0 - BAYESIAN_WEIGHT) * current_probs[i];
    }
}
```

---

## Extended BeliefSnapshot

Add first-principles fields to capture the mathematical state:

```rust
pub struct BeliefSnapshot {
    // Existing fields...

    // === First-Principles Extensions ===

    /// Threshold-dependent kappa regime
    pub kappa_regime: KappaRegime,
    /// Effective kappa after threshold adjustment
    pub kappa_threshold_adjusted: f64,

    /// Fill process intensity (Hawkes λ)
    pub fill_intensity: f64,
    /// Fill intensity percentile [0,1]
    pub fill_intensity_percentile: f64,
    /// P(fill in next 1 minute)
    pub p_fill_1m: f64,

    /// Adaptive hazard rate for BOCPD
    pub bocpd_hazard: f64,
    /// Market stress index driving hazard
    pub market_stress_index: f64,

    /// Trade size likelihood ratio (Cascade/Quiet)
    pub trade_size_likelihood_ratio: f64,
    /// Last trade size in sigmas
    pub last_trade_sigma: f64,
}
```

---

## Implementation Priority

| Priority | Component | Impact | Effort | Mathematical Grounding |
|----------|-----------|--------|--------|------------------------|
| P0 | Threshold Kappa (TAR) | HIGH | MEDIUM | Captures regime transition |
| P0 | Adaptive Hazard | HIGH | LOW | Non-stationary changepoints |
| P1 | Fill Hawkes | MEDIUM | MEDIUM | Autocorrelated toxicity |
| P1 | Bayesian Trade Update | MEDIUM | LOW | Proper likelihood |
| P2 | Explicit Terminal Time | LOW | LOW | Complete HJB formulation |

---

## Verification

### Mathematical Invariants to Test

```rust
#[test]
fn test_threshold_kappa_monotonic_decay() {
    let tk = ThresholdKappa::new(1000.0, 10.0, 0.5);
    // Kappa should decrease as deviation increases beyond threshold
    for deviation in [5.0, 10.0, 15.0, 20.0, 30.0] {
        let kappa = tk.kappa_at_deviation(deviation);
        // Should be monotonically decreasing beyond threshold
    }
}

#[test]
fn test_hawkes_intensity_clustering() {
    let mut hawkes = FillHawkesProcess::new(1.0, 0.5, 0.1);
    let now = 0i64;

    // Record burst of fills
    for i in 0..5 {
        hawkes.on_fill(now + i * 1000, false);
    }

    // Intensity should be elevated
    let lambda = hawkes.intensity(now + 5000);
    assert!(lambda > hawkes.mu * 2.0, "Intensity should cluster after burst");
}

#[test]
fn test_bayesian_regime_update() {
    let mut beliefs = CentralBeliefState::new();
    // Start with uniform priors

    // Observe 3σ trade
    beliefs.update_regime_from_trade(3.0);

    // P(Extreme) should increase significantly
    let probs = beliefs.regime_probs();
    assert!(probs[3] > 0.4, "Extreme regime prob should increase after 3σ trade");
}
```

### Live Validation

```bash
# Check threshold kappa in logs
grep "kappa_regime\|threshold_kappa" $LOG | tail -20

# Check adaptive hazard
grep "bocpd_hazard\|market_stress" $LOG | tail -20

# Check fill clustering detection
grep "fill_intensity\|intensity_percentile" $LOG | tail -20
```

---

## Summary: From Heuristics to First Principles

| Heuristic | First-Principles Replacement |
|-----------|------------------------------|
| "Widen spreads when VPIN high" | P(Cascade) × E[Loss|Cascade] enters HJB |
| "Constant changepoint probability" | h(t) = f(VPIN, λ_hawkes, σ_trade) |
| "Kappa is fill rate" | κ(δ) = κ_base × exp(-decay × excess_deviation) |
| "Fills are independent" | λ_fill(t) = μ + Σ α×exp(-β×dt) |
| "3σ trade = danger" | P(Regime|Trade) ∝ P(Trade|Regime) × P(Regime) |

The goal is not to remove intuition but to **ground intuition in mathematics** so that every decision can be traced to a probability distribution or utility function.
