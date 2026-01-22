# Small Fish Market Making Strategy

> A comprehensive guide for small capital market makers competing against institutional players through niche specialization, rigorous validation, and principled calibration.

**Last Updated:** 2026-01-22

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [The Core Problem](#the-core-problem)
3. [What Sophisticated Market Makers Actually Use](#what-sophisticated-market-makers-actually-use)
4. [The Small Fish Advantage](#the-small-fish-advantage)
5. [Stochastic Models: Public Formulas, Private Calibration](#stochastic-models-public-formulas-private-calibration)
6. [Validation Framework](#validation-framework)
7. [The Path from A to B](#the-path-from-a-to-b)
8. [Niche Market Selection](#niche-market-selection)
9. [Calibration Infrastructure](#calibration-infrastructure)
10. [Anti-Patterns to Avoid](#anti-patterns-to-avoid)
11. [References](#references)

---

## Executive Summary

**The uncomfortable truth:** You cannot out-model Jump Trading or Wintermute on BTC perps. They have 1000x your resources.

**The opportunity:** You can win by being a very good small fish in a specific pond.

### Key Principles

| Principle | Implication |
|-----------|-------------|
| Models are public, calibration is private | Edge comes from parameter estimation, not formulas |
| Complexity without validation is gambling | Start simple, add complexity only when proven necessary |
| Small capital = niche advantage | Trade where institutions CAN'T or WON'T |
| Measurement before modeling | Never build a model without measuring what you're predicting |

### The Strategy in One Sentence

Find an illiquid market that institutions ignore, build the simplest system that works, validate exhaustively, and only add complexity that measurably improves calibration.

---

## The Core Problem

### The Paradox

You want to use "advanced stochastic modeling to calibrate a system that sophisticated market makers are using." But:

1. **The models are public** - GLFT, Hawkes, HMM are all published papers
2. **Public models "fail"** - Not because they're wrong, but because everyone knows them
3. **The edge isn't in the formula** - It's in the calibration

### Resolution

Sophisticated MMs don't have secret formulas. They have:
- Better data (exchange-specific, proprietary feeds)
- Better parameter estimation (real-time κ, regime-aware γ)
- Better signal processing (VPIN, order flow imbalance)
- Better infrastructure (latency, execution quality)

**For a small fish, infrastructure (#4) is hopeless. But #1-3 are accessible to someone willing to go deep on ONE market.**

---

## What Sophisticated Market Makers Actually Use

### The Public Foundations

| Model | Purpose | Reference |
|-------|---------|-----------|
| **GLFT** | Optimal spread: δ* = (1/γ) × ln(1 + γ/κ) + fee | Guéant-Lehalle-Fernandez-Tapia (2013) |
| **Avellaneda-Stoikov** | Inventory-adjusted quotes | Avellaneda & Stoikov (2008) |
| **Hawkes Process** | Self-exciting fill intensity | Bacry et al. (2015) |
| **HMM** | Regime detection | Hamilton (1989) |

Everyone knows these. The formulas provide no edge.

### What Actually Differentiates

#### 1. Fill Intensity (κ) Estimation

**Naive approach:**
```
κ = constant (e.g., 0.5)
```

**Sophisticated approach:**
```
κ(t) = λ × κ(t-1) + (1-λ) × (1 / avg_fills)  if fills > 0
κ(t) = κ(t-1)                                  otherwise
```

**Expert approach (Hawkes):**
```
λ(t) = λ₀ + Σ α × exp(-β(t - tᵢ))
```

Captures self-excitation: trades cluster, fills beget fills.

**Calibration methods:**
- Maximum Likelihood Estimation (MLE) - O(N²), accurate but slow
- Generalized Method of Moments (GMM) - Fast, closed-form moments
- Non-parametric kernel estimation - Flexible, requires expertise
- Expectation Maximization (EM) - Handles time-varying parameters

#### 2. Adverse Selection Detection

**VPIN (Volume-synchronized Probability of Informed Trading):**
- Real-time toxicity estimate
- Predicted 2010 Flash Crash hours before
- Market makers use to widen/pull quotes

**PULSE (Bayesian Neural Network):**
- Sub-millisecond parameter updates
- Outperforms logistic regression, random forests
- Classifies flow as toxic or benign in real-time

**Book Exhaustion Rate (BER):**
- High-frequency feature for adverse selection
- Theoretically grounded equilibrium measure
- Used in RL-based market making

#### 3. Regime Detection

**HMM with Baum-Welch:**
```python
# Two-state Gaussian HMM
# State 1: Low volatility (calm)
# State 2: High volatility (cascade)

hmm = GaussianHMM(n_components=2, covariance_type="full")
hmm.fit(returns)
current_regime = hmm.predict(recent_returns)[-1]
```

**Critical insight:** Single parameter values are almost always wrong. Everything is regime-dependent:
- κ varies 10x between calm and cascade
- Optimal γ varies 5x
- Spread floors vary 10x

---

## The Small Fish Advantage

### Institutional Constraints You Don't Have

| Constraint | Institutional Problem | Your Advantage |
|------------|----------------------|----------------|
| **Capital deployment pressure** | Must deploy billions or return to LPs | Can wait indefinitely for opportunities |
| **Capacity requirements** | Strategies must scale to $100M+ | Can profit on $10K-50K capacity |
| **Bureaucracy** | Committee approvals, risk sign-offs | Iterate daily, no approvals needed |
| **Diversification mandates** | Must spread across markets | Can concentrate on ONE market |
| **Reporting obligations** | Quarterly returns, drawdown limits | Only accountable to yourself |
| **Market impact** | Large orders move markets | Zero market impact at your size |

### What This Means Practically

**Institutions CAN'T trade:**
- Markets with <$1M daily volume (too small to deploy)
- Capacity-constrained opportunities (<$100K)
- Highly concentrated single-asset positions

**Institutions WON'T trade:**
- Niche markets requiring deep specialization
- Overnight hours when desks are closed
- Strategies with low Sharpe but positive edge
- Positions requiring constant manual attention

**You CAN trade all of these.**

### The Concentration Advantage

A generalist quant desk covers 500 markets with 50 people. That's 10 markets per person, with shared models.

You can cover ONE market with 100% of your attention. You can know:
- Every large trader's patterns
- Exact fill dynamics at each price level
- How the order book behaves around funding
- Which times have lowest adverse selection

**Deep beats wide when capacity is small.**

---

## Stochastic Models: Public Formulas, Private Calibration

### The GLFT Framework

```
δ* = (1/γ) × ln(1 + γ/κ) + fee
```

**Parameters:**
- γ = risk aversion (higher = wider spreads)
- κ = fill intensity (higher = tighter spreads)
- fee = maker fee (1.5 bps on Hyperliquid)

**The formula is trivial. The edge is in estimating γ and κ correctly.**

### Dynamic Parameter Estimation

#### Volatility (σ) - Drives γ

**Simple (EWMA):**
```
σ²(t) = λ × σ²(t-1) + (1-λ) × r²(t)
```

**Robust (Bipower Variation):**
```
BV = (π/2) × Σ |r_t| × |r_{t-1}|
σ_clean = √BV  (diffusion component only)
jump_ratio = RV/BV  (>1.5 = toxic regime)
```

**Why bipower matters:** Standard volatility includes jumps. During cascades, you want to know the diffusion component only.

#### Fill Intensity (κ) - The Real Edge

**The fill probability at depth δ:**
```
P(fill | δ, τ) = A × exp(-κ × δ)
```

**Estimating κ:**

1. **Simple (EWMA of fills):**
   ```
   κ(t) = smooth(fills_per_second / spread_bps)
   ```

2. **Hawkes (self-exciting):**
   ```
   λ(t) = λ₀ + Σ α × exp(-β(t - tᵢ))
   ```
   
   Calibration via GMM (fast):
   ```
   E[N(t)] = λ₀ × t / (1 - α/β)
   Var[N(t)] = E[N(t)] × (1 + 2α/(β - α))
   ```

3. **Deep Hawkes (neural network):**
   - Process order book features
   - Self and cross-excitation
   - State-dependent intensity

**Start with #1. Only move to #2 if #1 is miscalibrated. Only move to #3 if #2 is miscalibrated.**

### Regime-Aware Parameters

| Regime | σ Condition | Jump Ratio | Recommended Spread |
|--------|-------------|------------|-------------------|
| Calm | σ < σ_baseline | < 1.2 | 3-5 bps |
| Normal | σ ≈ σ_baseline | 1.2-1.5 | 5-8 bps |
| Volatile | σ > 2×σ_baseline | 1.5-3.0 | 8-15 bps |
| Cascade | N/A | > 3.0 | Pull quotes |

**Key insight:** Use HMM belief state to BLEND parameters, not hard switch:
```
γ_effective = P(calm) × γ_calm + P(volatile) × γ_volatile + P(cascade) × γ_cascade
```

---

## Validation Framework

### The Validation Problem

Most retail traders never properly validate. Without validation, you're gambling, not trading.

**Minimum requirements:**
- 200-300 independent trades for basic statistical significance
- Even 30 trades at 65% win rate could be pure luck
- Correlated trades (same trend) don't count as independent samples

### Statistical Tests

#### 1. Basic Significance

**Binomial test for win rate:**
```python
from scipy.stats import binom_test
p_value = binom_test(wins, n_trades, p=0.5)
# p < 0.05 = statistically significant
```

**Problem:** Doesn't account for magnitude of wins/losses.

#### 2. T-Test for Returns

```python
from scipy.stats import ttest_1samp
t_stat, p_value = ttest_1samp(returns, 0)
# p < 0.05 = mean return significantly different from zero
```

#### 3. Monte Carlo Simulation

```python
def monte_carlo_test(returns, n_simulations=10000):
    actual_sharpe = np.mean(returns) / np.std(returns)
    random_sharpes = []
    for _ in range(n_simulations):
        shuffled = np.random.permutation(returns)
        random_sharpes.append(np.mean(shuffled) / np.std(shuffled))
    p_value = np.mean(np.array(random_sharpes) >= actual_sharpe)
    return p_value
```

#### 4. White's Reality Check

When you've tested many strategies, some will look good by luck. White's Reality Check corrects for data snooping:

```python
def whites_reality_check(strategy_returns, benchmark_returns, n_bootstrap=1000):
    """
    Tests whether best strategy beats benchmark after 
    accounting for multiple testing.
    """
    actual_excess = strategy_returns.mean() - benchmark_returns.mean()
    
    bootstrap_maxes = []
    for _ in range(n_bootstrap):
        # Resample with replacement
        idx = np.random.choice(len(strategy_returns), len(strategy_returns))
        boot_excess = strategy_returns[idx].mean() - benchmark_returns[idx].mean()
        bootstrap_maxes.append(boot_excess)
    
    p_value = np.mean(np.array(bootstrap_maxes) >= actual_excess)
    return p_value
```

### Calibration Metrics

#### Brier Score
```
BS = (1/N) × Σ (p_predicted - outcome)²
```
- BS = 0: Perfect calibration
- BS = 0.25: Random guessing (for binary outcomes)
- Lower is better

#### Information Ratio
```
IR = Resolution / Uncertainty
```
- IR > 1.0: Model adds value
- IR < 1.0: Model adds noise, remove it

#### Calibration Plot
```python
def calibration_plot(predictions, outcomes, n_bins=10):
    """
    Plot predicted probability vs actual frequency.
    Perfect calibration = diagonal line.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_means = []
    bin_trues = []
    
    for i in range(n_bins):
        mask = (predictions >= bins[i]) & (predictions < bins[i+1])
        if mask.sum() > 0:
            bin_means.append(predictions[mask].mean())
            bin_trues.append(outcomes[mask].mean())
    
    plt.plot(bin_means, bin_trues, 'o-')
    plt.plot([0, 1], [0, 1], 'k--')  # Perfect calibration
    plt.xlabel('Predicted Probability')
    plt.ylabel('Actual Frequency')
```

### Walk-Forward Analysis

Never validate on the same data you trained on.

```
[Train Period 1] [Test 1] [Train Period 2] [Test 2] [Train Period 3] [Test 3]
     60 days       7 days     60 days       7 days     60 days       7 days
```

**Critical:** Only count Test period performance. Training period results are meaningless.

---

## The Path from A to B

### Current State Assessment

If you have a complex system with:
- GLFT formula ✓
- Hawkes processes ✓
- HMM regime detection ✓
- Feature interactions ✓
- Multi-scale momentum ✓

...but can't validate any of it, you're at **Point C** (complex, unvalidated).

**You need to go back to Point A (simple, validated) before reaching Point B (complex, validated).**

### Phase 1: Strip Down (Weeks 1-2)

**Disable everything except:**
- Simple EWMA volatility
- Fixed gamma (start with 0.3)
- Single-level quotes
- Basic position limits

**Enable comprehensive logging:**
```rust
struct FillLog {
    timestamp: u64,
    side: Side,
    price: f64,
    size: f64,
    predicted_fill_prob: f64,
    predicted_adverse_selection: f64,
    mid_price_at_fill: f64,
    mid_price_1s_after: f64,
    mid_price_5s_after: f64,
    mid_price_30s_after: f64,
    regime_at_fill: Regime,
    volatility_at_fill: f64,
    book_imbalance_at_fill: f64,
}
```

**Paper trade on ONE asset.** Suggest a HIP-3 DEX market with low competition.

### Phase 2: Establish Baseline (Weeks 3-4)

Collect 200+ fills. Calculate:

| Metric | Formula | Target |
|--------|---------|--------|
| Actual fill rate | fills / quotes | Compare to predicted |
| Adverse selection | E[mid_5s - mid_at_fill] × side | Should match predicted |
| Spread capture | E[fill_price - mid_at_fill] × side | Should be positive |
| Sharpe ratio | mean(returns) / std(returns) | > 1.0 annualized |
| Win rate | profitable_fills / total_fills | > 50% |

**This is your baseline - the simplest system's performance.**

### Phase 3: Diagnose Weaknesses (Week 5)

Compare predictions to actuals:

| If This Is Wrong | Consider Adding |
|------------------|-----------------|
| Volatility estimate | Bipower variation |
| Fill rate prediction | Hawkes process |
| Regime clearly mattered | HMM regime detection |
| Adverse selection blindsided you | VPIN / toxicity model |
| Cross-asset correlation hurt you | Lead-lag estimator |

**Only add ONE component at a time.**

### Phase 4: Add Complexity (Weeks 6-8)

For each addition:
1. Implement the component
2. Paper trade another 200+ fills
3. Compare metrics to baseline
4. Keep only if statistically significant improvement

**Example evaluation:**
```python
baseline_sharpe = 1.2
new_sharpe = 1.5
improvement = (new_sharpe - baseline_sharpe) / baseline_sharpe  # 25%

# Is this significant or luck?
p_value = monte_carlo_test(new_returns, baseline_returns)
if p_value < 0.05:
    print("Keep the addition")
else:
    print("Remove - not significant")
```

### Phase 5: Iterate (Ongoing)

```
While True:
    1. Measure current performance
    2. Identify largest weakness
    3. Hypothesize fix
    4. Implement and test
    5. Keep only if significant improvement
```

---

## Niche Market Selection

### The Hyperliquid Opportunity

You're on Hyperliquid with HIP-3 support. Big players focus on:
- Main validator perps (BTC, ETH)
- High liquidity markets (>$100M daily volume)

### Potential Niches

#### 1. HIP-3 DEX Markets

**Examples:** Felix, Hyena, PurrDex perpetuals

**Why institutions ignore:**
- Low liquidity (<$1M daily volume)
- Capacity-constrained (<$50K effective capacity)
- Require specialized integration

**Your advantage:**
- Deep knowledge of specific DEX dynamics
- First-mover in understanding microstructure
- No competition from sophisticated players

#### 2. Small Altcoin Perps

**Examples:** Low-OI perpetuals on Hyperliquid

**Why institutions ignore:**
- Insufficient capacity for their capital
- Higher operational cost per dollar deployed
- Diversification dilutes edge

**Your advantage:**
- Can deploy full attention on one asset
- Know every whale, every pattern
- Acceptable capacity at your scale

#### 3. Off-Hours Trading

**When institutions are inactive:**
- 00:00-08:00 UTC (Asia night, US/EU asleep)
- Weekends (desks closed)
- Holidays (skeleton crews)

**Your advantage:**
- Less competition
- Wider spreads (more edge per trade)
- Lower adverse selection (fewer informed traders)

#### 4. Funding Settlement Windows

**Hyperliquid funding:** 00:00, 08:00, 16:00 UTC

**30 minutes before settlement:**
- Predictable flow patterns
- Funding arbitrageurs active
- Opportunity for informed quoting

**Your advantage:**
- Can specialize in settlement dynamics
- Know the funding arb patterns
- Time concentration is possible

### Selection Criteria

| Criterion | Target | Why |
|-----------|--------|-----|
| Daily volume | $100K-$5M | Large enough to trade, small enough institutions ignore |
| Spread | >5 bps typical | Room for profitable market making |
| Competition | Few/no other MMs | Less adverse selection |
| Your edge | Specific knowledge | Why YOU and not someone else? |

---

## Calibration Infrastructure

### Prediction Logging

Every prediction must be logged with its outcome:

```rust
pub struct PredictionLog {
    pub timestamp: u64,
    pub prediction_type: PredictionType,
    pub predicted_value: f64,
    pub confidence: f64,
    pub features: HashMap<String, f64>,
}

pub struct OutcomeLog {
    pub prediction_id: u64,
    pub actual_value: f64,
    pub measurement_delay_ms: u64,
}

pub enum PredictionType {
    FillProbability,
    AdverseSelection,
    RegimeChange,
    PriceDirection,
    Volatility,
}
```

### Real-Time Calibration Metrics

```rust
pub struct CalibrationMetrics {
    pub brier_score: f64,
    pub information_ratio: f64,
    pub calibration_error: f64,  // Mean absolute deviation from diagonal
    pub resolution: f64,
    pub reliability: f64,
    pub n_samples: usize,
    pub last_updated: u64,
}

impl CalibrationMetrics {
    pub fn is_well_calibrated(&self) -> bool {
        self.information_ratio > 1.0 && 
        self.calibration_error < 0.1 &&
        self.n_samples >= 100
    }
}
```

### Signal Quality Tracking

```rust
pub struct SignalQualityTracker {
    pub signal_name: String,
    pub mutual_information: f64,  // Bits of information about target
    pub mi_trend: f64,            // Slope of MI over time
    pub half_life_days: f64,      // Time to 50% MI decay
    pub last_n_mi: VecDeque<f64>, // Rolling MI history
}

impl SignalQualityTracker {
    pub fn is_stale(&self) -> bool {
        self.half_life_days < 7.0  // Critical: decaying fast
    }
    
    pub fn is_useful(&self) -> bool {
        self.mutual_information > 0.01  // >0.01 bits = signal
    }
}
```

### Automated Alerts

```rust
pub enum CalibrationAlert {
    ModelMiscalibrated { model: String, brier_score: f64 },
    SignalDecaying { signal: String, half_life_days: f64 },
    InsufficientSamples { model: String, n_samples: usize },
    RegimeShift { old_regime: Regime, new_regime: Regime },
    InformationRatioBelowOne { model: String, ir: f64 },
}
```

---

## Anti-Patterns to Avoid

### 1. Complexity Without Validation

**Bad:**
```rust
// Adding Hawkes because it's "sophisticated"
let kappa = hawkes_intensity();  // Never validated
```

**Good:**
```rust
// Adding Hawkes because EWMA kappa was miscalibrated
if ewma_kappa_brier_score > 0.15 {
    // Validated that EWMA is inadequate
    let kappa = hawkes_intensity();
    // Will validate Hawkes next
}
```

### 2. Overfitting

**Bad:**
```rust
// 50 parameters tuned on 100 trades
let config = Config {
    param1: 0.123456,  // Suspiciously precise
    param2: 0.789012,
    // ... 48 more
};
```

**Good:**
```rust
// 5 parameters, at least 100 samples per parameter
let config = Config {
    gamma: 0.3,      // 500+ trades
    vol_halflife: 60.0,
    // Validated each independently
};
```

### 3. Ignoring Regime

**Bad:**
```rust
let spread = glft_spread(FIXED_GAMMA, FIXED_KAPPA);
// Works in calm, fails in cascade
```

**Good:**
```rust
let regime = hmm.current_belief();
let gamma = regime.blend_gamma(gamma_calm, gamma_volatile, gamma_cascade);
let spread = glft_spread(gamma, dynamic_kappa);
```

### 4. Trusting Backtest Results

**Bad:**
```rust
// Backtest shows 500% annual return
// Going live immediately
```

**Good:**
```rust
// Backtest shows 50% annual return
// Paper trade for 4 weeks
// Live with 10% of intended capital
// Scale up over 3 months
```

### 5. Single Point Estimates

**Bad:**
```rust
let kappa = 0.5;  // "The" fill intensity
```

**Good:**
```rust
let kappa_estimate = 0.5;
let kappa_uncertainty = 0.15;  // Standard error

// Widen spread when uncertain
let gamma_adjusted = gamma * (1.0 + kappa_uncertainty / kappa_estimate);
```

---

## References

### Academic Papers

1. **Guéant, Lehalle, Fernandez-Tapia (2013)** - "Dealing with the Inventory Risk: A Solution to the Market Making Problem"
   - The GLFT optimal spread formula

2. **Avellaneda & Stoikov (2008)** - "High-Frequency Trading in a Limit Order Book"
   - Foundation of modern market making theory

3. **Bacry et al. (2015)** - "Hawkes Processes in Finance"
   - Comprehensive overview of Hawkes applications
   - [arXiv:1502.04592](https://arxiv.org/pdf/1502.04592)

4. **Easley, López de Prado, O'Hara (2012)** - "Flow Toxicity and Liquidity in a High Frequency World"
   - VPIN methodology for adverse selection

5. **Cartea, Jaimungal, Penalva (2015)** - "Algorithmic and High-Frequency Trading"
   - Cambridge University Press textbook

### Online Resources

6. **Deep Hawkes Process for Market Making**
   - [Springer](https://link.springer.com/article/10.1007/s42786-024-00049-8)

7. **Detecting Toxic Flow (PULSE)**
   - [arXiv:2312.05827](https://arxiv.org/abs/2312.05827)

8. **Market Regime Detection with HMMs**
   - [QuantStart](https://www.quantstart.com/articles/market-regime-detection-using-hidden-markov-models-in-qstrader/)

9. **Statistical Significance in Backtesting**
   - [Medium](https://medium.com/@trading.dude/how-many-trades-are-enough-a-guide-to-statistical-significance-in-backtesting-093c2eac6f05)

10. **Retail vs Institutional Trader Advantages**
    - [AlgoTrading101](https://algotrading101.com/learn/retail-traders-vs-institutional-traders/)
    - [Traders Magazine](https://www.tradersmagazine.com/news/retail-traders-are-gaining-an-edge-over-institutions/)

---

## Summary

### The Path Forward

1. **Accept your constraints** - You're small, make it an advantage
2. **Find your niche** - One market you can know better than anyone
3. **Start simple** - Prove the basics work before adding complexity
4. **Measure everything** - Predictions without outcomes are worthless
5. **Validate rigorously** - 200+ trades, walk-forward, White's Reality Check
6. **Add complexity sparingly** - Only when simple demonstrably fails
7. **Stay patient** - Edge builds slowly, destroys quickly

### The Ultimate Test

Before adding ANY feature, ask:

> "Do I have statistical evidence that the simple version fails, AND evidence that this addition fixes it?"

If no to either, don't add it.

---

*"The models are not the secret. The calibration is. And calibration requires measurement, samples, and patience. This is not sexy. But it's how you actually build edge."*
