# First Principles Stochastic Processes for Market Making

## Part 0: What Are We Actually Doing?

Before any math, let's be precise about the problem.

### The Market Maker's Function

You are a **liquidity transformer**. You convert:
- **Temporal liquidity demand** (someone wants to trade NOW)
- Into **immediate execution**
- For a **fee** (the spread)

This is a service. The spread is compensation for costs you bear.

### The Costs You Bear

1. **Inventory Risk**: Holding a position exposes you to price movement
2. **Adverse Selection**: Some counterparties know more than you
3. **Funding Cost**: On perpetuals, you pay/receive funding
4. **Execution Risk**: Your quotes might not get filled, or get filled at wrong times

### The Optimization Problem

Without leverage:
```
maximize E[∫₀^T spread_capture(t) - inventory_cost(t) - adverse_selection(t) - funding(t) dt]
```

With leverage (your case):
```
maximize E[PnL]
subject to: P(drawdown > X) < ε
            P(liquidation) ≈ 0
```

**Critical insight**: The constraint dominates. At 50x leverage, a 2% adverse move = 100% loss. Expected value is meaningless when a single tail event is terminal.

### What We Need to Model

To solve this optimization, we need to model:
1. How prices move (to understand inventory risk)
2. How volatility changes (risk varies over time)
3. How our inventory evolves (controlled by our quotes)
4. How spreads behave (our revenue source)
5. How order flow arrives (when we get filled)
6. How adverse selection varies (our main cost)
7. Where we are in the queue (fill probability)
8. How assets correlate (portfolio effects)
9. How funding rates move (carry cost/revenue)
10. How liquidations cascade (tail risk)

Each of these is a stochastic process. Let's derive each from first principles.

---

## Part 1: The Price Process

### First Principles Derivation

Price moves because of **information arrival**. Information arrives in two ways:
1. **Continuously**: Gradual revelation through trading activity (diffusion)
2. **Discretely**: News events, large orders, liquidations (jumps)

Additionally, **volatility itself is uncertain**—it clusters (high vol follows high vol) and mean-reverts.

### The Model

Let p(t) be the log mid-price. We model:

```
dp = μ dt + σ(t) dWₚ + J dN

where:
- μ: drift (usually ≈ 0 for short horizons, or = -funding_rate for perpetuals)
- σ(t): time-varying volatility (see volatility process)
- Wₚ: standard Brownian motion
- J: jump size ~ N(μⱼ, σⱼ²) or double-exponential
- N: Poisson process with intensity λ
```

### Discretization for Implementation

At each tick (or fixed time step Δt):

```
p(t+Δt) = p(t) + μΔt + σ(t)√Δt × Z + J × 1{jump}

where:
- Z ~ N(0,1)
- 1{jump} ~ Bernoulli(λΔt)
- J ~ N(μⱼ, σⱼ²) if jump occurs
```

### What We Observe vs. What We Model

We don't simulate prices—we observe them. The model tells us:
1. **Expected variance over horizon h**: Var[p(t+h) - p(t)] = σ²h + λh×E[J²]
2. **Tail probabilities**: P(|Δp| > x) includes both diffusion and jump components
3. **Parameter estimation**: We estimate σ, λ, μⱼ, σⱼ from observed returns

### State Variables

```rust
struct PriceState {
    mid: f64,                    // Current mid price (observed)
    log_mid: f64,                // log(mid) for return calculations
    last_update: Instant,        // When we last observed
    
    // Estimated parameters (updated online)
    drift: f64,                  // μ, usually from funding rate
    
    // Jump detection
    recent_returns: RingBuffer<f64>,  // For detecting jumps
    jump_threshold: f64,         // Returns beyond this are "jumps"
}
```

### Update Function

```rust
impl PriceState {
    fn update(&mut self, new_mid: f64, timestamp: Instant) {
        let dt = timestamp.duration_since(self.last_update).as_secs_f64();
        if dt < 1e-9 { return; }  // Avoid division by zero
        
        let old_log = self.log_mid;
        self.mid = new_mid;
        self.log_mid = new_mid.ln();
        
        // Compute return
        let log_return = self.log_mid - old_log;
        
        // Store for jump detection and volatility estimation
        self.recent_returns.push(log_return);
        
        // Update drift estimate (EWMA)
        // For perpetuals, this should track -funding_rate
        self.drift = 0.99 * self.drift + 0.01 * (log_return / dt);
        
        self.last_update = timestamp;
    }
    
    fn is_jump(&self, log_return: f64) -> bool {
        // A return is a "jump" if it exceeds threshold standard deviations
        // Threshold typically 3-5σ
        let recent_vol = self.recent_returns.std_dev();
        log_return.abs() > self.jump_threshold * recent_vol
    }
}
```

---

## Part 2: The Volatility Process

### First Principles Derivation

Volatility is not constant because:
1. **Information arrival is clustered**: News begets news, trades beget trades
2. **Market regimes exist**: Calm periods vs. turbulent periods
3. **Mean reversion**: Extreme volatility doesn't persist indefinitely

Empirical facts ("stylized facts"):
- Volatility is persistent (autocorrelated)
- Volatility clusters (high vol follows high vol)
- Volatility mean-reverts
- Volatility has "leverage effect" (negative correlation with returns)

### The Model

We use a regime-switching stochastic volatility model:

**Within regime r:**
```
dv = κᵣ(θᵣ - v)dt + ξᵣ√v dWᵥ

where:
- v: instantaneous variance (σ² = v)
- κᵣ: mean reversion speed in regime r
- θᵣ: long-run variance in regime r
- ξᵣ: vol-of-vol in regime r
- Wᵥ: Brownian motion with Corr(dWₚ, dWᵥ) = ρ (leverage effect)
```

**Regime transitions:**
```
Regimes: {Low, Normal, High, Extreme}
Transition matrix Q where Qᵢⱼ = P(regime j at t+dt | regime i at t)
```

### Why Regimes?

Pure stochastic vol models (Heston) don't capture the **discontinuous** shifts in market behavior. Regimes capture:
- Pre-announcement calm → post-announcement chaos
- Normal trading → liquidation cascade
- Trending market → mean-reverting market

### Discretization for Implementation

**Variance update (within regime):**
```
v(t+Δt) = v(t) + κ(θ - v(t))Δt + ξ√(v(t))√Δt × Zᵥ

where Zᵥ is correlated with Zₚ: Zᵥ = ρZₚ + √(1-ρ²)Z⊥
```

**Regime detection:**
Use realized volatility to classify current regime.

### State Variables

```rust
#[derive(Clone, Copy, PartialEq)]
enum VolRegime {
    Low,      // σ < 0.5 × historical average
    Normal,   // 0.5 - 1.5 × historical average
    High,     // 1.5 - 3 × historical average
    Extreme,  // > 3 × historical average
}

struct VolatilityState {
    // Current estimates
    instantaneous_var: f64,      // v(t), our best estimate of current variance
    realized_vol_1m: f64,        // Realized vol over last 1 minute
    realized_vol_5m: f64,        // Realized vol over last 5 minutes
    realized_vol_1h: f64,        // Realized vol over last 1 hour
    
    // Regime
    current_regime: VolRegime,
    regime_duration: Duration,    // How long in current regime
    
    // Parameters (per regime)
    regime_params: HashMap<VolRegime, VolParams>,
    
    // For estimation
    squared_returns: RingBuffer<(Instant, f64)>,  // (timestamp, r²)
    
    // Leverage effect
    return_vol_correlation: f64,  // ρ, estimated from data
}

struct VolParams {
    kappa: f64,      // Mean reversion speed
    theta: f64,      // Long-run variance
    xi: f64,         // Vol of vol
}
```

### Update Function

```rust
impl VolatilityState {
    fn update(&mut self, log_return: f64, dt: f64, timestamp: Instant) {
        // Store squared return for realized vol calculation
        self.squared_returns.push((timestamp, log_return.powi(2)));
        
        // Update realized vol estimates at different scales
        self.realized_vol_1m = self.compute_realized_vol(Duration::from_secs(60));
        self.realized_vol_5m = self.compute_realized_vol(Duration::from_secs(300));
        self.realized_vol_1h = self.compute_realized_vol(Duration::from_secs(3600));
        
        // Update instantaneous variance estimate
        // Use EWMA of squared returns, annualized
        let lambda = 0.94;  // RiskMetrics decay factor
        self.instantaneous_var = lambda * self.instantaneous_var 
                                + (1.0 - lambda) * log_return.powi(2) / dt;
        
        // Detect regime
        let new_regime = self.classify_regime();
        if new_regime != self.current_regime {
            self.current_regime = new_regime;
            self.regime_duration = Duration::ZERO;
        } else {
            self.regime_duration += Duration::from_secs_f64(dt);
        }
        
        // Update leverage effect correlation estimate
        self.update_leverage_effect(log_return);
    }
    
    fn compute_realized_vol(&self, window: Duration) -> f64 {
        let cutoff = Instant::now() - window;
        let sum_sq: f64 = self.squared_returns.iter()
            .filter(|(t, _)| *t > cutoff)
            .map(|(_, r2)| r2)
            .sum();
        let n = self.squared_returns.iter()
            .filter(|(t, _)| *t > cutoff)
            .count();
        
        if n < 2 { return self.instantaneous_var.sqrt(); }
        
        // Annualize: assume returns are per-second, multiply by √(seconds_per_year)
        let seconds_per_year = 365.25 * 24.0 * 3600.0;
        (sum_sq / n as f64 * seconds_per_year).sqrt()
    }
    
    fn classify_regime(&self) -> VolRegime {
        // Use 1h realized vol relative to long-term average
        let historical_avg = self.regime_params[&VolRegime::Normal].theta.sqrt();
        let ratio = self.realized_vol_1h / historical_avg;
        
        match ratio {
            r if r < 0.5 => VolRegime::Low,
            r if r < 1.5 => VolRegime::Normal,
            r if r < 3.0 => VolRegime::High,
            _ => VolRegime::Extreme,
        }
    }
    
    fn current_sigma(&self) -> f64 {
        self.instantaneous_var.sqrt()
    }
    
    fn sigma_for_horizon(&self, seconds: f64) -> f64 {
        // Blend short-term and long-term vol based on horizon
        // Short horizons: use instantaneous
        // Long horizons: mean reversion toward regime theta
        let params = &self.regime_params[&self.current_regime];
        let theta = params.theta;
        let kappa = params.kappa;
        
        // Expected average variance over horizon [0, T]:
        // E[∫v dt]/T = θ + (v₀ - θ)(1 - e^(-κT))/(κT)
        let v0 = self.instantaneous_var;
        let decay = 1.0 - (-kappa * seconds).exp();
        let avg_var = theta + (v0 - theta) * decay / (kappa * seconds);
        
        avg_var.sqrt()
    }
}
```

---

## Part 3: The Inventory Process

### First Principles Derivation

Your inventory q(t) evolves based on fills. Fills happen when:
1. Price reaches your quote level
2. Sufficient volume trades through your queue position

This is a **controlled stochastic process**—your quotes influence but don't determine outcomes.

### The Model

```
dq = (fill_rate_bid(t) - fill_rate_ask(t)) dt

where fill_rate depends on:
- Your quote depths (δ_bid, δ_ask)
- Order flow intensity λ(t)
- Your queue position
```

More precisely:
```
fill_rate_bid = λ_bid(t) × P(fill | order arrival on bid) × size_bid
fill_rate_ask = λ_ask(t) × P(fill | order arrival on ask) × size_ask
```

### Fill Probability Model

P(fill) depends on:
1. **P(price touches your level)**: Function of depth from mid and volatility
2. **P(execute | touch)**: Function of queue position and volume at touch

```
P(fill in dt) = P(touch in dt) × P(execute | touch)
```

**P(touch)** - Using reflection principle for Brownian motion:
```
P(touch level δ below mid in time T) ≈ 2Φ(-δ/(σ√T))

where Φ is standard normal CDF
```

**P(execute | touch)** - Depends on queue:
```
P(execute | touch) = P(volume at touch > queue_position)
                   ≈ exp(-queue_position / expected_volume_at_touch)
```

### State Variables

```rust
struct InventoryState {
    // Current inventory per asset
    positions: HashMap<String, f64>,
    
    // Entry prices for PnL tracking
    entry_prices: HashMap<String, f64>,
    average_entry: HashMap<String, f64>,
    
    // Fill tracking
    recent_fills: VecDeque<Fill>,
    fill_rate_bid: HashMap<String, f64>,  // Estimated fill rate
    fill_rate_ask: HashMap<String, f64>,
    
    // Limits
    max_position: HashMap<String, f64>,
    target_position: HashMap<String, f64>,  // Usually 0
}

struct Fill {
    asset: String,
    side: Side,
    price: f64,
    size: f64,
    timestamp: Instant,
    queue_position_at_fill: f64,  // For model validation
}
```

### Update Function

```rust
impl InventoryState {
    fn record_fill(&mut self, fill: Fill) {
        let pos = self.positions.entry(fill.asset.clone()).or_insert(0.0);
        let sign = match fill.side {
            Side::Buy => 1.0,
            Side::Sell => -1.0,
        };
        
        // Update position
        let new_pos = *pos + sign * fill.size;
        
        // Update average entry price
        if sign * new_pos > sign * *pos {
            // Adding to position
            let avg = self.average_entry.entry(fill.asset.clone()).or_insert(fill.price);
            *avg = (*avg * pos.abs() + fill.price * fill.size) / (pos.abs() + fill.size);
        }
        
        *pos = new_pos;
        
        // Track for fill rate estimation
        self.recent_fills.push_back(fill);
        while self.recent_fills.len() > 1000 {
            self.recent_fills.pop_front();
        }
        
        // Update fill rate estimates
        self.update_fill_rates();
    }
    
    fn update_fill_rates(&mut self) {
        // Compute fills per second over recent window
        let window = Duration::from_secs(300);  // 5 minutes
        let cutoff = Instant::now() - window;
        
        for (asset, _) in &self.positions {
            let bid_fills: f64 = self.recent_fills.iter()
                .filter(|f| f.asset == *asset && f.side == Side::Buy && f.timestamp > cutoff)
                .map(|f| f.size)
                .sum();
            let ask_fills: f64 = self.recent_fills.iter()
                .filter(|f| f.asset == *asset && f.side == Side::Sell && f.timestamp > cutoff)
                .map(|f| f.size)
                .sum();
            
            self.fill_rate_bid.insert(asset.clone(), bid_fills / window.as_secs_f64());
            self.fill_rate_ask.insert(asset.clone(), ask_fills / window.as_secs_f64());
        }
    }
    
    fn inventory_risk(&self, asset: &str, sigma: f64, horizon: f64) -> f64 {
        // Risk = |position| × σ × √horizon
        let pos = self.positions.get(asset).unwrap_or(&0.0);
        let price = self.entry_prices.get(asset).unwrap_or(&1.0);
        pos.abs() * price * sigma * horizon.sqrt()
    }
    
    fn inventory_skew(&self, asset: &str) -> f64 {
        // How much to skew quotes to reduce inventory
        // Positive = want to sell (long inventory), negative = want to buy
        let pos = self.positions.get(asset).unwrap_or(&0.0);
        let max = self.max_position.get(asset).unwrap_or(&1.0);
        let target = self.target_position.get(asset).unwrap_or(&0.0);
        
        (pos - target) / max  // Normalized to [-1, 1]
    }
}
```

---

## Part 4: The Spread Process

### First Principles Derivation

The market spread is set by competition among market makers. It reflects:
1. **Inventory costs** of the marginal market maker
2. **Adverse selection costs**
3. **Operating costs**

The spread is NOT constant—it widens during:
- High volatility (inventory risk increases)
- News events (adverse selection increases)
- Low liquidity (fewer competitors)

### The Model

```
d(spread) = κ_s(θ_s - spread)dt + σ_s dW_s + J_s dN_s

where:
- κ_s: mean reversion speed
- θ_s: "fair" spread (calibrated to historical median)
- σ_s: spread volatility
- J_s: jump size (positive, spreads widen on jumps)
- N_s: Poisson process for spread jumps
```

**Conditional dynamics**: The fair spread θ_s depends on volatility:
```
θ_s(v) = θ₀ + β_v × σ(v)
```

Higher vol → wider fair spread.

### State Variables

```rust
struct SpreadState {
    current_spread: f64,         // Observed spread
    fair_spread: f64,            // Our estimate of equilibrium spread
    
    // Model parameters
    mean_reversion: f64,         // κ_s
    base_fair_spread: f64,       // θ₀
    vol_sensitivity: f64,        // β_v
    spread_volatility: f64,      // σ_s
    
    // For estimation
    spread_history: RingBuffer<(Instant, f64)>,
    
    // Regime-dependent
    regime_spread_mult: HashMap<VolRegime, f64>,  // Multiplier per regime
}
```

### Update Function

```rust
impl SpreadState {
    fn update(&mut self, best_bid: f64, best_ask: f64, vol_state: &VolatilityState, timestamp: Instant) {
        self.current_spread = best_ask - best_bid;
        self.spread_history.push((timestamp, self.current_spread));
        
        // Update fair spread estimate based on current volatility
        let sigma = vol_state.current_sigma();
        let regime_mult = self.regime_spread_mult
            .get(&vol_state.current_regime)
            .unwrap_or(&1.0);
        
        self.fair_spread = (self.base_fair_spread + self.vol_sensitivity * sigma) * regime_mult;
    }
    
    fn spread_percentile(&self) -> f64 {
        // Where is current spread in historical distribution?
        let mut sorted: Vec<f64> = self.spread_history.iter()
            .map(|(_, s)| *s)
            .collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let rank = sorted.iter()
            .filter(|&&s| s <= self.current_spread)
            .count();
        
        rank as f64 / sorted.len() as f64
    }
    
    fn is_spread_tight(&self) -> bool {
        self.current_spread < self.fair_spread * 0.8
    }
    
    fn is_spread_wide(&self) -> bool {
        self.current_spread > self.fair_spread * 1.5
    }
}
```

---

## Part 5: The Order Flow Process

### First Principles Derivation

Order flow (buy/sell market orders) is NOT uniformly distributed in time. It exhibits:
1. **Clustering**: Trades beget trades (self-excitation)
2. **Asymmetry**: Buy vs. sell pressure imbalances
3. **Periodicity**: Higher activity at certain times

The Hawkes process captures self-excitation.

### The Model

**Hawkes process for order arrivals:**
```
λ(t) = μ + ∫₀ᵗ α e^(-β(t-s)) dN(s)

where:
- μ: baseline intensity (orders per second)
- α: excitation parameter (how much each order increases future intensity)
- β: decay rate (how fast excitation dies)
- N(s): counting process of past orders
```

**Interpretation**:
- Each order increases intensity by α
- This excitation decays exponentially with rate β
- "Branching ratio" α/β < 1 for stability

### Separate Processes for Bids/Asks

We need two Hawkes processes:
```
λ_buy(t) = μ_buy + ∫ α_buy e^(-β(t-s)) dN_buy(s) + ∫ γ e^(-β(t-s)) dN_sell(s)
λ_sell(t) = μ_sell + ∫ α_sell e^(-β(t-s)) dN_sell(s) + ∫ γ e^(-β(t-s)) dN_buy(s)
```

The cross-excitation term γ captures "both sides heat up together."

### State Variables

```rust
struct OrderFlowState {
    // Current intensities
    intensity_buy: f64,
    intensity_sell: f64,
    
    // Baseline intensities
    mu_buy: f64,
    mu_sell: f64,
    
    // Excitation parameters
    alpha_buy: f64,   // Self-excitation
    alpha_sell: f64,
    gamma: f64,       // Cross-excitation
    beta: f64,        // Decay rate
    
    // Recent events for computing intensity
    recent_buys: VecDeque<Instant>,
    recent_sells: VecDeque<Instant>,
    
    // Order flow imbalance
    imbalance: f64,   // (buy_volume - sell_volume) / total_volume
    imbalance_ema: f64,
}
```

### Update Function

```rust
impl OrderFlowState {
    fn update(&mut self, trade: Option<&Trade>, now: Instant) {
        // Add new trade to history
        if let Some(t) = trade {
            match t.side {
                Side::Buy => self.recent_buys.push_back(now),
                Side::Sell => self.recent_sells.push_back(now),
            }
        }
        
        // Prune old events (keep last 5 minutes)
        let cutoff = now - Duration::from_secs(300);
        while self.recent_buys.front().map_or(false, |&t| t < cutoff) {
            self.recent_buys.pop_front();
        }
        while self.recent_sells.front().map_or(false, |&t| t < cutoff) {
            self.recent_sells.pop_front();
        }
        
        // Compute current intensities
        self.intensity_buy = self.compute_intensity(&self.recent_buys, &self.recent_sells, 
                                                     self.mu_buy, self.alpha_buy, now);
        self.intensity_sell = self.compute_intensity(&self.recent_sells, &self.recent_buys,
                                                      self.mu_sell, self.alpha_sell, now);
        
        // Update imbalance
        if let Some(t) = trade {
            let sign = match t.side { Side::Buy => 1.0, Side::Sell => -1.0 };
            self.imbalance_ema = 0.99 * self.imbalance_ema + 0.01 * sign;
        }
    }
    
    fn compute_intensity(&self, same_side: &VecDeque<Instant>, other_side: &VecDeque<Instant>,
                         mu: f64, alpha: f64, now: Instant) -> f64 {
        let mut intensity = mu;
        
        // Self-excitation from same-side trades
        for &t in same_side {
            let elapsed = now.duration_since(t).as_secs_f64();
            intensity += alpha * (-self.beta * elapsed).exp();
        }
        
        // Cross-excitation from other-side trades
        for &t in other_side {
            let elapsed = now.duration_since(t).as_secs_f64();
            intensity += self.gamma * (-self.beta * elapsed).exp();
        }
        
        intensity
    }
    
    fn expected_trades_next(&self, seconds: f64) -> (f64, f64) {
        // Expected number of trades in next `seconds`
        // For Hawkes: E[N(t+s) - N(t)] ≈ λ(t) × s for small s
        (self.intensity_buy * seconds, self.intensity_sell * seconds)
    }
    
    fn flow_imbalance(&self) -> f64 {
        // Normalized to [-1, 1]
        // Positive = buy pressure, negative = sell pressure
        self.imbalance_ema.clamp(-1.0, 1.0)
    }
}
```

---

## Part 6: The Adverse Selection Process

### First Principles Derivation

**What is adverse selection?**
Some trades are "informed"—the counterparty knows something you don't. After these trades, the price moves against you.

**Measurable definition:**
```
Adverse Selection Cost = E[price_change_after_fill | you got filled]
```

If this is negative (for buys) / positive (for sells), you're being adversely selected.

**Why it varies over time:**
- More informed trading around news events
- Arbitrageurs active when cross-exchange spreads diverge
- Liquidations create informed flow (cascade direction is known)

### The Model

Let α(t) ∈ [0,1] be the probability that the next trade is informed.

```
dα = κ_α(ᾱ - α)dt + σ_α dW_α + jump on events

where:
- ᾱ: baseline informed fraction (~0.1-0.2 typically)
- κ_α: mean reversion speed
- Events that cause jumps: funding rate changes, liquidations, news
```

**Predictive signals for α:**
1. **Funding rate divergence**: |funding_this_exchange - funding_other| → arbitrage opportunity
2. **Volatility surprise**: realized_vol >> implied_vol → information arriving
3. **Flow imbalance**: Sustained one-sided flow → informed direction
4. **Liquidation proximity**: High leverage + price near liquidation prices

### State Variables

```rust
struct AdverseSelectionState {
    // Current estimate
    alpha: f64,  // P(next trade is informed)
    
    // Realized adverse selection (ground truth for calibration)
    realized_as_bid: ExponentialMovingAverage,  // E[Δp | fill on bid]
    realized_as_ask: ExponentialMovingAverage,  // E[Δp | fill on ask]
    
    // Pending fills waiting for price measurement
    pending_fills: VecDeque<PendingFillMeasurement>,
    measurement_horizon: Duration,  // How long after fill to measure (e.g., 1 second)
    
    // Predictive signals
    funding_divergence: f64,
    volatility_surprise: f64,
    flow_imbalance: f64,
    liquidation_intensity: f64,
    
    // Model coefficients (calibrated offline)
    beta_0: f64,  // Intercept
    beta_funding: f64,
    beta_vol: f64,
    beta_flow: f64,
    beta_liq: f64,
}

struct PendingFillMeasurement {
    fill: Fill,
    price_at_fill: f64,
    measure_at: Instant,
}
```

### Update Function

```rust
impl AdverseSelectionState {
    fn update(&mut self, current_mid: f64, now: Instant) {
        // Check if any pending fills are ready to measure
        while let Some(pending) = self.pending_fills.front() {
            if now < pending.measure_at {
                break;
            }
            
            let pending = self.pending_fills.pop_front().unwrap();
            let price_change = (current_mid - pending.price_at_fill) / pending.price_at_fill;
            
            // Adverse selection is price moving against us
            match pending.fill.side {
                Side::Buy => {
                    // We bought; if price dropped, we were adversely selected
                    self.realized_as_bid.update(-price_change);  // Negative = bad
                }
                Side::Sell => {
                    // We sold; if price rose, we were adversely selected
                    self.realized_as_ask.update(price_change);  // Positive = bad
                }
            }
        }
        
        // Update alpha estimate using predictive model
        self.alpha = self.predict_alpha();
    }
    
    fn record_fill(&mut self, fill: Fill, current_mid: f64, now: Instant) {
        self.pending_fills.push_back(PendingFillMeasurement {
            fill,
            price_at_fill: current_mid,
            measure_at: now + self.measurement_horizon,
        });
    }
    
    fn predict_alpha(&self) -> f64 {
        // Logistic regression
        let z = self.beta_0
            + self.beta_funding * self.funding_divergence
            + self.beta_vol * self.volatility_surprise
            + self.beta_flow * self.flow_imbalance.abs()
            + self.beta_liq * self.liquidation_intensity;
        
        1.0 / (1.0 + (-z).exp())  // Sigmoid
    }
    
    fn update_signals(&mut self, 
                      funding_here: f64, 
                      funding_reference: f64,
                      realized_vol: f64,
                      implied_vol: f64,
                      flow: &OrderFlowState,
                      liq: &LiquidationState) {
        self.funding_divergence = (funding_here - funding_reference).abs();
        self.volatility_surprise = (realized_vol - implied_vol) / implied_vol;
        self.flow_imbalance = flow.flow_imbalance();
        self.liquidation_intensity = liq.current_intensity / liq.base_intensity;
    }
    
    fn realized_adverse_selection(&self) -> f64 {
        // Average of bid and ask AS (both should be ~0 if no AS)
        // Positive means we're getting picked off
        (self.realized_as_bid.value().abs() + self.realized_as_ask.value().abs()) / 2.0
    }
    
    fn spread_adjustment(&self) -> f64 {
        // How much to widen spread due to adverse selection
        // Higher alpha → wider spread needed
        // Rule of thumb: spread should cover expected AS cost
        self.alpha * self.realized_adverse_selection() * 2.0  // 2x for safety margin
    }
}
```

---

## Part 7: The Queue Position Process

### First Principles Derivation

Your order sits in a queue. You only get filled when:
1. Price reaches your level, AND
2. All orders ahead of you get filled

Queue position evolves due to:
- New orders joining behind you (doesn't affect you)
- Orders ahead cancelling (you move up)
- Executions clearing the queue (you move up)
- Your own cancellations (you lose position)

**This is critical**: Two orders at the same price have very different fill probabilities based on queue position.

### The Model

Let Q(t) = your queue position (0 = at front, ∞ = at back).

When you place/replace an order:
```
Q(0) = current_depth_at_level  // You join at the back
```

Evolution:
```
dQ = -execution_rate × dt - cancel_rate × dt

where:
- execution_rate: rate at which volume ahead of you gets executed
- cancel_rate: rate at which orders ahead of you cancel
```

**Fill probability:**
```
P(fill in dt | order at level δ) = P(price touches δ in dt) × P(Q reaches 0 | touch)
```

### State Variables

```rust
struct QueueState {
    // Per-order queue tracking
    orders: HashMap<OrderId, QueuePosition>,
    
    // Aggregate depth model
    depth_ahead: HashMap<PriceLevel, f64>,  // Total depth ahead at each level
    
    // Model parameters (estimated from data)
    cancel_rate: f64,      // Fraction of queue that cancels per second
    execution_rate: f64,   // Volume executed per second at touch
}

struct QueuePosition {
    order_id: OrderId,
    price: f64,
    size: f64,
    estimated_position: f64,  // Depth ahead of us
    placed_at: Instant,
    last_update: Instant,
}
```

### Update Function

```rust
impl QueueState {
    fn order_placed(&mut self, order_id: OrderId, price: f64, size: f64, 
                    book: &OrderBook, now: Instant) {
        // Calculate depth ahead of our order
        let depth_ahead = book.depth_at_or_better(price);
        
        self.orders.insert(order_id, QueuePosition {
            order_id,
            price,
            size,
            estimated_position: depth_ahead,
            placed_at: now,
            last_update: now,
        });
    }
    
    fn order_cancelled(&mut self, order_id: &OrderId) {
        self.orders.remove(order_id);
    }
    
    fn update(&mut self, book: &OrderBook, trades: &[Trade], now: Instant) {
        for (_, pos) in self.orders.iter_mut() {
            let dt = now.duration_since(pos.last_update).as_secs_f64();
            
            // Volume traded at our level
            let traded_at_level: f64 = trades.iter()
                .filter(|t| t.price == pos.price)
                .map(|t| t.size)
                .sum();
            
            // Reduce position by traded volume
            pos.estimated_position = (pos.estimated_position - traded_at_level).max(0.0);
            
            // Also decay for cancellations (estimated)
            pos.estimated_position *= (1.0 - self.cancel_rate * dt);
            
            pos.last_update = now;
        }
    }
    
    fn fill_probability(&self, order_id: &OrderId, sigma: f64, horizon: f64) -> f64 {
        let pos = match self.orders.get(order_id) {
            Some(p) => p,
            None => return 0.0,
        };
        
        // P(touch) - simplified model using normal distribution
        // For a bid order at price B when mid is M:
        // P(touch in time T) ≈ 2Φ(-|M-B|/(σ√T))
        let depth = (pos.price - self.current_mid()).abs();
        let p_touch = 2.0 * normal_cdf(-depth / (sigma * horizon.sqrt()));
        
        // P(execute | touch) - exponential model
        // P(fill) = exp(-queue_position / expected_volume)
        let expected_vol = self.execution_rate * horizon;
        let p_execute = (-pos.estimated_position / expected_vol.max(1e-10)).exp();
        
        p_touch * p_execute
    }
}
```

---

## Part 8: The Correlation Process

### First Principles Derivation

Asset correlations matter because:
1. Portfolio risk depends on covariances, not just individual variances
2. Hedging effectiveness depends on correlation stability
3. Diversification benefit = 1 - correlation

**Critical fact**: Correlations are NOT constant. They:
- Vary across timescales (higher at longer horizons)
- Spike toward ±1 during crises ("correlation breakdown")
- Mean-revert slowly

### The Model

**DCC (Dynamic Conditional Correlation) approach:**
```
Rₜ = Dₜ⁻¹ Hₜ Dₜ⁻¹

where:
- Hₜ: conditional covariance matrix
- Dₜ: diagonal matrix of conditional standard deviations
- Rₜ: correlation matrix
```

**For practical implementation, use EWMA:**
```
Σₜ = λΣₜ₋₁ + (1-λ)rₜrₜ'

where:
- λ: decay factor (0.94 for daily, 0.99+ for intraday)
- rₜ: vector of returns at time t
```

### Multi-Scale Correlations

Different horizons need different decay factors:

| Horizon | λ | Effective Window |
|---------|---|------------------|
| 1 min   | 0.94 | ~15 observations |
| 5 min   | 0.97 | ~30 observations |
| 1 hour  | 0.99 | ~100 observations |
| 1 day   | 0.995 | ~200 observations |

### State Variables

```rust
struct CorrelationState {
    assets: Vec<String>,
    n_assets: usize,
    
    // Multi-scale covariance matrices
    cov_fast: Matrix,    // λ = 0.94
    cov_medium: Matrix,  // λ = 0.97
    cov_slow: Matrix,    // λ = 0.99
    
    // Derived correlation matrices
    corr_fast: Matrix,
    corr_medium: Matrix,
    corr_slow: Matrix,
    
    // Stress correlations (historical worst case)
    corr_stress: Matrix,
    
    // Return buffers for each asset
    returns: HashMap<String, RingBuffer<f64>>,
    
    // Diagnostics
    diversification_ratio: f64,  // 1 = no diversification, >1 = diversified
}
```

### Update Function

```rust
impl CorrelationState {
    fn update(&mut self, returns: &HashMap<String, f64>) {
        // Convert returns to vector in consistent order
        let ret_vec: Vec<f64> = self.assets.iter()
            .map(|a| *returns.get(a).unwrap_or(&0.0))
            .collect();
        
        // Outer product r × r'
        let outer = self.outer_product(&ret_vec);
        
        // Update EWMA covariances
        self.cov_fast = self.ewma_update(&self.cov_fast, &outer, 0.94);
        self.cov_medium = self.ewma_update(&self.cov_medium, &outer, 0.97);
        self.cov_slow = self.ewma_update(&self.cov_slow, &outer, 0.99);
        
        // Extract correlations
        self.corr_fast = self.cov_to_corr(&self.cov_fast);
        self.corr_medium = self.cov_to_corr(&self.cov_medium);
        self.corr_slow = self.cov_to_corr(&self.cov_slow);
        
        // Update diversification ratio
        self.diversification_ratio = self.compute_diversification_ratio();
        
        // Check for correlation breakdown
        if self.average_correlation() > 0.9 {
            // Update stress matrix (worst case)
            self.corr_stress = self.blend_toward_one(&self.corr_slow, 0.95);
        }
    }
    
    fn ewma_update(&self, old: &Matrix, new: &Matrix, lambda: f64) -> Matrix {
        old * lambda + new * (1.0 - lambda)
    }
    
    fn cov_to_corr(&self, cov: &Matrix) -> Matrix {
        let mut corr = cov.clone();
        for i in 0..self.n_assets {
            for j in 0..self.n_assets {
                let vol_i = cov[(i, i)].sqrt();
                let vol_j = cov[(j, j)].sqrt();
                corr[(i, j)] = cov[(i, j)] / (vol_i * vol_j);
            }
        }
        corr
    }
    
    fn correlation_for_horizon(&self, horizon_seconds: f64) -> &Matrix {
        match horizon_seconds {
            h if h < 60.0 => &self.corr_fast,
            h if h < 300.0 => &self.corr_medium,
            _ => &self.corr_slow,
        }
    }
    
    fn average_correlation(&self) -> f64 {
        let mut sum = 0.0;
        let mut count = 0;
        for i in 0..self.n_assets {
            for j in (i+1)..self.n_assets {
                sum += self.corr_slow[(i, j)].abs();
                count += 1;
            }
        }
        sum / count as f64
    }
    
    fn compute_diversification_ratio(&self) -> f64 {
        // DR = sum(w_i * σ_i) / σ_portfolio
        // For equal weights:
        let individual_vols: Vec<f64> = (0..self.n_assets)
            .map(|i| self.cov_slow[(i, i)].sqrt())
            .collect();
        
        let sum_vols: f64 = individual_vols.iter().sum();
        let portfolio_var = self.portfolio_variance(&vec![1.0 / self.n_assets as f64; self.n_assets]);
        
        sum_vols / (self.n_assets as f64 * portfolio_var.sqrt())
    }
    
    fn portfolio_variance(&self, weights: &[f64]) -> f64 {
        // w' Σ w
        let mut var = 0.0;
        for i in 0..self.n_assets {
            for j in 0..self.n_assets {
                var += weights[i] * weights[j] * self.cov_slow[(i, j)];
            }
        }
        var
    }
}
```

---

## Part 9: The Funding Rate Process

### First Principles Derivation

On perpetual futures:
- Funding = periodic payment from longs to shorts (or vice versa)
- Purpose: Keep perp price close to spot price
- Rate determined by: premium of perp over spot + interest rate component

**Why it matters for market making:**
1. Holding inventory costs/earns funding
2. Funding rates across exchanges diverge → arbitrage
3. Funding rate changes are predictable (they adjust slowly)

### The Model

```
df = κ_f(θ_f - f)dt + σ_f dW_f + jumps at funding intervals

where:
- f: current funding rate (annualized)
- θ_f: equilibrium funding rate (~0)
- κ_f: mean reversion speed (slow, ~0.1/hour)
- Jumps at 8-hour intervals when funding settles and resets
```

**Cross-exchange dynamics:**
```
df_i = κ_i(θ_i - f_i)dt + β_i df_BTC + ε_i dW_i
```

BTC funding often leads altcoin funding.

### State Variables

```rust
struct FundingState {
    // Current rates per asset
    rates: HashMap<String, f64>,
    
    // Predicted next funding
    predicted_rates: HashMap<String, f64>,
    
    // Time until next funding
    next_funding: Instant,
    funding_interval: Duration,  // Usually 8 hours
    
    // Historical for mean estimation
    rate_history: HashMap<String, RingBuffer<f64>>,
    
    // Cross-exchange comparison
    rates_other_exchange: HashMap<String, f64>,
    
    // Model parameters
    mean_reversion: f64,
    long_term_mean: HashMap<String, f64>,
}
```

### Update Function

```rust
impl FundingState {
    fn update(&mut self, new_rates: HashMap<String, f64>, now: Instant) {
        for (asset, rate) in new_rates {
            self.rates.insert(asset.clone(), rate);
            
            // Update history
            self.rate_history
                .entry(asset.clone())
                .or_insert_with(|| RingBuffer::new(1000))
                .push(rate);
            
            // Update long-term mean
            let history = &self.rate_history[&asset];
            let mean = history.iter().sum::<f64>() / history.len() as f64;
            self.long_term_mean.insert(asset, mean);
            
            // Predict next funding
            let predicted = self.predict_next_funding(&asset, rate);
            self.predicted_rates.insert(asset, predicted);
        }
        
        // Update next funding time
        if now > self.next_funding {
            self.next_funding = self.next_funding + self.funding_interval;
        }
    }
    
    fn predict_next_funding(&self, asset: &str, current: f64) -> f64 {
        let theta = self.long_term_mean.get(asset).unwrap_or(&0.0);
        let dt = self.time_to_next_funding().as_secs_f64() / 3600.0;  // in hours
        
        // Mean reversion: f_next = f + κ(θ - f)dt
        current + self.mean_reversion * (theta - current) * dt
    }
    
    fn time_to_next_funding(&self) -> Duration {
        self.next_funding.saturating_duration_since(Instant::now())
    }
    
    fn funding_cost(&self, asset: &str, position: f64, holding_hours: f64) -> f64 {
        // Expected funding cost over holding period
        let rate = self.rates.get(asset).unwrap_or(&0.0);
        let notional = position.abs();
        
        // Number of funding intervals in holding period
        let num_fundings = holding_hours / 8.0;
        
        // Cost = rate × notional × num_intervals × sign
        // Positive position + positive rate = you pay
        rate * notional * num_fundings * position.signum()
    }
    
    fn funding_arbitrage_opportunity(&self, asset: &str) -> Option<f64> {
        // Compare to other exchange
        let here = self.rates.get(asset)?;
        let there = self.rates_other_exchange.get(asset)?;
        
        let spread = (here - there).abs();
        if spread > 0.0005 {  // 0.05% annualized threshold
            Some(spread)
        } else {
            None
        }
    }
}
```

---

## Part 10: The Liquidation Cascade Process

### First Principles Derivation

On leveraged exchanges, liquidations are a **self-exciting** process:
1. Price moves → positions become underwater
2. Liquidations triggered → market orders in direction of move
3. Price moves further → more positions underwater
4. Cascade continues until leverage is flushed

This is where your **tail risk** lives. Model it explicitly.

### The Model

**Hawkes process for liquidations:**
```
λ_liq(t) = μ_liq + ∫₀ᵗ α_liq × size(s) × e^(-β(t-s)) dN_liq(s)

where:
- μ_liq: baseline liquidation intensity
- α_liq: excitation per unit liquidation size
- β: decay rate
- size(s): size of liquidation at time s
```

**Key insight**: Large liquidations excite more than small ones.

### State Variables

```rust
struct LiquidationState {
    // Current cascade intensity
    current_intensity: f64,
    base_intensity: f64,
    
    // Model parameters
    excitation: f64,   // α: how much each liq increases future intensity
    decay: f64,        // β: how fast excitation decays
    
    // Recent liquidations
    recent_liquidations: VecDeque<LiquidationEvent>,
    
    // Aggregate metrics
    total_liquidated_24h: f64,
    cascade_active: bool,
    cascade_start: Option<Instant>,
    
    // Predictive signals
    aggregate_leverage: f64,      // Estimated market leverage
    distance_to_liquidations: f64, // How close is price to major liq levels
}

struct LiquidationEvent {
    timestamp: Instant,
    size: f64,       // Notional value liquidated
    direction: Side, // Was this a long or short being liquidated
    price: f64,
}
```

### Update Function

```rust
impl LiquidationState {
    fn update(&mut self, event: Option<LiquidationEvent>, now: Instant) {
        // Add new event
        if let Some(liq) = event {
            self.recent_liquidations.push_back(liq.clone());
            self.total_liquidated_24h += liq.size;
            
            // Check if cascade is starting
            if !self.cascade_active && self.current_intensity > 3.0 * self.base_intensity {
                self.cascade_active = true;
                self.cascade_start = Some(now);
            }
        }
        
        // Prune old events (keep 1 hour)
        let cutoff = now - Duration::from_secs(3600);
        while self.recent_liquidations.front().map_or(false, |e| e.timestamp < cutoff) {
            let old = self.recent_liquidations.pop_front().unwrap();
            // Decay 24h total (rough approximation)
            self.total_liquidated_24h = (self.total_liquidated_24h - old.size).max(0.0);
        }
        
        // Compute current intensity
        self.current_intensity = self.compute_intensity(now);
        
        // Check if cascade has ended
        if self.cascade_active && self.current_intensity < 1.5 * self.base_intensity {
            self.cascade_active = false;
            self.cascade_start = None;
        }
    }
    
    fn compute_intensity(&self, now: Instant) -> f64 {
        let mut intensity = self.base_intensity;
        
        for liq in &self.recent_liquidations {
            let elapsed = now.duration_since(liq.timestamp).as_secs_f64();
            intensity += self.excitation * liq.size * (-self.decay * elapsed).exp();
        }
        
        intensity
    }
    
    fn tail_risk_multiplier(&self) -> f64 {
        // Use this to scale down positions and widen quotes
        // 1.0 = normal, >1.0 = elevated risk
        (self.current_intensity / self.base_intensity).max(1.0)
    }
    
    fn should_pull_quotes(&self) -> bool {
        // Pull quotes entirely if cascade is severe
        self.cascade_active && self.tail_risk_multiplier() > 5.0
    }
    
    fn cascade_direction(&self) -> Option<Side> {
        // Which direction is the cascade going?
        // Longs being liquidated = price falling = cascade down
        if !self.cascade_active {
            return None;
        }
        
        let recent: Vec<_> = self.recent_liquidations.iter()
            .filter(|e| e.timestamp > Instant::now() - Duration::from_secs(60))
            .collect();
        
        let long_liqs: f64 = recent.iter()
            .filter(|e| e.direction == Side::Buy)  // Long being liquidated
            .map(|e| e.size)
            .sum();
        
        let short_liqs: f64 = recent.iter()
            .filter(|e| e.direction == Side::Sell)  // Short being liquidated
            .map(|e| e.size)
            .sum();
        
        if long_liqs > short_liqs * 1.5 {
            Some(Side::Sell)  // Cascade is downward
        } else if short_liqs > long_liqs * 1.5 {
            Some(Side::Buy)   // Cascade is upward
        } else {
            None
        }
    }
}
```

---

## Part 11: Integration - The Quote Function

### From State to Quote

Now we connect everything. Given all state variables, compute the quote:

```rust
struct QuoteCalculator {
    // Risk parameters
    gamma: f64,            // Risk aversion
    max_inventory: f64,    // Position limit
    
    // Quote parameters
    min_spread: f64,       // Floor on spread (fees + min profit)
    max_spread: f64,       // Ceiling (beyond this, no fills)
}

struct Quote {
    bid_price: f64,
    bid_size: f64,
    ask_price: f64,
    ask_size: f64,
}

impl QuoteCalculator {
    fn compute_quote(&self, state: &MarketState) -> Quote {
        // 1. BASE SPREAD (Avellaneda-Stoikov style)
        // spread = γ × σ² × T + 2/γ × ln(1 + γ/k)
        // Simplified: spread ≈ γ × σ × √T (for small γ)
        let sigma = state.volatility.current_sigma();
        let T = 1.0 / 3600.0;  // 1 second holding horizon
        let base_spread = self.gamma * sigma * T.sqrt();
        
        // 2. ADVERSE SELECTION ADJUSTMENT
        // Widen spread proportional to informed flow probability
        let as_adjustment = state.adverse_selection.spread_adjustment();
        
        // 3. REGIME ADJUSTMENT
        let regime_mult = match state.volatility.current_regime {
            VolRegime::Low => 0.8,
            VolRegime::Normal => 1.0,
            VolRegime::High => 1.5,
            VolRegime::Extreme => 3.0,
        };
        
        // 4. LIQUIDATION RISK ADJUSTMENT
        let liq_mult = state.liquidation.tail_risk_multiplier().min(3.0);
        
        // 5. TOTAL SPREAD
        let total_spread = (base_spread + as_adjustment) * regime_mult * liq_mult;
        let half_spread = total_spread.max(self.min_spread / 2.0).min(self.max_spread / 2.0);
        
        // 6. INVENTORY SKEW
        // If long, make ask tighter (want to sell), bid wider (don't want more)
        // reservation_price = mid - γ × σ² × q × T
        let q = state.inventory.position();
        let inventory_skew = self.gamma * sigma.powi(2) * q * T;
        let reservation = state.price.mid - inventory_skew;
        
        // 7. FLOW IMBALANCE ADJUSTMENT
        // If heavy buy flow, tighten ask (capture spread before price rises)
        let flow_adj = state.order_flow.flow_imbalance() * 0.0001;  // Small adjustment
        
        // 8. FINAL PRICES
        let bid_price = reservation - half_spread - flow_adj;
        let ask_price = reservation + half_spread - flow_adj;
        
        // 9. SIZES
        // Reduce size when:
        // - Near inventory limits
        // - High adverse selection
        // - Cascade active
        let base_size = self.compute_base_size(state);
        let size_mult = self.compute_size_multiplier(state);
        
        // Asymmetric sizing based on inventory
        let inventory_ratio = q / self.max_inventory;  // [-1, 1]
        let bid_size_mult = (1.0 - inventory_ratio.max(0.0)).max(0.1);  // Reduce bid if long
        let ask_size_mult = (1.0 + inventory_ratio.min(0.0)).max(0.1);  // Reduce ask if short
        
        Quote {
            bid_price: self.round_to_tick(bid_price, state.tick_size),
            bid_size: base_size * size_mult * bid_size_mult,
            ask_price: self.round_to_tick(ask_price, state.tick_size),
            ask_size: base_size * size_mult * ask_size_mult,
        }
    }
    
    fn compute_base_size(&self, state: &MarketState) -> f64 {
        // Size based on:
        // - Available margin
        // - Typical fill rate (don't over-quote)
        // - Risk budget
        let margin_available = state.account.available_margin();
        let margin_per_unit = state.price.mid / state.leverage;
        
        // Don't use more than 10% of margin per quote
        let margin_size = margin_available * 0.1 / margin_per_unit;
        
        // Limit to fraction of daily volume
        let volume_size = state.daily_volume * 0.001;  // 0.1% of daily
        
        margin_size.min(volume_size)
    }
    
    fn compute_size_multiplier(&self, state: &MarketState) -> f64 {
        let mut mult = 1.0;
        
        // Reduce for high adverse selection
        mult *= 1.0 / (1.0 + state.adverse_selection.alpha);
        
        // Reduce for cascade risk
        mult *= 1.0 / state.liquidation.tail_risk_multiplier();
        
        // Reduce for extreme regime
        if state.volatility.current_regime == VolRegime::Extreme {
            mult *= 0.25;
        }
        
        mult.max(0.1)  // Never go below 10% of base
    }
    
    fn round_to_tick(&self, price: f64, tick_size: f64) -> f64 {
        (price / tick_size).round() * tick_size
    }
}
```

---

## Part 12: Integration - The Update Loop

### The Complete Pipeline

```rust
struct MarketMakerEngine {
    // State
    state: MarketState,
    
    // Quote calculator
    quoter: QuoteCalculator,
    
    // Connection
    exchange: ExchangeConnection,
    
    // Current quotes
    live_quotes: Option<Quote>,
}

struct MarketState {
    // All our process states
    price: PriceState,
    volatility: VolatilityState,
    inventory: InventoryState,
    spread: SpreadState,
    order_flow: OrderFlowState,
    adverse_selection: AdverseSelectionState,
    queue: QueueState,
    correlation: CorrelationState,  // For multi-asset
    funding: FundingState,
    liquidation: LiquidationState,
    
    // Static config
    tick_size: f64,
    leverage: f64,
    daily_volume: f64,
    
    // Account state
    account: AccountState,
}

impl MarketMakerEngine {
    async fn run(&mut self) {
        loop {
            tokio::select! {
                // L2 book update
                Some(book) = self.exchange.recv_book() => {
                    self.handle_book_update(book);
                }
                
                // Trade (public)
                Some(trade) = self.exchange.recv_trade() => {
                    self.handle_trade(trade);
                }
                
                // Our fill
                Some(fill) = self.exchange.recv_fill() => {
                    self.handle_fill(fill);
                }
                
                // Funding update
                Some(funding) = self.exchange.recv_funding() => {
                    self.handle_funding(funding);
                }
                
                // Liquidation event
                Some(liq) = self.exchange.recv_liquidation() => {
                    self.handle_liquidation(liq);
                }
                
                // Periodic housekeeping
                _ = tokio::time::sleep(Duration::from_millis(100)) => {
                    self.periodic_update();
                }
            }
        }
    }
    
    fn handle_book_update(&mut self, book: OrderBook) {
        let now = Instant::now();
        let mid = (book.best_bid + book.best_ask) / 2.0;
        
        // Update price state
        self.state.price.update(mid, now);
        
        // Update spread state
        self.state.spread.update(book.best_bid, book.best_ask, 
                                  &self.state.volatility, now);
        
        // Update queue positions
        self.state.queue.update(&book, &[], now);
        
        // Maybe update quotes
        self.maybe_requote();
    }
    
    fn handle_trade(&mut self, trade: Trade) {
        let now = Instant::now();
        let dt = now.duration_since(self.state.price.last_update).as_secs_f64();
        
        // Update volatility with return
        let log_return = (trade.price / self.state.price.mid).ln();
        self.state.volatility.update(log_return, dt, now);
        
        // Update order flow
        self.state.order_flow.update(Some(&trade), now);
        
        // Update adverse selection signals
        self.state.adverse_selection.update(trade.price, now);
        
        // Update price
        self.state.price.update(trade.price, now);
        
        // Maybe update quotes
        self.maybe_requote();
    }
    
    fn handle_fill(&mut self, fill: Fill) {
        let now = Instant::now();
        
        // Record fill in inventory
        self.state.inventory.record_fill(fill.clone());
        
        // Record for adverse selection measurement
        self.state.adverse_selection.record_fill(fill.clone(), 
                                                  self.state.price.mid, now);
        
        // Update queue (our order was removed)
        self.state.queue.order_cancelled(&fill.order_id);
        
        // Must update quotes (we need to requote the filled side)
        self.requote();
    }
    
    fn handle_funding(&mut self, funding: FundingUpdate) {
        self.state.funding.update(funding.rates, Instant::now());
        
        // Update adverse selection signal
        if let Some(ref_rate) = funding.reference_rate {
            let our_rate = funding.rates.values().next().unwrap_or(&0.0);
            self.state.adverse_selection.funding_divergence = (our_rate - ref_rate).abs();
        }
    }
    
    fn handle_liquidation(&mut self, liq: LiquidationEvent) {
        self.state.liquidation.update(Some(liq), Instant::now());
        
        // Check if we should pull quotes
        if self.state.liquidation.should_pull_quotes() {
            self.pull_quotes();
        } else {
            // Widen quotes
            self.requote();
        }
    }
    
    fn periodic_update(&mut self) {
        let now = Instant::now();
        
        // Decay liquidation intensity
        self.state.liquidation.update(None, now);
        
        // Update order flow (decay Hawkes intensity)
        self.state.order_flow.update(None, now);
        
        // Check model health
        self.validate_models();
    }
    
    fn maybe_requote(&mut self) {
        // Only requote if:
        // 1. Price moved enough
        // 2. Spread changed significantly
        // 3. Enough time passed
        
        if let Some(ref current) = self.live_quotes {
            let mid = self.state.price.mid;
            let current_mid = (current.bid_price + current.ask_price) / 2.0;
            
            // Requote if mid moved more than 10% of spread
            let spread = current.ask_price - current.bid_price;
            if (mid - current_mid).abs() > spread * 0.1 {
                self.requote();
            }
        } else {
            self.requote();
        }
    }
    
    fn requote(&mut self) {
        let quote = self.quoter.compute_quote(&self.state);
        
        // Validate quote makes sense
        if !self.validate_quote(&quote) {
            return;
        }
        
        // Send to exchange
        self.exchange.update_quotes(&quote);
        self.live_quotes = Some(quote);
    }
    
    fn pull_quotes(&mut self) {
        self.exchange.cancel_all();
        self.live_quotes = None;
    }
    
    fn validate_quote(&self, quote: &Quote) -> bool {
        // Sanity checks
        if quote.bid_price >= quote.ask_price {
            return false;  // Crossed quote
        }
        if quote.bid_price <= 0.0 || quote.ask_price <= 0.0 {
            return false;
        }
        if quote.bid_size < 0.0 || quote.ask_size < 0.0 {
            return false;
        }
        
        // Check spread is reasonable
        let spread = quote.ask_price - quote.bid_price;
        let mid = (quote.bid_price + quote.ask_price) / 2.0;
        let spread_bps = spread / mid * 10000.0;
        
        if spread_bps < 1.0 || spread_bps > 500.0 {
            return false;  // Spread too tight or too wide
        }
        
        true
    }
    
    fn validate_models(&self) {
        // Check if models are performing as expected
        // Log warnings if calibration seems stale
        
        // Check volatility estimate vs realized
        let predicted_vol = self.state.volatility.current_sigma();
        let realized_vol = self.state.volatility.realized_vol_1m;
        let vol_error = (predicted_vol - realized_vol).abs() / realized_vol;
        
        if vol_error > 0.5 {
            log::warn!("Volatility model error: {:.1}%", vol_error * 100.0);
        }
        
        // Check adverse selection predictions
        let predicted_as = self.state.adverse_selection.alpha;
        let realized_as = self.state.adverse_selection.realized_adverse_selection();
        
        // More validation...
    }
}
```

---

## Part 13: Calibration

### What Needs Calibration

Each process has parameters that must be estimated from data:

| Process | Parameters | Calibration Method |
|---------|------------|-------------------|
| Volatility | κ, θ, ξ per regime | MLE on historical returns |
| Spread | κ_s, θ_s, β_v | Linear regression of spread on vol |
| Order Flow | μ, α, β, γ | MLE on trade timestamps |
| Adverse Selection | β coefficients | Logistic regression on fill outcomes |
| Queue | cancel_rate, exec_rate | Empirical from book changes |
| Correlation | λ values | Chosen by desired horizon |
| Funding | κ_f | Regression on funding history |
| Liquidation | μ, α, β | MLE on liquidation events |

### Calibration Code Structure

```rust
struct ModelCalibrator {
    historical_trades: Vec<Trade>,
    historical_books: Vec<OrderBook>,
    historical_funding: Vec<FundingSnapshot>,
    historical_liquidations: Vec<LiquidationEvent>,
}

impl ModelCalibrator {
    fn calibrate_volatility(&self) -> VolatilityParams {
        // 1. Compute returns
        let returns: Vec<f64> = self.historical_trades
            .windows(2)
            .map(|w| (w[1].price / w[0].price).ln())
            .collect();
        
        // 2. Classify into regimes based on realized vol
        let regime_returns = self.classify_by_regime(&returns);
        
        // 3. For each regime, estimate parameters
        let mut params = HashMap::new();
        for (regime, regime_rets) in regime_returns {
            let (kappa, theta, xi) = self.estimate_ou_params(&regime_rets);
            params.insert(regime, VolParams { kappa, theta, xi });
        }
        
        VolatilityParams { regime_params: params }
    }
    
    fn calibrate_hawkes(&self) -> HawkesParams {
        // Maximum likelihood estimation for Hawkes process
        // log-likelihood: Σ log(λ(t_i)) - ∫λ(t)dt
        
        let timestamps: Vec<f64> = self.historical_trades
            .iter()
            .map(|t| t.timestamp.as_secs_f64())
            .collect();
        
        // Optimize over (μ, α, β)
        let (mu, alpha, beta) = self.optimize_hawkes_likelihood(&timestamps);
        
        HawkesParams { mu, alpha, beta }
    }
    
    fn calibrate_adverse_selection(&self, fills: &[FillWithOutcome]) -> ASParams {
        // Logistic regression: P(adverse) = σ(β'x)
        // Features: funding_div, vol_surprise, flow_imb, liq_intensity
        // Target: did price move against us after fill?
        
        let features: Vec<Vec<f64>> = fills.iter()
            .map(|f| vec![
                f.funding_divergence,
                f.volatility_surprise,
                f.flow_imbalance,
                f.liquidation_intensity,
            ])
            .collect();
        
        let targets: Vec<f64> = fills.iter()
            .map(|f| if f.was_adverse { 1.0 } else { 0.0 })
            .collect();
        
        // Fit logistic regression (use external library)
        let betas = logistic_regression(&features, &targets);
        
        ASParams {
            beta_0: betas[0],
            beta_funding: betas[1],
            beta_vol: betas[2],
            beta_flow: betas[3],
            beta_liq: betas[4],
        }
    }
}
```

---

## Part 14: Summary - The Complete Picture

### What We've Built

1. **Price Process**: Jump-diffusion for price evolution
2. **Volatility Process**: Regime-switching stochastic volatility
3. **Inventory Process**: Controlled by quotes, subject to fill randomness
4. **Spread Process**: Mean-reverting with vol dependency
5. **Order Flow Process**: Hawkes (self-exciting) for trade arrivals
6. **Adverse Selection Process**: Time-varying informed flow, measured and predicted
7. **Queue Position Process**: For fill probability estimation
8. **Correlation Process**: Multi-scale EWMA with stress detection
9. **Funding Process**: Mean-reverting with arbitrage detection
10. **Liquidation Process**: Self-exciting cascade risk

### The Data Flow

```
TICK → PARSE → UPDATE STATE → COMPUTE QUOTE → VALIDATE → EXECUTE
         ↓
    [Price, Vol, Flow, Spread, Queue, AS, Liq, Funding, Corr]
                    ↓
              STATE VECTOR
                    ↓
         [Avellaneda-Stoikov + Adjustments]
                    ↓
              (bid, ask, sizes)
                    ↓
              EXCHANGE
```

### Key Principles Applied

1. **PnL Decomposition**: Each cost driver has its own process
2. **Survival First**: Leverage means constraints dominate expected value
3. **Adaptive**: Every parameter is time-varying, not fixed
4. **Measurable**: Adverse selection, queue position, etc. are measured, not assumed
5. **Calibratable**: Each model can be fit to historical data
6. **Validatable**: Each prediction can be compared to realized outcomes

### What's Next

1. **Implementation**: Turn this spec into actual Rust code
2. **Backtesting**: Replay historical data through the engine
3. **Paper Trading**: Run against live data with no real orders
4. **Calibration**: Fit all parameters to your data
5. **Production**: Deploy with real capital (start small)

---

## Appendix: Mathematical Notation Summary

| Symbol | Meaning |
|--------|---------|
| p | Log price |
| σ, v | Volatility, variance |
| q | Inventory |
| λ | Intensity (arrivals per unit time) |
| α (Hawkes) | Self-excitation parameter |
| α (AS) | Informed flow probability |
| β | Decay rate |
| κ | Mean reversion speed |
| θ | Long-run mean |
| ρ | Correlation |
| Σ | Covariance matrix |
| γ | Risk aversion |
| δ | Quote depth (distance from mid) |
| W | Brownian motion |
| N | Counting process (Poisson or Hawkes) |
| J | Jump size |
