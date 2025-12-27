# The Complete Market Making System Manual
## From Zero to Production on Hyperliquid

---

# Table of Contents

1. [Philosophy & Mental Models](#1-philosophy--mental-models)
2. [System Architecture](#2-system-architecture)
3. [Data Infrastructure](#3-data-infrastructure)
4. [State Estimation Engine](#4-state-estimation-engine)
5. [The Quoting Engine](#5-the-quoting-engine)
6. [Order Management](#6-order-management)
7. [Risk Management](#7-risk-management)
8. [Calibration Pipeline](#8-calibration-pipeline)
9. [Backtesting Framework](#9-backtesting-framework)
10. [Paper Trading](#10-paper-trading)
11. [Production Deployment](#11-production-deployment)
12. [Monitoring & Alerting](#12-monitoring--alerting)
13. [Performance Analysis](#13-performance-analysis)
14. [Continuous Improvement](#14-continuous-improvement)
15. [Common Failure Modes](#15-common-failure-modes)
16. [Appendix: Mathematical Reference](#16-appendix-mathematical-reference)

---

# 1. Philosophy & Mental Models

## 1.1 What You're Actually Doing

You are running a **liquidity transformation business**. 

Customers (traders) want to trade NOW. You provide that service. Your fee is the spread. Your costs are:
1. Inventory risk (holding positions that move against you)
2. Adverse selection (trading against people who know more)
3. Funding (carrying perpetual positions)
4. Infrastructure (servers, data, development)

**Profit = Revenue - Costs**

This is a business with thin margins and high volume. A 0.5bp edge on $10M daily volume = $500/day = $180K/year. But a 2% drawdown on $200K capital = $4K loss in seconds.

## 1.2 The Hierarchy of Concerns

```
SURVIVAL (don't blow up)
    ↓
RISK MANAGEMENT (limit drawdowns)
    ↓
SPREAD CAPTURE (make money)
    ↓
OPTIMIZATION (make more money)
```

Never optimize before you've ensured survival. A strategy with 100% annual return and 50% drawdown is worse than one with 30% return and 5% drawdown—because you'll eventually hit the drawdown at the wrong time.

## 1.3 Key Mental Models

### The Leverage Trap
At 50x leverage, a 2% move = 100% loss. You WILL see 2% moves. Therefore, you MUST size positions such that even extreme moves don't kill you.

**Rule**: Maximum position such that a 5-sigma move in your expected holding time leaves you with >50% of capital.

### Adverse Selection is the Real Enemy
Spread capture is easy. Avoiding adverse selection is hard. The market has a way of filling your bids right before crashes and your asks right before rallies.

**Rule**: If you're getting filled more than expected, you're probably being adversely selected.

### Regimes Are Real
Markets are not stationary. The same strategy that makes money in low volatility will lose money in high volatility. You need to detect and adapt.

**Rule**: Always know what regime you're in. Have different parameters for each.

### Fill Rate ≠ Profit
Getting filled is not success. Getting filled and making money is success. A strategy that quotes tight and gets filled constantly can lose money if adverse selection exceeds spread.

**Rule**: Measure realized P&L per fill, not just fill count.

## 1.4 Success Metrics

| Metric | Target | Red Flag |
|--------|--------|----------|
| Sharpe Ratio | > 3.0 | < 1.0 |
| Max Drawdown | < 5% | > 10% |
| Win Rate | > 55% | < 45% |
| Profit per Fill | > 0.5bp | < 0 |
| Inventory Turnover | > 10x/day | < 2x/day |
| Time at Max Inventory | < 10% | > 30% |

---

# 2. System Architecture

## 2.1 High-Level Components

```
┌─────────────────────────────────────────────────────────────────────┐
│                         MARKET DATA                                  │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐       │
│  │ L2 Book │ │ Trades  │ │ Funding │ │ Liqs    │ │ Oracle  │       │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘       │
│       │           │           │           │           │             │
│       └───────────┴───────────┴───────────┴───────────┘             │
│                               │                                      │
│                               ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    DATA NORMALIZATION                        │   │
│  │   • Timestamp alignment                                      │   │
│  │   • Price/size normalization                                 │   │
│  │   • Missing data handling                                    │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      STATE ESTIMATION                                │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │ Price    │ │ Vol      │ │ Flow     │ │ Adverse  │ │ Liq      │  │
│  │ Process  │ │ Process  │ │ Process  │ │ Selection│ │ Cascade  │  │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘  │
│       └────────────┴────────────┴────────────┴────────────┘         │
│                               │                                      │
│                               ▼                                      │
│                    ┌─────────────────┐                              │
│                    │  STATE VECTOR   │                              │
│                    └────────┬────────┘                              │
└─────────────────────────────┼───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       DECISION ENGINE                                │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                  RISK CHECKS (First!)                        │   │
│  │   • Position limits                                          │   │
│  │   • Margin constraints                                       │   │
│  │   • Regime gates                                             │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                               │                                      │
│                    ┌──────────┴──────────┐                          │
│                    │   Pass risk checks? │                          │
│                    └──────────┬──────────┘                          │
│                          No   │   Yes                                │
│                    ┌──────────┴──────────┐                          │
│                    ▼                     ▼                          │
│           ┌─────────────┐      ┌─────────────────┐                  │
│           │ PULL QUOTES │      │ QUOTE CALCULATOR │                  │
│           └─────────────┘      │  • Compute ladder│                  │
│                                │  • Apply skew    │                  │
│                                │  • Apply limits  │                  │
│                                └────────┬────────┘                  │
│                                         │                            │
│                                         ▼                            │
│                              ┌─────────────────┐                     │
│                              │  TARGET LADDER  │                     │
│                              └────────┬────────┘                     │
└───────────────────────────────────────┼─────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      ORDER MANAGEMENT                                │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    ORDER DIFFER                              │   │
│  │   • Compare target vs live                                   │   │
│  │   • Minimize churn                                           │   │
│  │   • Prioritize cancels over new                              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                               │                                      │
│                               ▼                                      │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐                         │
│  │  CANCELS  │ │  AMENDS   │ │   NEWS    │                         │
│  └─────┬─────┘ └─────┬─────┘ └─────┬─────┘                         │
│        └─────────────┴─────────────┘                                │
│                      │                                               │
│                      ▼                                               │
│              ┌───────────────┐                                      │
│              │   EXCHANGE    │                                      │
│              └───────────────┘                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## 2.2 Module Responsibilities

| Module | Input | Output | Frequency |
|--------|-------|--------|-----------|
| Data Normalization | Raw websocket | Typed events | Every message |
| State Estimation | Events | State vector | Every event |
| Risk Checks | State | Pass/Fail | Every quote cycle |
| Quote Calculator | State | Target ladder | Every quote cycle |
| Order Differ | Target + Live | Order actions | Every quote cycle |
| Order Executor | Actions | Confirmations | Async |

## 2.3 Technology Choices

### Language: Rust

**Why Rust:**
- Zero-cost abstractions (fast as C, safe as Java)
- No garbage collection pauses
- Excellent async ecosystem (tokio)
- Strong type system catches bugs at compile time
- Memory safety prevents crashes

**Why not Python:**
- GC pauses in hot path
- GIL limits parallelism
- Too slow for tick-by-tick processing

**Why not C++:**
- Memory unsafety
- Build system complexity
- Less ergonomic async

### Key Dependencies

```toml
[dependencies]
tokio = { version = "1", features = ["full"] }  # Async runtime
serde = { version = "1", features = ["derive"] } # Serialization
serde_json = "1"                                 # JSON
tokio-tungstenite = "0.20"                       # WebSocket
reqwest = { version = "0.11", features = ["json"] } # HTTP
tracing = "0.1"                                  # Logging
metrics = "0.21"                                 # Metrics
rust_decimal = "1"                               # Precise decimals
dashmap = "5"                                    # Concurrent hashmap
crossbeam = "0.8"                                # Lock-free channels
```

## 2.4 Latency Budget

| Component | Target | Max |
|-----------|--------|-----|
| Websocket receive | <1ms | 5ms |
| State update | <100μs | 500μs |
| Quote calculation | <50μs | 200μs |
| Order diffing | <20μs | 100μs |
| Network to exchange | <10ms | 50ms |
| **Total tick-to-order** | **<15ms** | **60ms** |

If you're consistently over budget, profile and optimize.

---

# 3. Data Infrastructure

## 3.1 Data Sources

### Primary (Real-time)

| Source | Data | Update Frequency | Criticality |
|--------|------|------------------|-------------|
| L2 Order Book | Bids/asks at each level | Every change | CRITICAL |
| Trades | Price, size, side, time | Every trade | HIGH |
| User Fills | Your fills + queue info | Every fill | CRITICAL |
| Funding Rate | Current + predicted | Every update | MEDIUM |
| Liquidations | Size, price, direction | Every event | HIGH |
| Account | Margin, positions | On change | HIGH |

### Secondary (Periodic)

| Source | Data | Update Frequency | Use |
|--------|------|------------------|-----|
| Historical trades | Full trade history | Daily batch | Calibration |
| Historical books | Book snapshots | Hourly | Backtesting |
| Cross-exchange | Binance/other prices | 100ms | Arb detection |
| Oracle prices | Spot reference | 1s | Fair value |

## 3.2 Hyperliquid WebSocket Streams

```rust
// Subscription message format
pub enum Subscription {
    L2Book { coin: String },          // Order book
    Trades { coin: String },          // Public trades
    UserFills { user: String },       // Your fills
    UserFunding { user: String },     // Your funding
    Liquidations,                     // All liquidations
    AllMids,                          // All mid prices
}

// Connection management
pub struct DataManager {
    ws: WebSocketStream,
    subscriptions: Vec<Subscription>,
    reconnect_delay: Duration,
    last_message: Instant,
}

impl DataManager {
    pub async fn connect(&mut self) -> Result<()> {
        // 1. Connect to websocket
        self.ws = connect_ws("wss://api.hyperliquid.xyz/ws").await?;
        
        // 2. Subscribe to all streams
        for sub in &self.subscriptions {
            self.ws.send(sub.to_message()).await?;
        }
        
        // 3. Wait for confirmation
        self.await_subscribed().await?;
        
        Ok(())
    }
    
    pub async fn run(&mut self, tx: Sender<MarketEvent>) {
        loop {
            tokio::select! {
                msg = self.ws.next() => {
                    match msg {
                        Some(Ok(Message::Text(text))) => {
                            if let Some(event) = self.parse(&text) {
                                tx.send(event).await.ok();
                            }
                            self.last_message = Instant::now();
                        }
                        Some(Err(e)) => {
                            tracing::error!("WebSocket error: {}", e);
                            self.reconnect().await;
                        }
                        None => {
                            self.reconnect().await;
                        }
                        _ => {}
                    }
                }
                _ = tokio::time::sleep(Duration::from_secs(5)) => {
                    // Heartbeat check
                    if self.last_message.elapsed() > Duration::from_secs(10) {
                        tracing::warn!("No messages for 10s, reconnecting");
                        self.reconnect().await;
                    }
                }
            }
        }
    }
}
```

## 3.3 Data Normalization

Raw data from exchanges is messy. Normalize it.

```rust
pub struct NormalizedL2 {
    pub symbol: Symbol,
    pub timestamp: Timestamp,
    pub bids: Vec<PriceLevel>,   // Sorted best-to-worst
    pub asks: Vec<PriceLevel>,   // Sorted best-to-worst
    pub sequence: u64,           // For ordering
}

pub struct PriceLevel {
    pub price: Decimal,          // Use Decimal, not f64
    pub size: Decimal,
    pub num_orders: Option<u32>, // If available
}

pub struct NormalizedTrade {
    pub symbol: Symbol,
    pub timestamp: Timestamp,
    pub price: Decimal,
    pub size: Decimal,
    pub side: Side,              // Taker side
    pub trade_id: TradeId,
}

impl DataNormalizer {
    pub fn normalize_book(&self, raw: RawL2Update) -> NormalizedL2 {
        // 1. Parse prices and sizes to Decimal
        let bids: Vec<PriceLevel> = raw.bids.iter()
            .map(|[p, s]| PriceLevel {
                price: Decimal::from_str(p).unwrap(),
                size: Decimal::from_str(s).unwrap(),
                num_orders: None,
            })
            .filter(|l| l.size > Decimal::ZERO)  // Filter zero sizes
            .collect();
        
        // 2. Sort (should already be sorted, but verify)
        // Bids: highest first
        // Asks: lowest first
        
        // 3. Validate
        if let (Some(best_bid), Some(best_ask)) = (bids.first(), asks.first()) {
            if best_bid.price >= best_ask.price {
                tracing::warn!("Crossed book detected!");
            }
        }
        
        NormalizedL2 {
            symbol: raw.coin.into(),
            timestamp: Timestamp::now(),
            bids,
            asks,
            sequence: raw.sequence,
        }
    }
}
```

## 3.4 Data Quality Checks

| Check | Action if Failed |
|-------|------------------|
| Crossed book | Log warning, use previous |
| Stale data (>1s old) | Log warning, reduce confidence |
| Missing sequence | Request snapshot |
| Price outside bounds | Reject, use previous |
| Size negative | Reject |
| Timestamp in future | Use local time |

```rust
pub struct DataQualityMonitor {
    last_sequences: HashMap<Symbol, u64>,
    last_timestamps: HashMap<Symbol, Timestamp>,
    anomaly_counts: HashMap<AnomalyType, u64>,
}

impl DataQualityMonitor {
    pub fn check(&mut self, event: &MarketEvent) -> DataQuality {
        let mut issues = vec![];
        
        // Sequence check
        if let Some(&last_seq) = self.last_sequences.get(&event.symbol) {
            if event.sequence <= last_seq {
                issues.push(Issue::OutOfOrder);
            } else if event.sequence > last_seq + 1 {
                issues.push(Issue::GapDetected(event.sequence - last_seq - 1));
            }
        }
        
        // Staleness check
        if let Some(&last_ts) = self.last_timestamps.get(&event.symbol) {
            if event.timestamp < last_ts {
                issues.push(Issue::TimestampRegression);
            }
        }
        
        // Latency check
        let latency = Timestamp::now() - event.timestamp;
        if latency > Duration::from_secs(1) {
            issues.push(Issue::StaleData(latency));
        }
        
        // Update tracking
        self.last_sequences.insert(event.symbol.clone(), event.sequence);
        self.last_timestamps.insert(event.symbol.clone(), event.timestamp);
        
        if issues.is_empty() {
            DataQuality::Good
        } else {
            DataQuality::Degraded(issues)
        }
    }
}
```

## 3.5 Data Storage

For calibration and analysis, you need historical data.

```rust
// Schema for historical storage
pub struct TradeRecord {
    pub timestamp: i64,      // Unix micros
    pub symbol: String,
    pub price: f64,
    pub size: f64,
    pub side: i8,            // 1 = buy, -1 = sell
}

pub struct BookSnapshot {
    pub timestamp: i64,
    pub symbol: String,
    pub bids: Vec<(f64, f64)>,  // (price, size)
    pub asks: Vec<(f64, f64)>,
}

// Storage backend options:
// 1. Parquet files (best for batch analysis)
// 2. TimescaleDB (best for time-series queries)
// 3. ClickHouse (best for OLAP)
// 4. Redis (best for recent data cache)
```

---

# 4. State Estimation Engine

## 4.1 The State Vector

Everything we need to know to quote:

```rust
#[derive(Clone, Debug)]
pub struct MarketState {
    // === PRICE STATE ===
    pub mid: Decimal,
    pub bid: Decimal,
    pub ask: Decimal,
    pub spread: Decimal,
    pub last_trade_price: Decimal,
    pub last_update: Timestamp,
    
    // === VOLATILITY STATE ===
    pub sigma_instant: f64,      // EWMA of squared returns
    pub sigma_1m: f64,           // 1-minute realized
    pub sigma_5m: f64,           // 5-minute realized
    pub sigma_1h: f64,           // 1-hour realized
    pub regime: VolRegime,
    pub regime_confidence: f64,  // How confident in regime
    
    // === INVENTORY STATE ===
    pub position: Decimal,
    pub entry_price: Decimal,
    pub unrealized_pnl: Decimal,
    pub position_age: Duration,  // How long held
    
    // === ORDER FLOW STATE ===
    pub lambda_buy: f64,         // Hawkes intensity
    pub lambda_sell: f64,
    pub flow_imbalance: f64,     // [-1, 1]
    pub trade_intensity: f64,    // Trades per second
    
    // === ADVERSE SELECTION STATE ===
    pub alpha: f64,              // Informed flow probability
    pub realized_as: f64,        // Measured AS at touch
    pub as_decay_rate: f64,      // How fast AS decays with depth
    
    // === BOOK STATE ===
    pub book_imbalance: f64,     // (bid_depth - ask_depth) / total
    pub depth_bid_1: Decimal,    // Depth at 1bp
    pub depth_ask_1: Decimal,
    pub depth_bid_5: Decimal,    // Depth at 5bp
    pub depth_ask_5: Decimal,
    
    // === FUNDING STATE ===
    pub funding_rate: f64,       // Current rate
    pub predicted_funding: f64,  // Predicted next
    pub time_to_funding: Duration,
    
    // === RISK STATE ===
    pub liquidation_intensity: f64,  // Cascade risk
    pub margin_used: Decimal,
    pub margin_available: Decimal,
    pub leverage: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VolRegime {
    Low,      // σ < 0.5 × historical
    Normal,   // 0.5 - 1.5 × historical
    High,     // 1.5 - 3.0 × historical
    Extreme,  // > 3.0 × historical
}
```

## 4.2 Volatility Estimation

### EWMA for Instantaneous Vol

```rust
pub struct VolatilityEstimator {
    // EWMA variance
    ewma_var: f64,
    lambda: f64,  // Decay factor, typically 0.94
    
    // Realized vol at multiple scales
    returns_1m: RingBuffer<(Timestamp, f64)>,
    returns_5m: RingBuffer<(Timestamp, f64)>,
    returns_1h: RingBuffer<(Timestamp, f64)>,
    
    // Historical baseline for regime detection
    historical_vol: f64,
    
    // For annualization
    samples_per_year: f64,
}

impl VolatilityEstimator {
    pub fn new(historical_vol: f64, lambda: f64) -> Self {
        Self {
            ewma_var: historical_vol.powi(2),
            lambda,
            returns_1m: RingBuffer::new(1000),
            returns_5m: RingBuffer::new(1000),
            returns_1h: RingBuffer::new(10000),
            historical_vol,
            samples_per_year: 365.25 * 24.0 * 3600.0,  // Per-second returns
        }
    }
    
    pub fn update(&mut self, log_return: f64, timestamp: Timestamp) {
        // EWMA update
        self.ewma_var = self.lambda * self.ewma_var 
                      + (1.0 - self.lambda) * log_return.powi(2);
        
        // Store for realized vol
        self.returns_1m.push((timestamp, log_return));
        self.returns_5m.push((timestamp, log_return));
        self.returns_1h.push((timestamp, log_return));
        
        // Prune old
        self.prune_old(timestamp);
    }
    
    pub fn instant_vol(&self) -> f64 {
        (self.ewma_var * self.samples_per_year).sqrt()
    }
    
    pub fn realized_vol(&self, window: Duration) -> f64 {
        let buffer = match window.as_secs() {
            0..=60 => &self.returns_1m,
            61..=300 => &self.returns_5m,
            _ => &self.returns_1h,
        };
        
        let cutoff = Timestamp::now() - window;
        let returns: Vec<f64> = buffer.iter()
            .filter(|(t, _)| *t > cutoff)
            .map(|(_, r)| *r)
            .collect();
        
        if returns.len() < 2 {
            return self.instant_vol();
        }
        
        let sum_sq: f64 = returns.iter().map(|r| r.powi(2)).sum();
        let var = sum_sq / returns.len() as f64;
        
        // Annualize
        (var * self.samples_per_year).sqrt()
    }
    
    pub fn regime(&self) -> VolRegime {
        let current = self.realized_vol(Duration::from_secs(300));  // 5-min
        let ratio = current / self.historical_vol;
        
        match ratio {
            r if r < 0.5 => VolRegime::Low,
            r if r < 1.5 => VolRegime::Normal,
            r if r < 3.0 => VolRegime::High,
            _ => VolRegime::Extreme,
        }
    }
    
    fn prune_old(&mut self, now: Timestamp) {
        self.returns_1m.retain(|(t, _)| now - *t < Duration::from_secs(60));
        self.returns_5m.retain(|(t, _)| now - *t < Duration::from_secs(300));
        self.returns_1h.retain(|(t, _)| now - *t < Duration::from_secs(3600));
    }
}
```

### Regime Detection with Hysteresis

```rust
pub struct RegimeDetector {
    current_regime: VolRegime,
    regime_start: Timestamp,
    vol_estimator: VolatilityEstimator,
    
    // Thresholds (with hysteresis)
    low_enter: f64,      // Enter Low when ratio < this
    low_exit: f64,       // Exit Low when ratio > this
    high_enter: f64,     // Enter High when ratio > this
    high_exit: f64,      // Exit High when ratio < this
    extreme_enter: f64,
    extreme_exit: f64,
    
    // Minimum time in regime before switching
    min_regime_duration: Duration,
}

impl RegimeDetector {
    pub fn update(&mut self, log_return: f64, timestamp: Timestamp) -> Option<RegimeChange> {
        self.vol_estimator.update(log_return, timestamp);
        
        let ratio = self.vol_estimator.realized_vol(Duration::from_secs(300)) 
                  / self.vol_estimator.historical_vol;
        
        let new_regime = self.determine_regime(ratio);
        
        if new_regime != self.current_regime {
            // Check minimum duration
            if timestamp - self.regime_start >= self.min_regime_duration {
                let old = self.current_regime;
                self.current_regime = new_regime;
                self.regime_start = timestamp;
                
                return Some(RegimeChange { from: old, to: new_regime, at: timestamp });
            }
        }
        
        None
    }
    
    fn determine_regime(&self, ratio: f64) -> VolRegime {
        use VolRegime::*;
        
        match self.current_regime {
            Low => {
                if ratio > self.low_exit {
                    if ratio > self.extreme_enter { Extreme }
                    else if ratio > self.high_enter { High }
                    else { Normal }
                } else { Low }
            }
            Normal => {
                if ratio < self.low_enter { Low }
                else if ratio > self.extreme_enter { Extreme }
                else if ratio > self.high_enter { High }
                else { Normal }
            }
            High => {
                if ratio < self.high_exit {
                    if ratio < self.low_enter { Low }
                    else { Normal }
                } else if ratio > self.extreme_enter { Extreme }
                else { High }
            }
            Extreme => {
                if ratio < self.extreme_exit {
                    if ratio < self.high_exit { 
                        if ratio < self.low_enter { Low }
                        else { Normal }
                    }
                    else { High }
                } else { Extreme }
            }
        }
    }
}
```

## 4.3 Order Flow Estimation (Hawkes)

```rust
pub struct HawkesEstimator {
    // Current intensities
    lambda_buy: f64,
    lambda_sell: f64,
    
    // Parameters (calibrated)
    mu: f64,       // Baseline intensity
    alpha: f64,    // Self-excitation
    beta: f64,     // Decay rate
    gamma: f64,    // Cross-excitation
    
    // Event history
    buy_times: VecDeque<f64>,   // Times as seconds since epoch
    sell_times: VecDeque<f64>,
    
    // For imbalance calculation
    recent_buy_volume: f64,
    recent_sell_volume: f64,
    volume_decay: f64,
}

impl HawkesEstimator {
    pub fn update(&mut self, event: Option<&TradeEvent>, now: f64) {
        // Add new event
        if let Some(trade) = event {
            let volume = trade.size.to_f64().unwrap();
            
            match trade.side {
                Side::Buy => {
                    self.buy_times.push_back(now);
                    self.recent_buy_volume += volume;
                }
                Side::Sell => {
                    self.sell_times.push_back(now);
                    self.recent_sell_volume += volume;
                }
            }
        }
        
        // Prune old events
        let cutoff = now - 300.0;  // Keep 5 minutes
        while self.buy_times.front().map_or(false, |&t| t < cutoff) {
            self.buy_times.pop_front();
        }
        while self.sell_times.front().map_or(false, |&t| t < cutoff) {
            self.sell_times.pop_front();
        }
        
        // Compute intensities
        self.lambda_buy = self.compute_intensity(&self.buy_times, &self.sell_times, now);
        self.lambda_sell = self.compute_intensity(&self.sell_times, &self.buy_times, now);
        
        // Decay volumes
        self.recent_buy_volume *= self.volume_decay;
        self.recent_sell_volume *= self.volume_decay;
    }
    
    fn compute_intensity(&self, same: &VecDeque<f64>, other: &VecDeque<f64>, now: f64) -> f64 {
        let mut lambda = self.mu;
        
        // Self-excitation
        for &t in same {
            lambda += self.alpha * (-self.beta * (now - t)).exp();
        }
        
        // Cross-excitation
        for &t in other {
            lambda += self.gamma * (-self.beta * (now - t)).exp();
        }
        
        lambda
    }
    
    pub fn imbalance(&self) -> f64 {
        let total = self.recent_buy_volume + self.recent_sell_volume;
        if total < 1e-10 { return 0.0; }
        
        ((self.recent_buy_volume - self.recent_sell_volume) / total).clamp(-1.0, 1.0)
    }
    
    pub fn intensity_ratio(&self) -> f64 {
        // Ratio of current intensity to baseline
        // > 1 means market is "hot"
        (self.lambda_buy + self.lambda_sell) / (2.0 * self.mu)
    }
}
```

## 4.4 Adverse Selection Estimation

```rust
pub struct AdverseSelectionEstimator {
    // Realized measurements
    realized_as_bid: ExponentialMovingAverage,
    realized_as_ask: ExponentialMovingAverage,
    
    // Pending measurements
    pending_fills: VecDeque<PendingFill>,
    measurement_delay: Duration,
    
    // Predictive model coefficients
    alpha_baseline: f64,
    beta_funding: f64,
    beta_volatility: f64,
    beta_flow: f64,
    beta_liquidation: f64,
    
    // Current signals
    funding_divergence: f64,
    volatility_surprise: f64,
    flow_imbalance: f64,
    liquidation_intensity: f64,
}

struct PendingFill {
    side: Side,
    price: f64,
    mid_at_fill: f64,
    timestamp: Timestamp,
    measure_at: Timestamp,
}

impl AdverseSelectionEstimator {
    pub fn record_fill(&mut self, side: Side, price: f64, mid: f64, now: Timestamp) {
        self.pending_fills.push_back(PendingFill {
            side,
            price,
            mid_at_fill: mid,
            timestamp: now,
            measure_at: now + self.measurement_delay,
        });
    }
    
    pub fn update(&mut self, current_mid: f64, now: Timestamp) {
        // Process pending fills that are ready
        while let Some(pending) = self.pending_fills.front() {
            if now < pending.measure_at {
                break;
            }
            
            let pending = self.pending_fills.pop_front().unwrap();
            
            // Compute adverse selection
            // AS = how much price moved against us after fill
            let mid_change = (current_mid - pending.mid_at_fill) / pending.mid_at_fill;
            
            let as_value = match pending.side {
                Side::Buy => -mid_change,   // Bought, price dropped = adverse
                Side::Sell => mid_change,   // Sold, price rose = adverse
            };
            
            // Update running estimate
            match pending.side {
                Side::Buy => self.realized_as_bid.update(as_value),
                Side::Sell => self.realized_as_ask.update(as_value),
            }
        }
    }
    
    pub fn update_signals(
        &mut self,
        funding_here: f64,
        funding_reference: f64,
        realized_vol: f64,
        implied_vol: f64,
        flow_imbalance: f64,
        liq_intensity: f64,
    ) {
        self.funding_divergence = (funding_here - funding_reference).abs();
        self.volatility_surprise = (realized_vol - implied_vol) / implied_vol.max(0.01);
        self.flow_imbalance = flow_imbalance;
        self.liquidation_intensity = liq_intensity;
    }
    
    /// Predicted probability of informed flow
    pub fn alpha(&self) -> f64 {
        let z = self.alpha_baseline
            + self.beta_funding * self.funding_divergence
            + self.beta_volatility * self.volatility_surprise.abs()
            + self.beta_flow * self.flow_imbalance.abs()
            + self.beta_liquidation * (self.liquidation_intensity - 1.0).max(0.0);
        
        // Sigmoid
        1.0 / (1.0 + (-z).exp())
    }
    
    /// Realized adverse selection (at touch)
    pub fn realized_as(&self) -> f64 {
        (self.realized_as_bid.value().abs() + self.realized_as_ask.value().abs()) / 2.0
    }
    
    /// Spread widening needed to compensate for AS
    pub fn as_spread_adjustment(&self) -> f64 {
        self.alpha() * self.realized_as() * 2.0  // 2x for safety
    }
}
```

## 4.5 Liquidation Cascade Detection

```rust
pub struct LiquidationCascadeDetector {
    // Hawkes model for liquidations
    current_intensity: f64,
    base_intensity: f64,
    excitation: f64,
    decay: f64,
    
    // Recent liquidations
    recent_liqs: VecDeque<LiquidationEvent>,
    
    // Cascade detection
    cascade_active: bool,
    cascade_direction: Option<Side>,
    cascade_start: Option<Timestamp>,
    
    // Thresholds
    cascade_start_threshold: f64,  // Intensity multiple to start
    cascade_end_threshold: f64,    // Intensity multiple to end
}

impl LiquidationCascadeDetector {
    pub fn update(&mut self, liq: Option<&LiquidationEvent>, now: Timestamp) {
        // Add new liquidation
        if let Some(l) = liq {
            self.recent_liqs.push_back(l.clone());
        }
        
        // Prune old
        let cutoff = now - Duration::from_secs(3600);
        while self.recent_liqs.front().map_or(false, |l| l.timestamp < cutoff) {
            self.recent_liqs.pop_front();
        }
        
        // Compute intensity
        self.current_intensity = self.compute_intensity(now);
        
        // Detect cascade state
        let intensity_ratio = self.current_intensity / self.base_intensity;
        
        if !self.cascade_active {
            if intensity_ratio > self.cascade_start_threshold {
                self.cascade_active = true;
                self.cascade_start = Some(now);
                self.cascade_direction = self.detect_direction();
            }
        } else {
            if intensity_ratio < self.cascade_end_threshold {
                self.cascade_active = false;
                self.cascade_start = None;
                self.cascade_direction = None;
            } else {
                // Update direction
                self.cascade_direction = self.detect_direction();
            }
        }
    }
    
    fn compute_intensity(&self, now: Timestamp) -> f64 {
        let mut intensity = self.base_intensity;
        
        for liq in &self.recent_liqs {
            let elapsed = (now - liq.timestamp).as_secs_f64();
            intensity += self.excitation * liq.size * (-self.decay * elapsed).exp();
        }
        
        intensity
    }
    
    fn detect_direction(&self) -> Option<Side> {
        let recent: Vec<_> = self.recent_liqs.iter()
            .filter(|l| l.timestamp > Timestamp::now() - Duration::from_secs(60))
            .collect();
        
        let long_liq_size: f64 = recent.iter()
            .filter(|l| l.direction == Side::Buy)  // Long being liquidated
            .map(|l| l.size)
            .sum();
        
        let short_liq_size: f64 = recent.iter()
            .filter(|l| l.direction == Side::Sell)  // Short being liquidated
            .map(|l| l.size)
            .sum();
        
        if long_liq_size > short_liq_size * 1.5 {
            Some(Side::Sell)  // Price is falling (longs getting rekt)
        } else if short_liq_size > long_liq_size * 1.5 {
            Some(Side::Buy)   // Price is rising (shorts getting rekt)
        } else {
            None
        }
    }
    
    pub fn risk_multiplier(&self) -> f64 {
        (self.current_intensity / self.base_intensity).max(1.0)
    }
    
    pub fn should_pull_quotes(&self) -> bool {
        self.cascade_active && self.risk_multiplier() > 5.0
    }
}
```

## 4.6 Unified State Update

```rust
pub struct StateEstimationEngine {
    // Component estimators
    vol_estimator: VolatilityEstimator,
    regime_detector: RegimeDetector,
    flow_estimator: HawkesEstimator,
    as_estimator: AdverseSelectionEstimator,
    liq_detector: LiquidationCascadeDetector,
    
    // Current state
    state: MarketState,
    
    // Price tracking
    last_mid: f64,
    last_trade_time: Timestamp,
}

impl StateEstimationEngine {
    pub fn on_book_update(&mut self, book: &NormalizedL2) {
        let mid = (book.bids[0].price + book.asks[0].price).to_f64().unwrap() / 2.0;
        
        // Update price state
        self.state.mid = book.bids[0].price.midpoint(&book.asks[0].price);
        self.state.bid = book.bids[0].price;
        self.state.ask = book.asks[0].price;
        self.state.spread = book.asks[0].price - book.bids[0].price;
        
        // Update book imbalance
        let bid_depth: f64 = book.bids.iter().take(5).map(|l| l.size.to_f64().unwrap()).sum();
        let ask_depth: f64 = book.asks.iter().take(5).map(|l| l.size.to_f64().unwrap()).sum();
        self.state.book_imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth);
        
        // Volatility update (if price changed)
        if (mid - self.last_mid).abs() > 1e-10 {
            let log_return = (mid / self.last_mid).ln();
            self.vol_estimator.update(log_return, book.timestamp);
            self.regime_detector.update(log_return, book.timestamp);
            self.last_mid = mid;
        }
        
        // Update state
        self.state.sigma_instant = self.vol_estimator.instant_vol();
        self.state.sigma_1m = self.vol_estimator.realized_vol(Duration::from_secs(60));
        self.state.sigma_5m = self.vol_estimator.realized_vol(Duration::from_secs(300));
        self.state.regime = self.regime_detector.current_regime;
        
        self.state.last_update = book.timestamp;
    }
    
    pub fn on_trade(&mut self, trade: &NormalizedTrade) {
        let now = trade.timestamp.as_secs_f64();
        
        // Update flow
        self.flow_estimator.update(Some(&trade), now);
        
        // Update state
        self.state.lambda_buy = self.flow_estimator.lambda_buy;
        self.state.lambda_sell = self.flow_estimator.lambda_sell;
        self.state.flow_imbalance = self.flow_estimator.imbalance();
        self.state.trade_intensity = self.flow_estimator.intensity_ratio();
        
        self.state.last_trade_price = trade.price;
        self.last_trade_time = trade.timestamp;
    }
    
    pub fn on_fill(&mut self, fill: &UserFill) {
        let mid = self.state.mid.to_f64().unwrap();
        
        // Record for AS measurement
        self.as_estimator.record_fill(
            fill.side,
            fill.price.to_f64().unwrap(),
            mid,
            fill.timestamp,
        );
        
        // Update inventory
        let sign = match fill.side {
            Side::Buy => Decimal::ONE,
            Side::Sell => -Decimal::ONE,
        };
        self.state.position += sign * fill.size;
    }
    
    pub fn on_liquidation(&mut self, liq: &LiquidationEvent) {
        self.liq_detector.update(Some(liq), liq.timestamp);
        self.state.liquidation_intensity = self.liq_detector.risk_multiplier();
    }
    
    pub fn on_funding(&mut self, funding: &FundingUpdate) {
        self.state.funding_rate = funding.rate;
        self.state.predicted_funding = funding.predicted;
        self.state.time_to_funding = funding.time_to_next;
    }
    
    pub fn periodic_update(&mut self, now: Timestamp) {
        // Update AS estimates
        self.as_estimator.update(self.state.mid.to_f64().unwrap(), now);
        self.state.alpha = self.as_estimator.alpha();
        self.state.realized_as = self.as_estimator.realized_as();
        
        // Decay flow intensity
        self.flow_estimator.update(None, now.as_secs_f64());
        
        // Update liquidation
        self.liq_detector.update(None, now);
    }
    
    pub fn state(&self) -> &MarketState {
        &self.state
    }
}
```

---

# 5. The Quoting Engine

## 5.1 The Quoting Algorithm

```rust
pub struct QuotingEngine {
    // Configuration
    config: QuotingConfig,
    
    // Calculator
    calculator: LadderCalculator,
}

pub struct QuotingConfig {
    // Number of levels per side
    pub num_levels: usize,
    
    // Size parameters
    pub base_size: Decimal,
    pub min_size: Decimal,
    pub max_size: Decimal,
    
    // Spread parameters
    pub min_spread_bps: f64,
    pub max_spread_bps: f64,
    
    // Risk aversion
    pub gamma_base: f64,
    
    // Holding horizon (seconds)
    pub holding_horizon: f64,
    
    // Position limits
    pub max_position: Decimal,
    
    // Tick/lot sizes
    pub tick_size: Decimal,
    pub lot_size: Decimal,
}

impl QuotingEngine {
    pub fn compute_ladder(&self, state: &MarketState) -> Ladder {
        // 0. Check if we should even quote
        if self.should_pull_quotes(state) {
            return Ladder::empty();
        }
        
        // 1. Compute base parameters
        let params = self.compute_params(state);
        
        // 2. Compute level depths
        let depths = self.compute_depths(state, &params);
        
        // 3. Compute sizes at each level
        let (bid_sizes, ask_sizes) = self.compute_sizes(state, &params, &depths);
        
        // 4. Convert to prices
        let reservation = params.reservation;
        
        let bids: Vec<Level> = depths.iter().zip(bid_sizes.iter())
            .filter(|(_, &s)| s >= self.config.min_size)
            .map(|(&d, &s)| Level {
                price: self.round_down((reservation - d).to_decimal()),
                size: self.round_lot(s),
            })
            .filter(|l| l.price > Decimal::ZERO)
            .collect();
        
        let asks: Vec<Level> = depths.iter().zip(ask_sizes.iter())
            .filter(|(_, &s)| s >= self.config.min_size)
            .map(|(&d, &s)| Level {
                price: self.round_up((reservation + d).to_decimal()),
                size: self.round_lot(s),
            })
            .collect();
        
        Ladder { bids, asks }
    }
    
    fn should_pull_quotes(&self, state: &MarketState) -> bool {
        // Pull quotes if:
        // 1. Liquidation cascade is severe
        if state.liquidation_intensity > 5.0 {
            return true;
        }
        
        // 2. Extreme volatility regime with high intensity
        if state.regime == VolRegime::Extreme && state.trade_intensity > 3.0 {
            return true;
        }
        
        // 3. Near margin limit
        if state.margin_used / state.margin_available > Decimal::new(95, 2) {
            return true;
        }
        
        // 4. Data staleness
        if state.last_update.elapsed() > Duration::from_secs(5) {
            return true;
        }
        
        false
    }
    
    fn compute_params(&self, state: &MarketState) -> QuoteParams {
        let mid = state.mid.to_f64().unwrap();
        let sigma = state.sigma_instant;
        let q = state.position.to_f64().unwrap();
        
        // Dynamic risk aversion
        let gamma = self.dynamic_gamma(state);
        
        // Holding horizon
        let T = self.config.holding_horizon;
        
        // Reservation price (A-S formula)
        let reservation = mid - gamma * sigma.powi(2) * q * T;
        
        // Base half-spread
        // δ* = (1/γ) × ln(1 + γ/k)
        // Simplified for typical γ: δ* ≈ σ√T
        let base_half_spread = sigma * T.sqrt();
        
        // Adverse selection adjustment
        let as_adjustment = state.alpha * state.realized_as * 2.0;
        
        // Regime multiplier
        let regime_mult = match state.regime {
            VolRegime::Low => 0.8,
            VolRegime::Normal => 1.0,
            VolRegime::High => 1.5,
            VolRegime::Extreme => 3.0,
        };
        
        // Liquidation multiplier
        let liq_mult = state.liquidation_intensity.sqrt().min(2.0);
        
        // Final half-spread
        let half_spread = (base_half_spread + as_adjustment) * regime_mult * liq_mult;
        
        // Clamp to bounds
        let half_spread = half_spread
            .max(self.config.min_spread_bps / 10000.0 / 2.0)
            .min(self.config.max_spread_bps / 10000.0 / 2.0);
        
        QuoteParams {
            mid,
            reservation,
            sigma,
            gamma,
            T,
            half_spread,
            regime_mult,
            liq_mult,
        }
    }
    
    fn dynamic_gamma(&self, state: &MarketState) -> f64 {
        let mut gamma = self.config.gamma_base;
        
        // Increase when inventory is large
        let inv_ratio = (state.position / self.config.max_position)
            .abs()
            .to_f64()
            .unwrap();
        gamma *= 1.0 + inv_ratio;
        
        // Increase when margin is tight
        let margin_ratio = (state.margin_used / state.margin_available)
            .to_f64()
            .unwrap();
        gamma *= 1.0 + margin_ratio;
        
        // Increase when liquidation risk is high
        gamma *= state.liquidation_intensity.sqrt();
        
        gamma
    }
    
    fn compute_depths(&self, state: &MarketState, params: &QuoteParams) -> Vec<f64> {
        let K = self.config.num_levels;
        
        // Minimum depth: half the spread
        let delta_min = params.half_spread;
        
        // Maximum depth: ~3σ for holding horizon
        let delta_max = 3.0 * params.sigma * params.T.sqrt();
        
        // Ensure delta_max > delta_min
        let delta_max = delta_max.max(delta_min * 3.0);
        
        // Geometric spacing
        let r = (delta_max / delta_min).powf(1.0 / (K - 1).max(1) as f64);
        
        (0..K).map(|k| delta_min * r.powi(k as i32)).collect()
    }
    
    fn compute_sizes(
        &self, 
        state: &MarketState, 
        params: &QuoteParams,
        depths: &[f64],
    ) -> (Vec<Decimal>, Vec<Decimal>) {
        // Compute weight for each level
        let weights: Vec<f64> = depths.iter().map(|&d| {
            let fill_intensity = self.fill_intensity(d, params);
            let spread_capture = self.spread_capture(d, state, params);
            
            if spread_capture > 0.0 {
                fill_intensity * spread_capture
            } else {
                0.0
            }
        }).collect();
        
        let total_weight: f64 = weights.iter().sum();
        if total_weight < 1e-10 {
            return (vec![Decimal::ZERO; depths.len()], vec![Decimal::ZERO; depths.len()]);
        }
        
        // Total size budget
        let size_budget = self.size_budget(state);
        
        // Allocate sizes
        let base_sizes: Vec<Decimal> = weights.iter()
            .map(|&w| size_budget * Decimal::from_f64(w / total_weight).unwrap())
            .collect();
        
        // Apply inventory skew
        let q_norm = (state.position / self.config.max_position).to_f64().unwrap();
        
        let bid_sizes: Vec<Decimal> = base_sizes.iter()
            .map(|&s| {
                let mult = 1.0 - q_norm.max(0.0);  // Reduce if long
                let mult = mult * self.risk_size_mult(state);
                s * Decimal::from_f64(mult).unwrap()
            })
            .map(|s| s.min(self.config.max_size))
            .collect();
        
        let ask_sizes: Vec<Decimal> = base_sizes.iter()
            .map(|&s| {
                let mult = 1.0 + q_norm.min(0.0);  // Reduce if short
                let mult = mult * self.risk_size_mult(state);
                s * Decimal::from_f64(mult).unwrap()
            })
            .map(|s| s.min(self.config.max_size))
            .collect();
        
        (bid_sizes, ask_sizes)
    }
    
    /// Fill intensity at depth δ
    /// λ(δ) ∝ σ²/δ²
    fn fill_intensity(&self, delta: f64, params: &QuoteParams) -> f64 {
        let sigma = params.sigma;
        sigma.powi(2) / delta.powi(2)
    }
    
    /// Spread capture at depth δ
    /// SC(δ) = δ - AS(δ)
    fn spread_capture(&self, delta: f64, state: &MarketState, params: &QuoteParams) -> f64 {
        let AS_0 = state.alpha * state.realized_as;
        let delta_char = state.as_decay_rate;
        
        delta - AS_0 * (-delta / delta_char).exp()
    }
    
    fn size_budget(&self, state: &MarketState) -> Decimal {
        // Based on margin available and position room
        let margin_size = state.margin_available * Decimal::new(10, 2);  // 10% of available
        let position_room = self.config.max_position - state.position.abs();
        
        self.config.base_size
            .min(margin_size)
            .min(position_room)
            .max(Decimal::ZERO)
    }
    
    fn risk_size_mult(&self, state: &MarketState) -> f64 {
        let regime_mult = match state.regime {
            VolRegime::Low => 1.3,
            VolRegime::Normal => 1.0,
            VolRegime::High => 0.5,
            VolRegime::Extreme => 0.1,
        };
        
        let liq_mult = 1.0 / state.liquidation_intensity.max(1.0);
        
        regime_mult * liq_mult
    }
    
    fn round_down(&self, price: Decimal) -> Decimal {
        (price / self.config.tick_size).floor() * self.config.tick_size
    }
    
    fn round_up(&self, price: Decimal) -> Decimal {
        (price / self.config.tick_size).ceil() * self.config.tick_size
    }
    
    fn round_lot(&self, size: Decimal) -> Decimal {
        (size / self.config.lot_size).floor() * self.config.lot_size
    }
}

struct QuoteParams {
    mid: f64,
    reservation: f64,
    sigma: f64,
    gamma: f64,
    T: f64,
    half_spread: f64,
    regime_mult: f64,
    liq_mult: f64,
}
```

## 5.2 Ladder Data Structure

```rust
#[derive(Clone, Debug, Default)]
pub struct Ladder {
    pub bids: Vec<Level>,  // Sorted best (highest) first
    pub asks: Vec<Level>,  // Sorted best (lowest) first
}

#[derive(Clone, Debug)]
pub struct Level {
    pub price: Decimal,
    pub size: Decimal,
}

impl Ladder {
    pub fn empty() -> Self {
        Self::default()
    }
    
    pub fn best_bid(&self) -> Option<&Level> {
        self.bids.first()
    }
    
    pub fn best_ask(&self) -> Option<&Level> {
        self.asks.first()
    }
    
    pub fn spread(&self) -> Option<Decimal> {
        match (self.best_bid(), self.best_ask()) {
            (Some(b), Some(a)) => Some(a.price - b.price),
            _ => None,
        }
    }
    
    pub fn mid(&self) -> Option<Decimal> {
        match (self.best_bid(), self.best_ask()) {
            (Some(b), Some(a)) => Some((b.price + a.price) / Decimal::TWO),
            _ => None,
        }
    }
    
    pub fn total_bid_size(&self) -> Decimal {
        self.bids.iter().map(|l| l.size).sum()
    }
    
    pub fn total_ask_size(&self) -> Decimal {
        self.asks.iter().map(|l| l.size).sum()
    }
    
    pub fn is_valid(&self) -> bool {
        // No crossed prices
        if let (Some(b), Some(a)) = (self.best_bid(), self.best_ask()) {
            if b.price >= a.price {
                return false;
            }
        }
        
        // Bids descending
        for w in self.bids.windows(2) {
            if w[0].price <= w[1].price {
                return false;
            }
        }
        
        // Asks ascending
        for w in self.asks.windows(2) {
            if w[0].price >= w[1].price {
                return false;
            }
        }
        
        true
    }
}
```

---

# 6. Order Management

## 6.1 Order State Tracking

```rust
pub struct OrderManager {
    // Live orders
    live_orders: HashMap<OrderId, LiveOrder>,
    
    // Order-to-level mapping
    order_levels: HashMap<OrderId, LevelKey>,
    
    // Pending actions
    pending_actions: VecDeque<OrderAction>,
    
    // Exchange client
    exchange: ExchangeClient,
    
    // Configuration
    config: OrderManagerConfig,
}

#[derive(Clone, Debug)]
pub struct LiveOrder {
    pub order_id: OrderId,
    pub client_id: ClientOrderId,
    pub side: Side,
    pub price: Decimal,
    pub size: Decimal,
    pub filled: Decimal,
    pub status: OrderStatus,
    pub created_at: Timestamp,
    pub updated_at: Timestamp,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OrderStatus {
    Pending,    // Sent, not confirmed
    Open,       // Confirmed, resting
    PartiallyFilled,
    Filled,
    Cancelled,
    Rejected,
}

#[derive(Clone, Debug)]
pub enum OrderAction {
    Place { side: Side, price: Decimal, size: Decimal, client_id: ClientOrderId },
    Amend { order_id: OrderId, new_price: Decimal, new_size: Decimal },
    Cancel { order_id: OrderId },
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct LevelKey {
    side: Side,
    level_index: usize,
}
```

## 6.2 Ladder Diffing

```rust
impl OrderManager {
    /// Compute the minimal set of actions to move from current state to target ladder
    pub fn diff(&self, target: &Ladder) -> Vec<OrderAction> {
        let mut actions = vec![];
        
        // Group live orders by side
        let live_bids: Vec<_> = self.live_orders.values()
            .filter(|o| o.side == Side::Buy && o.status.is_active())
            .collect();
        let live_asks: Vec<_> = self.live_orders.values()
            .filter(|o| o.side == Side::Sell && o.status.is_active())
            .collect();
        
        // Diff bids
        actions.extend(self.diff_side(Side::Buy, &live_bids, &target.bids));
        
        // Diff asks
        actions.extend(self.diff_side(Side::Sell, &live_asks, &target.asks));
        
        // Sort: cancels first, then amends, then new
        actions.sort_by_key(|a| match a {
            OrderAction::Cancel { .. } => 0,
            OrderAction::Amend { .. } => 1,
            OrderAction::Place { .. } => 2,
        });
        
        actions
    }
    
    fn diff_side(
        &self,
        side: Side,
        live: &[&LiveOrder],
        target: &[Level],
    ) -> Vec<OrderAction> {
        let mut actions = vec![];
        
        // Sort live orders by price (best first)
        let mut sorted_live: Vec<_> = live.iter().map(|o| *o).collect();
        match side {
            Side::Buy => sorted_live.sort_by(|a, b| b.price.cmp(&a.price)),
            Side::Sell => sorted_live.sort_by(|a, b| a.price.cmp(&b.price)),
        }
        
        // Match target levels to live orders
        let mut used_orders: HashSet<OrderId> = HashSet::new();
        
        for (i, target_level) in target.iter().enumerate() {
            // Find best matching live order
            let best_match = sorted_live.iter()
                .filter(|o| !used_orders.contains(&o.order_id))
                .min_by_key(|o| {
                    // Prefer orders close in price
                    let price_diff = (o.price - target_level.price).abs();
                    OrderedFloat(price_diff.to_f64().unwrap())
                });
            
            match best_match {
                Some(order) if self.should_amend(order, target_level) => {
                    // Amend existing order
                    actions.push(OrderAction::Amend {
                        order_id: order.order_id,
                        new_price: target_level.price,
                        new_size: target_level.size,
                    });
                    used_orders.insert(order.order_id);
                }
                Some(order) => {
                    // Order is close enough, keep it
                    used_orders.insert(order.order_id);
                }
                None => {
                    // Need new order
                    actions.push(OrderAction::Place {
                        side,
                        price: target_level.price,
                        size: target_level.size,
                        client_id: self.generate_client_id(),
                    });
                }
            }
        }
        
        // Cancel unused orders
        for order in sorted_live {
            if !used_orders.contains(&order.order_id) {
                actions.push(OrderAction::Cancel {
                    order_id: order.order_id,
                });
            }
        }
        
        actions
    }
    
    fn should_amend(&self, order: &LiveOrder, target: &Level) -> bool {
        let price_diff = (order.price - target.price).abs();
        let size_diff = (order.size - target.size).abs();
        
        // Amend if price differs by more than 1 tick or size by more than 5%
        let price_threshold = self.config.tick_size;
        let size_threshold = order.size * Decimal::new(5, 2);
        
        price_diff > price_threshold || size_diff > size_threshold
    }
}
```

## 6.3 Order Execution

```rust
impl OrderManager {
    pub async fn execute_actions(&mut self, actions: Vec<OrderAction>) -> Vec<ActionResult> {
        let mut results = vec![];
        
        for action in actions {
            let result = match action {
                OrderAction::Cancel { order_id } => {
                    self.execute_cancel(order_id).await
                }
                OrderAction::Amend { order_id, new_price, new_size } => {
                    self.execute_amend(order_id, new_price, new_size).await
                }
                OrderAction::Place { side, price, size, client_id } => {
                    self.execute_place(side, price, size, client_id).await
                }
            };
            
            results.push(result);
        }
        
        results
    }
    
    async fn execute_cancel(&mut self, order_id: OrderId) -> ActionResult {
        match self.exchange.cancel_order(order_id).await {
            Ok(_) => {
                if let Some(order) = self.live_orders.get_mut(&order_id) {
                    order.status = OrderStatus::Cancelled;
                }
                ActionResult::Success
            }
            Err(e) => {
                tracing::warn!("Cancel failed for {}: {}", order_id, e);
                ActionResult::Failed(e.to_string())
            }
        }
    }
    
    async fn execute_amend(
        &mut self, 
        order_id: OrderId, 
        new_price: Decimal, 
        new_size: Decimal
    ) -> ActionResult {
        // Check if exchange supports amend
        if self.config.supports_amend {
            match self.exchange.amend_order(order_id, new_price, new_size).await {
                Ok(_) => {
                    if let Some(order) = self.live_orders.get_mut(&order_id) {
                        order.price = new_price;
                        order.size = new_size;
                        order.updated_at = Timestamp::now();
                    }
                    ActionResult::Success
                }
                Err(e) => {
                    // Fall back to cancel + new
                    self.execute_cancel(order_id).await;
                    // Place new order...
                    ActionResult::Failed(e.to_string())
                }
            }
        } else {
            // Cancel and replace
            self.execute_cancel(order_id).await;
            let side = self.live_orders.get(&order_id).map(|o| o.side).unwrap_or(Side::Buy);
            self.execute_place(side, new_price, new_size, self.generate_client_id()).await
        }
    }
    
    async fn execute_place(
        &mut self,
        side: Side,
        price: Decimal,
        size: Decimal,
        client_id: ClientOrderId,
    ) -> ActionResult {
        match self.exchange.place_order(side, price, size, client_id).await {
            Ok(order_id) => {
                self.live_orders.insert(order_id, LiveOrder {
                    order_id,
                    client_id,
                    side,
                    price,
                    size,
                    filled: Decimal::ZERO,
                    status: OrderStatus::Open,
                    created_at: Timestamp::now(),
                    updated_at: Timestamp::now(),
                });
                ActionResult::Success
            }
            Err(e) => {
                tracing::warn!("Place failed: {}", e);
                ActionResult::Failed(e.to_string())
            }
        }
    }
    
    pub fn on_fill(&mut self, fill: &ExchangeFill) {
        if let Some(order) = self.live_orders.get_mut(&fill.order_id) {
            order.filled += fill.size;
            order.updated_at = Timestamp::now();
            
            if order.filled >= order.size {
                order.status = OrderStatus::Filled;
            } else {
                order.status = OrderStatus::PartiallyFilled;
            }
        }
    }
}
```

---

# 7. Risk Management

## 7.1 Pre-Trade Risk Checks

```rust
pub struct RiskManager {
    config: RiskConfig,
    state: RiskState,
}

pub struct RiskConfig {
    // Position limits
    pub max_position: Decimal,
    pub max_notional: Decimal,
    
    // Loss limits
    pub max_daily_loss: Decimal,
    pub max_drawdown: f64,
    
    // Rate limits
    pub max_orders_per_second: u32,
    pub max_fills_per_minute: u32,
    
    // Margin
    pub min_margin_buffer: f64,  // Keep this fraction of margin unused
    
    // Regime limits
    pub extreme_regime_size_mult: f64,
}

pub struct RiskState {
    // P&L tracking
    pub daily_pnl: Decimal,
    pub peak_equity: Decimal,
    pub current_equity: Decimal,
    
    // Rate tracking
    pub orders_last_second: u32,
    pub fills_last_minute: u32,
    pub last_order_time: Timestamp,
    
    // Position tracking
    pub total_position_notional: Decimal,
}

impl RiskManager {
    /// Check if a ladder can be quoted
    pub fn check_ladder(&self, ladder: &Ladder, state: &MarketState) -> RiskResult {
        let mut issues = vec![];
        
        // 1. Position limit check
        let max_new_position = ladder.total_bid_size().max(ladder.total_ask_size());
        let potential_position = state.position.abs() + max_new_position;
        
        if potential_position > self.config.max_position {
            issues.push(RiskIssue::PositionLimit {
                current: state.position.abs(),
                potential: potential_position,
                limit: self.config.max_position,
            });
        }
        
        // 2. Notional limit check
        let mid = state.mid;
        let potential_notional = potential_position * mid;
        
        if potential_notional > self.config.max_notional {
            issues.push(RiskIssue::NotionalLimit {
                potential: potential_notional,
                limit: self.config.max_notional,
            });
        }
        
        // 3. Margin check
        let margin_ratio = (state.margin_used / state.margin_available).to_f64().unwrap();
        if margin_ratio > 1.0 - self.config.min_margin_buffer {
            issues.push(RiskIssue::MarginLimit {
                used_ratio: margin_ratio,
                max_ratio: 1.0 - self.config.min_margin_buffer,
            });
        }
        
        // 4. Daily loss check
        if self.state.daily_pnl < -self.config.max_daily_loss {
            issues.push(RiskIssue::DailyLossLimit {
                loss: -self.state.daily_pnl,
                limit: self.config.max_daily_loss,
            });
        }
        
        // 5. Drawdown check
        let drawdown = 1.0 - (self.state.current_equity / self.state.peak_equity).to_f64().unwrap();
        if drawdown > self.config.max_drawdown {
            issues.push(RiskIssue::DrawdownLimit {
                drawdown,
                limit: self.config.max_drawdown,
            });
        }
        
        // 6. Rate limit check
        if self.state.orders_last_second >= self.config.max_orders_per_second {
            issues.push(RiskIssue::RateLimit);
        }
        
        if issues.is_empty() {
            RiskResult::Approved
        } else {
            RiskResult::Rejected(issues)
        }
    }
    
    /// Apply risk-based modifications to a ladder
    pub fn adjust_ladder(&self, ladder: &mut Ladder, state: &MarketState) {
        // Reduce size in extreme regime
        if state.regime == VolRegime::Extreme {
            let mult = Decimal::from_f64(self.config.extreme_regime_size_mult).unwrap();
            for level in &mut ladder.bids {
                level.size *= mult;
            }
            for level in &mut ladder.asks {
                level.size *= mult;
            }
        }
        
        // Reduce size near position limit
        let position_utilization = (state.position.abs() / self.config.max_position)
            .to_f64().unwrap();
        if position_utilization > 0.8 {
            let mult = Decimal::from_f64(1.0 - position_utilization).unwrap();
            
            // Only reduce the side that increases position
            if state.position > Decimal::ZERO {
                for level in &mut ladder.bids {
                    level.size *= mult;
                }
            } else {
                for level in &mut ladder.asks {
                    level.size *= mult;
                }
            }
        }
        
        // Reduce size near margin limit
        let margin_utilization = (state.margin_used / state.margin_available)
            .to_f64().unwrap();
        if margin_utilization > 0.7 {
            let mult = Decimal::from_f64((1.0 - margin_utilization) / 0.3).unwrap();
            for level in &mut ladder.bids {
                level.size *= mult;
            }
            for level in &mut ladder.asks {
                level.size *= mult;
            }
        }
    }
}
```

## 7.2 Position Management

```rust
pub struct PositionManager {
    positions: HashMap<Symbol, Position>,
    config: PositionConfig,
}

pub struct Position {
    pub symbol: Symbol,
    pub size: Decimal,
    pub entry_price: Decimal,
    pub unrealized_pnl: Decimal,
    pub realized_pnl: Decimal,
    pub opened_at: Timestamp,
}

impl PositionManager {
    pub fn on_fill(&mut self, fill: &Fill) {
        let position = self.positions.entry(fill.symbol.clone())
            .or_insert_with(|| Position::new(fill.symbol.clone()));
        
        let sign = match fill.side {
            Side::Buy => Decimal::ONE,
            Side::Sell => -Decimal::ONE,
        };
        
        let old_size = position.size;
        let new_size = old_size + sign * fill.size;
        
        // Update entry price (weighted average)
        if new_size.signum() == old_size.signum() || old_size == Decimal::ZERO {
            // Adding to position
            let old_notional = old_size.abs() * position.entry_price;
            let new_notional = fill.size * fill.price;
            position.entry_price = (old_notional + new_notional) / (old_size.abs() + fill.size);
        } else if new_size.signum() != old_size.signum() {
            // Flipping position
            position.entry_price = fill.price;
        }
        // Else: reducing position, keep entry price
        
        // Realize P&L on size reduction
        if old_size.signum() != Decimal::ZERO && new_size.abs() < old_size.abs() {
            let closed_size = old_size.abs() - new_size.abs();
            let pnl_per_unit = match old_size.signum() {
                s if s > Decimal::ZERO => fill.price - position.entry_price,
                _ => position.entry_price - fill.price,
            };
            position.realized_pnl += closed_size * pnl_per_unit;
        }
        
        position.size = new_size;
    }
    
    pub fn update_mark(&mut self, symbol: &Symbol, mark_price: Decimal) {
        if let Some(position) = self.positions.get_mut(symbol) {
            let pnl_per_unit = if position.size > Decimal::ZERO {
                mark_price - position.entry_price
            } else {
                position.entry_price - mark_price
            };
            position.unrealized_pnl = position.size.abs() * pnl_per_unit;
        }
    }
    
    pub fn total_unrealized_pnl(&self) -> Decimal {
        self.positions.values().map(|p| p.unrealized_pnl).sum()
    }
    
    pub fn total_realized_pnl(&self) -> Decimal {
        self.positions.values().map(|p| p.realized_pnl).sum()
    }
}
```

## 7.3 Kill Switch

```rust
pub struct KillSwitch {
    triggered: AtomicBool,
    trigger_reasons: Mutex<Vec<KillReason>>,
    config: KillSwitchConfig,
}

pub struct KillSwitchConfig {
    pub max_loss_usd: f64,
    pub max_drawdown_pct: f64,
    pub max_position_usd: f64,
    pub stale_data_seconds: u64,
    pub max_fills_per_minute: u32,
}

#[derive(Debug, Clone)]
pub enum KillReason {
    MaxLoss(f64),
    MaxDrawdown(f64),
    MaxPosition(f64),
    StaleData(Duration),
    RateLimit,
    Manual,
}

impl KillSwitch {
    pub fn check(&self, state: &RiskState, market: &MarketState) -> bool {
        if self.triggered.load(Ordering::Relaxed) {
            return true;
        }
        
        let mut reasons = vec![];
        
        // Loss check
        let loss = -state.daily_pnl.to_f64().unwrap();
        if loss > self.config.max_loss_usd {
            reasons.push(KillReason::MaxLoss(loss));
        }
        
        // Drawdown check
        let drawdown = 1.0 - (state.current_equity / state.peak_equity).to_f64().unwrap();
        if drawdown > self.config.max_drawdown_pct {
            reasons.push(KillReason::MaxDrawdown(drawdown));
        }
        
        // Position check
        let position_usd = (market.position * market.mid).abs().to_f64().unwrap();
        if position_usd > self.config.max_position_usd {
            reasons.push(KillReason::MaxPosition(position_usd));
        }
        
        // Data staleness check
        let stale = market.last_update.elapsed();
        if stale > Duration::from_secs(self.config.stale_data_seconds) {
            reasons.push(KillReason::StaleData(stale));
        }
        
        // Rate limit check
        if state.fills_last_minute > self.config.max_fills_per_minute {
            reasons.push(KillReason::RateLimit);
        }
        
        if !reasons.is_empty() {
            self.trigger(reasons);
            return true;
        }
        
        false
    }
    
    pub fn trigger(&self, reasons: Vec<KillReason>) {
        self.triggered.store(true, Ordering::SeqCst);
        let mut guard = self.trigger_reasons.lock().unwrap();
        guard.extend(reasons);
        
        tracing::error!("KILL SWITCH TRIGGERED: {:?}", guard);
    }
    
    pub fn is_triggered(&self) -> bool {
        self.triggered.load(Ordering::Relaxed)
    }
    
    pub fn reset(&self) {
        self.triggered.store(false, Ordering::SeqCst);
        self.trigger_reasons.lock().unwrap().clear();
    }
}
```

---

# 8. Calibration Pipeline

## 8.1 Data Collection

```rust
pub struct CalibrationDataCollector {
    trades: Vec<TradeRecord>,
    books: Vec<BookSnapshot>,
    fills: Vec<FillRecord>,
    funding: Vec<FundingRecord>,
    liquidations: Vec<LiquidationRecord>,
}

pub struct FillRecord {
    pub timestamp: Timestamp,
    pub side: Side,
    pub price: f64,
    pub size: f64,
    pub mid_at_fill: f64,
    pub mid_after_1s: f64,
    pub mid_after_5s: f64,
    pub depth_from_mid: f64,  // In bps
}
```

## 8.2 Volatility Calibration

```rust
pub fn calibrate_volatility(trades: &[TradeRecord]) -> VolatilityParams {
    // Compute log returns
    let returns: Vec<f64> = trades.windows(2)
        .map(|w| (w[1].price / w[0].price).ln())
        .collect();
    
    // Compute annualized volatility
    let var: f64 = returns.iter().map(|r| r.powi(2)).sum::<f64>() / returns.len() as f64;
    let samples_per_year = 365.25 * 24.0 * 3600.0;  // Assuming per-second returns
    let annual_vol = (var * samples_per_year).sqrt();
    
    // Regime thresholds (percentiles of rolling vol)
    let rolling_vols = compute_rolling_volatility(&returns, 300);  // 5-minute windows
    let sorted: Vec<f64> = rolling_vols.iter().copied().sorted().collect();
    
    let p25 = sorted[sorted.len() / 4];
    let p75 = sorted[sorted.len() * 3 / 4];
    let p95 = sorted[sorted.len() * 95 / 100];
    
    VolatilityParams {
        historical_vol: annual_vol,
        low_threshold: p25,
        high_threshold: p75,
        extreme_threshold: p95,
    }
}
```

## 8.3 Hawkes Calibration

```rust
pub fn calibrate_hawkes(trade_times: &[f64]) -> HawkesParams {
    // Maximum likelihood estimation
    // L = Σᵢ log(λ(tᵢ)) - ∫λ(t)dt
    
    let objective = |params: &[f64]| {
        let (mu, alpha, beta) = (params[0], params[1], params[2]);
        
        // Validity check
        if mu <= 0.0 || alpha <= 0.0 || beta <= 0.0 || alpha >= beta {
            return f64::INFINITY;
        }
        
        let mut ll = 0.0;
        
        for (i, &t) in trade_times.iter().enumerate() {
            let mut lambda = mu;
            for j in 0..i {
                lambda += alpha * (-beta * (t - trade_times[j])).exp();
            }
            ll += lambda.ln();
        }
        
        // Integral term
        let T = trade_times.last().unwrap() - trade_times.first().unwrap();
        let mut integral = mu * T;
        for &t in trade_times {
            let remaining = trade_times.last().unwrap() - t;
            integral += (alpha / beta) * (1.0 - (-beta * remaining).exp());
        }
        ll -= integral;
        
        -ll  // Minimize negative log-likelihood
    };
    
    // Optimize
    let initial = [0.1, 0.5, 1.0];
    let result = nelder_mead_optimize(&objective, &initial);
    
    HawkesParams {
        mu: result[0],
        alpha: result[1],
        beta: result[2],
    }
}
```

## 8.4 Adverse Selection Calibration

```rust
pub fn calibrate_adverse_selection(fills: &[FillRecord]) -> AdverseSelectionParams {
    // Group fills by depth bucket
    let mut by_depth: HashMap<usize, Vec<f64>> = HashMap::new();
    
    for fill in fills {
        let bucket = (fill.depth_from_mid * 10.0) as usize;  // 0.1bp buckets
        let as_value = match fill.side {
            Side::Buy => fill.mid_at_fill - fill.mid_after_1s,
            Side::Sell => fill.mid_after_1s - fill.mid_at_fill,
        };
        by_depth.entry(bucket).or_default().push(as_value);
    }
    
    // Compute average AS at each depth
    let points: Vec<(f64, f64)> = by_depth.iter()
        .filter(|(_, vals)| vals.len() >= 10)  // Minimum sample size
        .map(|(&bucket, vals)| {
            let depth = bucket as f64 * 0.0001;  // Convert back to proportion
            let avg_as = vals.iter().sum::<f64>() / vals.len() as f64;
            (depth, avg_as)
        })
        .collect();
    
    // Fit exponential: AS(δ) = AS₀ × exp(-δ/δ_char)
    // Take log: ln(AS) = ln(AS₀) - δ/δ_char
    let log_points: Vec<(f64, f64)> = points.iter()
        .filter(|(_, as_val)| *as_val > 0.0)
        .map(|(d, as_val)| (*d, as_val.ln()))
        .collect();
    
    if log_points.len() < 2 {
        return AdverseSelectionParams::default();
    }
    
    let (intercept, slope) = linear_regression(&log_points);
    
    AdverseSelectionParams {
        as_0: intercept.exp(),
        delta_char: -1.0 / slope,
    }
}
```

## 8.5 Full Calibration Pipeline

```rust
pub async fn run_calibration(
    data_source: &DataSource,
    start: Timestamp,
    end: Timestamp,
) -> CalibrationResult {
    // 1. Load historical data
    let trades = data_source.load_trades(start, end).await?;
    let books = data_source.load_books(start, end).await?;
    let fills = data_source.load_fills(start, end).await?;
    
    // 2. Calibrate each component
    let vol_params = calibrate_volatility(&trades);
    let hawkes_params = calibrate_hawkes(&extract_times(&trades));
    let as_params = calibrate_adverse_selection(&fills);
    
    // 3. Validate (out-of-sample)
    let validation_start = end;
    let validation_end = end + Duration::from_secs(3600);
    let validation_trades = data_source.load_trades(validation_start, validation_end).await?;
    
    let vol_error = validate_volatility(&vol_params, &validation_trades);
    let as_error = validate_adverse_selection(&as_params, &fills);
    
    // 4. Return results
    CalibrationResult {
        vol_params,
        hawkes_params,
        as_params,
        validation: ValidationResult {
            vol_error,
            as_error,
        },
    }
}
```

---

# 9. Backtesting Framework

## 9.1 Event-Driven Backtester

```rust
pub struct Backtester {
    // Strategy
    strategy: Box<dyn Strategy>,
    
    // Simulated state
    account: SimulatedAccount,
    order_book: SimulatedOrderBook,
    
    // Configuration
    config: BacktestConfig,
    
    // Results
    results: BacktestResults,
}

pub struct BacktestConfig {
    pub initial_capital: f64,
    pub maker_fee: f64,
    pub taker_fee: f64,
    pub slippage_bps: f64,
    pub latency_ms: u64,
}

impl Backtester {
    pub fn run(&mut self, events: Vec<HistoricalEvent>) -> BacktestResults {
        for event in events {
            match event {
                HistoricalEvent::Book(book) => {
                    self.on_book(book);
                }
                HistoricalEvent::Trade(trade) => {
                    self.on_trade(trade);
                }
                HistoricalEvent::Funding(funding) => {
                    self.on_funding(funding);
                }
            }
            
            // Check for fills
            self.check_fills();
            
            // Record state
            self.record_snapshot();
        }
        
        self.compute_results()
    }
    
    fn on_book(&mut self, book: HistoricalBook) {
        // Update order book
        self.order_book.update(&book);
        
        // Let strategy react
        let actions = self.strategy.on_book(&book, &self.account);
        self.execute_actions(actions);
    }
    
    fn check_fills(&mut self) {
        // Check if any resting orders would have filled
        for order in self.account.open_orders() {
            if self.would_fill(order) {
                let fill_price = self.simulate_fill_price(order);
                self.account.fill(order.id, fill_price);
                self.strategy.on_fill(order, fill_price);
            }
        }
    }
    
    fn would_fill(&self, order: &Order) -> bool {
        match order.side {
            Side::Buy => self.order_book.best_ask() <= order.price,
            Side::Sell => self.order_book.best_bid() >= order.price,
        }
    }
    
    fn simulate_fill_price(&self, order: &Order) -> f64 {
        // Add slippage
        let slippage = self.config.slippage_bps / 10000.0;
        match order.side {
            Side::Buy => order.price * (1.0 + slippage),
            Side::Sell => order.price * (1.0 - slippage),
        }
    }
}
```

## 9.2 Fill Simulation

This is critical. Naive backtesting assumes you get filled whenever price touches your level. Reality is harder.

```rust
pub struct FillSimulator {
    // Queue model
    cancel_rate: f64,     // Fraction of queue that cancels per second
    
    // Our queue positions
    queue_positions: HashMap<OrderId, QueuePosition>,
}

struct QueuePosition {
    depth_ahead: f64,
    placed_at: Timestamp,
}

impl FillSimulator {
    pub fn place_order(&mut self, order: &Order, book: &OrderBook) {
        // We join at the back of the queue
        let depth_ahead = book.depth_at_price(order.price);
        self.queue_positions.insert(order.id, QueuePosition {
            depth_ahead,
            placed_at: book.timestamp,
        });
    }
    
    pub fn would_fill(
        &mut self, 
        order: &Order, 
        trade: &HistoricalTrade,
        book: &OrderBook,
    ) -> bool {
        // Only consider trades at our level or better
        let at_our_level = match order.side {
            Side::Buy => trade.price <= order.price,
            Side::Sell => trade.price >= order.price,
        };
        
        if !at_our_level {
            return false;
        }
        
        // Update queue position
        if let Some(pos) = self.queue_positions.get_mut(&order.id) {
            // Decay for cancellations
            let elapsed = (book.timestamp - pos.placed_at).as_secs_f64();
            pos.depth_ahead *= (1.0 - self.cancel_rate).powf(elapsed);
            pos.placed_at = book.timestamp;
            
            // Reduce for trades at our level
            if trade.price == order.price {
                pos.depth_ahead -= trade.size;
            }
            
            // Filled if queue cleared
            if pos.depth_ahead <= 0.0 {
                self.queue_positions.remove(&order.id);
                return true;
            }
        }
        
        false
    }
}
```

## 9.3 Performance Metrics

```rust
pub struct BacktestResults {
    // Returns
    pub total_return: f64,
    pub annualized_return: f64,
    pub daily_returns: Vec<f64>,
    
    // Risk
    pub volatility: f64,
    pub max_drawdown: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    
    // Trading
    pub num_trades: u64,
    pub win_rate: f64,
    pub avg_win: f64,
    pub avg_loss: f64,
    pub profit_factor: f64,
    
    // Market making specific
    pub spread_capture: f64,
    pub adverse_selection: f64,
    pub inventory_cost: f64,
    pub funding_pnl: f64,
    pub avg_inventory: f64,
    pub max_inventory: f64,
    pub inventory_turnover: f64,
}

impl BacktestResults {
    pub fn compute(equity_curve: &[f64], trades: &[BacktestTrade]) -> Self {
        let daily_returns = compute_daily_returns(equity_curve);
        let volatility = daily_returns.std_dev() * (252.0_f64).sqrt();
        let total_return = (equity_curve.last().unwrap() / equity_curve.first().unwrap()) - 1.0;
        let days = equity_curve.len() as f64 / (24.0 * 3600.0);  // Assuming per-second data
        let annualized_return = (1.0 + total_return).powf(365.0 / days) - 1.0;
        
        let sharpe = if volatility > 0.0 {
            annualized_return / volatility
        } else {
            0.0
        };
        
        let drawdowns = compute_drawdowns(equity_curve);
        let max_drawdown = drawdowns.iter().cloned().fold(0.0, f64::max);
        
        // Trading metrics
        let winning_trades: Vec<_> = trades.iter().filter(|t| t.pnl > 0.0).collect();
        let losing_trades: Vec<_> = trades.iter().filter(|t| t.pnl < 0.0).collect();
        
        let win_rate = winning_trades.len() as f64 / trades.len() as f64;
        let avg_win = winning_trades.iter().map(|t| t.pnl).sum::<f64>() / winning_trades.len() as f64;
        let avg_loss = losing_trades.iter().map(|t| t.pnl.abs()).sum::<f64>() / losing_trades.len() as f64;
        
        Self {
            total_return,
            annualized_return,
            daily_returns,
            volatility,
            max_drawdown,
            sharpe_ratio: sharpe,
            sortino_ratio: compute_sortino(&daily_returns),
            num_trades: trades.len() as u64,
            win_rate,
            avg_win,
            avg_loss,
            profit_factor: avg_win * win_rate / (avg_loss * (1.0 - win_rate)),
            // ... other metrics
            spread_capture: 0.0,
            adverse_selection: 0.0,
            inventory_cost: 0.0,
            funding_pnl: 0.0,
            avg_inventory: 0.0,
            max_inventory: 0.0,
            inventory_turnover: 0.0,
        }
    }
}
```

---

# 10. Paper Trading

## 10.1 Paper Trading Architecture

```rust
pub struct PaperTrader {
    // Real market data
    data_feed: DataFeed,
    
    // Strategy (same as production)
    strategy: Box<dyn Strategy>,
    
    // Simulated exchange
    simulated_exchange: SimulatedExchange,
    
    // Metrics
    metrics: PaperTradingMetrics,
}

pub struct SimulatedExchange {
    // Simulated account
    account: SimulatedAccount,
    
    // Order management
    orders: HashMap<OrderId, SimulatedOrder>,
    next_order_id: u64,
    
    // Fill simulation
    fill_simulator: FillSimulator,
    
    // Latency simulation
    latency: Duration,
    pending_actions: VecDeque<(Timestamp, OrderAction)>,
}

impl PaperTrader {
    pub async fn run(&mut self) {
        loop {
            tokio::select! {
                Some(event) = self.data_feed.next() => {
                    // Process real market data
                    self.on_market_event(event).await;
                }
                
                _ = tokio::time::sleep(Duration::from_millis(10)) => {
                    // Process pending actions
                    self.process_pending_actions();
                    
                    // Check for simulated fills
                    self.check_fills();
                }
            }
        }
    }
    
    async fn on_market_event(&mut self, event: MarketEvent) {
        // Update simulated exchange with real market data
        self.simulated_exchange.update_market(&event);
        
        // Let strategy react (produces order actions)
        let actions = self.strategy.on_event(&event);
        
        // Submit actions with simulated latency
        let execute_at = Timestamp::now() + self.simulated_exchange.latency;
        for action in actions {
            self.simulated_exchange.pending_actions.push_back((execute_at, action));
        }
    }
    
    fn check_fills(&mut self) {
        let fills = self.simulated_exchange.check_fills();
        for fill in fills {
            self.strategy.on_fill(&fill);
            self.metrics.record_fill(&fill);
        }
    }
}
```

## 10.2 Paper vs Live Comparison

```rust
pub struct PaperLiveComparator {
    paper_state: TradingState,
    live_state: TradingState,
    divergences: Vec<Divergence>,
}

impl PaperLiveComparator {
    pub fn compare(&mut self) -> ComparisonReport {
        let mut report = ComparisonReport::default();
        
        // Position divergence
        let position_diff = self.paper_state.position - self.live_state.position;
        if position_diff.abs() > Decimal::new(1, 2) {  // > 0.01
            self.divergences.push(Divergence::Position(position_diff));
        }
        
        // P&L divergence
        let pnl_diff = self.paper_state.pnl - self.live_state.pnl;
        report.pnl_divergence = pnl_diff;
        
        // Fill rate divergence
        let paper_fill_rate = self.paper_state.fills_per_hour();
        let live_fill_rate = self.live_state.fills_per_hour();
        report.fill_rate_divergence = (paper_fill_rate - live_fill_rate) / live_fill_rate;
        
        // If divergence is large, paper trading model needs calibration
        if report.fill_rate_divergence.abs() > 0.2 {
            tracing::warn!("Paper fill rate diverges from live by {:.1}%", 
                           report.fill_rate_divergence * 100.0);
        }
        
        report
    }
}
```

---

# 11. Production Deployment

## 11.1 Deployment Checklist

```markdown
## Pre-Deployment Checklist

### Code Review
- [ ] All changes reviewed by at least one other person
- [ ] No TODO comments in critical paths
- [ ] Error handling for all external calls
- [ ] Logging at appropriate levels

### Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Backtest on last 7 days shows expected performance
- [ ] Paper trading for at least 24 hours with acceptable metrics

### Configuration
- [ ] API keys are in secure vault, not in code
- [ ] Position limits set correctly
- [ ] Risk limits set correctly
- [ ] Kill switch thresholds configured

### Infrastructure
- [ ] Server has sufficient resources (CPU, memory, network)
- [ ] Server is in same region as exchange
- [ ] Monitoring and alerting configured
- [ ] Log aggregation configured

### Operations
- [ ] Runbook documented
- [ ] On-call schedule set
- [ ] Escalation path defined
- [ ] Recovery procedures documented
```

## 11.2 Configuration Management

```rust
#[derive(Debug, Clone, Deserialize)]
pub struct ProductionConfig {
    // Exchange
    pub exchange: ExchangeConfig,
    
    // Strategy
    pub strategy: StrategyConfig,
    
    // Risk
    pub risk: RiskConfig,
    
    // Monitoring
    pub monitoring: MonitoringConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ExchangeConfig {
    pub api_endpoint: String,
    pub ws_endpoint: String,
    // API keys loaded from environment/vault
    #[serde(skip)]
    pub api_key: String,
    #[serde(skip)]
    pub api_secret: String,
}

impl ProductionConfig {
    pub fn load() -> Result<Self> {
        // Load from file
        let config_path = std::env::var("CONFIG_PATH")
            .unwrap_or_else(|_| "config.toml".to_string());
        let config_str = std::fs::read_to_string(&config_path)?;
        let mut config: ProductionConfig = toml::from_str(&config_str)?;
        
        // Load secrets from environment
        config.exchange.api_key = std::env::var("API_KEY")
            .expect("API_KEY must be set");
        config.exchange.api_secret = std::env::var("API_SECRET")
            .expect("API_SECRET must be set");
        
        // Validate
        config.validate()?;
        
        Ok(config)
    }
    
    pub fn validate(&self) -> Result<()> {
        // Check risk limits are sane
        if self.risk.max_position.is_zero() {
            return Err(anyhow!("max_position cannot be zero"));
        }
        
        if self.risk.max_daily_loss > self.risk.initial_capital * Decimal::new(10, 2) {
            return Err(anyhow!("max_daily_loss too high relative to capital"));
        }
        
        // ... more validation
        
        Ok(())
    }
}
```

## 11.3 Graceful Shutdown

```rust
pub struct ProductionRunner {
    engine: MarketMakingEngine,
    shutdown_signal: broadcast::Receiver<()>,
}

impl ProductionRunner {
    pub async fn run(&mut self) {
        loop {
            tokio::select! {
                _ = self.shutdown_signal.recv() => {
                    tracing::info!("Shutdown signal received");
                    self.graceful_shutdown().await;
                    break;
                }
                
                result = self.engine.tick() => {
                    if let Err(e) = result {
                        tracing::error!("Engine error: {}", e);
                        if e.is_fatal() {
                            self.emergency_shutdown().await;
                            break;
                        }
                    }
                }
            }
        }
    }
    
    async fn graceful_shutdown(&mut self) {
        tracing::info!("Starting graceful shutdown");
        
        // 1. Stop quoting
        self.engine.pull_all_quotes().await;
        
        // 2. Wait for pending orders to be cancelled
        let timeout = Duration::from_secs(10);
        let start = Instant::now();
        
        while self.engine.has_open_orders() && start.elapsed() < timeout {
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        
        // 3. Log final state
        tracing::info!("Final position: {:?}", self.engine.position());
        tracing::info!("Final P&L: {:?}", self.engine.pnl());
        
        // 4. Flush metrics
        self.engine.flush_metrics().await;
        
        tracing::info!("Graceful shutdown complete");
    }
    
    async fn emergency_shutdown(&mut self) {
        tracing::error!("EMERGENCY SHUTDOWN");
        
        // Try to cancel all orders
        self.engine.cancel_all_orders().await;
        
        // Log state
        tracing::error!("Emergency state: {:?}", self.engine.state());
    }
}
```

---

# 12. Monitoring & Alerting

## 12.1 Key Metrics

```rust
pub struct MetricsCollector {
    // Counters
    orders_placed: Counter,
    orders_cancelled: Counter,
    orders_filled: Counter,
    orders_rejected: Counter,
    
    // Gauges
    position: Gauge,
    unrealized_pnl: Gauge,
    spread_quoted: Gauge,
    margin_utilization: Gauge,
    
    // Histograms
    fill_latency: Histogram,
    quote_latency: Histogram,
    tick_processing_time: Histogram,
}

impl MetricsCollector {
    pub fn record_fill(&self, fill: &Fill, latency: Duration) {
        self.orders_filled.inc();
        self.fill_latency.observe(latency.as_millis() as f64);
        self.position.set(fill.resulting_position.to_f64().unwrap());
    }
    
    pub fn record_quote_cycle(&self, duration: Duration) {
        self.quote_latency.observe(duration.as_micros() as f64);
    }
    
    pub fn record_tick(&self, duration: Duration) {
        self.tick_processing_time.observe(duration.as_micros() as f64);
    }
}
```

## 12.2 Alerting Rules

```yaml
# Prometheus alerting rules

groups:
  - name: market_maker
    rules:
      # Position limit approaching
      - alert: PositionLimitWarning
        expr: abs(position) / max_position > 0.8
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "Position at {{ $value | humanizePercentage }} of limit"
      
      # High adverse selection
      - alert: HighAdverseSelection
        expr: adverse_selection_rate > 0.002  # 20bps
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Adverse selection at {{ $value | humanizePercentage }}"
      
      # No fills for extended period
      - alert: NoFills
        expr: increase(orders_filled[10m]) == 0 AND position == 0
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "No fills in 10 minutes while flat"
      
      # High latency
      - alert: HighLatency
        expr: histogram_quantile(0.99, rate(tick_processing_time_bucket[5m])) > 1000  # 1ms
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "P99 tick processing latency is {{ $value }}μs"
      
      # Kill switch triggered
      - alert: KillSwitchTriggered
        expr: kill_switch_active == 1
        labels:
          severity: critical
        annotations:
          summary: "Kill switch has been triggered!"
      
      # Websocket disconnected
      - alert: WebsocketDisconnected
        expr: websocket_connected == 0
        for: 10s
        labels:
          severity: critical
        annotations:
          summary: "Websocket disconnected for {{ $value }}s"
```

## 12.3 Dashboard

```markdown
## Market Maker Dashboard Panels

### Real-Time
- Position (line chart, last hour)
- P&L (line chart, last hour)
- Current spread quoted vs market spread
- Order book with our quotes highlighted
- Recent fills table

### Performance
- P&L by hour (bar chart)
- Cumulative P&L (line chart)
- Sharpe ratio (rolling 24h)
- Win rate (rolling 100 trades)

### Risk
- Margin utilization (gauge)
- Drawdown from peak (gauge)
- Position utilization (gauge)
- Volatility regime (indicator)
- Liquidation cascade risk (gauge)

### System
- Tick processing latency (histogram)
- Orders per second (line chart)
- Websocket message rate (line chart)
- Queue depths (line chart)
- Memory/CPU usage (line charts)

### Diagnostics
- Adverse selection per fill (scatter plot)
- Fill rate by level (bar chart)
- Spread capture vs adverse selection (line chart)
- Regime distribution (pie chart)
```

---

# 13. Performance Analysis

## 13.1 P&L Attribution

```rust
pub struct PnLAttributor {
    trades: Vec<Trade>,
    inventory_snapshots: Vec<InventorySnapshot>,
    funding_payments: Vec<FundingPayment>,
}

impl PnLAttributor {
    pub fn attribute(&self) -> PnLAttribution {
        let mut spread_capture = 0.0;
        let mut adverse_selection = 0.0;
        let mut inventory_carry = 0.0;
        let mut funding = 0.0;
        let mut fees = 0.0;
        
        // Spread capture and adverse selection
        for trade in &self.trades {
            // Spread capture = distance from mid at time of trade
            let mid_at_trade = trade.mid_at_fill;
            let capture = match trade.side {
                Side::Buy => mid_at_trade - trade.price,
                Side::Sell => trade.price - mid_at_trade,
            };
            spread_capture += capture * trade.size.to_f64().unwrap();
            
            // Adverse selection = price move after trade
            let as_move = match trade.side {
                Side::Buy => trade.mid_at_fill - trade.mid_after_1s,
                Side::Sell => trade.mid_after_1s - trade.mid_at_fill,
            };
            adverse_selection += as_move * trade.size.to_f64().unwrap();
            
            fees += trade.fee.to_f64().unwrap();
        }
        
        // Inventory carry = P&L from price moves while holding inventory
        for i in 1..self.inventory_snapshots.len() {
            let prev = &self.inventory_snapshots[i - 1];
            let curr = &self.inventory_snapshots[i];
            
            let price_change = curr.mid_price - prev.mid_price;
            let avg_inventory = (prev.position + curr.position) / 2.0;
            
            inventory_carry += price_change.to_f64().unwrap() * avg_inventory.to_f64().unwrap();
        }
        
        // Funding
        for payment in &self.funding_payments {
            funding += payment.amount.to_f64().unwrap();
        }
        
        PnLAttribution {
            spread_capture,
            adverse_selection,
            inventory_carry,
            funding,
            fees,
            total: spread_capture - adverse_selection + inventory_carry + funding - fees,
        }
    }
}

#[derive(Debug)]
pub struct PnLAttribution {
    pub spread_capture: f64,    // Gross spread earned
    pub adverse_selection: f64, // Cost from informed flow
    pub inventory_carry: f64,   // P&L from inventory + price moves
    pub funding: f64,           // Net funding received/paid
    pub fees: f64,              // Exchange fees
    pub total: f64,
}
```

## 13.2 Fill Analysis

```rust
pub fn analyze_fills(fills: &[FillRecord]) -> FillAnalysis {
    // By level
    let mut by_level: HashMap<usize, LevelStats> = HashMap::new();
    
    for fill in fills {
        let level = (fill.depth_from_mid * 100.0) as usize;  // 1bp buckets
        let stats = by_level.entry(level).or_default();
        
        stats.count += 1;
        stats.total_size += fill.size;
        
        let as_value = match fill.side {
            Side::Buy => fill.mid_at_fill - fill.mid_after_1s,
            Side::Sell => fill.mid_after_1s - fill.mid_at_fill,
        };
        
        let net_pnl = fill.depth_from_mid - as_value;
        stats.total_as += as_value;
        stats.total_pnl += net_pnl * fill.size;
    }
    
    // By time of day
    let mut by_hour: [u64; 24] = [0; 24];
    for fill in fills {
        let hour = fill.timestamp.hour();
        by_hour[hour as usize] += 1;
    }
    
    // By regime
    let mut by_regime: HashMap<VolRegime, Vec<f64>> = HashMap::new();
    for fill in fills {
        by_regime.entry(fill.regime).or_default().push(fill.size);
    }
    
    FillAnalysis {
        by_level,
        by_hour,
        by_regime,
    }
}
```

---

# 14. Continuous Improvement

## 14.1 A/B Testing Framework

```rust
pub struct ABTest {
    name: String,
    variants: Vec<Variant>,
    allocation: Vec<f64>,  // Fraction of time for each variant
    current_variant: usize,
    results: HashMap<usize, VariantResults>,
}

pub struct Variant {
    name: String,
    config_override: StrategyConfig,
}

impl ABTest {
    pub fn select_variant(&mut self) -> &StrategyConfig {
        // Weighted random selection
        let r: f64 = rand::random();
        let mut cumsum = 0.0;
        
        for (i, &weight) in self.allocation.iter().enumerate() {
            cumsum += weight;
            if r < cumsum {
                self.current_variant = i;
                break;
            }
        }
        
        &self.variants[self.current_variant].config_override
    }
    
    pub fn record_result(&mut self, pnl: f64, fills: u64) {
        let results = self.results.entry(self.current_variant).or_default();
        results.total_pnl += pnl;
        results.total_fills += fills;
        results.samples += 1;
    }
    
    pub fn analyze(&self) -> ABTestResult {
        let mut variant_stats: Vec<_> = self.results.iter()
            .map(|(&variant, results)| {
                let avg_pnl = results.total_pnl / results.samples as f64;
                let avg_fills = results.total_fills as f64 / results.samples as f64;
                
                (variant, avg_pnl, avg_fills, results.samples)
            })
            .collect();
        
        variant_stats.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        ABTestResult {
            winner: variant_stats[0].0,
            stats: variant_stats,
        }
    }
}
```

## 14.2 Parameter Optimization

```rust
pub async fn optimize_parameters(
    historical_data: &HistoricalData,
    param_ranges: &ParamRanges,
) -> OptimizationResult {
    let mut best_params = Params::default();
    let mut best_sharpe = f64::NEG_INFINITY;
    
    // Grid search (simple but effective for small parameter spaces)
    for gamma in param_ranges.gamma.iter() {
        for num_levels in param_ranges.num_levels.iter() {
            for spread_mult in param_ranges.spread_mult.iter() {
                let params = Params {
                    gamma: *gamma,
                    num_levels: *num_levels,
                    spread_mult: *spread_mult,
                };
                
                let result = backtest(historical_data, &params).await;
                
                if result.sharpe_ratio > best_sharpe {
                    best_sharpe = result.sharpe_ratio;
                    best_params = params;
                }
            }
        }
    }
    
    OptimizationResult {
        best_params,
        best_sharpe,
    }
}
```

## 14.3 Post-Mortem Template

```markdown
# Post-Mortem: [Date] [Issue Name]

## Summary
Brief description of what happened.

## Timeline
- HH:MM: First indication of problem
- HH:MM: Alert fired
- HH:MM: Engineer paged
- HH:MM: Root cause identified
- HH:MM: Mitigation applied
- HH:MM: Normal operation restored

## Impact
- Duration: X minutes
- P&L impact: $X
- Positions affected: X

## Root Cause
Technical explanation of what went wrong.

## Contributing Factors
- Factor 1
- Factor 2

## Lessons Learned
What we learned from this incident.

## Action Items
- [ ] Action 1 (Owner, Due Date)
- [ ] Action 2 (Owner, Due Date)

## Detection
How was this detected? Could we detect it faster?

## Prevention
What would have prevented this?
```

---

# 15. Common Failure Modes

## 15.1 Technical Failures

| Failure | Symptom | Prevention | Recovery |
|---------|---------|------------|----------|
| Websocket disconnect | No updates | Heartbeat monitoring | Auto-reconnect |
| API rate limit | Order rejections | Rate limiting | Exponential backoff |
| Stale data | Old timestamps | Staleness check | Pull quotes |
| Order stuck | Order never fills/cancels | Timeout | Force cancel |
| Position mismatch | Local != exchange | Periodic reconciliation | Sync from exchange |

## 15.2 Strategy Failures

| Failure | Symptom | Prevention | Recovery |
|---------|---------|------------|----------|
| Adverse selection spike | High AS per fill | AS monitoring | Widen spreads |
| Inventory accumulation | Position grows | Inventory limits | Aggressive skew |
| Regime misclassification | Wrong vol estimate | Multiple estimators | Conservative default |
| Cascade caught | Large loss | Cascade detection | Pull quotes |

## 15.3 Operational Failures

| Failure | Symptom | Prevention | Recovery |
|---------|---------|------------|----------|
| Config error | Wrong behavior | Validation, review | Rollback |
| Deployment issue | System crash | Staging, canary | Rollback |
| Capacity issue | High latency | Load testing | Scale up |
| Dependency failure | Feature unavailable | Circuit breaker | Graceful degradation |

---

# 16. Appendix: Mathematical Reference

## Key Formulas

### Avellaneda-Stoikov

Reservation price:
```
r = S - γσ²qT
```

Optimal spread:
```
δ* = γσ²T + (2/γ)ln(1 + γ/k)
```

### Fill Intensity

For Brownian motion at depth δ:
```
λ(δ) ≈ σ²/δ²
```

### Adverse Selection Decay

```
AS(δ) = AS₀ × exp(-δ/δ_char)
```

### Hawkes Intensity

```
λ(t) = μ + Σᵢ α × exp(-β(t - tᵢ))
```

### Portfolio Risk

```
VaR = z_α × √(w'Σw)
```

### Sharpe Ratio

```
SR = (E[R] - r_f) / σ(R)
```

## Notation

| Symbol | Meaning |
|--------|---------|
| S, M | Mid price |
| σ | Volatility |
| q | Inventory |
| γ | Risk aversion |
| T | Time horizon |
| δ | Depth from mid |
| λ | Intensity (Hawkes or fill) |
| α | Informed flow probability / Hawkes excitation |
| β | Hawkes decay rate |
| AS | Adverse selection |

---

# End of Manual

This manual covers the complete system from first principles to production. Key areas for improvement as you iterate:

1. **Fill simulation accuracy** — Paper trading should match live as closely as possible
2. **Adverse selection model** — The exponential decay is a simplification; calibrate to your data
3. **Regime detection** — Add more sophisticated detection (HMM, online changepoint)
4. **Cross-asset effects** — For multi-asset MM, model correlations
5. **Funding arbitrage** — Systematic extraction across exchanges

Build incrementally. Start with single-asset, single-level, then add complexity.
