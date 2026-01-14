---
name: Adverse Selection Classifier
description: Predict which incoming trades are informed before they hit you, enabling dynamic spread/kappa adjustment
---

# Adverse Selection Classifier Skill

## Purpose

Predict which incoming trades are informed before they hit you, enabling dynamic spread/kappa adjustment to protect against toxic flow. This is often the highest-value model for PnL improvement because adverse selection is typically the largest drag on market maker profits.

## When to Use

- Building adverse selection prediction from scratch
- Debugging why you're losing money on fills
- Adding liquidation cascade detection
- Tuning when to widen spreads vs stay tight
- Any time adverse selection is your biggest PnL drag

## Prerequisites

- `measurement-infrastructure` with post-fill price tracking
- `signal-audit` to identify predictive features
- Historical trades with ex-post price evolution

---

## Core Concept: Trade Classification

Every trade comes from a latent source:

1. **Noise traders**: No information, mean-reverting flow
2. **Informed traders**: Have directional information, toxic to market makers
3. **Liquidations**: Forced selling/buying, extremely toxic but predictable
4. **Arbitrageurs**: Cross-exchange spread closure, predictable timing

You can't observe the true type, but you can:
- Infer it from ex-post price movement (for training)
- Predict it from observable features (for real-time use)

---

## Labeling Strategy

Since true intent is unobservable, label based on ex-post outcomes:

```rust
#[derive(Clone, Copy, PartialEq, Eq)]
enum TradeLabel {
    Informed,     // Price moved significantly in trade direction
    Noise,        // Price stayed flat or reversed
    Liquidation,  // Part of a liquidation cascade
    Arbitrage,    // Cross-exchange spread closure
}

fn label_trade(
    trade: &Trade,
    price_10s_later: f64,
    market_context: &MarketContext,
) -> TradeLabel {
    // Check for liquidation cascade first (highest priority)
    if is_liquidation_context(market_context) {
        return TradeLabel::Liquidation;
    }
    
    // Check for arbitrage
    if is_arbitrage_context(trade, market_context) {
        return TradeLabel::Arbitrage;
    }
    
    // Informed vs Noise based on ex-post price move
    let price_move_bps = (price_10s_later - trade.price) / trade.price * 10000.0;
    let signed_move = match trade.side {
        Side::Bid => price_move_bps,   // Buy: positive move = informed
        Side::Ask => -price_move_bps,  // Sell: negative move = informed
    };
    
    if signed_move > 5.0 {  // >5 bps in trade direction
        TradeLabel::Informed
    } else {
        TradeLabel::Noise
    }
}

fn is_liquidation_context(ctx: &MarketContext) -> bool {
    // OI dropped significantly + extreme funding
    ctx.oi_change_1m < -0.02  // >2% OI drop
    && (ctx.funding_rate > 0.0005 || ctx.funding_rate < -0.0005)  // Extreme funding
}

fn is_arbitrage_context(trade: &Trade, ctx: &MarketContext) -> bool {
    // Cross-exchange spread was significant and closed within 500ms
    let spread_before = ctx.binance_hl_spread_at_trade.abs();
    let spread_500ms_later = ctx.binance_hl_spread_500ms_later.abs();
    
    spread_before > 5.0  // >5 bps spread
    && spread_500ms_later < 2.0  // Closed to <2 bps
    && (trade.timestamp - ctx.spread_opened_timestamp) < Duration::from_millis(500)
}
```

### Label Distribution (Typical)

| Label | Frequency | Toxicity |
|-------|-----------|----------|
| Noise | 60-70% | Low (slightly positive expectation) |
| Informed | 20-30% | Medium (negative expectation) |
| Arbitrage | 5-10% | Medium-High (predictable timing) |
| Liquidation | 1-5% | Extreme (can be catastrophic) |

---

## Feature Engineering

### Size Features

```rust
struct SizeFeatures {
    size_raw: f64,
    size_zscore: f64,              // (size - μ) / σ, rolling 1h window
    size_quantile: f64,            // Percentile rank in recent trades
    size_vs_depth: f64,            // size / top_of_book_depth
    size_vs_typical: f64,          // size / median_trade_size
}

fn compute_size_features(trade: &Trade, history: &TradeHistory) -> SizeFeatures {
    let mean = history.mean_size_1h();
    let std = history.std_size_1h();
    let median = history.median_size_1h();
    
    SizeFeatures {
        size_raw: trade.size,
        size_zscore: (trade.size - mean) / std.max(1e-6),
        size_quantile: history.quantile_rank(trade.size),
        size_vs_depth: trade.size / trade.top_of_book_depth,
        size_vs_typical: trade.size / median,
    }
}
```

### Timing Features

```rust
struct TimingFeatures {
    time_since_last_trade_ms: f64,
    trades_in_last_1s: u32,
    trades_in_last_10s: u32,
    trade_rate_zscore: f64,        // Is trade rate abnormally high?
    is_burst: bool,                // Part of rapid sequence
}

fn compute_timing_features(trade: &Trade, history: &TradeHistory) -> TimingFeatures {
    let time_since = trade.timestamp - history.last_trade_time();
    let rate_1s = history.trades_in_window(Duration::from_secs(1));
    let rate_10s = history.trades_in_window(Duration::from_secs(10));
    
    let mean_rate = history.mean_trade_rate();
    let std_rate = history.std_trade_rate();
    let current_rate = rate_1s as f64;
    
    TimingFeatures {
        time_since_last_trade_ms: time_since.as_millis() as f64,
        trades_in_last_1s: rate_1s,
        trades_in_last_10s: rate_10s,
        trade_rate_zscore: (current_rate - mean_rate) / std_rate.max(0.1),
        is_burst: rate_1s > 10,  // >10 trades/sec is a burst
    }
}
```

### Aggression Features

```rust
struct AggressionFeatures {
    is_aggressor: bool,            // Market order (crossed the spread)
    crossed_spread_bps: f64,       // How far into book did it go
    depth_consumed_pct: f64,       // What % of top level consumed
    sweeping_multiple_levels: bool,
}

fn compute_aggression_features(trade: &Trade, book: &OrderBook) -> AggressionFeatures {
    let mid = book.mid_price();
    let crossed = match trade.side {
        Side::Bid => (trade.price - mid) / mid * 10000.0,  // Positive = aggressive buy
        Side::Ask => (mid - trade.price) / mid * 10000.0,  // Positive = aggressive sell
    };
    
    let top_level_size = match trade.side {
        Side::Bid => book.best_ask_size(),
        Side::Ask => book.best_bid_size(),
    };
    
    AggressionFeatures {
        is_aggressor: crossed > 0.0,
        crossed_spread_bps: crossed.max(0.0),
        depth_consumed_pct: trade.size / top_level_size,
        sweeping_multiple_levels: crossed > book.spread_bps() / 2.0,
    }
}
```

### Directional Flow Features

```rust
struct FlowFeatures {
    signed_volume_imbalance_1s: f64,   // Net buy - sell volume
    signed_volume_imbalance_10s: f64,
    flow_autocorrelation: f64,          // Are trades clustering on one side?
    trade_aligns_with_flow: bool,       // Does this trade match recent direction?
}

fn compute_flow_features(trade: &Trade, history: &TradeHistory) -> FlowFeatures {
    let imbalance_1s = history.signed_volume_imbalance(Duration::from_secs(1));
    let imbalance_10s = history.signed_volume_imbalance(Duration::from_secs(10));
    
    let trade_sign = match trade.side {
        Side::Bid => 1.0,
        Side::Ask => -1.0,
    };
    
    FlowFeatures {
        signed_volume_imbalance_1s: imbalance_1s,
        signed_volume_imbalance_10s: imbalance_10s,
        flow_autocorrelation: history.flow_autocorr(),
        trade_aligns_with_flow: (trade_sign * imbalance_1s) > 0.0,
    }
}
```

### Hyperliquid-Specific Features

```rust
struct HyperliquidFeatures {
    funding_rate: f64,
    funding_rate_percentile: f64,       // How extreme is funding?
    trade_against_funding: bool,        // Fighting the squeeze?
    oi_change_1m: f64,
    oi_change_5m: f64,
    near_liquidation_level: bool,       // Significant OI near this price?
    time_to_settlement_s: f64,
}

fn compute_hl_features(trade: &Trade, market: &MarketState) -> HyperliquidFeatures {
    let trade_sign = match trade.side {
        Side::Bid => 1.0,
        Side::Ask => -1.0,
    };
    
    // Trade against funding = buying when funding is positive (longs pay shorts)
    // This often indicates informed flow (fighting the consensus)
    let against_funding = (trade_sign * market.funding_rate) > 0.0;
    
    HyperliquidFeatures {
        funding_rate: market.funding_rate,
        funding_rate_percentile: market.funding_percentile,
        trade_against_funding: against_funding,
        oi_change_1m: market.oi_change_1m,
        oi_change_5m: market.oi_change_5m,
        near_liquidation_level: check_liquidation_levels(trade.price, market),
        time_to_settlement_s: market.time_to_settlement_s,
    }
}
```

### Cross-Exchange Features

```rust
struct CrossExchangeFeatures {
    binance_hl_spread_bps: f64,
    binance_price_change_100ms: f64,    // Did Binance just move?
    binance_price_change_500ms: f64,
    binance_leading: bool,               // Is Binance ahead of HL?
}

fn compute_cross_features(trade: &Trade, cross: &CrossExchangeState) -> CrossExchangeFeatures {
    let binance_change_100ms = (cross.binance_mid - cross.binance_mid_100ms_ago) 
        / cross.binance_mid_100ms_ago * 10000.0;
    
    CrossExchangeFeatures {
        binance_hl_spread_bps: cross.binance_hl_spread_bps,
        binance_price_change_100ms: binance_change_100ms,
        binance_price_change_500ms: cross.binance_change_500ms_bps,
        binance_leading: binance_change_100ms.abs() > 2.0,  // Binance moved >2 bps
    }
}
```

### Combined Feature Vector

```rust
struct TradeFeatures {
    size: SizeFeatures,
    timing: TimingFeatures,
    aggression: AggressionFeatures,
    flow: FlowFeatures,
    hyperliquid: HyperliquidFeatures,
    cross_exchange: CrossExchangeFeatures,
}

impl TradeFeatures {
    fn to_vector(&self) -> Vec<f64> {
        vec![
            // Size (5)
            self.size.size_zscore,
            self.size.size_quantile,
            self.size.size_vs_depth,
            self.size.size_vs_typical,
            
            // Timing (4)
            (self.timing.time_since_last_trade_ms / 1000.0).ln(),  // Log scale
            self.timing.trades_in_last_1s as f64,
            self.timing.trade_rate_zscore,
            if self.timing.is_burst { 1.0 } else { 0.0 },
            
            // Aggression (4)
            if self.aggression.is_aggressor { 1.0 } else { 0.0 },
            self.aggression.crossed_spread_bps,
            self.aggression.depth_consumed_pct.min(1.0),
            if self.aggression.sweeping_multiple_levels { 1.0 } else { 0.0 },
            
            // Flow (4)
            self.flow.signed_volume_imbalance_1s / 1000.0,  // Normalize
            self.flow.signed_volume_imbalance_10s / 10000.0,
            self.flow.flow_autocorrelation,
            if self.flow.trade_aligns_with_flow { 1.0 } else { 0.0 },
            
            // Hyperliquid (5)
            self.hyperliquid.funding_rate * 10000.0,  // Scale to bps
            self.hyperliquid.oi_change_1m * 100.0,    // Scale to %
            if self.hyperliquid.trade_against_funding { 1.0 } else { 0.0 },
            if self.hyperliquid.near_liquidation_level { 1.0 } else { 0.0 },
            (self.hyperliquid.time_to_settlement_s / 3600.0).sin(),  // Cyclical
            
            // Cross-exchange (4)
            self.cross_exchange.binance_hl_spread_bps,
            self.cross_exchange.binance_price_change_100ms,
            self.cross_exchange.binance_price_change_500ms,
            if self.cross_exchange.binance_leading { 1.0 } else { 0.0 },
        ]
    }
}
```

---

## Classifier Architecture

### Option 1: Logistic Regression (Interpretable)

```rust
struct LogisticClassifier {
    // P(informed) = σ(w · x + b)
    weights: Vec<f64>,
    bias: f64,
}

impl LogisticClassifier {
    fn predict_informed_prob(&self, features: &TradeFeatures) -> f64 {
        let x = features.to_vector();
        let logit: f64 = self.weights.iter()
            .zip(&x)
            .map(|(w, xi)| w * xi)
            .sum::<f64>() + self.bias;
        
        sigmoid(logit)
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
```

### Option 2: Small MLP (More Expressive)

```rust
struct MLPClassifier {
    layer1_weights: Vec<Vec<f64>>,  // input_dim x hidden1
    layer1_bias: Vec<f64>,
    layer2_weights: Vec<Vec<f64>>,  // hidden1 x hidden2
    layer2_bias: Vec<f64>,
    output_weights: Vec<Vec<f64>>,  // hidden2 x 4 (num classes)
    output_bias: Vec<f64>,
}

impl MLPClassifier {
    fn predict(&self, features: &TradeFeatures) -> ClassProbabilities {
        let x = features.to_vector();
        
        // Layer 1
        let h1 = self.layer1_weights.iter()
            .zip(&self.layer1_bias)
            .map(|(w, b)| relu(dot(w, &x) + b))
            .collect::<Vec<_>>();
        
        // Layer 2
        let h2 = self.layer2_weights.iter()
            .zip(&self.layer2_bias)
            .map(|(w, b)| relu(dot(w, &h1) + b))
            .collect::<Vec<_>>();
        
        // Output
        let logits = self.output_weights.iter()
            .zip(&self.output_bias)
            .map(|(w, b)| dot(w, &h2) + b)
            .collect::<Vec<_>>();
        
        softmax(&logits)
    }
    
    fn predict_informed_prob(&self, features: &TradeFeatures) -> f64 {
        let probs = self.predict(features);
        probs.informed + probs.liquidation  // Both are toxic
    }
}

fn relu(x: f64) -> f64 { x.max(0.0) }

fn softmax(logits: &[f64]) -> ClassProbabilities {
    let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_sum: f64 = logits.iter().map(|l| (l - max_logit).exp()).sum();
    let probs: Vec<f64> = logits.iter()
        .map(|l| (l - max_logit).exp() / exp_sum)
        .collect();
    
    ClassProbabilities {
        noise: probs[0],
        informed: probs[1],
        liquidation: probs[2],
        arbitrage: probs[3],
    }
}
```

---

## Training

```rust
fn train_classifier(
    labeled_data: &[(TradeFeatures, TradeLabel)],
    validation_data: &[(TradeFeatures, TradeLabel)],
) -> MLPClassifier {
    let mut model = MLPClassifier::random_init(INPUT_DIM, 32, 16, 4);
    let mut optimizer = Adam::new(0.001);
    
    let mut best_auc = 0.0;
    let mut best_model = model.clone();
    
    for epoch in 0..100 {
        // Shuffle training data
        let mut shuffled = labeled_data.to_vec();
        shuffled.shuffle(&mut rng);
        
        // Mini-batch training
        for batch in shuffled.chunks(64) {
            let mut grad_sum = Gradients::zeros(&model);
            
            for (features, label) in batch {
                let probs = model.predict(features);
                let grad = backpropagate(&model, features, label);
                grad_sum += grad;
            }
            
            optimizer.step(&mut model, grad_sum / batch.len());
        }
        
        // Validation
        let auc = compute_auc(&model, validation_data);
        if auc > best_auc {
            best_auc = auc;
            best_model = model.clone();
        }
        
        println!("Epoch {}: validation AUC = {:.4}", epoch, auc);
    }
    
    best_model
}

fn compute_auc(
    model: &MLPClassifier,
    data: &[(TradeFeatures, TradeLabel)],
) -> f64 {
    // Binary classification: toxic (Informed + Liquidation) vs not toxic
    let mut predictions = Vec::new();
    let mut labels = Vec::new();
    
    for (features, label) in data {
        let prob = model.predict_informed_prob(features);
        let is_toxic = matches!(label, TradeLabel::Informed | TradeLabel::Liquidation);
        
        predictions.push(prob);
        labels.push(is_toxic);
    }
    
    roc_auc(&predictions, &labels)
}
```

---

## Real-Time Integration

```rust
struct AdverseSelectionAdjuster {
    classifier: MLPClassifier,
    
    // Running estimate of informed flow intensity
    informed_intensity: ExponentialMovingAverage,
    
    // Configuration
    kappa_discount_per_10pct: f64,   // How much to reduce kappa per 10% informed
    spread_premium_per_10pct: f64,   // How much to widen spread per 10% informed
    
    // Thresholds
    high_toxicity_threshold: f64,    // Above this, go defensive
}

impl AdverseSelectionAdjuster {
    fn on_trade(&mut self, trade: &Trade, features: &TradeFeatures) {
        let informed_prob = self.classifier.predict_informed_prob(features);
        
        // Update EMA (α = 0.1 typical, decays over ~10 trades)
        self.informed_intensity.update(informed_prob);
    }
    
    fn get_kappa_adjustment(&self) -> f64 {
        // If recent trades are 30% informed, reduce kappa by 30% * discount_factor
        let informed_pct = self.informed_intensity.value();
        let adjustment = 1.0 - informed_pct * self.kappa_discount_per_10pct * 10.0;
        adjustment.max(0.3)  // Don't reduce by more than 70%
    }
    
    fn get_spread_adjustment_bps(&self) -> f64 {
        let informed_pct = self.informed_intensity.value();
        informed_pct * self.spread_premium_per_10pct * 10.0
    }
    
    fn should_go_defensive(&self) -> bool {
        self.informed_intensity.value() > self.high_toxicity_threshold
    }
}
```

---

## Liquidation Detector (Specialized Subsystem)

Liquidations deserve their own detector because they're:
- Highly predictable from OI + funding
- Extremely toxic
- Need fast response (pull quotes, don't just widen)

```rust
struct LiquidationDetector {
    // State tracking
    oi_history: RingBuffer<(u64, f64)>,
    funding_history: RingBuffer<(u64, f64)>,
    
    // Thresholds
    oi_drop_threshold_1m: f64,    // -2% typical
    oi_drop_threshold_5m: f64,   // -5% typical
    funding_extreme_threshold: f64,  // |0.0005| typical
    
    // Current estimate
    liquidation_probability: f64,
}

impl LiquidationDetector {
    fn update(&mut self, current_oi: f64, current_funding: f64, timestamp: u64) {
        self.oi_history.push((timestamp, current_oi));
        self.funding_history.push((timestamp, current_funding));
        
        // OI change rates
        let oi_1m_ago = self.oi_history.interpolate_at(timestamp - 60_000_000_000);
        let oi_5m_ago = self.oi_history.interpolate_at(timestamp - 300_000_000_000);
        
        let oi_change_1m = (current_oi - oi_1m_ago) / oi_1m_ago;
        let oi_change_5m = (current_oi - oi_5m_ago) / oi_5m_ago;
        
        // Funding extremity
        let funding_percentile = self.compute_funding_percentile(current_funding);
        
        // Liquidation probability model
        self.liquidation_probability = self.compute_prob(
            oi_change_1m,
            oi_change_5m,
            current_funding,
            funding_percentile,
        );
    }
    
    fn compute_prob(
        &self,
        oi_change_1m: f64,
        oi_change_5m: f64,
        funding: f64,
        funding_percentile: f64,
    ) -> f64 {
        let mut prob = 0.0;
        
        // Rapid OI decrease
        if oi_change_1m < self.oi_drop_threshold_1m {
            prob += 0.3;
        }
        if oi_change_5m < self.oi_drop_threshold_5m {
            prob += 0.3;
        }
        
        // Extreme funding
        if funding_percentile > 0.95 || funding_percentile < 0.05 {
            prob += 0.2;
        }
        
        // Funding direction alignment (squeeze)
        // Positive funding + OI drop = longs getting liquidated
        // Negative funding + OI drop = shorts getting liquidated
        if (funding > self.funding_extreme_threshold && oi_change_1m < -0.01)
            || (funding < -self.funding_extreme_threshold && oi_change_1m < -0.01)
        {
            prob += 0.2;
        }
        
        prob.min(0.95)
    }
    
    fn is_cascade_active(&self) -> bool {
        self.liquidation_probability > 0.5
    }
    
    fn should_pull_quotes(&self) -> bool {
        self.liquidation_probability > 0.7
    }
}
```

---

## Validation

### Key Metrics

```rust
struct ClassifierValidation {
    // Overall
    auc_roc: f64,
    precision_at_50pct_recall: f64,
    
    // By class
    confusion_matrix: [[usize; 4]; 4],
    class_precision: [f64; 4],
    class_recall: [f64; 4],
    
    // Calibration
    brier_score: f64,
    information_ratio: f64,
    
    // Economic
    pnl_with_classifier: f64,
    pnl_without_classifier: f64,
    pnl_improvement: f64,
}
```

### Acceptance Criteria

- AUC > 0.65 (informed vs noise classification)
- Information Ratio > 1.0 (adds value vs base rate)
- PnL improvement > 10% vs no classifier

---

## Dependencies

- **Requires**: measurement-infrastructure, signal-audit
- **Enables**: Dynamic kappa adjustment, spread widening on toxic flow

## Common Mistakes

1. **Overfitting to size**: Large trades aren't always informed
2. **Ignoring timing**: Clusters of trades are more toxic than isolated ones
3. **Missing cross-exchange**: Arbitrage flow is predictable from Binance
4. **Static thresholds**: Liquidation conditions vary with market state
5. **Binary thinking**: Use probabilities, not hard classifications

## Next Steps

1. Build labeled dataset from historical fills + price evolution
2. Run signal audit to identify predictive features
3. Train classifier (start with logistic regression for interpretability)
4. Validate AUC > 0.65, IR > 1.0
5. Integrate into quote engine
6. Build liquidation detector as separate fast path
7. Monitor classifier decay (retrain monthly)
