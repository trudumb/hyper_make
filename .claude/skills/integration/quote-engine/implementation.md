# Quote Engine - Implementation Details

All Rust code blocks extracted from [SKILL.md](./SKILL.md).

---

## Market Data Input Struct

The input data structure containing all market data feeds needed for quote generation.

```rust
struct MarketData {
    timestamp_ns: u64,

    // Hyperliquid L2
    hl_bids: Vec<(f64, f64)>,  // (price, size)
    hl_asks: Vec<(f64, f64)>,

    // Hyperliquid trades (recent)
    hl_recent_trades: Vec<Trade>,

    // Hyperliquid state
    funding_rate: f64,
    funding_rate_predicted: f64,
    time_to_settlement_s: f64,
    open_interest: f64,

    // Binance
    binance_mid: f64,
    binance_spread_bps: f64,
    binance_recent_trades: Vec<Trade>,

    // Position
    current_inventory: f64,
    inventory_age_s: f64,
}
```

---

## Quote Output Struct

The output data structure with final quotes and diagnostics for logging.

```rust
struct QuoteSet {
    timestamp_ns: u64,

    bid_price: f64,
    bid_size: f64,
    ask_price: f64,
    ask_size: f64,

    // Diagnostics (for logging)
    microprice: f64,
    adjusted_microprice: f64,
    half_spread_bps: f64,
    inventory_skew_bps: f64,
    lead_lag_skew_bps: f64,
    regime: MarketRegime,
    regime_confidence: f64,
    kappa: f64,
    gamma: f64,
    liquidation_probability: f64,
}
```

---

## QuoteEngine Struct and Full Implementation

Core engine struct with all model components, state estimators, tracking, configuration, and measurement. Includes the complete `generate_quotes` pipeline (Steps 1-11), `update_models`, `compute_kappa`, `compute_inventory_skew`, `compute_sizes`, `defensive_quotes`, and `log_predictions`.

```rust
struct QuoteEngine {
    // === Model Components ===
    hmm_filter: OnlineHMMFilter,
    lead_lag: RegimeConditionedLeadLag,
    hawkes_model: HyperliquidFillIntensityModel,
    adverse_classifier: MLPClassifier,
    liquidation_detector: LiquidationDetector,

    // === State Estimators ===
    microprice_estimator: MicropriceEstimator,
    volatility_estimator: BiPowerVolatilityEstimator,

    // === Tracking ===
    adverse_selection_tracker: AdverseSelectionAdjuster,
    recent_trades: RingBuffer<Trade>,

    // === Configuration ===
    config: QuoteEngineConfig,

    // === Measurement ===
    prediction_logger: PredictionLogger,
}

struct QuoteEngineConfig {
    // Base parameters
    base_gamma: f64,
    maker_fee_bps: f64,

    // Limits
    max_inventory: f64,
    max_quote_size: f64,

    // Floors
    min_spread_bps: f64,

    // Skew limits
    max_inventory_skew_bps: f64,
    max_lead_lag_skew_bps: f64,

    // Defensive thresholds
    liquidation_threshold: f64,
    defensive_spread_bps: f64,
}

impl QuoteEngine {
    pub fn generate_quotes(&mut self, data: &MarketData) -> QuoteSet {
        let cycle_id = self.next_cycle_id();

        // === Step 1: Update All Models ===
        self.update_models(data);

        // === Step 2: Check Liquidation Cascade ===
        if self.liquidation_detector.should_pull_quotes() {
            return self.defensive_quotes(data, cycle_id);
        }

        // === Step 3: Get Regime-Blended Parameters ===
        let regime_params = blend_params_by_belief(&self.hmm_filter);
        let gamma = regime_params.gamma;

        // === Step 4: Compute Adjusted Microprice ===
        let local_microprice = self.microprice_estimator.get_microprice();
        let (binance_return, time_since_move) = self.get_recent_binance_move(data);
        let lead_lag_adjustment = self.lead_lag.predict_remaining_move(
            binance_return, time_since_move
        );
        let adjusted_microprice = local_microprice * (1.0 + lead_lag_adjustment);

        // === Step 5: Compute Kappa ===
        let kappa = self.compute_kappa(data, &regime_params);

        // === Step 6: GLFT Optimal Spread ===
        let fee = self.config.maker_fee_bps / 10000.0;
        let glft_half_spread = (1.0 / gamma) * (1.0 + gamma / kappa).ln() + fee;

        // === Step 7: Apply Adjustments ===

        // Spread floor
        let spread_floor = regime_params.spread_floor_bps / 10000.0;
        let half_spread = glft_half_spread.max(spread_floor / 2.0);

        // Uncertainty multiplier (widen when HMM is uncertain)
        let regime_entropy = compute_entropy(&self.hmm_filter.belief);
        let uncertainty_mult = 1.0 + regime_entropy * 0.2;  // Up to 20% wider
        let half_spread = half_spread * uncertainty_mult;

        // Inventory skew
        let inventory_skew = self.compute_inventory_skew(data, &regime_params);

        // Lead-lag skew
        let lead_lag_skew = compute_lead_lag_skew_bps(
            &self.lead_lag,
            binance_return,
            time_since_move,
            self.config.max_lead_lag_skew_bps,
        ) / 10000.0;

        // === Step 8: Compute Final Prices ===
        let bid_depth = half_spread + inventory_skew - lead_lag_skew;
        let ask_depth = half_spread - inventory_skew + lead_lag_skew;

        let bid_price = adjusted_microprice * (1.0 - bid_depth);
        let ask_price = adjusted_microprice * (1.0 + ask_depth);

        // === Step 9: Compute Sizes ===
        let (bid_size, ask_size) = self.compute_sizes(data, &regime_params);

        // === Step 10: Build Output ===
        let quotes = QuoteSet {
            timestamp_ns: data.timestamp_ns,
            bid_price,
            bid_size,
            ask_price,
            ask_size,
            microprice: local_microprice,
            adjusted_microprice,
            half_spread_bps: half_spread * 10000.0,
            inventory_skew_bps: inventory_skew * 10000.0,
            lead_lag_skew_bps: lead_lag_skew * 10000.0,
            regime: self.hmm_filter.most_likely_regime(),
            regime_confidence: self.hmm_filter.belief.iter().cloned()
                .fold(0.0, f64::max),
            kappa,
            gamma,
            liquidation_probability: self.liquidation_detector.liquidation_probability,
        };

        // === Step 11: Log Predictions ===
        self.log_predictions(cycle_id, data, &quotes);

        quotes
    }

    fn update_models(&mut self, data: &MarketData) {
        // HMM filter
        let obs = ObservationVector {
            timestamp_ns: data.timestamp_ns,
            volatility: self.volatility_estimator.get_sigma(),
            trade_intensity: data.hl_recent_trades.len() as f64,
            imbalance: compute_trade_imbalance(&data.hl_recent_trades),
            adverse_selection_rate: self.adverse_selection_tracker.informed_intensity.value(),
        };
        self.hmm_filter.update(&obs);

        // Lead-lag
        self.lead_lag.update_regime(self.volatility_estimator.get_sigma());
        self.lead_lag.on_prices(
            data.binance_mid,
            self.microprice_estimator.get_microprice(),
            data.timestamp_ns,
        );

        // Liquidation detector
        self.liquidation_detector.update(
            data.open_interest,
            data.funding_rate,
            data.timestamp_ns,
        );

        // Microprice
        self.microprice_estimator.update(&data.hl_bids, &data.hl_asks);

        // Volatility
        for trade in &data.hl_recent_trades {
            self.volatility_estimator.on_trade(trade);
        }

        // Update trade buffer
        for trade in &data.hl_recent_trades {
            self.recent_trades.push(trade.clone());
        }

        // Adverse selection tracker
        for trade in &data.hl_recent_trades {
            let features = extract_trade_features(trade, data, &self.recent_trades);
            self.adverse_selection_tracker.on_trade(trade, &features);
        }
    }

    fn compute_kappa(&self, data: &MarketData, regime_params: &RegimeParams) -> f64 {
        // Base kappa from Hawkes intensity
        let market_state = self.extract_market_state(data);
        let intensity_kappa = intensity_to_kappa(
            &self.hawkes_model,
            &market_state,
            self.recent_trades.as_slice(),
            10.0,  // Reference depth
            Side::Bid,
        );

        // Adverse selection adjustment
        let as_adjustment = self.adverse_selection_tracker.get_kappa_adjustment();

        // Regime multiplier
        let regime_mult = regime_params.kappa_multiplier;

        let final_kappa = intensity_kappa * as_adjustment * regime_mult;

        // Floor to prevent division issues
        final_kappa.max(100.0)
    }

    fn compute_inventory_skew(&self, data: &MarketData, regime_params: &RegimeParams) -> f64 {
        let inventory = data.current_inventory;
        let max_inv = self.config.max_inventory * regime_params.max_inventory;

        // Linear skew
        let inventory_fraction = inventory / max_inv;
        let skew_bps = inventory_fraction * self.config.max_inventory_skew_bps;

        // Cap at maximum
        skew_bps.max(-self.config.max_inventory_skew_bps)
            .min(self.config.max_inventory_skew_bps) / 10000.0
    }

    fn compute_sizes(&self, data: &MarketData, regime_params: &RegimeParams) -> (f64, f64) {
        let base_size = self.config.max_quote_size * regime_params.quote_size_multiplier;
        let max_inv = self.config.max_inventory * regime_params.max_inventory;

        // Reduce size on side that would increase inventory
        let inventory = data.current_inventory;

        let bid_size = if inventory > 0.0 {
            // Long, reduce bid size
            let reduction = (inventory / max_inv).min(0.8);
            base_size * (1.0 - reduction)
        } else {
            base_size
        };

        let ask_size = if inventory < 0.0 {
            // Short, reduce ask size
            let reduction = (-inventory / max_inv).min(0.8);
            base_size * (1.0 - reduction)
        } else {
            base_size
        };

        (bid_size.max(0.001), ask_size.max(0.001))
    }

    fn defensive_quotes(&self, data: &MarketData, cycle_id: u64) -> QuoteSet {
        let microprice = self.microprice_estimator.get_microprice();
        let defensive_spread = self.config.defensive_spread_bps / 10000.0;

        QuoteSet {
            timestamp_ns: data.timestamp_ns,
            bid_price: microprice * (1.0 - defensive_spread),
            bid_size: self.config.max_quote_size * 0.1,  // Minimal size
            ask_price: microprice * (1.0 + defensive_spread),
            ask_size: self.config.max_quote_size * 0.1,
            microprice,
            adjusted_microprice: microprice,
            half_spread_bps: self.config.defensive_spread_bps,
            inventory_skew_bps: 0.0,
            lead_lag_skew_bps: 0.0,
            regime: MarketRegime::Cascade,
            regime_confidence: self.liquidation_detector.liquidation_probability,
            kappa: 1000.0,
            gamma: 2.0,
            liquidation_probability: self.liquidation_detector.liquidation_probability,
        }
    }

    fn log_predictions(&mut self, cycle_id: u64, data: &MarketData, quotes: &QuoteSet) {
        let record = PredictionRecord {
            timestamp_ns: data.timestamp_ns,
            quote_cycle_id: cycle_id,
            market_state: self.extract_market_state(data),
            predictions: self.extract_predictions(quotes),
            outcomes: None,
        };

        self.prediction_logger.log(record);
    }
}
```

---

## Configuration Defaults

Default configuration values for the quote engine.

```rust
impl Default for QuoteEngineConfig {
    fn default() -> Self {
        QuoteEngineConfig {
            base_gamma: 0.5,
            maker_fee_bps: 1.5,
            max_inventory: 1.0,  // 1 BTC
            max_quote_size: 0.1,  // 0.1 BTC per side
            min_spread_bps: 5.0,
            max_inventory_skew_bps: 10.0,
            max_lead_lag_skew_bps: 5.0,
            liquidation_threshold: 0.5,
            defensive_spread_bps: 50.0,
        }
    }
}
```

---

## Unit Tests

Core invariant tests: spread positivity, quotes straddle microprice, inventory skew direction, and cascade defensive behavior.

```rust
#[cfg(test)]
mod tests {
    #[test]
    fn spread_is_always_positive() {
        let mut engine = QuoteEngine::default();

        for data in generate_random_market_data(1000) {
            let quotes = engine.generate_quotes(&data);
            assert!(quotes.ask_price > quotes.bid_price);
        }
    }

    #[test]
    fn quotes_straddle_microprice() {
        let mut engine = QuoteEngine::default();

        for data in generate_random_market_data(1000) {
            let quotes = engine.generate_quotes(&data);
            assert!(quotes.bid_price < quotes.adjusted_microprice);
            assert!(quotes.ask_price > quotes.adjusted_microprice);
        }
    }

    #[test]
    fn inventory_skew_direction() {
        let mut engine = QuoteEngine::default();

        // Long inventory should widen bid, tighten ask
        let mut data = default_market_data();
        data.current_inventory = 0.5;  // Long
        let quotes = engine.generate_quotes(&data);

        let mid = quotes.adjusted_microprice;
        let bid_depth = (mid - quotes.bid_price) / mid;
        let ask_depth = (quotes.ask_price - mid) / mid;

        assert!(bid_depth > ask_depth);  // Bid wider when long
    }

    #[test]
    fn cascade_triggers_defensive() {
        let mut engine = QuoteEngine::default();

        // Simulate cascade conditions
        let mut data = default_market_data();
        data.open_interest = 1000.0;

        // Feed OI drop
        for _ in 0..10 {
            data.open_interest *= 0.97;  // 3% drop each tick
            data.funding_rate = 0.001;  // Extreme funding
            let quotes = engine.generate_quotes(&data);
        }

        let quotes = engine.generate_quotes(&data);
        assert!(quotes.half_spread_bps >= 25.0);  // Defensive
    }
}
```

---

## Integration Tests

Historical replay test for end-to-end PnL simulation.

```rust
#[test]
fn replay_historical_day() {
    let data = load_market_data("2024-01-15");
    let mut engine = QuoteEngine::default();

    let mut total_spread_capture = 0.0;
    let mut total_adverse = 0.0;

    for tick in data {
        let quotes = engine.generate_quotes(&tick);

        // Simulate fills based on historical trades
        for trade in &tick.hl_recent_trades {
            if would_fill(&quotes, trade) {
                let spread_capture = compute_spread_capture(&quotes, trade);
                let adverse = compute_adverse_selection(&quotes, trade, &data);

                total_spread_capture += spread_capture;
                total_adverse += adverse;
            }
        }
    }

    let pnl = total_spread_capture + total_adverse;
    println!("Replay PnL: ${:.2}", pnl);
    println!("Spread capture: ${:.2}", total_spread_capture);
    println!("Adverse selection: ${:.2}", total_adverse);
}
```
