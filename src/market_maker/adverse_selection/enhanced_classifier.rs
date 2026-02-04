//! Enhanced Adverse Selection Classifier using Microstructure Features
//!
//! This classifier combines statistically grounded microstructure features
//! with online learning to predict adverse selection before fills occur.
//!
//! # Key Improvements Over PreFillASClassifier
//!
//! 1. **Statistically Meaningful Features**: Z-scores and normalized metrics
//!    instead of raw values
//! 2. **Theory-Driven Weights**: Based on Kyle, Easley-O'Hara, Hasbrouck
//! 3. **Proper Normalization**: All features comparable in scale
//! 4. **Online Learning**: Adapts weights via gradient descent
//! 5. **Feature Diversity**: Features designed to be uncorrelated

use super::microstructure_features::{
    MicrostructureConfig, MicrostructureDiagnostics, MicrostructureExtractor,
    MicrostructureFeatures, TradeObservation,
};

/// Configuration for the enhanced classifier
#[derive(Debug, Clone)]
pub struct EnhancedClassifierConfig {
    /// Microstructure feature extraction config
    pub micro_config: MicrostructureConfig,
    /// Learning rate for online weight updates
    pub learning_rate: f64,
    /// Minimum samples before using learned weights
    pub min_samples_for_learning: usize,
    /// L2 regularization strength (pulls toward theory-driven weights)
    pub regularization: f64,
    /// Enable/disable online learning
    pub enable_learning: bool,
}

impl Default for EnhancedClassifierConfig {
    fn default() -> Self {
        Self {
            micro_config: MicrostructureConfig::default(),
            learning_rate: 0.01,
            min_samples_for_learning: 500,
            regularization: 0.001,
            enable_learning: true,
        }
    }
}

/// Feature weights for toxicity prediction
/// Based on market microstructure theory
const THEORY_WEIGHTS: [f64; 8] = [
    0.15,  // intensity_zscore - Hawkes/information events
    0.25,  // price_impact_zscore - Kyle's lambda (most important)
    0.20,  // run_length_zscore - informed trader clustering
    0.10,  // volume_imbalance - directional pressure
    0.10,  // spread_widening - MM response
    0.05,  // book_velocity_zscore - order flow dynamics
    0.10,  // arrival_speed_zscore - information processing
    0.05,  // size_zscore - large trade indicator
];

/// Enhanced adverse selection classifier
#[derive(Debug, Clone)]
pub struct EnhancedASClassifier {
    config: EnhancedClassifierConfig,
    extractor: MicrostructureExtractor,

    // Online learning state
    learned_weights: [f64; 8],
    weight_gradients: [f64; 8],
    learning_samples: usize,

    // Performance tracking
    prediction_sum: f64,
    outcome_sum: f64,
    correct_predictions: usize,
    total_predictions: usize,

    // Last features for learning
    last_features: Option<(MicrostructureFeatures, bool)>, // (features, is_bid)
}

impl EnhancedASClassifier {
    pub fn new(config: EnhancedClassifierConfig) -> Self {
        Self {
            extractor: MicrostructureExtractor::new(config.micro_config.clone()),
            learned_weights: THEORY_WEIGHTS,
            weight_gradients: [0.0; 8],
            learning_samples: 0,
            prediction_sum: 0.0,
            outcome_sum: 0.0,
            correct_predictions: 0,
            total_predictions: 0,
            last_features: None,
            config,
        }
    }

    pub fn default_config() -> Self {
        Self::new(EnhancedClassifierConfig::default())
    }

    /// Process a new trade
    pub fn on_trade(&mut self, trade: TradeObservation) {
        self.extractor.on_trade(trade);
    }

    /// Process a book update
    pub fn on_book_update(&mut self, bid: f64, ask: f64, bid_size: f64, ask_size: f64, timestamp_ms: u64) {
        self.extractor.on_book_update(bid, ask, bid_size, ask_size, timestamp_ms);
    }

    /// Get current features
    pub fn features(&self) -> MicrostructureFeatures {
        self.extractor.extract()
    }

    /// Predict toxicity for a potential fill
    /// Returns probability in [0, 1] that the fill will be adverse
    pub fn predict_toxicity(&mut self, is_bid: bool) -> f64 {
        let features = self.extractor.extract();

        if features.confidence < 0.5 {
            // Not enough data, return neutral
            return 0.5;
        }

        let weights = self.effective_weights();
        let feature_vec = features.as_vector();

        // For bids, we care about price going DOWN after we buy
        // For asks, we care about price going UP after we sell
        // The features are designed to be high when information is present
        // For bids: same direction as features (informed selling = toxic to our bid)
        // For asks: same direction as features (informed buying = toxic to our ask)

        // Directional adjustment for volume imbalance
        let mut adjusted_features = feature_vec;
        if is_bid {
            // For bids, positive imbalance (more buy pressure) is actually good
            // Negative imbalance (more sell pressure) is bad
            adjusted_features[3] = -adjusted_features[3]; // Invert imbalance for bids
        }
        // For asks, positive imbalance (buy pressure) is toxic, which is default

        // Compute weighted sum
        let raw_score: f64 = weights
            .iter()
            .zip(adjusted_features.iter())
            .map(|(w, f)| w * f)
            .sum();

        // Transform to probability using sigmoid
        let toxicity = 1.0 / (1.0 + (-raw_score).exp());

        // Store for learning
        self.last_features = Some((features, is_bid));

        toxicity.clamp(0.0, 1.0)
    }

    /// Record the outcome of a fill for online learning
    pub fn record_outcome(&mut self, is_bid: bool, was_adverse: bool, adverse_magnitude_bps: Option<f64>) {
        self.total_predictions += 1;
        self.outcome_sum += if was_adverse { 1.0 } else { 0.0 };

        // Get the features that were used for prediction
        let (features, _predicted_is_bid) = match &self.last_features {
            Some((f, b)) if *b == is_bid => (f.clone(), *b),
            _ => return, // No matching prediction
        };

        if features.confidence < 0.5 {
            return; // Don't learn from low-confidence predictions
        }

        let weights = self.effective_weights();
        let mut feature_vec = features.as_vector();

        // Apply same directional adjustment as in predict
        if is_bid {
            feature_vec[3] = -feature_vec[3];
        }

        // Current prediction
        let raw_score: f64 = weights
            .iter()
            .zip(feature_vec.iter())
            .map(|(w, f)| w * f)
            .sum();
        let prediction = 1.0 / (1.0 + (-raw_score).exp());

        // Target (optionally weighted by magnitude)
        let target = if was_adverse {
            let magnitude_weight = adverse_magnitude_bps
                .map(|m| (m / 10.0).clamp(0.5, 2.0))
                .unwrap_or(1.0);
            magnitude_weight.min(1.0)
        } else {
            0.0
        };

        // Track calibration
        self.prediction_sum += prediction;
        if (prediction > 0.5) == was_adverse {
            self.correct_predictions += 1;
        }

        // Online learning via gradient descent
        if self.config.enable_learning {
            self.learning_samples += 1;

            // Gradient of cross-entropy loss
            let error = prediction - target;

            // Update gradients with momentum
            for (i, (grad, feat)) in self.weight_gradients.iter_mut().zip(feature_vec.iter()).enumerate() {
                let new_grad = error * feat;
                *grad = 0.9 * *grad + 0.1 * new_grad; // Momentum

                // Apply gradient with regularization toward theory weights
                let regularization_pull = self.config.regularization * (self.learned_weights[i] - THEORY_WEIGHTS[i]);
                self.learned_weights[i] -= self.config.learning_rate * (*grad + regularization_pull);
            }

            // Keep weights positive and normalized
            self.normalize_weights();
        }

        self.last_features = None;
    }

    /// Normalize weights to sum to 1 and be non-negative
    fn normalize_weights(&mut self) {
        // Clamp to positive
        for w in &mut self.learned_weights {
            *w = w.max(0.01);
        }
        // Normalize to sum to 1
        let sum: f64 = self.learned_weights.iter().sum();
        for w in &mut self.learned_weights {
            *w /= sum;
        }
    }

    /// Get effective weights (learned or theory-driven)
    pub fn effective_weights(&self) -> [f64; 8] {
        if self.config.enable_learning && self.learning_samples >= self.config.min_samples_for_learning {
            self.learned_weights
        } else {
            THEORY_WEIGHTS
        }
    }

    /// Get spread multiplier based on toxicity
    pub fn spread_multiplier(&mut self, is_bid: bool) -> f64 {
        let toxicity = self.predict_toxicity(is_bid);

        // Map toxicity to spread multiplier
        // Low toxicity (< 0.3): tighten slightly (0.9x - 1.0x)
        // Medium toxicity (0.3-0.7): neutral (1.0x)
        // High toxicity (> 0.7): widen (1.0x - 1.5x)

        if toxicity < 0.3 {
            0.9 + toxicity * 0.33 // 0.9 to 1.0
        } else if toxicity > 0.7 {
            1.0 + (toxicity - 0.7) * 1.67 // 1.0 to 1.5
        } else {
            1.0
        }
    }

    /// Get diagnostic information
    pub fn diagnostics(&self) -> EnhancedClassifierDiagnostics {
        let accuracy = if self.total_predictions > 0 {
            self.correct_predictions as f64 / self.total_predictions as f64
        } else {
            0.5
        };

        let base_rate = if self.total_predictions > 0 {
            self.outcome_sum / self.total_predictions as f64
        } else {
            0.5
        };

        let calibration = if self.total_predictions > 0 {
            self.prediction_sum / self.total_predictions as f64
        } else {
            0.5
        };

        EnhancedClassifierDiagnostics {
            learning_samples: self.learning_samples,
            total_predictions: self.total_predictions,
            accuracy,
            base_rate,
            avg_prediction: calibration,
            calibration_gap: (calibration - base_rate).abs(),
            effective_weights: self.effective_weights(),
            is_using_learned: self.learning_samples >= self.config.min_samples_for_learning,
            micro_diagnostics: self.extractor.diagnostics(),
        }
    }

    /// Reset the classifier
    pub fn reset(&mut self) {
        self.extractor.reset();
        self.learned_weights = THEORY_WEIGHTS;
        self.weight_gradients = [0.0; 8];
        self.learning_samples = 0;
        self.prediction_sum = 0.0;
        self.outcome_sum = 0.0;
        self.correct_predictions = 0;
        self.total_predictions = 0;
        self.last_features = None;
    }
}

/// Diagnostic information for the enhanced classifier
#[derive(Debug, Clone)]
pub struct EnhancedClassifierDiagnostics {
    pub learning_samples: usize,
    pub total_predictions: usize,
    pub accuracy: f64,
    pub base_rate: f64,
    pub avg_prediction: f64,
    pub calibration_gap: f64,
    pub effective_weights: [f64; 8],
    pub is_using_learned: bool,
    pub micro_diagnostics: MicrostructureDiagnostics,
}

impl EnhancedClassifierDiagnostics {
    pub fn summary(&self) -> String {
        let weight_str: Vec<String> = self.effective_weights
            .iter()
            .map(|w| format!("{:.2}", w))
            .collect();

        format!(
            "samples={}/{} acc={:.1}% base={:.1}% cal_gap={:.1}% weights=[{}]{}",
            self.learning_samples,
            500, // min_samples
            self.accuracy * 100.0,
            self.base_rate * 100.0,
            self.calibration_gap * 100.0,
            weight_str.join(","),
            if self.is_using_learned { " LEARNED" } else { " default" }
        )
    }

    pub fn feature_summary(&self) -> String {
        let names = MicrostructureFeatures::feature_names();
        let weights = self.effective_weights;

        let mut pairs: Vec<_> = names.iter().zip(weights.iter()).collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

        pairs.iter()
            .take(4)
            .map(|(n, w)| format!("{}={:.0}%", n, *w * 100.0))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_trade(ts: u64, price: f64, size: f64, is_buy: bool) -> TradeObservation {
        TradeObservation {
            timestamp_ms: ts,
            price,
            size,
            is_buy,
        }
    }

    #[test]
    fn test_basic_prediction() {
        let mut classifier = EnhancedASClassifier::default_config();

        // Warmup
        for i in 0..60 {
            classifier.on_trade(make_trade(i * 100, 100.0, 1.0, i % 2 == 0));
            classifier.on_book_update(99.99, 100.01, 10.0, 10.0, i * 100);
        }

        let toxicity = classifier.predict_toxicity(true);
        assert!(toxicity >= 0.0 && toxicity <= 1.0);
    }

    #[test]
    fn test_learning() {
        let mut classifier = EnhancedASClassifier::default_config();

        // Warmup
        for i in 0..60 {
            classifier.on_trade(make_trade(i * 100, 100.0, 1.0, i % 2 == 0));
            classifier.on_book_update(99.99, 100.01, 10.0, 10.0, i * 100);
        }

        // Record some outcomes
        for i in 0..100 {
            let _ = classifier.predict_toxicity(true);
            classifier.record_outcome(true, i % 3 == 0, Some(5.0));
        }

        assert!(classifier.learning_samples > 0);
    }

    #[test]
    fn test_high_toxicity_scenario() {
        let mut classifier = EnhancedASClassifier::default_config();

        // Create high-toxicity scenario: burst of same-side trades
        let base_ts = 0u64;

        // Normal warmup
        for i in 0..50 {
            classifier.on_trade(make_trade(base_ts + i * 100, 100.0, 1.0, i % 2 == 0));
            classifier.on_book_update(99.99, 100.01, 10.0, 10.0, base_ts + i * 100);
        }

        let normal_toxicity = classifier.predict_toxicity(true);

        // Now create information event: burst of sells with falling price
        for i in 0..20 {
            classifier.on_trade(make_trade(
                base_ts + 5000 + i * 10, // Very fast
                100.0 - i as f64 * 0.01, // Falling price
                3.0, // Larger size
                false, // All sells
            ));
        }
        // Widen spread (MM response)
        classifier.on_book_update(99.8, 100.2, 5.0, 15.0, base_ts + 5200);

        let high_toxicity = classifier.predict_toxicity(true);

        assert!(
            high_toxicity > normal_toxicity,
            "Toxicity should increase during information event: {} vs {}",
            high_toxicity,
            normal_toxicity
        );
    }

    #[test]
    fn test_spread_multiplier() {
        let mut classifier = EnhancedASClassifier::default_config();

        // Warmup
        for i in 0..60 {
            classifier.on_trade(make_trade(i * 100, 100.0, 1.0, i % 2 == 0));
            classifier.on_book_update(99.99, 100.01, 10.0, 10.0, i * 100);
        }

        let mult = classifier.spread_multiplier(true);
        assert!(mult >= 0.9 && mult <= 1.5, "Multiplier out of range: {}", mult);
    }
}
