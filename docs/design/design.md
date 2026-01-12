Exactly. The architecture is:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         LAYER 0: DATA                                   │
│                    (WebSocket, trades, L2, fills)                       │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    LAYER 1: PARAMETER ESTIMATION                        │
│              (ParameterEstimator - what you had originally)             │
│                                                                         │
│   Outputs: σ, κ, regime, microprice, AS estimates (point estimates)    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      LAYER 2: LEARNING MODULE                           │
│                    (What you just implemented)                          │
│                                                                         │
│   ModelConfidenceTracker ──► "How accurate are my predictions?"         │
│   ModelEnsemble ──────────► "Which model is working now?"               │
│   DecisionEngine ─────────► "Is there edge right now?"                  │
│   ExecutionOptimizer ─────► "Best ladder given current beliefs"         │
│                                                                         │
│   Outputs: calibrated predictions, uncertainty, model health            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   LAYER 3: STOCHASTIC CONTROLLER                        │
│                         (The new layer)                                 │
│                                                                         │
│   Inputs FROM Layer 2:                                                  │
│     - edge_prediction: (mean, std) from ensemble                        │
│     - model_health: is the learning module trustworthy?                 │
│     - calibration_score: how well do predictions match reality?         │
│                                                                         │
│   Adds:                                                                 │
│     - BeliefState: wraps L2 uncertainty in proper Bayesian form         │
│     - ValueFunction: "what's the value of this state?"                  │
│     - OptimalController: "what action maximizes expected future value?" │
│     - ChangepointDetector: "should I distrust Layer 2 right now?"       │
│     - InformationValue: "should I wait to learn more?"                  │
│                                                                         │
│   Outputs: optimal action considering FUTURE, not just NOW              │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       LAYER 4: EXECUTION                                │
│              (OrderManager, position tracking, risk)                    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

**The interface between Layer 2 and Layer 3:**

```rust
/// What Layer 2 (Learning Module) provides to Layer 3
pub struct LearningModuleOutput {
    // === From ModelEnsemble ===
    /// Edge prediction with uncertainty
    pub edge_prediction: GaussianEstimate,
    
    /// Individual model contributions (for belief updates)
    pub model_predictions: Vec<ModelPrediction>,
    
    /// Disagreement between models (epistemic uncertainty)
    pub model_disagreement: f64,
    
    // === From ModelConfidenceTracker ===
    /// Overall model health
    pub model_health: ModelHealth,
    
    /// Calibration score (are predictions accurate?)
    pub calibration: CalibrationScore,
    
    /// Bias in AS estimation (positive = underestimating = dangerous)
    pub as_bias: f64,
    
    // === From DecisionEngine ===
    /// Myopic decision (what Layer 2 would do)
    pub myopic_decision: QuoteDecision,
    
    /// Confidence in positive edge
    pub p_positive_edge: f64,
    
    // === From ExecutionOptimizer ===
    /// Best ladder given current state (myopic)
    pub myopic_ladder: Ladder,
}

#[derive(Clone)]
pub struct GaussianEstimate {
    pub mean: f64,
    pub std: f64,
}

impl GaussianEstimate {
    pub fn confidence_interval(&self, z: f64) -> (f64, f64) {
        (self.mean - z * self.std, self.mean + z * self.std)
    }
    
    pub fn p_positive(&self) -> f64 {
        normal_cdf(self.mean / self.std.max(1e-9))
    }
}
```

---

**How Layer 3 uses Layer 2:**

```rust
impl StochasticController {
    /// Main entry point - called each quote cycle
    pub fn act(
        &mut self,
        learning_output: &LearningModuleOutput,
        state: &TradingState,
    ) -> Action {
        // 1. Update belief state from learning module output
        self.update_beliefs_from_learning(learning_output);
        
        // 2. Check for changepoint (regime shift)
        //    If detected, we DISTRUST the learning module temporarily
        let trust_learning = self.assess_learning_trust(learning_output);
        
        // 3. Build control state
        let control_state = ControlState {
            wealth: state.wealth,
            position: state.position,
            time: state.session_time_fraction(),
            
            // Belief comes from learning module, wrapped in Bayesian form
            belief: self.belief.clone(),
            
            // Meta-state: how much do we trust Layer 2?
            learning_trust: trust_learning,
            model_health: learning_output.model_health.clone(),
        };
        
        // 4. Compute optimal action (this is the new part)
        let (optimal_action, expected_value) = self.compute_optimal_action(&control_state);
        
        // 5. Compare to myopic action from Layer 2
        //    Sometimes Layer 2 is right, sometimes we override
        let action = self.reconcile_with_myopic(
            optimal_action,
            &learning_output.myopic_decision,
            &control_state,
        );
        
        // 6. Information value check: should we wait?
        if self.should_wait_to_learn(&control_state, &action) {
            return Action::NoQuote;
        }
        
        action
    }
    
    /// Convert learning module output to proper belief state
    fn update_beliefs_from_learning(&mut self, output: &LearningModuleOutput) {
        // The learning module gives us (mean, std) for edge
        // We convert this to a proper posterior
        
        // Edge belief: use ensemble prediction
        self.belief.edge = BetaDistribution::from_gaussian(
            output.edge_prediction.mean,
            output.edge_prediction.std,
        );
        
        // AS belief: incorporate bias estimate
        // If learning module has bias, we adjust our belief accordingly
        self.belief.as_bias_correction = output.as_bias;
        
        // Model weights: convert to Dirichlet
        self.belief.model_weights = DirichletDistribution::from_predictions(
            &output.model_predictions,
        );
        
        // Uncertainty about learning module itself (meta-uncertainty)
        self.belief.epistemic_uncertainty = output.model_disagreement;
    }
    
    /// Should we trust the learning module right now?
    fn assess_learning_trust(&mut self, output: &LearningModuleOutput) -> f64 {
        // Changepoint detection: did regime just change?
        let changepoint_prob = self.changepoint.probability_recent(5);
        
        // If high changepoint probability, distrust learning module
        // (it's trained on old regime data)
        if changepoint_prob > 0.5 {
            return 0.5 * (1.0 - changepoint_prob);
        }
        
        // Check calibration
        let calibration_trust = match output.model_health.overall {
            Health::Good => 1.0,
            Health::Degraded => 0.5,
            Health::Failed => 0.1,
        };
        
        // Check for dangerous AS bias
        let as_trust = if output.as_bias > 1.0 {
            // Underestimating AS - dangerous
            0.5
        } else {
            1.0
        };
        
        calibration_trust * as_trust * (1.0 - changepoint_prob)
    }
}
```

---

**When Layer 3 overrides Layer 2:**

```rust
impl StochasticController {
    /// Decide whether to use Layer 3's optimal action or Layer 2's myopic action
    fn reconcile_with_myopic(
        &self,
        optimal: Action,
        myopic: &QuoteDecision,
        state: &ControlState,
    ) -> Action {
        // Layer 3 overrides Layer 2 in these cases:
        
        // 1. Terminal zone: near end of session, need to unwind inventory
        if state.time > 0.95 && state.position.abs() > 0.5 {
            return self.terminal_inventory_action(state);
        }
        
        // 2. Funding approaching: adjust position for funding capture
        if state.time_to_funding < 0.1 && self.funding_opportunity(state) {
            return self.funding_action(state);
        }
        
        // 3. High model uncertainty: be more conservative than Layer 2 suggests
        if state.belief.epistemic_uncertainty > 0.5 {
            return self.conservative_action(myopic, state);
        }
        
        // 4. Information value: wait to learn
        if matches!(optimal, Action::NoQuote) && !matches!(myopic, QuoteDecision::NoQuote { .. }) {
            // Layer 3 says wait, Layer 2 says quote
            // Check if the value of waiting exceeds myopic action value
            if self.value_of_waiting(state) > self.myopic_value(myopic, state) {
                return Action::NoQuote;
            }
        }
        
        // 5. Position limits: Layer 2 might not account for sequential effects
        if self.position_would_exceed_optimal(myopic, state) {
            return self.position_constrained_action(myopic, state);
        }
        
        // Otherwise, trust Layer 2
        self.myopic_to_action(myopic)
    }
    
    /// Layer 2 doesn't know about session terminal conditions
    /// Layer 3 adds urgency to reduce inventory as session ends
    fn terminal_inventory_action(&self, state: &ControlState) -> Action {
        let time_remaining = 1.0 - state.time;
        let urgency = 1.0 / time_remaining.max(0.01);
        
        // Urgency scales how aggressively we reduce position
        Action::DumpInventory { 
            urgency: urgency.min(10.0) 
        }
    }
    
    /// Layer 3 knows to build position before favorable funding
    fn funding_action(&self, state: &ControlState) -> Action {
        let funding_rate = state.predicted_funding;
        
        // If funding is positive and we're flat, go short to collect
        // If funding is negative and we're flat, go long to collect
        let target = -funding_rate.signum() * self.max_funding_position();
        
        if (state.position - target).abs() > 0.1 {
            Action::BuildInventory { target }
        } else {
            Action::NoQuote  // Already positioned
        }
    }
}
```

---

**The feedback loop (outcomes update both layers):**

```rust
impl TradingSystem {
    pub fn on_fill(&mut self, fill: &Fill) {
        // 1. Layer 1: Update parameter estimates
        self.parameter_estimator.on_fill(fill);
        
        // 2. Layer 2: Update learning module
        //    (model confidence, ensemble weights, calibration)
        let params = self.parameter_estimator.to_market_params();
        self.learning_module.on_fill(fill, &params);
        
        // 3. Layer 3: Update stochastic controller
        //    (belief state, changepoint detection, value function)
        let learning_output = self.learning_module.output();
        self.stochastic_controller.on_fill(fill, &learning_output);
    }
    
    pub fn quote_cycle(&mut self) -> Option<Ladder> {
        // Layer 1: Get current parameter estimates
        let params = self.parameter_estimator.to_market_params();
        
        // Layer 2: Get learning module output
        let learning_output = self.learning_module.evaluate(&params);
        
        // Layer 3: Get optimal action from stochastic controller
        let state = self.current_trading_state();
        let action = self.stochastic_controller.act(&learning_output, &state);
        
        // Convert action to ladder (if quoting)
        match action {
            Action::Quote { ladder } => Some(ladder),
            Action::NoQuote => None,
            Action::DumpInventory { urgency } => {
                Some(self.aggressive_unwind_ladder(urgency))
            }
            Action::BuildInventory { target } => {
                Some(self.inventory_building_ladder(target))
            }
        }
    }
}
```

---

**Summary of responsibilities:**

| Layer | Responsibility | Answers |
|-------|----------------|---------|
| L1: Parameter Estimator | Raw estimation | "What is σ, κ, AS right now?" |
| L2: Learning Module | Calibration & uncertainty | "How confident am I? Which model works?" |
| L3: Stochastic Controller | Sequential optimization | "What should I do considering the future?" |

**Layer 3 adds value over Layer 2 when:**
- Near terminal conditions (session end, funding)
- High uncertainty (wait to learn more)
- Regime changes (distrust learned models)
- Position management (sequential effects)

**Layer 2 is sufficient when:**
- Mid-session, stable regime
- High model confidence
- No upcoming events
- Position near zero

The stochastic controller is the "adult supervision" layer that knows about time and consequences.