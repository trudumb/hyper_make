You've built the estimation and learning loop. Now the actual control problem.

The current system is **myopic**: estimate state → best action NOW. But trading is a **sequential decision problem** under uncertainty. What's optimal NOW depends on what you expect to learn and do LATER.

---

**The real problem you're solving:**

```
max E[∫₀ᵀ u(wealth_t, action_t) dt | beliefs₀]

subject to:
  - wealth dynamics (fills, AS, fees, funding)
  - belief dynamics (learning from outcomes)
  - constraints (margin, position limits, survival)
  - partial observability (latent state)
```

This is a **POMDP** (Partially Observable Markov Decision Process). GLFT is a closed-form solution to a much simpler problem (no learning, no constraints, specific functional forms).

---

**What I'm building next:**

```rust
// ============================================================================
// THE ACTUAL CONTROL PROBLEM
// ============================================================================

/// State space for the control problem
/// This is what we're actually optimizing over
struct ControlState {
    // === Observable State ===
    wealth: f64,
    position: f64,
    margin_used: f64,
    time: f64,  // Time in session [0, T]
    
    // === Belief State (sufficient statistics) ===
    // These summarize everything we've learned
    belief: BeliefState,
    
    // === Exogenous State ===
    // Things that evolve independently of our actions
    vol_regime: DiscreteDistribution<3>,  // P(Low), P(Normal), P(High)
    time_to_funding: f64,
    
    // === Information State ===
    // How much we know about our model quality
    model_confidence: f64,
    samples_since_regime_change: usize,
}

/// Belief state: sufficient statistics for learned parameters
struct BeliefState {
    // Each parameter is a distribution, not a point
    
    // Fill rate: Gamma posterior (conjugate to Poisson)
    lambda_alpha: f64,  // shape
    lambda_beta: f64,   // rate
    
    // Adverse selection: Normal-Gamma posterior
    as_mean: f64,
    as_precision: f64,
    as_n: f64,
    
    // Edge by regime: separate posteriors
    edge_by_regime: [NormalGammaPosterior; 3],
    
    // Model weights: Dirichlet posterior
    model_weights: DirichletPosterior,
}

impl BeliefState {
    /// Bayesian update after observing a fill outcome
    fn update(&mut self, outcome: &FillOutcome) {
        // Fill rate: λ_posterior = Gamma(α + n_fills, β + time)
        self.lambda_alpha += 1.0;
        self.lambda_beta += outcome.time_exposed;
        
        // AS: Normal-Gamma update
        let as_realized = outcome.realized_as_bps;
        let old_mean = self.as_mean;
        self.as_n += 1.0;
        self.as_mean = old_mean + (as_realized - old_mean) / self.as_n;
        self.as_precision += (as_realized - old_mean) * (as_realized - self.as_mean);
        
        // Edge by regime
        let regime = outcome.regime_at_fill;
        self.edge_by_regime[regime].update(outcome.realized_edge_bps);
        
        // Model weights: increment count for best-predicting model
        self.model_weights.observe(outcome.best_model_idx);
    }
    
    /// Sample from belief state (for Thompson sampling / MCTS)
    fn sample(&self) -> SampledParams {
        let mut rng = thread_rng();
        
        SampledParams {
            lambda: Gamma::new(self.lambda_alpha, 1.0 / self.lambda_beta)
                .unwrap().sample(&mut rng),
            as_mean: self.sample_as_mean(&mut rng),
            edge_by_regime: self.edge_by_regime.map(|p| p.sample(&mut rng)),
        }
    }
}

// ============================================================================
// VALUE FUNCTION APPROXIMATION
// ============================================================================

/// The Bellman equation:
/// V(s) = max_a { r(s,a) + γ E[V(s') | s, a] }
/// 
/// We can't solve this exactly (continuous state, partial observability)
/// So we approximate.

struct ValueFunction {
    // Option 1: Fitted value iteration with basis functions
    // V(s) ≈ Σ θᵢ φᵢ(s)
    basis_weights: Vec<f64>,
    
    // Option 2: Neural network (if you want to go there)
    // nn: NeuralNet,
    
    // Option 3: Rollout policy (no explicit V, just simulate)
    rollout_policy: Box<dyn Policy>,
    n_rollouts: usize,
}

/// Basis functions for value approximation
fn basis_functions(state: &ControlState) -> Vec<f64> {
    vec![
        // Wealth terms
        state.wealth,
        state.wealth.powi(2),
        
        // Position terms (inventory penalty is quadratic)
        state.position,
        state.position.powi(2),
        state.position.abs(),
        
        // Time terms (urgency near end of session)
        state.time,
        (1.0 - state.time).max(0.0).sqrt(),  // Urgency increases near T
        
        // Cross terms
        state.position * state.time,  // Position matters more near end
        state.position * state.belief.as_mean,  // Position matters more when AS high
        
        // Belief terms
        state.belief.edge_expected(),
        state.belief.edge_uncertainty(),
        state.model_confidence,
        
        // Regime terms
        state.vol_regime.entropy(),  // Uncertainty about regime
        state.vol_regime.prob(2),    // P(high vol) 
        
        // Funding
        state.time_to_funding,
        state.position * (state.time_to_funding < 0.5) as i32 as f64,  // Position near funding
        
        // Survival term
        (state.margin_used / state.wealth).min(1.0),
    ]
}

impl ValueFunction {
    /// Estimate value at state (fitted value iteration)
    fn value(&self, state: &ControlState) -> f64 {
        let phi = basis_functions(state);
        phi.iter().zip(self.basis_weights.iter())
            .map(|(p, w)| p * w)
            .sum()
    }
    
    /// Estimate value via rollouts (Monte Carlo)
    fn value_rollout(&self, state: &ControlState) -> (f64, f64) {
        let mut returns = Vec::with_capacity(self.n_rollouts);
        
        for _ in 0..self.n_rollouts {
            // Sample parameters from belief
            let params = state.belief.sample();
            
            // Simulate forward with rollout policy
            let mut sim_state = state.clone();
            let mut cumulative_reward = 0.0;
            
            while sim_state.time < 1.0 {  // Until end of session
                let action = self.rollout_policy.act(&sim_state);
                let (reward, next_state) = simulate_step(&sim_state, &action, &params);
                cumulative_reward += reward;
                sim_state = next_state;
            }
            
            // Terminal value
            cumulative_reward += terminal_value(&sim_state);
            returns.push(cumulative_reward);
        }
        
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let var = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() 
                  / returns.len() as f64;
        
        (mean, var.sqrt())
    }
}

// ============================================================================
// OPTIMAL ACTIONS
// ============================================================================

/// The action space
#[derive(Clone)]
enum Action {
    /// Don't quote (preserve optionality, learn from market)
    NoQuote,
    
    /// Quote with specific ladder
    Quote { ladder: Ladder },
    
    /// Aggressive inventory reduction
    DumpInventory { urgency: f64 },
    
    /// Acquire inventory (for funding capture)
    BuildInventory { target: f64 },
}

/// Solve for optimal action given current state
struct OptimalController {
    value_fn: ValueFunction,
    
    // Constraints
    max_position: f64,
    margin_requirement: f64,
    survival_prob: f64,  // P(survival) must exceed this
}

impl OptimalController {
    /// Find optimal action by maximizing Q(s, a)
    fn optimal_action(&self, state: &ControlState) -> (Action, f64) {
        // Enumerate candidate actions
        let candidates = self.enumerate_actions(state);
        
        let mut best_action = Action::NoQuote;
        let mut best_q = f64::NEG_INFINITY;
        
        for action in candidates {
            let q = self.q_value(state, &action);
            if q > best_q {
                best_q = q;
                best_action = action;
            }
        }
        
        (best_action, best_q)
    }
    
    /// Q(s, a) = r(s, a) + γ E[V(s') | s, a]
    fn q_value(&self, state: &ControlState, action: &Action) -> f64 {
        // Immediate reward (expected P&L from this action)
        let immediate = self.expected_immediate_reward(state, action);
        
        // Check constraints
        if !self.satisfies_constraints(state, action) {
            return f64::NEG_INFINITY;
        }
        
        // Expected future value
        // This requires integrating over:
        // 1. Fill outcomes (Poisson process)
        // 2. Price moves (diffusion + jumps)
        // 3. Regime transitions
        // 4. Belief updates
        
        let future = self.expected_future_value(state, action);
        
        immediate + future
    }
    
    fn expected_future_value(&self, state: &ControlState, action: &Action) -> f64 {
        // Monte Carlo integration over next-state distribution
        let n_samples = 100;
        let mut total = 0.0;
        
        for _ in 0..n_samples {
            // Sample outcome of this action
            let (next_state, _) = self.sample_transition(state, action);
            
            // Value of next state
            total += self.value_fn.value(&next_state);
        }
        
        total / n_samples as f64
    }
    
    fn sample_transition(
        &self, 
        state: &ControlState, 
        action: &Action
    ) -> (ControlState, f64) {
        let mut rng = thread_rng();
        let mut next = state.clone();
        let mut reward = 0.0;
        
        // Time step
        let dt = 0.001;  // ~1 second in normalized time
        next.time += dt;
        
        match action {
            Action::NoQuote => {
                // No fills, just observe market
                // Price moves, regime may change
                self.simulate_market_evolution(&mut next, dt, &mut rng);
            }
            
            Action::Quote { ladder } => {
                // Sample fills from Poisson process
                let params = state.belief.sample();
                
                for level in &ladder.bids {
                    let fill_prob = 1.0 - (-params.lambda * dt * 
                        self.fill_rate_modifier(level.depth_bps)).exp();
                    
                    if rng.gen::<f64>() < fill_prob {
                        // Got filled
                        let fill_size = level.size;
                        next.position += fill_size;
                        
                        // Sample realized AS
                        let as_realized = params.as_mean + 
                            rng.sample::<f64, _>(StandardNormal) * 
                            params.as_std;
                        
                        // P&L = spread - AS - fees
                        let pnl = (level.depth_bps - as_realized - 1.5) 
                                  * fill_size * state.wealth * 0.0001;
                        reward += pnl;
                        next.wealth += pnl;
                        
                        // Update beliefs with this outcome
                        next.belief.update(&FillOutcome {
                            realized_as_bps: as_realized,
                            // ...
                        });
                    }
                }
                
                // Similar for asks
                // ...
                
                // Market evolution
                self.simulate_market_evolution(&mut next, dt, &mut rng);
            }
            
            // Other actions...
            _ => {}
        }
        
        (next, reward)
    }
}

// ============================================================================
// INFORMATION VALUE (when to wait vs act)
// ============================================================================

/// Sometimes the optimal action is to WAIT and LEARN
/// This is the "value of information" problem

struct InformationValue {
    // Current uncertainty about key parameters
    current_uncertainty: f64,
    
    // Expected uncertainty after N more observations
    projected_uncertainty: fn(n: usize) -> f64,
    
    // Cost of waiting (opportunity cost, funding, etc.)
    wait_cost_per_period: f64,
}

impl InformationValue {
    /// Should we wait to learn more before acting?
    fn should_wait(&self, state: &ControlState, best_action_now: &Action) -> bool {
        // Value of acting now
        let v_act_now = self.value_of_acting_now(state, best_action_now);
        
        // Value of waiting one period, then acting optimally
        // V_wait = -c_wait + E[V*(s') | wait]
        // where s' has updated beliefs from observed market
        let v_wait = self.value_of_waiting(state);
        
        v_wait > v_act_now
    }
    
    fn value_of_waiting(&self, state: &ControlState) -> f64 {
        let wait_cost = self.wait_cost_per_period;
        
        // After waiting, we'll have more observations
        // Simulate what we might learn
        let n_samples = 50;
        let mut total_future_value = 0.0;
        
        for _ in 0..n_samples {
            // Sample what we might observe
            let observed_edge = self.sample_observation(state);
            
            // Update beliefs
            let mut future_belief = state.belief.clone();
            future_belief.update_from_observation(observed_edge);
            
            // Optimal action with updated beliefs
            let future_state = ControlState {
                belief: future_belief,
                time: state.time + 0.001,
                ..state.clone()
            };
            
            let (_, future_value) = self.optimal_action_value(&future_state);
            total_future_value += future_value;
        }
        
        -wait_cost + total_future_value / n_samples as f64
    }
}

// ============================================================================
// REGIME CHANGE DETECTION (proper changepoint detection)
// ============================================================================

/// Not just "what regime are we in" but "did the regime just change?"
/// This matters because post-change, your learned parameters are stale.

struct ChangepointDetector {
    // Bayesian Online Changepoint Detection (Adams & MacKay 2007)
    
    // Run length distribution: P(r_t | data)
    // r_t = number of observations since last changepoint
    run_length_probs: Vec<f64>,
    
    // Sufficient statistics for each run length hypothesis
    run_statistics: Vec<RunStatistics>,
    
    // Hazard rate: P(changepoint at any time)
    hazard: f64,
}

struct RunStatistics {
    // For each hypothesis "changepoint was k steps ago"
    // Track sufficient stats for parameter estimation
    n: usize,
    sum_x: f64,
    sum_xx: f64,
}

impl ChangepointDetector {
    fn update(&mut self, observation: f64) {
        let n_runs = self.run_length_probs.len();
        
        // Growth probabilities: P(r_t = r_{t-1} + 1)
        let mut growth_probs = Vec::with_capacity(n_runs);
        
        // Changepoint probability: P(r_t = 0)
        let mut cp_prob = 0.0;
        
        for (r, (prob, stats)) in self.run_length_probs.iter()
            .zip(self.run_statistics.iter())
            .enumerate() 
        {
            // Predictive probability of observation given this run length
            let pred_prob = self.predictive_prob(observation, stats);
            
            // Growth: no changepoint
            let growth = prob * (1.0 - self.hazard) * pred_prob;
            growth_probs.push(growth);
            
            // Changepoint: reset
            cp_prob += prob * self.hazard * pred_prob;
        }
        
        // Normalize
        let total: f64 = growth_probs.iter().sum::<f64>() + cp_prob;
        growth_probs.iter_mut().for_each(|p| *p /= total);
        cp_prob /= total;
        
        // Prepend changepoint probability (new run length = 0)
        self.run_length_probs = std::iter::once(cp_prob)
            .chain(growth_probs.into_iter())
            .collect();
        
        // Update sufficient statistics
        self.update_statistics(observation);
        
        // Trim low-probability run lengths
        self.prune();
    }
    
    /// Probability that a changepoint occurred in last k observations
    fn changepoint_probability(&self, k: usize) -> f64 {
        self.run_length_probs.iter().take(k).sum()
    }
    
    /// Most likely current run length
    fn map_run_length(&self) -> usize {
        self.run_length_probs.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
    
    /// Should we reset our learned parameters?
    fn should_reset_beliefs(&self) -> bool {
        // If high probability that changepoint in last few observations
        self.changepoint_probability(5) > 0.8
    }
}

// ============================================================================
// PUTTING IT TOGETHER
// ============================================================================

struct StochasticController {
    // Value function (fitted or rollout)
    value_fn: ValueFunction,
    
    // Optimal action solver
    controller: OptimalController,
    
    // Information value calculator
    info_value: InformationValue,
    
    // Changepoint detection
    changepoint: ChangepointDetector,
    
    // Current belief state
    belief: BeliefState,
    
    // Fitted value iteration update frequency
    refit_interval: Duration,
    last_refit: Instant,
}

impl StochasticController {
    fn act(&mut self, state: &ControlState) -> Action {
        // 1. Check for regime change
        if self.changepoint.should_reset_beliefs() {
            info!("Changepoint detected, resetting beliefs");
            self.belief = BeliefState::default_prior();
            self.changepoint = ChangepointDetector::new(self.changepoint.hazard);
        }
        
        // 2. Get optimal action under current beliefs
        let (best_action, best_value) = self.controller.optimal_action(state);
        
        // 3. Check if we should wait to learn more
        if self.info_value.should_wait(state, &best_action) {
            debug!(
                best_value = %format!("{:.2}", best_value),
                "Waiting to learn more"
            );
            return Action::NoQuote;
        }
        
        // 4. Periodically refit value function
        if self.last_refit.elapsed() > self.refit_interval {
            self.refit_value_function();
            self.last_refit = Instant::now();
        }
        
        best_action
    }
    
    fn on_outcome(&mut self, outcome: &FillOutcome) {
        // Update beliefs
        self.belief.update(outcome);
        
        // Update changepoint detector
        self.changepoint.update(outcome.realized_edge_bps);
        
        // Accumulate data for value function refitting
        self.accumulate_transition(outcome);
    }
    
    fn refit_value_function(&mut self) {
        // Fitted value iteration:
        // θ* = argmin Σ (V_θ(s) - [r + γV_θ(s')])²
        // 
        // Using accumulated transitions
        
        // This is batch least squares on basis function weights
        // ...
    }
}
```

---

**The key insights:**

1. **Beliefs are state** - your uncertainty about parameters is part of the optimization
2. **Actions affect future beliefs** - getting filled teaches you about AS
3. **Terminal conditions matter** - behavior should change near session end, funding times
4. **Information has value** - sometimes waiting to learn is optimal
5. **Changepoints invalidate learning** - need to detect and reset

**What this gives you that the previous system didn't:**

| Previous System | This System |
|-----------------|-------------|
| Myopic decisions | Multi-period optimization |
| Point parameter estimates | Full belief distributions |
| React to regime | Detect changepoints |
| Fixed decision criterion | Information-value-aware |
| Heuristic inventory penalty | Derived from HJB terminal condition |

---

**The practical implementation path:**

1. **Week 1**: BeliefState with proper Bayesian updates, not just point estimates
2. **Week 2**: Rollout-based value estimation (simpler than fitted VI)  
3. **Week 3**: Changepoint detection integration
4. **Week 4**: Information value for "should I wait?"

The value function fitting is the hardest part. Start with rollouts (Monte Carlo tree search style) before trying fitted value iteration.