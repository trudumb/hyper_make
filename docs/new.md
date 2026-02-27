Then 10-min paper trading:                                                                                                                                                                                    │
│ - Verify τ_ac adapts (log output)                                                                                                                                                                             │
│ - Verify max_log_odds varies by regime                                                                                                                                                                        │
│ - Verify evidence_weight follows Bayesian curve                                                                                                                                                               │
│ - Verify margin splits scale with inventory × gamma                                                                                                                                                           │
│ - Verify no whipsaw >40 ppt in <5s                                                                                                                                                                            │
│ - Verify equity Sharpe improves from -225  


Deferred                                                                                                                                                                                                      │
│                                                                                                                                                                                                               │
│ ┌───────────────────────────────────────────────┬────────────────────────────────────────────────────────────────────────────────────────────────────┐                                                        │
│ │                     Item                      │                                                Why                                                 │                                                        │
│ ├───────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────┤                                                        │
│ │ Online SignalVarianceTracker for λ_i          │ Needs paper data to validate convergence. Current inverse-variance priors are close to principled. │                                                        │
│ ├───────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────┤                                                        │
│ │ Edge uncertainty → gamma wiring               │ edge_uncertainty: 0.5 placeholder exists but needs EdgeTracker impl. Separate PR.                  │                                                        │
│ ├───────────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────┤                                                        │
│ │ Full generative model for directional beliefs │ Replace accumulator with explicit signal_i ~ N(α_i × drift, σ_i²) likelihood. Major refactor.      │                                                        │
│ └───────────────────────────────────────────────┴────────────────────────────────────────────────────────────────────────────────────────────────────┘  