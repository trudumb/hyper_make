//! Background thread orchestrator for the Shadow Tuner.
//!
//! Ties together the [`ShadowBufferConsumer`], [`CmaEsOptimizer`], and
//! [`DynamicParams`] to run continuous CMA-ES optimization on a dedicated OS thread.
//!
//! # Architecture
//!
//! ```text
//! Live Engine                Shadow Tuner Thread (this module)
//! -----------                --------------------------------
//! ShadowBufferProducer  -->  ShadowBufferConsumer::drain()
//!                                 |
//!                            CmaEsOptimizer::sample_population()
//!                                 |
//!                            ReplayEngine::process_event() x N  (parallel via rayon)
//!                                 |
//!                            CmaEsOptimizer::update()
//!                                 |
//!                            watch::Sender<DynamicParams>  -->  Live Engine (hot-swap)
//! ```
//!
//! The thread sleeps for `cycle_interval_s` between generations and only runs
//! when the calibration gate has passed (enough data for meaningful replay).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use portable_atomic::AtomicF64;
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

use super::cma_es::{CmaEsOptimizer, ParamBound};
use super::replay::{ReplayConfig, ReplayEngine, ReplayEvent};
use super::shadow_buffer::ShadowBufferConsumer;
use crate::market_maker::strategy::dynamic_params::DynamicParams;

// ---------------------------------------------------------------------------
// ShadowTunerCheckpoint
// ---------------------------------------------------------------------------

/// Checkpoint for persisting shadow tuner state across restarts.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ShadowTunerCheckpoint {
    /// Current best parameters found by the optimizer.
    #[serde(default)]
    pub current_best: Option<DynamicParams>,
    /// CMA-ES optimizer mean in unbounded space.
    #[serde(default)]
    pub optimizer_mean: Vec<f64>,
    /// CMA-ES step size.
    #[serde(default)]
    pub optimizer_sigma: f64,
    /// Current generation count.
    #[serde(default)]
    pub generation: u64,
    /// Number of completed optimization cycles.
    #[serde(default)]
    pub cycles_completed: u64,
}

// ---------------------------------------------------------------------------
// TunerConfig
// ---------------------------------------------------------------------------

/// Runtime configuration for the shadow tuner thread.
#[derive(Debug, Clone)]
pub struct TunerConfig {
    /// Seconds between CMA-ES generations.
    pub cycle_interval_s: u64,
    /// Minimum events required before running a generation.
    pub min_events_for_replay: usize,
    /// Minimum improvement ratio to inject new params.
    pub improvement_threshold: f64,
    /// Sigma threshold for convergence detection.
    pub convergence_sigma: f64,
    /// Maximum generations before forced reset.
    pub max_generations_before_reset: u64,
    /// Parameter bounds for the 8-dimensional search space.
    pub param_bounds: Vec<ParamBound>,
    /// Optional cap on rayon threads.
    pub rayon_threads: Option<usize>,
    /// Maximum position for replay config.
    pub replay_max_position: f64,
}

impl Default for TunerConfig {
    fn default() -> Self {
        Self {
            cycle_interval_s: 300,
            min_events_for_replay: 5000,
            improvement_threshold: 0.10,
            convergence_sigma: 0.01,
            max_generations_before_reset: 50,
            param_bounds: default_param_bounds(),
            rayon_threads: None,
            replay_max_position: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// default_param_bounds
// ---------------------------------------------------------------------------

/// Default bounds for the 8-dimensional search space.
///
/// Order matches [`DynamicParams::to_vec()`] / [`DynamicParams::from_vec()`].
pub fn default_param_bounds() -> Vec<ParamBound> {
    vec![
        ParamBound {
            min: 0.05,
            max: 5.0,
        }, // gamma_base
        ParamBound {
            min: 1.0,
            max: 20.0,
        }, // inventory_beta
        ParamBound {
            min: 1.5,
            max: 15.0,
        }, // spread_floor_bps  (note: > 1.5 fee)
        ParamBound { min: 1.0, max: 4.0 }, // toxic_hour_gamma_mult
        ParamBound {
            min: 0.05,
            max: 0.60,
        }, // alpha_touch
        ParamBound {
            min: 0.05,
            max: 0.50,
        }, // kelly_fraction
        ParamBound {
            min: 0.005,
            max: 0.10,
        }, // cascade_threshold
        ParamBound { min: 0.5, max: 5.0 }, // proactive_skew_sensitivity
    ]
}

// ---------------------------------------------------------------------------
// SharedEstimators
// ---------------------------------------------------------------------------

/// Shared atomic state from the live engine, used by the tuner for gating and replay.
pub struct SharedEstimators {
    /// Gate signal; tuner sleeps until this is `true`.
    pub calibration_gate_passed: Arc<AtomicBool>,
    /// Live order-arrival intensity from the online estimator.
    pub live_kappa: Arc<AtomicF64>,
    /// Live volatility estimate from the online estimator.
    pub live_sigma: Arc<AtomicF64>,
}

// ---------------------------------------------------------------------------
// ShadowTuner
// ---------------------------------------------------------------------------

/// Background optimization thread that continuously tunes macro hyperparameters.
///
/// Runs on a dedicated OS thread (not tokio) to avoid event loop starvation.
/// Communicates with the live engine via:
/// - [`ShadowBufferConsumer`] (receives market events via lock-free flume channel)
/// - [`tokio::sync::watch::Sender<DynamicParams>`] (publishes optimized params)
/// - [`SharedEstimators`] (calibration gate, live kappa and sigma)
pub struct ShadowTuner {
    consumer: ShadowBufferConsumer,
    params_tx: tokio::sync::watch::Sender<DynamicParams>,
    optimizer: CmaEsOptimizer,
    current_best: DynamicParams,
    current_best_score: f64,
    config: TunerConfig,
    cycle_count: u64,
    shared: SharedEstimators,
}

impl ShadowTuner {
    /// Create a new shadow tuner, optionally restoring from a checkpoint.
    ///
    /// # Arguments
    /// * `consumer` -- consumer end of the shadow buffer (from [`create_shadow_buffer`])
    /// * `params_tx` -- watch channel sender for publishing optimized params
    /// * `config` -- tuner runtime configuration
    /// * `shared` -- shared atomic state from the live engine
    /// * `initial_params` -- optional starting point; falls back to [`DynamicParams::default()`]
    /// * `checkpoint` -- optional checkpoint to restore optimizer state from
    pub fn new(
        consumer: ShadowBufferConsumer,
        params_tx: tokio::sync::watch::Sender<DynamicParams>,
        config: TunerConfig,
        shared: SharedEstimators,
        initial_params: Option<DynamicParams>,
        checkpoint: Option<ShadowTunerCheckpoint>,
    ) -> Self {
        let initial = initial_params.unwrap_or_default();
        let initial_vec = initial.to_vec();
        let mut optimizer = CmaEsOptimizer::new(config.param_bounds.clone(), &initial_vec);

        let (current_best, current_best_score, cycle_count) = if let Some(ref ckpt) = checkpoint {
            // Restore optimizer mean/sigma/generation if checkpoint has valid mean
            if !ckpt.optimizer_mean.is_empty() {
                optimizer.restore_state(
                    &ckpt.optimizer_mean,
                    ckpt.optimizer_sigma,
                    ckpt.generation,
                );
            }
            let best = ckpt.current_best.clone().unwrap_or(initial);
            // If we had a previous best, score is unknown — use NEG_INFINITY
            // so the first generation's result is always accepted
            (best, f64::NEG_INFINITY, ckpt.cycles_completed)
        } else {
            (initial, f64::NEG_INFINITY, 0)
        };

        Self {
            consumer,
            params_tx,
            optimizer,
            current_best,
            current_best_score,
            config,
            cycle_count,
            shared,
        }
    }

    /// Main optimization loop. Blocks the current thread.
    ///
    /// 1. Sleeps for `cycle_interval_s`
    /// 2. Checks calibration gate
    /// 3. Drains buffer, takes snapshot
    /// 4. Runs one CMA-ES generation with parallel evaluation
    /// 5. If improvement threshold met, publishes new params
    /// 6. Handles convergence reset
    pub fn run(&mut self) {
        info!("[ShadowTuner] Started on dedicated OS thread");

        // Optionally create a dedicated rayon thread pool
        let pool = self.config.rayon_threads.map(|n| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .expect("Failed to build rayon pool for shadow tuner")
        });

        loop {
            std::thread::sleep(Duration::from_secs(self.config.cycle_interval_s));

            // Check calibration gate
            if !self.shared.calibration_gate_passed.load(Ordering::Relaxed) {
                debug!("[ShadowTuner] Calibration gate not passed, sleeping");
                continue;
            }

            // Drain and snapshot
            self.consumer.drain();
            let event_count = self.consumer.len();
            if event_count < self.config.min_events_for_replay {
                debug!(
                    "[ShadowTuner] Insufficient events ({event_count} < {}), skipping",
                    self.config.min_events_for_replay
                );
                continue;
            }
            let events = self.consumer.snapshot();

            // Read live estimator values
            let kappa = self.shared.live_kappa.load(Ordering::Relaxed).max(1.0);
            let sigma = self.shared.live_sigma.load(Ordering::Relaxed);

            // Run one CMA-ES generation
            let (best_params, best_score) = if let Some(ref pool) = pool {
                pool.install(|| self.run_generation(&events, kappa, sigma))
            } else {
                self.run_generation(&events, kappa, sigma)
            };

            self.cycle_count += 1;

            info!(
                "[ShadowTuner] Gen {} | score={:.4} (best={:.4}) | sigma={:.6} | events={} | cycle={}",
                self.optimizer.generation(),
                best_score,
                self.current_best_score,
                self.optimizer.sigma(),
                event_count,
                self.cycle_count,
            );

            // Check improvement threshold
            let improved = if self.current_best_score <= f64::NEG_INFINITY {
                // First generation -- always accept
                best_score > f64::NEG_INFINITY
            } else {
                best_score > self.current_best_score * (1.0 + self.config.improvement_threshold)
            };

            if improved {
                info!(
                    "[ShadowTuner] Injecting new params: score {:.4} -> {:.4} (+{:.1}%)",
                    self.current_best_score,
                    best_score,
                    if self.current_best_score > 0.0 {
                        (best_score / self.current_best_score - 1.0) * 100.0
                    } else {
                        0.0
                    },
                );
                self.current_best = best_params;
                self.current_best_score = best_score;
                // Publish to live engine via watch channel (non-blocking)
                let _ = self.params_tx.send(self.current_best.clone());
            }

            // Check convergence -> reset
            if self.optimizer.has_converged(self.config.convergence_sigma)
                || self.optimizer.generation() > self.config.max_generations_before_reset
            {
                info!(
                    "[ShadowTuner] Resetting optimizer (sigma={:.6}, gen={})",
                    self.optimizer.sigma(),
                    self.optimizer.generation(),
                );
                let center = self.current_best.to_vec();
                self.optimizer.reset_around(&center);
            }
        }
    }

    /// Execute one CMA-ES generation: sample, evaluate in parallel, update.
    ///
    /// Each candidate is decoded from unbounded CMA-ES space into a
    /// [`DynamicParams`], then evaluated by replaying the event buffer through
    /// a fresh [`ReplayEngine`].
    fn run_generation(
        &mut self,
        events: &[ReplayEvent],
        kappa: f64,
        _sigma: f64,
    ) -> (DynamicParams, f64) {
        let population = self.optimizer.sample_population();

        // Evaluate each candidate by replaying events with its params.
        // When a dedicated rayon pool is installed, `pool.install(|| ...)` in the
        // caller ensures these closures execute on that pool.
        let fitnesses: Vec<f64> = population
            .iter()
            .map(|individual| {
                let decoded = self.optimizer.decode_individual(individual);
                let params = DynamicParams::from_vec(&decoded, 0, 0);

                let replay_config = ReplayConfig {
                    gamma: params.gamma_base,
                    kappa,
                    inventory_beta: params.inventory_beta,
                    spread_floor_bps: params.spread_floor_bps,
                    max_position: self.config.replay_max_position,
                    ..Default::default()
                };

                let mut engine = ReplayEngine::new(replay_config);
                for event in events {
                    engine.process_event(event);
                }
                engine.report().fitness_score()
            })
            .collect();

        // Update CMA-ES state
        self.optimizer.update(&population, &fitnesses);

        // Extract best from this generation
        let (best_decoded, best_fitness) = self
            .optimizer
            .best_params_from_population(&population, &fitnesses);

        let version = self.current_best.version + 1;
        let timestamp_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        let best_params = DynamicParams::from_vec(&best_decoded, version, timestamp_ns);
        (best_params, best_fitness)
    }

    /// Create a checkpoint of the current tuner state.
    pub fn checkpoint(&self) -> ShadowTunerCheckpoint {
        ShadowTunerCheckpoint {
            current_best: Some(self.current_best.clone()),
            optimizer_mean: self.optimizer.mean_vec(),
            optimizer_sigma: self.optimizer.sigma(),
            generation: self.optimizer.generation(),
            cycles_completed: self.cycle_count,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::market_maker::simulation::shadow_buffer::create_shadow_buffer;

    #[test]
    fn test_tuner_config_defaults() {
        let config = TunerConfig::default();
        assert_eq!(config.cycle_interval_s, 300);
        assert_eq!(config.min_events_for_replay, 5000);
        assert_eq!(config.param_bounds.len(), 8);
    }

    #[test]
    fn test_default_param_bounds() {
        let bounds = default_param_bounds();
        assert_eq!(bounds.len(), 8);
        // gamma_base
        assert!((bounds[0].min - 0.05).abs() < 1e-10);
        assert!((bounds[0].max - 5.0).abs() < 1e-10);
        // spread_floor_bps must be > maker fee (1.5)
        assert!(bounds[2].min > 1.5 - 1e-10);
    }

    #[test]
    fn test_checkpoint_default() {
        let checkpoint = ShadowTunerCheckpoint::default();
        assert!(checkpoint.current_best.is_none());
        assert_eq!(checkpoint.generation, 0);
        assert_eq!(checkpoint.cycles_completed, 0);
    }

    #[test]
    fn test_checkpoint_serde_roundtrip() {
        let checkpoint = ShadowTunerCheckpoint {
            current_best: Some(DynamicParams::default()),
            optimizer_sigma: 0.3,
            generation: 5,
            cycles_completed: 2,
            ..Default::default()
        };
        let json = serde_json::to_string(&checkpoint).unwrap();
        let restored: ShadowTunerCheckpoint = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.generation, 5);
        assert!(restored.current_best.is_some());
    }

    fn test_shared_estimators() -> SharedEstimators {
        SharedEstimators {
            calibration_gate_passed: Arc::new(AtomicBool::new(false)),
            live_kappa: Arc::new(AtomicF64::new(2000.0)),
            live_sigma: Arc::new(AtomicF64::new(0.0002)),
        }
    }

    fn test_shared_estimators_gated() -> SharedEstimators {
        SharedEstimators {
            calibration_gate_passed: Arc::new(AtomicBool::new(true)),
            live_kappa: Arc::new(AtomicF64::new(2000.0)),
            live_sigma: Arc::new(AtomicF64::new(0.0002)),
        }
    }

    #[test]
    fn test_tuner_creation() {
        let (_, consumer) = create_shadow_buffer(1000, 60_000_000_000);
        let (tx, _rx) = tokio::sync::watch::channel(DynamicParams::default());

        let tuner = ShadowTuner::new(
            consumer,
            tx,
            TunerConfig::default(),
            test_shared_estimators(),
            None,
            None,
        );
        assert_eq!(tuner.cycle_count, 0);
        assert!(tuner.current_best_score <= f64::NEG_INFINITY);
    }

    #[test]
    fn test_run_generation_with_synthetic_data() {
        let (producer, consumer) = create_shadow_buffer(10_000, 60_000_000_000);
        let (tx, _rx) = tokio::sync::watch::channel(DynamicParams::default());

        // Feed synthetic market data
        for i in 0..100 {
            let ts = (i + 1) * 1_000_000_000u64;
            let mid = 100.0 + (i as f64) * 0.001;
            producer.push(ReplayEvent::L2Update {
                timestamp_ns: ts,
                best_bid: mid - 0.05,
                best_ask: mid + 0.05,
                bid_depth: 10.0,
                ask_depth: 10.0,
            });
            if i % 3 == 0 {
                producer.push(ReplayEvent::Trade {
                    timestamp_ns: ts + 500_000_000,
                    price: mid + 0.03,
                    size: 0.1,
                    is_buy: true,
                });
            }
        }

        let mut tuner = ShadowTuner::new(
            consumer,
            tx,
            TunerConfig {
                min_events_for_replay: 10,
                ..Default::default()
            },
            test_shared_estimators_gated(),
            None,
            None,
        );

        // Drain buffer so events are available
        tuner.consumer.drain();
        let events = tuner.consumer.snapshot();
        assert!(!events.is_empty());

        // Run a single generation
        let (best_params, best_score) = tuner.run_generation(&events, 2000.0, 0.0002);
        // Should produce valid params with positive gamma_base
        assert!(best_params.gamma_base > 0.0);
        // Score should be finite
        assert!(
            best_score.is_finite(),
            "Score should be finite: {best_score}"
        );
    }

    #[test]
    fn test_checkpoint_stores_mean() {
        let (_, consumer) = create_shadow_buffer(1000, 60_000_000_000);
        let (tx, _rx) = tokio::sync::watch::channel(DynamicParams::default());

        let tuner = ShadowTuner::new(
            consumer,
            tx,
            TunerConfig::default(),
            test_shared_estimators(),
            None,
            None,
        );

        let ckpt = tuner.checkpoint();
        // optimizer_mean should be non-empty (8 dimensions)
        assert_eq!(
            ckpt.optimizer_mean.len(),
            8,
            "checkpoint mean should have 8 elements, got {}",
            ckpt.optimizer_mean.len()
        );
        // All values should be finite
        for (i, &v) in ckpt.optimizer_mean.iter().enumerate() {
            assert!(v.is_finite(), "mean[{i}] is not finite: {v}");
        }
    }

    #[test]
    fn test_checkpoint_restore_roundtrip() {
        let (producer, consumer) = create_shadow_buffer(10_000, 60_000_000_000);
        let (tx, _rx) = tokio::sync::watch::channel(DynamicParams::default());

        // Feed synthetic data
        for i in 0..100 {
            let ts = (i + 1) * 1_000_000_000u64;
            let mid = 100.0 + (i as f64) * 0.001;
            producer.push(ReplayEvent::L2Update {
                timestamp_ns: ts,
                best_bid: mid - 0.05,
                best_ask: mid + 0.05,
                bid_depth: 10.0,
                ask_depth: 10.0,
            });
        }

        let mut tuner = ShadowTuner::new(
            consumer,
            tx,
            TunerConfig {
                min_events_for_replay: 10,
                ..Default::default()
            },
            test_shared_estimators_gated(),
            None,
            None,
        );

        // Run a generation to evolve optimizer state
        tuner.consumer.drain();
        let events = tuner.consumer.snapshot();
        let _ = tuner.run_generation(&events, 2000.0, 0.0002);
        tuner.cycle_count += 1;

        // Take checkpoint
        let ckpt = tuner.checkpoint();
        assert!(!ckpt.optimizer_mean.is_empty());
        assert!(ckpt.optimizer_sigma > 0.0);
        assert_eq!(ckpt.generation, 1);
        assert_eq!(ckpt.cycles_completed, 1);

        // Create a new tuner from the checkpoint
        let (_, consumer2) = create_shadow_buffer(1000, 60_000_000_000);
        let (tx2, _rx2) = tokio::sync::watch::channel(DynamicParams::default());

        let restored = ShadowTuner::new(
            consumer2,
            tx2,
            TunerConfig::default(),
            test_shared_estimators_gated(),
            None,
            Some(ckpt.clone()),
        );

        // Verify restored state
        assert_eq!(restored.cycle_count, ckpt.cycles_completed);
        assert_eq!(restored.optimizer.generation(), ckpt.generation);
        assert!((restored.optimizer.sigma() - ckpt.optimizer_sigma).abs() < 1e-12);

        // Verify restored optimizer mean matches checkpoint
        let restored_mean = restored.optimizer.mean_vec();
        assert_eq!(restored_mean.len(), ckpt.optimizer_mean.len());
        for (i, (&r, &c)) in restored_mean
            .iter()
            .zip(ckpt.optimizer_mean.iter())
            .enumerate()
        {
            assert!(
                (r - c).abs() < 1e-12,
                "dim {i}: restored={r}, checkpoint={c}"
            );
        }
    }
}
