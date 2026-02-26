//! CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimizer.
//!
//! Native implementation using nalgebra for the Shadow Tuner's parameter optimization.
//! Optimizes continuous parameters in unbounded space with sigmoid boundary mapping.
//!
//! # Architecture
//!
//! ```text
//! Unbounded ℝⁿ (CMA-ES internal) ←→ Bounded [min, max] (user space)
//!                sigmoid / inv_sigmoid
//! ```
//!
//! The optimizer operates in unbounded space and maps to bounded parameters
//! via sigmoid transforms, ensuring all evaluated parameters respect their bounds.

use nalgebra::{DMatrix, DVector};
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// ParamBound
// ---------------------------------------------------------------------------

/// Defines min/max bounds for a single parameter dimension.
///
/// Uses sigmoid mapping to transform between unbounded CMA-ES internal
/// space and the bounded parameter space the objective function expects.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParamBound {
    pub min: f64,
    pub max: f64,
}

impl ParamBound {
    /// Maps an unbounded real value to the bounded interval [min, max] via sigmoid.
    ///
    /// `min + (max - min) / (1 + exp(-raw))`
    pub fn map_sigmoid(&self, raw: f64) -> f64 {
        let sigmoid = 1.0 / (1.0 + (-raw).exp());
        self.min + (self.max - self.min) * sigmoid
    }

    /// Maps a bounded value in [min, max] back to unbounded space via logit.
    ///
    /// `ln(t / (1 - t))` where `t = (value - min) / (max - min)` clamped to [0.001, 0.999].
    pub fn inv_sigmoid(&self, value: f64) -> f64 {
        let t = ((value - self.min) / (self.max - self.min)).clamp(0.001, 0.999);
        (t / (1.0 - t)).ln()
    }
}

// ---------------------------------------------------------------------------
// CmaEsOptimizer
// ---------------------------------------------------------------------------

/// CMA-ES optimizer for continuous parameter optimization.
///
/// Maintains a multivariate normal distribution (mean, sigma, covariance)
/// that adapts over generations to concentrate around high-fitness regions.
/// All internal state lives in unbounded ℝⁿ; boundary enforcement happens
/// via sigmoid mapping at decode time.
pub struct CmaEsOptimizer {
    /// Number of parameters being optimized.
    dimension: usize,
    /// Number of candidate solutions sampled each generation.
    population_size: usize,
    /// Number of top individuals used for recombination (top half).
    mu: usize,
    /// Distribution mean in unbounded space (pre-sigmoid).
    mean: DVector<f64>,
    /// Global step size controlling the spread of sampling.
    sigma: f64,
    /// Covariance matrix encoding the shape of the search distribution.
    covariance: DMatrix<f64>,
    /// Evolution path for covariance matrix adaptation.
    p_c: DVector<f64>,
    /// Evolution path for step-size adaptation.
    p_sigma: DVector<f64>,
    /// Recombination weights for the top-mu individuals (sum to 1.0).
    weights: DVector<f64>,
    /// Variance-effective selection mass: `1 / sum(w_i^2)`.
    mu_eff: f64,
    /// Learning rate for p_c (covariance evolution path).
    c_c: f64,
    /// Learning rate for p_sigma (step-size evolution path).
    c_sigma: f64,
    /// Rank-one update weight for covariance.
    c_1: f64,
    /// Rank-mu update weight for covariance.
    c_mu: f64,
    /// Damping factor for step-size adaptation.
    d_sigma: f64,
    /// Expected norm of a standard normal vector: `E[||N(0, I)||]`.
    expected_norm: f64,
    /// Per-dimension bounds for sigmoid mapping.
    bounds: Vec<ParamBound>,
    /// Generation counter.
    generation: u64,
    /// Random number generator for sampling.
    rng: SmallRng,
}

impl CmaEsOptimizer {
    /// Creates a new CMA-ES optimizer.
    ///
    /// # Arguments
    /// * `bounds` — per-dimension [min, max] bounds
    /// * `initial_mean_bounded` — starting point in bounded (user) space
    ///
    /// # Panics
    /// Panics if `bounds.len() != initial_mean_bounded.len()`.
    pub fn new(bounds: Vec<ParamBound>, initial_mean_bounded: &[f64]) -> Self {
        assert_eq!(
            bounds.len(),
            initial_mean_bounded.len(),
            "bounds length ({}) must match initial_mean length ({})",
            bounds.len(),
            initial_mean_bounded.len()
        );

        let dimension = bounds.len();
        let n = dimension as f64;

        // Population sizing from Hansen's tutorial
        let population_size = 4 + (3.0 * n.ln()).floor() as usize;
        let mu = population_size / 2;

        // Recombination weights: w_i = ln(mu + 0.5) - ln(i + 1), normalized
        let raw_weights: Vec<f64> = (0..mu)
            .map(|i| (mu as f64 + 0.5).ln() - ((i + 1) as f64).ln())
            .collect();
        let w_sum: f64 = raw_weights.iter().sum();
        let weights = DVector::from_iterator(mu, raw_weights.iter().map(|w| w / w_sum));

        // Variance-effective selection mass
        let mu_eff = 1.0 / weights.iter().map(|w| w * w).sum::<f64>();

        // Learning rates and damping (Hansen's standard formulae)
        let c_sigma = (mu_eff + 2.0) / (n + mu_eff + 5.0);

        let d_sigma = 1.0 + 2.0 * (((mu_eff - 1.0) / (n + 1.0)).sqrt() - 1.0).max(0.0) + c_sigma;

        let c_c = (4.0 + mu_eff / n) / (n + 4.0 + 2.0 * mu_eff / n);

        let c_1 = 2.0 / ((n + 1.3).powi(2) + mu_eff);

        let c_mu =
            (2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((n + 2.0).powi(2) + mu_eff)).min(1.0 - c_1);

        // Expected norm of N(0, I) in n dimensions
        let expected_norm = n.sqrt() * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n.powi(2)));

        // Transform initial mean from bounded to unbounded space
        let mean = DVector::from_iterator(
            dimension,
            initial_mean_bounded
                .iter()
                .zip(bounds.iter())
                .map(|(&val, b)| b.inv_sigmoid(val)),
        );

        Self {
            dimension,
            population_size,
            mu,
            mean,
            sigma: 0.3,
            covariance: DMatrix::identity(dimension, dimension),
            p_c: DVector::zeros(dimension),
            p_sigma: DVector::zeros(dimension),
            weights,
            mu_eff,
            c_c,
            c_sigma,
            c_1,
            c_mu,
            d_sigma,
            expected_norm,
            bounds,
            generation: 0,
            rng: SmallRng::from_entropy(),
        }
    }

    /// Samples `population_size` candidate solutions from the current distribution.
    ///
    /// Returns individuals in unbounded ℝⁿ space. Use [`decode_individual`] to
    /// map them into bounded parameter space for fitness evaluation.
    pub fn sample_population(&mut self) -> Vec<DVector<f64>> {
        // Eigendecompose covariance: C = V * D^2 * V^T
        let eigen = nalgebra::SymmetricEigen::new(self.covariance.clone());

        // Clamp eigenvalues to prevent numerical issues
        let sqrt_eigenvalues = eigen.eigenvalues.map(|v| v.max(1e-20).sqrt());
        let sqrt_d = DMatrix::from_diagonal(&sqrt_eigenvalues);

        // Transform matrix: B * D where B = eigenvectors
        let bd = &eigen.eigenvectors * &sqrt_d;

        let mut population = Vec::with_capacity(self.population_size);
        for _ in 0..self.population_size {
            // Sample z ~ N(0, I)
            let z = DVector::from_iterator(
                self.dimension,
                (0..self.dimension).map(|_| StandardNormal.sample(&mut self.rng)),
            );
            // x_k = mean + sigma * B * D * z
            let x_k = &self.mean + self.sigma * &bd * &z;
            population.push(x_k);
        }

        population
    }

    /// Decodes an unbounded individual into bounded parameter space via sigmoid.
    pub fn decode_individual(&self, x: &DVector<f64>) -> Vec<f64> {
        x.iter()
            .zip(self.bounds.iter())
            .map(|(&raw, b)| b.map_sigmoid(raw))
            .collect()
    }

    /// Performs one generation of the CMA-ES update.
    ///
    /// # Arguments
    /// * `population` — candidate solutions from [`sample_population`] (unbounded)
    /// * `fitnesses` — fitness values for each individual (higher is better)
    ///
    /// # Panics
    /// Panics if `population.len() != fitnesses.len()` or
    /// `population.len() != population_size`.
    pub fn update(&mut self, population: &[DVector<f64>], fitnesses: &[f64]) {
        assert_eq!(population.len(), fitnesses.len());
        assert_eq!(population.len(), self.population_size);

        // 1. Sort by fitness descending (we maximize)
        let mut indices: Vec<usize> = (0..self.population_size).collect();
        indices.sort_by(|&a, &b| {
            fitnesses[b]
                .partial_cmp(&fitnesses[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // 2. Weighted mean of top-mu individuals
        let old_mean = self.mean.clone();
        let mut new_mean = DVector::zeros(self.dimension);
        for (w_idx, &pop_idx) in indices.iter().take(self.mu).enumerate() {
            new_mean += self.weights[w_idx] * &population[pop_idx];
        }

        // 3. Compute C^{-1/2} for the p_sigma update
        let eigen = nalgebra::SymmetricEigen::new(self.covariance.clone());
        let inv_sqrt_eigenvalues = eigen.eigenvalues.map(|v| 1.0 / v.max(1e-20).sqrt());
        let inv_sqrt_d = DMatrix::from_diagonal(&inv_sqrt_eigenvalues);
        let c_inv_sqrt = &eigen.eigenvectors * &inv_sqrt_d * eigen.eigenvectors.transpose();

        // Displacement vector (normalized by sigma)
        let mean_diff = (&new_mean - &old_mean) / self.sigma;

        // 4. Step-size evolution path
        self.p_sigma = (1.0 - self.c_sigma) * &self.p_sigma
            + (self.c_sigma * (2.0 - self.c_sigma) * self.mu_eff).sqrt() * &c_inv_sqrt * &mean_diff;

        // 5. h_sigma: stalling indicator
        let gen_factor = 1.0 - (1.0 - self.c_sigma).powi(2 * (self.generation as i32 + 1));
        let p_sigma_norm = self.p_sigma.norm();
        let h_sigma_threshold =
            (1.4 + 2.0 / (self.dimension as f64 + 1.0)) * self.expected_norm * gen_factor.sqrt();
        let h_sigma = if p_sigma_norm < h_sigma_threshold {
            1.0
        } else {
            0.0
        };

        // 6. Covariance evolution path
        self.p_c = (1.0 - self.c_c) * &self.p_c
            + h_sigma * (self.c_c * (2.0 - self.c_c) * self.mu_eff).sqrt() * &mean_diff;

        // 7. Rank-mu component: weighted outer products of displacement vectors
        let mut rank_mu_update = DMatrix::zeros(self.dimension, self.dimension);
        for (w_idx, &pop_idx) in indices.iter().take(self.mu).enumerate() {
            let y_i = (&population[pop_idx] - &old_mean) / self.sigma;
            rank_mu_update += self.weights[w_idx] * &y_i * y_i.transpose();
        }

        // 8. Covariance update
        //    C = (1 - c_1 - c_mu) * C + c_1 * (p_c * p_c^T + delta_h * C) + c_mu * rank_mu
        //    where delta_h = (1 - h_sigma) * c_c * (2 - c_c)
        let delta_h_sigma = (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c);
        let base_weight = 1.0 - self.c_1 - self.c_mu + self.c_1 * delta_h_sigma;
        self.covariance = base_weight * &self.covariance
            + self.c_1 * &self.p_c * self.p_c.transpose()
            + self.c_mu * &rank_mu_update;

        // Enforce symmetry (correct floating-point drift)
        self.covariance = (&self.covariance + self.covariance.transpose()) * 0.5;

        // 9. Step-size update
        self.sigma *=
            ((self.c_sigma / self.d_sigma) * (p_sigma_norm / self.expected_norm - 1.0)).exp();

        // Clamp sigma to prevent collapse or explosion
        self.sigma = self.sigma.clamp(1e-10, 10.0);

        // 10. Update mean
        self.mean = new_mean;

        // NaN safety: if mean or sigma went NaN, reset to a sane state
        if self.mean.iter().any(|v| v.is_nan()) || self.sigma.is_nan() {
            log::warn!(
                "CMA-ES generation {}: NaN detected, resetting distribution",
                self.generation
            );
            self.mean = DVector::zeros(self.dimension);
            self.sigma = 0.3;
            self.covariance = DMatrix::identity(self.dimension, self.dimension);
            self.p_c = DVector::zeros(self.dimension);
            self.p_sigma = DVector::zeros(self.dimension);
        }

        self.generation += 1;
    }

    /// Returns `true` if the step size has fallen below `threshold`,
    /// indicating the optimizer has converged.
    pub fn has_converged(&self, threshold: f64) -> bool {
        self.sigma < threshold
    }

    /// Resets the distribution around a new center point (in bounded space).
    ///
    /// Sigma, covariance, and evolution paths are reset; generation count is preserved.
    pub fn reset_around(&mut self, center_bounded: &[f64]) {
        assert_eq!(center_bounded.len(), self.dimension);
        self.mean = DVector::from_iterator(
            self.dimension,
            center_bounded
                .iter()
                .zip(self.bounds.iter())
                .map(|(&val, b)| b.inv_sigmoid(val)),
        );
        self.sigma = 0.3;
        self.covariance = DMatrix::identity(self.dimension, self.dimension);
        self.p_c = DVector::zeros(self.dimension);
        self.p_sigma = DVector::zeros(self.dimension);
    }

    /// Returns the current generation count.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Returns the current global step size.
    pub fn sigma(&self) -> f64 {
        self.sigma
    }

    /// Returns the current mean vector in unbounded space as a plain Vec.
    pub fn mean_vec(&self) -> Vec<f64> {
        self.mean.iter().copied().collect()
    }

    /// Restore optimizer state from a checkpoint.
    ///
    /// Restores mean (in unbounded space), sigma, and generation count.
    /// Covariance and evolution paths are reset to identity/zero — this is
    /// acceptable because the covariance adapts quickly (O(10) generations).
    pub fn restore_state(&mut self, mean_unbounded: &[f64], sigma: f64, generation: u64) {
        if mean_unbounded.len() == self.dimension {
            self.mean = DVector::from_column_slice(mean_unbounded);
            self.sigma = sigma.clamp(1e-10, 10.0);
            self.generation = generation;
            // Covariance and paths reset — they'll re-adapt in ~10 generations
            self.covariance = DMatrix::identity(self.dimension, self.dimension);
            self.p_c = DVector::zeros(self.dimension);
            self.p_sigma = DVector::zeros(self.dimension);
        }
    }

    /// Returns the decoded best individual and its fitness from a population evaluation.
    ///
    /// Fitness is maximized: the individual with the highest fitness wins.
    pub fn best_params_from_population(
        &self,
        population: &[DVector<f64>],
        fitnesses: &[f64],
    ) -> (Vec<f64>, f64) {
        assert_eq!(population.len(), fitnesses.len());
        assert!(!population.is_empty(), "population must be non-empty");

        let best_idx = fitnesses
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .expect("population is non-empty");

        let decoded = self.decode_individual(&population[best_idx]);
        (decoded, fitnesses[best_idx])
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_bounds() -> Vec<ParamBound> {
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
            }, // spread_floor_bps
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

    fn default_initial() -> Vec<f64> {
        vec![1.0, 10.0, 5.0, 2.0, 0.30, 0.25, 0.03, 2.0]
    }

    #[test]
    fn test_sigmoid_roundtrip() {
        let bound = ParamBound {
            min: -5.0,
            max: 5.0,
        };
        for &raw in &[-3.0, -1.0, 0.0, 0.5, 2.0, 5.0] {
            let mapped = bound.map_sigmoid(raw);
            let back = bound.inv_sigmoid(mapped);
            assert!(
                (back - raw).abs() < 1e-6,
                "roundtrip failed: raw={raw}, mapped={mapped}, back={back}"
            );
        }

        // Test with asymmetric bounds
        let bound2 = ParamBound {
            min: 0.05,
            max: 5.0,
        };
        for &raw in &[-2.0, 0.0, 1.0, 3.0] {
            let mapped = bound2.map_sigmoid(raw);
            let back = bound2.inv_sigmoid(mapped);
            assert!(
                (back - raw).abs() < 1e-3,
                "roundtrip failed for asymmetric bound: raw={raw}, back={back}"
            );
        }
    }

    #[test]
    fn test_sigmoid_bounds() {
        let bound = ParamBound {
            min: 1.5,
            max: 15.0,
        };
        for &raw in &[-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0] {
            let val = bound.map_sigmoid(raw);
            assert!(
                val >= bound.min && val <= bound.max,
                "out of bounds: raw={raw}, val={val}, min={}, max={}",
                bound.min,
                bound.max
            );
        }
    }

    #[test]
    fn test_population_in_bounds() {
        let bounds = default_bounds();
        let initial = default_initial();
        let mut optimizer = CmaEsOptimizer::new(bounds.clone(), &initial);

        let population = optimizer.sample_population();
        assert_eq!(population.len(), optimizer.population_size);

        for (k, individual) in population.iter().enumerate() {
            let decoded = optimizer.decode_individual(individual);
            assert_eq!(decoded.len(), bounds.len());
            for (i, (&val, b)) in decoded.iter().zip(bounds.iter()).enumerate() {
                assert!(
                    val >= b.min && val <= b.max,
                    "individual {k} dim {i} out of bounds: val={val}, min={}, max={}",
                    b.min,
                    b.max
                );
            }
        }
    }

    #[test]
    fn test_convergence_on_quadratic() {
        // 2D problem: maximize -(x-2)^2 - (y-1)^2
        // Optimum at (2, 1) with fitness 0.
        let bounds = vec![
            ParamBound {
                min: -5.0,
                max: 5.0,
            },
            ParamBound {
                min: -5.0,
                max: 5.0,
            },
        ];
        let initial = vec![0.0, 0.0];
        let mut optimizer = CmaEsOptimizer::new(bounds, &initial);

        let mut best_fitness = f64::NEG_INFINITY;
        let mut best_params = vec![0.0, 0.0];

        for _ in 0..30 {
            let population = optimizer.sample_population();
            let fitnesses: Vec<f64> = population
                .iter()
                .map(|x| {
                    let decoded = optimizer.decode_individual(x);
                    let dx = decoded[0] - 2.0;
                    let dy = decoded[1] - 1.0;
                    -(dx * dx + dy * dy)
                })
                .collect();

            let (params, fitness) = optimizer.best_params_from_population(&population, &fitnesses);
            if fitness > best_fitness {
                best_fitness = fitness;
                best_params = params;
            }

            optimizer.update(&population, &fitnesses);
        }

        // Should converge reasonably close to (2, 1)
        let dx = best_params[0] - 2.0;
        let dy = best_params[1] - 1.0;
        let dist = (dx * dx + dy * dy).sqrt();
        assert!(
            dist < 1.0,
            "did not converge near (2,1): best=({:.3}, {:.3}), dist={dist:.3}",
            best_params[0],
            best_params[1]
        );
    }

    #[test]
    fn test_no_nan_after_many_generations() {
        let bounds = default_bounds();
        let initial = default_initial();
        let mut optimizer = CmaEsOptimizer::new(bounds, &initial);

        for gen in 0..50 {
            let population = optimizer.sample_population();
            // Use a simple separable objective so fitnesses vary
            let fitnesses: Vec<f64> = population
                .iter()
                .map(|x| {
                    let decoded = optimizer.decode_individual(x);
                    -decoded.iter().map(|v| (v - 1.0).powi(2)).sum::<f64>()
                })
                .collect();

            optimizer.update(&population, &fitnesses);

            // Check for NaN in critical state
            assert!(
                !optimizer.sigma.is_nan(),
                "sigma is NaN at generation {gen}"
            );
            assert!(
                optimizer.mean.iter().all(|v| !v.is_nan()),
                "mean has NaN at generation {gen}"
            );
            for i in 0..optimizer.dimension {
                assert!(
                    !optimizer.covariance[(i, i)].is_nan(),
                    "covariance diagonal [{i},{i}] is NaN at generation {gen}"
                );
            }
        }
    }

    #[test]
    fn test_optimizer_reset() {
        let bounds = vec![
            ParamBound {
                min: -5.0,
                max: 5.0,
            },
            ParamBound {
                min: -5.0,
                max: 5.0,
            },
        ];
        let initial = vec![0.0, 0.0];
        let mut optimizer = CmaEsOptimizer::new(bounds, &initial);

        // Run a few generations to evolve away from initial
        for _ in 0..5 {
            let population = optimizer.sample_population();
            let fitnesses: Vec<f64> = population
                .iter()
                .map(|x| {
                    let decoded = optimizer.decode_individual(x);
                    -(decoded[0].powi(2) + decoded[1].powi(2))
                })
                .collect();
            optimizer.update(&population, &fitnesses);
        }

        let gen_before = optimizer.generation();
        assert!(gen_before >= 5);

        // Sigma should have changed from initial 0.3 (test is meaningful)
        let sigma_before_reset = optimizer.sigma();
        assert!(
            (sigma_before_reset - 0.3).abs() > 1e-6,
            "sigma did not evolve away from 0.3: {sigma_before_reset}"
        );

        // Reset around a new center
        let new_center = vec![3.0, -2.0];
        optimizer.reset_around(&new_center);

        // Sigma should be back to 0.3
        assert!(
            (optimizer.sigma() - 0.3).abs() < 1e-12,
            "sigma not reset: {}",
            optimizer.sigma()
        );

        // Generation count should be preserved
        assert_eq!(optimizer.generation(), gen_before);

        // Decoded mean should be near the new center
        let decoded_mean: Vec<f64> = optimizer
            .mean
            .iter()
            .zip(optimizer.bounds.iter())
            .map(|(&raw, b)| b.map_sigmoid(raw))
            .collect();
        assert!(
            (decoded_mean[0] - 3.0).abs() < 0.1,
            "mean x not near 3.0: {}",
            decoded_mean[0]
        );
        assert!(
            (decoded_mean[1] - (-2.0)).abs() < 0.1,
            "mean y not near -2.0: {}",
            decoded_mean[1]
        );
    }

    #[test]
    fn test_mean_vec_matches_decoded() {
        let bounds = default_bounds();
        let initial = default_initial();
        let optimizer = CmaEsOptimizer::new(bounds.clone(), &initial);

        let mean_vec = optimizer.mean_vec();
        assert_eq!(mean_vec.len(), bounds.len());

        // Decode via sigmoid and verify roundtrip with initial values
        let decoded: Vec<f64> = mean_vec
            .iter()
            .zip(bounds.iter())
            .map(|(&raw, b)| b.map_sigmoid(raw))
            .collect();
        for (i, (&dec, &ini)) in decoded.iter().zip(initial.iter()).enumerate() {
            assert!(
                (dec - ini).abs() < 1e-3,
                "dim {i}: decoded={dec}, initial={ini}"
            );
        }
    }

    #[test]
    fn test_restore_state() {
        let bounds = vec![
            ParamBound {
                min: -5.0,
                max: 5.0,
            },
            ParamBound {
                min: -5.0,
                max: 5.0,
            },
        ];
        let initial = vec![0.0, 0.0];
        let mut optimizer = CmaEsOptimizer::new(bounds, &initial);

        // Run a few generations to evolve state
        for _ in 0..5 {
            let population = optimizer.sample_population();
            let fitnesses: Vec<f64> = population
                .iter()
                .map(|x| {
                    let decoded = optimizer.decode_individual(x);
                    -(decoded[0].powi(2) + decoded[1].powi(2))
                })
                .collect();
            optimizer.update(&population, &fitnesses);
        }

        // Save state
        let saved_mean = optimizer.mean_vec();
        let saved_sigma = optimizer.sigma();
        let saved_gen = optimizer.generation();
        assert!(saved_gen >= 5);

        // Create a fresh optimizer and restore
        let bounds2 = vec![
            ParamBound {
                min: -5.0,
                max: 5.0,
            },
            ParamBound {
                min: -5.0,
                max: 5.0,
            },
        ];
        let mut restored = CmaEsOptimizer::new(bounds2, &[0.0, 0.0]);
        restored.restore_state(&saved_mean, saved_sigma, saved_gen);

        // Verify restored state
        assert_eq!(restored.generation(), saved_gen);
        assert!((restored.sigma() - saved_sigma).abs() < 1e-12);
        let restored_mean = restored.mean_vec();
        for (i, (&r, &s)) in restored_mean.iter().zip(saved_mean.iter()).enumerate() {
            assert!((r - s).abs() < 1e-12, "dim {i}: restored={r}, saved={s}");
        }

        // Verify sampling still works after restore
        let population = restored.sample_population();
        assert!(!population.is_empty());
        for individual in &population {
            let decoded = restored.decode_individual(individual);
            assert_eq!(decoded.len(), 2);
            for &v in &decoded {
                assert!(v.is_finite());
            }
        }
    }
}
