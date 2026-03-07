//! Counterfactual analysis for offline "what-if" regret measurement.
//!
//! Measures per-fill regret: how much better would the FQI policy have done
//! compared to the actually-executed action. Provides the data gate for
//! increasing `fqi_blend_weight` beyond conservative levels.

use serde::{Deserialize, Serialize};

use super::experience::ExperienceRecord;
use super::fqi::FQIResult;

/// Per-fill counterfactual analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FillCounterfactual {
    /// Timestamp of the fill
    pub timestamp_ms: u64,
    /// State at fill time
    pub state_idx: usize,
    /// Actually-executed action
    pub actual_action: usize,
    /// FQI-recommended action
    pub recommended_action: usize,
    /// Actual reward received
    pub actual_reward: f64,
    /// Q-value of actual action
    pub q_actual: f64,
    /// Q-value of recommended action
    pub q_recommended: f64,
    /// Per-fill regret: Q(s, a*) - Q(s, a_actual)
    pub regret: f64,
    /// Whether the policy would have chosen differently
    pub is_suboptimal: bool,
}

/// Aggregate counterfactual report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CounterfactualReport {
    /// Total fills analyzed
    pub total_fills: usize,
    /// Fraction of fills where FQI would have chosen differently
    pub fraction_suboptimal: f64,
    /// Mean expected improvement from full policy adoption (bps)
    pub mean_expected_improvement: f64,
    /// Per-state regret: (state_idx, mean_regret, count)
    pub top_regret_states: Vec<(usize, f64, usize)>,
    /// Per-fill counterfactual details
    pub fills: Vec<FillCounterfactual>,
}

/// Counterfactual analyzer.
pub struct CounterfactualAnalysis;

impl CounterfactualAnalysis {
    /// Analyze a set of experience records against an FQI result.
    ///
    /// For each fill, computes the regret = Q(s, a*) - Q(s, a_actual)
    /// where a* is the FQI-greedy action.
    pub fn analyze(records: &[ExperienceRecord], fqi: &FQIResult) -> CounterfactualReport {
        let mut fills = Vec::with_capacity(records.len());
        let mut suboptimal_count = 0usize;
        let mut total_improvement = 0.0_f64;

        // Per-state regret accumulator: (sum_regret, count)
        let mut state_regret: std::collections::HashMap<usize, (f64, usize)> =
            std::collections::HashMap::new();

        let policy = fqi.greedy_policy();

        for r in records {
            let s = r.state_idx.min(44); // Clamp to valid range
            let a_actual = r.action_idx.min(7);
            let a_recommended = policy[s];

            let q_actual = fqi.q_value(s, a_actual);
            let q_recommended = fqi.q_value(s, a_recommended);
            let regret = (q_recommended - q_actual).max(0.0);
            let is_suboptimal = a_recommended != a_actual && regret > 0.01;

            if is_suboptimal {
                suboptimal_count += 1;
            }
            total_improvement += regret;

            let entry = state_regret.entry(s).or_insert((0.0, 0));
            entry.0 += regret;
            entry.1 += 1;

            fills.push(FillCounterfactual {
                timestamp_ms: r.timestamp_ms,
                state_idx: s,
                actual_action: a_actual,
                recommended_action: a_recommended,
                actual_reward: r.reward_total,
                q_actual,
                q_recommended,
                regret,
                is_suboptimal,
            });
        }

        let total = records.len().max(1) as f64;
        let fraction_suboptimal = suboptimal_count as f64 / total;
        let mean_expected_improvement = total_improvement / total;

        // Top regret states (sorted by mean regret, descending)
        let mut top_regret_states: Vec<(usize, f64, usize)> = state_regret
            .into_iter()
            .map(|(s, (sum, count))| (s, sum / count as f64, count))
            .collect();
        top_regret_states
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        top_regret_states.truncate(10); // Keep top 10

        CounterfactualReport {
            total_fills: records.len(),
            fraction_suboptimal,
            mean_expected_improvement,
            top_regret_states,
            fills,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::market_maker::learning::experience::ExperienceSource;

    fn make_record(state_idx: usize, action_idx: usize, reward: f64) -> ExperienceRecord {
        ExperienceRecord {
            state_idx,
            action_idx,
            reward_total: reward,
            edge_component: reward,
            inventory_penalty: 0.0,
            volatility_penalty: 0.0,
            inventory_change_penalty: 0.0,
            next_state_idx: 0,
            done: false,
            timestamp_ms: 0,
            session_id: "test".to_string(),
            source: ExperienceSource::Paper,
            side: "buy".to_string(),
            fill_price: 100.0,
            mid_price: 100.0,
            fill_size: 1.0,
            inventory: 0.0,
            regime: "normal".to_string(),
            drift_penalty: None,
            bandit_multiplier: None,
            vol_ratio: None,
            mdp_state_idx: Some(state_idx),
            bandit_arm_idx: Some(action_idx),
            inventory_risk_at_fill: None,
        }
    }

    #[test]
    fn test_counterfactual_known_q_table() {
        const N_ACTIONS: usize = 8;
        const N_CELLS: usize = 45 * N_ACTIONS;

        // Q-table where arm 5 is best for state 0 (Q=10), arm 3 is Q=5
        let mut q1 = vec![0.0; N_CELLS];
        q1[5] = 10.0; // state 0, arm 5
        q1[3] = 5.0; // state 0, arm 3

        let result = FQIResult {
            q1: q1.clone(),
            q2: q1,
            visit_counts: vec![10; N_CELLS],
            iterations: 10,
            bellman_residual: 0.001,
            total_records: 100,
        };

        // Record where we chose arm 3 (suboptimal — arm 5 was better)
        let records = vec![make_record(0, 3, 2.0)];
        let report = CounterfactualAnalysis::analyze(&records, &result);

        assert_eq!(report.total_fills, 1);
        assert!((report.fraction_suboptimal - 1.0).abs() < 1e-10);
        // Regret = Q(0,5) - Q(0,3) = 10 - 5 = 5
        assert!((report.mean_expected_improvement - 5.0).abs() < 1e-10);
        assert!(report.fills[0].is_suboptimal);
        assert_eq!(report.fills[0].recommended_action, 5);
    }

    #[test]
    fn test_counterfactual_optimal_actions() {
        const N_ACTIONS: usize = 8;
        const N_CELLS: usize = 45 * N_ACTIONS;

        // Q-table where arm 3 is best for state 0
        let mut q1 = vec![0.0; N_CELLS];
        q1[3] = 10.0; // state 0, arm 3

        let result = FQIResult {
            q1: q1.clone(),
            q2: q1,
            visit_counts: vec![10; N_CELLS],
            iterations: 10,
            bellman_residual: 0.001,
            total_records: 100,
        };

        // Record where we already chose the optimal arm 3
        let records = vec![make_record(0, 3, 5.0)];
        let report = CounterfactualAnalysis::analyze(&records, &result);

        assert_eq!(report.total_fills, 1);
        assert!((report.fraction_suboptimal).abs() < 1e-10);
        assert!((report.mean_expected_improvement).abs() < 1e-10);
        assert!(!report.fills[0].is_suboptimal);
    }

    #[test]
    fn test_counterfactual_empty_records() {
        let result = FQIResult {
            q1: vec![0.0; 45 * 8],
            q2: vec![0.0; 45 * 8],
            visit_counts: vec![0; 45 * 8],
            iterations: 0,
            bellman_residual: 0.0,
            total_records: 0,
        };

        let report = CounterfactualAnalysis::analyze(&[], &result);
        assert_eq!(report.total_fills, 0);
        assert!(report.fills.is_empty());
    }

    #[test]
    fn test_top_regret_states_sorted() {
        const N_ACTIONS: usize = 8;
        const N_CELLS: usize = 45 * N_ACTIONS;

        let mut q1 = vec![0.0; N_CELLS];
        // State 0: arm 5 is best (regret = 10 - 2 = 8 if arm 3 chosen)
        q1[5] = 10.0; // state 0, arm 5
        q1[3] = 2.0; // state 0, arm 3
                     // State 1: arm 7 is best (regret = 20 - 1 = 19 if arm 3 chosen)
        q1[N_ACTIONS + 7] = 20.0; // state 1, arm 7
        q1[N_ACTIONS + 3] = 1.0; // state 1, arm 3

        let result = FQIResult {
            q1: q1.clone(),
            q2: q1,
            visit_counts: vec![10; N_CELLS],
            iterations: 10,
            bellman_residual: 0.001,
            total_records: 100,
        };

        let records = vec![make_record(0, 3, 1.0), make_record(1, 3, 1.0)];
        let report = CounterfactualAnalysis::analyze(&records, &result);

        // State 1 should have higher regret than state 0
        assert!(report.top_regret_states.len() >= 2);
        assert_eq!(report.top_regret_states[0].0, 1); // Highest regret first
        assert!(report.top_regret_states[0].1 > report.top_regret_states[1].1);
    }
}
