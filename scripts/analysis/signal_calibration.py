#!/usr/bin/env python3
"""
Signal Calibration Analysis for Market Maker Fill Diagnostics.

Analyzes fill signal snapshots exported from the market maker to determine:
1. Which signals are calibrated (Brier scores)
2. Which signals correlate with toxic fills (markouts)
3. Signal reliability diagrams

Usage:
    python signal_calibration.py fills_with_signals.json

Output:
    - Brier scores for each signal
    - Signal-outcome correlations
    - Reliability diagram plots (saved as PNG)
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Plots will be skipped.")


def load_fills(path: str) -> list[dict]:
    """Load fill signal snapshots from JSON file."""
    with open(path) as f:
        return json.load(f)


def compute_brier_score(predictions: list[float], outcomes: list[bool]) -> float:
    """
    Compute Brier score: mean squared error of probability predictions.
    
    Lower is better:
    - 0.0 = perfect
    - 0.25 = random (for 50% base rate)
    - Higher = worse than random
    """
    if not predictions:
        return float('nan')
    
    predictions = np.array(predictions)
    outcomes = np.array(outcomes, dtype=float)
    
    return float(np.mean((predictions - outcomes) ** 2))


def compute_information_ratio(predictions: list[float], outcomes: list[bool]) -> float:
    """
    Compute Information Ratio: resolution / uncertainty.
    
    IR > 1.0 means the signal adds value.
    IR < 1.0 means the signal adds noise (remove it).
    """
    if not predictions or len(predictions) < 10:
        return float('nan')
    
    predictions = np.array(predictions)
    outcomes = np.array(outcomes, dtype=float)
    
    # Uncertainty: variance of outcomes
    base_rate = outcomes.mean()
    uncertainty = base_rate * (1 - base_rate)
    
    if uncertainty < 1e-10:
        return float('nan')
    
    # Resolution: how much predictions explain outcomes
    # Group predictions into bins and compute variance explained
    n_bins = min(10, len(predictions) // 10)
    if n_bins < 2:
        return float('nan')
    
    bins = np.linspace(0, 1, n_bins + 1)
    resolution = 0.0
    
    for i in range(n_bins):
        mask = (predictions >= bins[i]) & (predictions < bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_mean = outcomes[mask].mean()
        resolution += mask.sum() * (bin_mean - base_rate) ** 2
    
    resolution /= len(predictions)
    
    return resolution / uncertainty


def analyze_signal(
    fills: list[dict],
    signal_name: str,
    toxic_threshold_bps: float = -5.0
) -> dict:
    """
    Analyze a single signal's predictive power.
    
    Args:
        fills: List of fill signal snapshots
        signal_name: Name of the signal field to analyze
        toxic_threshold_bps: Markout threshold for "toxic" classification
    
    Returns:
        Dict with Brier score, IR, correlation, etc.
    """
    predictions = []
    outcomes = []
    
    for fill in fills:
        if signal_name not in fill:
            continue
        if fill.get('markout_10s_bps') is None:
            continue
        
        pred = fill[signal_name]
        # Handle signals that aren't [0, 1] probabilities
        if signal_name in ['momentum_bps', 'spread_bps', 'volatility', 'kappa']:
            continue  # Skip non-probability signals for Brier
        
        outcome = fill['markout_10s_bps'] < toxic_threshold_bps
        
        predictions.append(pred)
        outcomes.append(outcome)
    
    if not predictions:
        return {
            'signal': signal_name,
            'n_samples': 0,
            'brier_score': float('nan'),
            'information_ratio': float('nan'),
            'correlation': float('nan'),
        }
    
    # Compute metrics
    brier = compute_brier_score(predictions, outcomes)
    ir = compute_information_ratio(predictions, outcomes)
    
    # Correlation with markout
    markouts = [f['markout_10s_bps'] for f in fills 
                if signal_name in f and f.get('markout_10s_bps') is not None]
    signal_vals = [f[signal_name] for f in fills 
                   if signal_name in f and f.get('markout_10s_bps') is not None]
    
    if len(markouts) > 1:
        corr = float(np.corrcoef(signal_vals, markouts)[0, 1])
    else:
        corr = float('nan')
    
    return {
        'signal': signal_name,
        'n_samples': len(predictions),
        'brier_score': brier,
        'information_ratio': ir,
        'correlation_with_markout': corr,
        'base_rate_toxic': sum(outcomes) / len(outcomes) if outcomes else 0,
    }


def plot_reliability_diagram(
    fills: list[dict],
    signal_name: str,
    toxic_threshold_bps: float = -5.0,
    output_path: str = None
):
    """
    Plot reliability diagram for a probability signal.
    
    X-axis: Predicted probability (binned)
    Y-axis: Observed frequency of toxic fills
    
    Perfect calibration = diagonal line.
    """
    if not HAS_MATPLOTLIB:
        return
    
    predictions = []
    outcomes = []
    
    for fill in fills:
        if signal_name not in fill:
            continue
        if fill.get('markout_10s_bps') is None:
            continue
        
        pred = fill[signal_name]
        if not 0 <= pred <= 1:
            continue  # Skip non-probability values
        
        outcome = fill['markout_10s_bps'] < toxic_threshold_bps
        
        predictions.append(pred)
        outcomes.append(outcome)
    
    if len(predictions) < 20:
        print(f"Skipping reliability diagram for {signal_name}: insufficient samples ({len(predictions)})")
        return
    
    predictions = np.array(predictions)
    outcomes = np.array(outcomes, dtype=float)
    
    # Bin predictions
    n_bins = 10
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_counts = []
    bin_outcomes = []
    
    for i in range(n_bins):
        mask = (predictions >= bins[i]) & (predictions < bins[i + 1])
        if mask.sum() > 0:
            bin_counts.append(mask.sum())
            bin_outcomes.append(outcomes[mask].mean())
        else:
            bin_counts.append(0)
            bin_outcomes.append(float('nan'))
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    
    # Reliability curve
    valid = ~np.isnan(bin_outcomes)
    ax.plot(bin_centers[valid], np.array(bin_outcomes)[valid], 'bo-', label=signal_name)
    
    # Add count annotations
    for i, (x, y, n) in enumerate(zip(bin_centers, bin_outcomes, bin_counts)):
        if n > 0 and not np.isnan(y):
            ax.annotate(f'n={n}', (x, y), textcoords="offset points", xytext=(0, 10), ha='center')
    
    ax.set_xlabel('Predicted probability')
    ax.set_ylabel('Observed frequency')
    ax.set_title(f'Reliability Diagram: {signal_name}')
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved reliability diagram to {output_path}")
    else:
        plt.show()
    
    plt.close()


def analyze_all_signals(fills: list[dict], toxic_threshold_bps: float = -5.0) -> list[dict]:
    """Analyze all probability signals in the fill data."""
    
    # Signals that should be analyzed as probabilities
    prob_signals = [
        'pre_fill_toxicity',
        'continuation_p',
        'drift_prob_bearish',
        'cp_prob',
        'hmm_confidence',
        'hawkes_p_cluster',
    ]
    
    results = []
    for signal in prob_signals:
        result = analyze_signal(fills, signal, toxic_threshold_bps)
        if result['n_samples'] > 0:
            results.append(result)
    
    # Sort by information ratio (descending)
    results.sort(key=lambda x: -x['information_ratio'] if not np.isnan(x['information_ratio']) else -999)
    
    return results


def print_report(results: list[dict], fills: list[dict]):
    """Print calibration report to stdout."""
    
    n_total = len(fills)
    n_with_markout = sum(1 for f in fills if f.get('markout_10s_bps') is not None)
    
    print("=" * 80)
    print("SIGNAL CALIBRATION REPORT")
    print("=" * 80)
    print(f"\nTotal fills: {n_total}")
    print(f"Fills with markouts: {n_with_markout}")
    print(f"Coverage: {100 * n_with_markout / n_total:.1f}%\n")
    
    print("-" * 80)
    print(f"{'Signal':<25} {'N':>8} {'Brier':>10} {'IR':>10} {'Corr':>10} {'Toxic%':>10}")
    print("-" * 80)
    
    for r in results:
        brier_str = f"{r['brier_score']:.4f}" if not np.isnan(r['brier_score']) else "N/A"
        ir_str = f"{r['information_ratio']:.3f}" if not np.isnan(r['information_ratio']) else "N/A"
        corr_str = f"{r['correlation_with_markout']:.3f}" if not np.isnan(r['correlation_with_markout']) else "N/A"
        toxic_str = f"{100 * r['base_rate_toxic']:.1f}%"
        
        print(f"{r['signal']:<25} {r['n_samples']:>8} {brier_str:>10} {ir_str:>10} {corr_str:>10} {toxic_str:>10}")
    
    print("-" * 80)
    print("\nInterpretation:")
    print("  Brier: Lower is better (0=perfect, 0.25=random for 50% base rate)")
    print("  IR: >1.0 = signal adds value, <1.0 = signal adds noise (REMOVE IT)")
    print("  Corr: Negative correlation with markout = signal predicts toxicity")
    print()
    
    # Highlight problem signals
    problem_signals = [r for r in results if r['information_ratio'] < 1.0 and not np.isnan(r['information_ratio'])]
    if problem_signals:
        print("⚠️  SIGNALS WITH IR < 1.0 (consider removing):")
        for r in problem_signals:
            print(f"    - {r['signal']}: IR = {r['information_ratio']:.3f}")
    
    good_signals = [r for r in results if r['information_ratio'] > 1.0]
    if good_signals:
        print("\n✅ SIGNALS WITH IR > 1.0 (adding value):")
        for r in good_signals:
            print(f"    - {r['signal']}: IR = {r['information_ratio']:.3f}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python signal_calibration.py fills_with_signals.json [toxic_threshold_bps]")
        print("\nExample: python signal_calibration.py fills_with_signals.json -5.0")
        sys.exit(1)
    
    fills_path = sys.argv[1]
    toxic_threshold = float(sys.argv[2]) if len(sys.argv) > 2 else -5.0
    
    print(f"Loading fills from {fills_path}...")
    fills = load_fills(fills_path)
    print(f"Loaded {len(fills)} fills")
    
    print(f"\nAnalyzing signals (toxic threshold: {toxic_threshold} bps)...")
    results = analyze_all_signals(fills, toxic_threshold)
    
    print_report(results, fills)
    
    # Generate reliability diagrams
    if HAS_MATPLOTLIB:
        output_dir = Path(fills_path).parent
        for r in results:
            if r['n_samples'] >= 20:
                output_path = output_dir / f"reliability_{r['signal']}.png"
                plot_reliability_diagram(fills, r['signal'], toxic_threshold, str(output_path))


if __name__ == '__main__':
    main()
