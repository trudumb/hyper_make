#!/usr/bin/env python3
"""
Toxic Fill Audit for Market Maker Signal Diagnostics.

Analyzes fill signal snapshots to find:
1. Toxic fills and their signal states at fill time
2. "Unwarned" toxic fills where all signals were low
3. Patterns in when signals fail to warn

Usage:
    python toxic_fill_audit.py fills_with_signals.json

Output:
    - List of toxic fills with signal values
    - Summary of signal failures
    - Recommendations for signal tuning
"""

import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


def load_fills(path: str) -> list[dict]:
    """Load fill signal snapshots from JSON file."""
    with open(path) as f:
        return json.load(f)


def classify_fill(fill: dict, toxic_threshold_bps: float = -5.0) -> str:
    """
    Classify a fill based on markout.
    
    Returns:
        'toxic': Large adverse selection (markout < threshold)
        'neutral': Small adverse selection (-5 to +5 bps)
        'favorable': Price moved in our favor (markout > +5)
        'unknown': No markout data
    """
    markout = fill.get('markout_10s_bps')
    if markout is None:
        return 'unknown'
    
    if markout < toxic_threshold_bps:
        return 'toxic'
    elif markout > 5.0:
        return 'favorable'
    else:
        return 'neutral'


def get_signal_warnings(fill: dict, warning_thresholds: dict = None) -> list[str]:
    """
    Get list of signals that warned about toxicity for this fill.
    
    Default thresholds (signal > threshold = warning):
    - pre_fill_toxicity: 0.3
    - cp_prob: 0.3
    - hawkes_p_cluster: 0.3
    - is_toxic_regime: True (bool)
    - falling_knife_score: 1.0 (if buying)
    - rising_knife_score: 1.0 (if selling)
    """
    if warning_thresholds is None:
        warning_thresholds = {
            'pre_fill_toxicity': 0.3,
            'cp_prob': 0.3,
            'hawkes_p_cluster': 0.3,
            'continuation_p': 0.7,  # High continuation = momentum might continue against us
        }
    
    warnings = []
    
    for signal, threshold in warning_thresholds.items():
        value = fill.get(signal)
        if value is not None and value > threshold:
            warnings.append(signal)
    
    # Special handling for boolean signals
    if fill.get('is_toxic_regime'):
        warnings.append('is_toxic_regime')
    
    # Direction-dependent warnings
    side = fill.get('side', '').lower()
    if side == 'bid':  # We bought
        if fill.get('falling_knife_score', 0) > 1.0:
            warnings.append('falling_knife_score')
    elif side == 'ask':  # We sold
        if fill.get('rising_knife_score', 0) > 1.0:
            warnings.append('rising_knife_score')
    
    return warnings


def audit_toxic_fills(
    fills: list[dict],
    toxic_threshold_bps: float = -5.0,
    severe_threshold_bps: float = -15.0
) -> dict:
    """
    Audit all toxic fills to understand signal behavior.
    
    Returns dict with:
    - toxic_fills: List of toxic fill details
    - unwarned_fills: Fills where NO signals warned
    - signal_coverage: For each signal, % of toxic fills it warned about
    """
    toxic_fills = []
    unwarned_fills = []
    signal_warn_counts = defaultdict(int)
    total_toxic = 0
    
    for fill in fills:
        classification = classify_fill(fill, toxic_threshold_bps)
        if classification != 'toxic':
            continue
        
        total_toxic += 1
        markout = fill['markout_10s_bps']
        warnings = get_signal_warnings(fill)
        
        fill_summary = {
            'tid': fill.get('tid'),
            'timestamp_ms': fill.get('timestamp_ms'),
            'side': fill.get('side'),
            'price': fill.get('price'),
            'size': fill.get('size'),
            'markout_10s_bps': markout,
            'warnings': warnings,
            'n_warnings': len(warnings),
            'is_severe': markout < severe_threshold_bps,
            
            # Key signals at fill time
            'pre_fill_toxicity': fill.get('pre_fill_toxicity'),
            'cp_prob': fill.get('cp_prob'),
            'hawkes_p_cluster': fill.get('hawkes_p_cluster'),
            'continuation_p': fill.get('continuation_p'),
            'is_toxic_regime': fill.get('is_toxic_regime'),
            'falling_knife_score': fill.get('falling_knife_score'),
            'rising_knife_score': fill.get('rising_knife_score'),
            'momentum_bps': fill.get('momentum_bps'),
            'volatility': fill.get('volatility'),
        }
        
        toxic_fills.append(fill_summary)
        
        # Track which signals warned
        for signal in warnings:
            signal_warn_counts[signal] += 1
        
        # Track unwarned fills
        if len(warnings) == 0:
            unwarned_fills.append(fill_summary)
    
    # Compute signal coverage
    signal_coverage = {}
    for signal, count in signal_warn_counts.items():
        signal_coverage[signal] = count / total_toxic if total_toxic > 0 else 0
    
    return {
        'total_fills': len(fills),
        'total_toxic': total_toxic,
        'toxic_rate': total_toxic / len(fills) if fills else 0,
        'toxic_fills': toxic_fills,
        'unwarned_fills': unwarned_fills,
        'unwarned_rate': len(unwarned_fills) / total_toxic if total_toxic > 0 else 0,
        'signal_coverage': signal_coverage,
        'severe_count': sum(1 for f in toxic_fills if f['is_severe']),
    }


def analyze_unwarned_patterns(unwarned_fills: list[dict]) -> dict:
    """
    Analyze patterns in fills that no signal warned about.
    
    What do these fills have in common?
    """
    if not unwarned_fills:
        return {'n_fills': 0}
    
    # Extract patterns
    momentum_values = [f['momentum_bps'] for f in unwarned_fills if f.get('momentum_bps') is not None]
    volatility_values = [f['volatility'] for f in unwarned_fills if f.get('volatility') is not None]
    pre_fill_values = [f['pre_fill_toxicity'] for f in unwarned_fills if f.get('pre_fill_toxicity') is not None]
    cp_prob_values = [f['cp_prob'] for f in unwarned_fills if f.get('cp_prob') is not None]
    markout_values = [f['markout_10s_bps'] for f in unwarned_fills]
    
    # Side distribution
    side_counts = defaultdict(int)
    for f in unwarned_fills:
        side_counts[f.get('side', 'unknown')] += 1
    
    return {
        'n_fills': len(unwarned_fills),
        'side_distribution': dict(side_counts),
        'avg_momentum_bps': np.mean(momentum_values) if momentum_values else None,
        'avg_volatility': np.mean(volatility_values) if volatility_values else None,
        'avg_pre_fill_toxicity': np.mean(pre_fill_values) if pre_fill_values else None,
        'max_pre_fill_toxicity': max(pre_fill_values) if pre_fill_values else None,
        'avg_cp_prob': np.mean(cp_prob_values) if cp_prob_values else None,
        'max_cp_prob': max(cp_prob_values) if cp_prob_values else None,
        'avg_markout': np.mean(markout_values) if markout_values else None,
        'worst_markout': min(markout_values) if markout_values else None,
    }


def print_report(audit: dict, patterns: dict):
    """Print audit report to stdout."""
    
    print("=" * 80)
    print("TOXIC FILL AUDIT REPORT")
    print("=" * 80)
    
    print(f"\n📊 OVERVIEW:")
    print(f"  Total fills analyzed: {audit['total_fills']}")
    print(f"  Toxic fills: {audit['total_toxic']} ({100 * audit['toxic_rate']:.1f}%)")
    print(f"  Severe toxic (< -15 bps): {audit['severe_count']}")
    print(f"  Unwarned toxic fills: {len(audit['unwarned_fills'])} ({100 * audit['unwarned_rate']:.1f}%)")
    
    print(f"\n📈 SIGNAL COVERAGE (% of toxic fills each signal warned about):")
    print("-" * 60)
    sorted_coverage = sorted(audit['signal_coverage'].items(), key=lambda x: -x[1])
    for signal, coverage in sorted_coverage:
        bar = '█' * int(coverage * 20)
        print(f"  {signal:<25} {100 * coverage:5.1f}% {bar}")
    print("-" * 60)
    
    # Warn about blind spots
    missing_signals = [s for s, c in audit['signal_coverage'].items() if c < 0.3]
    if missing_signals:
        print(f"\n⚠️  BLIND SPOTS (signals covering < 30% of toxic fills):")
        for s in missing_signals:
            print(f"    - {s}")
    
    print(f"\n🔍 UNWARNED FILL PATTERNS:")
    if patterns['n_fills'] == 0:
        print("  No unwarned fills! Signals are providing good coverage.")
    else:
        print(f"  Side distribution: {patterns['side_distribution']}")
        print(f"  Avg momentum at fill: {patterns['avg_momentum_bps']:.1f} bps" if patterns['avg_momentum_bps'] else "  Avg momentum: N/A")
        print(f"  Avg volatility at fill: {patterns['avg_volatility']:.6f}" if patterns['avg_volatility'] else "  Avg volatility: N/A")
        print(f"  Avg pre_fill_toxicity: {patterns['avg_pre_fill_toxicity']:.3f}" if patterns['avg_pre_fill_toxicity'] else "  Avg pre_fill_toxicity: N/A")
        print(f"  Max pre_fill_toxicity: {patterns['max_pre_fill_toxicity']:.3f}" if patterns['max_pre_fill_toxicity'] else "  Max pre_fill_toxicity: N/A")
        print(f"  Avg markout: {patterns['avg_markout']:.1f} bps" if patterns['avg_markout'] else "  Avg markout: N/A")
        print(f"  Worst markout: {patterns['worst_markout']:.1f} bps" if patterns['worst_markout'] else "  Worst markout: N/A")
    
    # Top 10 worst fills
    print("\n🔥 TOP 10 WORST TOXIC FILLS:")
    print("-" * 100)
    print(f"{'TID':<15} {'Side':<6} {'Markout':>10} {'PreFill':>10} {'CP_Prob':>10} {'Warnings':>10} {'Momentum':>10}")
    print("-" * 100)
    
    sorted_toxic = sorted(audit['toxic_fills'], key=lambda x: x['markout_10s_bps'])[:10]
    for fill in sorted_toxic:
        markout = fill['markout_10s_bps']
        pre_fill = fill.get('pre_fill_toxicity')
        cp = fill.get('cp_prob')
        momentum = fill.get('momentum_bps')
        
        pre_fill_str = f"{pre_fill:.3f}" if pre_fill is not None else "N/A"
        cp_str = f"{cp:.3f}" if cp is not None else "N/A"
        momentum_str = f"{momentum:.1f}" if momentum is not None else "N/A"
        
        print(f"{fill['tid']:<15} {fill['side']:<6} {markout:>10.1f} {pre_fill_str:>10} {cp_str:>10} {fill['n_warnings']:>10} {momentum_str:>10}")
    
    print("-" * 100)
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    
    if audit['unwarned_rate'] > 0.3:
        print("  ⚠️  >30% of toxic fills are unwarned. Consider:")
        print("      - Lowering warning thresholds")
        print("      - Adding new signals that capture the unwarned patterns")
    
    if patterns.get('max_pre_fill_toxicity') and patterns['max_pre_fill_toxicity'] < 0.3:
        print("  ⚠️  Even unwarned fills have low pre_fill_toxicity. Consider:")
        print("      - Lowering pre_fill_toxicity threshold from 0.3 to 0.2")
        print("      - Investigating what features are missing from the classifier")
    
    avg_momentum = patterns.get('avg_momentum_bps')
    if avg_momentum is not None and abs(avg_momentum) > 5:
        direction = "negative" if avg_momentum < 0 else "positive"
        print(f"  ⚠️  Unwarned fills have {direction} momentum bias ({avg_momentum:.1f} bps). Consider:")
        print(f"      - Increasing {'falling_knife_score' if avg_momentum < 0 else 'rising_knife_score'} sensitivity")


def export_unwarned_csv(unwarned_fills: list[dict], output_path: str):
    """Export unwarned fills to CSV for deeper analysis."""
    import csv
    
    if not unwarned_fills:
        print(f"No unwarned fills to export.")
        return
    
    fieldnames = [
        'tid', 'timestamp_ms', 'side', 'price', 'size', 'markout_10s_bps',
        'pre_fill_toxicity', 'cp_prob', 'hawkes_p_cluster', 'continuation_p',
        'is_toxic_regime', 'falling_knife_score', 'rising_knife_score',
        'momentum_bps', 'volatility'
    ]
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for fill in unwarned_fills:
            writer.writerow(fill)
    
    print(f"Exported {len(unwarned_fills)} unwarned fills to {output_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python toxic_fill_audit.py fills_with_signals.json [toxic_threshold_bps]")
        print("\nExample: python toxic_fill_audit.py fills_with_signals.json -5.0")
        sys.exit(1)
    
    fills_path = sys.argv[1]
    toxic_threshold = float(sys.argv[2]) if len(sys.argv) > 2 else -5.0
    
    print(f"Loading fills from {fills_path}...")
    fills = load_fills(fills_path)
    print(f"Loaded {len(fills)} fills")
    
    print(f"\nAuditing toxic fills (threshold: {toxic_threshold} bps)...")
    audit = audit_toxic_fills(fills, toxic_threshold)
    patterns = analyze_unwarned_patterns(audit['unwarned_fills'])
    
    print_report(audit, patterns)
    
    # Export unwarned fills for deeper analysis
    if audit['unwarned_fills']:
        output_dir = Path(fills_path).parent
        csv_path = output_dir / "unwarned_toxic_fills.csv"
        export_unwarned_csv(audit['unwarned_fills'], str(csv_path))


if __name__ == '__main__':
    main()
