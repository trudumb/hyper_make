#!/usr/bin/env python3
"""
Calibration Report Generator

Analyzes paper trading session data and generates calibration metrics.
Follows the Small Fish Strategy validation framework:
- Brier score decomposition
- Information ratio
- Calibration curves
- Conditional calibration by regime

Usage:
    python scripts/analysis/calibration_report.py <session_dir>
    python scripts/analysis/calibration_report.py logs/paper_trading_BTC_2026-01-22_15-30-00

Output:
    - calibration_report.json     - Machine-readable metrics
    - calibration_report.md       - Human-readable report
"""

import json
import sys
import os
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
import math

# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class BrierDecomposition:
    """Brier score decomposition following Small Fish methodology."""
    brier_score: float = 0.0
    reliability: float = 0.0  # Lower is better
    resolution: float = 0.0   # Higher is better
    uncertainty: float = 0.0  # Base rate variance
    information_ratio: float = 0.0  # Resolution / Uncertainty
    n_samples: int = 0

    def is_useful(self) -> bool:
        """IR > 1.0 means model adds value."""
        return self.information_ratio > 1.0 and self.n_samples >= 100


@dataclass
class CalibrationPoint:
    """Point on calibration curve."""
    mean_predicted: float
    realized_frequency: float
    count: int
    std_error: float


@dataclass
class CalibrationCurve:
    """Calibration curve for a prediction type."""
    points: List[CalibrationPoint] = field(default_factory=list)
    calibration_error: float = 0.0  # Mean absolute deviation from diagonal
    max_calibration_error: float = 0.0  # Worst bin


@dataclass
class RegimeMetrics:
    """Metrics for a specific regime."""
    regime: str
    n_samples: int
    fill_rate: float
    avg_adverse_selection: float
    avg_spread_capture: float
    win_rate: float


@dataclass
class CalibrationReport:
    """Complete calibration report."""
    session_dir: str
    timestamp: str
    duration_secs: int = 0

    # Overall metrics
    total_predictions: int = 0
    total_fills: int = 0
    total_quote_cycles: int = 0

    # Fill prediction calibration
    fill_prediction: BrierDecomposition = field(default_factory=BrierDecomposition)
    fill_calibration_curve: CalibrationCurve = field(default_factory=CalibrationCurve)

    # Adverse selection prediction calibration
    as_prediction: BrierDecomposition = field(default_factory=BrierDecomposition)

    # By-regime breakdown
    regime_metrics: List[RegimeMetrics] = field(default_factory=list)

    # PnL Attribution
    total_pnl: float = 0.0
    spread_capture: float = 0.0
    adverse_selection_cost: float = 0.0
    fee_cost: float = 0.0

    # Health indicators
    is_well_calibrated: bool = False
    has_sufficient_samples: bool = False
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


# ============================================================================
# Calibration Computation
# ============================================================================

def compute_brier_decomposition(
    predictions: List[float],
    outcomes: List[bool],
    num_bins: int = 10
) -> BrierDecomposition:
    """
    Compute Brier score decomposition (Murphy 1973).

    Brier Score = Reliability - Resolution + Uncertainty

    - Reliability: How well calibrated (lower better)
    - Resolution: How well discriminates (higher better)
    - Uncertainty: Base rate variance (not controllable)

    Information Ratio = Resolution / Uncertainty
    IR > 1.0 means model adds value over base rate
    """
    n = len(predictions)
    if n == 0:
        return BrierDecomposition()

    # Base rate
    o_bar = sum(1.0 if o else 0.0 for o in outcomes) / n

    # Create bins
    bins: List[List[Tuple[float, bool]]] = [[] for _ in range(num_bins)]
    for p, o in zip(predictions, outcomes):
        bin_idx = min(int(p * num_bins), num_bins - 1)
        bins[bin_idx].append((p, o))

    # Compute components
    reliability = 0.0
    resolution = 0.0

    for bin_data in bins:
        if not bin_data:
            continue

        n_k = len(bin_data)
        p_bar_k = sum(p for p, _ in bin_data) / n_k
        o_bar_k = sum(1.0 if o else 0.0 for _, o in bin_data) / n_k

        reliability += n_k * (p_bar_k - o_bar_k) ** 2
        resolution += n_k * (o_bar_k - o_bar) ** 2

    reliability /= n
    resolution /= n
    uncertainty = o_bar * (1.0 - o_bar)
    brier_score = reliability - resolution + uncertainty

    information_ratio = resolution / uncertainty if uncertainty > 0 else 0.0

    return BrierDecomposition(
        brier_score=brier_score,
        reliability=reliability,
        resolution=resolution,
        uncertainty=uncertainty,
        information_ratio=information_ratio,
        n_samples=n
    )


def compute_calibration_curve(
    predictions: List[float],
    outcomes: List[bool],
    num_bins: int = 10
) -> CalibrationCurve:
    """Build calibration curve from predictions and outcomes."""
    bins: List[List[Tuple[float, bool]]] = [[] for _ in range(num_bins)]

    for p, o in zip(predictions, outcomes):
        bin_idx = min(int(p * num_bins), num_bins - 1)
        bins[bin_idx].append((p, o))

    points = []
    total_error = 0.0
    max_error = 0.0
    total_count = 0

    for bin_data in bins:
        if not bin_data:
            continue

        count = len(bin_data)
        mean_pred = sum(p for p, _ in bin_data) / count
        realized = sum(1.0 if o else 0.0 for _, o in bin_data) / count

        # Standard error of proportion
        std_error = math.sqrt(realized * (1 - realized) / count) if count > 0 else 0

        error = abs(mean_pred - realized)
        total_error += error * count
        max_error = max(max_error, error)
        total_count += count

        points.append(CalibrationPoint(
            mean_predicted=mean_pred,
            realized_frequency=realized,
            count=count,
            std_error=std_error
        ))

    calibration_error = total_error / total_count if total_count > 0 else 0

    return CalibrationCurve(
        points=points,
        calibration_error=calibration_error,
        max_calibration_error=max_error
    )


# ============================================================================
# Log Parsing
# ============================================================================

def parse_predictions(session_dir: Path) -> List[Dict]:
    """Parse predictions.jsonl file."""
    pred_file = session_dir / "predictions.jsonl"
    if not pred_file.exists():
        return []

    predictions = []
    with open(pred_file, 'r') as f:
        for line in f:
            try:
                predictions.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return predictions


def parse_fills(session_dir: Path) -> List[Dict]:
    """Parse fills.jsonl file."""
    fills_file = session_dir / "fills.jsonl"
    if not fills_file.exists():
        return []

    fills = []
    with open(fills_file, 'r') as f:
        for line in f:
            try:
                fills.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return fills


def parse_log_stats(session_dir: Path) -> Dict:
    """Parse basic stats from log file."""
    log_file = session_dir / "paper_trader.log"
    stats = {
        'quote_cycles': 0,
        'errors': 0,
        'warnings': 0,
        'sim_fills': 0,
    }

    if not log_file.exists():
        return stats

    with open(log_file, 'r') as f:
        for line in f:
            if 'Quote cycle' in line:
                stats['quote_cycles'] += 1
            elif 'ERROR' in line:
                stats['errors'] += 1
            elif 'WARN' in line:
                stats['warnings'] += 1
            elif '[SIM] Fill' in line:
                stats['sim_fills'] += 1

    return stats


# ============================================================================
# Report Generation
# ============================================================================

def generate_report(session_dir: str) -> CalibrationReport:
    """Generate calibration report from session data."""
    session_path = Path(session_dir)

    report = CalibrationReport(
        session_dir=session_dir,
        timestamp=datetime.now().isoformat()
    )

    # Parse data
    predictions = parse_predictions(session_path)
    fills = parse_fills(session_path)
    log_stats = parse_log_stats(session_path)

    report.total_predictions = len(predictions)
    report.total_fills = len(fills)
    report.total_quote_cycles = log_stats['quote_cycles']

    # Check sample sufficiency
    report.has_sufficient_samples = report.total_predictions >= 200

    if not report.has_sufficient_samples:
        report.issues.append(
            f"Insufficient samples ({report.total_predictions} < 200 required)"
        )
        report.recommendations.append(
            "Run longer session (3600s recommended) for statistical significance"
        )

    # Extract fill predictions and outcomes
    if predictions:
        fill_predictions = []
        fill_outcomes = []

        for pred in predictions:
            if 'fill_probability' in pred and 'was_filled' in pred:
                fill_predictions.append(pred['fill_probability'])
                fill_outcomes.append(pred['was_filled'])

        if fill_predictions:
            report.fill_prediction = compute_brier_decomposition(
                fill_predictions, fill_outcomes
            )
            report.fill_calibration_curve = compute_calibration_curve(
                fill_predictions, fill_outcomes
            )

            # Check calibration quality
            if report.fill_prediction.information_ratio < 1.0:
                report.issues.append(
                    f"Fill prediction IR ({report.fill_prediction.information_ratio:.2f}) < 1.0 - model adds noise"
                )
                report.recommendations.append(
                    "Consider simplifying fill prediction model or improving features"
                )

    # Extract adverse selection predictions
    if predictions:
        as_predictions = []
        as_outcomes = []

        for pred in predictions:
            if 'adverse_selection_prob' in pred and 'was_adverse' in pred:
                as_predictions.append(pred['adverse_selection_prob'])
                as_outcomes.append(pred['was_adverse'])

        if as_predictions:
            report.as_prediction = compute_brier_decomposition(
                as_predictions, as_outcomes
            )

    # PnL attribution from fills
    if fills:
        for fill in fills:
            if 'spread_capture' in fill:
                report.spread_capture += fill['spread_capture']
            if 'adverse_selection' in fill:
                report.adverse_selection_cost += fill['adverse_selection']
            if 'fee_cost' in fill:
                report.fee_cost += fill['fee_cost']

        report.total_pnl = (
            report.spread_capture
            + report.adverse_selection_cost
            - report.fee_cost
        )

    # Overall calibration status
    report.is_well_calibrated = (
        report.has_sufficient_samples
        and report.fill_prediction.information_ratio >= 1.0
        and len(report.issues) == 0
    )

    # Add error warnings
    if log_stats['errors'] > 0:
        report.issues.append(f"{log_stats['errors']} errors detected in session")

    return report


def format_markdown_report(report: CalibrationReport) -> str:
    """Format report as markdown."""
    md = []
    md.append("# Calibration Report")
    md.append("")
    md.append(f"**Session:** `{report.session_dir}`")
    md.append(f"**Generated:** {report.timestamp}")
    md.append("")

    # Summary
    md.append("## Summary")
    md.append("")
    status = "WELL CALIBRATED" if report.is_well_calibrated else "NEEDS ATTENTION"
    status_emoji = "✅" if report.is_well_calibrated else "⚠️"
    md.append(f"**Status:** {status_emoji} {status}")
    md.append("")

    md.append("| Metric | Value |")
    md.append("|--------|-------|")
    md.append(f"| Total Predictions | {report.total_predictions} |")
    md.append(f"| Total Fills | {report.total_fills} |")
    md.append(f"| Quote Cycles | {report.total_quote_cycles} |")
    md.append(f"| Sample Sufficiency | {'✅ Yes' if report.has_sufficient_samples else '❌ No (<200)'} |")
    md.append("")

    # Fill Prediction Calibration
    md.append("## Fill Prediction Calibration")
    md.append("")
    fp = report.fill_prediction
    md.append(f"**Information Ratio:** {fp.information_ratio:.3f} "
              f"({'✅ Useful' if fp.information_ratio > 1.0 else '❌ Adds noise'})")
    md.append("")
    md.append("| Component | Value | Interpretation |")
    md.append("|-----------|-------|----------------|")
    md.append(f"| Brier Score | {fp.brier_score:.4f} | Overall error (lower better) |")
    md.append(f"| Reliability | {fp.reliability:.4f} | Calibration quality (lower better) |")
    md.append(f"| Resolution | {fp.resolution:.4f} | Discrimination ability (higher better) |")
    md.append(f"| Uncertainty | {fp.uncertainty:.4f} | Base rate variance (not controllable) |")
    md.append(f"| Samples | {fp.n_samples} | — |")
    md.append("")

    # Calibration Curve
    if report.fill_calibration_curve.points:
        md.append("### Calibration Curve")
        md.append("")
        md.append("| Predicted | Realized | Count | Std Error |")
        md.append("|-----------|----------|-------|-----------|")
        for pt in report.fill_calibration_curve.points:
            md.append(f"| {pt.mean_predicted:.2f} | {pt.realized_frequency:.2f} | "
                      f"{pt.count} | ±{pt.std_error:.3f} |")
        md.append("")
        md.append(f"**Calibration Error:** {report.fill_calibration_curve.calibration_error:.4f}")
        md.append(f"**Max Error:** {report.fill_calibration_curve.max_calibration_error:.4f}")
        md.append("")

    # PnL Attribution
    md.append("## PnL Attribution")
    md.append("")
    md.append("| Component | Value |")
    md.append("|-----------|-------|")
    md.append(f"| Spread Capture | ${report.spread_capture:.2f} |")
    md.append(f"| Adverse Selection | ${report.adverse_selection_cost:.2f} |")
    md.append(f"| Fee Cost | -${report.fee_cost:.2f} |")
    md.append(f"| **Total PnL** | **${report.total_pnl:.2f}** |")
    md.append("")

    # Issues and Recommendations
    if report.issues:
        md.append("## Issues")
        md.append("")
        for issue in report.issues:
            md.append(f"- ⚠️ {issue}")
        md.append("")

    if report.recommendations:
        md.append("## Recommendations")
        md.append("")
        for rec in report.recommendations:
            md.append(f"- 💡 {rec}")
        md.append("")

    # Small Fish Validation Checklist
    md.append("## Small Fish Validation Checklist")
    md.append("")
    md.append("| Requirement | Status |")
    md.append("|-------------|--------|")
    md.append(f"| 200+ independent samples | {'✅' if report.has_sufficient_samples else '❌'} |")
    md.append(f"| Fill prediction IR > 1.0 | "
              f"{'✅' if report.fill_prediction.information_ratio > 1.0 else '❌'} |")
    md.append(f"| Calibration error < 0.1 | "
              f"{'✅' if report.fill_calibration_curve.calibration_error < 0.1 else '❌'} |")
    md.append(f"| Positive PnL | {'✅' if report.total_pnl > 0 else '❌'} |")
    md.append("")

    return "\n".join(md)


# ============================================================================
# Main
# ============================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python calibration_report.py <session_dir>")
        print("Example: python calibration_report.py logs/paper_trading_BTC_2026-01-22_15-30-00")
        sys.exit(1)

    session_dir = sys.argv[1]

    if not os.path.isdir(session_dir):
        print(f"Error: Session directory not found: {session_dir}")
        sys.exit(1)

    print(f"Generating calibration report for: {session_dir}")

    report = generate_report(session_dir)

    # Save JSON report
    json_path = os.path.join(session_dir, "calibration_report.json")
    with open(json_path, 'w') as f:
        json.dump(asdict(report), f, indent=2)
    print(f"JSON report saved: {json_path}")

    # Save Markdown report
    md_content = format_markdown_report(report)
    md_path = os.path.join(session_dir, "calibration_report.md")
    with open(md_path, 'w') as f:
        f.write(md_content)
    print(f"Markdown report saved: {md_path}")

    # Print summary
    print("")
    print("=" * 60)
    print("CALIBRATION SUMMARY")
    print("=" * 60)
    print(f"Status: {'WELL CALIBRATED' if report.is_well_calibrated else 'NEEDS ATTENTION'}")
    print(f"Samples: {report.total_predictions} (need 200+)")
    print(f"Fill Prediction IR: {report.fill_prediction.information_ratio:.3f} (need >1.0)")
    print(f"Total PnL: ${report.total_pnl:.2f}")
    print("")

    if report.issues:
        print("Issues:")
        for issue in report.issues:
            print(f"  - {issue}")
        print("")

    if report.recommendations:
        print("Recommendations:")
        for rec in report.recommendations:
            print(f"  - {rec}")


if __name__ == "__main__":
    main()
