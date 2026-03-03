#!/usr/bin/env python3
"""
Side-by-Side Paper vs Live Session Comparison

Usage: python3 scripts/analysis/compare_sessions.py <side_by_side_dir>

Reads logs and dashboard API snapshots from a side-by-side validation run
and produces a comparison report highlighting where paper and live diverge.

Expected directory structure:
  logs/side_by_side_ASSET_TIMESTAMP/
    paper/paper_trader.log
    live/live_trader.log
    snapshots/paper_*.json
    snapshots/live_*.json
"""

import json
import os
import re
import sys
from pathlib import Path


def load_latest_snapshot(snapshot_dir: Path, prefix: str) -> dict:
    """Load the most recent dashboard snapshot for a given prefix (paper/live)."""
    # Prefer _final snapshot
    final = snapshot_dir / f"{prefix}_final.json"
    if final.exists() and final.stat().st_size > 0:
        try:
            return json.loads(final.read_text())
        except json.JSONDecodeError:
            pass

    # Fall back to highest-numbered snapshot
    candidates = sorted(
        snapshot_dir.glob(f"{prefix}_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for c in candidates:
        if c.stat().st_size > 0:
            try:
                return json.loads(c.read_text())
            except json.JSONDecodeError:
                continue
    return {}


def count_log_pattern(log_path: Path, pattern: str) -> int:
    """Count occurrences of a regex pattern in a log file."""
    if not log_path.exists():
        return 0
    count = 0
    regex = re.compile(pattern)
    with open(log_path) as f:
        for line in f:
            if regex.search(line):
                count += 1
    return count


def extract_kappa_sigma(log_path: Path) -> dict:
    """Extract latest kappa/sigma estimates from log."""
    result = {"kappa": None, "sigma": None}
    if not log_path.exists():
        return result

    kappa_re = re.compile(r"kappa[=: ]+([0-9.]+)")
    sigma_re = re.compile(r"sigma[=: ]+([0-9.]+)")

    with open(log_path) as f:
        for line in f:
            m = kappa_re.search(line)
            if m:
                result["kappa"] = float(m.group(1))
            m = sigma_re.search(line)
            if m:
                result["sigma"] = float(m.group(1))
    return result


def extract_dashboard_metrics(data: dict) -> dict:
    """Extract key metrics from dashboard API JSON."""
    metrics = {
        "fill_count": None,
        "avg_spread_bps": None,
        "spread_capture_usd": None,
        "adverse_selection_usd": None,
        "fees_usd": None,
        "net_pnl_usd": None,
        "position": None,
    }

    if not data:
        return metrics

    # Try common dashboard JSON structures
    pnl = data.get("pnl", data.get("pnl_attribution", {}))
    if isinstance(pnl, dict):
        metrics["fill_count"] = pnl.get("fill_count", pnl.get("total_fills"))
        metrics["spread_capture_usd"] = pnl.get(
            "spread_capture", pnl.get("spread_capture_usd")
        )
        metrics["adverse_selection_usd"] = pnl.get(
            "adverse_selection", pnl.get("adverse_selection_usd")
        )
        metrics["fees_usd"] = pnl.get("total_fees", pnl.get("fees_usd"))
        metrics["net_pnl_usd"] = pnl.get("net_pnl", pnl.get("total_pnl"))

    quotes = data.get("quotes", data.get("quoting", {}))
    if isinstance(quotes, dict):
        metrics["avg_spread_bps"] = quotes.get(
            "avg_spread_bps", quotes.get("spread_bps")
        )

    state = data.get("state", data.get("position", {}))
    if isinstance(state, dict):
        metrics["position"] = state.get("position", state.get("inventory"))

    return metrics


def format_val(val, fmt=".2f", suffix=""):
    """Format a value or return 'N/A'."""
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return f"{val:{fmt}}{suffix}"
    return f"{val}{suffix}"


def compute_edge_per_fill(metrics: dict) -> str:
    """Compute edge per fill in bps if we have the data."""
    pnl = metrics.get("net_pnl_usd")
    fills = metrics.get("fill_count")
    if pnl is not None and fills is not None and fills > 0:
        # This is a rough estimate; proper edge/fill needs notional
        edge = pnl / fills
        return f"{edge:.4f} $/fill"
    return "N/A"


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <side_by_side_dir>")
        sys.exit(1)

    base_dir = Path(sys.argv[1])
    if not base_dir.exists():
        print(f"Error: {base_dir} does not exist")
        sys.exit(1)

    paper_log = base_dir / "paper" / "paper_trader.log"
    live_log = base_dir / "live" / "live_trader.log"
    snapshot_dir = base_dir / "snapshots"

    # Load dashboard snapshots
    paper_snap = load_latest_snapshot(snapshot_dir, "paper")
    live_snap = load_latest_snapshot(snapshot_dir, "live")

    paper_metrics = extract_dashboard_metrics(paper_snap)
    live_metrics = extract_dashboard_metrics(live_snap)

    # Log-based counts as fallback
    paper_fill_count = paper_metrics["fill_count"]
    if paper_fill_count is None:
        paper_fill_count = count_log_pattern(paper_log, r"\[SIM\] Fill")
        paper_metrics["fill_count"] = paper_fill_count

    live_fill_count = live_metrics["fill_count"]
    if live_fill_count is None:
        live_fill_count = count_log_pattern(live_log, r"Trades processed")
        live_metrics["fill_count"] = live_fill_count

    paper_errors = count_log_pattern(paper_log, r"ERROR")
    live_errors = count_log_pattern(live_log, r"ERROR")

    paper_kappa_sigma = extract_kappa_sigma(paper_log)
    live_kappa_sigma = extract_kappa_sigma(live_log)

    # Duration from log timestamps
    paper_lines = count_log_pattern(paper_log, r".")
    live_lines = count_log_pattern(live_log, r".")

    # Compute divergence
    def divergence(paper_val, live_val):
        if paper_val is None or live_val is None:
            return "N/A"
        if isinstance(paper_val, (int, float)) and isinstance(live_val, (int, float)):
            if live_val == 0 and paper_val == 0:
                return "0"
            if live_val == 0:
                return "inf"
            ratio = paper_val / live_val
            return f"{ratio:.2f}x"
        return "N/A"

    # Print report
    w = 20  # column width
    print("")
    print("=" * 72)
    print("  SIDE-BY-SIDE PAPER vs LIVE COMPARISON")
    print("=" * 72)
    print(f"  Directory: {base_dir}")
    print(f"  Paper log: {'found' if paper_log.exists() else 'MISSING'}")
    print(f"  Live log:  {'found' if live_log.exists() else 'MISSING'}")
    snap_count = len(list(snapshot_dir.glob("*.json"))) if snapshot_dir.exists() else 0
    print(f"  Snapshots: {snap_count}")
    print("")

    header = f"{'Metric':<25} {'Paper':>{w}} {'Live':>{w}} {'Divergence':>{w}}"
    print(header)
    print("-" * len(header))

    rows = [
        (
            "Fill count",
            format_val(paper_metrics["fill_count"], ".0f"),
            format_val(live_metrics["fill_count"], ".0f"),
            divergence(paper_metrics["fill_count"], live_metrics["fill_count"]),
        ),
        (
            "Avg spread (bps)",
            format_val(paper_metrics["avg_spread_bps"], ".1f"),
            format_val(live_metrics["avg_spread_bps"], ".1f"),
            divergence(
                paper_metrics["avg_spread_bps"], live_metrics["avg_spread_bps"]
            ),
        ),
        (
            "Spread capture ($)",
            format_val(paper_metrics["spread_capture_usd"], ".4f"),
            format_val(live_metrics["spread_capture_usd"], ".4f"),
            divergence(
                paper_metrics["spread_capture_usd"],
                live_metrics["spread_capture_usd"],
            ),
        ),
        (
            "Adverse selection ($)",
            format_val(paper_metrics["adverse_selection_usd"], ".4f"),
            format_val(live_metrics["adverse_selection_usd"], ".4f"),
            divergence(
                paper_metrics["adverse_selection_usd"],
                live_metrics["adverse_selection_usd"],
            ),
        ),
        (
            "Fees ($)",
            format_val(paper_metrics["fees_usd"], ".4f"),
            format_val(live_metrics["fees_usd"], ".4f"),
            divergence(paper_metrics["fees_usd"], live_metrics["fees_usd"]),
        ),
        (
            "Net PnL ($)",
            format_val(paper_metrics["net_pnl_usd"], ".4f"),
            format_val(live_metrics["net_pnl_usd"], ".4f"),
            divergence(paper_metrics["net_pnl_usd"], live_metrics["net_pnl_usd"]),
        ),
        (
            "Edge per fill",
            compute_edge_per_fill(paper_metrics),
            compute_edge_per_fill(live_metrics),
            "",
        ),
        (
            "Position",
            format_val(paper_metrics["position"], ".4f"),
            format_val(live_metrics["position"], ".4f"),
            "",
        ),
        (
            "Errors",
            str(paper_errors),
            str(live_errors),
            "",
        ),
        (
            "Kappa (latest)",
            format_val(paper_kappa_sigma["kappa"], ".1f"),
            format_val(live_kappa_sigma["kappa"], ".1f"),
            divergence(paper_kappa_sigma["kappa"], live_kappa_sigma["kappa"]),
        ),
        (
            "Sigma (latest)",
            format_val(paper_kappa_sigma["sigma"], ".6f"),
            format_val(live_kappa_sigma["sigma"], ".6f"),
            divergence(paper_kappa_sigma["sigma"], live_kappa_sigma["sigma"]),
        ),
    ]

    for label, paper_val, live_val, div in rows:
        print(f"{label:<25} {paper_val:>{w}} {live_val:>{w}} {div:>{w}}")

    print("")

    # Diagnostic interpretation
    print("DIAGNOSTICS")
    print("-" * 40)

    p_fills = paper_metrics["fill_count"]
    l_fills = live_metrics["fill_count"]
    if (
        p_fills is not None
        and l_fills is not None
        and l_fills > 0
        and p_fills > l_fills * 2
    ):
        print(
            f"  [!] Paper fill rate is {p_fills / l_fills:.1f}x live — "
            "fill simulator is likely too optimistic"
        )
        print("      Consider reducing touch_fill_probability or enabling book depth")

    p_pnl = paper_metrics["net_pnl_usd"]
    l_pnl = live_metrics["net_pnl_usd"]
    if p_pnl is not None and l_pnl is not None:
        if p_pnl > 0 and l_pnl < 0:
            print(
                "  [!] Paper profitable but live negative — "
                "fill simulation likely overstates favorable fills"
            )
        elif p_pnl > 0 and l_pnl > 0 and p_pnl > l_pnl * 3:
            print(
                f"  [!] Paper PnL is {p_pnl / l_pnl:.1f}x live — "
                "significant sim optimism remains"
            )
        elif p_pnl > 0 and l_pnl > 0 and p_pnl < l_pnl * 1.5:
            print("  [OK] Paper and live PnL within 1.5x — reasonable calibration")

    p_as = paper_metrics["adverse_selection_usd"]
    l_as = live_metrics["adverse_selection_usd"]
    if p_as is not None and l_as is not None and l_as != 0:
        if abs(p_as) < abs(l_as) * 0.5:
            print(
                "  [!] Paper adverse selection much lower than live — "
                "sim misses toxic flow"
            )

    p_spread = paper_metrics["avg_spread_bps"]
    l_spread = live_metrics["avg_spread_bps"]
    if p_spread is not None and l_spread is not None:
        if abs(p_spread - l_spread) > 5:
            print(
                f"  [!] Spread mismatch: paper={p_spread:.1f} vs live={l_spread:.1f} bps — "
                "config parity issue"
            )
        else:
            print(
                f"  [OK] Spreads match: paper={p_spread:.1f} vs live={l_spread:.1f} bps"
            )

    p_kappa = paper_kappa_sigma["kappa"]
    l_kappa = live_kappa_sigma["kappa"]
    if p_kappa is not None and l_kappa is not None:
        if abs(p_kappa - l_kappa) / max(p_kappa, l_kappa, 1) > 0.3:
            print(
                f"  [!] Kappa divergence: paper={p_kappa:.0f} vs live={l_kappa:.0f} — "
                "different market data processing"
            )
        else:
            print(
                f"  [OK] Kappa aligned: paper={p_kappa:.0f} vs live={l_kappa:.0f}"
            )

    if not any(
        [
            paper_metrics["fill_count"],
            live_metrics["fill_count"],
            paper_metrics["net_pnl_usd"],
            live_metrics["net_pnl_usd"],
        ]
    ):
        print("  [?] No fill/PnL data found — check dashboard API was reachable")
        print("      Paper: http://localhost:9091/api/dashboard")
        print("      Live:  http://localhost:9090/api/dashboard")

    print("")
    print("=" * 72)
    print("")


if __name__ == "__main__":
    main()
