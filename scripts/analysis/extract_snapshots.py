#!/usr/bin/env python3
"""Extract key metrics from side-by-side snapshot pairs."""

import json
import os
import sys

SNAP_DIR = "/home/trudumb/hyper_make/logs/side_by_side_HYPE_2026-03-02_14-25-51/snapshots"

def extract_metrics(filepath):
    with open(filepath) as f:
        d = json.load(f)

    fills = d.get("fills", [])
    sig_hist = d.get("signal_history", [])
    dec_hist = d.get("decision_history", [])

    return {
        "timestamp_ms": d.get("timestamp_ms", 0),
        "mid": d["quotes"]["mid"],
        "spread_bps": d["quotes"]["spread_bps"],
        "inventory": d["quotes"]["inventory"],
        "regime": d["quotes"]["regime"],
        "kappa": d["quotes"]["kappa"],
        "gamma": d["quotes"]["gamma"],
        "fill_prob": d["quotes"]["fill_prob"],
        "adverse_prob": d["quotes"]["adverse_prob"],
        "pnl_total": d["pnl"]["total"],
        "pnl_spread_capture": d["pnl"]["spread_capture"],
        "pnl_adverse_selection": d["pnl"]["adverse_selection"],
        "pnl_inventory_cost": d["pnl"]["inventory_cost"],
        "pnl_fees": d["pnl"]["fees"],
        "fills_count": len(fills),
        "glft_spread_bps": d["pipeline"]["glft_spread_bps"],
        "risk_premium_bps": d["pipeline"]["risk_premium_bps"],
        "total_spread_bps": d["pipeline"]["total_spread_bps"],
        "bid_depth_bps": d["pipeline"]["bid_depth_bps"],
        "ask_depth_bps": d["pipeline"]["ask_depth_bps"],
        "depth_skew_bps": d["pipeline"]["depth_skew_bps"],
        "sigma_pct": d["pipeline"]["sigma_pct"],
        "toxicity": d["pipeline"]["toxicity"],
        "drift_bps_s": d["pipeline"]["drift_bps_s"],
        "microprice": d["pipeline"]["microprice"],
        "data_age_ms": d["pipeline"]["data_age_ms"],
        "kelly_fraction": d["pipeline"]["kelly_fraction"],
        "max_size": d["pipeline"]["max_size"],
        "bid_levels": d["pipeline"]["bid_levels"],
        "ask_levels": d["pipeline"]["ask_levels"],
        "position_pct": d["risk_summary"]["position_pct"],
        "drawdown_pct": d["risk_summary"]["drawdown_pct"],
        "kill_switch_headroom_pct": d["risk_summary"]["kill_switch_headroom_pct"],
        "cascade_severity": d["risk_summary"]["cascade_severity"],
        "kill_switch_triggered": d["risk_summary"]["kill_switch_triggered"],
        "position_velocity_1m": d["risk_summary"]["position_velocity_1m"],
        "price_velocity_1s": d["risk_summary"]["price_velocity_1s"],
        "regime_current": d["regime"]["current"],
        "regime_quiet": d["regime"]["probabilities"]["quiet"],
        "regime_trending": d["regime"]["probabilities"]["trending"],
        "regime_volatile": d["regime"]["probabilities"]["volatile"],
        "regime_cascade": d["regime"]["probabilities"]["cascade"],
        "calibration_fill_n": d["calibration"]["fill"]["n_samples"],
        "calibration_fill_brier": d["calibration"]["fill"]["brier_score"],
        "calibration_as_n": d["calibration"]["adverse_selection"]["n_samples"],
        "signal_count": len(sig_hist),
        "decision_count": len(dec_hist),
        "last_signal_kappa": sig_hist[-1]["kappa"] if sig_hist else None,
        "last_signal_sigma": sig_hist[-1]["sigma"] if sig_hist else None,
        "last_signal_momentum_bps": sig_hist[-1]["momentum_bps"] if sig_hist else None,
        "last_signal_flow_imbalance": sig_hist[-1]["flow_imbalance"] if sig_hist else None,
        "last_signal_jump_ratio": sig_hist[-1]["jump_ratio"] if sig_hist else None,
        "last_decision_bid_spread": dec_hist[-1]["bid_spread_bps"] if dec_hist else None,
        "last_decision_ask_spread": dec_hist[-1]["ask_spread_bps"] if dec_hist else None,
        "last_decision_regime": dec_hist[-1]["regime"] if dec_hist else None,
        "quote_history_count": len(d.get("quote_history", [])),
        "price_history_count": len(d.get("price_history", [])),
    }

# Collect all snapshot pairs
results = []
for n in range(1, 31):
    paper_files = [f for f in os.listdir(SNAP_DIR) if f.startswith(f"paper_{n}_")]
    live_files = [f for f in os.listdir(SNAP_DIR) if f.startswith(f"live_{n}_")]

    if not paper_files or not live_files:
        continue

    paper_path = os.path.join(SNAP_DIR, paper_files[0])
    live_path = os.path.join(SNAP_DIR, live_files[0])

    paper_m = extract_metrics(paper_path)
    live_m = extract_metrics(live_path)

    results.append({
        "snapshot": n,
        "paper": paper_m,
        "live": live_m
    })

# Also extract finals (skip if empty)
for tag in ["paper_final", "live_final"]:
    fp = os.path.join(SNAP_DIR, f"{tag}.json")
    if os.path.exists(fp) and os.path.getsize(fp) > 0:
        m = extract_metrics(fp)
        results.append({"snapshot": f"{tag}", "data": m})

# Output
print(json.dumps(results, indent=2))
