#!/usr/bin/env python3
"""Comprehensive side-by-side analysis of paper vs live trading snapshots."""

import json
import os

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

# Collect all pairs
pairs = []
for n in range(1, 31):
    paper_files = [f for f in os.listdir(SNAP_DIR) if f.startswith(f"paper_{n}_")]
    live_files = [f for f in os.listdir(SNAP_DIR) if f.startswith(f"live_{n}_")]
    if not paper_files or not live_files:
        continue
    p = extract_metrics(os.path.join(SNAP_DIR, paper_files[0]))
    l = extract_metrics(os.path.join(SNAP_DIR, live_files[0]))
    pairs.append((n, p, l))

# ========== TIME SERIES TABLE ==========
print("=" * 140)
print("TIME SERIES: PAPER vs LIVE (30 snapshots, ~60s apart, 14:26 - 14:53 UTC)")
print("=" * 140)

# 1. Fills + PnL
print()
print("--- CUMULATIVE FILLS & PnL ---")
print(f"{'Snap':>4} {'Time':>8} | {'P.Fills':>7} {'L.Fills':>7} {'Ratio':>7} | {'P.PnL':>10} {'L.PnL':>10} {'Delta':>10} | {'P.Spread':>9} {'L.Spread':>9} {'P.Cap':>7} {'L.Cap':>7}")
print("-" * 140)

for n, p, l in pairs:
    ts = p["timestamp_ms"]
    # Convert to HH:MM:SS
    import datetime
    t = datetime.datetime.fromtimestamp(ts / 1000.0)
    time_str = t.strftime("%H:%M:%S")

    pf = p["fills_count"]
    lf = l["fills_count"]
    ratio = f"{pf/lf:.2f}" if lf > 0 else ("inf" if pf > 0 else "n/a")

    pp = p["pnl_total"]
    lp = l["pnl_total"]
    delta = pp - lp

    ps = p["spread_bps"]
    ls = l["spread_bps"]
    psc = p["pnl_spread_capture"]
    lsc = l["pnl_spread_capture"]

    print(f"{n:4d} {time_str:>8} | {pf:7d} {lf:7d} {ratio:>7} | {pp:10.4f} {lp:10.4f} {delta:10.4f} | {ps:9.3f} {ls:9.3f} {psc:7.4f} {lsc:7.4f}")

# 2. Inventory + Position
print()
print("--- INVENTORY & POSITION ---")
print(f"{'Snap':>4} {'Time':>8} | {'P.Inv':>8} {'L.Inv':>8} {'P.Pos%':>8} {'L.Pos%':>8} | {'P.Skew':>8} {'L.Skew':>8} | {'P.DDn%':>8} {'L.DDn%':>8} | {'P.KS_Head':>10} {'L.KS_Head':>10}")
print("-" * 140)

for n, p, l in pairs:
    ts = p["timestamp_ms"]
    t = datetime.datetime.fromtimestamp(ts / 1000.0)
    time_str = t.strftime("%H:%M:%S")
    print(f"{n:4d} {time_str:>8} | {p['inventory']:8.3f} {l['inventory']:8.3f} {p['position_pct']:8.3f} {l['position_pct']:8.3f} | {p['depth_skew_bps']:8.3f} {l['depth_skew_bps']:8.3f} | {p['drawdown_pct']:8.4f} {l['drawdown_pct']:8.4f} | {p['kill_switch_headroom_pct']:10.2f} {l['kill_switch_headroom_pct']:10.2f}")

# 3. Spread decomposition
print()
print("--- SPREAD DECOMPOSITION (bps) ---")
print(f"{'Snap':>4} {'Time':>8} | {'P.GLFT':>8} {'L.GLFT':>8} | {'P.Risk':>8} {'L.Risk':>8} | {'P.Total':>8} {'L.Total':>8} | {'P.BidD':>8} {'L.BidD':>8} {'P.AskD':>8} {'L.AskD':>8}")
print("-" * 140)

for n, p, l in pairs:
    ts = p["timestamp_ms"]
    t = datetime.datetime.fromtimestamp(ts / 1000.0)
    time_str = t.strftime("%H:%M:%S")
    print(f"{n:4d} {time_str:>8} | {p['glft_spread_bps']:8.3f} {l['glft_spread_bps']:8.3f} | {p['risk_premium_bps']:8.3f} {l['risk_premium_bps']:8.3f} | {p['total_spread_bps']:8.3f} {l['total_spread_bps']:8.3f} | {p['bid_depth_bps']:8.3f} {l['bid_depth_bps']:8.3f} {p['ask_depth_bps']:8.3f} {l['ask_depth_bps']:8.3f}")

# 4. Signal values
print()
print("--- SIGNALS & REGIME ---")
print(f"{'Snap':>4} {'Time':>8} | {'P.Regime':>10} {'L.Regime':>10} | {'P.Kappa':>10} {'L.Kappa':>10} | {'P.Sigma':>10} {'L.Sigma':>10} | {'P.Tox':>8} {'L.Tox':>8} | {'P.Mom':>8} {'L.Mom':>8} | {'P.Flow':>8} {'L.Flow':>8}")
print("-" * 140)

for n, p, l in pairs:
    ts = p["timestamp_ms"]
    t = datetime.datetime.fromtimestamp(ts / 1000.0)
    time_str = t.strftime("%H:%M:%S")

    pk = p["last_signal_kappa"] or 0
    lk = l["last_signal_kappa"] or 0
    ps = p["last_signal_sigma"] or 0
    ls = l["last_signal_sigma"] or 0
    pm = p["last_signal_momentum_bps"] or 0
    lm = l["last_signal_momentum_bps"] or 0
    pfl = p["last_signal_flow_imbalance"] or 0
    lfl = l["last_signal_flow_imbalance"] or 0

    print(f"{n:4d} {time_str:>8} | {p['regime_current']:>10} {l['regime_current']:>10} | {pk:10.1f} {lk:10.1f} | {ps:10.7f} {ls:10.7f} | {p['toxicity']:8.4f} {l['toxicity']:8.4f} | {pm:8.2f} {lm:8.2f} | {pfl:8.4f} {lfl:8.4f}")

# 5. Decision spreads (actual bid/ask)
print()
print("--- DECISION SPREADS (bps) ---")
print(f"{'Snap':>4} {'Time':>8} | {'P.BidSprd':>10} {'L.BidSprd':>10} | {'P.AskSprd':>10} {'L.AskSprd':>10} | {'P.FillProb':>10} {'L.FillProb':>10} | {'P.AdvProb':>10} {'L.AdvProb':>10}")
print("-" * 140)

for n, p, l in pairs:
    ts = p["timestamp_ms"]
    t = datetime.datetime.fromtimestamp(ts / 1000.0)
    time_str = t.strftime("%H:%M:%S")

    pbs = p["last_decision_bid_spread"] or 0
    lbs = l["last_decision_bid_spread"] or 0
    pas = p["last_decision_ask_spread"] or 0
    las = l["last_decision_ask_spread"] or 0

    print(f"{n:4d} {time_str:>8} | {pbs:10.3f} {lbs:10.3f} | {pas:10.3f} {las:10.3f} | {p['fill_prob']:10.3f} {l['fill_prob']:10.3f} | {p['adverse_prob']:10.4f} {l['adverse_prob']:10.4f}")

# 6. Event-to-quote pipeline metrics
print()
print("--- EVENT-TO-QUOTE PIPELINE ---")
print(f"{'Snap':>4} {'Time':>8} | {'P.SigN':>7} {'L.SigN':>7} {'Ratio':>7} | {'P.DecN':>7} {'L.DecN':>7} {'Ratio':>7} | {'P.PrcN':>7} {'L.PrcN':>7} {'Ratio':>7} | {'P.Age':>6} {'L.Age':>6} | {'P.QuoN':>6} {'L.QuoN':>6}")
print("-" * 140)

for n, p, l in pairs:
    ts = p["timestamp_ms"]
    t = datetime.datetime.fromtimestamp(ts / 1000.0)
    time_str = t.strftime("%H:%M:%S")

    ps = p["signal_count"]
    ls = l["signal_count"]
    sr = f"{ps/ls:.2f}" if ls > 0 else "n/a"

    pd = p["decision_count"]
    ld = l["decision_count"]
    dr = f"{pd/ld:.2f}" if ld > 0 else "n/a"

    pp = p["price_history_count"]
    lp = l["price_history_count"]
    pr = f"{pp/lp:.2f}" if lp > 0 else "n/a"

    print(f"{n:4d} {time_str:>8} | {ps:7d} {ls:7d} {sr:>7} | {pd:7d} {ld:7d} {dr:>7} | {pp:7d} {lp:7d} {pr:>7} | {p['data_age_ms']:6d} {l['data_age_ms']:6d} | {p['quote_history_count']:6d} {l['quote_history_count']:6d}")

# 7. PnL Decomposition
print()
print("--- PnL DECOMPOSITION ---")
print(f"{'Snap':>4} {'Time':>8} | {'P.SprdCap':>10} {'L.SprdCap':>10} | {'P.AS':>10} {'L.AS':>10} | {'P.InvCost':>10} {'L.InvCost':>10} | {'P.Fees':>10} {'L.Fees':>10} | {'P.Total':>10} {'L.Total':>10}")
print("-" * 140)

for n, p, l in pairs:
    ts = p["timestamp_ms"]
    t = datetime.datetime.fromtimestamp(ts / 1000.0)
    time_str = t.strftime("%H:%M:%S")
    print(f"{n:4d} {time_str:>8} | {p['pnl_spread_capture']:10.4f} {l['pnl_spread_capture']:10.4f} | {p['pnl_adverse_selection']:10.4f} {l['pnl_adverse_selection']:10.4f} | {p['pnl_inventory_cost']:10.4f} {l['pnl_inventory_cost']:10.4f} | {p['pnl_fees']:10.4f} {l['pnl_fees']:10.4f} | {p['pnl_total']:10.4f} {l['pnl_total']:10.4f}")

# 8. Regime probabilities
print()
print("--- REGIME PROBABILITIES ---")
print(f"{'Snap':>4} {'Time':>8} | {'P.Quiet':>8} {'L.Quiet':>8} | {'P.Trend':>8} {'L.Trend':>8} | {'P.Volat':>8} {'L.Volat':>8} | {'P.Casc':>8} {'L.Casc':>8} | {'P.CascSev':>9} {'L.CascSev':>9}")
print("-" * 140)

for n, p, l in pairs:
    ts = p["timestamp_ms"]
    t = datetime.datetime.fromtimestamp(ts / 1000.0)
    time_str = t.strftime("%H:%M:%S")
    print(f"{n:4d} {time_str:>8} | {p['regime_quiet']:8.4f} {l['regime_quiet']:8.4f} | {p['regime_trending']:8.4f} {l['regime_trending']:8.4f} | {p['regime_volatile']:8.4f} {l['regime_volatile']:8.4f} | {p['regime_cascade']:8.4f} {l['regime_cascade']:8.4f} | {p['cascade_severity']:9.4f} {l['cascade_severity']:9.4f}")

# ========== SUMMARY STATISTICS ==========
print()
print("=" * 140)
print("SUMMARY STATISTICS")
print("=" * 140)

last = pairs[-1]
n, p_last, l_last = last
first = pairs[0]
_, p_first, l_first = first

duration_ms = p_last["timestamp_ms"] - p_first["timestamp_ms"]
duration_min = duration_ms / 60000.0

print(f"\nSession duration: {duration_min:.1f} minutes ({duration_ms/1000:.0f}s)")
print(f"Snapshots: {len(pairs)}")

print(f"\n--- Final State (Snapshot 30) ---")
print(f"  Paper fills: {p_last['fills_count']}, Live fills: {l_last['fills_count']}")
if l_last['fills_count'] > 0:
    print(f"  Fill ratio (paper/live): {p_last['fills_count']/l_last['fills_count']:.2f}")
print(f"  Paper PnL total: {p_last['pnl_total']:.6f}, Live PnL total: {l_last['pnl_total']:.6f}")
print(f"  Paper inventory: {p_last['inventory']:.4f}, Live inventory: {l_last['inventory']:.4f}")
print(f"  Paper position%: {p_last['position_pct']:.4f}, Live position%: {l_last['position_pct']:.4f}")
print(f"  Paper spread: {p_last['spread_bps']:.3f} bps, Live spread: {l_last['spread_bps']:.3f} bps")
print(f"  Paper regime: {p_last['regime_current']}, Live regime: {l_last['regime_current']}")

# Compute averages
avg_paper_spread = sum(p["spread_bps"] for _, p, _ in pairs) / len(pairs)
avg_live_spread = sum(l["spread_bps"] for _, _, l in pairs) / len(pairs)
avg_paper_glft = sum(p["glft_spread_bps"] for _, p, _ in pairs) / len(pairs)
avg_live_glft = sum(l["glft_spread_bps"] for _, _, l in pairs) / len(pairs)
avg_paper_risk = sum(p["risk_premium_bps"] for _, p, _ in pairs) / len(pairs)
avg_live_risk = sum(l["risk_premium_bps"] for _, _, l in pairs) / len(pairs)

print(f"\n--- Averages Across Session ---")
print(f"  Avg paper spread: {avg_paper_spread:.3f} bps, Avg live spread: {avg_live_spread:.3f} bps")
print(f"  Avg paper GLFT spread: {avg_paper_glft:.3f} bps, Avg live GLFT spread: {avg_live_glft:.3f} bps")
print(f"  Avg paper risk premium: {avg_paper_risk:.3f} bps, Avg live risk premium: {avg_live_risk:.3f} bps")

# Signal processing throughput
avg_paper_signals = sum(p["signal_count"] for _, p, _ in pairs) / len(pairs)
avg_live_signals = sum(l["signal_count"] for _, _, l in pairs) / len(pairs)
avg_paper_decisions = sum(p["decision_count"] for _, p, _ in pairs) / len(pairs)
avg_live_decisions = sum(l["decision_count"] for _, _, l in pairs) / len(pairs)
avg_paper_prices = sum(p["price_history_count"] for _, p, _ in pairs) / len(pairs)
avg_live_prices = sum(l["price_history_count"] for _, _, l in pairs) / len(pairs)

print(f"\n--- Signal Processing Throughput (cumulative counts in dashboard) ---")
print(f"  Avg signal history size: paper={avg_paper_signals:.0f}, live={avg_live_signals:.0f} (ratio={avg_paper_signals/avg_live_signals:.2f})")
print(f"  Avg decision history size: paper={avg_paper_decisions:.0f}, live={avg_live_decisions:.0f} (ratio={avg_paper_decisions/avg_live_decisions:.2f})")
print(f"  Avg price history size: paper={avg_paper_prices:.0f}, live={avg_live_prices:.0f} (ratio={avg_paper_prices/avg_live_prices:.2f})")

# Find divergence moments
print(f"\n--- DIVERGENCE ANALYSIS ---")

# Spread divergences > 2 bps
spread_divs = []
for n, p, l in pairs:
    diff = abs(p["spread_bps"] - l["spread_bps"])
    if diff > 2.0:
        spread_divs.append((n, p["spread_bps"], l["spread_bps"], diff))

if spread_divs:
    print(f"\n  Spread divergences > 2 bps ({len(spread_divs)} occurrences):")
    for sn, ps, ls, d in spread_divs:
        print(f"    Snap {sn}: paper={ps:.3f}, live={ls:.3f}, diff={d:.3f} bps")
else:
    print(f"\n  No spread divergences > 2 bps")

# Mid price divergences > 5 bps
mid_divs = []
for n, p, l in pairs:
    mid_avg = (p["mid"] + l["mid"]) / 2.0
    diff_bps = abs(p["mid"] - l["mid"]) / mid_avg * 10000
    if diff_bps > 0.5:
        mid_divs.append((n, p["mid"], l["mid"], diff_bps))

if mid_divs:
    print(f"\n  Mid price divergences > 0.5 bps ({len(mid_divs)} occurrences):")
    for sn, pm, lm, d in mid_divs:
        print(f"    Snap {sn}: paper={pm:.4f}, live={lm:.4f}, diff={d:.3f} bps")
else:
    print(f"\n  No mid price divergences > 0.5 bps")

# Inventory divergences
inv_divs = []
for n, p, l in pairs:
    diff = abs(p["inventory"] - l["inventory"])
    if diff > 0.001:
        inv_divs.append((n, p["inventory"], l["inventory"], diff))

if inv_divs:
    print(f"\n  Inventory divergences ({len(inv_divs)} occurrences):")
    for sn, pi, li, d in inv_divs:
        print(f"    Snap {sn}: paper={pi:.4f}, live={li:.4f}, diff={d:.4f}")
else:
    print(f"\n  No inventory divergences (both remain at zero)")

# Fill divergences
fill_divs = []
for n, p, l in pairs:
    if p["fills_count"] != l["fills_count"]:
        fill_divs.append((n, p["fills_count"], l["fills_count"]))

if fill_divs:
    print(f"\n  Fill count divergences ({len(fill_divs)} occurrences):")
    for sn, pf, lf in fill_divs:
        print(f"    Snap {sn}: paper={pf}, live={lf}")
else:
    print(f"\n  No fill count divergences")

# Regime divergences
regime_divs = []
for n, p, l in pairs:
    if p["regime_current"] != l["regime_current"]:
        regime_divs.append((n, p["regime_current"], l["regime_current"]))

if regime_divs:
    print(f"\n  Regime divergences ({len(regime_divs)} occurrences):")
    for sn, pr, lr in regime_divs:
        print(f"    Snap {sn}: paper={pr}, live={lr}")
else:
    print(f"\n  No regime divergences (both always agree)")

# Signal processing rate divergences
print(f"\n--- SIGNAL PROCESSING RATE ---")
for n, p, l in pairs:
    if n in [1, 5, 10, 15, 20, 25, 30]:
        ts = p["timestamp_ms"]
        t = datetime.datetime.fromtimestamp(ts / 1000.0)
        time_str = t.strftime("%H:%M:%S")
        sig_ratio = p["signal_count"] / l["signal_count"] if l["signal_count"] > 0 else 0
        dec_ratio = p["decision_count"] / l["decision_count"] if l["decision_count"] > 0 else 0
        prc_ratio = p["price_history_count"] / l["price_history_count"] if l["price_history_count"] > 0 else 0
        print(f"  Snap {n:2d} ({time_str}): signals {p['signal_count']:4d}/{l['signal_count']:4d} ({sig_ratio:.2f}x), decisions {p['decision_count']:4d}/{l['decision_count']:4d} ({dec_ratio:.2f}x), prices {p['price_history_count']:4d}/{l['price_history_count']:4d} ({prc_ratio:.2f}x)")

# Kappa evolution
print(f"\n--- KAPPA EVOLUTION ---")
for n, p, l in pairs:
    pk = p["last_signal_kappa"] or 0
    lk = l["last_signal_kappa"] or 0
    if n in [1, 5, 10, 15, 20, 25, 30]:
        print(f"  Snap {n:2d}: paper={pk:.1f}, live={lk:.1f}, diff={pk-lk:.1f}")

# Toxicity evolution
print(f"\n--- TOXICITY EVOLUTION ---")
for n, p, l in pairs:
    if n in [1, 5, 10, 15, 20, 25, 30]:
        print(f"  Snap {n:2d}: paper={p['toxicity']:.4f}, live={l['toxicity']:.4f}, diff={p['toxicity']-l['toxicity']:.4f}")

# Fill probability evolution
print(f"\n--- FILL PROBABILITY EVOLUTION ---")
for n, p, l in pairs:
    if n in [1, 5, 10, 15, 20, 25, 30]:
        print(f"  Snap {n:2d}: paper={p['fill_prob']:.2f}%, live={l['fill_prob']:.2f}%, diff={p['fill_prob']-l['fill_prob']:.2f}%")

# bid/ask levels
print(f"\n--- QUOTE LEVELS (bid_levels / ask_levels from pipeline) ---")
for n, p, l in pairs:
    if n in [1, 5, 10, 15, 20, 25, 30]:
        print(f"  Snap {n:2d}: paper bid={p['bid_levels']}/ask={p['ask_levels']}, live bid={l['bid_levels']}/ask={l['ask_levels']}")

print(f"\n--- KILL SWITCH STATUS ---")
any_triggered = False
for n, p, l in pairs:
    if p["kill_switch_triggered"] or l["kill_switch_triggered"]:
        any_triggered = True
        print(f"  Snap {n}: paper={p['kill_switch_triggered']}, live={l['kill_switch_triggered']}")
if not any_triggered:
    print(f"  Kill switch never triggered in either session")

print()
print("=" * 140)
print("END OF ANALYSIS")
print("=" * 140)
