#!/usr/bin/env python3
"""Check PnL accounting: dashboard pnl.total vs cumPnl from fills."""
import json, os

SNAP_DIR = "/home/trudumb/hyper_make/logs/side_by_side_HYPE_2026-03-02_14-25-51/snapshots"

for n in [14, 24, 30]:
    for mode in ["paper", "live"]:
        files = [f for f in os.listdir(SNAP_DIR) if f.startswith(f"{mode}_{n}_")]
        if not files:
            continue
        with open(os.path.join(SNAP_DIR, files[0])) as f:
            d = json.load(f)
        fills = d.get("fills", [])
        pnl = d["pnl"]
        last_cum = fills[-1]["cumPnl"] if fills else 0
        print(f"{mode}_{n}: dashboard_total={pnl['total']:.6f}, fills_cumPnl={last_cum:.6f}, spread_cap={pnl['spread_capture']:.6f}, fees={pnl['fees']:.6f}, inv_cost={pnl['inventory_cost']:.6f}, AS={pnl['adverse_selection']:.6f}")
        # Spread per fill
        if fills:
            avg_pnl = sum(f["pnl"] for f in fills) / len(fills)
            print(f"  fills={len(fills)}, avg_pnl_per_fill={avg_pnl:.6f}, total_fill_pnl={sum(f['pnl'] for f in fills):.6f}")
        print()
