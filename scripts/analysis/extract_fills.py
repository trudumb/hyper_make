#!/usr/bin/env python3
"""Extract fill details from snapshots where fills first appear."""
import json, os

SNAP_DIR = "/home/trudumb/hyper_make/logs/side_by_side_HYPE_2026-03-02_14-25-51/snapshots"

# Read snapshot 10 (first live fill), 14 (first paper fills), 24 (live jump to 9), 30 (final)
for n in [10, 14, 16, 24, 30]:
    for mode in ["paper", "live"]:
        files = [f for f in os.listdir(SNAP_DIR) if f.startswith(f"{mode}_{n}_")]
        if not files:
            continue
        with open(os.path.join(SNAP_DIR, files[0])) as f:
            d = json.load(f)
        fills = d.get("fills", [])
        print(f"=== {mode}_{n} ({len(fills)} fills) ===")
        for fill in fills:
            print(f"  {json.dumps(fill)}")
        # Also show adverse_prob and decision filter info
        dec = d.get("decision_history", [])
        if dec:
            last = dec[-1]
            print(f"  Last decision: bid_spread={last['bid_spread_bps']:.2f}, ask_spread={last['ask_spread_bps']:.2f}, regime={last['regime']}, reason={last.get('defensive_reason','')}")
        print()
