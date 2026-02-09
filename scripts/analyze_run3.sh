#!/bin/bash
# Post-run analysis script for paper trader run 3
# Extracts key metrics from logs and checkpoint

LOG_DIR="logs/paper_trading/run3"
LOG_FILE="$LOG_DIR/full_output.log"
CHECKPOINT_DIR="data/checkpoints/paper/HYPE"

echo "============================================"
echo "  Paper Trader Run 3 Analysis (P0+P1 fixes)"
echo "============================================"
echo ""

# --- Fill Analysis ---
echo "=== FILL STATISTICS ==="
total_fills=$(grep -c "\[FILL FEEDBACK\]" "$LOG_FILE" 2>/dev/null || echo 0)
buy_fills=$(grep "\[FILL FEEDBACK\]" "$LOG_FILE" | grep -c "side=Buy" 2>/dev/null || echo 0)
sell_fills=$(grep "\[FILL FEEDBACK\]" "$LOG_FILE" | grep -c "side=Sell" 2>/dev/null || echo 0)
echo "Total fills: $total_fills (Buy: $buy_fills, Sell: $sell_fills)"
echo ""

# --- PnL ---
echo "=== PNL SUMMARY ==="
grep "\[PNL\]" "$LOG_FILE" | tail -3
grep "net_pnl\|realized_pnl\|total_pnl\|cumulative" "$LOG_FILE" | tail -5
echo ""

# --- RL Agent ---
echo "=== RL AGENT STATE ==="
grep "RL agent updated\|Paper RL agent\|Q-table\|reward_total\|realized_edge_bps" "$LOG_FILE" | tail -10
grep "RL Summary\|q_table_size\|total_observations\|avg_reward" "$LOG_FILE" | tail -5
echo ""

# --- Book Imbalance (P0-2 fix) ---
echo "=== BOOK IMBALANCE (P0-2 fix) ==="
grep "book_imbalance" "$LOG_FILE" | head -5
non_zero=$(grep "book_imbalance" "$LOG_FILE" | grep -v "book_imbalance=0.000" | wc -l)
total_imb=$(grep -c "book_imbalance" "$LOG_FILE" 2>/dev/null || echo 0)
echo "Non-zero imbalance observations: $non_zero / $total_imb"
echo ""

# --- Learned Parameters (P0-3 fix) ---
echo "=== LEARNED PARAMETERS (P0-3 fix) ==="
grep "learned alpha_touch\|learned kappa\|total_fills_observed\|n_observations" "$LOG_FILE" | tail -10
echo ""

# --- Kelly Tracker (P1-3 fix) ---
echo "=== KELLY TRACKER (P1-3 fix) ==="
grep -i "kelly\|win_rate\|record_win\|record_loss" "$LOG_FILE" | tail -5
echo ""

# --- Warmup Progress ---
echo "=== WARMUP PROGRESSION ==="
grep "warmup_pct" "$LOG_FILE" | sed 's/.*warmup_pct=\([^ ]*\).*/\1/' | sort -u | tr '\n' ' '
echo ""
echo ""

# --- Edge Analysis ---
echo "=== EDGE VALIDATION ==="
grep "realized_edge_bps\|spread_capture\|depth_bps" "$LOG_FILE" | grep "FILL" | tail -10
echo ""

# --- Checkpoint ---
echo "=== CHECKPOINT STATE ==="
if [ -d "$CHECKPOINT_DIR" ]; then
    ls -la "$CHECKPOINT_DIR"/ 2>/dev/null
    latest=$(ls -t "$CHECKPOINT_DIR"/*.json 2>/dev/null | head -1)
    if [ -n "$latest" ]; then
        echo ""
        echo "Latest checkpoint: $latest"
        # Extract key fields
        python3 -c "
import json, sys
try:
    with open('$latest') as f:
        data = json.load(f)
    print(f'Session duration: {data.get(\"metadata\", {}).get(\"session_duration_s\", 0):.0f}s')
    lp = data.get('learned_params', {})
    at = lp.get('alpha_touch', {})
    print(f'alpha_touch: estimate={at.get(\"estimate\", \"?\")}, n_obs={at.get(\"n_observations\", 0)}')
    kp = lp.get('kappa', {})
    print(f'kappa: estimate={kp.get(\"estimate\", \"?\")}, n_obs={kp.get(\"n_observations\", 0)}')
    print(f'total_fills_observed: {lp.get(\"total_fills_observed\", 0)}')
    kt = data.get('kelly_tracker', {})
    print(f'Kelly: n_wins={kt.get(\"n_wins\", 0)}, n_losses={kt.get(\"n_losses\", 0)}, ewma_wins={kt.get(\"ewma_wins\", 0):.2f}, ewma_losses={kt.get(\"ewma_losses\", 0):.2f}')
    rl = data.get('rl_q_table', {})
    print(f'RL Q-table: entries={len(rl.get(\"q_entries\", []))}, total_obs={rl.get(\"total_observations\", 0)}, total_reward={rl.get(\"total_reward\", 0):.2f}')
except Exception as e:
    print(f'Error: {e}')
" 2>/dev/null
    fi
else
    echo "No checkpoint directory found"
fi
echo ""

# --- Signal Performance ---
echo "=== SIGNAL CONTRIBUTIONS ==="
grep "signal_contributions\|per-signal\|SignalContribution" "$LOG_FILE" | tail -5
echo ""

# --- Adverse Selection ---
echo "=== ADVERSE SELECTION ==="
grep "AS classifier\|informed.*uninformed\|as_realized\|adverse" "$LOG_FILE" | grep -v "SPREAD TRACE" | tail -10
echo ""

echo "=== ANALYSIS COMPLETE ==="
