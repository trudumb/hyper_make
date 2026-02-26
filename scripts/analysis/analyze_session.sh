#!/bin/bash
# Session Analysis Script
# Usage: ./scripts/analysis/analyze_session.sh <session_dir>
#
# Analyzes paper trading session output and generates calibration reports.
#
# Expected session directory structure:
#   logs/paper_trading_ASSET_TIMESTAMP/
#   ├── paper_trader.log          - Main simulation log
#   ├── predictions.jsonl         - Prediction records (if enabled)
#   ├── fills.jsonl               - Simulated fills (if enabled)
#   └── calibration_report.json   - Generated report (output)

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

if [ -z "$1" ]; then
    echo -e "${RED}Usage: $0 <session_dir>${NC}"
    echo "Example: $0 logs/paper_trading_BTC_2026-01-22_15-30-00"
    exit 1
fi

SESSION_DIR="$1"
LOG_FILE="${SESSION_DIR}/paper_trader.log"

if [ ! -d "$SESSION_DIR" ]; then
    echo -e "${RED}Error: Session directory not found: ${SESSION_DIR}${NC}"
    exit 1
fi

if [ ! -f "$LOG_FILE" ]; then
    echo -e "${RED}Error: Log file not found: ${LOG_FILE}${NC}"
    exit 1
fi

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  Paper Trading Session Analysis${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""
echo -e "Session: ${GREEN}${SESSION_DIR}${NC}"
echo ""

# ============================================================================
# Basic Statistics
# ============================================================================
echo -e "${YELLOW}[1/6] Basic Statistics${NC}"
echo ""

TOTAL_LINES=$(wc -l < "$LOG_FILE")
ERROR_COUNT=$(grep -c "ERROR" "$LOG_FILE" 2>/dev/null || echo 0)
WARN_COUNT=$(grep -c "WARN" "$LOG_FILE" 2>/dev/null || echo 0)
QUOTE_CYCLES=$(grep -c "Quote cycle" "$LOG_FILE" 2>/dev/null || echo 0)
SIM_FILLS=$(grep -c "\[SIM\] Fill" "$LOG_FILE" 2>/dev/null || echo 0)

echo -e "  Log lines:        ${GREEN}${TOTAL_LINES}${NC}"
echo -e "  Errors:           $([ $ERROR_COUNT -gt 0 ] && echo -e "${RED}${ERROR_COUNT}${NC}" || echo -e "${GREEN}${ERROR_COUNT}${NC}")"
echo -e "  Warnings:         $([ $WARN_COUNT -gt 5 ] && echo -e "${YELLOW}${WARN_COUNT}${NC}" || echo -e "${GREEN}${WARN_COUNT}${NC}")"
echo -e "  Quote cycles:     ${GREEN}${QUOTE_CYCLES}${NC}"
echo -e "  Simulated fills:  ${GREEN}${SIM_FILLS}${NC}"
echo ""

# ============================================================================
# Time Analysis
# ============================================================================
echo -e "${YELLOW}[2/6] Session Timing${NC}"
echo ""

FIRST_LINE=$(head -1 "$LOG_FILE")
LAST_LINE=$(tail -1 "$LOG_FILE")

# Extract timestamps (format: 2026-01-22T15:30:00.123456Z)
FIRST_TS=$(echo "$FIRST_LINE" | grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}' | head -1 || echo "N/A")
LAST_TS=$(echo "$LAST_LINE" | grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}' | tail -1 || echo "N/A")

echo -e "  Start:            ${GREEN}${FIRST_TS}${NC}"
echo -e "  End:              ${GREEN}${LAST_TS}${NC}"

if [ "$QUOTE_CYCLES" -gt 0 ]; then
    # Calculate quotes per minute
    DURATION_SECS=$(grep -oE "duration=[0-9]+" "$LOG_FILE" | head -1 | grep -oE "[0-9]+" || echo "60")
    QUOTES_PER_MIN=$((QUOTE_CYCLES * 60 / DURATION_SECS))
    echo -e "  Quotes/minute:    ${GREEN}${QUOTES_PER_MIN}${NC}"
fi
echo ""

# ============================================================================
# Fill Analysis
# ============================================================================
echo -e "${YELLOW}[3/6] Fill Analysis${NC}"
echo ""

if [ "$SIM_FILLS" -gt 0 ]; then
    # Count buy vs sell fills
    BUY_FILLS=$(grep -c "\[SIM\] Fill.*side=Buy" "$LOG_FILE" 2>/dev/null || echo 0)
    SELL_FILLS=$(grep -c "\[SIM\] Fill.*side=Sell" "$LOG_FILE" 2>/dev/null || echo 0)

    echo -e "  Buy fills:        ${GREEN}${BUY_FILLS}${NC}"
    echo -e "  Sell fills:       ${GREEN}${SELL_FILLS}${NC}"

    # Calculate fill rate
    if [ "$QUOTE_CYCLES" -gt 0 ]; then
        FILL_RATE=$(echo "scale=2; $SIM_FILLS * 100 / $QUOTE_CYCLES" | bc)
        echo -e "  Fill rate:        ${GREEN}${FILL_RATE}%${NC} per cycle"
    fi

    # Extract adverse selection if logged
    AS_LINES=$(grep "adverse_selection" "$LOG_FILE" | tail -10)
    if [ -n "$AS_LINES" ]; then
        echo ""
        echo -e "  ${BLUE}Recent Adverse Selection:${NC}"
        echo "$AS_LINES" | while read line; do
            echo "    $line"
        done | head -5
    fi
else
    echo -e "  ${YELLOW}No simulated fills recorded${NC}"
fi
echo ""

# ============================================================================
# Parameter Estimation
# ============================================================================
echo -e "${YELLOW}[4/6] Parameter Estimation${NC}"
echo ""

# Extract kappa estimates
KAPPA_LINES=$(grep "kappa=" "$LOG_FILE" | tail -5)
if [ -n "$KAPPA_LINES" ]; then
    echo -e "  ${BLUE}Recent Kappa (fill intensity):${NC}"
    echo "$KAPPA_LINES" | while read line; do
        KAPPA=$(echo "$line" | grep -oE "kappa=[0-9.]+" | head -1)
        echo "    $KAPPA"
    done
fi

# Extract sigma estimates
SIGMA_LINES=$(grep "sigma=" "$LOG_FILE" | tail -5)
if [ -n "$SIGMA_LINES" ]; then
    echo -e "  ${BLUE}Recent Sigma (volatility):${NC}"
    echo "$SIGMA_LINES" | while read line; do
        SIGMA=$(echo "$line" | grep -oE "sigma=[0-9.]+" | head -1)
        echo "    $SIGMA"
    done
fi
echo ""

# ============================================================================
# Risk Analysis
# ============================================================================
echo -e "${YELLOW}[5/6] Risk Analysis${NC}"
echo ""

# Check for reduce-only triggers
REDUCE_ONLY=$(grep -c "Reduce-only" "$LOG_FILE" 2>/dev/null || echo 0)
KILL_SWITCH=$(grep -c "Kill switch" "$LOG_FILE" 2>/dev/null || echo 0)
CASCADE=$(grep -c "cascade" "$LOG_FILE" 2>/dev/null || echo 0)

echo -e "  Reduce-only triggers:  $([ $REDUCE_ONLY -gt 0 ] && echo -e "${YELLOW}${REDUCE_ONLY}${NC}" || echo -e "${GREEN}${REDUCE_ONLY}${NC}")"
echo -e "  Kill switch events:    $([ $KILL_SWITCH -gt 0 ] && echo -e "${RED}${KILL_SWITCH}${NC}" || echo -e "${GREEN}${KILL_SWITCH}${NC}")"
echo -e "  Cascade detections:    $([ $CASCADE -gt 0 ] && echo -e "${YELLOW}${CASCADE}${NC}" || echo -e "${GREEN}${CASCADE}${NC}")"
echo ""

# ============================================================================
# Calibration Metrics (if predictions.jsonl exists)
# ============================================================================
echo -e "${YELLOW}[6/6] Calibration Status${NC}"
echo ""

PRED_FILE="${SESSION_DIR}/predictions.jsonl"
if [ -f "$PRED_FILE" ]; then
    PRED_COUNT=$(wc -l < "$PRED_FILE")
    echo -e "  Prediction records: ${GREEN}${PRED_COUNT}${NC}"

    if [ "$PRED_COUNT" -ge 200 ]; then
        echo -e "  Sample size:        ${GREEN}SUFFICIENT (>=200)${NC}"
    elif [ "$PRED_COUNT" -ge 100 ]; then
        echo -e "  Sample size:        ${YELLOW}MARGINAL (100-199)${NC}"
    else
        echo -e "  Sample size:        ${RED}INSUFFICIENT (<100)${NC}"
    fi

    # Show first and last prediction for context
    echo ""
    echo -e "  ${BLUE}Sample predictions:${NC}"
    echo "    First: $(head -1 "$PRED_FILE" | cut -c1-100)..."
    echo "    Last:  $(tail -1 "$PRED_FILE" | cut -c1-100)..."
else
    echo -e "  ${YELLOW}No predictions.jsonl found${NC}"
    echo -e "  Run with --report flag to enable prediction logging"
fi

FILLS_FILE="${SESSION_DIR}/fills.jsonl"
if [ -f "$FILLS_FILE" ]; then
    FILLS_COUNT=$(wc -l < "$FILLS_FILE")
    echo ""
    echo -e "  Fill records:       ${GREEN}${FILLS_COUNT}${NC}"
fi
echo ""

# ============================================================================
# Summary and Recommendations
# ============================================================================
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  Summary${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""

# Health check
HEALTH="HEALTHY"
ISSUES=""

if [ "$ERROR_COUNT" -gt 0 ]; then
    HEALTH="DEGRADED"
    ISSUES="${ISSUES}  - ${ERROR_COUNT} errors detected\n"
fi

if [ "$WARN_COUNT" -gt 10 ]; then
    HEALTH="DEGRADED"
    ISSUES="${ISSUES}  - High warning count (${WARN_COUNT})\n"
fi

if [ "$KILL_SWITCH" -gt 0 ]; then
    HEALTH="CRITICAL"
    ISSUES="${ISSUES}  - Kill switch triggered\n"
fi

if [ "$SIM_FILLS" -eq 0 ] && [ "$QUOTE_CYCLES" -gt 10 ]; then
    HEALTH="WARNING"
    ISSUES="${ISSUES}  - No fills despite quoting (check spread/size)\n"
fi

echo -e "Session Health: $([ "$HEALTH" = "HEALTHY" ] && echo -e "${GREEN}${HEALTH}${NC}" || ([ "$HEALTH" = "DEGRADED" ] && echo -e "${YELLOW}${HEALTH}${NC}" || ([ "$HEALTH" = "WARNING" ] && echo -e "${YELLOW}${HEALTH}${NC}" || echo -e "${RED}${HEALTH}${NC}")))"

if [ -n "$ISSUES" ]; then
    echo ""
    echo -e "${YELLOW}Issues:${NC}"
    echo -e "$ISSUES"
fi

echo ""
echo -e "${BLUE}Next Steps:${NC}"
if [ ! -f "$PRED_FILE" ]; then
    echo "  1. Re-run with --report flag: ./scripts/paper_trading.sh ASSET DURATION --report"
fi
if [ "$SIM_FILLS" -lt 50 ]; then
    echo "  2. Run longer session for more samples (recommend 3600s minimum)"
fi
echo "  3. Review errors: grep ERROR ${LOG_FILE}"
echo "  4. Check fills: grep '\[SIM\] Fill' ${LOG_FILE}"
echo ""
