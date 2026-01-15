#!/bin/bash
# Paper Trading Simulation Script
# Usage: ./scripts/paper_trading.sh [ASSET] [DURATION_SECS] [OPTIONS]
#
# Examples:
#   ./scripts/paper_trading.sh BTC 60                  # 1 minute simulation
#   ./scripts/paper_trading.sh BTC 300                 # 5 minute simulation
#   ./scripts/paper_trading.sh BTC 3600 --report       # 1 hour with calibration report
#   ./scripts/paper_trading.sh BTC 300 --verbose       # With verbose logging
#   ./scripts/paper_trading.sh BTC 3600 --dashboard    # With live dashboard
#   ./scripts/paper_trading.sh BTC 300 --capture       # With dashboard + screenshot capture
#
# Options:
#   --report    Generate calibration report at end
#   --verbose   Enable verbose logging
#   --testnet   Use testnet instead of mainnet data
#   --dashboard Enable live dashboard at http://localhost:3000
#   --capture   Enable dashboard screenshots for Claude vision (implies --dashboard)
#
# Screenshot Capture:
#   When --capture is enabled:
#   - Screenshots saved to tools/dashboard-capture/screenshots/YYYY-MM-DD/
#   - Captures all 6 tabs every 5 seconds (optimized for Claude vision)
#   - Requires: cd tools/dashboard-capture && npm install (first time only)
#
# Description:
#   This script runs the paper trading simulator which:
#   - Subscribes to real market data (L2 book, trades, mids)
#   - Generates quotes as the live system would
#   - Simulates fills based on market trade flow
#   - Logs all predictions for calibration analysis
#   - Tracks PnL attribution (spread capture, adverse selection, fees)
#   - NO REAL ORDERS ARE PLACED - safe for testing
#
# Output:
#   - logs/paper_trading_ASSET_TIMESTAMP.log  - Main log file
#   - logs/paper_trading_ASSET_TIMESTAMP/predictions.jsonl - Prediction records
#   - logs/paper_trading_ASSET_TIMESTAMP/fills.jsonl - Simulated fills

set -e

# Parse arguments
ASSET="${1:-BTC}"
DURATION="${2:-60}"
REPORT=false
VERBOSE=false
NETWORK="mainnet"
DASHBOARD=false
CAPTURE=false
METRICS_PORT=8080

for arg in "$@"; do
    case $arg in
        --report)
            REPORT=true
            ;;
        --verbose)
            VERBOSE=true
            ;;
        --testnet)
            NETWORK="testnet"
            ;;
        --dashboard)
            DASHBOARD=true
            ;;
        --capture)
            CAPTURE=true
            DASHBOARD=true  # --capture implies --dashboard
            ;;
    esac
done

# Handle case where second arg is a flag
if [[ "$2" == "--report" ]] || [[ "$2" == "--verbose" ]] || [[ "$2" == "--testnet" ]] || [[ "$2" == "--dashboard" ]] || [[ "$2" == "--capture" ]]; then
    DURATION=60  # Default duration if only flag passed
fi

TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
LOG_DIR="logs"
OUTPUT_DIR="${LOG_DIR}/paper_trading_${ASSET}_${TIMESTAMP}"
LOG_FILE="${OUTPUT_DIR}/paper_trader.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  Paper Trading Simulator${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""
echo -e "Mode:      ${CYAN}SIMULATION${NC} (no real orders)"
echo -e "Network:   ${GREEN}${NETWORK}${NC} (data source)"
echo -e "Asset:     ${GREEN}${ASSET}${NC}"
echo -e "Duration:  ${GREEN}${DURATION}s${NC}"
echo -e "Output:    ${GREEN}${OUTPUT_DIR}${NC}"
if [ "$DASHBOARD" = true ]; then
    echo -e "Dashboard: ${GREEN}http://localhost:3000/mm-dashboard-fixed.html${NC}"
    echo -e "API:       ${GREEN}http://localhost:${METRICS_PORT}/api/dashboard${NC}"
fi
if [ "$CAPTURE" = true ]; then
    echo -e "Capture:   ${GREEN}tools/dashboard-capture/screenshots/${NC}"
fi
echo -e "Started:   ${GREEN}$(date)${NC}"
echo ""

# Ensure output directory exists
mkdir -p "${OUTPUT_DIR}"

# Pre-flight check: verify build
echo -e "${YELLOW}[1/3] Building paper_trader...${NC}"
if ! cargo build --bin paper_trader 2>/dev/null; then
    echo -e "${RED}Build failed! Run 'cargo build --bin paper_trader' to see errors.${NC}"
    exit 1
fi
echo -e "${GREEN}Build OK${NC}"

# Pre-flight check: verify we have market data access
echo -e "${YELLOW}[2/3] Verifying configuration...${NC}"
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}Note: No .env file found (API key optional for paper trading)${NC}"
fi
echo -e "${GREEN}Config OK${NC}"

# Run paper trader
echo -e "${YELLOW}[3/3] Running paper trading simulation for ${DURATION}s...${NC}"
echo -e "${YELLOW}       Press Ctrl+C to stop early${NC}"
echo ""

# Set RUST_LOG based on verbose mode
if [ "$VERBOSE" = true ]; then
    export RUST_LOG="hyperliquid_rust_sdk::market_maker::simulation=debug,paper_trader=debug"
    echo -e "${MAGENTA}VERBOSE MODE ENABLED - detailed logging active${NC}"
else
    export RUST_LOG="hyperliquid_rust_sdk::market_maker::simulation=info,paper_trader=info"
fi

# Build command args
PT_ARGS="--asset ${ASSET} --duration ${DURATION} --output-dir ${OUTPUT_DIR}"
if [ "$REPORT" = true ]; then
    PT_ARGS="${PT_ARGS} --report"
fi
if [ "$VERBOSE" = true ]; then
    PT_ARGS="${PT_ARGS} --verbose"
fi
if [ "$NETWORK" = "testnet" ]; then
    PT_ARGS="${PT_ARGS} --network testnet"
fi
if [ "$DASHBOARD" = true ]; then
    PT_ARGS="${PT_ARGS} --dashboard --metrics-port ${METRICS_PORT}"
fi

# Start dashboard HTTP server if requested
DASHBOARD_PID=""
CAPTURE_PID=""
if [ "$DASHBOARD" = true ]; then
    echo -e "${YELLOW}Starting dashboard server...${NC}"
    python3 -m http.server 3000 &>/dev/null &
    DASHBOARD_PID=$!
    echo -e "${GREEN}Dashboard server started (PID: ${DASHBOARD_PID})${NC}"
    echo -e "${GREEN}Open: http://localhost:3000/mm-dashboard-fixed.html${NC}"
    echo ""
fi

# Start screenshot capture if requested
if [ "$CAPTURE" = true ]; then
    echo -e "${YELLOW}Starting screenshot capture...${NC}"
    CAPTURE_DIR="tools/dashboard-capture"

    # Check if node_modules exists
    if [ ! -d "${CAPTURE_DIR}/node_modules" ]; then
        echo -e "${YELLOW}Installing capture tool dependencies...${NC}"
        (cd "${CAPTURE_DIR}" && npm install --silent)
    fi

    # Start capture tool in background
    (cd "${CAPTURE_DIR}" && node src/index.js) &
    CAPTURE_PID=$!

    # Wait a moment for browser to launch
    sleep 3
    echo -e "${GREEN}Screenshot capture started (PID: ${CAPTURE_PID})${NC}"
    echo -e "${GREEN}Screenshots: ${CAPTURE_DIR}/screenshots/${NC}"
    echo ""
fi

# Run paper trader with tee to capture output
./target/debug/paper_trader ${PT_ARGS} 2>&1 | tee "${LOG_FILE}" || true

# Stop capture tool if started
if [ -n "$CAPTURE_PID" ]; then
    echo ""
    echo -e "${YELLOW}Stopping screenshot capture...${NC}"
    kill $CAPTURE_PID 2>/dev/null || true
    sleep 1
fi

# Stop dashboard server if started
if [ -n "$DASHBOARD_PID" ]; then
    echo ""
    echo -e "${YELLOW}Stopping dashboard server...${NC}"
    kill $DASHBOARD_PID 2>/dev/null || true
fi

echo ""
echo -e "${YELLOW}Simulation complete${NC}"
echo ""

# Summary
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  Simulation Summary${NC}"
echo -e "${CYAN}========================================${NC}"
echo -e "Output Dir:  ${GREEN}${OUTPUT_DIR}${NC}"
echo -e "Log File:    ${GREEN}${LOG_FILE}${NC}"
if [ "$CAPTURE" = true ]; then
    SCREENSHOT_DIR="tools/dashboard-capture/screenshots/$(date +%Y-%m-%d)"
    SCREENSHOT_COUNT=$(find "${SCREENSHOT_DIR}" -name "*.png" 2>/dev/null | wc -l || echo 0)
    echo -e "Screenshots: ${GREEN}${SCREENSHOT_DIR}/${NC} (${SCREENSHOT_COUNT} files)"
fi
echo -e "Ended:       ${GREEN}$(date)${NC}"
echo ""

# Quick analysis
if [ -f "${LOG_FILE}" ]; then
    echo -e "${YELLOW}Quick Analysis:${NC}"
    echo -e "  Total lines:    $(wc -l < "${LOG_FILE}")"
    echo -e "  Errors:         $(grep -c "ERROR" "${LOG_FILE}" 2>/dev/null || echo 0)"
    echo -e "  Warnings:       $(grep -c "WARN" "${LOG_FILE}" 2>/dev/null || echo 0)"
    echo -e "  Quote cycles:   $(grep -c "Quote cycle" "${LOG_FILE}" 2>/dev/null || echo 0)"
    echo -e "  Simulated fills:$(grep -c "\[SIM\] Fill" "${LOG_FILE}" 2>/dev/null || echo 0)"
fi

# Check for prediction file
if [ -f "${OUTPUT_DIR}/predictions.jsonl" ]; then
    PRED_COUNT=$(wc -l < "${OUTPUT_DIR}/predictions.jsonl")
    echo -e "  Predictions:    ${PRED_COUNT}"
fi

echo ""
echo -e "${BLUE}Output Files:${NC}"
echo -e "  ${GREEN}${LOG_FILE}${NC}             - Full simulation log"
if [ -f "${OUTPUT_DIR}/predictions.jsonl" ]; then
    echo -e "  ${GREEN}${OUTPUT_DIR}/predictions.jsonl${NC} - Prediction records (for calibration)"
fi

echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo -e "  1. Review log:      ${GREEN}less ${LOG_FILE}${NC}"
echo -e "  2. Analyze fills:   ${GREEN}grep '\[SIM\] Fill' ${LOG_FILE}${NC}"
if [ "$CAPTURE" = true ]; then
    echo -e "  3. Vision:          Feed screenshots to Claude for visual analysis"
    echo -e "  4. Calibration:     Run ${GREEN}./scripts/paper_trading.sh ${ASSET} 3600 --report${NC} for full analysis"
    echo -e "  5. Live trading:    Run ${GREEN}./scripts/test_testnet.sh${NC} (testnet) or ${GREEN}./scripts/test_mainnet.sh${NC}"
else
    echo -e "  3. Calibration:     Run ${GREEN}./scripts/paper_trading.sh ${ASSET} 3600 --report${NC} for full analysis"
    echo -e "  4. Live trading:    Run ${GREEN}./scripts/test_testnet.sh${NC} (testnet) or ${GREEN}./scripts/test_mainnet.sh${NC}"
fi
echo ""
