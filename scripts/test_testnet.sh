#!/bin/bash
# Testnet Market Maker Testing Script
# Usage: ./scripts/test_testnet.sh [ASSET] [DURATION_SECS] [--diagnostics] [--dashboard] [--capture]
#
# Examples:
#   ./scripts/test_testnet.sh BTC 60                  # 1 minute test
#   ./scripts/test_testnet.sh BTC 300                 # 5 minute test
#   ./scripts/test_testnet.sh BTC 7200                # 2 hour test
#   ./scripts/test_testnet.sh BTC 720 --diagnostics   # With L2/L3 diagnostics
#   ./scripts/test_testnet.sh BTC 300 --dashboard     # With live dashboard
#   ./scripts/test_testnet.sh BTC 3600 --capture       # With dashboard + screenshot capture
#
# Options:
#   --diagnostics  Enables additional log targets for Layer 2/3 analysis
#   --dashboard    Starts HTTP server for live dashboard at http://localhost:3000
#   --capture      Enables dashboard screenshots for Claude vision (implies --dashboard)
#
# Dashboard:
#   When --dashboard is enabled:
#   - Metrics API runs at http://localhost:8080/api/dashboard
#   - Dashboard HTML served at http://localhost:3000/mm-dashboard-fixed.html
#   - Shows live quotes, P&L, regime, fills, and calibration data
#
# Screenshot Capture:
#   When --capture is enabled:
#   - Screenshots saved to tools/dashboard-capture/screenshots/YYYY-MM-DD/
#   - Captures all 6 tabs every 5 seconds (optimized for Claude vision)
#   - Requires: cd tools/dashboard-capture && npm install (first time only)
#
# Notes:
#   - Uses testnet (no real funds at risk)
#   - Standard validator perps (cross margin)
#   - Default spread profile (40-50 bps)
#   - Good for testing new features before mainnet

set -e

# Parse arguments
ASSET="${1:-BTC}"
DURATION="${2:-60}"
DIAGNOSTICS=false
DASHBOARD=false
CAPTURE=false
METRICS_PORT=8080

for arg in "$@"; do
    case $arg in
        --diagnostics)
            DIAGNOSTICS=true
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
if [[ "$2" == "--diagnostics" ]] || [[ "$2" == "--dashboard" ]] || [[ "$2" == "--capture" ]]; then
    DURATION=60  # Default duration if only flag passed
fi
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/mm_testnet_${ASSET}_${TIMESTAMP}.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

echo -e "${MAGENTA}========================================${NC}"
echo -e "${MAGENTA}  Testnet Market Maker Test${NC}"
echo -e "${MAGENTA}========================================${NC}"
echo ""
echo -e "Network:   ${MAGENTA}TESTNET${NC} (no real funds)"
echo -e "Asset:     ${GREEN}${ASSET}${NC}"
echo -e "Duration:  ${GREEN}${DURATION}s${NC}"
echo -e "Log File:  ${GREEN}${LOG_FILE}${NC}"
if [ "$DASHBOARD" = true ]; then
    echo -e "Dashboard: ${GREEN}http://localhost:3000/mm-dashboard-fixed.html${NC}"
    echo -e "API:       ${GREEN}http://localhost:${METRICS_PORT}/api/dashboard${NC}"
fi
if [ "$CAPTURE" = true ]; then
    echo -e "Capture:   ${GREEN}tools/dashboard-capture/screenshots/${NC}"
fi
echo -e "Started:   ${GREEN}$(date)${NC}"
echo ""

# Ensure log directory exists
mkdir -p "${LOG_DIR}"

# Pre-flight check: verify build
echo -e "${YELLOW}[1/4] Verifying build...${NC}"
if ! cargo build --bin market_maker 2>/dev/null; then
    echo -e "${RED}Build failed! Run 'cargo build' to see errors.${NC}"
    exit 1
fi
echo -e "${GREEN}Build OK${NC}"

# Pre-flight check: verify config
echo -e "${YELLOW}[2/4] Checking configuration...${NC}"
if [ ! -f "market_maker.toml" ]; then
    echo -e "${YELLOW}Warning: No market_maker.toml found, using CLI defaults${NC}"
fi
if [ ! -f ".env" ]; then
    echo -e "${RED}Error: .env file not found (required for wallet key)${NC}"
    exit 1
fi
echo -e "${GREEN}Config OK${NC}"

# Start dashboard HTTP server if requested
DASHBOARD_PID=""
CAPTURE_PID=""
if [ "$DASHBOARD" = true ]; then
    if [ "$CAPTURE" = true ]; then
        echo -e "${YELLOW}[3/6] Starting dashboard server...${NC}"
    else
        echo -e "${YELLOW}[3/5] Starting dashboard server...${NC}"
    fi
    python3 -m http.server 3000 &>/dev/null &
    DASHBOARD_PID=$!
    echo -e "${GREEN}Dashboard server started (PID: ${DASHBOARD_PID})${NC}"
    echo -e "${GREEN}Open: http://localhost:3000/mm-dashboard-fixed.html${NC}"
    echo ""
fi

# Start screenshot capture if requested
if [ "$CAPTURE" = true ]; then
    echo -e "${YELLOW}[4/6] Starting screenshot capture...${NC}"
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

# Run market maker with timeout
if [ "$CAPTURE" = true ]; then
    echo -e "${YELLOW}[5/6] Running market maker for ${DURATION}s...${NC}"
elif [ "$DASHBOARD" = true ]; then
    echo -e "${YELLOW}[4/5] Running market maker for ${DURATION}s...${NC}"
else
    echo -e "${YELLOW}[3/4] Running market maker for ${DURATION}s...${NC}"
fi
echo -e "${YELLOW}       Press Ctrl+C to stop early (graceful shutdown)${NC}"
echo ""

# Set RUST_LOG based on diagnostics mode
if [ "$DIAGNOSTICS" = true ]; then
    export RUST_LOG="hyperliquid_rust_sdk::market_maker=debug,layer2::calibration=info,layer3::changepoint=info,layer3::trace=info,layer3::health=info,learning::scatter=info"
    echo -e "${MAGENTA}DIAGNOSTIC MODE ENABLED - extra logging active${NC}"
else
    export RUST_LOG="hyperliquid_rust_sdk::market_maker=debug"
fi

# Build command with optional metrics port
MM_ARGS="--network testnet --asset ${ASSET} --log-file ${LOG_FILE}"
if [ "$DASHBOARD" = true ]; then
    MM_ARGS="${MM_ARGS} --metrics-port ${METRICS_PORT}"
fi

# Use timeout with --foreground to ensure signals are forwarded to the child
timeout --foreground "${DURATION}" ./target/debug/market_maker ${MM_ARGS} \
    2>&1 | tee -a "${LOG_FILE}.console" || true

# Stop capture tool if started
if [ -n "$CAPTURE_PID" ]; then
    echo ""
    echo -e "${YELLOW}Stopping screenshot capture...${NC}"
    kill $CAPTURE_PID 2>/dev/null || true
    # Give it a moment to close browser gracefully
    sleep 1
fi

# Stop dashboard server if started
if [ -n "$DASHBOARD_PID" ]; then
    echo ""
    echo -e "${YELLOW}Stopping dashboard server...${NC}"
    kill $DASHBOARD_PID 2>/dev/null || true
fi

echo ""
if [ "$CAPTURE" = true ]; then
    echo -e "${YELLOW}[6/6] Test complete${NC}"
elif [ "$DASHBOARD" = true ]; then
    echo -e "${YELLOW}[5/5] Test complete${NC}"
else
    echo -e "${YELLOW}[4/4] Test complete${NC}"
fi
echo ""

# Summary
echo -e "${MAGENTA}========================================${NC}"
echo -e "${MAGENTA}  Test Summary${NC}"
echo -e "${MAGENTA}========================================${NC}"
echo -e "Network:     ${MAGENTA}TESTNET${NC}"
echo -e "Log File:    ${GREEN}${LOG_FILE}${NC}"
echo -e "Console Log: ${GREEN}${LOG_FILE}.console${NC}"
if [ "$CAPTURE" = true ]; then
    SCREENSHOT_DIR="tools/dashboard-capture/screenshots/$(date +%Y-%m-%d)"
    SCREENSHOT_COUNT=$(find "${SCREENSHOT_DIR}" -name "*.png" 2>/dev/null | wc -l || echo 0)
    echo -e "Screenshots: ${GREEN}${SCREENSHOT_DIR}/${NC} (${SCREENSHOT_COUNT} files)"
fi
echo -e "Ended:       ${GREEN}$(date)${NC}"
echo ""

# Quick log analysis
if [ -f "${LOG_FILE}" ]; then
    echo -e "${YELLOW}Quick Analysis:${NC}"
    echo -e "  Total lines:  $(wc -l < "${LOG_FILE}")"
    echo -e "  Errors:       $(grep -c "ERROR" "${LOG_FILE}" 2>/dev/null || echo 0)"
    echo -e "  Warnings:     $(grep -c "WARN" "${LOG_FILE}" 2>/dev/null || echo 0)"
    echo -e "  Trades:       $(grep -c "Trades processed" "${LOG_FILE}" 2>/dev/null || echo 0)"
    echo -e "  Quotes:       $(grep -c "Quote cycle" "${LOG_FILE}" 2>/dev/null || echo 0)"

    # Diagnostic mode analysis
    if [ "$DIAGNOSTICS" = true ]; then
        echo ""
        echo -e "${MAGENTA}Layer 2/3 Diagnostic Analysis:${NC}"

        # Calibration metrics
        CALIBRATION_COUNT=$(grep -c "\[Calibration\]" "${LOG_FILE}" 2>/dev/null || echo 0)
        echo -e "  [Calibration] entries:  ${CALIBRATION_COUNT}"

        # Changepoint detection
        CHANGEPOINT_COUNT=$(grep -c "\[Changepoint\]" "${LOG_FILE}" 2>/dev/null || echo 0)
        echo -e "  [Changepoint] entries:  ${CHANGEPOINT_COUNT}"

        # L1->L2->L3 trace
        TRACE_COUNT=$(grep -c "\[Trace\]" "${LOG_FILE}" 2>/dev/null || echo 0)
        echo -e "  [Trace] entries:        ${TRACE_COUNT}"

        # Edge scatter data
        SCATTER_COUNT=$(grep -c "\[EdgeScatter\]" "${LOG_FILE}" 2>/dev/null || echo 0)
        echo -e "  [EdgeScatter] entries:  ${SCATTER_COUNT}"

        echo ""
        echo -e "${YELLOW}Sample Diagnostic Output:${NC}"

        # Show first calibration entry
        if [ "$CALIBRATION_COUNT" -gt 0 ]; then
            echo -e "${BLUE}[Calibration] (first):${NC}"
            grep "\[Calibration\]" "${LOG_FILE}" | head -1 | sed 's/^/    /'
        fi

        # Show first changepoint entry
        if [ "$CHANGEPOINT_COUNT" -gt 0 ]; then
            echo -e "${BLUE}[Changepoint] (first):${NC}"
            grep "\[Changepoint\]" "${LOG_FILE}" | head -1 | sed 's/^/    /'
        fi

        # Show first trace entry
        if [ "$TRACE_COUNT" -gt 0 ]; then
            echo -e "${BLUE}[Trace] (first):${NC}"
            grep "\[Trace\]" "${LOG_FILE}" | head -1 | sed 's/^/    /'
        fi

        # Show scatter stats if available
        if [ "$SCATTER_COUNT" -gt 0 ]; then
            echo -e "${BLUE}[EdgeScatter] summary (predicted vs realized):${NC}"
            echo "    Total observations: ${SCATTER_COUNT}"
            # Extract and show min/max predicted values if possible
            grep "\[EdgeScatter\]" "${LOG_FILE}" | head -5 | sed 's/^/    /'
            echo "    ..."
        fi

        echo ""
        echo -e "${YELLOW}Diagnostic Grep Commands:${NC}"
        echo -e "  ${GREEN}grep '\[Calibration\]' ${LOG_FILE}${NC}     # Model calibration metrics"
        echo -e "  ${GREEN}grep '\[Changepoint\]' ${LOG_FILE}${NC}     # BOCD changepoint state"
        echo -e "  ${GREEN}grep '\[Trace\]' ${LOG_FILE}${NC}           # L1->L2->L3 pipeline traces"
        echo -e "  ${GREEN}grep '\[EdgeScatter\]' ${LOG_FILE}${NC}     # Prediction vs realized data"
    fi
fi

echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo -e "  1. Review log:  ${GREEN}less ${LOG_FILE}${NC}"
echo -e "  2. Analyze:     Ask Claude to run ${GREEN}sc:analyze${NC} on the log"
if [ "$CAPTURE" = true ]; then
    echo -e "  3. Vision:      Feed screenshots to Claude for visual analysis"
    echo -e "  4. Mainnet:     Run ${GREEN}./scripts/test_mainnet.sh${NC} for production"
else
    echo -e "  3. Mainnet:     Run ${GREEN}./scripts/test_mainnet.sh${NC} for production"
fi
echo ""
