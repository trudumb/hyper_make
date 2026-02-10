#!/bin/bash
# Mainnet Validator Perps Market Maker Testing Script
# Usage: ./scripts/test_mainnet.sh [ASSET] [DURATION_SECS] [--dashboard] [--capture]
#
# Examples:
#   ./scripts/test_mainnet.sh BTC 60             # 1 minute test
#   ./scripts/test_mainnet.sh BTC 14400          # 4 hour test
#   ./scripts/test_mainnet.sh HYPE 14400         # 4 hour test
#   ./scripts/test_mainnet.sh SOL 3600           # 1 hour test
#   ./scripts/test_mainnet.sh BTC 300 --dashboard # With live dashboard
#   ./scripts/test_mainnet.sh BTC 300 --capture   # With dashboard + screenshot capture
#
# Options:
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
#   - Uses MAINNET (real funds at risk!)
#   - Standard validator perps (cross margin, USDC collateral)
#   - Default spread profile (40-50 bps for liquid perps)
#   - For HIP-3 DEX testing, use test_hip3.sh instead

set -e

# Configuration
ASSET="${1:-BTC}"
DURATION="${2:-60}"
DASHBOARD=false
CAPTURE=false
METRICS_PORT=8080

for arg in "$@"; do
    case $arg in
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
if [[ "$2" == "--dashboard" ]] || [[ "$2" == "--capture" ]]; then
    DURATION=60  # Default duration if only flag passed
fi
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/mm_mainnet_${ASSET}_${TIMESTAMP}.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${RED}========================================${NC}"
echo -e "${RED}  MAINNET Market Maker Test${NC}"
echo -e "${RED}========================================${NC}"
echo ""
echo -e "${RED}*** WARNING: REAL FUNDS AT RISK ***${NC}"
echo ""
echo -e "Network:   ${RED}MAINNET${NC}"
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

# Confirm mainnet operation
echo -e "${YELLOW}Press Enter to continue or Ctrl+C to abort...${NC}"
read -r

# Ensure log directory exists
mkdir -p "${LOG_DIR}"

# Pre-flight check: verify build
echo -e "${YELLOW}[1/4] Verifying build...${NC}"
if ! cargo build --release --bin market_maker 2>/dev/null; then
    echo -e "${RED}Build failed! Run 'cargo build --release' to see errors.${NC}"
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

# Build command with optional metrics port
MM_ARGS="--network mainnet --asset ${ASSET} --log-file ${LOG_FILE}"
if [ "$DASHBOARD" = true ]; then
    MM_ARGS="${MM_ARGS} --metrics-port ${METRICS_PORT}"
fi

# Use timeout with --foreground to ensure signals are forwarded to the child
RUST_LOG=hyperliquid_rust_sdk::market_maker=debug \
timeout --foreground "${DURATION}" ./target/release/market_maker ${MM_ARGS} \
    2>&1 | tee -a "${LOG_FILE}.console" || true

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
if [ "$CAPTURE" = true ]; then
    echo -e "${YELLOW}[6/6] Test complete${NC}"
elif [ "$DASHBOARD" = true ]; then
    echo -e "${YELLOW}[5/5] Test complete${NC}"
else
    echo -e "${YELLOW}[4/4] Test complete${NC}"
fi
echo ""

# Summary
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Test Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Network:     ${RED}MAINNET${NC}"
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
    echo -e "  Fills:        $(grep -c "Fill processed" "${LOG_FILE}" 2>/dev/null || echo 0)"
fi

echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo -e "  1. Review log:  ${GREEN}less ${LOG_FILE}${NC}"
echo -e "  2. Analyze:     Ask Claude to run ${GREEN}sc:analyze${NC} on the log"
if [ "$CAPTURE" = true ]; then
    echo -e "  3. Vision:      Feed screenshots to Claude for visual analysis"
    echo -e "  4. HIP-3 DEX:   Run ${GREEN}./scripts/test_hip3.sh HYPE hyna 60 hip3${NC}"
else
    echo -e "  3. HIP-3 DEX:   Run ${GREEN}./scripts/test_hip3.sh HYPE hyna 60 hip3${NC}"
fi
echo ""
