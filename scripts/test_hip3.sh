#!/bin/bash
# HIP-3 DEX Testing Workflow Script
# Usage: ./scripts/test_hip3.sh [ASSET] [DEX] [DURATION_SECS] [SPREAD_PROFILE]
#
# Examples:
#   ./scripts/test_hip3.sh BTC hyna 60         # 1 minute test (default profile)
#   ./scripts/test_hip3.sh HYPE hyna 600 hip3  # 5 minute test with tight spreads
#   ./scripts/test_hip3.sh HYPE hyna 14400 hip3   # 4 hour test with 15-25 bps target
#   ./scripts/test_hip3.sh ETH flx 120         # 2 minute test on Felix
#
# Spread Profiles:
#   default    - 40-50 bps spreads (standard)
#   hip3       - 15-25 bps spreads (optimized for HIP-3 DEX)
#   aggressive - 10-20 bps spreads (experimental)

set -e

# Configuration
ASSET="${1:-BTC}"
DEX="${2:-hyna}"
DURATION="${3:-60}"
SPREAD_PROFILE="${4:-hip3}"  # Default to hip3 for HIP-3 DEX testing
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/mm_hip3_${DEX}_${ASSET}_${SPREAD_PROFILE}_${TIMESTAMP}.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  HIP-3 DEX Market Maker Test${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "Asset:          ${GREEN}${ASSET}${NC}"
echo -e "DEX:            ${GREEN}${DEX}${NC}"
echo -e "Spread Profile: ${CYAN}${SPREAD_PROFILE}${NC}"
echo -e "Duration:       ${GREEN}${DURATION}s${NC}"
echo -e "Log File:       ${GREEN}${LOG_FILE}${NC}"
echo -e "Started:        ${GREEN}$(date)${NC}"
echo ""

# Show spread profile details
case "${SPREAD_PROFILE}" in
    hip3)
        echo -e "${CYAN}Profile: HIP-3 (kappa=1500, gamma=0.15, target=15-25 bps)${NC}"
        ;;
    aggressive)
        echo -e "${CYAN}Profile: Aggressive (kappa=2000, gamma=0.10, target=10-20 bps)${NC}"
        ;;
    default)
        echo -e "${CYAN}Profile: Default (kappa=500, gamma=0.30, target=40-50 bps)${NC}"
        ;;
    *)
        echo -e "${YELLOW}Profile: Unknown '${SPREAD_PROFILE}', using default${NC}"
        ;;
esac
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

# Run market maker with timeout
# Note: We run the binary directly (not via cargo run) so Ctrl+C signals
# are delivered correctly to the market_maker process
echo -e "${YELLOW}[3/4] Running market maker for ${DURATION}s...${NC}"
echo -e "${YELLOW}       Press Ctrl+C to stop early (graceful shutdown)${NC}"
echo ""

# Use timeout with --foreground to ensure signals are forwarded to the child
# The binary is run directly for proper signal handling
RUST_LOG=hyperliquid_rust_sdk::market_maker=debug \
timeout --foreground "${DURATION}" ./target/debug/market_maker \
    --network mainnet \
    --asset "${ASSET}" \
    --dex "${DEX}" \
    --spread-profile "${SPREAD_PROFILE}" \
    --log-file "${LOG_FILE}" \
    2>&1 | tee -a "${LOG_FILE}.console" || true

echo ""
echo -e "${YELLOW}[4/4] Test complete${NC}"
echo ""

# Summary
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Test Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Log File:       ${GREEN}${LOG_FILE}${NC}"
echo -e "Console Log:    ${GREEN}${LOG_FILE}.console${NC}"
echo -e "Spread Profile: ${CYAN}${SPREAD_PROFILE}${NC}"
echo -e "Ended:          ${GREEN}$(date)${NC}"
echo ""

# Quick log analysis
if [ -f "${LOG_FILE}" ]; then
    echo -e "${YELLOW}Quick Analysis:${NC}"
    echo -e "  Total lines:  $(wc -l < "${LOG_FILE}")"
    echo -e "  Errors:       $(grep -c "ERROR" "${LOG_FILE}" 2>/dev/null || echo 0)"
    echo -e "  Warnings:     $(grep -c "WARN" "${LOG_FILE}" 2>/dev/null || echo 0)"
    echo -e "  Trades:       $(grep -c "Trades processed" "${LOG_FILE}" 2>/dev/null || echo 0)"
    echo -e "  Quotes:       $(grep -c "Quote cycle" "${LOG_FILE}" 2>/dev/null || echo 0)"

    # Check for spread profile confirmation
    if grep -q "spread_profile.*${SPREAD_PROFILE}" "${LOG_FILE}" 2>/dev/null; then
        echo -e "  Profile:      ${GREEN}${SPREAD_PROFILE} (confirmed in logs)${NC}"
    fi
fi

echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo -e "  1. Review log:  ${GREEN}less ${LOG_FILE}${NC}"
echo -e "  2. Analyze:     Ask Claude to run ${GREEN}sc:analyze${NC} on the log"
echo -e "  3. Checkpoint:  Claude will create Serena session memory"
echo ""
