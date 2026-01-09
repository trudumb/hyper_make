#!/bin/bash
# Mainnet Validator Perps Market Maker Testing Script
# Usage: ./scripts/test_mainnet.sh [ASSET] [DURATION_SECS]
#
# Examples:
#   ./scripts/test_mainnet.sh BTC 60       # 1 minute test
#   ./scripts/test_mainnet.sh ETH 7200     # 2 hour test
#   ./scripts/test_mainnet.sh SOL 3600     # 1 hour test
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
echo -e "Started:   ${GREEN}$(date)${NC}"
echo ""

# Confirm mainnet operation
echo -e "${YELLOW}Press Enter to continue or Ctrl+C to abort...${NC}"
read -r

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
echo -e "${YELLOW}[3/4] Running market maker for ${DURATION}s...${NC}"
echo -e "${YELLOW}       Press Ctrl+C to stop early (graceful shutdown)${NC}"
echo ""

# Use timeout with --foreground to ensure signals are forwarded to the child
RUST_LOG=hyperliquid_rust_sdk::market_maker=debug \
timeout --foreground "${DURATION}" ./target/debug/market_maker \
    --network mainnet \
    --asset "${ASSET}" \
    --log-file "${LOG_FILE}" \
    2>&1 | tee -a "${LOG_FILE}.console" || true

echo ""
echo -e "${YELLOW}[4/4] Test complete${NC}"
echo ""

# Summary
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Test Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Network:     ${RED}MAINNET${NC}"
echo -e "Log File:    ${GREEN}${LOG_FILE}${NC}"
echo -e "Console Log: ${GREEN}${LOG_FILE}.console${NC}"
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
echo -e "  3. HIP-3 DEX:   Run ${GREEN}./scripts/test_hip3.sh HYPE hyna 60 hip3${NC}"
echo ""
