#!/bin/bash
# Side-by-Side Paper vs Live Validation
# Usage: ./scripts/validate_side_by_side.sh ASSET DEX DURATION_SECS [SPREAD_PROFILE]
#
# Runs paper trading and live trading simultaneously on the same asset/market
# with identical configurations to isolate where PnL divergence originates.
#
# Examples:
#   ./scripts/validate_side_by_side.sh HYPE hyna 3600          # 1 hour, hip3 profile
#   ./scripts/validate_side_by_side.sh HYPE hyna 3600 hip3     # 1 hour explicit profile
#   ./scripts/validate_side_by_side.sh ETH hyna 1800 default   # ETH with default profile
#
# What happens:
#   1. Builds release binary (shared by both)
#   2. Starts paper trading in background (metrics on port 9091)
#   3. Starts live trading in foreground (metrics on port 9090, requires Enter confirmation)
#   4. Captures dashboard API snapshots every 60s from both
#   5. After completion, runs comparison analysis
#
# Output:
#   logs/side_by_side_ASSET_TIMESTAMP/
#     paper/       - Paper trading logs
#     live/        - Live trading logs
#     snapshots/   - Periodic dashboard API captures
#     comparison.txt - Side-by-side analysis report

set -e

# === Parse arguments ===
ASSET="${1:?Usage: $0 ASSET DEX DURATION_SECS [SPREAD_PROFILE]}"
DEX="${2:?Usage: $0 ASSET DEX DURATION_SECS [SPREAD_PROFILE]}"
DURATION="${3:?Usage: $0 ASSET DEX DURATION_SECS [SPREAD_PROFILE]}"
SPREAD_PROFILE="${4:-hip3}"

TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
BASE_DIR="logs/side_by_side_${ASSET}_${TIMESTAMP}"
PAPER_DIR="${BASE_DIR}/paper"
LIVE_DIR="${BASE_DIR}/live"
SNAPSHOT_DIR="${BASE_DIR}/snapshots"

PAPER_PORT=9091
LIVE_PORT=9090

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  Side-by-Side Paper vs Live Validation${NC}"
echo -e "${CYAN}========================================${NC}"
echo ""
echo -e "Asset:          ${GREEN}${ASSET}${NC}"
echo -e "DEX:            ${GREEN}${DEX}${NC}"
echo -e "Profile:        ${GREEN}${SPREAD_PROFILE}${NC}"
echo -e "Duration:       ${GREEN}${DURATION}s${NC}"
echo -e "Paper port:     ${GREEN}${PAPER_PORT}${NC}"
echo -e "Live port:      ${GREEN}${LIVE_PORT}${NC}"
echo -e "Output:         ${GREEN}${BASE_DIR}/${NC}"
echo -e "Started:        ${GREEN}$(date)${NC}"
echo ""

# === Create output directories ===
mkdir -p "${PAPER_DIR}" "${LIVE_DIR}" "${SNAPSHOT_DIR}"

# === Build release binary (shared) ===
echo -e "${YELLOW}[1/5] Building release binary...${NC}"
if ! cargo build --release --bin market_maker 2>/dev/null; then
    echo -e "${RED}Build failed! Run 'cargo build --release --bin market_maker' to see errors.${NC}"
    exit 1
fi
echo -e "${GREEN}Build OK${NC}"
echo ""

# === Resolve config file ===
if [ -f "market_maker_live.toml" ]; then
    CONFIG_FILE="market_maker_live.toml"
elif [ -f "market_maker.toml" ]; then
    CONFIG_FILE="market_maker.toml"
else
    echo -e "${RED}No config file found (market_maker_live.toml or market_maker.toml)${NC}"
    exit 1
fi
echo -e "Config:         ${GREEN}${CONFIG_FILE}${NC}"
echo ""

# === Common CLI args (identical for both) ===
COMMON_ARGS="--config ${CONFIG_FILE} --network mainnet --asset ${ASSET} --dex ${DEX} --spread-profile ${SPREAD_PROFILE}"

# === Start dashboard snapshot collector in background ===
echo -e "${YELLOW}[2/5] Starting snapshot collector (every 60s)...${NC}"
(
    sleep 10  # Wait for processes to start
    SNAP_NUM=0
    while true; do
        SNAP_NUM=$((SNAP_NUM + 1))
        SNAP_TS=$(date +%Y-%m-%d_%H-%M-%S)

        # Capture paper dashboard
        curl -s "http://localhost:${PAPER_PORT}/api/dashboard" \
            > "${SNAPSHOT_DIR}/paper_${SNAP_NUM}_${SNAP_TS}.json" 2>/dev/null || true

        # Capture live dashboard
        curl -s "http://localhost:${LIVE_PORT}/api/dashboard" \
            > "${SNAPSHOT_DIR}/live_${SNAP_NUM}_${SNAP_TS}.json" 2>/dev/null || true

        sleep 60
    done
) &
SNAPSHOT_PID=$!

# === Start paper trading in background ===
echo -e "${YELLOW}[3/5] Starting paper trading (background, port ${PAPER_PORT})...${NC}"
PAPER_LOG="${PAPER_DIR}/paper_trader.log"
RUST_LOG=hyperliquid_rust_sdk::market_maker::simulation=info,market_maker=info \
    ./target/release/market_maker ${COMMON_ARGS} --metrics-port ${PAPER_PORT} \
    paper --duration ${DURATION} \
    > "${PAPER_LOG}" 2>&1 &
PAPER_PID=$!
echo -e "${GREEN}Paper trading started (PID: ${PAPER_PID})${NC}"
echo -e "${GREEN}Paper dashboard: http://localhost:${PAPER_PORT}/api/dashboard${NC}"
echo ""

# === Start live trading in foreground (needs Enter confirmation) ===
echo -e "${YELLOW}[4/5] Starting live trading (foreground, port ${LIVE_PORT})...${NC}"
echo ""
echo -e "${RED}+======================================================+${NC}"
echo -e "${RED}|  WARNING: This will trade REAL FUNDS on mainnet!      |${NC}"
echo -e "${RED}|  Asset: ${ASSET} / DEX: ${DEX} / Profile: ${SPREAD_PROFILE}$(printf '%*s' $((17 - ${#ASSET} - ${#DEX} - ${#SPREAD_PROFILE})) '')|${NC}"
echo -e "${RED}+======================================================+${NC}"
echo ""
echo -e "${YELLOW}Press Enter to start live trading, or Ctrl+C to abort (paper will stop too)...${NC}"
read -r

LIVE_LOG="${LIVE_DIR}/live_trader.log"

# Trap to clean up background processes on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}Cleaning up...${NC}"
    kill $PAPER_PID 2>/dev/null || true
    kill $SNAPSHOT_PID 2>/dev/null || true
    wait $PAPER_PID 2>/dev/null || true
    wait $SNAPSHOT_PID 2>/dev/null || true
}
trap cleanup EXIT

RUST_LOG=hyperliquid_rust_sdk::market_maker=debug \
    timeout --foreground "${DURATION}" ./target/release/market_maker ${COMMON_ARGS} \
    --metrics-port ${LIVE_PORT} --log-file "${LIVE_LOG}" \
    2>&1 | tee "${LIVE_LOG}.console" || true

# Wait for paper to finish (it has its own duration timer)
echo ""
echo -e "${YELLOW}Waiting for paper trading to finish...${NC}"
wait $PAPER_PID 2>/dev/null || true

# Stop snapshot collector
kill $SNAPSHOT_PID 2>/dev/null || true
wait $SNAPSHOT_PID 2>/dev/null || true

# Remove the trap since we cleaned up manually
trap - EXIT

# === Capture final snapshots ===
echo -e "${YELLOW}[5/5] Capturing final dashboard snapshots...${NC}"
curl -s "http://localhost:${PAPER_PORT}/api/dashboard" \
    > "${SNAPSHOT_DIR}/paper_final.json" 2>/dev/null || true
curl -s "http://localhost:${LIVE_PORT}/api/dashboard" \
    > "${SNAPSHOT_DIR}/live_final.json" 2>/dev/null || true

# === Summary ===
echo ""
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  Side-by-Side Complete${NC}"
echo -e "${CYAN}========================================${NC}"
echo -e "Output Dir:     ${GREEN}${BASE_DIR}${NC}"
echo -e "Paper Log:      ${GREEN}${PAPER_LOG}${NC}"
echo -e "Live Log:       ${GREEN}${LIVE_LOG}${NC}"
SNAP_COUNT=$(find "${SNAPSHOT_DIR}" -name "*.json" 2>/dev/null | wc -l)
echo -e "Snapshots:      ${GREEN}${SNAP_COUNT} files${NC}"
echo -e "Ended:          ${GREEN}$(date)${NC}"
echo ""

# === Quick analysis ===
echo -e "${YELLOW}Quick Analysis:${NC}"
if [ -f "${PAPER_LOG}" ]; then
    PAPER_FILLS=$(grep -c "\[SIM\] Fill" "${PAPER_LOG}" 2>/dev/null || echo 0)
    PAPER_ERRORS=$(grep -c "ERROR" "${PAPER_LOG}" 2>/dev/null || echo 0)
    echo -e "  Paper fills:    ${GREEN}${PAPER_FILLS}${NC}"
    echo -e "  Paper errors:   ${GREEN}${PAPER_ERRORS}${NC}"
fi
if [ -f "${LIVE_LOG}" ]; then
    LIVE_FILLS=$(grep -c "Trades processed" "${LIVE_LOG}" 2>/dev/null || echo 0)
    LIVE_ERRORS=$(grep -c "ERROR" "${LIVE_LOG}" 2>/dev/null || echo 0)
    echo -e "  Live fills:     ${GREEN}${LIVE_FILLS}${NC}"
    echo -e "  Live errors:    ${GREEN}${LIVE_ERRORS}${NC}"
fi
echo ""

# === Run comparison if analysis script exists ===
COMPARE_SCRIPT="scripts/analysis/compare_sessions.py"
if command -v python3 &> /dev/null && [ -f "${COMPARE_SCRIPT}" ]; then
    echo -e "${YELLOW}Running comparison analysis...${NC}"
    python3 "${COMPARE_SCRIPT}" "${BASE_DIR}" | tee "${BASE_DIR}/comparison.txt"
    echo ""
    echo -e "${GREEN}Comparison report: ${BASE_DIR}/comparison.txt${NC}"
else
    echo -e "${YELLOW}To analyze: python3 ${COMPARE_SCRIPT} ${BASE_DIR}${NC}"
fi

echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo -e "  1. Review comparison:  ${GREEN}cat ${BASE_DIR}/comparison.txt${NC}"
echo -e "  2. Paper log:          ${GREEN}less ${PAPER_LOG}${NC}"
echo -e "  3. Live log:           ${GREEN}less ${LIVE_LOG}${NC}"
echo -e "  4. Dashboard data:     ${GREEN}ls ${SNAPSHOT_DIR}/${NC}"
echo ""
