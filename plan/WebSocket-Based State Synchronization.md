Phase 2 Implementation Plan: WebSocket-Based State Synchronization
Goal
Eliminate the "race condition" in order tracking by switching the source of truth from periodic REST snapshots to the real-time WebSocket state manager. This aligns with the user's request to rely on WsOrderUpdates and WsUserFills.

User Review Required
IMPORTANT

This is a significant logic change. SafetySync will no longer poll the REST API for open orders in the hot loop. It will instead reconcile 
OrderManager
 (local strategy state) against 
WsOrderStateManager
 (websocket state). Ideally, we should perform a full system test on testnet/sim-mode after this change.

Proposed Changes
[Market Maker Core]
[MODIFY] 
mod.rs
Modify 
safety_sync
 to:
REMOVE: The REST call info_client.open_orders_for_dex().
ADD: Logic to fetch open orders from self.ws_state.order_ids().
ADD: Logic to trust ws_state as the "Exchange State".
[NEW] [Support OpenOrders Subscription]
MODIFY 
src/ws/ws_manager.rs
: Add OpenOrders to 
Subscription
 enum and Message enum.

MODIFY 
src/market_maker/tracking/ws_order_state/manager.rs
: Handle OpenOrders message to populate initial state or re-sync state (Snapshot).

This allows us to replace the start-up REST call 
sync_open_orders
 and 
safety_sync
 REST calls with WebSocket snapshots.

Ensure 
handle_message
 correctly propagates all updates to ws_state (verified in analysis).

[MODIFY] 
auditor.rs
Update 
find_stale_local
 call signatures if necessary (likely not, just passing different HashSets).
Verification Plan
Automated Tests
Run existing unit tests.
Create a new integration test simulating:
Order Placed.
WebSocket OrderUpdate(Open) arrives -> ws_state tracks it.
SafetySync runs -> sees order in ws_state, matches local state. (No "stale" removal).
WebSocket 
Fill
 arrives -> ws_state marks filled.
SafetySync runs -> sees order filled in ws_state, matches local filled state.
Manual Verification
Deploy to a test environment (if available) or run with minimal size/risk.
Monitor logs for "market maker started".
Verify "Untracked order filled" errors are GONE.
Verify [SafetySync] Order sync status logs show is_synced: true.