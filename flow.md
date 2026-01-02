Plan: First-Principles Refactor of Order State Management

 Executive Summary

 Root Cause: When finalize_pending_by_cloid() returns false (race condition with WebSocket fills), the return value is silently ignored, causing orders to exist on exchange but not in
 local tracking. This creates state mismatches like exchange=28, local_active=8.

 Solution: Create a single UnifiedOrderBook that guarantees tracking for every order on exchange.

 ---
 Current Architecture (The Mess)

                     ┌─────────────────────┐
                     │   MarketMaker       │
                     └─────────┬───────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
         ▼                     ▼                     ▼
 ┌───────────────┐    ┌───────────────┐    ┌───────────────┐
 │ OrderManager  │    │WsOrderState   │    │ OrphanTracker │
 │ (primary)     │    │Manager        │    │               │
 ├───────────────┤    │(parallel/dead)│    ├───────────────┤
 │ orders: Map   │    ├───────────────┤    │expected_cloids│
 │ pending: Map  │    │ orders: Map   │◄───┤recently_final │
 │ pending_cloid │    │ cloid_to_oid  │    │orphan_first   │
 │ fill_tids/ord │    │ processed_tids│    └───────────────┘
 └───────┬───────┘    └───────────────┘
         │
         │ (also)
         ▼
 ┌───────────────┐    ┌───────────────┐
 │FillDeduplicator│   │PositionTracker│
 │ (centralized) │    │ (simple)      │
 ├───────────────┤    ├───────────────┤
 │processed_tids │    │ position: f64 │
 └───────────────┘    └───────────────┘

 Problems Identified

 | #   | Problem                                                          | Impact                                |
 |-----|------------------------------------------------------------------|---------------------------------------|
 | 1   | Two parallel order tracking (OrderManager + WsOrderStateManager) | Confusion, dead code                  |
 | 2   | Three TID dedup mechanisms                                       | Redundant memory, inconsistent checks |
 | 3   | Two CLOID→OID mappings                                           | Race conditions, lookup failures      |
 | 4   | WsOrderStateManager never used for decisions                     | Dead weight                           |
 | 5   | finalize_pending_by_cloid() failure silently ignored             | ROOT CAUSE of state mismatches        |
 | 6   | No single source of truth                                        | Can't debug, can't reason about state |

 The Bug (Lines in mod.rs: 1691-1695, 2037-2041, 2261-2265)

 // CURRENT CODE - THE BUG:
 if let Some(c) = cloid {
     self.orders
         .finalize_pending_by_cloid(c, result.oid, result.resting_size);
     // ^^^^^^^^ Return value IGNORED! If false, order NEVER ADDED to tracking!
     self.infra.orphan_tracker.mark_finalized(c, result.oid);
 }

 What happens:
 1. add_pending_with_cloid() registers pending order
 2. API call succeeds, returns OID
 3. WS fill arrives BEFORE finalize_pending_by_cloid() (race!)
 4. WS handler calls process_fill() which consumes/removes the pending
 5. finalize_pending_by_cloid() returns false (pending not found)
 6. Order is on exchange but NOT in local orders map
 7. safety_sync sees mismatch, detects "orphan"

 ---
 Target Architecture (Clean)

                     ┌─────────────────────┐
                     │   MarketMaker       │
                     └─────────┬───────────┘
                               │
                               ▼
                     ┌─────────────────────┐
                     │  UnifiedOrderBook   │  ◄── Single Source of Truth
                     ├─────────────────────┤
                     │ orders: Map<OID>    │  Active orders by OID
                     │ pending: Map<CLOID> │  In-flight orders by CLOID
                     │ cloid_to_oid: Map   │  CLOID→OID resolution (persists!)
                     │ fill_dedup: TidSet  │  Single TID deduplication
                     │ position: f64       │  Authoritative position
                     │ orphan_guard: ...   │  Integrated orphan protection
                     └─────────┬───────────┘
                               │
               ┌───────────────┼───────────────┐
               │               │               │
               ▼               ▼               ▼
         ┌───────────┐  ┌───────────┐  ┌───────────┐
         │QueueTracker│  │SafetySync │  │ Metrics   │
         │(read-only)│  │(read-only)│  │(read-only)│
         └───────────┘  └───────────┘  └───────────┘

 Design Principles

 | Principle              | Implementation                                            |
 |------------------------|-----------------------------------------------------------|
 | Single source of truth | One UnifiedOrderBook owns all order state                 |
 | Explicit state machine | Clear transitions, no silent failures                     |
 | Guaranteed tracking    | finalize_order() ALWAYS adds order - never fails silently |
 | Atomic operations      | Position updates atomically with fills                    |
 | Unified deduplication  | One TID set, checked once per fill                        |

 ---
 Implementation Plan

 Phase 1: Create UnifiedOrderBook Module

 Create: src/market_maker/tracking/unified_book/mod.rs

 1.1 Core Data Structures

 use std::collections::{HashMap, HashSet, VecDeque};
 use std::time::{Duration, Instant};
 use smallvec::SmallVec;
 use uuid::Uuid;

 /// Configuration for UnifiedOrderBook.
 #[derive(Debug, Clone)]
 pub struct UnifiedBookConfig {
     /// Max TIDs to track for deduplication (FIFO eviction)
     pub max_tracked_tids: usize,           // default: 10_000
     /// Grace period for orphan detection
     pub orphan_grace_period: Duration,     // default: 5s
     /// TTL for expected CLOIDs
     pub expected_cloid_ttl: Duration,      // default: 30s
     /// Protection period after finalization
     pub finalized_protection: Duration,    // default: 10s
     /// Fill window before cleanup
     pub fill_window_duration: Duration,    // default: 60s
     /// Cancel timeout
     pub cancel_timeout: Duration,          // default: 5s
 }

 /// Single source of truth for all order state.
 ///
 /// ## Guarantees
 /// - Every order on exchange IS tracked locally
 /// - Every fill is processed exactly once
 /// - Position is always consistent with fills
 /// - CLOID→OID resolution is authoritative
 pub struct UnifiedOrderBook {
     // === Order State ===
     /// Active orders indexed by OID
     orders: HashMap<u64, TrackedOrder>,
     /// Pending orders indexed by CLOID (before OID assigned)
     pending: HashMap<String, PendingOrder>,
     /// CLOID→OID mapping (PERSISTS after finalization for late WS events!)
     cloid_to_oid: HashMap<String, u64>,

     // === Fill Deduplication (single source!) ===
     processed_tids: HashSet<u64>,
     tid_eviction: VecDeque<u64>,

     // === Position (authoritative) ===
     position: f64,
     fill_count: u64,

     // === Orphan Protection (embedded) ===
     orphan_guard: OrphanGuard,

     // === Config ===
     config: UnifiedBookConfig,
 }

 /// Embedded orphan protection (replaces standalone OrphanTracker).
 #[derive(Debug)]
 struct OrphanGuard {
     /// CLOIDs registered but not yet finalized
     expected_cloids: HashMap<String, (Instant, Option<u64>)>,
     /// OIDs finalized within protection window
     recently_finalized: HashMap<u64, Instant>,
     /// Potential orphans with first-seen time
     orphan_candidates: HashMap<u64, Instant>,
 }

 /// Result of finalize_order() - NEVER fails silently.
 #[derive(Debug, Clone, PartialEq)]
 pub enum FinalizeResult {
     /// Normal: pending found and converted to tracked
     FromPending,
     /// Recovery: pending not found (WS race), created directly
     DirectCreation { reason: &'static str },
     /// Idempotent: order already in active orders
     AlreadyTracked,
 }

 /// Result of process_fill() - includes position delta.
 #[derive(Debug, Clone)]
 pub struct FillResult {
     pub oid: u64,
     pub tid: u64,
     pub size: f64,
     pub price: f64,
     pub is_buy: bool,
     pub position_delta: f64,
     pub new_position: f64,
     pub order_complete: bool,
 }

 /// Result of reconcile() - actions to take.
 #[derive(Debug, Default)]
 pub struct ReconcileResult {
     /// Orphans that aged past grace period - should be cancelled
     pub orphans_to_cancel: Vec<u64>,
     /// Orders recovered from exchange state (weren't tracked locally)
     pub orders_recovered: usize,
     /// Stale orders removed (tracked locally but not on exchange)
     pub stale_removed: usize,
     /// New orphans detected this cycle (in grace period)
     pub new_orphans: usize,
 }

 1.2 Order Lifecycle Methods

 impl UnifiedOrderBook {
     // === Order Lifecycle ===

     /// Register pending order BEFORE API call. Returns CLOID.
     ///
     /// Call this for each order BEFORE the bulk order API call.
     pub fn register_pending(&mut self, side: Side, price: f64, size: f64) -> String {
         let cloid = Uuid::new_v4().to_string();
         let pending = PendingOrder::with_cloid(side, price, size, cloid.clone());

         // Store in pending map
         self.pending.insert(cloid.clone(), pending);

         // Register with orphan guard
         self.orphan_guard.expected_cloids.insert(cloid.clone(), (Instant::now(), None));

         cloid
     }

     /// Finalize pending→tracked after API response.
     ///
     /// **GUARANTEED to track order** - creates directly if pending not found.
     /// This is the KEY FIX: we never silently fail to track.
     pub fn finalize_order(
         &mut self,
         cloid: &str,
         oid: u64,
         resting_size: f64,
         side: Side,
         price: f64,
     ) -> FinalizeResult {
         // 1. Check if already tracked (idempotent)
         if self.orders.contains_key(&oid) {
             return FinalizeResult::AlreadyTracked;
         }

         // 2. Record CLOID→OID mapping (persists for late WS events!)
         self.cloid_to_oid.insert(cloid.to_string(), oid);

         // 3. Try to finalize from pending
         if let Some(pending) = self.pending.remove(cloid) {
             let order = TrackedOrder::with_cloid(
                 oid,
                 cloid.to_string(),
                 pending.side,
                 pending.price,
                 resting_size,
             );
             self.orders.insert(oid, order);
             self.orphan_guard.mark_finalized(cloid, oid);
             return FinalizeResult::FromPending;
         }

         // 4. RECOVERY: Pending not found (WS race consumed it) → create directly
         //    This is the fix for the root cause!
         let order = TrackedOrder::with_cloid(
             oid,
             cloid.to_string(),
             side,
             price,
             resting_size,
         );
         self.orders.insert(oid, order);
         self.orphan_guard.mark_finalized(cloid, oid);

         FinalizeResult::DirectCreation {
             reason: "pending consumed by WS race - recovered"
         }
     }

     /// Handle immediate fill (API returned filled=true, resting_size=0).
     pub fn handle_immediate_fill(
         &mut self,
         cloid: &str,
         oid: u64,
         filled_size: f64,
         fill_price: f64,
         is_buy: bool,
     ) {
         // Update position atomically
         let delta = if is_buy { filled_size } else { -filled_size };
         self.position += delta;
         self.fill_count += 1;

         // Remove from pending if present
         self.pending.remove(cloid);

         // Record CLOID→OID mapping
         self.cloid_to_oid.insert(cloid.to_string(), oid);

         // Mark finalized (order is already filled, no need to track)
         self.orphan_guard.mark_finalized(cloid, oid);
     }
 }

 1.3 Fill Processing (Unified Deduplication)

 impl UnifiedOrderBook {
     /// Process a fill. Returns None if duplicate, Some(details) if new.
     /// Atomically updates position + order state.
     pub fn process_fill(
         &mut self,
         oid: u64,
         tid: u64,
         size: f64,
         price: f64,
         is_buy: bool,
     ) -> Option<FillResult> {
         // 1. Global TID deduplication (single check!)
         if self.processed_tids.contains(&tid) {
             return None; // Duplicate
         }

         // Mark as processed (with FIFO eviction)
         self.processed_tids.insert(tid);
         self.tid_eviction.push_back(tid);
         while self.processed_tids.len() > self.config.max_tracked_tids {
             if let Some(old_tid) = self.tid_eviction.pop_front() {
                 self.processed_tids.remove(&old_tid);
             }
         }

         // 2. Update position atomically
         let position_delta = if is_buy { size } else { -size };
         self.position += position_delta;
         self.fill_count += 1;

         // 3. Update order state (if tracked)
         let order_complete = if let Some(order) = self.orders.get_mut(&oid) {
             order.record_fill_with_price(tid, size, price);
             let is_complete = order.is_filled();

             // Update state machine
             match order.state {
                 OrderState::Resting | OrderState::PartialFilled => {
                     if is_complete {
                         order.transition_to(OrderState::Filled);
                     } else {
                         order.transition_to(OrderState::PartialFilled);
                     }
                 }
                 OrderState::CancelPending | OrderState::CancelConfirmed => {
                     order.transition_to(OrderState::FilledDuringCancel);
                 }
                 _ => {}
             }
             is_complete
         } else {
             // Order not in tracking - this shouldn't happen after our fix,
             // but we still processed the fill correctly for position.
             tracing::warn!(oid = oid, tid = tid, "Fill for untracked order");
             false
         };

         Some(FillResult {
             oid,
             tid,
             size,
             price,
             is_buy,
             position_delta,
             new_position: self.position,
             order_complete,
         })
     }
 }

 1.4 State Queries & Reconciliation

 impl UnifiedOrderBook {
     // === State Queries ===

     pub fn get_order(&self, oid: u64) -> Option<&TrackedOrder> {
         self.orders.get(&oid)
     }

     pub fn get_order_mut(&mut self, oid: u64) -> Option<&mut TrackedOrder> {
         self.orders.get_mut(&oid)
     }

     pub fn get_active_oids(&self) -> HashSet<u64> {
         self.orders.keys().copied().collect()
     }

     pub fn position(&self) -> f64 {
         self.position
     }

     pub fn set_position(&mut self, pos: f64) {
         self.position = pos;
     }

     pub fn len(&self) -> usize {
         self.orders.len()
     }

     pub fn pending_count(&self) -> usize {
         self.pending.len()
     }

     /// Pending exposure for reduce-only checks.
     pub fn pending_exposure(&self, side: Side) -> f64 {
         self.orders.values()
             .filter(|o| o.side == side && o.is_active())
             .map(|o| o.remaining())
             .sum()
     }

     // === Reconciliation ===

     /// Reconcile with exchange state. Returns actions to take.
     pub fn reconcile(&mut self, exchange_oids: &HashSet<u64>) -> ReconcileResult {
         let mut result = ReconcileResult::default();
         let local_oids = self.get_active_oids();
         let protected = self.orphan_guard.protected_oids();

         // 1. Find orphans (on exchange but not local)
         let candidate_orphans: Vec<u64> = exchange_oids
             .difference(&local_oids)
             .copied()
             .collect();

         let (aged_orphans, new_orphans) = self.orphan_guard
             .filter_aged_orphans(&candidate_orphans);

         result.orphans_to_cancel = aged_orphans;
         result.new_orphans = new_orphans;

         // 2. Find stale (local but not on exchange, excluding terminal states)
         for &oid in local_oids.difference(exchange_oids) {
             if let Some(order) = self.orders.get(&oid) {
                 if !order.is_terminal() && !protected.contains(&oid) {
                     self.orders.remove(&oid);
                     result.stale_removed += 1;
                 }
             }
         }

         // 3. Cleanup expired entries
         self.orphan_guard.cleanup(&self.config);
         self.cleanup_filled_orders();

         result
     }

     fn cleanup_filled_orders(&mut self) {
         let now = Instant::now();
         let window = self.config.fill_window_duration;

         self.orders.retain(|_, order| {
             if order.is_terminal() && order.fill_window_expired(window, now) {
                 false // Remove
             } else {
                 true  // Keep
             }
         });
     }
 }

 1.5 OrphanGuard Implementation

 impl OrphanGuard {
     fn new() -> Self {
         Self {
             expected_cloids: HashMap::new(),
             recently_finalized: HashMap::new(),
             orphan_candidates: HashMap::new(),
         }
     }

     fn mark_finalized(&mut self, cloid: &str, oid: u64) {
         self.expected_cloids.remove(cloid);
         self.recently_finalized.insert(oid, Instant::now());
         self.orphan_candidates.remove(&oid);
     }

     fn protected_oids(&self) -> HashSet<u64> {
         let now = Instant::now();
         let mut protected = HashSet::new();

         // OIDs from expected CLOIDs
         for (_, maybe_oid) in self.expected_cloids.values() {
             if let Some(oid) = maybe_oid {
                 protected.insert(*oid);
             }
         }

         // Recently finalized (within protection period)
         protected.extend(self.recently_finalized.keys().copied());

         protected
     }

     fn filter_aged_orphans(&mut self, candidates: &[u64]) -> (Vec<u64>, usize) {
         let now = Instant::now();
         let protected = self.protected_oids();
         let grace = Duration::from_secs(5); // Use config

         let mut aged = Vec::new();
         let mut new_count = 0;

         for &oid in candidates {
             if protected.contains(&oid) {
                 continue;
             }

             let first_seen = self.orphan_candidates.entry(oid).or_insert_with(|| {
                 new_count += 1;
                 now
             });

             if now.duration_since(*first_seen) >= grace {
                 aged.push(oid);
             }
         }

         (aged, new_count)
     }

     fn cleanup(&mut self, config: &UnifiedBookConfig) {
         let now = Instant::now();

         // Expire old expected CLOIDs
         self.expected_cloids.retain(|_, (submitted, _)| {
             now.duration_since(*submitted) < config.expected_cloid_ttl
         });

         // Expire old recently finalized
         self.recently_finalized.retain(|_, finalized| {
             now.duration_since(*finalized) < config.finalized_protection
         });

         // Expire very old orphan candidates
         let stale_threshold = config.orphan_grace_period * 10;
         self.orphan_candidates.retain(|_, first_seen| {
             now.duration_since(*first_seen) < stale_threshold
         });
     }
 }

 Phase 2: Migrate MarketMaker to UnifiedOrderBook

 File: src/market_maker/mod.rs

 2.1 Replace Field Declarations

 // BEFORE (in MarketMaker struct):
 pub struct MarketMaker<S, E> {
     orders: OrderManager,
     position: PositionTracker,
     // ... (fill_deduplicator in safety.fill_processor)
 }

 // AFTER:
 pub struct MarketMaker<S, E> {
     order_book: UnifiedOrderBook,  // Single source of truth
     // position: REMOVED (now in order_book)
     // orders: REMOVED (now in order_book)
 }

 2.2 Migration Table

 | Old Pattern                                             | New Pattern                                        | Notes                      |
 |---------------------------------------------------------|----------------------------------------------------|----------------------------|
 | orders.add_pending_with_cloid(side, price, size, cloid) | order_book.register_pending(side, price, size)     | Returns CLOID              |
 | orders.finalize_pending_by_cloid(c, oid, sz)            | order_book.finalize_order(c, oid, sz, side, price) | GUARANTEED                 |
 | orders.process_fill(oid, tid, sz)                       | order_book.process_fill(oid, tid, sz, px, is_buy)  | Returns Option<FillResult> |
 | orders.get_order(oid)                                   | order_book.get_order(oid)                          | Same                       |
 | orders.get_order_mut(oid)                               | order_book.get_order_mut(oid)                      | Same                       |
 | orders.len()                                            | order_book.len()                                   | Same                       |
 | position.position()                                     | order_book.position()                              | UNIFIED                    |
 | position.process_fill(sz, is_buy)                       | (removed)                                          | Handled in process_fill()  |
 | fill_deduplicator.check_and_mark(tid)                   | (removed)                                          | Handled in process_fill()  |
 | orphan_tracker.register_expected_cloids(...)            | (automatic)                                        | Done in register_pending() |
 | orphan_tracker.mark_finalized(...)                      | (automatic)                                        | Done in finalize_order()   |

 2.3 Key Locations to Update

 | Line Range    | Current Code                                | Change                             |
 |---------------|---------------------------------------------|------------------------------------|
 | ~1691-1695    | finalize_pending_by_cloid() (ignore result) | Use finalize_order() (logs result) |
 | ~2037-2041    | finalize_pending_by_cloid() (ignore result) | Use finalize_order() (logs result) |
 | ~2261-2265    | finalize_pending_by_cloid() (ignore result) | Use finalize_order() (logs result) |
 | Fill handlers | process_fill() + position.process_fill()    | Single order_book.process_fill()   |
 | safety_sync   | Complex orphan detection                    | order_book.reconcile()             |

 2.4 Example Migration (place_single_order)

 // BEFORE:
 let cloid = Uuid::new_v4().to_string();
 self.orders.add_pending_with_cloid(side, price, size, cloid.clone());
 self.infra.orphan_tracker.register_expected_cloids(&[cloid.clone()]);

 // ... API call ...

 if let Some(c) = cloid {
     self.orders.finalize_pending_by_cloid(c, result.oid, result.resting_size);
     self.infra.orphan_tracker.mark_finalized(c, result.oid);
 }

 // AFTER:
 let cloid = self.order_book.register_pending(side, price, size);

 // ... API call ...

 if let Some(c) = cloid {
     let result = self.order_book.finalize_order(&c, result.oid, result.resting_size, side, price);
     match result {
         FinalizeResult::DirectCreation { reason } => {
             warn!(oid = result.oid, reason = reason, "Order recovered from race condition");
         }
         _ => {}
     }
 }

 Phase 3: Remove Deprecated Code

 3.1 Delete WsOrderStateManager

 Delete entire directory: src/market_maker/tracking/ws_order_state/

 This parallel tracking system was never fully integrated and duplicates functionality now unified in UnifiedOrderBook.

 3.2 Deprecate but Keep (for reference)

 Keep but mark deprecated:
 - src/market_maker/tracking/order_manager/ - Keep types (TrackedOrder, PendingOrder, Side, OrderState)
 - src/market_maker/tracking/position.rs - Can be deleted (position now in UnifiedOrderBook)
 - src/market_maker/fills/dedup.rs - Can be deleted (dedup now in UnifiedOrderBook)
 - src/market_maker/infra/orphan_tracker.rs - Can be deleted (merged into UnifiedOrderBook)

 Phase 4: Update Safety Sync

 File: src/market_maker/mod.rs (safety_sync method)

 async fn safety_sync(&mut self) -> Result<()> {
     // 1. Get exchange state
     let exchange_orders = self.info_client.open_orders(self.user_address).await?;
     let exchange_oids: HashSet<u64> = exchange_orders.iter()
         .filter(|o| o.coin == *self.config.asset)
         .map(|o| o.oid)
         .collect();

     // 2. Single reconciliation call handles everything
     let result = self.order_book.reconcile(&exchange_oids);

     // 3. Log state
     let local_active = self.order_book.len();
     if exchange_oids.len() != local_active || result.stale_removed > 0 || result.new_orphans > 0 {
         info!(
             exchange = exchange_oids.len(),
             local_active = local_active,
             stale_removed = result.stale_removed,
             orphans_in_grace = result.new_orphans,
             orphans_to_cancel = result.orphans_to_cancel.len(),
             "[SafetySync] State reconciliation"
         );
     }

     // 4. Cancel aged orphans
     for oid in result.orphans_to_cancel {
         warn!(oid = oid, "[SafetySync] Cancelling aged orphan");
         if let Err(e) = self.executor.cancel_order(&self.config.asset, oid).await {
             error!(oid = oid, error = %e, "[SafetySync] Failed to cancel orphan");
         }
     }

     // 5. Sync position with exchange if needed
     let exchange_position = self.get_exchange_position().await?;
     let local_position = self.order_book.position();
     if (exchange_position - local_position).abs() > 1e-9 {
         warn!(
             exchange = exchange_position,
             local = local_position,
             "[SafetySync] Position mismatch - syncing to exchange"
         );
         self.order_book.set_position(exchange_position);
     }

     Ok(())
 }

 Phase 5: Update Core Components

 File: src/market_maker/core/components.rs

 // BEFORE:
 pub struct Tier1Components {
     // ...
 }

 // AFTER: Remove orphan_tracker from InfraComponents
 pub struct InfraComponents {
     pub margin_sizer: MarginAwareSizer,
     pub prometheus: PrometheusMetrics,
     pub connection_health: ConnectionHealthMonitor,
     pub data_quality: DataQualityMonitor,
     pub exchange_limits: ExchangeLimits,
     // orphan_tracker: REMOVED (now in UnifiedOrderBook)
 }

 ---
 Files to Modify/Create

 | Priority | Action | File                                          | Description                                |
 |----------|--------|-----------------------------------------------|--------------------------------------------|
 | 1        | CREATE | src/market_maker/tracking/unified_book/mod.rs | UnifiedOrderBook + OrphanGuard             |
 | 2        | MODIFY | src/market_maker/tracking/mod.rs              | Export unified_book                        |
 | 3        | MODIFY | src/market_maker/mod.rs                       | Replace orders/position with order_book    |
 | 4        | MODIFY | src/market_maker/core/components.rs           | Remove orphan_tracker from InfraComponents |
 | 5        | DELETE | src/market_maker/tracking/ws_order_state/     | Remove dead parallel system                |
 | 6        | DELETE | src/market_maker/fills/dedup.rs               | Dedup now in UnifiedOrderBook              |
 | 7        | DELETE | src/market_maker/tracking/position.rs         | Position now in UnifiedOrderBook           |
 | 8        | DELETE | src/market_maker/infra/orphan_tracker.rs      | OrphanGuard now embedded                   |
 | 9        | MODIFY | src/market_maker/fills/processor.rs           | Use order_book.process_fill()              |
 | 10       | MODIFY | src/market_maker/messages/user_fills.rs       | Use order_book.process_fill()              |
 | 11       | MODIFY | src/market_maker/safety/auditor.rs            | Simplify to use reconcile()                |

 ---
 Migration Strategy

 Step 1: Create New Module (No Breaking Changes)

 1. Create unified_book/mod.rs with all types and methods
 2. Add export to tracking/mod.rs
 3. No changes to existing code yet

 Step 2: Add order_book Field (Dual-Write)

 1. Add order_book: UnifiedOrderBook field to MarketMaker
 2. Forward all writes to BOTH old and new systems
 3. Validate state matches on each quote cycle

 Step 3: Switch Reads

 1. Change position() calls to use order_book.position()
 2. Change orders.len() to order_book.len()
 3. Change fill processing to use order_book.process_fill()

 Step 4: Remove Old Systems

 1. Remove orders: OrderManager field
 2. Remove position: PositionTracker field
 3. Remove fill_deduplicator from FillProcessor
 4. Remove orphan_tracker from InfraComponents
 5. Delete deprecated files

 Step 5: Cleanup

 1. Delete ws_order_state/ directory
 2. Delete standalone dedup/position/orphan_tracker files
 3. Update all imports

 ---
 Unit Tests for UnifiedOrderBook

 #[cfg(test)]
 mod tests {
     use super::*;

     #[test]
     fn test_register_and_finalize_happy_path() {
         let mut book = UnifiedOrderBook::new();
         let cloid = book.register_pending(Side::Buy, 100.0, 1.0);

         let result = book.finalize_order(&cloid, 12345, 1.0, Side::Buy, 100.0);
         assert_eq!(result, FinalizeResult::FromPending);
         assert!(book.get_order(12345).is_some());
     }

     #[test]
     fn test_finalize_without_pending_creates_directly() {
         // Simulates WS race condition
         let mut book = UnifiedOrderBook::new();

         // Finalize without ever registering pending
         let result = book.finalize_order("unknown-cloid", 12345, 1.0, Side::Buy, 100.0);

         // Should create directly, not fail silently
         assert!(matches!(result, FinalizeResult::DirectCreation { .. }));
         assert!(book.get_order(12345).is_some());
     }

     #[test]
     fn test_finalize_idempotent() {
         let mut book = UnifiedOrderBook::new();
         let cloid = book.register_pending(Side::Buy, 100.0, 1.0);

         book.finalize_order(&cloid, 12345, 1.0, Side::Buy, 100.0);
         let result = book.finalize_order(&cloid, 12345, 1.0, Side::Buy, 100.0);

         assert_eq!(result, FinalizeResult::AlreadyTracked);
     }

     #[test]
     fn test_fill_deduplication() {
         let mut book = UnifiedOrderBook::new();
         let cloid = book.register_pending(Side::Buy, 100.0, 1.0);
         book.finalize_order(&cloid, 12345, 1.0, Side::Buy, 100.0);

         // First fill succeeds
         let r1 = book.process_fill(12345, 1, 0.5, 100.0, true);
         assert!(r1.is_some());
         assert!((book.position() - 0.5).abs() < f64::EPSILON);

         // Duplicate fill rejected
         let r2 = book.process_fill(12345, 1, 0.5, 100.0, true);
         assert!(r2.is_none());
         assert!((book.position() - 0.5).abs() < f64::EPSILON); // Position unchanged
     }

     #[test]
     fn test_position_updated_atomically_with_fill() {
         let mut book = UnifiedOrderBook::new();
         let cloid = book.register_pending(Side::Sell, 100.0, 1.0);
         book.finalize_order(&cloid, 12345, 1.0, Side::Sell, 100.0);

         let result = book.process_fill(12345, 1, 1.0, 100.0, false);
         assert!(result.is_some());

         let fill = result.unwrap();
         assert!((fill.position_delta - (-1.0)).abs() < f64::EPSILON);
         assert!((fill.new_position - (-1.0)).abs() < f64::EPSILON);
         assert!((book.position() - (-1.0)).abs() < f64::EPSILON);
     }

     #[test]
     fn test_reconcile_detects_orphans() {
         let mut book = UnifiedOrderBook::new();

         // Simulate exchange having orders we don't know about
         let exchange_oids: HashSet<u64> = [100, 200, 300].into_iter().collect();

         // First reconcile - orphans enter grace period
         let r1 = book.reconcile(&exchange_oids);
         assert_eq!(r1.new_orphans, 3);
         assert!(r1.orphans_to_cancel.is_empty());

         // Wait for grace period (in real test, use mock time)
         std::thread::sleep(Duration::from_secs(6));

         // Second reconcile - orphans aged past grace
         let r2 = book.reconcile(&exchange_oids);
         assert_eq!(r2.orphans_to_cancel.len(), 3);
     }
 }

 ---
 Success Criteria

 | Metric                                        | Before   | Target   |
 |-----------------------------------------------|----------|----------|
 | State mismatches (exchange=N, local_active=M) | Frequent | Zero     |
 | Orphan cancellations                          | Many     | < 1/hour |
 | "Modify failed on unknown order" errors       | Frequent | Zero     |
 | TID dedup systems                             | 3        | 1        |
 | CLOID→OID mappings                            | 2        | 1        |
 | Code lines in tracking                        | ~2500    | ~1500    |
 | Places to debug order state                   | 6+       | 1        |

 ---
 Risk Assessment

 | Risk                               | Likelihood       | Mitigation                             |
 |------------------------------------|------------------|----------------------------------------|
 | Regression in fill processing      | Medium           | Dual-write validation phase            |
 | Position drift                     | Low              | Periodic exchange sync in safety_sync  |
 | Missing edge case in state machine | Low              | Comprehensive unit tests               |
 | Build failures from imports        | High (temporary) | Incremental migration with cargo check |

 ---
 Estimated Effort

 | Phase                            | Effort     | Notes                    |
 |----------------------------------|------------|--------------------------|
 | Phase 1: Create UnifiedOrderBook | 2-3 hours  | Core implementation      |
 | Phase 2: Migrate MarketMaker     | 2-3 hours  | Many call sites          |
 | Phase 3: Remove deprecated       | 1 hour     | Delete + fix imports     |
 | Phase 4: Update safety_sync      | 30 min     | Simplification           |
 | Phase 5: Update components       | 30 min     | Remove fields            |
 | Testing                          | 1-2 hours  | Unit + manual validation |
 | Total                            | 7-10 hours | 