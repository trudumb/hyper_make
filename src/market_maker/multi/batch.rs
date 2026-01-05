//! Batch accumulator for multi-asset order operations.
//!
//! Accumulates order operations across multiple assets within a quote cycle,
//! then executes them efficiently in a single batch to minimize API calls.
//!
//! # Execution Order
//!
//! Operations are executed in the optimal order:
//! 1. **Cancels** - Free up order slots first
//! 2. **Modifies** - Update existing orders in place
//! 3. **Places** - Add new orders
//!
//! This ordering ensures we never exceed the order limit during reconciliation.
//!
//! # Example
//!
//! ```ignore
//! let mut batch = BatchAccumulator::new();
//!
//! // Accumulate operations from multiple assets
//! batch.add_cancel(btc_id, 12345);
//! batch.add_modify(eth_id, ModifySpec { oid: 67890, price: 3000.0, size: 0.1 });
//! batch.add_place(sol_id, OrderSpec { price: 150.0, size: 10.0, is_buy: true });
//!
//! // Execute all operations
//! let results = batch.flush(&executor).await;
//! ```

use std::collections::HashMap;
use std::time::Instant;

use crate::market_maker::infra::{CancelResult, ModifyResult, ModifySpec, OrderResult, OrderSpec};
use crate::market_maker::tracking::AssetId;

/// Entry for a cancel operation.
#[derive(Debug, Clone)]
pub struct CancelEntry {
    /// Asset this cancel belongs to.
    pub asset_id: AssetId,
    /// Order ID to cancel.
    pub oid: u64,
}

/// Entry for a modify operation.
#[derive(Debug, Clone)]
pub struct ModifyEntry {
    /// Asset this modify belongs to.
    pub asset_id: AssetId,
    /// Modification specification.
    pub spec: ModifySpec,
}

/// Entry for a place operation.
#[derive(Debug, Clone)]
pub struct PlaceEntry {
    /// Asset this place belongs to.
    pub asset_id: AssetId,
    /// Order specification.
    pub spec: OrderSpec,
}

/// Generic batch entry that can be any operation type.
#[derive(Debug, Clone)]
pub enum BatchEntry {
    Cancel(CancelEntry),
    Modify(ModifyEntry),
    Place(PlaceEntry),
}

/// Results from batch execution.
#[derive(Debug, Default)]
pub struct BatchResults {
    /// Cancel results: (asset_id, oid, result).
    pub cancels: Vec<(AssetId, u64, CancelResult)>,
    /// Modify results: (asset_id, spec, result).
    pub modifies: Vec<(AssetId, ModifySpec, ModifyResult)>,
    /// Place results: (asset_id, spec, result).
    pub places: Vec<(AssetId, OrderSpec, OrderResult)>,
    /// Execution duration.
    pub duration: std::time::Duration,
}

impl BatchResults {
    /// Total operations executed.
    pub fn total_ops(&self) -> usize {
        self.cancels.len() + self.modifies.len() + self.places.len()
    }

    /// Count of successful operations.
    pub fn successful_ops(&self) -> usize {
        let cancel_ok = self
            .cancels
            .iter()
            .filter(|(_, _, r)| r.order_is_gone())
            .count();
        let modify_ok = self.modifies.iter().filter(|(_, _, r)| r.success).count();
        let place_ok = self.places.iter().filter(|(_, _, r)| r.oid > 0).count();
        cancel_ok + modify_ok + place_ok
    }

    /// Count of failed operations.
    pub fn failed_ops(&self) -> usize {
        self.total_ops() - self.successful_ops()
    }

    /// Get failed modifies (for fallback to cancel+replace).
    pub fn failed_modifies(&self) -> Vec<(AssetId, ModifySpec)> {
        self.modifies
            .iter()
            .filter(|(_, _, r)| !r.success)
            .map(|(id, spec, _)| (*id, spec.clone()))
            .collect()
    }
}

/// Accumulates order operations for batched execution.
pub struct BatchAccumulator {
    /// Cancel operations.
    cancels: Vec<CancelEntry>,
    /// Modify operations.
    modifies: Vec<ModifyEntry>,
    /// Place operations.
    places: Vec<PlaceEntry>,
    /// Cycle start time.
    cycle_start: Instant,
}

impl BatchAccumulator {
    /// Create a new batch accumulator.
    pub fn new() -> Self {
        Self {
            cancels: Vec::with_capacity(100),
            modifies: Vec::with_capacity(200),
            places: Vec::with_capacity(200),
            cycle_start: Instant::now(),
        }
    }

    /// Add a cancel operation.
    pub fn add_cancel(&mut self, asset_id: AssetId, oid: u64) {
        self.cancels.push(CancelEntry { asset_id, oid });
    }

    /// Add a modify operation.
    pub fn add_modify(&mut self, asset_id: AssetId, spec: ModifySpec) {
        self.modifies.push(ModifyEntry { asset_id, spec });
    }

    /// Add a place operation.
    pub fn add_place(&mut self, asset_id: AssetId, spec: OrderSpec) {
        self.places.push(PlaceEntry { asset_id, spec });
    }

    /// Add multiple cancels.
    pub fn add_cancels(&mut self, cancels: impl IntoIterator<Item = (AssetId, u64)>) {
        for (asset_id, oid) in cancels {
            self.add_cancel(asset_id, oid);
        }
    }

    /// Add multiple modifies.
    pub fn add_modifies(&mut self, modifies: impl IntoIterator<Item = (AssetId, ModifySpec)>) {
        for (asset_id, spec) in modifies {
            self.add_modify(asset_id, spec);
        }
    }

    /// Add multiple places.
    pub fn add_places(&mut self, places: impl IntoIterator<Item = (AssetId, OrderSpec)>) {
        for (asset_id, spec) in places {
            self.add_place(asset_id, spec);
        }
    }

    /// Number of pending operations.
    pub fn pending_count(&self) -> usize {
        self.cancels.len() + self.modifies.len() + self.places.len()
    }

    /// Number of pending cancels.
    pub fn cancel_count(&self) -> usize {
        self.cancels.len()
    }

    /// Number of pending modifies.
    pub fn modify_count(&self) -> usize {
        self.modifies.len()
    }

    /// Number of pending places.
    pub fn place_count(&self) -> usize {
        self.places.len()
    }

    /// Check if batch is empty.
    pub fn is_empty(&self) -> bool {
        self.pending_count() == 0
    }

    /// Clear all pending operations.
    pub fn clear(&mut self) {
        self.cancels.clear();
        self.modifies.clear();
        self.places.clear();
        self.cycle_start = Instant::now();
    }

    /// Time since cycle start.
    pub fn cycle_duration(&self) -> std::time::Duration {
        self.cycle_start.elapsed()
    }

    /// Get cancels grouped by asset.
    pub fn cancels_by_asset(&self) -> HashMap<AssetId, Vec<u64>> {
        let mut by_asset: HashMap<AssetId, Vec<u64>> = HashMap::new();
        for entry in &self.cancels {
            by_asset.entry(entry.asset_id).or_default().push(entry.oid);
        }
        by_asset
    }

    /// Get modifies grouped by asset.
    pub fn modifies_by_asset(&self) -> HashMap<AssetId, Vec<ModifySpec>> {
        let mut by_asset: HashMap<AssetId, Vec<ModifySpec>> = HashMap::new();
        for entry in &self.modifies {
            by_asset
                .entry(entry.asset_id)
                .or_default()
                .push(entry.spec.clone());
        }
        by_asset
    }

    /// Get places grouped by asset.
    pub fn places_by_asset(&self) -> HashMap<AssetId, Vec<OrderSpec>> {
        let mut by_asset: HashMap<AssetId, Vec<OrderSpec>> = HashMap::new();
        for entry in &self.places {
            by_asset
                .entry(entry.asset_id)
                .or_default()
                .push(entry.spec.clone());
        }
        by_asset
    }

    /// Drain all entries (for manual execution).
    pub fn drain(&mut self) -> (Vec<CancelEntry>, Vec<ModifyEntry>, Vec<PlaceEntry>) {
        let cancels = std::mem::take(&mut self.cancels);
        let modifies = std::mem::take(&mut self.modifies);
        let places = std::mem::take(&mut self.places);
        self.cycle_start = Instant::now();
        (cancels, modifies, places)
    }

    /// Get all cancel OIDs (flat list for bulk cancel API).
    pub fn all_cancel_oids(&self) -> Vec<u64> {
        self.cancels.iter().map(|e| e.oid).collect()
    }

    /// Get summary for logging.
    pub fn summary(&self) -> String {
        format!(
            "BatchAccumulator {{ cancels: {}, modifies: {}, places: {}, duration: {:?} }}",
            self.cancels.len(),
            self.modifies.len(),
            self.places.len(),
            self.cycle_duration()
        )
    }
}

impl Default for BatchAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_order_spec(price: f64, size: f64, is_buy: bool) -> OrderSpec {
        OrderSpec::new(price, size, is_buy)
    }

    fn make_modify_spec(oid: u64, price: f64, size: f64, is_buy: bool) -> ModifySpec {
        ModifySpec {
            oid,
            new_price: price,
            new_size: size,
            is_buy,
        }
    }

    #[test]
    fn test_accumulator_basic() {
        let mut batch = BatchAccumulator::new();
        let btc = AssetId::new("BTC", None);
        let eth = AssetId::new("ETH", None);

        batch.add_cancel(btc, 12345);
        batch.add_modify(eth, make_modify_spec(67890, 3000.0, 0.1, true));
        batch.add_place(btc, make_order_spec(50000.0, 0.01, true));

        assert_eq!(batch.cancel_count(), 1);
        assert_eq!(batch.modify_count(), 1);
        assert_eq!(batch.place_count(), 1);
        assert_eq!(batch.pending_count(), 3);
    }

    #[test]
    fn test_group_by_asset() {
        let mut batch = BatchAccumulator::new();
        let btc = AssetId::new("BTC", None);
        let eth = AssetId::new("ETH", None);

        batch.add_cancel(btc, 1);
        batch.add_cancel(btc, 2);
        batch.add_cancel(eth, 3);

        let by_asset = batch.cancels_by_asset();

        assert_eq!(by_asset.get(&btc).map(|v| v.len()), Some(2));
        assert_eq!(by_asset.get(&eth).map(|v| v.len()), Some(1));
    }

    #[test]
    fn test_clear() {
        let mut batch = BatchAccumulator::new();
        let btc = AssetId::new("BTC", None);

        batch.add_cancel(btc, 1);
        batch.add_place(btc, make_order_spec(50000.0, 0.01, true));

        assert!(!batch.is_empty());

        batch.clear();

        assert!(batch.is_empty());
        assert_eq!(batch.pending_count(), 0);
    }

    #[test]
    fn test_drain() {
        let mut batch = BatchAccumulator::new();
        let btc = AssetId::new("BTC", None);

        batch.add_cancel(btc, 1);
        batch.add_cancel(btc, 2);

        let (cancels, modifies, places) = batch.drain();

        assert_eq!(cancels.len(), 2);
        assert!(modifies.is_empty());
        assert!(places.is_empty());
        assert!(batch.is_empty());
    }

    #[test]
    fn test_batch_results() {
        let mut results = BatchResults::default();

        // Add some mock results
        let btc = AssetId::new("BTC", None);
        results.cancels.push((btc, 1, CancelResult::Cancelled));
        results
            .cancels
            .push((btc, 2, CancelResult::AlreadyCancelled));

        assert_eq!(results.total_ops(), 2);
        assert_eq!(results.successful_ops(), 2); // Both are "success" in the sense of not failing
    }
}
