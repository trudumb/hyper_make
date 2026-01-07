# Session: 2026-01-06 FIFO Queue Priority Rules

## Summary
Documented Hyperliquid's price-time priority (FIFO) order book mechanics, specifically clarifying what preserves vs loses queue position during order modifications.

## Key Finding: Queue Priority Preservation Rules

### Actions That PRESERVE Queue Position
| Action | Result |
|--------|--------|
| Reduce size | ✅ Queue position preserved |
| Wait | ✅ Preserved (with natural decay) |
| Partial fill | ✅ Preserved for remaining size |

### Actions That LOSE Queue Position  
| Action | Result |
|--------|--------|
| Change price | ❌ Goes to back of new price level |
| Increase size | ❌ Treated as new order |
| Cancel + replace | ❌ Fresh order at back |

## Hyperliquid Documentation Reference
From Context7 query of official docs:
> "Orders are matched in price-time priority"

The `modify` API behavior:
- **Price change** = loses time priority (back of new level)
- **Size reduction only** = keeps time priority ✅

## Implementation Implications

### Current Queue Tracker Gap
The `QueuePositionTracker` in `src/market_maker/tracking/queue/tracker.rs` doesn't have an explicit `on_order_modified` method that distinguishes between:
1. Price modifications (should reset queue position)
2. Size-only reductions (should preserve queue position)

### Recommended Enhancement
```rust
/// Handle order modification - only price changes lose queue priority
pub fn order_modified(&mut self, oid: u64, old_price: f64, new_price: f64, new_size: f64) {
    if let Some(position) = self.positions.get_mut(&oid) {
        let price_changed = (old_price - new_price).abs() > 1e-10;
        
        if price_changed {
            // Price changed - reset to back of queue at new level
            position.price = new_price;
            position.depth_ahead = /* estimate depth at new level */;
            position.placed_at = Instant::now(); // Reset queue time
        } else {
            // Size-only change - preserve queue position
            position.size = new_size;
            // depth_ahead unchanged - we keep our spot!
        }
    }
}
```

## Market Making Value

Queue position = "free edge" earned by waiting:
- Front of queue at $100.00 often beats back of queue at $100.01
- Size-only modifications should be strongly preferred over cancel/replace
- The reconciliation logic should check if a size reduction suffices before repricing

## Files Relevant
- `src/market_maker/tracking/queue/tracker.rs` - Queue position tracking
- `src/market_maker/tracking/queue/position.rs` - Queue state types
- `src/market_maker/infra/executor.rs` - Order modification execution

## Next Steps
1. Add `order_modified()` method to `QueuePositionTracker`
2. Update order reconciliation to prefer size-only modifications
3. Add metrics to track queue preservation rate vs price modifications
