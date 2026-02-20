# Implement RFC Extensions for Spread Extraction (Additive Restructure)

- [ ] Phase 2: Asymmetric Quote Staleness
  - [ ] Add `StalenessConfig` in [config/risk.rs](file:///home/jcritch22/projects/hyper_make/src/market_maker/config/risk.rs) (use `max_addon_bps` instead of `max_mult`).
  - [ ] Add staleness addons in [market_params.rs](file:///home/jcritch22/projects/hyper_make/src/market_maker/strategy/market_params.rs) and [aggregator.rs](file:///home/jcritch22/projects/hyper_make/src/market_maker/risk/aggregator.rs).
  - [ ] Compute per-side staleness addons in [quote_engine.rs](file:///home/jcritch22/projects/hyper_make/src/market_maker/orchestrator/quote_engine.rs).
- [ ] Phase 3: Directional Flow Toxicity
  - [ ] Add per-side toxicity tracking in [informed_flow.rs](file:///home/jcritch22/projects/hyper_make/src/market_maker/estimator/informed_flow.rs).
  - [ ] Compute toxicity spread addons (e.g., up to +15 bps) in [quote_engine.rs](file:///home/jcritch22/projects/hyper_make/src/market_maker/orchestrator/quote_engine.rs).
- [ ] Phase 4a: Governor Hardening
  - [ ] Configure thresholds via `GovernorConfig` in [config/risk.rs](file:///home/jcritch22/projects/hyper_make/src/market_maker/config/risk.rs).
  - [ ] Change [inventory_governor.rs](file:///home/jcritch22/projects/hyper_make/src/market_maker/risk/inventory_governor.rs) to use `increasing_side_addon_bps` instead of [spread_mult](file:///home/jcritch22/projects/hyper_make/src/market_maker/risk/inventory_governor.rs#499-514).
- [ ] Phase 4b: Drift Hardening
  - [ ] Add `boost_responsiveness` in [drift_estimator.rs](file:///home/jcritch22/projects/hyper_make/src/market_maker/strategy/drift_estimator.rs).
  - [ ] Call boost upon cascade event in [handlers.rs](file:///home/jcritch22/projects/hyper_make/src/market_maker/orchestrator/handlers.rs).
- [ ] Refactor Ladder Strat (Additive Cap)
  - [ ] Replace remaining governor MULTIPLIER logic in [ladder_strat.rs](file:///home/jcritch22/projects/hyper_make/src/market_maker/strategy/ladder_strat.rs) with additive sum capping.
  - [ ] Combine governor, cascade, staleness, and toxicity addons into a single capped additive adjustment per side.
- [ ] Testing and Validation

ensure you are efficient with resources as possible
