# Lead Agent Memory

## Project State
Post-architecture-redesign phase (Feb 25). Edge accountability + dual-timescale stochastic control implemented. 3300+ tests, clippy clean. Next: paper trading validation.

## Coordination State
- File ownership enforced via `.claude/ownership.json` + hooks
- `signal_integration.rs` → strategy-only
- `mod.rs` re-exports → lead-only
- Plan approval required for: `orchestrator/`, `risk/`, `safety/`, `exchange/`, `src/bin/`

## Key Decisions Made
- Additive spread composition (not multiplicative) — Feb 15
- `fills_measured() < N` as canonical warm-start indicator — Feb 17
- Capital-aware policy system with 4 tiers — Feb 16
- Dual-timescale decomposition resolving gamma-bifurcation — Feb 24
- Edge accountability through gamma (not spread addons) — Feb 25

## Open Work
- Paper trading validation of edge accountability changes
- RL agent wiring (observe-only, needs gamma/omega consumed by GLFT)
- Dashboard data pipeline recently fixed (Feb 27) — needs validation
