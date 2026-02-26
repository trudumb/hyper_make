# RFC Supplement: From Entropy Heuristics to Endogenous Impact

**Status**: Draft  
**Context**: An in-depth analysis of how the current Entropy-Based Distribution system in `hyper_make` maps to the Clean-Slate Architecture (Engineer 5), and the theoretical path to replacing it with Endogenous Impact Modeling.

---

## 1. The Core Problem: Concentration Collapse

A critical problem in high-frequency market making is determining how to distribute order size across the depth of the limit order book (the "ladder").

If an algorithm is purely greedy, it calculates the Expected PnL (`E[PnL]`) for every depth increment. If a specific depth (e.g., 5 bps from mid) has the highest expected return, a naive optimizer will concentrate 100% of its allowed risk or capital at that single price level. 

This leads to **Concentration Collapse**:
- The market maker places a single massive order.
- This creates obvious signals for informed traders (predictable and gameable).
- If the market moves adversely, the entire position is picked off instantly at a single price point.

To survive, a market maker *must* disperse its orders across the book, maintaining a ladder even if the theoretical `E[PnL]` at deeper levels is sub-optimal compared to the touch.

---

## 2. The Current Solution: The Entropy Heuristic

Currently, `hyper_make` solves the Concentration Collapse problem using an **Entropy-Based Distribution System** (`src/market_maker/quoting/ladder/entropy_distribution.rs`).

This maps to **Engineer 5** in the Clean-Slate Architecture setup, acting as the logic that prevents the bot from stacking all its risk at a single price.

### How it Works:
1. **Utility Ranking:** It calculates a utility score for each depth level based on fill probability, spread capture, and toxicity.
2. **Softmax with Temperature:** It converts these utilities into a probability distribution using a temperature-scaled softmax function. High temperature = spread out; Low temperature = concentrated.
3. **Shannon Entropy Floor:** It calculates the Shannon Entropy (`H = -Σ p * ln(p)`) of the resulting distribution. **Crucially, if the entropy falls below a hard-coded floor (e.g., `H = 1.5`), the system iteratively increases the temperature until the distribution is forced to spread out.**
4. **Thompson Sampling:** It uses a Dirichlet prior and Thompson sampling to introduce stochastic noise, making the exact size and placement slightly randomized to prevent adversaries from perfectly predicting our ladder.

### Why It's a Heuristic (Not First-Principles)
While highly effective at preventing collapse, the Entropy system is a mathematical trick. It enforces diversity because *we know diversity is safe*, not because the underlying economic math dictated it. The system literally says: *"I want to put all my size at 5 bps, but the config file demands my Entropy must be at least 1.5, so I am forced to allocate to 10 bps and 15 bps as well."*

---

## 3. The Clean-Slate Ideal: Endogenous Impact

In the **Clean-Slate Foundational Architecture**, we strive for everything to be derived from First-Principles Mathematics (Options-Theoretic Pricing). Artificial constraints like "Minimum Entropy = 1.5" should not exist. 

Instead of artificially forcing the ladder to spread out, the clean-slate architecture achieves natural dispersion by correctly modeling **Endogenous Impact** (Engineer 4).

### What is Endogenous Impact?
If you place a $100,000 buy order at 5 bps, you are no longer an infinitesimal participant. You *are* the market. Placing that order changes the latent fair value of the asset.
1. **Forward Impact:** Other participants see your massive order, assume you know something, and front-run you.
2. **Round-Trip Impact (Slippage):** If you get filled on that $100,000, you will eventually have to exit. Exiting a concentrated position of that size will incur massive slippage.

### How Endogenous Impact Naturally Spreads the Ladder
In the clean-slate architecture, the Actuary (Engineer 4) subtracts the Endogenous Impact penalty from the overall `E[PnL]`. 

Crucially, **Impact scales non-linearly (usually close to the square root) with order size at a single price level.**

If you attempt to allocate your entire risk budget $Q$ to the touch (e.g., 5 bps):
1. The expected instantaneous edge is high.
2. BUT the impact penalty of placing $Q$ at one level is massive.
3. `E[PnL]` = `Edge - Adverse Selection - Impact(Q)`.
4. Because `Impact(Q)` is so large, `E[PnL]` at 5 bps goes negative.

If you instead distribute $Q$ across five levels ($Q/5$ at 5 bps, $Q/5$ at 10 bps, etc.):
1. `Impact(Q/5)` is drastically lower per level.
2. The total sum of `E[PnL]` across the five dispersed levels is now strictly positive and mathematically higher than concentrating at a single level.

**Conclusion:** A Convex Optimizer (Clean-Slate Engineer 5) natively disperses orders across the limit order book simply by maximizing true `Sum(E[PnL])`. It doesn't need to know what "Entropy" is; the economics of self-impact automatically punish concentration and reward dispersion.

---

## 4. The Transition Path

To move `hyper_make` from the current Entropy heuristic to the Clean-Slate Endogenous Impact model, the following architectural steps would be taken:

### Step 1: Implement the Impact Cost Function
Introduce a new module in `adverse_selection/` or `strategy/` that calculates continuous self-impact. 
`Impact(size, depth) = c * volatility * sqrt(size / average_daily_volume_at_depth)`

### Step 2: Refactor the Utility Payout Surface
Modify the input into the optimizer. Instead of `LevelOptimizationParams` providing a static `spread_capture`, it must provide a continuous payout curve: `E[PnL](size)`.

### Step 3: Replace Entropy Projection with Convex Optimization
Deprecate the Shannon Entropy projection loop (`project_to_min_entropy`) in `entropy_distribution.rs`. 
Replace it with a constrained convex optimizer (like a lightweight SciPy `minimize_slsqp` equivalent in Rust) that solves:
`Maximize: Σ (E[PnL]_i(size_i))` 
`Subject to: Σ size_i <= max_position`

### Step 4: Retain Thompson Sampling for Stealth
While the endogenous impact solves the *dispersion* problem natively, it remains completely deterministic. To prevent adversaries from reverse-engineering the exact shape of our impact function, the Thompson Sampling (Dirichlet Prior) mechanism from the current Entropy system should be kept as a final obfuscation layer applied over the optimal economic allocation.
