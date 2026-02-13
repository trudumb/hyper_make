# Market Maker Strategy Redesign: Foundational Assumptions

Each section examines one of five core strategies as if it were the **foundational assumption** of the entire system — restructuring the architecture so that strategy becomes the central pillar influencing all other components.

---

## 1. Spread as Primary Risk Buffer

**Existing approach:** Spread widening is an add-on calculation based on volatility forecasts, applied after computing a base spread. Spread is treated as a secondary adjustment rather than the core mechanism for handling uncertainty.

**Redesign premise:** The system is a *risk-pricing engine* where the spread is the starting point. All quoting derives from a spread-optimization function that dynamically scales with expected price swings. Fixed bases are eliminated; the spread becomes the heartbeat of liquidity provision.

### Key Properties

- Spread is computed **first** from a GARCH volatility forecast
- A configurable `vol_multiplier` scales conditional volatility into the core spread
- A `min_spread` floor prevents collapse in low-vol regimes
- Inventory modulates spread **directly** — wider when imbalanced, not via a separate overlay
- Quotes are outputs of the spread, not inputs to it

### Reference Implementation

```python
import pandas as pd
import numpy as np
from arch import arch_model

def spread_centric_market_maker(prices, vol_multiplier=2, min_spread=0.005, inventory_limit=100):
    """
    Spread is computed first from vol, then quotes are built around it,
    with inventory influencing spread asymmetry.
    """
    df = pd.DataFrame({'price': prices})
    df['returns'] = np.log(df['price'] / df['price'].shift(1)).dropna()

    # Core: Fit GARCH for vol, making spread the primary output
    model = arch_model(df['returns'], vol='Garch', p=1, q=1)
    fitted = model.fit(disp='off')
    vol_forecast = fitted.conditional_volatility.iloc[-1]
    core_spread = max(min_spread, vol_multiplier * vol_forecast)

    # Inventory modulates spread directly (wider if imbalanced)
    inventory = 0
    spread_adjust = abs(inventory) / inventory_limit * core_spread * 0.5 if inventory else 0

    # Quotes derived from spread, with mid-price as anchor
    mid_price = df['price'].iloc[-1]
    bid = mid_price - (core_spread / 2 + spread_adjust if inventory > 0 else core_spread / 2)
    ask = mid_price + (core_spread / 2 + spread_adjust if inventory < 0 else core_spread / 2)

    return {
        'bid': bid, 'ask': ask,
        'spread': core_spread + spread_adjust,
        'forecasted_vol': vol_forecast,
        'inventory': inventory
    }
```

### Architectural Implications

- Eliminates the concept of a "base spread" — all spread is risk-derived
- Volatility model becomes a first-class dependency, not a side calculation
- Inventory management is embedded in the spread computation, not layered on top
- Naturally efficient in volatile regimes where spread scales with risk

---

## 2. Asymmetry as Core Pricing Principle

**Existing approach:** Skew is added as a post-processing step based on trend detection. Symmetry is the default assumption, making skew a patch for directional markets.

**Redesign premise:** Symmetric quoting is abandoned entirely. The system is a *leaning engine* that always incorporates probabilistic trend signals into bid/ask placement. The mid-price is offset by expected drift, minimizing inventory buildup by design.

### Key Properties

- All markets are treated as **potentially directional** — neutral is a special case, not the default
- Momentum (e.g., MA crossover) drives a `core_skew` that offsets the mid-price
- Inventory reinforces the lean — amplifying skew when imbalanced in the direction of momentum
- The skewed mid-price is the anchor; spread is applied symmetrically around it

### Reference Implementation

```python
import pandas as pd
import numpy as np

def skew_centric_market_maker(prices, base_spread=0.01, skew_factor=0.005, inventory_limit=100):
    """
    Asymmetry based on trend/momentum is the default,
    with inventory reinforcing the lean.
    """
    df = pd.DataFrame({'price': prices})

    # Core: Compute trend skew first (MA crossover for momentum)
    short_ma = df['price'].rolling(5).mean().iloc[-1]
    long_ma = df['price'].rolling(20).mean().iloc[-1]
    momentum = (short_ma - long_ma) / long_ma
    core_skew = skew_factor * momentum

    # Inventory integrates into skew (amplify lean if imbalanced)
    inventory = 0
    inventory_skew = (inventory / inventory_limit) * skew_factor * 2

    total_skew = core_skew + inventory_skew

    # Quotes built asymmetrically around skewed mid
    mid_price = df['price'].iloc[-1] + total_skew
    bid = mid_price - base_spread / 2
    ask = mid_price + base_spread / 2

    return {
        'bid': bid, 'ask': ask,
        'spread': ask - bid,
        'total_skew': total_skew,
        'inventory': inventory
    }
```

### Architectural Implications

- Removes the notion of a "neutral" mid-price — all pricing is directional
- Trend/momentum signals are upstream of quoting, not downstream adjustments
- Inventory management becomes a natural extension of skew rather than a separate system
- Particularly effective in trending markets; requires dampening logic for mean-reverting regimes

---

## 3. Inventory as System Governor

**Existing approach:** Inventory checks are threshold-based overrides applied late in the pipeline, allowing imbalances to build before intervention.

**Redesign premise:** The system is an *inventory-first balancer*. Position limits dictate all operations. Quoting is conditional on current inventory, with algorithms optimizing for zero-net exposure as the primary objective. Spreads and skews are tools to enforce limits, preventing overloads proactively.

### Key Properties

- **Hard stop**: if `abs(inventory) >= inventory_limit`, quoting is paused entirely
- `exposure_ratio` (`abs(inventory) / inventory_limit`) is the master scaling variable
- Spread widens proportionally to exposure ratio — reducing fill probability as risk grows
- Skew is proportional to exposure ratio and direction — actively pushing toward rebalance
- All other logic is subordinate to inventory state

### Reference Implementation

```python
import pandas as pd
import numpy as np

def inventory_centric_market_maker(prices, base_spread=0.01, inventory_limit=100, max_adjust=0.02):
    """
    All quoting is gated by inventory state,
    with adjustments to force rebalancing.
    """
    df = pd.DataFrame({'price': prices})
    mid_price = df['price'].iloc[-1]

    # Core: Inventory is the governor
    inventory = 50

    if abs(inventory) >= inventory_limit:
        return {'status': 'paused', 'reason': 'Inventory limit exceeded'}

    # Adjustments scale with inventory proximity to limit
    exposure_ratio = abs(inventory) / inventory_limit
    spread = base_spread * (1 + exposure_ratio)
    skew = max_adjust * exposure_ratio * (1 if inventory > 0 else -1)

    bid = mid_price - spread / 2 + skew
    ask = mid_price + spread / 2 + skew

    return {
        'bid': bid, 'ask': ask,
        'spread': spread, 'skew': skew,
        'inventory': inventory,
        'exposure_ratio': exposure_ratio
    }
```

### Architectural Implications

- Quoting becomes a **constrained optimization** problem, not a free-form calculation
- Inventory state is checked before any computation — fail-fast on limits
- Spread and skew are instruments of the inventory governor, not independent concerns
- Eliminates the risk of late-stage overrides conflicting with earlier logic
- Aligns with the safety-first principle: missing a trade is cheap, getting run over is not

---

## 4. Hedging as Integrated Neutralizer

**Existing approach:** Hedging is mentioned but not implemented. It is treated as an optional external step rather than a core system component.

**Redesign premise:** The system is a *hedged liquidity provider*. All positions are automatically paired with offsets in correlated instruments. Inventory is actively neutralized via delta hedging, making the maker directionally indifferent by design. This requires a multi-asset architecture with hedging logic embedded in the quoting loop.

### Key Properties

- Every inventory position triggers an **immediate hedge** in a correlated asset (e.g., futures)
- `hedge_ratio` controls the fraction of inventory offset (e.g., 0.8 = 80% hedged)
- `residual_risk` = `inventory + hedge_position` — the unhedged remainder
- Spread and skew are functions of residual risk, not raw inventory
- Quoting becomes more aggressive when well-hedged (residual near zero)

### Reference Implementation

```python
import pandas as pd
import numpy as np

def hedging_centric_market_maker(prices, correlated_prices, base_spread=0.01, hedge_ratio=0.8):
    """
    Inventory is auto-hedged using correlated asset,
    with quotes adjusted based on post-hedge residual risk.
    """
    df = pd.DataFrame({'price': prices, 'corr_price': correlated_prices})
    mid_price = df['price'].iloc[-1]

    # Core: Simulate inventory and immediate hedge
    inventory = 50
    hedge_position = -inventory * hedge_ratio
    residual_risk = inventory + hedge_position

    # Adjust spread/skew based on residual (wider/more skewed if unhedged)
    risk_factor = abs(residual_risk) / abs(inventory) if inventory else 0
    spread = base_spread * (1 + risk_factor)
    skew = risk_factor * 0.005 * (1 if residual_risk > 0 else -1)

    bid = mid_price - spread / 2 + skew
    ask = mid_price + spread / 2 + skew

    return {
        'bid': bid, 'ask': ask,
        'spread': spread,
        'inventory': inventory,
        'hedge_position': hedge_position,
        'residual_risk': residual_risk
    }
```

### Architectural Implications

- Requires a **multi-asset** data pipeline — correlated instrument prices must be available in the quoting loop
- Risk is measured in residual terms, not gross inventory — fundamentally changes how position limits work
- Enables tighter spreads when hedging is effective, improving competitiveness
- Introduces hedge execution as a new latency-sensitive path alongside quoting
- Correlation stability becomes a critical monitoring target (hedge breaks down if correlation shifts)

---

## 5. Regimes as State Machine

**Existing approach:** Volatility forecasting and simple MA-based trend detection are used, but regime detection is peripheral — it does not drive state transitions or parameter selection.

**Redesign premise:** The system is a *regime-adaptive automaton* operating as a finite state machine. Detected regimes (mean-reversion, trending, high-vol) are the entry point. All parameters switch based on the current state, with quoting as state-dependent functions for seamless transitions.

### Key Properties

- **Three regimes** detected from volatility and trend strength:
  - `mean_reversion` — low vol, low trend: tight spread, no skew
  - `trending` — high trend strength: wider spread, directional skew
  - `high_vol` — elevated volatility: wide spread, neutral skew
- Regime detection runs **before** any quoting logic
- Each regime defines its own parameter set (spread multiplier, skew direction/magnitude)
- Transitions between regimes are implicit — re-evaluated on every tick

### Reference Implementation

```python
import pandas as pd
import numpy as np
from arch import arch_model

def regime_centric_market_maker(prices, base_spread=0.01, vol_threshold=0.015):
    """
    System operates in states (low-vol, trending, high-vol),
    with quoting params per regime.
    """
    df = pd.DataFrame({'price': prices})
    df['returns'] = np.log(df['price'] / df['price'].shift(1)).dropna()

    # Core: Detect regime first
    model = arch_model(df['returns'], vol='Garch', p=1, q=1)
    fitted = model.fit(disp='off')
    vol_forecast = fitted.conditional_volatility.iloc[-1]

    short_ma = df['price'].rolling(5).mean().iloc[-1]
    long_ma = df['price'].rolling(20).mean().iloc[-1]
    trend_strength = abs(short_ma - long_ma) / long_ma

    if vol_forecast < vol_threshold and trend_strength < 0.01:
        regime = 'mean_reversion'
        spread = base_spread * 0.5
        skew = 0
    elif trend_strength > 0.02:
        regime = 'trending'
        spread = base_spread * 1.5
        skew = 0.005 * (1 if short_ma > long_ma else -1)
    else:
        regime = 'high_vol'
        spread = base_spread * 2
        skew = 0

    mid_price = df['price'].iloc[-1]
    bid = mid_price - spread / 2 + skew
    ask = mid_price + spread / 2 + skew

    return {
        'bid': bid, 'ask': ask,
        'spread': spread,
        'regime': regime,
        'forecasted_vol': vol_forecast
    }
```

### Architectural Implications

- The system is a **state machine** — regime detection is the transition function
- All parameter tuning is per-regime, eliminating single-value assumptions
- Aligns with the core rule: "everything is regime-dependent"
- Regime persistence/hysteresis should be added to prevent rapid oscillation between states
- Enables clear monitoring: log regime transitions and per-regime P&L for calibration
- Natural extension to HMM-based detection for probabilistic regime assignment

---

## Cross-Cutting Observations

| Strategy | Central Abstraction | Primary Benefit | Key Risk |
|---|---|---|---|
| Spread as Risk Buffer | Volatility-derived spread | Scales naturally with uncertainty | Over-widening in calm markets loses volume |
| Asymmetric Pricing | Directional mid-price | Reduces adverse inventory buildup | Wrong-way lean amplifies losses |
| Inventory Governor | Exposure ratio | Prevents catastrophic position sizes | Over-conservative limits reduce profitability |
| Integrated Hedging | Residual risk after offset | Enables tighter spreads via risk transfer | Hedge correlation breakdown |
| Regime State Machine | Finite state detection | Adapts all params to market condition | Misclassification causes wrong parameter set |

### Composing the Strategies

In practice, these are not mutually exclusive. A production system composes them in a dependency order:

1. **Regime detection** runs first — determines which parameter set is active
2. **Spread computation** uses regime-specific volatility scaling
3. **Skew/asymmetry** applies regime-specific momentum signals
4. **Inventory governor** constrains the resulting quotes against position limits
5. **Hedging** neutralizes residual risk after fills

This ordering ensures each strategy operates on well-defined inputs from the layer above, minimizing coupling and making each component independently testable and calibratable.
