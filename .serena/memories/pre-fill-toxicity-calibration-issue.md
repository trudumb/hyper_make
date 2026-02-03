# Pre-Fill Toxicity Classifier Calibration Issue

## Discovery Date: 2026-02-03

## Problem Statement
The Pre-Fill AS Classifier has a **critical blind spot on bid-side toxicity**.

### Evidence
From live test on hyna:HYPE (16 fills):
- 6 bid fills with 0-2% toxicity signal had -50 bps actual markout
- 1 ask fill with 67% toxicity signal had -12 bps actual markout
- **Classifier recall on toxic fills: 14%**

### Hypothesis
The classifier may be trained primarily on ask-side toxic fills (e.g., during sell-offs, cascades). Bid-side toxicity (e.g., during short squeezes, liquidation hunts) has different feature signatures that the classifier hasn't learned.

### Features to Investigate
1. **Directional asymmetry**: Add explicit bid/ask feature
2. **OI change direction**: Increasing OI + bid fills = different than decreasing OI
3. **Funding rate sign**: Positive funding + bid fills = crowded long
4. **Recent fill direction**: Ratio of recent buys vs sells

### Files to Modify
- `src/market_maker/adverse_selection/pre_fill_classifier.rs`
- `src/market_maker/strategy/params/adverse_selection.rs`

### Validation Approach
1. Collect 50+ fills with signal diagnostic infrastructure
2. Compute Brier score separately for bid vs ask fills
3. If bid Brier >> ask Brier, confirms asymmetry
4. Retrain with directional features

### Priority
**HIGH** - 44% of fills are toxic, losing -16.57 bps average. Fixing this could flip PnL positive.
