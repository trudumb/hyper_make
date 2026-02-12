# Asset Characteristics Reference

Known Hyperliquid perpetual assets with key trading characteristics for market making.

## Asset Table

| Asset | Binance Pair | Typical Spread | Typical Volume | Leverage | SpreadProfile | Notes |
|-------|-------------|---------------|----------------|----------|---------------|-------|
| BTC | btcusdt | 1-3 bps | $1B+ | 50x | Default | Best lead-lag signal, highest volume |
| ETH | ethusdt | 2-5 bps | $500M+ | 50x | Default | Strong lead-lag, second-most liquid |
| SOL | solusdt | 3-8 bps | $100M+ | 20x | Default | Moderate lead-lag, good volume |
| DOGE | dogeusdt | 5-15 bps | $50M+ | 20x | Default | Moderate volume, meme volatility |
| AVAX | avaxusdt | 5-12 bps | $30M+ | 20x | Default | Decent lead-lag |
| LINK | linkusdt | 5-12 bps | $30M+ | 20x | Default | Stable oracle token |
| ARB | arbusdt | 5-15 bps | $20M+ | 20x | Default | L2 ecosystem token |
| OP | opusdt | 5-15 bps | $20M+ | 20x | Default | L2 ecosystem token |
| SUI | suiusdt | 5-15 bps | $20M+ | 20x | Default | Newer L1 |
| APT | aptusdt | 5-15 bps | $15M+ | 20x | Default | Move-based L1 |
| NEAR | nearusdt | 5-15 bps | $15M+ | 20x | Default | Sharded L1 |
| ATOM | atomusdt | 5-15 bps | $15M+ | 20x | Default | Cosmos hub |
| DOT | dotusdt | 5-15 bps | $15M+ | 20x | Default | Polkadot ecosystem |
| XRP | xrpusdt | 3-8 bps | $50M+ | 20x | Default | High volume, tight spreads |
| ADA | adausdt | 5-12 bps | $20M+ | 20x | Default | Cardano ecosystem |
| WIF | wifusdt | 10-30 bps | $10M+ | 10x | Default | Meme coin, volatile |
| PEPE | pepeusdt | 10-30 bps | $10M+ | 10x | Default | Meme coin, volatile |
| BONK | bonkusdt | 10-30 bps | $5M+ | 10x | Default | Meme coin, volatile |
| HYPE | None | 10-30 bps | $10-50M | 5x | Hip3 | HL-native, no Binance, HIP-3 token |
| PURR | None | 15-50 bps | $1-10M | 3x | Hip3 | HL-native, no Binance, HIP-3 meme |
| JEFF | None | 20-80 bps | $1-5M | 3x | Hip3 | HL-native, no Binance, very thin |

## Key Observations

- **Binance lead-lag**: BTC has strongest signal (50-500ms lead), degrades for smaller assets
- **HIP-3 tokens**: No Binance pair means `use_lead_lag` and `use_cross_venue` MUST be false
- **Meme coins**: Higher spreads but also higher adverse selection risk and volatility
- **Leverage**: Higher leverage allows larger position per dollar of capital; lower leverage (3-5x on HIP-3) means more capital needed per unit of exposure
- **Volume**: Below $1M/day is a red flag for viability; fills will be infrequent and learning loops slow

## HIP-3 DEX Names

| DEX Name | Description | Collateral |
|----------|-------------|------------|
| hyna | Hyena DEX | USDE |
| felix | Felix DEX | USDC |

*Run `market_maker --list-dexs` to see current live DEXs on the network.*
