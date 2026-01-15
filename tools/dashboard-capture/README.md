# Dashboard Screenshot Capture

Automated screenshot capture of the trading dashboard for Claude vision analysis.

## Features

- Captures all 6 dashboard tabs every 5 seconds
- Optimized viewport (1400x788) for Claude vision token efficiency
- Timestamped archive format for historical analysis
- Automatic browser restart for long-running sessions
- Graceful shutdown handling (Ctrl+C)

## Prerequisites

1. **Node.js 18+** - ES modules support required
2. **Dashboard running** - HTML file served on port 3000
3. **Market maker/Paper trader** - WebSocket server on port 8080

## Quick Start

```bash
# Install dependencies
npm install

# Start capture (headless)
npm start

# Start with visible browser (debugging)
npm run start:visible
```

## Serving the Dashboard

```bash
# Option 1: Using serve
npx serve -l 3000 /path/to/mm-dashboard-fixed.html

# Option 2: Using Python
cd /path/to/project
python3 -m http.server 3000
```

## Configuration

All settings can be overridden via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `DASHBOARD_URL` | `http://localhost:3000` | Dashboard URL |
| `CAPTURE_INTERVAL_MS` | `5000` | Capture frequency (ms) |
| `TAB_SWITCH_DELAY_MS` | `500` | Delay after tab click (ms) |
| `VIEWPORT_WIDTH` | `1400` | Screenshot width (px) |
| `VIEWPORT_HEIGHT` | `788` | Screenshot height (px) |
| `OUTPUT_DIR` | `./screenshots` | Output directory |
| `TABS` | `overview,book,...` | Comma-separated tab list |
| `HEADLESS` | `true` | Run browser headless |
| `BROWSER_RESTART_CYCLES` | `100` | Restart browser after N cycles |

### Examples

```bash
# Capture only overview and pnl tabs
TABS=overview,pnl npm start

# Slower capture rate (10 seconds)
CAPTURE_INTERVAL_MS=10000 npm start

# Custom output directory
OUTPUT_DIR=/path/to/output npm start

# Full example
DASHBOARD_URL=http://localhost:3001 \
CAPTURE_INTERVAL_MS=10000 \
TABS=overview,book,pnl \
npm start
```

## Output Format

Screenshots are saved with this structure:

```
screenshots/
├── 2026-01-15/
│   ├── 10-30-00_overview.png
│   ├── 10-30-00_book.png
│   ├── 10-30-00_calibration.png
│   ├── 10-30-00_regime.png
│   ├── 10-30-00_signals.png
│   ├── 10-30-00_pnl.png
│   ├── 10-30-05_overview.png
│   └── ...
└── 2026-01-16/
    └── ...
```

## Claude Vision Integration

### Token Costs

- **Per image** (1400x788): ~1470 tokens
- **Per cycle** (6 tabs): ~8,820 tokens
- **Per minute** (12 cycles): ~105,840 tokens
- **Per hour**: ~6.35M tokens

### Recommended Prompt Structure

```python
import anthropic
import base64

client = anthropic.Anthropic()

# Load images
def load_image(path):
    with open(path, 'rb') as f:
        return base64.standard_b64encode(f.read()).decode('utf-8')

# Build message with images before text
message = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=4096,
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Dashboard snapshot at 10:30:00:"},
            {"type": "text", "text": "Tab 1 - Overview:"},
            {"type": "image", "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": load_image("screenshots/2026-01-15/10-30-00_overview.png")
            }},
            {"type": "text", "text": "Tab 2 - Order Book:"},
            {"type": "image", "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": load_image("screenshots/2026-01-15/10-30-00_book.png")
            }},
            # ... remaining tabs ...
            {"type": "text", "text": "Identify any anomalies or concerning patterns."}
        ]
    }]
)
```

### Files API for Repeated Analysis

For analyzing the same images multiple times:

```bash
# Upload once
curl -X POST https://api.anthropic.com/v1/files \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "anthropic-beta: files-api-2025-04-14" \
  -F "file=@screenshot.png"

# Returns file_id for reuse in messages
```

## Dashboard Tabs

| Tab | ID | Content |
|-----|----|---------|
| Overview | `overview` | PnL summary, regime history |
| Order Book | `book` | Book heatmap, price history, depth |
| Calibration | `calibration` | Fill/AS calibration scatter plots |
| Regime | `regime` | Regime probabilities over time |
| Signals | `signals` | Signal audit table, MI charts |
| PnL | `pnl` | PnL attribution breakdown |

## Troubleshooting

### Browser won't launch
```bash
# Install Chromium dependencies (Linux)
npx puppeteer browsers install chrome
```

### Dashboard not loading
1. Check dashboard is served: `curl http://localhost:3000`
2. Check WebSocket server: `curl http://localhost:8080/api/dashboard`

### Screenshots are blank
- Increase `TAB_SWITCH_DELAY_MS` to allow more time for charts to render
- Check for JavaScript errors in browser console

### Memory issues in long runs
- Reduce `BROWSER_RESTART_CYCLES` to restart browser more frequently
- Consider capturing fewer tabs or at lower frequency
