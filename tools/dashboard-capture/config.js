// config.js - Configuration with environment variable overrides
// Optimized for Claude vision analysis (max 1.15MP images)

export const config = {
  // Dashboard connection
  dashboardUrl: process.env.DASHBOARD_URL || 'http://localhost:3000/mm-dashboard-fixed.html',
  wsUrl: process.env.WS_URL || 'ws://localhost:9090/ws/dashboard',

  // Capture settings
  captureIntervalMs: parseInt(process.env.CAPTURE_INTERVAL_MS) || 5000,
  tabSwitchDelayMs: parseInt(process.env.TAB_SWITCH_DELAY_MS) || 100,
  pageLoadTimeoutMs: parseInt(process.env.PAGE_LOAD_TIMEOUT_MS) || 30000,

  // Viewport - Claude vision optimized (max 1.15MP before auto-scaling)
  // 1400x788 = ~1.1MP = ~1470 tokens per image
  viewportWidth: parseInt(process.env.VIEWPORT_WIDTH) || 1400,
  viewportHeight: parseInt(process.env.VIEWPORT_HEIGHT) || 788,

  // Output
  outputDir: process.env.OUTPUT_DIR || './screenshots',

  // Tabs to capture (can be subset for debugging)
  tabsToCapture: (process.env.TABS || 'overview,book,calibration,regime,signals,pnl').split(','),

  // Browser settings
  headless: process.env.HEADLESS !== 'false',
  browserRestartCycles: parseInt(process.env.BROWSER_RESTART_CYCLES) || 100,

  // Retry settings
  maxRetries: parseInt(process.env.MAX_RETRIES) || 3,
  retryDelayMs: parseInt(process.env.RETRY_DELAY_MS) || 1000,
};
