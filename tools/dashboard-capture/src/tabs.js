// src/tabs.js - Tab definitions with selectors and wait conditions
// Based on mm-dashboard-fixed.html structure

/**
 * Tab definitions matching the dashboard React component
 * Each tab has:
 * - id: URL-safe identifier used in filenames
 * - label: Button text to click (case-sensitive match)
 * - waitSelector: Element to wait for after tab switch
 * - waitForContent: Optional async function to wait for specific content
 */
export const TABS = {
  overview: {
    id: 'overview',
    label: 'Overview',
    waitSelector: 'canvas',
  },

  book: {
    id: 'book',
    label: 'Order Book',
    waitSelector: 'canvas',
  },

  calibration: {
    id: 'calibration',
    label: 'Calibration',
    waitSelector: 'canvas',
  },

  regime: {
    id: 'regime',
    label: 'Regime',
    waitSelector: 'canvas',
  },

  signals: {
    id: 'signals',
    label: 'Signals',
    waitSelector: 'table',
  },

  pnl: {
    id: 'pnl',
    label: 'PnL',
    waitSelector: 'canvas',
  }
};

/**
 * Click a tab button by its label text
 * Uses text content matching since tab buttons are rendered by React
 * @param {import('puppeteer').Page} page
 * @param {string} tabLabel
 */
export async function clickTab(page, tabLabel) {
  // Find button containing the exact tab label
  const clicked = await page.evaluate((label) => {
    const buttons = Array.from(document.querySelectorAll('button'));
    const targetButton = buttons.find(btn => btn.textContent.trim() === label);
    if (targetButton) {
      targetButton.click();
      return true;
    }
    return false;
  }, tabLabel);

  if (!clicked) {
    throw new Error(`Tab button not found: ${tabLabel}`);
  }
}
