// src/capturer.js - Browser management and screenshot capture loop
import puppeteer from 'puppeteer';
import path from 'path';
import fs from 'fs/promises';
import { TABS, clickTab } from './tabs.js';
import { log, ensureOutputDir, formatDate, formatTime, sleep } from './utils.js';

export class DashboardCapturer {
  constructor(config) {
    this.config = config;
    this.browser = null;
    this.page = null;
    this.isRunning = false;
    this.cycleCount = 0;
    // Session ID from start time (matches MM log naming convention)
    this.sessionId = formatTime(new Date());
  }

  /**
   * Start the capture loop
   */
  async start() {
    this.isRunning = true;
    await this.launchBrowser();
    await this.navigateToDashboard();

    // Main capture loop
    while (this.isRunning) {
      const cycleStart = Date.now();

      try {
        await this.captureAllTabs();
        this.cycleCount++;

        // Restart browser periodically to prevent memory leaks
        if (this.cycleCount >= this.config.browserRestartCycles) {
          log('info', 'Restarting browser for memory cleanup...');
          await this.restartBrowser();
          this.cycleCount = 0;
        }
      } catch (error) {
        log('error', `Capture cycle failed: ${error.message}`);
        await this.handleError(error);
      }

      // Wait for next interval
      const elapsed = Date.now() - cycleStart;
      const waitTime = Math.max(0, this.config.captureIntervalMs - elapsed);

      if (waitTime > 0 && this.isRunning) {
        await sleep(waitTime);
      }
    }
  }

  /**
   * Stop the capture loop and close browser
   */
  async stop() {
    this.isRunning = false;
    if (this.browser) {
      await this.browser.close();
      this.browser = null;
      this.page = null;
    }
  }

  /**
   * Launch headless browser
   */
  async launchBrowser() {
    log('info', 'Launching browser...');

    this.browser = await puppeteer.launch({
      headless: this.config.headless,
      args: [
        '--no-sandbox',
        '--disable-setuid-sandbox',
        '--disable-dev-shm-usage',
        '--disable-gpu',
      ]
    });

    this.page = await this.browser.newPage();

    // Set viewport optimized for Claude vision (max 1.15MP)
    await this.page.setViewport({
      width: this.config.viewportWidth,
      height: this.config.viewportHeight,
    });

    log('info', `Browser launched (viewport: ${this.config.viewportWidth}x${this.config.viewportHeight})`);
  }

  /**
   * Navigate to dashboard and wait for it to load
   */
  async navigateToDashboard() {
    log('info', `Navigating to ${this.config.dashboardUrl}...`);

    await this.page.goto(this.config.dashboardUrl, {
      waitUntil: 'networkidle2',
      timeout: this.config.pageLoadTimeoutMs,
    });

    // Wait for React to render and WebSocket to connect
    await this.waitForDashboardReady();

    log('info', 'Dashboard loaded successfully');
  }

  /**
   * Wait for dashboard to be ready (React rendered + WebSocket connected)
   */
  async waitForDashboardReady() {
    // Wait for the header to be present (React rendered)
    await this.page.waitForSelector('header', { timeout: 10000 });

    // Wait for connection indicator (pulse-glow animation class)
    try {
      await this.page.waitForFunction(
        () => {
          const indicator = document.querySelector('.pulse-glow');
          return indicator !== null;
        },
        { timeout: 10000 }
      );
    } catch {
      log('warn', 'Connection indicator not found, proceeding anyway');
    }

    // Give WebSocket time to receive initial data
    await sleep(1000);
  }

  /**
   * Capture screenshots of all configured tabs
   */
  async captureAllTabs() {
    const timestamp = new Date();
    const dateDir = formatDate(timestamp);
    const timePrefix = formatTime(timestamp);

    // Ensure date/session directory exists
    const outputPath = path.join(this.config.outputDir, dateDir, this.sessionId);
    await ensureOutputDir(outputPath);

    log('info', `Starting capture cycle at ${timePrefix}...`);

    let capturedCount = 0;
    for (const tabId of this.config.tabsToCapture) {
      const tab = TABS[tabId];
      if (!tab) {
        log('warn', `Unknown tab: ${tabId}, skipping`);
        continue;
      }

      try {
        await this.captureTab(tab, outputPath, timePrefix);
        capturedCount++;
      } catch (error) {
        log('error', `Failed to capture ${tabId}: ${error.message}`);
        // Continue with other tabs
      }
    }

    log('info', `Capture cycle complete: ${capturedCount}/${this.config.tabsToCapture.length} tabs`);
  }

  /**
   * Wait for browser to complete painting (React render + canvas draw)
   * Uses double-RAF to ensure layout + paint are both complete
   */
  async waitForPaint() {
    await this.page.evaluate(() => {
      return new Promise(resolve => {
        // Double requestAnimationFrame ensures paint is complete
        requestAnimationFrame(() => {
          requestAnimationFrame(() => {
            resolve();
          });
        });
      });
    });
  }

  /**
   * Capture a single tab (async file write - non-blocking)
   * @param {object} tab - Tab definition from TABS
   * @param {string} outputPath - Base directory (date folder)
   * @param {string} timePrefix - Time prefix for filename
   */
  async captureTab(tab, outputPath, timePrefix) {
    // Click the tab button
    await clickTab(this.page, tab.label);

    // Brief delay for React state update
    await sleep(this.config.tabSwitchDelayMs);

    // Wait for paint to complete (replaces slow element counting)
    await this.waitForPaint();

    // Create tab-specific subfolder: screenshots/YYYY-MM-DD/tab_name/
    const tabDir = path.join(outputPath, tab.id);
    await ensureOutputDir(tabDir);

    // Capture screenshot to buffer (fast - in memory)
    const buffer = await this.page.screenshot({
      type: 'png',
      fullPage: false,
      encoding: 'binary',
    });

    // Write to disk async (fire-and-forget - non-blocking)
    const filename = `${timePrefix}.png`;
    const filepath = path.join(tabDir, filename);
    fs.writeFile(filepath, buffer).catch(err => {
      log('error', `Failed to write ${tab.id}/${filename}: ${err.message}`);
    });

    log('debug', `Captured: ${tab.id}/${filename}`);
  }

  /**
   * Restart browser to free memory
   */
  async restartBrowser() {
    await this.browser.close();
    await this.launchBrowser();
    await this.navigateToDashboard();
  }

  /**
   * Handle errors during capture
   */
  async handleError(error) {
    // Check if browser is still alive
    if (!this.browser.isConnected()) {
      log('warn', 'Browser disconnected, restarting...');
      await this.launchBrowser();
      await this.navigateToDashboard();
      return;
    }

    // Check if page is responsive
    try {
      await this.page.evaluate(() => true);
    } catch {
      log('warn', 'Page unresponsive, reloading...');
      await this.navigateToDashboard();
    }
  }
}
