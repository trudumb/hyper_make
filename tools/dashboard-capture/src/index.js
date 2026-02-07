#!/usr/bin/env node
// src/index.js - Main entry point for dashboard screenshot capture
import { config } from '../config.js';
import { DashboardCapturer } from './capturer.js';
import { log, ensureOutputDir } from './utils.js';

async function main() {
  log('info', '='.repeat(60));
  log('info', 'Dashboard Screenshot Capture for Claude Vision');
  log('info', '='.repeat(60));
  log('info', `Dashboard URL: ${config.dashboardUrl}`);
  log('info', `Capture interval: ${config.captureIntervalMs}ms`);
  log('info', `Viewport: ${config.viewportWidth}x${config.viewportHeight} (~${Math.round(config.viewportWidth * config.viewportHeight / 750)} tokens/image)`);
  log('info', `Tabs: ${config.tabsToCapture.join(', ')}`);
  log('info', `Output: ${config.outputDir}`);
  log('info', `Headless: ${config.headless}`);
  log('info', '='.repeat(60));

  // Ensure output directory exists
  await ensureOutputDir(config.outputDir);

  // Create capturer instance
  const capturer = new DashboardCapturer(config);

  // Graceful shutdown handling
  let isShuttingDown = false;

  const shutdown = async (signal) => {
    if (isShuttingDown) return;
    isShuttingDown = true;

    log('info', `Received ${signal}, shutting down gracefully...`);
    await capturer.stop();
    log('info', 'Shutdown complete');
    process.exit(0);
  };

  process.on('SIGINT', () => shutdown('SIGINT'));
  process.on('SIGTERM', () => shutdown('SIGTERM'));

  // Handle uncaught errors
  process.on('uncaughtException', (error) => {
    log('error', `Uncaught exception: ${error.message}`);
    console.error(error.stack);
    shutdown('uncaughtException');
  });

  process.on('unhandledRejection', (reason) => {
    log('error', `Unhandled rejection: ${reason}`);
    shutdown('unhandledRejection');
  });

  // Start capture loop
  try {
    await capturer.start();
  } catch (error) {
    log('error', `Fatal error: ${error.message}`);
    console.error(error.stack);
    await capturer.stop();
    process.exit(1);
  }
}

main().catch((error) => {
  console.error('Failed to start:', error);
  process.exit(1);
});
