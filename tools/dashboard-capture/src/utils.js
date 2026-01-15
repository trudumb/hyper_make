// src/utils.js - Utility functions
import fs from 'fs/promises';
import path from 'path';

// Log levels with timestamps and colors
const LOG_LEVELS = {
  debug: { color: '\x1b[90m', label: 'DEBUG' },
  info: { color: '\x1b[36m', label: 'INFO' },
  warn: { color: '\x1b[33m', label: 'WARN' },
  error: { color: '\x1b[31m', label: 'ERROR' },
};

/**
 * Log a message with timestamp and level
 * @param {'debug'|'info'|'warn'|'error'} level
 * @param {string} message
 */
export function log(level, message) {
  const config = LOG_LEVELS[level] || LOG_LEVELS.info;
  const timestamp = new Date().toISOString();
  const reset = '\x1b[0m';

  console.log(`${config.color}[${timestamp}] [${config.label}]${reset} ${message}`);
}

/**
 * Ensure a directory exists, creating it if necessary
 * @param {string} dirPath
 */
export async function ensureOutputDir(dirPath) {
  try {
    await fs.mkdir(dirPath, { recursive: true });
  } catch (error) {
    if (error.code !== 'EEXIST') {
      throw error;
    }
  }
}

/**
 * Format date as YYYY-MM-DD
 * @param {Date} date
 * @returns {string}
 */
export function formatDate(date) {
  return date.toISOString().split('T')[0];
}

/**
 * Format time as HH-MM-SS
 * @param {Date} date
 * @returns {string}
 */
export function formatTime(date) {
  return date.toTimeString().slice(0, 8).replace(/:/g, '-');
}

/**
 * Sleep for a given number of milliseconds
 * @param {number} ms
 * @returns {Promise<void>}
 */
export function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}
