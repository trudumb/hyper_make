//! Structured logging infrastructure.
//!
//! This module provides multi-stream logging with:
//! - Component-based log targets for filtering
//! - Log rotation via tracing-appender
//! - Separate streams for operational, diagnostic, and error logs
//! - Performance-safe async writes
//!
//! # Component Targets
//!
//! Use these log targets for component-specific filtering:
//!
//! | Target | Description |
//! |--------|-------------|
//! | `market_maker::core` | Main orchestrator lifecycle |
//! | `market_maker::strategy` | Strategy decisions |
//! | `market_maker::estimator` | Parameter estimation (σ, κ, microprice) |
//! | `market_maker::risk` | Risk management & kill switch |
//! | `market_maker::execution` | Order placement/cancellation |
//! | `market_maker::fills` | Fill processing & P&L |
//! | `market_maker::tracking` | Position & order tracking |
//! | `market_maker::infra` | Infrastructure (rate limits, reconnection) |
//! | `market_maker::metrics` | Prometheus metrics emission |
//!
//! # Example Usage
//!
//! ```bash
//! # Debug only estimator module
//! RUST_LOG=hyperliquid_rust_sdk::market_maker::estimator=debug cargo run --bin market_maker
//!
//! # Warn for all, debug for risk
//! RUST_LOG=warn,hyperliquid_rust_sdk::market_maker::risk=debug cargo run --bin market_maker
//! ```

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tracing_appender::non_blocking::WorkerGuard;
use tracing_appender::rolling::{RollingFileAppender, Rotation};
use tracing_subscriber::filter::EnvFilter;
use tracing_subscriber::fmt;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::Layer;

/// Log output format.
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum LogFormat {
    /// Human-readable format with colors (default for development)
    #[default]
    Pretty,
    /// JSON format (best for log aggregation)
    Json,
    /// Compact single-line format
    Compact,
}

/// Logging configuration for the market maker.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LogConfig {
    /// Directory for log files (if multi-stream enabled)
    #[serde(default = "default_log_dir")]
    pub log_dir: PathBuf,

    /// Enable multi-stream logging (operational/diagnostic/errors)
    #[serde(default)]
    pub enable_multi_stream: bool,

    /// Log level for operational logs (INFO+ by default)
    #[serde(default = "default_operational_level")]
    pub operational_level: String,

    /// Log level for diagnostic logs (DEBUG+ by default)
    #[serde(default = "default_diagnostic_level")]
    pub diagnostic_level: String,

    /// Log level for error logs (WARN+ by default)
    #[serde(default = "default_error_level")]
    pub error_level: String,

    /// Enable stdout logging (default: true)
    #[serde(default = "default_enable_stdout")]
    pub enable_stdout: bool,

    /// Format for stdout logging
    #[serde(default)]
    pub stdout_format: LogFormat,

    /// Optional single log file path (legacy mode)
    /// When set, logs to this file instead of multi-stream
    #[serde(default)]
    pub log_file: Option<String>,
}

fn default_log_dir() -> PathBuf {
    PathBuf::from("logs")
}

fn default_operational_level() -> String {
    "info".to_string()
}

fn default_diagnostic_level() -> String {
    "debug".to_string()
}

fn default_error_level() -> String {
    "warn".to_string()
}

fn default_enable_stdout() -> bool {
    true
}

impl Default for LogConfig {
    fn default() -> Self {
        Self {
            log_dir: default_log_dir(),
            enable_multi_stream: false,
            operational_level: default_operational_level(),
            diagnostic_level: default_diagnostic_level(),
            error_level: default_error_level(),
            enable_stdout: default_enable_stdout(),
            stdout_format: LogFormat::default(),
            log_file: None,
        }
    }
}

impl LogConfig {
    /// Create a config for development (pretty stdout, no files).
    pub fn development() -> Self {
        Self::default()
    }

    /// Create a config for production (JSON, multi-stream files).
    pub fn production(log_dir: PathBuf) -> Self {
        Self {
            log_dir,
            enable_multi_stream: true,
            stdout_format: LogFormat::Json,
            ..Default::default()
        }
    }

    /// Create a config with a single log file (legacy mode).
    pub fn with_log_file(log_file: String) -> Self {
        Self {
            log_file: Some(log_file),
            ..Default::default()
        }
    }
}

/// Initialize logging based on configuration.
///
/// Returns a vector of `WorkerGuard` that must be kept alive for the duration
/// of the program to ensure logs are flushed.
///
/// # Example
///
/// ```ignore
/// let config = LogConfig::production(PathBuf::from("logs"));
/// let _guards = init_logging(&config, None)?;
/// // Guards must be kept in scope for logging to work
/// ```
pub fn init_logging(
    config: &LogConfig,
    env_filter_override: Option<&str>,
) -> Result<Vec<WorkerGuard>, Box<dyn std::error::Error>> {
    let mut guards = Vec::new();

    // Build the base filter from RUST_LOG or override
    let base_filter = if let Some(filter) = env_filter_override {
        EnvFilter::new(filter)
    } else {
        EnvFilter::try_from_default_env().unwrap_or_else(|_| {
            EnvFilter::new("info")
                .add_directive("hyper=warn".parse().unwrap())
                .add_directive("reqwest=warn".parse().unwrap())
                .add_directive("tokio_tungstenite=warn".parse().unwrap())
        })
    };

    if config.enable_multi_stream {
        // Multi-stream mode: separate files for operational, diagnostic, errors
        std::fs::create_dir_all(&config.log_dir)?;

        // Operational log (INFO+, all components)
        let operational_appender =
            RollingFileAppender::new(Rotation::DAILY, &config.log_dir, "mm-operational.log");
        let (operational_writer, guard1) = tracing_appender::non_blocking(operational_appender);
        guards.push(guard1);

        let operational_filter = EnvFilter::new(&config.operational_level)
            .add_directive("hyper=warn".parse().unwrap())
            .add_directive("reqwest=warn".parse().unwrap());

        let operational_layer = fmt::layer()
            .with_writer(operational_writer)
            .with_ansi(false)
            .json()
            .with_filter(operational_filter);

        // Diagnostic log (DEBUG+, estimator/tracking/fills components)
        let diagnostic_appender =
            RollingFileAppender::new(Rotation::DAILY, &config.log_dir, "mm-diagnostic.log");
        let (diagnostic_writer, guard2) = tracing_appender::non_blocking(diagnostic_appender);
        guards.push(guard2);

        // Filter for diagnostic components only
        let diagnostic_filter = EnvFilter::new(
            "hyperliquid_rust_sdk::market_maker::estimator=debug,\
             hyperliquid_rust_sdk::market_maker::tracking=debug,\
             hyperliquid_rust_sdk::market_maker::fills=debug,\
             hyperliquid_rust_sdk::market_maker::process_models=debug",
        );

        let diagnostic_layer = fmt::layer()
            .with_writer(diagnostic_writer)
            .with_ansi(false)
            .json()
            .with_filter(diagnostic_filter);

        // Error log (WARN+, all components)
        let error_appender =
            RollingFileAppender::new(Rotation::DAILY, &config.log_dir, "mm-errors.log");
        let (error_writer, guard3) = tracing_appender::non_blocking(error_appender);
        guards.push(guard3);

        let error_filter = EnvFilter::new(&config.error_level);

        let error_layer = fmt::layer()
            .with_writer(error_writer)
            .with_ansi(false)
            .json()
            .with_filter(error_filter);

        // Stdout layer (optional)
        if config.enable_stdout {
            match config.stdout_format {
                LogFormat::Json => {
                    tracing_subscriber::registry()
                        .with(operational_layer)
                        .with(diagnostic_layer)
                        .with(error_layer)
                        .with(fmt::layer().json().with_filter(base_filter))
                        .init();
                }
                LogFormat::Compact => {
                    tracing_subscriber::registry()
                        .with(operational_layer)
                        .with(diagnostic_layer)
                        .with(error_layer)
                        .with(fmt::layer().compact().with_filter(base_filter))
                        .init();
                }
                LogFormat::Pretty => {
                    tracing_subscriber::registry()
                        .with(operational_layer)
                        .with(diagnostic_layer)
                        .with(error_layer)
                        .with(fmt::layer().with_target(false).with_filter(base_filter))
                        .init();
                }
            }
        } else {
            tracing_subscriber::registry()
                .with(operational_layer)
                .with(diagnostic_layer)
                .with(error_layer)
                .init();
        }

        eprintln!("Multi-stream logging enabled: {}", config.log_dir.display());
    } else if let Some(ref log_file) = config.log_file {
        // Single file mode (legacy) - always use JSON for both file and stdout
        // This ensures type compatibility between layers
        let file = std::fs::File::create(log_file)?;
        let file = std::sync::Mutex::new(file);

        let file_layer = fmt::layer().with_writer(file).with_ansi(false).json();

        let stdout_layer = fmt::layer().json();

        tracing_subscriber::registry()
            .with(base_filter)
            .with(stdout_layer)
            .with(file_layer)
            .init();

        eprintln!(
            "Logging to file: {} (using JSON format for both stdout and file)",
            log_file
        );
    } else {
        // Stdout only mode
        match config.stdout_format {
            LogFormat::Json => {
                tracing_subscriber::fmt()
                    .with_env_filter(base_filter)
                    .json()
                    .init();
            }
            LogFormat::Compact => {
                tracing_subscriber::fmt()
                    .with_env_filter(base_filter)
                    .compact()
                    .init();
            }
            LogFormat::Pretty => {
                tracing_subscriber::fmt()
                    .with_env_filter(base_filter)
                    .with_target(false)
                    .init();
            }
        }
    }

    Ok(guards)
}

/// Log target constants for component-specific logging.
///
/// Use these with the `target:` field in tracing macros:
/// ```ignore
/// tracing::debug!(target: LOG_TARGET_ESTIMATOR, sigma = %sigma, "Volatility updated");
/// ```
pub mod targets {
    /// Main orchestrator lifecycle
    pub const CORE: &str = "market_maker::core";
    /// Strategy decisions (GLFT, ladder)
    pub const STRATEGY: &str = "market_maker::strategy";
    /// Parameter estimation (σ, κ, microprice)
    pub const ESTIMATOR: &str = "market_maker::estimator";
    /// Risk management & kill switch
    pub const RISK: &str = "market_maker::risk";
    /// Order placement/cancellation
    pub const EXECUTION: &str = "market_maker::execution";
    /// Fill processing & P&L
    pub const FILLS: &str = "market_maker::fills";
    /// Position & order tracking
    pub const TRACKING: &str = "market_maker::tracking";
    /// Infrastructure (rate limits, reconnection)
    pub const INFRA: &str = "market_maker::infra";
    /// Prometheus metrics emission
    pub const METRICS: &str = "market_maker::metrics";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_config_defaults() {
        let config = LogConfig::default();
        assert!(!config.enable_multi_stream);
        assert!(config.enable_stdout);
        assert_eq!(config.stdout_format, LogFormat::Pretty);
        assert!(config.log_file.is_none());
    }

    #[test]
    fn test_log_config_production() {
        let config = LogConfig::production(PathBuf::from("/var/log/mm"));
        assert!(config.enable_multi_stream);
        assert_eq!(config.stdout_format, LogFormat::Json);
        assert_eq!(config.log_dir, PathBuf::from("/var/log/mm"));
    }

    #[test]
    fn test_log_config_with_file() {
        let config = LogConfig::with_log_file("mm.log".to_string());
        assert!(!config.enable_multi_stream);
        assert_eq!(config.log_file, Some("mm.log".to_string()));
    }

    #[test]
    fn test_log_format_serde() {
        let json = serde_json::to_string(&LogFormat::Json).unwrap();
        assert_eq!(json, "\"json\"");

        let parsed: LogFormat = serde_json::from_str("\"compact\"").unwrap();
        assert_eq!(parsed, LogFormat::Compact);
    }
}
