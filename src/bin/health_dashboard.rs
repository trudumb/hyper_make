//! Market Maker Health Dashboard
//!
//! Terminal-based real-time health monitoring.
//!
//! Usage:
//!   health_dashboard [OPTIONS]
//!
//! Options:
//!   --refresh <MS>     Refresh interval in milliseconds (default: 1000)

use clap::Parser;
use std::io::{self, Write};
use std::thread;
use std::time::Duration;

#[derive(Parser)]
#[command(name = "health_dashboard")]
#[command(about = "Real-time market maker health dashboard")]
struct Cli {
    /// Refresh interval in milliseconds
    #[arg(short, long, default_value = "1000")]
    refresh: u64,

    /// Demo mode with simulated data
    #[arg(long)]
    demo: bool,
}

/// Simulated health state for demonstration
struct HealthState {
    // Position
    position: f64,
    position_notional: f64,
    pnl_bps: f64,
    pnl_usd: f64,

    // Regime
    regime: &'static str,
    regime_confidence: f64,

    // Circuit Breakers
    circuit_breaker_active: bool,
    breaker_type: Option<&'static str>,

    // Model Health
    fill_model_ir: f64,
    as_model_ir: f64,
    regime_model_ir: f64,

    // Execution
    fill_rate: f64,
    latency_p50: f64,
    latency_p99: f64,

    // Alerts
    alerts: Vec<String>,

    // Uptime
    uptime_secs: u64,
}

impl HealthState {
    fn demo() -> Self {
        Self {
            position: 0.05,
            position_notional: 2150.0,
            pnl_bps: 12.5,
            pnl_usd: 45.30,
            regime: "Normal",
            regime_confidence: 0.72,
            circuit_breaker_active: false,
            breaker_type: None,
            fill_model_ir: 1.32,
            as_model_ir: 1.18,
            regime_model_ir: 1.05,
            fill_rate: 0.65,
            latency_p50: 45.0,
            latency_p99: 120.0,
            alerts: vec![],
            uptime_secs: 3600,
        }
    }

    fn update_demo(&mut self) {
        // Simulate small changes
        use std::time::{SystemTime, UNIX_EPOCH};
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .subsec_millis();
        let noise = (seed as f64 / 1000.0 - 0.5) * 0.1;

        self.position += noise * 0.01;
        self.position_notional = self.position.abs() * 43000.0;
        self.pnl_bps += noise * 2.0;
        self.pnl_usd += noise * 5.0;
        self.regime_confidence = (self.regime_confidence + noise * 0.05).clamp(0.5, 0.95);
        self.fill_rate = (self.fill_rate + noise * 0.1).clamp(0.3, 0.9);
        self.uptime_secs += 1;
    }
}

fn main() {
    let cli = Cli::parse();
    let refresh = Duration::from_millis(cli.refresh);

    let mut state = HealthState::demo();

    // Clear screen
    print!("\x1B[2J\x1B[1;1H");

    loop {
        // Move cursor to top
        print!("\x1B[1;1H");

        render_dashboard(&state);

        io::stdout().flush().unwrap();

        state.update_demo();
        thread::sleep(refresh);
    }
}

fn render_dashboard(state: &HealthState) {
    let position_color = if state.position > 0.0 {
        "\x1B[32m"
    // green
    } else if state.position < 0.0 {
        "\x1B[31m"
    // red
    } else {
        "\x1B[37m"
    }; // white

    let pnl_color = if state.pnl_bps > 0.0 {
        "\x1B[32m"
    } else {
        "\x1B[31m"
    };

    let regime_color = match state.regime {
        "Low" => "\x1B[34m",     // blue
        "Normal" => "\x1B[32m",  // green
        "High" => "\x1B[33m",    // yellow
        "Extreme" => "\x1B[31m", // red
        _ => "\x1B[37m",
    };

    let reset = "\x1B[0m";

    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║              MARKET MAKER HEALTH DASHBOARD                           ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!(
        "║ Uptime: {:5}s                                                       ║",
        state.uptime_secs
    );
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!("║ POSITION & P&L                                                       ║");
    println!(
        "║ Position: {}{:+8.4} BTC{} (${:.0})                                   ║",
        position_color, state.position, reset, state.position_notional
    );
    println!(
        "║ P&L:      {}{:+8.2} bps{} (${:+.2})                                   ║",
        pnl_color, state.pnl_bps, reset, state.pnl_usd
    );
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!("║ REGIME                                                               ║");
    println!(
        "║ Current: {}{:8}{} (confidence: {:5.1}%)                            ║",
        regime_color,
        state.regime,
        reset,
        state.regime_confidence * 100.0
    );

    // Regime probability bar
    let bar_len = 40;
    let filled = (state.regime_confidence * bar_len as f64) as usize;
    let bar: String = "\u{2588}".repeat(filled) + &"\u{2591}".repeat(bar_len - filled);
    println!(
        "║ [{}]                            ║",
        bar
    );

    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!("║ CIRCUIT BREAKERS                                                     ║");
    if state.circuit_breaker_active {
        println!(
            "║ Status: \x1B[31m⚠ ACTIVE - {}\x1B[0m                                       ║",
            state.breaker_type.unwrap_or("Unknown")
        );
    } else {
        println!("║ Status: \x1B[32m✓ All Clear\x1B[0m                                              ║");
    }

    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!("║ MODEL HEALTH (Information Ratio)                                     ║");
    println!(
        "║ Fill Model:   {} │ AS Model:   {} │ Regime:   {}         ║",
        format_ir(state.fill_model_ir),
        format_ir(state.as_model_ir),
        format_ir(state.regime_model_ir)
    );

    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!("║ EXECUTION                                                            ║");
    println!(
        "║ Fill Rate: {:5.1}%  │  Latency P50: {:5.1}ms  │  P99: {:5.1}ms        ║",
        state.fill_rate * 100.0,
        state.latency_p50,
        state.latency_p99
    );

    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!("║ ALERTS                                                               ║");
    if state.alerts.is_empty() {
        println!("║ \x1B[32mNo active alerts\x1B[0m                                                    ║");
    } else {
        for alert in &state.alerts {
            println!(
                "║ \x1B[31m⚠ {}\x1B[0m                                                     ║",
                alert
            );
        }
    }
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!("\nPress Ctrl+C to exit");
}

fn format_ir(ir: f64) -> String {
    let color = if ir >= 1.2 {
        "\x1B[32m"
    // green
    } else if ir >= 1.0 {
        "\x1B[33m"
    // yellow
    } else {
        "\x1B[31m"
    }; // red
    format!("{}{:4.2}\x1B[0m", color, ir)
}
