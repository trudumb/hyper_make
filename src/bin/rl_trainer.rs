//! Offline RL trainer binary.
//!
//! Reads experience JSONL files from paper/live trading, performs multi-epoch
//! Q-learning replay, and outputs a trained RLCheckpoint for deployment.
//!
//! Usage:
//! ```
//! cargo run --release --bin rl_trainer -- logs/experience/*.jsonl --epochs 10
//! ```

use clap::Parser;
use hyperliquid_rust_sdk::market_maker::checkpoint::types::RLCheckpoint;
use hyperliquid_rust_sdk::market_maker::learning::offline_trainer::{
    read_experience_file, OfflineTrainer, OfflineTrainerConfig,
};

#[derive(Parser)]
#[command(name = "rl_trainer", about = "Offline RL trainer for market making")]
struct Cli {
    /// Experience JSONL files to train on
    #[arg(required = true)]
    experience_files: Vec<String>,

    /// Maximum number of training epochs
    #[arg(long, default_value = "10")]
    epochs: usize,

    /// Shuffle experiences each epoch
    #[arg(long, default_value = "true")]
    shuffle: bool,

    /// Discount factor (gamma)
    #[arg(long, default_value = "0.95")]
    gamma: f64,

    /// Output path for trained checkpoint
    #[arg(long, default_value = "rl_trained.json")]
    output: String,

    /// Path to prior checkpoint to warm-start from
    #[arg(long)]
    prior: Option<String>,

    /// Weight for prior Q-values (0.0-1.0)
    #[arg(long, default_value = "0.3")]
    prior_weight: f64,

    /// Convergence threshold (stop when Q-delta < this)
    #[arg(long, default_value = "0.01")]
    convergence: f64,
}

fn main() {
    let cli = Cli::parse();

    // Read all experience files
    println!("=== RL Offline Trainer ===\n");
    let mut all_experiences = Vec::new();

    for file_path in &cli.experience_files {
        match read_experience_file(file_path) {
            Ok(records) => {
                println!("  Read {} experiences from {file_path}", records.len());
                all_experiences.extend(records);
            }
            Err(e) => {
                eprintln!("  Error reading {file_path}: {e}");
            }
        }
    }

    if all_experiences.is_empty() {
        eprintln!("\nNo experiences loaded. Nothing to train on.");
        std::process::exit(1);
    }

    println!("\nTotal experiences: {}", all_experiences.len());

    // Build config
    let config = OfflineTrainerConfig {
        max_epochs: cli.epochs,
        shuffle: cli.shuffle,
        gamma: cli.gamma,
        convergence_threshold: cli.convergence,
        ..Default::default()
    };

    // Create trainer (with optional prior)
    let mut trainer = if let Some(prior_path) = &cli.prior {
        println!("Loading prior checkpoint from: {prior_path}");
        let prior_json = std::fs::read_to_string(prior_path)
            .unwrap_or_else(|e| {
                eprintln!("Error reading prior checkpoint: {e}");
                std::process::exit(1);
            });
        let prior: RLCheckpoint = serde_json::from_str(&prior_json)
            .unwrap_or_else(|e| {
                eprintln!("Error parsing prior checkpoint: {e}");
                std::process::exit(1);
            });
        println!(
            "  Prior: {} Q-entries, {} total observations",
            prior.q_entries.len(),
            prior.total_observations
        );
        OfflineTrainer::with_prior(config, &prior, cli.prior_weight)
    } else {
        OfflineTrainer::new(config)
    };

    // Train
    println!("\nTraining...");
    println!(
        "  Config: epochs={}, shuffle={}, gamma={}, convergence={}",
        cli.epochs, cli.shuffle, cli.gamma, cli.convergence
    );

    let history = trainer.train(&all_experiences);

    // Print epoch summary
    println!("\n--- Epoch Summary ---");
    for metrics in &history.epoch_metrics {
        println!(
            "  Epoch {}: avg_reward={:.4}, states={}, updates={}, delta={:.6}",
            metrics.epoch,
            metrics.avg_reward,
            metrics.states_visited,
            metrics.total_updates,
            metrics.convergence_metric
        );
    }

    // Print final summary
    println!("\n--- Training Complete ---");
    println!("  Epochs completed: {}", history.epochs_completed);
    println!("  Converged: {}", history.converged);
    println!("  Final states visited: {}", history.final_states_visited);
    println!(
        "  Total experiences processed: {}",
        history.total_experiences_processed
    );

    // Export checkpoint
    let checkpoint = trainer.to_checkpoint();
    let json = serde_json::to_string_pretty(&checkpoint).unwrap();
    std::fs::write(&cli.output, &json).unwrap_or_else(|e| {
        eprintln!("Error writing output: {e}");
        std::process::exit(1);
    });

    println!(
        "\n  Checkpoint written to: {} ({} Q-entries, {} observations)",
        cli.output,
        checkpoint.q_entries.len(),
        checkpoint.total_observations
    );
}
