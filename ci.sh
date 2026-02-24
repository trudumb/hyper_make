#!/bin/bash

set -e

# Build
cargo build

# Check formatting
cargo fmt -- --check

# Run Clippy
cargo clippy -- -D warnings

# Run tests
export CARGO_BUILD_JOBS=4
cargo nextest run

echo "CI checks passed successfully."