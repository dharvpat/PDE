#!/bin/bash
# Quantitative Trading System - Run Script
# Usage: ./scripts/run.sh [command] [options]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_ROOT/.venv"

# Activate virtual environment
if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
else
    echo "Error: Virtual environment not found at $VENV_DIR"
    echo "Run ./scripts/setup.sh first"
    exit 1
fi

# Change to project root
cd "$PROJECT_ROOT"

# Run the trading system
if [ $# -eq 0 ]; then
    # No arguments - show help
    quant-trading --help
else
    quant-trading "$@"
fi
