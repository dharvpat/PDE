#!/bin/bash
# Quantitative Trading System - Setup Script
# This script sets up the development/deployment environment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_ROOT/.venv"

echo "=================================================="
echo "  Quantitative Trading System - Setup"
echo "=================================================="
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
REQUIRED_VERSION="3.11"

if [[ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]]; then
    echo "Error: Python $REQUIRED_VERSION or higher is required (found $PYTHON_VERSION)"
    exit 1
fi

echo "Python version: $PYTHON_VERSION"

# Create or activate virtual environment
if [ ! -d "$VENV_DIR" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install the package in editable mode
echo ""
echo "Installing quant_trading package..."
cd "$PROJECT_ROOT"
pip install -e ".[dev]"

# Build C++ extensions
echo ""
echo "Building C++ extensions..."
if [ -d "build" ]; then
    rm -rf build
fi
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --target quant_cpp

# Return to project root
cd "$PROJECT_ROOT"

# Verify installation
echo ""
echo "Verifying installation..."
python3 -c "import quant_trading; print(f'quant_trading version: {quant_trading.__version__}')"
python3 -c "from quant_trading.cpp import quant_cpp; print(f'C++ bindings loaded successfully')"

echo ""
echo "=================================================="
echo "  Setup Complete!"
echo "=================================================="
echo ""
echo "To activate the environment, run:"
echo "  source .venv/bin/activate"
echo ""
echo "To run the trading system:"
echo "  quant-trading --help"
echo "  quant-trading demo"
echo ""
echo "To run tests:"
echo "  pytest tests/python/"
echo ""
