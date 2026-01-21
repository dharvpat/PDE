#!/bin/bash
# Smoke test script for quant_trading
# Verifies the build system and basic functionality

set -e

echo "=== Smoke Test ==="
echo ""

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT"

echo "Project root: $PROJECT_ROOT"
echo ""

# Check Python version
echo "1. Checking Python version..."
python --version
PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "   Python $PYTHON_VERSION"
echo ""

# Clean any previous build
echo "2. Cleaning previous build artifacts..."
rm -rf build/ dist/ *.egg-info
find . -type f -name "*.so" -delete 2>/dev/null || true
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
echo "   Done."
echo ""

# Install package
echo "3. Installing package in development mode..."
pip install -e . --quiet
echo "   Done."
echo ""

# Test import
echo "4. Testing Python import..."
python -c "import quant_trading; print(f'   Version: {quant_trading.__version__}')"
echo ""

# Run Python tests
echo "5. Running Python tests..."
pytest tests/python -v --tb=short
echo ""

# Optional: Build C++ with CMake
echo "6. Building C++ components with CMake..."
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF
cmake --build .
cd ..
echo "   Done."
echo ""

echo "=== All smoke tests passed! ==="
