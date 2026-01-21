# Quantitative Trading System

A sophisticated quantitative trading platform leveraging stochastic volatility models, optimal stopping theory, and PDE-based pricing for swing trading strategies.

## Overview

This system implements academic research-backed strategies for 1-2 month swing trading horizons, combining:

- **Heston and SABR stochastic volatility models** for volatility surface calibration
- **Ornstein-Uhlenbeck mean-reversion** for optimal entry/exit timing
- **C++ accelerated numerical computations** via pybind11 for performance-critical operations

**Target Performance:** Sharpe ratio 0.5-1.2 with max drawdown <25%

## Features

- Heston model calibration using FFT-based option pricing (Carr-Madan 1999)
- SABR model for smile risk management (Hagan et al. 2002)
- Optimal mean-reversion trading with transaction costs (Leung & Li 2015)
- Volatility-managed position sizing (Moreira & Muir 2017)
- Python/C++ hybrid architecture for both flexibility and speed

## Installation

### Prerequisites

- Python 3.11+
- CMake 3.18+
- C++17 compatible compiler (GCC 9+, Clang 10+, or AppleClang 12+)
- Eigen 3.4+

### macOS

```bash
brew install cmake ninja eigen pybind11
```

### Ubuntu/Debian

```bash
sudo apt-get install cmake ninja-build libeigen3-dev pybind11-dev
```

### Python Setup

```bash
# Clone the repository
git clone <repository-url>
cd PDE

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install in development mode
pip install -e ".[dev]"

# Verify installation
python -c "import quant_trading; print(quant_trading.__version__)"
```

## Quick Start

```python
import quant_trading

# Package is ready for model implementations
print(f"Version: {quant_trading.__version__}")
```

## Project Structure

```
PDE/
├── src/
│   ├── cpp/              # C++ implementations
│   │   ├── core/         # Common utilities
│   │   ├── models/       # Heston, SABR, OU models
│   │   ├── solvers/      # PDE solvers, FFT
│   │   └── bindings/     # pybind11 bindings
│   └── python/
│       └── quant_trading/
│           ├── calibration/  # Model calibration
│           ├── models/       # Python model wrappers
│           ├── signals/      # Signal generation
│           ├── risk/         # Risk management
│           └── execution/    # Order execution
├── tests/
│   ├── cpp/              # C++ unit tests
│   └── python/           # Python tests
├── docs/                 # Documentation
└── benchmarks/           # Performance benchmarks
```

## Development

See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for development workflow.

```bash
# Build
make build

# Run tests
make test

# Format code
make format

# Run linters
make lint
```

## Documentation

- [Design Document](docs/design-doc.md) - System architecture and mathematical foundations
- [Build Guide](docs/BUILD.md) - Detailed build instructions
- [Development Guide](docs/DEVELOPMENT.md) - Development workflow
- [Bibliography](docs/bibliography.md) - Academic references

## License

MIT License
