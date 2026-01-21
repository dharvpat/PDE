# Build Guide

Detailed instructions for building the quant_trading package.

## Prerequisites

### Required Tools

| Tool | Minimum Version | Purpose |
|------|----------------|---------|
| Python | 3.11 | Runtime and package management |
| CMake | 3.18 | C++ build system |
| C++ Compiler | C++17 | GCC 9+, Clang 10+, or AppleClang 12+ |
| Eigen | 3.4 | Linear algebra library |

### Optional Tools

| Tool | Purpose |
|------|---------|
| Ninja | Faster builds than Make |
| ccache | Compiler cache for faster rebuilds |
| OpenMP | Parallel execution |
| Google Test | C++ unit testing |

## Platform-Specific Setup

### macOS

```bash
# Install Homebrew if not present
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install cmake ninja eigen pybind11

# Verify installations
cmake --version   # Should be >= 3.18
```

### Ubuntu/Debian

```bash
# Update package list
sudo apt-get update

# Install dependencies
sudo apt-get install -y \
    cmake \
    ninja-build \
    libeigen3-dev \
    pybind11-dev \
    libomp-dev

# Verify installations
cmake --version   # Should be >= 3.18
```

## Build Methods

### Method 1: pip install (Recommended)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install in development mode
pip install -e ".[dev]"
```

### Method 2: Makefile

```bash
# Build C++ extensions
make build

# Build with debug symbols
make build-debug

# Build optimized release
make build-release
```

### Method 3: CMake Directly

```bash
# Create build directory
mkdir build && cd build

# Configure
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON

# Build
cmake --build .

# Run tests
ctest --output-on-failure
```

## Build Options

CMake options can be set during configuration:

| Option | Default | Description |
|--------|---------|-------------|
| `CMAKE_BUILD_TYPE` | Release | Debug or Release |
| `BUILD_TESTS` | ON | Build unit tests |
| `ENABLE_OPENMP` | ON | Enable OpenMP parallelization |
| `ENABLE_LTO` | ON | Enable link-time optimization |

Example:

```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=ON -DENABLE_OPENMP=OFF
```

## Verifying the Build

```bash
# Verify Python package
python -c "import quant_trading; print(quant_trading.__version__)"

# Run Python tests
pytest tests/python -v

# Run C++ tests (if built)
cd build && ctest --output-on-failure
```

## Troubleshooting

### Eigen not found

```bash
# macOS
brew install eigen

# Ubuntu
sudo apt-get install libeigen3-dev

# Or download and place in third_party/
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
tar -xzf eigen-3.4.0.tar.gz
mv eigen-3.4.0 third_party/eigen
```

### CMake version too old

```bash
# Ubuntu: Install from Kitware APT repository
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc | sudo apt-key add -
sudo apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal main'
sudo apt-get update
sudo apt-get install cmake
```

### OpenMP not working on macOS

AppleClang doesn't ship with OpenMP. Options:

1. Install libomp: `brew install libomp`
2. Use GCC: `brew install gcc && export CXX=g++-13`
3. Disable OpenMP: `cmake .. -DENABLE_OPENMP=OFF`

## Clean Build

```bash
# Remove all build artifacts
make clean

# Or manually
rm -rf build/ dist/ *.egg-info
find . -name "*.so" -delete
find . -name "__pycache__" -type d -exec rm -rf {} +
```
