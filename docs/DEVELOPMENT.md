# Development Guide

Guide for contributing to the quant_trading project.

## Getting Started

### Initial Setup

```bash
# Clone the repository
git clone <repository-url>
cd PDE

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install in development mode with all dev dependencies
pip install -e ".[dev]"
```

### Development Workflow

1. Create a feature branch
2. Make changes
3. Run tests and linters
4. Commit with descriptive message
5. Push and create pull request

## Code Style

### Python

- **Formatter:** Black (line length 100)
- **Linter:** Ruff
- **Type Checker:** mypy (strict mode)

```bash
# Format code
black src/python tests/python

# Run linter
ruff check src/python tests/python

# Type check
mypy src/python
```

### C++

- **Standard:** C++17
- **Formatter:** clang-format (LLVM-based style)
- **Line length:** 100 characters

```bash
# Format C++ code
find src/cpp -name "*.cpp" -o -name "*.hpp" | xargs clang-format -i
```

## Testing

### Python Tests

```bash
# Run all Python tests
pytest tests/python -v

# Run with coverage
pytest tests/python -v --cov=src/python/quant_trading --cov-report=term-missing

# Run specific test file
pytest tests/python/test_import.py -v
```

### C++ Tests

```bash
# Build with tests enabled
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=ON
cmake --build .

# Run tests
ctest --output-on-failure
```

## Project Architecture

### Python/C++ Integration

The project uses pybind11 for Python/C++ integration:

1. **C++ Core:** Performance-critical code in `src/cpp/`
2. **Python Bindings:** pybind11 bindings in `src/cpp/bindings/`
3. **Python Wrappers:** High-level Python API in `src/python/quant_trading/`

### When to Use C++ vs Python

**Use C++ for:**
- Inner loops with millions of iterations
- PDE solvers (finite difference, Monte Carlo)
- Mathematical functions called repeatedly
- Real-time computations during market hours

**Use Python for:**
- Data I/O and validation
- API calls and database interaction
- Orchestration and workflow logic
- Testing and prototyping

## Adding New Features

### Adding a New Model

1. Implement core algorithm in C++ (`src/cpp/models/`)
2. Write C++ unit tests (`tests/cpp/`)
3. Create pybind11 bindings (`src/cpp/bindings/`)
4. Write Python wrapper (`src/python/quant_trading/models/`)
5. Add Python tests (`tests/python/`)
6. Update documentation

### Adding a New Signal

1. Add signal logic to `src/python/quant_trading/signals/`
2. Write comprehensive tests
3. Document expected inputs/outputs

## Debugging

### Python Debugging

```python
import pdb; pdb.set_trace()  # Add breakpoint
```

### C++ Debugging

Build with debug symbols:

```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug
```

Use GDB or LLDB:

```bash
lldb python
(lldb) run -c "import quant_trading"
```

## Performance Profiling

### Python Profiling

```python
import cProfile
import pstats

cProfile.run('function_to_profile()', 'output.prof')
stats = pstats.Stats('output.prof')
stats.sort_stats('cumulative').print_stats(20)
```

### C++ Profiling

Use Instruments (macOS) or perf (Linux):

```bash
# Linux
perf record ./build/bin/benchmark
perf report
```

## Common Tasks

### Makefile Targets

```bash
make build        # Build C++ extensions
make test         # Run all tests
make clean        # Remove build artifacts
make format       # Format all code
make lint         # Run linters
make dev-install  # Install in dev mode
```

### Updating Dependencies

1. Update version in `requirements.txt` or `pyproject.toml`
2. Run `pip install -r requirements.txt`
3. Test thoroughly
4. Update CI workflow if needed

## Resources

- [Design Document](design-doc.md) - System architecture
- [Build Guide](BUILD.md) - Detailed build instructions
- [Bibliography](bibliography.md) - Academic references
