# Makefile for quant_trading
# Common development tasks

.PHONY: all build build-debug build-release test test-cpp test-python clean format lint install dev-install help

# Default target
all: build

# Build C++ extensions in-place
build:
	python setup.py build_ext --inplace

# Build with CMake directly (for development)
build-debug:
	mkdir -p build
	cd build && cmake .. -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTS=ON
	cd build && cmake --build .

build-release:
	mkdir -p build
	cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON
	cd build && cmake --build .

# Run all tests
test: test-python

# Run C++ tests (requires build-debug or build-release first)
test-cpp:
	cd build && ctest --output-on-failure

# Run Python tests
test-python:
	pytest tests/python -v

# Run tests with coverage
test-cov:
	pytest tests/python -v --cov=src/python/quant_trading --cov-report=term-missing

# Clean build artifacts
clean:
	rm -rf build/ dist/ *.egg-info
	find . -type f -name "*.so" -delete
	find . -type f -name "*.dylib" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# Format code
format:
	black src/python tests/python
	find src/cpp -name "*.cpp" -o -name "*.hpp" | xargs clang-format -i

# Run linters
lint:
	ruff check src/python tests/python
	mypy src/python

# Install package
install:
	pip install .

# Install package in development mode with dev dependencies
dev-install:
	pip install -e ".[dev]"

# Help
help:
	@echo "Available targets:"
	@echo "  build         - Build C++ extensions in-place"
	@echo "  build-debug   - Build with CMake (Debug mode)"
	@echo "  build-release - Build with CMake (Release mode)"
	@echo "  test          - Run all tests"
	@echo "  test-cpp      - Run C++ tests"
	@echo "  test-python   - Run Python tests"
	@echo "  test-cov      - Run Python tests with coverage"
	@echo "  clean         - Remove build artifacts"
	@echo "  format        - Format code (black + clang-format)"
	@echo "  lint          - Run linters (ruff + mypy)"
	@echo "  install       - Install package"
	@echo "  dev-install   - Install in development mode with dev deps"
