# Development Setup Guide

## Overview

This guide walks you through setting up a local development environment for the Quantitative Trading System.

## Prerequisites

### Required Software

| Software | Version | Purpose |
|----------|---------|---------|
| Python | 3.11+ | Core runtime |
| PostgreSQL | 15+ | Database (with TimescaleDB) |
| Redis | 7+ | Caching |
| Docker | 24+ | Containerization |
| CMake | 3.20+ | C++ build system |
| GCC/Clang | 12+ / 15+ | C++ compiler |

### Optional Software

| Software | Purpose |
|----------|---------|
| Docker Compose | Local services orchestration |
| RabbitMQ | Message queue (optional for local) |
| VS Code | Recommended IDE |

## Quick Start with Docker

The fastest way to get started is using Docker Compose:

```bash
# Clone repository
git clone https://github.com/company/quant-trading.git
cd quant-trading

# Start all services
docker-compose -f deploy/docker/docker-compose.yml up -d

# Verify services
docker-compose ps

# View logs
docker-compose logs -f api
```

## Manual Setup

### Step 1: Clone Repository

```bash
git clone https://github.com/company/quant-trading.git
cd quant-trading
```

### Step 2: Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/macOS)
source venv/bin/activate

# Activate (Windows)
.\venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install package in editable mode
pip install -e .
```

### Step 3: C++ Extensions

The system includes C++ extensions for performance-critical code.

```bash
# Install C++ dependencies (Ubuntu/Debian)
sudo apt-get install -y \
    build-essential \
    cmake \
    libeigen3-dev \
    libboost-all-dev

# Install C++ dependencies (macOS)
brew install cmake eigen boost

# Build C++ extensions
python setup.py build_ext --inplace

# Verify build
python -c "from quant_trading.cpp import heston_pricer; print('C++ extensions loaded')"
```

### Step 4: Database Setup

#### Option A: Docker (Recommended)

```bash
# Start TimescaleDB
docker run -d \
    --name timescaledb \
    -p 5432:5432 \
    -e POSTGRES_PASSWORD=postgres \
    -v timescaledb_data:/var/lib/postgresql/data \
    timescale/timescaledb:latest-pg15

# Start Redis
docker run -d \
    --name redis \
    -p 6379:6379 \
    redis:7-alpine
```

#### Option B: Native Installation

```bash
# Ubuntu/Debian
sudo apt-get install postgresql-15 postgresql-contrib-15
# Follow TimescaleDB installation guide for your OS

# Initialize database
sudo -u postgres createuser -s $USER
createdb trading
```

### Step 5: Initialize Database

```bash
# Run migrations
cd src/python
alembic upgrade head

# Or use the init script
python -m quant_trading.database.init
```

### Step 6: Configuration

```bash
# Copy example configuration
cp config/config.example.yaml config/config.yaml

# Edit configuration
# Set database URL, API keys, etc.
```

Example configuration:

```yaml
# config/config.yaml
database:
  url: postgresql://postgres:postgres@localhost:5432/trading
  pool_size: 10

redis:
  url: redis://localhost:6379/0

api:
  host: 0.0.0.0
  port: 8000
  debug: true

logging:
  level: DEBUG
  format: json
```

### Step 7: Verify Installation

```bash
# Run tests
pytest tests/ -v

# Start API server
python -m quant_trading.api.server

# In another terminal, test the API
curl http://localhost:8000/health
```

## IDE Setup

### VS Code

Install recommended extensions:

```json
// .vscode/extensions.json
{
  "recommendations": [
    "ms-python.python",
    "ms-python.vscode-pylance",
    "ms-vscode.cpptools",
    "ms-azuretools.vscode-docker",
    "redhat.vscode-yaml"
  ]
}
```

Configure settings:

```json
// .vscode/settings.json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
  "python.formatting.provider": "black",
  "python.linting.enabled": true,
  "python.linting.ruffEnabled": true,
  "python.linting.mypyEnabled": true,
  "editor.formatOnSave": true,
  "[python]": {
    "editor.codeActionsOnSave": {
      "source.organizeImports": true
    }
  }
}
```

### PyCharm

1. Open project directory
2. Configure interpreter: `File > Settings > Project > Python Interpreter`
3. Select the `venv` environment
4. Enable pytest: `File > Settings > Tools > Python Integrated Tools > Default test runner: pytest`

## Running Services

### Development Server

```bash
# API server with hot reload
uvicorn quant_trading.api.main:app --reload --host 0.0.0.0 --port 8000

# Or using the module
python -m quant_trading.api.server
```

### Background Workers

```bash
# Start calibration worker
python -m quant_trading.workers.calibration

# Start signal generator
python -m quant_trading.workers.signals
```

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=quant_trading --cov-report=html

# Specific test file
pytest tests/python/models/test_heston.py -v

# Specific test
pytest tests/python/models/test_heston.py::test_price_call -v

# Run C++ tests
cd build && ctest --output-on-failure
```

### Code Quality

```bash
# Format code
black src/python tests
isort src/python tests

# Lint
ruff check src/python tests

# Type check
mypy src/python

# All quality checks
make lint
```

## Common Issues

### C++ Build Failures

**Issue**: `fatal error: Eigen/Dense: No such file or directory`

**Solution**:
```bash
# Ubuntu
sudo apt-get install libeigen3-dev

# macOS
brew install eigen

# Or specify path
cmake .. -DEIGEN3_INCLUDE_DIR=/path/to/eigen
```

### Database Connection Issues

**Issue**: `connection refused`

**Solution**:
```bash
# Check PostgreSQL is running
docker ps | grep timescaledb

# Check connection
psql -h localhost -U postgres -d trading

# Reset if needed
docker restart timescaledb
```

### Import Errors

**Issue**: `ModuleNotFoundError: No module named 'quant_trading'`

**Solution**:
```bash
# Install in editable mode
pip install -e .

# Verify
python -c "import quant_trading; print(quant_trading.__version__)"
```

### Permission Issues (macOS)

**Issue**: `Permission denied: /var/run/docker.sock`

**Solution**:
```bash
# Add user to docker group (Linux)
sudo usermod -aG docker $USER

# macOS: ensure Docker Desktop is running
```

## Development Workflow

### Feature Development

1. Create feature branch: `git checkout -b feature/my-feature`
2. Write tests first (TDD)
3. Implement feature
4. Run tests: `pytest tests/`
5. Check code quality: `make lint`
6. Commit with descriptive message
7. Push and create PR

### Debugging

```python
# Add breakpoint
import pdb; pdb.set_trace()

# Or use VS Code debugger with launch.json:
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: API Server",
      "type": "python",
      "request": "launch",
      "module": "uvicorn",
      "args": ["quant_trading.api.main:app", "--reload"],
      "jinja": true
    }
  ]
}
```

### Profiling

```python
# Profile function
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Your code here
result = calibrator.calibrate(options_data)

profiler.disable()
stats = pstats.Stats(profiler).sort_stats('cumtime')
stats.print_stats(20)
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection URL | - |
| `REDIS_URL` | Redis connection URL | redis://localhost:6379 |
| `LOG_LEVEL` | Logging level | INFO |
| `API_DEBUG` | Enable debug mode | false |
| `JWT_SECRET` | JWT signing secret | - |

## Useful Make Commands

```bash
make install       # Install dependencies
make build         # Build C++ extensions
make test          # Run all tests
make lint          # Run linters
make format        # Format code
make docs          # Build documentation
make docker-build  # Build Docker images
make clean         # Clean build artifacts
```
