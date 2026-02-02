#!/bin/bash
#
# Database Setup Script for Quantitative Trading System
#
# This script sets up the TimescaleDB database for local development or testing.
#
# Usage:
#   ./scripts/setup_database.sh [options]
#
# Options:
#   --create-user    Create database user
#   --create-db      Create database
#   --run-migrations Run Alembic migrations
#   --load-schema    Load SQL schema directly
#   --all            Run all setup steps
#   --drop           Drop existing database first
#
# Environment Variables:
#   QUANT_DB_HOST     Database host (default: localhost)
#   QUANT_DB_PORT     Database port (default: 5432)
#   QUANT_DB_NAME     Database name (default: quant_trading_db)
#   QUANT_DB_USER     Application user (default: quant_app)
#   QUANT_DB_PASSWORD Application password
#   PGPASSWORD        Admin password for postgres user
#

set -e

# Configuration
DB_HOST="${QUANT_DB_HOST:-localhost}"
DB_PORT="${QUANT_DB_PORT:-5432}"
DB_NAME="${QUANT_DB_NAME:-quant_trading_db}"
DB_USER="${QUANT_DB_USER:-quant_app}"
DB_PASSWORD="${QUANT_DB_PASSWORD:-quant_password}"
ADMIN_USER="${PGUSER:-postgres}"

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Options
CREATE_USER=false
CREATE_DB=false
RUN_MIGRATIONS=false
LOAD_SCHEMA=false
DROP_DB=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --create-user)
            CREATE_USER=true
            shift
            ;;
        --create-db)
            CREATE_DB=true
            shift
            ;;
        --run-migrations)
            RUN_MIGRATIONS=true
            shift
            ;;
        --load-schema)
            LOAD_SCHEMA=true
            shift
            ;;
        --all)
            CREATE_USER=true
            CREATE_DB=true
            RUN_MIGRATIONS=true
            shift
            ;;
        --drop)
            DROP_DB=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--create-user] [--create-db] [--run-migrations] [--load-schema] [--all] [--drop]"
            exit 1
            ;;
    esac
done

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Check PostgreSQL connectivity
log "Checking PostgreSQL connectivity..."
if ! pg_isready -h "$DB_HOST" -p "$DB_PORT" -U "$ADMIN_USER" > /dev/null 2>&1; then
    log "ERROR: Cannot connect to PostgreSQL at $DB_HOST:$DB_PORT"
    log "Make sure PostgreSQL is running and accessible."
    exit 1
fi
log "PostgreSQL is running"

# Drop database if requested
if [ "$DROP_DB" = true ]; then
    log "Dropping existing database: $DB_NAME"
    psql -h "$DB_HOST" -p "$DB_PORT" -U "$ADMIN_USER" -c "DROP DATABASE IF EXISTS $DB_NAME;" 2>/dev/null || true
    log "Database dropped"
fi

# Create user
if [ "$CREATE_USER" = true ]; then
    log "Creating database user: $DB_USER"

    psql -h "$DB_HOST" -p "$DB_PORT" -U "$ADMIN_USER" << EOF
DO \$\$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = '$DB_USER') THEN
        CREATE ROLE $DB_USER WITH LOGIN PASSWORD '$DB_PASSWORD';
    END IF;
END
\$\$;

-- Grant necessary privileges
ALTER ROLE $DB_USER CREATEDB;
EOF

    log "User created/verified: $DB_USER"
fi

# Create database
if [ "$CREATE_DB" = true ]; then
    log "Creating database: $DB_NAME"

    # Check if database exists
    if psql -h "$DB_HOST" -p "$DB_PORT" -U "$ADMIN_USER" -lqt | cut -d \| -f 1 | grep -qw "$DB_NAME"; then
        log "Database $DB_NAME already exists"
    else
        psql -h "$DB_HOST" -p "$DB_PORT" -U "$ADMIN_USER" -c "CREATE DATABASE $DB_NAME OWNER $DB_USER;"
        log "Database created: $DB_NAME"
    fi

    # Enable TimescaleDB extension
    log "Enabling TimescaleDB extension..."
    psql -h "$DB_HOST" -p "$DB_PORT" -U "$ADMIN_USER" -d "$DB_NAME" << EOF
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;
CREATE EXTENSION IF NOT EXISTS pgcrypto;
EOF

    # Verify TimescaleDB
    VERSION=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$ADMIN_USER" -d "$DB_NAME" -t -c "SELECT extversion FROM pg_extension WHERE extname='timescaledb';")
    log "TimescaleDB version: $VERSION"

    # Grant privileges
    psql -h "$DB_HOST" -p "$DB_PORT" -U "$ADMIN_USER" -d "$DB_NAME" << EOF
GRANT ALL PRIVILEGES ON DATABASE $DB_NAME TO $DB_USER;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO $DB_USER;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO $DB_USER;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO $DB_USER;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO $DB_USER;
EOF

    log "Privileges granted to $DB_USER"
fi

# Load SQL schema directly
if [ "$LOAD_SCHEMA" = true ]; then
    log "Loading SQL schema..."

    SCHEMA_FILE="$PROJECT_ROOT/sql/schema.sql"
    if [ -f "$SCHEMA_FILE" ]; then
        psql -h "$DB_HOST" -p "$DB_PORT" -U "$ADMIN_USER" -d "$DB_NAME" -f "$SCHEMA_FILE"
        log "Schema loaded from: $SCHEMA_FILE"
    else
        log "ERROR: Schema file not found: $SCHEMA_FILE"
        exit 1
    fi
fi

# Run Alembic migrations
if [ "$RUN_MIGRATIONS" = true ]; then
    log "Running Alembic migrations..."

    cd "$PROJECT_ROOT"

    # Set database URL for Alembic
    export QUANT_DB_URL="postgresql://$DB_USER:$DB_PASSWORD@$DB_HOST:$DB_PORT/$DB_NAME"

    # Run migrations
    if command -v alembic &> /dev/null; then
        alembic upgrade head
        log "Migrations completed"
    else
        log "WARNING: Alembic not found. Install with: pip install alembic"
        log "Skipping migrations"
    fi
fi

# Print summary
log "Setup completed successfully!"
log ""
log "Database connection details:"
log "  Host:     $DB_HOST"
log "  Port:     $DB_PORT"
log "  Database: $DB_NAME"
log "  User:     $DB_USER"
log ""
log "Connection URL:"
log "  postgresql://$DB_USER:****@$DB_HOST:$DB_PORT/$DB_NAME"
log ""
log "To connect:"
log "  psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME"
