#!/bin/bash
# =============================================================================
# Database Restore Script for Quantitative Trading System
# =============================================================================
#
# This script performs:
# - Database restoration from backup
# - Point-in-time recovery using WAL
# - Model parameters restoration
#
# Reference: Section 12.3 of design-doc.md (Disaster Recovery)
# RTO Target: <4 hours
# RPO Target: <15 minutes
#
# Usage:
#   ./restore.sh [backup_file] [--pit <timestamp>]
#
# =============================================================================

set -euo pipefail

# Configuration
BACKUP_FILE="${1:-}"
RESTORE_DIR="${RESTORE_DIR:-/var/restore/quant-trading}"
S3_BUCKET="${S3_BUCKET:-quant-trading-backups}"
AWS_REGION="${AWS_REGION:-us-east-1}"

# Database connection
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-quant_trading}"
DB_USER="${DB_USER:-quant}"

# Logging
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

error() {
    log "ERROR: $*" >&2
    exit 1
}

warning() {
    log "WARNING: $*" >&2
}

# =============================================================================
# Pre-flight Checks
# =============================================================================
preflight_checks() {
    log "Running pre-flight checks..."

    # Check required tools
    for cmd in pg_restore psql aws; do
        if ! command -v "${cmd}" &> /dev/null; then
            error "Required command not found: ${cmd}"
        fi
    done

    # Check database connectivity
    if ! PGPASSWORD="${DB_PASSWORD}" psql \
        -h "${DB_HOST}" \
        -p "${DB_PORT}" \
        -U "${DB_USER}" \
        -d postgres \
        -c "SELECT 1" &> /dev/null; then
        error "Cannot connect to PostgreSQL server"
    fi

    log "Pre-flight checks passed"
}

# =============================================================================
# Download Backup from S3
# =============================================================================
download_from_s3() {
    local s3_path="$1"
    local local_path="$2"

    log "Downloading backup from S3: ${s3_path}"

    mkdir -p "$(dirname "${local_path}")"

    aws s3 cp \
        "s3://${S3_BUCKET}/${s3_path}" \
        "${local_path}" \
        --region "${AWS_REGION}"

    # Download checksum if available
    if aws s3 ls "s3://${S3_BUCKET}/${s3_path}.sha256" &> /dev/null; then
        aws s3 cp \
            "s3://${S3_BUCKET}/${s3_path}.sha256" \
            "${local_path}.sha256" \
            --region "${AWS_REGION}"
    fi

    log "Download completed"
}

# =============================================================================
# List Available Backups
# =============================================================================
list_backups() {
    log "Available backups in S3:"
    echo ""

    aws s3 ls "s3://${S3_BUCKET}/full/" --region "${AWS_REGION}" | \
        awk '{print "  " $1, $2, $3, $4}' | \
        sort -r | \
        head -20

    echo ""
    log "Use: ./restore.sh <backup_filename>"
}

# =============================================================================
# Verify Backup Integrity
# =============================================================================
verify_backup() {
    local backup_file="$1"

    log "Verifying backup integrity..."

    if [[ ! -f "${backup_file}" ]]; then
        error "Backup file not found: ${backup_file}"
    fi

    # Verify checksum
    if [[ -f "${backup_file}.sha256" ]]; then
        log "Verifying checksum..."
        if ! sha256sum -c "${backup_file}.sha256" &> /dev/null; then
            error "Checksum verification failed!"
        fi
        log "Checksum verified"
    else
        warning "No checksum file found, skipping verification"
    fi

    # Verify backup format
    log "Verifying backup format..."
    if ! pg_restore -l "${backup_file}" &> /dev/null; then
        error "Invalid backup format"
    fi
    log "Backup format verified"
}

# =============================================================================
# Stop Application Services
# =============================================================================
stop_services() {
    log "Stopping application services..."

    # If running in Kubernetes
    if command -v kubectl &> /dev/null; then
        kubectl scale deployment -n quant-trading --all --replicas=0 2>/dev/null || true
    fi

    # If running with Docker Compose
    if [[ -f "/app/deploy/docker/docker-compose.yml" ]]; then
        docker-compose -f /app/deploy/docker/docker-compose.yml stop \
            api calibration data-ingestion execution signals 2>/dev/null || true
    fi

    log "Services stopped"
}

# =============================================================================
# Restore Database
# =============================================================================
restore_database() {
    local backup_file="$1"

    log "Starting database restoration..."
    log "Backup file: ${backup_file}"

    # Drop and recreate database
    log "Dropping existing database..."
    PGPASSWORD="${DB_PASSWORD}" psql \
        -h "${DB_HOST}" \
        -p "${DB_PORT}" \
        -U "${DB_USER}" \
        -d postgres \
        -c "DROP DATABASE IF EXISTS ${DB_NAME};"

    log "Creating new database..."
    PGPASSWORD="${DB_PASSWORD}" psql \
        -h "${DB_HOST}" \
        -p "${DB_PORT}" \
        -U "${DB_USER}" \
        -d postgres \
        -c "CREATE DATABASE ${DB_NAME} OWNER ${DB_USER};"

    # Enable extensions
    log "Enabling extensions..."
    PGPASSWORD="${DB_PASSWORD}" psql \
        -h "${DB_HOST}" \
        -p "${DB_PORT}" \
        -U "${DB_USER}" \
        -d "${DB_NAME}" \
        -c "CREATE EXTENSION IF NOT EXISTS timescaledb;"

    # Restore from backup
    log "Restoring data from backup..."
    PGPASSWORD="${DB_PASSWORD}" pg_restore \
        -h "${DB_HOST}" \
        -p "${DB_PORT}" \
        -U "${DB_USER}" \
        -d "${DB_NAME}" \
        --verbose \
        --no-owner \
        --no-privileges \
        "${backup_file}" \
        2>&1 | while read line; do log "pg_restore: $line"; done

    log "Database restoration completed"
}

# =============================================================================
# Point-in-Time Recovery
# =============================================================================
restore_pit() {
    local target_time="$1"

    log "Starting point-in-time recovery to: ${target_time}"

    # This requires WAL archiving to be enabled
    # In production, use pg_basebackup and recovery.conf

    warning "Point-in-time recovery requires manual WAL configuration"
    warning "Target time: ${target_time}"

    # Create recovery.conf (PostgreSQL < 12) or recovery.signal (>= 12)
    cat > /tmp/recovery.signal << EOF
# Recovery configuration for point-in-time recovery
# Generated by restore.sh

restore_command = 'aws s3 cp s3://${S3_BUCKET}/wal/%f %p --region ${AWS_REGION}'
recovery_target_time = '${target_time}'
recovery_target_action = 'promote'
EOF

    log "Recovery configuration created"
    log "Please apply this configuration to PostgreSQL and restart"
}

# =============================================================================
# Verify Restoration
# =============================================================================
verify_restoration() {
    log "Verifying restoration..."

    # Check table counts
    local tables=("market_data" "options_data" "heston_params" "sabr_params" "ou_params" "signals" "trades")

    for table in "${tables[@]}"; do
        count=$(PGPASSWORD="${DB_PASSWORD}" psql \
            -h "${DB_HOST}" \
            -p "${DB_PORT}" \
            -U "${DB_USER}" \
            -d "${DB_NAME}" \
            -t -A \
            -c "SELECT COUNT(*) FROM ${table};" 2>/dev/null || echo "0")
        log "  ${table}: ${count} rows"
    done

    # Check hypertables
    log "Checking TimescaleDB hypertables..."
    PGPASSWORD="${DB_PASSWORD}" psql \
        -h "${DB_HOST}" \
        -p "${DB_PORT}" \
        -U "${DB_USER}" \
        -d "${DB_NAME}" \
        -c "SELECT hypertable_name, num_chunks FROM timescaledb_information.hypertables;"

    log "Restoration verification completed"
}

# =============================================================================
# Start Application Services
# =============================================================================
start_services() {
    log "Starting application services..."

    # If running in Kubernetes
    if command -v kubectl &> /dev/null; then
        kubectl scale deployment -n quant-trading api-server --replicas=2 2>/dev/null || true
        kubectl scale deployment -n quant-trading calibration-service --replicas=3 2>/dev/null || true
        kubectl scale deployment -n quant-trading data-ingestion --replicas=2 2>/dev/null || true
        kubectl scale deployment -n quant-trading execution-service --replicas=2 2>/dev/null || true
        kubectl scale deployment -n quant-trading signals-service --replicas=2 2>/dev/null || true
    fi

    # If running with Docker Compose
    if [[ -f "/app/deploy/docker/docker-compose.yml" ]]; then
        docker-compose -f /app/deploy/docker/docker-compose.yml start \
            api calibration data-ingestion execution signals 2>/dev/null || true
    fi

    log "Services started"
}

# =============================================================================
# Main
# =============================================================================
main() {
    log "=========================================="
    log "Quant Trading System Restore"
    log "=========================================="

    # Parse arguments
    local pit_time=""
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --pit)
                pit_time="$2"
                shift 2
                ;;
            --list)
                list_backups
                exit 0
                ;;
            --help)
                echo "Usage: $0 [backup_file] [--pit <timestamp>] [--list]"
                exit 0
                ;;
            *)
                BACKUP_FILE="$1"
                shift
                ;;
        esac
    done

    if [[ -z "${BACKUP_FILE}" ]]; then
        list_backups
        error "Please specify a backup file"
    fi

    # Setup
    mkdir -p "${RESTORE_DIR}"

    # Pre-flight checks
    preflight_checks

    # Download if S3 path
    if [[ "${BACKUP_FILE}" == s3://* ]] || [[ ! -f "${BACKUP_FILE}" ]]; then
        local local_backup="${RESTORE_DIR}/$(basename "${BACKUP_FILE}")"
        download_from_s3 "full/${BACKUP_FILE}" "${local_backup}"
        BACKUP_FILE="${local_backup}"
    fi

    # Verify backup
    verify_backup "${BACKUP_FILE}"

    # Confirm with user
    echo ""
    warning "This will DESTROY the current database and restore from backup!"
    warning "Backup file: ${BACKUP_FILE}"
    if [[ -n "${pit_time}" ]]; then
        warning "Point-in-time recovery to: ${pit_time}"
    fi
    echo ""
    read -p "Are you sure you want to continue? (yes/no): " confirm
    if [[ "${confirm}" != "yes" ]]; then
        log "Restore cancelled"
        exit 0
    fi

    # Stop services
    stop_services

    # Perform restoration
    restore_database "${BACKUP_FILE}"

    # Point-in-time recovery if requested
    if [[ -n "${pit_time}" ]]; then
        restore_pit "${pit_time}"
    fi

    # Verify restoration
    verify_restoration

    # Start services
    read -p "Start application services? (yes/no): " start_confirm
    if [[ "${start_confirm}" == "yes" ]]; then
        start_services
    fi

    log "=========================================="
    log "Restore completed successfully"
    log "=========================================="
}

main "$@"
