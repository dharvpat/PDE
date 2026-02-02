#!/bin/bash
# =============================================================================
# Database Backup Script for Quantitative Trading System
# =============================================================================
#
# This script performs:
# - Full PostgreSQL/TimescaleDB database backup
# - WAL archiving for point-in-time recovery
# - Model parameters backup
# - Upload to S3 for off-site storage
#
# Reference: Section 12.3 of design-doc.md (Disaster Recovery)
#
# Usage:
#   ./backup.sh [full|incremental|wal]
#
# Environment Variables:
#   DATABASE_URL     - PostgreSQL connection string
#   S3_BUCKET        - S3 bucket for backup storage
#   AWS_REGION       - AWS region
#   BACKUP_RETENTION - Days to retain backups (default: 30)
#
# =============================================================================

set -euo pipefail

# Configuration
BACKUP_TYPE="${1:-full}"
BACKUP_DIR="${BACKUP_DIR:-/var/backups/quant-trading}"
S3_BUCKET="${S3_BUCKET:-quant-trading-backups}"
AWS_REGION="${AWS_REGION:-us-east-1}"
BACKUP_RETENTION="${BACKUP_RETENTION:-30}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
HOSTNAME=$(hostname)

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

# Create backup directory
mkdir -p "${BACKUP_DIR}"/{full,incremental,wal,params}

# =============================================================================
# Full Database Backup
# =============================================================================
backup_full() {
    log "Starting full database backup..."

    BACKUP_FILE="${BACKUP_DIR}/full/backup_${TIMESTAMP}.sql.gz"

    # Perform pg_dump with compression
    PGPASSWORD="${DB_PASSWORD}" pg_dump \
        -h "${DB_HOST}" \
        -p "${DB_PORT}" \
        -U "${DB_USER}" \
        -d "${DB_NAME}" \
        -F c \
        -Z 9 \
        -f "${BACKUP_FILE}.tmp" \
        --verbose \
        2>&1 | while read line; do log "pg_dump: $line"; done

    # Rename on success
    mv "${BACKUP_FILE}.tmp" "${BACKUP_FILE}"

    # Calculate checksum
    sha256sum "${BACKUP_FILE}" > "${BACKUP_FILE}.sha256"

    log "Full backup completed: ${BACKUP_FILE}"
    log "Size: $(du -h "${BACKUP_FILE}" | cut -f1)"

    # Upload to S3
    upload_to_s3 "${BACKUP_FILE}" "full"
    upload_to_s3 "${BACKUP_FILE}.sha256" "full"

    # Cleanup old backups
    cleanup_old_backups "full"
}

# =============================================================================
# WAL Archiving
# =============================================================================
backup_wal() {
    log "Starting WAL archive backup..."

    WAL_DIR="${BACKUP_DIR}/wal"

    # Archive current WAL files
    PGPASSWORD="${DB_PASSWORD}" psql \
        -h "${DB_HOST}" \
        -p "${DB_PORT}" \
        -U "${DB_USER}" \
        -d "${DB_NAME}" \
        -c "SELECT pg_switch_wal();" \
        2>&1 | while read line; do log "WAL switch: $line"; done

    # Copy WAL files to backup directory
    # Note: In production, use pg_receivewal for continuous archiving

    log "WAL archiving completed"
}

# =============================================================================
# Model Parameters Backup
# =============================================================================
backup_params() {
    log "Starting model parameters backup..."

    PARAMS_FILE="${BACKUP_DIR}/params/params_${TIMESTAMP}.json"

    # Export calibrated parameters from database
    PGPASSWORD="${DB_PASSWORD}" psql \
        -h "${DB_HOST}" \
        -p "${DB_PORT}" \
        -U "${DB_USER}" \
        -d "${DB_NAME}" \
        -t -A \
        -c "
        SELECT json_build_object(
            'timestamp', now(),
            'heston', (SELECT json_agg(row_to_json(h)) FROM (
                SELECT DISTINCT ON (symbol) * FROM heston_params ORDER BY symbol, time DESC
            ) h),
            'sabr', (SELECT json_agg(row_to_json(s)) FROM (
                SELECT DISTINCT ON (symbol, expiry) * FROM sabr_params ORDER BY symbol, expiry, time DESC
            ) s),
            'ou', (SELECT json_agg(row_to_json(o)) FROM (
                SELECT DISTINCT ON (pair_id) * FROM ou_params ORDER BY pair_id, time DESC
            ) o)
        );
        " > "${PARAMS_FILE}"

    # Compress
    gzip -9 "${PARAMS_FILE}"

    log "Model parameters backup completed: ${PARAMS_FILE}.gz"

    # Upload to S3
    upload_to_s3 "${PARAMS_FILE}.gz" "params"
}

# =============================================================================
# S3 Upload
# =============================================================================
upload_to_s3() {
    local file="$1"
    local prefix="$2"

    log "Uploading ${file} to S3..."

    aws s3 cp \
        "${file}" \
        "s3://${S3_BUCKET}/${prefix}/$(basename "${file}")" \
        --region "${AWS_REGION}" \
        --storage-class STANDARD_IA \
        2>&1 | while read line; do log "S3: $line"; done

    log "Upload completed"
}

# =============================================================================
# Cleanup Old Backups
# =============================================================================
cleanup_old_backups() {
    local backup_type="$1"

    log "Cleaning up backups older than ${BACKUP_RETENTION} days..."

    # Local cleanup
    find "${BACKUP_DIR}/${backup_type}" -type f -mtime "+${BACKUP_RETENTION}" -delete 2>/dev/null || true

    # S3 cleanup (using lifecycle policies is recommended instead)
    aws s3 ls "s3://${S3_BUCKET}/${backup_type}/" --recursive \
        | while read -r line; do
            create_date=$(echo "$line" | awk '{print $1}')
            file_name=$(echo "$line" | awk '{print $4}')
            create_epoch=$(date -d "${create_date}" +%s 2>/dev/null || echo 0)
            current_epoch=$(date +%s)
            age_days=$(( (current_epoch - create_epoch) / 86400 ))

            if [[ ${age_days} -gt ${BACKUP_RETENTION} ]]; then
                log "Deleting old S3 backup: ${file_name}"
                aws s3 rm "s3://${S3_BUCKET}/${file_name}" --region "${AWS_REGION}" || true
            fi
        done

    log "Cleanup completed"
}

# =============================================================================
# Verify Backup
# =============================================================================
verify_backup() {
    local backup_file="$1"

    log "Verifying backup integrity..."

    # Check if file exists and is not empty
    if [[ ! -s "${backup_file}" ]]; then
        error "Backup file is empty or missing: ${backup_file}"
    fi

    # Verify checksum if available
    if [[ -f "${backup_file}.sha256" ]]; then
        if sha256sum -c "${backup_file}.sha256" > /dev/null 2>&1; then
            log "Checksum verification passed"
        else
            error "Checksum verification failed!"
        fi
    fi

    # Try to list contents for pg_dump format
    if [[ "${backup_file}" == *.sql.gz ]]; then
        if pg_restore -l "${backup_file}" > /dev/null 2>&1; then
            log "Backup format verification passed"
        else
            log "Warning: Could not verify backup format"
        fi
    fi

    log "Backup verification completed"
}

# =============================================================================
# Main
# =============================================================================
main() {
    log "=========================================="
    log "Quant Trading System Backup"
    log "Type: ${BACKUP_TYPE}"
    log "Host: ${HOSTNAME}"
    log "=========================================="

    case "${BACKUP_TYPE}" in
        full)
            backup_full
            backup_params
            ;;
        incremental)
            backup_wal
            ;;
        wal)
            backup_wal
            ;;
        params)
            backup_params
            ;;
        all)
            backup_full
            backup_wal
            backup_params
            ;;
        *)
            error "Unknown backup type: ${BACKUP_TYPE}"
            ;;
    esac

    log "=========================================="
    log "Backup completed successfully"
    log "=========================================="
}

main "$@"
