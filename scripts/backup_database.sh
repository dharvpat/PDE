#!/bin/bash
#
# Database Backup Script for Quantitative Trading System
#
# This script performs full database backups with optional S3 upload.
#
# Usage:
#   ./scripts/backup_database.sh [options]
#
# Options:
#   --upload-s3     Upload backup to S3 bucket
#   --cleanup       Remove old backups (keeps last 30 days)
#   --verbose       Enable verbose output
#
# Environment Variables:
#   QUANT_DB_HOST       Database host (default: localhost)
#   QUANT_DB_PORT       Database port (default: 5432)
#   QUANT_DB_NAME       Database name (default: quant_trading_db)
#   QUANT_DB_USER       Database user (default: postgres)
#   PGPASSWORD          Database password
#   BACKUP_DIR          Backup directory (default: /var/backups/postgres)
#   S3_BUCKET           S3 bucket for backup uploads
#   RETENTION_DAYS      Days to keep backups (default: 30)
#

set -e

# Configuration
DB_HOST="${QUANT_DB_HOST:-localhost}"
DB_PORT="${QUANT_DB_PORT:-5432}"
DB_NAME="${QUANT_DB_NAME:-quant_trading_db}"
DB_USER="${QUANT_DB_USER:-postgres}"
BACKUP_DIR="${BACKUP_DIR:-/var/backups/postgres}"
S3_BUCKET="${S3_BUCKET:-}"
RETENTION_DAYS="${RETENTION_DAYS:-30}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="quant_trading_${TIMESTAMP}"

# Options
UPLOAD_S3=false
CLEANUP=false
VERBOSE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --upload-s3)
            UPLOAD_S3=true
            shift
            ;;
        --cleanup)
            CLEANUP=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

log() {
    if [ "$VERBOSE" = true ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    fi
}

log_always() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Create backup directory
mkdir -p "$BACKUP_DIR"

log_always "Starting backup of database: $DB_NAME"
log "Host: $DB_HOST:$DB_PORT"
log "User: $DB_USER"
log "Backup directory: $BACKUP_DIR"

# Perform backup
BACKUP_FILE="$BACKUP_DIR/${BACKUP_NAME}.sql.gz"

log "Running pg_dump..."

pg_dump \
    -h "$DB_HOST" \
    -p "$DB_PORT" \
    -U "$DB_USER" \
    -d "$DB_NAME" \
    --format=custom \
    --verbose \
    --file="$BACKUP_DIR/${BACKUP_NAME}.dump" \
    2>&1 | while read -r line; do log "$line"; done

# Also create SQL dump for human-readable backup
pg_dump \
    -h "$DB_HOST" \
    -p "$DB_PORT" \
    -U "$DB_USER" \
    -d "$DB_NAME" \
    --format=plain \
    | gzip > "$BACKUP_FILE"

BACKUP_SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
log_always "Backup created: $BACKUP_FILE ($BACKUP_SIZE)"

# Create checksum
sha256sum "$BACKUP_FILE" > "${BACKUP_FILE}.sha256"
sha256sum "$BACKUP_DIR/${BACKUP_NAME}.dump" > "$BACKUP_DIR/${BACKUP_NAME}.dump.sha256"
log "Checksums created"

# Upload to S3 if requested
if [ "$UPLOAD_S3" = true ] && [ -n "$S3_BUCKET" ]; then
    log_always "Uploading to S3: s3://$S3_BUCKET/"

    aws s3 cp "$BACKUP_FILE" "s3://$S3_BUCKET/backups/sql/${BACKUP_NAME}.sql.gz"
    aws s3 cp "${BACKUP_FILE}.sha256" "s3://$S3_BUCKET/backups/sql/${BACKUP_NAME}.sql.gz.sha256"
    aws s3 cp "$BACKUP_DIR/${BACKUP_NAME}.dump" "s3://$S3_BUCKET/backups/dump/${BACKUP_NAME}.dump"
    aws s3 cp "$BACKUP_DIR/${BACKUP_NAME}.dump.sha256" "s3://$S3_BUCKET/backups/dump/${BACKUP_NAME}.dump.sha256"

    log_always "Upload complete"
fi

# Cleanup old backups if requested
if [ "$CLEANUP" = true ]; then
    log_always "Cleaning up backups older than $RETENTION_DAYS days..."

    find "$BACKUP_DIR" -name "quant_trading_*.sql.gz" -mtime +$RETENTION_DAYS -delete 2>/dev/null || true
    find "$BACKUP_DIR" -name "quant_trading_*.dump" -mtime +$RETENTION_DAYS -delete 2>/dev/null || true
    find "$BACKUP_DIR" -name "quant_trading_*.sha256" -mtime +$RETENTION_DAYS -delete 2>/dev/null || true

    REMAINING=$(find "$BACKUP_DIR" -name "quant_trading_*.sql.gz" | wc -l)
    log "Remaining backups: $REMAINING"
fi

# Summary
log_always "Backup completed successfully"
log_always "Files:"
log_always "  - $BACKUP_FILE"
log_always "  - $BACKUP_DIR/${BACKUP_NAME}.dump"

# Return backup file path (useful for scripts)
echo "$BACKUP_FILE"
