# Database Recovery Procedures

This document describes backup, recovery, and disaster recovery procedures for the quantitative trading database.

## Overview

The system uses TimescaleDB (PostgreSQL extension) for time-series data storage. Recovery procedures follow PostgreSQL best practices with TimescaleDB-specific considerations.

## Backup Strategy

### Backup Types

| Type | Frequency | Retention | Use Case |
|------|-----------|-----------|----------|
| Full backup | Daily | 30 days | Complete database restore |
| WAL archiving | Continuous | 7 days | Point-in-time recovery |
| Logical backup | Weekly | 90 days | Cross-version migration |

### Backup Locations

- **Local**: `/var/backups/postgres/`
- **S3**: `s3://quant-trading-backups/`
- **Cross-region replica**: `s3://quant-trading-backups-dr/` (US-West-2)

## Daily Backup Procedure

### Automatic Backup

Daily backups run via cron at 02:00 UTC:

```cron
0 2 * * * /opt/quant-trading/scripts/backup_database.sh --upload-s3 --cleanup >> /var/log/backup.log 2>&1
```

### Manual Backup

```bash
# Full backup with S3 upload
./scripts/backup_database.sh --upload-s3 --verbose

# Local backup only
./scripts/backup_database.sh --verbose

# With cleanup of old backups
./scripts/backup_database.sh --cleanup
```

### Verify Backup

```bash
# Check backup file integrity
sha256sum -c /var/backups/postgres/quant_trading_YYYYMMDD_HHMMSS.sql.gz.sha256

# List recent backups
ls -la /var/backups/postgres/quant_trading_*.sql.gz | head -10

# Check S3 backups
aws s3 ls s3://quant-trading-backups/backups/sql/ --human-readable | tail -10
```

## Recovery Procedures

### Scenario 1: Single Table Recovery

For recovering a single table from backup:

```bash
# Extract specific table from backup
pg_restore \
    -h localhost \
    -U postgres \
    -d quant_trading_db \
    --table=market_prices \
    --clean \
    /var/backups/postgres/quant_trading_YYYYMMDD.dump
```

### Scenario 2: Full Database Recovery

For complete database restore:

```bash
# 1. Stop application connections
sudo systemctl stop quant-trading-api

# 2. Drop and recreate database
sudo -u postgres psql -c "DROP DATABASE IF EXISTS quant_trading_db;"
sudo -u postgres psql -c "CREATE DATABASE quant_trading_db;"

# 3. Enable TimescaleDB extension
sudo -u postgres psql -d quant_trading_db -c "CREATE EXTENSION IF NOT EXISTS timescaledb;"

# 4. Restore from backup
pg_restore \
    -h localhost \
    -U postgres \
    -d quant_trading_db \
    --verbose \
    /var/backups/postgres/quant_trading_YYYYMMDD.dump

# 5. Verify restoration
sudo -u postgres psql -d quant_trading_db -c "SELECT count(*) FROM market_prices;"

# 6. Restart application
sudo systemctl start quant-trading-api
```

### Scenario 3: Point-in-Time Recovery (PITR)

For recovery to a specific point in time:

```bash
# 1. Stop PostgreSQL
sudo systemctl stop postgresql

# 2. Clear current data directory
rm -rf /var/lib/postgresql/14/main/*

# 3. Restore base backup
pg_basebackup -D /var/lib/postgresql/14/main -F tar -z -P

# 4. Configure recovery
cat > /var/lib/postgresql/14/main/recovery.signal << EOF
EOF

cat >> /var/lib/postgresql/14/main/postgresql.auto.conf << EOF
restore_command = 'aws s3 cp s3://quant-trading-backups/wal/%f %p'
recovery_target_time = '2026-01-22 14:30:00 UTC'
recovery_target_action = 'promote'
EOF

# 5. Start PostgreSQL (will recover to target time)
sudo systemctl start postgresql

# 6. Verify recovery
sudo -u postgres psql -d quant_trading_db -c "SELECT max(time) FROM market_prices;"
```

### Scenario 4: Restore from S3

```bash
# Download backup from S3
aws s3 cp s3://quant-trading-backups/backups/dump/quant_trading_YYYYMMDD.dump /tmp/

# Verify checksum
aws s3 cp s3://quant-trading-backups/backups/dump/quant_trading_YYYYMMDD.dump.sha256 /tmp/
sha256sum -c /tmp/quant_trading_YYYYMMDD.dump.sha256

# Restore
pg_restore -h localhost -U postgres -d quant_trading_db /tmp/quant_trading_YYYYMMDD.dump
```

## TimescaleDB-Specific Recovery

### Restore Hypertables

When restoring TimescaleDB data:

```sql
-- Verify hypertables after restore
SELECT hypertable_name, num_chunks
FROM timescaledb_information.hypertables;

-- Check compression status
SELECT hypertable_name, total_bytes, compressed_total_bytes
FROM timescaledb_information.hypertable_compression_stats;

-- Recompress if needed
SELECT compress_chunk(c.chunk_full_name)
FROM timescaledb_information.chunks c
WHERE c.is_compressed = false;
```

### Restore Continuous Aggregates

```sql
-- Refresh continuous aggregates after restore
CALL refresh_continuous_aggregate('market_prices_1min', NULL, NULL);
CALL refresh_continuous_aggregate('market_prices_5min', NULL, NULL);
CALL refresh_continuous_aggregate('market_prices_daily', NULL, NULL);

-- Verify data
SELECT * FROM market_prices_daily ORDER BY bucket DESC LIMIT 10;
```

## Disaster Recovery

### Recovery Time Objective (RTO)

| Scenario | Target RTO |
|----------|------------|
| Single table | 15 minutes |
| Full database | 1 hour |
| PITR | 2 hours |
| Cross-region failover | 4 hours |

### Recovery Point Objective (RPO)

- **WAL archiving enabled**: < 5 minutes data loss
- **Daily backup only**: < 24 hours data loss

### Cross-Region Failover

1. Promote read replica in DR region:
```bash
aws rds promote-read-replica --db-instance-identifier quant-trading-dr
```

2. Update DNS/connection strings:
```bash
./scripts/failover_dns.sh production us-west-2
```

3. Verify connectivity:
```bash
psql -h quant-trading-db.us-west-2.rds.amazonaws.com -U quant_app -d quant_trading_db -c "SELECT 1;"
```

## Monitoring

### Backup Monitoring

```sql
-- Check last backup time (from application log)
SELECT max(time) as last_backup
FROM monitoring.backup_history;

-- Monitor database size growth
SELECT
    pg_size_pretty(pg_database_size('quant_trading_db')) as db_size,
    pg_size_pretty(pg_total_relation_size('market_prices')) as market_prices_size,
    pg_size_pretty(pg_total_relation_size('option_quotes')) as option_quotes_size;
```

### Alerts

Configure alerts for:
- Backup failure (no backup in 48 hours)
- WAL archive lag > 1 hour
- Database size > 80% of disk
- Replication lag > 5 minutes

## Testing

### Monthly Recovery Test

1. Restore to test environment
2. Verify data integrity
3. Run application smoke tests
4. Document results

### Quarterly DR Test

1. Failover to DR region
2. Run full test suite
3. Failback to primary
4. Document lessons learned

## Troubleshooting

### Backup Failures

```bash
# Check PostgreSQL logs
tail -100 /var/log/postgresql/postgresql-14-main.log

# Verify disk space
df -h /var/backups/postgres/

# Test database connectivity
pg_isready -h localhost -p 5432
```

### Restore Failures

```bash
# Check for active connections blocking restore
SELECT pid, usename, application_name, state, query
FROM pg_stat_activity
WHERE datname = 'quant_trading_db';

# Terminate connections
SELECT pg_terminate_backend(pid)
FROM pg_stat_activity
WHERE datname = 'quant_trading_db' AND pid <> pg_backend_pid();
```

### TimescaleDB Issues

```sql
-- Check TimescaleDB version
SELECT extversion FROM pg_extension WHERE extname = 'timescaledb';

-- Repair chunk metadata
SELECT _timescaledb_internal.repair_relation_acl();

-- Rebuild chunk index
REINDEX TABLE market_prices;
```

## Contact

For emergency database issues:
- On-call DBA: Check PagerDuty rotation
- Escalation: #database-emergency Slack channel
