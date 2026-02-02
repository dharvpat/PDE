# Incident Response Runbook

## Severity Levels

| Severity | Impact | Response Time | Examples |
|----------|--------|---------------|----------|
| P1 | Critical | 15 minutes | System down, data loss, execution failures |
| P2 | Major | 1 hour | Degraded performance, partial outage |
| P3 | Minor | 4 hours | Non-critical feature broken |
| P4 | Low | 24 hours | Cosmetic issues, documentation |

## Incident Response Process

```
┌──────────────────────────────────────────────────────────────────────┐
│                         Incident Timeline                            │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Alert        Acknowledge    Investigate    Mitigate    Resolve     │
│    │              │              │             │           │        │
│    ▼              ▼              ▼             ▼           ▼        │
│  ┌───┐        ┌───┐          ┌───┐        ┌───┐       ┌───┐       │
│  │ 0 │───────►│ 5 │─────────►│30 │───────►│60 │──────►│ ? │ mins  │
│  └───┘        └───┘          └───┘        └───┘       └───┘       │
│                                                                      │
│  • Page on-call  • Confirm issue   • Root cause    • Apply fix     │
│  • Create incident • Notify team   • Impact scope  • Verify        │
│                    • Start comms   • Identify fix  • Post-mortem   │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## P1: System Unavailable

### Symptoms
- API returning 5xx errors
- Health checks failing
- No trading activity
- Dashboard unreachable

### Initial Response (First 5 Minutes)

#### 1. Confirm Scope

```bash
# Check API health
curl -s https://api.trading.example.com/health | jq

# Check all services
kubectl get pods -n trading

# Check ingress
kubectl get ingress -n trading

# Check recent events
kubectl get events -n trading --sort-by='.lastTimestamp' | tail -20
```

#### 2. Check for Obvious Causes

```bash
# Recent deployments
kubectl rollout history deployment/api-service -n trading

# Node status
kubectl get nodes

# Resource exhaustion
kubectl top pods -n trading
kubectl top nodes
```

#### 3. Immediate Mitigation

**If recent deployment caused issue:**
```bash
# Rollback deployment
kubectl rollout undo deployment/api-service -n trading

# Verify rollback
kubectl rollout status deployment/api-service -n trading
```

**If pod crashes:**
```bash
# Get logs from crashed pod
kubectl logs <pod-name> -n trading --previous

# Delete problematic pod (will be recreated)
kubectl delete pod <pod-name> -n trading
```

**If resource exhaustion:**
```bash
# Scale up
kubectl scale deployment/api-service --replicas=5 -n trading

# Request emergency node addition
# Contact cloud provider or infrastructure team
```

### Investigation (Minutes 5-30)

#### Check Logs

```bash
# Aggregate logs from all API pods
kubectl logs -l app=api-service -n trading --tail=100 --since=10m

# Check specific pod
kubectl logs <pod-name> -n trading -f

# Check events
kubectl describe pod <pod-name> -n trading
```

#### Check Dependencies

```bash
# Database connectivity
kubectl exec -it deploy/api-service -n trading -- \
  python -c "from quant_trading.database import engine; print(engine.connect())"

# Redis connectivity
kubectl exec -it deploy/api-service -n trading -- \
  redis-cli -h redis PING

# Check database status
kubectl exec -it timescaledb-0 -n trading -- \
  psql -U postgres -c "SELECT count(*) FROM pg_stat_activity;"
```

#### Check External Dependencies

```bash
# Market data provider
curl -s https://api.marketdata.provider.com/status

# Broker API
curl -s https://api.broker.com/health
```

### Resolution

Document the root cause and fix applied:

```markdown
## Incident Summary
- **Incident ID**: INC-2024-001
- **Severity**: P1
- **Duration**: 45 minutes
- **Impact**: Complete API outage

## Timeline
- 09:15 - Alert triggered
- 09:17 - On-call acknowledged
- 09:25 - Root cause identified (OOM in calibration pod)
- 09:35 - Memory limits increased, pods restarted
- 09:50 - Service restored, monitoring for stability
- 10:00 - Incident closed

## Root Cause
Calibration service exceeded memory limits during large batch calibration.

## Action Items
- [ ] Increase memory limits for calibration pods
- [ ] Add memory-based autoscaling
- [ ] Implement batch size limits
```

---

## P1: Execution Failures

### Symptoms
- Orders not being filled
- Error responses from broker
- Position mismatches
- Missing execution reports

### Immediate Response

#### 1. Stop New Order Submission

```bash
# Emergency: pause all strategies
curl -X POST https://api.trading.example.com/v1/admin/pause-all \
  -H "Authorization: Bearer $ADMIN_TOKEN"

# Or via kubectl
kubectl scale deployment/execution-service --replicas=0 -n trading
```

#### 2. Reconcile Positions

```bash
# Get positions from broker
kubectl exec -it deploy/api-service -n trading -- \
  python -m quant_trading.tools.reconcile_positions

# Compare with internal records
# MANUALLY VERIFY before taking action
```

#### 3. Investigate Broker Connection

```bash
# Check broker API status
curl -s https://api.broker.com/v1/status

# Check our connection logs
kubectl logs -l app=execution-service -n trading --tail=200 | grep -i "broker\|error\|fail"

# Test broker authentication
kubectl exec -it deploy/api-service -n trading -- \
  python -c "from quant_trading.execution import BrokerClient; c = BrokerClient(); print(c.test_connection())"
```

### Resolution Checklist

- [ ] All pending orders cancelled or confirmed
- [ ] Positions reconciled with broker
- [ ] P&L verified
- [ ] Root cause identified
- [ ] Preventive measures implemented

---

## P2: High Latency

### Symptoms
- API response time >5 seconds
- Dashboard loading slowly
- Calibration taking >5 minutes
- User complaints

### Investigation

#### 1. Identify Bottleneck

```bash
# Check API latency metrics
curl -s localhost:9090/api/v1/query?query=histogram_quantile\(0.99,rate\(http_request_duration_seconds_bucket[5m]\)\)

# Check database query latency
kubectl exec -it timescaledb-0 -n trading -- \
  psql -U postgres -c "
    SELECT query, calls, mean_time, max_time
    FROM pg_stat_statements
    ORDER BY mean_time DESC
    LIMIT 10;"

# Check for slow queries
kubectl exec -it timescaledb-0 -n trading -- \
  psql -U postgres -c "
    SELECT pid, now() - pg_stat_activity.query_start AS duration, query
    FROM pg_stat_activity
    WHERE state = 'active'
    ORDER BY duration DESC;"
```

#### 2. Check Resource Utilization

```bash
# Pod resources
kubectl top pods -n trading

# Database connections
kubectl exec -it timescaledb-0 -n trading -- \
  psql -U postgres -c "SELECT count(*) FROM pg_stat_activity;"

# Redis memory
kubectl exec -it redis-0 -n trading -- redis-cli INFO memory
```

#### 3. Common Fixes

**Slow database queries:**
```sql
-- Add missing index
CREATE INDEX CONCURRENTLY idx_xxx ON table_name (column);

-- Update statistics
ANALYZE table_name;

-- Kill long-running query (caution!)
SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE pid = <pid>;
```

**Memory pressure:**
```bash
# Scale horizontally
kubectl scale deployment/api-service --replicas=5 -n trading

# Restart pods with memory leaks
kubectl rollout restart deployment/api-service -n trading
```

**Connection pool exhaustion:**
```bash
# Check pool status
kubectl exec -it deploy/api-service -n trading -- \
  python -c "from quant_trading.database import get_pool_status; print(get_pool_status())"

# Restart to reset connections
kubectl rollout restart deployment/api-service -n trading
```

---

## P2: Calibration Failures

### Symptoms
- Model parameters stale (>24h old)
- High calibration RMSE
- Missing volatility surfaces
- Signal generation using old data

### Investigation

```bash
# Check calibration service status
kubectl get pods -l app=calibration-service -n trading

# Check logs for errors
kubectl logs -l app=calibration-service -n trading --tail=100

# Check last successful calibration
curl -s https://api.trading.example.com/v1/calibration/status | jq

# Manual calibration test
kubectl exec -it deploy/calibration-service -n trading -- \
  python -m quant_trading.tools.test_calibration --symbol SPY
```

### Common Issues

**Market data unavailable:**
```bash
# Check data ingestion
kubectl logs -l app=data-ingestion -n trading --tail=50

# Verify recent market data
curl -s "https://api.trading.example.com/v1/market-data/SPY/latest"
```

**Optimization not converging:**
```bash
# Check calibration parameters
kubectl exec -it deploy/calibration-service -n trading -- \
  python -c "
    from quant_trading.calibration import HestonCalibrator
    c = HestonCalibrator()
    print(c.get_default_config())
  "

# Try with relaxed constraints
# May need code change or config update
```

### Recovery

```bash
# Force recalibration
curl -X POST https://api.trading.example.com/v1/calibration/heston \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "SPY", "force": true}'

# Use cached/previous parameters as fallback (automatic)
# Verify fallback is working
curl -s https://api.trading.example.com/v1/calibration/heston?symbol=SPY | jq '.calibrated_at'
```

---

## Communication Templates

### Initial Alert (Internal)

```
Subject: [P1] Trading System Incident - API Unavailable

Status: INVESTIGATING
Impact: Trading API is returning 5xx errors. No new orders being processed.
Started: 2024-01-15 09:15 UTC

Incident Commander: [Name]
Current Actions: Investigating root cause
ETA for Update: 15 minutes

War Room: [Slack Channel]
```

### Status Update

```
Subject: [P1] Trading System Incident - UPDATE 1

Status: MITIGATING
Impact: API restored, monitoring for stability

Progress:
- Root cause identified: Memory exhaustion in calibration pods
- Fix applied: Increased memory limits, restarted pods
- Monitoring: Latency back to normal levels

ETA for Resolution: 15 minutes
Next Update: 30 minutes or on resolution
```

### Resolution

```
Subject: [P1] Trading System Incident - RESOLVED

Status: RESOLVED
Duration: 45 minutes (09:15 - 10:00 UTC)
Impact: Complete API outage affecting all users

Root Cause: Calibration service OOM due to large batch processing

Resolution: Memory limits increased, batch size limits added

Post-Incident Review: Scheduled for 2024-01-16 14:00 UTC
```

---

## Escalation Path

| Time | Action |
|------|--------|
| 0-15 min | On-call engineer investigates |
| 15-30 min | Escalate to engineering lead |
| 30-60 min | Escalate to VP Engineering |
| 60+ min | Escalate to CTO |
| Trading impact | Notify compliance immediately |

### Contact Information

| Role | Primary | Backup |
|------|---------|--------|
| On-call | PagerDuty rotation | - |
| Engineering Lead | [Phone] | [Phone] |
| VP Engineering | [Phone] | [Phone] |
| Compliance | [Phone] | [Phone] |
