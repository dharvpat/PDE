# Kubernetes Deployment Guide

## Overview

The Quantitative Trading System is deployed on Kubernetes for production workloads. This guide covers the deployment architecture, configuration, and operational procedures.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Kubernetes Cluster                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐       │
│  │   Ingress       │────►│   API Service   │────►│  Calibration    │       │
│  │   (NGINX)       │     │   (2-5 pods)    │     │  (1-3 pods)     │       │
│  └─────────────────┘     └─────────────────┘     └─────────────────┘       │
│                                   │                      │                 │
│                                   ▼                      ▼                 │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐       │
│  │   Data Ingest   │────►│   TimescaleDB   │◄────│   Signals       │       │
│  │   (1-2 pods)    │     │  (StatefulSet)  │     │   (1-2 pods)    │       │
│  └─────────────────┘     └─────────────────┘     └─────────────────┘       │
│                                   │                      │                 │
│                                   ▼                      ▼                 │
│  ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐       │
│  │    Redis        │◄───►│   Risk Mgmt     │◄───►│   Execution     │       │
│  │  (StatefulSet)  │     │   (1 pod)       │     │   (1 pod)       │       │
│  └─────────────────┘     └─────────────────┘     └─────────────────┘       │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────┐       │
│  │                    Monitoring Stack                              │       │
│  │  Prometheus │ Grafana │ AlertManager │ Loki                     │       │
│  └─────────────────────────────────────────────────────────────────┘       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Prerequisites

- Kubernetes cluster (1.28+)
- kubectl configured
- Helm 3.13+
- Storage class with dynamic provisioning
- DNS configured for ingress

## Namespace Setup

```bash
# Create namespace
kubectl create namespace trading

# Set as default (optional)
kubectl config set-context --current --namespace=trading
```

## Deployment Methods

### Method 1: Kustomize

```bash
# Development environment
kubectl apply -k deploy/k8s/overlays/dev

# Production environment
kubectl apply -k deploy/k8s/overlays/prod
```

### Method 2: Helm

```bash
# Add local chart
helm install quant-trading deploy/helm/quant-trading \
  --namespace trading \
  -f deploy/helm/quant-trading/values-prod.yaml

# Upgrade
helm upgrade quant-trading deploy/helm/quant-trading \
  --namespace trading \
  -f deploy/helm/quant-trading/values-prod.yaml
```

## Resource Configuration

### API Service

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-service
  namespace: trading
spec:
  replicas: 2
  selector:
    matchLabels:
      app: api-service
  template:
    metadata:
      labels:
        app: api-service
    spec:
      containers:
      - name: api
        image: quant-trading/api:latest
        resources:
          requests:
            cpu: "500m"
            memory: "512Mi"
          limits:
            cpu: "2000m"
            memory: "2Gi"
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: url
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

### Calibration Service (High CPU)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: calibration-service
  namespace: trading
spec:
  replicas: 1
  selector:
    matchLabels:
      app: calibration-service
  template:
    spec:
      containers:
      - name: calibration
        image: quant-trading/calibration:latest
        resources:
          requests:
            cpu: "2000m"
            memory: "4Gi"
          limits:
            cpu: "4000m"
            memory: "8Gi"
        env:
        - name: OMP_NUM_THREADS
          value: "4"
```

### StatefulSets (Databases)

#### TimescaleDB

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: timescaledb
  namespace: trading
spec:
  serviceName: timescaledb
  replicas: 1
  selector:
    matchLabels:
      app: timescaledb
  template:
    spec:
      containers:
      - name: timescaledb
        image: timescale/timescaledb:latest-pg15
        resources:
          requests:
            cpu: "1000m"
            memory: "4Gi"
          limits:
            cpu: "4000m"
            memory: "16Gi"
        volumeMounts:
        - name: data
          mountPath: /var/lib/postgresql/data
        env:
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: password
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: ["ReadWriteOnce"]
      storageClassName: fast-ssd
      resources:
        requests:
          storage: 500Gi
```

## Secrets Management

### Creating Secrets

```bash
# Database credentials
kubectl create secret generic db-credentials \
  --namespace trading \
  --from-literal=url="postgresql://user:password@timescaledb:5432/trading" \
  --from-literal=password="secure_password"

# API keys
kubectl create secret generic api-keys \
  --namespace trading \
  --from-literal=jwt-secret="your-jwt-secret" \
  --from-literal=market-data-key="your-market-data-api-key"
```

### External Secrets (Production)

```yaml
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: db-credentials
  namespace: trading
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: vault-backend
    kind: SecretStore
  target:
    name: db-credentials
  data:
  - secretKey: url
    remoteRef:
      key: trading/database
      property: url
```

## Ingress Configuration

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: trading-ingress
  namespace: trading
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
spec:
  tls:
  - hosts:
    - api.trading.example.com
    secretName: trading-tls
  rules:
  - host: api.trading.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: api-service
            port:
              number: 8000
```

## Horizontal Pod Autoscaling

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api-service-hpa
  namespace: trading
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api-service
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## Network Policies

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: api-network-policy
  namespace: trading
spec:
  podSelector:
    matchLabels:
      app: api-service
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: timescaledb
    ports:
    - protocol: TCP
      port: 5432
  - to:
    - podSelector:
        matchLabels:
          app: redis
    ports:
    - protocol: TCP
      port: 6379
```

## Monitoring Setup

### Prometheus ServiceMonitor

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: trading-services
  namespace: trading
spec:
  selector:
    matchLabels:
      monitoring: enabled
  endpoints:
  - port: metrics
    interval: 30s
    path: /metrics
```

### Grafana Dashboard

Import dashboards from `deploy/k8s/monitoring/dashboards/`.

## Rolling Updates

```bash
# Update image
kubectl set image deployment/api-service \
  api=quant-trading/api:v1.2.0 \
  --namespace trading

# Watch rollout
kubectl rollout status deployment/api-service --namespace trading

# Rollback if needed
kubectl rollout undo deployment/api-service --namespace trading
```

## Troubleshooting

### Check Pod Status

```bash
# Get all pods
kubectl get pods -n trading

# Describe problem pod
kubectl describe pod <pod-name> -n trading

# Get logs
kubectl logs <pod-name> -n trading --tail=100

# Previous container logs
kubectl logs <pod-name> -n trading --previous
```

### Database Connection Issues

```bash
# Test database connectivity
kubectl run -it --rm debug \
  --image=postgres:15 \
  --restart=Never \
  --namespace trading \
  -- psql -h timescaledb -U postgres -d trading

# Check service endpoints
kubectl get endpoints timescaledb -n trading
```

### Resource Issues

```bash
# Check resource usage
kubectl top pods -n trading

# Check node resources
kubectl top nodes

# Describe node for events
kubectl describe node <node-name>
```

## Disaster Recovery

### Backup Procedures

```bash
# Database backup (CronJob)
kubectl create -f deploy/k8s/jobs/database-backup.yaml

# Manual backup
kubectl exec -it timescaledb-0 -n trading -- \
  pg_dump -Fc trading > backup.dump
```

### Restore Procedures

```bash
# Restore database
kubectl exec -i timescaledb-0 -n trading -- \
  pg_restore -d trading < backup.dump
```

## Production Checklist

- [ ] All secrets stored in external secrets manager
- [ ] TLS enabled on ingress
- [ ] Network policies applied
- [ ] HPA configured for API service
- [ ] PodDisruptionBudgets set
- [ ] Monitoring and alerting configured
- [ ] Backup jobs scheduled
- [ ] Resource limits set on all containers
- [ ] Health probes configured
- [ ] RBAC configured for service accounts
