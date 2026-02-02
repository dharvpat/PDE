"""
Grafana Dashboard Configuration Module.

This module provides:
- Dashboard JSON generation for Grafana
- Pre-configured trading system dashboards
- Dashboard templating and variables
- Provisioning helpers

Dashboards included:
- Trading Overview: P&L, positions, orders
- Risk Dashboard: VaR, Greeks, drawdown
- System Health: CPU, memory, latency
- Data Quality: Gaps, validation, freshness
- Model Performance: Calibration, predictions
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class PanelType(Enum):
    """Grafana panel types."""
    GRAPH = "graph"
    STAT = "stat"
    GAUGE = "gauge"
    BAR_GAUGE = "bargauge"
    TABLE = "table"
    HEATMAP = "heatmap"
    TEXT = "text"
    ROW = "row"
    TIME_SERIES = "timeseries"
    PIE_CHART = "piechart"
    ALERT_LIST = "alertlist"
    LOGS = "logs"


class AggregationType(Enum):
    """Prometheus aggregation types."""
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    RATE = "rate"
    IRATE = "irate"


@dataclass
class PrometheusTarget:
    """Prometheus query target."""
    expr: str
    legend_format: str = ""
    ref_id: str = "A"
    instant: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expr": self.expr,
            "legendFormat": self.legend_format,
            "refId": self.ref_id,
            "instant": self.instant
        }


@dataclass
class Threshold:
    """Panel threshold configuration."""
    value: float
    color: str  # "green", "yellow", "red", or hex color
    op: str = "gt"  # gt, lt

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "color": self.color,
            "op": self.op
        }


@dataclass
class Panel:
    """Grafana panel configuration."""
    title: str
    panel_type: PanelType
    targets: List[PrometheusTarget]
    grid_pos: Dict[str, int]
    description: str = ""
    unit: str = ""
    thresholds: List[Threshold] = field(default_factory=list)
    options: Dict[str, Any] = field(default_factory=dict)
    field_config: Dict[str, Any] = field(default_factory=dict)

    _id_counter: int = field(default=1, repr=False)

    def to_dict(self, panel_id: int) -> Dict[str, Any]:
        panel = {
            "id": panel_id,
            "title": self.title,
            "type": self.panel_type.value,
            "gridPos": self.grid_pos,
            "targets": [t.to_dict() for t in self.targets],
            "description": self.description,
        }

        if self.options:
            panel["options"] = self.options

        if self.field_config:
            panel["fieldConfig"] = self.field_config
        else:
            # Default field config
            panel["fieldConfig"] = {
                "defaults": {
                    "unit": self.unit,
                    "thresholds": {
                        "mode": "absolute",
                        "steps": [{"value": None, "color": "green"}] + [
                            {"value": t.value, "color": t.color}
                            for t in self.thresholds
                        ]
                    }
                },
                "overrides": []
            }

        return panel


@dataclass
class DashboardVariable:
    """Dashboard template variable."""
    name: str
    label: str
    var_type: str = "query"  # query, custom, constant, datasource
    query: str = ""
    options: List[Dict[str, str]] = field(default_factory=list)
    multi: bool = False
    include_all: bool = False

    def to_dict(self) -> Dict[str, Any]:
        var = {
            "name": self.name,
            "label": self.label,
            "type": self.var_type,
            "multi": self.multi,
            "includeAll": self.include_all
        }

        if self.var_type == "query":
            var["query"] = self.query
            var["datasource"] = {"type": "prometheus", "uid": "${DS_PROMETHEUS}"}
        elif self.var_type == "custom":
            var["options"] = self.options
            var["query"] = ",".join([o.get("value", "") for o in self.options])

        return var


@dataclass
class Dashboard:
    """Grafana dashboard configuration."""
    title: str
    uid: str
    description: str = ""
    tags: List[str] = field(default_factory=list)
    panels: List[Panel] = field(default_factory=list)
    variables: List[DashboardVariable] = field(default_factory=list)
    refresh: str = "30s"
    time_from: str = "now-1h"
    time_to: str = "now"
    timezone: str = "browser"
    editable: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert dashboard to Grafana JSON format."""
        return {
            "title": self.title,
            "uid": self.uid,
            "description": self.description,
            "tags": self.tags,
            "timezone": self.timezone,
            "editable": self.editable,
            "refresh": self.refresh,
            "time": {
                "from": self.time_from,
                "to": self.time_to
            },
            "templating": {
                "list": [v.to_dict() for v in self.variables]
            },
            "panels": [
                p.to_dict(i + 1) for i, p in enumerate(self.panels)
            ],
            "schemaVersion": 38,
            "version": 1
        }

    def to_json(self, indent: int = 2) -> str:
        """Export dashboard as JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, filepath: str) -> None:
        """Save dashboard to JSON file."""
        with open(filepath, 'w') as f:
            f.write(self.to_json())
        logger.info(f"Saved dashboard to {filepath}")


# =============================================================================
# Pre-built Dashboard Definitions
# =============================================================================

def create_trading_overview_dashboard() -> Dashboard:
    """Create the main trading overview dashboard."""
    return Dashboard(
        title="Trading System - Overview",
        uid="trading-overview",
        description="Overview of trading system performance, P&L, and positions",
        tags=["trading", "overview"],
        variables=[
            DashboardVariable(
                name="strategy",
                label="Strategy",
                var_type="query",
                query='label_values(trading_realized_pnl_total, strategy)',
                multi=True,
                include_all=True
            )
        ],
        panels=[
            # Row: Key Metrics
            Panel(
                title="Daily P&L",
                panel_type=PanelType.STAT,
                grid_pos={"x": 0, "y": 0, "w": 6, "h": 4},
                targets=[
                    PrometheusTarget(
                        expr='sum(trading_daily_pnl{strategy=~"$strategy"})',
                        legend_format="Daily P&L"
                    )
                ],
                unit="currencyUSD",
                thresholds=[
                    Threshold(value=-10000, color="red"),
                    Threshold(value=0, color="yellow"),
                    Threshold(value=10000, color="green")
                ]
            ),
            Panel(
                title="Cumulative P&L",
                panel_type=PanelType.STAT,
                grid_pos={"x": 6, "y": 0, "w": 6, "h": 4},
                targets=[
                    PrometheusTarget(
                        expr='sum(trading_cumulative_pnl{strategy=~"$strategy"})',
                        legend_format="Cumulative P&L"
                    )
                ],
                unit="currencyUSD"
            ),
            Panel(
                title="Open Positions",
                panel_type=PanelType.STAT,
                grid_pos={"x": 12, "y": 0, "w": 4, "h": 4},
                targets=[
                    PrometheusTarget(
                        expr='sum(trading_open_positions_count{strategy=~"$strategy"})',
                        legend_format="Positions"
                    )
                ],
                unit="short"
            ),
            Panel(
                title="Total Exposure",
                panel_type=PanelType.GAUGE,
                grid_pos={"x": 16, "y": 0, "w": 4, "h": 4},
                targets=[
                    PrometheusTarget(
                        expr='sum(trading_total_exposure_dollars{strategy=~"$strategy"})',
                        legend_format="Exposure"
                    )
                ],
                unit="currencyUSD",
                thresholds=[
                    Threshold(value=500000, color="yellow"),
                    Threshold(value=1000000, color="red")
                ]
            ),
            Panel(
                title="System Health",
                panel_type=PanelType.GAUGE,
                grid_pos={"x": 20, "y": 0, "w": 4, "h": 4},
                targets=[
                    PrometheusTarget(
                        expr='trading_system_health',
                        legend_format="Health"
                    )
                ],
                unit="percentunit",
                thresholds=[
                    Threshold(value=0.5, color="red"),
                    Threshold(value=0.8, color="yellow"),
                    Threshold(value=0.95, color="green")
                ]
            ),

            # Row: P&L Chart
            Panel(
                title="P&L Over Time",
                panel_type=PanelType.TIME_SERIES,
                grid_pos={"x": 0, "y": 4, "w": 16, "h": 8},
                targets=[
                    PrometheusTarget(
                        expr='trading_cumulative_pnl{strategy=~"$strategy"}',
                        legend_format="{{strategy}}"
                    )
                ],
                unit="currencyUSD"
            ),
            Panel(
                title="Orders by Status",
                panel_type=PanelType.PIE_CHART,
                grid_pos={"x": 16, "y": 4, "w": 8, "h": 8},
                targets=[
                    PrometheusTarget(
                        expr='sum by (status) (increase(trading_orders_total{strategy=~"$strategy"}[24h]))',
                        legend_format="{{status}}"
                    )
                ]
            ),

            # Row: Orders
            Panel(
                title="Orders Created (Rate)",
                panel_type=PanelType.TIME_SERIES,
                grid_pos={"x": 0, "y": 12, "w": 8, "h": 6},
                targets=[
                    PrometheusTarget(
                        expr='sum(rate(trading_orders_created_total{strategy=~"$strategy"}[5m])) by (side)',
                        legend_format="{{side}}"
                    )
                ],
                unit="ops"
            ),
            Panel(
                title="Order Fill Rate",
                panel_type=PanelType.TIME_SERIES,
                grid_pos={"x": 8, "y": 12, "w": 8, "h": 6},
                targets=[
                    PrometheusTarget(
                        expr='sum(rate(trading_orders_filled_total{strategy=~"$strategy"}[5m])) / sum(rate(trading_orders_created_total{strategy=~"$strategy"}[5m]))',
                        legend_format="Fill Rate"
                    )
                ],
                unit="percentunit"
            ),
            Panel(
                title="Order Rejections",
                panel_type=PanelType.TIME_SERIES,
                grid_pos={"x": 16, "y": 12, "w": 8, "h": 6},
                targets=[
                    PrometheusTarget(
                        expr='sum(rate(trading_orders_rejected_total{strategy=~"$strategy"}[5m])) by (reason)',
                        legend_format="{{reason}}"
                    )
                ],
                unit="ops"
            )
        ]
    )


def create_risk_dashboard() -> Dashboard:
    """Create the risk monitoring dashboard."""
    return Dashboard(
        title="Trading System - Risk",
        uid="trading-risk",
        description="Risk metrics including VaR, Greeks, and drawdown",
        tags=["trading", "risk"],
        variables=[
            DashboardVariable(
                name="strategy",
                label="Strategy",
                var_type="query",
                query='label_values(trading_portfolio_delta, strategy)',
                multi=True,
                include_all=True
            )
        ],
        panels=[
            # Row: Key Risk Metrics
            Panel(
                title="95% VaR (1-Day)",
                panel_type=PanelType.STAT,
                grid_pos={"x": 0, "y": 0, "w": 6, "h": 4},
                targets=[
                    PrometheusTarget(
                        expr='trading_portfolio_var{confidence_level="95", time_horizon="1d"}',
                        legend_format="VaR"
                    )
                ],
                unit="currencyUSD",
                thresholds=[
                    Threshold(value=15000, color="yellow"),
                    Threshold(value=25000, color="red")
                ]
            ),
            Panel(
                title="Max Drawdown",
                panel_type=PanelType.STAT,
                grid_pos={"x": 6, "y": 0, "w": 6, "h": 4},
                targets=[
                    PrometheusTarget(
                        expr='max(trading_max_drawdown_percent{strategy=~"$strategy"})',
                        legend_format="Max DD"
                    )
                ],
                unit="percent",
                thresholds=[
                    Threshold(value=10, color="yellow"),
                    Threshold(value=20, color="red")
                ]
            ),
            Panel(
                title="Portfolio Volatility",
                panel_type=PanelType.STAT,
                grid_pos={"x": 12, "y": 0, "w": 6, "h": 4},
                targets=[
                    PrometheusTarget(
                        expr='trading_portfolio_volatility{strategy=~"$strategy"}',
                        legend_format="Vol"
                    )
                ],
                unit="percentunit"
            ),
            Panel(
                title="Sharpe Ratio (30d)",
                panel_type=PanelType.STAT,
                grid_pos={"x": 18, "y": 0, "w": 6, "h": 4},
                targets=[
                    PrometheusTarget(
                        expr='trading_rolling_sharpe_ratio{strategy=~"$strategy", window="30d"}',
                        legend_format="Sharpe"
                    )
                ],
                unit="short",
                thresholds=[
                    Threshold(value=0, color="red"),
                    Threshold(value=0.5, color="yellow"),
                    Threshold(value=1.0, color="green")
                ]
            ),

            # Row: Greeks
            Panel(
                title="Portfolio Delta",
                panel_type=PanelType.TIME_SERIES,
                grid_pos={"x": 0, "y": 4, "w": 6, "h": 6},
                targets=[
                    PrometheusTarget(
                        expr='trading_portfolio_delta{strategy=~"$strategy"}',
                        legend_format="{{strategy}}"
                    )
                ],
                unit="short"
            ),
            Panel(
                title="Portfolio Gamma",
                panel_type=PanelType.TIME_SERIES,
                grid_pos={"x": 6, "y": 4, "w": 6, "h": 6},
                targets=[
                    PrometheusTarget(
                        expr='trading_portfolio_gamma{strategy=~"$strategy"}',
                        legend_format="{{strategy}}"
                    )
                ],
                unit="short"
            ),
            Panel(
                title="Portfolio Vega",
                panel_type=PanelType.TIME_SERIES,
                grid_pos={"x": 12, "y": 4, "w": 6, "h": 6},
                targets=[
                    PrometheusTarget(
                        expr='trading_portfolio_vega{strategy=~"$strategy"}',
                        legend_format="{{strategy}}"
                    )
                ],
                unit="short"
            ),
            Panel(
                title="Portfolio Theta",
                panel_type=PanelType.TIME_SERIES,
                grid_pos={"x": 18, "y": 4, "w": 6, "h": 6},
                targets=[
                    PrometheusTarget(
                        expr='trading_portfolio_theta{strategy=~"$strategy"}',
                        legend_format="{{strategy}}"
                    )
                ],
                unit="short"
            ),

            # Row: Drawdown
            Panel(
                title="Drawdown Over Time",
                panel_type=PanelType.TIME_SERIES,
                grid_pos={"x": 0, "y": 10, "w": 24, "h": 6},
                targets=[
                    PrometheusTarget(
                        expr='trading_current_drawdown_percent{strategy=~"$strategy"}',
                        legend_format="{{strategy}} Current DD"
                    ),
                    PrometheusTarget(
                        expr='trading_max_drawdown_percent{strategy=~"$strategy"}',
                        legend_format="{{strategy}} Max DD",
                        ref_id="B"
                    )
                ],
                unit="percent"
            )
        ]
    )


def create_system_health_dashboard() -> Dashboard:
    """Create the system health dashboard."""
    return Dashboard(
        title="Trading System - System Health",
        uid="trading-system-health",
        description="System resource usage and performance metrics",
        tags=["trading", "system", "infrastructure"],
        panels=[
            # Row: Key System Metrics
            Panel(
                title="CPU Usage",
                panel_type=PanelType.GAUGE,
                grid_pos={"x": 0, "y": 0, "w": 6, "h": 4},
                targets=[
                    PrometheusTarget(
                        expr='trading_cpu_usage_percent',
                        legend_format="CPU"
                    )
                ],
                unit="percent",
                thresholds=[
                    Threshold(value=70, color="yellow"),
                    Threshold(value=90, color="red")
                ]
            ),
            Panel(
                title="Memory Usage",
                panel_type=PanelType.GAUGE,
                grid_pos={"x": 6, "y": 0, "w": 6, "h": 4},
                targets=[
                    PrometheusTarget(
                        expr='trading_memory_usage_percent',
                        legend_format="Memory"
                    )
                ],
                unit="percent",
                thresholds=[
                    Threshold(value=70, color="yellow"),
                    Threshold(value=85, color="red")
                ]
            ),
            Panel(
                title="Process Uptime",
                panel_type=PanelType.STAT,
                grid_pos={"x": 12, "y": 0, "w": 6, "h": 4},
                targets=[
                    PrometheusTarget(
                        expr='trading_process_uptime_seconds',
                        legend_format="Uptime"
                    )
                ],
                unit="s"
            ),
            Panel(
                title="Active Threads",
                panel_type=PanelType.STAT,
                grid_pos={"x": 18, "y": 0, "w": 6, "h": 4},
                targets=[
                    PrometheusTarget(
                        expr='trading_thread_count',
                        legend_format="Threads"
                    )
                ],
                unit="short"
            ),

            # Row: Latency Metrics
            Panel(
                title="Signal Generation Latency",
                panel_type=PanelType.TIME_SERIES,
                grid_pos={"x": 0, "y": 4, "w": 8, "h": 6},
                targets=[
                    PrometheusTarget(
                        expr='histogram_quantile(0.95, rate(trading_signal_generation_latency_seconds_bucket[5m]))',
                        legend_format="p95"
                    ),
                    PrometheusTarget(
                        expr='histogram_quantile(0.99, rate(trading_signal_generation_latency_seconds_bucket[5m]))',
                        legend_format="p99",
                        ref_id="B"
                    )
                ],
                unit="s"
            ),
            Panel(
                title="Order Submission Latency",
                panel_type=PanelType.TIME_SERIES,
                grid_pos={"x": 8, "y": 4, "w": 8, "h": 6},
                targets=[
                    PrometheusTarget(
                        expr='histogram_quantile(0.95, rate(trading_order_submission_latency_seconds_bucket[5m]))',
                        legend_format="p95"
                    ),
                    PrometheusTarget(
                        expr='histogram_quantile(0.99, rate(trading_order_submission_latency_seconds_bucket[5m]))',
                        legend_format="p99",
                        ref_id="B"
                    )
                ],
                unit="s"
            ),
            Panel(
                title="Data Ingestion Latency",
                panel_type=PanelType.TIME_SERIES,
                grid_pos={"x": 16, "y": 4, "w": 8, "h": 6},
                targets=[
                    PrometheusTarget(
                        expr='histogram_quantile(0.95, rate(trading_data_ingestion_latency_seconds_bucket[5m])) by (source)',
                        legend_format="{{source}} p95"
                    )
                ],
                unit="s"
            ),

            # Row: Component Status
            Panel(
                title="Component Health",
                panel_type=PanelType.TABLE,
                grid_pos={"x": 0, "y": 10, "w": 12, "h": 6},
                targets=[
                    PrometheusTarget(
                        expr='trading_component_status',
                        legend_format="{{component}}",
                        instant=True
                    )
                ]
            ),
            Panel(
                title="API Latency by Endpoint",
                panel_type=PanelType.TIME_SERIES,
                grid_pos={"x": 12, "y": 10, "w": 12, "h": 6},
                targets=[
                    PrometheusTarget(
                        expr='histogram_quantile(0.95, rate(trading_api_latency_seconds_bucket[5m])) by (endpoint)',
                        legend_format="{{endpoint}}"
                    )
                ],
                unit="s"
            )
        ]
    )


def create_data_quality_dashboard() -> Dashboard:
    """Create the data quality dashboard."""
    return Dashboard(
        title="Trading System - Data Quality",
        uid="trading-data-quality",
        description="Data quality metrics including gaps, validation, and freshness",
        tags=["trading", "data"],
        panels=[
            # Row: Data Freshness
            Panel(
                title="Data Age (seconds)",
                panel_type=PanelType.TIME_SERIES,
                grid_pos={"x": 0, "y": 0, "w": 12, "h": 6},
                targets=[
                    PrometheusTarget(
                        expr='trading_data_age_seconds',
                        legend_format="{{symbol}} {{data_type}}"
                    )
                ],
                unit="s",
                thresholds=[
                    Threshold(value=30, color="yellow"),
                    Threshold(value=60, color="red")
                ]
            ),
            Panel(
                title="Data Completeness",
                panel_type=PanelType.TIME_SERIES,
                grid_pos={"x": 12, "y": 0, "w": 12, "h": 6},
                targets=[
                    PrometheusTarget(
                        expr='trading_data_completeness_percent',
                        legend_format="{{symbol}}"
                    )
                ],
                unit="percent"
            ),

            # Row: Gaps and Validation
            Panel(
                title="Data Gaps Detected (Rate)",
                panel_type=PanelType.TIME_SERIES,
                grid_pos={"x": 0, "y": 6, "w": 12, "h": 6},
                targets=[
                    PrometheusTarget(
                        expr='sum(rate(trading_data_gaps_detected_total[1h])) by (symbol)',
                        legend_format="{{symbol}}"
                    )
                ],
                unit="ops"
            ),
            Panel(
                title="Validation Failures (Rate)",
                panel_type=PanelType.TIME_SERIES,
                grid_pos={"x": 12, "y": 6, "w": 12, "h": 6},
                targets=[
                    PrometheusTarget(
                        expr='sum(rate(trading_data_validation_failures_total[1h])) by (check_type)',
                        legend_format="{{check_type}}"
                    )
                ],
                unit="ops"
            ),

            # Row: Model Quality
            Panel(
                title="Calibration RMSE",
                panel_type=PanelType.TIME_SERIES,
                grid_pos={"x": 0, "y": 12, "w": 12, "h": 6},
                targets=[
                    PrometheusTarget(
                        expr='trading_calibration_rmse',
                        legend_format="{{model}} {{symbol}}"
                    )
                ],
                unit="percent"
            ),
            Panel(
                title="Model Parameters",
                panel_type=PanelType.TIME_SERIES,
                grid_pos={"x": 12, "y": 12, "w": 12, "h": 6},
                targets=[
                    PrometheusTarget(
                        expr='trading_model_parameter',
                        legend_format="{{model}} {{parameter}}"
                    )
                ]
            )
        ]
    )


class DashboardProvisioner:
    """
    Provisions dashboards to Grafana.

    Handles both file-based provisioning and API-based deployment.
    """

    def __init__(
        self,
        output_dir: str = "/etc/grafana/provisioning/dashboards",
        grafana_url: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """
        Initialize provisioner.

        Args:
            output_dir: Directory for file-based provisioning
            grafana_url: Grafana API URL for API-based deployment
            api_key: Grafana API key
        """
        self.output_dir = output_dir
        self.grafana_url = grafana_url
        self.api_key = api_key

    def provision_all(self) -> None:
        """Provision all standard dashboards."""
        dashboards = [
            create_trading_overview_dashboard(),
            create_risk_dashboard(),
            create_system_health_dashboard(),
            create_data_quality_dashboard()
        ]

        for dashboard in dashboards:
            self.provision(dashboard)

    def provision(self, dashboard: Dashboard) -> None:
        """
        Provision a dashboard.

        Args:
            dashboard: Dashboard to provision
        """
        import os

        # File-based provisioning
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            filepath = os.path.join(self.output_dir, f"{dashboard.uid}.json")
            dashboard.save(filepath)

        # API-based deployment
        if self.grafana_url and self.api_key:
            self._deploy_via_api(dashboard)

    def _deploy_via_api(self, dashboard: Dashboard) -> bool:
        """Deploy dashboard via Grafana API."""
        try:
            import requests

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "dashboard": dashboard.to_dict(),
                "overwrite": True,
                "message": f"Automated deployment at {datetime.utcnow().isoformat()}"
            }

            response = requests.post(
                f"{self.grafana_url}/api/dashboards/db",
                json=payload,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            logger.info(f"Deployed dashboard: {dashboard.title}")
            return True

        except Exception as e:
            logger.error(f"Failed to deploy dashboard {dashboard.title}: {e}")
            return False

    def get_all_dashboards(self) -> List[Dashboard]:
        """Get all standard dashboard configurations."""
        return [
            create_trading_overview_dashboard(),
            create_risk_dashboard(),
            create_system_health_dashboard(),
            create_data_quality_dashboard()
        ]
