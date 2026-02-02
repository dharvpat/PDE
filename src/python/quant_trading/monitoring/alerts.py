"""
Alerting System for Trading Platform.

This module provides intelligent alerting with:
- Multiple severity levels (INFO, WARNING, CRITICAL)
- Alert rule engine with customizable conditions
- Multiple notification channels (email, Slack, PagerDuty, SMS)
- Alert suppression and deduplication
- Escalation policies
- Acknowledgment workflow
- Alert history and analytics

Designed for production trading systems requiring reliable alert delivery.
"""

import json
import logging
import smtplib
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

import requests

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class AlertStatus(Enum):
    """Alert lifecycle status."""
    FIRING = "firing"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


class AlertCategory(Enum):
    """Alert categories for routing and filtering."""
    RISK = "risk"
    EXECUTION = "execution"
    DATA = "data"
    MODEL = "model"
    SYSTEM = "system"
    COMPLIANCE = "compliance"


@dataclass
class Alert:
    """
    Alert object representing a triggered alert.

    Attributes:
        alert_id: Unique alert identifier
        title: Alert title/name
        description: Detailed description
        severity: Alert severity level
        category: Alert category
        component: Component that triggered alert
        metric_name: Name of the metric that triggered alert
        metric_value: Current metric value
        threshold_value: Threshold that was breached
        timestamp: Alert creation time
        status: Alert status
        acknowledged_by: User who acknowledged
        acknowledged_at: Acknowledgment time
        resolved_at: Resolution time
        metadata: Additional alert data
        labels: Alert labels for routing
    """
    alert_id: str
    title: str
    description: str
    severity: AlertSeverity
    category: AlertCategory
    component: str
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    threshold_value: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    status: AlertStatus = AlertStatus.FIRING
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)
    notification_count: int = 0
    last_notification: Optional[datetime] = None

    def acknowledge(self, user: str) -> None:
        """Acknowledge the alert."""
        self.status = AlertStatus.ACKNOWLEDGED
        self.acknowledged_by = user
        self.acknowledged_at = datetime.utcnow()

    def resolve(self) -> None:
        """Mark alert as resolved."""
        self.status = AlertStatus.RESOLVED
        self.resolved_at = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        data = asdict(self)
        data['severity'] = self.severity.value
        data['category'] = self.category.value
        data['status'] = self.status.value
        data['timestamp'] = self.timestamp.isoformat()
        if self.acknowledged_at:
            data['acknowledged_at'] = self.acknowledged_at.isoformat()
        if self.resolved_at:
            data['resolved_at'] = self.resolved_at.isoformat()
        if self.last_notification:
            data['last_notification'] = self.last_notification.isoformat()
        return data

    @property
    def age_minutes(self) -> float:
        """Get alert age in minutes."""
        return (datetime.utcnow() - self.timestamp).total_seconds() / 60


@dataclass
class AlertRule:
    """
    Alert rule definition.

    Defines conditions under which alerts should be triggered.

    Example:
        rule = AlertRule(
            name="High VaR",
            condition=lambda metrics: metrics.get('var_95', 0) > 20000,
            severity=AlertSeverity.WARNING,
            category=AlertCategory.RISK,
            cooldown_minutes=60
        )
    """
    name: str
    condition: Callable[[Dict[str, Any]], bool]
    severity: AlertSeverity
    category: AlertCategory
    description: str = ""
    component: str = "system"
    cooldown_minutes: int = 60
    notification_channels: List[str] = field(default_factory=lambda: ["log"])
    metric_name: Optional[str] = None
    threshold_value: Optional[float] = None
    labels: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True

    # Runtime state
    last_alert_time: Optional[datetime] = field(default=None, repr=False)
    is_active: bool = field(default=False, repr=False)
    fire_count: int = field(default=0, repr=False)

    def evaluate(self, metrics: Dict[str, Any]) -> Optional[Alert]:
        """
        Evaluate rule against current metrics.

        Args:
            metrics: Current system metrics

        Returns:
            Alert if condition met and not in cooldown, else None
        """
        if not self.enabled:
            return None

        # Check condition
        try:
            should_alert = self.condition(metrics)
        except Exception as e:
            logger.error(f"Error evaluating rule {self.name}: {e}")
            return None

        if not should_alert:
            self.is_active = False
            return None

        # Check cooldown
        if self.last_alert_time:
            time_since_last = datetime.utcnow() - self.last_alert_time
            if time_since_last < timedelta(minutes=self.cooldown_minutes):
                return None

        # Extract metric value for the alert
        metric_value = None
        if self.metric_name and self.metric_name in metrics:
            metric_value = metrics[self.metric_name]

        # Create alert
        alert = Alert(
            alert_id=str(uuid.uuid4()),
            title=self.name,
            description=self.description,
            severity=self.severity,
            category=self.category,
            component=self.component,
            metric_name=self.metric_name,
            metric_value=metric_value,
            threshold_value=self.threshold_value,
            labels=self.labels.copy(),
            metadata={'metrics_snapshot': {k: v for k, v in metrics.items()
                                           if isinstance(v, (int, float, str, bool))}}
        )

        self.last_alert_time = datetime.utcnow()
        self.is_active = True
        self.fire_count += 1

        return alert


# =============================================================================
# Notification Channels
# =============================================================================

class NotificationChannel(ABC):
    """Abstract base class for notification channels."""

    @abstractmethod
    def send(self, alert: Alert) -> bool:
        """
        Send alert notification.

        Args:
            alert: Alert to send

        Returns:
            True if successful
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Channel name."""
        pass


class LogChannel(NotificationChannel):
    """Log-based notification channel."""

    @property
    def name(self) -> str:
        return "log"

    def send(self, alert: Alert) -> bool:
        """Log the alert."""
        log_level = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.CRITICAL: logging.ERROR
        }.get(alert.severity, logging.WARNING)

        logger.log(
            log_level,
            f"ALERT [{alert.severity.value}] [{alert.category.value}] "
            f"{alert.title}: {alert.description} "
            f"(metric={alert.metric_value}, threshold={alert.threshold_value})"
        )
        return True


class SlackChannel(NotificationChannel):
    """Slack notification channel."""

    def __init__(
        self,
        webhook_url: str,
        channel: Optional[str] = None,
        username: str = "Trading Alerts"
    ):
        """
        Initialize Slack channel.

        Args:
            webhook_url: Slack webhook URL
            channel: Optional channel override
            username: Bot username
        """
        self.webhook_url = webhook_url
        self.channel = channel
        self.username = username

    @property
    def name(self) -> str:
        return "slack"

    def send(self, alert: Alert) -> bool:
        """Send alert to Slack."""
        try:
            color = {
                AlertSeverity.INFO: "#36a64f",  # Green
                AlertSeverity.WARNING: "#ff9800",  # Orange
                AlertSeverity.CRITICAL: "#ff0000"  # Red
            }.get(alert.severity, "#808080")

            payload = {
                "username": self.username,
                "attachments": [{
                    "color": color,
                    "title": f"[{alert.severity.value}] {alert.title}",
                    "text": alert.description,
                    "fields": [
                        {"title": "Category", "value": alert.category.value, "short": True},
                        {"title": "Component", "value": alert.component, "short": True},
                    ],
                    "footer": f"Alert ID: {alert.alert_id}",
                    "ts": int(alert.timestamp.timestamp())
                }]
            }

            if alert.metric_value is not None:
                payload["attachments"][0]["fields"].append({
                    "title": "Metric Value",
                    "value": f"{alert.metric_value}",
                    "short": True
                })

            if alert.threshold_value is not None:
                payload["attachments"][0]["fields"].append({
                    "title": "Threshold",
                    "value": f"{alert.threshold_value}",
                    "short": True
                })

            if self.channel:
                payload["channel"] = self.channel

            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            return True

        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False


class EmailChannel(NotificationChannel):
    """Email notification channel."""

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        username: str,
        password: str,
        from_addr: str,
        to_addrs: List[str],
        use_tls: bool = True
    ):
        """
        Initialize email channel.

        Args:
            smtp_host: SMTP server host
            smtp_port: SMTP server port
            username: SMTP username
            password: SMTP password
            from_addr: Sender email address
            to_addrs: Recipient email addresses
            use_tls: Whether to use TLS
        """
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_addr = from_addr
        self.to_addrs = to_addrs
        self.use_tls = use_tls

    @property
    def name(self) -> str:
        return "email"

    def send(self, alert: Alert) -> bool:
        """Send alert via email."""
        try:
            subject = f"[{alert.severity.value}] Trading Alert: {alert.title}"

            body = f"""
Trading System Alert

Title: {alert.title}
Severity: {alert.severity.value}
Category: {alert.category.value}
Component: {alert.component}

Description:
{alert.description}

Details:
- Metric: {alert.metric_name or 'N/A'}
- Current Value: {alert.metric_value}
- Threshold: {alert.threshold_value}
- Time: {alert.timestamp.isoformat()}
- Alert ID: {alert.alert_id}

---
This is an automated alert from the Trading System.
"""

            msg = MIMEMultipart()
            msg['From'] = self.from_addr
            msg['To'] = ', '.join(self.to_addrs)
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))

            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                server.login(self.username, self.password)
                server.sendmail(self.from_addr, self.to_addrs, msg.as_string())

            return True

        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False


class PagerDutyChannel(NotificationChannel):
    """PagerDuty notification channel."""

    def __init__(
        self,
        routing_key: str,
        service_name: str = "Trading System"
    ):
        """
        Initialize PagerDuty channel.

        Args:
            routing_key: PagerDuty Events API routing key
            service_name: Service name for context
        """
        self.routing_key = routing_key
        self.service_name = service_name
        self.api_url = "https://events.pagerduty.com/v2/enqueue"

    @property
    def name(self) -> str:
        return "pagerduty"

    def send(self, alert: Alert) -> bool:
        """Send alert to PagerDuty."""
        try:
            severity_map = {
                AlertSeverity.INFO: "info",
                AlertSeverity.WARNING: "warning",
                AlertSeverity.CRITICAL: "critical"
            }

            payload = {
                "routing_key": self.routing_key,
                "event_action": "trigger",
                "dedup_key": f"{alert.title}-{alert.component}",
                "payload": {
                    "summary": f"{alert.title}: {alert.description}",
                    "severity": severity_map.get(alert.severity, "warning"),
                    "source": self.service_name,
                    "component": alert.component,
                    "group": alert.category.value,
                    "class": alert.category.value,
                    "custom_details": {
                        "metric_name": alert.metric_name,
                        "metric_value": alert.metric_value,
                        "threshold_value": alert.threshold_value,
                        "alert_id": alert.alert_id
                    }
                }
            }

            response = requests.post(
                self.api_url,
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            return True

        except Exception as e:
            logger.error(f"Failed to send PagerDuty alert: {e}")
            return False


class WebhookChannel(NotificationChannel):
    """Generic webhook notification channel."""

    def __init__(
        self,
        webhook_url: str,
        headers: Optional[Dict[str, str]] = None,
        channel_name: str = "webhook"
    ):
        """
        Initialize webhook channel.

        Args:
            webhook_url: Webhook URL
            headers: Optional HTTP headers
            channel_name: Channel name identifier
        """
        self.webhook_url = webhook_url
        self.headers = headers or {"Content-Type": "application/json"}
        self._channel_name = channel_name

    @property
    def name(self) -> str:
        return self._channel_name

    def send(self, alert: Alert) -> bool:
        """Send alert to webhook."""
        try:
            response = requests.post(
                self.webhook_url,
                json=alert.to_dict(),
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
            return False


# =============================================================================
# Alert Manager
# =============================================================================

@dataclass
class EscalationPolicy:
    """
    Escalation policy for alerts.

    Defines how alerts should be escalated over time.
    """
    name: str
    escalation_minutes: List[int]  # Minutes after alert to escalate
    escalation_channels: List[List[str]]  # Channels for each escalation level
    severity_filter: Optional[List[AlertSeverity]] = None

    def get_channels_for_age(self, age_minutes: float) -> List[str]:
        """Get notification channels based on alert age."""
        channels = []
        for i, minutes in enumerate(self.escalation_minutes):
            if age_minutes >= minutes and i < len(self.escalation_channels):
                channels.extend(self.escalation_channels[i])
        return list(set(channels))


class AlertManager:
    """
    Central manager for alerts and notifications.

    Coordinates alert rules, notification channels, and escalation policies.

    Example:
        alert_mgr = AlertManager()

        # Add notification channel
        alert_mgr.register_channel(SlackChannel(webhook_url="..."))

        # Add alert rule
        alert_mgr.add_rule(AlertRule(
            name="Daily Loss Limit",
            condition=lambda m: m.get('daily_pnl', 0) < -50000,
            severity=AlertSeverity.CRITICAL,
            category=AlertCategory.RISK,
            notification_channels=["slack", "pagerduty"]
        ))

        # Evaluate rules periodically
        alert_mgr.evaluate_rules(current_metrics)
    """

    def __init__(
        self,
        dedup_window_minutes: int = 60,
        max_alerts_per_rule: int = 100
    ):
        """
        Initialize alert manager.

        Args:
            dedup_window_minutes: Window for alert deduplication
            max_alerts_per_rule: Maximum alerts to keep per rule in history
        """
        self.dedup_window_minutes = dedup_window_minutes
        self.max_alerts_per_rule = max_alerts_per_rule

        self.rules: List[AlertRule] = []
        self.channels: Dict[str, NotificationChannel] = {}
        self.escalation_policies: List[EscalationPolicy] = []

        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self._suppressed_rules: Set[str] = set()

        # Register default log channel
        self.register_channel(LogChannel())

        # Add default rules
        self._add_default_rules()

    def _add_default_rules(self) -> None:
        """Add default trading system alert rules."""
        # Risk alerts
        self.add_rule(AlertRule(
            name="Daily Loss Limit Breached",
            condition=lambda m: m.get('daily_pnl', 0) < -50000,
            severity=AlertSeverity.CRITICAL,
            category=AlertCategory.RISK,
            description="Daily P&L has dropped below -$50,000 limit",
            component="risk_manager",
            metric_name="daily_pnl",
            threshold_value=-50000,
            notification_channels=["log", "slack", "pagerduty"]
        ))

        self.add_rule(AlertRule(
            name="Maximum Drawdown Warning",
            condition=lambda m: m.get('max_drawdown_pct', 0) > 15,
            severity=AlertSeverity.WARNING,
            category=AlertCategory.RISK,
            description="Portfolio drawdown exceeds 15% from peak",
            component="risk_manager",
            metric_name="max_drawdown_pct",
            threshold_value=15,
            notification_channels=["log", "slack"]
        ))

        self.add_rule(AlertRule(
            name="High VaR Alert",
            condition=lambda m: m.get('var_95', 0) > 25000,
            severity=AlertSeverity.WARNING,
            category=AlertCategory.RISK,
            description="95% VaR exceeds $25,000 threshold",
            component="risk_manager",
            metric_name="var_95",
            threshold_value=25000,
            notification_channels=["log", "slack"]
        ))

        # Execution alerts
        self.add_rule(AlertRule(
            name="High Order Rejection Rate",
            condition=lambda m: m.get('order_rejection_rate', 0) > 0.1,
            severity=AlertSeverity.WARNING,
            category=AlertCategory.EXECUTION,
            description="More than 10% of orders are being rejected",
            component="execution_engine",
            metric_name="order_rejection_rate",
            threshold_value=0.1,
            cooldown_minutes=30
        ))

        self.add_rule(AlertRule(
            name="Order Fill Latency High",
            condition=lambda m: m.get('avg_fill_latency_ms', 0) > 1000,
            severity=AlertSeverity.WARNING,
            category=AlertCategory.EXECUTION,
            description="Average order fill latency exceeds 1 second",
            component="execution_engine",
            metric_name="avg_fill_latency_ms",
            threshold_value=1000
        ))

        # Data alerts
        self.add_rule(AlertRule(
            name="Stale Market Data",
            condition=lambda m: m.get('data_age_seconds', 0) > 60,
            severity=AlertSeverity.WARNING,
            category=AlertCategory.DATA,
            description="Market data not updated in 60 seconds",
            component="data_pipeline",
            metric_name="data_age_seconds",
            threshold_value=60,
            cooldown_minutes=5
        ))

        self.add_rule(AlertRule(
            name="Data Validation Failures",
            condition=lambda m: m.get('validation_failure_rate', 0) > 0.05,
            severity=AlertSeverity.WARNING,
            category=AlertCategory.DATA,
            description="More than 5% of data failing validation",
            component="data_pipeline",
            metric_name="validation_failure_rate",
            threshold_value=0.05
        ))

        # Model alerts
        self.add_rule(AlertRule(
            name="Model Calibration Degraded",
            condition=lambda m: m.get('calibration_rmse', 0) > 0.05,
            severity=AlertSeverity.WARNING,
            category=AlertCategory.MODEL,
            description="Model calibration RMSE exceeds 5%",
            component="model_calibrator",
            metric_name="calibration_rmse",
            threshold_value=0.05
        ))

        # System alerts
        self.add_rule(AlertRule(
            name="High Memory Usage",
            condition=lambda m: m.get('memory_usage_pct', 0) > 85,
            severity=AlertSeverity.WARNING,
            category=AlertCategory.SYSTEM,
            description="Memory usage exceeds 85%",
            component="system",
            metric_name="memory_usage_pct",
            threshold_value=85
        ))

        self.add_rule(AlertRule(
            name="High CPU Usage",
            condition=lambda m: m.get('cpu_usage_pct', 0) > 90,
            severity=AlertSeverity.WARNING,
            category=AlertCategory.SYSTEM,
            description="CPU usage exceeds 90%",
            component="system",
            metric_name="cpu_usage_pct",
            threshold_value=90,
            cooldown_minutes=5
        ))

    def register_channel(self, channel: NotificationChannel) -> None:
        """Register a notification channel."""
        self.channels[channel.name] = channel
        logger.info(f"Registered notification channel: {channel.name}")

    def add_rule(self, rule: AlertRule) -> None:
        """Add an alert rule."""
        self.rules.append(rule)
        logger.debug(f"Added alert rule: {rule.name}")

    def remove_rule(self, rule_name: str) -> bool:
        """Remove an alert rule by name."""
        for i, rule in enumerate(self.rules):
            if rule.name == rule_name:
                del self.rules[i]
                logger.info(f"Removed alert rule: {rule_name}")
                return True
        return False

    def add_escalation_policy(self, policy: EscalationPolicy) -> None:
        """Add an escalation policy."""
        self.escalation_policies.append(policy)
        logger.info(f"Added escalation policy: {policy.name}")

    def suppress_rule(self, rule_name: str, duration_minutes: int = 60) -> None:
        """
        Temporarily suppress a rule.

        Args:
            rule_name: Rule name to suppress
            duration_minutes: How long to suppress
        """
        self._suppressed_rules.add(rule_name)
        logger.info(f"Suppressed rule: {rule_name} for {duration_minutes} minutes")

        # TODO: Add timer to unsuppress

    def unsuppress_rule(self, rule_name: str) -> None:
        """Unsuppress a rule."""
        self._suppressed_rules.discard(rule_name)
        logger.info(f"Unsuppressed rule: {rule_name}")

    def evaluate_rules(self, metrics: Dict[str, Any]) -> List[Alert]:
        """
        Evaluate all rules against current metrics.

        Args:
            metrics: Current system metrics

        Returns:
            List of triggered alerts
        """
        triggered_alerts = []

        for rule in self.rules:
            if rule.name in self._suppressed_rules:
                continue

            alert = rule.evaluate(metrics)
            if alert:
                self._handle_alert(alert, rule.notification_channels)
                triggered_alerts.append(alert)

        # Check escalations for active alerts
        self._check_escalations()

        return triggered_alerts

    def _handle_alert(self, alert: Alert, channels: List[str]) -> None:
        """
        Handle a triggered alert.

        Args:
            alert: Alert to handle
            channels: Notification channels to use
        """
        # Store alert
        self.active_alerts[alert.alert_id] = alert
        self.alert_history.append(alert)

        # Trim history if needed
        if len(self.alert_history) > self.max_alerts_per_rule * len(self.rules):
            self.alert_history = self.alert_history[-self.max_alerts_per_rule * len(self.rules):]

        # Send notifications
        for channel_name in channels:
            channel = self.channels.get(channel_name)
            if channel:
                try:
                    if channel.send(alert):
                        alert.notification_count += 1
                        alert.last_notification = datetime.utcnow()
                except Exception as e:
                    logger.error(f"Failed to send alert via {channel_name}: {e}")
            else:
                logger.warning(f"No handler registered for channel: {channel_name}")

    def _check_escalations(self) -> None:
        """Check and process escalations for active alerts."""
        for alert in self.active_alerts.values():
            if alert.status != AlertStatus.FIRING:
                continue

            for policy in self.escalation_policies:
                # Check severity filter
                if policy.severity_filter and alert.severity not in policy.severity_filter:
                    continue

                # Get channels for current alert age
                channels = policy.get_channels_for_age(alert.age_minutes)
                for channel_name in channels:
                    channel = self.channels.get(channel_name)
                    if channel:
                        # Avoid duplicate notifications
                        if alert.last_notification:
                            minutes_since_last = (
                                datetime.utcnow() - alert.last_notification
                            ).total_seconds() / 60
                            if minutes_since_last < 5:  # 5 minute minimum between notifications
                                continue

                        try:
                            if channel.send(alert):
                                alert.notification_count += 1
                                alert.last_notification = datetime.utcnow()
                        except Exception as e:
                            logger.error(f"Escalation notification failed: {e}")

    def acknowledge_alert(self, alert_id: str, user: str) -> bool:
        """
        Acknowledge an alert.

        Args:
            alert_id: Alert ID
            user: User acknowledging

        Returns:
            True if successful
        """
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].acknowledge(user)
            logger.info(f"Alert {alert_id} acknowledged by {user}")
            return True
        return False

    def resolve_alert(self, alert_id: str) -> bool:
        """
        Resolve an alert.

        Args:
            alert_id: Alert ID

        Returns:
            True if successful
        """
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].resolve()
            del self.active_alerts[alert_id]
            logger.info(f"Alert {alert_id} resolved")
            return True
        return False

    def get_active_alerts(
        self,
        severity: Optional[AlertSeverity] = None,
        category: Optional[AlertCategory] = None
    ) -> List[Alert]:
        """
        Get active alerts with optional filtering.

        Args:
            severity: Filter by severity
            category: Filter by category

        Returns:
            List of matching alerts
        """
        alerts = list(self.active_alerts.values())

        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        if category:
            alerts = [a for a in alerts if a.category == category]

        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)

    def get_alert_history(
        self,
        hours: int = 24,
        severity: Optional[AlertSeverity] = None
    ) -> List[Alert]:
        """
        Get alert history.

        Args:
            hours: Hours of history to retrieve
            severity: Filter by severity

        Returns:
            List of historical alerts
        """
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        alerts = [a for a in self.alert_history if a.timestamp >= cutoff]

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)

    def get_alert_stats(self) -> Dict[str, Any]:
        """Get alert statistics."""
        now = datetime.utcnow()
        last_24h = now - timedelta(hours=24)

        recent_alerts = [a for a in self.alert_history if a.timestamp >= last_24h]

        by_severity = defaultdict(int)
        by_category = defaultdict(int)
        for alert in recent_alerts:
            by_severity[alert.severity.value] += 1
            by_category[alert.category.value] += 1

        return {
            'active_count': len(self.active_alerts),
            'active_by_severity': {
                s.value: len([a for a in self.active_alerts.values() if a.severity == s])
                for s in AlertSeverity
            },
            'last_24h_count': len(recent_alerts),
            'last_24h_by_severity': dict(by_severity),
            'last_24h_by_category': dict(by_category),
            'rules_count': len(self.rules),
            'suppressed_rules': list(self._suppressed_rules)
        }

    def get_rules_status(self) -> List[Dict[str, Any]]:
        """Get status of all rules."""
        return [
            {
                'name': rule.name,
                'severity': rule.severity.value,
                'category': rule.category.value,
                'enabled': rule.enabled,
                'suppressed': rule.name in self._suppressed_rules,
                'is_active': rule.is_active,
                'fire_count': rule.fire_count,
                'last_alert_time': rule.last_alert_time.isoformat() if rule.last_alert_time else None
            }
            for rule in self.rules
        ]


def create_default_alert_rules() -> List[AlertRule]:
    """
    Create default alert rules for trading systems.

    Returns:
        List of preconfigured alert rules covering common trading scenarios.
    """
    rules = [
        # Risk alerts
        AlertRule(
            name="high_drawdown",
            condition=lambda m: m.get("drawdown", 0) > 0.20,
            severity=AlertSeverity.CRITICAL,
            category=AlertCategory.RISK,
            description="Portfolio drawdown exceeds 20%",
            component="risk_manager",
            metric_name="drawdown",
            threshold_value=0.20,
            cooldown_minutes=60,
        ),
        AlertRule(
            name="warning_drawdown",
            condition=lambda m: 0.15 < m.get("drawdown", 0) <= 0.20,
            severity=AlertSeverity.WARNING,
            category=AlertCategory.RISK,
            description="Portfolio drawdown exceeds 15%",
            component="risk_manager",
            metric_name="drawdown",
            threshold_value=0.15,
            cooldown_minutes=30,
        ),
        AlertRule(
            name="var_breach",
            condition=lambda m: abs(m.get("var_95", 0)) > m.get("var_limit", float('inf')),
            severity=AlertSeverity.CRITICAL,
            category=AlertCategory.RISK,
            description="VaR exceeds limit",
            component="risk_manager",
            metric_name="var_95",
            cooldown_minutes=120,
        ),
        AlertRule(
            name="position_limit_breach",
            condition=lambda m: m.get("position_value", 0) > m.get("position_limit", float('inf')),
            severity=AlertSeverity.WARNING,
            category=AlertCategory.RISK,
            description="Position size exceeds limit",
            component="risk_manager",
            metric_name="position_value",
            cooldown_minutes=30,
        ),

        # Execution alerts
        AlertRule(
            name="high_slippage",
            condition=lambda m: m.get("slippage_bps", 0) > 50,
            severity=AlertSeverity.WARNING,
            category=AlertCategory.EXECUTION,
            description="Order slippage exceeds 50 basis points",
            component="execution_engine",
            metric_name="slippage_bps",
            threshold_value=50,
            cooldown_minutes=15,
        ),
        AlertRule(
            name="order_rejection_rate",
            condition=lambda m: m.get("rejection_rate", 0) > 0.10,
            severity=AlertSeverity.WARNING,
            category=AlertCategory.EXECUTION,
            description="Order rejection rate exceeds 10%",
            component="execution_engine",
            metric_name="rejection_rate",
            threshold_value=0.10,
            cooldown_minutes=30,
        ),

        # Data alerts
        AlertRule(
            name="data_staleness",
            condition=lambda m: m.get("data_age_seconds", 0) > 300,
            severity=AlertSeverity.WARNING,
            category=AlertCategory.DATA,
            description="Market data is stale (>5 minutes old)",
            component="data_feed",
            metric_name="data_age_seconds",
            threshold_value=300,
            cooldown_minutes=10,
        ),
        AlertRule(
            name="data_gap",
            condition=lambda m: m.get("data_gaps", 0) > 0,
            severity=AlertSeverity.WARNING,
            category=AlertCategory.DATA,
            description="Data gaps detected in feed",
            component="data_feed",
            metric_name="data_gaps",
            cooldown_minutes=15,
        ),

        # Model alerts
        AlertRule(
            name="calibration_error",
            condition=lambda m: m.get("calibration_rmse", 0) > 0.05,
            severity=AlertSeverity.WARNING,
            category=AlertCategory.MODEL,
            description="Model calibration RMSE exceeds 5%",
            component="model_calibrator",
            metric_name="calibration_rmse",
            threshold_value=0.05,
            cooldown_minutes=60,
        ),
        AlertRule(
            name="signal_degradation",
            condition=lambda m: m.get("signal_ic", 1) < 0.02,
            severity=AlertSeverity.WARNING,
            category=AlertCategory.MODEL,
            description="Signal information coefficient below 2%",
            component="signal_generator",
            metric_name="signal_ic",
            threshold_value=0.02,
            cooldown_minutes=120,
        ),

        # System alerts
        AlertRule(
            name="high_cpu",
            condition=lambda m: m.get("cpu_percent", 0) > 90,
            severity=AlertSeverity.WARNING,
            category=AlertCategory.SYSTEM,
            description="CPU usage exceeds 90%",
            component="system",
            metric_name="cpu_percent",
            threshold_value=90,
            cooldown_minutes=15,
        ),
        AlertRule(
            name="high_memory",
            condition=lambda m: m.get("memory_percent", 0) > 85,
            severity=AlertSeverity.WARNING,
            category=AlertCategory.SYSTEM,
            description="Memory usage exceeds 85%",
            component="system",
            metric_name="memory_percent",
            threshold_value=85,
            cooldown_minutes=15,
        ),
        AlertRule(
            name="low_disk",
            condition=lambda m: m.get("disk_free_percent", 100) < 10,
            severity=AlertSeverity.CRITICAL,
            category=AlertCategory.SYSTEM,
            description="Disk space below 10%",
            component="system",
            metric_name="disk_free_percent",
            threshold_value=10,
            cooldown_minutes=60,
        ),
    ]

    return rules
