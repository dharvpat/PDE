"""
Operational Runbooks for Quantitative Trading System.

This module provides automated runbook procedures for common operational scenarios:
- Alert response procedures
- System recovery procedures
- Performance troubleshooting
- Incident escalation
- Health check procedures
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class RunbookCategory(Enum):
    """Categories of runbooks."""
    ALERT_RESPONSE = "alert_response"
    SYSTEM_RECOVERY = "system_recovery"
    PERFORMANCE = "performance"
    DATA_QUALITY = "data_quality"
    MODEL_ISSUES = "model_issues"
    TRADING_OPERATIONS = "trading_operations"
    INCIDENT_MANAGEMENT = "incident_management"


class StepStatus(Enum):
    """Status of a runbook step."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class RunbookStep:
    """Individual step in a runbook."""

    name: str
    description: str
    action: Optional[Callable[[], Tuple[bool, str]]] = None
    manual_instructions: str = ""
    requires_approval: bool = False
    timeout_seconds: int = 300
    rollback_action: Optional[Callable[[], bool]] = None

    status: StepStatus = field(default=StepStatus.PENDING)
    result_message: str = ""
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def execute(self) -> Tuple[bool, str]:
        """Execute the step."""
        self.started_at = datetime.now()
        self.status = StepStatus.IN_PROGRESS

        if self.action is None:
            # Manual step
            self.status = StepStatus.COMPLETED
            self.completed_at = datetime.now()
            return True, f"Manual step: {self.manual_instructions}"

        try:
            success, message = self.action()
            self.result_message = message
            self.status = StepStatus.COMPLETED if success else StepStatus.FAILED
            self.completed_at = datetime.now()
            return success, message
        except Exception as e:
            self.status = StepStatus.FAILED
            self.result_message = str(e)
            self.completed_at = datetime.now()
            return False, str(e)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "manual_instructions": self.manual_instructions,
            "requires_approval": self.requires_approval,
            "status": self.status.value,
            "result_message": self.result_message,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


@dataclass
class Runbook:
    """Complete runbook definition."""

    name: str
    description: str
    category: RunbookCategory
    severity: str  # "low", "medium", "high", "critical"
    steps: List[RunbookStep]
    tags: List[str] = field(default_factory=list)
    owner: str = ""
    last_updated: datetime = field(default_factory=datetime.now)

    # Execution state
    current_step: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def execute_next_step(self) -> Optional[Tuple[bool, str]]:
        """Execute the next pending step."""
        if self.current_step >= len(self.steps):
            return None

        if self.started_at is None:
            self.started_at = datetime.now()

        step = self.steps[self.current_step]
        success, message = step.execute()

        if success:
            self.current_step += 1

        if self.current_step >= len(self.steps):
            self.completed_at = datetime.now()

        return success, message

    def execute_all(self, stop_on_failure: bool = True) -> List[Tuple[str, bool, str]]:
        """Execute all steps."""
        results = []
        self.started_at = datetime.now()

        for step in self.steps:
            success, message = step.execute()
            results.append((step.name, success, message))
            logger.info(f"Runbook step '{step.name}': {'SUCCESS' if success else 'FAILED'} - {message}")

            if not success and stop_on_failure:
                break

        self.completed_at = datetime.now()
        return results

    def get_progress(self) -> Dict[str, Any]:
        """Get execution progress."""
        completed = sum(1 for s in self.steps if s.status == StepStatus.COMPLETED)
        failed = sum(1 for s in self.steps if s.status == StepStatus.FAILED)
        pending = sum(1 for s in self.steps if s.status == StepStatus.PENDING)

        return {
            "total_steps": len(self.steps),
            "completed": completed,
            "failed": failed,
            "pending": pending,
            "progress_percent": (completed / len(self.steps) * 100) if self.steps else 0,
            "current_step": self.current_step,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "severity": self.severity,
            "tags": self.tags,
            "owner": self.owner,
            "steps": [s.to_dict() for s in self.steps],
            "progress": self.get_progress(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class RunbookRegistry:
    """Registry for all available runbooks."""

    def __init__(self):
        self._runbooks: Dict[str, Runbook] = {}
        self._execution_history: List[Dict[str, Any]] = []

    def register(self, runbook: Runbook) -> None:
        """Register a runbook."""
        self._runbooks[runbook.name] = runbook

    def get(self, name: str) -> Optional[Runbook]:
        """Get a runbook by name."""
        return self._runbooks.get(name)

    def list_runbooks(
        self,
        category: Optional[RunbookCategory] = None,
        tag: Optional[str] = None,
    ) -> List[Runbook]:
        """List runbooks with optional filtering."""
        runbooks = list(self._runbooks.values())

        if category:
            runbooks = [r for r in runbooks if r.category == category]
        if tag:
            runbooks = [r for r in runbooks if tag in r.tags]

        return runbooks

    def execute_runbook(
        self,
        name: str,
        stop_on_failure: bool = True,
    ) -> Optional[List[Tuple[str, bool, str]]]:
        """Execute a runbook by name."""
        runbook = self.get(name)
        if runbook is None:
            logger.error(f"Runbook '{name}' not found")
            return None

        logger.info(f"Starting runbook: {name}")
        results = runbook.execute_all(stop_on_failure)

        self._execution_history.append({
            "runbook_name": name,
            "started_at": runbook.started_at.isoformat() if runbook.started_at else None,
            "completed_at": runbook.completed_at.isoformat() if runbook.completed_at else None,
            "results": results,
            "success": all(r[1] for r in results),
        })

        return results

    def get_execution_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent execution history."""
        return self._execution_history[-limit:]


# =============================================================================
# Pre-built Runbooks
# =============================================================================

def create_high_drawdown_runbook() -> Runbook:
    """Create runbook for high drawdown alert response."""
    return Runbook(
        name="high_drawdown_response",
        description="Response procedure when portfolio drawdown exceeds threshold",
        category=RunbookCategory.ALERT_RESPONSE,
        severity="critical",
        tags=["risk", "drawdown", "emergency"],
        owner="risk_team",
        steps=[
            RunbookStep(
                name="assess_situation",
                description="Assess current market conditions and portfolio state",
                manual_instructions="""
                1. Check current drawdown level and duration
                2. Review market conditions (VIX, market direction)
                3. Identify positions contributing to drawdown
                4. Check if stop-loss levels are being approached
                """,
            ),
            RunbookStep(
                name="notify_stakeholders",
                description="Send notifications to relevant stakeholders",
                manual_instructions="""
                1. Alert risk manager via Slack/PagerDuty
                2. Notify portfolio manager
                3. CC compliance if drawdown > 15%
                """,
            ),
            RunbookStep(
                name="reduce_exposure",
                description="Consider reducing portfolio exposure",
                manual_instructions="""
                1. Review current position sizes
                2. Identify positions to reduce
                3. Execute partial position closes if needed
                4. Update risk limits if appropriate
                """,
                requires_approval=True,
            ),
            RunbookStep(
                name="document_actions",
                description="Document all actions taken",
                manual_instructions="""
                1. Record drawdown level and timestamp
                2. Document positions affected
                3. Record actions taken and rationale
                4. Update risk report
                """,
            ),
        ],
    )


def create_data_staleness_runbook() -> Runbook:
    """Create runbook for stale data alert response."""
    return Runbook(
        name="data_staleness_response",
        description="Response procedure when market data becomes stale",
        category=RunbookCategory.DATA_QUALITY,
        severity="high",
        tags=["data", "market_data", "staleness"],
        owner="data_team",
        steps=[
            RunbookStep(
                name="verify_staleness",
                description="Confirm data staleness and identify source",
                manual_instructions="""
                1. Check data feed timestamps
                2. Verify which symbols are affected
                3. Check feed provider status page
                4. Review network connectivity
                """,
            ),
            RunbookStep(
                name="pause_trading",
                description="Consider pausing automated trading",
                manual_instructions="""
                1. If staleness > 5 minutes, pause signal generation
                2. If staleness > 15 minutes, pause order execution
                3. Set trading mode to manual-only if needed
                """,
                requires_approval=True,
            ),
            RunbookStep(
                name="attempt_recovery",
                description="Attempt to recover data feed",
                manual_instructions="""
                1. Restart data feed connection
                2. If primary fails, switch to backup feed
                3. Verify data freshness after reconnection
                4. Validate data integrity
                """,
            ),
            RunbookStep(
                name="resume_operations",
                description="Resume normal operations if data recovered",
                manual_instructions="""
                1. Verify data is current (< 30 seconds old)
                2. Re-enable signal generation
                3. Re-enable order execution
                4. Monitor for recurrence
                """,
            ),
        ],
    )


def create_model_calibration_failure_runbook() -> Runbook:
    """Create runbook for model calibration failure."""
    return Runbook(
        name="calibration_failure_response",
        description="Response procedure when model calibration fails or produces poor results",
        category=RunbookCategory.MODEL_ISSUES,
        severity="medium",
        tags=["model", "calibration", "heston", "sabr"],
        owner="quant_team",
        steps=[
            RunbookStep(
                name="identify_failure",
                description="Identify the nature of calibration failure",
                manual_instructions="""
                1. Check calibration logs for errors
                2. Review RMSE and R-squared metrics
                3. Check if Feller condition is violated (Heston)
                4. Verify input data quality (option prices)
                """,
            ),
            RunbookStep(
                name="use_cached_params",
                description="Fall back to cached parameters",
                manual_instructions="""
                1. Load previous day's calibrated parameters
                2. Verify parameters are within reasonable bounds
                3. Apply cached parameters to model
                4. Note: Performance may be degraded
                """,
            ),
            RunbookStep(
                name="adjust_constraints",
                description="Try calibration with relaxed constraints",
                manual_instructions="""
                1. Widen parameter bounds slightly
                2. Increase optimization iterations
                3. Try alternative starting points
                4. Re-run calibration
                """,
            ),
            RunbookStep(
                name="escalate_if_needed",
                description="Escalate if calibration continues to fail",
                manual_instructions="""
                1. If failure persists > 2 days, escalate to lead quant
                2. Consider model review/update
                3. Document regime change if identified
                """,
            ),
        ],
    )


def create_system_high_cpu_runbook() -> Runbook:
    """Create runbook for high CPU usage."""
    return Runbook(
        name="high_cpu_response",
        description="Response procedure for high CPU utilization",
        category=RunbookCategory.SYSTEM_RECOVERY,
        severity="medium",
        tags=["system", "cpu", "performance"],
        owner="ops_team",
        steps=[
            RunbookStep(
                name="identify_process",
                description="Identify process causing high CPU",
                manual_instructions="""
                1. Run `top` or `htop` to identify high CPU processes
                2. Check if calibration or signal generation is running
                3. Review recent deployments or configuration changes
                """,
            ),
            RunbookStep(
                name="assess_impact",
                description="Assess impact on trading operations",
                manual_instructions="""
                1. Check signal generation latency
                2. Verify order execution times
                3. Check for any failed operations
                """,
            ),
            RunbookStep(
                name="take_action",
                description="Take corrective action",
                manual_instructions="""
                1. If non-critical process, consider killing it
                2. If critical process stuck, restart service
                3. Scale up resources if needed (cloud)
                4. Consider load balancing if persistent
                """,
                requires_approval=True,
            ),
            RunbookStep(
                name="post_incident",
                description="Post-incident actions",
                manual_instructions="""
                1. Document root cause
                2. Review CPU thresholds
                3. Consider optimization if recurring
                4. Update monitoring if needed
                """,
            ),
        ],
    )


def create_order_rejection_runbook() -> Runbook:
    """Create runbook for high order rejection rate."""
    return Runbook(
        name="order_rejection_response",
        description="Response procedure for elevated order rejection rate",
        category=RunbookCategory.TRADING_OPERATIONS,
        severity="high",
        tags=["execution", "orders", "rejection"],
        owner="trading_team",
        steps=[
            RunbookStep(
                name="analyze_rejections",
                description="Analyze rejection reasons",
                manual_instructions="""
                1. Review rejection codes from broker
                2. Identify patterns (symbol, time, order type)
                3. Check position limits and buying power
                4. Verify market hours and trading halts
                """,
            ),
            RunbookStep(
                name="check_connectivity",
                description="Verify broker connectivity",
                manual_instructions="""
                1. Test broker API connectivity
                2. Check for broker system status
                3. Verify credentials are valid
                4. Test with small market order
                """,
            ),
            RunbookStep(
                name="adjust_orders",
                description="Adjust order parameters if needed",
                manual_instructions="""
                1. Review order sizes vs lot requirements
                2. Check price limits vs current market
                3. Verify symbol mapping
                4. Adjust time-in-force if needed
                """,
            ),
            RunbookStep(
                name="resume_trading",
                description="Verify and resume normal trading",
                manual_instructions="""
                1. Execute test order successfully
                2. Monitor rejection rate for 15 minutes
                3. Re-enable full automated trading
                4. Alert if rejections continue
                """,
            ),
        ],
    )


def create_cointegration_breakdown_runbook() -> Runbook:
    """Create runbook for pairs trading cointegration breakdown."""
    return Runbook(
        name="cointegration_breakdown_response",
        description="Response procedure when pairs cointegration breaks down",
        category=RunbookCategory.MODEL_ISSUES,
        severity="medium",
        tags=["pairs", "cointegration", "mean_reversion"],
        owner="quant_team",
        steps=[
            RunbookStep(
                name="verify_breakdown",
                description="Confirm cointegration breakdown",
                manual_instructions="""
                1. Run ADF test on spread residuals
                2. Check half-life of mean reversion
                3. Compare current spread to historical range
                4. Verify with multiple testing windows
                """,
            ),
            RunbookStep(
                name="assess_positions",
                description="Review current pair positions",
                manual_instructions="""
                1. List all positions in affected pairs
                2. Calculate current P&L on each pair
                3. Estimate loss if spread continues diverging
                4. Review position sizes
                """,
            ),
            RunbookStep(
                name="reduce_exposure",
                description="Consider reducing pair exposure",
                manual_instructions="""
                1. Reduce position sizes by 50% if spread > 3 sigma
                2. Close positions if spread > 4 sigma
                3. Disable new entries for this pair
                4. Set tighter stop-losses
                """,
                requires_approval=True,
            ),
            RunbookStep(
                name="investigate_cause",
                description="Investigate fundamental cause",
                manual_instructions="""
                1. Check for corporate actions (M&A, spin-offs)
                2. Review sector/industry changes
                3. Check for regulatory changes
                4. Document findings for model review
                """,
            ),
        ],
    )


def create_default_runbooks() -> List[Runbook]:
    """Create all default runbooks."""
    return [
        create_high_drawdown_runbook(),
        create_data_staleness_runbook(),
        create_model_calibration_failure_runbook(),
        create_system_high_cpu_runbook(),
        create_order_rejection_runbook(),
        create_cointegration_breakdown_runbook(),
    ]


def get_default_registry() -> RunbookRegistry:
    """Get a registry with all default runbooks."""
    registry = RunbookRegistry()
    for runbook in create_default_runbooks():
        registry.register(runbook)
    return registry
