"""Execution log writer and structured entries."""

from datetime import datetime
from typing import Any, Dict, List

from autoviz_agent.utils.logging import get_logger

logger = get_logger(__name__)


class ExecutionLogEntry:
    """Single execution log entry."""

    def __init__(
        self,
        sequence: int,
        tool: str,
        args: Dict[str, Any],
        result: Dict[str, Any],
        timestamp: datetime,
        duration_ms: float,
        status: str,
    ):
        """
        Initialize log entry.

        Args:
            sequence: Execution sequence number
            tool: Tool name
            args: Tool arguments
            result: Tool result
            timestamp: Execution timestamp
            duration_ms: Duration in milliseconds
            status: Status (success, error, warning)
        """
        self.sequence = sequence
        self.tool = tool
        self.args = args
        self.result = result
        self.timestamp = timestamp
        self.duration_ms = duration_ms
        self.status = status

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sequence": self.sequence,
            "tool": self.tool,
            "args": self.args,
            "result": self.result,
            "timestamp": self.timestamp.isoformat(),
            "duration_ms": self.duration_ms,
            "status": self.status,
        }


class ExecutionLog:
    """Execution log manager."""

    def __init__(self):
        """Initialize execution log."""
        self.entries: List[ExecutionLogEntry] = []

    def add_entry(
        self,
        sequence: int,
        tool: str,
        args: Dict[str, Any],
        result: Dict[str, Any],
        duration_ms: float,
        status: str = "success",
    ) -> None:
        """
        Add an execution log entry.

        Args:
            sequence: Execution sequence number
            tool: Tool name
            args: Tool arguments
            result: Tool result
            duration_ms: Duration in milliseconds
            status: Status (success, error, warning)
        """
        entry = ExecutionLogEntry(
            sequence=sequence,
            tool=tool,
            args=args,
            result=result,
            timestamp=datetime.utcnow(),
            duration_ms=duration_ms,
            status=status,
        )
        self.entries.append(entry)
        logger.debug(f"Added log entry: seq={sequence}, tool={tool}, status={status}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert log to dictionary."""
        return {
            "entries": [entry.to_dict() for entry in self.entries],
            "total_entries": len(self.entries),
            "total_duration_ms": sum(e.duration_ms for e in self.entries),
        }

    def get_summary(self) -> Dict[str, Any]:
        """
        Get execution summary.

        Returns:
            Summary statistics
        """
        success_count = sum(1 for e in self.entries if e.status == "success")
        error_count = sum(1 for e in self.entries if e.status == "error")
        warning_count = sum(1 for e in self.entries if e.status == "warning")

        return {
            "total_tools_executed": len(self.entries),
            "success_count": success_count,
            "error_count": error_count,
            "warning_count": warning_count,
            "total_duration_ms": sum(e.duration_ms for e in self.entries),
        }

    def add_validation_error(
        self,
        tool_name: str,
        error_type: str,
        error_message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a validation error entry.

        Args:
            tool_name: Tool name
            error_type: Error type
            error_message: Error message
            details: Additional error details
        """
        entry = ExecutionLogEntry(
            sequence=-1,  # Validation errors don't have sequence
            tool=tool_name,
            args={},
            result={
                "validation_error": True,
                "error_type": error_type,
                "error_message": error_message,
                "details": details or {},
            },
            timestamp=datetime.utcnow(),
            duration_ms=0.0,
            status="error",
        )
        self.entries.append(entry)
        logger.error(f"Validation error: {tool_name} - {error_message}")

    def add_repair_attempt(
        self, tool_name: str, strategy: str, success: bool, details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a repair attempt entry.

        Args:
            tool_name: Tool name
            strategy: Repair strategy
            success: Whether repair succeeded
            details: Additional details
        """
        entry = ExecutionLogEntry(
            sequence=-1,
            tool=tool_name,
            args={},
            result={
                "repair_attempt": True,
                "strategy": strategy,
                "success": success,
                "details": details or {},
            },
            timestamp=datetime.utcnow(),
            duration_ms=0.0,
            status="success" if success else "error",
        )
        self.entries.append(entry)
        logger.info(f"Repair attempt: {tool_name} - {strategy} - {'succeeded' if success else 'failed'}")
