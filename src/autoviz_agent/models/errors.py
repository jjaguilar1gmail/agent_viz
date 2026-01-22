"""Error models and logging."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ValidationError(BaseModel):
    """Validation error model."""

    error_type: str = Field(..., description="Type of error (e.g., unknown_tool, invalid_param)")
    message: str = Field(..., description="Error message")
    tool_name: Optional[str] = Field(None, description="Tool name if applicable")
    parameter: Optional[str] = Field(None, description="Parameter name if applicable")
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional error details")


class RepairAttempt(BaseModel):
    """Repair attempt model."""

    attempt_id: str = Field(..., description="Unique attempt identifier")
    error: ValidationError = Field(..., description="Original error")
    repair_strategy: str = Field(..., description="Strategy used for repair")
    success: bool = Field(..., description="Whether repair succeeded")
    repaired_value: Optional[Any] = Field(None, description="Repaired value if successful")
    error_message: Optional[str] = Field(None, description="Error message if repair failed")


class ExecutionError(BaseModel):
    """Execution error model."""

    error_type: str = Field(..., description="Type of error")
    message: str = Field(..., description="Error message")
    tool_name: str = Field(..., description="Tool that failed")
    sequence: int = Field(..., description="Sequence number")
    stack_trace: Optional[str] = Field(None, description="Stack trace if available")
    recoverable: bool = Field(default=False, description="Whether error is recoverable")


def create_validation_error(
    error_type: str, message: str, tool_name: Optional[str] = None, **kwargs
) -> ValidationError:
    """
    Create a validation error.

    Args:
        error_type: Error type
        message: Error message
        tool_name: Tool name
        **kwargs: Additional details

    Returns:
        ValidationError instance
    """
    return ValidationError(
        error_type=error_type, message=message, tool_name=tool_name, details=kwargs
    )


def log_repair_attempt(
    attempt_id: str, error: ValidationError, strategy: str, success: bool, **kwargs
) -> RepairAttempt:
    """
    Log a repair attempt.

    Args:
        attempt_id: Attempt identifier
        error: Original error
        strategy: Repair strategy
        success: Whether repair succeeded
        **kwargs: Additional fields

    Returns:
        RepairAttempt instance
    """
    return RepairAttempt(
        attempt_id=attempt_id,
        error=error,
        repair_strategy=strategy,
        success=success,
        **kwargs,
    )


def format_error_for_log(error: ValidationError) -> Dict[str, Any]:
    """
    Format error for execution log.

    Args:
        error: Validation error

    Returns:
        Dictionary representation
    """
    return {
        "error_type": error.error_type,
        "message": error.message,
        "tool_name": error.tool_name,
        "parameter": error.parameter,
        "details": error.details,
    }
