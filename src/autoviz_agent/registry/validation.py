"""Tool call schema validation and unknown tool rejection."""

from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError

from autoviz_agent.registry.schemas import ToolCallRequest
from autoviz_agent.registry.tools import TOOL_REGISTRY, ToolParameter
from autoviz_agent.utils.logging import get_logger

logger = get_logger(__name__)


class ValidationResult:
    """Result of tool call validation."""

    def __init__(
        self,
        is_valid: bool,
        tool_name: str,
        errors: Optional[List[str]] = None,
        warnings: Optional[List[str]] = None,
    ):
        """
        Initialize validation result.

        Args:
            is_valid: Whether validation passed
            tool_name: Tool name
            errors: List of error messages
            warnings: List of warning messages
        """
        self.is_valid = is_valid
        self.tool_name = tool_name
        self.errors = errors or []
        self.warnings = warnings or []

    def __repr__(self) -> str:
        status = "VALID" if self.is_valid else "INVALID"
        return f"ValidationResult({status}, tool={self.tool_name}, errors={len(self.errors)})"


def validate_tool_call(tool_call: Dict[str, Any]) -> ValidationResult:
    """
    Validate a tool call against schema.

    Args:
        tool_call: Tool call specification

    Returns:
        ValidationResult
    """
    tool_name = tool_call.get("tool", "unknown")
    errors = []
    warnings = []

    # Check if tool exists
    if not TOOL_REGISTRY.get_tool(tool_name):
        errors.append(f"Unknown tool: {tool_name}")
        return ValidationResult(is_valid=False, tool_name=tool_name, errors=errors)

    # Validate using Pydantic model
    try:
        ToolCallRequest(**tool_call)
    except ValidationError as e:
        for error in e.errors():
            field = ".".join(str(loc) for loc in error["loc"])
            errors.append(f"{field}: {error['msg']}")

    # Get tool schema for parameter validation
    schema = TOOL_REGISTRY.get_schema(tool_name)
    if schema:
        param_errors, param_warnings = _validate_parameters(tool_call.get("args", {}), schema.parameters)
        errors.extend(param_errors)
        warnings.extend(param_warnings)

    is_valid = len(errors) == 0

    result = ValidationResult(
        is_valid=is_valid, tool_name=tool_name, errors=errors, warnings=warnings
    )

    if not is_valid:
        logger.warning(f"Tool call validation failed: {result}")

    return result


def _validate_parameters(
    args: Dict[str, Any], parameters: List[ToolParameter]
) -> Tuple[List[str], List[str]]:
    """
    Validate tool call arguments against parameter schema.

    Args:
        args: Tool call arguments
        parameters: Parameter schema

    Returns:
        Tuple of (errors, warnings)
    """
    errors = []
    warnings = []

    # Build parameter map
    param_map = {p.name: p for p in parameters}

    # Check required parameters
    for param in parameters:
        if param.required and param.name not in args:
            errors.append(f"Missing required parameter: {param.name}")

    # Check for unknown parameters (strict validation)
    for arg_name in args.keys():
        if arg_name not in param_map:
            errors.append(f"Unknown parameter: {arg_name}")

    return errors, warnings


def validate_tool_call_batch(tool_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate a batch of tool calls.

    Args:
        tool_calls: List of tool calls

    Returns:
        Validation summary
    """
    results = [validate_tool_call(tc) for tc in tool_calls]

    valid_count = sum(1 for r in results if r.is_valid)
    invalid_count = len(results) - valid_count

    summary = {
        "total": len(results),
        "valid": valid_count,
        "invalid": invalid_count,
        "results": results,
    }

    logger.info(f"Batch validation: {valid_count}/{len(results)} valid")

    return summary


def reject_invalid_tool_calls(tool_calls: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Separate valid and invalid tool calls.

    Args:
        tool_calls: List of tool calls

    Returns:
        Tuple of (valid_calls, invalid_calls)
    """
    valid = []
    invalid = []

    for tc in tool_calls:
        result = validate_tool_call(tc)
        if result.is_valid:
            valid.append(tc)
        else:
            invalid.append({"tool_call": tc, "errors": result.errors})

    logger.info(f"Rejected {len(invalid)} invalid tool calls")

    return valid, invalid
