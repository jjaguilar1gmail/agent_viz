"""Tool call schema validation and unknown tool rejection."""

import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError

from autoviz_agent.models.state import SchemaProfile
from autoviz_agent.registry.schemas import ToolCallRequest
from autoviz_agent.registry.tools import TOOL_REGISTRY, ToolParameter, ToolSchema
from autoviz_agent.utils.logging import get_logger

logger = get_logger(__name__)

_COLUMN_SIMILARITY_THRESHOLD = 0.7
_COLUMN_SIMILARITY_MARGIN = 0.05


def _normalize_column_name(name: str) -> str:
    normalized = name.strip().lower()
    normalized = normalized.replace("-", "_").replace(" ", "_")
    normalized = re.sub(r"[^a-z0-9_]", "", normalized)
    normalized = re.sub(r"_+", "_", normalized)
    return normalized


def _columns_for_role(schema_profile: SchemaProfile, role: str) -> List[str]:
    columns = [c.name for c in schema_profile.columns]
    if role == "any":
        return columns

    role_matches = [c.name for c in schema_profile.columns if role in c.roles]
    if role_matches:
        return role_matches

    numeric_dtypes = {"int64", "int32", "int", "float64", "float32", "float", "number"}
    categorical_dtypes = {"object", "string", "category", "bool", "boolean"}

    if role == "numeric":
        return [c.name for c in schema_profile.columns if c.dtype in numeric_dtypes] or columns
    if role == "categorical":
        return [c.name for c in schema_profile.columns if c.dtype in categorical_dtypes] or columns
    if role == "temporal":
        return [
            c.name
            for c in schema_profile.columns
            if "date" in c.dtype.lower() or "time" in c.dtype.lower()
        ] or columns

    return columns


def _best_column_match(value: str, candidates: List[str]) -> Optional[str]:
    if not candidates:
        return None

    normalized_candidates = {
        _normalize_column_name(candidate): candidate for candidate in candidates
    }
    normalized_value = _normalize_column_name(value)

    exact_match = normalized_candidates.get(normalized_value)
    if exact_match:
        return exact_match

    best = None
    best_score = 0.0
    second_best = 0.0

    for candidate in candidates:
        score = SequenceMatcher(
            None, normalized_value, _normalize_column_name(candidate)
        ).ratio()
        if score > best_score:
            second_best = best_score
            best_score = score
            best = candidate
        elif score > second_best:
            second_best = score

    if best_score >= _COLUMN_SIMILARITY_THRESHOLD and (best_score - second_best) >= _COLUMN_SIMILARITY_MARGIN:
        return best

    return None


def _repair_column_params(
    args: Dict[str, Any],
    schema_profile: SchemaProfile,
    tool_schema: ToolSchema,
) -> Dict[str, Any]:
    repaired = args.copy()

    for param in tool_schema.parameters:
        if not param.role or param.name not in repaired:
            continue

        role = param.role
        candidates = _columns_for_role(schema_profile, role)
        if not candidates:
            continue

        value = repaired.get(param.name)
        if value is None or value == "auto":
            continue

        if isinstance(value, list):
            new_values = []
            changed = False
            for item in value:
                if isinstance(item, str) and item not in candidates:
                    match = _best_column_match(item, candidates)
                    if match:
                        new_values.append(match)
                        changed = True
                        continue
                new_values.append(item)
            if changed:
                logger.info(
                    f"Repaired column list '{param.name}': {value} -> {new_values}"
                )
                repaired[param.name] = new_values
        elif isinstance(value, str) and value not in candidates:
            match = _best_column_match(value, candidates)
            if match:
                logger.info(
                    f"Repaired column '{param.name}': '{value}' -> '{match}'"
                )
                repaired[param.name] = match

    return repaired


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

    # Validate using Pydantic model - only validate the core fields
    try:
        # Create a minimal tool call for validation (exclude metadata fields)
        minimal_call = {
            "tool": tool_call.get("tool"),
            "sequence": tool_call.get("sequence", 1),
            "args": tool_call.get("args", {})
        }
        ToolCallRequest(**minimal_call)
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


def repair_tool_call(
    tool_call: Dict[str, Any],
    schema_profile: Optional[SchemaProfile] = None,
) -> Dict[str, Any]:
    """
    Attempt to repair an invalid tool call by filling missing required parameters.

    Args:
        tool_call: Tool call specification

    Returns:
        Repaired tool call (may still be invalid)
    """
    tool_name = tool_call.get("tool", "unknown")
    schema = TOOL_REGISTRY.get_schema(tool_name)
    
    if not schema:
        return tool_call
    
    repaired_call = tool_call.copy()
    args = repaired_call.get("args", {}).copy()

    allowed_params = {param.name for param in schema.parameters}
    invalid_params = [name for name in args.keys() if name not in allowed_params]
    if invalid_params:
        for name in invalid_params:
            args.pop(name, None)
        logger.info(f"Removed invalid params for {tool_name}: {invalid_params}")

    if schema_profile:
        args = _repair_column_params(args, schema_profile, schema)
    
    # Fill missing required parameters with schema defaults
    for param in schema.parameters:
        if param.required and param.name not in args:
            # Try to provide a sensible default
            if param.name == "df":
                args[param.name] = "$dataframe"
            elif param.type == "string":
                args[param.name] = "auto"
            elif param.type == "integer":
                args[param.name] = 0
            elif param.type == "boolean":
                args[param.name] = False
            elif param.type == "array":
                args[param.name] = []
            elif param.type == "object":
                args[param.name] = {}
            else:
                args[param.name] = None
            
            logger.info(f"Repaired {tool_name}: filled missing '{param.name}' with default")
    
    repaired_call["args"] = args
    return repaired_call

