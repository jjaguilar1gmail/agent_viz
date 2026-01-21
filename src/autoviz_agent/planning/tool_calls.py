"""Tool call compiler."""

from typing import Any, Dict, List

from autoviz_agent.utils.logging import get_logger

logger = get_logger(__name__)


def compile_tool_calls(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Compile tool calls from a plan.

    Args:
        plan: Execution plan with steps

    Returns:
        List of tool call specifications
    """
    tool_calls = []

    for idx, step in enumerate(plan.get("steps", [])):
        tool_call = {
            "sequence": idx,
            "step_id": step.get("step_id"),
            "tool": step.get("tool"),
            "args": step.get("params", {}),
            "description": step.get("description", ""),
        }
        tool_calls.append(tool_call)

    logger.info(f"Compiled {len(tool_calls)} tool calls from plan")
    return tool_calls


def validate_tool_call_sequence(tool_calls: List[Dict[str, Any]]) -> bool:
    """
    Validate that tool calls have proper sequence numbers.

    Args:
        tool_calls: List of tool calls

    Returns:
        True if valid, False otherwise
    """
    if not tool_calls:
        return True

    sequences = [tc["sequence"] for tc in tool_calls]

    # Check for duplicates
    if len(sequences) != len(set(sequences)):
        logger.error("Duplicate sequence numbers found in tool calls")
        return False

    # Check for gaps
    expected = list(range(len(tool_calls)))
    if sorted(sequences) != expected:
        logger.error("Tool call sequences are not contiguous")
        return False

    return True


def reorder_tool_calls(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Reorder tool calls by sequence number.

    Args:
        tool_calls: List of tool calls

    Returns:
        Sorted list of tool calls
    """
    return sorted(tool_calls, key=lambda x: x["sequence"])
