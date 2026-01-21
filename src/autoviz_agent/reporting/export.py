"""Export tools for tool_calls.json and execution_log.json."""

from pathlib import Path
from typing import Any, Dict, List

from autoviz_agent.io.artifacts import ArtifactManager
from autoviz_agent.reporting.execution_log import ExecutionLog
from autoviz_agent.utils.logging import get_logger

logger = get_logger(__name__)


def export_tool_calls(
    tool_calls: List[Dict[str, Any]], artifact_manager: ArtifactManager
) -> Path:
    """
    Export tool calls to JSON.

    Args:
        tool_calls: List of tool calls
        artifact_manager: Artifact manager

    Returns:
        Path to saved file
    """
    path = artifact_manager.save_json(
        {"tool_calls": tool_calls, "count": len(tool_calls)}, "tool_calls", "tool_calls.json"
    )
    logger.info(f"Exported {len(tool_calls)} tool calls")
    return path


def export_execution_log(execution_log: ExecutionLog, artifact_manager: ArtifactManager) -> Path:
    """
    Export execution log to JSON.

    Args:
        execution_log: Execution log
        artifact_manager: Artifact manager

    Returns:
        Path to saved file
    """
    log_data = execution_log.to_dict()
    path = artifact_manager.save_json(log_data, "execution_log", "execution_log.json")
    logger.info(f"Exported execution log with {len(execution_log.entries)} entries")
    return path


def export_run_metadata(
    run_id: str,
    dataset_path: str,
    question: str,
    status: str,
    artifacts: List[Dict[str, Any]],
    artifact_manager: ArtifactManager,
) -> Path:
    """
    Export run metadata.

    Args:
        run_id: Run identifier
        dataset_path: Path to dataset
        question: User question
        status: Run status
        artifacts: List of artifacts
        artifact_manager: Artifact manager

    Returns:
        Path to saved file
    """
    metadata = {
        "run_id": run_id,
        "dataset_path": dataset_path,
        "question": question,
        "status": status,
        "artifacts": artifacts,
    }

    path = artifact_manager.save_json(metadata, "metadata", "run_metadata.json")
    logger.info("Exported run metadata")
    return path
