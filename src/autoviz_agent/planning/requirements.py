"""Requirement label helpers for planning."""

from typing import List

from autoviz_agent.llm.llm_contracts import RequirementExtractionOutput


def build_required_labels(requirements: RequirementExtractionOutput) -> List[str]:
    """Build normalized requirement labels used for capability matching."""
    labels: List[str] = []
    if requirements.analysis:
        labels.extend([f"analysis.{label}" for label in requirements.analysis])
    if requirements.outputs:
        labels.extend([f"output.{label}" for label in requirements.outputs])
    if requirements.group_by:
        labels.append("group_by")
    if requirements.time and requirements.time.column:
        labels.append("time")
    return labels
