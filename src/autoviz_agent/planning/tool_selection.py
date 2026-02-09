"""Deterministic tool selection based on capability coverage."""

from typing import Dict, List, Set, Tuple

from autoviz_agent.planning.schema_tags import get_required_capabilities, normalize_capability
from autoviz_agent.registry.tools import TOOL_REGISTRY, ensure_default_tools_registered


def _tool_coverage(required_labels: List[str]) -> Dict[str, Set[str]]:
    ensure_default_tools_registered()
    coverage: Dict[str, Set[str]] = {}
    schemas = TOOL_REGISTRY.get_all_schemas()

    for name, schema in schemas.items():
        tool_caps = {normalize_capability(c) for c in schema.capabilities}
        covered: Set[str] = set()
        for label in required_labels:
            requirement_key = label.split(".", 1)[-1]
            try:
                required_caps = get_required_capabilities(requirement_key)
            except ValueError:
                continue
            normalized_required = {normalize_capability(c) for c in required_caps}
            if tool_caps & normalized_required:
                covered.add(label)
        coverage[name] = covered

    return coverage


def select_tools_by_capabilities(
    required_labels: List[str],
    candidate_tools: List[str],
) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Select a minimal tool set that covers required labels.

    Returns:
        (selected_tools, tool_to_labels)
    """
    coverage = _tool_coverage(required_labels)
    remaining = set(required_labels)
    selected: List[str] = []
    tool_to_labels: Dict[str, List[str]] = {}

    candidates = [t for t in candidate_tools if t in coverage]
    while remaining:
        best_tool = None
        best_cover: Set[str] = set()
        for tool in candidates:
            if tool in selected:
                continue
            covered = coverage.get(tool, set()) & remaining
            if len(covered) > len(best_cover):
                best_tool = tool
                best_cover = covered
        if not best_tool:
            break
        selected.append(best_tool)
        tool_to_labels[best_tool] = sorted(best_cover)
        remaining -= best_cover

    # If we failed to cover everything, fall back to candidate list order
    if remaining:
        for tool in candidates:
            if tool not in selected:
                selected.append(tool)
                tool_to_labels.setdefault(tool, sorted(coverage.get(tool, set())))

    return selected, tool_to_labels
