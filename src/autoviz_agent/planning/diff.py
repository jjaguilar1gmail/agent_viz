"""Plan diff generation with rationale."""

from typing import Any, Dict, List


def generate_diff(
    template_plan: Dict[str, Any], adapted_plan: Dict[str, Any], rationale: str = ""
) -> str:
    """
    Generate a markdown diff between template and adapted plan.

    Args:
        template_plan: Original template plan
        adapted_plan: Adapted plan
        rationale: Optional rationale for changes

    Returns:
        Markdown diff string
    """
    lines = ["# Plan Diff\n"]

    if rationale:
        lines.append(f"## Rationale\n\n{rationale}\n")

    lines.append("## Changes\n")

    # Compare steps
    template_steps = {step["step_id"]: step for step in template_plan.get("steps", [])}
    adapted_steps = {step["step_id"]: step for step in adapted_plan.get("steps", [])}

    # Added steps
    added = set(adapted_steps.keys()) - set(template_steps.keys())
    if added:
        lines.append("\n### Added Steps\n")
        for step_id in sorted(added):
            step = adapted_steps[step_id]
            lines.append(f"- **{step_id}**: {step.get('tool', 'unknown')} - {step.get('description', '')}\n")

    # Removed steps
    removed = set(template_steps.keys()) - set(adapted_steps.keys())
    if removed:
        lines.append("\n### Removed Steps\n")
        for step_id in sorted(removed):
            step = template_steps[step_id]
            lines.append(f"- **{step_id}**: {step.get('tool', 'unknown')} - {step.get('description', '')}\n")

    # Modified steps
    common = set(template_steps.keys()) & set(adapted_steps.keys())
    modified = []
    for step_id in sorted(common):
        if template_steps[step_id] != adapted_steps[step_id]:
            modified.append(step_id)

    if modified:
        lines.append("\n### Modified Steps\n")
        for step_id in modified:
            lines.append(f"\n#### {step_id}\n")
            lines.append("**Before:**\n")
            lines.append(f"```json\n{_format_step(template_steps[step_id])}\n```\n")
            lines.append("**After:**\n")
            lines.append(f"```json\n{_format_step(adapted_steps[step_id])}\n```\n")

    if not added and not removed and not modified:
        lines.append("\nNo changes between template and adapted plan.\n")

    return "".join(lines)


def _format_step(step: Dict[str, Any]) -> str:
    """Format a step for display."""
    import json

    return json.dumps(step, indent=2)


def extract_diff_summary(
    template_plan: Dict[str, Any], adapted_plan: Dict[str, Any]
) -> Dict[str, List[str]]:
    """
    Extract a structured summary of differences.

    Args:
        template_plan: Original template plan
        adapted_plan: Adapted plan

    Returns:
        Dictionary with added_steps, removed_steps, modified_steps lists
    """
    template_steps = {step["step_id"]: step for step in template_plan.get("steps", [])}
    adapted_steps = {step["step_id"]: step for step in adapted_plan.get("steps", [])}

    added = sorted(set(adapted_steps.keys()) - set(template_steps.keys()))
    removed = sorted(set(template_steps.keys()) - set(adapted_steps.keys()))

    common = set(template_steps.keys()) & set(adapted_steps.keys())
    modified = sorted([s for s in common if template_steps[s] != adapted_steps[s]])

    return {"added_steps": added, "removed_steps": removed, "modified_steps": modified}
