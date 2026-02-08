"""Plan diff generation with rationale and coverage validation."""

from typing import Any, Dict, List, Set, Tuple

from autoviz_agent.planning.schema_tags import (
    REQUIREMENT_TO_CAPABILITY_MAP,
    get_required_capabilities,
)
from autoviz_agent.registry.tools import TOOL_REGISTRY, ensure_default_tools_registered
from autoviz_agent.llm.llm_contracts import RequirementExtractionOutput
from autoviz_agent.utils.logging import get_logger

logger = get_logger(__name__)


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


# =============================================================================
# Coverage Validation
# =============================================================================

def validate_plan_coverage(
    plan: Dict[str, Any],
    requirements: RequirementExtractionOutput,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate that plan covers all requirements.
    
    Args:
        plan: Adapted plan with steps
        requirements: Extracted requirements
    
    Returns:
        Tuple of (is_valid, validation_report)
    """
    ensure_default_tools_registered()
    
    # Get tool capabilities from registry
    tool_capabilities = {}
    schemas = TOOL_REGISTRY.get_all_schemas()
    for name, schema in schemas.items():
        tool_capabilities[name] = set(schema.capabilities)
    
    # Extract capabilities from plan steps
    plan_capabilities = set()
    step_to_requirements = {}
    unjustified_steps = []
    
    for step in plan.get("steps", []):
        tool_name = step.get("tool")
        step_id = step.get("step_id")
        
        # Track what requirements this step claims to satisfy
        satisfies = step.get("satisfies", [])
        if satisfies:
            step_to_requirements[step_id] = satisfies
        else:
            # Step doesn't declare what it satisfies - potential unjustified step
            unjustified_steps.append(step_id)
        
        # Add tool capabilities to plan
        if tool_name and tool_name in tool_capabilities:
            plan_capabilities.update(tool_capabilities[tool_name])
    
    # Check coverage for each requirement type
    missing_coverage = {}
    
    # Check analysis requirements
    for analysis_type in requirements.analysis:
        try:
            required_caps = get_required_capabilities(analysis_type)
            if not any(cap in plan_capabilities for cap in required_caps):
                missing_coverage[f"analysis.{analysis_type}"] = required_caps
        except ValueError as e:
            logger.warning(f"Unknown analysis type: {analysis_type}")
    
    # Check output requirements
    for output_type in requirements.outputs:
        try:
            required_caps = get_required_capabilities(output_type)
            if not any(cap in plan_capabilities for cap in required_caps):
                missing_coverage[f"output.{output_type}"] = required_caps
        except ValueError:
            pass
    
    # Check group_by requirement
    if requirements.group_by:
        required_caps = get_required_capabilities("group_by")
        if not any(cap in plan_capabilities for cap in required_caps):
            missing_coverage["group_by"] = required_caps
    
    # Check time requirement
    if requirements.time and requirements.time.column:
        required_caps = get_required_capabilities("time")
        if not any(cap in plan_capabilities for cap in required_caps):
            missing_coverage["time"] = required_caps
    
    # Build validation report
    is_valid = len(missing_coverage) == 0 and len(unjustified_steps) == 0
    
    report = {
        "valid": is_valid,
        "missing_coverage": missing_coverage,
        "unjustified_steps": unjustified_steps,
        "plan_capabilities": list(plan_capabilities),
        "step_to_requirements": step_to_requirements,
    }
    
    if not is_valid:
        logger.warning(f"Plan coverage validation failed: "
                      f"missing={list(missing_coverage.keys())}, "
                      f"unjustified={unjustified_steps}")
    else:
        logger.info("Plan coverage validation passed")
    
    return is_valid, report


def generate_coverage_error_payload(
    validation_report: Dict[str, Any]
) -> str:
    """
    Generate error payload for failed coverage validation.
    
    Args:
        validation_report: Report from validate_plan_coverage
    
    Returns:
        Error message with actionable feedback
    """
    lines = ["Coverage validation failed:\n"]
    
    missing = validation_report.get("missing_coverage", {})
    if missing:
        lines.append("\nMissing coverage for requirements:")
        for req_label, required_caps in missing.items():
            lines.append(f"  - {req_label}: needs capabilities {required_caps}")
    
    unjustified = validation_report.get("unjustified_steps", [])
    if unjustified:
        lines.append("\nUnjustified steps (no requirement mapping):")
        for step_id in unjustified:
            lines.append(f"  - {step_id}")
        lines.append("\nPlease add 'satisfies' field to each step or remove unjustified steps.")
    
    return "\n".join(lines)
