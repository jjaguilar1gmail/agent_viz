"""Graph node implementations with actual LLM integration."""

import json
from pathlib import Path
from typing import Any, Dict

import yaml

from autoviz_agent.graph.state import GraphState
from autoviz_agent.io.artifacts import ArtifactManager
from autoviz_agent.llm.factory import create_llm_client
from autoviz_agent.models.state import Intent, RunStatus, SchemaProfile
from autoviz_agent.planning.diff import generate_diff
from autoviz_agent.planning.requirements import build_required_labels
from autoviz_agent.planning.tool_selection import select_tools_by_capabilities
from autoviz_agent.planning.retrieval import PlanRetrieval, get_tool_retriever
from autoviz_agent.planning.template_loader import TemplateLoader
from autoviz_agent.reporting.execution_log import ExecutionLog
from autoviz_agent.reporting.report_writer import ReportWriter
from autoviz_agent.runtime.executor import ToolExecutor
from autoviz_agent.tools.data_io import load_dataset
from autoviz_agent.tools.schema import infer_schema
from autoviz_agent.utils.logging import get_logger

logger = get_logger(__name__)


def initialize_node(state: GraphState) -> Dict[str, Any]:
    """
    Initialize the analysis run.

    Args:
        state: Current graph state

    Returns:
        Updated state
    """
    logger.info(f"Initializing run: {state.run_id}")
    state.current_node = "initialize"
    
    # Load configuration
    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
            state.config = config
    
    # Initialize components
    artifact_manager = ArtifactManager(state.run_id, Path("outputs"))
    state.artifact_manager = artifact_manager
    
    execution_log = ExecutionLog()
    state.execution_log = execution_log
    
    return {
        "current_node": "initialize",
        "config": state.config,
        "artifact_manager": artifact_manager,
        "execution_log": execution_log,
    }


def infer_schema_node(state: GraphState) -> Dict[str, Any]:
    """
    Load dataset and infer schema.

    Args:
        state: Current graph state

    Returns:
        Updated state with schema
    """
    logger.info("Loading dataset and inferring schema")
    state.current_node = "infer_schema"
    
    try:
        # Load dataset
        df = load_dataset(state.dataset_path)
        state.dataframe = df
        
        # Infer schema
        schema = infer_schema(df)
        state.schema = schema
        
        logger.info(f"Schema inferred: {len(schema.columns)} columns, {schema.row_count} rows")
        return {
            "current_node": "infer_schema",
            "dataframe": df,
            "schema": schema,
        }
    except Exception as e:
        logger.error(f"Schema inference failed: {e}")
        return {"current_node": "error", "error_message": str(e)}


def classify_intent_node(state: GraphState) -> Dict[str, Any]:
    """
    Classify user intent using LLM.

    Args:
        state: Current graph state

    Returns:
        Updated state with intent
    """
    logger.info("Classifying user intent")
    state.current_node = "classify_intent"
    
    try:
        # Initialize LLM client using factory
        model_config = state.config["models"][state.config["default_model"]]
        llm_client = create_llm_client(model_config)
        state.llm_client = llm_client
        
        # Classify intent
        intent = llm_client.classify_intent(
            state.question,
            state.schema,
            max_intents=state.config.get("intent", {}).get("max_intents", 3),
        )
        state.intent = intent
        
        # Track LLM interaction
        state.llm_interactions.append({
            "step": "intent_classification",
            "input": state.question,
            "output": {
                "intent": intent.label.value,
                "confidence": float(intent.confidence),
            },
            "node": "classify_intent"
        })
        state.llm_requests.append({
            "step": "intent_classification",
            "prompt": getattr(llm_client, "last_prompt", None),
            "response": getattr(llm_client, "last_response", None),
            "node": "classify_intent",
        })
        state.artifact_manager.save_json(
            state.llm_requests,
            "llm_requests",
            "llm_requests.json",
        )
        if state.llm_requests[-1].get("prompt"):
            state.artifact_manager.save_text(
                state.llm_requests[-1]["prompt"],
                "llm_requests",
                "llm_intent_prompt.txt",
            )
        if state.llm_requests[-1].get("response"):
            state.artifact_manager.save_text(
                state.llm_requests[-1]["response"],
                "llm_requests",
                "llm_intent_response.txt",
            )
        
        logger.info(f"Intent: {intent.label} (confidence={intent.confidence:.2f})")
        print(f"\n[Intent] Classified as '{intent.label.value}'")
        print(f"         Confidence: {intent.confidence:.0%}")
        return {
            "current_node": "classify_intent",
            "llm_client": llm_client,
            "intent": intent,
            "llm_interactions": state.llm_interactions,
            "llm_requests": state.llm_requests,
        }
    except Exception as e:
        logger.error(f"Intent classification failed: {e}")
        return {"current_node": "error", "error_message": str(e)}


def extract_requirements_node(state: GraphState) -> Dict[str, Any]:
    """
    Extract structured requirements from user question.

    Args:
        state: Current graph state

    Returns:
        Updated state with requirements
    """
    logger.info("Extracting requirements from user question")
    state.current_node = "extract_requirements"
    
    try:
        # Extract requirements using LLM
        requirements = state.llm_client.extract_requirements(
            state.question,
            state.schema,
        )
        state.requirements = requirements
        
        # Track LLM interaction
        state.llm_interactions.append({
            "step": "requirement_extraction",
            "input": state.question,
            "output": {
                "metrics": requirements.metrics,
                "group_by": requirements.group_by,
                "analysis": requirements.analysis,
            },
            "node": "extract_requirements"
        })
        state.llm_requests.append({
            "step": "requirement_extraction",
            "prompt": getattr(state.llm_client, "last_prompt", None),
            "response": getattr(state.llm_client, "last_response", None),
            "node": "extract_requirements",
        })
        
        # Save requirements
        state.artifact_manager.save_json(
            requirements.dict(),
            "requirements",
            "requirements.json",
        )
        
        logger.info(f"Requirements extracted: metrics={requirements.metrics}, "
                   f"analysis={requirements.analysis}")
        print(f"\n[Requirements] Extracted structured requirements")
        if requirements.metrics:
            print(f"               Metrics: {', '.join(requirements.metrics)}")
        if requirements.analysis:
            print(f"               Analysis: {', '.join(requirements.analysis)}")
        
        return {
            "current_node": "extract_requirements",
            "requirements": requirements,
            "llm_interactions": state.llm_interactions,
            "llm_requests": state.llm_requests,
        }
    except Exception as e:
        logger.warning(f"Requirement extraction failed: {e}. Continuing with empty requirements.")
        logger.debug(f"Exception type: {type(e).__name__}, traceback:", exc_info=True)
        # Don't fail the entire pipeline, just continue with empty requirements
        from autoviz_agent.llm.llm_contracts import RequirementExtractionOutput
        return {
            "current_node": "extract_requirements",
            "requirements": RequirementExtractionOutput(),
        }


def select_template_node(state: GraphState) -> Dict[str, Any]:
    """
    Select plan template based on intent and schema.

    Args:
        state: Current graph state

    Returns:
        Updated state with selected template
    """
    logger.info("Selecting plan template")
    state.current_node = "select_template"
    
    try:
        # Load templates
        templates_dir = Path(state.config.get("templates_dir", "templates"))
        template_loader = TemplateLoader(templates_dir)
        templates = template_loader.load_all()
        
        # Use retrieval to select best template
        retrieval = PlanRetrieval(templates)
        selected_template_id = retrieval.select_best(state.intent, state.schema)
        
        # Get the full template from the dict
        selected_template = templates[selected_template_id]
        state.template_plan = selected_template
        
        # Narrow tools using requirements
        template_tools = selected_template.get("curated_tools", [])
        if state.requirements:
            try:
                retriever = get_tool_retriever()
                narrowed_tools = retriever.retrieve_tools(
                    state.requirements,
                    template_tools,
                    top_k=5,
                    cap=12
                )
                state.narrowed_tools = narrowed_tools
                logger.info(f"Narrowed to {len(narrowed_tools)} tools")
            except Exception as e:
                logger.warning(f"Tool narrowing failed: {e}. Using template tools only.")
                state.narrowed_tools = template_tools
        else:
            # No requirements, use template tools
            state.narrowed_tools = template_tools
        
        # Save template to artifacts
        template_path = state.artifact_manager.save_json(
            selected_template, "plan_template", "plan_template.json"
        )
        
        logger.info(f"Selected template: {selected_template.get('template_id')}")
        print(f"\n[Template] I selected the '{selected_template.get('template_id')}' template")
        template_name = selected_template.get('name', selected_template.get('template_id'))
        print(f"   Template: {template_name}")
        print(f"   Reason: Best match for {state.intent.label.value} intent")
        return {
            "current_node": "select_template",
            "template_plan": selected_template,
            "narrowed_tools": state.narrowed_tools,
        }
    except Exception as e:
        logger.error(f"Template selection failed: {e}")
        return {"current_node": "error", "error_message": str(e)}


def derive_capabilities_node(state: GraphState) -> Dict[str, Any]:
    """
    Derive capability targets from requirements.
    """
    logger.info("Deriving capability targets")
    state.current_node = "derive_capabilities"

    if not state.requirements:
        state.capability_targets = []
    else:
        state.capability_targets = build_required_labels(state.requirements)

    if state.artifact_manager:
        state.artifact_manager.save_json(
            state.capability_targets,
            "capabilities",
            "capability_targets.json",
        )

    return {
        "current_node": "derive_capabilities",
        "capability_targets": state.capability_targets,
    }


def select_tools_node(state: GraphState) -> Dict[str, Any]:
    """
    Select tools to satisfy capability targets.
    """
    logger.info("Selecting tools by capability coverage")
    state.current_node = "select_tools"

    candidate_tools = state.narrowed_tools or []
    if not candidate_tools and state.template_plan:
        candidate_tools = state.template_plan.get("curated_tools", [])

    tool_to_labels = {}
    if not state.capability_targets:
        selected_tools = candidate_tools
    else:
        tool_catalog = state.llm_client.prompt_builder._build_tool_catalog(candidate_tools)
        selection = state.llm_client.select_tools(
            state.capability_targets,
            candidate_tools,
            tool_catalog,
        )
        selected_tools = selection.get("selected_tools", candidate_tools)
        # Guardrail: if selection misses coverage, add minimal tools to cover missing labels.
        selected_tools, tool_to_labels = select_tools_by_capabilities(
            state.capability_targets,
            selected_tools,
        )
        missing_labels = [
            label for label in state.capability_targets
            if not any(label in labels for labels in tool_to_labels.values())
        ]
        if missing_labels:
            expanded_tools, tool_to_labels = select_tools_by_capabilities(
                state.capability_targets,
                candidate_tools,
            )
            selected_tools = expanded_tools
        state.llm_requests.append({
            "step": "tool_selection",
            "prompt": getattr(state.llm_client, "last_prompt", None),
            "response": getattr(state.llm_client, "last_response", None),
            "node": "select_tools",
        })
        if state.llm_requests[-1].get("prompt"):
            state.artifact_manager.save_text(
                state.llm_requests[-1]["prompt"],
                "logs",
                "llm_tool_selection_prompt.txt",
            )
        if state.llm_requests[-1].get("response"):
            state.artifact_manager.save_text(
                state.llm_requests[-1]["response"],
                "logs",
                "llm_tool_selection_response.txt",
            )
        state.artifact_manager.save_json(
            state.llm_requests,
            "llm_requests",
            "llm_requests.json",
        )
        # tool_to_labels already computed above

    state.selected_tools = selected_tools
    state.llm_interactions.append({
        "step": "tool_selection",
        "input": {
            "capability_targets": state.capability_targets,
            "candidate_tools": candidate_tools,
        },
        "output": {
            "selected_tools": selected_tools,
        },
        "node": "select_tools",
    })

    if state.artifact_manager:
        state.artifact_manager.save_json(
            {
                "selected_tools": selected_tools,
                "tool_to_labels": tool_to_labels,
            },
            "tool_selection",
            "tool_selection.json",
        )

    logger.info(f"Selected {len(selected_tools)} tools for coverage")
    return {
        "current_node": "select_tools",
        "selected_tools": selected_tools,
    }


def build_plan_skeleton_node(state: GraphState) -> Dict[str, Any]:
    """
    Build a plan skeleton from selected tools and template steps.
    """
    logger.info("Building plan skeleton")
    state.current_node = "build_plan_skeleton"

    from autoviz_agent.registry.tools import TOOL_REGISTRY, ensure_default_tools_registered
    from autoviz_agent.planning.diff import validate_plan_coverage

    ensure_default_tools_registered()
    schemas = TOOL_REGISTRY.get_all_schemas()

    template_steps = state.template_plan.get("steps", []) if state.template_plan else []
    selected_tools = state.selected_tools or []

    steps = []
    used_step_ids = set()
    used_tools = set()

    # Keep template steps as the authoritative baseline.
    for step in template_steps:
        steps.append(step)
        used_step_ids.add(step.get("step_id"))
        tool_name = step.get("tool")
        if tool_name:
            used_tools.add(tool_name)

    for tool_name in selected_tools:
        if tool_name in used_tools:
            continue
        schema = schemas.get(tool_name)
        step_id = tool_name
        suffix = 1
        while step_id in used_step_ids:
            suffix += 1
            step_id = f"{tool_name}_{suffix}"
        used_step_ids.add(step_id)

        params = {}
        if schema:
            for param in schema.parameters:
                if param.name == "df":
                    params["df"] = "$dataframe"
                    break

        steps.append(
            {
                "step_id": step_id,
                "tool": tool_name,
                "description": (schema.description if schema else tool_name),
                "params": params,
            }
        )

    plan_skeleton = {
        **(state.template_plan or {}),
        "steps": steps,
    }

    if state.requirements:
        validate_plan_coverage(plan_skeleton, state.requirements)

    state.plan_skeleton = plan_skeleton

    if state.artifact_manager:
        state.artifact_manager.save_json(
            plan_skeleton,
            "plan_skeleton",
            "plan_skeleton.json",
        )

    return {
        "current_node": "build_plan_skeleton",
        "plan_skeleton": plan_skeleton,
    }


def fill_params_node(state: GraphState) -> Dict[str, Any]:
    """
    Fill parameters for plan steps using LLM, then validate coverage.
    """
    logger.info("Filling plan parameters")
    state.current_node = "fill_params"

    from autoviz_agent.planning.diff import (
        validate_plan_coverage,
        generate_coverage_error_payload,
        prune_unjustified_steps,
    )
    from autoviz_agent.registry.validation import (
        normalize_plan_param_types,
        repair_plan_step_columns,
        validate_plan_step_columns,
    )

    def apply_time_groupby_guardrails(plan: Dict[str, Any]) -> list[str]:
        fixes: list[str] = []
        requirements = getattr(state, "requirements", None)
        if not requirements or not requirements.group_by:
            return fixes
        time_info = getattr(requirements, "time", None)
        time_col = getattr(time_info, "column", "") if time_info else ""
        if not time_col:
            return fixes

        metrics = list(requirements.metrics or [])
        if not metrics:
            numeric_cols = [
                c.name for c in state.schema.columns
                if c.dtype in ["int64", "float64", "float", "int"]
            ]
            if numeric_cols:
                metrics = [numeric_cols[0]]
        if not metrics:
            return fixes

        group_by = [col for col in requirements.group_by if col and col != time_col]
        agg_group_by = [time_col] + group_by

        steps = plan.get("steps", [])
        aggregate_step = None
        for step in steps:
            if step.get("tool") == "aggregate":
                aggregate_step = step
                break

        if not aggregate_step:
            aggregate_step = {
                "step_id": "aggregate",
                "tool": "aggregate",
                "description": "Aggregate data by time and groups",
                "params": {"df": "$dataframe"},
                "satisfies": ["analysis.total", "analysis.compare", "output.table", "group_by", "time"],
            }
            steps.append(aggregate_step)
            fixes.append("added aggregate step for time+group_by")

        agg_params = aggregate_step.setdefault("params", {})
        agg_params["df"] = "$dataframe"
        agg_params["group_by"] = agg_group_by
        agg_params["agg_map"] = {metric: "sum" for metric in metrics}
        fixes.append("set aggregate group_by to include time + group_by")

        plot_step = next(
            (step for step in steps if step.get("tool") == "plot_line"),
            None,
        )
        if plot_step:
            plot_params = plot_step.setdefault("params", {})
            plot_params["df"] = "$aggregate_result"
            plot_params["x"] = time_col
            plot_params["y"] = metrics[0]
            plot_params["group_by"] = group_by
            fixes.append("bound plot_line to aggregate_result with time x-axis")

            agg_index = next(
                (idx for idx, step in enumerate(steps) if step is aggregate_step),
                None,
            )
            plot_index = next(
                (idx for idx, step in enumerate(steps) if step is plot_step),
                None,
            )
            if agg_index is not None and plot_index is not None and agg_index > plot_index:
                steps.pop(agg_index)
                parse_index = next(
                    (idx for idx, step in enumerate(steps) if step.get("tool") == "parse_datetime"),
                    None,
                )
                target_index = plot_index
                if parse_index is not None and target_index <= parse_index:
                    target_index = parse_index + 1
                steps.insert(target_index, aggregate_step)
                fixes.append("moved aggregate step before plot_line")

        plan["steps"] = steps
        return fixes

    def apply_compare_groups_guardrails(plan: Dict[str, Any]) -> list[str]:
        fixes: list[str] = []
        df = state.dataframe
        if df is None:
            return fixes
        requirements = getattr(state, "requirements", None)
        group_by = list(getattr(requirements, "group_by", []) or [])
        steps = plan.get("steps", [])
        for step in steps:
            if step.get("tool") != "compare_groups":
                continue
            params = step.setdefault("params", {})
            params["df"] = "$aggregate_result"
            group_col = params.get("group_col") or (group_by[0] if group_by else None)
            if group_col:
                params["group_col"] = group_col
            if not group_col or group_col not in df.columns:
                continue
            if "groups" in params:
                valid_groups = set(df[group_col].dropna().unique().tolist())
                filtered = [g for g in params.get("groups", []) if g in valid_groups]
                if len(filtered) >= 2:
                    params["groups"] = filtered
                    fixes.append(f"filtered compare_groups groups for {group_col}")
                else:
                    params.pop("groups", None)
                    fixes.append(f"removed compare_groups groups for {group_col}")
            if "groups" not in params:
                params["groups"] = sorted(df[group_col].dropna().unique().tolist())
                fixes.append(f"set compare_groups groups for {group_col}")
        plan["steps"] = steps
        return fixes

    def apply_distribution_guardrails(plan: Dict[str, Any]) -> list[str]:
        fixes: list[str] = []
        requirements = getattr(state, "requirements", None)
        analysis = set(getattr(requirements, "analysis", []) or [])
        if "distribution" in analysis:
            return fixes
        steps = plan.get("steps", [])
        filtered_steps = [step for step in steps if step.get("tool") != "plot_histogram"]
        if len(filtered_steps) != len(steps):
            fixes.append("removed plot_histogram (distribution not requested)")
        plan["steps"] = filtered_steps
        return fixes

    plan = state.plan_skeleton or state.template_plan
    if not plan:
        return {"current_node": "error", "error_message": "Missing plan skeleton"}

    filled_plan = state.llm_client.fill_plan_params(
        plan,
        state.schema,
        requirements=getattr(state, "requirements", None),
    )
    state.adapted_plan = filled_plan

    state.llm_requests.append({
        "step": "plan_param_fill",
        "prompt": getattr(state.llm_client, "last_prompt", None),
        "response": getattr(state.llm_client, "last_response", None),
        "node": "fill_params",
    })
    if state.llm_requests[-1].get("prompt"):
        state.artifact_manager.save_text(
            state.llm_requests[-1]["prompt"],
            "logs",
            "llm_param_fill_prompt.txt",
        )
    if state.llm_requests[-1].get("response"):
        state.artifact_manager.save_text(
            state.llm_requests[-1]["response"],
            "logs",
            "llm_param_fill_response.txt",
        )

    state.llm_interactions.append({
        "step": "plan_param_fill",
        "input": {
            "template": state.template_plan.get("template_id") if state.template_plan else None,
            "question": state.question,
        },
        "output": {
            "steps": len(filled_plan.get("steps", [])),
            "rationale": filled_plan.get("param_fill_rationale", ""),
        },
        "node": "fill_params",
    })

    repairs = repair_plan_step_columns(filled_plan, state.schema)
    if repairs:
        logger.info("Repaired plan step columns: %s", "; ".join(repairs[:5]))

    type_fixes = normalize_plan_param_types(filled_plan)
    if type_fixes:
        logger.info("Normalized plan param types: %s", "; ".join(type_fixes[:5]))

    guardrail_fixes = apply_time_groupby_guardrails(filled_plan)
    if guardrail_fixes:
        logger.info("Applied time/group_by guardrails: %s", "; ".join(guardrail_fixes[:5]))

    compare_fixes = apply_compare_groups_guardrails(filled_plan)
    if compare_fixes:
        logger.info("Applied compare_groups guardrails: %s", "; ".join(compare_fixes[:5]))

    distribution_fixes = apply_distribution_guardrails(filled_plan)
    if distribution_fixes:
        logger.info("Applied distribution guardrails: %s", "; ".join(distribution_fixes[:5]))

    removed_steps = prune_unjustified_steps(filled_plan, state.requirements)
    if removed_steps:
        logger.info("Removed unjustified steps: %s", ", ".join(removed_steps))

    column_errors = validate_plan_step_columns(filled_plan, state.schema)
    is_valid, coverage_report = validate_plan_coverage(filled_plan, state.requirements)
    if column_errors:
        coverage_report["column_errors"] = column_errors
        is_valid = False

    if not is_valid:
        error_payload = generate_coverage_error_payload(coverage_report)
        return {"current_node": "error", "error_message": error_payload}

    state.artifact_manager.save_json(
        state.llm_requests,
        "llm_requests",
        "llm_requests.json",
    )
    state.artifact_manager.save_json(
        filled_plan,
        "plan_adapted",
        "plan_adapted.json",
    )

    rationale = filled_plan.get("adaptation_rationale", "")
    diff_text = generate_diff(state.template_plan, filled_plan, rationale)
    state.artifact_manager.save_text(diff_text, "plan_diff", "plan_diff.md")

    return {
        "current_node": "fill_params",
        "adapted_plan": filled_plan,
    }


def adapt_plan_node(state: GraphState) -> Dict[str, Any]:
    """
    Adapt template plan using LLM.

    Args:
        state: Current graph state

    Returns:
        Updated state with adapted plan
    """
    logger.info("Adapting plan template")
    state.current_node = "adapt_plan"
    
    try:
        # Adapt plan using LLM with narrowed tools
        adapted_plan = state.llm_client.adapt_plan(
            state.template_plan,
            state.schema,
            state.intent,
            state.question,
            narrowed_tools=getattr(state, 'narrowed_tools', None),
            requirements=getattr(state, 'requirements', None),
        )
        state.adapted_plan = adapted_plan

        state.llm_requests.append({
            "step": "plan_adaptation",
            "prompt": getattr(state.llm_client, "last_prompt", None),
            "response": getattr(state.llm_client, "last_response", None),
            "node": "adapt_plan",
        })
        if state.llm_requests[-1].get("prompt"):
            state.artifact_manager.save_text(
                state.llm_requests[-1]["prompt"],
                "logs",
                "llm_adapt_prompt.txt",
            )
        if state.llm_requests[-1].get("response"):
            state.artifact_manager.save_text(
                state.llm_requests[-1]["response"],
                "logs",
                "llm_adapt_response.txt",
            )
        
        # Validate coverage if requirements exist
        if state.requirements and hasattr(state.requirements, 'metrics'):
            from autoviz_agent.planning.diff import (
                validate_plan_coverage,
                generate_coverage_error_payload,
                prune_unjustified_steps,
            )
            from autoviz_agent.registry.validation import (
                repair_plan_step_columns,
                validate_plan_step_columns,
            )

            repairs = repair_plan_step_columns(adapted_plan, state.schema)
            if repairs:
                logger.info(
                    "Repaired plan step columns: %s",
                    "; ".join(repairs[:5]) + ("..." if len(repairs) > 5 else ""),
                )

            removed_steps = prune_unjustified_steps(adapted_plan, state.requirements)
            if removed_steps:
                logger.info(
                    "Removed unjustified steps: %s",
                    ", ".join(removed_steps),
                )

            column_errors = validate_plan_step_columns(adapted_plan, state.schema)
            is_valid, coverage_report = validate_plan_coverage(adapted_plan, state.requirements)
            if column_errors:
                coverage_report["column_errors"] = column_errors
                is_valid = False
            
            if not is_valid:
                logger.warning(f"Plan coverage validation failed: {coverage_report}")
                
                # Implement retrieval fallback: expand tool list and retry once
                if state.coverage_retry_count == 0:
                    logger.info("Attempting tool expansion fallback...")
                    state.coverage_retry_count += 1
                    
                    try:
                        # Re-narrow with expanded parameters (no cap, higher top_k)
                        retriever = get_tool_retriever()
                        template_tools = state.template_plan.get("curated_tools", [])
                        expanded_tools = retriever.retrieve_tools(
                            state.requirements,
                            template_tools,
                            top_k=15,  # Increase from 5
                            cap=None   # Remove cap to allow more tools
                        )
                        
                        logger.info(f"Expanded from {len(state.narrowed_tools)} to {len(expanded_tools)} tools")
                        print(
                            f"\n[Fallback] Expanding tool list to improve coverage "
                            f"({len(state.narrowed_tools)} -> {len(expanded_tools)} tools)"
                        )
                        
                        # Re-adapt with expanded tools
                        error_payload = generate_coverage_error_payload(coverage_report)
                        adapted_plan = state.llm_client.adapt_plan(
                            state.template_plan,
                            state.schema,
                            state.intent,
                            state.question,
                            narrowed_tools=expanded_tools,
                            requirements=getattr(state, 'requirements', None),
                            validation_errors=error_payload,
                        )
                        state.adapted_plan = adapted_plan
                        state.narrowed_tools = expanded_tools

                        state.llm_requests.append({
                            "step": "plan_adaptation_retry",
                            "prompt": getattr(state.llm_client, "last_prompt", None),
                            "response": getattr(state.llm_client, "last_response", None),
                            "node": "adapt_plan",
                        })
                        if state.llm_requests[-1].get("prompt"):
                            state.artifact_manager.save_text(
                                state.llm_requests[-1]["prompt"],
                                "logs",
                                "llm_adapt_prompt_retry.txt",
                            )
                        if state.llm_requests[-1].get("response"):
                            state.artifact_manager.save_text(
                                state.llm_requests[-1]["response"],
                                "logs",
                                "llm_adapt_response_retry.txt",
                            )
                        
                        # Re-validate coverage
                        repairs_retry = repair_plan_step_columns(adapted_plan, state.schema)
                        if repairs_retry:
                            logger.info(
                                "Repaired plan step columns (retry): %s",
                                "; ".join(repairs_retry[:5]) + ("..." if len(repairs_retry) > 5 else ""),
                            )

                        removed_retry = prune_unjustified_steps(adapted_plan, state.requirements)
                        if removed_retry:
                            logger.info(
                                "Removed unjustified steps (retry): %s",
                                ", ".join(removed_retry),
                            )

                        column_errors_retry = validate_plan_step_columns(adapted_plan, state.schema)
                        is_valid_retry, coverage_report_retry = validate_plan_coverage(adapted_plan, state.requirements)
                        if column_errors_retry:
                            coverage_report_retry["column_errors"] = column_errors_retry
                            is_valid_retry = False
                        if is_valid_retry:
                            logger.info("Coverage improved after tool expansion")
                            print("[Fallback] Coverage validation passed after expansion")
                        else:
                            logger.warning(f"Coverage still insufficient after expansion: {coverage_report_retry}")
                            error_payload = generate_coverage_error_payload(coverage_report_retry)
                            return {"current_node": "error", "error_message": error_payload}
                    except Exception as e:
                        logger.error(f"Tool expansion fallback failed: {e}")
                        return {"current_node": "error", "error_message": str(e)}
                else:
                    logger.warning("Coverage retry limit reached, proceeding with current plan")
                    error_payload = generate_coverage_error_payload(coverage_report)
                    return {"current_node": "error", "error_message": error_payload}
        
        # Track LLM interaction
        state.llm_interactions.append({
            "step": "plan_adaptation",
            "input": {
                "template": state.template_plan.get('template_id'),
                "question": state.question
            },
            "output": {
                "changes_applied": adapted_plan.get('changes_applied', 0),
                "rationale": adapted_plan.get('adaptation_rationale', '')
            },
            "node": "adapt_plan"
        })
        state.artifact_manager.save_json(
            state.llm_requests,
            "llm_requests",
            "llm_requests.json",
        )
        
        # Save adapted plan
        adapted_path = state.artifact_manager.save_json(adapted_plan, "plan_adapted", "plan_adapted.json")
        
        # Generate and save diff with rationale
        rationale = adapted_plan.get('adaptation_rationale', '')
        diff_text = generate_diff(state.template_plan, adapted_plan, rationale)
        diff_path = state.artifact_manager.save_text(diff_text, "plan_diff", "plan_diff.md")
        
        changes_applied = adapted_plan.get('changes_applied', 0)
        logger.info(f"Plan adapted: {changes_applied} changes")
        
        adaptation_rationale = adapted_plan.get('adaptation_rationale', '')
        if changes_applied > 0:
            print(f"\n[Adapted] Plan modified with {changes_applied} change(s)")
            if adaptation_rationale:
                print(f"          Reason: {adaptation_rationale}")
        else:
            print(f"\n[Ready] Template fits your question - no adaptations needed")
            if adaptation_rationale:
                print(f"        Note: {adaptation_rationale}")
        
        return {
            "current_node": "adapt_plan",
            "adapted_plan": adapted_plan,
            "llm_interactions": state.llm_interactions,
            "llm_requests": state.llm_requests,
        }
    except Exception as e:
        logger.error(f"Plan adaptation failed: {e}")
        return {"current_node": "error", "error_message": str(e)}


def compile_tool_calls_node(state: GraphState) -> Dict[str, Any]:
    """
    Compile tool calls from adapted plan.

    Args:
        state: Current graph state

    Returns:
        Updated state with tool calls
    """
    logger.info("Compiling tool calls")
    state.current_node = "compile_tool_calls"
    
    try:
        # Generate tool calls from plan
        tool_calls = state.llm_client.generate_tool_calls(
            state.adapted_plan, 
            state.schema,
            state.artifact_manager,
            state.question,  # Pass user question for column extraction
            execution_log=state.execution_log,
        )
        state.tool_calls = tool_calls
        
        # Save tool calls
        tool_calls_path = state.artifact_manager.save_json(tool_calls, "tool_calls", "tool_calls.json")
        
        logger.info(f"Compiled {len(tool_calls)} tool calls")
        print(f"\n[Tools] Prepared {len(tool_calls)} analysis steps")
        for i, tc in enumerate(tool_calls[:3], 1):
            print(f"        {i}. {tc['tool']} - {tc.get('description', '')}")
        if len(tool_calls) > 3:
            print(f"        ... and {len(tool_calls) - 3} more steps")
        return {
            "current_node": "compile_tool_calls",
            "tool_calls": tool_calls,
        }
    except Exception as e:
        logger.error(f"Tool call compilation failed: {e}")
        return {"current_node": "error", "error_message": str(e)}


def execute_tools_node(state: GraphState) -> Dict[str, Any]:
    """
    Execute compiled tool calls.

    Args:
        state: Current graph state

    Returns:
        Updated state with execution results
    """
    logger.info("Executing tool calls")
    state.current_node = "execute_tools"
    
    try:
        # Initialize tool executor
        executor = ToolExecutor(state.artifact_manager)
        
        # Execute each tool call
        results = []
        context = {"dataframe": state.dataframe, "schema": state.schema}
        semantic_repair_detected = False
        
        for tool_call in state.tool_calls:
            try:
                result = executor.execute(tool_call, context)
                results.append({
                    "tool": tool_call["tool"],
                    "success": result.success,
                    "result": result.outputs,
                    "duration_ms": result.duration_ms,
                })
                
                # Phase 6: Check if repairs were semantic
                if result.outputs and isinstance(result.outputs, dict):
                    repair_details = result.outputs.get("details", {})
                    if repair_details.get("changed_params"):
                        # Classify repairs that occurred
                        from autoviz_agent.registry.validation import classify_repair, RepairType
                        for param_name, change_info in repair_details.get("changed_params", {}).items():
                            old_val = change_info.get("old")
                            new_val = change_info.get("new")
                            repair_type = classify_repair(param_name, old_val, new_val, tool_call["tool"])
                            
                            if repair_type == RepairType.SEMANTIC:
                                logger.warning(f"Semantic repair detected: {param_name} changed from {old_val} to {new_val}")
                                print(f"\n[Semantic Repair] Parameter '{param_name}' was changed (may alter analysis intent)")
                                state.last_repair_type = "semantic"
                                semantic_repair_detected = True
                            
            except Exception as e:
                logger.error(f"Tool {tool_call['tool']} failed: {e}")
                results.append({"tool": tool_call["tool"], "success": False, "error": str(e)})
        
        state.execution_results = results
        
        # Phase 6: Recommend replan if semantic repairs detected
        if semantic_repair_detected:
            logger.warning("Semantic repairs detected - replan recommended to validate intent alignment")
            print("[Recommendation] Semantic repairs detected. Consider re-running with adjusted parameters.")
            # Note: Full replan trigger would require graph cycle back to adapt_plan_node
            # Current implementation logs recommendation for user awareness
        
        # Save execution log
        log_data = executor.execution_log.to_dict()
        log_data["run_id"] = state.run_id
        log_data["status"] = "completed" if all(r.get("success") for r in results) else "partial_failure"
        log_data["semantic_repair_detected"] = semantic_repair_detected
        state.artifact_manager.save_json(log_data, "execution_log", "execution_log.json")
        
        logger.info(f"Executed {len(results)} tools")
        
        successful = sum(1 for r in results if r.get('success'))
        failed = len(results) - successful
        print(f"\n[Complete] {successful} successful, {failed} failed")
        
        return {
            "current_node": "execute_tools",
            "execution_results": results,
        }
    except Exception as e:
        logger.error(f"Tool execution failed: {e}")
        return {"current_node": "error", "error_message": str(e)}


def summarize_node(state: GraphState) -> Dict[str, Any]:
    """
    Generate final report and summary.

    Args:
        state: Current graph state

    Returns:
        Updated state with report
    """
    logger.info("Generating final report")
    state.current_node = "summarize"
    
    try:
        # Create report writer
        report = ReportWriter()
        
        # Add report sections
        report.add_header("AutoViz Agent Analysis Report", level=1)
        report.add_text(f"**Question**: {state.question}")
        report.add_text(f"**Run ID**: {state.run_id}")
        report.add_text(f"**Intent**: {state.intent.label.value if state.intent else 'unknown'}")
        
        report.add_header("Dataset Overview", level=2)
        from autoviz_agent.tools.schema import get_schema_summary
        schema_summary = get_schema_summary(state.schema)
        report.add_text(schema_summary)
        
        # Add LLM interactions section
        if state.llm_interactions:
            report.add_llm_interactions_section(state.llm_interactions)
        if not state.llm_requests and state.artifact_manager:
            try:
                state.llm_requests = state.artifact_manager.load_json(
                    "llm_requests",
                    "llm_requests.json",
                )
            except Exception as e:
                logger.debug(f"Failed to load llm_requests.json: {e}")
        if state.artifact_manager and state.llm_requests is not None:
            has_tool_selection = any(
                entry.get("step") == "tool_selection"
                for entry in state.llm_requests
            )
            if not has_tool_selection:
                tool_prompt_path = state.artifact_manager.get_path(
                    "logs",
                    "llm_tool_selection_prompt.txt",
                )
                tool_response_path = state.artifact_manager.get_path(
                    "logs",
                    "llm_tool_selection_response.txt",
                )
                if tool_prompt_path.exists() or tool_response_path.exists():
                    entry = {
                        "step": "tool_selection",
                        "prompt": tool_prompt_path.read_text(encoding="utf-8") if tool_prompt_path.exists() else None,
                        "response": tool_response_path.read_text(encoding="utf-8") if tool_response_path.exists() else None,
                        "node": "select_tools",
                    }
                    state.llm_requests.append(entry)
        if state.llm_requests:
            report.add_llm_request_details(state.llm_requests)
        
        # Add key metrics section
        if state.execution_results:
            report.add_key_metrics_section(state.execution_results)

        # Add charts section
        if state.execution_results:
            report.add_charts_section(state.execution_results)
        
        repair_entries = []
        if state.execution_log:
            for entry in state.execution_log.entries:
                result = entry.result or {}
                if result.get("repair_attempt"):
                    details = result.get("details", {})
                    removed = details.get("removed_params") or []
                    added = details.get("added_params") or {}
                    changed = details.get("changed_params") or {}
                    summary = [f"tool={entry.tool}"]
                    if removed:
                        summary.append(f"removed={', '.join(removed)}")
                    if added:
                        summary.append(f"added={', '.join(added.keys())}")
                    if changed:
                        summary.append(f"changed={', '.join(changed.keys())}")
                    repair_entries.append(", ".join(summary))

        if repair_entries:
            report.add_header("Repairs and Validation", level=2)
            report.add_text(
                "Some tool calls required repairs due to invalid parameters or "
                "column name mismatches."
            )
            report.add_list(repair_entries)

        # Add provenance section
        report.add_header("Plan Provenance", level=2)
        report.add_text(f"**Template**: {state.template_plan.get('template_id')}")
        report.add_text(f"**Adaptations**: {state.adapted_plan.get('changes_applied', 0)}")
        report.add_text(f"**Rationale**: {state.adapted_plan.get('adaptation_rationale', '')}")
        
        # Save report
        report_path = state.artifact_manager.get_path("report", "report.md")
        report.write(report_path)
        
        logger.info(f"Report saved: {report_path}")
        return {
            "current_node": "complete",
        }
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        return {"current_node": "error", "error_message": str(e)}


def repair_or_clarify_node(state: GraphState) -> Dict[str, Any]:
    """
    Handle validation errors and repair attempts.

    Args:
        state: Current graph state

    Returns:
        Updated state
    """
    logger.info("Attempting repair or requesting clarification")
    state.current_node = "repair_or_clarify"
    
    # Placeholder - would implement repair logic
    return {"current_node": "error", "error_message": "Repair not implemented"}


def error_node(state: GraphState) -> Dict[str, Any]:
    """
    Handle errors.

    Args:
        state: Current graph state

    Returns:
        Updated state
    """
    error_msg = getattr(state, 'error_message', 'Unknown error')
    logger.error(f"Error node reached: {error_msg}")
    
    # Show user-friendly error message
    print(f"\n[ERROR] {error_msg}")
    
    # Provide helpful hints based on error type
    if "not found" in error_msg.lower():
        print("        Tip: Check that the file path is correct and the file exists")
    elif "permission" in error_msg.lower():
        print("   [TIP] Check file permissions or try running with appropriate access")
    elif "columns" in error_msg.lower() or "schema" in error_msg.lower():
        print("   [TIP] The dataset may be empty or improperly formatted")
    
    state.current_node = "error"
    
    return {
        "current_node": "error",
    }


def complete_node(state: GraphState) -> Dict[str, Any]:
    """
    Complete the analysis run.

    Args:
        state: Current graph state

    Returns:
        Updated state
    """
    logger.info("Analysis complete")
    state.current_node = "complete"
    
    return {
        "current_node": "complete",
    }
