"""Graph node implementations with actual LLM integration."""

import json
from pathlib import Path
from typing import Any, Dict

import yaml

from autoviz_agent.graph.state import GraphState
from autoviz_agent.io.artifacts import ArtifactManager
from autoviz_agent.llm.client import LLMClient
from autoviz_agent.models.state import Intent, RunStatus, SchemaProfile
from autoviz_agent.planning.diff import generate_diff
from autoviz_agent.planning.retrieval import PlanRetrieval
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
        # Initialize LLM client
        model_config = state.config["models"][state.config["default_model"]]
        llm_client = LLMClient(model_config)
        state.llm_client = llm_client
        
        # Classify intent
        intent = llm_client.classify_intent(
            state.question,
            state.schema,
            max_intents=state.config.get("intent", {}).get("max_intents", 3),
        )
        state.intent = intent
        
        logger.info(f"Intent: {intent.label} (confidence={intent.confidence:.2f})")
        print(f"\n🎯 I classified your question as '{intent.label.value}'")
        print(f"   Confidence: {intent.confidence:.0%}")
        return {
            "current_node": "classify_intent",
            "llm_client": llm_client,
            "intent": intent,
        }
    except Exception as e:
        logger.error(f"Intent classification failed: {e}")
        return {"current_node": "error", "error_message": str(e)}


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
        
        # Save template to artifacts
        template_path = state.artifact_manager.save_json(
            selected_template, "plan_template", "plan_template.json"
        )
        
        logger.info(f"Selected template: {selected_template.get('template_id')}")
        print(f"\n📋 I selected the '{selected_template.get('template_id')}' template")
        template_name = selected_template.get('name', selected_template.get('template_id'))
        print(f"   Template: {template_name}")
        print(f"   Reason: Best match for {state.intent.label.value} intent")
        return {
            "current_node": "select_template",
            "template_plan": selected_template,
        }
    except Exception as e:
        logger.error(f"Template selection failed: {e}")
        return {"current_node": "error", "error_message": str(e)}


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
        # Adapt plan using LLM
        adapted_plan = state.llm_client.adapt_plan(
            state.template_plan, state.schema, state.intent, state.question
        )
        state.adapted_plan = adapted_plan
        
        # Save adapted plan
        adapted_path = state.artifact_manager.save_json(adapted_plan, "plan_adapted", "plan_adapted.json")
        
        # Generate and save diff
        diff_text = generate_diff(state.template_plan, adapted_plan)
        diff_path = state.artifact_manager.save_text(diff_text, "plan_diff", "plan_diff.md")
        
        changes_applied = adapted_plan.get('changes_applied', 0)
        logger.info(f"Plan adapted: {changes_applied} changes")
        
        adaptation_rationale = adapted_plan.get('adaptation_rationale', '')
        if changes_applied > 0:
            print(f"\n🔧 I adapted the plan with {changes_applied} change(s)")
            if adaptation_rationale:
                print(f"   Reason: {adaptation_rationale}")
        else:
            print(f"\n✅ The template fits your question well - no adaptations needed")
            if adaptation_rationale:
                print(f"   Note: {adaptation_rationale}")
        
        return {
            "current_node": "adapt_plan",
            "adapted_plan": adapted_plan,
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
            state.artifact_manager
        )
        state.tool_calls = tool_calls
        
        # Save tool calls
        tool_calls_path = state.artifact_manager.save_json(tool_calls, "tool_calls", "tool_calls.json")
        
        logger.info(f"Compiled {len(tool_calls)} tool calls")
        print(f"\n🔨 I prepared {len(tool_calls)} analysis steps")
        for i, tc in enumerate(tool_calls[:3], 1):
            print(f"   {i}. {tc['tool']} - {tc.get('description', '')}")
        if len(tool_calls) > 3:
            print(f"   ... and {len(tool_calls) - 3} more steps")
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
        
        for tool_call in state.tool_calls:
            try:
                result = executor.execute(tool_call, context)
                results.append({
                    "tool": tool_call["tool"],
                    "success": result.success,
                    "result": result.outputs,
                    "duration_ms": result.duration_ms,
                })
            except Exception as e:
                logger.error(f"Tool {tool_call['tool']} failed: {e}")
                results.append({"tool": tool_call["tool"], "success": False, "error": str(e)})
        
        state.execution_results = results
        
        # Save execution log
        log_data = executor.execution_log.to_dict()
        log_data["run_id"] = state.run_id
        log_data["status"] = "completed" if all(r.get("success") for r in results) else "partial_failure"
        state.artifact_manager.save_json(log_data, "execution_log", "execution_log.json")
        
        logger.info(f"Executed {len(results)} tools")
        
        successful = sum(1 for r in results if r.get('success'))
        failed = len(results) - successful
        print(f"\n✨ Execution complete: {successful} successful, {failed} failed")
        
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
    logger.error(f"Error node reached: {state.get('error_message', 'Unknown error')}")
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
