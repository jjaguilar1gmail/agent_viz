"""LangGraph pipeline wiring."""

from typing import Any, Dict

from langgraph.graph import END, StateGraph

from autoviz_agent.graph.nodes import (
    adapt_plan_node,
    classify_intent_node,
    compile_tool_calls_node,
    complete_node,
    error_node,
    execute_tools_node,
    extract_requirements_node,
    infer_schema_node,
    initialize_node,
    repair_or_clarify_node,
    select_template_node,
    summarize_node,
)
from autoviz_agent.graph.state import GraphState
from autoviz_agent.utils.logging import get_logger

logger = get_logger(__name__)


def should_continue(state: GraphState) -> str:
    """
    Determine if pipeline should continue or handle error.
    
    Args:
        state: Current graph state
        
    Returns:
        Next node name
    """
    if hasattr(state, "current_node") and state.current_node == "error":
        return "error"
    if hasattr(state, "error_message") and state.error_message:
        return "error"
    return "continue"


def build_graph() -> StateGraph:
    """
    Build the LangGraph execution pipeline.

    Returns:
        Configured StateGraph
    """
    # Create graph
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("initialize", initialize_node)
    workflow.add_node("infer_schema", infer_schema_node)
    workflow.add_node("classify_intent", classify_intent_node)
    workflow.add_node("extract_requirements", extract_requirements_node)
    workflow.add_node("select_template", select_template_node)
    workflow.add_node("adapt_plan", adapt_plan_node)
    workflow.add_node("compile_tool_calls", compile_tool_calls_node)
    workflow.add_node("execute_tools", execute_tools_node)
    workflow.add_node("summarize", summarize_node)
    workflow.add_node("repair_or_clarify", repair_or_clarify_node)
    workflow.add_node("error", error_node)
    workflow.add_node("complete", complete_node)

    # Set entry point
    workflow.set_entry_point("initialize")

    # Add conditional edges for error handling
    workflow.add_conditional_edges(
        "initialize",
        should_continue,
        {"continue": "infer_schema", "error": "error"}
    )
    workflow.add_conditional_edges(
        "infer_schema",
        should_continue,
        {"continue": "classify_intent", "error": "error"}
    )
    workflow.add_conditional_edges(
        "classify_intent",
        should_continue,
        {"continue": "extract_requirements", "error": "error"}
    )
    workflow.add_conditional_edges(
        "extract_requirements",
        should_continue,
        {"continue": "select_template", "error": "error"}
    )
    workflow.add_conditional_edges(
        "select_template",
        should_continue,
        {"continue": "adapt_plan", "error": "error"}
    )
    workflow.add_conditional_edges(
        "adapt_plan",
        should_continue,
        {"continue": "compile_tool_calls", "error": "error"}
    )
    workflow.add_conditional_edges(
        "compile_tool_calls",
        should_continue,
        {"continue": "execute_tools", "error": "error"}
    )
    workflow.add_conditional_edges(
        "execute_tools",
        should_continue,
        {"continue": "summarize", "error": "error"}
    )
    workflow.add_conditional_edges(
        "summarize",
        should_continue,
        {"continue": "complete", "error": "error"}
    )
    
    workflow.add_edge("complete", END)

    # Error handling edges
    workflow.add_edge("error", END)
    workflow.add_edge("repair_or_clarify", END)

    logger.info("Built LangGraph pipeline")
    return workflow


def create_pipeline():
    """
    Create and compile the execution pipeline.

    Returns:
        Compiled graph
    """
    graph = build_graph()
    compiled = graph.compile()
    logger.info("Compiled LangGraph pipeline")
    return compiled
