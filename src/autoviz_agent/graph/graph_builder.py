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
    infer_schema_node,
    initialize_node,
    repair_or_clarify_node,
    select_template_node,
    summarize_node,
)
from autoviz_agent.graph.state import GraphState
from autoviz_agent.utils.logging import get_logger

logger = get_logger(__name__)


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

    # Add edges (linear flow for now, will add conditional routing later)
    workflow.add_edge("initialize", "infer_schema")
    workflow.add_edge("infer_schema", "classify_intent")
    workflow.add_edge("classify_intent", "select_template")
    workflow.add_edge("select_template", "adapt_plan")
    workflow.add_edge("adapt_plan", "compile_tool_calls")
    workflow.add_edge("compile_tool_calls", "execute_tools")
    workflow.add_edge("execute_tools", "summarize")
    workflow.add_edge("summarize", "complete")
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
