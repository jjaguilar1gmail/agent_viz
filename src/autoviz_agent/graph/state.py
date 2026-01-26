"""Graph state schema and routing enums."""

from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class GraphNodeType(str, Enum):
    """Graph node types."""

    INITIALIZE = "initialize"
    INFER_SCHEMA = "infer_schema"
    CLASSIFY_INTENT = "classify_intent"
    SELECT_TEMPLATE = "select_template"
    ADAPT_PLAN = "adapt_plan"
    COMPILE_TOOL_CALLS = "compile_tool_calls"
    EXECUTE_TOOLS = "execute_tools"
    SUMMARIZE = "summarize"
    REPAIR_OR_CLARIFY = "repair_or_clarify"
    COMPLETE = "complete"
    ERROR = "error"


class GraphEdge(str, Enum):
    """Graph edge types for routing."""

    SUCCESS = "success"
    FAILURE = "failure"
    VALIDATION_ERROR = "validation_error"
    NEEDS_REPAIR = "needs_repair"
    COMPLETE = "complete"


class GraphState(BaseModel):
    """State passed between graph nodes."""

    run_id: str = Field(..., description="Unique run identifier")
    dataset_path: Optional[str] = Field(None, description="Path to dataset")
    question: str = Field(..., description="User question")

    # Configuration and components
    config: Optional[Dict[str, Any]] = Field(None, description="Configuration")
    artifact_manager: Optional[Any] = Field(None, description="Artifact manager")
    execution_log: Optional[Any] = Field(None, description="Execution log")
    llm_client: Optional[Any] = Field(None, description="LLM client")
    dataframe: Optional[Any] = Field(None, description="Loaded dataframe")

    # Processing state
    schema: Optional[Any] = Field(None, description="Inferred schema")
    intent: Optional[Any] = Field(None, description="Classified intent")
    template_plan: Optional[Dict[str, Any]] = Field(None, description="Template plan")
    adapted_plan: Optional[Dict[str, Any]] = Field(None, description="Adapted plan")
    tool_calls: list[Dict[str, Any]] = Field(default_factory=list, description="Tool calls")
    execution_results: list[Dict[str, Any]] = Field(
        default_factory=list, description="Execution results"
    )
    llm_interactions: list[Dict[str, Any]] = Field(
        default_factory=list, description="Log of LLM interactions"
    )

    # Status tracking
    current_node: str = Field(default="initialize", description="Current node")
    error_message: Optional[str] = Field(None, description="Error message if failed")

    class Config:
        arbitrary_types_allowed = True
