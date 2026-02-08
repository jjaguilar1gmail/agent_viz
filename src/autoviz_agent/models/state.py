"""Core state models for the AutoViz Agent."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from autoviz_agent.registry.intents import get_intent_labels


def _build_intent_label() -> Enum:
    labels = get_intent_labels(exposed_only=False)
    return Enum("IntentLabel", {label.upper(): label for label in labels}, type=str)


IntentLabel = _build_intent_label()
IntentLabel.__doc__ = "Intent classification labels."


class RunStatus(str, Enum):
    """Run execution status."""

    INITIALIZED = "initialized"
    SCHEMA_INFERRED = "schema_inferred"
    INTENT_CLASSIFIED = "intent_classified"
    TEMPLATE_SELECTED = "template_selected"
    PLAN_ADAPTED = "plan_adapted"
    TOOL_CALLS_COMPILED = "tool_calls_compiled"
    TOOLS_EXECUTED = "tools_executed"
    SUMMARIZED = "summarized"
    COMPLETED = "completed"
    REPAIR_OR_CLARIFY = "repair_or_clarify"
    FAILED = "failed"


class UserRequest(BaseModel):
    """User's analysis request."""

    question: str = Field(..., description="Natural language question")
    output_goal: Optional[str] = Field(None, description="Expected output format or goal")
    constraints: Dict[str, Any] = Field(default_factory=dict, description="Additional constraints")


class DatasetSource(BaseModel):
    """Dataset source information."""

    source_type: str = Field(..., description="Type: csv or dataframe")
    path: Optional[str] = Field(None, description="Path to dataset file")
    checksum: Optional[str] = Field(None, description="Dataset checksum for validation")


class ColumnProfile(BaseModel):
    """Profile for a single column."""

    name: str = Field(..., description="Column name")
    dtype: str = Field(..., description="Data type")
    missing_rate: float = Field(..., description="Proportion of missing values")
    cardinality: int = Field(..., description="Number of unique values")
    roles: List[str] = Field(default_factory=list, description="Inferred roles (e.g., id, metric)")


class SchemaProfile(BaseModel):
    """Dataset schema profile."""

    columns: List[ColumnProfile] = Field(default_factory=list, description="Column profiles")
    row_count: int = Field(..., description="Number of rows")
    data_shape: str = Field(..., description="Detected shape (e.g., wide, long, time_series)")


class Intent(BaseModel):
    """Classified user intent."""

    label: IntentLabel = Field(..., description="Primary intent label")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    top_intents: List[Dict[str, float]] = Field(
        default_factory=list, description="Top N intent candidates with scores"
    )


class Artifact(BaseModel):
    """Output artifact."""

    artifact_type: str = Field(
        ...,
        description="Type: chart, report, plan_template, plan_adapted, plan_diff, tool_calls, execution_log",
    )
    path: str = Field(..., description="File path")
    mime_type: Optional[str] = Field(None, description="MIME type")


class RunState(BaseModel):
    """Complete run state."""

    run_id: str = Field(..., description="Unique run identifier")
    user_request: UserRequest = Field(..., description="User's original request")
    dataset_source: DatasetSource = Field(..., description="Dataset source info")
    inferred_schema: Optional[SchemaProfile] = Field(None, description="Inferred schema profile")
    intent: Optional[Intent] = Field(None, description="Classified intent")
    template_id: Optional[str] = Field(None, description="Selected plan template ID")
    template_plan: Optional[Dict[str, Any]] = Field(None, description="Original template plan")
    adapted_plan: Optional[Dict[str, Any]] = Field(None, description="Adapted plan")
    plan_diff: Optional[str] = Field(None, description="Plan diff with rationale")
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list, description="Tool calls")
    execution_log: List[Dict[str, Any]] = Field(
        default_factory=list, description="Execution log entries"
    )
    artifacts: List[Artifact] = Field(default_factory=list, description="Output artifacts")
    status: RunStatus = Field(default=RunStatus.INITIALIZED, description="Current status")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update")
    error_message: Optional[str] = Field(None, description="Error message if failed")
