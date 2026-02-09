"""LLM contracts for structured outputs with xgrammar2 support.

This module defines JSON schemas that can be used for:
1. Validating LLM outputs
2. Grammar-constrained generation with xgrammar2
3. Type hints and documentation
"""

from typing import Any, Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel, Field, field_validator

from autoviz_agent.registry.intents import get_intent_labels
from autoviz_agent.models.state import SchemaProfile


# =============================================================================
# Intent Classification Contracts
# =============================================================================

INTENT_LABELS = get_intent_labels(exposed_only=True)


class IntentOutput(BaseModel):
    """Intent classification output contract."""

    primary: str = Field(..., description="Primary intent label")
    
    confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="Confidence score between 0.0 and 1.0"
    )
    
    reasoning: str = Field(
        ...,
        description="Brief explanation of why this intent was chosen"
    )

    @field_validator("primary")
    @classmethod
    def validate_primary(cls, value: str) -> str:
        if value not in INTENT_LABELS:
            raise ValueError(f"Unknown intent label: {value}")
        return value


# =============================================================================
# Requirement Extraction Contracts
# =============================================================================

ALLOWED_ANALYSIS_TYPES = [
    "total",
    "compare",
    "trend",
    "distribution",
    "anomaly",
    "correlation"
]


class TimeRequirement(BaseModel):
    """Time-related requirements."""
    
    column: str = Field(
        default="",
        description="Time column name, empty if not temporal analysis"
    )
    
    grain: str = Field(
        default="unknown",
        description="Time grain (daily, weekly, monthly, yearly, or unknown)"
    )


class RequirementExtractionOutput(BaseModel):
    """Structured requirements extracted from user question."""
    
    metrics: List[str] = Field(
        default_factory=list,
        description="Numeric columns to analyze (e.g., revenue, sales)"
    )
    
    group_by: List[str] = Field(
        default_factory=list,
        description="Categorical columns for grouping/segmentation"
    )
    
    time: TimeRequirement = Field(
        default_factory=TimeRequirement,
        description="Time-related requirements"
    )
    
    analysis: List[str] = Field(
        default_factory=list,
        description="Analysis types from allowed set"
    )
    
    outputs: List[str] = Field(
        default_factory=list,
        description="Output types: chart, table, or both"
    )
    
    constraints: List[str] = Field(
        default_factory=list,
        description="Special conditions, filters, or requirements"
    )
    
    @field_validator("analysis")
    @classmethod
    def validate_analysis(cls, values: List[str]) -> List[str]:
        """Validate that all analysis types are from allowed set."""
        for value in values:
            if value not in ALLOWED_ANALYSIS_TYPES:
                raise ValueError(
                    f"Unknown analysis type: {value}. "
                    f"Allowed: {ALLOWED_ANALYSIS_TYPES}"
                )
        return values


# =============================================================================
# Plan Adaptation Contracts
# =============================================================================

class PlanChange(BaseModel):
    """A single change to apply to a plan template."""
    
    action: Literal["add", "remove", "modify"] = Field(
        ...,
        description="Type of change: add a step, remove a step, or modify a step"
    )
    
    step_id: str = Field(
        ...,
        description="Unique identifier for the step being changed"
    )
    
    tool: str | None = Field(
        None,
        description="Tool name (required for 'add' actions)"
    )
    
    description: str | None = Field(
        None,
        description="Human-readable description of this change"
    )
    
    params: Dict[str, Any] | None = Field(
        None,
        description="Parameters for the tool (required for 'add' actions)"
    )

    satisfies: List[str] = Field(
        ...,
        description="Requirement labels this step satisfies (e.g., analysis.total, group_by)"
    )


class AdaptationOutput(BaseModel):
    """Plan adaptation output contract."""
    
    changes: List[PlanChange] = Field(
        default_factory=list,
        description="List of changes to apply to the template"
    )
    
    rationale: str = Field(
        ...,
        description="Overall explanation of why these changes were made"
    )


# =============================================================================
# JSON Schema Generation (for xgrammar2)
# =============================================================================

def get_intent_schema() -> Dict[str, Any]:
    """
    Get JSON schema for intent classification.
    
    Compatible with xgrammar2 grammar generation and JSON Schema Draft 7.
    
    Returns:
        JSON schema dict
    """
    return {
        "type": "object",
        "properties": {
            "primary": {
                "type": "string",
                "enum": INTENT_LABELS,
                "description": "Primary intent label"
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Confidence score"
            },
            "reasoning": {
                "type": "string",
                "description": "Brief explanation of intent choice"
            }
        },
        "required": ["primary", "confidence", "reasoning"],
        "additionalProperties": False
    }


def get_adaptation_schema(
    allowed_columns: Optional[List[str]] = None,
    numeric_columns: Optional[List[str]] = None,
    categorical_columns: Optional[List[str]] = None,
    temporal_columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Get JSON schema for plan adaptation.
    
    Compatible with xgrammar2 grammar generation and JSON Schema Draft 7.
    
    Returns:
        JSON schema dict
    """
    def _string_enum(candidates: Optional[List[str]]) -> Dict[str, Any]:
        if candidates:
            return {"type": "string", "enum": candidates}
        return {"type": "string"}

    def _array_enum(candidates: Optional[List[str]]) -> Dict[str, Any]:
        return {"type": "array", "items": _string_enum(candidates)}

    column_enum = allowed_columns or []
    numeric_enum = numeric_columns or column_enum
    categorical_enum = categorical_columns or column_enum
    temporal_enum = temporal_columns or column_enum

    return {
        "type": "object",
        "properties": {
            "changes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["add", "remove", "modify"],
                            "description": "Type of change"
                        },
                        "step_id": {
                            "type": "string",
                            "description": "Step identifier"
                        },
                        "tool": {
                            "type": ["string", "null"],
                            "description": "Tool name (for add actions)"
                        },
                        "description": {
                            "type": ["string", "null"],
                            "description": "Change description"
                        },
                        "params": {
                            "type": ["object", "null"],
                            "description": "Tool parameters",
                            "properties": {
                                "metrics": _array_enum(numeric_enum),
                                "group_by": _array_enum(categorical_enum),
                                "x": _string_enum(column_enum),
                                "y": _string_enum(column_enum),
                                "column": _string_enum(column_enum),
                                "columns": _array_enum(temporal_enum),
                                "time_column": _string_enum(temporal_enum),
                                "date_column": _string_enum(temporal_enum),
                            },
                            "additionalProperties": True,
                        },
                        "satisfies": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Requirement labels satisfied by this step"
                        }
                    },
                    "required": ["action", "step_id", "satisfies"],
                    "additionalProperties": False
                }
            },
            "rationale": {
                "type": "string",
                "description": "Overall explanation"
            }
        },
        "required": ["changes", "rationale"],
        "additionalProperties": False
    }


def get_requirement_extraction_schema(
    allowed_columns: Optional[List[str]] = None,
    numeric_columns: Optional[List[str]] = None,
    categorical_columns: Optional[List[str]] = None,
    temporal_columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Get JSON schema for requirement extraction.
    
    Compatible with xgrammar2 grammar generation and JSON Schema Draft 7.
    All fields are optional since not all analyses require all requirement types.
    
    Returns:
        JSON schema dict
    """
    def _items_schema(candidates: Optional[List[str]]) -> Dict[str, Any]:
        if candidates:
            return {"type": "string", "enum": candidates}
        return {"type": "string"}

    time_column_enum = None
    if temporal_columns:
        time_column_enum = [""] + temporal_columns

    return {
        "type": "object",
        "properties": {
            "metrics": {
                "type": "array",
                "items": _items_schema(numeric_columns or allowed_columns),
                "description": "Numeric columns to analyze",
                "default": []
            },
            "group_by": {
                "type": "array",
                "items": _items_schema(categorical_columns or allowed_columns),
                "description": "Categorical columns for grouping",
                "default": []
            },
            "time": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        **({"enum": time_column_enum} if time_column_enum else {}),
                        "description": "Time column name",
                        "default": ""
                    },
                    "grain": {
                        "type": "string",
                        "description": "Time grain or unknown",
                        "default": "unknown"
                    }
                },
                "additionalProperties": False,
                "default": {"column": "", "grain": "unknown"}
            },
            "analysis": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ALLOWED_ANALYSIS_TYPES
                },
                "description": "Analysis types from allowed set",
                "default": []
            },
            "outputs": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Output types: chart, table",
                "default": ["chart", "table"]
            },
            "constraints": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Special conditions or filters",
                "default": []
            }
        },
        "additionalProperties": False
    }


def validate_requirement_columns(
    requirements: RequirementExtractionOutput,
    schema: SchemaProfile,
) -> Tuple[bool, List[str]]:
    """
    Validate extracted requirement columns against schema columns.

    Returns:
        (is_valid, error_messages)
    """
    schema_columns = [col.name for col in schema.columns]
    temporal_columns = [c.name for c in schema.columns if 'temporal' in c.roles]
    numeric_columns = [c.name for c in schema.columns if c.dtype in ['int64', 'float64', 'float', 'int']]
    categorical_columns = [c.name for c in schema.columns if c.dtype in ['object', 'string', 'category']]

    errors: List[str] = []

    for metric in requirements.metrics:
        if metric not in numeric_columns:
            errors.append(f"metrics: '{metric}' is not a numeric column")

    for group_col in requirements.group_by:
        if group_col not in categorical_columns:
            errors.append(f"group_by: '{group_col}' is not a categorical column")

    if requirements.time and requirements.time.column:
        if requirements.time.column not in temporal_columns:
            errors.append(f"time.column: '{requirements.time.column}' is not a temporal column")

    for output in requirements.outputs:
        if output not in ("chart", "table"):
            errors.append(f"outputs: '{output}' is not a valid output type")

    unknown_columns = [
        col for col in (requirements.metrics + requirements.group_by)
        if col not in schema_columns
    ]
    for col in unknown_columns:
        errors.append(f"unknown_column: '{col}' not in schema columns")

    return len(errors) == 0, errors


# =============================================================================
# Validation Helpers
# =============================================================================

def validate_intent_output(data: Dict[str, Any]) -> IntentOutput:
    """
    Validate and parse intent classification output.
    
    Args:
        data: Raw dict from LLM
        
    Returns:
        Validated IntentOutput
        
    Raises:
        ValidationError if data doesn't match schema
    """
    return IntentOutput(**data)


def validate_adaptation_output(data: Dict[str, Any]) -> AdaptationOutput:
    """
    Validate and parse plan adaptation output.
    
    Args:
        data: Raw dict from LLM
        
    Returns:
        Validated AdaptationOutput
        
    Raises:
        ValidationError if data doesn't match schema
    """
    return AdaptationOutput(**data)


def validate_requirement_extraction_output(
    data: Dict[str, Any]
) -> RequirementExtractionOutput:
    """
    Validate and parse requirement extraction output.
    
    Args:
        data: Raw dict from LLM
        
    Returns:
        Validated RequirementExtractionOutput
        
    Raises:
        ValidationError if data doesn't match schema
    """
    return RequirementExtractionOutput(**data)
