"""LLM contracts for structured outputs with xgrammar2 support.

This module defines JSON schemas that can be used for:
1. Validating LLM outputs
2. Grammar-constrained generation with xgrammar2
3. Type hints and documentation
"""

from typing import Any, Dict, List, Literal
from pydantic import BaseModel, Field, field_validator

from autoviz_agent.registry.intents import get_intent_labels


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


def get_adaptation_schema() -> Dict[str, Any]:
    """
    Get JSON schema for plan adaptation.
    
    Compatible with xgrammar2 grammar generation and JSON Schema Draft 7.
    
    Returns:
        JSON schema dict
    """
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
                            "description": "Tool parameters"
                        }
                    },
                    "required": ["action", "step_id"],
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


def get_requirement_extraction_schema() -> Dict[str, Any]:
    """
    Get JSON schema for requirement extraction.
    
    Compatible with xgrammar2 grammar generation and JSON Schema Draft 7.
    
    Returns:
        JSON schema dict
    """
    return {
        "type": "object",
        "properties": {
            "metrics": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Numeric columns to analyze"
            },
            "group_by": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Categorical columns for grouping"
            },
            "time": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": "Time column name"
                    },
                    "grain": {
                        "type": "string",
                        "description": "Time grain or unknown"
                    }
                },
                "required": ["column", "grain"],
                "additionalProperties": False
            },
            "analysis": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ALLOWED_ANALYSIS_TYPES
                },
                "description": "Analysis types from allowed set"
            },
            "outputs": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Output types: chart, table"
            },
            "constraints": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Special conditions or filters"
            }
        },
        "required": ["metrics", "group_by", "time", "analysis", "outputs", "constraints"],
        "additionalProperties": False
    }


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
