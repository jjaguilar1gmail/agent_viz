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
