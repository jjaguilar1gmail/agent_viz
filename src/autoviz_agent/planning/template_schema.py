"""Plan template JSON schema definition."""

from typing import Any, Dict

from autoviz_agent.registry.intents import get_intent_labels

# JSON Schema for plan templates
PLAN_TEMPLATE_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["template_id", "version", "intents", "data_shape", "steps"],
    "additionalProperties": False,
    "properties": {
        "template_id": {"type": "string", "description": "Unique template identifier"},
        "version": {"type": "string", "description": "Template version"},
        "name": {"type": "string", "description": "Human-readable template name"},
        "description": {"type": "string", "description": "Template description"},
        "intents": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": get_intent_labels(exposed_only=False),
            },
            "minItems": 1,
            "description": "Supported intent labels",
        },
        "data_shape": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Required data shapes (e.g., wide, long, time_series)",
        },
        "requires": {
            "type": "object",
            "properties": {
                "min_rows": {"type": "integer", "minimum": 0},
                "min_columns": {"type": "integer", "minimum": 0},
                "column_types": {"type": "array", "items": {"type": "string"}},
            },
            "description": "Hard requirements for template applicability",
        },
        "prefers": {
            "type": "object",
            "properties": {
                "has_datetime": {"type": "boolean"},
                "has_categorical": {"type": "boolean"},
                "has_numeric": {"type": "boolean"},
            },
            "description": "Preferred characteristics (scoring factors)",
        },
        "supports": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Supported analysis types",
        },
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["step_id", "tool", "description"],
                "additionalProperties": False,
                "properties": {
                    "step_id": {"type": "string"},
                    "tool": {"type": "string"},
                    "description": {"type": "string"},
                    "params": {"type": "object"},
                    "depends_on": {"type": "array", "items": {"type": "string"}},
                },
            },
            "description": "Ordered list of plan steps",
        },
    },
}
