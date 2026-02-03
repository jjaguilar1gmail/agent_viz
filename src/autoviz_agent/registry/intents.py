"""Intent definitions and helpers for prompt/schema generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence


@dataclass(frozen=True)
class IntentExample:
    question: str
    primary: str
    confidence: float
    reasoning: str


@dataclass(frozen=True)
class IntentDefinition:
    label: str
    description: str
    keywords: Sequence[str] = ()
    examples: Sequence[IntentExample] = ()
    is_fallback: bool = False
    fallback_text: str | None = None
    expose_to_llm: bool = True


INTENT_DEFINITIONS: List[IntentDefinition] = [
    IntentDefinition(
        label="general_eda",
        description='Broad exploration (e.g., "summarize this data", "what\'s in here?")',
        examples=[
            IntentExample(
                question="Summarize this dataset",
                primary="general_eda",
                confidence=0.9,
                reasoning="General request without a specific analysis type",
            )
        ],
        is_fallback=True,
        fallback_text="General questions",
    ),
    IntentDefinition(
        label="time_series_investigation",
        description='Temporal patterns (e.g., "trends over time", "seasonal patterns")',
        keywords=["time", "trend", "over time", "temporal", "seasonal"],
        examples=[
            IntentExample(
                question="Analyze revenue trends over time",
                primary="time_series_investigation",
                confidence=0.95,
                reasoning="Keywords 'trends' and 'over time' indicate temporal analysis",
            )
        ],
    ),
    IntentDefinition(
        label="anomaly_detection",
        description='Outliers and unusual values (e.g., "find anomalies", "detect outliers")',
        keywords=["anomaly", "outlier", "unusual", "abnormal"],
        examples=[
            IntentExample(
                question="Find unusual sales patterns",
                primary="anomaly_detection",
                confidence=0.9,
                reasoning="User explicitly asks for unusual patterns",
            )
        ],
    ),
    IntentDefinition(
        label="comparative_analysis",
        description='Compare groups/categories (e.g., "compare by region", "revenue by product")',
        keywords=["compare", "by", "across", "versus", "difference between"],
        examples=[
            IntentExample(
                question="Compare revenue by region and product",
                primary="comparative_analysis",
                confidence=0.95,
                reasoning="Keywords 'compare' and 'by' indicate comparison across categories",
            )
        ],
    ),
    IntentDefinition(
        label="segmentation_drivers",
        description="Explain what factors or segments drive outcomes",
        keywords=["segment", "driver", "drivers", "factor", "why"],
        expose_to_llm=False,
    ),
]


def get_intent_definitions(exposed_only: bool = False) -> List[IntentDefinition]:
    if exposed_only:
        return [intent for intent in INTENT_DEFINITIONS if intent.expose_to_llm]
    return list(INTENT_DEFINITIONS)


def get_intent_labels(exposed_only: bool = False) -> List[str]:
    return [intent.label for intent in get_intent_definitions(exposed_only)]


def render_intent_catalog(exposed_only: bool = True) -> str:
    intents = get_intent_definitions(exposed_only)
    lines = [
        f"{idx}. **{intent.label}** - {intent.description}"
        for idx, intent in enumerate(intents, start=1)
    ]
    return "\n".join(lines)


def render_intent_rules(exposed_only: bool = True) -> str:
    intents = get_intent_definitions(exposed_only)
    lines: List[str] = []
    for intent in intents:
        if intent.keywords:
            keywords = ", ".join(f'"{keyword}"' for keyword in intent.keywords)
            lines.append(f"- {keywords} -> {intent.label}")
    for intent in intents:
        if intent.is_fallback:
            fallback_text = intent.fallback_text or "General questions"
            lines.append(f"- {fallback_text} -> {intent.label}")
    return "\n".join(lines)


def render_intent_examples(exposed_only: bool = True) -> str:
    intents = get_intent_definitions(exposed_only)
    lines: List[str] = []
    for intent in intents:
        for example in intent.examples:
            lines.append(
                'Question: "{question}" -> {response}'.format(
                    question=example.question,
                    response=(
                        f'{{"primary": "{example.primary}", '
                        f'"confidence": {example.confidence:.2f}, '
                        f'"reasoning": "{example.reasoning}"}}'
                    ),
                )
            )
    if not lines:
        return "None."
    return "\n\n".join(lines)


def classify_intent_by_keywords(question: str, exposed_only: bool = True) -> str:
    question_lower = question.lower()
    intents = get_intent_definitions(exposed_only)
    for intent in intents:
        if any(keyword in question_lower for keyword in intent.keywords):
            return intent.label
    for intent in intents:
        if intent.is_fallback:
            return intent.label
    return "general_eda"
