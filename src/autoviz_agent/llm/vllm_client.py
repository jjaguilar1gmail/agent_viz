"""vLLM client with OpenAI-compatible API and xgrammar2 support."""

import json
import requests
from pathlib import Path
from typing import Any, Dict, List, Optional

from autoviz_agent.models.state import Intent, IntentLabel, SchemaProfile
from autoviz_agent.utils.logging import get_logger
from autoviz_agent.llm.prompts import PromptBuilder
from autoviz_agent.llm.llm_contracts import (
    get_intent_schema,
    get_adaptation_schema,
    validate_intent_output,
    validate_adaptation_output,
)

logger = get_logger(__name__)


class VLLMClient:
    """Client for vLLM server with OpenAI-compatible API."""

    def __init__(self, model_config: Dict[str, Any]):
        """
        Initialize vLLM client.

        Args:
            model_config: Model configuration from config.yaml
                Expected keys:
                - backend: "vllm"
                - url: "http://localhost:8000" (vLLM server endpoint)
                - model_name: Model identifier (optional, for tracking)
                - temperature: Generation temperature (default: 0.1)
                - max_tokens: Max tokens to generate (default: 512)
                - use_grammar: Whether to use xgrammar2 (default: True)
        """
        self.model_config = model_config
        self.base_url = model_config.get("url", "http://localhost:8000")
        self.model_name = model_config.get("model_name", "unknown")
        self.temperature = model_config.get("temperature", 0.1)
        self.max_tokens = model_config.get("max_tokens", 512)
        self.use_grammar = model_config.get("use_grammar", True)
        
        # Initialize prompt builder with optional template directory
        templates_dir = Path(__file__).parent.parent.parent.parent / "templates" / "prompts"
        self.prompt_builder = PromptBuilder(template_dir=templates_dir if templates_dir.exists() else None)
        
        # Verify server is accessible
        self._verify_connection()

    def _verify_connection(self) -> None:
        """Verify vLLM server is accessible."""
        try:
            response = requests.get(f"{self.base_url}/v1/models", timeout=5)
            response.raise_for_status()
            models = response.json()
            logger.info(f"Connected to vLLM server at {self.base_url}")
            logger.debug(f"Available models: {models}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to vLLM server at {self.base_url}: {e}")
            logger.warning("vLLM client will fail on inference. Make sure vLLM server is running.")

    def _generate(
        self, 
        prompt: str, 
        max_tokens: int = 512,
        json_schema: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate text from prompt using vLLM server.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            json_schema: Optional JSON schema for grammar-constrained generation

        Returns:
            Generated text

        Raises:
            requests.exceptions.RequestException: If request fails
        """
        # Build request payload
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature,
            "max_tokens": max_tokens,
        }
        
        # Add grammar constraint if schema provided and grammar is enabled
        if json_schema and self.use_grammar:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response_schema",
                    "schema": json_schema,
                    "strict": True
                }
            }
            logger.debug("Using grammar-constrained generation with xgrammar2")
        
        # Make request to vLLM server
        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            if "choices" not in result or len(result["choices"]) == 0:
                raise ValueError("No choices in vLLM response")
            
            content = result["choices"][0]["message"]["content"]
            logger.debug(f"vLLM response: {content[:200]}...")
            
            return content
            
        except requests.exceptions.Timeout:
            logger.error("vLLM request timed out")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"vLLM request failed: {e}")
            raise
        except (KeyError, ValueError) as e:
            logger.error(f"Failed to parse vLLM response: {e}")
            raise

    def classify_intent(
        self, user_question: str, schema: SchemaProfile, max_intents: int = 3
    ) -> Intent:
        """
        Classify user intent from question and schema.

        Args:
            user_question: User's analytical question
            schema: Inferred schema profile
            max_intents: Maximum number of intents to return (unused for now)

        Returns:
            Intent classification
        """
        # Build prompt for intent classification using PromptBuilder
        prompt = self.prompt_builder.build_intent_prompt(user_question, schema)
        
        # Get JSON schema for grammar constraint
        intent_schema = get_intent_schema()
        
        logger.info("Classifying user intent with vLLM")
        response = self._generate(prompt, max_tokens=200, json_schema=intent_schema)
        
        # Parse and validate response
        try:
            result = json.loads(response)
            validated = validate_intent_output(result)
            
            primary = IntentLabel(validated.primary)
            confidence = validated.confidence
            reasoning = validated.reasoning
            
            intent = Intent(
                label=primary,
                confidence=float(confidence),
                top_intents=[{primary.value: float(confidence)}],
            )
            logger.info(f"Classified intent: {intent.label} (confidence={confidence:.2f})")
            if reasoning:
                logger.info(f"Reasoning: {reasoning}")
            return intent
            
        except Exception as e:
            logger.warning(f"Failed to parse intent response: {e}. Using fallback.")
            # Fallback to general_eda
            return Intent(
                label=IntentLabel.GENERAL_EDA,
                confidence=0.5,
                top_intents=[{"general_eda": 0.5}],
            )

    def adapt_plan(
        self,
        template_plan: Dict[str, Any],
        schema: SchemaProfile,
        intent: Intent,
        user_question: str,
    ) -> Dict[str, Any]:
        """
        Adapt plan template based on schema and intent.

        Args:
            template_plan: Original plan template
            schema: Inferred schema
            intent: Classified intent
            user_question: User's question

        Returns:
            Adapted plan with modifications
        """
        # Build prompt for plan adaptation using PromptBuilder
        prompt = self.prompt_builder.build_adaptation_prompt(template_plan, schema, intent, user_question)
        
        # Get JSON schema for grammar constraint
        adaptation_schema = get_adaptation_schema()
        
        logger.info("Adapting plan template with vLLM")
        response = self._generate(prompt, max_tokens=400, json_schema=adaptation_schema)
        
        # Parse and validate response
        try:
            result = json.loads(response)
            validated = validate_adaptation_output(result)
            
            changes = [change.dict(exclude_none=True) for change in validated.changes]
            rationale = validated.rationale
            
            logger.info(f"vLLM suggested {len(changes)} changes: {rationale}")
            
            # Apply changes to template (reuse existing logic)
            adapted = self._apply_adaptations(template_plan, changes)
            adapted["adaptation_rationale"] = rationale
            adapted["changes_applied"] = len(changes)
            
            logger.info(f"Applied {len(changes)} adaptations to plan")
            return adapted
            
        except Exception as e:
            logger.warning(f"Failed to parse adaptation response: {e}. Using template as-is.")
            return {**template_plan, "adaptation_rationale": "No adaptation applied", "changes_applied": 0}

    def _apply_adaptations(self, template: Dict[str, Any], changes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Apply adaptation changes to template.
        
        This is shared logic with the gpt4all client - could be extracted to a common module.
        """
        import copy
        adapted = copy.deepcopy(template)
        
        # Valid tool names for validation
        valid_tools = [
            "load_dataset", "sample_rows", "save_dataframe",
            "infer_schema",
            "handle_missing", "parse_datetime", "cast_types", "normalize_column_names",
            "compute_summary_stats", "compute_correlations", "compute_value_counts", 
            "compute_percentiles", "aggregate",
            "detect_anomalies", "segment_metric", "compute_distributions", 
            "compare_groups", "compute_time_series_features",
            "plot_line", "plot_bar", "plot_scatter", "plot_histogram", 
            "plot_heatmap", "plot_boxplot"
        ]
        
        for change in changes:
            action = change.get("action")
            step_id = change.get("step_id")
            
            if action == "remove":
                adapted["steps"] = [s for s in adapted.get("steps", []) if s.get("step_id") != step_id]
            elif action == "modify":
                for step in adapted.get("steps", []):
                    if step.get("step_id") == step_id:
                        step["params"].update(change.get("params", {}))
            elif action == "add":
                # Get tool name and validate/correct it
                tool_name = change.get("tool", "unknown")
                
                # Fix common LLM mistakes in tool names
                if tool_name not in valid_tools:
                    # Try to extract valid tool name from description
                    if "group" in tool_name.lower() or "compare" in tool_name.lower():
                        tool_name = "aggregate"
                        # Add group_by parameter if not present
                        if "params" not in change or "group_by" not in change.get("params", {}):
                            if "params" not in change:
                                change["params"] = {"df": "$dataframe"}
                            change["params"]["group_by"] = ["auto"]  # Will be filled by generate_tool_calls
                    elif "summary" in tool_name.lower() or "stats" in tool_name.lower():
                        tool_name = "compute_summary_stats"
                    elif "anomal" in tool_name.lower() or "outlier" in tool_name.lower():
                        tool_name = "detect_anomalies"
                    else:
                        logger.warning(f"Invalid tool name '{tool_name}', using 'unknown'")
                        tool_name = "unknown"
                
                # Add new step from change specification
                new_step = {
                    "step_id": step_id,
                    "tool": tool_name,
                    "description": change.get("description", change.get("reason", "")),
                    "params": change.get("params", {"df": "$dataframe"})
                }
                # Insert at the end or at a specific position if provided
                insert_position = change.get("position", len(adapted.get("steps", [])))
                adapted.setdefault("steps", []).insert(insert_position, new_step)
        
        return adapted

    def generate_tool_calls(
        self,
        plan: Dict[str, Any],
        schema: SchemaProfile,
        artifact_manager=None,
        user_question: str = "",
        execution_log=None,
    ) -> List[Dict[str, Any]]:
        """
        Generate tool calls from plan, using ParamResolver for parameter defaults.

        Args:
            plan: Execution plan
            schema: Dataset schema
            artifact_manager: Artifact manager for generating output paths
            user_question: User's original question for extracting mentioned columns

        Returns:
            List of tool call specifications
        """
        from autoviz_agent.runtime.param_resolver import ParamResolver
        from autoviz_agent.registry.validation import validate_tool_call, repair_tool_call
        
        # Create parameter resolver with user question for column extraction
        resolver = ParamResolver(schema, artifact_manager, user_question)
        
        tool_calls = []
        dropped_calls = []
        
        for idx, step in enumerate(plan.get("steps", []), start=1):
            tool_name = step.get("tool")
            params = step.get("params", {}).copy()
            
            # Resolve parameters using the resolver
            resolved_params = resolver.resolve(tool_name, params, sequence=idx)
            
            tool_call = {
                "sequence": idx,
                "step_id": step.get("step_id"),
                "tool": tool_name,
                "args": resolved_params,
                "description": step.get("description", ""),
            }
            
            repaired_call = repair_tool_call(tool_call, schema_profile=schema)
            if execution_log and repaired_call.get("args") != tool_call.get("args"):
                original_args = tool_call.get("args", {})
                new_args = repaired_call.get("args", {})
                removed = sorted([k for k in original_args.keys() if k not in new_args])
                added = {k: new_args[k] for k in new_args.keys() if k not in original_args}
                changed = {
                    k: {"from": original_args[k], "to": new_args[k]}
                    for k in original_args.keys()
                    if k in new_args and original_args[k] != new_args[k]
                }
                details = {
                    "removed_params": removed,
                    "added_params": added,
                    "changed_params": changed,
                }
                execution_log.add_repair_attempt(
                    tool_name,
                    strategy="tool_call_repair",
                    success=True,
                    details=details,
                )

            validation_result = validate_tool_call(repaired_call)
            if validation_result.is_valid:
                if repaired_call.get("args") != tool_call.get("args"):
                    logger.info(f"Successfully repaired tool call: {tool_name}")
                tool_calls.append(repaired_call)
            else:
                logger.error(f"Cannot repair tool call for {tool_name}, dropping it")
                dropped_calls.append({"tool": tool_name, "errors": validation_result.errors})
        
        if dropped_calls:
            logger.warning(f"Dropped {len(dropped_calls)} invalid tool calls: {[d['tool'] for d in dropped_calls]}")
        
        logger.info(f"Generated {len(tool_calls)} valid tool calls from plan")
        return tool_calls
