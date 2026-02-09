"""vLLM client with OpenAI-compatible API and xgrammar2 support."""

import json
import math
import re
import requests
from pathlib import Path
from typing import Any, Dict, List, Optional

from autoviz_agent.models.state import Intent, IntentLabel, SchemaProfile
from autoviz_agent.utils.logging import get_logger
from autoviz_agent.llm.prompts import PromptBuilder
from autoviz_agent.llm.llm_contracts import (
    get_intent_schema,
    get_adaptation_schema,
    get_requirement_extraction_schema,
    validate_intent_output,
    validate_adaptation_output,
    validate_requirement_extraction_output,
    validate_requirement_columns,
    RequirementExtractionOutput,
)
from autoviz_agent.registry.tools import TOOL_REGISTRY, ensure_default_tools_registered

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
        self.max_context_tokens = (
            model_config.get("max_context_tokens")
            or model_config.get("max_context_length")
            or model_config.get("context_length")
        )
        self.use_grammar = model_config.get("use_grammar", True)
        self.last_prompt: Optional[str] = None
        self.last_response: Optional[str] = None
        
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
        json_schema: Optional[Dict[str, Any]] = None,
        _retry_on_validation: bool = True,
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
        max_tokens = self._cap_max_tokens(prompt, max_tokens)

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
                timeout=60,
            )
            if response.status_code == 400 and _retry_on_validation:
                capped = self._cap_max_tokens_from_response(response, max_tokens)
                if capped != max_tokens:
                    return self._generate(
                        prompt,
                        max_tokens=capped,
                        json_schema=json_schema,
                        _retry_on_validation=False,
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

    def _estimate_prompt_tokens(self, prompt: str) -> int:
        # Heuristic for token estimation without tokenizer dependency.
        return max(1, math.ceil(len(prompt) / 4) + 16)

    def _cap_max_tokens(self, prompt: str, requested_tokens: int) -> int:
        if not self.max_context_tokens:
            return requested_tokens
        prompt_tokens = self._estimate_prompt_tokens(prompt)
        available = max(1, self.max_context_tokens - prompt_tokens)
        if requested_tokens > available:
            logger.warning(
                "Capping max_tokens from %s to %s based on estimated prompt tokens (%s)",
                requested_tokens,
                available,
                prompt_tokens,
            )
            return available
        return requested_tokens

    def _cap_max_tokens_from_response(
        self, response: requests.Response, requested_tokens: int
    ) -> int:
        try:
            payload = response.json()
            message = payload.get("error", {}).get("message", "")
        except ValueError:
            message = response.text or ""
        match = re.search(
            r"maximum context length is (\d+) tokens.*?request has (\d+) input tokens",
            message,
        )
        if not match:
            return requested_tokens
        max_context = int(match.group(1))
        input_tokens = int(match.group(2))
        available = max(1, max_context - input_tokens)
        if requested_tokens > available:
            logger.warning(
                "Capping max_tokens from %s to %s based on vLLM validation error",
                requested_tokens,
                available,
            )
            return available
        return requested_tokens

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
        self.last_prompt = prompt
        
        # Get JSON schema for grammar constraint
        intent_schema = get_intent_schema()
        
        logger.info("Classifying user intent with vLLM")
        response = self._generate(prompt, max_tokens=200, json_schema=intent_schema)
        self.last_response = response
        
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

    def extract_requirements(
        self, user_question: str, schema: SchemaProfile
    ) -> RequirementExtractionOutput:
        """
        Extract structured requirements from user question.

        Args:
            user_question: User's analytical question
            schema: Inferred schema profile

        Returns:
            RequirementExtractionOutput with structured requirements
        """
        numeric_cols = [c.name for c in schema.columns if c.dtype in ['int64', 'float64', 'float', 'int']]
        categorical_cols = [c.name for c in schema.columns if c.dtype in ['object', 'string', 'category']]
        temporal_cols = [c.name for c in schema.columns if 'temporal' in c.roles]

        # Build prompt for requirement extraction using PromptBuilder
        prompt = self.prompt_builder.build_requirement_extraction_prompt(user_question, schema)
        self.last_prompt = prompt
        
        # Get JSON schema for grammar constraint with dynamic enums
        requirements_schema = get_requirement_extraction_schema(
            numeric_columns=numeric_cols,
            categorical_columns=categorical_cols,
            temporal_columns=temporal_cols,
        )
        
        logger.info("Extracting requirements from user question with vLLM")
        try:
            response = self._generate(prompt, max_tokens=300, json_schema=requirements_schema)
            self.last_response = response
        except Exception as e:
            logger.error(f"Failed to generate requirement extraction response: {e}")
            self.last_response = f"ERROR: {e}"
            return RequirementExtractionOutput()
        
        # Parse and validate response
        try:
            result = json.loads(response)
            validated = validate_requirement_extraction_output(result)

            is_valid, errors = validate_requirement_columns(validated, schema)
            if not is_valid:
                validation_errors = "Invalid columns:\n- " + "\n- ".join(errors)
                retry_prompt = self.prompt_builder.build_requirement_extraction_prompt(
                    user_question, schema, validation_errors=validation_errors
                )
                self.last_prompt = retry_prompt
                response = self._generate(
                    retry_prompt,
                    max_tokens=300,
                    json_schema=requirements_schema,
                )
                self.last_response = response
                result = json.loads(response)
                validated = validate_requirement_extraction_output(result)

            logger.info(
                f"Extracted requirements: metrics={validated.metrics}, "
                f"group_by={validated.group_by}, analysis={validated.analysis}"
            )
            return validated
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse requirement extraction response as JSON: {e}")
            logger.warning(f"Raw response: {response[:500]}")
            # Return empty requirements on failure
            return RequirementExtractionOutput()
        except Exception as e:
            logger.warning(f"Failed to validate requirement extraction response: {e}")
            logger.warning(f"Parsed data: {result if 'result' in locals() else 'N/A'}")
            # Return empty requirements on failure
            return RequirementExtractionOutput()

    def adapt_plan(
        self,
        template_plan: Dict[str, Any],
        schema: SchemaProfile,
        intent: Intent,
        user_question: str,
        narrowed_tools: Optional[List[str]] = None,
        requirements: Optional[Any] = None,
        validation_errors: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Adapt plan template based on schema and intent.

        Args:
            template_plan: Original plan template
            schema: Inferred schema
            intent: Classified intent
            user_question: User's question
            narrowed_tools: Optional narrowed tool list from retrieval

        Returns:
            Adapted plan with modifications
        """
        # Build prompt for plan adaptation using PromptBuilder
        prompt = self.prompt_builder.build_adaptation_prompt(
            template_plan,
            schema,
            intent,
            user_question,
            narrowed_tools,
            requirements,
            validation_errors,
        )
        self.last_prompt = prompt
        
        allowed_columns = [c.name for c in schema.columns]
        numeric_cols = [c.name for c in schema.columns if c.dtype in ['int64', 'float64', 'float', 'int']]
        categorical_cols = [c.name for c in schema.columns if c.dtype in ['object', 'string', 'category']]
        temporal_cols = [c.name for c in schema.columns if 'temporal' in c.roles]

        # Get JSON schema for grammar constraint
        adaptation_schema = get_adaptation_schema(
            allowed_columns=allowed_columns,
            numeric_columns=numeric_cols,
            categorical_columns=categorical_cols,
            temporal_columns=temporal_cols,
        )
        
        logger.info("Adapting plan template with vLLM")
        response = self._generate(prompt, max_tokens=400, json_schema=adaptation_schema)
        self.last_response = response
        
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
        ensure_default_tools_registered()
        valid_tools = set(TOOL_REGISTRY.list_tools())
        
        for change in changes:
            action = change.get("action")
            step_id = change.get("step_id")
            existing_step = next(
                (s for s in adapted.get("steps", []) if s.get("step_id") == step_id),
                None,
            )
            
            if action == "remove":
                adapted["steps"] = [s for s in adapted.get("steps", []) if s.get("step_id") != step_id]
            elif action == "modify":
                for step in adapted.get("steps", []):
                    if step.get("step_id") == step_id:
                        step["params"].update(change.get("params", {}))
                        if change.get("description"):
                            step["description"] = change.get("description")
                        if change.get("tool"):
                            step["tool"] = change.get("tool")
            elif action == "add":
                if existing_step is not None:
                    logger.warning(
                        "Step_id '%s' already exists; converting add -> modify",
                        step_id,
                    )
                    existing_step["params"].update(change.get("params", {}))
                    if change.get("description"):
                        existing_step["description"] = change.get("description")
                    if change.get("tool"):
                        existing_step["tool"] = change.get("tool")
                    continue
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
