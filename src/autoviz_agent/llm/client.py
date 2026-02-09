"""LLM client for intent classification and plan adaptation using gpt4all."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from autoviz_agent.models.state import Intent, IntentLabel, SchemaProfile
from autoviz_agent.utils.logging import get_logger
from autoviz_agent.llm.prompts import PromptBuilder
from autoviz_agent.llm.llm_contracts import (
    RequirementExtractionOutput,
    validate_requirement_extraction_output,
    validate_requirement_columns,
)
from autoviz_agent.registry.intents import classify_intent_by_keywords
from autoviz_agent.registry.tools import TOOL_REGISTRY, ensure_default_tools_registered

logger = get_logger(__name__)


class LLMClient:
    """Client for LLM operations using gpt4all."""

    def __init__(self, model_config: Dict[str, Any]):
        """
        Initialize LLM client.

        Args:
            model_config: Model configuration from config.yaml
        """
        self.model_config = model_config
        self.model_path = Path(model_config.get("path", ""))
        self._llm = None
        self.last_prompt: Optional[str] = None
        self.last_response: Optional[str] = None
        
        # Initialize prompt builder with optional template directory
        templates_dir = Path(__file__).parent.parent.parent.parent / "templates" / "prompts"
        self.prompt_builder = PromptBuilder(template_dir=templates_dir if templates_dir.exists() else None)
        
        # Check if model file exists
        if not self.model_path.exists():
            logger.warning(
                f"Model file not found: {self.model_path}. "
                f"Download from HuggingFace and place in models/ directory. "
                f"Falling back to keyword-based classification."
            )
            self._use_fallback = True
        else:
            self._use_fallback = False
            self._load_model()

    def _load_model(self) -> None:
        """Load the GGUF model using gpt4all."""
        try:
            from gpt4all import GPT4All
            
            logger.info(f"Loading model: {self.model_config['name']}")
            # gpt4all expects just the filename, not the full path
            self._llm = GPT4All(
                model_name=self.model_path.name,  # Just the filename
                model_path=str(self.model_path.parent),  # Just the directory
                allow_download=False,
            )
            logger.info("Model loaded successfully with gpt4all")
        except ImportError:
            logger.warning("gpt4all not installed. Run: pip install gpt4all")
            self._use_fallback = True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self._use_fallback = True

    def _generate(self, prompt: str, max_tokens: int = 512, stop: Optional[List[str]] = None) -> str:
        """
        Generate text from prompt.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            stop: Stop sequences

        Returns:
            Generated text
        """
        if self._use_fallback or self._llm is None:
            return self._fallback_generate(prompt)
        
        try:
            response = self._llm.generate(
                prompt,
                max_tokens=max_tokens,
                temp=self.model_config.get("temperature", 0.1),
                top_p=self.model_config.get("top_p", 0.9),
            )
            
            # Extract first complete JSON object from LLM response
            response = response.strip()
            
            # Strip everything before the first '{'
            json_start = response.find('{')
            if json_start == -1:
                return response  # No JSON found, return as-is
            
            response = response[json_start:]
            
            # Find the end of the first complete JSON object
            brace_count = 0
            for i, char in enumerate(response):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        return response[:i+1]
            
            return response
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return self._fallback_generate(prompt)

    def _fallback_generate(self, prompt: str) -> str:
        """Keyword-based fallback when LLM unavailable."""
        prompt_lower = prompt.lower()
        
        # Intent classification fallback - check user question specifically
        if "classify" in prompt_lower or "intent" in prompt_lower:
            # Extract just the user question from the prompt
            if "user question:" in prompt_lower:
                question_part = prompt_lower.split("user question:")[1].split("\n")[0]
            else:
                question_part = prompt_lower
                
            primary = classify_intent_by_keywords(question_part, exposed_only=True)
            return json.dumps(
                {
                    "primary": primary,
                    "confidence": 0.7 if primary != "general_eda" else 0.6,
                    "reasoning": "Keyword-based fallback classification",
                }
            )
        
        # Plan adaptation fallback - return minimal changes
        if "adapt" in prompt_lower or "modify" in prompt_lower:
            return '{"changes": [], "rationale": "No adaptation needed - using template as-is (fallback mode)"}'
        
        return "{}"

    def classify_intent(
        self, user_question: str, schema: SchemaProfile, max_intents: int = 3
    ) -> Intent:
        """
        Classify user intent from question and schema.

        Args:
            user_question: User's analytical question
            schema: Inferred schema profile
            max_intents: Maximum number of intents to return

        Returns:
            Intent classification
        """
        # Build prompt for intent classification using PromptBuilder
        prompt = self.prompt_builder.build_intent_prompt(user_question, schema)
        self.last_prompt = prompt
        
        logger.info("Classifying user intent")
        response = self._generate(prompt, max_tokens=200, stop=["\n\n"])
        self.last_response = response
        
        # Parse response
        try:
            result = json.loads(response)
            primary = IntentLabel(result.get("primary", "general_eda"))
            confidence = result.get("confidence", 0.5)
            reasoning = result.get("reasoning", "")
            
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
        # Build prompt for requirement extraction using PromptBuilder
        prompt = self.prompt_builder.build_requirement_extraction_prompt(user_question, schema)
        self.last_prompt = prompt
        
        logger.info("Extracting requirements from user question")
        response = self._generate(prompt, max_tokens=300, stop=["\n\n"])
        self.last_response = response
        
        # Parse response
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
                response = self._generate(retry_prompt, max_tokens=300, stop=["\n\n"])
                self.last_response = response
                result = json.loads(response)
                validated = validate_requirement_extraction_output(result)

            logger.info(
                f"Extracted requirements: metrics={validated.metrics}, "
                f"group_by={validated.group_by}, analysis={validated.analysis}"
            )
            return validated
        except Exception as e:
            logger.warning(f"Failed to parse requirement extraction response: {e}. Using empty requirements.")
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
        
        logger.info("Adapting plan template")
        response = self._generate(prompt, max_tokens=400, stop=["Here are", "Here is", "\n\nHere", "Additional"])
        self.last_response = response
        
        # Debug: log LLM response
        logger.debug(f"LLM adaptation response: {response[:200]}...")
        
        # Parse adaptation instructions
        try:
            result = json.loads(response)
            changes = result.get("changes", [])
            rationale = result.get("rationale", "")
            
            logger.info(f"LLM suggested {len(changes)} changes: {rationale}")
            
            # Apply changes to template
            adapted = self._apply_adaptations(template_plan, changes)
            adapted["adaptation_rationale"] = rationale
            adapted["changes_applied"] = len(changes)
            
            logger.info(f"Applied {len(changes)} adaptations to plan")
            return adapted
        except Exception as e:
            logger.warning(f"Failed to parse adaptation response: {e}. Using template as-is.")
            return {**template_plan, "adaptation_rationale": "No adaptation applied", "changes_applied": 0}

    def _apply_adaptations(self, template: Dict[str, Any], changes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply adaptation changes to template."""
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
