"""LLM client for intent classification and plan adaptation using gpt4all."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from autoviz_agent.models.state import Intent, IntentLabel, SchemaProfile
from autoviz_agent.utils.logging import get_logger

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
                
            if any(word in question_part for word in ["time", "trend", "temporal", "series", "over time"]):
                return '{"primary": "time_series_investigation", "secondary": ["general_eda"], "confidence": 0.7, "reasoning": "Time-based keywords detected"}'
            elif any(word in question_part for word in ["anomaly", "outlier", "unusual"]):
                return '{"primary": "anomaly_detection", "secondary": ["general_eda"], "confidence": 0.7, "reasoning": "Anomaly keywords detected"}'
            elif any(word in question_part for word in ["segment", "group", "compare", "difference"]):
                return '{"primary": "comparative_analysis", "secondary": ["segmentation_drivers"], "confidence": 0.7, "reasoning": "Comparison keywords detected"}'
            else:
                return '{"primary": "general_eda", "secondary": [], "confidence": 0.6, "reasoning": "General exploration"}'
        
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
        # Build prompt for intent classification
        prompt = self._build_intent_prompt(user_question, schema)
        
        logger.info("Classifying user intent")
        response = self._generate(prompt, max_tokens=200, stop=["\n\n"])
        
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

    def _build_intent_prompt(self, question: str, schema: SchemaProfile) -> str:
        """Build prompt for intent classification."""
        column_info = ', '.join(f"{c.name}({c.dtype})" for c in schema.columns[:5])
        if len(schema.columns) > 5:
            column_info += '...'
        
        temporal_cols = [c.name for c in schema.columns if 'temporal' in c.roles]
        numeric_cols = [c.name for c in schema.columns if c.dtype in ['int64', 'float64']]
        
        return f"""Classify the user's analytical intent based on their question and the dataset structure.

AVAILABLE INTENTS (with templates):
1. general_eda - Broad exploration (e.g., "summarize this data", "what's in here?") [ALWAYS AVAILABLE]
2. time_series_investigation - Temporal patterns (e.g., "trends over time", "seasonal patterns")
3. anomaly_detection - Outliers and unusual values (e.g., "find anomalies", "detect outliers")

IMPORTANT: If the question doesn't clearly match time_series or anomaly_detection, choose general_eda.
For comparison questions like "compare regions" or "segment analysis", use general_eda.

USER QUESTION: "{question}"

DATASET SCHEMA:
- Rows: {schema.row_count}, Columns: {len(schema.columns)}
- Column names: {column_info}
- Temporal columns: {', '.join(temporal_cols) if temporal_cols else 'none'}
- Numeric columns: {len(numeric_cols)}

EXAMPLES:
Question: "Analyze revenue trends over time" → {{"primary": "time_series_investigation", "confidence": 0.95, "reasoning": "Keywords 'trends' and 'over time' indicate temporal analysis"}}
Question: "Find unusual sales patterns" → {{"primary": "anomaly_detection", "confidence": 0.90, "reasoning": "User explicitly asks for 'unusual' patterns"}}

Your response (JSON only, no other text):
{{"primary": "<intent>", "confidence": 0.0-1.0, "reasoning": "<why you chose this intent>"}}

Response:"""

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
        prompt = self._build_adaptation_prompt(template_plan, schema, intent, user_question)
        
        logger.info("Adapting plan template")
        response = self._generate(prompt, max_tokens=400, stop=["Here are", "Here is", "\n\nHere", "Additional"])
        
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

    def _build_adaptation_prompt(
        self, template: Dict[str, Any], schema: SchemaProfile, intent: Intent, question: str
    ) -> str:
        """Build prompt for plan adaptation."""
        template_steps = template.get('steps', [])
        step_summary = "\n".join([f"  - {s.get('step_id')}: {s.get('tool')} - {s.get('description', '')}" for s in template_steps[:5]])
        
        temporal_cols = [c.name for c in schema.columns if 'temporal' in c.roles]
        
        return f"""Review this analysis plan template and suggest modifications based on the user's specific question.

USER QUESTION: "{question}"
INTENT: {intent.label}

TEMPLATE: {template.get('template_id')} ({len(template_steps)} steps)
Current steps:
{step_summary}

DATASET CONTEXT:
- Rows: {schema.row_count}, Columns: {len(schema.columns)}
- Temporal columns: {', '.join(temporal_cols) if temporal_cols else 'none'}
- Shape: {schema.data_shape}

YOUR TASK:
1. Check if the user question mentions specific requirements not in the template
2. Look for keywords like "anomaly", "outlier", "unusual", "compare", "segment", "correlate"
3. Suggest adding, removing, or modifying steps to better match the question

EXAMPLES:
- If user asks for "outliers" but template lacks anomaly detection → add detect_anomalies step
- If user asks for "trends" and template has histogram → change to line plot
- If template fits perfectly → return empty changes array

RESPONSE FORMAT (JSON only, no preamble):
{{"changes": [{{"action": "add|remove|modify", "step_id": "<id>", "tool": "<tool_name>", "description": "<reason>", "params": {{"df": "$dataframe"}}}}], "rationale": "<overall explanation of changes or 'Template fits user question well'>"}}

IMPORTANT: When action is "add", you MUST provide:
- step_id: unique identifier (e.g., "detect_outliers")
- tool: exact tool name (e.g., "detect_anomalies", "plot_line", "compute_summary_stats")
- description: what this step does
- params: at minimum {{"df": "$dataframe"}}

Your JSON response:"""

    def _apply_adaptations(self, template: Dict[str, Any], changes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply adaptation changes to template."""
        import copy
        adapted = copy.deepcopy(template)
        
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
                # Add new step from change specification
                new_step = {
                    "step_id": step_id,
                    "tool": change.get("tool", "unknown"),
                    "description": change.get("description", change.get("reason", "")),
                    "params": change.get("params", {"df": "$dataframe"})
                }
                # Insert at the end or at a specific position if provided
                insert_position = change.get("position", len(adapted.get("steps", [])))
                adapted.setdefault("steps", []).insert(insert_position, new_step)
        
        return adapted

    def generate_tool_calls(
        self, plan: Dict[str, Any], schema: SchemaProfile, artifact_manager=None
    ) -> List[Dict[str, Any]]:
        """
        Generate tool calls from plan, filling in schema-derived parameters.

        Args:
            plan: Execution plan
            schema: Dataset schema
            artifact_manager: Artifact manager for generating output paths

        Returns:
            List of tool call specifications
        """
        from pathlib import Path
        
        tool_calls = []
        
        # Get temporal and numeric columns from schema
        temporal_cols = [c.name for c in schema.columns if 'temporal' in c.roles]
        numeric_cols = [c.name for c in schema.columns if c.dtype in ['int64', 'float64', 'int32', 'float32']]
        
        for idx, step in enumerate(plan.get("steps", []), start=1):
            tool_name = step.get("tool")
            params = step.get("params", {}).copy()
            
            # Auto-fill missing parameters based on tool type and schema
            if tool_name == "parse_datetime" and "columns" not in params:
                # Use first temporal column or try to detect date column
                if temporal_cols:
                    params["columns"] = temporal_cols
                else:
                    date_cols = [c.name for c in schema.columns if 'date' in c.name.lower() or 'time' in c.name.lower()]
                    if date_cols:
                        params["columns"] = date_cols
                # Remove auto_detect if present (not a valid parameter)
                params.pop("auto_detect", None)
            
            elif tool_name == "plot_line":
                # Need x, y, and output_path
                if "x" not in params and temporal_cols:
                    params["x"] = temporal_cols[0]
                if "y" not in params and numeric_cols:
                    params["y"] = numeric_cols[0]
                if "output_path" not in params:
                    if artifact_manager:
                        params["output_path"] = str(artifact_manager.get_path("chart", f"line_plot_{idx}.png"))
                    else:
                        params["output_path"] = f"line_plot_{idx}.png"
                # Remove show_trend if present (not a valid parameter)
                params.pop("show_trend", None)
            
            elif tool_name == "plot_histogram":
                # Need column and output_path
                if "column" not in params and numeric_cols:
                    params["column"] = numeric_cols[0]
                if "output_path" not in params:
                    if artifact_manager:
                        params["output_path"] = str(artifact_manager.get_path("chart", f"histogram_{idx}.png"))
                    else:
                        params["output_path"] = f"histogram_{idx}.png"
                # Remove max_columns if present (not a valid parameter)
                params.pop("max_columns", None)
            
            elif tool_name == "plot_bar":
                # Need x, y, and output_path
                if "output_path" not in params:
                    if artifact_manager:
                        params["output_path"] = str(artifact_manager.get_path("chart", f"bar_plot_{idx}.png"))
                    else:
                        params["output_path"] = f"bar_plot_{idx}.png"
            
            elif tool_name == "plot_scatter":
                # Need x, y, and output_path
                if "x" not in params and numeric_cols:
                    params["x"] = numeric_cols[0] if len(numeric_cols) > 0 else "index"
                if "y" not in params and numeric_cols:
                    params["y"] = numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0]
                if "output_path" not in params:
                    if artifact_manager:
                        params["output_path"] = str(artifact_manager.get_path("chart", f"scatter_{idx}.png"))
                    else:
                        params["output_path"] = f"scatter_{idx}.png"
                # Remove highlight_anomalies if present (not a valid parameter)
                params.pop("highlight_anomalies", None)
            
            elif tool_name == "plot_boxplot":
                # Need column and output_path
                if "column" not in params and numeric_cols:
                    params["column"] = numeric_cols[0]
                if "output_path" not in params:
                    if artifact_manager:
                        params["output_path"] = str(artifact_manager.get_path("chart", f"boxplot_{idx}.png"))
                    else:
                        params["output_path"] = f"boxplot_{idx}.png"
            
            elif tool_name == "plot_heatmap":
                # Need data (DataFrame) and output_path
                # data comes from df parameter, need to rename it
                if "df" in params:
                    params["data"] = params.pop("df")
                if "output_path" not in params:
                    if artifact_manager:
                        params["output_path"] = str(artifact_manager.get_path("chart", f"heatmap_{idx}.png"))
                    else:
                        params["output_path"] = f"heatmap_{idx}.png"
                # Rename annotation to annot
                if "annotation" in params:
                    params["annot"] = params.pop("annotation")
                # Add select_numeric flag to filter only numeric columns
                if "select_numeric" not in params:
                    params["select_numeric"] = True
            
            elif tool_name == "compute_distributions":
                # Need column
                if "column" not in params and numeric_cols:
                    params["column"] = numeric_cols[0]
            
            elif tool_name == "detect_anomalies":
                # Need column
                if "column" not in params and numeric_cols:
                    params["column"] = numeric_cols[0]
            
            tool_call = {
                "sequence": idx,
                "step_id": step.get("step_id"),
                "tool": tool_name,
                "args": params,
                "description": step.get("description", ""),
            }
            tool_calls.append(tool_call)
        
        logger.info(f"Generated {len(tool_calls)} tool calls from plan")
        return tool_calls
