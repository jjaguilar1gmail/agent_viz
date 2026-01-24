# Prompt Strategy (Proposed)

## Purpose
Make prompts easy to find, edit, and reason about without digging into core code.

## Why change the current approach
Today prompts are embedded in `src/autoviz_agent/llm/client.py`. This makes:
- High-level review difficult (prompts are mixed with logic).
- Iteration slower (editing code to tweak prompt text).
- Schema alignment harder (prompt and JSON schema are not co-located).

## Goals
- Centralize prompt content in one obvious place.
- Keep prompt structure consistent across steps.
- Enable quick iteration without refactoring logic.

## Proposed structure
### 1) Centralize prompt templates
Option A (code-based):
- Create `src/autoviz_agent/llm/prompts.py`
- Store prompt templates as plain strings or format functions

Option B (file-based):
- Create `templates/prompts/`
- Store prompts as `.md` or `.j2` template files

File-based is best if non-code edits are common.

### 2) Use a single prompt builder interface
Create a helper that takes:
- `prompt_name` (intent, adapt_plan, tool_call)
- `variables` (schema summary, user question, tool list)

This keeps the LLM client focused on:
- selecting a prompt
- sending to backend
- parsing output

### 3) Standardize output formats
Define JSON schemas for:
- Intent output
- Plan adaptation output
- (optional) Tool-call output

Include the schema or a JSON example in the prompt to keep the LLM aligned.
If using xgrammar2, use the schema directly as a decoding constraint.

## Suggested prompt inventory
1) Intent classification prompt
2) Plan adaptation prompt
3) (Optional) Direct tool-call prompt

## Proposed file layout (example)
```
src/autoviz_agent/llm/
  client.py
  prompts.py
templates/
  prompts/
    intent.md
    adapt_plan.md
    tool_calls.md
```

## Migration plan
1) Extract `_build_intent_prompt()` and `_build_adaptation_prompt()` into
   `prompts.py` or template files.
2) Update `LLMClient` to import and use those templates.
3) Add schema definitions (JSON or pydantic) for parsing and xgrammar2.
4) (Optional) add hot-reload for file-based templates during development.

## Benefits
- Prompts become visible, searchable, and editable.
- Easier collaboration and iteration.
- Cleaner separation between prompt content and execution logic.
