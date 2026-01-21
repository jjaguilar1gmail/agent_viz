"""Schema models for tool validation."""

from typing import Any, Dict

from pydantic import BaseModel, Field, model_validator


class ToolCallRequest(BaseModel):
    """Tool call request with validation."""

    model_config = {"extra": "forbid"}

    tool: str = Field(..., description="Tool name")
    args: Dict[str, Any] = Field(default_factory=dict, description="Tool arguments")
    sequence: int = Field(..., description="Execution sequence number")

    @model_validator(mode="after")
    def validate_tool_exists(self):
        """Validate that the tool exists in registry."""
        from autoviz_agent.registry.tools import TOOL_REGISTRY

        if not TOOL_REGISTRY.get_tool(self.tool):
            raise ValueError(f"Unknown tool: {self.tool}")
        return self


class ToolCallResult(BaseModel):
    """Tool call result."""

    tool: str = Field(..., description="Tool name")
    sequence: int = Field(..., description="Execution sequence number")
    success: bool = Field(..., description="Whether execution succeeded")
    outputs: Dict[str, Any] = Field(default_factory=dict, description="Tool outputs")
    duration_ms: float = Field(..., description="Execution duration in milliseconds")
    warnings: list[str] = Field(default_factory=list, description="Warning messages")
    errors: list[str] = Field(default_factory=list, description="Error messages")
