"""Settings and configuration defaults."""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Model configuration."""

    provider: str = Field(default="llama.cpp", description="Model provider")
    model_path: Optional[Path] = Field(default=None, description="Path to model weights")
    context_size: int = Field(default=4096, description="Context window size")
    temperature: float = Field(default=0.0, description="Temperature for generation")
    seed: Optional[int] = Field(default=42, description="Random seed for reproducibility")


class RuntimeConfig(BaseModel):
    """Runtime configuration."""

    output_dir: Path = Field(default=Path("outputs"), description="Output directory")
    templates_dir: Path = Field(
        default=Path("src/autoviz_agent/templates"), description="Templates directory"
    )
    max_execution_time: int = Field(default=300, description="Max execution time in seconds")
    enable_cache: bool = Field(default=True, description="Enable caching")


class Settings(BaseModel):
    """Application settings."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    log_level: str = Field(default="INFO", description="Logging level")


# Default settings instance
DEFAULT_SETTINGS = Settings()
