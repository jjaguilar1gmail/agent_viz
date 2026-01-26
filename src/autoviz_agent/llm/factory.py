"""LLM client factory for selecting appropriate backend."""

from typing import Any, Dict

from autoviz_agent.utils.logging import get_logger

logger = get_logger(__name__)


def create_llm_client(model_config: Dict[str, Any]):
    """
    Create LLM client based on backend configuration.
    
    Args:
        model_config: Model configuration from config.yaml
            Must include 'backend' key with value 'gpt4all', 'llama.cpp', or 'vllm'
    
    Returns:
        LLMClient instance (either LLMClient for gpt4all/llama.cpp or VLLMClient for vllm)
    
    Raises:
        ValueError: If backend is unsupported
    """
    backend = model_config.get("backend", "llama.cpp")
    
    logger.info(f"Creating LLM client with backend: {backend}")
    
    if backend in ["llama.cpp", "gpt4all"]:
        # Use gpt4all client (supports both llama.cpp and gpt4all)
        from autoviz_agent.llm.client import LLMClient
        return LLMClient(model_config)
    
    elif backend == "vllm":
        # Use vLLM client with OpenAI-compatible API
        from autoviz_agent.llm.vllm_client import VLLMClient
        return VLLMClient(model_config)
    
    else:
        raise ValueError(
            f"Unsupported backend: {backend}. "
            f"Supported backends: llama.cpp, gpt4all, vllm"
        )
