"""
Inference module for Bun-Bench.

This module provides functionality for running LLM inference on benchmark tasks,
including support for OpenAI and Anthropic APIs, prompt management, and result processing.
"""

from bunbench.inference.run_api import (
    run_inference,
    OpenAIClient,
    AnthropicClient,
)
from bunbench.inference.prompts import (
    SYSTEM_PROMPT,
    USER_PROMPT,
    format_prompt,
)
from bunbench.inference.utils import (
    extract_patch,
    repair_patch,
    count_tokens,
)

__all__ = [
    "run_inference",
    "OpenAIClient",
    "AnthropicClient",
    "SYSTEM_PROMPT",
    "USER_PROMPT",
    "format_prompt",
    "extract_patch",
    "repair_patch",
    "count_tokens",
]
