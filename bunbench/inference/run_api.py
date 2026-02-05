"""
API inference runner for Bun-Bench.

This module provides clients for running inference with OpenAI and Anthropic APIs,
including retry logic, token counting, cost tracking, and resumable execution.
"""

import json
import os
import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Iterator
import argparse

from bunbench.inference.prompts import format_for_openai, format_for_anthropic
from bunbench.inference.utils import (
    extract_patch,
    repair_patch,
    count_tokens,
    estimate_cost,
)

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """Result from a single inference call."""

    instance_id: str
    model: str
    raw_response: str
    extracted_patch: Optional[str]
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_seconds: float
    timestamp: str
    success: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class InferenceStats:
    """Aggregated statistics for an inference run."""

    total_instances: int = 0
    successful: int = 0
    failed: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    total_latency_seconds: float = 0.0
    patches_extracted: int = 0

    def update(self, result: InferenceResult):
        """Update stats with a new result."""
        self.total_instances += 1
        if result.success:
            self.successful += 1
        else:
            self.failed += 1
        self.total_input_tokens += result.input_tokens
        self.total_output_tokens += result.output_tokens
        self.total_cost_usd += result.cost_usd
        self.total_latency_seconds += result.latency_seconds
        if result.extracted_patch:
            self.patches_extracted += 1

    def __str__(self) -> str:
        return (
            f"InferenceStats(\n"
            f"  total={self.total_instances}, success={self.successful}, failed={self.failed}\n"
            f"  tokens: in={self.total_input_tokens}, out={self.total_output_tokens}\n"
            f"  cost=${self.total_cost_usd:.4f}, latency={self.total_latency_seconds:.1f}s\n"
            f"  patches_extracted={self.patches_extracted}\n"
            f")"
        )


class APIClient(ABC):
    """Abstract base class for API clients."""

    def __init__(
        self,
        model: str,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: float = 120.0,
    ):
        """
        Initialize the API client.

        Args:
            model: Model identifier to use.
            max_retries: Maximum number of retry attempts.
            retry_delay: Initial delay between retries (exponential backoff).
            timeout: Request timeout in seconds.
        """
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout

    @abstractmethod
    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> Dict[str, Any]:
        """
        Make an API call.

        Args:
            system_prompt: System message content.
            user_prompt: User message content.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            Dictionary with 'content', 'input_tokens', 'output_tokens'.
        """
        pass

    def call_with_retry(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> Dict[str, Any]:
        """
        Make an API call with retry logic.

        Args:
            system_prompt: System message content.
            user_prompt: User message content.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            Dictionary with 'content', 'input_tokens', 'output_tokens'.

        Raises:
            Exception: If all retries fail.
        """
        last_error = None
        delay = self.retry_delay

        for attempt in range(self.max_retries):
            try:
                return self.call(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except Exception as e:
                last_error = e
                logger.warning(
                    f"API call failed (attempt {attempt + 1}/{self.max_retries}): {e}"
                )

                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff

        raise last_error


class OpenAIClient(APIClient):
    """OpenAI API client."""

    def __init__(
        self,
        model: str = "gpt-4-turbo",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize OpenAI client.

        Args:
            model: Model name (e.g., 'gpt-4', 'gpt-4-turbo', 'gpt-3.5-turbo').
            api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
            base_url: Optional custom base URL for API-compatible endpoints.
            **kwargs: Additional arguments passed to parent.
        """
        super().__init__(model=model, **kwargs)

        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not provided. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.base_url = base_url
        logger.info(f"Initialized OpenAI client with model='{model}', base_url='{base_url}'")

        # Lazy import to avoid dependency issues
        try:
            from openai import OpenAI

            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        except ImportError:
            raise ImportError(
                "openai package not installed. Install with: pip install openai"
            )

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> Dict[str, Any]:
        """Make an OpenAI API call."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=self.timeout,
        )

        return {
            "content": response.choices[0].message.content,
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        }


class AnthropicClient(APIClient):
    """Anthropic API client."""

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20261022",
        api_key: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize Anthropic client.

        Args:
            model: Model name (e.g., 'claude-3-opus', 'claude-3-sonnet').
            api_key: Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.
            **kwargs: Additional arguments passed to parent.
        """
        super().__init__(model=model, **kwargs)

        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key not provided. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )

        # Lazy import to avoid dependency issues
        try:
            from anthropic import Anthropic

            self.client = Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError(
                "anthropic package not installed. Install with: pip install anthropic"
            )

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> Dict[str, Any]:
        """Make an Anthropic API call."""
        response = self.client.messages.create(
            model=self.model,
            system=system_prompt if system_prompt else "",
            messages=[{"role": "user", "content": user_prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return {
            "content": response.content[0].text,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }


def load_dataset(path: str) -> List[Dict[str, Any]]:
    """
    Load benchmark dataset from JSON file.

    Args:
        path: Path to JSON file containing benchmark instances.

    Returns:
        List of instance dictionaries.

    Raises:
        FileNotFoundError: If file doesn't exist.
        json.JSONDecodeError: If file is invalid JSON.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle both list format and dict with "instances" key
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "instances" in data:
        return data["instances"]
    else:
        raise ValueError(
            f"Invalid dataset format. Expected list or dict with 'instances' key."
        )


def load_processed_ids(output_path: str) -> set:
    """
    Load IDs of already processed instances from output file.

    Args:
        output_path: Path to JSONL output file.

    Returns:
        Set of instance IDs that have been processed.
    """
    processed = set()
    output_path = Path(output_path)

    if not output_path.exists():
        return processed

    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    result = json.loads(line)
                    if "instance_id" in result:
                        processed.add(result["instance_id"])
                except json.JSONDecodeError:
                    continue

    logger.info(f"Found {len(processed)} already processed instances")
    return processed


def save_result(result: InferenceResult, output_path: str):
    """
    Append result to JSONL output file.

    Args:
        result: Inference result to save.
        output_path: Path to output file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result.to_dict()) + "\n")


def run_inference(
    dataset_path: str,
    output_path: str,
    provider: str = "openai",
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    prompt_style: str = "default",
    resume: bool = True,
    max_instances: Optional[int] = None,
    instance_ids: Optional[List[str]] = None,
    repair_patches: bool = True,
    verbose: bool = False,
) -> InferenceStats:
    """
    Run inference on benchmark dataset.

    Args:
        dataset_path: Path to JSON dataset file.
        output_path: Path to JSONL output file.
        provider: API provider ('openai' or 'anthropic').
        model: Model name (uses provider default if not specified).
        base_url: Optional custom base URL for OpenAI-compatible APIs.
        api_key: Optional API key (falls back to environment variables).
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
        prompt_style: Prompt formatting style.
        resume: Skip already processed instances.
        max_instances: Maximum number of instances to process.
        instance_ids: Specific instance IDs to process (None = all).
        repair_patches: Attempt to repair extracted patches.
        verbose: Enable verbose logging.

    Returns:
        InferenceStats with run statistics.
    """
    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Initialize client
    if provider.lower() == "openai":
        default_model = "gpt-4-turbo"
        client = OpenAIClient(
            model=model or default_model,
            base_url=base_url,
            api_key=api_key
        )
    elif provider.lower() == "anthropic":
        default_model = "claude-3-5-sonnet-20261022"
        client = AnthropicClient(model=model or default_model, api_key=api_key)
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'openai' or 'anthropic'.")

    # Load dataset
    logger.info(f"Loading dataset from {dataset_path}")
    instances = load_dataset(dataset_path)
    logger.info(f"Loaded {len(instances)} instances")

    # Filter by instance IDs if specified
    if instance_ids:
        instance_set = set(instance_ids)
        instances = [i for i in instances if i.get("instance_id") in instance_set]
        logger.info(f"Filtered to {len(instances)} specified instances")

    # Load already processed for resumption
    processed_ids = set()
    if resume:
        processed_ids = load_processed_ids(output_path)

    # Track statistics
    stats = InferenceStats()

    # Process instances
    processed_count = 0
    for instance in instances:
        instance_id = instance.get("instance_id", "unknown")

        # Skip if already processed
        if instance_id in processed_ids:
            logger.debug(f"Skipping already processed: {instance_id}")
            continue

        # Check max instances limit
        if max_instances and processed_count >= max_instances:
            logger.info(f"Reached max instances limit: {max_instances}")
            break

        logger.info(f"Processing instance: {instance_id}")

        # Build prompts
        problem_statement = instance.get("problem_statement", "")
        code_context = instance.get("code_context", "")

        if provider.lower() == "openai":
            prompts = format_for_openai(
                problem_statement=problem_statement,
                code_context=code_context,
                style=prompt_style,
            )
            system_prompt = ""
            user_prompt = ""
            for msg in prompts:
                if msg["role"] == "system":
                    system_prompt = msg["content"]
                elif msg["role"] == "user":
                    user_prompt = msg["content"]
        else:
            prompt_data = format_for_anthropic(
                problem_statement=problem_statement,
                code_context=code_context,
                style=prompt_style,
            )
            system_prompt = prompt_data["system"]
            user_prompt = prompt_data["messages"][0]["content"]

        # Make API call
        start_time = time.time()
        error_msg = None
        success = True

        try:
            response = client.call_with_retry(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            raw_response = response["content"]
            input_tokens = response["input_tokens"]
            output_tokens = response["output_tokens"]

        except Exception as e:
            logger.error(f"Failed to process {instance_id}: {e}")
            raw_response = ""
            input_tokens = count_tokens(system_prompt + user_prompt, client.model)
            output_tokens = 0
            error_msg = str(e)
            success = False

        latency = time.time() - start_time

        # Extract and optionally repair patch
        extracted_patch = None
        if success and raw_response:
            extracted_patch = extract_patch(raw_response)
            if extracted_patch and repair_patches:
                extracted_patch = repair_patch(extracted_patch)

        # Calculate cost
        cost = estimate_cost(input_tokens, output_tokens, client.model)

        # Create result
        result = InferenceResult(
            instance_id=instance_id,
            model=client.model,
            raw_response=raw_response,
            extracted_patch=extracted_patch,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            latency_seconds=latency,
            timestamp=datetime.utcnow().isoformat(),
            success=success,
            error=error_msg,
            metadata={
                "temperature": temperature,
                "max_tokens": max_tokens,
                "prompt_style": prompt_style,
            },
        )

        # Save result
        save_result(result, output_path)

        # Update stats
        stats.update(result)
        processed_count += 1

        # Log progress
        logger.info(
            f"  Completed: tokens={input_tokens}+{output_tokens}, "
            f"cost=${cost:.4f}, patch={'yes' if extracted_patch else 'no'}"
        )

    logger.info(f"\nInference complete!\n{stats}")
    return stats


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run LLM inference on Bun-Bench dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with OpenAI GPT-4
  python -m bunbench.inference.run_api --dataset data/bun-bench.json --output results/gpt4.jsonl --provider openai --model gpt-4-turbo

  # Run with Anthropic Claude
  python -m bunbench.inference.run_api --dataset data/bun-bench.json --output results/claude.jsonl --provider anthropic --model claude-3-5-sonnet-20261022

  # Process specific instances
  python -m bunbench.inference.run_api --dataset data/bun-bench.json --output results/test.jsonl --instances BUN-001 BUN-002

  # Resume interrupted run
  python -m bunbench.inference.run_api --dataset data/bun-bench.json --output results/gpt4.jsonl --resume
        """,
    )

    parser.add_argument(
        "--dataset",
        "-d",
        required=True,
        help="Path to JSON dataset file",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Path to JSONL output file",
    )
    parser.add_argument(
        "--provider",
        "-p",
        choices=["openai", "anthropic"],
        default="openai",
        help="API provider (default: openai)",
    )
    parser.add_argument(
        "--model",
        "-m",
        help="Model name (uses provider default if not specified)",
    )
    parser.add_argument(
        "--base-url",
        "-b",
        help="Custom base URL for OpenAI-compatible API (e.g., OpenRouter)",
    )
    parser.add_argument(
        "--api-key",
        "-k",
        help="API key (falls back to OPENAI_API_KEY or ANTHROPIC_API_KEY env vars)",
    )
    parser.add_argument(
        "--temperature",
        "-t",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum tokens to generate (default: 4096)",
    )
    parser.add_argument(
        "--prompt-style",
        choices=["default", "minimal", "detailed", "chat"],
        default="default",
        help="Prompt formatting style (default: default)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't skip already processed instances",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        help="Maximum number of instances to process",
    )
    parser.add_argument(
        "--instances",
        nargs="+",
        help="Specific instance IDs to process",
    )
    parser.add_argument(
        "--no-repair",
        action="store_true",
        help="Don't attempt to repair extracted patches",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    try:
        stats = run_inference(
            dataset_path=args.dataset,
            output_path=args.output,
            provider=args.provider,
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            prompt_style=args.prompt_style,
            resume=not args.no_resume,
            max_instances=args.max_instances,
            instance_ids=args.instances,
            repair_patches=not args.no_repair,
            verbose=args.verbose,
        )

        print(f"\nFinal Statistics:\n{stats}")
        print(f"\nResults saved to: {args.output}")

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
