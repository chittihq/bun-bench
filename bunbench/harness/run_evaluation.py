"""
Main evaluation harness for Bun-Bench.

This module provides the core functionality for running evaluations of
model-generated patches against the Bun-Bench benchmark suite.
"""

import argparse
import json
import logging
import os
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not installed
    def tqdm(iterable, **kwargs):
        return iterable


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class EvaluationStatus(str, Enum):
    """Status of an evaluation run."""

    PENDING = "pending"
    RUNNING = "running"
    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass
class EvaluationConfig:
    """Configuration for running evaluations.

    Attributes:
        dataset_path: Path to JSON dataset or HuggingFace dataset identifier.
        predictions_path: Path to JSON file with instance_id -> patch mapping.
        output_dir: Directory to save evaluation results.
        max_workers: Maximum number of parallel workers.
        timeout: Timeout in seconds for each evaluation.
        docker_image_prefix: Prefix for Docker image names.
        force_rebuild: Whether to force rebuild Docker images.
        verbose: Enable verbose output.
        instance_ids: Optional list of specific instance IDs to evaluate.
    """

    dataset_path: str
    predictions_path: str
    output_dir: str = "./results"
    max_workers: int = 4
    timeout: int = 300
    docker_image_prefix: str = "bunbench"
    force_rebuild: bool = False
    verbose: bool = False
    instance_ids: list[str] | None = None


@dataclass
class TestResult:
    """Result of running tests for a single instance.

    Attributes:
        passed: Number of tests passed.
        failed: Number of tests failed.
        skipped: Number of tests skipped.
        total: Total number of tests.
        output: Raw test output.
        error: Error message if any.
    """

    passed: int = 0
    failed: int = 0
    skipped: int = 0
    total: int = 0
    output: str = ""
    error: str | None = None


@dataclass
class EvaluationResult:
    """Result of evaluating a single instance.

    Attributes:
        instance_id: Unique identifier for the instance.
        status: Status of the evaluation.
        patch_applied: Whether the patch was successfully applied.
        test_result: Results from running tests.
        duration: Time taken in seconds.
        error_message: Error message if evaluation failed.
        metadata: Additional metadata about the evaluation.
    """

    instance_id: str
    status: EvaluationStatus = EvaluationStatus.PENDING
    patch_applied: bool = False
    test_result: TestResult | None = None
    duration: float = 0.0
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary for JSON serialization."""
        result = asdict(self)
        result["status"] = self.status.value
        return result


def load_dataset(dataset_path: str) -> list[dict[str, Any]]:
    """Load dataset from JSON file or HuggingFace.

    Args:
        dataset_path: Path to JSON file or HuggingFace dataset identifier.

    Returns:
        List of dataset instances.

    Raises:
        FileNotFoundError: If the dataset file doesn't exist.
        ValueError: If the dataset format is invalid.
    """
    logger.info(f"Loading dataset from: {dataset_path}")

    # Check if it's a local JSON file
    if os.path.exists(dataset_path):
        with open(dataset_path, encoding="utf-8") as f:
            data = json.load(f)

        # Handle both list format and dict with "instances" key
        if isinstance(data, list):
            instances = data
        elif isinstance(data, dict) and "instances" in data:
            instances = data["instances"]
        else:
            raise ValueError("Dataset must be a list or dict with 'instances' key")

        logger.info(f"Loaded {len(instances)} instances from JSON file")
        return instances

    # Try loading from HuggingFace
    try:
        from datasets import load_dataset as hf_load_dataset

        logger.info(f"Loading dataset from HuggingFace: {dataset_path}")
        dataset = hf_load_dataset(dataset_path)

        # Convert to list of dicts
        if "test" in dataset:
            instances = [dict(item) for item in dataset["test"]]
        elif "train" in dataset:
            instances = [dict(item) for item in dataset["train"]]
        else:
            # Use first available split
            split_name = list(dataset.keys())[0]
            instances = [dict(item) for item in dataset[split_name]]

        logger.info(f"Loaded {len(instances)} instances from HuggingFace")
        return instances

    except ImportError:
        raise ImportError(
            "datasets package required for HuggingFace loading. "
            "Install with: pip install datasets"
        )
    except Exception as e:
        raise FileNotFoundError(f"Could not load dataset from '{dataset_path}': {e}")


def load_predictions(predictions_path: str) -> dict[str, str]:
    """Load predictions from JSON file.

    Args:
        predictions_path: Path to JSON file with instance_id -> patch mapping.

    Returns:
        Dictionary mapping instance IDs to patches.

    Raises:
        FileNotFoundError: If the predictions file doesn't exist.
        ValueError: If the predictions format is invalid.
    """
    logger.info(f"Loading predictions from: {predictions_path}")

    if not os.path.exists(predictions_path):
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

    with open(predictions_path, encoding="utf-8") as f:
        predictions = json.load(f)

    if not isinstance(predictions, dict):
        raise ValueError("Predictions must be a dict mapping instance_id to patch")

    logger.info(f"Loaded {len(predictions)} predictions")
    return predictions


def get_docker_image_name(instance: dict[str, Any], prefix: str) -> str:
    """Generate Docker image name for an instance.

    Args:
        instance: Dataset instance.
        prefix: Image name prefix.

    Returns:
        Docker image name.
    """
    instance_id = instance.get("instance_id", "unknown")
    # Sanitize instance_id for Docker image naming
    safe_id = instance_id.replace("/", "-").replace(":", "-").lower()
    return f"{prefix}:{safe_id}"


def build_docker_image(
    instance: dict[str, Any], image_name: str, force_rebuild: bool = False
) -> bool:
    """Build or retrieve Docker image for an instance.

    Args:
        instance: Dataset instance.
        image_name: Docker image name.
        force_rebuild: Whether to force rebuild.

    Returns:
        True if image is available, False otherwise.
    """
    # Check if image already exists
    if not force_rebuild:
        result = subprocess.run(
            ["docker", "image", "inspect", image_name], capture_output=True, text=True
        )
        if result.returncode == 0:
            logger.debug(f"Image {image_name} already exists")
            return True

    # Build the image
    logger.info(f"Building Docker image: {image_name}")

    # Get Dockerfile content from instance or use default
    dockerfile_content = instance.get("dockerfile", None)

    if dockerfile_content:
        # Create temporary directory with Dockerfile
        with tempfile.TemporaryDirectory() as tmpdir:
            dockerfile_path = os.path.join(tmpdir, "Dockerfile")
            with open(dockerfile_path, "w") as f:
                f.write(dockerfile_content)

            result = subprocess.run(
                ["docker", "build", "-t", image_name, tmpdir], capture_output=True, text=True
            )

            if result.returncode != 0:
                logger.error(f"Failed to build image: {result.stderr}")
                return False
    else:
        # Use default Bun image if no Dockerfile specified
        logger.info(f"Using default bun image for {image_name}")
        result = subprocess.run(
            ["docker", "pull", "oven/bun:latest"], capture_output=True, text=True
        )

        if result.returncode != 0:
            logger.error(f"Failed to pull bun image: {result.stderr}")
            return False

        # Tag it with our image name
        subprocess.run(["docker", "tag", "oven/bun:latest", image_name], capture_output=True)

    return True


def apply_patch(container_id: str, patch: str) -> tuple[bool, str]:
    """Apply a git patch inside the container.

    Args:
        container_id: Docker container ID.
        patch: Git patch content.

    Returns:
        Tuple of (success, error_message).
    """
    logger.debug(f"Applying patch to container {container_id}")

    # Write patch to a temp file in the container
    result = subprocess.run(
        ["docker", "exec", "-i", container_id, "sh", "-c", "cat > /tmp/model.patch"],
        input=patch,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        return False, f"Failed to write patch: {result.stderr}"

    # Apply the patch
    result = subprocess.run(
        ["docker", "exec", container_id, "git", "apply", "--allow-empty", "/tmp/model.patch"],
        capture_output=True,
        text=True,
        cwd="/app",
    )

    if result.returncode != 0:
        # Try with --3way for more lenient patching
        result = subprocess.run(
            [
                "docker",
                "exec",
                container_id,
                "git",
                "apply",
                "--3way",
                "--allow-empty",
                "/tmp/model.patch",
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            return False, f"Failed to apply patch: {result.stderr}"

    return True, ""


def run_tests(container_id: str, timeout: int) -> TestResult:
    """Run bun tests inside the container.

    Args:
        container_id: Docker container ID.
        timeout: Timeout in seconds.

    Returns:
        TestResult with test outcomes.
    """
    logger.debug(f"Running tests in container {container_id}")

    try:
        result = subprocess.run(
            ["docker", "exec", container_id, "bun", "test", "--json"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        output = result.stdout + result.stderr

        # Try to parse JSON output
        return parse_test_output(output)

    except subprocess.TimeoutExpired:
        return TestResult(error=f"Test execution timed out after {timeout} seconds")
    except Exception as e:
        return TestResult(error=str(e))


def parse_test_output(output: str) -> TestResult:
    """Parse bun test output.

    Args:
        output: Raw test output.

    Returns:
        TestResult with parsed outcomes.
    """
    result = TestResult(output=output)

    try:
        # Try to find JSON output
        json_start = output.find("{")
        json_end = output.rfind("}") + 1

        if json_start >= 0 and json_end > json_start:
            json_str = output[json_start:json_end]
            data = json.loads(json_str)

            result.passed = data.get("passed", 0)
            result.failed = data.get("failed", 0)
            result.skipped = data.get("skipped", 0)
            result.total = data.get("total", result.passed + result.failed + result.skipped)
            return result
    except json.JSONDecodeError:
        pass

    # Fallback: parse text output
    lines = output.split("\n")
    for line in lines:
        line_lower = line.lower()

        # Look for common patterns like "X passed", "X failed"
        if "pass" in line_lower:
            parts = line.split()
            for i, part in enumerate(parts):
                if part.isdigit() and i + 1 < len(parts) and "pass" in parts[i + 1].lower():
                    result.passed = int(part)
                    break

        if "fail" in line_lower:
            parts = line.split()
            for i, part in enumerate(parts):
                if part.isdigit() and i + 1 < len(parts) and "fail" in parts[i + 1].lower():
                    result.failed = int(part)
                    break

        if "skip" in line_lower:
            parts = line.split()
            for i, part in enumerate(parts):
                if part.isdigit() and i + 1 < len(parts) and "skip" in parts[i + 1].lower():
                    result.skipped = int(part)
                    break

    result.total = result.passed + result.failed + result.skipped
    return result


def grade_result(test_result: TestResult, instance: dict[str, Any]) -> EvaluationStatus:
    """Grade the evaluation result.

    Args:
        test_result: Results from running tests.
        instance: Dataset instance with expected outcomes.

    Returns:
        EvaluationStatus indicating success or failure.
    """
    if test_result.error:
        return EvaluationStatus.ERROR

    # Get expected test outcomes from instance
    expected_pass = instance.get("expected_pass_tests", [])
    expected_fail = instance.get("expected_fail_tests", [])

    # Simple grading: all tests must pass
    if test_result.failed == 0 and test_result.passed > 0:
        return EvaluationStatus.RESOLVED

    # More nuanced grading could compare specific test names
    return EvaluationStatus.UNRESOLVED


def run_single_evaluation(
    instance: dict[str, Any], patch: str, config: EvaluationConfig
) -> EvaluationResult:
    """Run evaluation for a single instance.

    Args:
        instance: Dataset instance.
        patch: Model-generated patch.
        config: Evaluation configuration.

    Returns:
        EvaluationResult with outcomes.
    """
    instance_id = instance.get("instance_id", "unknown")
    result = EvaluationResult(instance_id=instance_id)
    start_time = time.time()

    container_id = None

    try:
        result.status = EvaluationStatus.RUNNING

        # Get or build Docker image
        image_name = get_docker_image_name(instance, config.docker_image_prefix)
        if not build_docker_image(instance, image_name, config.force_rebuild):
            result.status = EvaluationStatus.ERROR
            result.error_message = "Failed to build Docker image"
            return result

        # Get the repo setup from instance
        repo_url = instance.get("repo", "")
        base_commit = instance.get("base_commit", "")
        workdir = instance.get("workdir", "/app")

        # Create and start container
        logger.debug(f"Starting container for {instance_id}")

        # Build docker run command
        docker_run_cmd = [
            "docker",
            "run",
            "-d",
            "--name",
            f"bunbench-{instance_id.replace('/', '-')}",
            "-w",
            workdir,
            image_name,
            "tail",
            "-f",
            "/dev/null",  # Keep container running
        ]

        run_result = subprocess.run(docker_run_cmd, capture_output=True, text=True)

        if run_result.returncode != 0:
            result.status = EvaluationStatus.ERROR
            result.error_message = f"Failed to start container: {run_result.stderr}"
            return result

        container_id = run_result.stdout.strip()

        # Clone repo and checkout base commit if specified
        if repo_url:
            clone_cmd = ["docker", "exec", container_id, "git", "clone", repo_url, workdir]
            subprocess.run(clone_cmd, capture_output=True)

            if base_commit:
                checkout_cmd = [
                    "docker",
                    "exec",
                    "-w",
                    workdir,
                    container_id,
                    "git",
                    "checkout",
                    base_commit,
                ]
                subprocess.run(checkout_cmd, capture_output=True)

        # Install dependencies
        subprocess.run(
            ["docker", "exec", "-w", workdir, container_id, "bun", "install"],
            capture_output=True,
            timeout=120,
        )

        # Apply the patch
        patch_success, patch_error = apply_patch(container_id, patch)
        result.patch_applied = patch_success

        if not patch_success:
            result.status = EvaluationStatus.ERROR
            result.error_message = patch_error
            return result

        # Run tests
        test_result = run_tests(container_id, config.timeout)
        result.test_result = test_result

        # Grade the result
        result.status = grade_result(test_result, instance)

    except Exception as e:
        logger.exception(f"Error evaluating {instance_id}")
        result.status = EvaluationStatus.ERROR
        result.error_message = str(e)

    finally:
        # Cleanup: stop and remove container
        if container_id:
            try:
                subprocess.run(
                    ["docker", "rm", "-f", container_id], capture_output=True, timeout=30
                )
            except Exception:
                pass

        result.duration = time.time() - start_time

    return result


def run_evaluation(config: EvaluationConfig) -> list[EvaluationResult]:
    """Run evaluation for all instances.

    Args:
        config: Evaluation configuration.

    Returns:
        List of EvaluationResult for each instance.
    """
    logger.info("Starting Bun-Bench evaluation")
    logger.info(f"Configuration: {config}")

    # Load dataset and predictions
    dataset = load_dataset(config.dataset_path)
    predictions = load_predictions(config.predictions_path)

    # Filter instances if specific IDs requested
    if config.instance_ids:
        dataset = [inst for inst in dataset if inst.get("instance_id") in config.instance_ids]
        logger.info(f"Filtered to {len(dataset)} instances")

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    results: list[EvaluationResult] = []

    # Run evaluations in parallel
    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        # Submit all tasks
        future_to_instance = {}

        for instance in dataset:
            instance_id = instance.get("instance_id")

            if instance_id not in predictions:
                logger.warning(f"No prediction for {instance_id}, skipping")
                results.append(
                    EvaluationResult(
                        instance_id=instance_id,
                        status=EvaluationStatus.SKIPPED,
                        error_message="No prediction provided",
                    )
                )
                continue

            patch = predictions[instance_id]
            future = executor.submit(run_single_evaluation, instance, patch, config)
            future_to_instance[future] = instance_id

        # Collect results with progress bar
        futures = list(future_to_instance.keys())

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Evaluating", disable=not config.verbose
        ):
            try:
                result = future.result()
                results.append(result)

                if config.verbose:
                    status_emoji = {
                        EvaluationStatus.RESOLVED: "PASS",
                        EvaluationStatus.UNRESOLVED: "FAIL",
                        EvaluationStatus.ERROR: "ERROR",
                        EvaluationStatus.SKIPPED: "SKIP",
                    }
                    status = status_emoji.get(result.status, "?")
                    logger.info(f"[{status}] {result.instance_id}")

            except Exception as e:
                instance_id = future_to_instance[future]
                logger.error(f"Error processing {instance_id}: {e}")
                results.append(
                    EvaluationResult(
                        instance_id=instance_id, status=EvaluationStatus.ERROR, error_message=str(e)
                    )
                )

    logger.info(f"Evaluation complete. {len(results)} instances processed.")
    return results


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Bun-Bench Evaluation Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run evaluation with default settings
  python -m bunbench.harness.run_evaluation \\
      --dataset ./data/bun-bench.json \\
      --predictions ./predictions.json

  # Run with parallel workers and verbose output
  python -m bunbench.harness.run_evaluation \\
      --dataset ./data/bun-bench.json \\
      --predictions ./predictions.json \\
      --workers 8 \\
      --verbose

  # Evaluate specific instances
  python -m bunbench.harness.run_evaluation \\
      --dataset ./data/bun-bench.json \\
      --predictions ./predictions.json \\
      --instance-ids bun-123 bun-456
        """,
    )

    parser.add_argument(
        "--dataset",
        "-d",
        required=True,
        help="Path to dataset JSON file or HuggingFace dataset identifier",
    )
    parser.add_argument(
        "--predictions",
        "-p",
        required=True,
        help="Path to predictions JSON file (instance_id -> patch mapping)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="./results",
        help="Output directory for results (default: ./results)",
    )
    parser.add_argument(
        "--workers", "-w", type=int, default=4, help="Number of parallel workers (default: 4)"
    )
    parser.add_argument(
        "--timeout",
        "-t",
        type=int,
        default=300,
        help="Timeout in seconds per evaluation (default: 300)",
    )
    parser.add_argument(
        "--docker-prefix", default="bunbench", help="Docker image name prefix (default: bunbench)"
    )
    parser.add_argument(
        "--force-rebuild", action="store_true", help="Force rebuild of Docker images"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--instance-ids", nargs="+", help="Specific instance IDs to evaluate")

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create configuration
    config = EvaluationConfig(
        dataset_path=args.dataset,
        predictions_path=args.predictions,
        output_dir=args.output,
        max_workers=args.workers,
        timeout=args.timeout,
        docker_image_prefix=args.docker_prefix,
        force_rebuild=args.force_rebuild,
        verbose=args.verbose,
        instance_ids=args.instance_ids,
    )

    # Run evaluation
    results = run_evaluation(config)

    # Generate and save report
    from bunbench.harness.reporting import generate_report, save_report

    report = generate_report(results)
    report_path = os.path.join(args.output, "evaluation_report.json")
    save_report(report, report_path)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total instances:    {report.total}")
    print(f"Resolved:           {report.resolved} ({report.resolved_rate:.1%})")
    print(f"Unresolved:         {report.unresolved}")
    print(f"Errors:             {report.errors}")
    print(f"Skipped:            {report.skipped}")
    print("=" * 60)
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
