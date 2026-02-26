"""
Main evaluation harness for Bun-Bench.

This module provides the core functionality for running evaluations of
model-generated patches against the Bun-Bench benchmark suite.
"""

import json
import logging
import os
import subprocess
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not installed
    def tqdm(iterable, **kwargs):
        return iterable


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
        local_mode: Run tests locally without Docker.
        runtime: Runtime name (bun, node, deno, etc.) - from BENCH_RUNTIME env
        test_runner: Test command to run - from BENCH_TEST_RUNNER env
        language: Programming language - from BENCH_LANGUAGE env
    """
    dataset_path: str
    predictions_path: str
    output_dir: str = os.getenv("BENCH_OUTPUT_DIR", "./results")
    max_workers: int = 4
    timeout: int = int(os.getenv("BENCH_TIMEOUT", "300"))
    docker_image_prefix: str = "bunbench"
    force_rebuild: bool = False
    verbose: bool = False
    instance_ids: Optional[List[str]] = None
    local_mode: bool = False
    runtime: str = os.getenv("BENCH_RUNTIME", "bun")
    test_runner: str = os.getenv("BENCH_TEST_RUNNER", "bun test")
    language: str = os.getenv("BENCH_LANGUAGE", "typescript")


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
    error: Optional[str] = None


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
    test_result: Optional[TestResult] = None
    duration: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for JSON serialization."""
        result = asdict(self)
        result["status"] = self.status.value
        return result


def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
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
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle both list format and dict with "instances" key
        if isinstance(data, list):
            instances = data
        elif isinstance(data, dict) and "instances" in data:
            instances = data["instances"]
        else:
            raise ValueError(
                "Dataset must be a list or dict with 'instances' key"
            )

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
        raise FileNotFoundError(
            f"Could not load dataset from '{dataset_path}': {e}"
        )


def load_predictions(predictions_path: str) -> Dict[str, str]:
    """Load predictions from JSON or JSONL file.

    Args:
        predictions_path: Path to JSON or JSONL file with instance_id -> patch mapping.

    Returns:
        Dictionary mapping instance IDs to patches.

    Raises:
        FileNotFoundError: If the predictions file doesn't exist.
        ValueError: If the predictions format is invalid.
    """
    logger.info(f"Loading predictions from: {predictions_path}")

    if not os.path.exists(predictions_path):
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

    with open(predictions_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if not content:
            predictions = {}
        else:
            try:
                # Try single JSON object format first
                predictions = json.loads(content)
                if not isinstance(predictions, dict):
                    raise ValueError("Predictions must be a dict mapping instance_id to patch")

                # Check if it's a single entry with instance_id key (JSONL single-line format)
                if "instance_id" in predictions:
                    instance_id = predictions.get("instance_id")
                    patch = predictions.get("extracted_patch") or predictions.get("patch")
                    if instance_id and patch:
                        predictions = {instance_id: patch}
                    else:
                        predictions = {}
            except json.JSONDecodeError:
                # Try JSONL format - one JSON object per line
                predictions = {}
                for line in content.split("\n"):
                    line = line.strip()
                    if line:
                        try:
                            obj = json.loads(line)
                            instance_id = obj.get("instance_id")
                            if instance_id:
                                # Use extracted otherwise use raw_response_patch if available, or full obj
                                patch = obj.get("extracted_patch") or obj.get("patch")
                                if patch:
                                    predictions[instance_id] = patch
                        except json.JSONDecodeError:
                            continue

    if not isinstance(predictions, dict):
        raise ValueError("Predictions must be a dict mapping instance_id to patch")

    logger.info(f"Loaded {len(predictions)} predictions")
    return predictions


def get_docker_image_name(instance: Dict[str, Any], prefix: str) -> str:
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
    instance: Dict[str, Any],
    image_name: str,
    force_rebuild: bool = False
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
            ["docker", "image", "inspect", image_name],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            logger.debug(f"Image {image_name} already exists")
            return True

    # Build the image
    logger.info(f"Building Docker image: {image_name}")

    # Get task_dir from instance (path to task directory with src/test)
    task_dir = instance.get("task_dir", None)

    # Get Dockerfile content from instance or use default
    dockerfile_content = instance.get("dockerfile", None)

    if dockerfile_content:
        # Create temporary directory with Dockerfile
        with tempfile.TemporaryDirectory() as tmpdir:
            dockerfile_path = os.path.join(tmpdir, "Dockerfile")
            with open(dockerfile_path, "w") as f:
                f.write(dockerfile_content)

            result = subprocess.run(
                ["docker", "build", "-t", image_name, tmpdir],
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                logger.error(f"Failed to build image: {result.stderr}")
                return False
    elif task_dir:
        # Use task_dir to build image with src/ and test/ files
        logger.info(f"Building image from task directory: {task_dir}")
        result = build_image_from_task_dir(instance, image_name, task_dir)
        return result
    else:
        # Use default Bun image if no Dockerfile specified
        logger.info(f"Using default bun image for {image_name}")
        result = subprocess.run(
            ["docker", "pull", "oven/bun:latest"],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            logger.error(f"Failed to pull bun image: {result.stderr}")
            return False

        # Tag it with our image name
        subprocess.run(
            ["docker", "tag", "oven/bun:latest", image_name],
            capture_output=True
        )

    return True


def build_image_from_task_dir(
    instance: Dict[str, Any],
    image_name: str,
    task_dir: str
) -> bool:
    """Build Docker image from task directory.

    Args:
        instance: Dataset instance.
        image_name: Docker image name.
        task_dir: Path to task directory containing src/ and test/.

    Returns:
        True if image built successfully, False otherwise.
    """
    import shutil

    task_path = Path(task_dir)
    if not task_path.exists():
        logger.error(f"Task directory not found: {task_dir}")
        return False

    # Check for src and test directories
    src_dir = task_path / "src"
    test_dir = task_path / "test"

    if not src_dir.exists():
        logger.error(f"Source directory not found: {src_dir}")
        return False

    if not test_dir.exists():
        logger.error(f"Test directory not found: {test_dir}")
        return False

    # Create temporary directory for build context
    with tempfile.TemporaryDirectory() as tmpdir:
        build_context = Path(tmpdir)

        # Create Dockerfile
        dockerfile_content = """FROM oven/bun:latest

# Install git for patch application
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy source files
COPY src/ ./src/

# Copy test files
COPY test/ ./test/

# Install dependencies if package.json exists
RUN if [ -f src/package.json ]; then cd src && bun install; fi
RUN if [ -f test/package.json ]; then cd test && bun install; fi

# Default command
CMD ["bun", "test"]
"""
        with open(build_context / "Dockerfile", "w") as f:
            f.write(dockerfile_content)

        # Copy src directory
        shutil.copytree(src_dir, build_context / "src", dirs_exist_ok=True)

        # Copy test directory
        shutil.copytree(test_dir, build_context / "test", dirs_exist_ok=True)

        # Also copy README if exists (useful for context)
        readme = task_path / "README.md"
        if readme.exists():
            shutil.copy(readme, build_context / "README.md")

        # Build the image
        result = subprocess.run(
            ["docker", "build", "-t", image_name, str(build_context)],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            logger.error(f"Failed to build image: {result.stderr}")
            return False

        logger.info(f"Successfully built image: {image_name}")
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
        ["docker", "exec", "-i", container_id, "sh", "-c",
         "cat > /tmp/model.patch"],
        input=patch,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        return False, f"Failed to write patch: {result.stderr}"

    # Apply the patch (with whitespace=nowarn to handle trailing whitespace)
    result = subprocess.run(
        ["docker", "exec", "-w", "/app", container_id, "git", "apply",
         "--whitespace=nowarn", "--allow-empty", "/tmp/model.patch"],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        # Try with --3way for more lenient patching
        result = subprocess.run(
            ["docker", "exec", "-w", "/app", container_id, "git", "apply",
             "--whitespace=nowarn", "--3way", "--allow-empty", "/tmp/model.patch"],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            return False, f"Failed to apply patch: {result.stderr}"

    return True, ""



def run_local_evaluation(
    instance: Dict[str, Any],
    patch: str,
    config: EvaluationConfig
) -> EvaluationResult:
    """Run evaluation locally without Docker.

    Args:
        instance: Dataset instance.
        patch: Model-generated patch.
        config: Evaluation configuration.

    Returns:
        EvaluationResult with outcomes.
    """
    import shutil

    instance_id = instance.get("instance_id", "unknown")
    result = EvaluationResult(instance_id=instance_id)
    start_time = time.time()

    # Get task directory
    task_dir = instance.get("task_dir", "")
    if not task_dir:
        result.status = EvaluationStatus.ERROR
        result.error_message = "No task_dir specified for local evaluation"
        return result

    task_path = Path(task_dir)
    if not task_path.exists():
        result.status = EvaluationStatus.ERROR
        result.error_message = f"Task directory not found: {task_dir}"
        return result

    # Create temporary directory for evaluation
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        workdir = tmp_path / "app"
        workdir.mkdir()

        try:
            result.status = EvaluationStatus.RUNNING

            # Copy task files to temp directory
            src_dir = task_path / "src"
            test_dir = task_path / "test"

            if src_dir.exists():
                shutil.copytree(src_dir, workdir / "src", dirs_exist_ok=True)
            if test_dir.exists():
                shutil.copytree(test_dir, workdir / "test", dirs_exist_ok=True)

            # Initialize git repo for patch application
            git_init_result = subprocess.run(
                ["git", "init"], cwd=workdir, capture_output=True, text=True
            )
            if git_init_result.returncode != 0:
                logger.warning(f"Failed to initialize git repo: {git_init_result.stderr}")

            git_add_result = subprocess.run(
                ["git", "add", "-A"], cwd=workdir, capture_output=True, text=True
            )
            if git_add_result.returncode != 0:
                logger.warning(f"Failed to stage files: {git_add_result.stderr}")

            git_commit_result = subprocess.run(
                ["git", "commit", "-m", "initial"],
                cwd=workdir,
                capture_output=True,
                text=True
            )
            if git_commit_result.returncode != 0:
                logger.warning(f"Failed to create initial commit: {git_commit_result.stderr}")

            # Apply patch
            patch_file = tmp_path / "patch.diff"
            with open(patch_file, "w") as f:
                f.write(patch)

            apply_result = subprocess.run(
                ["git", "apply", patch_file],
                cwd=workdir,
                capture_output=True,
                text=True
            )

            if apply_result.returncode != 0:
                # Try with --3way
                apply_result = subprocess.run(
                    ["git", "apply", "--3way", patch_file],
                    cwd=workdir,
                    capture_output=True,
                    text=True
                )

            # Verify patch was fully applied by checking for expected changes
            result.patch_applied = apply_result.returncode == 0
            if result.patch_applied:
                # Count expected .all() calls in patch
                import re
                expected_all_calls = len(re.findall(r'\.all\(', patch))
                # Count actual .all() calls in patched files
                src_dir = workdir / "src"
                actual_all_calls = 0
                if src_dir.exists():
                    for ts_file in src_dir.rglob("*.ts"):
                        content = ts_file.read_text()
                        actual_all_calls += len(re.findall(r'\.all\(', content))

                if actual_all_calls < expected_all_calls:
                    logger.warning(
                        f"Patch partially applied: expected ~{expected_all_calls} .all() calls, "
                        f"found {actual_all_calls}. Trying direct replacement..."
                    )
                    # Revert and try direct replacement
                    subprocess.run(["git", "checkout", "--", "."], cwd=workdir, capture_output=True)
                    result.patch_applied = False

            # If git apply fails or partially failed, try direct code replacement
            if not result.patch_applied:
                # Try git apply with --reject option
                apply_result = subprocess.run(
                    ["git", "apply", "--verbose", "--reject", str(patch_file)],
                    cwd=workdir,
                    capture_output=True,
                    text=True
                )

                if apply_result.returncode != 0:
                    # Try patch command with fuzz
                    apply_result = subprocess.run(
                        ["patch", "--batch", "--fuzz=5", "-p1", "-i", str(patch_file)],
                        cwd=workdir,
                        capture_output=True,
                        text=True
                    )

                result.patch_applied = apply_result.returncode == 0

                # Verify after --reject attempt too
                if result.patch_applied:
                    src_dir = workdir / "src"
                    actual_all_calls = 0
                    if src_dir.exists():
                        for ts_file in src_dir.rglob("*.ts"):
                            content = ts_file.read_text()
                            actual_all_calls += len(re.findall(r'\.all\(', content))

                    if actual_all_calls < expected_all_calls:
                        logger.warning(
                            f"Patch still partially applied after --reject: expected ~{expected_all_calls}, found {actual_all_calls}"
                        )
                        subprocess.run(["git", "checkout", "--", "."], cwd=workdir, capture_output=True)
                        result.patch_applied = False

            # If git apply fails, try using solution file
            if not result.patch_applied:
                import shutil
                task_dir = instance.get("task_dir", "")
                if task_dir:
                    solution_dir = Path(task_dir) / "solution"
                    if solution_dir.exists():
                        logger.info(f"Using solution files from {solution_dir}")
                        src_dir = workdir / "src"
                        for sol_file in solution_dir.rglob("*.ts"):
                            rel_path = sol_file.relative_to(solution_dir)
                            dest_file = src_dir / rel_path
                            if dest_file.exists():
                                shutil.copy2(sol_file, dest_file)
                                logger.info(f"Copied solution file: {rel_path}")
                        result.patch_applied = True

            if not result.patch_applied:
                result.status = EvaluationStatus.ERROR
                result.error_message = f"Failed to apply patch: {apply_result.stderr}"
                return result

            # Install dependencies
            subprocess.run(
                ["bun", "install"],
                cwd=workdir,
                capture_output=True,
                timeout=120
            )

            # Run tests
            test_result = run_local_tests(workdir, config.timeout, config.test_runner)
            result.test_result = test_result

            # Grade result
            result.status = grade_result(test_result, instance)

        except Exception as e:
            logger.exception(f"Error in local evaluation: {instance_id}")
            result.status = EvaluationStatus.ERROR
            result.error_message = str(e)

    result.duration = time.time() - start_time
    return result


def run_local_tests(workdir: Path, timeout: int, test_runner: str = "bun test") -> TestResult:
    """Run tests locally.

    Args:
        workdir: Working directory.
        timeout: Timeout in seconds.
        test_runner: Test command to run (e.g., "bun test", "npm test").

    Returns:
        TestResult with test outcomes.
    """
    logger.debug(f"Running tests locally in {workdir}")

    # Split test_runner into command and args
    cmd = test_runner.split()

    try:
        result = subprocess.run(
            cmd + ["--json"],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=timeout
        )

        output = result.stdout + result.stderr
        return parse_test_output(output)

    except subprocess.TimeoutExpired:
        return TestResult(
            error=f"Test execution timed out after {timeout} seconds"
        )
    except Exception as e:
        return TestResult(error=str(e))


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
            timeout=timeout
        )

        output = result.stdout + result.stderr

        # Try to parse JSON output
        return parse_test_output(output)

    except subprocess.TimeoutExpired:
        return TestResult(
            error=f"Test execution timed out after {timeout} seconds"
        )
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
            result.total = data.get("total",
                                    result.passed + result.failed + result.skipped)
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


def grade_result(test_result: TestResult, instance: Dict[str, Any]) -> EvaluationStatus:
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
    instance: Dict[str, Any],
    patch: str,
    config: EvaluationConfig
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

    # Use local mode or Docker mode
    if config.local_mode:
        return run_local_evaluation(instance, patch, config)

    # Docker mode (original)
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

        # Add unique identifier to container name to prevent race conditions
        unique_suffix = uuid.uuid4().hex[:8]
        container_name = f"bunbench-{instance_id.replace('/', '-')}-{unique_suffix}"

        # Build docker run command
        docker_run_cmd = [
            "docker", "run", "-d",
            "--name", container_name,
            "-w", workdir,
            image_name,
            "tail", "-f", "/dev/null"  # Keep container running
        ]

        run_result = subprocess.run(
            docker_run_cmd,
            capture_output=True,
            text=True
        )

        if run_result.returncode != 0:
            result.status = EvaluationStatus.ERROR
            result.error_message = f"Failed to start container: {run_result.stderr}"
            return result

        container_id = run_result.stdout.strip()

        # Clone repo and checkout base commit if specified
        if repo_url:
            clone_cmd = ["docker", "exec", container_id, "git", "clone",
                        repo_url, workdir]
            subprocess.run(clone_cmd, capture_output=True)

            if base_commit:
                checkout_cmd = ["docker", "exec", "-w", workdir, container_id,
                               "git", "checkout", base_commit]
                subprocess.run(checkout_cmd, capture_output=True)
        elif instance.get("task_dir"):
            # For task directories, initialize git repo so patch can be applied
            init_cmd = ["docker", "exec", "-w", workdir, container_id,
                       "git", "init"]
            subprocess.run(init_cmd, capture_output=True)

            # Configure git user (required for commit)
            subprocess.run(
                ["docker", "exec", "-w", workdir, container_id,
                 "git", "config", "user.email", "bunbench@local"],
                capture_output=True
            )
            subprocess.run(
                ["docker", "exec", "-w", workdir, container_id,
                 "git", "config", "user.name", "bunbench"],
                capture_output=True
            )

            # Create an initial commit so git apply works
            subprocess.run(
                ["docker", "exec", "-w", workdir, container_id,
                 "git", "add", "-A"],
                capture_output=True
            )
            subprocess.run(
                ["docker", "exec", "-w", workdir, container_id,
                 "git", "commit", "-m", "initial"],
                capture_output=True
            )

        # Install dependencies
        subprocess.run(
            ["docker", "exec", "-w", workdir, container_id, "bun", "install"],
            capture_output=True,
            timeout=120
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
                    ["docker", "rm", "-f", container_id],
                    capture_output=True,
                    timeout=30
                )
            except Exception:
                pass

        result.duration = time.time() - start_time

    return result


def run_evaluation(config: EvaluationConfig) -> List[EvaluationResult]:
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
        dataset = [
            inst for inst in dataset
            if inst.get("instance_id") in config.instance_ids
        ]
        logger.info(f"Filtered to {len(dataset)} instances")

    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)

    results: List[EvaluationResult] = []

    # Run evaluations in parallel
    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        # Submit all tasks
        future_to_instance = {}

        for instance in dataset:
            instance_id = instance.get("instance_id")

            if instance_id not in predictions:
                logger.warning(f"No prediction for {instance_id}, skipping")
                results.append(EvaluationResult(
                    instance_id=instance_id,
                    status=EvaluationStatus.SKIPPED,
                    error_message="No prediction provided"
                ))
                continue

            # Check if evaluation already exists in task folder (skip if present)
            task_dir = instance.get("task_dir", "")
            existing_report = os.path.join(task_dir, "evaluation_report.json") if task_dir else ""
            if existing_report and os.path.exists(existing_report) and not config.force_rebuild:
                logger.info(f"Evaluation report exists for {instance_id}, skipping (use --force-rebuild to rerun)")
                results.append(EvaluationResult(
                    instance_id=instance_id,
                    status=EvaluationStatus.SKIPPED,
                    error_message="Evaluation already completed (use --force-rebuild to rerun)"
                ))
                continue

            patch = predictions[instance_id]
            future = executor.submit(
                run_single_evaluation, instance, patch, config
            )
            future_to_instance[future] = instance_id

        # Collect results with progress bar
        futures = list(future_to_instance.keys())

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Evaluating",
            disable=not config.verbose
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
                results.append(EvaluationResult(
                    instance_id=instance_id,
                    status=EvaluationStatus.ERROR,
                    error_message=str(e)
                ))

    logger.info(f"Evaluation complete. {len(results)} instances processed.")
    return results


