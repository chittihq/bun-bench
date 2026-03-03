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
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from bunbench.inference.utils import extract_snapshot_files

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
    """
    dataset_path: str
    predictions_path: str
    output_dir: str = "./results"
    max_workers: int = 4
    timeout: int = 300
    docker_image_prefix: str = "bunbench"
    force_rebuild: bool = False
    verbose: bool = False
    instance_ids: Optional[List[str]] = None
    local_mode: bool = False


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
                    raw_response = predictions.get("raw_response") or ""
                    if instance_id and (patch or raw_response):
                        # Store as dict to preserve raw_response
                        predictions = {instance_id: {"patch": patch or "", "raw_response": raw_response}}
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
                                # Store both patch and raw_response
                                patch = obj.get("extracted_patch") or obj.get("patch")
                                raw_response = obj.get("raw_response") or ""
                                # Store if patch exists OR if raw_response exists (for full code format)
                                if patch or raw_response:
                                    predictions[instance_id] = {"patch": patch or "", "raw_response": raw_response}
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


def apply_full_content_then_diff(workdir: Path, raw_response: str) -> bool:
    """Extract full file content from model output, write it, then generate diff ourselves.

    This solves the issue where model-generated diffs have wrong line numbers.
    Instead of relying on the model's diff, we:
    1. Extract the full fixed file content from model output
    2. Write it directly to the source file
    3. Generate diff ourselves using `git diff`
    4. Apply that diff to create a proper commit state

    Args:
        workdir: Working directory containing the source files.
        raw_response: Raw model response.

    Returns:
        True if content was extracted and written successfully.
    """
    import re

    if not workdir.exists():
        return False

    src_dir = workdir / "src"
    if not src_dir.exists():
        return False

    # Method 1: Try to extract full file content from snapshot format
    # (// File: src/filename.ts ... full content ...)
    # This is the BEST method - model gives us complete fixed file
    snapshot_files = extract_snapshot_files(raw_response)

    if snapshot_files:
        logger.debug(f"Found {len(snapshot_files)} snapshot file(s)")
        files_written = []
        for filepath, content in snapshot_files.items():
            # filepath can be "src/filename.ts" or "test/filename.ts"
            parts = filepath.split("/")
            if len(parts) >= 2 and parts[0] in ("src", "test"):
                # Determine target directory based on filepath prefix
                target_dir = workdir / parts[0]
                filename = parts[-1]
            else:
                # Default to src directory
                target_dir = src_dir
                filename = filepath.split("/")[-1]

            target_file = target_dir / filename

            # Create parent directories if needed and write the file
            target_dir.mkdir(parents=True, exist_ok=True)
            target_file.write_text(content + "\n")
            logger.debug(f"Wrote snapshot content to {target_dir.name}/{filename}")
            files_written.append(f"{target_dir.name}/{filename}")

        if files_written:
            logger.info(f"Applied patch via content extraction: {', '.join(files_written)}")

        if files_written:
            # Generate diff ourselves from the written files
            diff_result = subprocess.run(
                ["git", "diff"],
                cwd=workdir,
                capture_output=True,
                text=True
            )

            if diff_result.stdout:
                logger.debug(f"Generated diff with {len(diff_result.stdout)} chars")

            return True

    # Method 2: Try any code blocks without diff markers (full code format)
    # Look for code blocks that are NOT diffs
    code_blocks = re.findall(r'```typescript\n(.*?)```', raw_response, re.DOTALL)
    if not code_blocks:
        code_blocks = re.findall(r'```ts\n(.*?)```', raw_response, re.DOTALL)
    if not code_blocks:
        code_blocks = re.findall(r'```javascript\n(.*?)```', raw_response, re.DOTALL)
    if not code_blocks:
        code_blocks = re.findall(r'```js\n(.*?)```', raw_response, re.DOTALL)

    if code_blocks:
        logger.debug(f"Found {len(code_blocks)} code block(s) without diff markers")
        for code_content in code_blocks:
            # Extract function/class names from code
            code_funcs = set(re.findall(
                r'(?:function|class|export\s+function|export\s+const|export\s+class)\s+(\w+)',
                code_content
            ))

            if not code_funcs:
                continue

            # Find matching file
            for f in src_dir.glob('*.ts'):
                file_content = f.read_text()
                file_funcs = set(re.findall(
                    r'(?:function|class|export\s+function|export\s+const|export\s+class)\s+(\w+)',
                    file_content
                ))
                matches = code_funcs & file_funcs

                if len(matches) >= 1:  # Match on at least 1 function
                    # Remove any remaining File: comments
                    code_lines = code_content.splitlines()
                    code_clean = '\n'.join(
                        line for line in code_lines
                        if not line.strip().startswith('// File:')
                    )
                    f.write_text(code_clean + '\n')
                    logger.debug(f"Wrote code to {f.name} (matched: {matches})")

        # Check if any files changed
        diff_check = subprocess.run(
            ["git", "diff", "--name-only"],
            cwd=workdir,
            capture_output=True,
            text=True
        )
        if diff_check.stdout.strip():
            logger.debug(f"Files modified via code blocks: {diff_check.stdout.strip()}")
            return True

    # Method 3: For diff format, we can't easily reconstruct full file
    # Just return False and let the error handling show the issue
    logger.debug("No full content found in response, cannot apply patch")

    return False


def run_local_evaluation(
    instance: Dict[str, Any],
    patch: str,
    raw_response: str = "",
    config: EvaluationConfig = None
) -> EvaluationResult:
    """Run evaluation locally without Docker.

    Args:
        instance: Dataset instance.
        patch: Model-generated patch.
        raw_response: Raw model response (for fallback file replacement).
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

            # Try git apply if patch is available
            result.patch_applied = False
            apply_stderr = ""
            if patch:
                # Apply patch using git apply
                patch_file = tmp_path / "patch.diff"
                with open(patch_file, "w") as f:
                    f.write(patch)

                apply_result = subprocess.run(
                    ["git", "apply", patch_file],
                    cwd=workdir,
                    capture_output=True,
                    text=True
                )

                result.patch_applied = apply_result.returncode == 0
                apply_stderr = apply_result.stderr

            # If git apply failed, try extracting full content and generating diff ourselves
            if not result.patch_applied and raw_response:
                if apply_full_content_then_diff(workdir, raw_response):
                    # Verify the changes are applied by checking git diff
                    diff_check = subprocess.run(
                        ["git", "diff", "--name-only"],
                        cwd=workdir,
                        capture_output=True,
                        text=True
                    )
                    if diff_check.stdout.strip():
                        result.patch_applied = True
                        logger.info(f"Applied patch via content extraction: {diff_check.stdout.strip()}")

            # If patch still failed, mark as error
            if not result.patch_applied:
                result.status = EvaluationStatus.ERROR
                result.error_message = f"Failed to apply patch: {apply_stderr}"
                return result

            # Install dependencies
            subprocess.run(
                ["bun", "install"],
                cwd=workdir,
                capture_output=True,
                timeout=120
            )

            # Run tests
            test_result = run_local_tests(workdir, config.timeout)
            result.test_result = test_result

            # Grade result
            result.status = grade_result(test_result, instance)

        except Exception as e:
            logger.exception(f"Error in local evaluation: {instance_id}")
            result.status = EvaluationStatus.ERROR
            result.error_message = str(e)

    result.duration = time.time() - start_time
    return result


def run_local_tests(workdir: Path, timeout: int) -> TestResult:
    """Run bun tests locally.

    Args:
        workdir: Working directory.
        timeout: Timeout in seconds.

    Returns:
        TestResult with test outcomes.
    """
    logger.debug(f"Running tests locally in {workdir}")

    try:
        result = subprocess.run(
            ["bun", "test", "--json"],
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

    # Strip ANSI escape sequences
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    output_clean = ansi_escape.sub('', output)
    
    lines = output_clean.split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        line_lower = line.lower()

        # Look for the summary line: "1 pass", "3 fail"
        # Bun standard output: " 1 pass", " 3 fail"
        match_pass = re.search(r'(\d+)\s+pass', line_lower)
        if match_pass:
            result.passed = int(match_pass.group(1))
            
        match_fail = re.search(r'(\d+)\s+fail', line_lower)
        if match_fail:
            result.failed = int(match_fail.group(1))
            
        match_skip = re.search(r'(\d+)\s+skip', line_lower)
        if match_skip:
            result.skipped = int(match_skip.group(1))

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
    raw_response: str = "",
    config: EvaluationConfig = None
) -> EvaluationResult:
    """Run evaluation for a single instance.

    Args:
        instance: Dataset instance.
        patch: Model-generated patch.
        raw_response: Raw model response (for fallback file replacement).
        config: Evaluation configuration.

    Returns:
        EvaluationResult with outcomes.
    """
    instance_id = instance.get("instance_id", "unknown")
    result = EvaluationResult(instance_id=instance_id)
    start_time = time.time()

    # Use local mode or Docker mode
    if config.local_mode:
        return run_local_evaluation(instance, patch, raw_response, config)

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
    # Set logging level if verbose
    if config.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)

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

            pred = predictions[instance_id]
            # Handle both old format (string) and new format (dict)
            if isinstance(pred, dict):
                patch = pred.get("patch") or pred.get("extracted_patch") or ""
                raw_response = pred.get("raw_response", "")
            else:
                patch = pred
                raw_response = ""

            future = executor.submit(
                run_single_evaluation, instance, patch, raw_response, config
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


