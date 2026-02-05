#!/usr/bin/env python3
"""
Simple example demonstrating how to use bun-bench programmatically.

This script shows how to:
1. Load a benchmark task
2. Apply a model's patch
3. Run tests
4. Evaluate the results

Usage:
    python examples/run_simple_eval.py
"""

import json
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Any
import sys
import os

# Add the project root to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.test_harness import LogParser, grade_instance, validate_instance, BunTestSummary


# -----------------------------------------------------------------------------
# Sample Data (for demonstration purposes)
# -----------------------------------------------------------------------------

SAMPLE_INSTANCE = {
    "instance_id": "bun__serve__content_length_1",
    "repo": "oven-sh/bun",
    "base_commit": "abc123def456",
    "problem_statement": """
[Bug Fix] Bun.serve() returns incorrect Content-Length header when response
body contains multi-byte UTF-8 characters. The server reports byte length as
character length, causing truncated responses for non-ASCII content.

Reproduce with: `new Response("こんにちは")` returns Content-Length: 5 instead of 15.

Fix the response header calculation to use byte length.
""",
    "hints_text": "Look at the Response header calculation in src/http.zig",
    "created_at": "2026-01-15T10:30:00Z",
    "version": "1.0.0",
    "FAIL_TO_PASS": [
        "test/http/content-length.test.ts::should calculate correct byte length for UTF-8",
        "test/http/content-length.test.ts::should handle multi-byte characters",
    ],
    "PASS_TO_PASS": [
        "test/http/content-length.test.ts::should work for ASCII content",
        "test/http/basic.test.ts::should start server",
    ],
    "environment_setup_commit": "def789abc012",
    "difficulty": "medium",
    "category": "bug_fix",
}

SAMPLE_PREDICTION = {
    "instance_id": "bun__serve__content_length_1",
    "model_name_or_path": "claude-3-opus",
    "model_patch": """
diff --git a/src/http.zig b/src/http.zig
index 1234567..abcdefg 100644
--- a/src/http.zig
+++ b/src/http.zig
@@ -100,7 +100,7 @@ fn calculateContentLength(body: []const u8) usize {
-    return body.len;
+    return std.mem.len(body);  // Use byte length, not character count
 }
""",
    "full_output": "I analyzed the issue and found that the Content-Length calculation was using string length instead of byte length.",
}


class BunBenchEvaluator:
    """
    Evaluator for bun-bench instances.

    This class provides methods to:
    - Load and validate instances
    - Apply patches to a repository
    - Run tests
    - Grade results
    """

    def __init__(self, workspace_dir: Path | None = None):
        """
        Initialize the evaluator.

        Args:
            workspace_dir: Directory for cloning repos and running tests.
                          If None, a temporary directory will be used.
        """
        self.workspace_dir = workspace_dir or Path(tempfile.mkdtemp())
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

    def load_instances(self, instances_file: Path) -> list[dict[str, Any]]:
        """
        Load benchmark instances from a JSON file.

        Args:
            instances_file: Path to the JSON file containing instances

        Returns:
            List of instance dictionaries
        """
        with open(instances_file) as f:
            instances = json.load(f)

        # Validate all instances
        for instance in instances:
            is_valid, errors = validate_instance(instance)
            if not is_valid:
                raise ValueError(
                    f"Invalid instance {instance.get('instance_id', 'unknown')}: {errors}"
                )

        return instances

    def load_predictions(self, predictions_file: Path) -> list[dict[str, Any]]:
        """
        Load model predictions from a JSON file.

        Args:
            predictions_file: Path to the JSON file containing predictions

        Returns:
            List of prediction dictionaries
        """
        with open(predictions_file) as f:
            return json.load(f)

    def setup_repo(self, instance: dict[str, Any]) -> Path:
        """
        Clone and set up the repository for an instance.

        Args:
            instance: The benchmark instance

        Returns:
            Path to the cloned repository
        """
        repo_name = instance["repo"].replace("/", "_")
        repo_path = self.workspace_dir / repo_name

        if repo_path.exists():
            shutil.rmtree(repo_path)

        # Clone the repository
        subprocess.run(
            ["git", "clone", f"https://github.com/{instance['repo']}.git", str(repo_path)],
            check=True,
            capture_output=True,
        )

        # Checkout the base commit
        subprocess.run(
            ["git", "checkout", instance["base_commit"]],
            cwd=repo_path,
            check=True,
            capture_output=True,
        )

        return repo_path

    def apply_patch(self, repo_path: Path, patch: str) -> bool:
        """
        Apply a patch to the repository.

        Args:
            repo_path: Path to the repository
            patch: The patch content as a string

        Returns:
            True if the patch was applied successfully
        """
        try:
            result = subprocess.run(
                ["git", "apply", "--check", "-"],
                input=patch.encode(),
                cwd=repo_path,
                capture_output=True,
            )

            if result.returncode != 0:
                print(f"Patch check failed: {result.stderr.decode()}")
                return False

            subprocess.run(
                ["git", "apply", "-"],
                input=patch.encode(),
                cwd=repo_path,
                check=True,
                capture_output=True,
            )
            return True

        except subprocess.CalledProcessError as e:
            print(f"Failed to apply patch: {e}")
            return False

    def run_tests(
        self,
        repo_path: Path,
        test_files: list[str] | None = None,
        timeout: int = 300,
    ) -> BunTestSummary:
        """
        Run bun tests in the repository.

        Args:
            repo_path: Path to the repository
            test_files: Optional list of specific test files to run
            timeout: Timeout in seconds for test execution

        Returns:
            Parsed test results
        """
        cmd = ["bun", "test"]
        if test_files:
            cmd.extend(test_files)

        try:
            result = subprocess.run(
                cmd,
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            output = result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            output = "Test execution timed out"
        except Exception as e:
            output = f"Test execution failed: {e}"

        return LogParser.parse_bun_test_output(output)

    def evaluate_instance(
        self,
        instance: dict[str, Any],
        prediction: dict[str, Any],
        cleanup: bool = True,
    ) -> dict[str, Any]:
        """
        Evaluate a single instance with a model's prediction.

        Args:
            instance: The benchmark instance
            prediction: The model's prediction (including patch)
            cleanup: Whether to clean up the repository after evaluation

        Returns:
            Evaluation result dictionary
        """
        result = {
            "instance_id": instance["instance_id"],
            "model": prediction.get("model_name_or_path", "unknown"),
            "setup_success": False,
            "patch_applied": False,
            "tests_run": False,
            "grade": None,
            "error": None,
        }

        try:
            # Set up the repository
            repo_path = self.setup_repo(instance)
            result["setup_success"] = True

            # Apply the patch
            patch_success = self.apply_patch(repo_path, prediction["model_patch"])
            result["patch_applied"] = patch_success

            if not patch_success:
                result["error"] = "Failed to apply patch"
                return result

            # Run tests
            test_results = self.run_tests(repo_path)
            result["tests_run"] = True
            result["test_summary"] = {
                "passed": test_results.passed,
                "failed": test_results.failed,
                "skipped": test_results.skipped,
                "total": test_results.total,
            }

            # Grade the results
            result["grade"] = grade_instance(instance, test_results)

        except Exception as e:
            result["error"] = str(e)

        finally:
            if cleanup and "repo_path" in dir():
                try:
                    shutil.rmtree(repo_path)
                except Exception:
                    pass

        return result

    def evaluate_all(
        self,
        instances: list[dict[str, Any]],
        predictions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Evaluate all instances with their corresponding predictions.

        Args:
            instances: List of benchmark instances
            predictions: List of model predictions

        Returns:
            List of evaluation results
        """
        # Create a mapping from instance_id to prediction
        prediction_map = {p["instance_id"]: p for p in predictions}

        results = []
        for instance in instances:
            instance_id = instance["instance_id"]
            prediction = prediction_map.get(instance_id)

            if prediction is None:
                results.append({
                    "instance_id": instance_id,
                    "error": "No prediction found for this instance",
                })
                continue

            result = self.evaluate_instance(instance, prediction)
            results.append(result)

        return results


def demo_parsing():
    """Demonstrate the log parser with sample output."""
    print("=" * 60)
    print("Demo: Parsing Bun Test Output")
    print("=" * 60)

    sample_output = """bun test v1.0.0

test/http/content-length.test.ts:
  should calculate correct byte length for UTF-8 ... [PASS] (2.31ms)
  should handle multi-byte characters ... [PASS] (1.45ms)
  should work for ASCII content ... [PASS] (0.89ms)

test/http/basic.test.ts:
  should start server ... [PASS] (5.21ms)
  should handle requests ... [FAIL] (3.12ms)
    Error: Expected 200 but got 500

 4 pass
 1 fail
 0 skip
 5 expect() calls

Ran 5 tests in 12.98ms
"""

    results = LogParser.parse_bun_test_output(sample_output)

    print(f"\nParsed Results:")
    print(f"  Passed: {results.passed}")
    print(f"  Failed: {results.failed}")
    print(f"  Skipped: {results.skipped}")
    print(f"  Total: {results.total}")
    print(f"  Pass Rate: {results.pass_rate:.1%}")
    print(f"  Duration: {results.total_duration_ms}ms")

    print(f"\nFailed Tests:")
    for test in results.get_failed_tests():
        print(f"  - {test.file_path}::{test.name}")
        if test.error_message:
            print(f"    Error: {test.error_message}")


def demo_grading():
    """Demonstrate the grading function."""
    print("\n" + "=" * 60)
    print("Demo: Grading Instance")
    print("=" * 60)

    # Create mock test results that match the FAIL_TO_PASS and PASS_TO_PASS
    from tests.test_harness import BunTestResult

    results = BunTestSummary()

    # All FAIL_TO_PASS tests now pass (bug is fixed)
    for test_name in SAMPLE_INSTANCE["FAIL_TO_PASS"]:
        parts = test_name.split("::")
        results.tests.append(BunTestResult(
            name=parts[1],
            status="PASS",
            file_path=parts[0],
            duration_ms=1.0,
        ))

    # All PASS_TO_PASS tests still pass (no regressions)
    for test_name in SAMPLE_INSTANCE["PASS_TO_PASS"]:
        parts = test_name.split("::")
        results.tests.append(BunTestResult(
            name=parts[1],
            status="PASS",
            file_path=parts[0],
            duration_ms=1.0,
        ))

    results.passed = len(results.tests)

    grade = grade_instance(SAMPLE_INSTANCE, results)

    print(f"\nGrading Results:")
    print(f"  Resolved: {grade['resolved']}")
    print(f"  FAIL_TO_PASS Score: {grade['fail_to_pass_score']:.1%}")
    print(f"  PASS_TO_PASS Score: {grade['pass_to_pass_score']:.1%}")
    print(f"  Total Tests Run: {grade['total_tests_run']}")


def demo_validation():
    """Demonstrate instance validation."""
    print("\n" + "=" * 60)
    print("Demo: Instance Validation")
    print("=" * 60)

    # Valid instance
    is_valid, errors = validate_instance(SAMPLE_INSTANCE)
    print(f"\nValid Instance:")
    print(f"  Is Valid: {is_valid}")
    print(f"  Errors: {errors}")

    # Invalid instance (missing field)
    invalid_instance = {"instance_id": "test", "repo": "test/repo"}
    is_valid, errors = validate_instance(invalid_instance)
    print(f"\nInvalid Instance (missing fields):")
    print(f"  Is Valid: {is_valid}")
    print(f"  Errors: {errors}")


def demo_full_evaluation():
    """Demonstrate full evaluation (without actual repo cloning)."""
    print("\n" + "=" * 60)
    print("Demo: Full Evaluation Pipeline (Simulated)")
    print("=" * 60)

    print(f"\nInstance ID: {SAMPLE_INSTANCE['instance_id']}")
    print(f"Problem: {SAMPLE_INSTANCE['problem_statement'][:100]}...")
    print(f"Model: {SAMPLE_PREDICTION['model_name_or_path']}")

    print("\nSteps that would be executed:")
    print("  1. Clone repository oven-sh/bun")
    print(f"  2. Checkout commit {SAMPLE_INSTANCE['base_commit']}")
    print("  3. Apply model's patch")
    print("  4. Run bun test")
    print("  5. Parse test results")
    print("  6. Grade against FAIL_TO_PASS and PASS_TO_PASS criteria")

    print("\nNote: Actual repo cloning skipped in demo mode.")
    print("      Use BunBenchEvaluator.evaluate_instance() for real evaluation.")


def main():
    """Main entry point for the demo."""
    print("\n" + "#" * 60)
    print("# Bun-Bench: Simple Evaluation Demo")
    print("#" * 60)

    # Run all demos
    demo_parsing()
    demo_grading()
    demo_validation()
    demo_full_evaluation()

    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)

    # Show how to run actual evaluation
    print("\nTo run actual evaluation:")
    print("""
    from examples.run_simple_eval import BunBenchEvaluator

    evaluator = BunBenchEvaluator()
    instances = evaluator.load_instances(Path("data/instances.json"))
    predictions = evaluator.load_predictions(Path("predictions/model_output.json"))
    results = evaluator.evaluate_all(instances, predictions)

    for result in results:
        print(f"{result['instance_id']}: Resolved={result['grade']['resolved']}")
    """)


if __name__ == "__main__":
    main()
