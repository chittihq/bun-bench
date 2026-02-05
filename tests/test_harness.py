"""
Test harness for bun-bench evaluation framework.

Tests the core functionality including:
- Bun test output parsing
- Grading functions
- Instance validation
"""

import pytest
import json
from pathlib import Path
from typing import Any
import re


# -----------------------------------------------------------------------------
# Log Parser Module (to be tested)
# -----------------------------------------------------------------------------

class BunTestResult:
    """Represents the result of a single test case."""

    def __init__(
        self,
        name: str,
        status: str,
        duration_ms: float = 0.0,
        error_message: str | None = None,
        file_path: str | None = None,
    ):
        self.name = name
        self.status = status  # PASS, FAIL, SKIP
        self.duration_ms = duration_ms
        self.error_message = error_message
        self.file_path = file_path

    def __repr__(self) -> str:
        return f"BunTestResult(name={self.name!r}, status={self.status!r})"


class BunTestSummary:
    """Summary of all test results from a bun test run."""

    def __init__(self):
        self.tests: list[BunTestResult] = []
        self.passed: int = 0
        self.failed: int = 0
        self.skipped: int = 0
        self.total_duration_ms: float = 0.0
        self.expect_calls: int = 0

    @property
    def total(self) -> int:
        return self.passed + self.failed + self.skipped

    @property
    def pass_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.passed / self.total

    def get_failed_tests(self) -> list[BunTestResult]:
        return [t for t in self.tests if t.status == "FAIL"]

    def get_passed_tests(self) -> list[BunTestResult]:
        return [t for t in self.tests if t.status == "PASS"]

    def get_skipped_tests(self) -> list[BunTestResult]:
        return [t for t in self.tests if t.status == "SKIP"]


class LogParser:
    """Parser for bun test output."""

    # Regex patterns for parsing
    TEST_FILE_PATTERN = re.compile(r"^(test/[^\s:]+\.test\.ts):$", re.MULTILINE)
    TEST_RESULT_PATTERN = re.compile(
        r"^\s+(.+?)\s+\.\.\.\s+\[(PASS|FAIL|SKIP)\]\s+\((\d+\.?\d*)ms\)",
        re.MULTILINE,
    )
    ERROR_PATTERN = re.compile(r"^\s+Error:\s+(.+)$", re.MULTILINE)
    SUMMARY_PATTERN = re.compile(
        r"^\s*(\d+)\s+pass\s*\n\s*(\d+)\s+fail\s*\n\s*(\d+)\s+skip",
        re.MULTILINE,
    )
    EXPECT_PATTERN = re.compile(r"(\d+)\s+expect\(\)\s+calls", re.MULTILINE)
    DURATION_PATTERN = re.compile(r"Ran\s+\d+\s+tests?\s+in\s+(\d+\.?\d*)ms")

    @classmethod
    def parse_bun_test_output(cls, output: str) -> BunTestSummary:
        """
        Parse bun test output and return a structured summary.

        Args:
            output: Raw output from bun test command

        Returns:
            BunTestSummary object containing parsed results
        """
        summary = BunTestSummary()
        current_file = None
        lines = output.split("\n")

        i = 0
        while i < len(lines):
            line = lines[i]

            # Check for test file header
            file_match = cls.TEST_FILE_PATTERN.match(line)
            if file_match:
                current_file = file_match.group(1)
                i += 1
                continue

            # Check for test result
            result_match = cls.TEST_RESULT_PATTERN.match(line)
            if result_match:
                name = result_match.group(1).strip()
                status = result_match.group(2)
                duration = float(result_match.group(3))

                error_message = None
                # Look for error message on subsequent lines if test failed
                if status == "FAIL":
                    j = i + 1
                    while j < len(lines):
                        error_match = cls.ERROR_PATTERN.match(lines[j])
                        if error_match:
                            error_message = error_match.group(1)
                            break
                        # Stop if we hit another test result or file header
                        if cls.TEST_RESULT_PATTERN.match(lines[j]) or cls.TEST_FILE_PATTERN.match(lines[j]):
                            break
                        j += 1

                test_result = BunTestResult(
                    name=name,
                    status=status,
                    duration_ms=duration,
                    error_message=error_message,
                    file_path=current_file,
                )
                summary.tests.append(test_result)
                i += 1
                continue

            i += 1

        # Parse summary statistics
        summary_match = cls.SUMMARY_PATTERN.search(output)
        if summary_match:
            summary.passed = int(summary_match.group(1))
            summary.failed = int(summary_match.group(2))
            summary.skipped = int(summary_match.group(3))

        # Parse expect calls
        expect_match = cls.EXPECT_PATTERN.search(output)
        if expect_match:
            summary.expect_calls = int(expect_match.group(1))

        # Parse total duration
        duration_match = cls.DURATION_PATTERN.search(output)
        if duration_match:
            summary.total_duration_ms = float(duration_match.group(1))

        return summary


# -----------------------------------------------------------------------------
# Grading Functions (to be tested)
# -----------------------------------------------------------------------------

def grade_instance(
    instance: dict[str, Any],
    test_results: BunTestSummary,
) -> dict[str, Any]:
    """
    Grade a model's solution based on test results.

    Args:
        instance: The benchmark instance with FAIL_TO_PASS and PASS_TO_PASS
        test_results: Parsed test results from running bun test

    Returns:
        Grading result with scores and details
    """
    fail_to_pass = set(instance.get("FAIL_TO_PASS", []))
    pass_to_pass = set(instance.get("PASS_TO_PASS", []))

    # Build a map of test name to result
    test_map = {}
    for test in test_results.tests:
        full_name = f"{test.file_path}::{test.name}" if test.file_path else test.name
        test_map[full_name] = test

    # Check FAIL_TO_PASS: tests that should now pass
    f2p_passed = 0
    f2p_results = []
    for test_name in fail_to_pass:
        result = test_map.get(test_name)
        if result and result.status == "PASS":
            f2p_passed += 1
            f2p_results.append({"test": test_name, "status": "PASS", "correct": True})
        else:
            status = result.status if result else "NOT_FOUND"
            f2p_results.append({"test": test_name, "status": status, "correct": False})

    # Check PASS_TO_PASS: tests that should still pass (no regressions)
    p2p_passed = 0
    p2p_results = []
    for test_name in pass_to_pass:
        result = test_map.get(test_name)
        if result and result.status == "PASS":
            p2p_passed += 1
            p2p_results.append({"test": test_name, "status": "PASS", "correct": True})
        else:
            status = result.status if result else "NOT_FOUND"
            p2p_results.append({"test": test_name, "status": status, "correct": False})

    # Calculate scores
    f2p_score = f2p_passed / len(fail_to_pass) if fail_to_pass else 1.0
    p2p_score = p2p_passed / len(pass_to_pass) if pass_to_pass else 1.0

    # Overall success: all FAIL_TO_PASS tests pass AND no regressions
    resolved = f2p_passed == len(fail_to_pass) and p2p_passed == len(pass_to_pass)

    return {
        "resolved": resolved,
        "fail_to_pass_score": f2p_score,
        "pass_to_pass_score": p2p_score,
        "fail_to_pass_results": f2p_results,
        "pass_to_pass_results": p2p_results,
        "total_tests_run": test_results.total,
        "test_pass_rate": test_results.pass_rate,
    }


def validate_instance(instance: dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Validate that an instance has all required fields.

    Args:
        instance: The benchmark instance to validate

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    required_fields = [
        "instance_id",
        "repo",
        "base_commit",
        "problem_statement",
        "FAIL_TO_PASS",
        "PASS_TO_PASS",
    ]

    errors = []

    for field in required_fields:
        if field not in instance:
            errors.append(f"Missing required field: {field}")

    # Validate types
    if "instance_id" in instance and not isinstance(instance["instance_id"], str):
        errors.append("instance_id must be a string")

    if "FAIL_TO_PASS" in instance and not isinstance(instance["FAIL_TO_PASS"], list):
        errors.append("FAIL_TO_PASS must be a list")

    if "PASS_TO_PASS" in instance and not isinstance(instance["PASS_TO_PASS"], list):
        errors.append("PASS_TO_PASS must be a list")

    if "FAIL_TO_PASS" in instance and isinstance(instance["FAIL_TO_PASS"], list):
        if len(instance["FAIL_TO_PASS"]) == 0:
            errors.append("FAIL_TO_PASS must not be empty")

    return len(errors) == 0, errors


# -----------------------------------------------------------------------------
# Tests for LogParser
# -----------------------------------------------------------------------------

class TestLogParser:
    """Tests for the bun test output parser."""

    def test_parse_basic_output(self, sample_bun_test_output: str):
        """Test parsing of basic bun test output."""
        result = LogParser.parse_bun_test_output(sample_bun_test_output)

        assert result.passed == 8
        assert result.failed == 1
        assert result.skipped == 1
        assert result.total == 10

    def test_parse_all_pass(self, sample_bun_test_output_all_pass: str):
        """Test parsing when all tests pass."""
        result = LogParser.parse_bun_test_output(sample_bun_test_output_all_pass)

        assert result.passed == 3
        assert result.failed == 0
        assert result.skipped == 0
        assert result.pass_rate == 1.0

    def test_parse_all_fail(self, sample_bun_test_output_all_fail: str):
        """Test parsing when all tests fail."""
        result = LogParser.parse_bun_test_output(sample_bun_test_output_all_fail)

        assert result.passed == 0
        assert result.failed == 3
        assert result.skipped == 0
        assert result.pass_rate == 0.0

    def test_parse_test_details(self, sample_bun_test_output: str):
        """Test that individual test details are captured correctly."""
        result = LogParser.parse_bun_test_output(sample_bun_test_output)

        passed_tests = result.get_passed_tests()
        assert len(passed_tests) > 0

        failed_tests = result.get_failed_tests()
        assert len(failed_tests) > 0
        assert failed_tests[0].error_message is not None

    def test_parse_expect_calls(self, sample_bun_test_output: str):
        """Test parsing of expect() call count."""
        result = LogParser.parse_bun_test_output(sample_bun_test_output)
        assert result.expect_calls == 10  # fixture has 10 expect() calls

    def test_parse_empty_output(self):
        """Test parsing of empty output."""
        result = LogParser.parse_bun_test_output("")
        assert result.total == 0
        assert result.passed == 0
        assert result.failed == 0

    def test_parse_from_file(self, test_data_dir: Path):
        """Test parsing from the sample file."""
        sample_file = test_data_dir / "sample_bun_output.txt"
        if sample_file.exists():
            output = sample_file.read_text()
            result = LogParser.parse_bun_test_output(output)

            assert result.passed == 28
            assert result.failed == 3
            assert result.skipped == 2
            assert result.total == 33


# -----------------------------------------------------------------------------
# Tests for Grading Functions
# -----------------------------------------------------------------------------

class TestGrading:
    """Tests for the grading functions."""

    def test_grade_fully_resolved(self, sample_instance: dict):
        """Test grading when all required tests pass."""
        # Create test results where all tests pass
        summary = BunTestSummary()
        for test_name in sample_instance["FAIL_TO_PASS"] + sample_instance["PASS_TO_PASS"]:
            parts = test_name.split("::")
            file_path = parts[0] if len(parts) > 1 else None
            name = parts[1] if len(parts) > 1 else parts[0]
            summary.tests.append(BunTestResult(
                name=name,
                status="PASS",
                file_path=file_path,
                duration_ms=1.0,
            ))
        summary.passed = len(summary.tests)

        result = grade_instance(sample_instance, summary)

        assert result["resolved"] is True
        assert result["fail_to_pass_score"] == 1.0
        assert result["pass_to_pass_score"] == 1.0

    def test_grade_partially_resolved(self, sample_instance: dict):
        """Test grading when only some FAIL_TO_PASS tests pass."""
        summary = BunTestSummary()

        # Only first FAIL_TO_PASS test passes
        f2p_tests = sample_instance["FAIL_TO_PASS"]
        parts = f2p_tests[0].split("::")
        summary.tests.append(BunTestResult(
            name=parts[1],
            status="PASS",
            file_path=parts[0],
            duration_ms=1.0,
        ))

        # Second FAIL_TO_PASS test fails
        parts = f2p_tests[1].split("::")
        summary.tests.append(BunTestResult(
            name=parts[1],
            status="FAIL",
            file_path=parts[0],
            duration_ms=1.0,
        ))

        # All PASS_TO_PASS tests pass
        for test_name in sample_instance["PASS_TO_PASS"]:
            parts = test_name.split("::")
            summary.tests.append(BunTestResult(
                name=parts[1] if len(parts) > 1 else parts[0],
                status="PASS",
                file_path=parts[0] if len(parts) > 1 else None,
                duration_ms=1.0,
            ))

        summary.passed = len([t for t in summary.tests if t.status == "PASS"])
        summary.failed = len([t for t in summary.tests if t.status == "FAIL"])

        result = grade_instance(sample_instance, summary)

        assert result["resolved"] is False
        assert result["fail_to_pass_score"] == 0.5
        assert result["pass_to_pass_score"] == 1.0

    def test_grade_with_regression(self, sample_instance: dict):
        """Test grading when a PASS_TO_PASS test fails (regression)."""
        summary = BunTestSummary()

        # All FAIL_TO_PASS tests pass
        for test_name in sample_instance["FAIL_TO_PASS"]:
            parts = test_name.split("::")
            summary.tests.append(BunTestResult(
                name=parts[1] if len(parts) > 1 else parts[0],
                status="PASS",
                file_path=parts[0] if len(parts) > 1 else None,
                duration_ms=1.0,
            ))

        # First PASS_TO_PASS test regresses (fails)
        p2p_tests = sample_instance["PASS_TO_PASS"]
        parts = p2p_tests[0].split("::")
        summary.tests.append(BunTestResult(
            name=parts[1] if len(parts) > 1 else parts[0],
            status="FAIL",
            file_path=parts[0] if len(parts) > 1 else None,
            duration_ms=1.0,
        ))

        # Second PASS_TO_PASS test passes
        parts = p2p_tests[1].split("::")
        summary.tests.append(BunTestResult(
            name=parts[1] if len(parts) > 1 else parts[0],
            status="PASS",
            file_path=parts[0] if len(parts) > 1 else None,
            duration_ms=1.0,
        ))

        summary.passed = len([t for t in summary.tests if t.status == "PASS"])
        summary.failed = len([t for t in summary.tests if t.status == "FAIL"])

        result = grade_instance(sample_instance, summary)

        assert result["resolved"] is False
        assert result["fail_to_pass_score"] == 1.0
        assert result["pass_to_pass_score"] == 0.5

    def test_grade_empty_test_results(self, sample_instance: dict):
        """Test grading with no test results."""
        summary = BunTestSummary()
        result = grade_instance(sample_instance, summary)

        assert result["resolved"] is False
        assert result["fail_to_pass_score"] == 0.0
        assert result["pass_to_pass_score"] == 0.0


# -----------------------------------------------------------------------------
# Tests for Instance Validation
# -----------------------------------------------------------------------------

class TestValidation:
    """Tests for instance validation."""

    def test_validate_valid_instance(self, sample_instance: dict):
        """Test validation of a valid instance."""
        is_valid, errors = validate_instance(sample_instance)
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_missing_required_field(self, sample_instance: dict):
        """Test validation when a required field is missing."""
        del sample_instance["instance_id"]
        is_valid, errors = validate_instance(sample_instance)
        assert is_valid is False
        assert "Missing required field: instance_id" in errors

    def test_validate_wrong_type(self, sample_instance: dict):
        """Test validation when a field has wrong type."""
        sample_instance["FAIL_TO_PASS"] = "not a list"
        is_valid, errors = validate_instance(sample_instance)
        assert is_valid is False
        assert "FAIL_TO_PASS must be a list" in errors

    def test_validate_empty_fail_to_pass(self, sample_instance: dict):
        """Test validation when FAIL_TO_PASS is empty."""
        sample_instance["FAIL_TO_PASS"] = []
        is_valid, errors = validate_instance(sample_instance)
        assert is_valid is False
        assert "FAIL_TO_PASS must not be empty" in errors

    def test_validate_minimal_instance(self):
        """Test validation with minimal valid instance."""
        minimal = {
            "instance_id": "test_1",
            "repo": "test/repo",
            "base_commit": "abc123",
            "problem_statement": "Fix the bug",
            "FAIL_TO_PASS": ["test1"],
            "PASS_TO_PASS": [],
        }
        is_valid, errors = validate_instance(minimal)
        assert is_valid is True


# -----------------------------------------------------------------------------
# Integration Tests
# -----------------------------------------------------------------------------

class TestIntegration:
    """Integration tests for the full evaluation pipeline."""

    def test_full_pipeline(self, sample_instance: dict, sample_bun_test_output: str):
        """Test the full evaluation pipeline from parsing to grading."""
        # Parse the test output
        test_results = LogParser.parse_bun_test_output(sample_bun_test_output)

        # Validate the instance
        is_valid, errors = validate_instance(sample_instance)
        assert is_valid is True

        # Grade the instance
        grade_result = grade_instance(sample_instance, test_results)

        # Verify the result structure
        assert "resolved" in grade_result
        assert "fail_to_pass_score" in grade_result
        assert "pass_to_pass_score" in grade_result
        assert isinstance(grade_result["resolved"], bool)

    def test_load_and_validate_instances(self, sample_instances_file: Path):
        """Test loading and validating instances from a file."""
        with open(sample_instances_file) as f:
            instances = json.load(f)

        for instance in instances:
            is_valid, errors = validate_instance(instance)
            assert is_valid is True, f"Instance {instance.get('instance_id')} invalid: {errors}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
