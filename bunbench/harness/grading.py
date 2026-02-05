"""
Grading functions for Bun-Bench evaluation.

This module provides functions to:
- Parse and analyze bun test output
- Grade FAIL_TO_PASS and PASS_TO_PASS test results
- Determine resolution status (FULL/PARTIAL/NO)
- Generate evaluation reports
"""

from dataclasses import dataclass, field

from .constants import (
    BunBenchInstance,
    EvaluationResult,
    ResolvedStatus,
    TestResult,
    TestStatus,
)
from .log_parser import ParsedTestOutput, parse_bun_test_output


@dataclass
class GradingResult:
    """Result of grading a single test category."""

    expected_tests: list[str]
    actual_results: list[TestResult]
    passed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        """Check if all expected tests passed."""
        return len(self.passed) == len(self.expected_tests) and len(self.failed) == 0

    @property
    def any_passed(self) -> bool:
        """Check if any expected tests passed."""
        return len(self.passed) > 0

    @property
    def pass_rate(self) -> float:
        """Calculate pass rate as a fraction."""
        if not self.expected_tests:
            return 1.0
        return len(self.passed) / len(self.expected_tests)


def grade_tests(
    expected_tests: list[str], parsed_output: ParsedTestOutput, require_pass: bool = True
) -> GradingResult:
    """
    Grade a set of tests against parsed output.

    Args:
        expected_tests: List of test names that should be checked
        parsed_output: Parsed output from bun test
        require_pass: If True, tests must pass; if False, tests must exist

    Returns:
        GradingResult with categorized test outcomes
    """
    result = GradingResult(
        expected_tests=expected_tests,
        actual_results=parsed_output.tests,
    )

    # Build a map of test name to result for lookup
    test_map: dict[str, TestResult] = {}
    for test in parsed_output.tests:
        test_map[test["name"]] = test
        # Also map normalized names (lowercase, stripped)
        normalized = _normalize_test_name(test["name"])
        if normalized not in test_map:
            test_map[normalized] = test

    for expected in expected_tests:
        # Try exact match first
        test_result = test_map.get(expected)

        # Try normalized match
        if test_result is None:
            normalized = _normalize_test_name(expected)
            test_result = test_map.get(normalized)

        # Try partial match (test name contains expected)
        if test_result is None:
            test_result = _find_partial_match(expected, parsed_output.tests)

        if test_result is None:
            result.missing.append(expected)
        elif require_pass:
            if test_result["status"] == TestStatus.PASSED.value:
                result.passed.append(expected)
            else:
                result.failed.append(expected)
        else:
            # Just checking existence
            result.passed.append(expected)

    return result


def grade_fail_to_pass(
    fail_to_pass_tests: list[str], parsed_output: ParsedTestOutput
) -> GradingResult:
    """
    Grade FAIL_TO_PASS tests.

    These are tests that should fail before the patch and pass after.
    We check that they pass in the post-patch evaluation.

    Args:
        fail_to_pass_tests: List of test names expected to pass after patch
        parsed_output: Parsed output from bun test after applying patch

    Returns:
        GradingResult indicating which FAIL_TO_PASS tests now pass
    """
    return grade_tests(fail_to_pass_tests, parsed_output, require_pass=True)


def grade_pass_to_pass(
    pass_to_pass_tests: list[str], parsed_output: ParsedTestOutput
) -> GradingResult:
    """
    Grade PASS_TO_PASS tests.

    These are tests that should pass both before and after the patch.
    We check that they still pass to ensure no regressions.

    Args:
        pass_to_pass_tests: List of test names expected to continue passing
        parsed_output: Parsed output from bun test after applying patch

    Returns:
        GradingResult indicating which PASS_TO_PASS tests still pass
    """
    return grade_tests(pass_to_pass_tests, parsed_output, require_pass=True)


def determine_resolution_status(
    fail_to_pass_result: GradingResult, pass_to_pass_result: GradingResult
) -> ResolvedStatus:
    """
    Determine the overall resolution status based on grading results.

    Resolution status is determined as follows:
    - FULL: All FAIL_TO_PASS tests pass AND all PASS_TO_PASS tests pass
    - PARTIAL: Some FAIL_TO_PASS tests pass AND all PASS_TO_PASS tests pass
    - NO: No FAIL_TO_PASS tests pass OR any PASS_TO_PASS tests fail

    Args:
        fail_to_pass_result: Grading result for FAIL_TO_PASS tests
        pass_to_pass_result: Grading result for PASS_TO_PASS tests

    Returns:
        ResolvedStatus enum value
    """
    # If any PASS_TO_PASS tests fail, it's a regression - NO resolution
    if not pass_to_pass_result.all_passed:
        return ResolvedStatus.NO

    # If no FAIL_TO_PASS tests are defined, treat as FULL if PASS_TO_PASS all pass
    if not fail_to_pass_result.expected_tests:
        return ResolvedStatus.FULL

    # Check FAIL_TO_PASS results
    if fail_to_pass_result.all_passed:
        return ResolvedStatus.FULL
    elif fail_to_pass_result.any_passed:
        return ResolvedStatus.PARTIAL
    else:
        return ResolvedStatus.NO


def get_eval_report(
    instance: BunBenchInstance,
    test_output: str,
    model_patch: str | None = None,
    total_duration_ms: float | None = None,
    error: str | None = None,
) -> EvaluationResult:
    """
    Generate a complete evaluation report for an instance.

    This function parses the test output, grades both FAIL_TO_PASS and
    PASS_TO_PASS tests, determines the resolution status, and returns
    a comprehensive evaluation result.

    Args:
        instance: The BunBenchInstance being evaluated
        test_output: Raw output from running bun test
        model_patch: Optional patch generated by the model (for reference)
        total_duration_ms: Optional total evaluation duration
        error: Optional error message if evaluation failed

    Returns:
        EvaluationResult with complete grading information
    """
    # Handle error case
    if error:
        return EvaluationResult(
            instance_id=instance["instance_id"],
            resolved=ResolvedStatus.NO.value,
            fail_to_pass_results=[],
            pass_to_pass_results=[],
            fail_to_pass_passed=0,
            fail_to_pass_total=len(instance.get("FAIL_TO_PASS", [])),
            pass_to_pass_passed=0,
            pass_to_pass_total=len(instance.get("PASS_TO_PASS", [])),
            error=error,
        )

    # Parse test output
    parsed = parse_bun_test_output(test_output)

    # Grade both categories
    fail_to_pass_tests = instance.get("FAIL_TO_PASS", [])
    pass_to_pass_tests = instance.get("PASS_TO_PASS", [])

    f2p_result = grade_fail_to_pass(fail_to_pass_tests, parsed)
    p2p_result = grade_pass_to_pass(pass_to_pass_tests, parsed)

    # Determine resolution status
    resolved = determine_resolution_status(f2p_result, p2p_result)

    # Build test results for report
    f2p_test_results = _build_test_results(fail_to_pass_tests, f2p_result, parsed)
    p2p_test_results = _build_test_results(pass_to_pass_tests, p2p_result, parsed)

    result: EvaluationResult = {
        "instance_id": instance["instance_id"],
        "resolved": resolved.value,
        "fail_to_pass_results": f2p_test_results,
        "pass_to_pass_results": p2p_test_results,
        "fail_to_pass_passed": len(f2p_result.passed),
        "fail_to_pass_total": len(fail_to_pass_tests),
        "pass_to_pass_passed": len(p2p_result.passed),
        "pass_to_pass_total": len(pass_to_pass_tests),
    }

    if total_duration_ms is not None:
        result["total_duration_ms"] = total_duration_ms

    # Add parse errors as warnings if any
    if parsed.parse_errors:
        result["error"] = f"Parse warnings: {'; '.join(parsed.parse_errors)}"

    return result


def _build_test_results(
    expected_tests: list[str], grading_result: GradingResult, parsed: ParsedTestOutput
) -> list[TestResult]:
    """Build detailed test results for the evaluation report."""
    results: list[TestResult] = []

    # Build map of actual results
    actual_map: dict[str, TestResult] = {test["name"]: test for test in parsed.tests}

    for test_name in expected_tests:
        # Try to find actual result
        actual = actual_map.get(test_name)

        if actual is None:
            # Try normalized lookup
            normalized = _normalize_test_name(test_name)
            for name, result in actual_map.items():
                if _normalize_test_name(name) == normalized:
                    actual = result
                    break

        if actual is None:
            # Test was missing from output
            results.append(
                {
                    "name": test_name,
                    "status": TestStatus.ERROR.value,
                    "error_message": "Test not found in output",
                }
            )
        else:
            results.append(actual)

    return results


def _normalize_test_name(name: str) -> str:
    """Normalize test name for comparison."""
    # Remove common prefixes and suffixes
    name = name.strip()
    name = name.lower()
    # Remove "test" prefix if present
    if name.startswith("test "):
        name = name[5:]
    # Remove quotes
    name = name.replace('"', "").replace("'", "")
    # Normalize whitespace
    name = " ".join(name.split())
    return name


def _find_partial_match(expected: str, actual_tests: list[TestResult]) -> TestResult | None:
    """Find a partial match for a test name."""
    expected_normalized = _normalize_test_name(expected)
    expected_words = set(expected_normalized.split())

    best_match: TestResult | None = None
    best_score = 0.0

    for test in actual_tests:
        actual_normalized = _normalize_test_name(test["name"])

        # Check if expected is contained in actual
        if expected_normalized in actual_normalized:
            return test

        # Check if actual is contained in expected
        if actual_normalized in expected_normalized:
            return test

        # Calculate word overlap score
        actual_words = set(actual_normalized.split())
        overlap = len(expected_words & actual_words)
        total = len(expected_words | actual_words)

        if total > 0:
            score = overlap / total
            if score > best_score and score > 0.5:  # At least 50% overlap
                best_score = score
                best_match = test

    return best_match


def summarize_evaluation(result: EvaluationResult) -> str:
    """
    Generate a human-readable summary of an evaluation result.

    Args:
        result: EvaluationResult to summarize

    Returns:
        Formatted string summary
    """
    lines = [
        f"Instance: {result['instance_id']}",
        f"Resolution: {result['resolved'].upper()}",
        "",
        f"FAIL_TO_PASS: {result['fail_to_pass_passed']}/{result['fail_to_pass_total']} passed",
        f"PASS_TO_PASS: {result['pass_to_pass_passed']}/{result['pass_to_pass_total']} passed",
    ]

    if "total_duration_ms" in result:
        duration_s = result["total_duration_ms"] / 1000
        lines.append(f"Duration: {duration_s:.2f}s")

    if "error" in result:
        lines.append(f"Error: {result['error']}")

    # Add failed test details
    failed_f2p = [
        t for t in result["fail_to_pass_results"] if t["status"] != TestStatus.PASSED.value
    ]
    if failed_f2p:
        lines.append("")
        lines.append("Failed FAIL_TO_PASS tests:")
        for test in failed_f2p:
            lines.append(f"  - {test['name']}: {test['status']}")
            if "error_message" in test:
                lines.append(f"    Error: {test['error_message']}")

    failed_p2p = [
        t for t in result["pass_to_pass_results"] if t["status"] != TestStatus.PASSED.value
    ]
    if failed_p2p:
        lines.append("")
        lines.append("Failed PASS_TO_PASS tests (regressions):")
        for test in failed_p2p:
            lines.append(f"  - {test['name']}: {test['status']}")
            if "error_message" in test:
                lines.append(f"    Error: {test['error_message']}")

    return "\n".join(lines)
