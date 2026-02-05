"""
Log parser for Bun test output.

This module provides functions to parse the output of `bun test` commands
and extract test results, including test names, pass/fail status, and error details.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from .constants import TestStatus, TestResult


@dataclass
class ParsedTestOutput:
    """Parsed results from bun test output."""

    tests: list[TestResult] = field(default_factory=list)
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    skipped_tests: int = 0
    total_duration_ms: float = 0.0
    raw_output: str = ""
    parse_errors: list[str] = field(default_factory=list)


# Regular expressions for parsing bun test output
# Bun test output format:
# ✓ test name [duration]
# ✗ test name [duration]
#   error message
# ○ test name (skipped)

# Pass pattern: ✓ or (pass)
PASS_PATTERN = re.compile(
    r'^[\s]*[✓✔][\s]+(.+?)(?:\s+\[([0-9.]+)\s*m?s\])?$',
    re.MULTILINE
)

# Fail pattern: ✗ or (fail)
FAIL_PATTERN = re.compile(
    r'^[\s]*[✗✘×][\s]+(.+?)(?:\s+\[([0-9.]+)\s*m?s\])?$',
    re.MULTILINE
)

# Skip pattern: ○ or (skip)
SKIP_PATTERN = re.compile(
    r'^[\s]*[○◯][\s]+(.+?)(?:\s+\(skipped\))?$',
    re.MULTILINE
)

# Alternative format: test name ... pass/fail
ALT_PASS_PATTERN = re.compile(
    r'^[\s]*(.+?)\s+\.{2,}\s+(?:pass|passed|ok)(?:\s+\[([0-9.]+)\s*m?s\])?$',
    re.MULTILINE | re.IGNORECASE
)

ALT_FAIL_PATTERN = re.compile(
    r'^[\s]*(.+?)\s+\.{2,}\s+(?:fail|failed|error)(?:\s+\[([0-9.]+)\s*m?s\])?$',
    re.MULTILINE | re.IGNORECASE
)

# Summary line pattern
SUMMARY_PATTERN = re.compile(
    r'(\d+)\s+pass(?:ed)?.*?(\d+)\s+fail(?:ed)?',
    re.IGNORECASE
)

# Duration pattern from summary
DURATION_PATTERN = re.compile(
    r'(?:total\s+)?(?:time|duration)[:\s]+([0-9.]+)\s*(ms|s|m)',
    re.IGNORECASE
)

# Error block pattern (indented error messages after failed tests)
ERROR_BLOCK_PATTERN = re.compile(
    r'^[\s]*[✗✘×][\s]+(.+?)(?:\s+\[.*?\])?\n((?:[\s]+.+\n?)+)',
    re.MULTILINE
)

# Describe block pattern for nested test names
DESCRIBE_PATTERN = re.compile(
    r'^[\s]*(?:describe|suite)[\s]+["\'](.+?)["\']',
    re.MULTILINE
)


def parse_bun_test_output(output: str) -> ParsedTestOutput:
    """
    Parse the output of a `bun test` command.

    Args:
        output: Raw string output from bun test command

    Returns:
        ParsedTestOutput containing parsed test results
    """
    result = ParsedTestOutput(raw_output=output)

    if not output or not output.strip():
        result.parse_errors.append("Empty output received")
        return result

    tests: list[TestResult] = []
    seen_tests: set[str] = set()

    # Parse passed tests
    for match in PASS_PATTERN.finditer(output):
        test_name = match.group(1).strip()
        duration_str = match.group(2)

        if test_name and test_name not in seen_tests:
            seen_tests.add(test_name)
            test_result: TestResult = {
                "name": test_name,
                "status": TestStatus.PASSED.value,
            }
            if duration_str:
                test_result["duration_ms"] = _parse_duration(duration_str)
            tests.append(test_result)

    # Parse failed tests
    for match in FAIL_PATTERN.finditer(output):
        test_name = match.group(1).strip()
        duration_str = match.group(2)

        if test_name and test_name not in seen_tests:
            seen_tests.add(test_name)
            test_result: TestResult = {
                "name": test_name,
                "status": TestStatus.FAILED.value,
            }
            if duration_str:
                test_result["duration_ms"] = _parse_duration(duration_str)
            tests.append(test_result)

    # Parse skipped tests
    for match in SKIP_PATTERN.finditer(output):
        test_name = match.group(1).strip()

        if test_name and test_name not in seen_tests:
            seen_tests.add(test_name)
            test_result: TestResult = {
                "name": test_name,
                "status": TestStatus.SKIPPED.value,
            }
            tests.append(test_result)

    # Try alternative format if no tests found
    if not tests:
        tests = _parse_alternative_format(output, seen_tests)

    # Extract error messages for failed tests
    _extract_error_messages(output, tests)

    result.tests = tests
    result.total_tests = len(tests)
    result.passed_tests = sum(1 for t in tests if t["status"] == TestStatus.PASSED.value)
    result.failed_tests = sum(1 for t in tests if t["status"] == TestStatus.FAILED.value)
    result.skipped_tests = sum(1 for t in tests if t["status"] == TestStatus.SKIPPED.value)

    # Try to extract total duration from summary
    duration_match = DURATION_PATTERN.search(output)
    if duration_match:
        result.total_duration_ms = _parse_duration_with_unit(
            duration_match.group(1),
            duration_match.group(2)
        )

    # Validate against summary if present
    summary_match = SUMMARY_PATTERN.search(output)
    if summary_match:
        expected_passed = int(summary_match.group(1))
        expected_failed = int(summary_match.group(2))

        if result.passed_tests != expected_passed:
            result.parse_errors.append(
                f"Parsed {result.passed_tests} passed tests, but summary shows {expected_passed}"
            )
        if result.failed_tests != expected_failed:
            result.parse_errors.append(
                f"Parsed {result.failed_tests} failed tests, but summary shows {expected_failed}"
            )

    return result


def _parse_alternative_format(output: str, seen_tests: set[str]) -> list[TestResult]:
    """Parse tests in alternative format (name ... pass/fail)."""
    tests: list[TestResult] = []

    for match in ALT_PASS_PATTERN.finditer(output):
        test_name = match.group(1).strip()
        duration_str = match.group(2)

        if test_name and test_name not in seen_tests:
            seen_tests.add(test_name)
            test_result: TestResult = {
                "name": test_name,
                "status": TestStatus.PASSED.value,
            }
            if duration_str:
                test_result["duration_ms"] = _parse_duration(duration_str)
            tests.append(test_result)

    for match in ALT_FAIL_PATTERN.finditer(output):
        test_name = match.group(1).strip()
        duration_str = match.group(2)

        if test_name and test_name not in seen_tests:
            seen_tests.add(test_name)
            test_result: TestResult = {
                "name": test_name,
                "status": TestStatus.FAILED.value,
            }
            if duration_str:
                test_result["duration_ms"] = _parse_duration(duration_str)
            tests.append(test_result)

    return tests


def _extract_error_messages(output: str, tests: list[TestResult]) -> None:
    """Extract and attach error messages to failed tests."""
    for match in ERROR_BLOCK_PATTERN.finditer(output):
        test_name = match.group(1).strip()
        error_block = match.group(2)

        # Find matching test and add error message
        for test in tests:
            if test["name"] == test_name and test["status"] == TestStatus.FAILED.value:
                error_lines = [line.strip() for line in error_block.split('\n') if line.strip()]
                if error_lines:
                    test["error_message"] = error_lines[0]
                    if len(error_lines) > 1:
                        test["stack_trace"] = '\n'.join(error_lines[1:])
                break


def _parse_duration(duration_str: str) -> float:
    """Parse duration string to milliseconds (assumes ms if no unit)."""
    try:
        return float(duration_str)
    except ValueError:
        return 0.0


def _parse_duration_with_unit(value: str, unit: str) -> float:
    """Parse duration with explicit unit to milliseconds."""
    try:
        num = float(value)
        unit = unit.lower()
        if unit == 's':
            return num * 1000
        elif unit == 'm':
            return num * 60 * 1000
        else:  # ms
            return num
    except ValueError:
        return 0.0


def extract_test_names(output: str) -> list[str]:
    """
    Extract just the test names from bun test output.

    Args:
        output: Raw string output from bun test command

    Returns:
        List of test names found in the output
    """
    parsed = parse_bun_test_output(output)
    return [test["name"] for test in parsed.tests]


def get_test_status_map(output: str) -> dict[str, TestStatus]:
    """
    Get a mapping of test names to their status.

    Args:
        output: Raw string output from bun test command

    Returns:
        Dictionary mapping test names to TestStatus
    """
    parsed = parse_bun_test_output(output)
    return {
        test["name"]: TestStatus(test["status"])
        for test in parsed.tests
    }


def filter_tests_by_status(
    output: str,
    status: TestStatus
) -> list[str]:
    """
    Get test names filtered by status.

    Args:
        output: Raw string output from bun test command
        status: TestStatus to filter by

    Returns:
        List of test names with the specified status
    """
    parsed = parse_bun_test_output(output)
    return [
        test["name"]
        for test in parsed.tests
        if test["status"] == status.value
    ]


def get_failed_test_details(output: str) -> list[TestResult]:
    """
    Get detailed information about failed tests.

    Args:
        output: Raw string output from bun test command

    Returns:
        List of TestResult dicts for failed tests with error details
    """
    parsed = parse_bun_test_output(output)
    return [
        test for test in parsed.tests
        if test["status"] == TestStatus.FAILED.value
    ]
