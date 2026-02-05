"""
Reporting module for Bun-Bench evaluations.

This module provides functionality for generating evaluation reports,
aggregating statistics, and saving results.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from bunbench.harness.run_evaluation import EvaluationResult, EvaluationStatus


@dataclass
class InstanceReport:
    """Detailed report for a single instance.

    Attributes:
        instance_id: Unique identifier for the instance.
        status: Final status of the evaluation.
        patch_applied: Whether the patch was successfully applied.
        tests_passed: Number of tests passed.
        tests_failed: Number of tests failed.
        tests_skipped: Number of tests skipped.
        tests_total: Total number of tests.
        duration: Time taken in seconds.
        error_message: Error message if evaluation failed.
        test_output: Raw test output (truncated if too long).
    """
    instance_id: str
    status: str
    patch_applied: bool = False
    tests_passed: int = 0
    tests_failed: int = 0
    tests_skipped: int = 0
    tests_total: int = 0
    duration: float = 0.0
    error_message: Optional[str] = None
    test_output: Optional[str] = None

    @classmethod
    def from_evaluation_result(
        cls,
        result: EvaluationResult,
        max_output_length: int = 10000
    ) -> "InstanceReport":
        """Create an InstanceReport from an EvaluationResult.

        Args:
            result: The evaluation result to convert.
            max_output_length: Maximum length of test output to include.

        Returns:
            InstanceReport with data from the evaluation result.
        """
        test_output = None
        tests_passed = 0
        tests_failed = 0
        tests_skipped = 0
        tests_total = 0

        if result.test_result:
            tests_passed = result.test_result.passed
            tests_failed = result.test_result.failed
            tests_skipped = result.test_result.skipped
            tests_total = result.test_result.total

            if result.test_result.output:
                test_output = result.test_result.output
                if len(test_output) > max_output_length:
                    test_output = (
                        test_output[:max_output_length // 2] +
                        "\n... [truncated] ...\n" +
                        test_output[-max_output_length // 2:]
                    )

        return cls(
            instance_id=result.instance_id,
            status=result.status.value,
            patch_applied=result.patch_applied,
            tests_passed=tests_passed,
            tests_failed=tests_failed,
            tests_skipped=tests_skipped,
            tests_total=tests_total,
            duration=result.duration,
            error_message=result.error_message,
            test_output=test_output,
        )


@dataclass
class ReportSummary:
    """Summary statistics for an evaluation run.

    Attributes:
        total: Total number of instances evaluated.
        resolved: Number of instances successfully resolved.
        unresolved: Number of instances that failed tests.
        errors: Number of instances with errors.
        skipped: Number of instances skipped.
        resolved_rate: Percentage of resolved instances.
        avg_duration: Average evaluation duration in seconds.
        total_duration: Total evaluation duration in seconds.
        timestamp: When the evaluation was run.
        config: Configuration used for the evaluation.
        instances: Detailed reports for each instance.
    """
    total: int = 0
    resolved: int = 0
    unresolved: int = 0
    errors: int = 0
    skipped: int = 0
    resolved_rate: float = 0.0
    avg_duration: float = 0.0
    total_duration: float = 0.0
    timestamp: str = ""
    config: Dict[str, Any] = field(default_factory=dict)
    instances: List[InstanceReport] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for JSON serialization."""
        return {
            "summary": {
                "total": self.total,
                "resolved": self.resolved,
                "unresolved": self.unresolved,
                "errors": self.errors,
                "skipped": self.skipped,
                "resolved_rate": self.resolved_rate,
                "avg_duration": self.avg_duration,
                "total_duration": self.total_duration,
            },
            "metadata": {
                "timestamp": self.timestamp,
                "config": self.config,
            },
            "instances": [asdict(inst) for inst in self.instances],
        }


def generate_report(
    results: List[EvaluationResult],
    config: Optional[Dict[str, Any]] = None,
    include_test_output: bool = True,
    max_output_length: int = 10000,
) -> ReportSummary:
    """Generate an evaluation report from results.

    Args:
        results: List of evaluation results.
        config: Optional configuration dict to include in report.
        include_test_output: Whether to include raw test output.
        max_output_length: Maximum length of test output per instance.

    Returns:
        ReportSummary with aggregated statistics and instance details.
    """
    report = ReportSummary(
        timestamp=datetime.utcnow().isoformat() + "Z",
        config=config or {},
    )

    total_duration = 0.0

    for result in results:
        report.total += 1
        total_duration += result.duration

        if result.status == EvaluationStatus.RESOLVED:
            report.resolved += 1
        elif result.status == EvaluationStatus.UNRESOLVED:
            report.unresolved += 1
        elif result.status == EvaluationStatus.ERROR:
            report.errors += 1
        elif result.status == EvaluationStatus.SKIPPED:
            report.skipped += 1

        # Create instance report
        instance_report = InstanceReport.from_evaluation_result(
            result,
            max_output_length=max_output_length if include_test_output else 0
        )

        if not include_test_output:
            instance_report.test_output = None

        report.instances.append(instance_report)

    # Calculate aggregate statistics
    report.total_duration = total_duration

    if report.total > 0:
        report.avg_duration = total_duration / report.total
        # Calculate resolved rate excluding skipped instances
        evaluated = report.total - report.skipped
        if evaluated > 0:
            report.resolved_rate = report.resolved / evaluated
        else:
            report.resolved_rate = 0.0
    else:
        report.avg_duration = 0.0
        report.resolved_rate = 0.0

    return report


def save_report(
    report: ReportSummary,
    output_path: str,
    indent: int = 2,
) -> str:
    """Save evaluation report to JSON file.

    Args:
        report: The report to save.
        output_path: Path to save the report.
        indent: JSON indentation level.

    Returns:
        Path to the saved report file.
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=indent)

    return output_path


def load_report(report_path: str) -> ReportSummary:
    """Load an evaluation report from JSON file.

    Args:
        report_path: Path to the report JSON file.

    Returns:
        ReportSummary loaded from the file.

    Raises:
        FileNotFoundError: If the report file doesn't exist.
        ValueError: If the report format is invalid.
    """
    if not os.path.exists(report_path):
        raise FileNotFoundError(f"Report file not found: {report_path}")

    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Parse the report structure
    summary = data.get("summary", {})
    metadata = data.get("metadata", {})
    instances_data = data.get("instances", [])

    instances = [
        InstanceReport(**inst_data) for inst_data in instances_data
    ]

    return ReportSummary(
        total=summary.get("total", 0),
        resolved=summary.get("resolved", 0),
        unresolved=summary.get("unresolved", 0),
        errors=summary.get("errors", 0),
        skipped=summary.get("skipped", 0),
        resolved_rate=summary.get("resolved_rate", 0.0),
        avg_duration=summary.get("avg_duration", 0.0),
        total_duration=summary.get("total_duration", 0.0),
        timestamp=metadata.get("timestamp", ""),
        config=metadata.get("config", {}),
        instances=instances,
    )


def print_report_summary(report: ReportSummary) -> None:
    """Print a formatted summary of the evaluation report.

    Args:
        report: The report to print.
    """
    print("\n" + "=" * 70)
    print("BUN-BENCH EVALUATION REPORT")
    print("=" * 70)
    print(f"Timestamp: {report.timestamp}")
    print("-" * 70)
    print("SUMMARY")
    print("-" * 70)
    print(f"  Total Instances:    {report.total}")
    print(f"  Resolved:           {report.resolved} ({report.resolved_rate:.1%})")
    print(f"  Unresolved:         {report.unresolved}")
    print(f"  Errors:             {report.errors}")
    print(f"  Skipped:            {report.skipped}")
    print(f"  Avg Duration:       {report.avg_duration:.2f}s")
    print(f"  Total Duration:     {report.total_duration:.2f}s")
    print("-" * 70)

    # Print instance details
    if report.instances:
        print("INSTANCE DETAILS")
        print("-" * 70)

        for inst in sorted(report.instances, key=lambda x: x.status):
            status_marker = {
                "resolved": "[PASS]",
                "unresolved": "[FAIL]",
                "error": "[ERR ]",
                "skipped": "[SKIP]",
            }.get(inst.status, "[????]")

            print(f"  {status_marker} {inst.instance_id}")

            if inst.status == "resolved" or inst.status == "unresolved":
                print(f"           Tests: {inst.tests_passed} passed, "
                      f"{inst.tests_failed} failed, "
                      f"{inst.tests_skipped} skipped")

            if inst.error_message:
                error_short = inst.error_message[:60]
                if len(inst.error_message) > 60:
                    error_short += "..."
                print(f"           Error: {error_short}")

            print(f"           Duration: {inst.duration:.2f}s")

    print("=" * 70)


def compare_reports(
    report1: ReportSummary,
    report2: ReportSummary,
    label1: str = "Report 1",
    label2: str = "Report 2",
) -> Dict[str, Any]:
    """Compare two evaluation reports.

    Args:
        report1: First report to compare.
        report2: Second report to compare.
        label1: Label for the first report.
        label2: Label for the second report.

    Returns:
        Dictionary with comparison statistics.
    """
    comparison = {
        "labels": [label1, label2],
        "summary": {
            "total": [report1.total, report2.total],
            "resolved": [report1.resolved, report2.resolved],
            "unresolved": [report1.unresolved, report2.unresolved],
            "errors": [report1.errors, report2.errors],
            "skipped": [report1.skipped, report2.skipped],
            "resolved_rate": [report1.resolved_rate, report2.resolved_rate],
        },
        "resolved_rate_diff": report2.resolved_rate - report1.resolved_rate,
    }

    # Find instances that changed status
    instances1 = {inst.instance_id: inst for inst in report1.instances}
    instances2 = {inst.instance_id: inst for inst in report2.instances}

    all_ids = set(instances1.keys()) | set(instances2.keys())

    improvements = []
    regressions = []
    unchanged = []

    status_order = {"resolved": 3, "unresolved": 2, "error": 1, "skipped": 0}

    for instance_id in all_ids:
        inst1 = instances1.get(instance_id)
        inst2 = instances2.get(instance_id)

        if inst1 and inst2:
            s1 = status_order.get(inst1.status, 0)
            s2 = status_order.get(inst2.status, 0)

            if s2 > s1:
                improvements.append({
                    "instance_id": instance_id,
                    "from": inst1.status,
                    "to": inst2.status,
                })
            elif s2 < s1:
                regressions.append({
                    "instance_id": instance_id,
                    "from": inst1.status,
                    "to": inst2.status,
                })
            else:
                unchanged.append(instance_id)

    comparison["improvements"] = improvements
    comparison["regressions"] = regressions
    comparison["unchanged_count"] = len(unchanged)

    return comparison
