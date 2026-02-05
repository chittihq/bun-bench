"""
Harness module for running Bun-Bench evaluations.

This module provides the core functionality for:
- Loading datasets and predictions
- Running evaluations in Docker containers
- Parsing test results
- Generating reports
- Docker image building and container management
"""

from bunbench.harness.docker_build import (
    BASE_IMAGE_NAME,
    ENV_IMAGE_PREFIX,
    EVAL_IMAGE_PREFIX,
    IMAGE_PREFIX,
    BuildResult,
    build_base_image,
    build_env_image,
    build_instance_image,
    cleanup_images,
    get_env_image_name,
    get_eval_image_name,
)
from bunbench.harness.docker_utils import (
    ContainerConfig,
    ContainerInfo,
    ExecutionResult,
    cleanup_container,
    cleanup_containers_by_prefix,
    copy_content_to_container,
    copy_from_container,
    copy_to_container,
    create_container,
    execute_command,
    execute_script,
    get_container_logs,
    get_container_status,
    list_containers,
    start_container,
    stop_container,
    wait_for_container,
)
from bunbench.harness.reporting import (
    InstanceReport,
    ReportSummary,
    compare_reports,
    generate_report,
    load_report,
    print_report_summary,
    save_report,
)
from bunbench.harness.run_evaluation import (
    EvaluationConfig,
    EvaluationResult,
    EvaluationStatus,
    TestResult,
    load_dataset,
    load_predictions,
    run_evaluation,
    run_single_evaluation,
)

__all__ = [
    # Run evaluation
    "run_evaluation",
    "run_single_evaluation",
    "load_dataset",
    "load_predictions",
    "EvaluationConfig",
    "EvaluationResult",
    "EvaluationStatus",
    "TestResult",
    # Reporting
    "generate_report",
    "ReportSummary",
    "InstanceReport",
    "save_report",
    "load_report",
    "print_report_summary",
    "compare_reports",
    # Docker build
    "build_base_image",
    "build_env_image",
    "build_instance_image",
    "get_env_image_name",
    "get_eval_image_name",
    "cleanup_images",
    "BuildResult",
    "IMAGE_PREFIX",
    "BASE_IMAGE_NAME",
    "ENV_IMAGE_PREFIX",
    "EVAL_IMAGE_PREFIX",
    # Docker utils
    "create_container",
    "start_container",
    "stop_container",
    "copy_to_container",
    "copy_from_container",
    "copy_content_to_container",
    "execute_command",
    "execute_script",
    "get_container_logs",
    "get_container_status",
    "cleanup_container",
    "cleanup_containers_by_prefix",
    "list_containers",
    "wait_for_container",
    "ContainerInfo",
    "ContainerConfig",
    "ExecutionResult",
]
