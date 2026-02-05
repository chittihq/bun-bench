"""
Bun-Bench: A benchmark suite for evaluating LLM code generation on Bun.js tasks.

This package provides tools for:
- Running evaluations of model-generated patches
- Building and managing Docker containers for isolated testing
- Parsing test results and generating reports
- Running inference with various LLM APIs (OpenAI, Anthropic)
- Collecting and building benchmark datasets
"""

from bunbench.harness import generate_report, run_evaluation

__version__ = "0.1.0"
__author__ = "Bun-Bench Team"

VERSION_INFO = {
    "major": 0,
    "minor": 1,
    "patch": 0,
    "release": "alpha",
}


def get_version() -> str:
    """Return the current version string."""
    return __version__


def get_version_info() -> dict:
    """Return detailed version information as a dictionary."""
    return VERSION_INFO.copy()


__all__ = [
    "run_evaluation",
    "generate_report",
    "__version__",
    "get_version",
    "get_version_info",
    "VERSION_INFO",
]
