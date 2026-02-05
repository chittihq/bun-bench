"""
Collection module for Bun-Bench.

This module provides utilities for creating and managing benchmark datasets,
including instance creation, validation, and dataset building.
"""

from bunbench.collect.build_dataset import (
    BenchmarkInstance,
    DatasetBuilder,
    create_instance,
    validate_instance,
)

__all__ = [
    "BenchmarkInstance",
    "DatasetBuilder",
    "validate_instance",
    "create_instance",
]
