"""
Constants and type definitions for the Bun-Bench evaluation harness.

This module defines:
- TypedDict schemas for benchmark instances
- Enums for test and resolution statuses
- Component categories and difficulty levels
- Bun version mappings
"""

from enum import Enum
from typing import NotRequired, TypedDict


class TestStatus(Enum):
    """Status of an individual test case."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"

    def __str__(self) -> str:
        return self.value


class ResolvedStatus(Enum):
    """Resolution status for an evaluation instance."""

    FULL = "full"       # All FAIL_TO_PASS tests pass, all PASS_TO_PASS tests pass
    PARTIAL = "partial" # Some FAIL_TO_PASS tests pass, all PASS_TO_PASS tests pass
    NO = "no"           # No FAIL_TO_PASS tests pass, or PASS_TO_PASS tests fail

    def __str__(self) -> str:
        return self.value


class DifficultyLevel(Enum):
    """Difficulty classification for benchmark instances."""

    EASY = "easy"           # Simple, single-file changes
    MEDIUM = "medium"       # Multi-file or moderate complexity
    HARD = "hard"           # Complex, requires deep understanding
    EXPERT = "expert"       # Very complex, architectural changes

    def __str__(self) -> str:
        return self.value


class ComponentCategory(Enum):
    """Categories of Bun runtime components."""

    # Core runtime
    RUNTIME = "runtime"
    BUNDLER = "bundler"
    TRANSPILER = "transpiler"
    TEST_RUNNER = "test_runner"
    PACKAGE_MANAGER = "package_manager"

    # APIs and features
    HTTP_SERVER = "http_server"
    WEBSOCKET = "websocket"
    FILE_SYSTEM = "file_system"
    SHELL = "shell"
    FFI = "ffi"
    SQLITE = "sqlite"

    # Compatibility
    NODE_COMPAT = "node_compat"
    WEB_APIS = "web_apis"

    # Other
    CLI = "cli"
    PLUGIN = "plugin"
    OTHER = "other"

    def __str__(self) -> str:
        return self.value


class BunBenchInstance(TypedDict):
    """Schema for a Bun-Bench evaluation instance."""

    # Required fields
    instance_id: str
    repo: str
    base_commit: str
    patch: str
    test_patch: str
    problem_statement: str
    hints_text: str
    created_at: str

    # Test specifications
    FAIL_TO_PASS: list[str]
    PASS_TO_PASS: list[str]

    # Metadata
    version: NotRequired[str]
    environment_setup_commit: NotRequired[str]

    # Classification
    component: NotRequired[str]
    difficulty: NotRequired[str]

    # Additional context
    related_issues: NotRequired[list[str]]
    related_prs: NotRequired[list[str]]


class TestResult(TypedDict):
    """Result of a single test execution."""

    name: str
    status: str  # TestStatus value
    duration_ms: NotRequired[float]
    error_message: NotRequired[str]
    stack_trace: NotRequired[str]


class EvaluationResult(TypedDict):
    """Result of evaluating an instance."""

    instance_id: str
    resolved: str  # ResolvedStatus value

    # Test results
    fail_to_pass_results: list[TestResult]
    pass_to_pass_results: list[TestResult]

    # Statistics
    fail_to_pass_passed: int
    fail_to_pass_total: int
    pass_to_pass_passed: int
    pass_to_pass_total: int

    # Timing
    total_duration_ms: NotRequired[float]

    # Errors
    error: NotRequired[str]


# Bun version mappings for different benchmark time periods
BUN_VERSION_MAPPINGS: dict[str, str] = {
    # Major releases
    "1.0": "1.0.0",
    "1.1": "1.1.0",
    "1.2": "1.2.0",

    # Specific versions for historical commits
    "2023-09": "1.0.0",
    "2023-10": "1.0.7",
    "2023-11": "1.0.14",
    "2023-12": "1.0.18",
    "2026-01": "1.0.25",
    "2026-02": "1.0.29",
    "2026-03": "1.0.35",
    "2026-04": "1.1.0",
    "2026-05": "1.1.7",
    "2026-06": "1.1.12",
    "2026-07": "1.1.17",
    "2026-08": "1.1.21",
    "2026-09": "1.1.26",
    "2026-10": "1.1.30",
    "2026-11": "1.1.34",
    "2026-12": "1.1.38",
    "2025-01": "1.2.0",

    # Latest
    "latest": "latest",
}

# Default Bun version for evaluation
DEFAULT_BUN_VERSION = "1.1.0"

# Container configuration
DEFAULT_TIMEOUT_SECONDS = 300
DEFAULT_MEMORY_LIMIT = "4g"
DEFAULT_CPU_LIMIT = "2"

# Test patterns
BUN_TEST_FILE_PATTERNS = [
    "*.test.ts",
    "*.test.js",
    "*.spec.ts",
    "*.spec.js",
    "test/*.ts",
    "test/*.js",
    "tests/*.ts",
    "tests/*.js",
    "__tests__/*.ts",
    "__tests__/*.js",
]

# Component keywords for auto-classification
COMPONENT_KEYWORDS: dict[str, list[str]] = {
    ComponentCategory.RUNTIME.value: ["runtime", "jsc", "javascript", "execution"],
    ComponentCategory.BUNDLER.value: ["bundler", "bundle", "esbuild", "build"],
    ComponentCategory.TRANSPILER.value: ["transpile", "transform", "compile", "typescript"],
    ComponentCategory.TEST_RUNNER.value: ["test", "jest", "expect", "describe", "it"],
    ComponentCategory.PACKAGE_MANAGER.value: ["install", "npm", "package", "registry", "lockfile"],
    ComponentCategory.HTTP_SERVER.value: ["serve", "http", "server", "request", "response"],
    ComponentCategory.WEBSOCKET.value: ["websocket", "ws", "socket"],
    ComponentCategory.FILE_SYSTEM.value: ["file", "fs", "read", "write", "directory"],
    ComponentCategory.SHELL.value: ["shell", "spawn", "exec", "process", "$"],
    ComponentCategory.FFI.value: ["ffi", "native", "c", "dlopen"],
    ComponentCategory.SQLITE.value: ["sqlite", "database", "sql", "db"],
    ComponentCategory.NODE_COMPAT.value: ["node:", "require", "commonjs", "module"],
    ComponentCategory.WEB_APIS.value: ["fetch", "blob", "stream", "url", "formdata"],
    ComponentCategory.CLI.value: ["cli", "command", "args", "argv"],
    ComponentCategory.PLUGIN.value: ["plugin", "loader", "hook"],
}
