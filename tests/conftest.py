"""
Pytest configuration and fixtures for bun-bench tests.
"""

import pytest
from pathlib import Path
from typing import Any


@pytest.fixture
def sample_instance() -> dict[str, Any]:
    """
    Provides a sample benchmark instance for testing.

    Returns a dictionary representing a single benchmark task
    with all required fields.
    """
    return {
        "instance_id": "bun__bun__1",
        "repo": "oven-sh/bun",
        "base_commit": "abc123def456",
        "problem_statement": (
            "[Bug Fix] Bun.serve() returns incorrect Content-Length header "
            "when response body contains multi-byte UTF-8 characters. "
            "The server reports byte length as character length, causing "
            "truncated responses for non-ASCII content."
        ),
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


@pytest.fixture
def sample_prediction() -> dict[str, Any]:
    """
    Provides a sample model prediction for testing.

    Returns a dictionary representing an LLM's attempted
    solution to a benchmark instance.
    """
    return {
        "instance_id": "bun__bun__1",
        "model_name_or_path": "claude-3-opus",
        "model_patch": '''
diff --git a/src/http.zig b/src/http.zig
index 1234567..abcdefg 100644
--- a/src/http.zig
+++ b/src/http.zig
@@ -100,7 +100,7 @@ fn calculateContentLength(body: []const u8) usize {
-    return body.len;
+    return std.mem.len(body);  // Use byte length, not character count
 }
''',
        "full_output": "I analyzed the issue and found that...",
        "run_id": "eval_run_001",
    }


@pytest.fixture
def sample_bun_test_output() -> str:
    """
    Provides sample bun test output for parser testing.
    """
    return """bun test v1.0.0

test/http/content-length.test.ts:
  should calculate correct byte length for UTF-8 ... [PASS] (2.31ms)
  should handle multi-byte characters ... [PASS] (1.45ms)
  should work for ASCII content ... [PASS] (0.89ms)

test/http/basic.test.ts:
  should start server ... [PASS] (5.21ms)
  should handle requests ... [FAIL] (3.12ms)
    Error: Expected 200 but got 500
      at test/http/basic.test.ts:25:5
  should stop server ... [SKIP] (0.00ms)
    Reason: Depends on previous test

test/websocket/connection.test.ts:
  should connect to websocket server ... [PASS] (10.32ms)
  should send and receive messages ... [PASS] (8.76ms)

 8 pass
 1 fail
 1 skip
 10 expect() calls

Ran 10 tests in 32.06ms
"""


@pytest.fixture
def sample_bun_test_output_all_pass() -> str:
    """
    Provides sample bun test output where all tests pass.
    """
    return """bun test v1.0.0

test/http/content-length.test.ts:
  should calculate correct byte length for UTF-8 ... [PASS] (2.31ms)
  should handle multi-byte characters ... [PASS] (1.45ms)
  should work for ASCII content ... [PASS] (0.89ms)

 3 pass
 0 fail
 0 skip
 3 expect() calls

Ran 3 tests in 4.65ms
"""


@pytest.fixture
def sample_bun_test_output_all_fail() -> str:
    """
    Provides sample bun test output where all tests fail.
    """
    return """bun test v1.0.0

test/http/content-length.test.ts:
  should calculate correct byte length for UTF-8 ... [FAIL] (2.31ms)
    Error: Expected 15 but got 5
      at test/http/content-length.test.ts:10:5
  should handle multi-byte characters ... [FAIL] (1.45ms)
    Error: Content-Length mismatch
      at test/http/content-length.test.ts:20:5
  should work for ASCII content ... [FAIL] (0.89ms)
    Error: Server not responding
      at test/http/content-length.test.ts:30:5

 0 pass
 3 fail
 0 skip
 3 expect() calls

Ran 3 tests in 4.65ms
"""


@pytest.fixture
def test_data_dir() -> Path:
    """
    Returns the path to the test data directory.
    """
    return Path(__file__).parent / "test_data"


@pytest.fixture
def sample_instances_file(tmp_path: Path, sample_instance: dict) -> Path:
    """
    Creates a temporary JSON file with sample instances.
    """
    import json

    instances_file = tmp_path / "instances.json"
    instances_file.write_text(json.dumps([sample_instance], indent=2))
    return instances_file


@pytest.fixture
def sample_predictions_file(tmp_path: Path, sample_prediction: dict) -> Path:
    """
    Creates a temporary JSON file with sample predictions.
    """
    import json

    predictions_file = tmp_path / "predictions.json"
    predictions_file.write_text(json.dumps([sample_prediction], indent=2))
    return predictions_file


@pytest.fixture
def mock_subprocess_success(monkeypatch):
    """
    Mocks subprocess.run to simulate successful test execution.
    """
    import subprocess

    class MockCompletedProcess:
        def __init__(self):
            self.returncode = 0
            self.stdout = """bun test v1.0.0

test/example.test.ts:
  test case 1 ... [PASS] (1.00ms)
  test case 2 ... [PASS] (1.00ms)

 2 pass
 0 fail
 0 skip

Ran 2 tests in 2.00ms
"""
            self.stderr = ""

    def mock_run(*args, **kwargs):
        return MockCompletedProcess()

    monkeypatch.setattr(subprocess, "run", mock_run)


@pytest.fixture
def mock_subprocess_failure(monkeypatch):
    """
    Mocks subprocess.run to simulate failed test execution.
    """
    import subprocess

    class MockCompletedProcess:
        def __init__(self):
            self.returncode = 1
            self.stdout = """bun test v1.0.0

test/example.test.ts:
  test case 1 ... [FAIL] (1.00ms)
    Error: Assertion failed

 0 pass
 1 fail
 0 skip

Ran 1 test in 1.00ms
"""
            self.stderr = ""

    def mock_run(*args, **kwargs):
        return MockCompletedProcess()

    monkeypatch.setattr(subprocess, "run", mock_run)
