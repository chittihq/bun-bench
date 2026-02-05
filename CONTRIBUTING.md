# Contributing to Bun-Bench

Thank you for your interest in contributing to Bun-Bench! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
- [Adding New Tasks](#adding-new-tasks)
- [Task Format Requirements](#task-format-requirements)
- [Testing Requirements](#testing-requirements)
- [Pull Request Process](#pull-request-process)
- [Development Setup](#development-setup)

---

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment. Please:

- Be respectful and constructive in discussions
- Welcome newcomers and help them get started
- Focus on what is best for the community and the project

---

## How to Contribute

There are many ways to contribute to Bun-Bench:

### 1. Add New Tasks
Expand the benchmark with new Bun runtime issues. See [Adding New Tasks](#adding-new-tasks).

### 2. Improve Documentation
Help improve documentation, examples, and guides.

### 3. Report Issues
Found a bug or have a suggestion? [Open an issue](https://github.com/chittihq/bun-bench/issues).

### 4. Submit Fixes
Fix bugs in the evaluation harness or task definitions.

### 5. Improve Test Coverage
Add tests for existing tasks or improve the testing framework.

---

## Adding New Tasks

### Step 1: Identify a Valid Task

Good tasks for Bun-Bench should:

- Be **real issues** from the Bun runtime or reasonable feature requests
- Be **self-contained** enough to evaluate independently
- Have **clear acceptance criteria** that can be tested
- Be **reproducible** across different environments
- Require **meaningful code changes** (not just configuration)

Tasks should NOT:

- Require external services that cannot be mocked
- Depend on non-deterministic behavior
- Be trivially solvable with simple text matching
- Require human judgment to evaluate

### Step 2: Create the Task File

Create a new task file in `tasks/` following the naming convention:

```
tasks/
  001_response_content_length.json
  002_sqlite_memory_leak.json
  ...
```

### Step 3: Define the Task

Each task file should contain:

```json
{
  "id": 101,
  "title": "Short descriptive title",
  "category": "bug_fix | feature",
  "area": "core_api | fetch | sqlite | postgresql | mysql | build | test | pm",
  "difficulty": "easy | medium | hard",
  "description": "Detailed description of the issue...",
  "reproduction": "Steps to reproduce the issue...",
  "expected_behavior": "What should happen after the fix...",
  "test_file": "tests/101_test.ts",
  "files_to_modify": ["src/relevant_file.ts"],
  "hints": ["Optional hints for solving"],
  "tags": ["tag1", "tag2"]
}
```

### Step 4: Write Tests

Create a test file in `tests/` that validates the fix:

```typescript
// tests/101_test.ts
import { expect, test } from "bun:test";

test("issue 101: description of what is being tested", () => {
  // Test that reproduces the bug (should fail before fix)
  // Test that validates the fix (should pass after fix)
});
```

### Step 5: Submit PR

Submit your task following the [Pull Request Process](#pull-request-process).

---

## Task Format Requirements

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | number | Unique task identifier |
| `title` | string | Short, descriptive title (max 80 chars) |
| `category` | string | Either `bug_fix` or `feature` |
| `area` | string | Functional area (see valid values below) |
| `difficulty` | string | `easy`, `medium`, or `hard` |
| `description` | string | Full description of the issue |
| `expected_behavior` | string | What correct behavior looks like |
| `test_file` | string | Path to the test file |

### Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `reproduction` | string | Steps to reproduce |
| `files_to_modify` | array | Hint about which files need changes |
| `hints` | array | Optional hints for solving |
| `tags` | array | Searchable tags |
| `related_issues` | array | Links to related GitHub issues |
| `bun_version` | string | Minimum Bun version required |

### Valid Area Values

- `core_api` - Bun.* core APIs
- `fetch` - HTTP client and fetch API
- `websocket` - WebSocket client and server
- `sqlite` - bun:sqlite module
- `postgresql` - PostgreSQL driver
- `mysql` - MySQL driver
- `build` - bun build and bundling
- `test` - bun test framework
- `pm` - Package manager (bun install, etc.)
- `ffi` - Foreign function interface
- `shell` - Bun.$ shell API

### Difficulty Guidelines

**Easy:**
- Single file modification
- Clear, isolated bug
- Well-documented expected behavior
- Estimated time: < 30 minutes

**Medium:**
- 2-3 files may need modification
- Requires understanding of Bun internals
- May involve edge cases
- Estimated time: 30-90 minutes

**Hard:**
- Multiple files across different modules
- Requires deep runtime knowledge
- Complex architectural changes
- Estimated time: > 90 minutes

---

## Testing Requirements

### All Tasks Must Have Tests

Every task must include a test file that:

1. **Reproduces the issue** - Test should fail on unpatched code
2. **Validates the fix** - Test should pass after correct fix
3. **Tests edge cases** - Include relevant boundary conditions
4. **Is deterministic** - No flaky tests

### Test File Structure

```typescript
import { expect, test, describe } from "bun:test";

describe("Task XXX: Title", () => {
  test("reproduces the original issue", () => {
    // This test demonstrates the bug
  });

  test("validates the fix works correctly", () => {
    // This test passes with correct implementation
  });

  test("handles edge case: description", () => {
    // Edge case testing
  });
});
```

### Running Tests Locally

```bash
# Run all tests
bun test

# Run specific task test
bun test tests/001_test.ts

# Run tests with coverage
bun test --coverage
```

### Test Validation

Before submitting, ensure:

```bash
# Validate task format
bun-bench validate-task tasks/XXX_your_task.json

# Run the task's tests
bun-bench test-task XXX

# Check for conflicts with existing tasks
bun-bench check-conflicts
```

---

## Pull Request Process

### 1. Fork and Clone

```bash
git clone https://github.com/your-username/bun-bench.git
cd bun-bench
```

### 2. Create a Branch

```bash
git checkout -b add-task-101-description
```

Use descriptive branch names:
- `add-task-XXX-short-description` for new tasks
- `fix-task-XXX-description` for task fixes
- `docs-description` for documentation
- `fix-description` for bug fixes

### 3. Make Your Changes

- Follow the task format requirements
- Write comprehensive tests
- Update documentation if needed

### 4. Validate Your Changes

```bash
# Format code
bun run format

# Run linting
bun run lint

# Run all tests
bun test

# Validate task (if adding a task)
bun-bench validate-task tasks/XXX_your_task.json
```

### 5. Commit Your Changes

Write clear, descriptive commit messages:

```bash
git commit -m "Add task 101: WebSocket close frame buffer overflow

- Add task definition with reproduction steps
- Include test file validating the fix
- Categorize as medium difficulty bug fix"
```

### 6. Push and Create PR

```bash
git push origin add-task-101-description
```

Then create a Pull Request on GitHub with:

- **Title**: Clear description of the change
- **Description**: What the PR does and why
- **Testing**: How you tested the changes
- **Checklist**: Confirm all requirements are met

### PR Checklist

- [ ] Task follows the required format
- [ ] Tests are included and pass
- [ ] No conflicts with existing tasks
- [ ] Documentation updated (if applicable)
- [ ] Code is formatted and linted
- [ ] Commit messages are clear

### Review Process

1. A maintainer will review your PR
2. Address any requested changes
3. Once approved, your PR will be merged
4. Your contribution will be acknowledged in the changelog

---

## Development Setup

### Prerequisites

- Python 3.9+
- Bun (latest version)
- Git

### Setup

```bash
# Clone the repository
git clone https://github.com/chittihq/bun-bench.git
cd bun-bench

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install Bun (if not already installed)
curl -fsSL https://bun.sh/install | bash
```

### Project Structure

```
bun-bench/
  tasks/              # Task definitions (JSON)
  tests/              # Task test files (TypeScript)
  src/                # Python source code
    bun_bench/
      __init__.py
      benchmark.py    # Core benchmark class
      evaluator.py    # Evaluation harness
      cli.py          # Command-line interface
  docs/               # Documentation
  scripts/            # Utility scripts
  results/            # Evaluation results (gitignored)
```

### Useful Commands

```bash
# Run tests
bun test
pytest

# Format code
bun run format
black src/

# Lint
bun run lint
ruff src/

# Build documentation
mkdocs serve
```

---

## Questions?

- Open a [GitHub Discussion](https://github.com/chittihq/bun-bench/discussions)
- Check existing [Issues](https://github.com/chittihq/bun-bench/issues)
- Read the [Documentation](docs/index.md)

Thank you for contributing to Bun-Bench!
