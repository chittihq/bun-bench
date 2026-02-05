# Bun-Bench Documentation

Welcome to the Bun-Bench documentation. Bun-Bench is a comprehensive benchmark for evaluating Large Language Models (LLMs) on their ability to resolve real-world issues in the Bun JavaScript/TypeScript runtime.

## What is Bun-Bench?

Bun-Bench provides a standardized framework for assessing AI coding assistants on 100 carefully curated tasks derived from realistic Bun runtime issues. Each task includes:

- A detailed problem description
- Reproducible test cases
- Expected behavior specifications
- Difficulty classification

## Documentation

### Getting Started

- [Quick Start Guide](quickstart.md) - Get up and running in minutes
- [Installation](quickstart.md#installation) - Detailed installation instructions

### Evaluation

- [Evaluation Guide](evaluation.md) - How to run evaluations
- [Understanding Results](evaluation.md#understanding-results) - Interpreting evaluation output
- [Submitting Results](evaluation.md#submitting-results) - How to submit to the leaderboard

### Reference

- [Task Format](../CONTRIBUTING.md#task-format-requirements) - Task specification format
- [API Reference](#api-reference) - Python API documentation

---

## Quick Links

| Resource | Description |
|----------|-------------|
| [GitHub Repository](https://github.com/chittihq/bun-bench) | Source code and issues |
| [Quick Start](quickstart.md) | Get started in 5 minutes |
| [Evaluation Guide](evaluation.md) | Detailed evaluation instructions |
| [Contributing](../CONTRIBUTING.md) | How to contribute tasks |
| [Leaderboard](https://github.com/chittihq/bun-bench#leaderboard) | Current benchmark results |

---

## Dataset Overview

Bun-Bench contains **100 tasks** spanning multiple categories:

### Task Categories

| Category | Tasks | Description |
|----------|-------|-------------|
| Core APIs | 25 | `Bun.serve()`, `Bun.file()`, `Bun.spawn()`, etc. |
| Fetch & Network | 12 | HTTP client, WebSocket, DNS resolution |
| SQLite | 8 | `bun:sqlite` module issues |
| PostgreSQL | 25 | PostgreSQL driver bugs and features |
| MySQL | 25 | MySQL driver bugs and features |
| Build & Bundle | 10 | `bun build`, transpilation, HMR |
| Testing | 8 | `bun test` framework issues |
| Package Manager | 7 | `bun install`, `bun pm` commands |

### Task Types

- **Bug Fix** (70 tasks): Resolve existing incorrect behavior
- **Feature** (30 tasks): Implement new functionality

### Difficulty Levels

| Level | Count | Description |
|-------|-------|-------------|
| Easy | 30 | Single-file fixes, clear steps |
| Medium | 45 | Multi-file changes, moderate complexity |
| Hard | 25 | Architectural changes, deep knowledge required |

---

## API Reference

### BunBench Class

```python
from bun_bench import BunBench

# Initialize the benchmark
benchmark = BunBench()

# Get all tasks
tasks = benchmark.tasks

# Get a specific task by ID
task = benchmark.get_task(1)

# Filter tasks by category
postgres_tasks = benchmark.filter(area="postgresql")

# Filter by difficulty
easy_tasks = benchmark.filter(difficulty="easy")
```

### Task Object

```python
task = benchmark.get_task(1)

# Access task properties
task.id           # int: Unique task identifier
task.title        # str: Short description
task.category     # str: "bug_fix" or "feature"
task.area         # str: Functional area
task.difficulty   # str: "easy", "medium", or "hard"
task.description  # str: Full problem description
task.test_file    # str: Path to test file
```

### Evaluator Class

```python
from bun_bench import Evaluator

# Create an evaluator for a specific model
evaluator = Evaluator(
    model="claude-3-opus",
    max_iterations=3,
    timeout=300
)

# Evaluate a single task
result = evaluator.evaluate_task(task)

# Evaluate multiple tasks
results = evaluator.evaluate_all(tasks)

# Check if task was resolved
result.resolved   # bool: Whether the task was successfully completed
result.iterations # int: Number of attempts made
result.time       # float: Time taken in seconds
result.output     # str: Model's final output
```

### Results Class

```python
# Generate reports
results.to_json("results.json")
results.to_report("report.json")

# Get summary statistics
summary = results.summary()
print(f"Resolved: {summary['resolved']}/{summary['total']}")
print(f"Success rate: {summary['percentage']:.1f}%")
```

---

## Example Workflow

```python
from bun_bench import BunBench, Evaluator

# 1. Load the benchmark
benchmark = BunBench()

# 2. Select tasks to evaluate
tasks = benchmark.filter(difficulty="easy", area="core_api")

# 3. Create evaluator
evaluator = Evaluator(model="your-model")

# 4. Run evaluation
results = evaluator.evaluate_all(tasks)

# 5. Generate report
results.to_report("my_results.json")

# 6. Print summary
print(results.summary())
```

---

## Related Resources

- [Bun Documentation](https://bun.sh/docs)
- [SWE-bench](https://github.com/princeton-nlp/SWE-bench)
- [Bun GitHub Repository](https://github.com/oven-sh/bun)

---

## Support

- **Issues**: [GitHub Issues](https://github.com/chittihq/bun-bench/issues)
- **Discussions**: [GitHub Discussions](https://github.com/chittihq/bun-bench/discussions)
- **Contributing**: [CONTRIBUTING.md](../CONTRIBUTING.md)
