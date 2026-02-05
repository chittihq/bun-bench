# Evaluation Guide

This guide provides comprehensive documentation for running evaluations with Bun-Bench, understanding results, and submitting to the leaderboard.

## Table of Contents

- [Overview](#overview)
- [Evaluation Process](#evaluation-process)
- [Running Evaluations](#running-evaluations)
- [Evaluation Options](#evaluation-options)
- [Understanding Results](#understanding-results)
- [Submitting Results](#submitting-results)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

---

## Overview

Bun-Bench evaluates LLMs on their ability to resolve real-world Bun runtime issues. The evaluation process:

1. Presents a task description to the model
2. Allows the model to analyze and generate a solution
3. Runs test cases to validate the solution
4. Records whether the task was successfully resolved

### Evaluation Criteria

A task is considered **resolved** if:
- All test cases pass
- The solution doesn't break existing functionality
- The implementation matches expected behavior

### Metrics

| Metric | Description |
|--------|-------------|
| **Resolved** | Number of tasks successfully completed |
| **Resolution Rate** | Percentage of tasks resolved (Resolved / Total) |
| **Avg. Iterations** | Average attempts per resolved task |
| **Avg. Time** | Average time per task |

---

## Evaluation Process

### How It Works

```
+-------------------------------------------------------------+
|                    EVALUATION PIPELINE                       |
+-------------------------------------------------------------+
|                                                              |
|  1. LOAD TASK                                                |
|     +-- Read task description                                |
|     +-- Load test file                                       |
|     +-- Set up evaluation environment                        |
|                                                              |
|  2. PROMPT MODEL                                             |
|     +-- Send task description to model                       |
|     +-- Include context (file structure, etc.)               |
|     +-- Request solution                                     |
|                                                              |
|  3. APPLY SOLUTION                                           |
|     +-- Extract code changes from model output               |
|     +-- Apply changes to codebase                            |
|     +-- Handle any syntax errors                             |
|                                                              |
|  4. RUN TESTS                                                |
|     +-- Execute test file with Bun                           |
|     +-- Capture test results                                 |
|     +-- Check for passing/failing tests                      |
|                                                              |
|  5. RECORD RESULT                                            |
|     +-- Mark task as resolved or failed                      |
|     +-- Record iterations and time                           |
|     +-- Save model output and logs                           |
|                                                              |
+-------------------------------------------------------------+
```

### Iteration Flow

If a task isn't resolved on the first attempt, the evaluator can retry:

```
Attempt 1 -> Test fails -> Provide error feedback -> Attempt 2 -> Test fails -> ... -> Max iterations
```

The evaluator provides test failure information to help the model refine its solution.

---

## Running Evaluations

### Command Line Interface

#### Basic Evaluation

```bash
# Evaluate a model on all tasks
bun-bench evaluate --model gpt-4 --output results/

# Evaluate specific tasks
bun-bench evaluate --model claude-3-opus --tasks 1,2,3,4,5 --output results/
```

#### With Options

```bash
bun-bench evaluate \
  --model claude-3-opus \
  --output results/claude-opus-full/ \
  --parallel 4 \
  --max-iterations 3 \
  --timeout 300 \
  --verbose
```

### Python API

#### Basic Usage

```python
from bun_bench import BunBench, Evaluator

# Load the benchmark
benchmark = BunBench()

# Create an evaluator
evaluator = Evaluator(model="claude-3-opus")

# Evaluate all tasks
results = evaluator.evaluate_all(benchmark.tasks)

# Save results
results.to_json("results/output.json")
```

#### Advanced Usage

```python
from bun_bench import BunBench, Evaluator, EvaluatorConfig

# Load benchmark
benchmark = BunBench()

# Configure evaluator
config = EvaluatorConfig(
    model="claude-3-opus",
    max_iterations=5,
    timeout=600,
    temperature=0,
    parallel_workers=4,
    retry_on_error=True,
    verbose=True
)

evaluator = Evaluator(config=config)

# Filter tasks
hard_tasks = benchmark.filter(difficulty="hard")

# Evaluate with progress callback
def on_task_complete(task, result):
    status = "PASS" if result.resolved else "FAIL"
    print(f"[{status}] Task {task.id}: {task.title}")

results = evaluator.evaluate_all(
    hard_tasks,
    on_complete=on_task_complete
)

# Generate detailed report
results.to_report("results/hard_tasks_report.json")
```

---

## Evaluation Options

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | Required | Model identifier to evaluate |
| `--output` | Required | Output directory for results |
| `--tasks` | All | Comma-separated task IDs |
| `--category` | All | Filter by category (bug_fix, feature) |
| `--area` | All | Filter by area (postgresql, mysql, etc.) |
| `--difficulty` | All | Filter by difficulty (easy, medium, hard) |
| `--max-iterations` | 3 | Maximum attempts per task |
| `--timeout` | 300 | Timeout per task in seconds |
| `--parallel` | 1 | Number of parallel workers |
| `--verbose` | false | Enable verbose output |
| `--dry-run` | false | List tasks without running |

### Python Configuration

```python
from bun_bench import EvaluatorConfig

config = EvaluatorConfig(
    # Model settings
    model="claude-3-opus",
    api_key=None,  # Uses environment variable if not set
    temperature=0,
    max_tokens=4096,

    # Evaluation settings
    max_iterations=3,
    timeout=300,
    retry_on_error=True,

    # Execution settings
    parallel_workers=4,
    verbose=True,

    # Output settings
    save_intermediate=True,
    log_level="INFO"
)
```

---

## Understanding Results

### Result Structure

```json
{
  "metadata": {
    "model": "claude-3-opus",
    "timestamp": "2026-01-15T10:30:00Z",
    "total_tasks": 100,
    "config": {}
  },
  "summary": {
    "resolved": 67,
    "failed": 33,
    "resolution_rate": 0.67,
    "avg_iterations": 1.8,
    "avg_time": 45.2
  },
  "results": [
    {
      "task_id": 1,
      "resolved": true,
      "iterations": 1,
      "time": 23.5,
      "output": "...",
      "test_results": {}
    }
  ]
}
```

### Generating Reports

```bash
# Generate summary report
bun-bench report results/ --format summary

# Generate detailed report
bun-bench report results/ --format detailed

# Generate comparison report
bun-bench compare results/model1/ results/model2/
```

### Python Report Generation

```python
# Basic summary
summary = results.summary()
print(f"Resolved: {summary['resolved']}/{summary['total']}")

# Detailed breakdown
breakdown = results.breakdown()
print(f"Easy: {breakdown['easy']['resolved']}/{breakdown['easy']['total']}")
print(f"Medium: {breakdown['medium']['resolved']}/{breakdown['medium']['total']}")
print(f"Hard: {breakdown['hard']['resolved']}/{breakdown['hard']['total']}")

# Export formats
results.to_json("results.json")
results.to_csv("results.csv")
results.to_markdown("results.md")
```

### Analyzing Failures

```python
# Get failed tasks
failed = results.filter(resolved=False)

for result in failed:
    print(f"Task {result.task_id}: {result.error_message}")
    print(f"  Last output: {result.output[:200]}...")
```

---

## Submitting Results

### Prerequisites

Before submitting:

1. Run the **complete benchmark** (all 100 tasks)
2. Ensure results are **reproducible**
3. Document your **model configuration**

### Validation

```bash
# Validate your results
bun-bench validate results/

# This checks:
# - All 100 tasks are present
# - Result format is correct
# - No corrupted data
```

### Package Submission

```bash
# Create submission package
bun-bench package results/ --output submission.json
```

This creates a submission file containing:
- Complete results
- Model configuration
- Evaluation metadata
- Checksums for verification

### Submit via GitHub

1. **Fork** the [bun-bench-results](https://github.com/chittihq/bun-bench-results) repository

2. **Add** your submission:
   ```bash
   git clone https://github.com/your-username/bun-bench-results.git
   cd bun-bench-results
   cp /path/to/submission.json submissions/your-model-name.json
   ```

3. **Create PR** with the following information:
   - Model name and version
   - Link to model (if public)
   - Any special configuration
   - Date of evaluation

### Submission Requirements

| Requirement | Description |
|-------------|-------------|
| Complete | All 100 tasks evaluated |
| Reproducible | Results can be independently verified |
| Documented | Model and configuration clearly described |
| Unmodified | Tasks and tests unchanged from official version |
| Honest | No task-specific fine-tuning or data leakage |

---

## Best Practices

### For Accurate Evaluation

1. **Use consistent settings**
   - Temperature = 0 for reproducibility
   - Same timeout for all tasks
   - Document any model-specific parameters

2. **Avoid contamination**
   - Don't fine-tune on Bun-Bench tasks
   - Don't provide task solutions in context
   - Evaluate on the official task set

3. **Handle failures gracefully**
   - Allow retries for transient errors
   - Log all errors for debugging
   - Report incomplete evaluations honestly

### For Performance

1. **Parallelize carefully**
   ```bash
   # Balance parallelism with API rate limits
   bun-bench evaluate --model gpt-4 --parallel 4
   ```

2. **Use appropriate timeouts**
   - Easy tasks: 120-180 seconds
   - Medium tasks: 180-300 seconds
   - Hard tasks: 300-600 seconds

3. **Monitor resource usage**
   - Watch for memory leaks
   - Monitor API costs
   - Use --dry-run first

### For Reproducibility

1. **Pin versions**
   ```bash
   pip freeze > requirements.txt
   bun --version > bun-version.txt
   ```

2. **Save all logs**
   ```bash
   bun-bench evaluate --model gpt-4 --output results/ --verbose 2>&1 | tee evaluation.log
   ```

3. **Document environment**
   - OS and version
   - Python version
   - Bun version
   - Model version/date

---

## Troubleshooting

### Common Issues

#### "API rate limit exceeded"

```bash
# Reduce parallelism
bun-bench evaluate --model gpt-4 --parallel 1

# Add delay between requests (Python)
evaluator = Evaluator(model="gpt-4", request_delay=1.0)
```

#### "Task timeout exceeded"

```bash
# Increase timeout
bun-bench evaluate --model gpt-4 --timeout 600
```

#### "Tests fail unexpectedly"

```bash
# Run single task in verbose mode
bun-bench evaluate --model gpt-4 --tasks 1 --verbose

# Check Bun version compatibility
bun --version
```

#### "Out of memory"

```bash
# Reduce parallelism
bun-bench evaluate --model gpt-4 --parallel 1

# Increase system swap space
```

### Debugging

```bash
# Enable debug logging
export BUN_BENCH_DEBUG=1
bun-bench evaluate --model gpt-4 --output results/ --verbose

# Check specific task
bun-bench show 42  # View task details
bun-bench test 42  # Run task tests manually
```

### Getting Help

- Check [GitHub Issues](https://github.com/chittihq/bun-bench/issues)
- Ask in [GitHub Discussions](https://github.com/chittihq/bun-bench/discussions)
- Review [examples](https://github.com/chittihq/bun-bench/tree/main/examples)
