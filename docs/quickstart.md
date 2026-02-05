# Quick Start Guide

Get up and running with Bun-Bench in just a few minutes.

## Prerequisites

Before you begin, ensure you have:

- **Python 3.9+** installed
- **Bun runtime** installed ([install Bun](https://bun.sh))
- **Git** installed

## Installation

### Option 1: Install from PyPI (Recommended)

```bash
pip install bun-bench
```

### Option 2: Install from Source

```bash
# Clone the repository
git clone https://github.com/chittihq/bun-bench.git
cd bun-bench

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .
```

### Verify Installation

```bash
# Check that bun-bench is installed
bun-bench --version

# Verify Bun is available
bun --version
```

---

## Your First Evaluation

### Step 1: Explore the Dataset

```python
from bun_bench import BunBench

# Load the benchmark
benchmark = BunBench()

# See how many tasks are available
print(f"Total tasks: {len(benchmark.tasks)}")

# View a specific task
task = benchmark.get_task(1)
print(f"Task 1: {task.title}")
print(f"Category: {task.category}")
print(f"Difficulty: {task.difficulty}")
print(f"\nDescription:\n{task.description}")
```

### Step 2: Run a Simple Evaluation

Using the command line:

```bash
# Evaluate a model on a single task
bun-bench evaluate --model gpt-4 --tasks 1 --output results/

# Evaluate on multiple tasks
bun-bench evaluate --model claude-3-opus --tasks 1,2,3,4,5 --output results/
```

Using Python:

```python
from bun_bench import BunBench, Evaluator

# Load benchmark and create evaluator
benchmark = BunBench()
evaluator = Evaluator(model="gpt-4")

# Evaluate a single task
task = benchmark.get_task(1)
result = evaluator.evaluate_task(task)

print(f"Task resolved: {result.resolved}")
print(f"Iterations: {result.iterations}")
print(f"Time: {result.time:.2f}s")
```

### Step 3: View Results

```bash
# Generate a report from results
bun-bench report results/

# View summary statistics
bun-bench summary results/
```

---

## Running Full Evaluation

To run the complete benchmark:

```bash
# Evaluate all 100 tasks
bun-bench evaluate \
  --model your-model \
  --output results/full-run/ \
  --parallel 4 \
  --timeout 300
```

This will:
1. Run all 100 tasks against your model
2. Save detailed results to `results/full-run/`
3. Generate a summary report

---

## Filtering Tasks

### By Difficulty

```bash
# Easy tasks only
bun-bench evaluate --model gpt-4 --difficulty easy --output results/

# Hard tasks only
bun-bench evaluate --model gpt-4 --difficulty hard --output results/
```

### By Category

```bash
# PostgreSQL tasks only
bun-bench evaluate --model gpt-4 --category postgresql --output results/

# Bug fixes only
bun-bench evaluate --model gpt-4 --type bug_fix --output results/
```

### By Area

```bash
# Core API tasks
bun-bench evaluate --model gpt-4 --area core_api --output results/

# Database tasks (PostgreSQL + MySQL)
bun-bench evaluate --model gpt-4 --area postgresql,mysql --output results/
```

---

## Configuration

### Environment Variables

```bash
# Set your API keys
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"

# Set default output directory
export BUN_BENCH_OUTPUT="./results"
```

### Configuration File

Create `bun-bench.yaml` in your project root:

```yaml
# bun-bench.yaml
evaluation:
  max_iterations: 3
  timeout: 300
  parallel: 4

models:
  gpt-4:
    api_key: ${OPENAI_API_KEY}
    temperature: 0
  claude-3-opus:
    api_key: ${ANTHROPIC_API_KEY}
    temperature: 0

output:
  directory: ./results
  format: json
```

---

## Common Commands

| Command | Description |
|---------|-------------|
| `bun-bench evaluate` | Run evaluation on tasks |
| `bun-bench report` | Generate report from results |
| `bun-bench summary` | Show summary statistics |
| `bun-bench list` | List available tasks |
| `bun-bench show <id>` | Show details of a specific task |
| `bun-bench validate` | Validate result files |

### Command Help

```bash
# Get help for any command
bun-bench --help
bun-bench evaluate --help
```

---

## Example Scripts

### Evaluate and Report

```bash
#!/bin/bash
# evaluate_model.sh

MODEL=$1
OUTPUT_DIR="results/${MODEL}-$(date +%Y%m%d)"

# Run evaluation
bun-bench evaluate \
  --model $MODEL \
  --output $OUTPUT_DIR \
  --parallel 4

# Generate report
bun-bench report $OUTPUT_DIR

# Show summary
bun-bench summary $OUTPUT_DIR
```

### Python Evaluation Script

```python
#!/usr/bin/env python3
"""evaluate_model.py - Run Bun-Bench evaluation"""

import sys
from bun_bench import BunBench, Evaluator

def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "gpt-4"

    # Load benchmark
    benchmark = BunBench()

    # Create evaluator
    evaluator = Evaluator(
        model=model,
        max_iterations=3,
        timeout=300
    )

    # Run evaluation
    print(f"Evaluating {model} on {len(benchmark.tasks)} tasks...")
    results = evaluator.evaluate_all(benchmark.tasks)

    # Save results
    output_path = f"results/{model}_results.json"
    results.to_json(output_path)

    # Print summary
    summary = results.summary()
    print(f"\nResults:")
    print(f"  Resolved: {summary['resolved']}/{summary['total']}")
    print(f"  Success rate: {summary['percentage']:.1f}%")
    print(f"  Saved to: {output_path}")

if __name__ == "__main__":
    main()
```

---

## Troubleshooting

### Common Issues

**"bun: command not found"**
```bash
# Install Bun
curl -fsSL https://bun.sh/install | bash
# Restart your terminal or run:
source ~/.bashrc  # or ~/.zshrc
```

**"Model API key not set"**
```bash
# Set your API key
export OPENAI_API_KEY="your-key"
# Or for Anthropic:
export ANTHROPIC_API_KEY="your-key"
```

**"Task timeout exceeded"**
```bash
# Increase timeout
bun-bench evaluate --model gpt-4 --timeout 600 --output results/
```

**"Memory error during evaluation"**
```bash
# Reduce parallelism
bun-bench evaluate --model gpt-4 --parallel 1 --output results/
```

---

## Next Steps

- Read the [Evaluation Guide](evaluation.md) for detailed evaluation options
- Learn about [task format](../CONTRIBUTING.md) to understand tasks better
- [Submit your results](evaluation.md#submitting-results) to the leaderboard
- [Contribute new tasks](../CONTRIBUTING.md) to expand the benchmark
