# Bun-Bench

**A benchmark for evaluating LLM coding capabilities on real-world Bun runtime issues**

<!-- Badges -->
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Tasks](https://img.shields.io/badge/tasks-100-green.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)
<!-- ![PyPI](https://img.shields.io/pypi/v/bun-bench.svg) -->
<!-- ![Downloads](https://img.shields.io/pypi/dm/bun-bench.svg) -->

---

## Overview

Bun-Bench is a curated benchmark designed to evaluate Large Language Models (LLMs) on their ability to resolve real-world issues in the [Bun](https://bun.sh) JavaScript/TypeScript runtime. Inspired by [SWE-bench](https://github.com/SWE-bench/SWE-bench), Bun-Bench provides a standardized framework for assessing AI coding assistants on tasks ranging from bug fixes to feature implementations.

### What is Bun-Bench?

Bun-Bench consists of **100 carefully curated tasks** derived from realistic Bun runtime issues, including:

- **Bug fixes** in core APIs (`Bun.serve()`, `Bun.file()`, `fetch()`, etc.)
- **Feature implementations** for new Bun APIs
- **Database driver issues** for PostgreSQL and MySQL
- **Build tooling problems** with bundling, transpilation, and HMR
- **Testing framework issues** with `bun test`

Each task includes:
- A detailed problem description
- The issue category (Bug Fix or Feature)
- Reproducible test cases
- Expected behavior specifications

---

## Installation

### From PyPI (Recommended)

```bash
pip install bun-bench
```

### From Source

```bash
git clone https://github.com/chittihq/bun-bench.git
cd bun-bench
pip install -e .
```

### Requirements

- Python 3.9+
- Bun runtime (latest version recommended)
- Git

---

## Quick Start

### 1. Install the package

```bash
pip install bun-bench
```

### 2. Configure Environment

Edit `.env` file to set runtime and test configuration:

```bash
# Runtime Configuration
BENCH_RUNTIME=bun
BENCH_TEST_RUNNER="bun test"
BENCH_LANGUAGE=typescript

# Dataset
BENCH_DATASET=dataset/tasks_from_dirs.json

# Model Configuration
BENCH_PROVIDER=openrouter
BENCH_MODEL=minimax/minimax-m2.5
```

### 3. Run Inference

```bash
python3 -m bunbench.inference.run_api --instances task-001 task-002
```

### 4. Run Evaluation

```bash
python3 -m bunbench evaluate --local --instance-ids task-001
```

### 5. View Results

Reports are saved in each task folder:
- `dataset/tasks/<task-id>/evaluation_report.json`
- `dataset/tasks/<task-id>/predictions.jsonl`

---

## Dataset Overview

Bun-Bench contains **100 tasks** organized across multiple categories:

### Categories

| Category | Count | Description |
|----------|-------|-------------|
| Core APIs | 25 | `Bun.serve()`, `Bun.file()`, `Bun.spawn()`, etc. |
| Fetch & Network | 12 | HTTP client, WebSocket, DNS |
| SQLite | 8 | `bun:sqlite` driver issues |
| PostgreSQL | 25 | PostgreSQL driver bugs and features |
| MySQL | 25 | MySQL driver bugs and features |
| Build & Bundle | 10 | `bun build`, transpilation, HMR |
| Testing | 8 | `bun test` framework issues |
| Package Manager | 7 | `bun install`, `bun pm` |

### Difficulty Distribution

| Difficulty | Count | Description |
|------------|-------|-------------|
| Easy | 30 | Single-file fixes, clear reproduction steps |
| Medium | 45 | Multi-file changes, moderate complexity |
| Hard | 25 | Complex architectural changes, deep runtime knowledge |

### Task Types

- **Bug Fix** (70 tasks): Resolve existing incorrect behavior
- **Feature** (30 tasks): Implement new functionality

---

## Running Evaluation

### Basic Evaluation

```bash
# Run inference on specific tasks
python3 -m bunbench.inference.run_api --instances task-001 task-002

# Evaluate predictions (local mode)
python3 -m bunbench evaluate --local --instance-ids task-001

# Evaluate all predictions
python3 -m bunbench evaluate --local
```

### Evaluation Options

```bash
python3 -m bunbench evaluate \
  --dataset <path> \              # Dataset JSON file
  --predictions <path> \           # Predictions JSONL file
  --output <directory> \           # Output directory
  --local \                        # Run locally (no Docker)
  --instance-ids <ids> \           # Specific tasks to evaluate
  --force-rebuild \                # Re-run even if report exists
  --workers <n> \                  # Parallel workers (default: 4)
  --timeout <seconds> \            # Timeout per task (default: 300)
  --verbose                       # Verbose output
```

### Commands Summary

| Command | Description |
|---------|-------------|
| `python3 -m bunbench.inference.run_api` | Run inference |
| `python3 -m bunbench evaluate` | Run evaluation |
| `python3 -m bunbench report` | View/compare reports |
| `python3 -m bunbench build-images` | Build Docker images |

### Programmatic Evaluation

```python
from bun_bench import BunBench, Evaluator

benchmark = BunBench()
evaluator = Evaluator(model="claude-3-opus")

# Evaluate single task
result = evaluator.evaluate_task(benchmark.get_task(1))

# Evaluate all tasks
results = evaluator.evaluate_all(benchmark.tasks)

# Generate report
results.to_report("results/report.json")
```

---

## Retry Flow (Multiple Attempts)

Bun-Bench supports up to 3 attempts per task with error feedback:

### How It Works

```
Attempt 1:
├── Input: Bug code + README
├── Run Inference → attempt-1.json
├── Run Evaluation
└── If PASS → Save report, STOP

Attempt 2 (if failed):
├── Input: Bug code + README + Attempt 1 Error
├── Run Inference → attempt-2.json
├── Run Evaluation
└── If PASS → Save report, STOP

Attempt 3 (if failed):
├── Input: Bug code + README + Attempt 1 & 2 Errors
├── Run Inference → attempt-3.json
└── Run Evaluation → Save report if PASS
```

### Task File Structure

```
task-001/
├── README.md              # Problem description
├── src/                  # Buggy source code
├── test/                 # Test files
├── solution/             # Fixed code
├── attempt-1.json        # 1st inference response
├── attempt-2.json        # 2nd inference (if needed)
├── attempt-3.json        # 3rd inference (if needed)
├── inference-response.json  # Latest response
└── evaluation_report.json  # Final result (only if passed)
```

### Skip Logic

| Condition | Action |
|-----------|--------|
| `attempt-N.json` exists | Skip inference, use existing |
| `evaluation_report.json` exists | Skip evaluation |
| `--force-rebuild` flag | Re-run everything |

---

## Configuration

### Environment Variables

All configuration is done via environment variables (in `.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `BENCH_RUNTIME` | `bun` | Runtime (bun, node, deno) |
| `BENCH_TEST_RUNNER` | `bun test` | Test command |
| `BENCH_LANGUAGE` | `typescript` | Language (js, ts) |
| `BENCH_DATASET` | `dataset/tasks_from_dirs.json` | Dataset path |
| `BENCH_OUTPUT` | `predictions.jsonl` | Output file |
| `BENCH_TIMEOUT` | `300` | Timeout in seconds |
| `BENCH_PROVIDER` | `openai` | API provider |
| `BENCH_MODEL` | `gpt-4-turbo` | Model name |

### Example: Node.js Benchmark

```bash
BENCH_RUNTIME=node
BENCH_TEST_RUNNER="npm test"
BENCH_LANGUAGE=javascript
```

---

## Submitting Results

### Official Leaderboard Submission

To submit your results to the official leaderboard:

1. **Run the full benchmark**
   ```bash
   bun-bench evaluate --model your-model --output results/
   ```

2. **Validate your results**
   ```bash
   bun-bench validate results/
   ```

3. **Generate submission file**
   ```bash
   bun-bench package results/ --output submission.json
   ```

4. **Submit via GitHub**
   - Fork the [bun-bench-results](https://github.com/chittihq/bun-bench-results) repository
   - Add your `submission.json` to `submissions/`
   - Open a Pull Request with your model details

### Submission Requirements

- All 100 tasks must be evaluated
- Results must be reproducible
- Include model configuration details
- Provide inference logs for verification

---

## Leaderboard

<!-- LEADERBOARD_START -->

| Rank | Model | Resolved | % | Date |
|------|-------|----------|---|------|
| - | *Your model here* | - | - | - |

<!-- LEADERBOARD_END -->

*Submit your results to appear on the leaderboard!*

---

## Citation

If you use Bun-Bench in your research, please cite:

```bibtex
@misc{bunbench2026,
  title={Bun-Bench: A Benchmark for Evaluating LLMs on Bun Runtime Issues},
  author={Bun-Bench Contributors},
  year={2026},
  howpublished={\url{https://github.com/chittihq/bun-bench}}
}
```

---

## Related Work

Bun-Bench is inspired by and builds upon the methodology of:

- **[SWE-bench](https://github.com/princeton-nlp/SWE-bench)** - The original benchmark for evaluating LLMs on real-world software engineering tasks from GitHub issues.
- **[SWE-bench Lite](https://github.com/princeton-nlp/SWE-bench)** - A filtered subset of SWE-bench for faster evaluation.

We extend our thanks to the SWE-bench team for pioneering this approach to LLM evaluation.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:

- Adding new tasks
- Improving documentation
- Reporting issues
- Submitting fixes

---

## Links

- [Documentation](docs/index.md)
- [Quick Start Guide](docs/quickstart.md)
- [Evaluation Guide](docs/evaluation.md)
- [Bun Runtime](https://bun.sh)
- [SWE-bench](https://github.com/princeton-nlp/SWE-bench)
