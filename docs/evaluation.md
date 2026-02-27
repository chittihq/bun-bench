# Evaluation Guide

## Configure (.env)
```bash
BENCH_RUNTIME=bun
BENCH_TEST_RUNNER="bun test"
BENCH_LANGUAGE=typescript
BENCH_TIMEOUT=300
```

## Run Evaluation
```bash
# Single task
python3 -m bunbench evaluate --local --instance-ids task-001

# Multiple tasks
python3 -m bunbench evaluate --local

# With options
python3 -m bunbench evaluate --local --workers 4 --verbose
```

## Options
| Option | Description |
|--------|-------------|
| `--local` | Run locally (no Docker) |
| `--instance-ids` | Specific tasks |
| `--force-rebuild` | Re-run even if report exists |
| `--workers` | Parallel workers |
| `--timeout` | Timeout per task |

## Retry Flow

Up to 3 attempts per task:

```
Attempt 1: Input → Inference → Eval → PASS? → STOP
                         ↓ FAIL
Attempt 2: + Attempt 1 Error → Inference → Eval → PASS? → STOP
                         ↓ FAIL
Attempt 3: + Attempt 1&2 Errors → Inference → Eval
```

## Results

| File | Location |
|------|----------|
| Predictions | `dataset/tasks/<task-id>/predictions.jsonl` |
| Report | `dataset/tasks/<task-id>/evaluation_report.json` |
| Log | `logs/run_*.log` |

## Task Files

```
task-001/
├── attempt-1.json      # 1st attempt
├── attempt-2.json      # 2nd attempt (if needed)
├── attempt-3.json      # 3rd attempt (if needed)
├── inference-response.json
└── evaluation_report.json  # Only if passed
```
