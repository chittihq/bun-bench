# Quick Start

## Install
```bash
pip install bun-bench
```

## Configure (.env)
```bash
BENCH_RUNTIME=bun
BENCH_TEST_RUNNER="bun test"
BENCH_LANGUAGE=typescript
BENCH_PROVIDER=openrouter
BENCH_MODEL=minimax/minimax-m2.5
```

## Run
```bash
# 1. Inference
python3 -m bunbench.inference.run_api --instances task-001

# 2. Evaluate
python3 -m bunbench evaluate --local --instance-ids task-001
```

## Results
- `dataset/tasks/<task-id>/predictions.jsonl`
- `dataset/tasks/<task-id>/evaluation_report.json`
- `logs/run_*.log`

## Retry (up to 3 attempts)
- Saves each attempt to `attempt-1.json`, `attempt-2.json`, `attempt-3.json`
- Includes previous errors in prompt for next attempt
- Report saved only if all tests pass
