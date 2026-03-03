# Bun-Bench Code Guidelines

## Code Style
- Keep it simple and readable
- No over-engineering or premature abstractions
- Type hints where helpful

## What NOT to Do
- Don't add unused imports or variables
- Don't add docstrings/comments to code you didn't change
- Don't add error handling for impossible cases
- Don't create utility helpers for one-time use

## Patch Application
- **Never rely on model to generate valid unified diffs**
- Models often generate malformed patches (missing +/-, wrong line numbers)
- Solution: Ask model for **full file content**, then generate diff ourselves

## Environment
- Use `BENCH_` prefix for all env vars (not `BUNBENCH_`)
- Dataset in `dataset/tasks_mini2.5.json`
- Predictions saved to `predictions.json`

## Testing
- Run inference: `python3 -m bunbench.inference.run_api --instances <task_id>`
- Run evaluation: `python3 -m bunbench evaluate --local --instance-ids <task_id> --predictions predictions.json`
