#!/bin/bash
###############################################################################
# Run Bun-Bench Evaluation
# This script evaluates the generated patches against the benchmark
###############################################################################

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
DATASET_PATH="${DATASET_PATH:-dataset/tasks.json}"
PREDICTIONS_PATH="${1}"
OUTPUT_DIR="${OUTPUT_DIR:-eval_results}"

if [ -z "$PREDICTIONS_PATH" ]; then
    echo -e "${RED}Error: Please provide predictions file path${NC}"
    echo -e "Usage: ${GREEN}./run_bunbench_eval.sh <predictions.jsonl>${NC}"
    echo -e "Example: ${GREEN}./run_bunbench_eval.sh results/qwen3-vllm.jsonl${NC}"
    exit 1
fi

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}Bun-Bench Evaluation${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Check if predictions file exists
echo -e "${YELLOW}Checking predictions file...${NC}"
if [ ! -f "$PREDICTIONS_PATH" ]; then
    echo -e "${RED}✗ Predictions file not found: $PREDICTIONS_PATH${NC}"
    exit 1
fi

# Count predictions
PREDICTION_COUNT=$(wc -l < "$PREDICTIONS_PATH")
echo -e "  ${GREEN}✓ Found $PREDICTION_COUNT predictions${NC}"

# Check dataset
echo ""
echo -e "${YELLOW}Checking dataset...${NC}"
if [ ! -f "$DATASET_PATH" ]; then
    echo -e "${RED}✗ Dataset not found: $DATASET_PATH${NC}"
    exit 1
fi
echo -e "  ${GREEN}✓ Dataset found${NC}"

# Check Docker
echo ""
echo -e "${YELLOW}Checking Docker...${NC}"
if ! command -v docker &> /dev/null; then
    echo -e "${RED}✗ Docker not installed${NC}"
    exit 1
fi
if ! docker info &> /dev/null; then
    echo -e "${RED}✗ Docker daemon not running${NC}"
    echo -e "  Start Docker with: ${GREEN}sudo systemctl start docker${NC}"
    exit 1
fi
echo -e "  ${GREEN}✓ Docker is running${NC}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Generate output directory name with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EVAL_OUTPUT_DIR="$OUTPUT_DIR/eval_$TIMESTAMP"

echo ""
echo -e "${BLUE}Configuration:${NC}"
echo -e "  Dataset:      ${GREEN}$DATASET_PATH${NC}"
echo -e "  Predictions:  ${GREEN}$PREDICTIONS_PATH${NC}"
echo -e "  Output:       ${GREEN}$EVAL_OUTPUT_DIR${NC}"
echo -e "  Workers:      ${GREEN}${WORKERS:-4}${NC}"
echo ""

# Confirm before running
if [ "$AUTO_CONFIRM" != "true" ]; then
    echo -e "${YELLOW}This will take 1-3 hours. Press Ctrl+C to cancel.${NC}"
    sleep 3
fi

# Convert JSONL to JSON for evaluation (if needed)
echo -e "${YELLOW}Preparing predictions for evaluation...${NC}"
TEMP_PREDICTIONS="$OUTPUT_DIR/temp_predictions.json"

python -c "
import json
import sys

# Read JSONL and convert to JSON dict
predictions = {}
with open('$PREDICTIONS_PATH', 'r') as f:
    for line in f:
        if line.strip():
            data = json.loads(line)
            instance_id = data.get('instance_id')
            patch = data.get('extracted_patch')
            if instance_id and patch:
                predictions[instance_id] = patch

# Write as JSON dict
with open('$TEMP_PREDICTIONS', 'w') as f:
    json.dump(predictions, f, indent=2)

print(f'Converted {len(predictions)} predictions')
"

echo -e "  ${GREEN}✓ Predictions prepared${NC}"

# Run evaluation
echo ""
echo -e "${YELLOW}Starting evaluation...${NC}"
echo ""

python -m bunbench evaluate \
    --dataset "$DATASET_PATH" \
    --predictions "$TEMP_PREDICTIONS" \
    --output "$EVAL_OUTPUT_DIR" \
    --workers "${WORKERS:-4}" \
    --timeout "${TIMEOUT:-300}" \
    ${VERBOSE:+--verbose}

EVAL_EXIT_CODE=$?

# Cleanup temp file
rm -f "$TEMP_PREDICTIONS"

echo ""
echo -e "${BLUE}============================================${NC}"

if [ $EVAL_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ Evaluation completed successfully!${NC}"
    echo -e "${BLUE}============================================${NC}"
    echo ""
    echo -e "Results saved to: ${GREEN}$EVAL_OUTPUT_DIR${NC}"

    # Display summary if report exists
    REPORT_FILE="$EVAL_OUTPUT_DIR/evaluation_report.json"
    if [ -f "$REPORT_FILE" ]; then
        echo ""
        echo -e "${BLUE}Evaluation Summary:${NC}"
        python -c "
import json
with open('$REPORT_FILE', 'r') as f:
    report = json.load(f)

total = report.get('total', 0)
resolved = report.get('resolved', 0)
partial = report.get('partial', 0)
unresolved = report.get('unresolved', 0)
errors = report.get('errors', 0)

print(f'  Total Tasks:     {total}')
print(f'  FULL Resolved:   {resolved} ({resolved/total*100 if total > 0 else 0:.1f}%)')
print(f'  PARTIAL Resolved: {partial} ({partial/total*100 if total > 0 else 0:.1f}%)')
print(f'  NOT Resolved:    {unresolved} ({unresolved/total*100 if total > 0 else 0:.1f}%)')
print(f'  Errors:          {errors} ({errors/total*100 if total > 0 else 0:.1f}%)')
"
    fi

    echo ""
    echo -e "View detailed report: ${GREEN}python -m bunbench report \"$REPORT_FILE\"${NC}"
else
    echo -e "${RED}✗ Evaluation failed with exit code $EVAL_EXIT_CODE${NC}"
    echo -e "${BLUE}============================================${NC}"
fi

exit $EVAL_EXIT_CODE
