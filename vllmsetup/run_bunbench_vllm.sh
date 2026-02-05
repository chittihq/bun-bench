#!/bin/bash
###############################################################################
# Run Bun-Bench Inference with Local vLLM Server
# This script runs the complete Bun-Bench evaluation pipeline
###############################################################################

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Load configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
if [ -f "$SCRIPT_DIR/vllm_config.sh" ]; then
    source "$SCRIPT_DIR/vllm_config.sh"
fi

# Default values
DATASET_PATH="${DATASET_PATH:-dataset/tasks.json}"
OUTPUT_PATH="${OUTPUT_PATH:-results/qwen3-vllm.jsonl}"
MAX_INSTANCES="${MAX_INSTANCES:-}"
PORT=${PORT:-8000}
HOST=${HOST:-localhost}

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}Bun-Bench Inference with vLLM${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Check if vLLM server is running
echo -e "${YELLOW}Checking vLLM server status...${NC}"
if ! curl -s "http://$HOST:$PORT/health" | grep -q "OK" 2>/dev/null; then
    echo -e "${RED}✗ vLLM server is not running${NC}"
    echo -e "  Start the server first: ${GREEN}./start_vllm_server.sh${NC}"
    echo -e "  Or run in background: ${GREEN}./start_vllm_server.sh &${NC}"
    exit 1
fi
echo -e "  ${GREEN}✓ vLLM server is running${NC}"

# Check dataset
echo ""
echo -e "${YELLOW}Checking dataset...${NC}"
if [ ! -f "$DATASET_PATH" ]; then
    echo -e "${RED}✗ Dataset not found: $DATASET_PATH${NC}"
    exit 1
fi
TASK_COUNT=$(python -c "import json; print(len(json.load(open('$DATASET_PATH'))))")
echo -e "  ${GREEN}✓ Dataset found with $TASK_COUNT tasks${NC}"

# Create output directory
mkdir -p "$(dirname "$OUTPUT_PATH")"

# Display configuration
echo ""
echo -e "${BLUE}Configuration:${NC}"
echo -e "  Dataset:        ${GREEN}$DATASET_PATH${NC}"
echo -e "  Output:         ${GREEN}$OUTPUT_PATH${NC}"
echo -e "  Max Instances:  ${GREEN}${MAX_INSTANCES:-All (unlimited)}${NC}"
echo -e "  Server:         ${GREEN}http://$HOST:$PORT/v1${NC}"
echo ""

# Confirm before running
if [ "$AUTO_CONFIRM" != "true" ]; then
    echo -e "${YELLOW}Ready to start inference. Press Ctrl+C to cancel.${NC}"
    sleep 2
fi

# Build inference command
INFERENCE_CMD="python -m bunbench.inference.run_api \
    --dataset \"$DATASET_PATH\" \
    --output \"$OUTPUT_PATH\" \
    --provider openai \
    --model \"$MODEL_NAME\" \
    --base-url \"http://$HOST:$PORT/v1\" \
    --api-key dummy"

# Add optional parameters
if [ -n "$MAX_INSTANCES" ]; then
    INFERENCE_CMD="$INFERENCE_CMD --max-instances $MAX_INSTANCES"
fi

if [ "$VERBOSE" == "true" ]; then
    INFERENCE_CMD="$INFERENCE_CMD --verbose"
fi

# Run inference
echo -e "${YELLOW}Starting inference...${NC}"
echo ""

eval $INFERENCE_CMD

INFERENCE_EXIT_CODE=$?

echo ""
echo -e "${BLUE}============================================${NC}"

if [ $INFERENCE_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ Inference completed successfully!${NC}"
    echo -e "${BLUE}============================================${NC}"
    echo ""
    echo -e "Results saved to: ${GREEN}$OUTPUT_PATH${NC}"
    echo ""
    echo -e "Next steps:"
    echo -e "  1. View results: ${GREEN}cat $OUTPUT_PATH${NC}"
    echo -e "  2. Run evaluation: ${GREEN}./run_bunbench_eval.sh \"$OUTPUT_PATH\"${NC}"
    echo ""

    # Display summary statistics
    if [ -f "$OUTPUT_PATH" ]; then
        TOTAL=$(wc -l < "$OUTPUT_PATH")
        SUCCESS=$(grep -c '"success":true' "$OUTPUT_PATH" || echo 0)
        FAILED=$(grep -c '"success":false' "$OUTPUT_PATH" || echo 0)

        echo -e "${BLUE}Quick Statistics:${NC}"
        echo -e "  Total tasks:     ${GREEN}$TOTAL${NC}"
        echo -e "  Successful:      ${GREEN}$SUCCESS${NC}"
        echo -e "  Failed:          ${RED}$FAILED${NC}"
    fi
else
    echo -e "${RED}✗ Inference failed with exit code $INFERENCE_EXIT_CODE${NC}"
    echo -e "${BLUE}============================================${NC}"
fi

exit $INFERENCE_EXIT_CODE
