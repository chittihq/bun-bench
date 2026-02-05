#!/bin/bash
###############################################################################
# Start vLLM Server for Bun-Bench
# This script launches vLLM with Qwen3-80B on L40 GPU
###############################################################################

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Load configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
if [ -f "$SCRIPT_DIR/vllm_config.sh" ]; then
    source "$SCRIPT_DIR/vllm_config.sh"
else
    # Default values if config file doesn't exist
    export MODEL_NAME="qwen/qwen3-next-80b-a3b-instruct"
    export QUANTIZATION="awq"
    export DTYPE="half"
    export HOST="0.0.0.0"
    export PORT=8000
    export MAX_MODEL_LEN=4096
    export GPU_MEMORY_UTILIZATION=0.95
    export TENSOR_PARALLEL_SIZE=1
fi

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}Bun-Bench vLLM Server Setup${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Check GPU availability
echo -e "${YELLOW}Checking GPU availability...${NC}"
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}Error: nvidia-smi not found. Is NVIDIA driver installed?${NC}"
    exit 1
fi

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits | \
while IFS=, read -r name total free; do
    echo -e "  GPU: ${GREEN}$name${NC}"
    echo -e "  Memory: ${GREEN}$free${NC} MB free / $total MB total"
done

echo ""
echo -e "${YELLOW}Checking vLLM installation...${NC}"
if ! python -c "import vllm" 2>/dev/null; then
    echo -e "${RED}Error: vLLM not installed.${NC}"
    echo -e "  Install with: ${GREEN}pip install -r vllm_requirements.txt${NC}"
    exit 1
fi
echo -e "  ${GREEN}✓ vLLM is installed${NC}"

# Check if HuggingFace token is needed
echo ""
echo -e "${YELLOW}Checking model access...${NC}"
if python -c "from huggingface_hub import login; login(token='dummy', new_session=False)" 2>&1 | grep -q "User Access Token"; then
    echo -e "${YELLOW}Warning: HuggingFace token may be required${NC}"
    echo -e "  1. Login at: ${GREEN}huggingface-cli login${NC}"
    echo -e "  2. Accept model terms at: https://huggingface.co/$MODEL_NAME"
    echo ""
    read -p "Press Enter to continue or Ctrl+C to exit..."
fi

# Display configuration
echo ""
echo -e "${BLUE}Configuration:${NC}"
echo -e "  Model:           ${GREEN}$MODEL_NAME${NC}"
echo -e "  Quantization:    ${GREEN}$QUANTIZATION${NC}"
echo -e "  Data Type:       ${GREEN}$DTYPE${NC}"
echo -e "  Max Length:      ${GREEN}$MAX_MODEL_LEN${NC}"
echo -e "  GPU Memory:      ${GREEN}${GPU_MEMORY_UTILIZATION}%${NC}"
echo -e "  Server:          ${GREEN}$HOST:$PORT${NC}"
echo ""

# Create logs directory
mkdir -p logs
LOG_FILE="logs/vllm_$(date +%Y%m%d_%H%M%S).log"

echo -e "${YELLOW}Starting vLLM server...${NC}"
echo -e "  Log file: ${GREEN}$LOG_FILE${NC}"
echo -e "${YELLOW}This will take 2-5 minutes on first run (downloading model)...${NC}"
echo ""

# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_NAME" \
    --host "$HOST" \
    --port "$PORT" \
    --quantization "$QUANTIZATION" \
    --dtype "$DTYPE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --block-size 16 \
    --max-num-seqs 256 \
    --trust-remote-code \
    --disable-log-requests \
    --log-level "$LOG_LEVEL" \
    2>&1 | tee "$LOG_FILE"

###############################################################################
# Server is now running and listening for API requests
# Test it with: curl http://localhost:$PORT/v1/models
###############################################################################
