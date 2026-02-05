#!/bin/bash
###############################################################################
# Test vLLM Server Connectivity
# This script verifies that the vLLM server is working correctly
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

# Use defaults if not set
PORT=${PORT:-8000}
HOST=${HOST:-localhost}

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}Testing vLLM Server${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""

# Test 1: Server health check
echo -e "${YELLOW}Test 1: Checking server health...${NC}"
HEALTH_URL="http://$HOST:$PORT/health"

if command -v curl &> /dev/null; then
    if curl -s "$HEALTH_URL" | grep -q "OK"; then
        echo -e "  ${GREEN}✓ Server is healthy${NC}"
    else
        echo -e "  ${RED}✗ Server health check failed${NC}"
        echo -e "  Make sure the server is running: ${GREEN}./start_vllm_server.sh${NC}"
        exit 1
    fi
else
    echo -e "  ${YELLOW}⚠ curl not found, skipping health check${NC}"
fi

echo ""

# Test 2: List available models
echo -e "${YELLOW}Test 2: Listing available models...${NC}"
MODELS_URL="http://$HOST:$PORT/v1/models"

if command -v curl &> /dev/null; then
    MODELS=$(curl -s "$MODELS_URL")
    if echo "$MODELS" | grep -q "qwen"; then
        echo -e "  ${GREEN}✓ Model is available${NC}"
        echo "$MODELS" | python -m json.tool 2>/dev/null || echo "$MODELS"
    else
        echo -e "  ${RED}✗ Model not found${NC}"
        echo "  Response: $MODELS"
        exit 1
    fi
else
    echo -e "  ${YELLOW}⚠ curl not found, skipping model check${NC}"
fi

echo ""

# Test 3: Simple inference test
echo -e "${YELLOW}Test 3: Testing inference...${NC}"

# Create a simple test request
TEST_PROMPT="Hello, how are you?"

if command -v curl &> /dev/null; then
    echo -e "  Sending test prompt: ${GREEN}\"$TEST_PROMPT\"${NC}"

    RESPONSE=$(curl -s -X POST "http://$HOST:$PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"$MODEL_NAME\"",
            \"messages\": [
                {\"role\": \"user\", \"content\": \"$TEST_PROMPT\"}
            ],
            \"max_tokens\": 50,
            \"temperature\": 0.0
        }")

    if echo "$RESPONSE" | grep -q "choices"; then
        echo -e "  ${GREEN}✓ Inference successful${NC}"
        echo ""
        echo -e "${BLUE}Response:${NC}"
        echo "$RESPONSE" | python -c "import sys, json; data=json.load(sys.stdin); print(data['choices'][0]['message']['content'])" 2>/dev/null || echo "$RESPONSE"
    else
        echo -e "  ${RED}✗ Inference failed${NC}"
        echo "  Response: $RESPONSE"
        exit 1
    fi
else
    echo -e "  ${YELLOW}⚠ curl not found, skipping inference test${NC}"
fi

echo ""
echo -e "${BLUE}============================================${NC}"
echo -e "${GREEN}All tests passed! ✓${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo -e "Server is ready for Bun-Bench inference."
echo -e "Run: ${GREEN}./run_bunbench_vllm.sh${NC}"
echo ""
