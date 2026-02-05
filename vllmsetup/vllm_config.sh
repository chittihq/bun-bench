#!/bin/bash
###############################################################################
# vLLM Configuration for Bun-Bench with Qwen3-80B on L40 GPU
###############################################################################

# GPU Configuration
export CUDA_VISIBLE_DEVICES=0
export GPU_MEMORY_UTILIZATION=0.95
export GPU_COUNT=1

# Model Configuration
export MODEL_NAME="qwen/qwen3-next-80b-a3b-instruct"
export QUANTIZATION="awq"  # Options: awq, gptq, None (no quantization)
export DTYPE="half"  # Options: half, bfloat16, float32

# Server Configuration
export HOST="0.0.0.0"
export PORT=8000
export MAX_MODEL_LEN=4096
export TOKENIZER_MODE=auto

# Performance Tuning
export TENSOR_PARALLEL_SIZE=1  # Number of GPUs (for multi-GPU setups)
export BLOCK_SIZE=16
export MAX_NUM_SEQS=256
export MAX_NUM_BATCHED_TOKENS=4096

# Logging
export LOG_LEVEL=info

###############################################################################
# Usage:
#   source vllm_config.sh
#   Then run start_vllm_server.sh
###############################################################################
