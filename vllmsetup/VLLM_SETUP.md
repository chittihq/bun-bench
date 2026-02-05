# vLLM Setup Guide for Bun-Bench on L40 GPU

Complete setup for running Qwen3-80B locally on NVIDIA L40 GPU using vLLM.

---

## Prerequisites

### Hardware
- **GPU**: NVIDIA L40 (48GB VRAM)
- **RAM**: 64GB+ recommended
- **Storage**: 200GB+ free space (for model weights)

### Software
- **OS**: Linux (Ubuntu 20.04+) or WSL2 on Windows
- **CUDA**: 12.x
- **NVIDIA Driver**: 525.x+
- **Python**: 3.9+

### Check Your Setup
```bash
# Check GPU
nvidia-smi

# Check CUDA
nvcc --version

# Check Python
python --version
```

---

## Installation Steps

### 1. Install Python Dependencies

```bash
cd /path/to/bun-bench
pip install -r vllm_requirements.txt
```

### 2. Install HuggingFace CLI

```bash
pip install huggingface_hub
huggingface-cli login
```

### 3. Accept Model Terms

Visit these URLs and accept the license:
- https://huggingface.co/qwen/qwen3-next-80b-a3b-instruct

---

## Quick Start

### Option A: Automatic Setup (Recommended)

```bash
# 1. Start vLLM server
chmod +x *.sh
./start_vllm_server.sh
```

The server will:
- Check your GPU
- Download Qwen3-80B model (first time only, ~40GB)
- Start API server on port 8000

**Wait for:** `Uvicorn running on http://0.0.0.0:8000`

### In a NEW terminal:

```bash
# 2. Test the server
./test_vllm_server.sh

# 3. Run Bun-Bench (test with 1 task first)
MAX_INSTANCES=1 ./run_bunbench_vllm.sh

# 4. If successful, run full benchmark
./run_bunbench_vllm.sh
```

---

## Option B: Manual Setup

### Step 1: Configure vLLM

Edit `vllm_config.sh` if needed:

```bash
# For L40 with 48GB VRAM
export GPU_MEMORY_UTILIZATION=0.95   # Use 95% of 48GB
export QUANTIZATION="awq"            # 4-bit quantization
export MAX_MODEL_LEN=4096            # Context window
export PORT=8000
```

### Step 2: Start Server

```bash
./start_vllm_server.sh
```

**Expected output:**
```
============================================
Bun-Bench vLLM Server Setup
============================================

Checking GPU availability...
  GPU: L40
  Memory: 48753 MB free / 48753 MB total

Checking vLLM installation...
  ✓ vLLM is installed

Configuration:
  Model:           qwen/qwen3-next-80b-a3b-instruct
  Quantization:    awq
  Data Type:       half
  Max Length:      4096
  GPU Memory:      95%
  Server:          0.0.0.0:8000

Starting vLLM server...
  Log file: logs/vllm_20240101_120000.log
  This will take 2-5 minutes on first run...
```

### Step 3: Verify Server

```bash
curl http://localhost:8000/v1/models
```

**Expected response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "qwen/qwen3-next-80b-a3b-instruct",
      "object": "model"
    }
  ]
}
```

---

## Running Bun-Bench

### Test Run (1 Task)

```bash
MAX_INSTANCES=1 ./run_bunbench_vllm.sh
```

**Expected output:**
```
============================================
Bun-Bench Inference with vLLM
============================================

Checking vLLM server status...
  ✓ vLLM server is running

Checking dataset...
  ✓ Dataset found with 100 tasks

Configuration:
  Dataset:        dataset/tasks.json
  Output:         results/qwen3-vllm.jsonl
  Max Instances:  1
  Server:         http://localhost:8000/v1

Starting inference...
2024-01-01 12:00:00 - INFO - Loading dataset from dataset/tasks.json
2024-01-01 12:00:00 - INFO - Loaded 100 instances
2024-01-01 12:00:01 - INFO - Processing instance: bun-bench-001
  Completed: tokens=1234+567, cost=$0.0000, patch=yes

✓ Inference completed successfully!

Quick Statistics:
  Total tasks:     1
  Successful:      1
  Failed:          0
```

### Full Benchmark (100 Tasks)

```bash
# Remove MAX_INSTANCES for full run
./run_bunbench_vllm.sh
```

**Expected time:** 1-2 hours on L40

### Monitoring Progress

In another terminal:

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Watch output file
tail -f results/qwen3-vllm.jsonl
```

---

## Evaluating Results

After inference completes, evaluate the patches:

```bash
./run_bunbench_eval.sh results/qwen3-vllm.jsonl
```

**Expected time:** 2-3 hours (runs Docker tests for each patch)

---

## Troubleshooting

### Out of Memory Error

**Symptom:** `OutOfMemoryError: CUDA out of memory`

**Solutions:**

1. Reduce GPU memory utilization:
```bash
# Edit vllm_config.sh
export GPU_MEMORY_UTILIZATION=0.85  # Instead of 0.95
```

2. Reduce context window:
```bash
export MAX_MODEL_LEN=2048  # Instead of 4096
```

3. Use stronger quantization:
```bash
export QUANTIZATION="gptq"  # Instead of awq
```

### Model Download Stuck

**Symptom:** Download hangs at same percentage

**Solution:**
```bash
# Clear cache and retry
rm -rf ~/.cache/huggingface/hub/
./start_vllm_server.sh
```

### Server Won't Start

**Check logs:**
```bash
tail -100 logs/vllm_*.log
```

**Common issues:**

1. **Port already in use:**
```bash
# Change port in vllm_config.sh
export PORT=8001
```

2. **Wrong CUDA version:**
```bash
# Reinstall PyTorch with correct CUDA
pip install torch --upgrade --index-url https://download.pytorch.org/whl/cu121
```

### Inference Fails

**Test server manually:**
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen/qwen3-next-80b-a3b-instruct",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 50
  }'
```

---

## Performance Tuning

### For Faster Inference

Edit `vllm_config.sh`:

```bash
# Increase batch size
export MAX_NUM_SEQS=512        # Default: 256
export MAX_NUM_BATCHED_TOKENS=8192  # Default: 4096

# Enable experimental features
export USE_V2_BLOCK_MANAGER=true
```

### For Better Quality

```bash
# Remove quantization (slower but better)
export QUANTIZATION=""

# Use full precision
export DTYPE="float32"
```

**Warning:** Without quantization, you need ~160GB VRAM (3-4 L40 GPUs).

---

## Advanced: Multi-GPU Setup

If you have multiple L40 GPUs:

```bash
# Edit vllm_config.sh
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Use 4 GPUs
export TENSOR_PARALLEL_SIZE=4         # Split model across GPUs
export QUANTIZATION=""                # No quantization needed
```

---

## Cost Comparison

| Setup | Cost per 100 tasks | Speed |
|-------|-------------------|-------|
| **OpenRouter** | ~$15-20 | Fast (datacenter GPUs) |
| **vLLM on L40** | $0 (free) | Medium (consumer GPU) |

**Break-even point:** ~1 full benchmark run pays for the L40 electricity.

---

## File Reference

### Scripts Created

- `vllm_requirements.txt` - Python dependencies
- `vllm_config.sh` - vLLM configuration
- `start_vllm_server.sh` - Start vLLM server
- `test_vllm_server.sh` - Test server connectivity
- `run_bunbench_vllm.sh` - Run Bun-Bench inference
- `run_bunbench_eval.sh` - Evaluate patches

### Output Files

- `logs/vllm_*.log` - vLLM server logs
- `results/qwen3-vllm.jsonl` - Inference results
- `eval_results/eval_*/` - Evaluation reports

---

## Getting Help

1. **Check logs:** `tail -f logs/vllm_*.log`
2. **Test server:** `./test_vllm_server.sh`
3. **Check GPU:** `nvidia-smi`
4. **Check Docker:** `docker ps`

---

## Next Steps

1. ✅ Install vLLM: `pip install -r vllm_requirements.txt`
2. ✅ Start server: `./start_vllm_server.sh`
3. ✅ Test server: `./test_vllm_server.sh`
4. ✅ Run inference: `./run_bunbench_vllm.sh`
5. ✅ Evaluate: `./run_bunbench_eval.sh results/qwen3-vllm.jsonl`

Good luck! 🚀
