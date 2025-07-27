# SimpleTIR Quick Start Guide

## 30-Minute Setup for 8xH100 Node

### Prerequisites Check
```bash
# Verify GPU availability
nvidia-smi  # Should show 8x H100 GPUs

# Check Python version
python3 --version  # Should be 3.8+

# Verify disk space
df -h .  # Need at least 100GB free
```

### Step 1: Environment Setup (5 minutes)
```bash
# You should already be on artifact-evaluation branch with .venv created
source .venv/bin/activate

# Quick dependency install
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install vllm==0.8.5
pip install -e .
```

### Step 2: Quick Test (5 minutes)
```bash
# Test imports
python -c "import verl, torch, vllm; print('All imports successful')"
python -c "import torch; print(f'GPUs available: {torch.cuda.device_count()}')"

# Set sandbox to local mode for testing
export SANDBOX_ENDPOINT=local
```

### Step 3: Download Minimal Test Data (10 minutes)
```bash
# Create minimal test setup
mkdir -p models datasets/test logs checkpoints

# For quick testing, you can use any Qwen model from HuggingFace
# Example: Download Qwen2.5-7B-Instruct (smaller, faster to download)
git clone https://huggingface.co/Qwen/Qwen2.5-7B-Instruct models/Qwen2.5-7B-Instruct
```

### Step 4: Run First Experiment (10 minutes)

#### Option A: If you have evaluation data
```bash
MODEL_PATH=./models \
DATA_PATH=./datasets \
LOG_PATH=./logs \
NNODES=1 GPUS_PER_NODE=8 \
CONFIG_NAME=simpletir_trainer \
bash train.sh \
  --val_only True \
  --model_name Qwen2.5-7B-Instruct \
  --valid_dataset "test/sample" \
  --n_val 4 \
  --val_sample_size 10
```

#### Option B: Training test run (verify setup)
```bash
MODEL_PATH=./models \
DATA_PATH=./datasets \
CHECKPOINT_PATH=./checkpoints \
LOG_PATH=./logs \
NNODES=1 GPUS_PER_NODE=8 \
CONFIG_NAME=simpletir_trainer \
bash train.sh \
  --model_name Qwen2.5-7B-Instruct \
  --train_batch_size 64 \
  --total_epochs 1 \
  --save_freq 1
```

## What to Expect

### Successful Setup Indicators:
- No import errors
- All 8 GPUs detected
- Training/evaluation starts without errors
- Logs appear in `logs/` directory
- GPU utilization visible in `nvidia-smi`

### Common Quick Fixes:

**CUDA Error**:
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

**Memory Error**:
```bash
# Reduce batch size
--train_batch_size 32  # Instead of 512
```

**vLLM Import Error**:
```bash
pip uninstall vllm
pip install vllm==0.8.5 --no-cache-dir
```

## Next Steps

1. **Get Real Datasets**: Contact authors or prepare datasets in parquet format
2. **Download Full Models**: Get complete Qwen2.5 models for better results  
3. **Configure Sandbox**: Set up proper code execution sandbox for tool use
4. **Launch Full Training**: Use parameters from EXPERIMENT_WORKFLOW.md

## Emergency Commands

```bash
# Kill all GPU processes
sudo fuser -v /dev/nvidia* | awk '{print $2}' | xargs -I {} kill -9 {}

# Clear GPU memory
nvidia-smi --gpu-reset

# Monitor GPU usage
watch -n 1 nvidia-smi

# Check training progress
tail -f logs/*.log
```

## Contact for Help

- Check existing issues in the repository
- Refer to detailed documentation in SETUP_AND_REQUIREMENTS.md
- For dataset access, contact paper authors

---

**Remember**: This quick start uses minimal settings. For paper reproduction, use full parameters from the detailed guides.