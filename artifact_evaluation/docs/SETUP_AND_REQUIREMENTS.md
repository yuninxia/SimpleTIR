# SimpleTIR Artifact Evaluation: Setup and Requirements Guide

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Setup Options by Priority](#setup-options-by-priority)
3. [Detailed Setup Instructions](#detailed-setup-instructions)
4. [Storage Planning](#storage-planning)
5. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Hardware Requirements

#### GPU Requirements
- **Minimum**: 1x NVIDIA GPU with 24GB+ VRAM (e.g., RTX 3090, A10)
- **Recommended**: 8x NVIDIA H100 80GB GPUs
- **CUDA Version**: 12.x (required for H100)

**GPU Memory Requirements by Model Size:**
- 3B models: ~6GB VRAM minimum
- 7B models: ~14-15GB VRAM (FP16)
- 14B models: ~28-30GB VRAM
- 32B models: ~65GB VRAM (requires multi-GPU or H100)

#### CPU and System Memory
- **CPU**: 16+ cores recommended for data preprocessing & inference management
- **RAM**: 
  - Minimum: 64GB
  - Recommended: 128GB+ (for tokenizers, caches, parallel requests)
  - For multi-GPU setups: 256GB+

#### Storage Requirements

**Disk Space Breakdown:**

1. **Base Installation** (~10GB)
   - SimpleTIR repository: ~54MB
   - Python virtual environment: ~25MB
   - PyTorch + CUDA: ~3-4GB
   - vLLM 0.8.5: ~1-2GB
   - Other dependencies: ~2-3GB

2. **Model Weights** (per model)
   - Qwen2.5-3B: ~6GB
   - Qwen2.5-7B: ~14-15GB
   - Qwen2.5-14B: ~28-30GB
   - Qwen2.5-32B: ~65GB

3. **Training Storage**
   - Checkpoints: Model size × 10 (includes optimizer states)
     - 7B model checkpoint: ~70GB each
     - 32B model checkpoint: ~320GB each
   - Training logs & wandb: ~2-5GB
   - Temporary files: ~10-20GB

4. **Datasets**
   - SimpleTIR datasets: ~500MB-2GB
   - Custom datasets: Variable

**Total Space Requirements:**
- **Evaluation only**: 30-50GB minimum
- **Training 7B model**: 200-300GB recommended
- **Training 32B model**: 1TB+ recommended

**Storage Type:**
- Required: SSD (for reasonable performance)
- Recommended: NVMe SSD (for optimal checkpoint I/O)

---

## Setup Options by Priority

### Priority 1: Quick Evaluation Demo (1-2 hours)
**Best for**: Quickly testing pre-trained models, verifying setup

**Requirements**:
- Pre-trained model weights
- 30-50GB disk space
- 1 GPU with 24GB+ VRAM

**Steps**:
```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
pip install vllm==0.8.5

# 3. Run evaluation
MODEL_PATH=/path/to/models DATA_PATH=/path/to/data \
bash train.sh --val_only True --valid_dataset "deepscaler/aime" \
  --model_name Qwen2.5-7B --n_val 32 --val_sample_size 100
```

### Priority 2: Docker Container Setup (2-3 hours)
**Best for**: Reproducible environment, avoiding dependency conflicts

**Requirements**:
- Docker with NVIDIA Container Toolkit
- 50-100GB disk space
- GPU access in Docker

**Steps**:
```bash
# 1. Build Docker image
docker build -f docker/Dockerfile.ngc.vllm0.8 -t simpletir:latest .

# 2. Run container with GPU access
docker run --gpus all \
  -v $(pwd):/workspace \
  -v /path/to/models:/models \
  -v /path/to/data:/data \
  -it simpletir:latest bash

# 3. Inside container, run training/evaluation
cd /workspace
bash train.sh [your options]
```

### Priority 3: Native Installation (1-2 hours)
**Best for**: Development, debugging, customization

**Requirements**:
- Python 3.8+
- CUDA 12.x
- 50GB+ disk space

**Steps**:
```bash
# 1. Setup environment
source .venv/bin/activate
pip install --upgrade pip

# 2. Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Install dependencies
pip install -r requirements.txt
pip install vllm==0.8.5

# 4. Install verl package
pip install -e .

# 5. Configure sandbox
export SANDBOX_ENDPOINT=local  # or your sandbox URL

# 6. Verify installation
python -c "import verl; print('VERL installed successfully')"
python -c "import torch; print(f'PyTorch CUDA available: {torch.cuda.is_available()}')"
```

### Priority 4: Full Training Run (Days to weeks)
**Best for**: Training new models, reproducing paper results

**Requirements**:
- 8x H100 GPUs (recommended)
- 500GB-1TB disk space
- Stable multi-day runtime

**Steps**:
```bash
# 1. Prepare datasets
cd datasets/
# Download or prepare your parquet files

# 2. Configure environment
export MODEL_PATH=/path/to/base/models
export DATA_PATH=/path/to/datasets
export CHECKPOINT_PATH=./checkpoints
export LOG_PATH=./logs
export SANDBOX_ENDPOINT=your_sandbox_url

# 3. Run multi-node training
NNODES=1 GPUS_PER_NODE=8 RESUME=False \
CONFIG_NAME=simpletir_trainer \
bash train.sh \
  --model_name Qwen2.5-7B \
  --train_batch_size 512 \
  --max_turns 5 \
  --total_epochs 100 \
  --train_dataset "simplelr_math_35/train deepscaler/train"
```

---

## Detailed Setup Instructions

### Step 1: Environment Preparation

```bash
# Clone and setup branch (if not already done)
git checkout -b artifact-evaluation

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### Step 2: CUDA Setup Verification

```bash
# Check CUDA availability
nvidia-smi

# Verify CUDA version (should be 12.x for H100)
nvcc --version

# If CUDA not found, add to PATH
export PATH=/usr/local/cuda-12/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12/lib64:$LD_LIBRARY_PATH
```

### Step 3: Dependency Installation

```bash
# Install PyTorch with appropriate CUDA version
# For CUDA 12.1 (H100):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
pip install -r requirements.txt

# Install vLLM (specific version required)
pip install vllm==0.8.5

# Install verl package in development mode
pip install -e .
```

### Step 4: Sandbox Configuration

SimpleTIR requires a code execution sandbox for tool-integrated reasoning:

```bash
# Option 1: Use local sandbox (limited security)
export SANDBOX_ENDPOINT=local

# Option 2: Use internal sandbox (if available)
export SANDBOX_ENDPOINT=http://your-sandbox-endpoint:port

# Option 3: Setup firejail sandbox (recommended for security)
sudo apt-get install firejail
# Configure sandbox/local_sandbox.py for firejail
```

### Step 5: Data Preparation

```bash
# Create data directory structure
mkdir -p datasets/deepscaler
mkdir -p datasets/simplelr_math_35

# Download datasets (example paths)
# The datasets should be in parquet format
# Place them in the appropriate directories
```

### Step 6: Model Preparation

```bash
# Create model directory
mkdir -p models

# Download base models (e.g., from HuggingFace)
# Example for Qwen2.5-7B:
git clone https://huggingface.co/Qwen/Qwen2.5-7B models/Qwen2.5-7B
```

---

## Storage Planning

### Recommended Directory Structure
```
SimpleTIR/
├── .venv/              # Python virtual environment
├── models/             # Base model weights
│   ├── Qwen2.5-7B/
│   └── Qwen2.5-32B/
├── datasets/           # Training/evaluation data
│   ├── deepscaler/
│   └── simplelr_math_35/
├── checkpoints/        # Training checkpoints
├── logs/               # Training logs
└── wandb/              # Weights & Biases logs
```

### Storage Management Tips

1. **Use Symbolic Links for Large Files**
   ```bash
   # Link models from external storage
   ln -s /mnt/storage/models ./models
   ```

2. **Enable Checkpoint Compression**
   - Reduces checkpoint size by 30-50%
   - Slight overhead during save/load

3. **Implement Checkpoint Rotation**
   - Keep only last N checkpoints
   - Archive older checkpoints to cold storage

4. **Monitor Disk Usage**
   ```bash
   # Check disk usage
   df -h .
   du -sh checkpoints/
   ```

---

## Troubleshooting

### Common Issues and Solutions

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   --train_batch_size 256  # Instead of 512
   
   # Enable gradient checkpointing
   --gradient_checkpointing True
   
   # Use model parallelism
   --rollout_tp 2  # Tensor parallel size
   ```

2. **vLLM Installation Failures**
   ```bash
   # Install specific CUDA toolkit
   conda install cuda-toolkit=12.1
   
   # Or use pre-built wheel
   pip install https://github.com/vllm-project/vllm/releases/download/v0.8.5/vllm-0.8.5+cu121-cp38-cp38-linux_x86_64.whl
   ```

3. **Sandbox Connection Errors**
   ```bash
   # Test sandbox connectivity
   curl $SANDBOX_ENDPOINT/health
   
   # Use local sandbox as fallback
   export SANDBOX_ENDPOINT=local
   ```

4. **Insufficient Disk Space**
   - Clear pip cache: `pip cache purge`
   - Remove old checkpoints
   - Use external storage for models/data

### Performance Optimization

1. **Multi-GPU Training**
   ```bash
   # Set visible GPUs
   export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
   
   # Enable NCCL optimizations
   export NCCL_IB_DISABLE=0
   export NCCL_IB_GID_INDEX=3
   ```

2. **Memory Optimization**
   ```bash
   # Adjust vLLM memory utilization
   --rollout_gpu_memory_util 0.9  # Use 90% of GPU memory
   
   # Enable CPU offloading
   --actor_parameter_offload True
   --actor_optimizer_offload True
   ```

---

## References

1. SimpleTIR Paper & Blog: https://simpletir.notion.site/report
2. VERL Documentation: https://github.com/volcengine/verl
3. vLLM Documentation: https://docs.vllm.ai/
4. NVIDIA H100 Specifications: https://www.nvidia.com/en-us/data-center/h100/

---

*Last Updated: 2025-07-27*