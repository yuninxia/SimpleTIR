# SimpleTIR Experiment Workflow Guide

## Overview

This guide provides step-by-step instructions for reproducing SimpleTIR experiments, from initial setup to result analysis.

## Experiment Types

### 1. Baseline Evaluation (Verify Pre-trained Models)
**Duration**: 1-2 hours  
**Purpose**: Test existing SimpleTIR models on benchmarks

```bash
# Evaluate on AIME dataset
CONFIG_NAME=simpletir_trainer \
bash train.sh \
  --val_only True \
  --model_name SimpleTIR-Qwen2.5-7B \
  --valid_dataset "deepscaler/aime" \
  --max_turns 10 \
  --n_val 32 \
  --val_sample_size 500 \
  --output_acc_to_file True
```

Expected outputs:
- Accuracy metrics in `logs/[RUN_NAME]_accuracy.json`
- Generation samples in `logs/[RUN_NAME]_generations.jsonl`

### 2. Single-Turn Training (Baseline Comparison)
**Duration**: 12-24 hours on 8xH100  
**Purpose**: Train models without tool integration

```bash
CONFIG_NAME=single_turn_math \
bash train.sh \
  --model_name Qwen2.5-7B \
  --train_dataset "simplelr_math_35/train deepscaler/train" \
  --tool_use False \
  --mask_void_turns False \
  --train_batch_size 512 \
  --total_epochs 20
```

### 3. Multi-Turn SimpleTIR Training (Main Experiment)
**Duration**: 3-7 days on 8xH100  
**Purpose**: Full SimpleTIR training with tool integration

```bash
CONFIG_NAME=simpletir_trainer \
bash train.sh \
  --model_name Qwen2.5-7B \
  --max_turns 5 \
  --train_batch_size 512 \
  --train_dataset "simplelr_math_35/train deepscaler/train" \
  --total_epochs 100 \
  --save_freq 20 \
  --test_freq 10
```

## Key Experimental Parameters

### Training Parameters
- `max_turns`: Maximum reasoning turns (default: 5)
- `train_batch_size`: Batch size per GPU (adjust based on memory)
- `mask_void_turns`: Enable SimpleTIR's void turn filtering (critical!)
- `oversample`: Oversample multiplier for rejection sampling (default: 3)

### Model Parameters
- `model_name`: Base model to use (Qwen2.5-7B, Qwen2.5-32B, etc.)
- `max_prompt_length`: Maximum input length (default: 16000)
- `max_response_length`: Maximum generation length (default: 8000)

### Optimization Parameters
- `grad_clip`: Gradient clipping value (default: 1.0)
- `remove_clip`: Whether to remove overlong responses
- `start_clip_step`: When to start clipping (default: 20)

## Monitoring Training Progress

### 1. Weights & Biases (Recommended)
```bash
# Training will automatically log to W&B if configured
wandb login  # One-time setup

# Monitor metrics:
# - reward/mean: Average reward across episodes
# - kl/mean: KL divergence from reference model
# - loss/actor: Actor loss
# - val/accuracy: Validation accuracy
```

### 2. Local Logs
```bash
# Monitor training logs
tail -f logs/[RUN_NAME].log

# Check validation results
cat logs/[RUN_NAME]_val_epoch_*.json
```

### 3. Checkpoint Management
```bash
# List checkpoints
ls -la checkpoints/[RUN_NAME]/

# Resume from checkpoint
RESUME=True bash train.sh [same parameters]

# Convert checkpoint to HuggingFace format
bash scripts/model_merger.sh \
  --ckpt_path checkpoints/[RUN_NAME]/epoch_100 \
  --output_path models/SimpleTIR-Qwen2.5-7B-final
```

## Evaluation Metrics

### Primary Metrics
1. **Pass@k Accuracy**: Percentage of problems solved within k attempts
2. **Average Turns**: Mean number of reasoning turns used
3. **Tool Usage Rate**: Frequency of code generation/execution

### Benchmark Datasets
- **AIME**: American Invitational Mathematics Examination problems
- **AIME25**: Updated AIME problems from 2025
- **SimpleLR Math**: Custom mathematical reasoning dataset

### Analyzing Results

```python
# Load and analyze results
import json
import pandas as pd

# Load accuracy results
with open('logs/run_name_accuracy.json', 'r') as f:
    results = json.load(f)

# Key metrics to report
print(f"Pass@1 Accuracy: {results['pass@1']:.2%}")
print(f"Pass@5 Accuracy: {results['pass@5']:.2%}")
print(f"Average Turns: {results['avg_turns']:.2f}")
print(f"Tool Usage Rate: {results['tool_usage_rate']:.2%}")
```

## Reproducing Paper Results

### Table 1: Main Results
To reproduce the main results table:

```bash
# 1. Train SimpleTIR models
for model in "Qwen2.5-7B" "Qwen2.5-32B"; do
  CONFIG_NAME=simpletir_trainer bash train.sh \
    --model_name $model \
    --train_dataset "simplelr_math_35/train deepscaler/train" \
    --total_epochs 100
done

# 2. Evaluate on all benchmarks
for dataset in "deepscaler/aime" "deepscaler/aime25" "simplelr_math_35/test"; do
  bash train.sh --val_only True \
    --valid_dataset $dataset \
    --output_acc_to_file True
done
```

### Figure 3: Training Stability
To reproduce the training curves:

```bash
# Run with and without void turn filtering
# With filtering (SimpleTIR)
bash train.sh --mask_void_turns True --experiment_name simpletir_stable

# Without filtering (baseline)
bash train.sh --mask_void_turns False --experiment_name baseline_unstable

# Plot results using wandb or tensorboard
```

## Experiment Checklist

- [ ] Environment setup complete (CUDA, dependencies installed)
- [ ] Sandbox configured and tested
- [ ] Base models downloaded
- [ ] Datasets prepared in parquet format
- [ ] Sufficient disk space allocated (>300GB for 7B model)
- [ ] W&B configured for logging
- [ ] Baseline evaluation run successfully
- [ ] Training launched with appropriate parameters
- [ ] Checkpoints being saved regularly
- [ ] Results being logged and monitored

## Tips for Successful Experiments

1. **Start Small**: Begin with shorter training runs (10-20 epochs) to verify setup
2. **Monitor Memory**: Use `nvidia-smi` to ensure GPU memory is efficiently utilized
3. **Save Checkpoints**: Set `save_freq` to save checkpoints regularly
4. **Track Experiments**: Use descriptive experiment names and log all parameters
5. **Backup Results**: Regularly backup logs and checkpoints to external storage

## Common Pitfalls

1. **Forgetting Sandbox Setup**: SimpleTIR requires sandbox for code execution
2. **Insufficient Disk Space**: Monitor disk usage, especially during training
3. **Wrong Dataset Format**: Ensure datasets are in parquet format
4. **Memory Issues**: Adjust batch size and sequence lengths based on GPU memory
5. **Version Mismatches**: Use exact versions specified (especially vLLM 0.8.5)

---

*For additional support, refer to the main documentation or create an issue in the repository.*