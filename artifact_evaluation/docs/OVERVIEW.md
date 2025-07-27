# SimpleTIR Project Overview

## Overview

SimpleTIR is an end-to-end reinforcement learning framework for training LLMs to perform multi-turn Tool-Integrated Reasoning (TIR). The framework enables models to iteratively generate code, execute it in a sandbox, and reason about the execution results to solve complex mathematical problems and conduct sophisticated data analysis.

## Core Architecture

The project is built on top of the VERL (Volcano Engine Reinforcement Learning) framework and consists of:

- **Recipe System**: Main training logic in `recipe/simpletir/` containing the SimpleTIR algorithm implementation
- **VERL Core**: Base RL infrastructure in `verl/` including trainers, workers, models, and utilities
- **Sandbox System**: Code execution environment in `sandbox/` with internal and local sandbox implementations
- **Worker Types**: Specialized workers for actors, critics, rollout generation, and reward computation
- **Model Support**: Implementations for LLaMA and Qwen2 models with Megatron and FSDP parallelism

## Common Commands

### Training
```bash
# 7B model training on 8xH100 node
MODEL_PATH=... DATA_PATH=... CHECKPOINT_PATH=... LOG_PATH=... \
NNODES=1 GPUS_PER_NODE=8 RESUME=False CONFIG_NAME=simpletir_trainer \
bash train.sh \
  --max_response_length 8000 \
  --max_prompt_length 16000 \
  --model_name Qwen2.5-7B \
  --max_turns 5 \
  --train_batch_size 512
```

### Single-turn Training
```bash
CONFIG_NAME=single_turn_math bash train.sh \
  --tool_use False \
  --mask_void_turns False
```

### Evaluation
```bash
CONFIG_NAME=simpletir_trainer bash train.sh \
  --val_only True \
  --valid_dataset "deepscaler/aime" \
  --output_acc_to_file True
```

### Testing
```bash
# Run specific test
python -m pytest tests/e2e/test_specific.py

# Run end-to-end tests
bash tests/e2e/run_qwen_gsm8k_model_rm.sh
```

### Code Formatting
```bash
bash scripts/format.sh  # Uses yapf with Google style
```

### Model Conversion
```bash
bash scripts/model_merger.sh  # Convert checkpoint to HuggingFace format
```

## Key Configurations

- **Training Config**: `recipe/simpletir/config/simpletir_trainer.yaml` - main PPO training configuration
- **Single-turn Config**: `recipe/simpletir/config/single_turn_math.yaml` - single-turn training without tool use
- **Hydra Config System**: Uses Hydra for configuration management with searchpath to `verl/trainer/config`

## Environment Variables

- `SANDBOX_ENDPOINT`: Endpoint for code execution sandbox (required for tool-integrated reasoning)
- `MODEL_PATH`: Parent directory containing model checkpoints
- `DATA_PATH`: Directory containing training/validation datasets in parquet format
- `CHECKPOINT_PATH`: Directory for saving training checkpoints
- `LOG_PATH`: Directory for training logs

## Development Notes

- The codebase uses vLLM 0.8.5 for inference
- Multi-node training is managed through Ray
- Training stability is achieved by filtering "void" turns (turns without code blocks or final answers)
- The framework supports both Megatron and FSDP parallelism strategies
- Sandbox execution is critical for tool-integrated reasoning - ensure SANDBOX_ENDPOINT is properly configured