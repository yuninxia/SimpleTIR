# SimpleTIR Artifact Evaluation

This directory contains comprehensive documentation and resources for evaluating SimpleTIR: End-to-End Reinforcement Learning for Multi-Turn Tool-Integrated Reasoning.

## Documentation Structure

- **[OVERVIEW.md](docs/OVERVIEW.md)** - Project architecture and common commands reference
- **[QUICK_START.md](docs/QUICK_START.md)** - 30-minute setup guide for getting started on 8xH100 nodes
- **[SETUP_AND_REQUIREMENTS.md](docs/SETUP_AND_REQUIREMENTS.md)** - Detailed system requirements, installation options, and troubleshooting
- **[EXPERIMENT_WORKFLOW.md](docs/EXPERIMENT_WORKFLOW.md)** - Step-by-step guide for reproducing paper experiments

## Getting Started

### For Quick Testing (30 minutes)
Follow [QUICK_START.md](docs/QUICK_START.md) to:
1. Set up the environment
2. Install dependencies
3. Run a test evaluation or training

### For Full Reproduction (3-7 days)
Follow [EXPERIMENT_WORKFLOW.md](docs/EXPERIMENT_WORKFLOW.md) to:
1. Train SimpleTIR models from scratch
2. Evaluate on benchmark datasets
3. Reproduce paper results

## System Requirements Summary

### Minimum Requirements
- **GPU**: 1x NVIDIA GPU with 24GB+ VRAM
- **CPU**: 16+ cores
- **RAM**: 64GB
- **Storage**: 100GB SSD

### Recommended (8xH100 Node)
- **GPU**: 8x NVIDIA H100 80GB
- **CPU**: 64+ cores  
- **RAM**: 256GB+
- **Storage**: 1TB NVMe SSD

## Key Experiments

1. **Baseline Evaluation**: Test pre-trained SimpleTIR models (~2 hours)
2. **Single-Turn Training**: Train without tool integration (~24 hours)
3. **Multi-Turn SimpleTIR**: Full training with tool use (3-7 days)

## Setup Options

We provide 4 setup options ranked by priority:

1. **Priority 1: Quick Evaluation** - Test with pre-trained models
2. **Priority 2: Docker Container** - Reproducible environment
3. **Priority 3: Native Installation** - For development
4. **Priority 4: Full Training** - Complete reproduction

## Expected Results

SimpleTIR achieves significant improvements over baselines:
- Stable multi-turn training through void turn filtering
- Diverse reasoning patterns (inductive, self-correction, cross-validation)
- State-of-the-art performance on mathematical reasoning benchmarks

## Troubleshooting

Common issues and solutions are covered in [SETUP_AND_REQUIREMENTS.md](docs/SETUP_AND_REQUIREMENTS.md#troubleshooting).

## Citation

```bibtex
@misc{xue2025simpletir,
  title={SimpleTIR: End-to-End Reinforcement Learning for Multi-Turn Tool-Integrated Reasoning},
  author={Zhenghai Xue and Longtao Zheng and Qian Liu and Yingru Li and Zejun Ma and Bo An},
  year={2025},
  howpublished={\url{https://simpletir.notion.site/report}},
}
```

## Support

- Repository: https://github.com/ZhenghaiXue/SimpleTIR
- Blog Post: https://simpletir.notion.site/report
- For dataset access and specific questions, please contact the authors

---

*Last Updated: 2025-07-27*