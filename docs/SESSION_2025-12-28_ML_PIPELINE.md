# ML Training Pipeline Overhaul - Session Breadcrumb
**Date:** 2025-12-28
**Session Focus:** Phases 4-6 of ML Training Pipeline Enhancement

---

## Executive Summary

Completed comprehensive overhaul of the Latin square neural solver training pipeline, adding precision control, multi-dimensional curriculum learning, and Zen3 CPU-specific optimizations.

---

## Phases Completed

### Phase 0-3 (Previous Session)
- P0: Checkpoint Resilience (crash recovery, rolling checkpoints)
- P1: Token Vocabulary (35-token unified vocab)
- P2: Multi-Mode Training (STANDARD, ZERO_INCLUSIVE, NEGATIVE)
- P3: Isotopism Augmentation (n!^3 grid variants)

### Phase 4: Precision & Memory Optimization
| Task | Status | Implementation |
|------|--------|----------------|
| P4.1: BF16/FP16/FP32 dtype selection | Complete | `--dtype {fp32,fp16,bf16}` CLI flag |
| P4.2: Gradient checkpointing | Complete | `--grad-checkpoint` (~30% memory savings) |
| P4.3: Memory monitoring | Complete | Per-epoch GPU peak memory logging |
| P4.4: 8-bit optimizer fallback | Complete | Graceful warning when bitsandbytes unavailable |

**Key Code:**
```python
# train_autoregressive.py
DTYPE_MAP = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
train_dtype = DTYPE_MAP[args.dtype]
scaler = torch.amp.GradScaler('cuda') if args.dtype == "fp16" else None
```

### Phase 5: Multi-Dimensional Curriculum v2
| Task | Status | Implementation |
|------|--------|----------------|
| P5.1: Mode curriculum | Complete | STANDARD -> ZERO -> NEGATIVE progression |
| P5.2: Fill-ratio curriculum | Complete | 70% -> 50% -> 30% -> 10% revealed |
| P5.3: CurriculumScheduler class | Complete | New `curriculum.py` module |
| P5.4: Checkpoint serialization | Complete | `curriculum_scheduler_state` in TrainingState |
| P5.5: Testing & documentation | Complete | README.md updated |

**New File:** `scripts/ai/curriculum.py`
```python
class CurriculumScheduler:
    """Multi-dimensional curriculum: size + mode + fill"""
    def step(epoch) -> CurriculumState
    def get_allowed_sizes() -> List[int]
    def get_allowed_modes() -> List[GameMode]
    def get_fill_range() -> Tuple[float, float]
```

### Phase 6: Zen3 CPU Optimization
| Task | Status | Implementation |
|------|--------|----------------|
| P6.1: Makefile audit | Complete | Found deprecated `train_massive_model.py` |
| P6.2: CachyOS packages | Complete | `python-pytorch-opt-cuda` already optimal |
| P6.3.1: Zen3 research | Complete | 96MB L3, AVX2, no AVX-512 |
| P6.3.2: GCC flags | Complete | `-march=znver3 -mtune=znver3` |
| P6.3.3: Hardware detection | Complete | `cpu_family`, `has_vcache` fields |
| P6.3.4: Inductor config | Complete | C++ wrapper, oneDNN enabled |
| P6.3.5: V-Cache optimizations | Complete | +32 batch boost, prefetch=6 |
| P6.4: Documentation | Complete | This document |
| P6.5: Makefile targets | Complete | `train`, `train-full`, `hw-report` |

---

## Key Insights

### Zen3 + 3D V-Cache Architecture
```
CPU:  AMD Ryzen 5 5600X3D
      Vendor: AMD, Family: zen3_vcache
      6 cores / 12 threads
      L3 Cache: 96 MB (3D V-Cache - 3x typical!)
      AVX2: True, AVX-512: False
      SHA-NI: True, VAES: True
```

**V-Cache Optimizations Applied:**
- Batch size: 160 (+32 boost from 128 baseline)
- DataLoader workers: 8 (up from 6)
- Prefetch factor: 6 (up from 4)
- Inductor: C++ wrapper + oneDNN enabled
- No AVX-512 (stick to 256-bit ops)

### PyTorch Environment
```
PyTorch: 2.9.1
CUDA: 13.0
cuDNN: 9.1.7
Package: python-pytorch-opt-cuda (Arch/CachyOS)
         Already includes MKL, oneDNN, MAGMA, NCCL
```

### Curriculum Learning Schedule (60 epochs)
| Epochs | Size Stage | Mode Stage | Fill Stage |
|--------|------------|------------|------------|
| 0-14   | 3-5        | STANDARD   | 50-70%     |
| 15-29  | 3-8        | STANDARD   | 30-50%     |
| 30-44  | 3-12       | +ZERO      | 10-30%     |
| 45-59  | 3-16       | +NEGATIVE  | 0-10%      |

---

## Files Modified

| File | Changes |
|------|---------|
| `scripts/ai/train_autoregressive.py` | Dtype, checkpoint, curriculum integration |
| `scripts/ai/checkpoint_manager.py` | `curriculum_scheduler_state` field |
| `scripts/ai/curriculum.py` | **NEW** - Multi-dim curriculum scheduler |
| `scripts/ai/hardware_config.py` | Zen3/V-Cache detection, optimizations |
| `scripts/ai/README.md` | Updated CLI args and usage |
| `Makefile` | New training targets |

---

## New Makefile Targets

```bash
make train          # Full curriculum training (production)
make train-full     # All optimizations (curriculum + augmentation)
make train-quick    # Quick test (5 epochs)
make train-resume   # Resume from checkpoint
make hw-report      # Print hardware detection
make generate-data-cuda  # GPU data generation (3-16)
```

---

## What Comes Next

### Immediate (Next Session)
1. **Run full training**: `make train` with all curriculum dimensions
2. **Validate model quality**: Check valid grid rates at each size
3. **ONNX export verification**: Ensure model works in Android app

### Short-term Enhancements
1. **Metrics dashboard**: TensorBoard integration for curriculum transitions
2. **Adaptive curriculum**: Advance stages based on loss/accuracy thresholds
3. **Model distillation**: Smaller model for mobile inference

### Research Directions
1. **Constraint satisfaction head**: Direct Latin constraint enforcement
2. **Beam search inference**: Multiple candidate solutions
3. **Reinforcement learning**: Self-play puzzle generation

---

## Quick Start (Next Session)

```bash
cd /home/eirikr/Github/KeenKenning

# Check hardware optimizations
make hw-report

# Start full training
make train TRAIN_EPOCHS=60 TARGET_LOSS=0.09

# Or resume if interrupted
make train-resume
```

---

## Session Statistics

- **Duration**: ~2 hours
- **Files created**: 1 (curriculum.py)
- **Files modified**: 5
- **Lines added**: ~400
- **Tests passed**: All syntax checks, curriculum unit test, hardware detection

---

*Generated by Claude Code - 2025-12-28*
