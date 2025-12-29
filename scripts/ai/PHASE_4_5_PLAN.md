# Phase 4-5 Implementation Plan: Precision & Curriculum v2

## Overview

This document outlines the hypergranular implementation plan for Phases 4 and 5
of the KeenKenning ML training pipeline overhaul.

**Dependencies:**
- Phase 4 is independent (can start immediately)
- Phase 5 depends on Phase 2 (multi-mode) being complete ✓

---

## PHASE 4: PRECISION & MEMORY OPTIMIZATION

### Current State (Baseline)
```python
# Already implemented:
torch.set_float32_matmul_precision('high')  # TF32
torch.backends.cudnn.benchmark = True
torch.amp.autocast('cuda')                  # FP16 AMP
torch.amp.GradScaler('cuda')                # Loss scaling
torch.compile(model)                        # Inductor
```

### 4.1 Mixed Precision Enhancement (BF16)

| Task | Description | File | Est. LOC |
|------|-------------|------|----------|
| P4.1.1 | Audit AMP implementation | train_autoregressive.py | 0 (read) |
| P4.1.2 | Add `--dtype {fp32,fp16,bf16}` arg | train_autoregressive.py | ~15 |
| P4.1.3 | Implement BF16 autocast | train_autoregressive.py | ~20 |
| P4.1.4 | Serialize dtype in checkpoint | checkpoint_manager.py | ~10 |

**Rationale:** BF16 has same exponent range as FP32 (no loss scaling needed)
but 8-bit mantissa. On Ampere+ GPUs, BF16 is often faster and more stable than FP16.

**Implementation:**
```python
# P4.1.2: New argument
parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp16")

# P4.1.3: BF16 autocast
dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
with torch.amp.autocast('cuda', dtype=dtype_map[args.dtype]):
    ...
```

### 4.2 Gradient Checkpointing

| Task | Description | File | Est. LOC |
|------|-------------|------|----------|
| P4.2.1 | Wrap transformer layers with checkpoint | train_autoregressive.py | ~25 |
| P4.2.2 | Add `--grad-checkpoint` flag | train_autoregressive.py | ~5 |
| P4.2.3 | Benchmark memory/speed tradeoff | (manual test) | ~0 |

**Rationale:** Trades compute for memory. Re-computes activations during backward
instead of storing them. ~30-50% memory reduction, ~20% slower.

**Implementation:**
```python
from torch.utils.checkpoint import checkpoint

class TransformerBlockWithCheckpoint(nn.Module):
    def forward(self, x):
        if self.training and self.use_checkpoint:
            return checkpoint(self._forward_impl, x, use_reentrant=False)
        return self._forward_impl(x)
```

### 4.3 Memory Monitoring

| Task | Description | File | Est. LOC |
|------|-------------|------|----------|
| P4.3.1 | Add memory monitoring hooks | train_autoregressive.py | ~20 |
| P4.3.2 | Log GPU memory per epoch | train_autoregressive.py | ~10 |
| P4.3.3 | Add `--profile` flag for torch.profiler | train_autoregressive.py | ~30 |

**Implementation:**
```python
def get_gpu_memory_mb():
    return torch.cuda.max_memory_allocated() / 1024 / 1024

# In training loop:
print(f"  Peak GPU Memory: {get_gpu_memory_mb():.0f} MB")
```

### 4.4 8-bit Optimizer Refinement

| Task | Description | File | Est. LOC |
|------|-------------|------|----------|
| P4.4.1 | Improve 8-bit integration | train_autoregressive.py | ~15 |
| P4.4.2 | Graceful fallback message | train_autoregressive.py | ~5 |
| P4.4.3 | Test training stability | (manual test) | ~0 |

**Current State:** 8-bit optimizer import exists but usage could be cleaner.

---

## PHASE 5: CURRICULUM v2 (MULTI-DIMENSIONAL)

### Current State (Baseline)
```python
# Size-only curriculum:
curriculum = [
    list(range(3, 6)),   # Stage 1: 3-5
    list(range(3, 9)),   # Stage 2: 3-8
    list(range(3, 13)),  # Stage 3: 3-12
    list(range(3, 17))   # Stage 4: 3-16
]
```

### 5.1 Mode Curriculum

| Task | Description | File | Est. LOC |
|------|-------------|------|----------|
| P5.1.1 | Design mode stages | (design doc) | ~0 |
| P5.1.2 | Create ModeCurriculumDataset | train_autoregressive.py | ~40 |
| P5.1.3 | Add `--mode-curriculum` flag | train_autoregressive.py | ~10 |
| P5.1.4 | Integrate with size curriculum | train_autoregressive.py | ~20 |

**Mode Curriculum Design:**
```
Stage 1: STANDARD only (50% of epochs)
Stage 2: STANDARD + ZERO_INCLUSIVE (30% of epochs)
Stage 3: All modes including NEGATIVE (20% of epochs)
```

**Rationale:** STANDARD is most common in real usage. NEGATIVE has the most
complex token space (symmetric around 0). Progressive introduction prevents
mode confusion early in training.

### 5.2 Fill-Ratio Curriculum

| Task | Description | File | Est. LOC |
|------|-------------|------|----------|
| P5.2.1 | Design fill stages | (design doc) | ~0 |
| P5.2.2 | Add progressive fill ratio | train_autoregressive.py | ~15 |
| P5.2.3 | Add `--fill-curriculum` flag | train_autoregressive.py | ~5 |

**Fill-Ratio Curriculum Design:**
```
Stage 1: 50-70% revealed (easy, many hints)
Stage 2: 30-50% revealed (medium)
Stage 3: 10-30% revealed (hard, few hints)
Stage 4: 0-10% revealed (expert, nearly empty)
```

**Rationale:** Matches inference scenario where user provides partial solution.
Starting with more hints helps model learn Latin square structure before
learning to generate from scratch.

### 5.3 Unified Curriculum Scheduler

| Task | Description | File | Est. LOC |
|------|-------------|------|----------|
| P5.3.1 | Create CurriculumScheduler class | curriculum.py (new) | ~80 |
| P5.3.2 | Support 2D curriculum (size × mode) | curriculum.py | ~30 |
| P5.3.3 | Support 3D curriculum (size × mode × fill) | curriculum.py | ~40 |

**Architecture:**
```python
@dataclass
class CurriculumState:
    size_stage: int
    mode_stage: int
    fill_stage: int
    epoch: int

class CurriculumScheduler:
    def __init__(self, size_stages, mode_stages, fill_stages):
        ...
    
    def step(self, epoch: int, metrics: dict) -> CurriculumState:
        # Advance stages based on epoch or metrics
        ...
    
    def get_dataset_params(self) -> dict:
        # Return current allowed_sizes, allowed_modes, fill_range
        ...
```

### 5.4 Curriculum Metrics & Transitions

| Task | Description | File | Est. LOC |
|------|-------------|------|----------|
| P5.4.1 | Per-stage validation metrics | train_autoregressive.py | ~25 |
| P5.4.2 | Stage transition criteria | curriculum.py | ~30 |
| P5.4.3 | Log curriculum progression | train_autoregressive.py | ~15 |

**Transition Criteria Options:**
1. **Epoch-based:** Fixed epochs per stage (current approach)
2. **Loss-based:** Advance when loss < threshold
3. **Valid-rate-based:** Advance when valid_grid_rate > threshold
4. **Combined:** Both loss AND valid_rate must meet criteria

### 5.5 Testing & Documentation

| Task | Description | File | Est. LOC |
|------|-------------|------|----------|
| P5.5.1 | Write curriculum unit tests | test_curriculum.py (new) | ~100 |
| P5.5.2 | Run full curriculum training | (manual test) | ~0 |
| P5.5.3 | Document CLI options in README | README.md | ~50 |

---

## Implementation Order (Critical Path)

```
Phase 4 (can run in parallel):
  4.1.1 → 4.1.2 → 4.1.3 → 4.1.4  (BF16 chain)
  4.2.1 → 4.2.2 → 4.2.3          (Checkpointing chain)
  4.3.1 → 4.3.2 → 4.3.3          (Monitoring chain)
  4.4.1 → 4.4.2 → 4.4.3          (8-bit chain)

Phase 5 (sequential, depends on 4.x complete):
  5.1.1 → 5.1.2 → 5.1.3 → 5.1.4  (Mode curriculum)
      ↓
  5.2.1 → 5.2.2 → 5.2.3          (Fill curriculum)
      ↓
  5.3.1 → 5.3.2 → 5.3.3          (Unified scheduler)
      ↓
  5.4.1 → 5.4.2 → 5.4.3          (Metrics & transitions)
      ↓
  5.5.1 → 5.5.2 → 5.5.3          (Testing & docs)
```

---

## Estimated Effort

| Phase | Tasks | Est. LOC | Est. Time |
|-------|-------|----------|-----------|
| 4.1 | 4 | ~45 | 1 hour |
| 4.2 | 3 | ~30 | 45 min |
| 4.3 | 3 | ~60 | 1 hour |
| 4.4 | 3 | ~20 | 30 min |
| **4 Total** | **13** | **~155** | **~3.25 hours** |
| 5.1 | 4 | ~70 | 1.5 hours |
| 5.2 | 3 | ~20 | 30 min |
| 5.3 | 3 | ~150 | 2 hours |
| 5.4 | 3 | ~70 | 1 hour |
| 5.5 | 3 | ~150 | 1.5 hours |
| **5 Total** | **16** | **~460** | **~6.5 hours** |
| **Grand Total** | **29** | **~615** | **~10 hours** |

---

## Success Criteria

### Phase 4
- [ ] BF16 training matches FP16 accuracy
- [ ] Gradient checkpointing reduces memory by >25%
- [ ] Memory logging shows peak/current per epoch
- [ ] 8-bit optimizer trains without NaN/divergence

### Phase 5
- [ ] Mode curriculum improves NEGATIVE mode accuracy
- [ ] Fill curriculum improves empty-grid generation
- [ ] Unified scheduler coordinates all dimensions
- [ ] Metrics-based transitions work correctly
- [ ] Full curriculum training reaches target loss

---

*Generated: 2025-12-28*
*Author: Claude Code (Opus 4.5)*
