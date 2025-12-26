# KeenKeenForAndroid - Synthesis Roadmap

## Executive Summary

Total codebase: **11,743 LOC** across 6 languages
- Java: 1,447 LOC (10 files)
- Kotlin: 584 LOC (7 files)
- C/Headers: 8,431 LOC (16 files)
- Python: 707 LOC (7 files)
- XML: ~200 LOC (13 files)
- Shell: ~100 LOC (4 files)

## Phase 1: Dead Code Removal (Est. -400 LOC)

### 1.1 Java/Kotlin Cleanup
- [ ] Remove `DSF.java` (61 LOC) - explicitly marked unused
- [ ] Remove `KeenController.java` (95 LOC) - replaced by GameViewModel
- [ ] Deprecate `KeenView.java` (610 LOC) - Compose replacement ready
- [ ] Consolidate `MenuActivity` into `KeenActivity` (future)

### 1.2 C/Native Cleanup
- [ ] Delete `unused/maxflow.c` - replaced by AVX2 version
- [ ] Delete `unused/maxflow_dinic_proto.c` - incomplete prototype
- [ ] Delete `unused/maxflow_simd.c` - abandoned variant
- [ ] Remove ~150 LOC of `#ifdef STANDALONE_*` debug blocks
- [ ] Remove commented UI code in `keen.c` (~200 LOC)

### 1.3 Python Pipeline Cleanup
- [ ] Remove `train_tiny_model.py` (40 LOC) - superseded
- [ ] Remove `train_real_model.py` (140 LOC) - superseded by massive
- [ ] Remove `param_sweep.py` (22 LOC) - incomplete skeleton

## Phase 2: Code Consolidation

### 2.1 Extract Shared Utilities
| Utility | Purpose | Saves |
|---------|---------|-------|
| `LatinSquareValidator` | Unify 3 validation impls | 25 LOC |
| `GameSettings` | Consolidate config storage | 60 LOC |
| `BaseLatinDataset` | Common dataset class | 50 LOC |

### 2.2 Modularize C Layer
| Module | Content | LOC |
|--------|---------|-----|
| `config.h` | Magic numbers (MAXBLK, etc) | 30 |
| `keen_layout.c` | Block structure generation | 140 |
| `keen_clues.c` | Clue calculation | 124 |

## Phase 3: UI/UX Synthesis

### 3.1 Fix Hardcoded Grid Sizes
- [ ] `GameScreen.kt:QuantumGrid()` - dynamic grid calculation
- [ ] `GameScreen.kt:NoteGrid()` - dynamic grid calculation
- [ ] `GameScreen.kt:InputPad()` - responsive button rows

### 3.2 UI Consolidation Path
```
Current: MenuActivity (Java) -> KeenActivity (Kotlin/Compose + Legacy)
Target:  SingleActivity (Kotlin/Compose only)
```

### 3.3 Accessibility
- [ ] Add contentDescription to all interactive elements
- [ ] Support high text scaling
- [ ] D-pad/keyboard navigation

## Phase 4: AI Pipeline Optimization

### 4.1 Training Improvements
- [ ] Add validation split (80/20)
- [ ] Enable mixed precision (torch.amp)
- [ ] Fix position embedding initialization
- [ ] Increase epochs from 2 to 10-15

### 4.2 Deployment Optimization
- [ ] Quantize ONNX model (4.8MB -> ~1.2MB)
- [ ] Remove unused `.pth` files from deployment

### 4.3 Dependency Cleanup
Remove from requirements.txt:
- torchvision (unused)
- tensorboard (unused)
- ruff, pytest (dev only)

## Phase 5: Build Infrastructure

### 5.1 Linting Configuration
| Language | Tool | Config File |
|----------|------|-------------|
| Java | Android Lint | `app/build.gradle` |
| Kotlin | ktlint | `.editorconfig` |
| C | clang-format | `.clang-format` |
| Python | ruff | `pyproject.toml` |
| XML | xmllint | (built-in) |

### 5.2 Pre-commit Hooks
```yaml
repos:
  - repo: local
    hooks:
      - id: ktlint
      - id: clang-format
      - id: ruff
```

### 5.3 CI Integration
- [ ] Add lint checks to GitHub Actions
- [ ] Add build verification step
- [ ] Add model export verification

## Phase 6: Quality Metrics

### Current State
| Metric | Value | Target |
|--------|-------|--------|
| Test Coverage | ~2% | 20% |
| Lint Warnings | 0 | 0 |
| Dead Code | 400 LOC | 0 |
| Duplicate Code | 3 instances | 0 |

### Testing Expansion
- [ ] Unit tests for KeenModel state transitions
- [ ] JNI parsing edge case tests
- [ ] AI fallback integration tests
- [ ] E2E game flow tests

## Execution Priority

### Immediate (This Session)
1. Remove dead files (Java, C, Python)
2. Set up formatting configs
3. Format all source files
4. Create pre-commit hooks
5. Commit and verify build

### Short-term (Next Session)
1. Extract shared utilities
2. Fix UI hardcoding
3. Optimize AI pipeline
4. Add validation splits

### Medium-term (Future)
1. Complete UI consolidation
2. Model quantization
3. Test coverage expansion
4. C code modularization
