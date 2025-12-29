# Master Task Tracker: Keen Kenning Refinement

## I. Quality Assurance & Build Discipline
- [x] **Build Baseline**: Execute `./gradlew assembleDebug` and capture all errors/warnings.
- [x] **Warning Zeroing**: Resolve all compiler warnings (Java/Kotlin/C/C++).
- [x] **Lint Zeroing**: Resolve all Android Lint warnings.
- [x] **Dependency Audit**: Verify all versions and suppress unresolvable upgrades (due to SDK 35 constraint).

## II. Structural Harmonization
- [x] **Documentation**: Move root markdown files (`roadmap.md`, `tasks.md`) to `docs/planning/`.
- [x] **Scripts**: Audit `scripts/` for executable permissions and headers.
- [x] **JNI**: Organized unused reference implementations into `app/src/main/jni/unused/`.

## III. Codebase Deep Dive
- [x] **Java/Kotlin**: Updated `KeenModelBuilder` Javadoc and removed dead code.
- [x] **C/C++**: Verified JNI Bridge `keen-android-jni.c` handles AI grid input correctly.
- [x] **AI Integration**:
    - [x] `NeuralKeenGenerator.java` implemented with ONNX Runtime.
    - [x] `latin_solver.onnx` verified in assets (supports 3x3-16x16 grids).
    - [x] Size check implemented for full range (3-16).

## IV. Synthesis & Expansion
- [x] **Feature**: Neural/Classic toggle is present in UI (`activity_menu.xml`) and wired in `KeenModelBuilder`.
- [x] **Feature**: AI Solver connected via `NeuralKeenGenerator`.

## V. Advanced Model Training (Pending)
- [ ] **Massive Dataset**: Generate 500k grids across sizes 2-20.
- [ ] **Parameter Sweep**: Run `scripts/ai/param_sweep.py` to optimize transformer depth/heads.
- [ ] **Multi-GPU Training**: Train production `MassiveBrain.pth` model.

## VI. Quality & Validation (Pending)
- [ ] **JNI Security Audit**: Implement boundary checks for grid arrays.
- [ ] **Unit Testing**:
    - [ ] C: Test maxflow correctness vs brute force.
    - [ ] Python: Verify ONNX model output matches PyTorch prototype.
    - [ ] Java: Robolectric tests for `KeenModel` state transitions.
- [ ] **Game Engine Refactor**: Extract `KeenController` into standalone logic unit.
