# Master Task Tracker: Keen Kenning Refinement

## I. Quality Assurance & Build Discipline
- [x] **Build Baseline**: `./gradlew assembleClassikDebug assembleKenningDebug`.
- [x] **Unit Tests**: `./gradlew testClassikDebugUnitTest testKenningDebugUnitTest`.
- [x] **Instrumented Tests**: `./gradlew connectedClassikDebugAndroidTest connectedKenningDebugAndroidTest`.
- [ ] **Lint Zeroing**: `./gradlew lintClassikDebug lintKenningDebug` (warnings are errors).
- [ ] **Build Profiling**: capture `--profile`, configuration cache, build cache behavior.
- [ ] **Dependency Audit**: verify versions and suppress unresolvable upgrades (SDK 35 constraint).

## II. Structural Harmonization
- [x] **Documentation**: Move root markdown files (`roadmap.md`, `tasks.md`) to `docs/planning/`.
- [x] **Scripts**: Audit `scripts/` for executable permissions and headers.
- [x] **JNI**: Organized unused reference implementations into `app/src/main/jni/unused/`.
- [x] **Module Split**: Add `:core` (interfaces + shared models).
- [x] **Kenning Module**: Add `:kenning` (ML + narrative + story) and wire `kenningImplementation project(':kenning')`.
- [x] **Flavor Services**: Define interfaces in `app/src/main`, implementations in `app/src/classik` + `app/src/kenning`, injected via factory.
- [x] **Classik Baseline**: Move Classik implementations into `app/src/main`; keep Kenning overrides in `app/src/kenning`.

## III. Codebase Deep Dive
- [x] **Java/Kotlin**: Updated `KeenModelBuilder` Javadoc and removed dead code.
- [x] **C/C++**: Verified JNI Bridge `keen-android-jni.c` handles AI grid input correctly.
- [x] **AI Integration**:
    - [x] `NeuralKeenGenerator.java` implemented with ONNX Runtime.
    - [x] `latin_solver.onnx` verified in assets (supports 3x3-16x16 grids).
    - [x] Size check implemented for full range (3-16).
- [x] **Story Mode Split**: Story UI in `app/src/main`, Kenning assets in `:kenning`, Classik stubs in `app/src/main`.
- [ ] **Asset Consistency**: verify ONNX asset names used by generator match packaged assets.

## IV. Synthesis & Expansion
- [x] **Feature**: Neural/Classic toggle is present in UI (`activity_menu.xml`) and wired in `KeenModelBuilder`.
- [x] **Feature**: AI Solver connected via `NeuralKeenGenerator`.

## V. Performance & Instrumentation (Pending)
- [x] **Sanitizers**: document and wire `keenSanitizers`, `keenCoverage`, `keenFuncTrace` in Gradle properties (ASan/TSan/UBSan where supported).
- [x] **Perf/Flamegraph**: add scripts for perf + flamegraph on native hot paths.
- [x] **Coverage**: add gcovr workflow for native code (host or device build).
- [ ] **Valgrind/Heaptrack**: add host-native build instructions to run memcheck (valgrind added, heaptrack pending).
- [x] **Infer**: add Infer static analysis workflow (see `~/Documents/Code-Analysis-Tooling`).
- [ ] **Perfetto/Tracing**: capture CPU/GPU traces for puzzle generation and rendering.

## VI. Emulator & UI Automation (Pending)
- [x] **Headless Emulator**: boot script + readiness checks (system_server + package service).
- [x] **Smoke Tests**: Espresso UI flow: launch -> new puzzle -> input -> solve/validate.
- [x] **Crash Repro**: automated test cases for navigation + story entry (Kenning).

## VII. Benchmarks & Telemetry (Pending)
- [x] **Benchmarks**: parse/layout/solver timing (AndroidX Benchmark or JMH).
- [x] **Memory/GPU Metrics**: memory stats and gfxinfo baseline.
- [x] **Logging Hooks**: structured logs for perf and crash triage.

## VIII. Advanced Model Training (Pending)
- [ ] **Massive Dataset**: Generate 500k grids across sizes 2-20.
- [ ] **Parameter Sweep**: Run `scripts/ai/param_sweep.py` to optimize transformer depth/heads.
- [ ] **Multi-GPU Training**: Train production `MassiveBrain.pth` model.

## IX. Quality & Validation (Pending)
- [ ] **JNI Security Audit**: Implement boundary checks for grid arrays.
- [ ] **Unit Testing**:
    - [ ] C: Test maxflow correctness vs brute force.
    - [ ] Python: Verify ONNX model output matches PyTorch prototype.
    - [ ] Java: Robolectric tests for `KeenModel` state transitions.
- [ ] **Game Engine Refactor**: Extract `KeenController` into standalone logic unit.
