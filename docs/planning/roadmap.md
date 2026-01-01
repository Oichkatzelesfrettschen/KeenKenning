# Keen Kenning - Development Roadmap

Target ISAs: `x86_64`, `arm64-v8a`, `armeabi-v7a` only. Warnings are errors.

## Phase 0: Flavor Architecture Baseline
- [x] **Classik Baseline**: Shared implementations live in `app/src/main`.
- [x] **Kenning Overrides**: Flavor-specific services in `app/src/kenning`.
- [x] **:core Module**: Shared interfaces + models.
- [x] **:kenning Module**: ML + narrative + story assets; wired via `kenningImplementation`.
- [x] **Kenning Assets**: Stored in `kenning/src/main/assets`; app/src/kenning reserved for overrides.
- [x] **Factory**: `FlavorServices` provides flavor implementations.
- [ ] **Asset Consistency**: verify ONNX asset names used by generators match packaged assets.

## Phase 1: Build & Test Stabilization
- [x] **Build Check**: `./gradlew assembleClassikDebug assembleKenningDebug`.
- [x] **Unit Tests**: `./gradlew testClassikDebugUnitTest testKenningDebugUnitTest`.
- [x] **Instrumented Tests**: headless emulator + `connected*AndroidTest` (ensure package service ready).
- [x] **Emulator Keepalive**: keepalive + logcat capture with timestamp window.
- [x] **Runner Split**: UI tests default to `AndroidJUnitRunner`, benchmarks via `-PkeenBenchmark`.
- [ ] **Lint**: `./gradlew lintClassikDebug lintKenningDebug`.
- [x] **Build Profiling**: configuration cache + build cache + parallel in `gradle.properties`.

## Phase 2: Performance Instrumentation
- [x] **Sanitizers**: document and gate via Gradle properties (`keenSanitizers`, `keenCoverage`, `keenFuncTrace`).
- [x] **Tracing**: perf + flamegraph capture for native hot paths.
- [x] **Coverage**: gcovr workflow (host or device, native only).
- [x] **Valgrind/Heaptrack**: valgrind workflow added (heaptrack pending).
- [x] **Infer**: static analysis workflow using pre-built Infer package.
- [x] **CTest**: host CMake + CTest harness for native smoke tests.

## Phase 3: Benchmarks & Telemetry
- [x] **Benchmarks**: parse/layout/solver timing harness.
- [x] **Memory/GPU**: baseline capture (Perfetto / gfxinfo).
- [x] **Logging**: structured logs for perf + crash triage.

## Phase 4: UI Automation & Crash Triage
- [x] **Espresso Flows**: launch -> new puzzle -> input -> validation.
- [x] **Story Mode Smoke**: Kenning narrative entry/exit.
- [x] **Crash Repro**: focused tests for regressions.

## Phase 5: Neural Integration (Follow-ups)
- [x] **Architecture Scope**: Define model size and integration point (ONNX/JNI).
- [x] **AI Model**: Exported `latin_solver.onnx` (supports 3x3-16x16 grids).
- [x] **Integration**: Implement `NeuralKeenGenerator.java` with ONNX Runtime.
- [x] **UI Toggle**: Add "Classic/Neural" switch.
- [ ] **Environment Setup**: refresh `scripts/ai` + `requirements.txt`.

## Phase 6: Synthesis (AI + C)
- [ ] **C Refactor**: Extract `new_game_desc_from_existing_grid` in `keen.c`.
- [ ] **JNI Bridge**: Expose `getZonesForGrid` to accept an `int[]` from Java.
- [ ] **AI Parser**: Convert ONNX `float[]` logits to `int[]` grid in Java.
- [ ] **Full Loop**: Connect AI -> Parser -> JNI -> Game.

## Phase 7: Expansion (Future)
- [ ] **Features**: Add more puzzle types or difficulty settings.
- [ ] **UI Polish**: Material Design 3 updates.
