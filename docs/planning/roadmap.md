# Keen Kenning - Development Roadmap

## Phase 1: Harmonization & Cleanup
- [ ] **Directory Organization**: Move root docs to `docs/`, standardize file structure.
- [ ] **Code Cleanup**: Remove dead code from `KeenModelBuilder.java`.
- [ ] **Modernization**: Evaluate moving `jni` to `cpp` standard structure.
- [ ] **Documentation**: Update `README.md` with build instructions and architecture overview.

## Phase 2: Build & Verify
- [ ] **Build Check**: Ensure `./gradlew assembleDebug` passes.
- [ ] **Test**: Verify JNI bridge functionality (via unit tests or log verification if possible).

## Phase 3: Expansion (Future)
- [ ] **Features**: Add more puzzle types or difficulty settings.
- [ ] **UI Polish**: Material Design 3 updates.

## Phase 4: Neural Integration
- [x] **Architecture Scope**: Define model size and integration point (ONNX/JNI).
- [ ] **Environment Setup**: Create `scripts/ai` and `requirements.txt`.
- [x] **AI Model**: Exported `latin_solver.onnx` (supports 3x3-16x16 grids).
- [x] **Integration**: Implement `NeuralKeenGenerator.java` with ONNX Runtime.
- [x] **UI Toggle**: Add "Classic/Neural" switch.

## Phase 5: Synthesis (AI + C)
- [ ] **C Refactor**: Extract `new_game_desc_from_existing_grid` in `keen.c`.
- [ ] **JNI Bridge**: Expose `getZonesForGrid` to accept an `int[]` from Java.
- [ ] **AI Parser**: Convert ONNX `float[]` logits to `int[]` grid in Java.
- [ ] **Full Loop**: Connect AI -> Parser -> JNI -> Game.
