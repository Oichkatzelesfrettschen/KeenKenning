# Neural Puzzle Generator - Architecture Scope

## Objective
Integrate a Machine Learning model (PyTorch/ONNX) to generate puzzle layouts, serving as an alternative to the "Classic" algorithmic generation (C/Simon Tatham).

## Architecture

### 1. Model Design (The "Tiny Torch")
- **Task**: Latin Square Completion / Puzzle Validty Prediction.
- **Input**: A partially filled grid or a seed vector.
- **Output**: A probability distribution over cell values (1-N).
- **Architecture**:
    - **Type**: Small Transformer or lightweight CNN (e.g., Mini-ResNet).
    - **Size**: < 5MB to ensure fast on-device inference and minimal APK bloat.
    - **Format**: Exported to **ONNX** (Open Neural Network Exchange).

### 2. Integration Strategy
- **Runtime**: **ONNX Runtime (ORT)** for C/C++.
    - *Why ORT?* Easier integration with the existing NDK C codebase than PyTorch Mobile, and often lighter.
- **JNI Layer**:
    - Existing: `getLevelFromC` (calls `keen.c`).
    - New: `getLevelFromAI` (calls `keen_ai.cpp` wrapping ORT).
- **Fallback**: Hybrid approach where AI proposes a grid, and the "Classic" C logic verifies/fixes constraints.

### 3. Implementation Plan
1.  **Environment**: Set up Python/PyTorch training environment in `scripts/ai/`.
2.  **Training**: Train a model to generate valid Latin Squares of size 4x4 to 9x9.
3.  **Export**: Convert `.pt` model to `.onnx`.
4.  **Native Integration**:
    - Add ONNX Runtime shared library to `app/src/main/jni/libs`.
    - Create `keen_ai.cpp`.
5.  **UI Update**: Add a "Generator Mode" toggle (Classic vs. Neural) in the Settings.

## Terminology
- **Classic Logic**: The existing deterministic C algorithms.
- **Neural Logic**: The new stochastic AI-based generation.
- **Synthesized Mode**: AI generates, C validates.
