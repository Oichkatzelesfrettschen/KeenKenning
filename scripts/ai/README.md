# Neural Puzzle Generator

This directory contains the experimental Neural Network based puzzle generator for KeenKeenForAndroid.

## Scripts

| Script | Purpose | Output |
|--------|---------|--------|
| `generate_data.py` | Generate Latin square training data (CPU) | `data/latin_squares.npz` |
| `generate_data_cuda.py` | GPU-accelerated data generation | `data/latin_squares.npz` |
| `train_massive_model.py` | **Production** 9x9 model | `keen_solver_9x9.onnx` |
| `final_massive_training.py` | Optimized training with Optuna HP tuning | `keen_solver_9x9.onnx` |

## Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Generate training data:
   ```bash
   python generate_data.py       # CPU (slow)
   python generate_data_cuda.py  # GPU (fast, requires CUDA)
   ```

3. Train and export the production model:
   ```bash
   python train_massive_model.py
   ```
   This generates `keen_solver_9x9.onnx` for deployment.

## Integration

- Copy `keen_solver_9x9.onnx` and `keen_solver_9x9.onnx.data` to `app/src/main/assets/`
- The Android app loads via ONNX Runtime (C++ API) through JNI
- See `NeuralKeenGenerator.java` for inference implementation
