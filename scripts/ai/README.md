# ML Puzzle Generator

This directory contains the Machine Learning training pipeline for Orthogon's Latin square solver.

## Scripts

| Script | Purpose | Output |
|--------|---------|--------|
| `generate_data.py` | Generate Latin square training data (CPU) | `data/latin_squares_massive.npz` |
| `generate_data_cuda.py` | GPU-accelerated data generation | `data/latin_squares.npz` |
| `train_massive_model.py` | Full 16x16 model training (3-16 grid sizes) | `keen_solver_16x16.onnx` |
| `train_enhanced.py` | **Enhanced** training with curriculum learning | `keen_solver_enhanced.onnx` |
| `final_massive_training.py` | Optuna hyperparameter tuning | `keen_solver_16x16.onnx` |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate training data (choose one)
python generate_data.py       # CPU (slow)
python generate_data_cuda.py  # GPU (fast, requires CUDA)

# Train model (choose one)
python train_massive_model.py   # Basic training
python train_enhanced.py        # Enhanced (recommended)
```

## Enhanced Training Pipeline

The `train_enhanced.py` script includes several improvements:

### Features

| Feature | Description |
|---------|-------------|
| **Curriculum Learning** | Starts with 3x3 grids, progressively adds larger sizes |
| **Data Augmentation** | Digit permutation, row/column swaps, transposition |
| **Constraint Loss** | Latin square constraint encouragement during training |
| **Early Stopping** | Prevents overfitting with patience-based stopping |
| **Gradient Clipping** | Stabilizes training with norm clipping |
| **Warmup Schedule** | LR warmup followed by cosine decay |

### Usage

```bash
# Basic enhanced training
python train_enhanced.py

# With curriculum learning (recommended for best accuracy)
python train_enhanced.py --curriculum --epochs 30

# Custom configuration
python train_enhanced.py \
    --data-path data/latin_squares_massive.npz \
    --output-prefix keen_solver_v2 \
    --d-model 256 \
    --layers 8 \
    --curriculum \
    --constraint-weight 0.15 \
    --epochs 25
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-path` | `data/latin_squares_massive.npz` | Training data path |
| `--output-prefix` | `keen_solver_enhanced` | Output file prefix |
| `--d-model` | 128 | Transformer hidden dimension |
| `--nhead` | 8 | Attention heads |
| `--layers` | 6 | Transformer layers |
| `--dropout` | 0.1 | Dropout rate |
| `--epochs` | 20 | Training epochs |
| `--batch-size` | 32 | Batch size |
| `--lr` | 3e-4 | Learning rate |
| `--curriculum` | False | Enable curriculum learning |
| `--constraint-weight` | 0.1 | Latin constraint loss weight |
| `--patience` | 5 | Early stopping patience |

## Model Architecture

The `RelationalTransformer` architecture (16x16 support):

```
Input Grid (16x16) -> Flatten -> Embedding -> Positional Encoding
    -> Pre-LN Transformer Encoder (6 layers)
    -> Residual Output Head -> Logits (17 classes per cell)
```

- **Input**: 16x16 grid with 0 for empty, 1-16 for filled cells (supports 3x3 to 16x16)
- **Output**: Per-cell probability distribution over 17 classes
- **Parameters**: ~2.5M (default config with 256 positions)

## Integration

1. Copy ONNX model to Android assets:
   ```bash
   cp keen_solver_enhanced.onnx ../../app/src/main/assets/keen_solver_9x9.onnx
   cp keen_solver_enhanced.onnx.data ../../app/src/main/assets/keen_solver_9x9.onnx.data
   ```

2. The Android app loads via ONNX Runtime through JNI
3. See `NeuralKenKenGenerator.java` for inference implementation

## Metrics

Training outputs cell-level and grid-level accuracy:
- **Cell Accuracy**: Percentage of individual cells correctly predicted
- **Grid Accuracy**: Percentage of entire grids correctly solved

Typical results with enhanced training:
- Cell accuracy: 95-98%
- Grid accuracy: 70-85% (varies by grid size)
