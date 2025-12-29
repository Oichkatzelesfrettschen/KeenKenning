# ML Puzzle Generator

This directory contains the Machine Learning training pipeline for Keen Kenning's Latin square solver.

## Scripts

| Script | Purpose | Output |
|--------|---------|--------|
| `generate_data.py` | Generate Latin square training data (CPU) | `data/latin_squares.npz` |
| `generate_data_cuda.py` | GPU-accelerated data generation (3-16) | `data/latin_squares_massive.npz` |
| `train_autoregressive.py` | **Recommended** autoregressive training | `latin_solver.onnx` |
| `train_enhanced.py` | Enhanced training with curriculum | `keen_solver_enhanced.onnx` |
| `train_massive_model.py` | Basic training (deprecated) | `keen_solver_16x16.onnx` |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate training data (GPU - 1.14M grids for sizes 3-16)
python generate_data_cuda.py --full --count 125000

# Train model (autoregressive, recommended)
python train_autoregressive.py --curriculum --epochs 60 --target-loss 0.09

# Deploy to Android
cp latin_solver.onnx ../../app/src/main/assets/
```

## Autoregressive Training Pipeline

The `train_autoregressive.py` script is the production training pipeline with:

### Architecture (nanoGPT-inspired)

| Component | Description |
|-----------|-------------|
| **RMSNorm** | Root Mean Square normalization (x-transformers) |
| **CausalSelfAttention** | Flash Attention via `scaled_dot_product_attention` |
| **Size Conditioning** | Explicit grid size embedding for multi-size support |
| **Constraint-Aware Loss** | Auxiliary penalty for Latin violations |

### GPU Optimizations

| Optimization | Benefit |
|--------------|---------|
| **Flash Attention** | O(N) memory, 2-4x faster attention |
| **TF32 Precision** | Tensor Core acceleration on Ampere+ |
| **cuDNN Benchmark** | Auto-tuned convolution kernels |
| **torch.compile** | Fused operations, reduced overhead |
| **pin_memory** | Async CPU-GPU transfer |

### Training Features

| Feature | Description |
|---------|-------------|
| **Multi-Dim Curriculum** | Size [3-5]→[3-16], Mode [STD→ALL], Fill [70%→0%] |
| **Variable Fill Ratios** | Training matches inference (empty → full) |
| **Cosine LR + Warmup** | nanoGPT-style schedule with min_lr floor |
| **AdamW β₂=0.95** | Faster gradient adaptation |
| **Precision Control** | FP32, FP16 (default), BF16 (Ampere+) |
| **Gradient Checkpointing** | ~30% memory savings, ~20% speed cost |

### Usage

```bash
# Full training with all curriculum dimensions (recommended)
python train_autoregressive.py --curriculum --mode-curriculum --fill-curriculum \
    --epochs 60 --target-loss 0.09

# Quick test run
python train_autoregressive.py --epochs 5 --batch-size 64

# BF16 precision with gradient checkpointing (memory-efficient)
python train_autoregressive.py --dtype bf16 --grad-checkpoint --curriculum

# Custom configuration
python train_autoregressive.py \
    --data data/latin_squares_massive.npz \
    --output latin_solver \
    --d-model 256 \
    --n-layer 8 \
    --curriculum --mode-curriculum --fill-curriculum \
    --constraint-weight 0.15 \
    --epochs 60 \
    --target-loss 0.09
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data` | `data/latin_squares_massive.npz` | Training data path |
| `--output` | `latin_solver` | Output file prefix |
| `--d-model` | 256 | Transformer hidden dim (multiple of 8) |
| `--n-layer` | 8 | Transformer layers |
| `--batch-size` | 128 | Batch size (tensor core saturation) |
| `--lr` | 3e-4 | Learning rate |
| `--epochs` | 60 | Training epochs |
| `--curriculum` | False | Enable size curriculum [3-5]→[3-8]→[3-12]→[3-16] |
| `--mode-curriculum` | False | Enable mode curriculum [STANDARD→ZERO→NEGATIVE] |
| `--fill-curriculum` | False | Enable fill curriculum [70%→50%→30%→10%] |
| `--dtype` | fp16 | Precision: fp32, fp16, bf16 |
| `--grad-checkpoint` | False | Gradient checkpointing (~30% memory savings) |
| `--profile` | False | Enable torch.profiler for first epoch |
| `--constraint-weight` | 0.15 | Latin constraint loss weight |
| `--target-loss` | 0.09 | Early stop target |
| `--use-8bit` | False | Use 8-bit AdamW (requires bitsandbytes) |
| `--prefetch` | 4 | DataLoader prefetch factor |
| `--num-workers` | 6 | DataLoader workers (use ~half CPU cores) |
| `--hw-report` | False | Print hardware detection report |

### Hardware Optimization

The `hardware_config.py` module provides automatic hardware detection:

```bash
# Run hardware detection report
python hardware_config.py

# Train with 8-bit optimizer (saves ~40% optimizer memory)
pip install bitsandbytes
python train_autoregressive.py --use-8bit --curriculum

# Custom DataLoader tuning for your CPU
python train_autoregressive.py --num-workers 4 --prefetch 2
```

Detected optimizations:
- **GPU**: TF32, BF16, Flash Attention (SM 8.0+)
- **CPU**: AVX2/AVX-512 vectorization via Inductor
- **Memory**: 8-bit optimizer, gradient checkpointing

## Model Architecture

The `ConstraintAwareTransformer` (6.4M parameters):

```
Inputs:
  - input_grid: [batch, 16, 16] INT64 (0=empty, 1-16=values)
  - grid_size: [batch] INT64 (actual size 3-16)

Processing:
  Grid -> Flatten -> Token Embedding + Position Embedding + Size Embedding
    -> Dropout
    -> 8x [RMSNorm -> CausalSelfAttention -> RMSNorm -> MLP]
    -> RMSNorm -> Output Head

Output:
  - cell_logits: [batch, 17, 16, 16] (17 classes per cell)
```

## Metrics Tracked

| Metric | Description |
|--------|-------------|
| **Valid Grid Rate** | % of generated grids with zero violations |
| **Per-Cell Accuracy** | % of cells matching target |
| **Avg Violations** | Mean row/column duplicates per grid |
| **Generation Entropy** | Diversity of predictions |

## Integration

1. Copy trained model to Android assets:
   ```bash
   cp latin_solver.onnx ../../app/src/main/assets/
   cp latin_solver.onnx.data ../../app/src/main/assets/  # if exists
   ```

2. The model expects **two inputs**:
   - `input_grid`: Zero-initialized 16x16 grid
   - `grid_size`: The target size (3-16)

3. See `NeuralKenKenGenerator.java` for inference implementation

## Performance Targets

| Metric | Target | Current |
|--------|--------|---------|
| Training Loss | < 0.09 | In progress |
| Valid Grid Rate (3x3) | > 99% | TBD |
| Valid Grid Rate (9x9) | > 85% | TBD |
| Valid Grid Rate (16x16) | > 60% | TBD |
