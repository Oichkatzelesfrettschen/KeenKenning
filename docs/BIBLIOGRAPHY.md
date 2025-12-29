# Research Bibliography

**Project:** Keen Kenning Latin Square Neural Solver
**Compiled:** December 2025

---

## Core Architecture Papers

### Recurrent Transformer for Constraint Satisfaction
- **Title:** Learning to Solve Constraint Satisfaction Problems with Recurrent Transformer
- **Authors:** Zhun Yang, Adam Ishay, Joohyung Lee
- **arXiv:** [2307.04895](https://arxiv.org/abs/2307.04895) (July 2023)
- **Key Insights:**
  - Recurrence: Hidden states from step r become input for step r+1
  - Constraint loss applied at EVERY recurrent layer (not just final)
  - Straight-Through Estimator (STE) for gradient flow through argmax
  - Fewer layers (1-4) but many recurrent steps (32+) outperforms deep non-recurrent

### nanoGPT Training Practices
- **Repository:** [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)
- **Author:** Andrej Karpathy
- **Key Hyperparameters Applied:**
  - Learning rate: 6e-4 (max) -> 6e-5 (min)
  - Beta2: 0.95 (faster gradient adaptation for transformers)
  - Weight decay: 0.1
  - Gradient clipping: 1.0
  - Linear warmup + cosine decay schedule

### x-transformers Architecture Components
- **Repository:** [lucidrains/x-transformers](https://github.com/lucidrains/x-transformers)
- **Author:** Phil Wang (lucidrains)
- **Components Used:**
  - RMSNorm (Root Mean Square Layer Normalization)
  - Rotary Position Embeddings concept
  - Pre-LN transformer block ordering

---

## Latin Square Theory

### Mathematical Foundations
- **Title:** Latin Squares: Mathematical Frontiers, Scientific Applications, and Research Synthesis
- **Location:** `docs/LATIN_SQUARE_RESEARCH_2025.md`
- **Key Topics:**
  - Mutually Keen Kenningal Latin Squares (MOLS)
  - Quantum Latin Squares (QLS)
  - Transversal properties
  - Quasigroup encryption

### Improved MOLS Bounds
- **Authors:** Abel, Janiszczak, Staszewski (2025)
- **Findings:**
  - N(54) >= 8 (previously 5)
  - N(96) >= 10 (previously 9)
  - N(108) >= 9 (previously 8)

---

## Training Optimization References

### Mixed Precision Training
- **Source:** PyTorch Documentation
- **URL:** https://pytorch.org/docs/stable/amp.html
- **Techniques Applied:**
  - `torch.amp.autocast` with BF16 dtype
  - `GradScaler` for loss scaling
  - TF32 enabled via `torch.backends.cuda.matmul.allow_tf32`

### Flash Attention
- **Paper:** FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
- **Authors:** Tri Dao et al.
- **arXiv:** [2205.14135](https://arxiv.org/abs/2205.14135) (2022)
- **Implementation:** PyTorch `scaled_dot_product_attention` (native since 2.0)

### Gradient Accumulation
- **Source:** HuggingFace Transformers Training Tips
- **URL:** https://huggingface.co/docs/transformers/perf_train_gpu_one
- **Technique:** Accumulate gradients over N micro-batches before optimizer step

---

## GPU Optimization

### CUDA Best Practices
- **Source:** NVIDIA CUDA C++ Best Practices Guide
- **Key Optimizations:**
  - cuDNN benchmark mode for auto-tuning
  - Tensor Core utilization (batch sizes multiple of 8)
  - Memory coalescing patterns

### PyTorch 2.x Compiler
- **Source:** PyTorch Documentation
- **URL:** https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
- **Usage:** `torch.compile(model)` for graph optimization

---

## Data Generation

### Jacobson-Matthews Algorithm
- **Reference:** MCMC sampling for Latin squares
- **Application:** Uniform random Latin square generation
- **Implementation:** GPU-parallelized in `generate_data_cuda.py`

---

## Hardware Optimization

### RTX 40-Series Tensor Core Optimization
- **Title:** Accelerating Llama3 FP8 Inference with Triton Kernels
- **Source:** PyTorch Blog (2024)
- **URL:** https://pytorch.org/blog/accelerating-llama3/
- **Key Insights:**
  - TK-GEMM with SplitK parallelization for small batch sizes
  - Up to 1.94x speedup over base Triton matmul
  - Roofline analysis for tensor core utilization

### Triton Kernel Autotuning
- **Title:** TritonForge: Profiling-Guided Framework for Automated Triton Kernel Optimization
- **arXiv:** [2512.09196](https://arxiv.org/abs/2512.09196) (December 2024)
- **Key Insights:**
  - Occupancy-aware autotuning (25% optimal with 3-4 CTAs/SM)
  - Shared memory below 76KB for better SM utilization
  - Integration with PyTorch 2.0 TorchInductor

### ReBAR (Resizable BAR)
- **Source:** NVIDIA Driver Documentation
- **Key Insights:**
  - 16GB BAR1 mapping for full VRAM access
  - Reduces CPU-GPU transfer overhead
  - Enabled via Above 4G Decoding in BIOS

### AMD 3D V-Cache Optimization
- **Source:** Phoronix (2024)
- **URL:** https://www.phoronix.com/review/amd-3d-vcache-optimizer-9950x3d
- **Key Insights:**
  - 96MB L3 cache benefits data preprocessing
  - Kernel scheduler awareness for multi-CCD layouts
  - Improved DataLoader prefetching with large cache

### PyTorch CUDA Memory Management
- **Source:** PyTorch Documentation (2024)
- **URLs:**
  - [CUDA semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html)
  - [CUDA Environment Variables](https://docs.pytorch.org/docs/stable/cuda_environment_variables.html)
- **Key Settings:**
  - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` - Reduces fragmentation
  - `garbage_collection_threshold:0.8` - Active memory reclamation
  - `CUDA_MODULE_LOADING=LAZY` - Faster startup

---

## Additional Resources

### ONNX Runtime
- **URL:** https://onnxruntime.ai/
- **Usage:** Mobile inference on Android via JNI bridge

### Android NDK
- **Documentation:** https://developer.android.com/ndk
- **Version:** NDK 27.x for native C compilation

---

## Citation Format

```bibtex
@article{yang2023recurrent,
  title={Learning to Solve Constraint Satisfaction Problems with Recurrent Transformer},
  author={Yang, Zhun and Ishay, Adam and Lee, Joohyung},
  journal={arXiv preprint arXiv:2307.04895},
  year={2023}
}

@software{karpathy2023nanogpt,
  title={nanoGPT},
  author={Karpathy, Andrej},
  url={https://github.com/karpathy/nanoGPT},
  year={2023}
}

@article{dao2022flashattention,
  title={FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness},
  author={Dao, Tri and Fu, Daniel Y and Ermon, Stefano and Rudra, Atri and R{\'e}, Christopher},
  journal={arXiv preprint arXiv:2205.14135},
  year={2022}
}
```
