#!/usr/bin/env python3
"""
Hardware-Optimized Training Configuration

Target Systems:
- Primary: AMD Ryzen 5 5600X3D + NVIDIA RTX 4070 Ti
- Fallback: Generic CUDA or CPU systems

ISA Features Used:
- GPU: TF32, BF16, Flash Attention, Tensor Cores (SM 8.9)
- CPU: AVX2, FMA, SHA-NI (Zen 3)

Optimizations:
1. GPU: TF32 matmul, cuDNN benchmark, torch.compile
2. Memory: 8-bit optimizer (bitsandbytes), gradient checkpointing
3. DataLoader: pin_memory, prefetch_factor, 6-core CPU overlap
4. Mixed Precision: BF16 autocast with FP32 weights
"""

import os
import platform
import subprocess
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import torch


@dataclass
class HardwareProfile:
    """Detected hardware capabilities."""
    # CPU
    cpu_model: str = ""
    cpu_vendor: str = ""  # "AMD", "Intel", "Unknown"
    cpu_family: str = ""  # "zen3", "zen4", "alderlake", etc.
    cpu_cores: int = 1
    cpu_threads: int = 1
    has_avx2: bool = False
    has_avx512: bool = False
    has_fma: bool = False
    has_sha_ni: bool = False
    has_vaes: bool = False

    # CPU Cache (for 3D V-Cache detection)
    l3_cache_mb: int = 0
    has_vcache: bool = False  # 3D V-Cache (L3 >= 64MB on desktop Zen3/4)

    # GPU
    gpu_model: str = ""
    gpu_memory_gb: float = 0.0
    compute_capability: tuple = (0, 0)
    has_tensor_cores: bool = False
    has_tf32: bool = False
    has_bf16: bool = False
    has_fp8: bool = False  # Hopper only

    # System
    platform: str = ""
    cuda_version: str = ""
    torch_version: str = ""


def detect_cpu_features() -> Dict[str, Any]:
    """Detect CPU model, vendor, family, and ISA features."""
    info = {
        "model": "Unknown",
        "vendor": "Unknown",
        "family": "unknown",
        "cores": os.cpu_count() or 1,
        "threads": os.cpu_count() or 1,
        "avx2": False,
        "avx512": False,
        "fma": False,
        "sha_ni": False,
        "vaes": False,
        "l3_cache_mb": 0,
        "has_vcache": False,
    }

    try:
        with open("/proc/cpuinfo", "r") as f:
            cpuinfo = f.read()

        # Extract model name and vendor
        for line in cpuinfo.split("\n"):
            if "model name" in line:
                info["model"] = line.split(":")[1].strip()
            elif "vendor_id" in line:
                vendor_raw = line.split(":")[1].strip()
                if "AMD" in vendor_raw or "AuthenticAMD" in vendor_raw:
                    info["vendor"] = "AMD"
                elif "Intel" in vendor_raw or "GenuineIntel" in vendor_raw:
                    info["vendor"] = "Intel"

        # Check flags
        flags_line = ""
        for line in cpuinfo.split("\n"):
            if "flags" in line:
                flags_line = line.lower()
                break

        info["avx2"] = "avx2" in flags_line
        info["avx512"] = "avx512f" in flags_line
        info["fma"] = "fma" in flags_line
        info["sha_ni"] = "sha_ni" in flags_line
        info["vaes"] = "vaes" in flags_line

        # Physical cores (Linux)
        try:
            result = subprocess.run(
                ["nproc", "--all"], capture_output=True, text=True
            )
            info["threads"] = int(result.stdout.strip())
            info["cores"] = info["threads"] // 2  # Assume SMT
        except:
            pass

        # L3 Cache detection (for 3D V-Cache)
        try:
            result = subprocess.run(
                ["lscpu"], capture_output=True, text=True
            )
            for line in result.stdout.split("\n"):
                if "L3 cache:" in line:
                    cache_str = line.split(":")[1].strip()
                    # Parse "96 MiB" or "32M" etc.
                    cache_val = ''.join(c for c in cache_str.split()[0] if c.isdigit())
                    if cache_val:
                        info["l3_cache_mb"] = int(cache_val)
                    break
        except:
            pass

        # Determine CPU family
        info["family"] = _detect_cpu_family(info["vendor"], info["model"], flags_line)

        # 3D V-Cache detection: Desktop Zen3/4 X3D chips have 64+ MB L3
        if info["vendor"] == "AMD" and info["l3_cache_mb"] >= 64:
            info["has_vcache"] = True

    except Exception as e:
        print(f"CPU detection error: {e}")

    return info


def _detect_cpu_family(vendor: str, model: str, flags: str) -> str:
    """Determine CPU microarchitecture family."""
    model_lower = model.lower()

    if vendor == "AMD":
        # Zen 4 (Ryzen 7000, EPYC Genoa) - has AVX-512
        if "7" in model and "ryzen" in model_lower and "avx512" in flags:
            return "zen4"
        # Zen 3 (Ryzen 5000, EPYC Milan) - no AVX-512
        if "5" in model and "ryzen" in model_lower:
            if "5600x3d" in model_lower or "5800x3d" in model_lower:
                return "zen3_vcache"  # 3D V-Cache variant
            return "zen3"
        # Zen 2 (Ryzen 3000, EPYC Rome)
        if "3" in model and "ryzen" in model_lower:
            return "zen2"
        # Zen+ (Ryzen 2000)
        if "2" in model and "ryzen" in model_lower:
            return "zenplus"
        # EPYC detection
        if "epyc" in model_lower:
            if "avx512" in flags:
                return "zen4"  # Genoa
            return "zen3"  # Milan/Rome
        return "zen"  # Generic AMD Zen

    elif vendor == "Intel":
        # Raptor Lake / Alder Lake (12th/13th/14th gen)
        if any(x in model_lower for x in ["13th", "14th", "raptor"]):
            return "raptorlake"
        if any(x in model_lower for x in ["12th", "alder"]):
            return "alderlake"
        # Rocket Lake (11th gen)
        if "11th" in model_lower:
            return "rocketlake"
        # Tiger Lake / Ice Lake
        if "tiger" in model_lower or "ice" in model_lower:
            return "icelake"
        return "intel"  # Generic Intel

    return "unknown"


def detect_gpu_features() -> Dict[str, Any]:
    """Detect GPU model and capabilities."""
    info = {
        "model": "None",
        "memory_gb": 0.0,
        "compute_cap": (0, 0),
        "tensor_cores": False,
        "tf32": False,
        "bf16": False,
        "fp8": False,
    }

    if not torch.cuda.is_available():
        return info

    try:
        info["model"] = torch.cuda.get_device_name(0)
        info["memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9

        major = torch.cuda.get_device_properties(0).major
        minor = torch.cuda.get_device_properties(0).minor
        info["compute_cap"] = (major, minor)

        # Feature availability by compute capability
        # Tensor Cores: Volta (7.0) and later
        info["tensor_cores"] = major >= 7

        # TF32: Ampere (8.0) and later
        info["tf32"] = major >= 8

        # BF16: Ampere (8.0) and later
        info["bf16"] = major >= 8

        # FP8: Hopper (9.0) and later
        info["fp8"] = major >= 9

    except Exception as e:
        print(f"GPU detection error: {e}")

    return info


def detect_hardware() -> HardwareProfile:
    """Detect full hardware profile."""
    cpu = detect_cpu_features()
    gpu = detect_gpu_features()

    return HardwareProfile(
        cpu_model=cpu["model"],
        cpu_vendor=cpu["vendor"],
        cpu_family=cpu["family"],
        cpu_cores=cpu["cores"],
        cpu_threads=cpu["threads"],
        has_avx2=cpu["avx2"],
        has_avx512=cpu["avx512"],
        has_fma=cpu["fma"],
        has_sha_ni=cpu["sha_ni"],
        has_vaes=cpu["vaes"],
        l3_cache_mb=cpu["l3_cache_mb"],
        has_vcache=cpu["has_vcache"],
        gpu_model=gpu["model"],
        gpu_memory_gb=gpu["memory_gb"],
        compute_capability=gpu["compute_cap"],
        has_tensor_cores=gpu["tensor_cores"],
        has_tf32=gpu["tf32"],
        has_bf16=gpu["bf16"],
        has_fp8=gpu["fp8"],
        platform=platform.system(),
        cuda_version=torch.version.cuda or "N/A",
        torch_version=torch.__version__,
    )


@dataclass
class TrainingConfig:
    """Hardware-optimized training configuration."""
    # Model
    d_model: int = 256
    n_layer: int = 8
    n_head: int = 8
    dropout: float = 0.1

    # Training
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    epochs: int = 60
    grad_clip: float = 1.0

    # Precision
    use_amp: bool = True
    amp_dtype: str = "bfloat16"  # or "float16"
    use_tf32: bool = True

    # DataLoader
    num_workers: int = 6
    pin_memory: bool = True
    prefetch_factor: int = 4
    persistent_workers: bool = True

    # Memory optimization
    use_8bit_optimizer: bool = False  # Requires bitsandbytes
    gradient_checkpointing: bool = False
    channels_last: bool = True

    # Compilation
    use_torch_compile: bool = True
    compile_mode: str = "default"  # reduce-overhead, max-autotune, default

    # Inductor (CPU backend)
    inductor_simd: int = 256  # AVX2 = 256, AVX512 = 512
    inductor_cpp_wrapper: bool = True  # Faster CPU code generation

    # CPU Architecture-specific
    cpu_march: str = "native"  # GCC -march flag
    cpu_mtune: str = "native"  # GCC -mtune flag
    use_onednn: bool = True  # Intel MKL-DNN / oneDNN
    use_mkl: bool = True  # Intel MKL

    # 3D V-Cache optimization
    vcache_batch_boost: int = 0  # Extra batch size for V-Cache CPUs
    vcache_prefetch: bool = False  # Enable aggressive prefetching

    # Hardware-specific
    gpu_sm_count: int = 60  # RTX 4070 Ti has 60 SMs


def get_optimized_config(hw: HardwareProfile) -> TrainingConfig:
    """Generate optimized config for detected hardware."""
    config = TrainingConfig()

    # === CPU Optimizations ===
    # Use half the threads for DataLoader (leave room for main process)
    config.num_workers = max(2, hw.cpu_threads // 2)

    # AVX2 vs AVX512
    if hw.has_avx512:
        config.inductor_simd = 512
    elif hw.has_avx2:
        config.inductor_simd = 256
    else:
        config.inductor_simd = 128  # SSE fallback

    # CPU architecture-specific flags
    config.cpu_march, config.cpu_mtune = _get_cpu_march_flags(hw)

    # === AMD Zen3 / V-Cache Optimizations ===
    if hw.cpu_family in ("zen3", "zen3_vcache"):
        # Zen3 specific: prefer 256-bit ops (AVX2), avoid 512-bit
        config.inductor_simd = 256
        config.use_onednn = True  # oneDNN has Zen3 kernels

        # 3D V-Cache specific optimizations
        if hw.has_vcache:
            # With 96MB L3, we can afford larger batches that stay cache-resident
            config.vcache_batch_boost = 32  # +32 to base batch size
            config.vcache_prefetch = True
            # More workers can share L3 cache
            config.num_workers = min(hw.cpu_threads - 2, 8)
            config.prefetch_factor = 6  # More aggressive prefetching

    elif hw.cpu_family == "zen4":
        # Zen4: has AVX-512, use it
        config.inductor_simd = 512
        config.cpu_march = "znver4"
        config.use_onednn = True

    elif hw.cpu_vendor == "Intel":
        # Intel: use MKL for best performance
        config.use_mkl = True
        config.use_onednn = True

    # === GPU Optimizations ===
    if hw.compute_capability >= (8, 0):
        # Ampere+ (RTX 30xx, 40xx, A100, etc.)
        config.use_tf32 = True
        config.amp_dtype = "bfloat16"
        config.batch_size = 128  # Larger batches for Tensor Cores
        config.channels_last = True
    elif hw.compute_capability >= (7, 0):
        # Volta/Turing (RTX 20xx, V100)
        config.use_tf32 = False
        config.amp_dtype = "float16"
        config.batch_size = 64
    else:
        # Older GPUs
        config.use_amp = False
        config.batch_size = 32

    # Apply V-Cache batch boost
    if hw.has_vcache and config.vcache_batch_boost > 0:
        config.batch_size += config.vcache_batch_boost

    # Adjust for VRAM
    if hw.gpu_memory_gb < 8:
        config.batch_size = min(config.batch_size, 32)
        config.gradient_checkpointing = True
    elif hw.gpu_memory_gb < 12:
        config.batch_size = min(config.batch_size, 64)

    # torch.compile tuning based on SM count
    if hw.compute_capability >= (8, 0):
        # RTX 4070 Ti has 60 SMs - not enough for max-autotune (needs 80+)
        config.compile_mode = "default"
    else:
        config.compile_mode = "reduce-overhead"

    # 8-bit optimizer if bitsandbytes available
    try:
        import bitsandbytes
        config.use_8bit_optimizer = True
    except ImportError:
        config.use_8bit_optimizer = False

    return config


def _get_cpu_march_flags(hw: HardwareProfile) -> tuple:
    """Get GCC -march and -mtune flags for CPU architecture."""
    # Map CPU family to GCC architecture names
    march_map = {
        "zen4": ("znver4", "znver4"),
        "zen3": ("znver3", "znver3"),
        "zen3_vcache": ("znver3", "znver3"),
        "zen2": ("znver2", "znver2"),
        "zenplus": ("znver1", "znver1"),
        "zen": ("znver1", "znver1"),
        "raptorlake": ("raptorlake", "raptorlake"),
        "alderlake": ("alderlake", "alderlake"),
        "rocketlake": ("rocketlake", "rocketlake"),
        "icelake": ("icelake-client", "icelake-client"),
        "intel": ("native", "native"),
    }
    return march_map.get(hw.cpu_family, ("native", "native"))


def apply_optimizations(config: TrainingConfig):
    """Apply hardware optimizations to PyTorch."""
    # TF32 for matmul
    if config.use_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("TF32 enabled for Tensor Cores")

    # cuDNN benchmark
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print("cuDNN benchmark enabled")

    # Set inductor SIMD width for CPU ops
    if config.inductor_simd >= 512:
        os.environ["ATEN_CPU_CAPABILITY"] = "avx512"
    elif config.inductor_simd >= 256:
        os.environ["ATEN_CPU_CAPABILITY"] = "avx2"
    else:
        os.environ["ATEN_CPU_CAPABILITY"] = "default"

    # Inductor C++ wrapper for faster CPU codegen
    if config.inductor_cpp_wrapper:
        os.environ["TORCHINDUCTOR_CPP_WRAPPER"] = "1"

    # oneDNN / MKL-DNN
    if config.use_onednn:
        os.environ["ONEDNN_VERBOSE"] = "0"  # Disable verbose logging
        torch.backends.mkldnn.enabled = True

    # V-Cache prefetching optimization
    if config.vcache_prefetch:
        # Enable software prefetching in inductor
        os.environ["TORCHINDUCTOR_FREEZING"] = "1"
        os.environ["TORCHINDUCTOR_DISABLE_CACHE"] = "0"

    # Flash Attention
    if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        print("Flash Attention available")


def print_hardware_report(hw: HardwareProfile, config: TrainingConfig):
    """Print hardware and config summary."""
    print("\n" + "=" * 60)
    print("HARDWARE PROFILE")
    print("=" * 60)
    print(f"CPU:  {hw.cpu_model}")
    print(f"      Vendor: {hw.cpu_vendor}, Family: {hw.cpu_family}")
    print(f"      {hw.cpu_cores} cores / {hw.cpu_threads} threads")
    print(f"      L3 Cache: {hw.l3_cache_mb} MB" + (" (3D V-Cache)" if hw.has_vcache else ""))
    print(f"      AVX2: {hw.has_avx2}, AVX512: {hw.has_avx512}, FMA: {hw.has_fma}")
    print(f"      SHA-NI: {hw.has_sha_ni}, VAES: {hw.has_vaes}")
    print(f"GPU:  {hw.gpu_model}")
    print(f"      {hw.gpu_memory_gb:.1f} GB VRAM, SM {hw.compute_capability[0]}.{hw.compute_capability[1]}")
    print(f"      TensorCores: {hw.has_tensor_cores}, TF32: {hw.has_tf32}, BF16: {hw.has_bf16}, FP8: {hw.has_fp8}")
    print(f"PyTorch: {hw.torch_version}, CUDA: {hw.cuda_version}")

    print("\n" + "-" * 60)
    print("OPTIMIZED CONFIG")
    print("-" * 60)
    print(f"Batch size:    {config.batch_size}" + (f" (+{config.vcache_batch_boost} V-Cache boost)" if config.vcache_batch_boost else ""))
    print(f"DataLoader:    {config.num_workers} workers, prefetch={config.prefetch_factor}")
    print(f"Precision:     {'AMP ' + config.amp_dtype if config.use_amp else 'FP32'}, TF32={config.use_tf32}")
    print(f"Optimizer:     {'AdamW8bit' if config.use_8bit_optimizer else 'AdamW'}")
    print(f"Compile:       {config.compile_mode}")
    print(f"CPU Target:    -march={config.cpu_march} -mtune={config.cpu_mtune}")
    print(f"CPU SIMD:      {config.inductor_simd}-bit, oneDNN={config.use_onednn}")
    if hw.has_vcache:
        print(f"V-Cache:       prefetch={config.vcache_prefetch}, batch_boost=+{config.vcache_batch_boost}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    hw = detect_hardware()
    config = get_optimized_config(hw)
    apply_optimizations(config)
    print_hardware_report(hw, config)
