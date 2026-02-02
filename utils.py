"""Utility functions for training."""

import random
import hashlib

import numpy as np
import torch


# -----------------
# GPU FLOPS (theoretical peak for bf16/fp16)
# -----------------
GPU_PEAK_TFLOPS = {
    "NVIDIA A100": 312,
    "NVIDIA A100-SXM4-80GB": 312,
    "NVIDIA A100-SXM4-40GB": 312,
    "NVIDIA A100-PCIE-40GB": 312,
    "NVIDIA A10": 125,
    "NVIDIA A10G": 125,
    "NVIDIA V100": 125,
    "NVIDIA T4": 65,
    "NVIDIA RTX 4090": 330,
    "NVIDIA RTX 3090": 142,
    "NVIDIA H100": 990,
    "NVIDIA L4": 121
}


def get_gpu_peak_tflops() -> float:
    """Get theoretical peak TFLOPs for the current GPU (bf16/fp16)."""
    if not torch.cuda.is_available():
        return 1.0  # Avoid division by zero
    
    gpu_name = torch.cuda.get_device_name(0)
    
    # Try exact match first
    if gpu_name in GPU_PEAK_TFLOPS:
        return GPU_PEAK_TFLOPS[gpu_name]
    
    # Try partial match
    for key, value in GPU_PEAK_TFLOPS.items():
        if key in gpu_name or gpu_name in key:
            return value
    
    # Default fallback
    print(f"Warning: Unknown GPU '{gpu_name}', using 100 TFLOPs as default")
    return 100.0


def estimate_flops_per_step(
    num_params: int,
    seq_len: int,
    global_batch_size: int,
    grad_accum_steps: int = 1,
) -> float:
    """
    Estimate FLOPs per optimizer step for transformer training.
    
    Formula from PyTorch blog (https://pytorch.org/blog/large-scale-training-hugging-face/):
        tokens_per_batch = global_batch_size * seq_len
        FLOPS_per_step = 6 * tokens_per_batch * num_params
    
    The factor of 6 comes from:
    - Forward pass: 2 * num_params * tokens (matmul = 2 ops per param)
    - Backward pass: 4 * num_params * tokens (grad computation = 2x forward)
    
    Note: This assumes d_model >> seq_len. If violated, self-attention FLOPs
    become significant and this will underestimate true MFU.
    
    Args:
        num_params: Number of model parameters
        seq_len: Sequence length (block_size)
        global_batch_size: Total batch size across all devices
        grad_accum_steps: Number of gradient accumulation steps
    
    Returns:
        FLOPs per optimizer step (not TFLOPs)
    """
    tokens_per_batch = global_batch_size * seq_len * grad_accum_steps
    flops_per_step = 6 * tokens_per_batch * num_params
    return flops_per_step


def compute_mfu(
    flops_per_step: float,
    step_time_seconds: float,
    gpu_peak_flops: float,
    num_devices: int = 1,
) -> float:
    """
    Compute Model FLOPs Utilization (MFU).
    
    Formula from PyTorch blog (https://pytorch.org/blog/large-scale-training-hugging-face/):
        MFU = FLOPS_per_step / step_time(s) / chip_count / FLOPS_per_chip
    
    MFU measures how effectively the implementation uses the hardware.
    100% MFU means the hardware is being used perfectly.
    
    Args:
        flops_per_step: FLOPs per optimizer step
        step_time_seconds: Time per optimizer step in seconds
        gpu_peak_flops: Peak FLOPS per GPU (in raw FLOPS, not TFLOPs)
        num_devices: Number of GPUs/chips
    
    Returns:
        MFU as a fraction (0.0 to 1.0)
    """
    if step_time_seconds <= 0:
        return 0.0
    
    # MFU = FLOPS_per_step / step_time / chip_count / FLOPS_per_chip
    mfu = flops_per_step / step_time_seconds / num_devices / gpu_peak_flops
    return mfu


def set_seed(seed: int, rank: int = 0):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Base seed value
        rank: Process rank (used to offset seed for different data per rank)
    """
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def global_l2_norm(tensors) -> float:
    """Compute global L2 norm over a list of tensors."""
    sq_sum = None
    device = None
    with torch.no_grad():
        for t in tensors:
            if t is None:
                continue
            if sq_sum is None:
                device = t.device
                sq_sum = torch.zeros((), device=device)
            sq_sum += (t.float() ** 2).sum()
    return float(torch.sqrt(sq_sum).item()) if sq_sum is not None else 0.0
