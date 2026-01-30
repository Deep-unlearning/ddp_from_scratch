"""Distributed training utilities."""

import os
import torch
import torch.distributed as dist


# -----------------
# Environment Helpers
# -----------------
def get_rank() -> int:
    """Get current process rank from environment."""
    return int(os.environ.get("RANK", 0))


def get_world_size() -> int:
    """Get total number of processes from environment."""
    return int(os.environ.get("WORLD_SIZE", 1))


def get_local_rank() -> int:
    """Get local rank (which GPU on this node)."""
    return int(os.environ.get("LOCAL_RANK", 0))


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    return get_rank() == 0


# -----------------
# Process Group Management
# -----------------
def setup_distributed(backend: str = "nccl"):
    """
    Initialize the distributed process group.
    
    Args:
        backend: Communication backend ("nccl" for GPU, "gloo" for CPU)
    """
    dist.init_process_group(backend=backend)


def cleanup_distributed():
    """Clean up distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


# -----------------
# Naive DDP Operations (Step 3)
# -----------------
def broadcast_model(model: torch.nn.Module, src: int = 0):
    """
    Broadcast all model parameters from source rank to all other ranks.
    
    This ensures all ranks start with identical weights.
    
    Args:
        model: The model whose parameters to broadcast
        src: Source rank to broadcast from (default: 0)
    """
    for p in model.parameters():
        dist.broadcast(p.data, src=src)


def sync_gradients(model: torch.nn.Module, world_size: int):
    """
    Synchronize gradients across all ranks using all-reduce.
    
    Performs sum reduction followed by division to compute average gradient.
    
    Args:
        model: The model whose gradients to synchronize
        world_size: Total number of processes
    """
    for p in model.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)
            p.grad.data /= world_size
