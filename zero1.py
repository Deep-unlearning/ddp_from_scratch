"""
Implements basic ZeRO-1 (optimizer state sharding only).
"""

import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import Adam

from ddp import get_rank, get_world_size, ZeRO1Optimizer
from utils import set_seed


def print_memory_stats(tag, model, optimizer, rank, device):
    allocated = torch.cuda.memory_allocated(device) / 1024**2
    reserved = torch.cuda.memory_reserved(device) / 1024**2
    params_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
    grads_mb = sum(
        p.grad.numel() * p.grad.element_size()
        for p in model.parameters()
        if p.grad is not None
    ) / 1024**2
    opt = optimizer.optimizer if hasattr(optimizer, "optimizer") else optimizer
    opt_state_mb = 0.0
    for state in opt.state.values():
        for v in state.values():
            if torch.is_tensor(v):
                opt_state_mb += v.numel() * v.element_size()
    opt_state_mb /= 1024**2
    print(
        f"[rank{rank}] {tag} | allocated={allocated:.2f}MB "
        f"reserved={reserved:.2f}MB params={params_mb:.2f}MB "
        f"grads={grads_mb:.2f}MB opt_state={opt_state_mb:.2f}MB"
    )


def train(model, optimizer, device, is_zero1=False):
    rank = get_rank()
    batch_size = 16
    x = torch.randn(batch_size, 10000, device=device)
    y = torch.randn(batch_size, 10000, device=device)

    # Warmup step to avoid first-step overhead
    optimizer.zero_grad()
    output = model(x)
    loss = nn.functional.mse_loss(output, y)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()

    if rank == 0:
        print_memory_stats("Initial state", model, optimizer, rank, device)
    dist.barrier()

    peak_memories = []
    for i in range(10):
        torch.cuda.reset_peak_memory_stats(device)
        optimizer.zero_grad()
        output = model(x)
        loss = nn.functional.mse_loss(output, y)

        if rank == 0 and i == 0:
            print(f"\nStep {i} memory:")
            print(
                f"Before backward: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB"
            )

        loss.backward()
        torch.cuda.synchronize()

        if rank == 0 and i == 0:
            grad_memory = sum(
                p.grad.numel() * p.grad.element_size() / 1024**2
                for p in model.parameters()
                if p.grad is not None
            )
            print(f"Gradient memory after backward: {grad_memory:.2f} MB")

        optimizer.step()
        current_peak = torch.cuda.max_memory_allocated(device) / 1024**2
        peak_memories.append(current_peak)

        if rank == 0 and i == 0:
            print(f"Peak memory this step: {current_peak:.2f} MB")
        dist.barrier()

    if rank == 0:
        print(f"\nFinal peak memory: {max(peak_memories):.2f} MB")
        if is_zero1:
            print("Note: ZeRO-1 keeps full params/grads; only optimizer state is sharded.")

    return model, optimizer, max(peak_memories)


def test_zero1():
    dist.init_process_group("nccl")
    rank = get_rank()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    set_seed(42)

    # Test with regular Adam
    print(f"\nGPU {rank} - Testing with regular Adam:")
    torch.cuda.reset_peak_memory_stats()
    model = nn.Sequential(
        nn.Linear(10_000, 10_000),
        nn.ReLU(),
        nn.Linear(10_000, 10_000),
        nn.ReLU(),
        nn.Linear(10_000, 10_000),
        nn.ReLU(),
        nn.Linear(10_000, 10_000),
        nn.ReLU(),
        nn.Linear(10_000, 10_000),
        nn.ReLU(),
        nn.Linear(10_000, 10_000),
    ).to(device)
    regular_optimizer = Adam(model.parameters(), lr=0.001)
    model, regular_optimizer, peak_memory_adam = train(
        model, regular_optimizer, device, is_zero1=False
    )

    # Clear memory before testing ZeRO-1 optimizer
    del model, regular_optimizer
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    dist.barrier()

    print(f"\nGPU {rank} - Testing with ZeRO-1 optimizer state sharding:")
    model = nn.Sequential(
        nn.Linear(10_000, 10_000),
        nn.ReLU(),
        nn.Linear(10_000, 10_000),
        nn.ReLU(),
        nn.Linear(10_000, 10_000),
        nn.ReLU(),
        nn.Linear(10_000, 10_000),
        nn.ReLU(),
        nn.Linear(10_000, 10_000),
        nn.ReLU(),
        nn.Linear(10_000, 10_000),
    ).to(device)
    base_optimizer = Adam(model.parameters(), lr=0.001)
    zero1_optimizer = ZeRO1Optimizer(base_optimizer)
    model, zero1_optimizer, peak_memory_z1 = train(
        model, zero1_optimizer, device, is_zero1=True
    )

    if rank == 0:
        print("\nMemory Usage Summary:")
        print("-" * 40)
        print(f"Peak memory with regular Adam: {peak_memory_adam:.2f} MB")
        print(f"Peak memory with ZeRO-1: {peak_memory_z1:.2f} MB")
        print(
            f"Memory reduction: {(peak_memory_adam - peak_memory_z1):.2f} MB "
            f"({((peak_memory_adam - peak_memory_z1) / peak_memory_adam * 100):.2f}%)"
        )


if __name__ == "__main__":
    test_zero1()