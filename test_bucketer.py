"""
Test the GradientBucketer against naive sync_gradients.

Usage:
    torchrun --nproc_per_node=2 test_bucketer.py
"""

import torch
import torch.distributed as dist
import copy

from ddp import (
    setup_distributed,
    cleanup_distributed,
    get_rank,
    get_world_size,
    broadcast_model,
    sync_gradients,
    GradientBucketer,
)


def create_test_model():
    """Create a small test model."""
    return torch.nn.Sequential(
        torch.nn.Linear(64, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 8),
    )


def test_bucketer_vs_naive():
    """
    Test that GradientBucketer produces the same averaged gradients
    as the naive sync_gradients approach.
    """
    rank = get_rank()
    world_size = get_world_size()
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    
    print(f"[Rank {rank}] Running on {device}")
    
    # Create two identical models
    torch.manual_seed(42)
    model_naive = create_test_model().to(device)
    model_bucketed = copy.deepcopy(model_naive)
    
    # Broadcast to ensure all ranks start with same weights
    broadcast_model(model_naive)
    broadcast_model(model_bucketed)
    
    # Create bucketer for the bucketed model
    bucketer = GradientBucketer(model_bucketed, bucket_size_mb=0.001)  # Small buckets for testing
    
    # Create rank-specific input (different data on each rank)
    torch.manual_seed(42 + rank)  # Different seed per rank
    x = torch.randn(4, 64, device=device)
    target = torch.randn(4, 8, device=device)
    
    # -------------------------------------------------------------------------
    # Test 1: Forward/backward with NAIVE sync
    # -------------------------------------------------------------------------
    model_naive.zero_grad()
    output_naive = model_naive(x)
    loss_naive = torch.nn.functional.mse_loss(output_naive, target)
    loss_naive.backward()
    sync_gradients(model_naive, world_size)  # Your naive implementation
    
    # Save naive gradients
    naive_grads = {name: p.grad.clone() for name, p in model_naive.named_parameters()}
    
    # -------------------------------------------------------------------------
    # Test 2: Forward/backward with BUCKETED sync
    # -------------------------------------------------------------------------
    model_bucketed.zero_grad()
    output_bucketed = model_bucketed(x)
    loss_bucketed = torch.nn.functional.mse_loss(output_bucketed, target)
    loss_bucketed.backward()  # Hooks fire here!
    bucketer.wait_for_all_reduces()  # Wait and copy back
    bucketer.reset()  # Ready for next iteration
    
    # Save bucketed gradients
    bucketed_grads = {name: p.grad.clone() for name, p in model_bucketed.named_parameters()}
    
    # -------------------------------------------------------------------------
    # Compare gradients
    # -------------------------------------------------------------------------
    print(f"\n[Rank {rank}] Comparing gradients:")
    all_close = True
    
    for name in naive_grads:
        naive_g = naive_grads[name]
        bucketed_g = bucketed_grads[name]
        
        max_diff = (naive_g - bucketed_g).abs().max().item()
        is_close = torch.allclose(naive_g, bucketed_g, rtol=1e-5, atol=1e-6)
        
        status = "✓" if is_close else "✗"
        print(f"  {status} {name}: max_diff = {max_diff:.2e}")
        
        if not is_close:
            all_close = False
    
    # -------------------------------------------------------------------------
    # Final verdict
    # -------------------------------------------------------------------------
    print()
    if all_close:
        print(f"[Rank {rank}] ✓ SUCCESS: Bucketed gradients match naive gradients!")
    else:
        print(f"[Rank {rank}] ✗ FAILED: Gradients don't match!")
    
    # Cleanup
    bucketer.remove_hooks()
    
    return all_close


def test_bucket_creation():
    """Test that buckets are created correctly."""
    rank = get_rank()
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    
    torch.manual_seed(42)
    model = create_test_model().to(device)
    
    # Very small bucket size to force multiple buckets
    bucketer = GradientBucketer(model, bucket_size_mb=0.0001)
    
    print(f"\n[Rank {rank}] Bucket creation test:")
    print(f"  Number of buckets: {len(bucketer.buckets)}")
    
    for i, bucket in enumerate(bucketer.buckets):
        num_params = len(bucket.params)
        buffer_size = bucket.buffer.numel()
        print(f"  Bucket {i}: {num_params} params, {buffer_size} elements")
    
    # Verify all params are assigned to a bucket
    total_params_in_buckets = sum(len(b.params) for b in bucketer.buckets)
    total_model_params = sum(1 for p in model.parameters() if p.requires_grad)
    
    if total_params_in_buckets == total_model_params:
        print(f"  ✓ All {total_model_params} parameters assigned to buckets")
    else:
        print(f"  ✗ Mismatch: {total_params_in_buckets} in buckets vs {total_model_params} in model")
    
    bucketer.remove_hooks()


def main():
    setup_distributed()
    rank = get_rank()
    
    try:
        print(f"\n{'='*60}")
        print(f"[Rank {rank}] GRADIENT BUCKETER TEST")
        print(f"{'='*60}")
        
        # Test 1: Bucket creation
        test_bucket_creation()
        
        # Sync before next test
        dist.barrier()
        
        # Test 2: Compare with naive
        print(f"\n{'-'*60}")
        success = test_bucketer_vs_naive()
        
        dist.barrier()
        
        if rank == 0:
            print(f"\n{'='*60}")
            print("TEST COMPLETE")
            print(f"{'='*60}")
            
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
