"""
Compare model parameters: naive DDP vs PyTorch DDP with tolerance.

Usage:
    python compare_checkpoints.py
"""

import os
import torch


def load_model_state(path: str) -> dict:
    """Load model state dict from checkpoint."""
    if not os.path.exists(path):
        return None
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt["model"]


def compare_state_dicts(state1: dict, state2: dict, rtol: float = 1e-5, atol: float = 1e-6):
    """
    Compare two state dicts with tolerance.
    
    Returns:
        dict with max_diff, mean_diff, num_params, all_close (bool)
    """
    max_diff = 0.0
    total_diff = 0.0
    total_elements = 0
    mismatched_params = []
    
    keys1 = set(state1.keys())
    keys2 = set(state2.keys())
    
    if keys1 != keys2:
        print(f"Warning: Different keys! Only in 1: {keys1 - keys2}, Only in 2: {keys2 - keys1}")
    
    common_keys = keys1 & keys2
    
    for key in sorted(common_keys):
        t1 = state1[key].float()
        t2 = state2[key].float()
        
        diff = (t1 - t2).abs()
        param_max_diff = diff.max().item()
        param_mean_diff = diff.mean().item()
        
        max_diff = max(max_diff, param_max_diff)
        total_diff += diff.sum().item()
        total_elements += t1.numel()
        
        # Check if this param is close
        if not torch.allclose(t1, t2, rtol=rtol, atol=atol):
            mismatched_params.append((key, param_max_diff, param_mean_diff))
    
    mean_diff = total_diff / total_elements if total_elements > 0 else 0
    all_close = len(mismatched_params) == 0
    
    return {
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "num_params": len(common_keys),
        "num_elements": total_elements,
        "all_close": all_close,
        "mismatched_params": mismatched_params[:5],  # Show first 5
    }


def main():
    runs_dir = "runs"
    steps = [5, 20, 100, "final"]
    
    rtol = 1e-5  # relative tolerance
    atol = 1e-6  # absolute tolerance
    
    print("=" * 70)
    print(f"Comparing: ddp_naive vs ddp_torch (rtol={rtol}, atol={atol})")
    print("=" * 70)
    
    for step in steps:
        if step == "final":
            naive_path = f"{runs_dir}/ddp_naive/final.pt"
            torch_path = f"{runs_dir}/ddp_torch/final.pt"
        else:
            naive_path = f"{runs_dir}/ddp_naive/ckpt_step_{step}.pt"
            torch_path = f"{runs_dir}/ddp_torch/ckpt_step_{step}.pt"
        
        naive_state = load_model_state(naive_path)
        torch_state = load_model_state(torch_path)
        
        print(f"\nStep {step}:")
        
        if naive_state is None:
            print(f"  ddp_naive: NOT FOUND")
            continue
        if torch_state is None:
            print(f"  ddp_torch: NOT FOUND")
            continue
        
        result = compare_state_dicts(naive_state, torch_state, rtol=rtol, atol=atol)
        
        print(f"  Max diff:  {result['max_diff']:.2e}")
        print(f"  Mean diff: {result['mean_diff']:.2e}")
        
        if result["all_close"]:
            print(f"  ✓ MATCH (within tolerance)")
        else:
            print(f"  ✗ MISMATCH ({len(result['mismatched_params'])} params exceed tolerance)")
            for name, max_d, mean_d in result["mismatched_params"]:
                print(f"    - {name}: max={max_d:.2e}, mean={mean_d:.2e}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
