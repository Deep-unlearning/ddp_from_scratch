"""
Gradient Bucketing for DDP - Step 4

This module implements gradient bucketing with autograd hooks
to overlap backward computation with gradient communication.
"""

import torch
import torch.distributed as dist
from dataclasses import dataclass, field
from typing import Any, Optional
from contextlib import contextmanager

from .distributed import get_world_size

@dataclass
class Bucket:
    """A bucket holds multiple parameter gradients in a flat buffer."""
    buffer: torch.Tensor                          # Flat tensor for gradients
    params: list = field(default_factory=list)    # Parameters in this bucket
    offsets: dict = field(default_factory=dict)   # param -> (start, end) indices
    num_params_ready: int = 0
    async_handle: Optional[Any] = None
    
    @property
    def num_params_expected(self) -> int:
        return len(self.params)
    
    def is_complete(self) -> bool:
        return self.num_params_ready == self.num_params_expected


class GradientBucketer:
    """
    Manages gradient bucketing and async all-reduce for DDP.
    
    Usage:
        bucketer = GradientBucketer(model, bucket_size_mb=25.0)
        
        # In training loop:
        loss.backward()                    # Hooks fire automatically
        bucketer.wait_for_all_reduces()    # Wait before optimizer step
        optimizer.step()
        bucketer.reset()                   # Reset for next iteration
    """
    
    def __init__(self, model: torch.nn.Module, bucket_size_mb: float = 25.0):
        self.world_size = get_world_size()
        self.buckets: list[Bucket] = []
        self.param_to_bucket_idx: dict = {}  # Maps param -> bucket index
        self._hooks = []  # Store hook handles for cleanup
        self.sync_enabled = True
        
        self._build_buckets(model, bucket_size_mb)
        self._register_hooks(model)
    
    def _build_buckets(self, model: torch.nn.Module, bucket_size_mb: float):
        """
        Create buckets by iterating parameters in REVERSE order.
        
        Algorithm:
        1. Iterate through model.parameters() in REVERSE order
        2. Add parameters to current bucket until size exceeds bucket_size_mb
        3. When bucket is "full", finalize it and start a new bucket
        4. Don't forget to finalize the last bucket!
        
        "Finalizing" a bucket means:
        - Create the flat buffer tensor (sum of all param sizes)
        - Record the offset (start, end) for each param in the buffer
        - Create the Bucket object and append to self.buckets
        - Update self.param_to_bucket_idx for each param
        """
        bucket_size_bytes = bucket_size_mb * 1024 * 1024
        
        current_bucket_params = []
        current_bucket_size = 0
        
        # Get device and dtype from first parameter
        first_param = next(model.parameters())
        device = first_param.device
        dtype = first_param.dtype
        
        # Iterate parameters in reverse order (last layers first)
        params = list(model.parameters())
        
        for param in reversed(params):
            if not param.requires_grad:
                continue
                
            param_size = param.numel() * param.element_size()  # size in bytes
            
            if current_bucket_size + param_size > bucket_size_bytes and len(current_bucket_params) > 0:
                self._finalize_bucket(current_bucket_params, device, dtype)
                current_bucket_params = []
                current_bucket_size = 0
            
            current_bucket_params.append(param)
            current_bucket_size += param_size
        

        if current_bucket_params:
            self._finalize_bucket(current_bucket_params, device, dtype)
        
        print(f"[GradientBucketer] Created {len(self.buckets)} buckets")
    
    def _finalize_bucket(self, params: list, device, dtype):
        """
        Helper: Create a Bucket object from a list of parameters.
        
        This method:
        1. Calculates total buffer size
        2. Creates the flat buffer tensor
        3. Records offsets for each parameter
        4. Creates and stores the Bucket object
        5. Updates param_to_bucket_idx mapping
        """
        if not params:
            return
        
        # Calculate total size and offsets
        total_numel = sum(p.numel() for p in params)
        buffer = torch.zeros(total_numel, device=device, dtype=dtype)
        
        offsets = {}
        offset = 0
        for p in params:
            offsets[p] = (offset, offset + p.numel())
            offset += p.numel()
        
        # Create bucket
        bucket = Bucket(
            buffer=buffer,
            params=list(params),
            offsets=offsets,
        )
        
        # Record bucket index for each param
        bucket_idx = len(self.buckets)
        for p in params:
            self.param_to_bucket_idx[p] = bucket_idx
        
        self.buckets.append(bucket)
    
    def _register_hooks(self, model: torch.nn.Module):
        """
        Register a backward hook on each parameter.
        
        Use param.register_post_accumulate_grad_hook() which fires
        AFTER the gradient is accumulated (important for gradient accumulation).
        
        The hook should call self._on_grad_ready(param) when triggered.
        
        Hint: The hook function signature is: hook(param) -> None
        """
        for param in model.parameters():
            if not param.requires_grad:
                continue

            hook = param.register_post_accumulate_grad_hook(self._on_grad_ready)
            self._hooks.append(hook)
    
    @contextmanager
    def no_sync(self):
        self.sync_enabled = False
        try:
            yield
        finally:
            self.sync_enabled = True

    def _on_grad_ready(self, param: torch.nn.Parameter):
        """
        Called when a parameter's gradient is computed.
        
        Steps:
        1. Find which bucket this param belongs to
        2. Copy the gradient into the bucket's buffer at the correct offset
        3. Increment the bucket's num_params_ready counter
        4. If bucket is complete, trigger async all-reduce
        
        For the all-reduce:
        - Use dist.all_reduce(bucket.buffer, op=dist.ReduceOp.SUM, async_op=True)
        - Store the returned handle in bucket.async_handle
        - Remember: we're summing, so we'll need to divide by world_size later
        """
        if not self.sync_enabled:
            return
        bucket_idx = self.param_to_bucket_idx[param]
        bucket = self.buckets[bucket_idx]
        
        start, end = bucket.offsets[param]
        bucket.buffer[start:end] = param.grad.flatten()

        bucket.num_params_ready += 1
        if bucket.is_complete():
            bucket.async_handle = dist.all_reduce(bucket.buffer, op=dist.ReduceOp.SUM, async_op=True)
    
    def wait_for_all_reduces(self):
        """
        Wait for all async all-reduce operations to complete,
        then copy gradients back from buckets to parameters.
        
        Steps:
        1. For each bucket with a pending async_handle, call handle.wait()
        2. Divide the buffer by world_size (we summed, need average)
        3. Copy gradients back from buffer to each param.grad
        """
        for bucket in self.buckets:

            if bucket.async_handle is not None:
                bucket.async_handle.wait()
            
            bucket.buffer /= self.world_size

            for param in bucket.params:
                start, end = bucket.offsets[param]
                param.grad = bucket.buffer[start:end].view_as(param.grad)
    
    def reset(self):
        """
        Reset state for the next backward pass.
        
        What needs to be reset?
        - Each bucket's num_params_ready -> 0
        - Each bucket's async_handle -> None
        """
        for bucket in self.buckets:

            bucket.num_params_ready = 0
            bucket.async_handle = None
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
