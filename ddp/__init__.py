"""DDP (Distributed Data Parallel) implementation from scratch."""

from .distributed import (
    get_rank,
    get_world_size,
    get_local_rank,
    is_main_process,
    setup_distributed,
    cleanup_distributed,
    broadcast_model,
    sync_gradients,
)
from .gradient_bucketer import GradientBucketer, Bucket
from .sampler import ShardSampler
