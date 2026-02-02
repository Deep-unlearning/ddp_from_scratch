"""Utility modules for training."""

from .config import (
    MODEL_NAME,
    DATASET_NAME,
    NUM_EPOCHS,
    BATCH_SIZE_SINGLE_GPU,
    BATCH_SIZE_DDP,
    GRAD_ACCUM_STEPS,
    MAX_LENGTH,
    LR,
    WEIGHT_DECAY,
    WARMUP_RATIO,
    MAX_GRAD_NORM,
    LOG_EVERY,
    SEED,
    USE_AMP,
    DTYPE,
)
from .utils import (
    set_seed,
    global_l2_norm,
    get_gpu_peak_tflops,
    estimate_flops_per_step,
    compute_mfu,
)

__all__ = [
    # config
    "MODEL_NAME", "DATASET_NAME", "NUM_EPOCHS", "BATCH_SIZE_SINGLE_GPU",
    "BATCH_SIZE_DDP", "GRAD_ACCUM_STEPS", "MAX_LENGTH", "LR", "WEIGHT_DECAY",
    "WARMUP_RATIO", "MAX_GRAD_NORM", "LOG_EVERY", "SEED", "USE_AMP", "DTYPE",
    # utils
    "set_seed", "global_l2_norm",
    "get_gpu_peak_tflops", "estimate_flops_per_step", "compute_mfu",
    # data
    "CausalLMCollator",
]
from .data import CausalLMCollator
