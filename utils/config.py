"""Shared configuration for all training scripts."""

import torch

# -----------------
# Model & Data
# -----------------
MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-instruct"
DATASET_NAME = "b-mc2/sql-create-context"

# -----------------
# Training Hyperparameters
# -----------------
NUM_EPOCHS = 3
BATCH_SIZE_SINGLE_GPU = 8  # For single GPU training
BATCH_SIZE_DDP = 8         # Per-GPU batch size for DDP
GRAD_ACCUM_STEPS = 8
MAX_LENGTH = 512

LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.03
MAX_GRAD_NORM = 1.0

# -----------------
# Logging & Checkpointing
# -----------------
LOG_EVERY = 10
SEED = 42

# -----------------
# Mixed Precision
# -----------------
USE_AMP = True
DTYPE = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else torch.float16
)
