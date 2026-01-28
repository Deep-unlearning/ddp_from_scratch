"""
Single-GPU reference training script (NO Trainer).
Produces a golden run to validate custom DDP against.
"""

import os
import json
import time
import random
import hashlib
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import wandb

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)

# -----------------
# Config
# -----------------
MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-instruct"
DATASET_NAME = "b-mc2/sql-create-context"

NUM_EPOCHS = 3
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 8
MAX_LENGTH = 512

LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.03

LOG_EVERY = 10
SAVE_DIR = "runs/single_gpu_baseline"
SEED = 42

USE_AMP = True
DTYPE = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    else torch.float16
)

# -----------------
# Determinism
# -----------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------
# Diagnostics
# -----------------
def tensor_checksum(model: torch.nn.Module) -> str:
    """Stable-ish checksum of model parameters."""
    h = hashlib.sha256()
    with torch.no_grad():
        for p in model.parameters():
            t = p.detach().float().cpu().contiguous().view(-1)
            h.update(t.numpy().tobytes())
    return h.hexdigest()


def global_l2_norm(tensors) -> float:
    """Global L2 norm over a list of tensors."""
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


# -----------------
# Tokenization
# -----------------
def make_text(example):
    parts = []
    for k in ["context", "question", "answer"]:
        if k in example and example[k] is not None:
            parts.append(str(example[k]))
    return "\n".join(parts)


@dataclass
class CausalLMCollator:
    tokenizer: AutoTokenizer
    max_length: int

    def __call__(self, batch):
        texts = [make_text(ex) for ex in batch]
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        enc["labels"] = enc["input_ids"].clone()
        return enc


# -----------------
# Checkpointing
# -----------------
def save_checkpoint(path, model, optimizer, scheduler, step, epoch):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "epoch": epoch,
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None,
            },
        },
        path,
    )


# -----------------
# Main
# -----------------
def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(
        project="smollm2-360m-instruct",
        name="single_gpu_baseline",
        config=dict(
            model=MODEL_NAME,
            batch_size=BATCH_SIZE,
            grad_accum_steps=GRAD_ACCUM_STEPS,
            max_length=MAX_LENGTH,
            lr=LR,
            dtype=str(DTYPE),
        ),
    )

    os.makedirs(SAVE_DIR, exist_ok=True)

    dataset = load_dataset(DATASET_NAME, split="train")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE if device.type == "cuda" else None,
    ).to(device)
    model.train()

    collator = CausalLMCollator(tokenizer, MAX_LENGTH)

    g = torch.Generator()
    g.manual_seed(SEED)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=g,
        num_workers=0,
        collate_fn=collator,
        pin_memory=device.type == "cuda",
    )

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    steps_per_epoch = (len(loader) + GRAD_ACCUM_STEPS - 1) // GRAD_ACCUM_STEPS
    total_steps = steps_per_epoch * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        warmup_steps,
        total_steps,
    )

    use_fp16 = USE_AMP and DTYPE == torch.float16 and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    optimizer.zero_grad(set_to_none=True)

    step = 0
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        for it, batch in enumerate(loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            with torch.cuda.amp.autocast(
                enabled=USE_AMP and device.type == "cuda", dtype=DTYPE
            ):
                out = model(**batch)
                loss = out.loss / GRAD_ACCUM_STEPS

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (it + 1) % GRAD_ACCUM_STEPS == 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                grad_norm = global_l2_norm(
                    p.grad for p in model.parameters() if p.grad is not None
                )
                param_norm = global_l2_norm(p.data for p in model.parameters())

                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                if device.type == "cuda":
                    torch.cuda.synchronize()
                    max_mem = int(torch.cuda.max_memory_allocated())
                else:
                    max_mem = 0

                record = {
                    "step": step,
                    "epoch": epoch,
                    "epoch_progress": f"{step / steps_per_epoch:.2f}",
                    "loss": float(loss.item() * GRAD_ACCUM_STEPS),
                    "grad_norm": grad_norm,
                    "param_norm": param_norm,
                    "lr": optimizer.param_groups[0]["lr"],
                    "max_mem_bytes": max_mem,
                    "time_s": time.time() - start_time,
                }

                if step % LOG_EVERY == 0:
                    print(json.dumps(record))
                    wandb.log({
                        "loss": record["loss"],
                        "grad_norm": record["grad_norm"],
                        "param_norm": record["param_norm"],
                        "lr": record["lr"],
                        "max_mem_bytes": record["max_mem_bytes"],
                        "time_s": record["time_s"],
                    })

                if step in (5, 20, 100):
                    save_checkpoint(
                        f"{SAVE_DIR}/ckpt_step_{step}.pt",
                        model,
                        optimizer,
                        scheduler,
                        step,
                        epoch,
                    )

                step += 1

    final_ckpt = f"{SAVE_DIR}/final.pt"
    save_checkpoint(final_ckpt, model, optimizer, scheduler, step, epoch)

    checksum = tensor_checksum(model)
    with open(f"{SAVE_DIR}/golden_metrics.json", "w") as f:
        json.dump({"final_checksum": checksum}, f, indent=2)

    wandb.finish()

    print("Training complete.")
    print("Final checksum:", checksum)
    print("Saved to:", SAVE_DIR)


if __name__ == "__main__":
    main()
