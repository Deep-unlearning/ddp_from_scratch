"""
Single-GPU reference training script (NO Trainer).
Produces a golden run to validate custom DDP against.
"""

import os
import json
import time

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

from utils import (
    MODEL_NAME, DATASET_NAME, NUM_EPOCHS, BATCH_SIZE_SINGLE_GPU as BATCH_SIZE,
    GRAD_ACCUM_STEPS, MAX_LENGTH, LR, WEIGHT_DECAY, WARMUP_RATIO,
    MAX_GRAD_NORM, LOG_EVERY, SEED, USE_AMP, DTYPE,
    set_seed, global_l2_norm,
    get_gpu_peak_tflops, estimate_flops_per_step, compute_mfu,
    CausalLMCollator,
)

# -----------------
# Config
# -----------------
SAVE_DIR = "runs/single_gpu_baseline"


# -----------------
# Checkpointing
# -----------------
def save_checkpoint(path, model, optimizer, scheduler, step, epoch):
    import random
    import numpy as np
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

    # Calculate model stats for MFU (formula from PyTorch blog)
    num_params = sum(p.numel() for p in model.parameters())
    gpu_peak_tflops = get_gpu_peak_tflops()
    gpu_peak_flops = gpu_peak_tflops * 1e12  # Convert TFLOPs to FLOPS
    flops_per_step = estimate_flops_per_step(
        num_params=num_params,
        seq_len=MAX_LENGTH,
        global_batch_size=BATCH_SIZE,
        grad_accum_steps=GRAD_ACCUM_STEPS,
    )
    print(f"Model parameters: {num_params:,}")
    print(f"Tokens per step: {BATCH_SIZE * GRAD_ACCUM_STEPS * MAX_LENGTH:,}")
    print(f"GPU peak TFLOPs: {gpu_peak_tflops}")
    print(f"Estimated FLOPs per step: {flops_per_step:.2e}")

    collator = CausalLMCollator(tokenizer, MAX_LENGTH)

    g = torch.Generator()
    g.manual_seed(SEED)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
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

    print(f"DEBUG: len(loader)={len(loader)}, steps_per_epoch={steps_per_epoch}, total_steps={total_steps}, warmup_steps={warmup_steps}")

    use_fp16 = USE_AMP and DTYPE == torch.float16 and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    optimizer.zero_grad(set_to_none=True)

    step = 0
    start_time = time.time()
    last_log_time = start_time
    last_log_step = 0

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

                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

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
                    }

                if step % LOG_EVERY == 0 and step > 0:
                    # Compute time per LOG_EVERY steps and MFU
                    current_time = time.time()
                    steps_since_last_log = step - last_log_step
                    time_per_step = (current_time - last_log_time) / steps_since_last_log if steps_since_last_log > 0 else 0
                    time_per_log_interval = current_time - last_log_time
                    
                    mfu = compute_mfu(
                        flops_per_step=flops_per_step,
                        step_time_seconds=time_per_step,
                        gpu_peak_flops=gpu_peak_flops,
                        num_devices=1,
                    )
                    
                        record["time_per_step"] = time_per_step
                        record["mfu"] = mfu
                        
                        last_log_time = current_time
                        last_log_step = step
                        
                        print(json.dumps(record))
                        wandb.log({
                            "loss": record["loss"],
                            "grad_norm": record["grad_norm"],
                            "param_norm": record["param_norm"],
                            "lr": record["lr"],
                            "max_mem_bytes": record["max_mem_bytes"],
                            "time_per_step": time_per_step,
                            "mfu": mfu,
                        }, step=step)
                elif step == 0:
                    # First step - just initialize timing
                    last_log_time = time.time()
                    last_log_step = 0
                    print(json.dumps(record))

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

    wandb.finish()

    print("Training complete.")
    print("Saved to:", SAVE_DIR)


if __name__ == "__main__":
    main()
