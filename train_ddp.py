"""
Naive DDP training script — Step 3
Implements manual broadcast and gradient synchronization.
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
    MODEL_NAME, DATASET_NAME, NUM_EPOCHS, BATCH_SIZE_DDP as BATCH_SIZE,
    GRAD_ACCUM_STEPS, MAX_LENGTH, LR, WEIGHT_DECAY, WARMUP_RATIO,
    MAX_GRAD_NORM, LOG_EVERY, SEED, USE_AMP, DTYPE,
    set_seed, global_l2_norm,
    get_gpu_peak_tflops, estimate_flops_per_step, compute_mfu,
    CausalLMCollator,
)
from ddp import (
    get_rank, get_world_size, get_local_rank, is_main_process,
    setup_distributed, cleanup_distributed,
    broadcast_model, sync_gradients,
    ShardSampler, GradientBucketer,
)
import torch.distributed as dist

# -----------------
# Config
# -----------------
SAVE_DIR = "runs/ddp_naive"


# -----------------
# Checkpointing (only rank 0 saves)
# -----------------
def save_checkpoint(path, model, optimizer, scheduler, step, epoch):
    if not is_main_process():
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "epoch": epoch,
        },
        path,
    )


# -----------------
# Main
# -----------------
def main():
    # === 1. Setup distributed ===
    setup_distributed()
    
    rank = get_rank()
    world_size = get_world_size()
    local_rank = get_local_rank()
    
    # Each process uses a different GPU
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    
    if is_main_process():
        print(f"Starting DDP training with {world_size} processes")
        wandb.init(
            project="smollm2-360m-instruct",
            name="ddp_bucket_with_gradacc",
            config=dict(
                model=MODEL_NAME,
                batch_size=BATCH_SIZE,
                grad_accum_steps=GRAD_ACCUM_STEPS,
                max_length=MAX_LENGTH,
                lr=LR,
                dtype=str(DTYPE),
                world_size=world_size,
            ),
        )
    
    # === 2. Seed (same base seed for model init) ===
    set_seed(SEED, rank=0)
    
    # === 3. Load model ===
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=DTYPE if device.type == "cuda" else None,
    ).to(device)
    model.train()
    
    # === 4. Broadcast model weights from rank 0 ===
    broadcast_model(model)
    bucketer = GradientBucketer(model, bucket_size_mb=25)  # Small buckets for testing

    # Calculate model stats for MFU (formula from PyTorch blog)
    num_params = sum(p.numel() for p in model.parameters())
    gpu_peak_tflops = get_gpu_peak_tflops()
    gpu_peak_flops = gpu_peak_tflops * 1e12  # Convert TFLOPs to FLOPS
    global_batch_size = BATCH_SIZE * world_size
    flops_per_step = estimate_flops_per_step(
        num_params=num_params,
        seq_len=MAX_LENGTH,
        global_batch_size=global_batch_size,
        grad_accum_steps=GRAD_ACCUM_STEPS,
    )
    
    if is_main_process():
        print(f"Model parameters: {num_params:,}")
        print(f"Global batch size: {global_batch_size} (per step: {global_batch_size * GRAD_ACCUM_STEPS})")
        print(f"Tokens per step: {global_batch_size * GRAD_ACCUM_STEPS * MAX_LENGTH:,}")
        print(f"GPU peak TFLOPs: {gpu_peak_tflops} (x{world_size} GPUs)")
        print(f"Estimated FLOPs per step: {flops_per_step:.2e}")
    
    # === 5. Dataset and DataLoader with ShardSampler ===
    dataset = load_dataset(DATASET_NAME, split="train")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    collator = CausalLMCollator(tokenizer, MAX_LENGTH)
    
    sampler = ShardSampler(
        dataset_len=len(dataset),
        rank=rank,
        world_size=world_size,
        shuffle=False,
        seed=SEED,
        drop_last=True,
    )
    
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=0,
        collate_fn=collator,
        pin_memory=device.type == "cuda",
    )
    
    # === 6. Optimizer and Scheduler ===
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    steps_per_epoch = (len(loader) + GRAD_ACCUM_STEPS - 1) // GRAD_ACCUM_STEPS
    total_steps = steps_per_epoch * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    if is_main_process():
        print(f"DEBUG: len(loader)={len(loader)}, steps_per_epoch={steps_per_epoch}, total_steps={total_steps}, warmup_steps={warmup_steps}")
    
    # === 7. AMP setup ===
    use_fp16 = USE_AMP and DTYPE == torch.float16 and device.type == "cuda"
    scaler = torch.amp.GradScaler('cuda', enabled=use_fp16)
    
    optimizer.zero_grad(set_to_none=True)
    
    # === 8. Training Loop ===
    step = 0
    start_time = time.time()
    last_log_time = start_time
    last_log_step = 0
    
    for epoch in range(NUM_EPOCHS):
        sampler.set_epoch(epoch)
        
        for it, batch in enumerate(loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            
            with torch.amp.autocast(
                'cuda', enabled=USE_AMP and device.type == "cuda", dtype=DTYPE
            ):
                out = model(**batch)
                loss = out.loss / GRAD_ACCUM_STEPS
            
            if (it + 1) % GRAD_ACCUM_STEPS != 0:
                with bucketer.no_sync():
                    if scaler.is_enabled():
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
            else:
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                        
            # Gradient accumulation boundary
            if (it + 1) % GRAD_ACCUM_STEPS == 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                
                bucketer.wait_for_all_reduces()
                bucketer.reset()
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
                
                # Logging (only rank 0)
                if is_main_process():
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                        max_mem = int(torch.cuda.max_memory_allocated())
                    else:
                        max_mem = 0
                    
                    record = {
                        "step": step,
                        "epoch": epoch,
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
                            num_devices=world_size,
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
                            model, optimizer, scheduler, step, epoch,
                        )
                step += 1
            
            dist.barrier()
    
    # === 9. Final checkpoint and checksum ===
    if is_main_process():
        os.makedirs(SAVE_DIR, exist_ok=True)
        save_checkpoint(
            f"{SAVE_DIR}/final.pt",
            model, optimizer, scheduler, step, epoch,
        )
        
        print("Training complete.")
        wandb.finish()
    dist.barrier()
    # === 10. Cleanup ===
    cleanup_distributed()


if __name__ == "__main__":
    main()
