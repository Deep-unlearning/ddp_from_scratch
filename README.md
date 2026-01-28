# Distributed Training From Scratch — Roadmap Checklist

## Step 1 — Single-GPU Golden Baseline ✅

* [x] Manual training loop (no `Trainer`)
* [x] Deterministic seed setup
* [x] Fixed tokenizer + max sequence length
* [x] Gradient accumulation
* [x] AMP (bf16 / fp16)
* [x] Gradient clipping
* [x] LR scheduler
* [x] Checkpoint + resume
* [x] Log loss, grad norm, param norm, LR, GPU memory
* [x] Final parameter checksum (SHA-256)
* [x] “Golden run” saved for later comparison

---

## Step 2 — Multi-Process Skeleton

* [ ] Spawn N processes
* [ ] Assign `rank`, `local_rank`, `world_size`
* [ ] One GPU per process
* [ ] Initialize process group
* [ ] Rank-0 logging only
* [ ] Global barrier utility

---

## Step 3 — Sharded Data Loading

* [ ] Deterministic dataset sharding per rank
* [ ] Same number of steps per rank
* [ ] Per-epoch seed control
* [ ] Verify no sample overlap across ranks

---

## Step 4 — Naive Data Parallelism (Correctness)

* [ ] Broadcast model parameters from rank 0
* [ ] Backward pass on each rank
* [ ] All-reduce gradients
* [ ] Divide gradients by `world_size`
* [ ] Optimizer step after synchronization
* [ ] Checksum matches single-GPU baseline

---

## Step 5 — NCCL Backend

* [ ] Use NCCL process group
* [ ] GPU all-reduce for gradients
* [ ] Explicit synchronization points
* [ ] NCCL debug logging enabled

---

## Step 6 — Autograd Hooks + Bucketing (Real DDP)

* [ ] Register backward hooks per parameter
* [ ] Gradient bucketing
* [ ] Async all-reduce per bucket
* [ ] Overlap backward compute with communication
* [ ] Single sync point before optimizer step

---

## Step 7 — Correctness Harness

* [ ] Per-rank parameter checksum
* [ ] Gradient norm scaling validation
* [ ] Single-GPU vs DDP loss curve comparison
* [ ] Resume-from-checkpoint equivalence

---

## Step 8 — Performance Tuning

* [ ] Bucket size tuning
* [ ] Gradient accumulation without early sync
* [ ] Mixed precision stability tests
* [ ] Throughput vs GPU count benchmark

---

## Step 9 — Polish / Extras

* [ ] Clean shutdown & error handling
* [ ] Timeout / deadlock detection
* [ ] Optional: unused parameter detection
* [ ] Optional: ZeRO / optimizer sharding
* [ ] Optional: gradient compression

---

### Definition of “Done”

* DDP run matches single-GPU golden run (within tolerance)
* Near-linear scaling on multiple GPUs
