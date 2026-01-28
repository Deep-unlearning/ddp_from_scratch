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

## Step 2 — Sharded Data Loading ✅

* [x] Deterministic dataset sharding per rank
* [x] Same number of steps per rank
* [x] Per-epoch seed control
* [x] Verify no sample overlap across ranks

---

## Step 3 — Naive Data Parallelism (Correctness)

* [ ] Broadcast model parameters from rank 0
* [ ] Backward pass on each rank
* [ ] All-reduce gradients
* [ ] Divide gradients by `world_size`
* [ ] Optimizer step after synchronization
* [ ] Checksum matches single-GPU baseline

---

## Step 4 — NCCL Backend

* [ ] Use NCCL process group
* [ ] GPU all-reduce for gradients
* [ ] Explicit synchronization points
* [ ] NCCL debug logging enabled

---

## Step 5 — Autograd Hooks + Bucketing (Real DDP)

* [ ] Register backward hooks per parameter
* [ ] Gradient bucketing
* [ ] Async all-reduce per bucket
* [ ] Overlap backward compute with communication
* [ ] Single sync point before optimizer step

---

## Step 6 — Correctness Harness

* [ ] Per-rank parameter checksum
* [ ] Gradient norm scaling validation
* [ ] Single-GPU vs DDP loss curve comparison
* [ ] Resume-from-checkpoint equivalence

---

## Step 7 — Performance Tuning

* [ ] Bucket size tuning
* [ ] Gradient accumulation without early sync
* [ ] Mixed precision stability tests
* [ ] Throughput vs GPU count benchmark

---

## Step 8 — Polish / Extras

* [ ] Clean shutdown & error handling
* [ ] Timeout / deadlock detection
* [ ] Optional: unused parameter detection
* [ ] Optional: ZeRO / optimizer sharding
* [ ] Optional: gradient compression

---

### Definition of “Done”

* DDP run matches single-GPU golden run (within tolerance)
* Near-linear scaling on multiple GPUs
