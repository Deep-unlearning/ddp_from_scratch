# Distributed Training From Scratch — Roadmap Checklist

## Step 0 — Single-GPU Baseline ✅

* [x] Manual training loop (no `Trainer`)
* [x] Deterministic seed setup
* [x] Fixed tokenizer + max sequence length
* [x] Gradient accumulation
* [x] AMP (bf16 / fp16)
* [x] Gradient clipping
* [x] LR scheduler
* [x] Checkpoint + resume
* [x] Log loss, grad norm, param norm, LR, GPU memory

---

## Step 1 — PyTorch DDP Golden Baseline ✅

* [x] Manual training loop (no `Trainer`)
* [x] Deterministic seed setup
* [x] Fixed tokenizer + max sequence length
* [x] Gradient accumulation
* [x] AMP (bf16 / fp16)
* [x] Gradient clipping
* [x] LR scheduler
* [x] Checkpoint + resume
* [x] Log loss, grad norm, param norm, LR, GPU memory
* [x] PyTorch DDP reference run saved for comparison

---

## Step 2 — Sharded Data Loading ✅

* [x] Deterministic dataset sharding per rank
* [x] Same number of steps per rank
* [x] Per-epoch seed control
* [x] Verify no sample overlap across ranks

---

## Step 3 — Naive Data Parallelism (Correctness)

* [x] Broadcast model parameters from rank 0
* [x] Backward pass on each rank
* [x] All-reduce gradients
* [x] Divide gradients by `world_size`
* [x] Optimizer step after synchronization
* [x] Weights match PyTorch DDP within tolerance (rtol=1e-5, atol=1e-6)
* [x] param_norm diff < 1e-6

---

## Step 4 — Autograd Hooks + Bucketing (Real DDP)

* [ ] Register backward hooks per parameter
* [ ] Gradient bucketing
* [ ] Async all-reduce per bucket
* [ ] Overlap backward compute with communication
* [ ] Single sync point before optimizer step

---

## Step 5 — Correctness Harness

* [ ] Per-rank parameter checksum
* [ ] Gradient norm scaling validation
* [ ] Single-GPU vs DDP loss curve comparison
* [ ] Resume-from-checkpoint equivalence

---

## Step 6 — Performance Tuning

* [ ] Bucket size tuning
* [ ] Gradient accumulation without early sync
* [ ] Mixed precision stability tests
* [ ] Throughput vs GPU count benchmark

---

## Step 7 — Polish / Extras

* [ ] Clean shutdown & error handling
* [ ] Timeout / deadlock detection
* [ ] Optional: unused parameter detection
* [ ] Optional: ZeRO / optimizer sharding
* [ ] Optional: gradient compression

---

### Definition of “Done”

* DDP run matches single-GPU golden run (within tolerance)
* Near-linear scaling on multiple GPUs
