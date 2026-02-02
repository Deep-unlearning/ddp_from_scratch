import os
from datetime import timedelta

import torch
import torch.distributed as dist

from ddp import ShardSampler


def rank0_print(rank: int, *msg):
    if rank == 0:
        print(*msg, flush=True)


def main():
    # --- dist init ---
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=timedelta(seconds=60),
        device_id=device,
    )

    # --- config ---
    SEED = 123
    dataset_len = 103  # intentionally not divisible by world_size
    K = 50             # how many indices to test per rank per epoch
    drop_last = True   # set False later to observe padding behavior

    sampler = ShardSampler(
        dataset_len=dataset_len,
        rank=rank,
        world_size=world_size,
        shuffle=True,
        seed=SEED,
        drop_last=drop_last,
    )

    # --------------- Test A: per-epoch no-overlap ---------------
    for epoch in [0, 1]:
        sampler.set_epoch(epoch)

        # Collect first K indices from this rank
        it = iter(sampler)
        local = []
        for _ in range(min(K, len(sampler))):
            local.append(next(it))

        # Pad to fixed length so all_gather works easily
        pad_val = -1
        if len(local) < K:
            local = local + [pad_val] * (K - len(local))

        t = torch.tensor(local, device=device, dtype=torch.int64)

        gathered = [torch.empty_like(t) for _ in range(world_size)]
        dist.all_gather(gathered, t)

        if rank == 0:
            per_rank = [[int(x) for x in g.cpu().tolist() if x != pad_val] for g in gathered]

            # Check each rank produced same count (for drop_last=True, len(sampler) is equal)
            lens = [len(x) for x in per_rank]
            rank0_print(rank, f"[epoch={epoch}] per-rank counts (first K): {lens}")

            # Check no overlap
            sets = [set(x) for x in per_rank]
            ok = True
            for i in range(world_size):
                for j in range(i + 1, world_size):
                    inter = sets[i].intersection(sets[j])
                    if len(inter) != 0:
                        ok = False
                        rank0_print(rank, f"❌ Overlap between rank {i} and {j}: {sorted(list(inter))[:20]}")
            if ok:
                rank0_print(rank, f"✅ [epoch={epoch}] No overlap across ranks (for first {min(K, len(sampler))} samples).")

            # Show a small preview
            rank0_print(rank, f"rank0 indices preview: {per_rank[0][:10]}")

        dist.barrier()

    # --------------- Test B: determinism within same epoch ---------------
    # Re-create the sampler with same seed + epoch and verify first K indices match.
    sampler2 = ShardSampler(
        dataset_len=dataset_len,
        rank=rank,
        world_size=world_size,
        shuffle=True,
        seed=SEED,
        drop_last=drop_last,
    )
    epoch = 0
    sampler.set_epoch(epoch)
    sampler2.set_epoch(epoch)

    def first_k(s):
        it = iter(s)
        out = []
        for _ in range(min(K, len(s))):
            out.append(next(it))
        return out

    a = first_k(sampler)
    b = first_k(sampler2)

    same = int(a == b)
    same_t = torch.tensor([same], device=device, dtype=torch.int32)
    dist.all_reduce(same_t, op=dist.ReduceOp.MIN)

    if rank == 0:
        if int(same_t.item()) == 1:
            rank0_print(rank, "✅ Determinism test passed: same seed+epoch => identical per-rank index sequence.")
        else:
            rank0_print(rank, "❌ Determinism test failed on at least one rank.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
