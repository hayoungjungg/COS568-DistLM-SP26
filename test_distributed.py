#!/usr/bin/env python3
"""
Test script to verify PyTorch distributed process group initialization on multiple nodes.
Run with torchrun to launch 4 processes (e.g., 4 nodes with 1 process each, or 1 node with 4 processes).

Example for 4 nodes (run on each node; set MASTER_ADDR to the master node's IP):
  Node 0: torchrun --nnodes=4 --node_rank=0 --nproc_per_node=1 --master_addr=<MASTER_IP> --master_port=29500 test_distributed.py
  Node 1: torchrun --nnodes=4 --node_rank=1 --nproc_per_node=1 --master_addr=<MASTER_IP> --master_port=29500 test_distributed.py
  Node 2: torchrun --nnodes=4 --node_rank=2 --nproc_per_node=1 --master_addr=<MASTER_IP> --master_port=29500 test_distributed.py
  Node 3: torchrun --nnodes=4 --node_rank=3 --nproc_per_node=1 --master_addr=<MASTER_IP> --master_port=29500 test_distributed.py

Example for 1 node with 4 processes:
  torchrun --nnodes=1 --nproc_per_node=4 test_distributed.py
"""

import os
import torch
import torch.distributed as dist


def main():
    # LOCAL_RANK, RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT are set by torchrun
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank == -1:
        print("Not running in distributed mode (LOCAL_RANK not set). Use torchrun to launch.")
        return

    # CPU only: gloo backend
    backend = "gloo"
    dist.init_process_group(backend=backend, init_method="env://")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print(f"[Rank {rank}] init_process_group OK: world_size={world_size}, backend={backend}, local_rank={local_rank}")

    # Synchronize all processes
    dist.barrier()

    # Optional: simple all_reduce to verify communication (CPU)
    t = torch.tensor([float(rank)], device="cpu")
    dist.all_reduce(t)
    expected_sum = sum(range(world_size))
    if rank == 0:
        print(f"[Rank 0] all_reduce test: sum of ranks = {t.item()} (expected {expected_sum})")

    dist.barrier()
    dist.destroy_process_group()
    print(f"[Rank {rank}] Process group destroyed. Test passed.")


if __name__ == "__main__":
    main()
