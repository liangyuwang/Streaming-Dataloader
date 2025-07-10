import os
import argparse
import psutil
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from dataset import SlidingTokenDataset
from tqdm.auto import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data/fineweb-edu-sample-10BT")
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()
    return args

def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_world_size()

def cleanup_distributed():
    dist.destroy_process_group()

def is_ddp_mode():
    return dist.is_available() and dist.is_initialized()

def main():
    args = get_args()
    dataset = SlidingTokenDataset(dataset_path=args.data_path, split="train", split_rate=1.0, batch_size=args.batch_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    for epoch in range(10):
        dataloader.dataset.set_epoch(epoch)
        for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
            input_ids, labels = batch["input_ids"], batch["labels"]

def main_ddp():
    args = get_args()
    rank, world_size = setup_distributed()

    dataset = SlidingTokenDataset(
        dataset_path=args.data_path,
        seq_len=args.seq_len,
        stride=args.stride,
        batch_size=args.batch_size,
        rank=rank,
        world_size=world_size
    )

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )

    for epoch in range(10):
        if rank == 0:
            print("Epoch: ", epoch)
        dataloader.dataset.set_epoch(epoch)
        for batch in tqdm(dataloader, desc=f"Epoch {epoch}", disable=(rank != 0)):
            input_ids, labels = batch["input_ids"], batch["labels"]

    cleanup_distributed()

if __name__ == "__main__":
    if not is_ddp_mode():
        main()
    else:
        main_ddp()