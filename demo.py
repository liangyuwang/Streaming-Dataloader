import os
import math
import argparse
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from dataset import DistributedDataset
from tqdm.auto import tqdm


def get_args():
    parser = argparse.ArgumentParser(description="Demo script for DistributedDataset")
    parser.add_argument("--data_path", type=str, default="./data/fineweb-edu-sample-10BT")
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def setup_distributed():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def get_training_info(args, dataset, dp_rank, dp_world_size, num_workers):
    rem = dataset.total_samples - dataset.global_skip_batches
    if rem <= 0:
        return 0, 0

    total_streams = dp_world_size * num_workers
    local_samples = 0

    for worker_id in range(num_workers):
        global_stream_id = dp_rank * num_workers + worker_id
        if global_stream_id >= rem:
            continue
        n = (rem - 1 - global_stream_id) // total_streams + 1
        local_samples += n

    local_num_batches = math.ceil(local_samples / args.batch_size)

    return local_samples, local_num_batches


def main():
    args = get_args()
    rank, local_rank, world_size = setup_distributed()
    
    dataset = DistributedDataset(
        data_dir=args.data_path,
        seq_len=args.seq_len,
        shuffle=True,
        seed=123,
        dp_rank=rank,
        dp_world_size=world_size,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,  # IterableDataset handles its own ordering
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    for epoch in range(10):
        dataloader.dataset.set_epoch(epoch)
        local_samples, local_num_batches = get_training_info(
            args, 
            dataset, 
            dp_rank=rank,
            dp_world_size=world_size,
            num_workers=args.num_workers,
        )
        for batch in tqdm(dataloader, total=local_num_batches, desc=f"Epoch {epoch}", disable=(rank != 0)):
            input_ids, labels = batch["input_ids"], batch["labels"]
            # input_ids = input_ids.cuda(local_rank, non_blocking=True)
            # labels = labels.cuda(local_rank, non_blocking=True)

    cleanup_distributed()


if __name__ == "__main__":
    main()