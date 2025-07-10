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
    """
    Parse command line arguments for the demo script.
    """
    parser = argparse.ArgumentParser(description="Demo script for Streaming-Dataloader")
    parser.add_argument("--data_path", type=str, default="./data/fineweb-edu-sample-10BT",
                       help="Path to the preprocessed dataset directory")
    parser.add_argument("--seq_len", type=int, default=1024,
                       help="Sequence length for each sample")
    parser.add_argument("--stride", type=int, default=512,
                       help="Sliding window stride (how much to move window each step)")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=0,
                       help="Number of DataLoader workers for parallel data loading")
    args = parser.parse_args()
    return args

def setup_distributed():
    """
    Initialize distributed training using NCCL backend.
    
    This function sets up the process group and configures the CUDA device
    for the current process based on the LOCAL_RANK environment variable.
    
    Returns:
        tuple: (local_rank, world_size) for this process
    """
    # Initialize the process group for distributed training
    dist.init_process_group(backend="nccl")
    
    # Get local rank from environment variable set by torchrun
    local_rank = int(os.environ["LOCAL_RANK"])
    
    # Set the CUDA device for this process
    torch.cuda.set_device(local_rank)
    
    return local_rank, dist.get_world_size()

def cleanup_distributed():
    """
    Clean up the distributed process group.
    Should be called at the end of distributed training.
    """
    dist.destroy_process_group()

def is_ddp_mode():
    """
    Check if we're running in distributed data parallel mode.
    
    Returns:
        bool: True if distributed training is active, False otherwise
    """
    return dist.is_available() and dist.is_initialized()

def main():
    """
    Main training loop for single GPU training.
    
    This function demonstrates basic usage of the SlidingTokenDataset
    without distributed training. It shows how to:
    1. Create the dataset
    2. Set up the DataLoader
    3. Iterate through epochs and batches
    4. Handle the epoch-specific randomization
    """
    args = get_args()
    
    # Create the streaming dataset
    # Note: rank and world_size default to 0 and 1 for single GPU training
    dataset = SlidingTokenDataset(
        dataset_path=args.data_path, 
        split="train", 
        split_rate=1.0,  # Use all data for training in this demo
        batch_size=args.batch_size
    )
    
    # Create DataLoader with appropriate settings
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False,  # Dataset handles its own shuffling via shifts
        num_workers=args.num_workers, 
        pin_memory=True  # Speeds up GPU transfer
    )
    
    # Training loop
    for epoch in range(10):
        # IMPORTANT: Set epoch for proper randomization of shifts
        # This ensures different shift patterns across epochs
        dataloader.dataset.set_epoch(epoch)
        
        # Iterate through batches
        for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
            input_ids, labels = batch["input_ids"], batch["labels"]
            
            # Here you would normally:
            # 1. Forward pass through your model
            # 2. Compute loss
            # 3. Backward pass and optimization
            # 
            # Example:
            # outputs = model(input_ids)
            # loss = criterion(outputs.view(-1, vocab_size), labels.view(-1))
            # loss.backward()
            # optimizer.step()

def main_ddp():
    """
    Main training loop for distributed data parallel (DDP) training.
    
    This function demonstrates how to use the SlidingTokenDataset
    in a multi-GPU distributed setting. Key differences from single GPU:
    1. Distributed process group setup
    2. Rank-aware dataset creation
    3. DistributedSampler usage
    4. Progress bar only on rank 0
    """
    args = get_args()
    
    # Set up distributed training
    rank, world_size = setup_distributed()

    # Create dataset with distributed training parameters
    dataset = SlidingTokenDataset(
        dataset_path=args.data_path,
        seq_len=args.seq_len,
        stride=args.stride,
        batch_size=args.batch_size,
        rank=rank,        # Current process rank
        world_size=world_size  # Total number of processes
    )

    # Create DistributedSampler to ensure each GPU gets different data
    sampler = DistributedSampler(
        dataset, 
        num_replicas=world_size, 
        rank=rank, 
        shuffle=False  # Must be False
    )

    # Create DataLoader with distributed sampler
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size // world_size,  # batch_size per GPU
        shuffle=False,  # Must be False when using DistributedSampler
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Training loop
    for epoch in range(10):
        # Print epoch info only on the main process to avoid cluttered output
        if rank == 0:
            print("Epoch: ", epoch)
            
        # Set epoch for both dataset and sampler
        # Dataset: for shift randomization
        # Sampler: for distributed data shuffling
        dataloader.dataset.set_epoch(epoch)
        
        # Iterate through batches
        # Progress bar only shown on rank 0 to avoid multiple bars
        for batch in tqdm(dataloader, desc=f"Epoch {epoch}", disable=(rank != 0)):
            input_ids, labels = batch["input_ids"], batch["labels"]
            
            # Your distributed training code here
            # Don't forget to:
            # 1. Use DistributedDataParallel wrapper for your model
            # 2. Handle gradient synchronization
            # 3. Average losses across processes if needed

    # Clean up distributed training
    cleanup_distributed()

if __name__ == "__main__":
    """
    Entry point that automatically detects whether to run in single GPU
    or distributed mode based on the environment.
    
    For single GPU: python demo.py --data_path ./data/your-dataset
    For multi-GPU: torchrun --nproc_per_node=4 demo.py --data_path ./data/your-dataset
    """
    if not is_ddp_mode():
        print("Running in single GPU mode")
        main()
    else:
        print("Running in distributed mode")
        main_ddp()