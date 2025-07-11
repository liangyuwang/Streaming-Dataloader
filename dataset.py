import os
import re
import numpy as np
from tqdm.auto import tqdm
from collections import OrderedDict
import collections
import threading
import torch
from torch.utils.data import Dataset
from datasets import load_from_disk

class SlidingTokenDataset(Dataset):
    """
    A memory-efficient streaming dataset for LLM pretraining that uses sliding window
    technique to maximize data utilization while keeping memory usage under control.
    
    Key Features:
    - Sliding window processing with configurable stride
    - LRU cache for memory-efficient chunk loading
    - Dynamic shift mechanism for data augmentation
    - Distributed training support
    - Thread-safe operations
    
    Args:
        dataset_path (str): Path to the directory containing preprocessed data chunks
        split (str): Dataset split, either "train" or "validation"
        split_rate (float): Ratio for train/validation split (e.g., 0.9 means 90% train, 10% val)
        seq_len (int): Sequence length for each sample
        stride (int): Sliding window stride (how much to move window each step)
        batch_size (int): Batch size for grouping samples with same shift
        m (int): Fixed shift value; default 1 for next token prediction, if None, uses proper divisors of seq_len
        seed (int): Random seed for reproducibility
        rank (int): Process rank for distributed training
        world_size (int): Total number of processes in distributed training
        cache_capacity (int): Maximum number of chunks to keep in LRU cache
        tokens_per_chunk (float): Expected number of tokens per chunk (for length estimation)
    """
    def __init__(self, dataset_path=None, split="train", split_rate=1.0,
                 seq_len=1024, stride=1024, batch_size=1, m=1, 
                 seed=42, rank=0, world_size=1, 
                 cache_capacity=2, tokens_per_chunk=1e8):
        
        # Get all shard directories and sort them by shard ID for consistent ordering
        all_shard_paths = sorted([
            os.path.join(dataset_path, d)
            for d in os.listdir(dataset_path)
            if os.path.isdir(os.path.join(dataset_path, d))
        ], key=lambda path: self.extract_shard_id(os.path.basename(path)))
        
        # Split dataset into train/validation based on split_rate
        split_idx = int(len(all_shard_paths) * split_rate)
        if split == "train":
            self.shard_paths = all_shard_paths[:split_idx]
        elif split == "validation":
            self.shard_paths = all_shard_paths[split_idx:]
        else:
            raise ValueError(f"Invalid split: {split}. Use 'train' or 'validation'.")
    
        # Store configuration parameters
        self.seq_len = seq_len
        self.stride = stride
        self.batch_size = batch_size
        self.m = m  # Fixed shift value, None means use dynamic shifts
        self.seed = seed
        self.rank = rank
        self.world_size = world_size

        # Load metadata: chunk lengths for efficient indexing
        # We only load lengths, not actual data, to minimize memory usage
        self.chunk_lens = []
        for i in tqdm(range(len(self.shard_paths)), desc=f"Loading {split} dataset", disable=(self.rank != 0)):
            ds = load_from_disk(self.shard_paths[i])
            assert len(ds) == 1, "Each chunk should contain exactly one sequence"
            
            # Use provided tokens_per_chunk for all but last chunk (which may be smaller)
            if tokens_per_chunk is not None:
                self.chunk_lens.append(int(tokens_per_chunk  
                                    if i != len(self.shard_paths)-1 
                                    else len(ds[0]['input_ids'])))
            else:
                # Fallback: use actual length (requires loading data)
                self.chunk_lens.append(int(len(ds[0]['input_ids'])))

        # Create cumulative sum for efficient binary search later
        self.cumsum_lens = []
        total = 0
        for l in self.chunk_lens:
            total += l
            self.cumsum_lens.append(total)
        self.total_tokens = total

        # Determine valid shift values for the shift mechanism
        # Shifts must be proper divisors of seq_len to ensure clean reshaping
        if m is not None:
            self.shift_candidates = [m]
        else:
            self.shift_candidates = self._proper_divisors(seq_len)
            assert self.shift_candidates, f"seq_len={seq_len} is prime, and no m was specified."

        # Pre-compute all valid starting positions for sequences
        # This avoids runtime computation and ensures we don't exceed data bounds
        full_starts = []
        for start in tqdm(range(0, self.total_tokens - seq_len - max(self.shift_candidates), stride), 
                         desc=f"Loading {split} dataset", disable=(self.rank != 0)):
            # Check if this start position allows for at least one valid shift
            valid_shifts = [s for s in self.shift_candidates if start + seq_len + s <= self.total_tokens]
            if valid_shifts:
                full_starts.append(start)

        self.starts = full_starts
        self.set_epoch(0)  # Initialize random state for first epoch

        # Initialize LRU cache for memory-efficient chunk loading
        self._chunk_cache = LRUCache(capacity=cache_capacity)

    def _load_chunk(self, idx):
        """
        Load a data chunk with LRU caching to minimize I/O operations.
        
        Args:
            idx (int): Index of the chunk to load
            
        Returns:
            list: Token IDs from the specified chunk
        """
        chunk = self._chunk_cache.get(idx)
        if chunk is None:
            # Cache miss: load chunk from disk
            ds = load_from_disk(self.shard_paths[idx])
            chunk = ds[0]["input_ids"]
            self._chunk_cache.put(idx, chunk)
        return chunk

    def set_epoch(self, epoch):
        """
        Set the current epoch and initialize random state for shift selection.
        This should be called at the beginning of each epoch to ensure
        proper randomization and reproducibility.
        
        Args:
            epoch (int): Current epoch number
        """
        # Create epoch-specific random number generator for reproducibility
        self.rng = np.random.default_rng(self.seed + epoch)
        
        # Pre-generate shift values for all samples in this epoch
        # Samples within the same batch share the same shift for efficient processing
        self.shifts = []
        i = 0
        while i < len(self.starts):
            # Find valid shifts for this starting position
            valid_shifts = [s for s in self.shift_candidates 
                           if self.starts[i] + self.seq_len + s <= self.total_tokens]
            shift = self.rng.choice(valid_shifts)
            
            # Assign the same shift to all samples in this batch
            for j in range(self.batch_size):
                if i + j < len(self.starts):
                    self.shifts.append(shift)
            i += self.batch_size

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.starts)

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        The sliding window mechanism works as follows:
        1. Extract seq_len + shift tokens starting from the predetermined position
        2. Split into input (first seq_len tokens) and labels (last seq_len tokens)
        3. Reshape both to [seq_len//shift, shift] for efficient batch processing
        
        Args:
            idx (int): Sample index
            
        Returns:
            dict: Contains 'input_ids', 'labels', and 'shift' tensors
        """
        start = self.starts[idx]
        shift = self.shifts[idx]
        
        # Extract tokens for this sample (input + shifted labels)
        data = torch.tensor(self._get_tokens_range(start, start + self.seq_len + shift), dtype=torch.long)
        
        return {
            "input_ids": data[:self.seq_len].reshape(self.seq_len//shift, shift),
            "labels": data[shift:].reshape(self.seq_len//shift, shift),
            "shift": shift,
        }

    def _get_tokens_range(self, start, end):
        """
        Extract tokens from a range that may span multiple chunks.
        Uses the LRU cache to minimize I/O operations.
        
        Args:
            start (int): Starting token index (global)
            end (int): Ending token index (global, exclusive)
            
        Returns:
            list: Token IDs in the specified range
        """
        tokens = []
        cur = start
        
        while cur < end:
            # Find which chunk contains the current position
            chunk_idx = self._find_chunk(cur)
            
            # Calculate offset within the chunk
            chunk_start_offset = cur - (self.cumsum_lens[chunk_idx - 1] if chunk_idx > 0 else 0)
            
            # Load chunk (may come from cache)
            chunk = self._load_chunk(chunk_idx)
            
            # Take as many tokens as needed from this chunk
            take = min(len(chunk) - chunk_start_offset, end - cur)
            tokens.extend(chunk[chunk_start_offset:chunk_start_offset + take])
            cur += take
            
        return tokens

    def _find_chunk(self, index):
        """
        Find which chunk contains the token at the given global index.
        Uses binary search for O(log n) efficiency.
        
        Args:
            index (int): Global token index
            
        Returns:
            int: Chunk index containing the specified token
        """
        left, right = 0, len(self.cumsum_lens) - 1
        
        while left <= right:
            mid = (left + right) // 2
            if index < self.cumsum_lens[mid]:
                if mid == 0 or index >= self.cumsum_lens[mid - 1]:
                    return mid
                right = mid - 1
            else:
                left = mid + 1
                
        raise IndexError("Index out of range")

    def _proper_divisors(self, n):
        """
        Find all proper divisors of n (divisors other than 1 and n itself).
        These are used as valid shift values to ensure clean tensor reshaping.
        
        Args:
            n (int): Number to find divisors for
            
        Returns:
            list: All proper divisors of n
        """
        return [i for i in range(2, n) if n % i == 0]

    def extract_shard_id(self, name):
        """
        Extract numeric shard ID from directory name for proper sorting.
        
        Args:
            name (str): Directory name (e.g., "chunk_000042")
            
        Returns:
            int: Numeric shard ID, or -1 if no number found
        """
        match = re.search(r"\d+", name)
        return int(match.group()) if match else -1


class LRUCache:
    """
    A thread-safe Least Recently Used (LRU) cache implementation.
    
    This cache is crucial for memory management in the streaming dataset.
    It ensures that only the most recently accessed chunks are kept in memory,
    preventing OOM errors while maintaining good performance.
    
    Features:
    - Thread-safe operations using locks
    - Automatic eviction of least recently used items
    - Picklable for use with multiprocessing
    
    Args:
        capacity (int): Maximum number of items to store in the cache
    """
    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer.")
        self.capacity = capacity
        self.cache = collections.OrderedDict()  # Maintains insertion/access order
        self.lock = threading.Lock()  # Ensures thread safety
    
    def __getstate__(self):
        """
        Custom pickling method to handle non-picklable lock objects.
        This is needed for multiprocessing with DataLoader workers.
        """
        state = self.__dict__.copy()
        del state["lock"]  # Remove the unpicklable lock
        return state

    def __setstate__(self, state):
        """
        Custom unpickling method to recreate the lock object.
        """
        self.__dict__.update(state)
        self.lock = threading.Lock()  # Create a new lock

    def get(self, key):
        """
        Retrieve an item from the cache.
        
        If the item exists, it's moved to the end (most recently used position).
        
        Args:
            key: Key to look up
            
        Returns:
            The cached value, or None if not found
        """
        with self.lock:
            if key not in self.cache:
                return None
            
            # Move to end to mark as most recently used
            self.cache.move_to_end(key)
            return self.cache[key]

    def put(self, key, value):
        """
        Add an item to the cache.
        
        If the cache is at capacity, the least recently used item is evicted.
        If the key already exists, its value is updated and it's marked as most recent.
        
        Args:
            key: Key to store
            value: Value to cache
        """
        with self.lock:
            if key in self.cache:
                # Key exists: update value and mark as most recent
                self.cache[key] = value
                self.cache.move_to_end(key)
            else:
                # New key: check capacity and add
                if len(self.cache) >= self.capacity:
                    # Evict least recently used item (from the front)
                    self.cache.popitem(last=False)
                
                # Add new item
                self.cache[key] = value
    
    def __len__(self):
        """Return the current number of items in the cache."""
        with self.lock:
            return len(self.cache)

    def __repr__(self):
        """String representation of the cache contents."""
        with self.lock:
            return repr(self.cache)