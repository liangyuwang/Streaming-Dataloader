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
    def __init__(self, dataset_path=None, split="train", split_rate=1.0,
                 seq_len=1024, stride=512, batch_size=1, m=1, 
                 seed=42, rank=0, world_size=1, 
                 cache_capacity=2, tokens_per_chunk=1e8):
        all_shard_paths = sorted([
            os.path.join(dataset_path, d)
            for d in os.listdir(dataset_path)
            if os.path.isdir(os.path.join(dataset_path, d))
        ], key=lambda path: self.extract_shard_id(os.path.basename(path)))
        
        # ---- Split train / val ----
        split_idx = int(len(all_shard_paths) * split_rate)
        if split == "train":
            self.shard_paths = all_shard_paths[:split_idx]
        elif split == "validation":
            self.shard_paths = all_shard_paths[split_idx:]
        else:
            raise ValueError(f"Invalid split: {split}. Use 'train' or 'validation'.")
    
        self.seq_len = seq_len
        self.stride = stride
        self.batch_size = batch_size
        self.m = m  # normally be 1 for next token prediction
        self.seed = seed
        self.rank = rank
        self.world_size = world_size

        # only load the length of every chunk
        self.chunk_lens = []
        for i in tqdm(range(len(self.shard_paths)), desc=f"Loading {split} dataset", disable=(self.rank != 0)):
            ds = load_from_disk(self.shard_paths[i])
            assert len(ds) == 1
            if tokens_per_chunk is not None:
                self.chunk_lens.append(int(tokens_per_chunk  # same as 'tokens_per_chunk' in "dataset/fineweb_edu.py"
                                    if i!=len(self.shard_paths)-1 
                                    else len(ds[0]['input_ids'])))
            else:
                self.chunk_lens.append(int(len(ds[0]['input_ids'])))

        self.cumsum_lens = []
        total = 0
        for l in self.chunk_lens:
            total += l
            self.cumsum_lens.append(total)
        self.total_tokens = total

        if m is not None:
            self.shift_candidates = [m]
        else:
            self.shift_candidates = self._proper_divisors(seq_len)
            assert self.shift_candidates, f"seq_len={seq_len} is prime, and no m was specified."

        full_starts = []
        for start in tqdm(range(0, self.total_tokens - seq_len - max(self.shift_candidates), stride), desc=f"Loading {split} dataset", disable=(self.rank != 0)):
            valid_shifts = [s for s in self.shift_candidates if start + seq_len + s <= self.total_tokens]
            if valid_shifts:
                full_starts.append(start)

        self.starts = full_starts
        self.set_epoch(0)

        self._chunk_cache = LRUCache(capacity=cache_capacity)   # high cache_capacity will take much memory

    def _load_chunk(self, idx):
        chunk = self._chunk_cache.get(idx)
        if chunk is None:
            ds = load_from_disk(self.shard_paths[idx])
            chunk = ds[0]["input_ids"]
            self._chunk_cache.put(idx, chunk)
        return chunk

    def set_epoch(self, epoch):
        self.rng = np.random.default_rng(self.seed + epoch) # more safe than random lib
        self.shifts = []
        i = 0
        while i < len(self.starts):
            valid_shifts = [s for s in self.shift_candidates if self.starts[i] + self.seq_len + s <= self.total_tokens]
            shift = self.rng.choice(valid_shifts)
            for j in range(self.batch_size):
                if i + j < len(self.starts):
                    self.shifts.append(shift)
            i += self.batch_size

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        start = self.starts[idx]
        shift = self.shifts[idx]
        data = torch.tensor(self._get_tokens_range(start, start + self.seq_len + shift), dtype=torch.long)
        return {
            "input_ids": data[:self.seq_len].reshape(self.seq_len//shift, shift),
            "labels": data[shift:].reshape(self.seq_len//shift, shift),
            "shift": shift,
        }

    def _get_tokens_range(self, start, end):
        tokens = []
        cur = start
        while cur < end:
            chunk_idx = self._find_chunk(cur)
            chunk_start_offset = cur - (self.cumsum_lens[chunk_idx - 1] if chunk_idx > 0 else 0)
            chunk = self._load_chunk(chunk_idx)
            take = min(len(chunk) - chunk_start_offset, end - cur)
            tokens.extend(chunk[chunk_start_offset:chunk_start_offset + take])
            cur += take
        return tokens

    def _find_chunk(self, index):
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
        return [i for i in range(2, n) if n % i == 0]

    def extract_shard_id(self, name):
        match = re.search(r"\d+", name)
        return int(match.group()) if match else -1


class LRUCache:
    """
    A thread-safe Least Recently Used (LRU) cache.
    
    Args:
        capacity (int): The maximum number of items to store in the cache.
    """
    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer.")
        self.capacity = capacity
        self.cache = collections.OrderedDict()
        self.lock = threading.Lock() # The key to making it thread-safe
    
    # --- make it picklable -----------------
    def __getstate__(self):
        state = self.__dict__.copy()
        del state["lock"]          # locks can't be pickled
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.lock = threading.Lock()   # recreate a fresh lock
    # ---------------------------------------

    def get(self, key):
        """
        Retrieves an item from the cache. Returns None if the item is not found.
        Moves the accessed item to the end to mark it as recently used.
        """
        with self.lock:
            if key not in self.cache:
                return None
            
            # Move the key to the end to mark it as most recently used
            self.cache.move_to_end(key)
            return self.cache[key]

    def put(self, key, value):
        """
        Adds an item to the cache. If the cache is full, it evicts the
        least recently used item.
        """
        with self.lock:
            # If key already exists, update it and move to the end
            if key in self.cache:
                self.cache[key] = value
                self.cache.move_to_end(key)
            else:
                # If cache is at full capacity, remove the oldest item
                if len(self.cache) >= self.capacity:
                    # popitem(last=False) pops from the front (Least Recently Used)
                    self.cache.popitem(last=False)
                
                # Add the new item
                self.cache[key] = value
    
    def __len__(self):
        with self.lock:
            return len(self.cache)

    def __repr__(self):
        with self.lock:
            return repr(self.cache)