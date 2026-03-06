import os
import glob
import json
import bisect
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset


class DistributedDataset(IterableDataset):
    """
    Treat multiple .bin shards as one logical continuous token stream, then shard
    fixed-length samples across (dp_rank * num_workers + worker_id).

    Sample definition:
        sample_id -> token range [sample_id * (seq_len + 1), sample_id * (seq_len + 1) + seq_len + 1)

    Each stream reads:
        sample_id = global_stream_id, global_stream_id + total_streams, ...

    Properties:
    - no per-chunk tail dropping
    - no need for external .idx dataset
    - supports distributed + dataloader workers
    - supports resume via global_skip_batches
    - optional deterministic sample shuffle per epoch
    """

    def __init__(
        self,
        data_dir,
        seq_len=2048,
        dtype=None,                 # None => infer from meta.json, else np.uint16 / np.uint32
        shuffle=False,
        seed=42,
        global_skip_batches=0,      # number of globally-consumed samples to skip
        strict=True,                # if True, require enough samples for all streams
    ):
        super().__init__()
        self.data_dir = data_dir
        self.seq_len = int(seq_len)
        self.block_size = self.seq_len + 1
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        self.global_skip_batches = int(global_skip_batches or 0)
        self.strict = bool(strict)

        # ---- load meta.json if available ----
        meta_path = os.path.join(data_dir, "meta.json")
        meta = None
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

        # ---- infer dtype ----
        if dtype is None:
            if meta is not None:
                meta_dtype = meta.get("dtype", "uint16")
                if meta_dtype == "uint16":
                    dtype = np.uint16
                elif meta_dtype == "uint32":
                    dtype = np.uint32
                else:
                    raise ValueError(f"Unsupported dtype in meta.json: {meta_dtype}")
            else:
                dtype = np.uint16
        self.dtype = dtype

        # ---- discover chunk files in deterministic order ----
        if meta is not None and "chunks" in meta:
            # Prefer meta.json order if present
            chunk_files = [os.path.join(data_dir, c["file"]) for c in meta["chunks"]]
            chunk_files = [p for p in chunk_files if os.path.exists(p)]
        else:
            chunk_files = sorted(glob.glob(os.path.join(data_dir, "*.bin")))

        if not chunk_files:
            raise FileNotFoundError(f"No .bin files found in {data_dir}")

        self.chunk_files = chunk_files

        # ---- token counts per chunk ----
        if meta is not None and "chunks" in meta:
            meta_map = {c["file"]: int(c["tokens"]) for c in meta["chunks"]}
            chunk_token_counts = []
            for p in self.chunk_files:
                fname = os.path.basename(p)
                if fname in meta_map:
                    chunk_token_counts.append(meta_map[fname])
                else:
                    # fallback to filesize
                    nbytes = os.path.getsize(p)
                    itemsize = np.dtype(self.dtype).itemsize
                    if nbytes % itemsize != 0:
                        raise ValueError(f"File size not divisible by dtype size: {p}")
                    chunk_token_counts.append(nbytes // itemsize)
        else:
            chunk_token_counts = []
            itemsize = np.dtype(self.dtype).itemsize
            for p in self.chunk_files:
                nbytes = os.path.getsize(p)
                if nbytes % itemsize != 0:
                    raise ValueError(f"File size not divisible by dtype size: {p}")
                chunk_token_counts.append(nbytes // itemsize)

        self.chunk_token_counts = [int(x) for x in chunk_token_counts]
        self.num_chunks = len(self.chunk_files)

        # ---- global layout ----
        # chunk_starts[i] = global token offset where chunk i starts
        self.chunk_starts = np.cumsum([0] + self.chunk_token_counts[:-1], dtype=np.int64).tolist()
        self.total_tokens = int(sum(self.chunk_token_counts))
        self.total_samples = self.total_tokens // self.block_size  # only final global tail is dropped

        if self.total_samples <= 0:
            raise ValueError(
                f"Not enough tokens to form one sample: total_tokens={self.total_tokens}, "
                f"block_size={self.block_size}"
            )

        # memmap cache per worker/process iterator
        self._memmaps = None

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def _ensure_memmaps(self):
        if self._memmaps is None:
            self._memmaps = [
                np.memmap(path, dtype=self.dtype, mode="r")
                for path in self.chunk_files
            ]

    def _read_global_range(self, start_token: int, length: int) -> np.ndarray:
        """
        Read tokens from the logical global stream [start_token, start_token + length),
        automatically crossing chunk boundaries when needed.

        Returns a fresh contiguous numpy array of shape [length], dtype=self.dtype.
        """
        end_token = start_token + length
        if start_token < 0 or end_token > self.total_tokens:
            raise IndexError(
                f"Requested token range [{start_token}, {end_token}) out of bounds "
                f"for total_tokens={self.total_tokens}"
            )

        # Fast path: range fully inside one chunk
        chunk_idx = bisect.bisect_right(self.chunk_starts, start_token) - 1
        chunk_start = self.chunk_starts[chunk_idx]
        chunk_len = self.chunk_token_counts[chunk_idx]
        local_offset = start_token - chunk_start

        if local_offset + length <= chunk_len:
            arr = self._memmaps[chunk_idx][local_offset : local_offset + length]
            return np.asarray(arr)  # materialize contiguous array view/copy as needed

        # Slow path: crosses chunk boundary
        out = np.empty(length, dtype=self.dtype)
        out_pos = 0
        remaining = length
        cur = start_token

        while remaining > 0:
            ci = bisect.bisect_right(self.chunk_starts, cur) - 1
            cs = self.chunk_starts[ci]
            clen = self.chunk_token_counts[ci]
            lo = cur - cs
            take = min(remaining, clen - lo)

            out[out_pos : out_pos + take] = self._memmaps[ci][lo : lo + take]

            out_pos += take
            cur += take
            remaining -= take

        return out

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        if dist.is_available() and dist.is_initialized():
            dp_rank = dist.get_rank()
            dp_world_size = dist.get_world_size()
        else:
            dp_rank = 0
            dp_world_size = 1

        total_streams = dp_world_size * num_workers
        global_stream_id = dp_rank * num_workers + worker_id

        if self.strict and self.total_samples < total_streams:
            raise ValueError(
                f"total_samples={self.total_samples} < total_streams={total_streams}. "
                f"Some streams would receive no data. Reduce num_workers/world_size "
                f"or use more data."
            )

        self._ensure_memmaps()

        # Global sample ids after resume skip
        start_sample = self.global_skip_batches + global_stream_id
        if start_sample >= self.total_samples:
            return

        # Deterministic epoch-aware order
        if not self.shuffle:
            sample_ids = range(start_sample, self.total_samples, total_streams)
        else:
            # Build global permutation once per iterator in this worker process
            rng = np.random.default_rng(self.seed + self.epoch * 1000003)
            perm = rng.permutation(self.total_samples)

            # Apply global skip first, then stride by stream id
            perm = perm[self.global_skip_batches:]
            sample_ids = perm[global_stream_id :: total_streams]

        for sample_id in sample_ids:
            sample_id = int(sample_id)
            token_offset = sample_id * self.block_size

            tokens = self._read_global_range(token_offset, self.block_size)

            # convert only this sample to torch
            # use int64 directly because embedding lookup / CE targets usually want long
            t = torch.from_numpy(tokens.astype(np.int64, copy=False))

            x = t[:-1]
            y = t[1:]
            yield x, y


def lm_collate(batch):
    xs, ys = zip(*batch)  # each is [seq_len]
    x = torch.stack(xs, dim=0).long()
    y = torch.stack(ys, dim=0).long()
    return x, y