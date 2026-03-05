import os, glob, json
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset

class DistributedDataset(IterableDataset):
    def __init__(
        self,
        data_dir,
        seq_len=2048,
        dtype=None,                 # None => try meta.json
        shuffle=False,
        seed=42,
        skip_batches_per_worker=0,  # per-stream skip
        global_skip_batches=None,   # optional: global skip across all streams
    ):
        super().__init__()
        self.data_dir = data_dir
        self.seq_len = int(seq_len)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

        # ---- load meta.json if exists ----
        meta_path = os.path.join(data_dir, "meta.json")
        if dtype is None and os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            meta_dtype = meta.get("dtype", "uint16")
            dtype = np.uint16 if meta_dtype == "uint16" else np.uint32
        self.dtype = dtype if dtype is not None else np.uint16

        self.chunk_files = sorted(glob.glob(os.path.join(data_dir, "*.bin")))
        assert len(self.chunk_files) > 0, f"No .bin file in {data_dir}."

        if dist.is_available() and dist.is_initialized():
            self.dp_rank = dist.get_rank()
            self.dp_world_size = dist.get_world_size()
        else:
            self.dp_rank = 0
            self.dp_world_size = 1

        self.skip_batches_per_worker = int(skip_batches_per_worker)
        self.global_skip_batches = None if global_skip_batches is None else int(global_skip_batches)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        total_streams = self.dp_world_size * num_workers
        global_stream_id = (self.dp_rank * num_workers) + worker_id

        # ---- compute per-stream skip from global_skip_batches if provided ----
        target_skip = self.skip_batches_per_worker
        if self.global_skip_batches is not None:
            base = self.global_skip_batches // total_streams
            rem = self.global_skip_batches % total_streams
            target_skip = base + (1 if global_stream_id < rem else 0)

        # ---- deterministic rng (epoch-aware) ----
        base_seed = self.seed + self.epoch * 1000003  # big-ish prime
        rng = np.random.default_rng(base_seed)

        chunk_list = rng.permutation(self.chunk_files).tolist() if self.shuffle else list(self.chunk_files)
        my_chunks = chunk_list[global_stream_id :: total_streams]

        batches_skipped = 0

        for chunk_file in my_chunks:
            mmap_array = np.memmap(chunk_file, dtype=self.dtype, mode="r")
            total_tokens = int(mmap_array.shape[0])
            stride = self.seq_len + 1
            num_batches = total_tokens // stride
            if num_batches <= 0:
                continue

            # chunk-level fast skip
            if batches_skipped + num_batches <= target_skip:
                batches_skipped += num_batches
                continue

            indices = np.arange(num_batches, dtype=np.int64) * stride

            if self.shuffle:
                # stream-specific but epoch-aware
                local_rng = np.random.default_rng(base_seed + 10007 * global_stream_id)
                local_rng.shuffle(indices)

            for idx in indices:
                if batches_skipped < target_skip:
                    batches_skipped += 1
                    continue

                # read view (uint16/uint32)
                tokens = mmap_array[idx : idx + stride]

                # convert to torch tensor WITHOUT numpy astype per-sample
                # keep int32 here; collate can convert whole batch to long once
                t = torch.from_numpy(np.asarray(tokens))  # shares/refs underlying slice as much as possible
                if t.dtype != torch.int32:
                    # uint16/uint32 -> int32 (still a copy, but cheaper than int64 per-sample)
                    t = t.to(torch.int32)

                x = t[:-1]
                y = t[1:]
                yield x, y


def lm_collate(batch):
    xs, ys = zip(*batch)  # each is [seq_len]
    x = torch.stack(xs, dim=0).long()
    y = torch.stack(ys, dim=0).long()
    return x, y