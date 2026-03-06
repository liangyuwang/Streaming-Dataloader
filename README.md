# Streaming-Dataloader

`Streaming-Dataloader` is a small prototype for LLM next-token training on large tokenized corpora.

It has two parts:

- a preprocessing script that streams text from Hugging Face, tokenizes it, and writes raw token shards to disk
- an `IterableDataset` that treats those shards as one logical token stream and splits samples across distributed ranks and `DataLoader` workers

This README reflects the current code in `prepare/fineweb_edu.py`, `dataset.py`, and `demo.py`.

## What The Repo Does

The project stores tokenized data as:

- `chunk_000000.bin`, `chunk_000001.bin`, ...
- `meta.json`

Each `.bin` file contains contiguous token ids in `uint16` or `uint32`. At runtime, `DistributedDataset` opens them with `numpy.memmap`, so the whole corpus does not need to be loaded into RAM.

Each training sample is defined as a contiguous block of `seq_len + 1` tokens:

```text
sample_id -> tokens[start : start + seq_len + 1]
input_ids = tokens[:-1]
labels    = tokens[1:]
```

This is standard autoregressive next-token training.

## Repository Layout

- `prepare/fineweb_edu.py`: stream a dataset, tokenize it, and write binary shards plus `meta.json`
- `dataset.py`: `DistributedDataset`, the core iterable dataset used at training time
- `demo.py`: a minimal distributed demo that builds the dataset and iterates over batches

## Install

There is no packaged installation yet, so install the runtime dependencies directly in your Python environment.

```bash
pip install torch numpy datasets transformers tqdm
```

If you plan to use Hugging Face streaming datasets behind a mirror, set `HF_ENDPOINT` before preprocessing.

## Data Format

After preprocessing, the output directory looks like this:

```text
data/your-dataset/
├── chunk_000000.bin
├── chunk_000001.bin
├── ...
└── meta.json
```

`meta.json` records:

- tokenizer name
- whether the tokenizer is fast
- storage dtype
- `eos_token_id`
- configured `tokens_per_chunk`
- per-chunk token counts
- total token count

The dataset code prefers the chunk order recorded in `meta.json`. If `meta.json` is missing, it falls back to sorted `*.bin` files and infers lengths from file sizes.

## Preprocess Data

The included preprocessing script targets FineWeb-Edu, but the output format is generic enough for the runtime dataset.

Example:

```bash
python prepare/fineweb_edu.py \
  --tokenizer gpt2 \
  --dataset HuggingFaceFW/fineweb-edu \
  --data_name sample-10BT \
  --text_field text \
  --output_path ./data/fineweb-edu-sample-10BT \
  --tokens_per_chunk 100000000 \
  --batch_size 1000
```

Useful flags:

- `--tokenizer`: tokenizer name or local path
- `--dataset`: Hugging Face dataset id
- `--data_name`: dataset config name
- `--text_field`: field containing raw text
- `--output_path`: directory to write `.bin` shards and `meta.json`
- `--tokens_per_chunk`: max number of tokens per shard
- `--batch_size`: text batch size sent to the tokenizer
- `--max_samples`: optional limit for debugging
- `--overwrite`: clear a non-empty output directory before writing

Implementation details:

- the script reads the dataset with `streaming=True`
- each example is tokenized without adding model-specific special tokens
- one `eos_token_id` is appended after every sample
- dtype is chosen automatically:
  - `uint16` when the tokenizer vocab fits in 65536 ids
  - otherwise `uint32`

## Use `DistributedDataset`

Minimal single-process example:

```python
from torch.utils.data import DataLoader
from dataset import DistributedDataset

dataset = DistributedDataset(
    data_dir="./data/fineweb-edu-sample-10BT",
    seq_len=1024,
    shuffle=True,
    seed=123,
)

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)

dataset.set_epoch(0)
for batch in loader:
    input_ids = batch["input_ids"]   # [batch, seq_len]
    labels = batch["labels"]         # [batch, seq_len]
```

### Dataset Behavior

`DistributedDataset` is an `IterableDataset` with the following behavior:

- all shards are treated as one logical global token stream
- only the final global tail is dropped
- samples can cross shard boundaries
- shard files are memory-mapped lazily with `numpy.memmap`
- work is split across:
  - distributed ranks
  - `DataLoader` workers inside each rank
- optional deterministic shuffle is controlled by `seed` and `set_epoch(epoch)`
- `global_skip_batches` can be used to skip already-consumed global samples during resume

### How Samples Are Assigned

Let:

- `block_size = seq_len + 1`
- `total_streams = dp_world_size * num_workers`
- `global_stream_id = dp_rank * num_workers + worker_id`

Then each stream reads:

```text
sample_ids = global_stream_id, global_stream_id + total_streams, ...
```

When shuffling is enabled, the dataset first builds a global permutation for the epoch, then takes every `total_streams`-th sample for each stream.

## Run The Demo

`demo.py` is a smoke test for distributed loading. It does not include a model, optimizer, loss computation, or checkpointing.

Single-process run:

```bash
python demo.py \
  --data_path ./data/fineweb-edu-sample-10BT \
  --seq_len 1024 \
  --batch_size 16 \
  --num_workers 0
```

Distributed run:

```bash
torchrun --nproc_per_node=2 demo.py \
  --data_path ./data/fineweb-edu-sample-10BT \
  --seq_len 1024 \
  --batch_size 16 \
  --num_workers 0
```

Notes:

- `demo.py` currently initializes `torch.distributed` with `gloo`
- CUDA placement lines are present but commented out
- for GPU training, you will likely want to switch the backend to `nccl` and move tensors to the correct device in your own training script

## Main `DistributedDataset` Arguments

| Argument | Default | Meaning |
| --- | --- | --- |
| `data_dir` | required | Directory containing shards and optional `meta.json` |
| `seq_len` | `2048` | Sequence length of `input_ids` and `labels` |
| `dtype` | `None` | Force storage dtype, otherwise infer from `meta.json` |
| `shuffle` | `False` | Whether to reshuffle sample order per epoch |
| `seed` | `42` | Base seed for deterministic shuffling |
| `global_skip_batches` | `0` | Number of global samples to skip before iteration |
| `strict` | `True` | Raise if there are fewer samples than total streams |
| `dp_rank` | `None` | Explicit distributed rank override |
| `dp_world_size` | `None` | Explicit distributed world size override |

If `dp_rank` and `dp_world_size` are not passed, the dataset will try to read them from `torch.distributed` when available.

## Design Choices

This repo deliberately uses a simple data format and loader design:

- no external index file
- no per-chunk tail dropping
- no in-memory corpus loading
- no custom C++ extension

The tradeoff is that this is a focused prototype for autoregressive language modeling, not a general-purpose dataset library.

## Limitations

Current limitations from the codebase:

- preprocessing is currently specialized around FineWeb-Edu-style text streaming
- the runtime dataset only supports fixed-length next-token samples
- there is no validation or test suite in the repo
- there is no packaged environment file such as `requirements.txt` or `pyproject.toml`
- `demo.py` demonstrates loading only; it is not a full trainer

## License

See `LICENSE` if you add one. The current repository contents do not define license terms inside the code itself.
