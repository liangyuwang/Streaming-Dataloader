# Streaming-Dataloader

A memory-efficient streaming data loader designed for LLM pretraining under limited CPU and GPU memory constraints.

## 🎯 Overview

Streaming-Dataloader is a high-performance data loading solution that enables training large language models on massive datasets without overwhelming system memory. It uses smart caching, sliding window techniques, and distributed processing to handle terabyte-scale datasets efficiently.

## ✨ Key Features

- **Memory Efficient**: LRU cache mechanism controls memory usage, preventing OOM errors
- **Streaming Processing**: Processes data chunks on-demand without loading entire datasets
- **Sliding Window**: Maximizes data utilization through configurable stride patterns  
- **Dynamic Shifts**: Increases training data diversity with randomized sequence shifts
- **Distributed Ready**: Built-in support for multi-GPU and multi-node training
- **Thread Safe**: Robust concurrent data loading with multiple workers

## 🚀 Performance Highlights

- **Low Memory Footprint**: Works efficiently with <32GB CPU RAM and <24GB GPU memory
- **Scalable**: Handles TB-scale datasets through intelligent chunking
- **Fast Loading**: Binary search optimization for rapid chunk location
- **High Throughput**: Optimized batch processing with shared shift values

## 📦 Installation

```bash
git clone https://github.com/your-username/Streaming-Dataloader.git
cd Streaming-Dataloader
```

### Dependencies

This project requires the following packages with tested versions:

- **PyTorch**: 2.4.0
- **CUDA**: 12.1
- **datasets**: 3.5.1
- **transformers**: 4.51.3
- **tqdm**: 4.66.5
- **numpy**: 1.26.4

Please install these dependencies according to your environment setup.

## 🔧 Data Preparation

First, prepare your dataset by tokenizing and chunking:

```bash
cd prepare/

# Prepare FineWeb-Edu dataset (example)
python fineweb_edu.py \
    --tokenizer gpt2 \
    --data_name sample-10BT \
    --output_path ./data/fineweb-edu-sample-10BT/ \
    --tokens_per_chunk 100000000 \
    --max_samples 1000
```

**Parameters:**
- `--tokenizer`: Tokenizer model (default: gpt2)
- `--data_name`: Dataset name from HuggingFace
- `--output_path`: Output directory for processed chunks
- `--tokens_per_chunk`: Tokens per chunk (default: 100M)
- `--max_samples`: Maximum samples to process (optional)

## 💻 Usage

### Single GPU Training

```python
from dataset import SlidingTokenDataset
from torch.utils.data import DataLoader

# Create dataset
dataset = SlidingTokenDataset(
    dataset_path="./data/fineweb-edu-sample-10BT",
    split="train",
    split_rate=0.9,
    seq_len=1024,
    stride=512,
    batch_size=16,
    cache_capacity=2
)

# Create dataloader
dataloader = DataLoader(
    dataset, 
    batch_size=16, 
    num_workers=4, 
    pin_memory=True
)

# Training loop
for epoch in range(10):
    dataset.set_epoch(epoch)  # Important: set epoch for randomization
    for batch in dataloader:
        input_ids = batch["input_ids"]  # [batch_size, seq_len//shift, shift]
        labels = batch["labels"]        # [batch_size, seq_len//shift, shift]
        shift = batch["shift"]          # shift value used for this batch
        
        # Your training code here
        loss = model(input_ids, labels=labels)
        loss.backward()
```

### Multi-GPU Training

```bash
# Run with torchrun
torchrun --nproc_per_node=4 demo.py \
    --data_path ./data/fineweb-edu-sample-10BT \
    --seq_len 2048 \
    --stride 1024 \
    --batch_size 8 \
    --num_workers 2
```

### Quick Demo

```bash
# Single GPU
python demo.py --data_path ./data/fineweb-edu-sample-10BT

# Multi-GPU (DDP)
torchrun --nproc_per_node=2 demo.py --data_path ./data/fineweb-edu-sample-10BT
```

## ⚙️ Configuration

### SlidingTokenDataset Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset_path` | str | None | Path to processed dataset chunks |
| `split` | str | "train" | Dataset split ("train" or "validation") |
| `split_rate` | float | 1.0 | Train/validation split ratio |
| `seq_len` | int | 1024 | Sequence length |
| `stride` | int | 512 | Sliding window stride |
| `batch_size` | int | 1 | Batch size for shift grouping |
| `m` | int | 1 | Fixed shift value (default 1 for next token prediction) |
| `seed` | int | 42 | Random seed |
| `rank` | int | 0 | Process rank for distributed training |
| `world_size` | int | 1 | Total number of processes |
| `cache_capacity` | int | 2 | LRU cache capacity (number of chunks) |
| `tokens_per_chunk` | float | 1e8 | Expected tokens per chunk |

## 🧠 How It Works

### 1. Data Chunking
```
Original Dataset → Tokenize → Split into Chunks → Save to Disk
[Raw Text] → [Tokens] → [Chunk_0, Chunk_1, ...] → [chunk_000000/, chunk_000001/, ...]
```

### 2. Sliding Window Processing
```
Sequence: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
stride=4, seq_len=8, shift=2

Sample 1: input=[0,1,2,3,4,5,6,7], label=[2,3,4,5,6,7,8,9]
Sample 2: input=[4,5,6,7,8,9,10,11], label=[6,7,8,9,10,11,12,13]
```

### 3. Memory Management
- **LRU Cache**: Only keeps recently accessed chunks in memory
- **Lazy Loading**: Loads chunks only when needed
- **Garbage Collection**: Automatic cleanup of unused chunks

## 🔍 Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Prep     │    │  Sliding Window │    │   LRU Cache     │
│   (streaming)   │───▶│   Processing    │───▶│   Management    │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Tokenized Chunks│    │ Batched Samples │    │ Memory Efficient│
│ chunk_000000/   │    │ with Shifts     │    │ Loading         │
│ chunk_000001/   │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📊 Memory Usage

| Component | Memory Impact | Notes |
|-----------|---------------|-------|
| Chunk Cache | `cache_capacity × chunk_size` | Typically 2 × 100M tokens |
| Batch Buffer | `batch_size × seq_len × dtype` | Temporary batch storage |
| Metadata | Minimal | Only stores chunk lengths and indices |

**Example**: With `cache_capacity=2`, `tokens_per_chunk=1e8`, each chunk ~400MB, total cache ~800MB.

## 🎛️ Advanced Usage

### Custom Shift Strategy
```python
# Use fixed shift for consistent batching
dataset = SlidingTokenDataset(
    dataset_path="./data/your-dataset",
    seq_len=1024,
    m=256,  # Fixed shift value
)

# Let the system choose optimal shifts automatically
dataset = SlidingTokenDataset(
    dataset_path="./data/your-dataset", 
    seq_len=1024,
    # m=None (default) - uses proper divisors of seq_len
)
```

### Memory Optimization
```python
# For memory-constrained environments
dataset = SlidingTokenDataset(
    dataset_path="./data/your-dataset",
    cache_capacity=1,      # Minimal cache
    tokens_per_chunk=5e7,  # Smaller chunks
)

# For high-memory environments  
dataset = SlidingTokenDataset(
    dataset_path="./data/your-dataset",
    cache_capacity=10,     # Larger cache
    tokens_per_chunk=2e8,  # Bigger chunks
)
```

## 🐛 Troubleshooting

### Common Issues

**Out of Memory (OOM)**
```python
# Reduce cache capacity
cache_capacity=1

# Reduce batch size
batch_size=8

# Use smaller chunks in preprocessing
tokens_per_chunk=5e7
```

**Slow Loading**
```python
# Increase cache capacity (if memory allows)
cache_capacity=5

# Increase number of workers
num_workers=4

# Use larger chunks
tokens_per_chunk=2e8
```

**Data Imbalance in DDP**
```python
# Ensure proper distributed setup
dataset = SlidingTokenDataset(
    dataset_path="./data/your-dataset",
    rank=rank,          # Must set rank
    world_size=world_size  # Must set world_size  
)
```

## 📈 Performance Tips

1. **Optimize Cache Size**: Balance between memory usage and I/O operations
2. **Tune Chunk Size**: Larger chunks = fewer files but more memory per chunk
3. **Use SSD Storage**: Significantly improves chunk loading speed
4. **Pin Memory**: Use `pin_memory=True` in DataLoader for GPU training
5. **Proper Workers**: Set `num_workers` based on CPU cores and I/O capacity

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Report bugs and issues
- Suggest new features  
- Submit pull requests
- Improve documentation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- HuggingFace for the datasets library
- PyTorch team for distributed training utilities
- FineWeb-Edu dataset for providing high-quality training data

---

**Built for efficient LLM pretraining** 🚀