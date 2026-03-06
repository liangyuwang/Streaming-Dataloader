import os
import json
import time
import shutil
import numpy as np
import argparse
from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

def build_binary_chunks_and_save(args, dataset, tokenizer, eos_token_id,
                                 output_dir, tokens_per_chunk=100_000_000,
                                 batch_size=1000, max_samples=None):
    """
    Stream-read data, tokenize in batches, and write results to raw binary (.bin) files.
    """
    os.makedirs(output_dir, exist_ok=True)

    existing = [p for p in os.listdir(output_dir) if not p.startswith(".")]
    if existing:
        if getattr(args, "overwrite", False):
            shutil.rmtree(output_dir)
            os.makedirs(output_dir, exist_ok=True)
        else:
            raise FileExistsError(
                f"Output directory '{output_dir}' is not empty, "
                f"use --overwrite to clear it first."
            )

    # Optimization 1: Automatically choose dtype based on actual vocab size (including added tokens)
    # Ensure the maximum possible token ID fits in uint16 (<= 65535)
    dtype = np.uint16 if len(tokenizer) <= 65536 else np.uint32
    DTYPE_MAX = np.iinfo(dtype).max  # 65535 for uint16, 4294967295 for uint32
    if eos_token_id > DTYPE_MAX:
        raise ValueError(
            f"eos_token_id={eos_token_id} exceeds dtype max {DTYPE_MAX}."
        )
    print(f"Vocab size (incl. special tokens): {len(tokenizer)}. Using dtype: {dtype}")

    chunk_token_counts = []  # tokens count for each completed chunk
    chunk_id = 0
    current_chunk_tokens = 0
    
    # Open the first binary file for writing
    current_filepath = os.path.join(output_dir, f"chunk_{chunk_id:06d}.bin")
    f = open(current_filepath, 'wb', buffering=1024 * 1024)

    def write_tokens_to_bin(tokens):
        """Write a 1-D sequence of token ids to file and auto-split by tokens_per_chunk."""
        nonlocal chunk_id, current_chunk_tokens, f

        # If tokens is numpy array, avoid extra conversion
        if isinstance(tokens, np.ndarray):
            arr = tokens.astype(dtype, copy=False)
            idx = 0
            n = arr.size
            while idx < n:
                space_left = tokens_per_chunk - current_chunk_tokens
                end = min(idx + space_left, n)
                f.write(arr[idx:end].tobytes())
                written = end - idx
                current_chunk_tokens += written
                idx = end

                if current_chunk_tokens >= tokens_per_chunk:
                    f.close()
                    chunk_token_counts.append(tokens_per_chunk)
                    chunk_id += 1
                    current_chunk_tokens = 0
                    next_filepath = os.path.join(output_dir, f"chunk_{chunk_id:06d}.bin")
                    f = open(next_filepath, 'wb', buffering=1024 * 1024)
            return

        # Otherwise treat as a Python sequence (list)
        idx = 0
        n = len(tokens)
        while idx < n:
            space_left = tokens_per_chunk - current_chunk_tokens
            end = min(idx + space_left, n)

            np_array = np.asarray(tokens[idx:end], dtype=dtype)
            f.write(np_array.tobytes())
            written = end - idx
            current_chunk_tokens += written
            idx = end

            if current_chunk_tokens >= tokens_per_chunk:
                f.close()
                chunk_token_counts.append(tokens_per_chunk)
                chunk_id += 1
                current_chunk_tokens = 0
                next_filepath = os.path.join(output_dir, f"chunk_{chunk_id:06d}.bin")
                f = open(next_filepath, 'wb', buffering=1024 * 1024)
    
    def _flush_encoded_batch(encoded_batch):
        batch_max = max((max(s) for s in encoded_batch if s), default=-1)
        if batch_max > DTYPE_MAX:
            raise ValueError(
                f"Token id overflow: batch max token id {batch_max} exceeds dtype max {DTYPE_MAX}. "
                f"Current dtype={dtype}. Use uint32 (e.g., force dtype) or a tokenizer with smaller ids."
            )
        total = sum(len(seq) + 1 for seq in encoded_batch)  # +1 for eos
        flat = np.empty(total, dtype=dtype)
        pos = 0
        for seq in encoded_batch:
            L = len(seq)
            if L:
                flat[pos:pos+L] = seq
                pos += L
            flat[pos] = eos_token_id
            pos += 1
        write_tokens_to_bin(flat)

    try:
        text_buffer = []
        
        # Optimization 2: Accumulate a text batch, then send it to the tokenizer at once to leverage internal multithreading
        for i, example in enumerate(tqdm(dataset, desc="Tokenizing to Binary")):
            if max_samples is not None and i >= max_samples:
                break
                
            if args.text_field not in example:
                raise KeyError(
                    f"Example missing field '{args.text_field}'. Available keys: {list(example.keys())}"
                )
            text_buffer.append(example[args.text_field])
            
            # Run batched processing when we have collected batch_size samples
            if len(text_buffer) >= batch_size:
                encoded_batch = tokenizer(text_buffer, add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False)["input_ids"]
                _flush_encoded_batch(encoded_batch)
                text_buffer = [] # Clear buffer

        # After the loop, process any remaining texts that do not make a full batch
        if text_buffer:
            encoded_batch = tokenizer(text_buffer, add_special_tokens=False, return_attention_mask=False, return_token_type_ids=False)["input_ids"]
            _flush_encoded_batch(encoded_batch)

    finally:
        # Close the last file handle
        if not f.closed:
            f.close()
    
    # Decide whether last chunk file is empty by checking file size (robust against state mismatches)
    last_path = os.path.join(output_dir, f"chunk_{chunk_id:06d}.bin")
    last_size = os.path.getsize(last_path) if os.path.exists(last_path) else 0
    if last_size == 0:
        # If the last file is truly empty (e.g., we exactly hit a chunk boundary), delete it
        if os.path.exists(last_path):
            os.remove(last_path)
    else:
        # Only record last chunk token count if file is non-empty
        chunk_token_counts.append(current_chunk_tokens)

    meta = {
        "created_at_unix": int(time.time()),
        "tokenizer_name_or_path": getattr(tokenizer, "name_or_path", None),
        "is_fast": getattr(tokenizer, "is_fast", None),
        "dtype": "uint16" if dtype == np.uint16 else "uint32",
        "eos_token_id": int(eos_token_id),
        "tokens_per_chunk": int(tokens_per_chunk),
        "chunks": [
            {"file": f"chunk_{i:06d}.bin", "tokens": int(n)}
            for i, n in enumerate(chunk_token_counts)
        ],
        "total_tokens": int(sum(chunk_token_counts)),
    }
    with open(os.path.join(output_dir, "meta.json"), "w", encoding="utf-8") as mf:
        json.dump(meta, mf, ensure_ascii=False, indent=2)

def main(args):
    dataset = load_dataset(args.dataset, 
                          name=args.data_name, 
                          split="train", 
                          streaming=True)
                          
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("Tokenizer has no eos_token_id; please set it or pass one explicitly.")

    build_binary_chunks_and_save(
        args,
        dataset, 
        tokenizer, 
        eos_token_id,
        output_dir=args.output_path,
        tokens_per_chunk=int(args.tokens_per_chunk),
        batch_size=args.batch_size, 
        max_samples=args.max_samples
    )
    print(f"Binary token chunks successfully saved to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess FineWeb-Edu dataset into binary chunks")
    parser.add_argument("--tokenizer", type=str, default="gpt2", help="Tokenizer model name or path (e.g., 'gpt2', 'meta-llama/Llama-2-7b-hf')")
    parser.add_argument("--dataset", type=str, default="HuggingFaceFW/fineweb-edu", help="FineWeb-Edu dataset")
    parser.add_argument("--data_name", type=str, default="sample-10BT", help="FineWeb-Edu dataset variant")
    parser.add_argument("--text_field", type=str, default="text", help="Field name containing text in dataset examples (default: 'text').")
    parser.add_argument("--output_path", type=str, default="./data/fineweb-edu-sample-10BT/", help="Output directory for processed binary chunks")
    parser.add_argument("--tokens_per_chunk", type=int, default=100_000_000, help="Number of tokens per chunk (default: 100M)")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for tokenizer (default: 1000)")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to process (for testing/debugging)")
    parser.add_argument("--overwrite", action="store_true", help="If set, allow writing into a non-empty output directory (may overwrite chunks).")
    args = parser.parse_args()
    main(args)