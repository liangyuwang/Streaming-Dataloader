import os
import gc
import argparse
from tqdm.auto import tqdm
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer


def build_token_chunks_and_save(dataset, tokenizer, eos_token_id,
                                 output_dir, tokens_per_chunk=1000000,
                                 max_samples=None):
    os.makedirs(output_dir, exist_ok=True)

    buffer = []
    chunk_id = 0

    for i, example in enumerate(tqdm(dataset, desc="Tokenizing")):
        if max_samples is not None and i >= max_samples:
            break

        tokens = tokenizer.encode(example["text"], add_special_tokens=False)
        tokens += [eos_token_id]
        buffer.extend(tokens)

        while len(buffer) >= tokens_per_chunk:
            chunk = buffer[:tokens_per_chunk]
            buffer = buffer[tokens_per_chunk:]

            # each shard only has one chunk
            ds = Dataset.from_dict({"input_ids": [chunk]})
            save_path = os.path.join(output_dir, f"chunk_{chunk_id:06d}")
            ds.save_to_disk(save_path, max_shard_size=tokens_per_chunk*64)
            chunk_id += 1

            del ds
            gc.collect()

    if buffer:
        ds = Dataset.from_dict({"input_ids": [buffer]})
        save_path = os.path.join(output_dir, f"chunk_{chunk_id:06d}")
        ds.save_to_disk(save_path)
        del ds
        gc.collect()


def main(args):
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", name=args.data_name, split="train", streaming=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    eos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token or "<|endoftext|>")

    build_token_chunks_and_save(dataset, tokenizer, eos_token_id,
                                output_dir=args.output_path,
                                tokens_per_chunk=int(args.tokens_per_chunk),
                                max_samples=args.max_samples)

    print(f"Token chunks saved to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument("--data_name", type=str, default="sample-10BT")
    parser.add_argument("--output_path", type=str, default="./data/fineweb-edu-sample-10BT/")
    parser.add_argument("--tokens_per_chunk", type=int, default=1e8)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()
    main(args)
