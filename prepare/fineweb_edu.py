import os
import gc
import argparse
from tqdm.auto import tqdm
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer


def build_token_chunks_and_save(dataset, tokenizer, eos_token_id,
                                 output_dir, tokens_per_chunk=1000000,
                                 max_samples=None):
    """
    Process a streaming dataset by tokenizing text and saving it in fixed-size chunks.
    
    This function is the core of the preprocessing pipeline. It:
    1. Streams through the dataset to avoid loading everything into memory
    2. Tokenizes text on-the-fly
    3. Accumulates tokens in a buffer
    4. Saves chunks when buffer reaches target size
    5. Handles memory cleanup to prevent OOM
    
    Args:
        dataset: HuggingFace streaming dataset
        tokenizer: Pre-trained tokenizer for text encoding
        eos_token_id (int): End-of-sequence token ID to add after each text
        output_dir (str): Directory to save processed chunks
        tokens_per_chunk (int): Target number of tokens per chunk
        max_samples (int, optional): Maximum number of samples to process (for testing)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    buffer = []  # Accumulate tokens here before saving
    chunk_id = 0  # Counter for naming chunk directories

    # Stream through dataset samples
    for i, example in enumerate(tqdm(dataset, desc="Tokenizing")):
        # Early termination for testing/debugging
        if max_samples is not None and i >= max_samples:
            break

        # Tokenize the text without special tokens (we'll add EOS manually)
        tokens = tokenizer.encode(example["text"], add_special_tokens=False)
        
        # Add end-of-sequence token to mark document boundaries
        # This is crucial for proper training as it signals where one document ends
        tokens += [eos_token_id]
        
        # Add tokens to buffer
        buffer.extend(tokens)

        # Check if buffer has enough tokens to form a complete chunk
        while len(buffer) >= tokens_per_chunk:
            # Extract exactly tokens_per_chunk tokens for this chunk
            chunk = buffer[:tokens_per_chunk]
            # Keep remaining tokens for next chunk
            buffer = buffer[tokens_per_chunk:]

            # Create HuggingFace Dataset object with the chunk
            # Each chunk contains a single sequence of token IDs
            ds = Dataset.from_dict({"input_ids": [chunk]})
            
            # Save chunk to disk with zero-padded naming for proper sorting
            save_path = os.path.join(output_dir, f"chunk_{chunk_id:06d}")
            ds.save_to_disk(save_path, max_shard_size=tokens_per_chunk*64)
            chunk_id += 1

            # Explicit memory cleanup to prevent accumulation
            del ds
            gc.collect()

    # Handle remaining tokens in buffer (last chunk may be smaller)
    if buffer:
        ds = Dataset.from_dict({"input_ids": [buffer]})
        save_path = os.path.join(output_dir, f"chunk_{chunk_id:06d}")
        ds.save_to_disk(save_path)
        del ds
        gc.collect()


def main(args):
    """
    Main preprocessing pipeline that orchestrates the entire process.
    
    This function:
    1. Loads the streaming dataset from HuggingFace
    2. Initializes the tokenizer
    3. Handles special token configuration
    4. Calls the chunking function
    
    Args:
        args: Parsed command line arguments
    """
    # Load dataset in streaming mode to avoid memory issues
    # Streaming=True means data is downloaded and processed on-demand
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", 
                          name=args.data_name, 
                          split="train", 
                          streaming=True)
    
    # Load pre-trained tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    
    # Get end-of-sequence token ID
    # Different tokenizers use different EOS tokens (e.g., GPT-2 uses <|endoftext|>)
    eos_token_id = tokenizer.convert_tokens_to_ids(
        tokenizer.eos_token or "<|endoftext|>"
    )

    # Process and save the dataset
    build_token_chunks_and_save(
        dataset, 
        tokenizer, 
        eos_token_id,
        output_dir=args.output_path,
        tokens_per_chunk=int(args.tokens_per_chunk),
        max_samples=args.max_samples
    )

    print(f"Token chunks saved to {args.output_path}")


if __name__ == "__main__":
    """
    Command line interface for the preprocessing script.
    
    Example usage:
    python fineweb_edu.py --tokenizer gpt2 --data_name sample-10BT --output_path ./data/processed/
    """
    parser = argparse.ArgumentParser(description="Preprocess FineWeb-Edu dataset for streaming training")
    
    parser.add_argument("--tokenizer", type=str, default="gpt2",
                       help="Tokenizer model name or path (e.g., 'gpt2', 'microsoft/DialoGPT-medium')")
    
    parser.add_argument("--data_name", type=str, default="sample-10BT",
                       help="FineWeb-Edu dataset variant (e.g., 'sample-10BT', 'sample-100BT')")
    
    parser.add_argument("--output_path", type=str, default="./data/fineweb-edu-sample-10BT/",
                       help="Output directory for processed chunks")
    
    parser.add_argument("--tokens_per_chunk", type=int, default=1e8,
                       help="Number of tokens per chunk (default: 100M)")
    
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to process (for testing)")
    
    args = parser.parse_args()
    main(args)
