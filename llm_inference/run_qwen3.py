#!/usr/bin/env python3
"""
Interactive chat interface for Qwen3 models using the InferenceEngine.

Example usage:
    python -m llm_inference.run_qwen3 model.safetensors --tokenizer-path /path/to/model
"""

import argparse
import os
import sys
import time

try:
    import numpy as np
except ImportError:
    print("Error: numpy is required. Install with: pip install numpy")
    sys.exit(1)

try:
    from llm_inference import InferenceEngine
    from llm_inference import Qwen3ChatInterface
    from llm_inference.chat_interface import VerbosityLevel
except ImportError as e:
    print(f"Error: Failed to import llm_inference package: {e}")
    print("Make sure the package is built and available in the Python path.")
    print("Build the package by running the CMake build process.")
    print(f"Checked paths: {sys.path[:5]}")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Interactive chat interface for Qwen3 models using InferenceEngine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start interactive chat:
  python -m llm_inference.run_qwen3 model.safetensors --tokenizer-path /path/to/model

  # Custom max tokens per response:
  python -m llm_inference.run_qwen3 model.safetensors --tokenizer-path /path/to/model --max-new-tokens 256

  # Disable memory mapping:
  python -m llm_inference.run_qwen3 model.safetensors --tokenizer-path /path/to/model --no-mmap

  # Specify model name:
  python -m llm_inference.run_qwen3 model.safetensors --tokenizer-path /path/to/model --model-name Qwen3-1.7B
        """
    )
    
    parser.add_argument(
        "safetensors_path",
        type=str,
        help="Path to the model safetensors file"
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        required=True,
        help="Path to tokenizer (required for text input/output)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen3-1.7B",
        help="Model name (e.g., Qwen3-1.7B, default: Qwen3-1.7B)"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate per response (default: 512)"
    )
    parser.add_argument(
        "--no-mmap",
        action="store_true",
        help="Disable memory mapping (default: use mmap)"
    )
    parser.add_argument(
        "--verbosity",
        type=str,
        choices=["error", "warning", "info", "debug"],
        default="info",
        help="Verbosity level: error (only errors), warning (errors+warnings), info (errors+warnings+info), debug (all messages). Default: info"
    )
    
    args = parser.parse_args()
    
    # Convert verbosity string to VerbosityLevel enum
    verbosity_map = {
        "error": VerbosityLevel.ERROR,
        "warning": VerbosityLevel.WARNING,
        "info": VerbosityLevel.INFO,
        "debug": VerbosityLevel.DEBUG
    }
    verbosity = verbosity_map[args.verbosity.lower()]
    
    # Check if safetensors file exists
    if not os.path.exists(args.safetensors_path):
        print(f"Error: Safetensors file not found: {args.safetensors_path}", file=sys.stderr)
        sys.exit(1)
    
    # Check if tokenizer path exists
    if not os.path.exists(args.tokenizer_path):
        print(f"Error: Tokenizer path not found: {args.tokenizer_path}", file=sys.stderr)
        sys.exit(1)
    
    # Load tokenizer
    try:
        from transformers import AutoTokenizer
        if verbosity >= VerbosityLevel.INFO:
            print(f"Loading tokenizer from: {args.tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        if verbosity >= VerbosityLevel.INFO:
            print("Tokenizer loaded successfully.")
    except ImportError:
        print("Error: transformers library is required.", file=sys.stderr)
        print("Install with: pip install transformers", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error loading tokenizer: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Create and load model (config is managed internally by the model)
    if verbosity >= VerbosityLevel.INFO:
        print(f"\nCreating InferenceEngine for model: {args.model_name}")
        print(f"Loading model from: {args.safetensors_path}")
    load_start = time.time()
    model = InferenceEngine(args.model_name)
    model.load_weights(args.safetensors_path, use_mmap=not args.no_mmap)
    load_time = time.time() - load_start
    if verbosity >= VerbosityLevel.INFO:
        print(f"Model loaded in {load_time:.3f} seconds\n")
    
    # Create Qwen3 chat interface and start interactive loop
    chat = Qwen3ChatInterface(model, tokenizer, max_new_tokens=args.max_new_tokens, verbosity=verbosity)
    chat.chat_loop()


if __name__ == "__main__":
    main()

