#!/usr/bin/env python3
"""
Interactive chat interface for Qwen3Model using the qwen3model bindings.
Supports conversational interactions with the model.
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Add the bindings directory and its subdirectories to Python path
# This allows importing qwen3model regardless of where the script is run from
_script_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(_script_dir))
# Also check for build output directories
for subdir in ['RelWithDebInfo', 'Release', 'Debug', 'build']:
    build_path = _script_dir / subdir
    if build_path.exists():
        sys.path.insert(0, str(build_path))

try:
    import numpy as np
except ImportError:
    print("Error: numpy is required. Install with: pip install numpy")
    sys.exit(1)

try:
    import qwen3model
except ImportError as e:
    print(f"Error: Failed to import qwen3model module: {e}")
    print("Make sure the qwen3model.pyd (Windows) or qwen3model.so (Linux) is in the bindings/ directory")
    print("or in your Python path.")
    print(f"Checked paths: {sys.path[:5]}")
    sys.exit(1)


class ChatInterface:
    """Interactive chat interface for Qwen3Model."""
    
    def __init__(self, model, tokenizer, config, max_new_tokens=512):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.max_new_tokens = max_new_tokens
        self.conversation_history = []
        self.processed_token_count = 0
        # Store the actual token IDs that were generated to verify re-encoding matches
        self.generated_token_ids = []
    
    def encode_conversation(self, add_generation_prompt=True):
        """Encode the full conversation using the tokenizer with chat template."""
        try:
            # Apply chat template to entire conversation
            formatted_text = self.tokenizer.apply_chat_template(
                self.conversation_history,
                tokenize=False,
                add_generation_prompt=add_generation_prompt
            )
        except Exception:
            # If chat template fails, format manually
            formatted_parts = []
            for msg in self.conversation_history:
                role = msg["role"]
                content = msg["content"]
                if role == "user":
                    formatted_parts.append(f"User: {content}")
                else:
                    formatted_parts.append(f"Assistant: {content}")
            formatted_text = "\n".join(formatted_parts)
            if add_generation_prompt:
                formatted_text += "\nAssistant: "
        
        # Tokenize
        inputs = self.tokenizer([formatted_text], return_tensors="pt")
        token_ids = inputs.input_ids[0].tolist()
        return token_ids
    
    def process_and_generate(self, user_message):
        """Process user message and generate response with streaming output."""
        # Add user message to conversation history
        self.conversation_history.append({"role": "user", "content": user_message})
        
        # Encode the full conversation (with generation prompt)
        encode_start = time.time()
        input_tokens = self.encode_conversation(add_generation_prompt=True)
        encode_time = time.time() - encode_start
        
        # Process only new tokens (those after what we've already processed)
        # Safety check: if processed_token_count is >= len(input_tokens), something is wrong
        if self.processed_token_count >= len(input_tokens):
            print(f"[WARNING] processed_token_count ({self.processed_token_count}) >= total tokens ({len(input_tokens)}), resetting to 0", file=sys.stderr)
            self.processed_token_count = 0
        
        new_tokens = input_tokens[self.processed_token_count:]
        num_new_tokens = len(new_tokens)
        
        # Debug: Log token counts to track the issue
        print(f"[DEBUG] processed_token_count={self.processed_token_count}, total_input_tokens={len(input_tokens)}, new_tokens={num_new_tokens}", file=sys.stderr)
        if num_new_tokens > 0:
            # Decode first few new tokens to see what we're processing
            first_few = new_tokens[:min(5, len(new_tokens))]
            decoded_preview = self.tokenizer.decode(first_few, skip_special_tokens=False)
            print(f"[DEBUG] First few new tokens decoded: {repr(decoded_preview[:100])}", file=sys.stderr)
            # Also show what tokens we're skipping
            if self.processed_token_count > 0:
                skipped_tokens = input_tokens[:self.processed_token_count]
                if len(skipped_tokens) > 0:
                    last_skipped = skipped_tokens[-min(3, len(skipped_tokens)):]
                    decoded_skipped = self.tokenizer.decode(last_skipped, skip_special_tokens=False)
                    print(f"[DEBUG] Last few skipped tokens decoded: {repr(decoded_skipped[:100])}", file=sys.stderr)
        else:
            print(f"[WARNING] No new tokens to process! This might cause issues.", file=sys.stderr)
        
        # Process all but the last new token as prompt tokens
        prompt_start = time.time()
        token_times = []
        for i, token_id in enumerate(new_tokens[:-1]):
            token_start = time.time()
            self.model.process_prompt_token(token_id)
            token_time = time.time() - token_start
            token_times.append(token_time)
            # Log first few tokens to identify initialization overhead
            if i < 3:
                print(f"[DEBUG] Prompt token {i}: {token_time*1000:.2f} ms", file=sys.stderr)
        prompt_time = time.time() - prompt_start
        
        # Update processed token count: we've processed all tokens up to (but not including) the last one
        # The last token will be used as the starting point for generation
        # We need to use the absolute position in the current encoding to ensure accuracy
        num_prompt_tokens_processed = len(new_tokens) - 1 if new_tokens else 0
        self.processed_token_count = len(input_tokens) - 1
        
        print(f"[DEBUG] After prompt processing: processed {num_prompt_tokens_processed} new prompt tokens, processed_token_count={self.processed_token_count}", file=sys.stderr)
        
        # Generate response starting from the last token
        if not new_tokens:
            # This shouldn't happen in normal flow, but handle it
            print(f"[ERROR] No new tokens to generate from! Using last token from input.", file=sys.stderr)
            if input_tokens:
                current_token = input_tokens[-1]
            else:
                current_token = self.config.bos_token_id
        else:
            current_token = new_tokens[-1]
        generated_tokens = []
        generated_text_parts = []
        
        generation_start = time.time()
        first_token_time = None
        
        for i in range(self.max_new_tokens):
            # Get logits and sample next token
            token_start = time.time()
            logits = self.model.predict_next_token(current_token)
            next_token = int(np.argmax(logits))
            token_time = time.time() - token_start
            
            # Record time to first token
            if first_token_time is None:
                first_token_time = time.time() - generation_start
            
            generated_tokens.append(next_token)
            self.generated_token_ids.append(next_token)  # Track for verification
            
            # Decode and print token immediately (streaming)
            token_text = self.tokenizer.decode([next_token], skip_special_tokens=True)
            generated_text_parts.append(token_text)
            print(token_text, end="", flush=True)
            
            # Check for EOS token
            if next_token == self.config.eos_token_id:
                break
            
            current_token = next_token
        
        generation_time = time.time() - generation_start
        
        # Combine all generated text
        response_text = "".join(generated_text_parts)
        
        # Add assistant response to history
        self.conversation_history.append({"role": "assistant", "content": response_text})
        
        # Update processed token count to include the generated tokens
        # The key insight: we need to track tokens consistently. After generation:
        # 1. We've processed all input tokens (up to len(input_tokens) - 1)
        # 2. We've generated len(generated_tokens) tokens
        # 3. The model's internal state now has all these tokens in its KV cache
        #
        # For the next turn, we'll encode the full conversation WITH generation prompt.
        # To know how many tokens to skip, we encode the conversation WITHOUT generation prompt
        # (which represents what's actually in the conversation), and that's our baseline.
        # Next time, encoding WITH generation prompt will add exactly the generation prompt tokens.
        conversation_tokens = self.encode_conversation(add_generation_prompt=False)
        
        # Verify that re-encoding the generated text matches the actual generated tokens
        # This helps catch tokenization mismatches
        if self.generated_token_ids:
            # Find where the assistant response starts in the conversation tokens
            # We'll look for a match of the generated tokens
            generated_text_tokens = self.tokenizer.encode(response_text, add_special_tokens=False)
            if generated_text_tokens != self.generated_token_ids:
                print(f"[WARNING] Tokenization mismatch! Generated {len(self.generated_token_ids)} tokens, but re-encoding gives {len(generated_text_tokens)} tokens", file=sys.stderr)
                print(f"[DEBUG] First 10 generated: {self.generated_token_ids[:10]}", file=sys.stderr)
                print(f"[DEBUG] First 10 re-encoded: {generated_text_tokens[:10]}", file=sys.stderr)
        
        # Use the conversation token count as our baseline
        self.processed_token_count = len(conversation_tokens)
        
        print(f"[DEBUG] After generation: conversation_tokens={len(conversation_tokens)}, processed_token_count={self.processed_token_count}", file=sys.stderr)
        # Verify: next encoding with generation prompt should have more tokens
        verify_tokens_with_prompt = self.encode_conversation(add_generation_prompt=True)
        print(f"[DEBUG] Verification: encoding with generation_prompt has {len(verify_tokens_with_prompt)} tokens, diff={len(verify_tokens_with_prompt) - len(conversation_tokens)}", file=sys.stderr)
        
        # If the difference is unexpected, warn
        if len(verify_tokens_with_prompt) <= len(conversation_tokens):
            print(f"[WARNING] Token count mismatch! This might cause issues on next turn.", file=sys.stderr)
        
        # Clear generated tokens for next turn
        self.generated_token_ids = []
        
        # Return timing information
        num_generated = len(generated_tokens)
        num_prompt_tokens = len(new_tokens) - 1 if new_tokens else 0
        timing_info = {
            "encode_time": encode_time,
            "prompt_time": prompt_time,
            "generation_time": generation_time,
            "first_token_time": first_token_time if first_token_time is not None else 0.0,
            "num_generated": num_generated,
            "num_prompt_tokens": num_prompt_tokens,
            "token_times": token_times if num_prompt_tokens > 0 else []
        }
        
        return response_text, timing_info
    
    def reset(self):
        """Reset the conversation and model cache."""
        self.model.reset_cache()
        self.conversation_history = []
        self.processed_token_count = 0
        self.generated_token_ids = []
        print("Conversation reset.")
    
    def chat_loop(self):
        """Main interactive chat loop."""
        print("\n" + "="*60)
        print("Interactive Chat Interface")
        print("="*60)
        print("Type your messages and press Enter to get a response.")
        print("Commands:")
        print("  /reset  - Reset the conversation")
        print("  /quit   - Exit the chat")
        print("="*60 + "\n")
        
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() == "/quit":
                    print("Goodbye!")
                    break
                elif user_input.lower() == "/reset":
                    self.reset()
                    continue
                
                # Generate response
                print("Assistant: ", end="", flush=True)
                response, timing = self.process_and_generate(user_input)
                
                # Print detailed timing breakdown
                print("\n" + "-" * 60)
                print("Timing Breakdown:")
                print(f"  Encoding time:     {timing['encode_time']*1000:.2f} ms")
                if timing['num_prompt_tokens'] > 0:
                    avg_prompt_time = timing['prompt_time'] / timing['num_prompt_tokens']
                    print(f"  Prompt processing: {timing['prompt_time']*1000:.2f} ms ({timing['num_prompt_tokens']} tokens, {avg_prompt_time*1000:.2f} ms/token)")
                    # Show first token time if available (likely has initialization overhead)
                    if timing.get('token_times') and len(timing['token_times']) > 0:
                        first_token_time = timing['token_times'][0] * 1000
                        if len(timing['token_times']) > 1:
                            avg_rest = sum(timing['token_times'][1:]) / (len(timing['token_times']) - 1) * 1000
                            print(f"    - First token:   {first_token_time:.2f} ms (may include initialization)")
                            print(f"    - Rest avg:      {avg_rest:.2f} ms/token")
                else:
                    print(f"  Prompt processing: {timing['prompt_time']*1000:.2f} ms (0 tokens)")
                print(f"  Token generation:  {timing['generation_time']*1000:.2f} ms ({timing['num_generated']} tokens)")
                print(f"  Time to first token: {timing['first_token_time']*1000:.2f} ms")
                if timing['num_generated'] > 0:
                    print(f"  Avg per token:     {timing['generation_time']/timing['num_generated']*1000:.2f} ms/token")
                print(f"  Total time:        {(timing['encode_time'] + timing['prompt_time'] + timing['generation_time'])*1000:.2f} ms")
                print("-" * 60 + "\n")
                
            except KeyboardInterrupt:
                print("\n\nInterrupted. Type /quit to exit or continue chatting.")
            except EOFError:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}\n")
                import traceback
                traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Interactive chat interface for Qwen3Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start interactive chat:
  python run_inference.py model.safetensors --tokenizer-path /path/to/model

  # Custom max tokens per response:
  python run_inference.py model.safetensors --tokenizer-path /path/to/model --max-new-tokens 256

  # Disable memory mapping:
  python run_inference.py model.safetensors --tokenizer-path /path/to/model --no-mmap
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
        "--config-hidden-size",
        type=int,
        default=2048,
        help="Model hidden size (default: 2048)"
    )
    parser.add_argument(
        "--config-intermediate-size",
        type=int,
        default=6144,
        help="Model intermediate size (default: 6144)"
    )
    parser.add_argument(
        "--config-max-position-embeddings",
        type=int,
        default=40960,
        help="Maximum position embeddings (default: 40960)"
    )
    parser.add_argument(
        "--config-num-attention-heads",
        type=int,
        default=16,
        help="Number of attention heads (default: 16)"
    )
    parser.add_argument(
        "--config-num-hidden-layers",
        type=int,
        default=28,
        help="Number of hidden layers (default: 28)"
    )
    parser.add_argument(
        "--config-num-key-value-heads",
        type=int,
        default=8,
        help="Number of key-value heads (default: 8)"
    )
    parser.add_argument(
        "--config-vocab-size",
        type=int,
        default=151936,
        help="Vocabulary size (default: 151936)"
    )
    parser.add_argument(
        "--config-bos-token-id",
        type=int,
        default=151643,
        help="BOS token ID (default: 151643)"
    )
    parser.add_argument(
        "--config-eos-token-id",
        type=int,
        default=151645,
        help="EOS token ID (default: 151645)"
    )
    
    args = parser.parse_args()
    
    # Check if safetensors file exists
    if not os.path.exists(args.safetensors_path):
        print(f"Error: Safetensors file not found: {args.safetensors_path}")
        sys.exit(1)
    
    # Check if tokenizer path exists
    if not os.path.exists(args.tokenizer_path):
        print(f"Error: Tokenizer path not found: {args.tokenizer_path}")
        sys.exit(1)
    
    # Load tokenizer
    try:
        from transformers import AutoTokenizer
        print(f"Loading tokenizer from: {args.tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        print("Tokenizer loaded successfully.")
    except ImportError:
        print("Error: transformers library is required.")
        print("Install with: pip install transformers")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        sys.exit(1)
    
    # Create model config (using defaults, can be extended with command-line args if needed)
    config = qwen3model.Qwen3Config()
    
    # Create and load model
    print(f"\nLoading model from: {args.safetensors_path}")
    load_start = time.time()
    model = qwen3model.Qwen3Model(config)
    model.load_weights(args.safetensors_path, use_mmap=not args.no_mmap)
    load_time = time.time() - load_start
    print(f"Model loaded in {load_time:.3f} seconds\n")
    
    # Create chat interface and start interactive loop
    chat = ChatInterface(model, tokenizer, config, max_new_tokens=args.max_new_tokens)
    chat.chat_loop()


if __name__ == "__main__":
    main()

