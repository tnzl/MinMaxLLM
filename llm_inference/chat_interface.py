"""
Base chat interface classes for LLM inference.

This module provides base classes for implementing chat interfaces
for different model families.
"""

import sys
import time
from abc import ABC, abstractmethod
from enum import IntEnum
from typing import List, Dict, Tuple, Optional


class VerbosityLevel(IntEnum):
    """Verbosity levels for controlling output."""
    ERROR = 0      # Only errors
    WARNING = 1    # Errors and warnings
    INFO = 2       # Errors, warnings, and info messages
    DEBUG = 3      # All messages including debug details


class ChatInterface(ABC):
    """
    Base class for interactive chat interfaces.
    
    This abstract base class defines the interface for chat interactions
    with LLM models. Subclasses should implement model-specific behavior.
    """
    
    def __init__(self, model, tokenizer, max_new_tokens: int = 512, verbosity: VerbosityLevel = VerbosityLevel.INFO):
        """
        Initialize the chat interface.
        
        Args:
            model: The inference engine model instance
            tokenizer: The tokenizer instance (from transformers)
            max_new_tokens: Maximum number of tokens to generate per response
            verbosity: Verbosity level for controlling output (default: INFO)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.verbosity = verbosity
        self.conversation_history: List[Dict[str, str]] = []
        self.processed_token_count = 0
        self.generated_token_ids: List[int] = []
    
    @abstractmethod
    def get_bos_token_id(self) -> Optional[int]:
        """
        Get the beginning-of-sequence token ID.
        
        Returns:
            BOS token ID or None if not available
        """
        pass
    
    @abstractmethod
    def get_eos_token_id(self) -> Optional[int]:
        """
        Get the end-of-sequence token ID.
        
        Returns:
            EOS token ID or None if not available
        """
        pass
    
    def _should_print(self, level: VerbosityLevel) -> bool:
        """Check if a message at the given verbosity level should be printed."""
        return self.verbosity >= level
    
    def _print_error(self, message: str, file=sys.stderr, **kwargs):
        """Print an error message."""
        if self._should_print(VerbosityLevel.ERROR):
            print(f"[ERROR] {message}", file=file, **kwargs)
    
    def _print_warning(self, message: str, file=sys.stderr, **kwargs):
        """Print a warning message."""
        if self._should_print(VerbosityLevel.WARNING):
            print(f"[WARNING] {message}", file=file, **kwargs)
    
    def _print_info(self, message: str, file=sys.stdout, **kwargs):
        """Print an info message."""
        if self._should_print(VerbosityLevel.INFO):
            print(message, file=file, **kwargs)
    
    def _print_debug(self, message: str, file=sys.stderr, **kwargs):
        """Print a debug message."""
        if self._should_print(VerbosityLevel.DEBUG):
            print(f"[DEBUG] {message}", file=file, **kwargs)
    
    def encode_conversation(self, add_generation_prompt: bool = True) -> List[int]:
        """
        Encode the full conversation using the tokenizer with chat template.
        
        Args:
            add_generation_prompt: Whether to add generation prompt
            
        Returns:
            List of token IDs
        """
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
    
    def process_and_generate(self, user_message: str) -> Tuple[str, Dict]:
        """
        Process user message and generate response with streaming output.
        
        Args:
            user_message: The user's message
            
        Returns:
            Tuple of (response_text, timing_info)
        """
        # Add user message to conversation history
        self.conversation_history.append({"role": "user", "content": user_message})
        
        # Encode the full conversation (with generation prompt)
        encode_start = time.time()
        input_tokens = self.encode_conversation(add_generation_prompt=True)
        encode_time = time.time() - encode_start
        
        # Process only new tokens (those after what we've already processed)
        if self.processed_token_count >= len(input_tokens):
            self._print_warning(f"processed_token_count ({self.processed_token_count}) >= total tokens ({len(input_tokens)}), resetting to 0")
            self.processed_token_count = 0
        
        new_tokens = input_tokens[self.processed_token_count:]
        num_new_tokens = len(new_tokens)
        
        # Debug: Log token counts
        self._print_debug(f"processed_token_count={self.processed_token_count}, total_input_tokens={len(input_tokens)}, new_tokens={num_new_tokens}")
        
        # Process all but the last new token as prompt tokens
        prompt_start = time.time()
        token_times = []
        for i, token_id in enumerate(new_tokens[:-1]):
            token_start = time.time()
            self.model.process_prompt_token(token_id)
            token_time = time.time() - token_start
            token_times.append(token_time)
            if i < 3:
                self._print_debug(f"Prompt token {i}: {token_time*1000:.2f} ms")
        prompt_time = time.time() - prompt_start
        
        # Update processed token count
        num_prompt_tokens_processed = len(new_tokens) - 1 if new_tokens else 0
        self.processed_token_count = len(input_tokens) - 1
        
        self._print_debug(f"After prompt processing: processed {num_prompt_tokens_processed} new prompt tokens, processed_token_count={self.processed_token_count}")
        
        # Generate response starting from the last token
        if not new_tokens:
            if input_tokens:
                current_token = input_tokens[-1]
            else:
                current_token = self.get_bos_token_id() or 0
        else:
            current_token = new_tokens[-1]
        
        generated_tokens = []
        generated_text_parts = []
        generation_start = time.time()
        first_token_time = None
        
        # Import numpy here to avoid requiring it at module level
        import numpy as np
        
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
            self.generated_token_ids.append(next_token)
            
            # Decode and print token immediately (streaming)
            token_text = self.tokenizer.decode([next_token], skip_special_tokens=True)
            generated_text_parts.append(token_text)
            # Always print generated tokens (streaming output) regardless of verbosity
            print(token_text, end="", flush=True)
            
            # Check for EOS token
            eos_token_id = self.get_eos_token_id()
            if eos_token_id is not None and next_token == eos_token_id:
                break
            
            current_token = next_token
        
        generation_time = time.time() - generation_start
        
        # Combine all generated text
        response_text = "".join(generated_text_parts)
        
        # Add assistant response to history
        self.conversation_history.append({"role": "assistant", "content": response_text})
        
        # Update processed token count
        conversation_tokens = self.encode_conversation(add_generation_prompt=False)
        self.processed_token_count = len(conversation_tokens)
        
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
    
    def reset(self) -> None:
        """Reset the conversation and model cache."""
        self.model.reset_cache()
        self.conversation_history = []
        self.processed_token_count = 0
        self.generated_token_ids = []
        self._print_info("Conversation reset.")
    
    def chat_loop(self) -> None:
        """Main interactive chat loop."""
        # Ensure clean output before starting
        sys.stdout.flush()
        sys.stderr.flush()
        
        self._print_info("\n" + "="*60)
        self._print_info("Interactive Chat Interface")
        self._print_info("="*60)
        self._print_info("Type your messages and press Enter to get a response.")
        self._print_info("Commands:")
        self._print_info("  /reset  - Reset the conversation")
        self._print_info("  /quit   - Exit the chat")
        self._print_info("="*60 + "\n")
        
        while True:
            try:
                # Ensure clean line before input prompt
                sys.stdout.flush()
                sys.stderr.flush()
                # Get user input
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() == "/quit":
                    self._print_info("Goodbye!")
                    break
                elif user_input.lower() == "/reset":
                    self.reset()
                    continue
                
                # Generate response
                self._print_info("Assistant: ", end="", flush=True)
                response, timing = self.process_and_generate(user_input)
                
                # Ensure response ends with newline for clean output
                if response and not response.endswith('\n'):
                    print()  # Add newline after response
                
                # Print detailed timing breakdown (INFO level)
                if self._should_print(VerbosityLevel.INFO):
                    print("\n" + "-" * 60)
                    print("Timing Breakdown:")
                    print(f"  Encoding time:     {timing['encode_time']*1000:.2f} ms")
                    if timing['num_prompt_tokens'] > 0:
                        avg_prompt_time = timing['prompt_time'] / timing['num_prompt_tokens']
                        print(f"  Prompt processing: {timing['prompt_time']*1000:.2f} ms ({timing['num_prompt_tokens']} tokens, {avg_prompt_time*1000:.2f} ms/token)")
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
                self._print_info("\n\nInterrupted. Type /quit to exit or continue chatting.")
            except EOFError:
                self._print_info("\nGoodbye!")
                break
            except Exception as e:
                self._print_error(f"\nError: {e}\n")
                if self._should_print(VerbosityLevel.DEBUG):
                    import traceback
                    traceback.print_exc()


class Qwen3ChatInterface(ChatInterface):
    """
    Chat interface implementation for Qwen3 models.
    
    This class provides Qwen3-specific token ID handling and chat formatting.
    """
    
    def __init__(self, model, tokenizer, max_new_tokens: int = 512, verbosity: VerbosityLevel = VerbosityLevel.INFO):
        """
        Initialize Qwen3 chat interface.
        
        Args:
            model: The InferenceEngine model instance
            tokenizer: The tokenizer instance (from transformers)
            max_new_tokens: Maximum number of tokens to generate per response
            verbosity: Verbosity level for controlling output (default: INFO)
        """
        super().__init__(model, tokenizer, max_new_tokens, verbosity)
        # Cache token IDs from tokenizer
        self._bos_token_id = self._get_token_id_from_tokenizer('bos_token_id')
        self._eos_token_id = self._get_token_id_from_tokenizer('eos_token_id')
    
    def _get_token_id_from_tokenizer(self, attr_name: str) -> Optional[int]:
        """
        Get token ID from tokenizer attribute.
        
        Args:
            attr_name: Name of the tokenizer attribute
            
        Returns:
            Token ID or None if not available
        """
        token_id = getattr(self.tokenizer, attr_name, None)
        if token_id is None:
            # Try getting from tokenizer config
            if hasattr(self.tokenizer, 'tokenizer') and hasattr(self.tokenizer.tokenizer, attr_name):
                token_id = getattr(self.tokenizer.tokenizer, attr_name)
        # Handle case where token_id might be a string
        if isinstance(token_id, str):
            # Try to convert using tokenizer
            try:
                token_id = self.tokenizer.convert_tokens_to_ids([token_id])[0]
            except Exception:
                return None
        return token_id
    
    def get_bos_token_id(self) -> Optional[int]:
        """Get the beginning-of-sequence token ID for Qwen3."""
        return self._bos_token_id
    
    def get_eos_token_id(self) -> Optional[int]:
        """Get the end-of-sequence token ID for Qwen3."""
        return self._eos_token_id
