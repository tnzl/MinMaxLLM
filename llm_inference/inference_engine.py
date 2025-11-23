"""
Python wrapper for the C++ InferenceEngine bindings.

This module provides a more Pythonic interface to the C++ inference engine,
with additional utilities and convenience methods.
"""

import numpy as np
from typing import Optional, List, Union
import sys
import importlib.util
from pathlib import Path

# Import the compiled C++ extension module
# We need to explicitly load the .pyd/.so file to avoid naming conflict with this Python module
def _load_cpp_extension():
    """Load the compiled C++ extension module, avoiding the Python file with the same name."""
    package_dir = Path(__file__).parent
    
    # Determine extension file name based on platform
    if sys.platform == "win32":
        ext_file = package_dir / "inference_engine.pyd"
        # Also try versioned .pyd files (e.g., inference_engine.cp312-win_amd64.pyd)
        for pyd_file in package_dir.glob("inference_engine.*.pyd"):
            ext_file = pyd_file
            break
    else:
        ext_file = package_dir / "inference_engine.so"
    
    # Try loading the compiled extension directly
    if ext_file.exists():
        try:
            spec = importlib.util.spec_from_file_location("_inference_engine_cpp", ext_file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module
        except Exception:
            pass
    
    # Fallback: try importing as installed module
    # This will work if the extension is in site-packages or on sys.path
    try:
        # Temporarily remove current module from sys.modules to avoid circular import
        current_module = sys.modules.get('llm_inference.inference_engine')
        if current_module:
            del sys.modules['llm_inference.inference_engine']
        
        # Try importing the compiled extension
        import inference_engine
        # Verify it's the C++ extension (check if it has the C++ class and is not our Python file)
        if hasattr(inference_engine, 'InferenceEngine'):
            # Check if it's a compiled extension (no __file__ or __file__ points to .pyd/.so)
            if not hasattr(inference_engine, '__file__') or (
                hasattr(inference_engine, '__file__') and 
                (inference_engine.__file__.endswith('.pyd') or inference_engine.__file__.endswith('.so'))
            ):
                return inference_engine
    except ImportError:
        pass
    finally:
        # Restore current module
        if current_module:
            sys.modules['llm_inference.inference_engine'] = current_module
    
    raise ImportError(
        "Could not find the compiled inference_engine extension module.\n"
        f"Expected to find it at: {ext_file}\n"
        "Please build the project using CMake to generate the extension module."
    )

try:
    _cext = _load_cpp_extension()
except ImportError as e:
    raise ImportError(
        "Could not import the compiled inference_engine extension.\n"
        "Please build the project using CMake to generate the extension module."
    ) from e


class InferenceEngine:
    """
    Python wrapper for the C++ InferenceEngine.
    
    This provides a more Pythonic interface with additional convenience methods.
    """
    
    def __init__(self, model_name: str):
        """
        Initialize the inference engine.
        
        Args:
            model_name: Name of the model (e.g., 'Qwen3-1.7B', 'qwen3')
        """
        self._engine = _cext.InferenceEngine(model_name)
        self._model_name = model_name
    
    def load_weights(self, safetensor_path: str, use_mmap: bool = True) -> None:
        """
        Load model weights from a safetensors file.
        
        Args:
            safetensor_path: Path to the safetensors file
            use_mmap: Whether to use memory mapping (default: True)
        """
        self._engine.load_weights(safetensor_path, use_mmap)
    
    def reset_cache(self) -> None:
        """Reset the KV cache."""
        self._engine.reset_cache()
    
    def process_prompt_token(self, token_id: int) -> None:
        """
        Process a single prompt token (without computing logits).
        
        This is used for efficient prompt processing where we don't need
        logits for each token, only for the final token.
        
        Args:
            token_id: Token ID to process
        """
        self._engine.process_prompt_token(token_id)
    
    def process_prompt_tokens(self, token_ids: Union[List[int], np.ndarray]) -> None:
        """
        Process multiple prompt tokens efficiently.
        
        Args:
            token_ids: List or array of token IDs to process
        """
        if isinstance(token_ids, np.ndarray):
            token_ids = token_ids.tolist()
        
        for token_id in token_ids:
            self.process_prompt_token(token_id)
    
    def predict_next_token(self, token_id: int) -> np.ndarray:
        """
        Predict next token logits for the given token ID.
        
        Args:
            token_id: Token ID to predict from
            
        Returns:
            NumPy array of logits (vocab_size,)
        """
        return self._engine.predict_next_token(token_id)
    
    def generate_token(self, token_id: int, sampling_strategy: str = "greedy", 
                      temperature: float = 1.0, top_k: Optional[int] = None,
                      top_p: Optional[float] = None) -> int:
        """
        Generate the next token using the specified sampling strategy.
        
        Args:
            token_id: Current token ID
            sampling_strategy: Sampling strategy ('greedy', 'top_k', 'top_p')
            temperature: Temperature for sampling (default: 1.0)
            top_k: Top-k sampling parameter (optional)
            top_p: Top-p (nucleus) sampling parameter (optional)
            
        Returns:
            Generated token ID
        """
        logits = self.predict_next_token(token_id)
        
        if sampling_strategy == "greedy":
            return int(np.argmax(logits))
        elif sampling_strategy == "top_k":
            if top_k is None:
                raise ValueError("top_k must be specified for top_k sampling")
            # Get top-k logits
            top_k_indices = np.argsort(logits)[-top_k:]
            top_k_logits = logits[top_k_indices]
            # Apply temperature
            if temperature != 1.0:
                top_k_logits = top_k_logits / temperature
            # Sample from top-k
            probs = np.exp(top_k_logits - np.max(top_k_logits))
            probs = probs / probs.sum()
            selected_idx = np.random.choice(len(top_k_indices), p=probs)
            return int(top_k_indices[selected_idx])
        elif sampling_strategy == "top_p":
            if top_p is None:
                raise ValueError("top_p must be specified for top_p sampling")
            # Sort logits
            sorted_indices = np.argsort(logits)[::-1]
            sorted_logits = logits[sorted_indices]
            # Apply temperature
            if temperature != 1.0:
                sorted_logits = sorted_logits / temperature
            # Compute probabilities
            probs = np.exp(sorted_logits - np.max(sorted_logits))
            probs = probs / probs.sum()
            # Cumulative sum
            cumsum_probs = np.cumsum(probs)
            # Find top-p threshold
            mask = cumsum_probs <= top_p
            if not mask.any():
                mask[0] = True  # At least one token
            # Sample from top-p
            filtered_probs = probs[mask]
            filtered_probs = filtered_probs / filtered_probs.sum()
            selected_idx = np.random.choice(np.sum(mask), p=filtered_probs)
            return int(sorted_indices[mask][selected_idx])
        else:
            raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")
    
    @property
    def tokens_processed(self) -> int:
        """Get the number of tokens processed so far."""
        return self._engine.tokens_processed
    
    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._engine.model_name
    
    @property
    def _cxx_engine(self):
        """Get the underlying C++ engine object (for internal use)."""
        return self._engine
    
    def __repr__(self) -> str:
        return f"InferenceEngine(model_name='{self.model_name}', tokens_processed={self.tokens_processed})"

