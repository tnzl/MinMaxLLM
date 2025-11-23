"""
llm_inference - Python bindings for MinMaxLLM inference engine.

This package provides Python bindings for the C++ inference engine,
allowing you to run LLM inference from Python.
"""

__version__ = "0.1.0"

# Import the Python wrapper (which handles the C++ extension import)
try:
    from .inference_engine import InferenceEngine
    from .chat_interface import ChatInterface, Qwen3ChatInterface, VerbosityLevel
    
    __all__ = [
        'InferenceEngine',
        'ChatInterface',
        'Qwen3ChatInterface',
        'VerbosityLevel',
    ]
except ImportError as e:
    import warnings
    warnings.warn(
        f"Failed to import llm_inference: {e}\n"
        "Make sure the C++ extension is built and available in the Python path.",
        ImportWarning
    )
    __all__ = []

