## Getting Started

### Build Steps
1. Install Visual Studio 2022 (Community Edition) or later
2. Run `build.ps1`
3. Run the application from the `build` folder

## Roadmap

### Features
1. Add AVX-optimized matmul for M=1
2. Implement op classes architecture
3. Add pybind11 bindings to run the model from Python
4. Implement 8-bit quantization
5. Add sampling strategies (e.g., implement top-k type ops)
6. Enable `tensor.to(device)` - device can be RAM, GPU, etc.
7. **Build Your LLM**: Make components highly configurable
   - Decoder block can be configured with type of RoPE, attention block, MLP, etc.
   - Requires reading more LLM architecture classes in transformers
   - Goal: Cover all architectures from config
8. Migrate build system from Visual Studio to Ninja
   - Use VS Code for debugging
   - Create a starter's guide for this setup

### Code Quality & Cleanup
1. Correct parameter name formatting in matmul AVX (should be M, K, N order)
2. Standardize function parameter order:
   - Inputs first, then outputs
   - Example: `bool parseString(const char* input, int start_pos, int length, char* output, int* output_length);`
3. Organize CPU ops properly:
   - Use namespaces based on implementation type (naive, AVX2, etc.)
   - Prepare for future implementations
4. Research scaling factor for rotary embedding (used for long context)
5. Investigate if softmax at the end of Qwen3 model is required
6. Document every component (ops, tensor, etc.)
7. Find solution for intermediate tensor memory management

### Performance Optimizations
1. Use MatMul instead of Linear (latency difference ~1.5x)
2. Optimize MLP block by fusing operations:
   - Combine MM gate, MM up, and SiLU operations
3. Write a dedicated MLP kernel:
   - Assert `input.numel() == embed_dim == output_dim`
4. Properly manage `prepare()` calls in Qwen3 model
5. Memory optimization improvements