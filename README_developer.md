## Roadmap

### Top TODOs :
0. Review py bind and run inference .py script properly
   - separate chat logic and model wrapping
   - properly name files and dirs i,e binding and qwn3 inference.
1. Create op class for all the ops 
2. MLP optimisation :
   - create an mlp op
   - we can fuse gate MM and up MM. 
      Do this by transposing and merging into a single weight during construction.
   - transpose down proj wt and use matmul instead
3. Use MatMul instead of Linear (latency difference ~1.5x)
   - we can map Linear ops with owned weights to matmul by : 
      - transposing weights at construction
      - calling matmul kernel during run




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
9. Enable all the qwen3 models of different sizes 

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
   - we can map Linear ops with owned weights to matmul by : 
      - transposing weights at construction
      - calling matmul kernel during run
2. MLP optimisation :
   - create a kernel and preprocess weight at the time of construction
   - create an mlp op
   - we can fuse gate MM and up MM. 
      Do this by transposing and merging into a single weight during construction.
   - transpose down proj wt and use matmul instead
   - Assert `input.numel() == embed_dim == output_dim`
4. Properly manage `prepare()` calls in Qwen3 model
5. Memory optimization improvements
6. Create special kernel for our M=1 ,K,N 