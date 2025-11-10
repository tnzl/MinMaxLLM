## Get Started
#### Steps to build: 
    1. Install Visual Studio 2022 (Community Edition) or later.
    2. run build.ps1
    3. Run the application from build folder.

## Next TODOs


### Features :
    1. Add an avx optimised matmul for M=1
    2. Op classes
    3. pybind to run the model from py
    4. 8 bit quantization
    5. sampling strategies, maube implement topk type ops.
    6. Enable tensor.to(device); // device can be ram gpu etc

### Cleanups :
    1. Correct the name formatting in matmul avx. Should be M K N.
    2. Follow standards in function parameter order : 
        * inputs first and then outputs 
        * ex : bool parseString(const char* input, int start_pos, int length, char* output, int* output_length);
    3. Organise cpu ops properly. maybe namespaces based on type of impl : naive / avx2 or more might come. +
    4. Read about scaling factor for rotary embedding, used for long context. 
    5. softmax at the end of qwen3 model. is it required ?

### Performance : 
    1. Use MatMul instead of Linear. lat diff ~1.5x
    2. MLP block can optimised by doing MM gate, MM up and Silu at the same time.
    3. write an mlp kernel directly instead. assert input numel() == embed_dim == output_dim
    4. properly manage the prepare() calls in qwen3