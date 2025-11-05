## Get Started
#### Steps to build: 
0. Install Visual Studio 2022 (Community Edition) or later.
1. run build.ps1
2. Run the application from build folder.

## Next TODOs
1. Implement required ops.

    1. GQA (Grouped Query Attention) [DONE]
        i. functional modular GQA 
        ii. optimise using avx2
    2. Layer Normalization 
    3. RMS normalization [DONE]
    4. Softmax function [DONE]
2. Write auto regressive QWEN3 model
    1. Create a test structure to test against a pytorch module
    2. Build modules :
        1. Exbedding 
        2. Qwen3DecoderLayer
            1. Qwen3Attention
            2. Qwen3MLP [DONE]
            3. Qwen3RMSNorm [DONE]
        3. Qwen3RotaryEmbedding

## Features :
    1. Add an avx optimised matmul for M=1

## Cleanups :
1. Correct the name formatting in matmul avx. Should be M K N.
2. Follow standards in function parameter order : 
    * inputs first and then outputs 
    * ex : bool parseString(const char* input, int start_pos, int length, char* output, int* output_length);
3. Organise cpu ops properly. maybe namespaces based on type of impl : naive / avx2 or more might come. +
4. Read about scaling factor for rotary embedding, used for long context. 

## Performance : 
1. Use MatMul instead of Linear. lat diff ~1.5x
2. MLP block can optimised by doing MM gate, MM up and Silu at the same time.