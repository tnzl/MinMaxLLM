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
            2. Qwen3MLP
            3. Qwen3RMSNorm [DONE]
        3. Qwen3RotaryEmbedding

3. Compare kernels to open source libs like libblas