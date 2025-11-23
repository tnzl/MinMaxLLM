* Defining project objective becomes critical as the field language modeling is expanding every day.
* This project aims to serve the LLMs on device. 
* Our primary device target is CPU (AVX supported).
* We will process only one token at a time as the user is running on device.
* We will implement separate classes for every LLM family.  
* All the heavy computation will be done in C++.
* For serving the model we prefer Python to, make it more user friendly.
* We will provide chat interface and OpenAI style APIs.
* For core LLM backbone inference we aim to implement all the required operators.
* To infer the model we will run our operators sequentially.
* There'll be a separate py module/directory which optimizes the model. Example optimizations: quantization, weight transpose, weight concat(eg MLP)etc.
* Model optimization module takes the fp32 model's directory(which contains safetensors) and model config as input and generates optimized model.
* We aim to pool memory, implement a custom scratch memory allocator.