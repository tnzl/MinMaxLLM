// Steps to run: 
// 1. nvcc .\simple_cuda.cu
// 2. .\a.exe

#include <iostream>

__global__ void hello_cuda() {
    printf("Hello from CUDA kernel!\n");
}

int main() {
    hello_cuda<<<1, 1>>>();
    cudaDeviceSynchronize(); // Wait for the kernel to finish
    std::cout << "Hello from CPU!" << std::endl;
    return 0;
}
