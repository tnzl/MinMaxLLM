#include <iostream>
#include "elew_add.cuh"

int main() {
    const int n = 1000000;
    
    // Allocate host memory
    float *a = new float[n];
    float *b = new float[n];
    float *c = new float[n];

    // Initialize input vectors
    for (int i = 0; i < n; ++i) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    // Perform elementwise addition
    elementwiseAdd(a, b, c, n);

    // Verify results (check first few elements)
    for (int i = 0; i < 10; ++i) {
        std::cout << "c[" << i << "] = " << c[i] << std::endl;
    }

    // Free host memory
    delete[] a;
    delete[] b;
    delete[] c;

    return 0;
}
