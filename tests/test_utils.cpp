// #include "test_util.h"
#include <iostream>
#include <algorithm>
#include <numeric>

template <typename T>
void print2DVector(const T* vec, size_t M, size_t N) {
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            // Calculate the index in the flattened 1D array
            std::cout << vec[i * N + j] << " ";
        }
        std::cout << "\n";
    }
}

bool validateResults(float* C1, float* C2, int M, int N, float epsilon) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float diff = std::fabs(C1[i * N + j] - C2[i * N + j]);
            if (diff > epsilon) {
                std::cerr << "Mismatch at (" << i << "," << j << "): "
                          << C1[i * N + j] << " vs " << C2[i * N + j] 
                          << " (diff: " << diff << ")\n";
                return false;
            }
        }
    }
    return true;
}

float calculateL2Error(float* C1, float* C2, int M, int N) {
    float sum_sq = 0.0f;
    int total_elements = M * N;
    
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float diff = C1[i * N + j] - C2[i * N + j];
            sum_sq += diff * diff;
        }
    }
    
    return std::sqrt(sum_sq / total_elements);
}

float calculateMaxError(float* C1, float* C2, int M, int N) {
    float max_error = 0.0f;
    
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float diff = std::fabs(C1[i * N + j] - C2[i * N + j]);
            if (diff > max_error) {
                max_error = diff;
            }
        }
    }
    
    return max_error;
}

void printErrorAnalysis(float* C1, float* C2, int M, int N) {
    // Calculate various error metrics
    float l2_error = calculateL2Error(C1, C2, M, N);
    float max_error = calculateMaxError(C1, C2, M, N);
    
    // Calculate relative errors
    float sum_ref = 0.0f;
    float sum_actual = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        sum_ref += C1[i] * C1[i];
        sum_actual += C2[i] * C2[i];
    }
    float norm_ref = std::sqrt(sum_ref);
    float norm_actual = std::sqrt(sum_actual);
    float relative_error = std::fabs(norm_ref - norm_actual) / norm_ref;
    
    std::cout << "\nError Analysis:\n";
    std::cout << "L2 Error: " << l2_error << "\n";
    std::cout << "Max Error: " << max_error << "\n";
    std::cout << "Relative Error (Frobenius norm): " << relative_error * 100 << "%\n";
    
    // Count number of elements with significant errors
    const float significant_error_threshold = 1e-4f;
    int significant_errors = 0;
    for (int i = 0; i < M * N; ++i) {
        if (std::fabs(C1[i] - C2[i]) > significant_error_threshold) {
            significant_errors++;
        }
    }
    
    std::cout << "Elements with error > " << significant_error_threshold << ": " 
              << significant_errors << " (" 
              << (100.0f * significant_errors) / (M * N) << "%)\n";
}