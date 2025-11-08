#include <iostream>
#include <algorithm>
#include <numeric>
#include <vector>
#include <sstream>
#include <fstream>

// Helper to parse comma-separated shape string
std::vector<int> parse_shape(const std::string& shape_str) {
    std::vector<int> shape;
    std::stringstream ss(shape_str);
    std::string item;
    while (std::getline(ss, item, ',')) {
        shape.push_back(std::stoi(item));
    }
    return shape;
}

// Helper to load flat float array from txt file
//add a deprecation warning
[[deprecated("Use load_txt that takes void* pointer as input")]]
std::vector<float> load_txt(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open file: " + path);
    std::vector<float> data;
    float val;
    while (f >> val) data.push_back(val);
    return data;
}

void load_txt(const std::string& path, void* data) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open file: " + path);
    float* fdata = static_cast<float*>(data);
    for (int i = 0; f >> fdata[i]; ++i);
}

void load_bin(const std::string& path, void* data, size_t size) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open file: " + path);
    f.read(static_cast<char*>(data), sizeof(float) * size);
}


// Helper to save flat array to txt file (templated)
template <typename T>
void save_txt(const std::string& path, const std::vector<T>& data) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot open file: " + path);
    for (const T& v : data) f << v << "\n";
}

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

void printErrorAnalysis1D(const float* ref, const float* actual, size_t N, float threshold = 1e-4f) {
    float l2_error = 0.0f, max_error = 0.0f, sum_ref = 0.0f, sum_actual = 0.0f;
    int significant_errors = 0;
    for (size_t i = 0; i < N; ++i) {
        float diff = ref[i] - actual[i];
        l2_error += diff * diff;
        float abs_diff = std::fabs(diff);
        if (abs_diff > max_error) max_error = abs_diff;
        if (abs_diff > threshold) significant_errors++;
        sum_ref += ref[i] * ref[i];
        sum_actual += actual[i] * actual[i];
    }
    l2_error = std::sqrt(l2_error / N);
    float norm_ref = std::sqrt(sum_ref);
    float norm_actual = std::sqrt(sum_actual);
    float relative_error = std::fabs(norm_ref - norm_actual) / (norm_ref + 1e-12f);
    std::cout << "\nError Analysis:\n";
    std::cout << "L2 Error: " << l2_error << "\n";
    std::cout << "Max Error: " << max_error << "\n";
    std::cout << "Relative Error (L2 norm): " << relative_error * 100 << "%\n";
    std::cout << "Elements with error > " << threshold << ": " << significant_errors << " ("
              << (100.0f * significant_errors) / N << "%)\n";
}