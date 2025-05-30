#ifndef TEST_UTIL_H
#define TEST_UTIL_H

#include <cmath>

// Validate results with epsilon tolerance (basic element-wise comparison)
bool validateResults(float* C1, float* C2, int M, int N, float epsilon = 1e-5f);

// Calculate L2 norm error between two matrices
float calculateL2Error(float* C1, float* C2, int M, int N);

// Calculate max error between two matrices
float calculateMaxError(float* C1, float* C2, int M, int N);

// Print detailed error analysis
void printErrorAnalysis(float* C1, float* C2, int M, int N);

#endif // TEST_UTIL_H