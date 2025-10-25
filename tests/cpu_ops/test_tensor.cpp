#include <tensor/tensor.h>
#include <tensor/safetensors.h>
#include <cassert>
#include <cmath>
#include <vector>
#include <iostream>
#include <random>
#include <immintrin.h>
#include <chrono>
#include <thread>
#include <psapi.h>
#include <windows.h>
#include <iomanip>

#pragma comment(lib, "psapi.lib")

using namespace std;
using namespace std::chrono;

// ------------------------------------------------------------
// Utility: print memory stats for current process
// ------------------------------------------------------------
void printMemoryStats(const string &label)
{
    PROCESS_MEMORY_COUNTERS_EX pmc{};
    if (GetProcessMemoryInfo(GetCurrentProcess(),
                             (PROCESS_MEMORY_COUNTERS *)&pmc, sizeof(pmc)))
    {
        cout << "\n[" << label << "]\n";
        cout << "  Working Set: " << fixed << setprecision(2)
             << pmc.WorkingSetSize / (1024.0 * 1024.0) << " MB\n";
        cout << "  Private Usage: "
             << pmc.PrivateUsage / (1024.0 * 1024.0) << " MB\n";
        cout << "  Pagefile Usage: "
             << pmc.PagefileUsage / (1024.0 * 1024.0) << " MB\n";
    }
}

// ------------------------------------------------------------
// Utility: check which pages are resident using QueryWorkingSetEx
// ------------------------------------------------------------
void checkWorkingSet(const void *ptr, size_t bytes)
{
    const size_t pageSize = 4096;
    size_t pageCount = bytes / pageSize;
    vector<PSAPI_WORKING_SET_EX_INFORMATION> wsInfo(pageCount);
    for (size_t i = 0; i < pageCount; ++i)
        wsInfo[i].VirtualAddress = (void *)((uintptr_t)ptr + i * pageSize);

    if (QueryWorkingSetEx(GetCurrentProcess(), wsInfo.data(),
                          sizeof(PSAPI_WORKING_SET_EX_INFORMATION) * pageCount))
    {
        size_t resident = 0;
        for (const auto &info : wsInfo)
            if (info.VirtualAttributes.Valid)
                resident++;
        double pct = 100.0 * resident / pageCount;
        cout << "  Resident pages: " << resident << "/" << pageCount
             << " (" << fixed << setprecision(2) << pct << "%)\n";
    }
    else
    {
        cerr << "QueryWorkingSetEx failed with error: " << GetLastError() << "\n";
    }
}

// ------------------------------------------------------------
// Utility: measure latency of reading one float per 4KB page
// ------------------------------------------------------------
template <typename T>
void measureAccessLatency(const Tensor<T> &tensor, const string &label)
{
    const T *ptr = tensor.data();
    size_t bytes = tensor.size() * sizeof(T);
    size_t step = 4096 / sizeof(T);
    volatile double sum = 0.0;

    auto start = high_resolution_clock::now();
    for (size_t i = 0; i < tensor.size(); i += step)
        sum += ptr[i];
    auto end = high_resolution_clock::now();

    auto dur = duration_cast<milliseconds>(end - start).count();
    cout << "  [" << label << "] Access latency: " << dur << " ms (sum=" << sum << ")\n";
}

// ------------------------------------------------------------
// Main Test
// ------------------------------------------------------------
int main(int argc, char **argv)
{
    if (argc != 2)
    {
        cerr << "Usage: " << argv[0] << " <safetensor_file>\n";
        return 1;
    }

    const string path = argv[1];
    Safetensor st(path, true);

    const string key1 = "tensor_0";
    const string key2 = "tensor_2";

    const auto *info1 = st.getTensorInfo(key1);
    const auto *info2 = st.getTensorInfo(key2);
    if (!info1 || !info2)
    {
        cerr << "Tensor keys not found.\n";
        return 1;
    }

    const float *data1 = st.tensorDataPtr<float>(key1);
    const float *data2 = st.tensorDataPtr<float>(key2);

    Tensor<float> tensor1(const_cast<float *>(data1), info1->shape, true);
    Tensor<float> tensor2(const_cast<float *>(data2), info2->shape, true);

    cout << "Tensor sizes:\n";
    cout << "  tensor_0: " << (tensor1.size() * sizeof(float)) / (1024.0 * 1024.0) << " MB\n";
    cout << "  tensor_2: " << (tensor2.size() * sizeof(float)) / (1024.0 * 1024.0) << " MB\n";

    // ------------------------------------------------------------
    // Baseline: before prefetch
    // ------------------------------------------------------------
    printMemoryStats("Before prefetch()");
    checkWorkingSet(tensor2.data(), tensor2.size() * sizeof(float));
    measureAccessLatency(tensor2, "Before prefetch()");

    // ------------------------------------------------------------
    // Synchronous prefetch()
    // ------------------------------------------------------------
    cout << "\nTesting prefetch()...\n";
    auto t0 = high_resolution_clock::now();
    bool ok = tensor2.prefetch();
    auto t1 = high_resolution_clock::now();

    cout << "  prefetch() result: " << (ok ? "OK" : "FAILED")
         << " | duration: " << duration_cast<milliseconds>(t1 - t0).count() << " ms\n";

    printMemoryStats("After prefetch()");
    checkWorkingSet(tensor2.data(), tensor2.size() * sizeof(float));
    measureAccessLatency(tensor2, "After prefetch()");

    // ------------------------------------------------------------
    // Async prefetch
    // ------------------------------------------------------------
    cout << "\nTesting async prefetch_async() ...\n";
    auto t2 = high_resolution_clock::now();
    tensor1.prefetch_async();
    tensor2.prefetch_async();
    auto t3 = high_resolution_clock::now();

    cout << "  Async enqueue time: "
         << duration_cast<microseconds>(t3 - t2).count() << " µs\n";
    cout << "  Waiting for async prefetch (3s)...\n";
    this_thread::sleep_for(seconds(3));

    printMemoryStats("After async prefetch()");
    checkWorkingSet(tensor1.data(), tensor1.size() * sizeof(float));
    checkWorkingSet(tensor2.data(), tensor2.size() * sizeof(float));
    measureAccessLatency(tensor1, "After async prefetch()");
    measureAccessLatency(tensor2, "After async prefetch()");

    cout << "\nTest completed.\n";
    return 0;
}
