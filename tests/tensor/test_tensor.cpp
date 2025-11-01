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
// Test modes
// ------------------------------------------------------------
void test_prefetch(Tensor<float> &tensor)
{
    printMemoryStats("Before prefetch()");
    checkWorkingSet(tensor.data(), tensor.size() * sizeof(float));
    measureAccessLatency(tensor, "Before prefetch()");

    cout << "\nTesting prefetch()...\n";
    auto t0 = high_resolution_clock::now();
    bool ok = tensor.prefetch();
    auto t1 = high_resolution_clock::now();

    cout << "  prefetch() result: " << (ok ? "OK" : "FAILED")
         << " | duration: " << duration_cast<milliseconds>(t1 - t0).count() << " ms\n";

    printMemoryStats("After prefetch()");
    checkWorkingSet(tensor.data(), tensor.size() * sizeof(float));
    measureAccessLatency(tensor, "After prefetch()");
}

void test_async_prefetch(Tensor<float> &tensor)
{
    printMemoryStats("Before async prefetch()");
    checkWorkingSet(tensor.data(), tensor.size() * sizeof(float));
    measureAccessLatency(tensor, "Before async prefetch()");

    cout << "\nTesting prefetch_async()...\n";
    auto t0 = high_resolution_clock::now();
    tensor.prefetch_async();
    auto t1 = high_resolution_clock::now();

    cout << "  Enqueued async prefetch in "
         << duration_cast<microseconds>(t1 - t0).count() << " us\n";
    cout << "  Waiting for async prefetch (5s)...\n";
    this_thread::sleep_for(seconds(5));

    printMemoryStats("After async prefetch()");
    checkWorkingSet(tensor.data(), tensor.size() * sizeof(float));
    measureAccessLatency(tensor, "After async prefetch()");
}

// ------------------------------------------------------------
// Main
// ------------------------------------------------------------
int main(int argc, char **argv)
{
    if (argc < 3)
    {
        cerr << "Usage: " << argv[0] << " <safetensor_file> <--prefetch | --async>\n";
        return 1;
    }

    const string path = argv[1];
    const string mode = argv[2];

    Safetensor st(path, true);

    const string key = "tensor_2"; // test big tensor for visibility
    const auto *info = st.getTensorInfo(key);
    if (!info)
    {
        cerr << "Tensor key not found: " << key << "\n";
        return 1;
    }

    const float *data = st.tensorDataPtr<float>(key);
    Tensor<float> tensor(const_cast<float *>(data), info->shape, true);

    cout << "Tensor under test: " << key << " | size = "
         << (tensor.size() * sizeof(float)) / (1024.0 * 1024.0) << " MB\n";

    if (mode == "--prefetch")
        test_prefetch(tensor);
    else if (mode == "--async")
        test_async_prefetch(tensor);
    else
        cerr << "Unknown mode: " << mode << " (use --prefetch or --async)\n";

    cout << "\nTest completed.\n";
    return 0;
}