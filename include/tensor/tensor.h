#pragma once

#include <queue>
#include <tuple>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <windows.h>
#include <vector>
#include <cstddef>
#include <stdexcept>
#include <cstdint>
#include <type_traits>
#include <cassert>

struct PrefetchManager
{
    using WorkItem = std::tuple<void *, size_t>;

    static PrefetchManager &instance();

    void enqueue(void *ptr, size_t bytes);
    void stop();

private:
    PrefetchManager();
    ~PrefetchManager();

    PrefetchManager(const PrefetchManager &) = delete;
    PrefetchManager &operator=(const PrefetchManager &) = delete;

    void worker_loop();

    std::queue<WorkItem> queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::thread worker_thread_;
    std::atomic<bool> running_{false};
};

enum class DataType
{
    F32,
    F64,
    I32,
    U8
};

class Tensor
{
public:
    Tensor();
    Tensor(DataType dtype, const std::vector<size_t> &shape);
    Tensor(void *data, const std::vector<size_t> &shape, DataType dtype, bool is_mmapped = false, bool take_ownership = false);
    Tensor(const void *data, const std::vector<size_t> &shape, DataType dtype, bool is_mmapped = false, bool take_ownership = false);

    Tensor(Tensor &&other) noexcept;
    Tensor(const Tensor &) = delete;
    Tensor &operator=(const Tensor &) = delete;
    Tensor &operator=(Tensor &&other) noexcept;
    ~Tensor();

    // Raw data access
    void *raw_data() noexcept { return data_; }
    const void *raw_data() const noexcept { return data_; }

    // Typed accessors with runtime validation
    template <typename T>
    T *data() noexcept
    {
        validate_dtype_for<T>();
        return reinterpret_cast<T *>(data_);
    }

    template <typename T>
    const T *data() const noexcept
    {
        validate_dtype_for<T>();
        return reinterpret_cast<const T *>(data_);
    }

    const std::vector<size_t> &shape() const noexcept { return shape_; }
    size_t size() const noexcept { return compute_num_elements(shape_); }
    size_t nbytes() const noexcept { return size() * element_size(dtype_); }

    bool prefetch() const noexcept;
    void prefetch_async() const noexcept;
    void reshape(const std::vector<size_t> &new_shape);
    void mark_mmapped(bool mmapped = true) noexcept { is_mmapped_ = mmapped; }
    void take_ownership(bool take = true) noexcept { is_mem_owner_ = take; }
    DataType dtype() const noexcept { return dtype_; }

private:
    static size_t compute_num_elements(const std::vector<size_t> &shape);
    static size_t element_size(DataType dt) noexcept;

    template <typename T>
    static DataType cpp_to_dtype();

    template <typename T>
    void validate_dtype_for() const noexcept
    {
        // In debug builds assert; in release builds only best-effort check via noexcept
        assert(dtype_ == cpp_to_dtype<T>());
        (void)0;
    }

    void release_owned();

    void *data_ = nullptr;
    std::vector<size_t> shape_;
    DataType dtype_ = DataType::F32;
    bool is_mmapped_ = false;
    bool is_mem_owner_ = false;
};

// Type mappings
template <>
inline DataType Tensor::cpp_to_dtype<float>()
{
    return DataType::F32;
}
template <>
inline DataType Tensor::cpp_to_dtype<double>()
{
    return DataType::F64;
}
template <>
inline DataType Tensor::cpp_to_dtype<int32_t>()
{
    return DataType::I32;
}
template <>
inline DataType Tensor::cpp_to_dtype<uint8_t>()
{
    return DataType::U8;
}