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

template <typename T>
class Tensor
{
public:
    Tensor();
    explicit Tensor(const std::vector<size_t> &shape);

    // Create from mutable pointer (non-const)
    Tensor(T *data, const std::vector<size_t> &shape, bool is_mmapped = false, bool take_ownership = false);

    // Create from const pointer (e.g., mmapped read-only memory)
    Tensor(const T *data, const std::vector<size_t> &shape, bool is_mmapped = false, bool take_ownership = false);

    Tensor(Tensor &&other) noexcept;
    Tensor(const Tensor &) = delete;
    Tensor &operator=(const Tensor &) = delete;
    Tensor &operator=(Tensor &&other) noexcept;
    ~Tensor();

    // non-const access (may return nullptr if created from const pointer and user treats it as mutable)
    T *data() noexcept;
    const T *data() const noexcept;

    const std::vector<size_t> &shape() const noexcept;
    size_t size() const noexcept;

    bool prefetch() const noexcept;
    void prefetch_async() const noexcept;
    void reshape(const std::vector<size_t> &new_shape);
    void mark_mmapped(bool mmapped = true) noexcept;
    void take_ownership(bool take = true) noexcept;

private:
    static size_t compute_num_elements(const std::vector<size_t> &shape);
    void release_owned();

    T *data_ = nullptr;
    std::vector<size_t> shape_;
    bool is_mmapped_ = false;
    bool is_mem_owner_ = false;
};