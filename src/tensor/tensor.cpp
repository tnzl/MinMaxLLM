#include <tensor/tensor.h>
#include <utility>

PrefetchManager &PrefetchManager::instance()
{
    static PrefetchManager inst;
    return inst;
}

PrefetchManager::PrefetchManager() : running_(true)
{
    worker_thread_ = std::thread(&PrefetchManager::worker_loop, this);
}

PrefetchManager::~PrefetchManager()
{
    stop();
}

void PrefetchManager::enqueue(void *ptr, size_t bytes)
{
    if (!ptr || bytes == 0)
        return;

    {
        std::lock_guard<std::mutex> lg(mutex_);
        queue_.emplace(ptr, bytes);
    }
    cv_.notify_one();
}

void PrefetchManager::stop()
{
    bool expected = true;
    if (!running_.compare_exchange_strong(expected, false))
        return;

    cv_.notify_one();
    if (worker_thread_.joinable())
        worker_thread_.join();
}

void PrefetchManager::worker_loop()
{
    while (running_.load())
    {
        WorkItem item;
        {
            std::unique_lock<std::mutex> ul(mutex_);
            cv_.wait(ul, [this]()
                     { return !queue_.empty() || !running_.load(); });

            if (!running_.load() && queue_.empty())
                break;

            if (queue_.empty())
                continue;

            item = queue_.front();
            queue_.pop();
        }

        void *ptr = std::get<0>(item);
        size_t bytes = std::get<1>(item);
        if (ptr && bytes > 0)
        {
            // Use WIN32_MEMORY_RANGE_ENTRY (correct struct) for PrefetchVirtualMemory
            WIN32_MEMORY_RANGE_ENTRY range;
            range.VirtualAddress = ptr;
            range.NumberOfBytes = static_cast<SIZE_T>(bytes);
            // ignore return - this is "best effort"
            (void)PrefetchVirtualMemory(GetCurrentProcess(), 1, &range, 0);
        }
    }
}

template <typename T>
Tensor<T>::Tensor() = default;

template <typename T>
Tensor<T>::Tensor(const std::vector<size_t> &shape)
    : shape_(shape), is_mmapped_(false), is_mem_owner_(true)
{
    size_t n = compute_num_elements(shape_);
    if (n > 0)
        data_ = new T[n];
}

template <typename T>
Tensor<T>::Tensor(T *data, const std::vector<size_t> &shape, bool is_mmapped, bool take_ownership)
    : data_(data), shape_(shape), is_mmapped_(is_mmapped), is_mem_owner_(take_ownership)
{
}

template <typename T>
Tensor<T>::Tensor(const T *data, const std::vector<size_t> &shape, bool is_mmapped, bool take_ownership)
    : shape_(shape), is_mmapped_(is_mmapped)
{
    if (take_ownership)
    {
        // cannot take ownership of const data safely
        throw std::invalid_argument("Cannot take ownership of const data pointer");
    }
    // store pointer (const_cast only for storage; we do NOT free it later)
    data_ = const_cast<T *>(data);
    is_mem_owner_ = false;
}

template <typename T>
Tensor<T>::Tensor(Tensor &&other) noexcept
    : data_(other.data_), shape_(std::move(other.shape_)), is_mmapped_(other.is_mmapped_), is_mem_owner_(other.is_mem_owner_)
{
    other.data_ = nullptr;
    other.is_mem_owner_ = false;
    other.is_mmapped_ = false;
}

template <typename T>
Tensor<T> &Tensor<T>::operator=(Tensor &&other) noexcept
{
    if (this != &other)
    {
        release_owned();
        data_ = other.data_;
        shape_ = std::move(other.shape_);
        is_mmapped_ = other.is_mmapped_;
        is_mem_owner_ = other.is_mem_owner_;
        other.data_ = nullptr;
        other.is_mem_owner_ = false;
        other.is_mmapped_ = false;
    }
    return *this;
}

template <typename T>
Tensor<T>::~Tensor()
{
    release_owned();
}

template <typename T>
T *Tensor<T>::data() noexcept { return data_; }

template <typename T>
const T *Tensor<T>::data() const noexcept { return data_; }

template <typename T>
const std::vector<size_t> &Tensor<T>::shape() const noexcept { return shape_; }

template <typename T>
size_t Tensor<T>::size() const noexcept { return compute_num_elements(shape_); }

template <typename T>
bool Tensor<T>::prefetch() const noexcept
{
    if (!data_ || size() == 0 || !is_mmapped_)
        return false;

    WIN32_MEMORY_RANGE_ENTRY range;
    range.VirtualAddress = static_cast<void *>(data_);
    range.NumberOfBytes = static_cast<SIZE_T>(size() * sizeof(T));

    BOOL result = PrefetchVirtualMemory(GetCurrentProcess(), 1, &range, 0);
    return result != FALSE;
}

template <typename T>
void Tensor<T>::prefetch_async() const noexcept
{
    if (!data_ || size() == 0 || !is_mmapped_)
        return;

    PrefetchManager::instance().enqueue(static_cast<void *>(data_), size() * sizeof(T));
}

template <typename T>
void Tensor<T>::reshape(const std::vector<size_t> &new_shape)
{
    size_t old_count = size();
    size_t new_count = compute_num_elements(new_shape);
    if (old_count != 0 && new_count != old_count)
        throw std::invalid_argument("reshape: total size must remain the same");
    shape_ = new_shape;
}

template <typename T>
void Tensor<T>::mark_mmapped(bool mmapped) noexcept { is_mmapped_ = mmapped; }

template <typename T>
void Tensor<T>::take_ownership(bool take) noexcept { is_mem_owner_ = take; }

template <typename T>
size_t Tensor<T>::compute_num_elements(const std::vector<size_t> &shape)
{
    size_t n = 1;
    for (size_t d : shape)
    {
        if (d == 0)
            return 0;
        n *= d;
    }
    return n;
}

template <typename T>
void Tensor<T>::release_owned()
{
    if (is_mem_owner_ && data_)
    {
        delete[] data_;
        data_ = nullptr;
        is_mem_owner_ = false;
    }
}

// Explicit template instantiation for common types
template class Tensor<float>;
template class Tensor<double>;
template class Tensor<int>;
template class Tensor<uint8_t>;
