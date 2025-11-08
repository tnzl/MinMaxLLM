#include <tensor/tensor.h>
#include <utility>
#include <cstring>
#include <malloc.h>

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

Tensor::Tensor() = default;

Tensor::Tensor(DataType dtype, const std::vector<size_t> &shape)
    : shape_(shape), dtype_(dtype), is_mmapped_(false), is_mem_owner_(true)
{
    size_t bytes = nbytes();
    if (bytes > 0)
    {
        data_ = _aligned_malloc(bytes, 64);
        if (!data_)
            throw std::bad_alloc();
    }
}

Tensor::Tensor(void *data, const std::vector<size_t> &shape, DataType dtype, bool is_mmapped, bool take_ownership)
    : data_(data), shape_(shape), dtype_(dtype), is_mmapped_(is_mmapped), is_mem_owner_(take_ownership)
{
}

Tensor::Tensor(const void *data, const std::vector<size_t> &shape, DataType dtype, bool is_mmapped, bool take_ownership)
    : shape_(shape), dtype_(dtype), is_mmapped_(is_mmapped)
{
    if (take_ownership)
    {
        throw std::invalid_argument("Cannot take ownership of const data pointer");
    }
    data_ = const_cast<void *>(data);
    is_mem_owner_ = false;
}

Tensor::Tensor(Tensor &&other) noexcept
    : data_(other.data_), shape_(std::move(other.shape_)), dtype_(other.dtype_), is_mmapped_(other.is_mmapped_), is_mem_owner_(other.is_mem_owner_)
{
    other.data_ = nullptr;
    other.is_mem_owner_ = false;
    other.is_mmapped_ = false;
}

Tensor &Tensor::operator=(Tensor &&other) noexcept
{
    if (this != &other)
    {
        release_owned();
        data_ = other.data_;
        shape_ = std::move(other.shape_);
        dtype_ = other.dtype_;
        is_mmapped_ = other.is_mmapped_;
        is_mem_owner_ = other.is_mem_owner_;
        other.data_ = nullptr;
        other.is_mem_owner_ = false;
        other.is_mmapped_ = false;
    }
    return *this;
}

Tensor::~Tensor()
{
    release_owned();
}

size_t Tensor::element_size(DataType dt) noexcept
{
    switch (dt)
    {
    case DataType::F32:
        return sizeof(float);
    case DataType::F64:
        return sizeof(double);
    case DataType::I32:
        return sizeof(int32_t);
    case DataType::U8:
        return sizeof(uint8_t);
    default:
        return 1;
    }
}

bool Tensor::prefetch() const noexcept
{
    if (!data_ || size() == 0 || !is_mmapped_)
        return false;

    WIN32_MEMORY_RANGE_ENTRY range;
    range.VirtualAddress = static_cast<void *>(data_);
    range.NumberOfBytes = static_cast<SIZE_T>(nbytes());

    BOOL result = PrefetchVirtualMemory(GetCurrentProcess(), 1, &range, 0);
    return result != FALSE;
}

void Tensor::prefetch_async() const noexcept
{
    if (!data_ || size() == 0 || !is_mmapped_)
        return;

    PrefetchManager::instance().enqueue(static_cast<void *>(data_), nbytes());
}

void Tensor::reshape(const std::vector<size_t> &new_shape)
{
    size_t old_count = size();
    size_t new_count = compute_num_elements(new_shape);
    if (old_count != 0 && new_count != old_count)
        throw std::invalid_argument("reshape: total size must remain the same");
    shape_ = new_shape;
}

size_t Tensor::compute_num_elements(const std::vector<size_t> &shape)
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

void Tensor::release_owned()
{
    if (is_mem_owner_ && data_)
    {
        _aligned_free(data_);
        data_ = nullptr;
        is_mem_owner_ = false;
    }
}

