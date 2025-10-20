#include <tensor/tensor.h>

template <typename T>
Tensor<T>::Tensor(const std::vector<size_t> &shape)
    : shape_(shape), is_mem_owner_(true), is_mmapped_(false)
{
    if (shape.empty())
    {
        data_ = nullptr;
        return;
    }

    size_t total_size = size() * sizeof(T);
    if (total_size > 0)
    {
        data_ = new T[total_size];
        if (!data_)
        {
            throw std::bad_alloc();
        }
    }
    else
    {
        data_ = nullptr;
    }
}

template <typename T>
Tensor<T>::Tensor(T *data, const std::vector<size_t> &shape, bool is_mmapped)
    : data_(data), shape_(shape), is_mmapped_(is_mmapped), is_mem_owner_(false)
{
}

template <typename T>
Tensor<T>::Tensor(Tensor &&other) noexcept
    : data_(other.data_),
      shape_(std::move(other.shape_)),
      is_mmapped_(other.is_mmapped_),
      is_mem_owner_(other.is_mem_owner_)
{
    other.data_ = nullptr;
    other.is_mem_owner_ = false;
}

template <typename T>
Tensor<T> &Tensor<T>::operator=(Tensor &&other) noexcept
{
    if (this != &other)
    {
        if (is_mem_owner_ && data_)
        {
            delete[] data_;
        }

        data_ = other.data_;
        shape_ = std::move(other.shape_);
        is_mmapped_ = other.is_mmapped_;
        is_mem_owner_ = other.is_mem_owner_;

        other.data_ = nullptr;
        other.is_mem_owner_ = false;
    }
    return *this;
}

template <typename T>
T *Tensor<T>::data()
{
    return data_;
}

template <typename T>
const std::vector<size_t> &Tensor<T>::shape() const
{
    return shape_;
}

template <typename T>
size_t Tensor<T>::size() const
{
    if (shape_.empty())
        return 0;

    size_t total = 1;
    for (const auto &dim : shape_)
    {
        total *= dim;
    }
    return total;
}

template <typename T>
bool Tensor<T>::prepare()
{
    if (!data_ || size() == 0)
        return false;

    if (!is_mmapped_)
        return true; // Already in memory

    using PrefetchVirtualMemoryFn = BOOL(WINAPI *)(HANDLE, ULONG, PWIN32_MEMORY_RANGE_ENTRY, ULONG);
    static auto fn = reinterpret_cast<PrefetchVirtualMemoryFn>(
        GetProcAddress(GetModuleHandleW(L"kernel32.dll"), "PrefetchVirtualMemory"));
    if (!fn)
        return false; // Not supported

    WIN32_MEMORY_RANGE_ENTRY range;
    range.VirtualAddress = data_;
    range.NumberOfBytes = size() * sizeof(T);

    return fn(GetCurrentProcess(), 1, &range, 0) != 0;
}

template <typename T>
void Tensor<T>::reshape(const std::vector<size_t> &new_shape)
{
    size_t new_size = 1;
    for (const auto &dim : new_shape)
    {
        new_size *= dim;
    }
    if (new_size != size())
    {
        throw std::runtime_error("Reshape size does not match original size: " << new_size << " != " << size() << ".");
    }
    shape_ = new_shape;
}
