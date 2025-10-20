#pragma once
#include <vector>
#include <windows.h>

template <typename T>
class Tensor
{
public:
    Tensor() = default;

    Tensor(const std::vector<size_t> &shape);

    Tensor(T *data, const std::vector<size_t> &shape, bool is_mmapped = false);

    Tensor(Tensor &&other) noexcept;

    Tensor(const Tensor &) = delete;

    Tensor &operator=(const Tensor &) = delete;

    Tensor &operator=(Tensor &&other) noexcept;

    T *data();

    const std::vector<size_t> &shape() const;

    size_t size() const;

    // Prepare the tensor for use ( i.e advised to load into memory if mmapped )
    bool prepare();

    void reshape(const std::vector<size_t> &new_shape);

    ~Tensor()
    {
        if (is_mem_owner_ && data_)
        {
            delete[] data_;
            data_ = nullptr;
        }
    }

private:
    T *data_ = nullptr;
    std::vector<size_t> shape_;
    bool is_mmapped_ = false;
    bool is_mem_owner_ = false;
};