#include <tensor/kvcache.h>

// Helper function to calculate memory offsets
size_t KVCache::get_key_offset(size_t layer, size_t group, size_t token_idx) const
{
    return ((layer * num_groups_ + group) * max_sequence_length_ + token_idx) * head_dim_;
}

size_t KVCache::get_value_offset(size_t layer, size_t group, size_t token_idx) const
{
    return ((layer * num_groups_ + group) * max_sequence_length_ + token_idx) * head_dim_;
}

void KVCache::check_indices(size_t layer, size_t group, size_t token_idx) const
{
    if (layer >= num_layers_)
    {
        throw std::out_of_range("Layer index out of range: " + std::to_string(layer));
    }
    if (group >= num_groups_)
    {
        throw std::out_of_range("Group index out of range: " + std::to_string(group));
    }
    if (token_idx >= max_sequence_length_)
    {
        throw std::out_of_range("Token index out of range: " + std::to_string(token_idx));
    }
}

KVCache::KVCache(size_t max_seq_len, size_t head_dim, size_t num_groups, size_t num_layers)
    : max_sequence_length_(max_seq_len),
      head_dim_(head_dim),
      num_layers_(num_layers),
      num_groups_(num_groups),
      current_token_idx_(0),
      key_cache_(nullptr),
      value_cache_(nullptr)
{

    // Calculate total memory needed
    size_t total_elements = num_layers_ * num_groups_ * max_sequence_length_ * head_dim_;

    // Allocate contiguous memory for key cache
    key_cache_ = (float *)malloc(total_elements * sizeof(float));
    if (!key_cache_)
        throw std::bad_alloc();
    memset(key_cache_, 0, total_elements * sizeof(float));

    // Allocate contiguous memory for value cache
    value_cache_ = (float *)malloc(total_elements * sizeof(float));
    if (!value_cache_)
        throw std::bad_alloc();
    memset(value_cache_, 0, total_elements * sizeof(float));
}

KVCache::~KVCache()
{
    if (key_cache_)
    {
        free(key_cache_);
    }
    if (value_cache_)
    {
        free(value_cache_);
    }
}

float *KVCache::get_key_ptr(size_t layer, size_t group)
{
    check_indices(layer, group);
    return key_cache_ + get_key_offset(layer, group, current_token_idx_);
}

float *KVCache::get_value_ptr(size_t layer, size_t group)
{
    check_indices(layer, group);
    return value_cache_ + get_value_offset(layer, group, current_token_idx_);
}

const float *KVCache::get_key_memory_ptr(size_t layer, size_t group) const
{
    check_indices(layer, group);
    return key_cache_ + get_key_offset(layer, group, 0);
}

const float *KVCache::get_value_memory_ptr(size_t layer, size_t group) const
{
    check_indices(layer, group);
    return value_cache_ + get_value_offset(layer, group, 0);
}

const float *KVCache::get_full_key_cache_ptr() const
{
    return key_cache_;
}

const float *KVCache::get_full_value_cache_ptr() const
{
    return value_cache_;
}

void KVCache::set_key(size_t layer, size_t group, size_t token_idx, const float *key_data)
{
    check_indices(layer, group, token_idx);
    float *dest = key_cache_ + get_key_offset(layer, group, token_idx);
    memcpy(dest, key_data, head_dim_ * sizeof(float));
}

void KVCache::set_value(size_t layer, size_t group, size_t token_idx, const float *value_data)
{
    check_indices(layer, group, token_idx);
    float *dest = value_cache_ + get_value_offset(layer, group, token_idx);
    memcpy(dest, value_data, head_dim_ * sizeof(float));
}

void KVCache::set_current_key(size_t layer, const float *key_data)
{
    if (layer >= num_layers_)
    {
        throw std::out_of_range("Layer index out of range: " + std::to_string(layer));
    }

    for (size_t group = 0; group < num_groups_; ++group)
    {
        float *dest = key_cache_ + get_key_offset(layer, group, current_token_idx_);
        const float *src = key_data + (group * head_dim_);
        memcpy(dest, src, head_dim_ * sizeof(float));
    }
}

void KVCache::set_current_value(size_t layer, const float *value_data)
{
    if (layer >= num_layers_)
    {
        throw std::out_of_range("Layer index out of range: " + std::to_string(layer));
    }

    for (size_t group = 0; group < num_groups_; ++group)
    {
        float *dest = value_cache_ + get_value_offset(layer, group, current_token_idx_);
        const float *src = value_data + (group * head_dim_);
        memcpy(dest, src, head_dim_ * sizeof(float));
    }
}

const float *KVCache::get_key_at(size_t layer, size_t group, size_t token_idx) const
{
    check_indices(layer, group, token_idx);
    return key_cache_ + get_key_offset(layer, group, token_idx);
}

const float *KVCache::get_value_at(size_t layer, size_t group, size_t token_idx) const
{
    check_indices(layer, group, token_idx);
    return value_cache_ + get_value_offset(layer, group, token_idx);
}

std::vector<const float *> KVCache::get_all_keys_up_to_current(size_t layer, size_t group) const
{
    check_indices(layer, group);
    std::vector<const float *> keys;
    for (size_t i = 0; i <= current_token_idx_; ++i)
    {
        keys.push_back(get_key_at(layer, group, i));
    }
    return keys;
}

std::vector<const float *> KVCache::get_all_values_up_to_current(size_t layer, size_t group) const
{
    check_indices(layer, group);
    std::vector<const float *> values;
    for (size_t i = 0; i <= current_token_idx_; ++i)
    {
        values.push_back(get_value_at(layer, group, i));
    }
    return values;
}

void KVCache::advance()
{
    if (current_token_idx_ >= max_sequence_length_ - 1)
    {
        throw std::runtime_error("Token limit reached: " + std::to_string(max_sequence_length_));
    }
    current_token_idx_++;
}

void KVCache::reset()
{
    current_token_idx_ = 0;
}

size_t KVCache::get_current_token_idx() const
{
    return current_token_idx_;
}

size_t KVCache::get_max_sequence_length() const
{
    return max_sequence_length_;
}

size_t KVCache::get_remaining_tokens() const
{
    return max_sequence_length_ - current_token_idx_;
}

size_t KVCache::get_head_dim() const
{
    return head_dim_;
}

size_t KVCache::get_num_layers() const
{
    return num_layers_;
}

size_t KVCache::get_num_groups() const
{
    return num_groups_;
}

size_t KVCache::get_total_memory_size() const
{
    return 2 * num_layers_ * num_groups_ * max_sequence_length_ * head_dim_ * sizeof(float);
}