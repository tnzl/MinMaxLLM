#pragma once

#include <cstdlib>
#include <cstring>
#include <vector>
#include <stdexcept>
#include <string>

class KVCache
{
private:
    size_t max_sequence_length_;
    size_t head_dim_;
    size_t num_layers_;
    size_t num_groups_;
    size_t current_token_idx_;

    // Contiguous KV cache storage
    float *key_cache_;   // [num_layers_ * num_groups_ * max_sequence_length_ * head_dim_]
    float *value_cache_; // [num_layers_ * num_groups_ * max_sequence_length_ * head_dim_]

    // Helper function to calculate memory offsets
    size_t get_key_offset(size_t layer, size_t group, size_t token_idx) const;
    size_t get_value_offset(size_t layer, size_t group, size_t token_idx) const;
    void check_indices(size_t layer, size_t group, size_t token_idx = 0) const;

public:
    KVCache(size_t max_seq_len, size_t head_dim, size_t num_groups, size_t num_layers);
    ~KVCache();

    // Delete copy constructor and assignment operator (rule of three)
    KVCache(const KVCache &) = delete;
    KVCache &operator=(const KVCache &) = delete;

    // Get key pointer for specific layer and group at current token
    float *get_key_ptr(size_t layer, size_t group=0);

    // Get value pointer for specific layer and group at current token
    float *get_value_ptr(size_t layer, size_t group=0);

    // Get const key pointer for full key memory of specific layer and group
    const float *get_key_memory_ptr(size_t layer, size_t group=0) const;

    // Get const value pointer for full value memory of specific layer and group
    const float *get_value_memory_ptr(size_t layer, size_t group=0) const;

    // Get const key pointer for entire key cache
    const float *get_full_key_cache_ptr() const;

    // Get const value pointer for entire value cache
    const float *get_full_value_cache_ptr() const;

    // Set key at specific layer, group and token index
    void set_key(size_t layer, size_t group, size_t token_idx, const float *key_data);

    // Set value at specific layer, group and token index
    void set_value(size_t layer, size_t group, size_t token_idx, const float *value_data);

    // Set current key for all groups in a layer (input: head_dim * num_groups elements)
    void set_current_key(size_t layer, const float *key_data);

    // Set current value for all groups in a layer (input: head_dim * num_groups elements)
    void set_current_value(size_t layer, const float *value_data);

    // Historical token data retrieval methods
    const float *get_key_at(size_t layer, size_t group, size_t token_idx) const;
    const float *get_value_at(size_t layer, size_t group, size_t token_idx) const;
    std::vector<const float *> get_all_keys_up_to_current(size_t layer, size_t group) const;
    std::vector<const float *> get_all_values_up_to_current(size_t layer, size_t group) const;

    // Sequence management
    void advance();
    void reset();

    // Getters
    size_t get_current_token_idx() const;
    size_t get_max_sequence_length() const;
    size_t get_remaining_tokens() const;
    size_t get_head_dim() const;
    size_t get_num_layers() const;
    size_t get_num_groups() const;
    size_t get_total_memory_size() const;
};