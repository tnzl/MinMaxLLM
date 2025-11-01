#include <iostream>
#include <cassert>
#include <vector>
#include <cstring>
#include <tensor/kvcache.h>

// Helper to compare two float arrays
bool float_array_equal(const float *a, const float *b, size_t n, float eps = 1e-6f)
{
    for (size_t i = 0; i < n; ++i)
    {
        if (std::abs(a[i] - b[i]) > eps)
            return false;
    }
    return true;
}

int main()
{
    // Test configuration
    const size_t max_seq_len = 8;
    const size_t head_dim = 4;
    const size_t num_groups = 2;
    const size_t num_layers = 3;

    KVCache cache(max_seq_len, head_dim, num_groups, num_layers);

    // Basic getter tests
    assert(cache.get_current_token_idx() == 0);
    assert(cache.get_max_sequence_length() == max_seq_len);
    assert(cache.get_head_dim() == head_dim);
    assert(cache.get_num_layers() == num_layers);
    assert(cache.get_num_groups() == num_groups);
    assert(cache.get_remaining_tokens() == max_seq_len);

    // Prepare dummy key/value data
    float key_data[head_dim * num_groups];
    float value_data[head_dim * num_groups];
    for (size_t i = 0; i < head_dim * num_groups; ++i)
    {
        key_data[i] = static_cast<float>(i + 1);
        value_data[i] = static_cast<float>((i + 1) * 10);
    }

    // Test set_current_key and set_current_value
    for (size_t layer = 0; layer < num_layers; ++layer)
    {
        cache.set_current_key(layer, key_data);
        cache.set_current_value(layer, value_data);
    }

    // Validate stored values
    for (size_t layer = 0; layer < num_layers; ++layer)
    {
        for (size_t group = 0; group < num_groups; ++group)
        {
            const float *kptr = cache.get_key_ptr(layer, group);
            const float *vptr = cache.get_value_ptr(layer, group);

            assert(float_array_equal(kptr, &key_data[group * head_dim], head_dim));
            assert(float_array_equal(vptr, &value_data[group * head_dim], head_dim));
        }
    }

    // Test advance()
    cache.advance();
    assert(cache.get_current_token_idx() == 1);
    assert(cache.get_remaining_tokens() == max_seq_len - 1);

    // Insert new data at new position
    float key_data2[head_dim * num_groups];
    float value_data2[head_dim * num_groups];
    for (size_t i = 0; i < head_dim * num_groups; ++i)
    {
        key_data2[i] = static_cast<float>(i + 100);
        value_data2[i] = static_cast<float>((i + 1) * 200);
    }

    cache.set_current_key(0, key_data2);
    cache.set_current_value(0, value_data2);

    // Check historical values: layer 0, token 0 vs token 1
    const float *hist_key0 = cache.get_key_at(0, 0, 0);
    const float *hist_key1 = cache.get_key_at(0, 0, 1);
    assert(float_array_equal(hist_key0, &key_data[0], head_dim));
    assert(float_array_equal(hist_key1, &key_data2[0], head_dim));

    // Test get_all_keys_up_to_current
    auto keys = cache.get_all_keys_up_to_current(0, 0);
    assert(keys.size() == 2);
    assert(float_array_equal(keys[0], &key_data[0], head_dim));
    assert(float_array_equal(keys[1], &key_data2[0], head_dim));

    // Reset and check
    cache.reset();
    assert(cache.get_current_token_idx() == 0);

    // Ensure after reset, the first write occurs at index 0 again
    cache.set_current_key(0, key_data);
    const float *reset_key_check = cache.get_key_at(0, 0, 0);
    assert(float_array_equal(reset_key_check, key_data, head_dim));

    std::cout << "✅ All KVCache tests passed successfully!" << std::endl;
    return 0;
}
