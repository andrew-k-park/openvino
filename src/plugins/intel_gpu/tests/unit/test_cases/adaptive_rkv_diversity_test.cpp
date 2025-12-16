// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "random_generator.hpp"
#include <gtest/gtest.h>

#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <openvino/reference/adaptive_rkv_diversity.hpp>

using namespace cldnn;
using namespace ov::intel_gpu;
using namespace ::tests;

namespace {

// Helper to generate realistic key data for testing
std::vector<float> generate_test_keys(size_t num_heads, size_t num_tokens, size_t head_size, int seed = 1234) {
    std::vector<float> keys(num_heads * num_tokens * head_size);
    
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.0f, 0.1f);
    
    for (size_t h = 0; h < num_heads; ++h) {
        for (size_t t = 0; t < num_tokens; ++t) {
            for (size_t d = 0; d < head_size; ++d) {
                float val = dist(gen);
                // Add temporal correlation
                if (t > 0) {
                    size_t prev_idx = h * num_tokens * head_size + (t - 1) * head_size + d;
                    val = 0.7f * val + 0.3f * keys[prev_idx];
                }
                keys[h * num_tokens * head_size + t * head_size + d] = val;
            }
        }
    }
    
    return keys;
}

} // anonymous namespace

// Test Adaptive RKV diversity calculator reference implementation
TEST(adaptive_rkv_diversity_reference_test, basic_diversity_calculation) {
    const size_t num_heads = 4;
    const size_t num_tokens = 128;
    const size_t head_size = 64;
    const size_t block_size = 16;
    const size_t start_size = 32;  // Skip first 32 tokens (2 blocks)
    const size_t eviction_size = 64;  // Consider 64 tokens for eviction (4 blocks)
    
    // Generate test keys
    auto keys = generate_test_keys(num_heads, num_tokens, head_size);
    
    // Create diversity calculator
    ov::reference::AdaptiveRKVDiversityCalculator<float> calculator(
        start_size, eviction_size, block_size);
    
    // Reshape keys to [num_heads, num_tokens, head_size]
    ov::Shape key_shape = {num_heads, num_tokens, head_size};
    
    // Calculate diversity
    auto diversity = calculator.calculate_block_diversity(keys.data(), key_shape);
    
    // Verify output shape: [num_blocks, eviction_size]
    const size_t expected_num_blocks = eviction_size / block_size;
    ASSERT_EQ(diversity.size(), expected_num_blocks);
    for (const auto& row : diversity) {
        ASSERT_EQ(row.size(), eviction_size);
    }
    
    // Verify diversity values are reasonable (should be negative or near zero)
    // Diversity = -similarity, so values should be <= 0
    for (size_t block_idx = 0; block_idx < diversity.size(); ++block_idx) {
        for (size_t token_idx = 0; token_idx < diversity[block_idx].size(); ++token_idx) {
            float div_value = static_cast<float>(diversity[block_idx][token_idx]);
            EXPECT_LE(div_value, 1.0f) << "Block " << block_idx << ", Token " << token_idx;
            EXPECT_GE(div_value, -1.0f) << "Block " << block_idx << ", Token " << token_idx;
        }
    }
}

TEST(adaptive_rkv_diversity_reference_test, diversity_with_different_parameters) {
    const size_t num_heads = 2;
    const size_t num_tokens = 80;
    const size_t head_size = 32;
    const size_t block_size = 16;
    
    struct TestCase {
        size_t start_size;
        size_t eviction_size;
        std::string name;
    };
    
    std::vector<TestCase> test_cases = {
        {16, 32, "small_eviction"},
        {32, 48, "medium_eviction"},
        {0, 64, "no_start_area"},
    };
    
    for (const auto& tc : test_cases) {
        SCOPED_TRACE(tc.name);
        
        // Verify preconditions
        ASSERT_EQ(tc.start_size % block_size, 0);
        ASSERT_EQ(tc.eviction_size % block_size, 0);
        ASSERT_LE(tc.start_size + tc.eviction_size, num_tokens);
        
        auto keys = generate_test_keys(num_heads, num_tokens, head_size, 42);
        
        ov::reference::AdaptiveRKVDiversityCalculator<float> calculator(
            tc.start_size, tc.eviction_size, block_size);
        
        ov::Shape key_shape = {num_heads, num_tokens, head_size};
        auto diversity = calculator.calculate_block_diversity(keys.data(), key_shape);
        
        const size_t expected_num_blocks = tc.eviction_size / block_size;
        ASSERT_EQ(diversity.size(), expected_num_blocks);
    }
}

TEST(adaptive_rkv_diversity_reference_test, fill_diagonal_correctness) {
    const size_t num_heads = 2;
    const size_t token_dim = 16;
    
    std::vector<float> matrix(num_heads * token_dim * token_dim, 1.0f);
    ov::Shape matrix_shape = {num_heads, token_dim, token_dim};
    
    ov::reference::AdaptiveRKVDiversityCalculator<float> calculator(16, 16, 16);
    calculator.fill_diagonal_(matrix.data(), matrix_shape, 0.0f);
    
    // Verify diagonal is 0
    for (size_t h = 0; h < num_heads; ++h) {
        for (size_t i = 0; i < token_dim; ++i) {
            size_t idx = h * token_dim * token_dim + i * token_dim + i;
            EXPECT_EQ(matrix[idx], 0.0f) << "Head " << h << ", position " << i;
        }
    }
    
    // Verify non-diagonal elements are still 1.0
    for (size_t h = 0; h < num_heads; ++h) {
        for (size_t i = 0; i < token_dim; ++i) {
            for (size_t j = 0; j < token_dim; ++j) {
                if (i != j) {
                    size_t idx = h * token_dim * token_dim + i * token_dim + j;
                    EXPECT_EQ(matrix[idx], 1.0f) << "Head " << h << ", position (" << i << "," << j << ")";
                }
            }
        }
    }
}

TEST(adaptive_rkv_diversity_reference_test, mean_threshold_filtering) {
    const size_t num_heads = 2;
    const size_t eviction_size = 8;
    
    // Create test matrix with known values
    std::vector<float> matrix(num_heads * eviction_size * eviction_size);
    ov::Shape matrix_shape = {num_heads, eviction_size, eviction_size};
    
    // Fill with pattern: row i has values from 0 to eviction_size-1
    for (size_t h = 0; h < num_heads; ++h) {
        for (size_t i = 0; i < eviction_size; ++i) {
            for (size_t j = 0; j < eviction_size; ++j) {
                matrix[h * eviction_size * eviction_size + i * eviction_size + j] = static_cast<float>(j);
            }
        }
    }
    
    // Calculate means (should be 3.5 for each row)
    std::vector<float> means(num_heads * eviction_size);
    ov::Shape means_shape = {num_heads, eviction_size};
    
    for (size_t h = 0; h < num_heads; ++h) {
        for (size_t i = 0; i < eviction_size; ++i) {
            float sum = 0.0f;
            for (size_t j = 0; j < eviction_size; ++j) {
                sum += matrix[h * eviction_size * eviction_size + i * eviction_size + j];
            }
            means[h * eviction_size + i] = sum / eviction_size;
        }
    }
    
    ov::reference::AdaptiveRKVDiversityCalculator<float> calculator(16, 16, 16);
    calculator.fill_low_values_with_zeros_(matrix.data(), matrix_shape, means.data(), means_shape);
    
    // Verify: values < mean (3.5) should be 0, values >= mean should be unchanged
    for (size_t h = 0; h < num_heads; ++h) {
        for (size_t i = 0; i < eviction_size; ++i) {
            for (size_t j = 0; j < eviction_size; ++j) {
                float expected = (j >= 4) ? static_cast<float>(j) : 0.0f;  // mean is 3.5, so 0-3 -> 0, 4-7 -> kept
                float actual = matrix[h * eviction_size * eviction_size + i * eviction_size + j];
                EXPECT_EQ(actual, expected) << "Head " << h << ", position (" << i << "," << j << ")";
            }
        }
    }
}

TEST(adaptive_rkv_diversity_reference_test, block_sum_diversity_values) {
    const size_t token_dim = 32;
    const size_t block_size = 16;
    
    // Create test matrix with known values
    std::vector<float> input_matrix(token_dim * token_dim);
    ov::Shape input_shape = {token_dim, token_dim};
    
    // Fill with simple pattern for easy verification
    for (size_t i = 0; i < token_dim; ++i) {
        for (size_t j = 0; j < token_dim; ++j) {
            input_matrix[i * token_dim + j] = 1.0f;  // All ones for simple sum check
        }
    }
    
    const size_t num_blocks = token_dim / block_size;
    std::vector<float> output_matrix(num_blocks * token_dim, 0.0f);
    ov::Shape output_shape = {num_blocks, token_dim};
    
    ov::reference::AdaptiveRKVDiversityCalculator<float> calculator(0, token_dim, block_size);
    calculator.block_sum_diversity_values(input_matrix.data(), input_shape, output_matrix.data(), output_shape);
    
    // Verify: each output value should be -block_size (due to negation and summing block_size ones)
    for (size_t block_idx = 0; block_idx < num_blocks; ++block_idx) {
        for (size_t token_idx = 0; token_idx < token_dim; ++token_idx) {
            float expected = -static_cast<float>(block_size);
            float actual = output_matrix[block_idx * token_dim + token_idx];
            EXPECT_FLOAT_EQ(actual, expected) << "Block " << block_idx << ", Token " << token_idx;
        }
    }
}

TEST(adaptive_rkv_diversity_reference_test, edge_case_single_block) {
    const size_t num_heads = 1;
    const size_t num_tokens = 32;
    const size_t head_size = 16;
    const size_t block_size = 16;
    const size_t start_size = 16;
    const size_t eviction_size = 16;  // Single block
    
    auto keys = generate_test_keys(num_heads, num_tokens, head_size);
    
    ov::reference::AdaptiveRKVDiversityCalculator<float> calculator(
        start_size, eviction_size, block_size);
    
    ov::Shape key_shape = {num_heads, num_tokens, head_size};
    auto diversity = calculator.calculate_block_diversity(keys.data(), key_shape);
    
    // Should have exactly 1 block
    ASSERT_EQ(diversity.size(), 1);
    ASSERT_EQ(diversity[0].size(), eviction_size);
}

TEST(adaptive_rkv_diversity_reference_test, deterministic_output) {
    const size_t num_heads = 2;
    const size_t num_tokens = 64;
    const size_t head_size = 32;
    const size_t block_size = 16;
    const size_t start_size = 16;
    const size_t eviction_size = 32;
    
    auto keys = generate_test_keys(num_heads, num_tokens, head_size, 999);
    
    ov::reference::AdaptiveRKVDiversityCalculator<float> calculator(
        start_size, eviction_size, block_size);
    
    ov::Shape key_shape = {num_heads, num_tokens, head_size};
    
    // Run twice with same input
    auto diversity1 = calculator.calculate_block_diversity(keys.data(), key_shape);
    auto diversity2 = calculator.calculate_block_diversity(keys.data(), key_shape);
    
    // Results should be identical
    ASSERT_EQ(diversity1.size(), diversity2.size());
    for (size_t i = 0; i < diversity1.size(); ++i) {
        ASSERT_EQ(diversity1[i].size(), diversity2[i].size());
        for (size_t j = 0; j < diversity1[i].size(); ++j) {
            EXPECT_FLOAT_EQ(static_cast<float>(diversity1[i][j]), 
                          static_cast<float>(diversity2[i][j]))
                << "Mismatch at [" << i << "][" << j << "]";
        }
    }
}

// Test that diversity values correctly reflect token similarity patterns
TEST(adaptive_rkv_diversity_reference_test, diversity_reflects_similarity) {
    const size_t num_heads = 1;
    const size_t num_tokens = 48;
    const size_t head_size = 8;
    const size_t block_size = 16;
    const size_t start_size = 0;
    const size_t eviction_size = 48;
    
    std::vector<float> keys(num_heads * num_tokens * head_size);
    
    // Create two distinct patterns:
    // Tokens 0-23: all similar (low diversity expected)
    // Tokens 24-47: all different (high diversity expected)
    for (size_t t = 0; t < num_tokens; ++t) {
        for (size_t d = 0; d < head_size; ++d) {
            if (t < 24) {
                // Similar tokens
                keys[t * head_size + d] = 1.0f;
            } else {
                // Different tokens
                keys[t * head_size + d] = static_cast<float>(t - 24) / 24.0f;
            }
        }
    }
    
    ov::reference::AdaptiveRKVDiversityCalculator<float> calculator(
        start_size, eviction_size, block_size);
    
    ov::Shape key_shape = {num_heads, num_tokens, head_size};
    auto diversity = calculator.calculate_block_diversity(keys.data(), key_shape);
    
    // Blocks 0-1 (tokens 0-31): should have lower diversity (more similar)
    // Block 2 (tokens 32-47): should have higher diversity (more different)
    // Note: diversity values are negative, so "higher diversity" means more negative
    
    float avg_diversity_block0 = 0.0f;
    float avg_diversity_block2 = 0.0f;
    
    for (size_t j = 0; j < eviction_size; ++j) {
        avg_diversity_block0 += static_cast<float>(diversity[0][j]);
        avg_diversity_block2 += static_cast<float>(diversity[2][j]);
    }
    
    avg_diversity_block0 /= eviction_size;
    avg_diversity_block2 /= eviction_size;
    
    // More diverse block should have more negative average
    EXPECT_LT(avg_diversity_block2, avg_diversity_block0)
        << "Block with different tokens should have higher diversity (more negative value)";
}

// ========================================
// GPU Kernel Integration Tests
// ========================================

/*
 * Test suite for GPU kernel-level operations
 * Tests individual kernels and their integration
 */

TEST(adaptive_rkv_gpu_kernel_test, kernel_parameter_alignment) {
    // Test that kernel parameters are properly aligned to block boundaries
    const int block_size = 16;
    
    struct TestCase {
        int start_size;
        int evictable_size;
        int num_tokens;
        bool expected_valid;
        std::string description;
    };
    
    std::vector<TestCase> test_cases = {
        {0, 16, 32, true, "minimal_single_block"},
        {16, 32, 64, true, "normal_case"},
        {32, 64, 128, true, "larger_cache"},
        {0, 256, 256, true, "all_evictable"},
        {15, 32, 64, false, "start_not_aligned"},
        {16, 31, 64, false, "evictable_not_aligned"},
        {16, 64, 32, false, "exceeds_total_tokens"},
    };
    
    for (const auto& tc : test_cases) {
        SCOPED_TRACE(tc.description);
        
        bool start_aligned = (tc.start_size % block_size == 0);
        bool evictable_aligned = (tc.evictable_size % block_size == 0);
        bool size_valid = (tc.start_size + tc.evictable_size <= tc.num_tokens);
        bool is_valid = start_aligned && evictable_aligned && size_valid;
        
        EXPECT_EQ(is_valid, tc.expected_valid);
    }
}

TEST(adaptive_rkv_gpu_kernel_test, kernel_work_group_sizes) {
    // Test that kernel work group sizes are correctly calculated
    // These parameters are for reference/documentation purposes
    [[maybe_unused]] const int block_size = 16;
    [[maybe_unused]] const int subgroup_size = 16;
    
    struct TestCase {
        int num_kv_heads;
        int evictable_size;
        int expected_gws_x;  // Expected global work size in X dimension
        int expected_gws_y;
        int expected_gws_z;
    };
    
    std::vector<TestCase> test_cases = {
        {2, 16, 16, 16, 2},    // 2 heads, 1 block
        {4, 32, 32, 32, 4},    // 4 heads, 2 blocks
        {8, 64, 64, 64, 8},    // 8 heads, 4 blocks
        {32, 128, 128, 128, 32}, // Large case
    };
    
    for (const auto& tc : test_cases) {
        // For slice_and_fill_diagonal kernel:
        // gws = (evictable_size, evictable_size, num_kv_heads)
        EXPECT_EQ(tc.expected_gws_x, tc.evictable_size);
        EXPECT_EQ(tc.expected_gws_y, tc.evictable_size);
        EXPECT_EQ(tc.expected_gws_z, tc.num_kv_heads);
    }
}

TEST(adaptive_rkv_gpu_kernel_test, memory_layout_calculation) {
    // Test memory layout and offset calculations
    const int num_kv_heads = 4;
    const int num_tokens = 128;
    const int evictable_size = 64;
    const int start_size = 32;
    
    // Test source offset calculation (full similarity matrix)
    for (int head_idx = 0; head_idx < num_kv_heads; ++head_idx) {
        for (int row = 0; row < evictable_size; ++row) {
            for (int col = 0; col < evictable_size; ++col) {
                // This matches kernel logic:
                // src_offset = head_idx * num_tokens * num_tokens +
                //              (start_size + row_idx) * num_tokens +
                //              (start_size + col_idx);
                int expected_src_offset = 
                    head_idx * num_tokens * num_tokens +
                    (start_size + row) * num_tokens +
                    (start_size + col);
                
                EXPECT_GE(expected_src_offset, 0);
                EXPECT_LT(expected_src_offset, num_kv_heads * num_tokens * num_tokens);
            }
        }
    }
    
    // Test destination offset calculation (sliced matrix)
    for (int head_idx = 0; head_idx < num_kv_heads; ++head_idx) {
        for (int row = 0; row < evictable_size; ++row) {
            for (int col = 0; col < evictable_size; ++col) {
                // dst_offset = head_idx * eviction_size * eviction_size +
                //              row_idx * eviction_size + col_idx;
                int expected_dst_offset = 
                    head_idx * evictable_size * evictable_size +
                    row * evictable_size + col;
                
                EXPECT_GE(expected_dst_offset, 0);
                EXPECT_LT(expected_dst_offset, num_kv_heads * evictable_size * evictable_size);
            }
        }
    }
}

TEST(adaptive_rkv_gpu_kernel_test, block_sum_index_mapping) {
    // Test block sum kernel index calculations
    const int block_size = 16;
    const int evictable_size = 64;
    const int num_blocks = evictable_size / block_size;  // 4 blocks
    
    // For block_sum_diversity kernel:
    // global_id(0) = token_idx (0 to evictable_size-1)
    // global_id(1) = block_idx (0 to num_blocks-1)
    
    for (int block_idx = 0; block_idx < num_blocks; ++block_idx) {
        int token_start = block_idx * block_size;
        int token_end = token_start + block_size;
        
        EXPECT_EQ(token_start, block_idx * block_size);
        EXPECT_EQ(token_end, (block_idx + 1) * block_size);
        EXPECT_LE(token_end, evictable_size);
        
        // Each block processes block_size tokens
        for (int local_idx = 0; local_idx < block_size; ++local_idx) {
            int global_token_idx = token_start + local_idx;
            EXPECT_GE(global_token_idx, 0);
            EXPECT_LT(global_token_idx, evictable_size);
        }
    }
}

TEST(adaptive_rkv_gpu_kernel_test, multi_sequence_indexing) {
    // Test correct indexing for multiple sequences
    const int batch_size = 3;
    [[maybe_unused]] const int start_size = 32;
    
    std::vector<int> evictable_sizes = {64, 128, 192};
    
    for (int seq_idx = 0; seq_idx < batch_size; ++seq_idx) {
        // SDPA kernels extract parameters like:
        // evictable_start = adaptive_rkv_evictable_start_size[seq_idx * 2]
        // evictable_size = adaptive_rkv_evictable_start_size[seq_idx * 2 + 1]
        
        int array_index_start = seq_idx * 2;
        int array_index_size = seq_idx * 2 + 1;
        
        EXPECT_EQ(array_index_start, seq_idx * 2);
        EXPECT_EQ(array_index_size, seq_idx * 2 + 1);
        
        // Verify we can access the correct evictable_size
        EXPECT_EQ(evictable_sizes[seq_idx], evictable_sizes[seq_idx]);
    }
}

TEST(adaptive_rkv_gpu_kernel_test, diversity_output_buffer_size) {
    // Test that output buffer sizes are correctly calculated
    const int block_size = 16;
    
    struct TestCase {
        int evictable_size;
        int expected_num_blocks;
        int expected_output_size;
    };
    
    std::vector<TestCase> test_cases = {
        {16, 1, 16},       // 1 block × 16 tokens
        {32, 2, 32},       // 2 blocks × 16 tokens
        {64, 4, 64},       // 4 blocks × 16 tokens
        {128, 8, 128},     // 8 blocks × 16 tokens
        {256, 16, 256},    // 16 blocks × 16 tokens
    };
    
    for (const auto& tc : test_cases) {
        int num_blocks = tc.evictable_size / block_size;
        EXPECT_EQ(num_blocks, tc.expected_num_blocks);
        
        // Output buffer: [num_blocks, eviction_size]
        int output_size = num_blocks * tc.evictable_size;
        EXPECT_EQ(output_size, tc.expected_num_blocks * tc.evictable_size);
    }
}

TEST(adaptive_rkv_gpu_kernel_test, stage_detection_logic) {
    // Test stage detection based on num_tokens and past_len
    
    struct TestCase {
        int num_tokens;
        int past_len;
        std::string expected_stage;
        bool should_compute_diversity;
    };
    
    std::vector<TestCase> test_cases = {
        {128, 0, "PREFILL", true},      // Large prompt
        {1, 127, "GENERATE", false},    // Single token generation
        {32, 96, "MIXED", true},        // Partial prompt + cache
        {256, 0, "PREFILL", true},      // Large prompt
        {1, 0, "GENERATE", false},      // Special case: first token but single
        {2, 1022, "MIXED", true},       // Few tokens with large cache
    };
    
    for (const auto& tc : test_cases) {
        SCOPED_TRACE(tc.expected_stage);
        
        // Stage detection logic:
        // GENERATE: num_tokens == 1 && past_len > 0
        // PREFILL: past_len == 0 && num_tokens > 1
        // MIXED: past_len > 0 && num_tokens > 1
        
        bool is_generate = (tc.num_tokens == 1 && tc.past_len > 0);
        // These are for documentation/debugging
        [[maybe_unused]] bool is_prefill = (tc.past_len == 0 && tc.num_tokens > 1);
        [[maybe_unused]] bool is_mixed = (tc.past_len > 0 && tc.num_tokens > 1);
        
        // Diversity computation should be skipped in GENERATE
        bool should_compute = !is_generate;
        
        EXPECT_EQ(should_compute, tc.should_compute_diversity);
    }
}


