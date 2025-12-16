// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/paged_attention.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "paged_attention_inst.h"

#include "program_wrapper.h"

#include <cmath>
#include <algorithm>

using namespace cldnn;
using namespace ::tests;

namespace shape_infer_tests {

struct paged_attention_test_params {
    // Input layouts
    layout query_layout;
    layout key_layout;
    layout value_layout;
    layout key_cache_layout;
    layout value_cache_layout;
    layout past_lens_layout;
    layout subsequence_begins_layout;
    layout block_indices_layout;
    layout block_indices_begins_layout;
    layout max_context_len_layout;
    
    // Optional: Adaptive R-KV layouts
    bool has_adaptive_rkv;
    layout adaptive_rkv_start_size_layout;
    layout adaptive_rkv_evictable_sizes_layout;
    
    // PagedAttention parameters
    uint32_t heads_num;
    uint32_t kv_heads_num;
    uint32_t k_head_size;
    uint32_t v_head_size;
    uint32_t block_size;
    bool has_score_output;
    
    // Expected outputs
    layout expected_output_layout;
    std::vector<layout> expected_score_layouts;  // Score output (optional) + Adaptive RKV output (optional)
};

class paged_attention_si_test : public testing::TestWithParam<paged_attention_test_params> { };

TEST_P(paged_attention_si_test, shape_infer) {
    auto p = GetParam();
    auto& engine = get_test_engine();

    // Create input primitives
    auto query_prim = std::make_shared<input_layout>("query", p.query_layout);
    auto key_prim = std::make_shared<input_layout>("key", p.key_layout);
    auto value_prim = std::make_shared<input_layout>("value", p.value_layout);
    auto key_cache_prim = std::make_shared<input_layout>("key_cache", p.key_cache_layout);
    auto value_cache_prim = std::make_shared<input_layout>("value_cache", p.value_cache_layout);
    auto past_lens_prim = std::make_shared<input_layout>("past_lens", p.past_lens_layout);
    auto subsequence_begins_prim = std::make_shared<input_layout>("subsequence_begins", p.subsequence_begins_layout);
    auto block_indices_prim = std::make_shared<input_layout>("block_indices", p.block_indices_layout);
    auto block_indices_begins_prim = std::make_shared<input_layout>("block_indices_begins", p.block_indices_begins_layout);
    auto max_context_len_prim = std::make_shared<input_layout>("max_context_len", p.max_context_len_layout);
    
    // Create placeholder for optional inputs (to satisfy 25 input requirement)
    auto empty_layout = layout{ov::PartialShape{0}, data_types::f16, format::bfyx};
    auto placeholder_10 = std::make_shared<input_layout>("placeholder_10", empty_layout);
    auto placeholder_11 = std::make_shared<input_layout>("placeholder_11", empty_layout);
    auto placeholder_12 = std::make_shared<input_layout>("placeholder_12", empty_layout);
    auto placeholder_13 = std::make_shared<input_layout>("placeholder_13", empty_layout);
    auto placeholder_14 = std::make_shared<input_layout>("placeholder_14", empty_layout);
    auto placeholder_15 = std::make_shared<input_layout>("placeholder_15", empty_layout);
    auto placeholder_16 = std::make_shared<input_layout>("placeholder_16", empty_layout);
    auto placeholder_17 = std::make_shared<input_layout>("placeholder_17", empty_layout);
    auto placeholder_18 = std::make_shared<input_layout>("placeholder_18", empty_layout);
    auto placeholder_19 = std::make_shared<input_layout>("placeholder_19", empty_layout);
    auto placeholder_20 = std::make_shared<input_layout>("placeholder_20", empty_layout);
    auto placeholder_23 = std::make_shared<input_layout>("placeholder_23", empty_layout);
    auto placeholder_24 = std::make_shared<input_layout>("placeholder_24", empty_layout);
    
    // Create adaptive RKV input primitives if needed
    std::shared_ptr<input_layout> adaptive_rkv_start_size_prim;
    std::shared_ptr<input_layout> adaptive_rkv_evictable_sizes_prim;
    if (p.has_adaptive_rkv) {
        adaptive_rkv_start_size_prim = std::make_shared<input_layout>("adaptive_rkv_start_size", p.adaptive_rkv_start_size_layout);
        adaptive_rkv_evictable_sizes_prim = std::make_shared<input_layout>("adaptive_rkv_evictable_sizes", p.adaptive_rkv_evictable_sizes_layout);
    } else {
        adaptive_rkv_start_size_prim = std::make_shared<input_layout>("placeholder_21", empty_layout);
        adaptive_rkv_evictable_sizes_prim = std::make_shared<input_layout>("placeholder_22", empty_layout);
    }
    
    // Create PagedAttention primitive with all inputs
    std::vector<input_info> pa_inputs = {
        input_info("query"),                    // 0: QUERY
        input_info("key"),                      // 1: KEY
        input_info("value"),                    // 2: VALUE
        input_info("key_cache"),                // 3: KEY_CACHE
        input_info("value_cache"),              // 4: VALUE_CACHE
        input_info("past_lens"),                // 5: PAST_LENS
        input_info("subsequence_begins"),       // 6: SUBSEQUENCE_BEGINS
        input_info("block_indices"),            // 7: BLOCK_INDICES
        input_info("block_indices_begins"),     // 8: BLOCK_INDICES_BEGINS
        input_info("max_context_len"),          // 9: MAX_CONTEXT_LEN
        input_info("placeholder_10"),           // 10: ALIBI_SLOPES (optional)
        input_info("placeholder_11"),           // 11: SCALE (optional)
        input_info("placeholder_12"),           // 12: SCORE_AGGREGATION (optional)
        input_info("placeholder_13"),           // 13: WINDOW_SCORES (optional)
        input_info("placeholder_14"),           // 14: SLIDING_WINDOW (optional)
        input_info("placeholder_15"),           // 15: SCORE_OUTPUT (optional)
        input_info("placeholder_16"),           // 16: ROTATED_BLOCK_INDICES (optional)
        input_info("placeholder_17"),           // 17: ROTATED_BLOCK_COUNT (optional)
        input_info("placeholder_18"),           // 18: SUBSEQUENCE_SORTED_INDICES (optional)
        input_info("placeholder_19"),           // 19: SINK_KEYS (optional)
        input_info("placeholder_20"),           // 20: SINK_VALUES (optional)
        input_info(p.has_adaptive_rkv ? "adaptive_rkv_start_size" : "placeholder_21"),           // 21: ADAPTIVE_RKV_START_SIZE
        input_info(p.has_adaptive_rkv ? "adaptive_rkv_evictable_sizes" : "placeholder_22"),      // 22: ADAPTIVE_RKV_EVICTABLE_SIZES
        input_info("placeholder_23"),           // 23: ADAPTIVE_RKV_DIVERSITY_BLOCK_SET_INDICES (optional)
        input_info("placeholder_24"),           // 24: ADAPTIVE_RKV_DIVERSITY_BLOCK_SET_INDICES_BEGINS (optional)
    };
    
    auto pa_prim = std::make_shared<paged_attention>("paged_attention", pa_inputs);
    pa_prim->heads_num = p.heads_num;
    pa_prim->kv_heads_num = p.kv_heads_num;
    pa_prim->k_head_size = p.k_head_size;
    pa_prim->v_head_size = p.v_head_size;
    pa_prim->num_outputs = p.has_score_output ? 2 : 1;
    pa_prim->has_adaptive_rkv = p.has_adaptive_rkv;
    pa_prim->is_key_by_channel = true;  // Match execution config expectation
    // block_size is no longer a field in paged_attention primitive

    cldnn::program prog(engine);

    // Add nodes
    auto& query_node = prog.get_or_create(query_prim);
    auto& key_node = prog.get_or_create(key_prim);
    auto& value_node = prog.get_or_create(value_prim);
    auto& key_cache_node = prog.get_or_create(key_cache_prim);
    auto& value_cache_node = prog.get_or_create(value_cache_prim);
    auto& past_lens_node = prog.get_or_create(past_lens_prim);
    auto& subsequence_begins_node = prog.get_or_create(subsequence_begins_prim);
    auto& block_indices_node = prog.get_or_create(block_indices_prim);
    auto& block_indices_begins_node = prog.get_or_create(block_indices_begins_prim);
    auto& max_context_len_node = prog.get_or_create(max_context_len_prim);
    auto& pa_node = prog.get_or_create(pa_prim);
    
    // Add placeholder nodes for optional inputs
    auto& placeholder_10_node = prog.get_or_create(placeholder_10);
    auto& placeholder_11_node = prog.get_or_create(placeholder_11);
    auto& placeholder_12_node = prog.get_or_create(placeholder_12);
    auto& placeholder_13_node = prog.get_or_create(placeholder_13);
    auto& placeholder_14_node = prog.get_or_create(placeholder_14);
    auto& placeholder_15_node = prog.get_or_create(placeholder_15);
    auto& placeholder_16_node = prog.get_or_create(placeholder_16);
    auto& placeholder_17_node = prog.get_or_create(placeholder_17);
    auto& placeholder_18_node = prog.get_or_create(placeholder_18);
    auto& placeholder_19_node = prog.get_or_create(placeholder_19);
    auto& placeholder_20_node = prog.get_or_create(placeholder_20);
    auto& placeholder_23_node = prog.get_or_create(placeholder_23);
    auto& placeholder_24_node = prog.get_or_create(placeholder_24);
    auto& adaptive_rkv_start_size_node = prog.get_or_create(adaptive_rkv_start_size_prim);
    auto& adaptive_rkv_evictable_sizes_node = prog.get_or_create(adaptive_rkv_evictable_sizes_prim);
    
    // Connect inputs
    program_wrapper::add_connection(prog, query_node, pa_node);
    program_wrapper::add_connection(prog, key_node, pa_node);
    program_wrapper::add_connection(prog, value_node, pa_node);
    program_wrapper::add_connection(prog, key_cache_node, pa_node);
    program_wrapper::add_connection(prog, value_cache_node, pa_node);
    program_wrapper::add_connection(prog, past_lens_node, pa_node);
    program_wrapper::add_connection(prog, subsequence_begins_node, pa_node);
    program_wrapper::add_connection(prog, block_indices_node, pa_node);
    program_wrapper::add_connection(prog, block_indices_begins_node, pa_node);
    program_wrapper::add_connection(prog, max_context_len_node, pa_node);
    
    // Connect placeholders for optional inputs
    program_wrapper::add_connection(prog, placeholder_10_node, pa_node);
    program_wrapper::add_connection(prog, placeholder_11_node, pa_node);
    program_wrapper::add_connection(prog, placeholder_12_node, pa_node);
    program_wrapper::add_connection(prog, placeholder_13_node, pa_node);
    program_wrapper::add_connection(prog, placeholder_14_node, pa_node);
    program_wrapper::add_connection(prog, placeholder_15_node, pa_node);
    program_wrapper::add_connection(prog, placeholder_16_node, pa_node);
    program_wrapper::add_connection(prog, placeholder_17_node, pa_node);
    program_wrapper::add_connection(prog, placeholder_18_node, pa_node);
    program_wrapper::add_connection(prog, placeholder_19_node, pa_node);
    program_wrapper::add_connection(prog, placeholder_20_node, pa_node);
    program_wrapper::add_connection(prog, adaptive_rkv_start_size_node, pa_node);
    program_wrapper::add_connection(prog, adaptive_rkv_evictable_sizes_node, pa_node);
    program_wrapper::add_connection(prog, placeholder_23_node, pa_node);
    program_wrapper::add_connection(prog, placeholder_24_node, pa_node);

    auto res = paged_attention_inst::calc_output_layouts<ov::PartialShape>(pa_node, *pa_node.get_kernel_impl_params());

    // Verify output count
    size_t expected_output_count = 1;  // Main output
    if (p.has_score_output) {
        expected_output_count++;  // Score output
        if (p.has_adaptive_rkv) {
            expected_output_count++;  // Adaptive R-KV diversity output
        }
    }
    
    ASSERT_EQ(res.size(), expected_output_count);
    
    // Verify main output layout
    ASSERT_EQ(res[0], p.expected_output_layout);
    
    // Verify additional outputs if present
    if (p.has_score_output) {
        for (size_t i = 0; i < p.expected_score_layouts.size(); i++) {
            ASSERT_EQ(res[i + 1], p.expected_score_layouts[i]);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(smoke, paged_attention_si_test,
    testing::ValuesIn(std::vector<paged_attention_test_params>{
        // Test 1: Basic PREFILL stage (static shapes)
        {
            layout{ov::PartialShape{128, 2048}, data_types::f16, format::bfyx},  // query: [num_tokens, heads*head_size]
            layout{ov::PartialShape{128, 1024}, data_types::f16, format::bfyx},  // key
            layout{ov::PartialShape{128, 1024}, data_types::f16, format::bfyx},  // value
            layout{ov::PartialShape{64, 16, 64, 16}, data_types::f16, format::bfyx},  // key_cache: [num_blocks, kv_heads, head_size, block_size]
            layout{ov::PartialShape{64, 16, 16, 64}, data_types::f16, format::bfyx},  // value_cache: [num_blocks, kv_heads, block_size, head_size]
            layout{ov::PartialShape{1}, data_types::i32, format::bfyx},  // past_lens
            layout{ov::PartialShape{2}, data_types::i32, format::bfyx},  // subsequence_begins
            layout{ov::PartialShape{8}, data_types::i32, format::bfyx},  // block_indices
            layout{ov::PartialShape{2}, data_types::i32, format::bfyx},  // block_indices_begins
            layout{ov::PartialShape{1}, data_types::i32, format::bfyx},  // max_context_len
            false,  // has_adaptive_rkv
            layout{ov::PartialShape{}, data_types::i32, format::bfyx},  // adaptive_rkv_start_size (unused)
            layout{ov::PartialShape{}, data_types::i32, format::bfyx},  // adaptive_rkv_evictable_sizes (unused)
            32, 16, 64, 64, 16,  // heads_num, kv_heads_num, k_head_size, v_head_size, block_size
            false,  // has_score_output
            layout{ov::PartialShape{128, 2048}, data_types::f16, format::bfyx},  // expected_output
            {}  // no score outputs
        },
        // Test 2: GENERATE stage with different k/v head sizes
        {
            layout{ov::PartialShape{1, 1024}, data_types::f16, format::bfyx},  // query: single token
            layout{ov::PartialShape{1, 512}, data_types::f16, format::bfyx},  // key
            layout{ov::PartialShape{1, 1024}, data_types::f16, format::bfyx},  // value
            layout{ov::PartialShape{32, 8, 64, 16}, data_types::f16, format::bfyx},  // key_cache
            layout{ov::PartialShape{32, 8, 16, 128}, data_types::f16, format::bfyx},  // value_cache
            layout{ov::PartialShape{1}, data_types::i32, format::bfyx},  // past_lens
            layout{ov::PartialShape{2}, data_types::i32, format::bfyx},  // subsequence_begins
            layout{ov::PartialShape{4}, data_types::i32, format::bfyx},  // block_indices
            layout{ov::PartialShape{2}, data_types::i32, format::bfyx},  // block_indices_begins
            layout{ov::PartialShape{1}, data_types::i32, format::bfyx},  // max_context_len
            false,  // has_adaptive_rkv
            layout{ov::PartialShape{}, data_types::i32, format::bfyx},
            layout{ov::PartialShape{}, data_types::i32, format::bfyx},
            16, 8, 64, 128, 16,  // Different k_head_size and v_head_size
            false,  // has_score_output
            layout{ov::PartialShape{1, 2048}, data_types::f16, format::bfyx},  // expected_output: [1, heads_num * v_head_size]
            {}
        },
        // Test 3: With score output
        {
            layout{ov::PartialShape{64, 1024}, data_types::f16, format::bfyx},
            layout{ov::PartialShape{64, 512}, data_types::f16, format::bfyx},
            layout{ov::PartialShape{64, 512}, data_types::f16, format::bfyx},
            layout{ov::PartialShape{16, 8, 64, 16}, data_types::f16, format::bfyx},
            layout{ov::PartialShape{16, 8, 16, 64}, data_types::f16, format::bfyx},
            layout{ov::PartialShape::dynamic(1), data_types::i32, format::bfyx},  // past_lens (dynamic to skip memory read)
            layout{ov::PartialShape{3}, data_types::i32, format::bfyx},
            layout{ov::PartialShape{8}, data_types::i32, format::bfyx},
            layout{ov::PartialShape{3}, data_types::i32, format::bfyx},
            layout{ov::PartialShape{1}, data_types::i32, format::bfyx},
            false,  // has_adaptive_rkv
            layout{ov::PartialShape{}, data_types::i32, format::bfyx},
            layout{ov::PartialShape{}, data_types::i32, format::bfyx},
            16, 8, 64, 64, 16,
            true,  // has_score_output
            layout{ov::PartialShape{64, 1024}, data_types::f16, format::bfyx},
            {layout{ov::PartialShape::dynamic(1), data_types::f16, format::bfyx}}  // score output (dynamic)
        },
        // Test 4: Dynamic shapes
        {
            layout{ov::PartialShape::dynamic(2), data_types::f16, format::bfyx},
            layout{ov::PartialShape::dynamic(2), data_types::f16, format::bfyx},
            layout{ov::PartialShape::dynamic(2), data_types::f16, format::bfyx},
            layout{ov::PartialShape::dynamic(4), data_types::f16, format::bfyx},
            layout{ov::PartialShape::dynamic(4), data_types::f16, format::bfyx},
            layout{ov::PartialShape::dynamic(1), data_types::i32, format::bfyx},
            layout{ov::PartialShape::dynamic(1), data_types::i32, format::bfyx},
            layout{ov::PartialShape::dynamic(1), data_types::i32, format::bfyx},
            layout{ov::PartialShape::dynamic(1), data_types::i32, format::bfyx},
            layout{ov::PartialShape{1}, data_types::i32, format::bfyx},
            false,  // has_adaptive_rkv
            layout{ov::PartialShape{}, data_types::i32, format::bfyx},
            layout{ov::PartialShape{}, data_types::i32, format::bfyx},
            32, 32, 64, 64, 16,
            false,  // has_score_output
            layout{ov::PartialShape::dynamic(2), data_types::f16, format::bfyx},
            {}
        },
        // Test 5: With Adaptive R-KV (single sequence)
        {
            layout{ov::PartialShape{256, 2048}, data_types::f16, format::bfyx},
            layout{ov::PartialShape{256, 1024}, data_types::f16, format::bfyx},
            layout{ov::PartialShape{256, 1024}, data_types::f16, format::bfyx},
            layout{ov::PartialShape{64, 16, 64, 16}, data_types::f16, format::bfyx},
            layout{ov::PartialShape{64, 16, 16, 64}, data_types::f16, format::bfyx},
            layout{ov::PartialShape::dynamic(1), data_types::i32, format::bfyx},  // past_lens (dynamic to skip memory read)
            layout{ov::PartialShape{2}, data_types::i32, format::bfyx},
            layout{ov::PartialShape{16}, data_types::i32, format::bfyx},
            layout{ov::PartialShape{2}, data_types::i32, format::bfyx},
            layout{ov::PartialShape{1}, data_types::i32, format::bfyx},
            true,  // has_adaptive_rkv
            layout{ov::PartialShape::dynamic(1), data_types::i32, format::bfyx},  // start_size (dynamic to skip memory read)
            layout{ov::PartialShape::dynamic(1), data_types::i32, format::bfyx},  // evictable_sizes (dynamic to skip memory read)
            32, 16, 64, 64, 16,
            true,  // has_score_output
            layout{ov::PartialShape{256, 2048}, data_types::f16, format::bfyx},
            {
                layout{ov::PartialShape::dynamic(1), data_types::f16, format::bfyx},  // score output
                layout{ov::PartialShape::dynamic(1), data_types::f16, format::bfyx}   // adaptive rkv output
            }
        },
        // Test 6: With Adaptive R-KV (multi-sequence)
        {
            layout{ov::PartialShape{320, 2048}, data_types::f16, format::bfyx},
            layout{ov::PartialShape{320, 1024}, data_types::f16, format::bfyx},
            layout{ov::PartialShape{320, 1024}, data_types::f16, format::bfyx},
            layout{ov::PartialShape{80, 16, 64, 16}, data_types::f16, format::bfyx},
            layout{ov::PartialShape{80, 16, 16, 64}, data_types::f16, format::bfyx},
            layout{ov::PartialShape::dynamic(1), data_types::i32, format::bfyx},  // past_lens (dynamic to skip memory read)
            layout{ov::PartialShape{4}, data_types::i32, format::bfyx},
            layout{ov::PartialShape{24}, data_types::i32, format::bfyx},
            layout{ov::PartialShape{4}, data_types::i32, format::bfyx},
            layout{ov::PartialShape{1}, data_types::i32, format::bfyx},
            true,  // has_adaptive_rkv
            layout{ov::PartialShape::dynamic(1), data_types::i32, format::bfyx},  // start_size (dynamic to skip memory read)
            layout{ov::PartialShape::dynamic(1), data_types::i32, format::bfyx},  // evictable_sizes (dynamic to skip memory read)
            32, 16, 64, 64, 16,
            true,  // has_score_output
            layout{ov::PartialShape{320, 2048}, data_types::f16, format::bfyx},
            {
                layout{ov::PartialShape::dynamic(1), data_types::f16, format::bfyx},  // score output
                layout{ov::PartialShape::dynamic(1), data_types::f16, format::bfyx}   // adaptive rkv diversity output
            }
        },
        // Test 7: GQA (Grouped Query Attention)
        {
            layout{ov::PartialShape{128, 4096}, data_types::f16, format::bfyx},  // 32 heads * 128 head_size
            layout{ov::PartialShape{128, 512}, data_types::f16, format::bfyx},   // 4 kv_heads * 128 head_size
            layout{ov::PartialShape{128, 512}, data_types::f16, format::bfyx},
            layout{ov::PartialShape{32, 4, 128, 16}, data_types::f16, format::bfyx},
            layout{ov::PartialShape{32, 4, 16, 128}, data_types::f16, format::bfyx},
            layout{ov::PartialShape{1}, data_types::i32, format::bfyx},
            layout{ov::PartialShape{2}, data_types::i32, format::bfyx},
            layout{ov::PartialShape{8}, data_types::i32, format::bfyx},
            layout{ov::PartialShape{2}, data_types::i32, format::bfyx},
            layout{ov::PartialShape{1}, data_types::i32, format::bfyx},
            false,  // has_adaptive_rkv
            layout{ov::PartialShape{}, data_types::i32, format::bfyx},
            layout{ov::PartialShape{}, data_types::i32, format::bfyx},
            32, 4, 128, 128, 16,  // GQA: 32 query heads, 4 KV heads
            false,  // has_score_output
            layout{ov::PartialShape{128, 4096}, data_types::f16, format::bfyx},
            {}
        },
        // Test 8: Compressed KV cache (i8) - block size must account for scale/zp storage
        {
            layout{ov::PartialShape{64, 1024}, data_types::f16, format::bfyx},
            layout{ov::PartialShape{64, 512}, data_types::f16, format::bfyx},
            layout{ov::PartialShape{64, 512}, data_types::f16, format::bfyx},
            layout{ov::PartialShape{16, 8, 64, 20}, data_types::i8, format::bfyx},  // Compressed key cache (block_size=16+4 for scale/zp)
            layout{ov::PartialShape{16, 8, 20, 64}, data_types::i8, format::bfyx},  // Compressed value cache (block_size=16+4 for scale/zp)
            layout{ov::PartialShape{1}, data_types::i32, format::bfyx},
            layout{ov::PartialShape{2}, data_types::i32, format::bfyx},
            layout{ov::PartialShape{4}, data_types::i32, format::bfyx},
            layout{ov::PartialShape{2}, data_types::i32, format::bfyx},
            layout{ov::PartialShape{1}, data_types::i32, format::bfyx},
            false,  // has_adaptive_rkv (should be disabled for compressed cache)
            layout{ov::PartialShape{}, data_types::i32, format::bfyx},
            layout{ov::PartialShape{}, data_types::i32, format::bfyx},
            16, 8, 64, 64, 16,
            false,  // has_score_output
            layout{ov::PartialShape{64, 1024}, data_types::f16, format::bfyx},
            {}
        },
    }));

}  // namespace shape_infer_tests
