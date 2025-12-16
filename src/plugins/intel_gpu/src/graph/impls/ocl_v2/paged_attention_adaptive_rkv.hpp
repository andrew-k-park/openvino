// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/primitives/paged_attention.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "adaptive_rkv_diversity.hpp"

#include <memory>
#include <vector>

using namespace cldnn;  // TODO: Remove once namespaces are aligned

namespace ov::intel_gpu::ocl {

// Forward declaration - should match paged_attention_opt.cpp
enum class PagedAttentionStage : uint8_t { GENERATE = 0, PREFILL = 1, MIXED = 2, UNKNOWN = 3 };

// Convenience alias for PagedAttention input indices
using PagedAttentionInputIdx = paged_attention::PagedAttentionInputIdx;

/**
 * @brief Integration point for Adaptive R-KV diversity calculation in Paged Attention
 * 
 * This class manages the integration of Adaptive R-KV diversity computation
 * with the existing Paged Attention implementation. It orchestrates:
 * 
 * Step 2: Compute Similarity
 * - Normalize keys
 * - Compute cosine similarity matrix
 * - Slice evictable region
 * - Fill diagonal with zeros
 * - Apply mean threshold filtering
 * - Aggregate across heads
 * 
 * Step 3: Group Diversity Filtering (Partial)
 * - Compute block-wise diversity sums
 * - Output: [num_blocks, eviction_size] matrix for genai layer
 * 
 * The final step (mask application and mean reduction) is delegated to
 * the genai layer where block retention information is available.
 */
class PagedAttentionAdaptiveRKVIntegration {
public:
    PagedAttentionAdaptiveRKVIntegration() = default;

    /**
     * @brief Check if Adaptive R-KV should be computed for this execution
     * 
     * @param impl_params Kernel implementation parameters
     * @param stage Current paged attention execution stage
     * @return true if diversity should be computed for this stage
     */
    static bool should_compute_diversity(const kernel_impl_params& impl_params, 
                                        PagedAttentionStage stage);

    /**
     * @brief Compute diversity values during paged attention execution
     * 
     * This is called from within the paged attention kernel execution
     * when has_adaptive_rkv is true. It executes Step 2 and partial Step 3.
     * 
     * @param impl_params Kernel implementation parameters
     * @param stage Current paged attention execution stage
     * @param diversity_output Output buffer for diversity values [num_blocks, eviction_size]
     * @param events Dependency events to wait for
     * @return Event representing completion of diversity computation
     */
    static event::ptr compute_diversity(
        const kernel_impl_params& impl_params,
        PagedAttentionStage stage,
        memory::ptr diversity_output,
        const std::vector<event::ptr>& events);

    /**
     * @brief Get required output buffer size for diversity computation
     * 
     * @param eviction_size Size of evictable region in tokens
     * @param block_size Paged attention block size
     * @return Layout for diversity output buffer
     */
    static layout get_diversity_output_layout(
        int eviction_size,
        int block_size);

private:
    /**
     * @brief Execute the full diversity computation pipeline
     */
    static event::ptr execute_diversity_pipeline(
        const kernel_impl_params& impl_params,
        const AdaptiveRKVDiversityKernelSelector::KernelParams& params,
        memory::ptr output);

    /**
     * @brief Optimized path using fused kernel
     */
    static event::ptr execute_diversity_fused(
        const kernel_impl_params& impl_params,
        const AdaptiveRKVDiversityKernelSelector::KernelParams& params,
        memory::ptr output);
};

/**
 * @brief Runtime parameters for Adaptive R-KV in paged attention
 */
struct AdaptiveRKVRuntimeParams {
    int start_size;            // From ADAPTIVE_RKV_START_SIZE input
    int eviction_size;         // From ADAPTIVE_RKV_EVICTABLE_SIZES input
    int block_size;            // Paged attention block size (16)
    int num_kv_heads;          // Number of KV heads
    int head_size;             // Head dimension size
    int num_tokens;            // Total number of tokens in cache
    int sequence_idx;          // Current sequence index
    
    bool is_valid() const {
        return eviction_size > 0 && 
               eviction_size % block_size == 0 &&
               start_size % block_size == 0;
    }
    
    int get_num_blocks() const {
        return eviction_size / block_size;
    }
};

/**
 * @brief Extract Adaptive R-KV parameters from paged attention inputs
 */
inline AdaptiveRKVRuntimeParams extract_adaptive_rkv_params(
    const kernel_impl_params& impl_params,
    int sequence_idx = 0) {
    
    AdaptiveRKVRuntimeParams params;
    
    const auto& desc = impl_params.typed_desc<paged_attention>();
    params.num_kv_heads = static_cast<int>(desc->kv_heads_num);
    params.head_size = static_cast<int>(desc->k_head_size);
    params.block_size = static_cast<int>(paged_attention::block_size);
    params.sequence_idx = sequence_idx;
    
    if (desc->has_adaptive_rkv) {
        // Read start_size (scalar)
        auto start_size_mem = impl_params.memory_deps.at(
            PagedAttentionInputIdx::ADAPTIVE_RKV_START_SIZE);
        mem_lock<int32_t, mem_lock_type::read> start_size_lock(
            start_size_mem, *impl_params.strm);
        params.start_size = start_size_lock[0];
        
        // Read eviction_size for this sequence
        auto evictable_sizes_mem = impl_params.memory_deps.at(
            PagedAttentionInputIdx::ADAPTIVE_RKV_EVICTABLE_SIZES);
        mem_lock<int32_t, mem_lock_type::read> evictable_sizes_lock(
            evictable_sizes_mem, *impl_params.strm);
        params.eviction_size = evictable_sizes_lock[sequence_idx];
        
        // Calculate num_tokens from block_indices
        auto block_indices_begins = impl_params.memory_deps.at(
            PagedAttentionInputIdx::BLOCK_INDICES_BEGINS);
        mem_lock<int32_t, mem_lock_type::read> block_begins_lock(
            block_indices_begins, *impl_params.strm);
        
        const int block_start = block_begins_lock[sequence_idx];
        const int block_end = block_begins_lock[sequence_idx + 1];
        const int num_blocks = block_end - block_start;
        params.num_tokens = num_blocks * params.block_size;
    } else {
        params.start_size = 0;
        params.eviction_size = 0;
        params.num_tokens = 0;
    }
    
    return params;
}

} // namespace ov::intel_gpu::ocl
