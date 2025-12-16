// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/primitives/paged_attention.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "primitive_inst.h"
#include "primitive_ocl_base.hpp"
#include "common_utils/jitter.hpp"

#include <memory>
#include <string>
#include <vector>

using namespace cldnn;  // TODO: Remove once namespaces are aligned

namespace ov::intel_gpu::ocl {

// Kernel selector for Adaptive R-KV diversity calculation
struct AdaptiveRKVDiversityKernelSelector {
    enum class KernelType {
        NORMALIZE_KEYS,
        COMPUTE_SIMILARITY,
        SLICE_AND_FILL_DIAGONAL,
        THRESHOLD_BY_MEAN,
        AGGREGATE_HEADS,
        BLOCK_SUM_DIVERSITY,
        APPLY_MASK_AND_REDUCE,
        COMPUTE_DIVERSITY_FUSED
    };

    struct KernelParams {
        int num_kv_heads;
        int head_size;
        int block_size;
        int num_tokens;
        int start_size;
        int eviction_size;
        int sequence_idx;
        ov::element::Type input_type;
        ov::element::Type output_type;
        bool use_fused = false;  // Use staged pipeline for debugging (fused kernel has argument issues)
    };

    static std::string get_kernel_name(KernelType type) {
        switch (type) {
            case KernelType::NORMALIZE_KEYS:
                return "adaptive_rkv_normalize_keys";
            case KernelType::COMPUTE_SIMILARITY:
                return "adaptive_rkv_compute_similarity";
            case KernelType::SLICE_AND_FILL_DIAGONAL:
                return "adaptive_rkv_slice_and_fill_diagonal";
            case KernelType::THRESHOLD_BY_MEAN:
                return "adaptive_rkv_threshold_by_mean";
            case KernelType::AGGREGATE_HEADS:
                return "adaptive_rkv_aggregate_heads";
            case KernelType::BLOCK_SUM_DIVERSITY:
                return "adaptive_rkv_block_sum_diversity";
            case KernelType::APPLY_MASK_AND_REDUCE:
                return "adaptive_rkv_apply_mask_and_reduce";
            case KernelType::COMPUTE_DIVERSITY_FUSED:
                return "adaptive_rkv_compute_diversity_fused";
            default:
                return "";
        }
    }

    static JitConstants get_jit_constants(const KernelParams& params) {
        JitConstants jit;
        
        jit.make("NUM_KV_HEADS", params.num_kv_heads);
        jit.make("HEAD_SIZE", params.head_size);
        jit.make("BLOCK_SIZE", params.block_size);
        jit.make("NUM_TOKENS", params.num_tokens);
        jit.make("START_SIZE", params.start_size);
        jit.make("EVICTION_SIZE", params.eviction_size);
        
        return jit;
    }

    static std::vector<size_t> get_global_work_size(KernelType type, const KernelParams& params) {
        constexpr size_t subgroup_size = 16;
        
        switch (type) {
            case KernelType::NORMALIZE_KEYS:
                return {
                    align_to(params.num_tokens, subgroup_size),
                    static_cast<size_t>(params.num_kv_heads),
                    1
                };
                
            case KernelType::COMPUTE_SIMILARITY:
                return {
                    align_to(params.num_tokens, subgroup_size),
                    align_to(params.num_tokens, subgroup_size),
                    static_cast<size_t>(params.num_kv_heads)
                };
                
            case KernelType::SLICE_AND_FILL_DIAGONAL:
                return {
                    align_to(params.eviction_size, subgroup_size),
                    align_to(params.eviction_size, subgroup_size),
                    static_cast<size_t>(params.num_kv_heads)
                };
                
            case KernelType::THRESHOLD_BY_MEAN:
                return {
                    align_to(params.eviction_size, subgroup_size),
                    static_cast<size_t>(params.num_kv_heads),
                    1
                };
                
            case KernelType::AGGREGATE_HEADS:
                return {
                    align_to(params.eviction_size, subgroup_size),
                    align_to(params.eviction_size, subgroup_size),
                    1
                };
                
            case KernelType::BLOCK_SUM_DIVERSITY: {
                const int num_blocks = params.eviction_size / params.block_size;
                return {
                    align_to(num_blocks, subgroup_size),
                    align_to(params.eviction_size, subgroup_size),
                    1
                };
            }
                
            case KernelType::APPLY_MASK_AND_REDUCE: {
                const int num_blocks = params.eviction_size / params.block_size;
                return {
                    align_to(num_blocks, subgroup_size),
                    1,
                    1
                };
            }
                
            case KernelType::COMPUTE_DIVERSITY_FUSED:
                // For fused kernel, optimize based on overall computation
                return {
                    align_to(params.eviction_size / params.block_size, subgroup_size),
                    align_to(params.eviction_size, subgroup_size),
                    1
                };
                
            default:
                return {subgroup_size, 1, 1};
        }
    }

    static std::vector<size_t> get_local_work_size(KernelType type) {
        constexpr size_t subgroup_size = 16;
        
        switch (type) {
            case KernelType::NORMALIZE_KEYS:
            case KernelType::THRESHOLD_BY_MEAN:
            case KernelType::APPLY_MASK_AND_REDUCE:
                return {subgroup_size, 1, 1};
                
            case KernelType::COMPUTE_SIMILARITY:
            case KernelType::SLICE_AND_FILL_DIAGONAL:
            case KernelType::AGGREGATE_HEADS:
            case KernelType::BLOCK_SUM_DIVERSITY:
                return {subgroup_size, 1, 1};
                
            case KernelType::COMPUTE_DIVERSITY_FUSED:
                return {subgroup_size, 1, 1};
                
            default:
                return {subgroup_size, 1, 1};
        }
    }

private:
    static size_t align_to(size_t value, size_t alignment) {
        return ((value + alignment - 1) / alignment) * alignment;
    }
};

// Helper class for Adaptive R-KV diversity calculation
// Not a standalone primitive - used by PagedAttention
class AdaptiveRKVDiversityImpl {
public:
    AdaptiveRKVDiversityImpl() = default;

    struct IntermediateBuffers {
        memory::ptr normalized_keys;
        memory::ptr similarity_matrix;
        memory::ptr evictable_sim;
        memory::ptr aggregated_sim;
        memory::ptr block_diversity;
    };

    // Note: Kernel creation and execution are handled by the PagedAttention implementation
    // These helper methods provide the structure for future integration
    static void execute(const std::vector<event::ptr>& events, 
                        primitive_inst& instance);

    static IntermediateBuffers allocate_intermediate_buffers(
        const kernel_impl_params& impl_params,
        int num_kv_heads,
        int num_tokens,
        int head_size,
        int eviction_size,
        int block_size);

    static kernel_arguments_data get_arguments(
        const AdaptiveRKVDiversityKernelSelector::KernelType type,
        const kernel_impl_params& impl_params,
        const IntermediateBuffers& buffers);
};

} // namespace ov::intel_gpu::ocl
