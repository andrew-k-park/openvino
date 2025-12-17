// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"
#include "kernel_selector_params.h"
#include "../utils/jitter.hpp"

namespace ov::intel_gpu::ocl {

struct AdaptiveRKVDiversityKernelParams : kernel_selector::Params {
    size_t batch_size = 0;
    size_t num_heads = 0;
    size_t head_size = 0;
    size_t max_evictable_size = 0;
    size_t paged_attention_block_size = 16;
    
    ov::element::Type input_dt = ov::element::f16;
    ov::element::Type output_dt = ov::element::f32;
};

class AdaptiveRKVDiversityKernel {
public:
    static std::string GetName() { return "adaptive_rkv_diversity"; }
    
    static JitConstants GetJitConstants(const AdaptiveRKVDiversityKernelParams& params) {
        JitConstants jit;
        
        jit.add(make_type_jit_constants("INPUT0", params.input_dt));
        jit.add(make_type_jit_constants("OUTPUT", params.output_dt));
        jit.add(make_type_jit_constants("SOFTMAX_ACCUMULATOR", ov::element::f32));
        
        jit.make("BATCH_SIZE", params.batch_size);
        jit.make("NUM_HEADS", params.num_heads);
        jit.make("HEAD_SIZE", params.head_size);
        jit.make("MAX_EVICTABLE_SIZE", params.max_evictable_size);
        jit.make("PAGED_ATTENTION_BLOCK_SIZE", params.paged_attention_block_size);
        jit.make("SUBGROUP_SIZE", 16);
        
        return jit;
    }
    
    static std::string GetKernelSource() {
        return "adaptive_rkv_diversity.cl";
    }
    
    static kernel_selector::WorkGroupSizes GetWorkGroupSizes(const AdaptiveRKVDiversityKernelParams& params) {
        // Global work size: [batch_size, num_heads, max_evictable_size]
        kernel_selector::WorkGroupSizes wgs;
        wgs.global = {params.batch_size, params.num_heads, params.max_evictable_size};
        wgs.local = {1, 1, std::min(params.max_evictable_size, static_cast<size_t>(256))};
        return wgs;
    }
};

} // namespace ov::intel_gpu::ocl
