// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "adaptive_rkv_diversity_kernel.hpp"
#include "../primitive_ocl_base.hpp"
#include "../utils/kernel_generator.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/paged_attention.hpp"
#include "paged_attention_inst.h"

namespace ov::intel_gpu::ocl {

struct AdaptiveRKVDiversityImpl : typed_primitive_impl_ocl<paged_attention> {
    using parent = typed_primitive_impl_ocl<paged_attention>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return std::make_unique<AdaptiveRKVDiversityImpl>(*this);
    }

    void save(BinaryOutputBuffer& ob) const override {
        parent::save(ob);
    }

    void load(BinaryInputBuffer& ib) override {
        parent::load(ib);
    }

    static std::unique_ptr<primitive_impl> create(const typed_program_node<paged_attention>& node,
                                                   const kernel_impl_params& params) {
        auto desc = params.typed_desc<paged_attention>();
        
        // Only create this implementation if adaptive R-KV is enabled
        if (!desc->has_adaptive_rkv) {
            return nullptr;
        }

        AdaptiveRKVDiversityKernelParams kernel_params;
        
        // Extract parameters from paged_attention descriptor
        kernel_params.batch_size = 1; // Will be set dynamically
        kernel_params.num_heads = desc->heads_num;
        kernel_params.head_size = desc->k_head_size;
        kernel_params.max_evictable_size = 512; // Conservative estimate, should be JIT constant
        kernel_params.paged_attention_block_size = 16;
        
        // Determine data types
        const auto& key_cache_layout = params.get_input_layout(PagedAttentionInputIdx::KEY_CACHE);
        kernel_params.input_dt = key_cache_layout.data_type;
        kernel_params.output_dt = ov::element::f32;

        auto jit = AdaptiveRKVDiversityKernel::GetJitConstants(kernel_params);
        auto entry_point = AdaptiveRKVDiversityKernel::GetName();
        auto kernel_source = AdaptiveRKVDiversityKernel::GetKernelSource();
        
        // Create kernel
        auto kernel_code = KernelGenerator::create_jit_source(kernel_source, jit);
        auto kernels_data = KernelData::create(params, {kernel_code}, entry_point);
        
        kernels_data.internalBufferSizes.clear();
        kernels_data.internalBufferDataType = kernel_params.output_dt;
        
        // Set work group sizes
        auto wgs = AdaptiveRKVDiversityKernel::GetWorkGroupSizes(kernel_params);
        kernels_data.params.workGroups = wgs;

        return cldnn::make_unique<AdaptiveRKVDiversityImpl>(kernels_data);
    }
};

namespace detail {

attach_paged_attention_impl(AdaptiveRKVDiversityImpl, "adaptive_rkv_diversity");

} // namespace detail
} // namespace ov::intel_gpu::ocl
