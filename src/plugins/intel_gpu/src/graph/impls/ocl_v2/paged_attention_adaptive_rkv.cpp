// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "paged_attention_adaptive_rkv.hpp"
#include "adaptive_rkv_diversity.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/event.hpp"
#include "utils/jitter.hpp"
#include "utils/kernels_db.hpp"
#include "../ocl/kernels_cache.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned

namespace ov::intel_gpu::ocl {

using PagedAttentionInputIdx = paged_attention::PagedAttentionInputIdx;

bool PagedAttentionAdaptiveRKVIntegration::should_compute_diversity(
    const kernel_impl_params& impl_params,
    PagedAttentionStage stage) {
    
    const auto& desc = impl_params.typed_desc<paged_attention>();
    
    // Only compute diversity if:
    // 1. Adaptive RKV is enabled
    // 2. We have the required inputs
    // 3. This is the appropriate execution stage
    
    if (!desc->has_adaptive_rkv) {
        return false;
    }
    
    // KV cache compression check:
    // Compressed caches (i8/u8) are quantized, making accurate similarity computation difficult
    // Options:
    //   1. Skip diversity calculation in compressed mode (current approach)
    //   2. Dequantize keys before diversity calculation (performance overhead)
    //   3. Approximate diversity using quantized values (accuracy loss)
    auto key_cache_layout = impl_params.get_input_layout(PagedAttentionInputIdx::KEY_CACHE);
    if (data_type_traits::is_i8_u8(key_cache_layout.data_type)) {
        // Skip diversity calculation for compressed KV cache
        // Quantization error would significantly impact similarity measurements
        return false;
    }
    
    // Stage-based filtering:
    // - GENERATE: Single token generation - diversity calculation is inefficient and unnecessary
    //             The KV cache grows by only 1 token, not enough to warrant full recomputation
    // - PREFILL: Initial prompt processing - diversity calculation is meaningful
    //            Process entire prompt, establish initial cache eviction strategy
    // - MIXED: Mixed prompt and generation - diversity calculation may be needed
    //          Some sequences are in prompt phase, requiring diversity computation
    // - UNKNOWN: Cannot determine stage - skip for safety
    
    switch (stage) {
        case PagedAttentionStage::GENERATE:
            // Skip diversity calculation during single token generation
            // Incremental updates are too expensive for single token addition
            return false;
            
        case PagedAttentionStage::PREFILL:
        case PagedAttentionStage::MIXED:
            // Compute diversity for prompt processing and mixed scenarios
            break;
            
        case PagedAttentionStage::UNKNOWN:
        default:
            // Unknown stage - skip for safety
            return false;
    }
    
    // Verify required inputs are present
    const auto& input_layouts = impl_params.input_layouts;
    if (input_layouts.size() < PagedAttentionInputIdx::ADAPTIVE_RKV_EVICTABLE_SIZES + 1) {
        return false;
    }
    
    return true;
}

layout PagedAttentionAdaptiveRKVIntegration::get_diversity_output_layout(
    int eviction_size,
    int block_size) {
    
    const int num_blocks = eviction_size / block_size;
    
    // Output layout: [num_blocks, eviction_size]
    // This represents per-block diversity values for each token position
    // The genai layer will apply row mask and mean reduction
    return layout(
        ov::element::f32,
        format::bfyx,
        tensor(batch(1), feature(num_blocks), spatial(eviction_size, 1)));
}

event::ptr PagedAttentionAdaptiveRKVIntegration::compute_diversity(
    const kernel_impl_params& impl_params,
    PagedAttentionStage stage,
    memory::ptr diversity_output,
    const std::vector<event::ptr>& events) {
    
    if (!should_compute_diversity(impl_params, stage)) {
        return nullptr;
    }
    
    // Extract runtime parameters
    auto params_runtime = extract_adaptive_rkv_params(impl_params, 0);
    
    if (!params_runtime.is_valid()) {
        // Invalid parameters, return without computing diversity
        return nullptr;
    }
    
    // Convert to kernel parameters
    // Note: GQA (Grouped Query Attention) is automatically supported
    // by using num_kv_heads instead of num_heads
    AdaptiveRKVDiversityKernelSelector::KernelParams kernel_params;
    kernel_params.num_kv_heads = params_runtime.num_kv_heads;
    kernel_params.head_size = params_runtime.head_size;
    kernel_params.block_size = params_runtime.block_size;
    kernel_params.num_tokens = params_runtime.num_tokens;
    kernel_params.start_size = params_runtime.start_size;
    kernel_params.eviction_size = params_runtime.eviction_size;
    kernel_params.sequence_idx = params_runtime.sequence_idx;
    kernel_params.input_type = impl_params.get_input_layout(
        PagedAttentionInputIdx::KEY_CACHE).data_type;
    kernel_params.output_type = ov::element::f32;
    
    // Execute diversity computation pipeline
    // Use fused kernel by default for better performance
    // Can be disabled by setting kernel_params.use_fused = false
    
    if (kernel_params.use_fused) {
        return execute_diversity_fused(impl_params, kernel_params, diversity_output);
    } else {
        return execute_diversity_pipeline(impl_params, kernel_params, diversity_output);
    }
}

event::ptr PagedAttentionAdaptiveRKVIntegration::execute_diversity_pipeline(
    const kernel_impl_params& impl_params,
    const AdaptiveRKVDiversityKernelSelector::KernelParams& params,
    memory::ptr output) {
    
    auto& stream = *impl_params.strm;
    auto& kernels_cache = const_cast<program*>(impl_params.prog)->get_kernels_cache();
    
    // Allocate intermediate buffers
    auto buffers = AdaptiveRKVDiversityImpl::allocate_intermediate_buffers(
        impl_params,
        params.num_kv_heads,
        params.num_tokens,
        params.head_size,
        params.eviction_size,
        params.block_size);
    
    std::vector<event::ptr> stage_events;
    
    // Get JIT constants for kernel compilation
    JitConstants jit_constants;
    jit_constants.add(make_jit_constant("INPUT0_TYPE", to_ocl_type(params.input_type)));
    jit_constants.add(make_jit_constant("INPUT1_TYPE", to_ocl_type(ov::element::i32)));
    jit_constants.add(make_jit_constant("INPUT2_TYPE", to_ocl_type(ov::element::i32)));
    jit_constants.add(make_jit_constant("OUTPUT_TYPE", to_ocl_type(params.output_type)));
    
    auto kernel_params_jit = AdaptiveRKVDiversityKernelSelector::get_jit_constants(params);
    jit_constants.add(kernel_params_jit);
    
    // Execute 6-stage pipeline
    std::vector<AdaptiveRKVDiversityKernelSelector::KernelType> kernel_types = {
        AdaptiveRKVDiversityKernelSelector::KernelType::NORMALIZE_KEYS,
        AdaptiveRKVDiversityKernelSelector::KernelType::COMPUTE_SIMILARITY,
        AdaptiveRKVDiversityKernelSelector::KernelType::SLICE_AND_FILL_DIAGONAL,
        AdaptiveRKVDiversityKernelSelector::KernelType::THRESHOLD_BY_MEAN,
        AdaptiveRKVDiversityKernelSelector::KernelType::AGGREGATE_HEADS,
        AdaptiveRKVDiversityKernelSelector::KernelType::BLOCK_SUM_DIVERSITY
    };
    
    // Prepare scalar parameters (shared across all kernels)
    scalars_desc scalars;
    
    scalar_desc num_kv_heads_desc;
    num_kv_heads_desc.t = scalar_desc::Types::INT32;
    num_kv_heads_desc.v.s32 = params.num_kv_heads;
    scalars.push_back(num_kv_heads_desc);
    
    scalar_desc head_size_desc;
    head_size_desc.t = scalar_desc::Types::INT32;
    head_size_desc.v.s32 = params.head_size;
    scalars.push_back(head_size_desc);
    
    scalar_desc block_size_desc;
    block_size_desc.t = scalar_desc::Types::INT32;
    block_size_desc.v.s32 = params.block_size;
    scalars.push_back(block_size_desc);
    
    scalar_desc sequence_idx_desc;
    sequence_idx_desc.t = scalar_desc::Types::INT32;
    sequence_idx_desc.v.s32 = params.sequence_idx;
    scalars.push_back(sequence_idx_desc);
    
    for (auto kernel_type : kernel_types) {
        auto kernel_name = AdaptiveRKVDiversityKernelSelector::get_kernel_name(kernel_type);
        
        // Build kernel source with JIT constants
        auto kernel_source = std::make_shared<kernel_string>();
        kernel_source->entry_point = kernel_name;
        kernel_source->batch_compilation = false;
        
        // Build code by replacing macros with JIT constants
        CodeBuilder code;
        for (const auto& jit_constant : jit_constants) {
            code.value_macro(jit_constant.name, jit_constant.value);
        }
        code.add_line(std::string(SourcesDB::get_kernel_template("adaptive_rkv_diversity")));
        for (const auto& jit_constant : jit_constants) {
            code.undef_macro(jit_constant.name);
        }
        kernel_source->str = code.str();
        
        // Compile kernel
        auto compiled_kernels = kernels_cache.compile(impl_params, {kernel_source}, false);
        
        if (!compiled_kernels.empty() && !compiled_kernels.begin()->second.empty()) {
            auto kernel = compiled_kernels.begin()->second[0].first;
            
            // Prepare arguments
            auto args = AdaptiveRKVDiversityImpl::get_arguments(kernel_type, impl_params, buffers);
            if (kernel_type == AdaptiveRKVDiversityKernelSelector::KernelType::BLOCK_SUM_DIVERSITY) {
                args.outputs = {output};
            }
            
            // Add scalar parameters
            args.scalars = &scalars;
            
            // Build kernel arguments descriptor manually
            kernel_arguments_desc kernel_params_desc;
            kernel_params_desc.workGroups.global = AdaptiveRKVDiversityKernelSelector::get_global_work_size(kernel_type, params);
            kernel_params_desc.workGroups.local = AdaptiveRKVDiversityKernelSelector::get_local_work_size(kernel_type);
            
            // Build arguments list based on kernel signature
            // Inputs, outputs, scalars in order
            for (size_t i = 0; i < args.inputs.size(); i++) {
                kernel_params_desc.arguments.push_back({ArgumentDescriptor::Types::INPUT, static_cast<uint32_t>(i)});
            }
            for (size_t i = 0; i < args.outputs.size(); i++) {
                kernel_params_desc.arguments.push_back({ArgumentDescriptor::Types::OUTPUT, static_cast<uint32_t>(i)});
            }
            for (size_t i = 0; i < scalars.size(); i++) {
                kernel_params_desc.arguments.push_back({ArgumentDescriptor::Types::SCALAR, static_cast<uint32_t>(i)});
            }
            
            // Execute kernel
            stream.set_arguments(*kernel, kernel_params_desc, args);
            auto event = stream.enqueue_kernel(*kernel, kernel_params_desc, args, stage_events, false);
            stage_events.clear();
            stage_events.push_back(event);
        }
    }
    
    return stage_events.empty() ? nullptr : stage_events.back();
}

event::ptr PagedAttentionAdaptiveRKVIntegration::execute_diversity_fused(
    const kernel_impl_params& impl_params,
    const AdaptiveRKVDiversityKernelSelector::KernelParams& params,
    memory::ptr output) {
    
    auto& stream = *impl_params.strm;
    auto& kernels_cache = const_cast<program*>(impl_params.prog)->get_kernels_cache();
    
    auto kernel_type = AdaptiveRKVDiversityKernelSelector::KernelType::COMPUTE_DIVERSITY_FUSED;
    auto kernel_name = AdaptiveRKVDiversityKernelSelector::get_kernel_name(kernel_type);
    
    // Get JIT constants for kernel compilation
    JitConstants jit_constants;
    jit_constants.add(make_jit_constant("INPUT0_TYPE", to_ocl_type(params.input_type)));
    jit_constants.add(make_jit_constant("INPUT1_TYPE", to_ocl_type(ov::element::i32)));
    jit_constants.add(make_jit_constant("INPUT2_TYPE", to_ocl_type(ov::element::i32)));
    jit_constants.add(make_jit_constant("OUTPUT_TYPE", to_ocl_type(params.output_type)));
    
    auto kernel_params_jit = AdaptiveRKVDiversityKernelSelector::get_jit_constants(params);
    for (const auto& constant : kernel_params_jit) {
        jit_constants.add(constant);
    }
    
    // Build kernel source with JIT constants
    auto kernel_source = std::make_shared<kernel_string>();
    kernel_source->entry_point = kernel_name;
    kernel_source->batch_compilation = false;
    
    // Build code by replacing macros with JIT constants
    CodeBuilder code;
    for (const auto& jit_constant : jit_constants) {
        code.value_macro(jit_constant.name, jit_constant.value);
    }
    code.add_line(std::string(SourcesDB::get_kernel_template("adaptive_rkv_diversity")));
    for (const auto& jit_constant : jit_constants) {
        code.undef_macro(jit_constant.name);
    }
    kernel_source->str = code.str();
    
    // Compile kernel
    auto compiled_kernels = kernels_cache.compile(impl_params, {kernel_source}, false);
    
    if (compiled_kernels.empty() || compiled_kernels.begin()->second.empty()) {
        return nullptr;
    }
    
    auto kernel = compiled_kernels.begin()->second[0].first;
    
    // Build kernel arguments
    kernel_arguments_data args;
    args.inputs = {
        impl_params.memory_deps.at(PagedAttentionInputIdx::KEY_CACHE),
        impl_params.memory_deps.at(PagedAttentionInputIdx::BLOCK_INDICES),
        impl_params.memory_deps.at(PagedAttentionInputIdx::BLOCK_INDICES_BEGINS)
    };
    args.outputs = {output};
    
    // Add scalar parameters
    scalars_desc scalars;
    
    scalar_desc num_kv_heads_desc;
    num_kv_heads_desc.t = scalar_desc::Types::INT32;
    num_kv_heads_desc.v.s32 = params.num_kv_heads;
    scalars.push_back(num_kv_heads_desc);
    
    scalar_desc head_size_desc;
    head_size_desc.t = scalar_desc::Types::INT32;
    head_size_desc.v.s32 = params.head_size;
    scalars.push_back(head_size_desc);
    
    scalar_desc block_size_desc;
    block_size_desc.t = scalar_desc::Types::INT32;
    block_size_desc.v.s32 = params.block_size;
    scalars.push_back(block_size_desc);
    
    scalar_desc start_size_desc;
    start_size_desc.t = scalar_desc::Types::INT32;
    start_size_desc.v.s32 = params.start_size;
    scalars.push_back(start_size_desc);
    
    scalar_desc eviction_size_desc;
    eviction_size_desc.t = scalar_desc::Types::INT32;
    eviction_size_desc.v.s32 = params.eviction_size;
    scalars.push_back(eviction_size_desc);
    
    scalar_desc sequence_idx_desc;
    sequence_idx_desc.t = scalar_desc::Types::INT32;
    sequence_idx_desc.v.s32 = params.sequence_idx;
    scalars.push_back(sequence_idx_desc);
    
    args.scalars = &scalars;
    
    // Set up local memory
    local_memory_args_desc local_mem_args;
    size_t local_mem_size = params.num_kv_heads * params.block_size * params.head_size * sizeof(float);
    local_mem_args.push_back(local_mem_size);
    args.local_memory_args = &local_mem_args;
    
    // Build kernel arguments descriptor manually
    kernel_arguments_desc kernel_params_desc;
    kernel_params_desc.workGroups.global = AdaptiveRKVDiversityKernelSelector::get_global_work_size(kernel_type, params);
    kernel_params_desc.workGroups.local = AdaptiveRKVDiversityKernelSelector::get_local_work_size(kernel_type);
    
    // Build arguments list based on kernel signature
    // adaptive_rkv_compute_diversity_fused(key_cache, block_indices, block_indices_begins, block_diversity, local_mem, scalars...)
    for (size_t i = 0; i < args.inputs.size(); i++) {
        kernel_params_desc.arguments.push_back({ArgumentDescriptor::Types::INPUT, static_cast<uint32_t>(i)});
    }
    for (size_t i = 0; i < args.outputs.size(); i++) {
        kernel_params_desc.arguments.push_back({ArgumentDescriptor::Types::OUTPUT, static_cast<uint32_t>(i)});
    }
    // Local memory argument
    kernel_params_desc.arguments.push_back({ArgumentDescriptor::Types::LOCAL_MEMORY_SIZE, 0});
    // Scalar arguments
    for (size_t i = 0; i < scalars.size(); i++) {
        kernel_params_desc.arguments.push_back({ArgumentDescriptor::Types::SCALAR, static_cast<uint32_t>(i)});
    }
    
    // Execute fused kernel
    stream.set_arguments(*kernel, kernel_params_desc, args);
    auto event = stream.enqueue_kernel(*kernel, kernel_params_desc, args, {}, false);
    
    return event;
}

} // namespace ov::intel_gpu::ocl
