# SDPA_OPT Adaptive R-KV Implementation Changes

## Overview
This document describes the changes made to `sdpa_opt.cl` and `sdpa_gen_opt.cpp` to support Adaptive R-KV diversity calculation in general SDPA (Scaled Dot Product Attention) kernels when used with paged attention mode.

## Modified Files

### 1. sdpa_opt.cl
**Location**: `src/plugins/intel_gpu/src/graph/impls/ocl_v2/sdpa_opt.cl`

#### Changes Made:
Extended the kernel signature to include adaptive R-KV parameters when both `IS_PAGED_ATTENTION` and `HAS_ADAPTIVE_RKV` are enabled:

```c
#if IS_PAGED_ATTENTION && HAS_ADAPTIVE_RKV
    , const __global int* adaptive_rkv_evictable_start_size
    , const __global int* adaptive_rkv_evictable_sizes
    , const __global int* adaptive_rkv_evictable_indices
    , const __global int* adaptive_rkv_evictable_begins
    , __global OUTPUT_TYPE* adaptive_rkv_diversity_output
#endif
```

#### Parameter Descriptions:

- **adaptive_rkv_evictable_start_size**: `[batch_size * 2]`
  - Contains pairs of (eviction_start, eviction_size) for each sequence in the batch

- **adaptive_rkv_evictable_sizes**: `[batch_size]`
  - Number of evictable tokens per sequence

- **adaptive_rkv_evictable_indices**: `[total_evictable_blocks]`
  - Block indices in the eviction area across all sequences

- **adaptive_rkv_evictable_begins**: `[batch_size + 1]`
  - Offset array indicating where each sequence's evictable blocks start in the indices array

- **adaptive_rkv_diversity_output**: `[total_diversity_elements]`
  - Output buffer for computed diversity values per block

#### Integration Point:
Added a placeholder marker after softmax computation where diversity calculation should be triggered:

```c
#if IS_PAGED_ATTENTION && HAS_ADAPTIVE_RKV
        // Adaptive R-KV diversity calculation
        // Note: This is a placeholder for integration with adaptive_rkv_diversity.cl kernel
        // The actual diversity computation should be performed via a separate kernel dispatch
        if (sgid == 0 && sglid == 0) {
            const uint seq_correspondence_idx = gws_seq_indexes_correspondence[target_seq_dim];
            const uint evictable_start = adaptive_rkv_evictable_start_size[seq_correspondence_idx * 2];
            const uint evictable_size = adaptive_rkv_evictable_start_size[seq_correspondence_idx * 2 + 1];
            
            if (evictable_size > 0) {
                // Placeholder: actual diversity calculation happens in adaptive_rkv_diversity.cl
                adaptive_rkv_diversity_output[seq_correspondence_idx] = OUTPUT_VAL_ZERO;
            }
        }
#endif
```

### 2. sdpa_gen_opt.cpp
**Location**: `src/plugins/intel_gpu/src/graph/impls/ocl_v2/sdpa/sdpa_gen_opt.cpp`

#### Changes Made:
Added JIT (Just-In-Time compilation) constants for adaptive R-KV support in the `get_jit_constants_base()` function:

```cpp
if (is_paged_attention) {
    auto desc = params.typed_desc<paged_attention>();
    // ... existing paged attention setup ...
    
    if (desc->has_adaptive_rkv) {
        jit.make("HAS_ADAPTIVE_RKV", 1);
        jit.make("PAGED_ATTENTION_BLOCK_SIZE", paged_attention::block_size);
    }
}
```

#### Purpose:
These JIT constants enable conditional compilation of adaptive R-KV code paths in the OpenCL kernels:
- `HAS_ADAPTIVE_RKV`: Enables the adaptive R-KV parameter extensions
- `PAGED_ATTENTION_BLOCK_SIZE`: Provides the block size constant (typically 16) for diversity calculations

## Compilation Flow

### When Adaptive R-KV is Enabled:
1. `paged_attention` primitive created with `has_adaptive_rkv = true`
2. `sdpa_gen_opt.cpp` adds JIT constants: `HAS_ADAPTIVE_RKV=1`, `PAGED_ATTENTION_BLOCK_SIZE=16`
3. OpenCL kernel `sdpa_opt.cl` compiled with extended parameters
4. At runtime, kernel receives 5 additional input buffers for adaptive R-KV

### When Adaptive R-KV is Disabled:
1. JIT constants not set
2. Kernel compiled without adaptive R-KV parameters
3. Standard SDPA execution without diversity calculation

## Integration with Separate Diversity Kernel

The current implementation in `sdpa_opt.cl` provides a placeholder for diversity calculation. The actual computation should be performed by the standalone `adaptive_rkv_diversity.cl` kernel through a separate dispatch:

### Recommended Execution Flow:
1. **SDPA Kernel**: Computes attention scores and stores in SLM/memory
2. **Diversity Kernel** (`adaptive_rkv_diversity.cl`): 
   - Reads attention scores (or key vectors)
   - Computes L2 normalization
   - Calculates cosine similarity matrix
   - Performs filtering and block aggregation
   - Outputs diversity values
3. **Eviction Logic** (at genai level): Uses diversity output to select blocks for eviction

## Argument Descriptor Updates

The argument descriptors are handled by `paged_attention_opt.cpp` for paged attention kernels. For general SDPA kernels (`sdpa_opt.cl`), the arguments are passed through the paged attention infrastructure when `IS_PAGED_ATTENTION` is enabled.

**Note**: `sdpa_gen_opt.cpp::get_arguments_desc_impl()` handles arguments for non-paged SDPA. Paged attention arguments (including adaptive R-KV) are managed separately in `paged_attention_opt.cpp`.

## Testing Recommendations

1. **Compilation Test**: Verify kernels compile with `HAS_ADAPTIVE_RKV=1`
2. **Parameter Passing**: Ensure all 5 adaptive R-KV buffers are correctly bound
3. **Diversity Kernel Integration**: Test separate dispatch of `adaptive_rkv_diversity.cl`
4. **Performance**: Measure impact of additional parameters on register pressure
5. **Correctness**: Validate diversity values match reference implementation

## Future Work

1. **Full Integration**: Implement automatic dispatch of diversity kernel after SDPA
2. **Optimization**: Fuse diversity calculation into main SDPA kernel if beneficial
3. **Dynamic Tiling**: Support larger eviction areas beyond SLM limits
4. **FP16 Support**: Add half-precision option for diversity calculations
5. **Multi-Head Optimization**: Use tree reduction instead of atomics for head aggregation

## Related Files

- `adaptive_rkv_diversity.cl`: Standalone diversity calculation kernel
- `paged_attention_opt.cl`: Paged attention kernel with adaptive R-KV support
- `paged_attention_opt.cpp`: Main implementation for paged attention argument handling
- `ADAPTIVE_RKV_IMPLEMENTATION_SUMMARY.md`: Complete implementation overview
