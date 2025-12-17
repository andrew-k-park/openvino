# Adaptive R-KV Implementation Summary for Intel GPU Plugin

## Overview
This document summarizes the implementation of Adaptive R-KV (Relevant Key-Value) diversity calculation for Paged Attention in the Intel GPU OpenCL backend, based on the reference implementation in `adaptive_rkv_diversity.hpp`.

## Reference Algorithm
Source: https://arxiv.org/pdf/2505.24133v3

The Adaptive R-KV mechanism calculates token diversity in the eviction area to identify which KV-cache blocks can be safely evicted during attention computation.

### Key Steps:
1. **L2 Normalization**: Normalize key vectors in the eviction area
2. **Cosine Similarity**: Compute pairwise cosine similarity matrix
3. **Diagonal Zeroing**: Set diagonal elements to zero (self-similarity)
4. **Mean Filtering**: Zero out values below row-wise mean
5. **Head Aggregation**: Average across attention heads
6. **Block Aggregation**: Sum diversity per block (negated cosine similarity)

## Implementation Files

### 1. New OpenCL Kernel: `adaptive_rkv_diversity.cl`
**Location**: `src/plugins/intel_gpu/src/graph/impls/ocl_v2/adaptive_rkv_diversity.cl`

**Purpose**: Standalone kernel for computing block diversity values

**Key Features**:
- L2 normalization of key vectors
- Cosine similarity matrix computation using SLM (Shared Local Memory)
- Diagonal filling and mean-based filtering
- Per-block diversity aggregation

**Inputs**:
- `key_cache`: KV-cache blocks `[num_blocks, num_heads, head_size, block_size]`
- `past_lens`: Sequence lengths `[batch_size]`
- `block_indices`: Block mapping `[total_blocks]`
- `adaptive_rkv_start_size`: Start of eviction area (scalar)
- `adaptive_rkv_evictable_sizes`: Evictable size per sequence `[batch_size]`
- `diversity_block_set_indices`: Block indices for diversity calculation
- `diversity_block_set_begins`: Offsets into diversity block set

**Outputs**:
- `diversity_output`: Per-block diversity values `[num_eviction_blocks, eviction_size]`
- `normalized_keys_buffer`: Temporary buffer for normalized keys

**Limitations of Current Implementation**:
- Simplified inline version in main kernel (full kernel can be invoked separately)
- MAX_EVICTABLE_SIZE compile-time constant needed for SLM allocation
- Atomic operations used for cross-head aggregation (may impact performance)

### 2. Modified: `paged_attention_opt.cl`
**Location**: `src/plugins/intel_gpu/src/graph/impls/ocl_v2/paged_attention_opt.cl`

**Changes**:
- Added `HAS_ADAPTIVE_RKV` preprocessor conditional
- Extended kernel signature with adaptive R-KV parameters:
  ```c
  #if HAS_ADAPTIVE_RKV
      , const __global int* adaptive_rkv_start_size
      , const __global int* adaptive_rkv_evictable_sizes
      , const __global int* diversity_block_set_indices
      , const __global int* diversity_block_set_begins
      , __global OUTPUT_TYPE* diversity_output
  #endif
  ```
- Added diversity calculation trigger at end of SDPA_STAGE_0

### 3. Modified: `paged_attention_opt.cpp`
**Location**: `src/plugins/intel_gpu/src/graph/impls/ocl_v2/sdpa/paged_attention_opt.cpp`

**Changes**:
- Added JIT constant `HAS_ADAPTIVE_RKV` when `desc->has_adaptive_rkv` is true
- Extended `get_arguments_desc()` to include adaptive R-KV inputs:
  - `ADAPTIVE_RKV_START_SIZE`
  - `ADAPTIVE_RKV_EVICTABLE_SIZES`
  - `ADAPTIVE_RKV_DIVERSITY_BLOCK_SET_INDICES`
  - `ADAPTIVE_RKV_DIVERSITY_BLOCK_SET_BEGINS`
  - Output port 2 for diversity results (when applicable)

### 4. Modified: `sdpa_gen_opt.cpp`
**Location**: `src/plugins/intel_gpu/src/graph/impls/ocl_v2/sdpa/sdpa_gen_opt.cpp`

**Changes**:
- Added JIT constants for paged attention with adaptive R-KV:
  ```cpp
  if (desc->has_adaptive_rkv) {
      jit.make("HAS_ADAPTIVE_RKV", 1);
      jit.make("PAGED_ATTENTION_BLOCK_SIZE", paged_attention::block_size);
  }
  ```

### 5. Modified: `sdpa_opt.cl`
**Location**: `src/plugins/intel_gpu/src/graph/impls/ocl_v2/sdpa_opt.cl`

**Changes**:
- Extended kernel signature for paged attention with adaptive R-KV parameters:
  ```c
  #if IS_PAGED_ATTENTION && HAS_ADAPTIVE_RKV
      , const __global int* adaptive_rkv_evictable_start_size
      , const __global int* adaptive_rkv_evictable_sizes
      , const __global int* adaptive_rkv_evictable_indices
      , const __global int* adaptive_rkv_evictable_begins
      , __global OUTPUT_TYPE* adaptive_rkv_diversity_output
  #endif
  ```
- Added placeholder for diversity calculation integration after softmax computation
- Note: Full diversity calculation should be performed via separate kernel dispatch

**Purpose**: Provides integration point for diversity calculation in general SDPA kernels when used with paged attention

### 6. Already Modified: `paged_attention.cpp`
**Location**: `src/plugins/intel_gpu/src/graph/paged_attention.cpp`

**Existing Changes** (from previous implementation):
- `calc_output_layouts()` already handles 3 outputs when `has_adaptive_rkv` is true:
  1. Main attention output
  2. Softmax scores (if `has_scores_output`)
  3. **Diversity values** `[total_diversity_elements]`

**Diversity Output Shape Calculation**:
```cpp
if (desc->has_adaptive_rkv) {
    size_t num_elements_in_output = 0;
    for (size_t i = 0; i < evictable_sizes.size(); i++) {
        size_t evictable_size = evictable_sizes[i];
        // [num_eviction_blocks, eviction_size] flattened
        num_elements_in_output += evictable_size * evictable_size / block_size;
    }
    output_layouts.push_back(layout{PartialShape{num_elements_in_output}, dtype, format::bfyx});
}
```

### 5. Modified: `sdpa_opt.cl`
**Location**: `src/plugins/intel_gpu/src/graph/impls/ocl_v2/sdpa_opt.cl`

**Changes**:
- Extended kernel signature for paged attention with adaptive R-KV parameters:
  ```c
  #if IS_PAGED_ATTENTION && HAS_ADAPTIVE_RKV
      , const __global int* adaptive_rkv_evictable_start_size
      , const __global int* adaptive_rkv_evictable_sizes
      , const __global int* adaptive_rkv_evictable_indices
      , const __global int* adaptive_rkv_evictable_begins
      , __global OUTPUT_TYPE* adaptive_rkv_diversity_output
  #endif
  ```
- Added placeholder marker for diversity calculation integration after softmax computation
- Note: Full diversity calculation should be performed via separate kernel dispatch using `adaptive_rkv_diversity.cl`

**Purpose**: Provides integration point for diversity calculation in general SDPA kernels when used with paged attention mode

### 6. Already Modified: `paged_attention.hpp`
**Location**: `src/plugins/intel_gpu/include/intel_gpu/primitives/paged_attention.hpp`

**Existing Changes**:
- Input enum extended with:
  - `ADAPTIVE_RKV_START_SIZE = 21`
  - `ADAPTIVE_RKV_EVICTABLE_SIZES = 22`
  - `ADAPTIVE_RKV_DIVERSITY_BLOCK_SET_INDICES = 23`
  - `ADAPTIVE_RKV_DIVERSITY_BLOCK_SET_BEGINS = 24`
- Primitive descriptor includes `bool has_adaptive_rkv = false`
- Serialization/deserialization support added

### 7. Already Modified: `paged_attention.cpp` (plugin ops)
**Location**: `src/plugins/intel_gpu/src/plugin/ops/paged_attention.cpp`

**Existing Changes**:
- Detection of adaptive R-KV from dynamic inputs:
  ```cpp
  auto adaptive_rkv_evictable_sizes_input = ov::as_type_ptr<ov::op::v0::Parameter>(
      op->get_input_node_shared_ptr(adaptive_rkv_evictable_sizes_idx));
  if (adaptive_rkv_evictable_sizes_input && 
      adaptive_rkv_evictable_sizes_input->get_output_partial_shape(0).is_dynamic()) {
      prim.has_adaptive_rkv = true;
  }
  ```

## Summary of All Modified Files

1. **adaptive_rkv_diversity.cl** (NEW) - Standalone diversity calculation kernel
2. **paged_attention_opt.cl** (MODIFIED) - Paged attention kernel with adaptive R-KV support
3. **paged_attention_opt.cpp** (MODIFIED) - JIT constants and arguments for paged attention
4. **sdpa_gen_opt.cpp** (MODIFIED) - JIT constants for SDPA with paged attention
5. **sdpa_opt.cl** (MODIFIED) - General SDPA kernel with adaptive R-KV parameters
6. **paged_attention.cpp** (graph) (ALREADY MODIFIED) - Output layout calculation
7. **paged_attention.hpp** (ALREADY MODIFIED) - Input enum and descriptor
8. **paged_attention.cpp** (plugin ops) (ALREADY MODIFIED) - Adaptive R-KV detection

## Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│ Input: key_cache [num_blocks, heads, head_size, block_size]│
│        evictable_sizes [batch_size]                         │
│        start_size (scalar)                                   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │ Step 1: L2 Normalization   │
        │ (per token in eviction area)│
        └────────────┬───────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │ Step 2: Cosine Similarity  │
        │ Matrix [evict_size²]       │
        └────────────┬───────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │ Step 3: Diagonal = 0       │
        └────────────┬───────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │ Step 4: Filter by Mean     │
        │ (zero if < row mean)       │
        └────────────┬───────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │ Step 5: Head Aggregation   │
        │ (mean across heads)        │
        └────────────┬───────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │ Step 6: Block Aggregation  │
        │ Sum per block, negate      │
        └────────────┬───────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ Output: diversity [num_blocks, evict_size]                  │
│         Higher values = more diverse = keep                  │
└─────────────────────────────────────────────────────────────┘
```

## Compilation Flags

### Required Defines:
- `HAS_ADAPTIVE_RKV=1`: Enables adaptive R-KV code paths
- `PAGED_ATTENTION_BLOCK_SIZE=16`: Block size constant
- `NUM_HEADS`: Number of attention heads
- `HEAD_SIZE`: Size of each attention head
- `MAX_EVICTABLE_SIZE`: Maximum evictable region size (for SLM allocation)

### Example JIT Constants:
```cpp
jit.make("HAS_ADAPTIVE_RKV", 1);
jit.make("PAGED_ATTENTION_BLOCK_SIZE", 16);
jit.make("NUM_HEADS", 32);
jit.make("HEAD_SIZE", 128);
jit.make("MAX_EVICTABLE_SIZE", 512);  // Adjust based on requirements
```

## Integration Points

### For openvino.genai:
1. The diversity output (output port 2) provides raw per-block, per-token diversity values
2. Shape: `[num_eviction_blocks, eviction_size]` per sequence, concatenated
3. Final filtering (mean reduction along rank-1) should be done at genai level after knowing which blocks to retain
4. Use diversity scores to rank and evict low-diversity blocks from KV-cache

### Runtime Flow:
```
PA Kernel Execute → Compute Attention → Calculate Diversity (if enabled)
                                              ↓
                                    diversity_output buffer
                                              ↓
                         openvino.genai reads diversity scores
                                              ↓
                              Filter retained blocks
                                              ↓
                        Mean-reduce diversity per block
                                              ↓
                          Evict low-scoring blocks
```

## Performance Considerations

### Optimization Opportunities:
1. **Separate Kernel Invocation**: Launch diversity calculation as separate kernel to avoid blocking main PA kernel
2. **SLM Usage**: Current implementation uses significant SLM for cosine similarity matrix
3. **Tiling**: For large eviction areas, implement tiled computation
4. **Atomic Operations**: Replace atomic adds with tree reduction for better performance
5. **Precision**: Consider FP16 for intermediate calculations (currently using SOFTMAX_ACCUMULATOR_TYPE = FP32)

### Memory Requirements:
- Normalized keys buffer: `batch_size × num_heads × evictable_size × head_size × sizeof(float)`
- Cosine similarity SLM: `evictable_size² × sizeof(float)` per work-group
- Output diversity: `sum(evictable_size × evictable_size / block_size) × sizeof(float)`

## Testing Recommendations

1. **Unit Tests**: Validate diversity calculation matches reference implementation
2. **Integration Tests**: Verify end-to-end flow with openvino.genai
3. **Performance Tests**: Measure overhead of diversity calculation
4. **Edge Cases**:
   - `evictable_size = 0` (no diversity calculation)
   - `seq_len < start_size + evictable_size` (sequence too short)
   - Dynamic batch sizes and sequence lengths
   - Multiple sequences with different evictable sizes

## Known Limitations

1. **Inline Implementation**: Current SDPA_STAGE_0 implementation is simplified
2. **Fixed MAX_EVICTABLE_SIZE**: Requires compile-time constant for SLM allocation
3. **No Dynamic Tiling**: Large eviction areas may exceed available SLM
4. **Synchronization**: Cross-head aggregation uses atomics (potential bottleneck)
5. **Rank-1 Aggregation**: Final mean-reduction deferred to openvino.genai level

## Future Enhancements

1. Implement full standalone diversity kernel with optimized memory access patterns
2. Add support for dynamic tiling based on available SLM
3. Optimize head aggregation using tree reduction instead of atomics
4. Add FP16 compute path for better performance on supported hardware
5. Implement block-wise processing to handle arbitrary eviction sizes
6. Add kernel fusion opportunities with main attention computation

## References

- Paper: https://arxiv.org/pdf/2505.24133v3
- Reference Implementation: `/home/andrew/work/openvino.dev/src/core/reference/include/openvino/reference/adaptive_rkv_diversity.hpp`
- GPU Plugin Docs: Intel GPU Plugin Developer Guide
