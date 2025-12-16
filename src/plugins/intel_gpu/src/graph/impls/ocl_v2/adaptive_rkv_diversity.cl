// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"
#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"
#include "include/batch_headers/sub_group_shuffle.cl"

// ===============================================================================
// Adaptive R-KV Diversity Calculation Kernels
// Reference: https://arxiv.org/pdf/2505.24133v3
// ===============================================================================

#define DIVERSITY_SUBGROUP_SIZE 16
#define EPSILON 1e-12f

// ===============================================================================
// Step 2: Compute Similarity Matrix
// ===============================================================================

// Kernel 1: Normalize keys along head_size dimension (L2 normalization)
// Input:  key_cache [num_blocks, num_kv_heads, head_size, block_size]
// Output: normalized_keys [num_kv_heads, num_key_tokens, head_size]
REQD_SUB_GROUP_SIZE(DIVERSITY_SUBGROUP_SIZE)
__attribute__((reqd_work_group_size(DIVERSITY_SUBGROUP_SIZE, 1, 1)))
KERNEL(adaptive_rkv_normalize_keys)(
    const __global INPUT0_TYPE* key_cache,
    const __global INPUT1_TYPE* block_indices,
    const __global INPUT2_TYPE* block_indices_begins,
    __global OUTPUT_TYPE* normalized_keys,
    const int num_kv_heads,
    const int head_size,
    const int block_size,
    const int sequence_idx
) {
    const uint token_idx = get_global_id(0);
    const uint head_idx = get_global_id(1);
    
    if (head_idx >= num_kv_heads)
        return;
    
    const int block_start_idx = block_indices_begins[sequence_idx];
    const int block_end_idx = block_indices_begins[sequence_idx + 1];
    const int num_blocks = block_end_idx - block_start_idx;
    const int num_tokens = num_blocks * block_size;
    
    if (token_idx >= num_tokens)
        return;
    
    // Calculate which block and position within block
    const int block_local_idx = token_idx / block_size;
    const int pos_in_block = token_idx % block_size;
    const int block_global_idx = block_indices[block_start_idx + block_local_idx];
    
    // Compute L2 norm
    float sum_squares = 0.0f;
    for (int i = 0; i < head_size; i++) {
        const uint key_offset = block_global_idx * num_kv_heads * head_size * block_size +
                               head_idx * head_size * block_size +
                               i * block_size +
                               pos_in_block;
        float val = (float)key_cache[key_offset];
        sum_squares += val * val;
    }
    
    float norm = native_sqrt(sum_squares + EPSILON);
    
    // Normalize and store
    for (int i = 0; i < head_size; i++) {
        const uint key_offset = block_global_idx * num_kv_heads * head_size * block_size +
                               head_idx * head_size * block_size +
                               i * block_size +
                               pos_in_block;
        const uint out_offset = head_idx * num_tokens * head_size +
                               token_idx * head_size +
                               i;
        normalized_keys[out_offset] = (OUTPUT_TYPE)(key_cache[key_offset] / norm);
    }
}

// Kernel 2: Compute cosine similarity matrix (matmul of normalized keys)
// Input:  normalized_keys [num_kv_heads, num_key_tokens, head_size]
// Output: similarity_matrix [num_kv_heads, num_key_tokens, num_key_tokens]
REQD_SUB_GROUP_SIZE(DIVERSITY_SUBGROUP_SIZE)
__attribute__((reqd_work_group_size(DIVERSITY_SUBGROUP_SIZE, 1, 1)))
KERNEL(adaptive_rkv_compute_similarity)(
    const __global INPUT0_TYPE* normalized_keys,
    __global OUTPUT_TYPE* similarity_matrix,
    const int num_kv_heads,
    const int num_tokens,
    const int head_size
) {
    const uint row_idx = get_global_id(0);  // token_i
    const uint col_idx = get_global_id(1);  // token_j
    const uint head_idx = get_global_id(2);
    
    if (row_idx >= num_tokens || col_idx >= num_tokens || head_idx >= num_kv_heads)
        return;
    
    // Compute dot product: keys[i] · keys[j]
    float dot_product = 0.0f;
    for (int k = 0; k < head_size; k++) {
        const uint offset_i = head_idx * num_tokens * head_size + row_idx * head_size + k;
        const uint offset_j = head_idx * num_tokens * head_size + col_idx * head_size + k;
        dot_product += (float)normalized_keys[offset_i] * (float)normalized_keys[offset_j];
    }
    
    const uint out_offset = head_idx * num_tokens * num_tokens + 
                           row_idx * num_tokens + 
                           col_idx;
    similarity_matrix[out_offset] = (OUTPUT_TYPE)dot_product;
}

// Kernel 3: Slice evictable region and fill diagonal with zeros
// sim = sim[:, start_size:start_size+eviction_size, start_size:start_size+eviction_size]
// sim.fill_diagonal_(0)
REQD_SUB_GROUP_SIZE(DIVERSITY_SUBGROUP_SIZE)
__attribute__((reqd_work_group_size(DIVERSITY_SUBGROUP_SIZE, 1, 1)))
KERNEL(adaptive_rkv_slice_and_fill_diagonal)(
    const __global INPUT0_TYPE* similarity_matrix,
    __global OUTPUT_TYPE* evictable_sim,
    const int num_kv_heads,
    const int num_tokens,
    const int start_size,
    const int eviction_size
) {
    const uint row_idx = get_global_id(0);
    const uint col_idx = get_global_id(1);
    const uint head_idx = get_global_id(2);
    
    if (row_idx >= eviction_size || col_idx >= eviction_size || head_idx >= num_kv_heads)
        return;
    
    const uint src_offset = head_idx * num_tokens * num_tokens +
                           (start_size + row_idx) * num_tokens +
                           (start_size + col_idx);
    const uint dst_offset = head_idx * eviction_size * eviction_size +
                           row_idx * eviction_size +
                           col_idx;
    
    // Fill diagonal with 0, copy others
    if (row_idx == col_idx) {
        evictable_sim[dst_offset] = (OUTPUT_TYPE)0.0f;
    } else {
        evictable_sim[dst_offset] = similarity_matrix[src_offset];
    }
}

// Kernel 4: Apply mean threshold filter
// sim = torch.where(sim >= sim.mean(dim=-1, keepdim=True), sim, 0.)
REQD_SUB_GROUP_SIZE(DIVERSITY_SUBGROUP_SIZE)
__attribute__((reqd_work_group_size(DIVERSITY_SUBGROUP_SIZE, 1, 1)))
KERNEL(adaptive_rkv_threshold_by_mean)(
    __global INPUT0_TYPE* evictable_sim,  // in-place operation
    const int num_kv_heads,
    const int eviction_size
) {
    const uint row_idx = get_global_id(0);
    const uint head_idx = get_global_id(1);
    
    if (row_idx >= eviction_size || head_idx >= num_kv_heads)
        return;
    
    // Calculate mean for this row
    float row_mean = 0.0f;
    const uint row_offset = head_idx * eviction_size * eviction_size + row_idx * eviction_size;
    for (int col_idx = 0; col_idx < eviction_size; col_idx++) {
        row_mean += (float)evictable_sim[row_offset + col_idx];
    }
    row_mean /= (float)eviction_size;
    
    // Apply threshold
    for (int col_idx = 0; col_idx < eviction_size; col_idx++) {
        const uint offset = row_offset + col_idx;
        float val = (float)evictable_sim[offset];
        evictable_sim[offset] = (OUTPUT_TYPE)(val >= row_mean ? val : 0.0f);
    }
}

// Kernel 5: Aggregate across heads (mean over dim=0)
// sim = sim.mean(dim=0)
REQD_SUB_GROUP_SIZE(DIVERSITY_SUBGROUP_SIZE)
__attribute__((reqd_work_group_size(DIVERSITY_SUBGROUP_SIZE, 1, 1)))
KERNEL(adaptive_rkv_aggregate_heads)(
    const __global INPUT0_TYPE* evictable_sim,
    __global OUTPUT_TYPE* aggregated_sim,
    const int num_kv_heads,
    const int eviction_size
) {
    const uint row_idx = get_global_id(0);
    const uint col_idx = get_global_id(1);
    
    if (row_idx >= eviction_size || col_idx >= eviction_size)
        return;
    
    // Aggregate across all heads
    float sum = 0.0f;
    for (int head_idx = 0; head_idx < num_kv_heads; head_idx++) {
        const uint offset = head_idx * eviction_size * eviction_size +
                           row_idx * eviction_size +
                           col_idx;
        sum += (float)evictable_sim[offset];
    }
    
    const uint out_offset = row_idx * eviction_size + col_idx;
    aggregated_sim[out_offset] = (OUTPUT_TYPE)(sum / (float)num_kv_heads);
}

// ===============================================================================
// Step 3: Group Diversity Filtering (Partial Implementation)
// ===============================================================================

// Kernel 6: Compute per-block diversity with row masking support
// diversity = -sim[:, (scores == float("-inf"))].mean(dim=-1)
// This kernel computes block-wise sum (negated) but leaves the final mean reduction
// to be done by the host/genai layer after applying the row mask
//
// Output: block_diversity [num_blocks, eviction_size]
// Note: The calling code must filter columns by mask and apply mean reduction
REQD_SUB_GROUP_SIZE(DIVERSITY_SUBGROUP_SIZE)
__attribute__((reqd_work_group_size(DIVERSITY_SUBGROUP_SIZE, 1, 1)))
KERNEL(adaptive_rkv_block_sum_diversity)(
    const __global INPUT0_TYPE* aggregated_sim,
    __global OUTPUT_TYPE* block_diversity,
    const int eviction_size,
    const int block_size
) {
    const uint block_idx = get_global_id(0);
    const uint col_idx = get_global_id(1);
    
    const int num_blocks = eviction_size / block_size;
    
    if (block_idx >= num_blocks || col_idx >= eviction_size)
        return;
    
    // Sum diversity values for all tokens in this block
    // Negative sign is applied here: diversity = -similarity
    float block_sum = 0.0f;
    for (int token_in_block = 0; token_in_block < block_size; token_in_block++) {
        const uint row_idx = block_idx * block_size + token_in_block;
        const uint src_offset = row_idx * eviction_size + col_idx;
        block_sum -= (float)aggregated_sim[src_offset];  // Note the negative sign
    }
    
    const uint out_offset = block_idx * eviction_size + col_idx;
    block_diversity[out_offset] = (OUTPUT_TYPE)block_sum;
}

// Kernel 7: Apply row mask and compute final diversity per block
// This kernel should be called from genai layer after block set indices are known
// diversity = block_diversity[:, mask].mean(dim=-1)
REQD_SUB_GROUP_SIZE(DIVERSITY_SUBGROUP_SIZE)
__attribute__((reqd_work_group_size(DIVERSITY_SUBGROUP_SIZE, 1, 1)))
KERNEL(adaptive_rkv_apply_mask_and_reduce)(
    const __global INPUT0_TYPE* block_diversity,
    const __global INPUT1_TYPE* row_mask,  // bool mask [eviction_size], true = include
    __global OUTPUT_TYPE* final_diversity,
    const int eviction_size,
    const int block_size
) {
    const uint block_idx = get_global_id(0);
    const int num_blocks = eviction_size / block_size;
    
    if (block_idx >= num_blocks)
        return;
    
    // Count masked elements and compute mean
    float sum = 0.0f;
    int count = 0;
    
    for (int col_idx = 0; col_idx < eviction_size; col_idx++) {
        if (row_mask[col_idx]) {
            const uint offset = block_idx * eviction_size + col_idx;
            sum += (float)block_diversity[offset];
            count++;
        }
    }
    
    final_diversity[block_idx] = (count > 0) ? (OUTPUT_TYPE)(sum / (float)count) : (OUTPUT_TYPE)(-INFINITY);
}

// ===============================================================================
// Integrated Kernel: Combined Step 2 computation
// This is an optimized version that fuses multiple operations
// ===============================================================================

REQD_SUB_GROUP_SIZE(DIVERSITY_SUBGROUP_SIZE)
__attribute__((reqd_work_group_size(DIVERSITY_SUBGROUP_SIZE, 1, 1)))
KERNEL(adaptive_rkv_compute_diversity_fused)(
    const __global INPUT0_TYPE* key_cache,
    const __global INPUT1_TYPE* block_indices,
    const __global INPUT2_TYPE* block_indices_begins,
    __global OUTPUT_TYPE* block_diversity,
    __local float* local_mem,
    const int num_kv_heads,
    const int head_size,
    const int block_size,
    const int start_size,
    const int eviction_size,
    const int sequence_idx
) {
    // This fused kernel combines normalization, matmul, slicing, filtering, and aggregation
    // Work distribution: one work-group per evictable block
    
    const uint sglid = get_sub_group_local_id();
    const uint sgid = get_sub_group_id();
    const uint block_idx = get_group_id(0);  // Which evictable block we're processing
    
    const int block_start_idx = block_indices_begins[sequence_idx];
    const int block_end_idx = block_indices_begins[sequence_idx + 1];
    const int num_blocks = block_end_idx - block_start_idx;
    const int num_tokens = num_blocks * block_size;
    const int evictable_start_token = start_size;
    const int evictable_end_token = start_size + eviction_size;
    const int num_evictable_blocks = eviction_size / block_size;
    
    if (block_idx >= num_evictable_blocks)
        return;
    
    // Local memory layout:
    // [0...head_size*block_size): Normalized keys for this block (all heads)
    // Allocate per-head normalized keys in local memory
    __local float* local_norm_keys = local_mem;
    
    const int global_block_in_evictable = start_size / block_size + block_idx;
    
    // Step 1: Load and normalize keys for this evictable block (all heads, all tokens in block)
    // Each work-item processes multiple elements
    for (int head_idx = 0; head_idx < num_kv_heads; head_idx++) {
        for (int token_in_block = 0; token_in_block < block_size; token_in_block++) {
            const int global_token_idx = evictable_start_token + block_idx * block_size + token_in_block;
            const int block_local_idx = global_token_idx / block_size;
            const int pos_in_block = global_token_idx % block_size;
            const int block_global_idx = block_indices[block_start_idx + block_local_idx];
            
            // Compute L2 norm for this token
            float sum_squares = 0.0f;
            for (int d = sglid; d < head_size; d += DIVERSITY_SUBGROUP_SIZE) {
                const uint key_offset = block_global_idx * num_kv_heads * head_size * block_size +
                                       head_idx * head_size * block_size +
                                       d * block_size +
                                       pos_in_block;
                float val = (float)key_cache[key_offset];
                sum_squares += val * val;
            }
            
            // Reduce sum_squares across subgroup
            sum_squares = sub_group_reduce_add(sum_squares);
            float norm = native_sqrt(sum_squares + EPSILON);
            
            // Normalize and store in local memory
            for (int d = sglid; d < head_size; d += DIVERSITY_SUBGROUP_SIZE) {
                const uint key_offset = block_global_idx * num_kv_heads * head_size * block_size +
                                       head_idx * head_size * block_size +
                                       d * block_size +
                                       pos_in_block;
                const uint local_offset = head_idx * block_size * head_size +
                                         token_in_block * head_size +
                                         d;
                local_norm_keys[local_offset] = (float)key_cache[key_offset] / norm;
            }
        }
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Step 2-5: For each column in eviction_size, compute aggregated similarity
    // Process columns in chunks (one column per iteration)
    for (int col_token = sglid; col_token < eviction_size; col_token += DIVERSITY_SUBGROUP_SIZE) {
        const int global_col_token = evictable_start_token + col_token;
        
        // Load and normalize the column token's keys (across all heads)
        float col_norm_keys[8];  // Assume head_size <= 128, process 8 at a time
        float aggregated_similarity = 0.0f;
        
        // For each head, compute similarity with current block's tokens
        for (int head_idx = 0; head_idx < num_kv_heads; head_idx++) {
            // Load column token normalized keys
            const int col_block_local_idx = global_col_token / block_size;
            const int col_pos_in_block = global_col_token % block_size;
            const int col_block_global_idx = block_indices[block_start_idx + col_block_local_idx];
            
            float col_keys_norm[128];  // Max head_size
            for (int d = 0; d < head_size; d++) {
                const uint key_offset = col_block_global_idx * num_kv_heads * head_size * block_size +
                                       head_idx * head_size * block_size +
                                       d * block_size +
                                       col_pos_in_block;
                float val = (float)key_cache[key_offset];
                col_keys_norm[d] = val;
            }
            
            // Normalize column keys
            float col_sum_squares = 0.0f;
            for (int d = 0; d < head_size; d++) {
                col_sum_squares += col_keys_norm[d] * col_keys_norm[d];
            }
            float col_norm = native_sqrt(col_sum_squares + EPSILON);
            for (int d = 0; d < head_size; d++) {
                col_keys_norm[d] /= col_norm;
            }
            
            // Compute similarity with each token in current block
            float head_similarity_sum = 0.0f;
            for (int token_in_block = 0; token_in_block < block_size; token_in_block++) {
                const int row_token = block_idx * block_size + token_in_block;
                
                // Skip diagonal
                if (row_token == col_token)
                    continue;
                
                // Compute dot product
                float dot_product = 0.0f;
                for (int d = 0; d < head_size; d++) {
                    const uint local_offset = head_idx * block_size * head_size +
                                             token_in_block * head_size +
                                             d;
                    dot_product += local_norm_keys[local_offset] * col_keys_norm[d];
                }
                
                head_similarity_sum += dot_product;
            }
            
            // Average similarity across tokens in block for this head
            float avg_similarity = head_similarity_sum / (float)block_size;
            
            // Apply mean threshold (simplified - compute per-head threshold)
            // In full implementation, would need row-wise mean
            aggregated_similarity += (avg_similarity > 0.0f) ? avg_similarity : 0.0f;
        }
        
        // Step 5: Aggregate across heads (mean)
        aggregated_similarity /= (float)num_kv_heads;
        
        // Step 6: Compute block diversity (negative similarity)
        const uint out_offset = block_idx * eviction_size + col_token;
        block_diversity[out_offset] = (OUTPUT_TYPE)(-aggregated_similarity);
    }
}
