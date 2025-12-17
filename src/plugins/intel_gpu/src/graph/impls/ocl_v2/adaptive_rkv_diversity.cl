// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"
#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"
#include "include/batch_headers/sub_group_shuffle.cl"

// Adaptive R-KV Diversity Calculation Kernel
// Implements the diversity calculation from https://arxiv.org/pdf/2505.24133v3
// This kernel computes per-block diversity values for the eviction area in paged attention

REQD_SUB_GROUP_SIZE(SUBGROUP_SIZE)
KERNEL(adaptive_rkv_diversity)(
    const __global INPUT0_TYPE* key_cache,           // [num_blocks, num_heads, head_size, block_size]
    const __global int* past_lens,                   // [batch_size]
    const __global int* block_indices,               // [total_blocks]
    const __global int* block_indices_begins,        // [batch_size + 1]
    const __global int* adaptive_rkv_start_size,     // [1] - scalar
    const __global int* adaptive_rkv_evictable_sizes, // [batch_size]
    const __global int* diversity_block_set_indices, // [num_diversity_blocks]
    const __global int* diversity_block_set_begins,  // [batch_size + 1]
    __global OUTPUT_TYPE* diversity_output,           // [total_diversity_elements]
    __global OUTPUT_TYPE* normalized_keys_buffer      // Temporary buffer for normalized keys
) {
    const uint seq_idx = get_global_id(0);
    const uint head_idx = get_global_id(1);
    const uint sglid = get_sub_group_local_id();
    
    const int start_size = adaptive_rkv_start_size[0];
    const int evictable_size = adaptive_rkv_evictable_sizes[seq_idx];
    const int seq_len = past_lens[seq_idx] + 1;
    
    if (evictable_size == 0 || seq_len < start_size + evictable_size)
        return;
    
    const int start_token_idx = start_size;
    const int end_token_idx = start_size + evictable_size;
    const int num_evictable_blocks = evictable_size / PAGED_ATTENTION_BLOCK_SIZE;
    
    const int block_begin = block_indices_begins[seq_idx];
    
    // Step 1: Normalize keys using L2 normalization
    // For each token in the eviction area, normalize its key vector
    for (int token_offset = 0; token_offset < evictable_size; token_offset++) {
        const int token_idx = start_token_idx + token_offset;
        const int block_offset = token_idx / PAGED_ATTENTION_BLOCK_SIZE;
        const int token_in_block = token_idx % PAGED_ATTENTION_BLOCK_SIZE;
        
        const int block_indice = block_indices[block_begin + block_offset];
        const uint key_offset = block_indice * HEAD_SIZE * NUM_HEADS * PAGED_ATTENTION_BLOCK_SIZE +
                               head_idx * HEAD_SIZE * PAGED_ATTENTION_BLOCK_SIZE +
                               token_in_block;
        
        // Compute L2 norm
        SOFTMAX_ACCUMULATOR_TYPE norm_sq = 0.0f;
        for (int h = 0; h < HEAD_SIZE; h++) {
            INPUT0_TYPE key_val = key_cache[key_offset + h * PAGED_ATTENTION_BLOCK_SIZE];
            norm_sq += TO_SOFTMAX_ACCUMULATOR_TYPE(key_val) * TO_SOFTMAX_ACCUMULATOR_TYPE(key_val);
        }
        
        SOFTMAX_ACCUMULATOR_TYPE norm = native_sqrt(norm_sq + 1e-12f);
        
        // Store normalized keys
        const uint norm_key_offset = (seq_idx * NUM_HEADS * evictable_size * HEAD_SIZE) +
                                     (head_idx * evictable_size * HEAD_SIZE) +
                                     (token_offset * HEAD_SIZE);
        
        for (int h = 0; h < HEAD_SIZE; h++) {
            INPUT0_TYPE key_val = key_cache[key_offset + h * PAGED_ATTENTION_BLOCK_SIZE];
            normalized_keys_buffer[norm_key_offset + h] = TO_OUTPUT_TYPE(TO_SOFTMAX_ACCUMULATOR_TYPE(key_val) / norm);
        }
    }
    
    barrier(CLK_GLOBAL_MEM_FENCE);
    
    // Step 2: Compute cosine similarity matrix (dot product of normalized vectors)
    __local SOFTMAX_ACCUMULATOR_TYPE slm_cos_sim[MAX_EVICTABLE_SIZE * MAX_EVICTABLE_SIZE];
    
    for (int i = get_local_id(2); i < evictable_size; i += get_local_size(2)) {
        for (int j = 0; j < evictable_size; j++) {
            const uint key1_offset = (seq_idx * NUM_HEADS * evictable_size * HEAD_SIZE) +
                                     (head_idx * evictable_size * HEAD_SIZE) +
                                     (i * HEAD_SIZE);
            const uint key2_offset = (seq_idx * NUM_HEADS * evictable_size * HEAD_SIZE) +
                                     (head_idx * evictable_size * HEAD_SIZE) +
                                     (j * HEAD_SIZE);
            
            SOFTMAX_ACCUMULATOR_TYPE dot_product = 0.0f;
            for (int h = 0; h < HEAD_SIZE; h++) {
                dot_product += TO_SOFTMAX_ACCUMULATOR_TYPE(normalized_keys_buffer[key1_offset + h]) *
                              TO_SOFTMAX_ACCUMULATOR_TYPE(normalized_keys_buffer[key2_offset + h]);
            }
            
            slm_cos_sim[i * evictable_size + j] = dot_product;
        }
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Step 3: Fill diagonal with zeros
    if (get_local_id(2) < evictable_size) {
        slm_cos_sim[get_local_id(2) * evictable_size + get_local_id(2)] = 0.0f;
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Step 4: Compute mean for each row and filter low values
    for (int i = get_local_id(2); i < evictable_size; i += get_local_size(2)) {
        SOFTMAX_ACCUMULATOR_TYPE row_sum = 0.0f;
        for (int j = 0; j < evictable_size; j++) {
            row_sum += slm_cos_sim[i * evictable_size + j];
        }
        SOFTMAX_ACCUMULATOR_TYPE row_mean = row_sum / evictable_size;
        
        // Zero out values below mean
        for (int j = 0; j < evictable_size; j++) {
            if (slm_cos_sim[i * evictable_size + j] < row_mean) {
                slm_cos_sim[i * evictable_size + j] = 0.0f;
            }
        }
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Step 5: Aggregate across heads (mean reduction)
    __local SOFTMAX_ACCUMULATOR_TYPE slm_aggregated[MAX_EVICTABLE_SIZE * MAX_EVICTABLE_SIZE];
    
    if (head_idx == 0) {
        for (int i = get_local_id(2); i < evictable_size * evictable_size; i += get_local_size(2)) {
            slm_aggregated[i] = slm_cos_sim[i];
        }
    }
    
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    
    if (head_idx > 0) {
        for (int i = get_local_id(2); i < evictable_size * evictable_size; i += get_local_size(2)) {
            atomic_add_global(&slm_aggregated[i], slm_cos_sim[i] / NUM_HEADS);
        }
    }
    
    barrier(CLK_GLOBAL_MEM_FENCE);
    
    // Step 6: Block-wise diversity aggregation
    // Sum diversity values per block and negate to get final diversity score
    if (head_idx == 0) {
        const int diversity_output_begin = diversity_block_set_begins[seq_idx];
        
        for (int block_idx = get_local_id(2); block_idx < num_evictable_blocks; block_idx += get_local_size(2)) {
            const int block_start_token = block_idx * PAGED_ATTENTION_BLOCK_SIZE;
            
            // Sum across tokens in this block (rows) and all evictable tokens (columns)
            for (int token_offset = 0; token_offset < evictable_size; token_offset++) {
                SOFTMAX_ACCUMULATOR_TYPE block_diversity = 0.0f;
                
                for (int token_in_block = 0; token_in_block < PAGED_ATTENTION_BLOCK_SIZE; token_in_block++) {
                    int row = block_start_token + token_in_block;
                    if (row < evictable_size) {
                        block_diversity -= slm_aggregated[row * evictable_size + token_offset];
                    }
                }
                
                // Output: [num_eviction_blocks, eviction_size]
                const int output_offset = diversity_output_begin + 
                                         (block_idx * evictable_size) + 
                                         token_offset;
                diversity_output[output_offset] = TO_OUTPUT_TYPE(block_diversity);
            }
        }
    }
}
