// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @file cpu_streams_calculation.hpp
 * @brief A header file for CPU streams calulation implementation.
 */

#pragma once

#include <memory>
#include <set>
#include <string>
#include <vector>

#include "config.h"
#include "openvino/core/model.hpp"
#include "openvino/runtime/properties.hpp"

namespace ov::intel_cpu {
/**
 * @brief      Generate streams information table according to processors type table.
 * @param[in]  input_streams is the targeted number of streams set by user via ov::num_streams or the default value.
 * @param[in]  input_streams_changed indicates if streams is set by user via ov::num_streams.
 * @param[in]  input_threads is the max number of threads set by user via ov::inference_num_threads or the default
 * value.
 *               - input "0" indicates that the function can use all resource in proc_type_table.
 *               - If user limits the max number of threads, the final number of streams output cannot exceed the max
 * number of threads.
 * @param[in]  input_infer_requests is max number of infer requests set by user via ov::hint::num_requests.
 *               - input "0" indicates that the function can use all resource in proc_type_table.
 *               - If user limits the max number of infer requests, the final number of streams output cannot exceed the
 * max number of infer requests.
 * @param[in]  model_prefer_threads is preferred number of threads per stream based on the model generated in previous
 * function.
 *               - input "0" indicates that the function generates the optimal number of threads per stream based on
 * processors type information.
 * @param[in]  enable_tensor_parallel is test flag used with hint_llm_distribution_policy to test data accurcy of
 * TENSOR_PARALLEL
 * @param[in]  input_perf_hint is performance hint set by user via ov::hint::performance_mode or the default value.
 * @param[in]  hint_llm_distribution_policy is the distribution policy for Large language models
 * @param[in]  proc_type_table is currently available candidate processors.
 *               - candidate processors have benn updated based on user input hints like ov::hint::scheduling_core_type
 * in previous function.
 * @return     streams information table which will be used by StreamsExecutor.
 */
std::vector<std::vector<int>> get_streams_info_table(
    int input_streams,
    bool input_streams_changed,
    int input_threads,
    int input_infer_requests,
    int model_prefer_threads,
    bool enable_tensor_parallel,
    const std::string& input_perf_hint,
    const std::set<ov::hint::ModelDistributionPolicy>& hint_model_distribution_policy,
    const std::vector<std::vector<int>>& proc_type_table);

/**
 * @brief      Generate streams rank table for tensor parallel according to streams info table.
 * @param[in]  streams_info_table is streams information table for tensor parallel.
 * @param[in]  input_rank_level is depth of rank nesting.
 * @param[out] num_sub_streams is number of sub streams for tensor parallel.
 * @return     streams rank table which will be used by StreamsExecutor.
 */
std::vector<std::vector<int>> get_streams_rank_table(const std::vector<std::vector<int>>& streams_info_table,
                                                     int input_rank_level,
                                                     int& num_sub_streams);

/**
 * @brief      Get model_prefer_threads
 * @param[in]  num_streams is target streams set by user via NUM_STREAMS or hints.
 *               - input "0" mean function generate the optimal number of streams
 *               - LATENCY hint equals 1 stream.
 * @param[in]  proc_type_table candidate processors available at this time
 *               - candidate processors have benn updated based on properties like "Ecore only" in previous function
 * @param[in]  model model
 * @param[in]  config intel cpu configuration
 * @return     model_prefer_threads "0" means generating the optimal threads per stream based on platform
 */
int get_model_prefer_threads(int num_streams,
                             const std::vector<std::vector<int>>& proc_type_table,
                             const std::shared_ptr<ov::Model>& model,
                             Config& config);

/**
 * @brief      Generate streams information according to processors type table
 * @param[in]  streams number of streams
 * @param[in]  input_numa_node_id is the numa node ID in cpu mapping table of the currently running thread
 *               - input "-1" indicates that the function get_streams_info_table will query this id internally.
 * @param[in]  model graph handle
 * @param[in]  config intel cpu configuration
 * @param[in]  proc_type_table candidate processors available at current platform
 * @param[in]  preferred_nthreads_per_stream is initial preferred number of threads per stream
 * @return     candidate processors have benn updated based on user input hints like ov::hint::scheduling_core_type and
 * ov::hint::enable_hyper_threading
 */
std::vector<std::vector<int>> generate_stream_info(int streams,
                                                   int input_numa_node_id,
                                                   const std::shared_ptr<ov::Model>& model,
                                                   Config& config,
                                                   std::vector<std::vector<int>>& proc_type_table,
                                                   int preferred_nthreads_per_stream = -1);

/**
 * @brief      Get information about number of streams, threads and pinning threads on different processors
 * @param[in]  streams number of streams
 * @param[in]  model graph handle
 * @param[in]  config intel cpu configuration
 */
void get_num_streams(int streams, const std::shared_ptr<ov::Model>& model, Config& config);

/**
 * @brief      Sort proc_type_table by numa node id on which application is running. The numa node will move to first
 * row.
 * @param[in]  current_numa_node numa node ID on which application is running.
 * @param[in]  proc_type_table summary table of number of processors per type
 */
void sort_table_by_numa_node_id(int current_numa_node, std::vector<std::vector<int>>& proc_type_table);

}  // namespace ov::intel_cpu
