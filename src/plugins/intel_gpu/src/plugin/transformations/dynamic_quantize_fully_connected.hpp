// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {

class DynamicQuantizeFullyConnected: public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("DynamicQuantizeFullyConnected");
    DynamicQuantizeFullyConnected(uint64_t group_size, bool asymmetric = false, bool precompute_sum = true, bool use_gs128_for_int8_per_token = false);
    static bool ShouldUseGs128(uint64_t is_wei_i8u8, bool use_gs128_for_int8_per_token, uint64_t group_size) {
        return (is_wei_i8u8 && use_gs128_for_int8_per_token && group_size == UINT64_MAX);
    }
    // WA: Qwen3-Next / Qwen3.5 hybrid models contain Mamba2-style "linear_attn" blocks whose
    // activations (gated SSM state outputs) have a very wide dynamic range. Per-token INT8
    // activation quantization on these FCs causes severe accuracy loss on dGPU. Force a smaller
    // group size (128) instead of disabling dyn-quant entirely so that performance is preserved.
    static bool ShouldUseGs128ForLinearAttn(const std::string& friendly_name, uint64_t group_size) {
        return group_size == UINT64_MAX && friendly_name.find("linear_attn") != std::string::npos;
    }
};

}   // namespace ov::intel_gpu
