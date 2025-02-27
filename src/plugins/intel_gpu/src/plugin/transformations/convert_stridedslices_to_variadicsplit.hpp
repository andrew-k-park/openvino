// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {

class ConvertStridedSlicesToVariadicSplit : public ov::pass::GraphRewrite {
public:
    OPENVINO_GRAPH_REWRITE_RTTI("ConvertStridedSlicesToVariadicSplit");
    ConvertStridedSlicesToVariadicSplit();
};

class ConvertFCStridedSlicesToVariadicSplitMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertFCStridedSlicesToVariadicSplitMatcher");
    ConvertFCStridedSlicesToVariadicSplitMatcher();
};

class ConvertReshapeStridedSlicesToVariadicSplitMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertReshapeStridedSlicesToVariadicSplitMatcher");
    ConvertReshapeStridedSlicesToVariadicSplitMatcher();
};

}   // namespace ov::intel_gpu
