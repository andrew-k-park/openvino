// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace intel_gpu {

class ReshapeFusion: public ov::pass::GraphRewrite {
public:
    OPENVINO_RTTI("ReshapeFusion", "0");
    ReshapeFusion();
};

class KVCacheReshapeMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("KVCacheReshapeMatcher", "0");
    KVCacheReshapeMatcher();
};

}   // namespace intel_gpu
}   // namespace ov
