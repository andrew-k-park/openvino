// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {

// Rewrites 1D GroupConvolutionBackpropData as Unsqueeze -> 2D GroupConvolutionBackpropData (H = 1) -> Squeeze.
// The deconvolution primitive reads 4D grouped weights with the kernel on X while 1D data is laid out on Y.
class ConvertGroupConvolutionBackpropData1DTo2D : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertGroupConvolutionBackpropData1DTo2D");
    ConvertGroupConvolutionBackpropData1DTo2D();
};

}  // namespace ov::intel_gpu
