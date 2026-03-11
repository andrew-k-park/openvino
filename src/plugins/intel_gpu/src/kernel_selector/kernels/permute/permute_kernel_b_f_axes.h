// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "permute_kernel_base.h"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// PermuteKernel_b_f_axes
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Optimized kernel for permutations that swap only the B and F axes while keeping
// all spatial axes in place.  Handles order [1, 0, 2], [1, 0, 2, 3], [1, 0, 2, 3, 4]
// (i.e. bfyx->fbyx, bfzyx->fbzyx, bfwzyx->fbwzyx).
// Because X is the innermost (contiguous) dimension, a simple vectorized load/store
// along X is sufficient — no SLM transpose required.
class PermuteKernel_b_f_axes : public PermuteKernelBase {
public:
    using Parent = PermuteKernelBase;
    using Parent::Parent;
    PermuteKernel_b_f_axes() : PermuteKernelBase("permute_b_f_axes") {}
    virtual ~PermuteKernel_b_f_axes() {}

    bool Validate(const Params& p) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    JitConstants GetJitConstants(const permute_params& params, const CommonDispatchData& dispatchData) const override;
    CommonDispatchData SetDefault(const permute_params& params) const override;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return {
            FusedOpType::REORDER,
            FusedOpType::ACTIVATION,
            FusedOpType::QUANTIZE,
            FusedOpType::ELTWISE
        };
    }
};
}  // namespace kernel_selector
