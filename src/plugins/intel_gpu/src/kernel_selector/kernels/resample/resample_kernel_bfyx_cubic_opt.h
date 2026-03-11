// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "resample_kernel_base.h"

namespace kernel_selector {

// Optimized bicubic (CUBIC) interpolation kernel for bfyx planar layout.
// Restricts itself to 2D spatial resize (no B/F/Z interpolation), which
// allows a tight 4x4 = 16-iteration inner loop instead of the 4^5 = 1024
// iterations in the generic resample_ref path.
class ResampleKernelBfyxCubicOpt : public ResampleKernelBase {
public:
    using Parent = ResampleKernelBase;
    ResampleKernelBfyxCubicOpt() : ResampleKernelBase("resample_bfyx_cubic_opt") {}
    virtual ~ResampleKernelBfyxCubicOpt() = default;

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;

    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return { FusedOpType::ELTWISE, FusedOpType::ACTIVATION };
    }

protected:
    bool Validate(const Params& p) const override;
    JitConstants GetJitConstants(const resample_params& params) const override;
    DispatchData SetDefault(const resample_params& arg) const override;
};

}  // namespace kernel_selector
