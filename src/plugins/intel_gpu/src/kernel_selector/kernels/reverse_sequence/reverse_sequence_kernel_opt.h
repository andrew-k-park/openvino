// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"
#include "reverse_sequence_kernel_ref.h"

namespace kernel_selector {
// Vectorized variant of reverse_sequence_ref.
//
// Status: WIP — currently disabled in the selector. Initial Y-axis
// vectorization attempt produced output-size mismatch on Kokoro because the
// inputs are dynamic-shape (X reported as 1 at compile time turned out to be
// a placeholder, and INPUT0_GET_INDEX did not give the unit-stride access
// the kernel assumed). Needs a redesign that does not rely on compile-time
// shape values.
//
// Intended target chains (from BiLSTM decomposition in Kokoro):
//   - bfyx 4D, B=1, F=seq, Y=hidden*dirs (256/512/640), X=1
//   - reverse axis = F (seq_axis = 1)
class ReverseSequenceKernelOpt : public KernelBaseOpenCL {
public:
    using Parent = KernelBaseOpenCL;
    ReverseSequenceKernelOpt() : KernelBaseOpenCL("reverse_sequence_opt") {}
    ~ReverseSequenceKernelOpt() override = default;

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    bool Validate(const Params& p) const override;
    JitConstants GetJitConstants(const reverse_sequence_params& params) const;
    CommonDispatchData SetDefault(const reverse_sequence_params& params) const;
};
}  // namespace kernel_selector
