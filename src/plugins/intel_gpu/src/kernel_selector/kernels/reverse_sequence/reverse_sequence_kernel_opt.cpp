// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reverse_sequence_kernel_opt.h"
#include "kernel_selector_utils.h"
#include <string>
#include <vector>

namespace kernel_selector {

constexpr size_t SUB_GROUP_SIZE = 16;
constexpr size_t VEC_SIZE = 8;
constexpr size_t TILE = SUB_GROUP_SIZE * VEC_SIZE;  // = 128 elements per subgroup

ParamsKey ReverseSequenceKernelOpt::GetSupportedKey() const {
    ParamsKey k;
    // Output data type is what determines the body op; restrict to fp16/fp32.
    // INT32 is also enabled as an input data type because seq_lengths is
    // INT32 — without it, a fp16-data + int32-seq_lengths instance would be
    // dropped at the supported-key match step. Validate() narrows actually
    // applicable cases by output element type.
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::INT32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDifferentTypes();
    return k;
}

bool ReverseSequenceKernelOpt::Validate(const Params& p) const {
    if (!Parent::Validate(p))
        DO_NOT_USE_THIS_KERNEL(p.layerID);

    const reverse_sequence_params& params = static_cast<const reverse_sequence_params&>(p);
    const auto& in = params.inputs[0];
    const auto& out = params.outputs[0];

    if (in.GetLayout() != DataLayout::bfyx)
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    if (in.GetDims().size() != 4)
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    if (in.is_dynamic() || out.is_dynamic())
        DO_NOT_USE_THIS_KERNEL(p.layerID);

    // Static-only path:
    // BiLSTM-decomposed reverse_sequence has X=1 and reverse on F axis.
    // The inner contiguous stride is then along Y. We require:
    //   - X == 1
    //   - SEQ_AXIS != 2 (so Y is not the reverse axis)
    //   - Y is divisible by VEC_SIZE
    //   - no padding on Y
    if (out.X().v != 1)
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    if (params.seq_axis == 2)
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    if (out.Y().v == 0 || out.Y().v % TILE != 0)
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    if (out.Y().pad.before != 0 || out.Y().pad.after != 0)
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    if (in.Y().pad.before != 0 || in.Y().pad.after != 0)
        DO_NOT_USE_THIS_KERNEL(p.layerID);

    return true;
}

JitConstants ReverseSequenceKernelOpt::GetJitConstants(const reverse_sequence_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);
    jit.AddConstant(MakeJitConstant("SEQ_AXIS", params.seq_axis));
    jit.AddConstant(MakeJitConstant("BATCH_AXIS", params.batch_axis));
    jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", SUB_GROUP_SIZE));
    return jit;
}

CommonDispatchData ReverseSequenceKernelOpt::SetDefault(const reverse_sequence_params& params) const {
    CommonDispatchData dispatchData;

    // GWS dim2 = Y / VEC_SIZE so each work-item nominally owns VEC_SIZE
    // consecutive Y elements. LWS dim2 = SUB_GROUP_SIZE makes one work-group
    // exactly one subgroup, which together loads/stores TILE = SG*VEC = 128
    // contiguous elements via intel_sub_group_block_read8/write8.
    dispatchData.gws = { params.outputs[0].Batch().v,
                         params.outputs[0].Feature().v,
                         params.outputs[0].Y().v / VEC_SIZE };
    dispatchData.lws = { 1, 1, SUB_GROUP_SIZE };

    return dispatchData;
}

KernelsData ReverseSequenceKernelOpt::GetKernelsData(const Params& params) const {
    if (!Validate(params))
        return {};

    KernelData kd = KernelData::Default<reverse_sequence_params>(params);
    reverse_sequence_params& newParams = *static_cast<reverse_sequence_params*>(kd.params.get());

    auto dispatchData = SetDefault(newParams);
    auto entry_point = GetEntryPoint(kernelName, newParams.layerID, params);
    auto cldnn_jit = GetJitConstants(newParams);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];
    FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point, "", false, false, 2);

    return {kd};
}

KernelsPriority ReverseSequenceKernelOpt::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_4;
}

}  // namespace kernel_selector
