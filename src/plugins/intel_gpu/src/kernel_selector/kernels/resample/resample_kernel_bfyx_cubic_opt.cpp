// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "resample_kernel_bfyx_cubic_opt.h"
#include <kernel_selector_utils.h>
#include <string>
#include <vector>

namespace kernel_selector {

ParamsKey ResampleKernelBfyxCubicOpt::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableDifferentTypes();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableReampleType(ResampleType::CUBIC);
    k.EnableDynamicShapesSupport();
    return k;
}

bool ResampleKernelBfyxCubicOpt::Validate(const Params& p) const {
    if (!Parent::Validate(p))
        DO_NOT_USE_THIS_KERNEL(p.layerID);

    const resample_params& params = static_cast<const resample_params&>(p);

    // Only 4D tensors (no Z axis).
    if (params.inputs[0].Dimentions() > 4)
        DO_NOT_USE_THIS_KERNEL(p.layerID);

    // Batch and feature dimensions must not be resized.
    const auto& input  = params.inputs[0];
    const auto& output = params.outputs[0];
    if (input.Batch().v != output.Batch().v || input.Feature().v != output.Feature().v)
        DO_NOT_USE_THIS_KERNEL(p.layerID);

    return true;
}

ResampleKernelBase::DispatchData ResampleKernelBfyxCubicOpt::SetDefault(const resample_params& arg) const {
    DispatchData dispatchData;

    const auto& out    = arg.outputs[0];
    auto in_layout     = arg.inputs[0].GetLayout();
    auto out_layout    = arg.outputs[0].GetLayout();

    // One work-item per output pixel; feature and batch packed in dim 2.
    dispatchData.gws = { out.X().v, out.Y().v, out.Feature().v * out.Batch().v };

    std::vector<std::vector<Tensor::DataChannelName>> dims_by_gws = {
        { Tensor::DataChannelName::X },
        { Tensor::DataChannelName::Y },
        { Tensor::DataChannelName::FEATURE, Tensor::DataChannelName::BATCH }
    };

    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, arg.engineInfo,
                                                     in_layout, out_layout, dims_by_gws);
    return dispatchData;
}

KernelsPriority ResampleKernelBfyxCubicOpt::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_3;
}

JitConstants ResampleKernelBfyxCubicOpt::GetJitConstants(const resample_params& params) const {
    JitConstants jit = Parent::GetJitConstants(params);

    if (!params.fused_ops.empty()) {
        // Variable names used inside the kernel: b, f, oy, ox.
        // Interpolation result variable: val.
        FusedOpsConfiguration conf = {"", {"b", "f", "oy", "ox"},
                                      "val", GetAccumulatorType(params), 1};
        jit.Merge(MakeFusedOpsJitConstants(params, {conf}));
    }

    return jit;
}

void ResampleKernelBfyxCubicOpt::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [this](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const resample_params&>(params);
        auto dispatchData = SetDefault(prim_params);
        OPENVINO_ASSERT(kd.kernels.size() == 1, "[GPU] Invalid kernels size for update dispatch data func");
        kd.kernels[0].params.workGroups.global = dispatchData.gws;
        kd.kernels[0].params.workGroups.local  = dispatchData.lws;
        kd.kernels[0].skip_execution = KernelData::SkipKernelExecution(prim_params);
    };
}

KernelsData ResampleKernelBfyxCubicOpt::GetKernelsData(const Params& params) const {
    if (!Validate(params))
        return {};

    KernelData kd = KernelData::Default<resample_params>(params);
    resample_params& newParams = *static_cast<resample_params*>(kd.params.get());

    auto dispatchData = SetDefault(newParams);
    auto entry_point  = GetEntryPoint(kernelName, newParams.layerID, params);
    auto cldnn_jit    = GetJitConstants(newParams);
    auto jit          = CreateJit(kernelName, cldnn_jit, entry_point);

    GetUpdateDispatchDataFunc(kd);

    auto& kernel = kd.kernels[0];
    FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point,
                     EXE_MODE_DEFAULT, false, false, 1, GetFusedPrimitiveInputsCount(params), 1,
                     newParams.is_shape_agnostic);
    return {kd};
}

}  // namespace kernel_selector
