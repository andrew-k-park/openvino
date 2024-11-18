// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iostream>
#include "tensor_type.h"
#include "concatenation_kernel_base.h"
#include <algorithm>
#include <vector>

namespace kernel_selector {

static std::string GetDimsOrder(const std::vector<int64_t>& order_idx) {
    auto get_order_idx = [](std::vector<int64_t> order_idx, int64_t dim_idx) {
        int loc = 0;
        for (auto idx : order_idx) {
            if (idx == dim_idx)
                break;
            loc += 1;
        }
        return loc;
    };
    std::string dims_order = "";
    if (order_idx.size() == 2) {
        const std::vector<std::string> dims2 = {"y", "x"};
        dims_order = "b,f,w,z,"
                    + dims2[get_order_idx(order_idx, 0)] + "," + dims2[get_order_idx(order_idx, 1)];
    } else if (order_idx.size() == 3) {
        const std::vector<std::string> dims3 = {"f", "y", "x"};
        dims_order = "b," + dims3[get_order_idx(order_idx, 0)] + ",w,z,"
                    + dims3[get_order_idx(order_idx, 1)] + "," + dims3[get_order_idx(order_idx, 2)];
    } else if (order_idx.size() == 4) {
        const std::vector<std::string> dims4 = {"b", "f", "y", "x"};
        dims_order = dims4[get_order_idx(order_idx, 0)] + "," + dims4[get_order_idx(order_idx, 1)] + ",w,z,"
                    + dims4[get_order_idx(order_idx, 2)] + "," + dims4[get_order_idx(order_idx, 3)];
    } else if (order_idx.size() == 5) {
        const std::vector<std::string> dims5 = {"b", "f", "z", "y", "x"};
        dims_order = dims5[get_order_idx(order_idx, 0)] + "," + dims5[get_order_idx(order_idx, 1)] + ",w,"
                    + dims5[get_order_idx(order_idx, 2)] + "," + dims5[get_order_idx(order_idx, 3)] + ","
                    + dims5[get_order_idx(order_idx, 4)];
    } else if (order_idx.size() == 6) {
        const std::vector<std::string> dims6 = {"b", "f", "w", "z", "y", "x"};
        dims_order = dims6[get_order_idx(order_idx, 0)] + "," + dims6[get_order_idx(order_idx, 1)] + ","
                    + dims6[get_order_idx(order_idx, 2)] + "," + dims6[get_order_idx(order_idx, 3)] + ","
                    + dims6[get_order_idx(order_idx, 4)] + "," + dims6[get_order_idx(order_idx, 5)];
    } else {
        dims_order = "b,f,w,z,y,x";
    }
    return dims_order;
}

Tensor::DataChannelName ConcatenationKernelBase::GetConcatChannel(const concatenation_params& params) const {
    switch (params.axis) {
        case ConcatAxis::X:
            return Tensor::DataChannelName::X;
        case ConcatAxis::Y:
            return Tensor::DataChannelName::Y;
        case ConcatAxis::Z:
            return Tensor::DataChannelName::Z;
        case ConcatAxis::W:
            return Tensor::DataChannelName::W;
        case ConcatAxis::FEATURE:
            return Tensor::DataChannelName::FEATURE;
        case ConcatAxis::BATCH:
            return Tensor::DataChannelName::BATCH;
        default:
            return Tensor::DataChannelName::X;
    }
}

int32_t ConcatenationKernelBase::GetConcatChannelIndex(const concatenation_params& params) const {
    return DataTensor::Channelndex(params.outputs[0].GetLayout(), GetConcatChannel(params));
}

bool ConcatenationKernelBase::Validate(const Params& p) const {
    if (p.GetType() != KernelType::CONCATENATION) {
        return false;
    }

    const concatenation_params& params = static_cast<const concatenation_params&>(p);

    for (auto& fused_op : params.fused_ops) {
        if (!IsFusedPrimitiveSupported(fused_op))
            return false;
    }

    if (GetConcatChannelIndex(params) == -1) {
        return false;
    }

    return true;
}

JitConstants ConcatenationKernelBase::GetJitConstants(const concatenation_params& params) const {
    auto& inputs = params.original_input_layouts;
    bool is_dynamic = std::any_of(inputs.begin(), inputs.end(), [](const DataTensor& t) { return t.is_dynamic(); }) ||
                      std::any_of(params.outputs.begin(), params.outputs.end(), [](const DataTensor& t) { return t.is_dynamic(); });
    JitConstants jit = MakeBaseParamsJitConstants(params, !is_dynamic);

    jit.AddConstants({
        MakeJitConstant("CONCAT_" + toString(params.axis), 1),
    });

    auto is_default_order = [](const std::vector<int64_t>& order) {
        for (size_t i = 0; i < order.size(); i++)
            if (order[i] != static_cast<int64_t>(i))
                return false;
        return true;
    };

    if (!params.present_order.empty() && !is_default_order(params.present_order)) {
        jit.AddConstant(MakeJitConstant("INPUT1_DIMS_ORDER", GetDimsOrder(params.present_order)));
    }

    if (is_dynamic) {
        jit.AddConstant(MakeJitConstant("INPUT0", params.inputs[0]));
        jit.AddConstant(MakeJitConstant("OUTPUT", params.outputs[0]));

        jit.AddConstant(MakeJitConstant("IS_DYNAMIC", 1));
        jit.AddConstant(MakeJitConstant("OPTIONAL_SHAPE_INFO_ARG", "__global const int* shape_info,"));
        jit.AddConstant(MakeJitConstant("OPTIONAL_SHAPE_INFO_TENSOR", "shape_info,"));
    }

    jit.AddConstant(MakeJitConstant("CONCAT_AXIS_INDEX", GetConcatChannelIndex(params)));
    return jit;
}

ConcatenationKernelBase::DispatchData ConcatenationKernelBase::SetDefault(const concatenation_params& params) const {
    DispatchData dispatchData;

    const auto& dims = params.inputs[0].GetDims();
    auto layout = params.inputs[0].GetLayout();

    std::vector<int> idx = { DataTensor::Channelndex(layout, Tensor::DataChannelName::BATCH),
                             DataTensor::Channelndex(layout, Tensor::DataChannelName::FEATURE),
                             DataTensor::Channelndex(layout, Tensor::DataChannelName::Y),
                             DataTensor::Channelndex(layout, Tensor::DataChannelName::X) };

    // Determine global work sizes.
    dispatchData.gws[0] = idx[2] != -1 ? dims[idx[2]].v : 1;  // Y
    dispatchData.gws[1] = idx[1] != -1 ? dims[idx[1]].v : 1;  // F
    dispatchData.gws[2] = idx[0] != -1 ? dims[idx[0]].v : 1;  // B

    dispatchData.lws[0] = std::min(std::max(dispatchData.gws[0], static_cast<size_t>(1)), static_cast<size_t>(32));
    while (dispatchData.gws[0] % dispatchData.lws[0] != 0) {
        --dispatchData.lws[0];
    }

    dispatchData.lws[1] = 1;
    dispatchData.lws[2] = 1;
    return dispatchData;
}

void ConcatenationKernelBase::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [this](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const concatenation_params&>(params);
        uint32_t lastOffset = 0;
        for (size_t i = 0; i < prim_params.inputs.size(); i++) {
            size_t ifm_offset = 0;

            const auto& input = prim_params.inputs[i];
            auto newParams = prim_params;
            newParams.inputs.resize(1);
            newParams.inputs[0] = input;
            size_t ifm = input.Feature().v;
            newParams.isAligned = ifm_offset % GetAlignment(newParams) == 0;
            newParams.misalignment = ifm_offset % GetAlignment(newParams);
            ifm_offset += ifm;
            if (i != 1)
                newParams.present_order = {};

            auto& kernel = kd.kernels[i];
            DispatchData dispatchData = SetDefault(newParams);
            kernel.params.workGroups.global = dispatchData.gws;
            kernel.params.workGroups.local = dispatchData.lws;
            kernel.skip_execution = KernelData::SkipKernelExecution(newParams);

            ScalarDescriptor s;
            s.t = ScalarDescriptor::Types::UINT32;
            s.v.u32 = lastOffset;
            kernel.params.scalars.resize(1);
            kernel.params.scalars[0] = s;

            auto concatChannelIndex = DataTensor::Channelndex(input.GetLayout(), GetConcatChannel(prim_params));
            OPENVINO_ASSERT(concatChannelIndex >= 0, "concatChannelIndex shouldn't be negative");
            lastOffset += (uint32_t)input.GetDims()[concatChannelIndex].v;
        }
    };
}

KernelsData ConcatenationKernelBase::GetCommonKernelsData(const Params& params) const {
    if (!Validate(params)) {
        return {};
    }

    const concatenation_params& orgParams = static_cast<const concatenation_params&>(params);
    KernelData kd = KernelData::Default<concatenation_params>(params, orgParams.inputs.size());
    kd.needs_sub_kernels_sync = false;
    GetUpdateDispatchDataFunc(kd);

    bool is_dynamic = orgParams.has_dynamic_tensors();
    uint32_t lastOffset = 0;
    size_t ifm_offset = 0;
    for (size_t i = 0; i < orgParams.inputs.size(); i++) {
        const auto& input = orgParams.inputs[i];
        auto newParams = orgParams;
        newParams.inputs.resize(1);
        newParams.inputs[0] = input;
        size_t ifm = input.Feature().v;
        newParams.isAligned = ifm_offset % GetAlignment(newParams) == 0;
        newParams.misalignment = ifm_offset % GetAlignment(newParams);
        ifm_offset += ifm;
        if (i != 1)
            newParams.present_order = {};

        newParams.kernel_split_id = i;
        newParams.original_input_layouts = orgParams.inputs;

        auto& kernel = kd.kernels[i];
        DispatchData dispatchData = SetDefault(newParams);
        auto cldnnJit = GetJitConstants(newParams);
        auto entryPoint = GetEntryPoint(kernelName, newParams.layerID, params, i);
        auto jit = CreateJit(kernelName, cldnnJit, entryPoint);

        kernel.code.kernelString = GetKernelString(kernelName, jit, entryPoint, params.engineInfo);
        kernel.params.workGroups.global = dispatchData.gws;
        kernel.params.workGroups.local = dispatchData.lws;
        kernel.skip_execution = KernelData::SkipKernelExecution(newParams);
        if (is_dynamic) {
            kernel.params.arguments.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
        }
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, (uint32_t) i});
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::OUTPUT, 0});

        ScalarDescriptor s;
        s.t = ScalarDescriptor::Types::UINT32;
        s.v.u32 = lastOffset;
        kernel.params.scalars.push_back(s);
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::SCALAR, 0});
        size_t concatChannelIndex = (size_t)DataTensor::Channelndex(orgParams.inputs[i].GetLayout(), GetConcatChannel(orgParams));
        lastOffset += (uint32_t)input.GetDims()[concatChannelIndex].v;
    }

    return {kd};
}
}  // namespace kernel_selector
