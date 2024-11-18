// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"
#include "kernel_selector_params.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {
struct TransposedDimensionAccessHelperBase : virtual DimensionAccessHelperBase {
    explicit TransposedDimensionAccessHelperBase(const DataTensor& t, std::vector<int64_t> order)
    : DimensionAccessHelperBase(t) {
        size_t total_dims_count = dims.size();
        size_t new_axis_count = total_dims_count - order.size();
        transposed_order.resize(total_dims_count);
        std::iota(transposed_order.begin(), transposed_order.end(), 0);
        for (size_t i = 0; i < order.size(); i++) {
            size_t transposed_order_pos = i < 2 ? i : i + new_axis_count;
            transposed_order[transposed_order_pos] = order[i] < 2 ? order[i] : order[i] + new_axis_count;
        }
    }
    Tensor::Dim& x_dim() { return dims[transposed_order[7]]; }
    Tensor::Dim& y_dim() { return dims[transposed_order[6]]; }
    Tensor::Dim& z_dim() { return dims[transposed_order[5]]; }
    Tensor::Dim& w_dim() { return dims[transposed_order[4]]; }
    Tensor::Dim& v_dim() { return dims[transposed_order[3]]; }
    Tensor::Dim& u_dim() { return dims[transposed_order[2]]; }
    Tensor::Dim& f_dim() { return dims[transposed_order[1]]; }
    Tensor::Dim& b_dim() { return dims[transposed_order[0]]; }
    std::vector<int64_t> transposed_order;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// concatenation_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct concatenation_params : public base_params {
    concatenation_params() : base_params(KernelType::CONCATENATION) {}

    ConcatAxis axis = ConcatAxis::FEATURE;
    bool isAligned = true;
    size_t misalignment = 0;

    size_t kernel_split_id = 0;
    MultiDataTensor original_input_layouts;
    bool kernelPerInput = true;
    std::vector<int64_t> present_order;

    ParamsKey GetParamsKey() const override {
        auto k = base_params::GetParamsKey();
        k.EnableConcatAxis(axis);
        if (kernelPerInput) {
            k.EnableConcatKernelPerInput();
        } else {
            k.EnableConcatOneKernel();
        }
        return k;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ConcatenationKernelBase
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class ConcatenationKernelBase : public KernelBaseOpenCL {
public:
    using KernelBaseOpenCL::KernelBaseOpenCL;
    virtual ~ConcatenationKernelBase() {}

    using DispatchData = CommonDispatchData;

protected:
    bool Validate(const Params&) const override;
    virtual JitConstants GetJitConstants(const concatenation_params& params) const;
    virtual DispatchData SetDefault(const concatenation_params& params) const;
    KernelsData GetCommonKernelsData(const Params& params) const;
    int32_t GetConcatChannelIndex(const concatenation_params& params) const;
    Tensor::DataChannelName GetConcatChannel(const concatenation_params& params) const;
    virtual size_t GetAlignment(const concatenation_params& /*params*/) const {
        return 1;
    }
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;
};
}  // namespace kernel_selector
