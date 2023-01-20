// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fully_connected_kernel_bf_tiled.h"

#include <vector>
#include <functional>

static constexpr size_t simd = 16;

namespace kernel_selector {

FullyConnected_bf_tiled::FullyConnected_bf_tiled() : FullyConnectedKernelBase("fully_connected_gpu_bf_tiled") {
    for (unsigned tile_b = 1; tile_b <= 32; ++tile_b)
    for (unsigned tile_ofm = 1; tile_ofm <= 4; tile_ofm *= 2)
    for (unsigned tile_ifm = 1; tile_ifm <= 2; tile_ifm *= 2)
    for (unsigned tile_k = 1; tile_k <= 8; tile_k *= 2)
    for (unsigned dispatch_bsv = 1; dispatch_bsv <= 16; ++dispatch_bsv)
    for (unsigned dispatch_fsv = 1; dispatch_fsv <= 16; ++dispatch_fsv)
    for (auto exec : Parent::autoTuneOptions) {
        // Block reads support at most vector size of 8.
        if (tile_k * tile_ofm > 8)
            continue;
        // For bsv == 1 dispatch order reduces to b_fsv, so anything other than fsv == 1 is redundant.
        if (dispatch_bsv == 1 && dispatch_fsv != 1)
            continue;

        auto_tune_params.emplace_back(tile_b, tile_ofm, tile_ifm, tile_k, dispatch_bsv, dispatch_fsv, exec);
    }
}

ParamsKey FullyConnected_bf_tiled::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableInputWeightsType(WeightsType::F16);
    k.EnableInputWeightsType(WeightsType::F32);
    k.EnableInputLayout(DataLayout::bf);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bf);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableBatching();
    k.EnableBiasPerFeature();
    k.EnableNonBiasTerm();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableDifferentTypes();
    k.EnableDifferentInputWeightsTypes();
    k.EnableDynamicShapesSupport();
    return k;
}

DeviceFeaturesKey FullyConnected_bf_tiled::get_required_device_features_key(const Params& params, const optional_params& options) const {
    auto k = get_common_subgroups_device_features_key(params, options);
    k.requires_subgroup_shuffle();

    return k;
}

bool FullyConnected_bf_tiled::Validate(const Params& params, const optional_params& options) const {
    if (!Parent::Validate(params, options))
        return false;

    // static size_t counter = 0;
    // if (params.layerID.find("MatMul_1159") != std::string::npos) {
    //     std::cout << "Layer DETECTED " << params.layerID << std::endl;
    //     return false;
    // }

    // counter++;
    // if (counter > 83) {
    //     std::cout << "Layer DETECTED2 " << params.layerID << "(" << counter << ")" << std::endl;
    //     return false;
    // }

    auto& fc_params = static_cast<const fully_connected_params&>(params);
    auto& input = fc_params.inputs[0];
    auto& output = fc_params.outputs[0];

    // Block reads must be aligned to 4 bytes, for fp16 we can correct for offset misalignment,
    // but we need to ensure that batch pitch preserves alignment.
    if (input.GetDType() == Datatype::F16) {
        if (input.Batch().pitch % 2 != 0 && input.Batch().v > 1)
            return false;
        // for 3d case we have to check feature alignment as well
        if (output.GetLayout() == DataLayout::bfyx && input.Feature().pitch % 2 != 0 && input.Feature().v > 1)
            return false;
    }

    if (input.GetLayout() == DataLayout::bfyx) {
        // Padding on input is not supported.
        // TODO: Enable by mirroring the padding in weights.
        if (input.X().pad.Total() != 0)
            return false;
        if (input.Y().pad.Total() != 0)
            return false;
    }

    // We don't support 4d output
    if (fc_params.outputs[0].GetLayout() == DataLayout::bfyx) {
        if (input.X().v > 1)
            return false;
    }

    return true;
}

namespace {

struct TuneParamsSelector {
    using tune_params = FullyConnected_bf_tiled::tune_params;
    using functional_case = std::function<tune_params(const fully_connected_params&)>;

    TuneParamsSelector(const fully_connected_params& params) : params(params), selected(false) {}

    TuneParamsSelector& Case(const tune_params& tparams) {
        if (!selected && VerifyTuneParams(params, tparams, ref)) {
            result = tparams;
            selected = true;
        }
        return *this;
    }

    TuneParamsSelector& Case(functional_case fun) {
        return Case(fun(params));
    }

    tune_params Default(const tune_params& tparams) {
        if (!selected) {
            selected = true;
            result = tparams;
        }
        return result;
    }

    static bool VerifyTuneParams(const fully_connected_params& params,
                                 const tune_params& tparams,
                                 std::shared_ptr<tune_params> dyn_tparams = nullptr);

    const fully_connected_params& params;
    bool selected;
    tune_params result;
    std::shared_ptr<tune_params> ref;
};

bool TuneParamsSelector::VerifyTuneParams(const fully_connected_params& params, const tune_params& tparams, std::shared_ptr<tune_params> dyn_tparams) {
    // Check divisibility by dispatch tile sizes.
    size_t output_f = params.outputs[0].Feature().v;
    size_t output_b = params.outputs[0].Batch().v;
    if (params.outputs[0].GetLayout() == DataLayout::bfyx) {
        output_b *= params.outputs[0].Feature().v;
        output_f = params.outputs[0].Y().v;
    }

    std::cout << "Verify (_" << tparams.tile_b
                << "_" << tparams.tile_ofm
                << "_" << tparams.tile_ifm
                << "_" << tparams.tile_k
                << "_" << tparams.dispatch_bsv
                << "_" << tparams.dispatch_fsv << ")" << std::endl;

    if (dyn_tparams) {
        bool supports = /* dyn_tparams->tile_b == tparams.tile_b && */
                        dyn_tparams->tile_ofm == tparams.tile_ofm &&
                        dyn_tparams->tile_ifm == tparams.tile_ifm &&
                        dyn_tparams->tile_k == tparams.tile_k;
                        // dyn_tparams->tile_k == tparams.tile_k &&
                        // dyn_tparams->dispatch_bsv == tparams.dispatch_bsv &&
                        // dyn_tparams->dispatch_fsv == tparams.dispatch_fsv;
        if (!supports) {
            std::cout << "Unsupported dyn_params\n";
            return false;
        } else {
            std::cout << "It's okay, prev params: "
                        << dyn_tparams->tile_b
                        << "_" << dyn_tparams->tile_ofm
                        << "_" << dyn_tparams->tile_ifm
                        << "_" << dyn_tparams->tile_k
                        << "_" << dyn_tparams->dispatch_bsv
                        << "_" << dyn_tparams->dispatch_fsv << "" << std::endl;
        }
    }

    if (dyn_tparams && output_b > 8) {
        if (Align(output_b, tparams.tile_b) % (tparams.tile_b * tparams.dispatch_bsv) != 0) {
            std::cout << "False 1.1 (Dynamic): " << Align(output_b, tparams.tile_b) << " % " << tparams.tile_b * tparams.dispatch_bsv << " != 0\n";
            return false;
        }
    } else {
        if (output_b % (tparams.tile_b * tparams.dispatch_bsv) != 0) {
            std::cout << "False 1.1: " << output_b << " % " << tparams.tile_b * tparams.dispatch_bsv << " != 0\n";
            return false;
        }
    }

    if (params.has_dynamic_tensors() && tparams.tile_b * tparams.dispatch_bsv > 32) {
        std::cout << "False 1.2: " << output_b << " % " << tparams.tile_b * tparams.dispatch_bsv << " != 0\n";
        return false;
    }


    if (CeilDiv(output_f, tparams.tile_ofm * simd) % tparams.dispatch_fsv != 0) {
        std::cout << "False 2: " << CeilDiv(output_f, tparams.tile_ofm * simd) << " % " << tparams.dispatch_fsv << " != 0\n";
        return false;
    }

    // Same result can be achieved with smaller tile_ofm.
    if (output_f <= (tparams.tile_ofm / 2) * simd) {
        std::cout << "False 3: " << output_f << " <= " << (tparams.tile_ofm / 2) * simd << "\n";
        return false;
    }
    // No weights layout for such huge tile ofm.
    if (tparams.tile_ofm * simd > 64) {
        std::cout << "False 4: " << tparams.tile_ofm * simd << " > " << "64\n";
        return false;
    }

    // Reject tile sizes that are guaranteed to spill out of registers.
    unsigned acc_register_bytes = tparams.tile_b * tparams.tile_ofm * simd * BytesPerElement(params.inputs[0].GetDType());
    unsigned in_register_bytes = tparams.tile_b * tparams.tile_ifm * simd * BytesPerElement(params.inputs[0].GetDType());
    unsigned wei_register_bytes = tparams.tile_ofm * tparams.tile_k * simd * BytesPerElement(params.weights.GetDType());

    unsigned total_register_bytes = acc_register_bytes + in_register_bytes + wei_register_bytes;
    unsigned max_register_bytes = 128 * 32;

    if (total_register_bytes > max_register_bytes) {
        std::cout << "False 5: registers\n";
        return false;
    }

    return true;
}

}  // namespace

FullyConnected_bf_tiled::tune_params
FullyConnected_bf_tiled::GetAutoTuneParams(const fully_connected_params& params, int idx, bool) const {
    if (idx >= 0 && idx < static_cast<int>(auto_tune_params.size())
        && TuneParamsSelector::VerifyTuneParams(params, auto_tune_params[idx]))
        return auto_tune_params[idx];

    size_t batch = params.outputs[0].Batch().v;
    size_t output_f = params.outputs[0].Feature().v;

    // 3d output
    if (params.outputs[0].GetLayout() == DataLayout::bfyx) {
        batch *= params.outputs[0].Feature().v;
        output_f = params.outputs[0].Y().v;
    }
    Datatype dtype = params.inputs[0].GetDType();

    auto selector = TuneParamsSelector(params);

    if (tune_params_dynamic) {
        selector.ref = tune_params_dynamic;
    }

    // if (tune_params_dynamic) {
    //     std::cout << "Load tuning params..." << std::endl;

    //     std::cout << "tparams.tile_b = " << tune_params_dynamic->tile_b << std::endl;
    //     std::cout << "tparams.tile_ofm = " << tune_params_dynamic->tile_ofm << std::endl;
    //     std::cout << "tparams.tile_ifm = " << tune_params_dynamic->tile_ifm << std::endl;
    //     std::cout << "tparams.tile_k = " << tune_params_dynamic->tile_k << std::endl;
    //     std::cout << "tparams.dispatch_bsv = " << tune_params_dynamic->dispatch_bsv << std::endl;
    //     std::cout << "tparams.dispatch_fsv = " << tune_params_dynamic->dispatch_fsv << std::endl;

    //     return *tune_params_dynamic;
    // }


    unsigned max_tile_ofm = 1;
    while (max_tile_ofm * 2 * simd <= output_f && max_tile_ofm < 4)
        max_tile_ofm *= 2;

    if (dtype == Datatype::F16) {
        // tune_params(tile_b, tile_ofm, tile_ifm, tile_k, dispatch_bsv, dispatch_fsv, exec_options)
        selector.Case(tune_params(8,  std::min(max_tile_ofm, 2u), 1, 2, 16, 2, EXE_MODE_AGE_BASED))
                .Case(tune_params(8,  std::min(max_tile_ofm, 2u), 1, 2, 16, 1, EXE_MODE_AGE_BASED))
                .Case(tune_params(16, std::min(max_tile_ofm, 2u), 1, 2, 4,  2, EXE_MODE_AGE_BASED))
                .Case(tune_params(8,  std::min(max_tile_ofm, 2u), 1, 2, 8,  1, EXE_MODE_AGE_BASED))
                .Case(tune_params(16, std::min(max_tile_ofm, 2u), 1, 2, 2,  2, EXE_MODE_AGE_BASED))
                .Case(tune_params(8,  std::min(max_tile_ofm, 2u), 1, 2, 4,  1, EXE_MODE_AGE_BASED))
                .Case(tune_params(16, std::min(max_tile_ofm, 2u), 1, 2, 1,  1, EXE_MODE_AGE_BASED))
                .Case(tune_params(8,  std::min(max_tile_ofm, 2u), 1, 2, 1,  1, EXE_MODE_AGE_BASED));
    }

    if (dtype == Datatype::F32) {
        // tune_params(tile_b, tile_ofm, tile_ifm, tile_k, dispatch_bsv, dispatch_fsv, exec_options)
        selector.Case(tune_params(8,  std::min(max_tile_ofm, 2u), 1, 1, 16, 2, EXE_MODE_AGE_BASED))
                .Case(tune_params(8,  std::min(max_tile_ofm, 2u), 1, 1, 16, 1, EXE_MODE_AGE_BASED))
                .Case(tune_params(8,  std::min(max_tile_ofm, 2u), 1, 1, 8,  1, EXE_MODE_AGE_BASED))
                .Case(tune_params(8,  std::min(max_tile_ofm, 2u), 1, 1, 4,  1, EXE_MODE_AGE_BASED))
                .Case(tune_params(8,  std::min(max_tile_ofm, 2u), 1, 1, 2,  1, EXE_MODE_AGE_BASED))
                .Case(tune_params(8,  std::min(max_tile_ofm, 2u), 1, 1, 1,  1, EXE_MODE_AGE_BASED));
    }

    selector.Case([&](const fully_connected_params&) -> tune_params {
        std::cout << "Auto params generation\n";
        tune_params result(8, std::min(max_tile_ofm, 2u), 1, 2, 1, 1, EXE_MODE_DEFAULT);

        while (batch % result.tile_b != 0)
            result.tile_b--;

        result.dispatch_bsv = 16;
        while (batch % (result.tile_b * result.dispatch_bsv) != 0)
            result.dispatch_bsv--;

        if (result.tile_b >= 8)
            result.exec_options = EXE_MODE_AGE_BASED;

        return result;
    });

    if (tune_params_dynamic) {
        selector.Case([&](const fully_connected_params&) -> tune_params {
            std::cout << "Auto params generation2\n";
            tune_params result(8, std::min(max_tile_ofm, 2u), 1, 1, 1, 1, EXE_MODE_DEFAULT);

            while (batch % result.tile_b != 0)
                result.tile_b--;

            result.dispatch_bsv = 16;
            while (batch % (result.tile_b * result.dispatch_bsv) != 0)
                result.dispatch_bsv--;

            if (result.tile_b >= 8)
                result.exec_options = EXE_MODE_AGE_BASED;

            return result;
        });
    }

    auto res_params = selector.Default(tune_params(1, 1, 1, 1, 1, 1, EXE_MODE_DEFAULT));
    if (params.has_dynamic_inputs() && !tune_params_dynamic) {
        std::cout << "Save tuning params\n" << std::endl;
        tune_params_dynamic = std::make_shared<tune_params>(res_params);
        selector.ref = tune_params_dynamic;
    }

    std::cout << "Tuning results:\n";

    std::cout << "tparams.tile_b = " << res_params.tile_b << std::endl;
    std::cout << "tparams.tile_ofm = " << res_params.tile_ofm << std::endl;
    std::cout << "tparams.tile_ifm = " << res_params.tile_ifm << std::endl;
    std::cout << "tparams.tile_k = " << res_params.tile_k << std::endl;
    std::cout << "tparams.dispatch_bsv = " << res_params.dispatch_bsv << std::endl;
    std::cout << "tparams.dispatch_fsv = " << res_params.dispatch_fsv << std::endl;

    return res_params;
}

FullyConnected_bf_tiled::DispatchData
FullyConnected_bf_tiled::SetDefault(const fully_connected_params& params, int autoTuneIndex, bool) const {
    auto dispatchData = Parent::SetDefault(params);
    auto tparams = GetAutoTuneParams(params, autoTuneIndex);

    size_t feature_threads = CeilDiv(params.outputs[0].Feature().v, tparams.tile_ofm * simd);
    size_t batch_threads = params.outputs[0].Batch().v;
    if (params.outputs[0].GetLayout() == DataLayout::bfyx) {
        feature_threads = CeilDiv(params.outputs[0].Y().v, tparams.tile_ofm * simd);
        batch_threads = params.outputs[0].Batch().v * params.outputs[0].Feature().v;
    }
    // Check if it is Ok for static path
    batch_threads = CeilDiv(batch_threads, tparams.tile_b);

    dispatchData.gws[0] = feature_threads * batch_threads * simd;
    dispatchData.gws[1] = 1;
    dispatchData.gws[2] = 1;

    dispatchData.lws[0] = simd;
    dispatchData.lws[1] = 1;
    dispatchData.lws[2] = 1;

    dispatchData.tile_m = tparams.tile_b;
    dispatchData.tile_n = tparams.tile_ofm;
    dispatchData.tile_mk = tparams.tile_ifm;
    dispatchData.tile_nk = tparams.tile_k;
    dispatchData.tile_ms = tparams.dispatch_bsv;
    dispatchData.tile_ns = tparams.dispatch_fsv;

    std::cout << "Update params for shape " << params.outputs[0].Batch().v << "x" << params.outputs[0].Feature().v << "x" << params.outputs[0].Y().v << "x" << params.outputs[0].X().v
              << " Input: " << params.inputs[0].Batch().v << "x" << params.inputs[0].Feature().v << "x" << params.inputs[0].Y().v << "x" << params.inputs[0].X().v << std::endl;
    std::cout << "GWS calc: feature_threads=" << feature_threads << " batch_threads=" << batch_threads << " - " << params.layerID << "(" << params.has_dynamic_tensors() << ")" << std::endl;
    std::cout << "GWS: " << dispatchData.gws[0] << "x" << dispatchData.gws[1] << "x" << dispatchData.gws[2] << std::endl;
    std::cout << "LWS: " << dispatchData.lws[0] << "x" << dispatchData.lws[1] << "x" << dispatchData.lws[2] << std::endl;

    return dispatchData;
}

void FullyConnected_bf_tiled::UpdateDynamicParams(const Params& params, KernelData& kd) const {
    const auto& prim_params = static_cast<const fully_connected_params&>(params);
    auto dispatchData = SetDefault(prim_params, -1, true);
    auto& kernel_params = kd.kernels[0].params;

    OPENVINO_ASSERT(kd.kernels.size() == 1, "[GPU] Invalid kernels size for update dispatch data func");
    kernel_params.workGroups.global = dispatchData.gws;
    kernel_params.workGroups.local = dispatchData.lws;

    OPENVINO_ASSERT(kernel_params.dynamic_params.size() == 3, "[GPU] Unexpected number of dynamic params for bf_tiled fc kernel");
    kernel_params.dynamic_params[0].v.u32 = dispatchData.tile_m;
    kernel_params.dynamic_params[1].v.u32 = dispatchData.tile_ms;
    kernel_params.dynamic_params[2].v.u32 = dispatchData.tile_ns;
}

KernelsPriority FullyConnected_bf_tiled::GetKernelsPriority(const Params& params, const optional_params& /*options*/) const {
    const auto& fc_params = static_cast<const fully_connected_params&>(params);

    size_t output_b = fc_params.outputs[0].Batch().v;
    if (fc_params.outputs[0].GetLayout() == DataLayout::bfyx)
        output_b *= fc_params.outputs[0].Feature().v;

    float estimated_time = FORCE_PRIORITY_9;
    if (output_b > 1 && fc_params.inputs[0].GetDType() == Datatype::F32)
        estimated_time = FORCE_PRIORITY_3;
    else if (output_b > 1 && fc_params.inputs[0].GetDType() == Datatype::F16)
        estimated_time = FORCE_PRIORITY_4;

    return estimated_time;
}

JitConstants FullyConnected_bf_tiled::GetJitConstants(const fully_connected_params& params, const DispatchData& dispatchData) const {
    JitConstants jit = Parent::GetJitConstants(params, dispatchData);

    jit.AddConstant(MakeJitConstant("SIMD", simd));
    jit.AddConstant(MakeJitConstant("TILE_B", dispatchData.tile_m));
    jit.AddConstant(MakeJitConstant("TILE_OFM", dispatchData.tile_n));
    jit.AddConstant(MakeJitConstant("TILE_IFM", dispatchData.tile_mk));
    jit.AddConstant(MakeJitConstant("TILE_K", dispatchData.tile_nk));
    jit.AddConstant(MakeJitConstant("TILE_K_OFM", dispatchData.tile_nk * dispatchData.tile_n));

    if (!params.has_dynamic_tensors()) {
        jit.AddConstant(MakeJitConstant("DISPATCH_BSV", dispatchData.tile_ms));
        jit.AddConstant(MakeJitConstant("DISPATCH_FSV", dispatchData.tile_ns));
    } else {
        std::cout << "Dynamic kernel!\n";
        std::vector<std::pair<std::string, ScalarDescriptor::Types>> dynamic_params = {
            {"TILE_B1", ScalarDescriptor::Types::UINT32},
            {"DISPATCH_BSV1", ScalarDescriptor::Types::UINT32},
            {"DISPATCH_FSV1", ScalarDescriptor::Types::UINT32}};
        jit.Merge(MakeDynamicParamsJitConstants(dynamic_params));
    }

    jit.Merge(MakeConstantLoopUnrollJitConstants(dispatchData.tile_m));

    bool realign_fp16_offset = params.inputs[0].GetDType() == Datatype::F16 && params.inputs[0].GetFirstElementOffset() % 2 != 0;
    jit.AddConstant(MakeJitConstant("REALIGN_FP16_OFFSET", realign_fp16_offset));

    auto activation_dt = GetActivationType(params);
    auto accumulator_dt = GetAccumulatorType(params);
    jit.Merge(MakeTypeJitConstants(activation_dt, "ACTIVATION"));
    jit.Merge(MakeActivationJitConstants(params.activations, activation_dt, "_TYPED"));
    jit.Merge(MakeTypeJitConstants(accumulator_dt, "ACCUMULATOR"));

    // for 3d output we are treating spatial as features
    if (params.outputs[0].GetLayout() == DataLayout::bfyx) {
        jit.AddConstant(MakeJitConstant("TILE_OUT_F_NUM", params.outputs[0].Y().v));
        jit.AddConstant(MakeJitConstant("TILE_OUT_F_PITCH", params.outputs[0].Y().pitch));
        jit.AddConstant(MakeJitConstant("TILE_IN_B_PITCH", params.inputs[0].Feature().pitch));
        jit.AddConstant(MakeJitConstant("TILE_OUT_B_PITCH", params.outputs[0].Feature().pitch));
        jit.AddConstant(MakeJitConstant("OUTPUT_3D", true));
        jit.AddConstant(MakeJitConstant("BATCH_SIZE", "(OUTPUT_BATCH_NUM * OUTPUT_FEATURE_NUM)"));
    } else {
        jit.AddConstant(MakeJitConstant("TILE_OUT_F_NUM", params.outputs[0].Feature().v));
        jit.AddConstant(MakeJitConstant("TILE_OUT_F_PITCH", params.outputs[0].Feature().pitch));
        jit.AddConstant(MakeJitConstant("TILE_IN_B_PITCH", params.inputs[0].Batch().pitch));
        jit.AddConstant(MakeJitConstant("TILE_OUT_B_PITCH", params.outputs[0].Batch().pitch));
        jit.AddConstant(MakeJitConstant("BATCH_SIZE", "(OUTPUT_BATCH_NUM)"));
    }

    if (!params.fused_ops.empty()) {
        std::vector<std::string> idx_order_scalar = { "(out_b + bi)", "(out_f + sglid)", "0", "0" };
        std::vector<std::string> idx_order_vec = { "(out_b + bi)", "(out_f + fi + sglid)", "0", "0" };
        if (params.outputs[0].GetLayout() == DataLayout::bfyx) {
            idx_order_scalar = { "(out_b + bi) / OUTPUT_FEATURE_NUM", "(out_b + bi) % OUTPUT_FEATURE_NUM", "sglid", "0" };
            idx_order_vec = { "(out_b + bi) / OUTPUT_FEATURE_NUM", "(out_b + bi) % OUTPUT_FEATURE_NUM", "sglid", "0" };
        }

        // Simplify fused ops configuration to prevent mixed layout exception in jitter
        // for common cases with bfyx -> bf layouts and eltwise fusing (such scenarios currently don't work for vectors)
        FusedOpsConfiguration conf_scalar = { "_SCALAR",
                                              idx_order_scalar,
                                              "activated[bi]",
                                              activation_dt,
                                              1 };
        FusedOpsConfiguration conf_vec = { "_VEC",
                                           idx_order_vec,
                                           "activated[bi][fi]",
                                           activation_dt,
                                           1 };
        jit.Merge(MakeFusedOpsJitConstants(params, { conf_scalar, conf_vec }));
    }

    return jit;
}

KernelsData FullyConnected_bf_tiled::GetTunedKernelsDataByIndex(const Params &params,
                                                                const optional_params &options,
                                                                const int autoTuneIndex) const {
    auto& fc_params = static_cast<const fully_connected_params&>(params);

    if (autoTuneIndex >= 0 && autoTuneIndex < static_cast<int>(auto_tune_params.size())
        && !TuneParamsSelector::VerifyTuneParams(fc_params, auto_tune_params[autoTuneIndex]))
        return {};

    tune_params tparams = GetAutoTuneParams(fc_params, autoTuneIndex);

    WeightsLayout weights_layout = WeightsLayout::os_iyx_osv16;
    if (tparams.tile_ofm * simd == 32)
        weights_layout = WeightsLayout::os_iyx_osv32;
    else if (tparams.tile_ofm * simd == 64)
        weights_layout = WeightsLayout::os_iyx_osv64;

    auto kernelsData = GetCommonKernelsData(params,
                                options,
                                fc_params.inputs[0].GetLayout(),
                                weights_layout,
                                tparams.exec_options,
                                autoTuneIndex);

    if (fc_params.has_dynamic_inputs() && kernelsData.size()) {
        auto& kernel_params = kernelsData[0].kernels[0].params;
        kernel_params.dynamic_params.emplace_back(ScalarDescriptor::Types::UINT32);
        kernel_params.dynamic_params.emplace_back(ScalarDescriptor::Types::UINT32);
        kernel_params.dynamic_params.emplace_back(ScalarDescriptor::Types::UINT32);

        kernel_params.arguments.push_back({ArgumentDescriptor::Types::DYNAMIC_PARAMS, 0});
    }

    return kernelsData;
}

KernelsData FullyConnected_bf_tiled::GetKernelsDataForAutoTune(const Params& params, const optional_params& options) const {
    KernelsData res = {};
    for (size_t idx = 0; idx < auto_tune_params.size(); ++idx) {
        KernelsData kds = GetTunedKernelsDataByIndex(params, options, static_cast<int>(idx));

        if (!kds.empty()) {
            res.emplace_back(kds[0]);
        }
    }

    return res;
}

KernelsData FullyConnected_bf_tiled::GetKernelsData(const Params& params, const optional_params& optParams) const {
    KernelsData res = {};
    auto& fc_params = static_cast<const fully_connected_params&>(params);
    auto tparams = GetAutoTuneParams(fc_params);

    KernelsData kds = GetTunedKernelsDataByIndex(params, optParams, -1);
    if (!kds.empty()) {
        res.emplace_back(kds[0]);
    }

    return res;
}

}  // namespace kernel_selector
