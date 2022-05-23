// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "ngraph/op/matmul.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/fake_quantize.hpp"

#include "intel_gpu/primitives/gemm.hpp"
#include "intel_gpu/primitives/fully_connected.hpp"
#include "intel_gpu/primitives/reshape.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include "intel_gpu/primitives/permute.hpp"

namespace ov {
namespace runtime {
namespace intel_gpu {

/*
*  get_aligned_shapes function align two input shapes to have the same size and
*  the same batch dimensions (last two dimensions are not comparable).
*  It also checks that dimensions are compatible so in case with two shapes
*  for example: [2, 32, 64] [3, 64, 64] it will raise an exception.
*/

static std::tuple<bool, ov::PartialShape, ov::PartialShape> get_aligned_shapes(const ov::PartialShape& shape_a,
                                                                               const ov::PartialShape& shape_b,
                                                                               const std::shared_ptr<ngraph::op::v0::MatMul>& matmul) {
    auto rank_a = shape_a.rank().get_length();
    auto rank_b = shape_b.rank().get_length();
    ov::PartialShape shape_a_aligned(shape_a), shape_b_aligned(shape_b);
    size_t max_size = std::max(rank_a, rank_b);
    for (size_t i = 0, cnt = max_size - rank_a; i < cnt; ++i) {
        shape_a_aligned.insert(shape_a_aligned.begin(), 1);
    }
    for (size_t i = 0, cnt = max_size - rank_b; i < cnt; ++i) {
        shape_b_aligned.insert(shape_b_aligned.begin(), 1);
    }

    if (matmul->get_transpose_a() && rank_a != 1) {
        std::swap(*(shape_a_aligned.end() - 1), *(shape_a_aligned.end() - 2));
    }
    if (matmul->get_transpose_b()) {
        std::swap(*(shape_b_aligned.end() - 1), *(shape_b_aligned.end() - 2));
    }

    for (size_t i = 0; i < max_size - 2; ++i) {
        auto a_dim = shape_a_aligned[i], b_dim = shape_b_aligned[i];
        if (a_dim.is_dynamic()) {
            if (b_dim == 1) {
                shape_a_aligned[i] = shape_b_aligned[i] = a_dim;
            } else {
                return std::make_tuple(false, ngraph::PartialShape{shape_a_aligned}, ngraph::PartialShape{shape_b_aligned});
            }
            continue;
        }
        // both dimensions are static
        if (a_dim != b_dim && a_dim.get_length() > 1 && b_dim.get_length() > 1) {
            std::ostringstream stream;
            stream << "Shapes can't be aligned: " << shape_a_aligned << " " << shape_b_aligned;
            throw ngraph::ngraph_error(stream.str());
        }
        size_t max_value = std::max(a_dim.get_length(), b_dim.get_length());
        shape_a_aligned[i] = shape_b_aligned[i] = max_value;
    }
    return std::make_tuple(true, shape_a_aligned, shape_b_aligned);
}

static void CreateMatMulOp(Program& p, const std::shared_ptr<ngraph::op::v0::MatMul>& op) {
    p.ValidateInputs(op, {2});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    auto shape_a = op->get_input_partial_shape(0);
    auto shape_b = op->get_input_partial_shape(1);

    bool is_fc = IsNodeOnConstPath(op->get_input_node_shared_ptr(1));
    is_fc &= std::count_if(shape_b.begin(), shape_b.end(), [](ov::Dimension x) { return x != 1; }) <= 2;
    // TODO: This conditions can be relaxed with proper handling in FC path
    is_fc &= shape_b.size() > 1 && shape_a.size() > 1 && shape_b.is_static();

    do {
        if (!is_fc)
            break;
        ov::PartialShape shape_a_aligned, shape_b_aligned;
        bool success;
        std::tie(success, shape_a_aligned, shape_b_aligned) = get_aligned_shapes(shape_a, shape_b, op);
        if (shape_a_aligned.size() < 2 || shape_b_aligned.size() < 2) {
            IE_THROW() << "MatMul " << op->get_friendly_name() << " shapes are inconsistent.";
        }
        if (!success)
            break;

        if (shape_a_aligned.size() > 3 && shape_a.is_dynamic())
            break;
        if (shape_b.size() != 2 && shape_b.is_dynamic())
            break;

        ov::Dimension K = *(shape_a_aligned.end() - 1);

        auto inputName = inputPrimitives[0];
        auto weightsName = inputPrimitives[1];

        // Weights normalization
        if (!op->get_transpose_b()) {
            std::vector<uint16_t> transpose_order(shape_b.size());
            std::iota(transpose_order.begin(), transpose_order.end(), 0);
            std::swap(*(transpose_order.end() - 1), *(transpose_order.end() - 2));

            auto permuteName = op->get_friendly_name() + "/transpose_b";
            auto permutePrim = cldnn::permute(permuteName,
                                              weightsName,
                                              transpose_order,
                                              op->get_friendly_name());
            p.AddPrimitive(permutePrim);
            p.AddInnerPrimitiveToProfiler(permuteName, layerName, op);
            weightsName = permuteName;
        }

        // Input normalization
        if (op->get_transpose_a()) {
            std::vector<uint16_t> transpose_order(shape_a.size());
            std::iota(transpose_order.begin(), transpose_order.end(), 0);
            std::swap(*(transpose_order.end() - 1), *(transpose_order.end() - 2));

            auto permuteName = op->get_friendly_name() + "/transpose_a";
            auto permutePrim = cldnn::permute(permuteName,
                                              inputName,
                                              transpose_order,
                                              op->get_friendly_name());
            p.AddPrimitive(permutePrim);
            p.AddInnerPrimitiveToProfiler(permuteName, layerName, op);
            inputName = permuteName;
        }

        bool reshape_fc = shape_a_aligned.size() > 3;

        auto reshape_to_2d = [&](const ov::PartialShape& shape, std::string inputName, size_t features, std::string suffix) -> std::string {
            auto total = std::accumulate(shape.begin(), shape.end(), 1, [](int b, const ov::Dimension& a){ return b * a.get_length(); });
            ov::PartialShape reshapeSize = { total / features, features, 1, 1 };

            if (total != reshapeSize[0].get_length() * reshapeSize[1].get_length())
                IE_THROW() << "Inconsistent reshape in Matmul op: " << op->get_friendly_name();

            auto reshapeInName = op->get_friendly_name() + suffix;
            auto reshapeInPrim = cldnn::reshape(reshapeInName,
                                                inputName,
                                                reshapeSize,
                                                op->get_friendly_name());
            p.AddPrimitive(reshapeInPrim);
            p.AddInnerPrimitiveToProfiler(reshapeInName, layerName, op);
            return reshapeInName;
        };

        if (reshape_fc) {
            inputName = reshape_to_2d(shape_a, inputName, (shape_a.end() - 1)->get_length(), "_cldnn_reshape_in");
        }

        if (shape_b.size() != 2) {
            weightsName = reshape_to_2d(shape_b, weightsName, K.get_length(), "_cldnn_reshape_weights");
        }

        auto input_rank = reshape_fc ? 2 : shape_a.size();
        auto fcPrim = cldnn::fully_connected(layerName,
                                             inputName,
                                             weightsName,
                                             "",
                                             DataTypeFromPrecision(op->get_output_element_type(0)),
                                             op->get_friendly_name(),
                                             cldnn::padding(),
                                             input_rank);

        p.AddPrimitive(fcPrim);

        auto lastLayerName = layerName;
        if (reshape_fc) {
            auto outReshapeName = layerName + "_cldnn_out_reshape";
            auto outReshapePrim = cldnn::reshape(outReshapeName, layerName, op->get_output_partial_shape(0), op->get_friendly_name());

            p.AddPrimitive(outReshapePrim);
            p.AddInnerPrimitiveToProfiler(outReshapeName, layerName, op);

            lastLayerName = outReshapeName;
        }

        p.AddPrimitiveToProfiler(op, lastLayerName);
        return;
    }  while (false);

    // auto output_pshape = op->get_output_partial_shape(0);
    // auto output_rank = output_pshape.rank().get_length();

    // // Preprocess inputs
    // for (size_t i = 0; i < inputPrimitives.size(); ++i) {
    //     auto input_pshape = op->get_input_partial_shape(i);
    //     auto input_rank = input_pshape.rank().get_length();

    //     // Add reorder if changing number of dimensions requires changing format
    //     auto target_format = cldnn::format::get_default_format(output_rank);

    //     if (target_format.value != cldnn::format::get_default_format(input_rank).value) {
    //         auto reorderName = layerName + "_cldnn_in" + std::to_string(i) + "_reorder";
    //         auto targetDatatype = DataTypeFromPrecision(op->get_output_element_type(0));
    //         auto reorderPrim = cldnn::reorder(reorderName,
    //                                           inputPrimitives[i],
    //                                           target_format,
    //                                           targetDatatype,
    //                                           std::vector<float>(),
    //                                           cldnn::reorder_mean_mode::subtract,
    //                                           op->get_friendly_name());

    //         p.AddPrimitive(reorderPrim);
    //         p.AddInnerPrimitiveToProfiler(reorderName, layerName, op);

    //         inputPrimitives[i] = reorderName;
    //     }

    //     // Reshape input if they differ or gemm specific shape matches default one
    //     if (input_rank != output_rank || input_rank < 4) {
    //         auto reshapeName = layerName + "_cldnn_in" + std::to_string(i) + "_reshape";

    //         // Extend input dimensions by prepending ones
    //         if (input_rank == 1) {
    //             // One-dimensional tensors unsqueezing is applied for each input independently.
    //             // The axes inserted in this step are not included in the output shape.
    //             // * If rank of the **first** input is equal to 1, it is always unsqueezed to 2D tensor **row vector** (regardless of `transpose_a`)
    //             // by adding axes with size 1 at ROW_INDEX_DIM, to the **left** of the shape. For example `[S]` will be reshaped to `[1, S]`.
    //             // * If rank of the **second** input is equal to 1, it is always unsqueezed to 2D tensor **column vector** (regardless of `transpose_b`)
    //             // by adding axes with size 1 at COL_INDEX_DIM, to the **right** of the shape. For example `[S]` will be reshaped to `[S, 1]`.
    //             bool transpose = false;
    //             if (i == 0) {
    //                 transpose = op->get_transpose_a();
    //                 input_pshape.insert(input_pshape.begin(), 1);
    //             } else {
    //                 transpose = op->get_transpose_b();
    //                 input_pshape.insert(input_pshape.end(), 1);
    //             }
    //             // Specs says that shapes must be unsqueezed regardless of tranpose flag, but primitive implementation always respects transposes
    //             // so we have to swap dimensions correspondingly to have consistent shapes.
    //             if (transpose) {
    //                 std::swap(input_pshape[0], input_pshape[1]);
    //             }
    //         }
    //         if (input_rank < output_rank)
    //             input_pshape.insert(input_pshape.begin(), output_rank - input_rank, 1ul);

    //         auto reshapePrim = cldnn::reshape(reshapeName, inputPrimitives[i], input_pshape, op->get_friendly_name());

    //         p.AddPrimitive(reshapePrim);
    //         p.AddInnerPrimitiveToProfiler(reshapeName, layerName, op);

    //         inputPrimitives[i] = reshapeName;
    //     }
    // }

    // Add actual gemm
    auto alpha = 1.0f;
    auto beta = 0.0f;
    auto transA = op->get_transpose_a();
    auto transB = op->get_transpose_b();

    auto gemmPrim = cldnn::gemm(layerName,
                                inputPrimitives,
                                DataTypeFromPrecision(op->get_output_element_type(0)),
                                transA,
                                transB,
                                alpha,
                                beta,
                                op->get_friendly_name());

    p.AddPrimitive(gemmPrim);

    auto lastLayerName = layerName;

    // Reshape output if gemm specific shape does not match default one
    // if (output_rank < 4) {
    //     auto outReshapeName = layerName + "_cldnn_out_reshape";
    //     auto outReshapePrim = cldnn::reshape(outReshapeName, layerName, output_pshape, op->get_friendly_name());

    //     p.AddPrimitive(outReshapePrim);
    //     p.AddInnerPrimitiveToProfiler(outReshapeName, layerName, op);

    //     lastLayerName = outReshapeName;
    // }

    p.AddPrimitiveToProfiler(op, lastLayerName);
}

REGISTER_FACTORY_IMPL(v0, MatMul);

}  // namespace intel_gpu
}  // namespace runtime
}  // namespace ov
