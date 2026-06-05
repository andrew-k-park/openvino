// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <openvino/core/model.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/clamp.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/gelu.hpp>
#include <openvino/op/matmul.hpp>
#include <openvino/op/minimum.hpp>
#include <openvino/op/moe.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/reduce_sum.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/slice.hpp>
#include <openvino/op/swish.hpp>
#include <openvino/op/tile.hpp>
#include <openvino/op/topk.hpp>
#include <openvino/pass/manager.hpp>
#include <openvino/pass/serialize.hpp>
#include <openvino/pass/visualize_tree.hpp>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "ov_ops/gather_matmul.hpp"
#include "ov_ops/gather_matmul_compressed.hpp"
#include "ov_ops/moe_compressed.hpp"
#include "transformations/common_optimizations/convert_tiled_moe_block_to_gather_matmuls.hpp"
#include "transformations/common_optimizations/moe_op_fusion.hpp"

using GatherMatmul = ov::op::internal::GatherMatmul;
using GatherMatmulCompressed = ov::op::internal::GatherMatmulCompressed;

// ============================================================================
// IR model builders (original MOE pattern before any transformation)
// ============================================================================

inline std::shared_ptr<ov::Model> build_2gemm_moe_pattern_model() {
    using namespace ov;

    const size_t batch = 2;
    const Dimension in_dim = Dimension::dynamic();
    const size_t hidden_size = 2048;
    const size_t intermediate_size = 4096;
    const size_t topk = 2;
    const size_t number_of_experts = 3;
    const size_t fusion_factor = 2;
    const auto expert_alpha = 1.702f;
    const auto expert_beta = 7.0f;

    auto input_shape = PartialShape{batch, in_dim, hidden_size};
    auto input = std::make_shared<op::v0::Parameter>(element::f32, input_shape);
    auto experts_reshape = std::make_shared<op::v1::Reshape>(
        input,
        op::v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{-1, hidden_size}),
        false);

    auto tile = std::make_shared<op::v0::Tile>(
        experts_reshape,
        op::v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{number_of_experts, 1}));
    auto after_tile_reshape = std::make_shared<op::v1::Reshape>(
        tile,
        op::v0::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{number_of_experts, batch, hidden_size}),
        false);

    auto gate_up_matmul = std::make_shared<op::v0::MatMul>(
        after_tile_reshape,
        op::v0::Constant::create(element::f32,
                                 Shape{number_of_experts, intermediate_size * fusion_factor, hidden_size},
                                 {1.0f}),
        false,
        true);
    auto gate_up_add = std::make_shared<op::v1::Add>(
        gate_up_matmul,
        op::v0::Constant::create(element::f32, Shape{number_of_experts, 1, intermediate_size * fusion_factor}, {0.0f}));

    auto slice1 = std::make_shared<op::v8::Slice>(
        gate_up_add,
        op::v0::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{0, 0, 0}),
        op::v0::Constant::create(element::i64,
                                 Shape{3},
                                 std::vector<int64_t>{number_of_experts, batch, intermediate_size * 2}),
        op::v0::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{1, 1, 2}),
        op::v0::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{0, 1, 2}));
    auto clamp = std::make_shared<op::v0::Clamp>(slice1, -expert_beta, expert_beta);
    auto add1 = std::make_shared<op::v1::Add>(clamp, op::v0::Constant::create(element::f32, Shape{1}, {1.0f}));

    auto slice2 = std::make_shared<op::v8::Slice>(
        gate_up_add,
        op::v0::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{0, 1, 0}),
        op::v0::Constant::create(element::i64,
                                 Shape{3},
                                 std::vector<int64_t>{number_of_experts, batch, intermediate_size * 2}),
        op::v0::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{1, 1, 2}),
        op::v0::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{0, 1, 2}));
    auto minimum1 =
        std::make_shared<op::v1::Minimum>(slice2, op::v0::Constant::create(element::f32, Shape{1}, {10.0f}));
    auto swish_beta = op::v0::Constant::create(element::f32, Shape{}, std::vector<float>{expert_alpha});
    auto swish = std::make_shared<op::v4::Swish>(minimum1, swish_beta);

    auto multiply2 = std::make_shared<op::v1::Multiply>(add1, swish);

    auto down_proj_matmul = std::make_shared<op::v0::MatMul>(
        multiply2,
        op::v0::Constant::create(element::f32, Shape{number_of_experts, hidden_size, intermediate_size}, {1.0f}),
        false,
        true);

    auto down_proj_add = std::make_shared<op::v1::Add>(
        down_proj_matmul,
        op::v0::Constant::create(element::f32, Shape{number_of_experts, 1, hidden_size}, {1.0f}));

    auto end_reshape = std::make_shared<op::v1::Reshape>(
        down_proj_add,
        op::v0::Constant::create(element::i64,
                                 Shape{4},
                                 std::vector<int64_t>{number_of_experts, batch, -1, hidden_size}),
        false);

    // Router subgraph
    auto reshape_2nd_consumer_router_matmul = std::make_shared<op::v0::MatMul>(
        experts_reshape,
        op::v0::Constant::create(element::f32, Shape{number_of_experts, hidden_size}, {1.0f}),
        false,
        true);

    auto router_bias =
        std::make_shared<op::v1::Add>(reshape_2nd_consumer_router_matmul,
                                      op::v0::Constant::create(element::f32, Shape{1, number_of_experts}, {1.0f}));

    auto router_topk_values_and_indices =
        std::make_shared<op::v11::TopK>(router_bias,
                                        op::v0::Constant::create(element::i64, Shape{}, {topk}),
                                        -1,
                                        op::v11::TopK::Mode::MAX,
                                        op::v11::TopK::SortType::SORT_VALUES,
                                        element::i64);

    auto router_topk_values = router_topk_values_and_indices->output(0);
    auto router_topk_indices = router_topk_values_and_indices->output(1);

    auto scatter_elements_update = std::make_shared<op::v12::ScatterElementsUpdate>(
        router_topk_values,
        router_topk_indices,
        op::v0::Constant::create(element::f32, Shape{batch, topk}, {0}),
        op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{1}));
    auto router_transpose = std::make_shared<op::v1::Transpose>(
        scatter_elements_update,
        op::v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{1, 0}));
    auto router_reshape = std::make_shared<op::v1::Reshape>(
        router_transpose,
        op::v0::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{number_of_experts, batch, -1}),
        true);
    auto unsqueeze_routing_weights =
        std::make_shared<op::v0::Unsqueeze>(router_reshape,
                                            op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{-1}));

    auto mul3 = std::make_shared<op::v1::Multiply>(end_reshape, unsqueeze_routing_weights);

    auto reduce_sum =
        std::make_shared<op::v1::ReduceSum>(mul3,
                                            op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{0}),
                                            false);

    return std::make_shared<ov::Model>(ov::OutputVector{reduce_sum}, ov::ParameterVector{input});
}

inline std::shared_ptr<ov::Model> build_3gemm_moe_pattern_model() {
    using namespace ov;

    const size_t batch = 2;
    const Dimension in_dim = Dimension::dynamic();
    const size_t hidden_size = 2048;
    const size_t intermediate_size = 4096;
    const size_t number_of_experts = 3;
    const size_t topk = 2;

    auto input_shape = PartialShape{batch, in_dim, hidden_size};
    auto input = std::make_shared<op::v0::Parameter>(element::f32, input_shape);
    auto experts_reshape = std::make_shared<op::v1::Reshape>(
        input,
        op::v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{-1, hidden_size}),
        false);

    auto tile = std::make_shared<op::v0::Tile>(
        experts_reshape,
        op::v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{number_of_experts, 1}));
    auto after_tile_reshape = std::make_shared<op::v1::Reshape>(
        tile,
        op::v0::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{number_of_experts, batch, hidden_size}),
        false);

    // First GEMM (gate)
    auto gate_matmul = std::make_shared<op::v0::MatMul>(
        after_tile_reshape,
        op::v0::Constant::create(element::f32, Shape{number_of_experts, intermediate_size, hidden_size}, {1.0f}),
        false,
        true);

    auto swish = std::make_shared<op::v4::Swish>(gate_matmul);

    // Second GEMM (up)
    auto up_matmul = std::make_shared<op::v0::MatMul>(
        after_tile_reshape,
        op::v0::Constant::create(element::f32, Shape{number_of_experts, intermediate_size, hidden_size}, {1.0f}),
        false,
        true);

    auto swiglu = std::make_shared<op::v1::Multiply>(swish, up_matmul);

    // Third GEMM (down)
    auto down_matmul = std::make_shared<op::v0::MatMul>(
        swiglu,
        op::v0::Constant::create(element::f32, Shape{number_of_experts, hidden_size, intermediate_size}, {1.0f}),
        false,
        true);

    auto experts_out_reshape = std::make_shared<op::v1::Reshape>(
        down_matmul,
        op::v0::Constant::create(element::i64,
                                 Shape{4},
                                 std::vector<int64_t>{number_of_experts, batch, -1, hidden_size}),
        false);

    // Router subgraph
    auto router_matmul = std::make_shared<op::v0::MatMul>(
        experts_reshape,
        op::v0::Constant::create(element::f32, Shape{number_of_experts, hidden_size}, {1.0f}),
        false,
        true);

    auto router_topk_values_and_indices =
        std::make_shared<op::v11::TopK>(router_matmul,
                                        op::v0::Constant::create(element::i64, Shape{}, {topk}),
                                        -1,
                                        op::v11::TopK::Mode::MAX,
                                        op::v11::TopK::SortType::SORT_VALUES,
                                        element::i64);

    auto router_topk_values = router_topk_values_and_indices->output(0);
    auto router_topk_indices = router_topk_values_and_indices->output(1);

    auto scatter_elements_update = std::make_shared<op::v12::ScatterElementsUpdate>(
        router_topk_values,
        router_topk_indices,
        op::v0::Constant::create(element::f32, Shape{batch, topk}, {0}),
        op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{1}));
    auto router_transpose = std::make_shared<op::v1::Transpose>(
        scatter_elements_update,
        op::v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{1, 0}));
    auto router_reshape = std::make_shared<op::v1::Reshape>(
        router_transpose,
        op::v0::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{number_of_experts, batch, -1}),
        true);
    auto unsqueeze_routing_weights =
        std::make_shared<op::v0::Unsqueeze>(router_reshape,
                                            op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{-1}));

    auto mul3 = std::make_shared<op::v1::Multiply>(experts_out_reshape, unsqueeze_routing_weights);

    auto reduce_sum =
        std::make_shared<op::v1::ReduceSum>(mul3,
                                            op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{0}),
                                            false);

    return std::make_shared<ov::Model>(ov::OutputVector{reduce_sum}, ov::ParameterVector{input});
}

// ============================================================================
// Post-BGM model builders (3 BGMs + compact routing + ReduceSum + Reshape)
// ============================================================================

inline std::shared_ptr<ov::Model> build_3gemm_bgm_model(
    ov::op::internal::MOE::Activation_type activation_type = ov::op::internal::MOE::Activation_type::SWIGLU) {
    using namespace ov;

    const size_t batch = 2;
    const Dimension in_dim = Dimension::dynamic();
    const size_t hidden_size = 2048;
    const size_t intermediate_size = 4096;
    const size_t number_of_experts = 3;
    const size_t topk = 2;

    auto input = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{batch, in_dim, hidden_size});
    auto experts_reshape = std::make_shared<op::v1::Reshape>(
        input,
        op::v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{-1, static_cast<int64_t>(hidden_size)}),
        false);

    // Unsqueeze to add experts dimension: [1, batch*seq, hidden]
    auto unsqueeze =
        std::make_shared<op::v0::Unsqueeze>(experts_reshape, op::v0::Constant::create(element::i32, Shape{}, {0}));

    // Router subgraph to produce topk_indices and chosen_experts
    auto router_matmul = std::make_shared<op::v0::MatMul>(
        experts_reshape,
        op::v0::Constant::create(element::f32, Shape{number_of_experts, hidden_size}, {1.0f}),
        false,
        true);
    auto router_topk = std::make_shared<op::v11::TopK>(router_matmul,
                                                       op::v0::Constant::create(element::i64, Shape{}, {topk}),
                                                       -1,
                                                       op::v11::TopK::Mode::MAX,
                                                       op::v11::TopK::SortType::SORT_VALUES,
                                                       element::i64);
    auto topk_indices = router_topk->output(1);    // [batch*seq, topk]
    auto chosen_experts = router_topk->output(0);  // [batch*seq, topk] (values used as routing weights)

    // Gate weights
    auto gate_w =
        op::v0::Constant::create(element::f32, Shape{number_of_experts, intermediate_size, hidden_size}, {1.0f});
    // Up weights
    auto up_w =
        op::v0::Constant::create(element::f32, Shape{number_of_experts, intermediate_size, hidden_size}, {1.0f});
    // Down weights
    auto down_w =
        op::v0::Constant::create(element::f32, Shape{number_of_experts, hidden_size, intermediate_size}, {1.0f});

    // 3 BGMs
    auto bgm_gate = std::make_shared<GatherMatmul>(unsqueeze, gate_w, topk_indices);
    std::shared_ptr<ov::Node> gate_act;
    if (activation_type == ov::op::internal::MOE::Activation_type::GEGLU_TANH) {
        gate_act = std::make_shared<op::v7::Gelu>(bgm_gate, ov::op::GeluApproximationMode::TANH);
    } else if (activation_type == ov::op::internal::MOE::Activation_type::GEGLU_ERF) {
        gate_act = std::make_shared<op::v7::Gelu>(bgm_gate, ov::op::GeluApproximationMode::ERF);
    } else {
        gate_act = std::make_shared<op::v4::Swish>(bgm_gate);
    }
    auto bgm_up = std::make_shared<GatherMatmul>(unsqueeze, up_w, topk_indices);
    auto swiglu = std::make_shared<op::v1::Multiply>(gate_act, bgm_up);
    auto bgm_down = std::make_shared<GatherMatmul>(swiglu, down_w, topk_indices);

    // Compact routing: chosen_experts → Transpose({1,0}) → Unsqueeze(-1)
    auto router_transpose = std::make_shared<op::v1::Transpose>(
        chosen_experts,
        op::v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{1, 0}));
    auto router_unsqueeze =
        std::make_shared<op::v0::Unsqueeze>(router_transpose, op::v0::Constant::create(element::i32, Shape{}, {-1}));

    // Final: Multiply → ReduceSum → Reshape
    auto final_mul = std::make_shared<op::v1::Multiply>(bgm_down, router_unsqueeze);
    auto reduce_sum =
        std::make_shared<op::v1::ReduceSum>(final_mul,
                                            op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{0}),
                                            false);

    auto end_reshape = std::make_shared<op::v1::Reshape>(
        reduce_sum,
        op::v0::Constant::create(
            element::i64,
            Shape{3},
            std::vector<int64_t>{static_cast<int64_t>(batch), -1, static_cast<int64_t>(hidden_size)}),
        true);

    return std::make_shared<ov::Model>(ov::OutputVector{end_reshape}, ov::ParameterVector{input});
}

inline std::shared_ptr<ov::Model> build_3gemm_bgm_to_moe_reference_model(
    ov::op::internal::MOE::Activation_type activation_type = ov::op::internal::MOE::Activation_type::SWIGLU) {
    using namespace ov;

    const size_t batch = 2;
    const Dimension in_dim = Dimension::dynamic();
    const size_t hidden_size = 2048;
    const size_t intermediate_size = 4096;
    const size_t number_of_experts = 3;
    const size_t topk = 2;

    auto input = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{batch, in_dim, hidden_size});

    // Router subgraph (not fused, remains in the graph)
    auto experts_reshape = std::make_shared<op::v1::Reshape>(
        input,
        op::v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{-1, static_cast<int64_t>(hidden_size)}),
        false);

    auto router_matmul = std::make_shared<op::v0::MatMul>(
        experts_reshape,
        op::v0::Constant::create(element::f32, Shape{number_of_experts, hidden_size}, {1.0f}),
        false,
        true);
    auto router_topk = std::make_shared<op::v11::TopK>(router_matmul,
                                                       op::v0::Constant::create(element::i64, Shape{}, {topk}),
                                                       -1,
                                                       op::v11::TopK::Mode::MAX,
                                                       op::v11::TopK::SortType::SORT_VALUES,
                                                       element::i64);
    auto topk_indices = router_topk->output(1);
    auto chosen_experts = router_topk->output(0);

    // Compact routing (stays as-is, becomes MOE input 1)
    auto router_transpose = std::make_shared<op::v1::Transpose>(
        chosen_experts,
        op::v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{1, 0}));
    auto router_unsqueeze =
        std::make_shared<op::v0::Unsqueeze>(router_transpose, op::v0::Constant::create(element::i32, Shape{}, {-1}));

    // Weights
    auto gate_w =
        op::v0::Constant::create(element::f32, Shape{number_of_experts, intermediate_size, hidden_size}, {1.0f});
    auto up_w =
        op::v0::Constant::create(element::f32, Shape{number_of_experts, intermediate_size, hidden_size}, {1.0f});
    auto down_w =
        op::v0::Constant::create(element::f32, Shape{number_of_experts, hidden_size, intermediate_size}, {1.0f});

    // MOE op with compact routing
    ov::OutputVector moe_inputs = {input, router_unsqueeze, topk_indices, gate_w, up_w, down_w};
    ov::op::internal::MOE::Config config;
    config.expert_type = ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU;
    config.activation_type = activation_type;
    auto moe = std::make_shared<ov::op::internal::MOE>(moe_inputs, config);

    return std::make_shared<ov::Model>(ov::OutputVector{moe}, ov::ParameterVector{input});
}

inline std::shared_ptr<ov::Model> build_2gemm_bgm_model() {
    using namespace ov;

    const size_t batch = 2;
    const Dimension in_dim = Dimension::dynamic();
    const size_t hidden_size = 2048;
    const size_t intermediate_size = 4096;
    const size_t topk = 2;
    const size_t number_of_experts = 3;
    const size_t fusion_factor = 2;
    const auto expert_alpha = 1.702f;
    const auto expert_beta = 7.0f;

    auto input = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{batch, in_dim, hidden_size});
    auto experts_reshape = std::make_shared<op::v1::Reshape>(
        input,
        op::v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{-1, static_cast<int64_t>(hidden_size)}),
        false);

    auto unsqueeze =
        std::make_shared<op::v0::Unsqueeze>(experts_reshape, op::v0::Constant::create(element::i32, Shape{}, {0}));

    // Router
    auto router_matmul = std::make_shared<op::v0::MatMul>(
        experts_reshape,
        op::v0::Constant::create(element::f32, Shape{number_of_experts, hidden_size}, {1.0f}),
        false,
        true);
    auto router_bias =
        std::make_shared<op::v1::Add>(router_matmul,
                                      op::v0::Constant::create(element::f32, Shape{1, number_of_experts}, {1.0f}));
    auto router_topk = std::make_shared<op::v11::TopK>(router_bias,
                                                       op::v0::Constant::create(element::i64, Shape{}, {topk}),
                                                       -1,
                                                       op::v11::TopK::Mode::MAX,
                                                       op::v11::TopK::SortType::SORT_VALUES,
                                                       element::i64);
    auto topk_indices = router_topk->output(1);
    auto chosen_experts = router_topk->output(0);

    // Gate/up weights and bias
    auto gate_up_w = op::v0::Constant::create(element::f32,
                                              Shape{number_of_experts, intermediate_size * fusion_factor, hidden_size},
                                              {1.0f});
    auto gate_up_bias =
        op::v0::Constant::create(element::f32, Shape{number_of_experts, 1, intermediate_size * fusion_factor}, {0.0f});

    // BGM gate_up (4 inputs: data, weight, indices, bias)
    auto bgm_gate_up = std::make_shared<GatherMatmul>(unsqueeze, gate_up_w, topk_indices, gate_up_bias);

    // Activation subgraph (same as in the original 2GEMM pattern)
    auto slice1 = std::make_shared<op::v8::Slice>(
        bgm_gate_up,
        op::v0::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{0, 0, 0}),
        op::v0::Constant::create(element::i64,
                                 Shape{3},
                                 std::vector<int64_t>{static_cast<int64_t>(topk),
                                                      static_cast<int64_t>(batch),
                                                      static_cast<int64_t>(intermediate_size * 2)}),
        op::v0::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{1, 1, 2}),
        op::v0::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{0, 1, 2}));
    auto clamp = std::make_shared<op::v0::Clamp>(slice1, -expert_beta, expert_beta);
    auto add1 = std::make_shared<op::v1::Add>(clamp, op::v0::Constant::create(element::f32, Shape{1}, {1.0f}));

    auto slice2 = std::make_shared<op::v8::Slice>(
        bgm_gate_up,
        op::v0::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{0, 1, 0}),
        op::v0::Constant::create(element::i64,
                                 Shape{3},
                                 std::vector<int64_t>{static_cast<int64_t>(topk),
                                                      static_cast<int64_t>(batch),
                                                      static_cast<int64_t>(intermediate_size * 2)}),
        op::v0::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{1, 1, 2}),
        op::v0::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{0, 1, 2}));
    auto minimum1 =
        std::make_shared<op::v1::Minimum>(slice2, op::v0::Constant::create(element::f32, Shape{1}, {10.0f}));
    auto swish_beta_const = op::v0::Constant::create(element::f32, Shape{}, std::vector<float>{expert_alpha});
    auto swish = std::make_shared<op::v4::Swish>(minimum1, swish_beta_const);
    auto multiply2 = std::make_shared<op::v1::Multiply>(add1, swish);

    // Down proj
    auto down_w =
        op::v0::Constant::create(element::f32, Shape{number_of_experts, hidden_size, intermediate_size}, {1.0f});
    auto down_bias = op::v0::Constant::create(element::f32, Shape{number_of_experts, 1, hidden_size}, {1.0f});
    auto bgm_down = std::make_shared<GatherMatmul>(multiply2, down_w, topk_indices, down_bias);

    // Compact routing
    auto router_transpose = std::make_shared<op::v1::Transpose>(
        chosen_experts,
        op::v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{1, 0}));
    auto router_unsqueeze =
        std::make_shared<op::v0::Unsqueeze>(router_transpose, op::v0::Constant::create(element::i32, Shape{}, {-1}));

    auto final_mul = std::make_shared<op::v1::Multiply>(bgm_down, router_unsqueeze);
    auto reduce_sum =
        std::make_shared<op::v1::ReduceSum>(final_mul,
                                            op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{0}),
                                            false);
    auto end_reshape = std::make_shared<op::v1::Reshape>(
        reduce_sum,
        op::v0::Constant::create(
            element::i64,
            Shape{3},
            std::vector<int64_t>{static_cast<int64_t>(batch), -1, static_cast<int64_t>(hidden_size)}),
        true);

    return std::make_shared<ov::Model>(ov::OutputVector{end_reshape}, ov::ParameterVector{input});
}

inline std::shared_ptr<ov::Model> build_2gemm_bgm_to_moe_reference_model() {
    using namespace ov;

    const size_t batch = 2;
    const Dimension in_dim = Dimension::dynamic();
    const size_t hidden_size = 2048;
    const size_t intermediate_size = 4096;
    const size_t topk = 2;
    const size_t number_of_experts = 3;
    const size_t fusion_factor = 2;
    const auto expert_alpha = 1.702f;
    const auto expert_beta = 7.0f;

    auto input = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{batch, in_dim, hidden_size});

    auto experts_reshape = std::make_shared<op::v1::Reshape>(
        input,
        op::v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{-1, static_cast<int64_t>(hidden_size)}),
        false);

    // Router (stays in graph)
    auto router_matmul = std::make_shared<op::v0::MatMul>(
        experts_reshape,
        op::v0::Constant::create(element::f32, Shape{number_of_experts, hidden_size}, {1.0f}),
        false,
        true);
    auto router_bias =
        std::make_shared<op::v1::Add>(router_matmul,
                                      op::v0::Constant::create(element::f32, Shape{1, number_of_experts}, {1.0f}));
    auto router_topk = std::make_shared<op::v11::TopK>(router_bias,
                                                       op::v0::Constant::create(element::i64, Shape{}, {topk}),
                                                       -1,
                                                       op::v11::TopK::Mode::MAX,
                                                       op::v11::TopK::SortType::SORT_VALUES,
                                                       element::i64);
    auto topk_indices = router_topk->output(1);
    auto chosen_experts = router_topk->output(0);

    // Convert2GatherMatmulMoeBlockToMoeOp bypasses Transpose+Unsqueeze (tokens-major).
    auto routing = chosen_experts;

    // Weights
    auto gate_up_w = op::v0::Constant::create(element::f32,
                                              Shape{number_of_experts, intermediate_size * fusion_factor, hidden_size},
                                              {1.0f});
    auto gate_up_bias =
        op::v0::Constant::create(element::f32, Shape{number_of_experts, 1, intermediate_size * fusion_factor}, {0.0f});
    auto down_w =
        op::v0::Constant::create(element::f32, Shape{number_of_experts, hidden_size, intermediate_size}, {1.0f});
    auto down_bias = op::v0::Constant::create(element::f32, Shape{number_of_experts, 1, hidden_size}, {1.0f});

    ov::OutputVector moe_inputs = {input, routing, topk_indices, gate_up_w, gate_up_bias, down_w, down_bias};

    ov::op::internal::MOE::Config config;
    config.expert_type = ov::op::internal::MOE::Expert_type::GEMM2_BIAS_SWIGLU_CLAMP;
    config.expert_alpha = expert_alpha;
    config.expert_beta = expert_beta;

    auto moe = std::make_shared<ov::op::internal::MOE>(moe_inputs, config);
    return std::make_shared<ov::Model>(ov::OutputVector{moe}, ov::ParameterVector{input});
}

// ============================================================================
// Post-BGM model builders (3 BGMs + compact routing) — Multiply instead of Reshape
// before Unsqueeze (gemma4 pattern: layernorm Multiply feeds expert path directly)
// ============================================================================

inline std::shared_ptr<ov::Model> build_3gemm_bgm_model_multiply_input(
    ov::op::internal::MOE::Activation_type activation_type = ov::op::internal::MOE::Activation_type::SWIGLU) {
    using namespace ov;

    const size_t batch = 2;
    const Dimension in_dim = Dimension::dynamic();
    const size_t hidden_size = 2048;
    const size_t intermediate_size = 4096;
    const size_t number_of_experts = 3;
    const size_t topk = 2;

    auto input = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{batch, in_dim, hidden_size});

    // Simulate layernorm output: Multiply(input, scale) — no Reshape
    auto norm_scale = op::v0::Constant::create(element::f32, Shape{1, 1, hidden_size}, {1.0f});
    auto layernorm_mul = std::make_shared<op::v1::Multiply>(input, norm_scale);

    // Reshape to [batch*seq, hidden] for router path
    auto experts_reshape = std::make_shared<op::v1::Reshape>(
        layernorm_mul,
        op::v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{-1, static_cast<int64_t>(hidden_size)}),
        false);

    // Unsqueeze feeds from Multiply (via Reshape) — but the pattern's optional<Reshape>
    // means hidden_states_m captures the Multiply output (layernorm_mul).
    auto unsqueeze =
        std::make_shared<op::v0::Unsqueeze>(experts_reshape, op::v0::Constant::create(element::i32, Shape{}, {0}));

    // Router subgraph
    auto router_matmul = std::make_shared<op::v0::MatMul>(
        experts_reshape,
        op::v0::Constant::create(element::f32, Shape{number_of_experts, hidden_size}, {1.0f}),
        false,
        true);
    auto router_topk = std::make_shared<op::v11::TopK>(router_matmul,
                                                       op::v0::Constant::create(element::i64, Shape{}, {topk}),
                                                       -1,
                                                       op::v11::TopK::Mode::MAX,
                                                       op::v11::TopK::SortType::SORT_VALUES,
                                                       element::i64);
    auto topk_indices = router_topk->output(1);
    auto chosen_experts = router_topk->output(0);

    // Weights
    auto gate_w =
        op::v0::Constant::create(element::f32, Shape{number_of_experts, intermediate_size, hidden_size}, {1.0f});
    auto up_w =
        op::v0::Constant::create(element::f32, Shape{number_of_experts, intermediate_size, hidden_size}, {1.0f});
    auto down_w =
        op::v0::Constant::create(element::f32, Shape{number_of_experts, hidden_size, intermediate_size}, {1.0f});

    // 3 BGMs
    auto bgm_gate = std::make_shared<GatherMatmul>(unsqueeze, gate_w, topk_indices);
    std::shared_ptr<ov::Node> gate_act;
    if (activation_type == ov::op::internal::MOE::Activation_type::GEGLU_TANH) {
        gate_act = std::make_shared<op::v7::Gelu>(bgm_gate, ov::op::GeluApproximationMode::TANH);
    } else if (activation_type == ov::op::internal::MOE::Activation_type::GEGLU_ERF) {
        gate_act = std::make_shared<op::v7::Gelu>(bgm_gate, ov::op::GeluApproximationMode::ERF);
    } else {
        gate_act = std::make_shared<op::v4::Swish>(bgm_gate);
    }
    auto bgm_up = std::make_shared<GatherMatmul>(unsqueeze, up_w, topk_indices);
    auto swiglu = std::make_shared<op::v1::Multiply>(gate_act, bgm_up);
    auto bgm_down = std::make_shared<GatherMatmul>(swiglu, down_w, topk_indices);

    // Compact routing
    auto router_transpose = std::make_shared<op::v1::Transpose>(
        chosen_experts,
        op::v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{1, 0}));
    auto router_unsqueeze =
        std::make_shared<op::v0::Unsqueeze>(router_transpose, op::v0::Constant::create(element::i32, Shape{}, {-1}));

    // Final: Multiply → ReduceSum → Reshape
    auto final_mul = std::make_shared<op::v1::Multiply>(bgm_down, router_unsqueeze);
    auto reduce_sum =
        std::make_shared<op::v1::ReduceSum>(final_mul,
                                            op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{0}),
                                            false);

    auto end_reshape = std::make_shared<op::v1::Reshape>(
        reduce_sum,
        op::v0::Constant::create(
            element::i64,
            Shape{3},
            std::vector<int64_t>{static_cast<int64_t>(batch), -1, static_cast<int64_t>(hidden_size)}),
        true);

    return std::make_shared<ov::Model>(ov::OutputVector{end_reshape}, ov::ParameterVector{input});
}

inline std::shared_ptr<ov::Model> build_3gemm_bgm_to_moe_reference_model_multiply_input(
    ov::op::internal::MOE::Activation_type activation_type = ov::op::internal::MOE::Activation_type::SWIGLU) {
    using namespace ov;

    const size_t batch = 2;
    const Dimension in_dim = Dimension::dynamic();
    const size_t hidden_size = 2048;
    const size_t intermediate_size = 4096;
    const size_t number_of_experts = 3;
    const size_t topk = 2;

    auto input = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{batch, in_dim, hidden_size});

    // Layernorm Multiply — its OUTPUT is the correct hidden_states for the MOE op
    auto norm_scale = op::v0::Constant::create(element::f32, Shape{1, 1, hidden_size}, {1.0f});
    auto layernorm_mul = std::make_shared<op::v1::Multiply>(input, norm_scale);

    // Router subgraph (not fused, remains in the graph)
    auto experts_reshape = std::make_shared<op::v1::Reshape>(
        layernorm_mul,
        op::v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{-1, static_cast<int64_t>(hidden_size)}),
        false);

    auto router_matmul = std::make_shared<op::v0::MatMul>(
        experts_reshape,
        op::v0::Constant::create(element::f32, Shape{number_of_experts, hidden_size}, {1.0f}),
        false,
        true);
    auto router_topk = std::make_shared<op::v11::TopK>(router_matmul,
                                                       op::v0::Constant::create(element::i64, Shape{}, {topk}),
                                                       -1,
                                                       op::v11::TopK::Mode::MAX,
                                                       op::v11::TopK::SortType::SORT_VALUES,
                                                       element::i64);
    auto topk_indices = router_topk->output(1);
    auto chosen_experts = router_topk->output(0);

    // Compact routing
    auto router_transpose = std::make_shared<op::v1::Transpose>(
        chosen_experts,
        op::v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{1, 0}));
    auto router_unsqueeze =
        std::make_shared<op::v0::Unsqueeze>(router_transpose, op::v0::Constant::create(element::i32, Shape{}, {-1}));

    // Weights
    auto gate_w =
        op::v0::Constant::create(element::f32, Shape{number_of_experts, intermediate_size, hidden_size}, {1.0f});
    auto up_w =
        op::v0::Constant::create(element::f32, Shape{number_of_experts, intermediate_size, hidden_size}, {1.0f});
    auto down_w =
        op::v0::Constant::create(element::f32, Shape{number_of_experts, hidden_size, intermediate_size}, {1.0f});

    // MOE op — hidden_states input is the layernorm Multiply output (NOT the Parameter)
    ov::OutputVector moe_inputs = {layernorm_mul, router_unsqueeze, topk_indices, gate_w, up_w, down_w};
    ov::op::internal::MOE::Config config;
    config.expert_type = ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU;
    config.activation_type = activation_type;
    auto moe = std::make_shared<ov::op::internal::MOE>(moe_inputs, config);

    return std::make_shared<ov::Model>(ov::OutputVector{moe}, ov::ParameterVector{input});
}

// Per-branch compression scheme used by the compressed-3gemm builder. NNCF mixed-precision
// quantization can emit different schemes per gate/up/down within a single MoE block — the
// fusion pass must reject those graphs so they fall back to BGMCompressed execution.
enum class CompressionScheme {
    GroupedSym,   // INT4 grouped-symmetric: weight rank-4, scale rank-4, ZP = dynamic placeholder
    GroupedAsym,  // INT4 grouped-asymmetric: weight rank-4, scale rank-4, ZP rank-4
    PerOcSym,     // INT8 per-output-channel symmetric: weight rank-3, scale rank-3, ZP = dynamic placeholder
};

inline std::shared_ptr<ov::Model> build_3gemm_compressed_bgm_model(CompressionScheme gate_s,
                                                                  CompressionScheme up_s,
                                                                  CompressionScheme down_s) {
    using namespace ov;

    const size_t batch = 2;
    const Dimension in_dim = Dimension::dynamic();
    const size_t hidden_size = 2048;
    const size_t intermediate_size = 4096;
    const size_t group_size = 128;
    const size_t number_of_experts = 3;
    const size_t topk = 2;

    // Build a BGMCompressed input bundle (B, scale, ZP) for one branch.
    // K is the contracting dim (hidden_size for gate/up, intermediate_size for down).
    // OFM is the output-feature dim (intermediate_size for gate/up, hidden_size for down).
    auto make_branch = [&](size_t OFM, size_t K, CompressionScheme s) {
        std::shared_ptr<ov::Node> w, scale, zp;
        switch (s) {
        case CompressionScheme::GroupedSym:
        case CompressionScheme::GroupedAsym: {
            const size_t G = K / group_size;
            w = op::v0::Constant::create(element::f32, Shape{number_of_experts, OFM, G, group_size}, {1.0f});
            scale = op::v0::Constant::create(element::f32, Shape{number_of_experts, OFM, G, 1}, {1.0f});
            if (s == CompressionScheme::GroupedAsym) {
                zp = op::v0::Constant::create(element::f32, Shape{number_of_experts, OFM, G, 1}, {0.0f});
            } else {
                zp = std::make_shared<op::v0::Constant>(element::dynamic, Shape{0});
            }
            break;
        }
        case CompressionScheme::PerOcSym: {
            w = op::v0::Constant::create(element::f32, Shape{number_of_experts, OFM, K}, {1.0f});
            scale = op::v0::Constant::create(element::f32, Shape{number_of_experts, OFM, 1}, {1.0f});
            zp = std::make_shared<op::v0::Constant>(element::dynamic, Shape{0});
            break;
        }
        }
        return std::make_tuple(w, scale, zp);
    };

    auto input = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{batch, in_dim, hidden_size});
    auto experts_reshape = std::make_shared<op::v1::Reshape>(
        input,
        op::v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{-1, static_cast<int64_t>(hidden_size)}),
        false);
    auto unsqueeze =
        std::make_shared<op::v0::Unsqueeze>(experts_reshape, op::v0::Constant::create(element::i32, Shape{}, {0}));

    // Router → topk
    auto router_matmul = std::make_shared<op::v0::MatMul>(
        experts_reshape,
        op::v0::Constant::create(element::f32, Shape{number_of_experts, hidden_size}, {1.0f}),
        false,
        true);
    auto router_topk = std::make_shared<op::v11::TopK>(router_matmul,
                                                       op::v0::Constant::create(element::i64, Shape{}, {topk}),
                                                       -1,
                                                       op::v11::TopK::Mode::MAX,
                                                       op::v11::TopK::SortType::SORT_VALUES,
                                                       element::i64);
    auto topk_indices = router_topk->output(1);
    auto chosen_experts = router_topk->output(0);

    auto bias = std::make_shared<op::v0::Constant>(element::dynamic, Shape{0});

    auto [gate_w, gate_scale, gate_zp] = make_branch(intermediate_size, hidden_size, gate_s);
    auto bgm_gate =
        std::make_shared<GatherMatmulCompressed>(unsqueeze, gate_w, topk_indices, bias, gate_scale, gate_zp);
    auto gate_act = std::make_shared<op::v4::Swish>(bgm_gate);

    auto [up_w, up_scale, up_zp] = make_branch(intermediate_size, hidden_size, up_s);
    auto bgm_up = std::make_shared<GatherMatmulCompressed>(unsqueeze, up_w, topk_indices, bias, up_scale, up_zp);
    auto swiglu = std::make_shared<op::v1::Multiply>(gate_act, bgm_up);

    auto [down_w, down_scale, down_zp] = make_branch(hidden_size, intermediate_size, down_s);
    auto bgm_down =
        std::make_shared<GatherMatmulCompressed>(swiglu, down_w, topk_indices, bias, down_scale, down_zp);

    auto router_transpose = std::make_shared<op::v1::Transpose>(
        chosen_experts,
        op::v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{1, 0}));
    auto router_unsqueeze =
        std::make_shared<op::v0::Unsqueeze>(router_transpose, op::v0::Constant::create(element::i32, Shape{}, {-1}));

    auto final_mul = std::make_shared<op::v1::Multiply>(bgm_down, router_unsqueeze);
    auto reduce_sum =
        std::make_shared<op::v1::ReduceSum>(final_mul,
                                            op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{0}),
                                            false);
    auto end_reshape = std::make_shared<op::v1::Reshape>(
        reduce_sum,
        op::v0::Constant::create(
            element::i64,
            Shape{3},
            std::vector<int64_t>{static_cast<int64_t>(batch), -1, static_cast<int64_t>(hidden_size)}),
        true);

    return std::make_shared<ov::Model>(ov::OutputVector{end_reshape}, ov::ParameterVector{input});
}

// ============================================================================
// Tests for BGM→MOE passes (Convert3GatherMatmulMoeBlockToMoeOp, Convert2GatherMatmulMoeBlockToMoeOp)
// ============================================================================

TEST_F(TransformationTestsF, Convert3GatherMatmulMoeBlockToMoeOp_basic) {
    model = build_3gemm_bgm_model();
    manager.register_pass<ov::pass::Convert3GatherMatmulMoeBlockToMoeOp>();
    model_ref = build_3gemm_bgm_to_moe_reference_model();
}

TEST_F(TransformationTestsF, Convert3GatherMatmulMoeBlockToMoeOp_gelu_tanh) {
    using AT = ov::op::internal::MOE::Activation_type;
    model = build_3gemm_bgm_model(AT::GEGLU_TANH);
    manager.register_pass<ov::pass::Convert3GatherMatmulMoeBlockToMoeOp>();
    model_ref = build_3gemm_bgm_to_moe_reference_model(AT::GEGLU_TANH);
}

TEST_F(TransformationTestsF, Convert3GatherMatmulMoeBlockToMoeOp_gelu_erf) {
    using AT = ov::op::internal::MOE::Activation_type;
    model = build_3gemm_bgm_model(AT::GEGLU_ERF);
    manager.register_pass<ov::pass::Convert3GatherMatmulMoeBlockToMoeOp>();
    model_ref = build_3gemm_bgm_to_moe_reference_model(AT::GEGLU_ERF);
}

TEST_F(TransformationTestsF, Convert3GatherMatmulMoeBlockToMoeOp_multiply_input) {
    model = build_3gemm_bgm_model_multiply_input();
    manager.register_pass<ov::pass::Convert3GatherMatmulMoeBlockToMoeOp>();
    model_ref = build_3gemm_bgm_to_moe_reference_model_multiply_input();
}

TEST_F(TransformationTestsF, Convert3GatherMatmulMoeBlockToMoeOp_multiply_input_gelu_tanh) {
    using AT = ov::op::internal::MOE::Activation_type;
    model = build_3gemm_bgm_model_multiply_input(AT::GEGLU_TANH);
    manager.register_pass<ov::pass::Convert3GatherMatmulMoeBlockToMoeOp>();
    model_ref = build_3gemm_bgm_to_moe_reference_model_multiply_input(AT::GEGLU_TANH);
}

TEST_F(TransformationTestsF, Convert2GatherMatmulMoeBlockToMoeOp_basic) {
    model = build_2gemm_bgm_model();
    manager.register_pass<ov::pass::Convert2GatherMatmulMoeBlockToMoeOp>();
    model_ref = build_2gemm_bgm_to_moe_reference_model();
}

// Negative cases — MOECompressed::Config carries one group_size and one has_zp shared by gate/up/down.
// NNCF mixed-precision IRs (e.g. Mixtral-8x7B INT4) violate this when only some experts within a layer
// are kept INT4-grouped while the rest fall back to INT8-per-OC, or symmetric and asymmetric coexist.
// Fusion must reject such graphs (no model_ref → expect graph unchanged) so they execute as plain
// BGMCompressed instead of throwing in MOECompressed::validate_and_infer_types.

TEST_F(TransformationTestsF, Convert3GatherMatmulMoeBlockToMoeOp_mixed_group_size_up_per_oc) {
    // Mixtral-8x7B INT4 layer-0 shape: gate grouped, up per-OC, down grouped.
    model = build_3gemm_compressed_bgm_model(CompressionScheme::GroupedSym,
                                             CompressionScheme::PerOcSym,
                                             CompressionScheme::GroupedSym);
    manager.register_pass<ov::pass::Convert3GatherMatmulMoeBlockToMoeOp>();
}

TEST_F(TransformationTestsF, Convert3GatherMatmulMoeBlockToMoeOp_mixed_group_size_down_per_oc) {
    model = build_3gemm_compressed_bgm_model(CompressionScheme::GroupedSym,
                                             CompressionScheme::GroupedSym,
                                             CompressionScheme::PerOcSym);
    manager.register_pass<ov::pass::Convert3GatherMatmulMoeBlockToMoeOp>();
}

TEST_F(TransformationTestsF, Convert3GatherMatmulMoeBlockToMoeOp_mixed_has_zp) {
    // gate/down symmetric, up asymmetric: ZP element-type disagrees across branches.
    model = build_3gemm_compressed_bgm_model(CompressionScheme::GroupedSym,
                                             CompressionScheme::GroupedAsym,
                                             CompressionScheme::GroupedSym);
    manager.register_pass<ov::pass::Convert3GatherMatmulMoeBlockToMoeOp>();
}

// Positive — uniform schemes still fuse; guards the negative checks against accidentally widening.
// Uses plain TEST (not TransformationTestsF) so we can assert directly on the post-pass graph
// without authoring a reference model for every BGM→MOECompressed shape detail.
namespace {
bool has_moe_compressed_op(const std::shared_ptr<ov::Model>& m) {
    for (const auto& op : m->get_ordered_ops()) {
        if (ov::is_type<ov::op::internal::MOECompressed>(op)) {
            return true;
        }
    }
    return false;
}
}  // namespace

TEST(Convert3GatherMatmulMoeBlockToMoeOp, compressed_uniform_grouped_sym_fuses) {
    auto model = build_3gemm_compressed_bgm_model(CompressionScheme::GroupedSym,
                                                  CompressionScheme::GroupedSym,
                                                  CompressionScheme::GroupedSym);
    ov::pass::Manager mgr;
    mgr.register_pass<ov::pass::Convert3GatherMatmulMoeBlockToMoeOp>();
    mgr.run_passes(model);
    EXPECT_TRUE(has_moe_compressed_op(model)) << "uniform grouped-sym scheme should fuse to MOECompressed";
}

TEST(Convert3GatherMatmulMoeBlockToMoeOp, compressed_uniform_per_oc_fuses) {
    auto model = build_3gemm_compressed_bgm_model(CompressionScheme::PerOcSym,
                                                  CompressionScheme::PerOcSym,
                                                  CompressionScheme::PerOcSym);
    ov::pass::Manager mgr;
    mgr.register_pass<ov::pass::Convert3GatherMatmulMoeBlockToMoeOp>();
    mgr.run_passes(model);
    EXPECT_TRUE(has_moe_compressed_op(model)) << "uniform per-OC scheme should fuse to MOECompressed";
}
