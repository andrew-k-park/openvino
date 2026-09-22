// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "intel_gpu/op/fully_connected_compressed.hpp"
#include "intel_gpu/op/placeholder.hpp"
#include "intel_gpu/op/sdpa.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/pass/manager.hpp"
#include "ov_ops/dynamic_quantize.hpp"
#include "plugin/transformations/dynamic_quantize_fully_connected.hpp"

namespace ov::test::intel_gpu {

static std::shared_ptr<ov::Model> make_model(float scale_value) {
    auto q = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 2, 4, 16});
    auto k = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 2, 4, 16});
    auto v = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 2, 4, 16});
    auto sdpa = std::make_shared<ov::intel_gpu::op::SDPA>(ov::OutputVector{q, k, v},
                                                          false,
                                                          std::vector<int64_t>{0, 1, 2, 3},
                                                          std::vector<int64_t>{0, 1, 2, 3},
                                                          std::vector<int64_t>{0, 1, 2, 3},
                                                          std::vector<int64_t>{0, 2, 1, 3});
    auto scale = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{}, {scale_value});
    auto multiply = std::make_shared<ov::op::v1::Multiply>(sdpa, scale);
    auto reshape_pattern = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {1, 4, 32});
    auto reshape = std::make_shared<ov::op::v1::Reshape>(multiply, reshape_pattern, false);
    auto weights = ov::op::v0::Constant::create(ov::element::u4, ov::Shape{64, 32}, {1});
    auto bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
    auto weight_scale = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{64, 1}, {1.0f});
    auto fc = std::make_shared<ov::intel_gpu::op::FullyConnectedCompressed>(reshape,
                                                                            weights,
                                                                            bias,
                                                                            weight_scale);
    return std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{q, k, v});
}

static std::shared_ptr<ov::op::internal::DynamicQuantize> run_pass(float scale_value) {
    auto model = make_model(scale_value);
    ov::pass::Manager manager;
    manager.register_pass<ov::intel_gpu::DynamicQuantizeFullyConnected>(32, false, false);
    manager.run_passes(model);

    for (const auto& node : model->get_ordered_ops()) {
        if (auto dq = ov::as_type_ptr<ov::op::internal::DynamicQuantize>(node))
            return dq;
    }
    return nullptr;
}

TEST(DynamicQuantizeFullyConnectedTest, FusesPostSDPAPowerOfTwoScale) {
    auto dq = run_pass(0.125f);

    ASSERT_NE(dq, nullptr);
    EXPECT_EQ(dq->get_attrs().input_scale, 0.125f);
    auto reshape = ov::as_type_ptr<ov::op::v1::Reshape>(dq->input_value(0).get_node_shared_ptr());
    ASSERT_NE(reshape, nullptr);
    EXPECT_TRUE(ov::is_type<ov::intel_gpu::op::SDPA>(reshape->input_value(0).get_node_shared_ptr()));
}

TEST(DynamicQuantizeFullyConnectedTest, RejectsPostSDPANonPowerOfTwoScale) {
    auto dq = run_pass(0.3f);

    ASSERT_NE(dq, nullptr);
    EXPECT_EQ(dq->get_attrs().input_scale, 1.0f);
    auto reshape = ov::as_type_ptr<ov::op::v1::Reshape>(dq->input_value(0).get_node_shared_ptr());
    ASSERT_NE(reshape, nullptr);
    EXPECT_TRUE(ov::is_type<ov::op::v1::Multiply>(reshape->input_value(0).get_node_shared_ptr()));
}

}  // namespace ov::test::intel_gpu
