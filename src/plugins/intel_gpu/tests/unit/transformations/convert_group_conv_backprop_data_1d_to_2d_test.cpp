// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/transformations/convert_group_conv_backprop_data_1d_to_2d.hpp"

#include <numeric>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/manager.hpp"

using namespace testing;
using namespace ov::intel_gpu;

namespace ov::test::intel_gpu {

namespace {
std::vector<float> make_weights(size_t size) {
    std::vector<float> values(size);
    std::iota(values.begin(), values.end(), 1.0f);
    return values;
}
}  // namespace

TEST_F(TransformationTestsF, ConvertGroupConvolutionBackpropData1DTo2D_Depthwise) {
    const auto weights_values = make_weights(16 * 12);
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 16, -1});
        auto weights = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{16, 1, 1, 12}, weights_values);
        auto deconv = std::make_shared<ov::op::v1::GroupConvolutionBackpropData>(input,
                                                                                 weights,
                                                                                 ov::Strides{2},
                                                                                 ov::CoordinateDiff{5},
                                                                                 ov::CoordinateDiff{5},
                                                                                 ov::Strides{1});
        model = std::make_shared<ov::Model>(ov::OutputVector{deconv}, ov::ParameterVector{input});
        manager.register_pass<ConvertGroupConvolutionBackpropData1DTo2D>();
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 16, -1});
        auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {2});
        auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(input, axis);
        auto weights = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{16, 1, 1, 1, 12}, weights_values);
        auto deconv = std::make_shared<ov::op::v1::GroupConvolutionBackpropData>(unsqueeze,
                                                                                 weights,
                                                                                 ov::Strides{1, 2},
                                                                                 ov::CoordinateDiff{0, 5},
                                                                                 ov::CoordinateDiff{0, 5},
                                                                                 ov::Strides{1, 1});
        auto squeeze = std::make_shared<ov::op::v0::Squeeze>(deconv, axis);
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{squeeze}, ov::ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, ConvertGroupConvolutionBackpropData1DTo2D_OutputShape) {
    const auto weights_values = make_weights(2 * 3 * 4 * 3);
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 6, -1});
        auto weights = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{2, 3, 4, 3}, weights_values);
        auto output_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {40});
        auto deconv = std::make_shared<ov::op::v1::GroupConvolutionBackpropData>(input,
                                                                                 weights,
                                                                                 output_shape,
                                                                                 ov::Strides{2},
                                                                                 ov::CoordinateDiff{0},
                                                                                 ov::CoordinateDiff{0},
                                                                                 ov::Strides{1},
                                                                                 ov::op::PadType::VALID);
        model = std::make_shared<ov::Model>(ov::OutputVector{deconv}, ov::ParameterVector{input});
        manager.register_pass<ConvertGroupConvolutionBackpropData1DTo2D>();
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 6, -1});
        auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {2});
        auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(input, axis);
        auto weights = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{2, 3, 4, 1, 3}, weights_values);
        auto output_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {1, 40});
        auto deconv = std::make_shared<ov::op::v1::GroupConvolutionBackpropData>(unsqueeze,
                                                                                 weights,
                                                                                 output_shape,
                                                                                 ov::Strides{1, 2},
                                                                                 ov::CoordinateDiff{0, 0},
                                                                                 ov::CoordinateDiff{0, 0},
                                                                                 ov::Strides{1, 1},
                                                                                 ov::op::PadType::VALID);
        auto squeeze = std::make_shared<ov::op::v0::Squeeze>(deconv, axis);
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{squeeze}, ov::ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, ConvertGroupConvolutionBackpropData1DTo2D_2DNotChanged) {
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 16, 1, -1});
        auto weights = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{16, 1, 1, 1, 12}, make_weights(16 * 12));
        auto deconv = std::make_shared<ov::op::v1::GroupConvolutionBackpropData>(input,
                                                                                 weights,
                                                                                 ov::Strides{1, 2},
                                                                                 ov::CoordinateDiff{0, 0},
                                                                                 ov::CoordinateDiff{0, 0},
                                                                                 ov::Strides{1, 1});
        model = std::make_shared<ov::Model>(ov::OutputVector{deconv}, ov::ParameterVector{input});
        manager.register_pass<ConvertGroupConvolutionBackpropData1DTo2D>();
    }
    model_ref = model->clone();
}

}  // namespace ov::test::intel_gpu
