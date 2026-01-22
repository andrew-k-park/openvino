// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/node_builders/constant.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/gather.hpp"

namespace {
using ov::test::InputShape;

struct RoPELtxVideoFusionParams {
    ov::PartialShape input_shape;        // [batch, seq_len, 2048]
    ov::PartialShape cos_sin_shape;      // [batch, seq_len, 2048]
    ov::element::Type model_type;
};

class RoPELtxVideoFusionTest : public testing::WithParamInterface<RoPELtxVideoFusionParams>,
                                virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<RoPELtxVideoFusionParams>& obj) {
        RoPELtxVideoFusionParams params = obj.param;
        std::ostringstream result;
        result << "IS=" << params.input_shape << "_";
        result << "CS=" << params.cos_sin_shape << "_";
        result << "Prc=" << params.model_type;
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;
        
        RoPELtxVideoFusionParams params = this->GetParam();
        
        // Initialize input shapes for dynamic execution
        std::vector<ov::Shape> concrete_shapes;
        if (params.input_shape.is_static()) {
            concrete_shapes = {params.input_shape.to_shape()};
        } else {
            // For dynamic shapes, provide concrete test shapes
            concrete_shapes = {ov::Shape{1, 256, 2048}};
        }
        
        InputShape input_shape = {params.input_shape, concrete_shapes};
        InputShape cos_shape = {params.cos_sin_shape, concrete_shapes};
        InputShape sin_shape = {params.cos_sin_shape, concrete_shapes};
        init_input_shapes({input_shape, cos_shape, sin_shape});
        
        // Create input parameters
        auto input_param = std::make_shared<ov::op::v0::Parameter>(params.model_type, params.input_shape);
        input_param->set_friendly_name("input");
        
        auto cos_param = std::make_shared<ov::op::v0::Parameter>(params.model_type, params.cos_sin_shape);
        cos_param->set_friendly_name("cos");
        
        auto sin_param = std::make_shared<ov::op::v0::Parameter>(params.model_type, params.cos_sin_shape);
        sin_param->set_friendly_name("sin");
        
        ov::ParameterVector params_vec = {input_param, cos_param, sin_param};
        
        // Build LTX-Video RoPE pattern (19 nodes)
        auto rope_output = buildLtxVideoRoPEPattern(input_param, cos_param, sin_param);
        
        auto result = std::make_shared<ov::op::v0::Result>(rope_output);
        result->set_friendly_name("output");
        
        function = std::make_shared<ov::Model>(ov::ResultVector{result}, params_vec, "RoPELtxVideoFusion");
    }
    
    void generate_inputs(const std::vector<ov::Shape>& target_shapes) override {
        inputs.clear();
        
        // Generate input tensor
        auto input_tensor = ov::test::utils::create_and_fill_tensor(
            function->get_parameters()[0]->get_element_type(), 
            target_shapes[0],
            10.0,  // range
            -5.0,  // start_from
            1);    // resolution
        inputs.insert({function->get_parameters()[0], input_tensor});
        
        // Generate cos tensor
        auto cos_tensor = ov::test::utils::create_and_fill_tensor(
            function->get_parameters()[1]->get_element_type(), 
            target_shapes[1],
            1.0,   // range (cos values in [-1, 1])
            -1.0,  // start_from
            1);
        inputs.insert({function->get_parameters()[1], cos_tensor});
        
        // Generate sin tensor
        auto sin_tensor = ov::test::utils::create_and_fill_tensor(
            function->get_parameters()[2]->get_element_type(), 
            target_shapes[2],
            1.0,   // range (sin values in [-1, 1])
            -1.0,  // start_from
            1);
        inputs.insert({function->get_parameters()[2], sin_tensor});
    }

private:
    // Build the exact LTX-Video RoPE pattern from IR dump
    std::shared_ptr<ov::Node> buildLtxVideoRoPEPattern(
        const std::shared_ptr<ov::Node>& input,
        const std::shared_ptr<ov::Node>& cos,
        const std::shared_ptr<ov::Node>& sin) {
        
        // Step 1: Reshape input to [batch, seq_len, 1024, 2]
        // Use ShapeOf to support dynamic shapes
        auto input_shape = std::make_shared<ov::op::v3::ShapeOf>(input);
        
        // Gather first 2 dimensions [batch, seq_len]
        auto indices_0_1 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {0, 1});
        auto gather_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
        auto batch_seq = std::make_shared<ov::op::v7::Gather>(input_shape, indices_0_1, gather_axis);
        
        // Append [1024, 2] to create [batch, seq_len, 1024, 2]
        auto new_dims = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {1024, 2});
        ov::OutputVector concat_inputs = {batch_seq, new_dims};
        auto reshape_shape_1 = std::make_shared<ov::op::v0::Concat>(concat_inputs, 0);
        auto x_reshape = std::make_shared<ov::op::v1::Reshape>(input, reshape_shape_1, false);
        x_reshape->set_friendly_name("x_reshape");
        
        // Step 2: Split into real and imaginary parts
        auto split_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {-1});
        auto split = std::make_shared<ov::op::v1::Split>(x_reshape, split_axis, 2);
        split->set_friendly_name("split");
        
        auto real = split->output(0);  // [batch, seq_len, 1024, 1]
        auto imag = split->output(1);  // [batch, seq_len, 1024, 1]
        
        // Step 3: Negate imaginary part: imag * (-1.0)
        auto neg_const = ov::op::v0::Constant::create(input->get_element_type(), ov::Shape{1}, {-1.0f});
        auto neg_imag_mul = std::make_shared<ov::op::v1::Multiply>(imag, neg_const);
        neg_imag_mul->set_friendly_name("neg_imag_mul");
        
        // Step 4: Squeeze and Unsqueeze (pattern from IR)
        auto squeeze_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
        auto squeeze_imag = std::make_shared<ov::op::v0::Squeeze>(neg_imag_mul, squeeze_axis);
        squeeze_imag->set_friendly_name("squeeze_imag");
        
        auto unsqueeze_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
        auto unsqueeze_imag = std::make_shared<ov::op::v0::Unsqueeze>(squeeze_imag, unsqueeze_axis);
        unsqueeze_imag->set_friendly_name("unsqueeze_imag");
        
        // Step 5: Concat [-imag, real] along axis=-1
        ov::OutputVector concat_rotated = {unsqueeze_imag, real};
        auto x_rotated_concat = std::make_shared<ov::op::v0::Concat>(concat_rotated, -1);
        x_rotated_concat->set_friendly_name("x_rotated_concat");
        
        // Step 6: Reshape back to [batch, seq_len, 2048]
        auto new_dims_2 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2048});
        ov::OutputVector concat_inputs_2 = {batch_seq, new_dims_2};
        auto reshape_shape_2 = std::make_shared<ov::op::v0::Concat>(concat_inputs_2, 0);
        auto x_rotated = std::make_shared<ov::op::v1::Reshape>(x_rotated_concat, reshape_shape_2, false);
        x_rotated->set_friendly_name("x_rotated");
        
        // Step 7: Apply RoPE formula: x * cos + x_rotated * sin
        auto real_mul_cos = std::make_shared<ov::op::v1::Multiply>(input, cos);
        real_mul_cos->set_friendly_name("real_mul_cos");
        
        auto imag_mul_sin = std::make_shared<ov::op::v1::Multiply>(x_rotated, sin);
        imag_mul_sin->set_friendly_name("imag_mul_sin");
        
        auto result = std::make_shared<ov::op::v1::Add>(real_mul_cos, imag_mul_sin);
        result->set_friendly_name("result");
        
        return result;
    }
};

TEST_P(RoPELtxVideoFusionTest, CompareWithRefs) {
    run();
}

const std::vector<RoPELtxVideoFusionParams> testParams = {
    // Static shapes
    {
        ov::PartialShape{1, 256, 2048},     // input: [batch=1, seq_len=256, 2048]
        ov::PartialShape{1, 256, 2048},     // cos/sin: [batch=1, seq_len=256, 2048]
        ov::element::f16
    },
    {
        ov::PartialShape{2, 512, 2048},     // input: [batch=2, seq_len=512, 2048]
        ov::PartialShape{2, 512, 2048},     // cos/sin: [batch=2, seq_len=512, 2048]
        ov::element::f32
    },
    // Dynamic shapes
    {
        ov::PartialShape{-1, -1, 2048},     // input: [batch=dynamic, seq_len=dynamic, 2048]
        ov::PartialShape{-1, -1, 2048},     // cos/sin: [batch=dynamic, seq_len=dynamic, 2048]
        ov::element::f16
    },
    {
        ov::PartialShape{ov::Dimension(1, 4), ov::Dimension(128, 1024), 2048},  // bounded dynamic
        ov::PartialShape{ov::Dimension(1, 4), ov::Dimension(128, 1024), 2048},
        ov::element::f32
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_RoPELtxVideoFusion,
                         RoPELtxVideoFusionTest,
                         ::testing::ValuesIn(testParams),
                         RoPELtxVideoFusionTest::getTestCaseName);

// Test with actual LTX-Video model dimensions
const std::vector<RoPELtxVideoFusionParams> ltxVideoModelParams = {
    {
        ov::PartialShape{1, 2520, 2048},    // 640x480@48frames: (48/4)*(640/16)*(480/16) = 2520
        ov::PartialShape{1, 2520, 2048},
        ov::element::f16
    },
    {
        ov::PartialShape{1, 1260, 2048},    // 640x480@24frames: (24/4)*(640/16)*(480/16) = 1260
        ov::PartialShape{1, 1260, 2048},
        ov::element::f16
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_RoPELtxVideoFusion_ModelDims,
                         RoPELtxVideoFusionTest,
                         ::testing::ValuesIn(ltxVideoModelParams),
                         RoPELtxVideoFusionTest::getTestCaseName);

} // namespace
