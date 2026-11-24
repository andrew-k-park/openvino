// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_group_conv_backprop_data_1d_to_2d.hpp"

#include <memory>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::intel_gpu {

ConvertGroupConvolutionBackpropData1DTo2D::ConvertGroupConvolutionBackpropData1DTo2D() {
    using namespace ov::pass::pattern;

    auto deconv_m = wrap_type<ov::op::v1::GroupConvolutionBackpropData>([](const ov::Output<ov::Node>& output) {
        return output.get_node()->get_input_partial_shape(0).rank() == 3;
    });

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        auto deconv = ov::as_type_ptr<ov::op::v1::GroupConvolutionBackpropData>(m.get_match_root());
        if (!deconv || transformation_callback(deconv)) {
            return false;
        }

        // data [N, C, W] -> [N, C, 1, W], weights [G, C_IN, C_OUT, K] -> [G, C_IN, C_OUT, 1, K]
        auto data_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {2});
        auto weights_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {3});
        auto data = std::make_shared<ov::op::v0::Unsqueeze>(deconv->input_value(0), data_axis);
        auto weights = ov::op::util::make_try_fold<ov::op::v0::Unsqueeze>(deconv->input_value(1), weights_axis);

        auto with_unit_height = [](auto values, auto unit) {
            if (!values.empty()) {
                values.insert(values.begin(), unit);
            }
            return values;
        };
        const auto strides = with_unit_height(deconv->get_strides(), size_t{1});
        const auto dilations = with_unit_height(deconv->get_dilations(), size_t{1});
        const auto pads_begin = with_unit_height(deconv->get_pads_begin(), std::ptrdiff_t{0});
        const auto pads_end = with_unit_height(deconv->get_pads_end(), std::ptrdiff_t{0});
        const auto output_padding = with_unit_height(deconv->get_output_padding(), std::ptrdiff_t{0});

        std::shared_ptr<ov::op::v1::GroupConvolutionBackpropData> deconv_2d;
        ov::NodeVector new_nodes{data, weights};
        if (deconv->get_input_size() == 3) {
            const auto output_shape = deconv->input_value(2);
            auto unit_height = ov::op::v0::Constant::create(output_shape.get_element_type(), ov::Shape{1}, {1});
            auto output_shape_2d = ov::op::util::make_try_fold<ov::op::v0::Concat>(ov::OutputVector{unit_height, output_shape}, 0);
            new_nodes.push_back(output_shape_2d);
            deconv_2d = std::make_shared<ov::op::v1::GroupConvolutionBackpropData>(data,
                                                                                   weights,
                                                                                   output_shape_2d,
                                                                                   strides,
                                                                                   pads_begin,
                                                                                   pads_end,
                                                                                   dilations,
                                                                                   deconv->get_auto_pad(),
                                                                                   output_padding);
        } else {
            deconv_2d = std::make_shared<ov::op::v1::GroupConvolutionBackpropData>(data,
                                                                                   weights,
                                                                                   strides,
                                                                                   pads_begin,
                                                                                   pads_end,
                                                                                   dilations,
                                                                                   deconv->get_auto_pad(),
                                                                                   output_padding);
        }
        auto output = std::make_shared<ov::op::v0::Squeeze>(deconv_2d, data_axis);
        new_nodes.push_back(deconv_2d);
        new_nodes.push_back(output);

        deconv_2d->set_friendly_name(deconv->get_friendly_name() + "/2d");
        output->set_friendly_name(deconv->get_friendly_name());
        ov::copy_runtime_info(deconv, new_nodes);
        ov::replace_node(deconv, output);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(deconv_m, "ConvertGroupConvolutionBackpropData1DTo2D");
    register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
