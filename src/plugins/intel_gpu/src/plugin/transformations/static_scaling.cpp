// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "static_scaling.hpp"

#include "intel_gpu/op/fully_connected_compressed.hpp"
#include "intel_gpu/op/placeholder.hpp"

#include "openvino/op/multiply.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "transformations/utils/utils.hpp"

namespace ov {
namespace intel_gpu {

StaticScaling::StaticScaling() {
    using namespace ov::pass::pattern;

    auto data_m = any_input();
    auto weights_m = any_input();
    auto bias_m = any_input();
    auto fc_compressed_wo_zp_m = wrap_type<op::FullyConnectedCompressed>({data_m, weights_m, bias_m, any_input()}, consumers_count(1));
    auto fc_compressed_w_zp_m = wrap_type<op::FullyConnectedCompressed>({data_m, weights_m, bias_m, any_input(), any_input()}, consumers_count(1));
    auto fc_compressed_m = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{fc_compressed_wo_zp_m, fc_compressed_w_zp_m});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        auto fc = std::dynamic_pointer_cast<op::FullyConnectedCompressed>(m.get_match_root());
        if (!fc || transformation_callback(fc))
            return false;

        const auto& pattern_map = m.get_pattern_value_map();
        const auto& data = pattern_map.at(data_m).get_node_shared_ptr();
        const auto& bias = pattern_map.at(bias_m).get_node_shared_ptr();

        const float scale_factor = 2.f;

        ov::Shape scale_const_shape = {1};
        std::vector<float> scale_down_value = {(1.f / scale_factor)};
        std::vector<float> scale_up_value = {scale_factor};
        std::shared_ptr<ov::Node> scale_down_const_f16 = std::make_shared<ov::op::v0::Constant>(ov::element::f16, scale_const_shape, scale_down_value);
        std::shared_ptr<ov::Node> scale_down_const_f32 = std::make_shared<ov::op::v0::Constant>(ov::element::f32, scale_const_shape, scale_down_value);
        std::shared_ptr<ov::Node> scale_up_const_f16 = std::make_shared<ov::op::v0::Constant>(ov::element::f16, scale_const_shape, scale_up_value);
        std::shared_ptr<ov::Node> scale_up_const_f32 = std::make_shared<ov::op::v0::Constant>(ov::element::f32, scale_const_shape, scale_up_value);

        std::shared_ptr<ov::Node> scale_down_const = (data->get_element_type() == ov::element::f16) ? scale_down_const_f16 : scale_down_const_f32;
        auto scale_down = std::make_shared<ov::op::v1::Multiply>(data, scale_down_const);
        fc->input(0).replace_source_output(scale_down->output(0));

        if (!std::dynamic_pointer_cast<op::Placeholder>(bias)) {
            std::shared_ptr<ov::Node> bias_scale_down_const = (bias->get_element_type() == ov::element::f16) ? scale_down_const_f16 : scale_down_const_f32;
            auto bias_scale_down = std::make_shared<ov::op::v1::Multiply>(bias, bias_scale_down_const);
            fc->input(2).replace_source_output(bias_scale_down->output(0));
        }

        std::shared_ptr<ov::Node> scale_up_const = (fc->get_element_type() == ov::element::f16) ? scale_up_const_f16 : scale_up_const_f32;
        auto scale_up = std::make_shared<ov::op::v1::Multiply>(fc, scale_up_const);
        ov::replace_node(fc, scale_up);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(fc_compressed_m, "StaticScaling");
    this->register_matcher(m, callback);
}

}  // namespace intel_gpu
}  // namespace ov
