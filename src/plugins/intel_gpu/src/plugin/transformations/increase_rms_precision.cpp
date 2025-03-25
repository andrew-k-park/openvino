// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "increase_rms_precision.hpp"

#include "intel_gpu/op/fully_connected_compressed.hpp"

#include "ov_ops/rms.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gelu.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::intel_gpu {

IncreaseRMSPrecision::IncreaseRMSPrecision() {
    using namespace ov::pass::pattern;

    // auto inputs_embeds_m = ov::pass::pattern::wrap_type<ov::op::v0::Parameter>(type_matches(element::f32));
    // auto convert_m = wrap_type<ov::op::v0::Convert>({inputs_embeds_m}, type_matches(element::f16));
    auto fc_o_proj_m = wrap_type<op::FullyConnectedCompressed>({any_input(), any_input(), any_input(), any_input()}, type_matches(element::f16));
    auto rms_post_attn_m = wrap_type<ov::op::internal::RMS>({fc_o_proj_m, wrap_type<ov::op::v0::Constant>()}, type_matches(element::f16));
    auto add_m = wrap_type<ov::op::v1::Add>({any_input(), rms_post_attn_m}, type_matches(element::f16));
    auto mul_m = wrap_type<ov::op::v1::Multiply>({any_input(), any_input()}, type_matches(element::f16));
    auto fc_down_proj_m = wrap_type<op::FullyConnectedCompressed>({mul_m, any_input(), any_input(), any_input()}, type_matches(element::f16));
    auto rms_post_ff_m = wrap_type<ov::op::internal::RMS>({fc_down_proj_m, wrap_type<ov::op::v0::Constant>()}, type_matches(element::f16));
    auto add_1_m = wrap_type<ov::op::v1::Add>({add_m, rms_post_ff_m}, type_matches(element::f16));

    auto rms_next_input_m = wrap_type<ov::op::internal::RMS>({add_1_m, wrap_type<ov::op::v0::Constant>()}, type_matches(element::f16));
    // auto fc_next_q_proj_m = wrap_type<op::FullyConnectedCompressed>({rms_next_input_m, any_input(), any_input(), any_input()}, type_matches(element::f16));
    auto add_next_m = wrap_type<ov::op::v1::Add>({add_1_m, any_input()}, type_matches(element::f16));
    // auto rms_next_pre_ff_m = wrap_type<ov::op::internal::RMS>({add_next_m, wrap_type<ov::op::v0::Constant>()}, type_matches(element::f16));
    // auto add_next_1_m = wrap_type<ov::op::v1::Add>({add_next_m, any_input()}, type_matches(element::f16));

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        std::cout << "IncreaseRMSPrecision::callback - START" << std::endl;
        auto add_1 = ov::as_type_ptr<ov::op::v1::Add>(m.get_match_root());
        if (!add_1 || transformation_callback(add_1)) {
            std::cout << "IncreaseRMSPrecision::callback | return -1" << std::endl;
            return false;
        }

        const auto desired_et = ov::element::f32;
        const auto original_et = add_1->get_output_element_type(0);
        if (original_et == desired_et) {
            std::cout << "IncreaseRMSPrecision::callback | return -2" << std::endl;
            return false;
        }

        // const auto& pattern_map = m.get_pattern_value_map();
        // auto fc_down_proj =  ov::as_type_ptr<op::FullyConnectedCompressed>(pattern_map.at(fc_down_proj_m).get_node_shared_ptr());
        // const auto mul = ov::as_type_ptr<ov::op::v1::Multiply>(fc_down_proj->get_input_node_shared_ptr(0));
        // if (!mul) {
        //     return false;
        // }
        // if (auto mul = ov::as_type_ptr<ov::op::v1::Multiply>(fc_down_proj->get_input_node_shared_ptr(0))) {
        //     std::cout << "IncreaseRMSPrecision::callback | mul=" << mul->get_friendly_name()
        //               << ", type=" << mul->get_autob().m_type << std::endl;
        // } else {
        //     return false;
        // }
        // const auto& add = pattern_map.at(add_m).get_node_shared_ptr();

        // auto data_const_convert = std::make_shared<ov::op::v0::Convert>(data_const, element::f32);
        std::cout << "IncreaseRMSPrecision::callback | add_1=" << add_1->get_friendly_name() << std::endl;

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(add_1_m, "IncreaseRMSPrecision");
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
