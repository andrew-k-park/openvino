// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_pricision_lrope.hpp"

#include "intel_gpu/op/gemm.hpp"

#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::intel_gpu {

ConvertPricisionLRoPE::ConvertPricisionLRoPE() {
    using namespace ov::pass::pattern;

    auto data_const_m = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto concat_m = wrap_type<ov::op::v0::Concat>({any_input(), any_input(), any_input()}, type_matches(element::i32));
    auto broadcast_m = wrap_type<ov::op::v3::Broadcast>({data_const_m, concat_m}, type_matches(element::f16));
    auto reshape_m = wrap_type<ov::op::v1::Reshape>({any_input(), wrap_type<ov::op::v0::Constant>()}, type_matches(element::i32));
    auto convert_m = wrap_type<ov::op::v0::Convert>({reshape_m}, type_matches(element::f16));
    auto gemm_m = wrap_type<op::Gemm>({broadcast_m, convert_m});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        auto gemm = ov::as_type_ptr<op::Gemm>(m.get_match_root());
        if (!gemm || transformation_callback(gemm))
            return false;

        // std::cout << "ConvertPricisionLRoPE::callback | name=" << gemm->get_friendly_name() << std::endl;

        const auto& pattern_map = m.get_pattern_value_map();
        const auto& data_const = pattern_map.at(data_const_m).get_node_shared_ptr();
        auto broadcast = ov::as_type_ptr<ov::op::v3::Broadcast>(pattern_map.at(broadcast_m).get_node_shared_ptr());
        auto convert =  ov::as_type_ptr<ov::op::v0::Convert>(pattern_map.at(convert_m).get_node_shared_ptr());
        auto data_const_convert = std::make_shared<ov::op::v0::Convert>(data_const, element::f32);
        data_const_convert->set_friendly_name(data_const->get_friendly_name() + "/Convert");
        broadcast->input(0).replace_source_output(data_const_convert);
        broadcast->set_output_type(0, element::f32, broadcast->get_output_partial_shape(0));
        convert->set_destination_type(element::f32);
        gemm->set_output_type(0, element::f32, gemm->get_output_partial_shape(0));

        auto target_inputs = gemm->get_output_target_inputs(0);
        auto gemm_convert = std::make_shared<ov::op::v0::Convert>(gemm, element::f16);
        gemm_convert->set_friendly_name(gemm->get_friendly_name() + "/Convert");
        ov::copy_runtime_info(gemm, gemm_convert);
        for (auto& in : target_inputs) {
            in.replace_source_output(gemm_convert);
        }

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(gemm_m, "ConvertPricisionLRoPE");
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
