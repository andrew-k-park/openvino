// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "move_fc_after_reshape.hpp"

#include "intel_gpu/op/fully_connected_compressed.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace ov {
namespace intel_gpu {

MoveFullyConnectedAfterReshape::MoveFullyConnectedAfterReshape() {
    using namespace ov::pass::pattern;

    auto one_consumer_rank_2 = [](const ov::Output<ov::Node>& out) {
        return consumers_count(1)(out) && rank_equals(2)(out);
    };
    auto pre_reshape = wrap_type<ov::op::v1::Reshape>({any_input(), any_input()}, one_consumer_rank_2);
    auto weights = any_input();
    auto scale = any_input();
    auto zp = any_input();
    auto fully_connected_compressed = wrap_type<op::FullyConnectedCompressed>({pre_reshape, weights, scale, zp}, consumers_count(1));
    auto one_consumer_rank_3 = [](const ov::Output<ov::Node>& out) {
        return consumers_count(1)(out) && rank_equals(3)(out);
    };
    auto post_reshape_sp = any_input();
    auto post_reshape = wrap_type<ov::op::v1::Reshape>({fully_connected_compressed, post_reshape_sp}, one_consumer_rank_3);
    auto add = wrap_type<ov::op::v1::Add>({any_input(), post_reshape});

    ov::matcher_pass_callback callback = [=](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        const auto& m_pre_reshape = pattern_map.at(pre_reshape).get_node_shared_ptr();
        const auto& m_weights = pattern_map.at(weights).get_node_shared_ptr();
        const auto& m_scale = pattern_map.at(weights).get_node_shared_ptr();
        const auto& m_zp = pattern_map.at(zp).get_node_shared_ptr();
        const auto& m_fc = pattern_map.at(fully_connected_compressed).get_node_shared_ptr();
        const auto& m_post_reshape_sp = pattern_map.at(post_reshape_sp).get_node_shared_ptr();
        auto m_post_reshape = std::dynamic_pointer_cast<ov::op::v1::Reshape>(pattern_map.at(post_reshape).get_node_shared_ptr());
        const auto& m_add = pattern_map.at(add).get_node_shared_ptr();
        // std::cout << "MoveFullyConnectedAfterReshape | m_fc=" << m_fc->get_friendly_name()
        //           << ", pre reshape's consumers cnt =" << m_pre_reshape->output(0).get_target_inputs().size()
        //           << ", pre reshape's output rank=" << m_pre_reshape->output(0).get_partial_shape().rank()
        //           << ", post reshape's consumers cnt =" << m_post_reshape->output(0).get_target_inputs().size()
        //           << ", post reshape's output rank=" << m_post_reshape->output(0).get_partial_shape().rank()
        //           << ", add's consumers cnt =" << m_add->output(0).get_target_inputs().size()
        //           << std::endl;

        auto new_post_reshape = std::make_shared<ov::op::v1::Reshape>(m_fc->input_value(0),
                                                                      m_post_reshape->input_value(1),
                                                                      m_post_reshape->get_special_zero());
        new_post_reshape->set_friendly_name(m_post_reshape->get_friendly_name());
        ov::copy_runtime_info({m_pre_reshape, m_post_reshape_sp}, new_post_reshape);
        ov::replace_node(m_post_reshape, new_post_reshape);

        auto new_fc = std::make_shared<op::FullyConnectedCompressed>(new_post_reshape,
                                                                     m_weights,
                                                                     m_scale,
                                                                     m_zp,
                                                                     m_fc->get_output_element_type(0));
        new_fc->set_friendly_name(m_fc->get_friendly_name());
        ov::copy_runtime_info({new_post_reshape, m_weights, m_scale, zp}, new_fc);
        ov::replace_node(m_fc, new_fc);
        // for (const auto& output_port : m_pre_reshape->outputs()) {
        //     std::cout << output_port->get_node()->get_friendly_name() << std::endl;
        // }

        // m_pre_reshape->clear_control_dependencies();
        // m_post_reshape->input(0).replace_source_output(m_pre_reshape->output(0));

        // m_fc->clear_control_dependencies();
        // const auto& post_reshape_target_inputs = m_post_reshape->get_output_target_inputs(0);
        // for (const auto& target_input : post_reshape_target_inputs) {
        //     target_input.replace_source_output(m_fc->output(0));
        // }

        // m_post_reshape->clear_control_dependencies();
        // m_fc->input(0).replace_source_output(m_post_reshape->output(0));

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(add);
    this->register_matcher(m, callback);
}

}  // namespace intel_gpu
}  // namespace ov
