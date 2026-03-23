// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lora_horizontal_simple_fusion.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

#include "intel_gpu/op/fully_connected_compressed.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/variadic_split.hpp"

namespace {
void copy_runtime_info_from_outputs(const ov::OutputVector& from, const std::shared_ptr<ov::Node>& to) {
    ov::NodeVector nodes_from;
    for (const auto& output : from) {
        nodes_from.emplace_back(output.get_node_shared_ptr());
    }
    ov::copy_runtime_info(nodes_from, to);
}
}  // namespace

namespace ov::intel_gpu {

LoRAHorizontalSimpleFusion::LoRAHorizontalSimpleFusion() {
    using namespace ov::pass::pattern;

    // Validate that every consumer of the VariadicSplit is a LoRA Add pattern
    auto is_target_pattern = [](const std::shared_ptr<Node>& split_node) {
        auto is_lora_pattern = [](const std::shared_ptr<Node>& node) {
            #define check(node) if (!node) return false;

            const auto& add = ov::as_type_ptr<ov::op::v1::Add>(node);                                                         check(add)

            size_t matmul2_idx = ov::is_type<ov::op::v0::MatMul>(add->get_input_node_shared_ptr(0)) ? 0 : 1;
            const auto& matmul2 = ov::as_type_ptr<ov::op::v0::MatMul>(add->get_input_node_shared_ptr(matmul2_idx));           check(matmul2)

            const auto& multiply = ov::as_type_ptr<ov::op::v1::Multiply>(matmul2->get_input_node_shared_ptr(0));              check(multiply)

            const auto& variable_b = ov::as_type_ptr<ov::op::util::ReadValueBase>(matmul2->get_input_node_shared_ptr(1));     check(variable_b)

            size_t matmul1_idx = ov::is_type<ov::op::v0::MatMul>(multiply->get_input_node_shared_ptr(0)) ? 0 : 1;
            const auto& matmul1 = ov::as_type_ptr<ov::op::v0::MatMul>(multiply->get_input_node_shared_ptr(matmul1_idx));      check(matmul1)

            size_t alpha_idx = (matmul1_idx + 1) % 2;
            const auto& variable_alpha =
                ov::as_type_ptr<ov::op::util::ReadValueBase>(multiply->get_input_node_shared_ptr(alpha_idx));                 check(variable_alpha)

            const auto& variable_a = ov::as_type_ptr<ov::op::util::ReadValueBase>(matmul1->get_input_node_shared_ptr(1));     check(variable_a)

            #undef check
            return true;
        };

        for (const auto& user : split_node->get_users()) {
            if (!is_lora_pattern(user)) {
                return false;
            }
        }

        return true;
    };

    auto lora_input = any_input();
    auto main_flow_1 = wrap_type<op::FullyConnectedCompressed>({lora_input, any_input(), any_input(), any_input()});
    auto main_flow_2 = wrap_type<op::FullyConnectedCompressed>({lora_input, any_input(), any_input(), any_input(), any_input()});
    auto main_flow = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{main_flow_1, main_flow_2});

    auto axis_const = wrap_type<ov::op::v0::Constant>();
    auto split_const = wrap_type<ov::op::v0::Constant>();
    auto split = wrap_type<ov::op::v1::VariadicSplit>({main_flow, axis_const, split_const}, is_target_pattern);

    ov::matcher_pass_callback callback = [=](Matcher& m) {
        const auto& split = m.get_match_root();

        ov::NodeVector add_nodes;
        ov::OutputVector matmul2_outputs;

        // Collect Add nodes and their MatMul2 (up-projection) inputs in split output order
        for (size_t i = 0; i < split->get_output_size(); ++i) {
            // Each split output should have exactly one consumer (the Add)
            const auto& target_inputs = split->output(i).get_target_inputs();
            if (target_inputs.size() != 1)
                return false;

            const auto& add = target_inputs.begin()->get_node()->shared_from_this();
            add_nodes.emplace_back(add);

            size_t matmul2_idx = ov::is_type<ov::op::v0::MatMul>(add->get_input_node_shared_ptr(0)) ? 0 : 1;
            matmul2_outputs.emplace_back(add->input_value(matmul2_idx));
        }

        // Concat all MatMul2 outputs along the last dimension
        auto last_dim = matmul2_outputs[0].get_partial_shape().size() - 1;
        auto fused_matmul2_concat = std::make_shared<ov::op::v0::Concat>(matmul2_outputs, last_dim);
        fused_matmul2_concat->set_friendly_name(
            matmul2_outputs[0].get_node()->get_friendly_name() +
            "_fused_" + std::to_string(matmul2_outputs.size()) + "_MatMul2_outputs");
        copy_runtime_info_from_outputs(matmul2_outputs, fused_matmul2_concat);

        // Create a single Add: FC_output + Concat(matmul2_outputs)
        // This Add can later be absorbed as FC post-sum by the runtime
        auto fused_add = std::make_shared<ov::op::v1::Add>(split->input_value(0), fused_matmul2_concat);
        fused_add->set_friendly_name(
            add_nodes[0]->get_friendly_name() +
            "_fused_" + std::to_string(add_nodes.size()) + "_Adds");
        ov::copy_runtime_info(add_nodes, fused_add);

        // Rewire: redirect each old Add's consumers to the corresponding split output
        for (size_t i = 0; i < add_nodes.size(); ++i) {
            const auto& old_add = add_nodes[i];
            for (auto u : old_add->get_users()) {
                for (size_t idx = 0; idx < u->inputs().size(); ++idx) {
                    if (u->get_input_node_shared_ptr(idx) == old_add) {
                        u->input(idx).replace_source_output(split->output(i));
                    }
                }
            }
            old_add->clear_control_dependencies();
        }

        // Replace the VariadicSplit's input from FC_output to fused_add
        split->input(0).replace_source_output(fused_add->output(0));
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(split, "LoRAHorizontalSimpleFusion");
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
