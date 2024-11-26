// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reshape_fusion.hpp"

#include "intel_gpu/op/kv_cache.hpp"
#include "intel_gpu/op/read_value.hpp"

#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov::pass::pattern;

namespace ov {
namespace intel_gpu {

ReshapeFusion::ReshapeFusion() {
    add_matcher<KVCacheReshapeMatcher>();
}

KVCacheReshapeMatcher::KVCacheReshapeMatcher() {
    auto present_m = any_input();
    auto transpose_present_order_m = wrap_type<ov::op::v0::Constant>(consumers_count(1));
    auto transpose_present_m = wrap_type<ov::op::v1::Transpose>({present_m, transpose_present_order_m});
    auto beam_idx_m = wrap_type<ov::op::v0::Parameter>();
    auto past_m = wrap_type<ov::intel_gpu::op::ReadValue>();
    auto gather_past_axis_m = wrap_type<ov::op::v0::Constant>(
        ov::op::util::constant_predicate<int64_t>([](const std::vector<int64_t>& value) -> bool {
            std::cout << "KVCacheReshapeMatcher | value.size()=" << value.size() << ", value[0]=" << value[0] << std::endl;
            return value.size() == 1 && (value[0] == 0 || value[0] == 1);
        }));
    auto gather_past_m = wrap_type<ov::op::v8::Gather>({past_m, beam_idx_m, gather_past_axis_m});
    auto kv_cache_m = wrap_type<ov::intel_gpu::op::KVCache>({gather_past_m, transpose_present_m});
    auto valid_concat_inputs = [](const Output<Node>& node) -> bool {
        const auto concat = node.get_node();
        size_t num_inputs = concat->get_input_size();
        if (num_inputs != 3)
            return false;

        const auto first_input = concat->get_input_node_ptr(0);
        if (!is_type<ov::op::v1::Multiply>(first_input))
            return false;

        const auto second_input = as_type<ov::op::v0::Constant>(concat->get_input_node_ptr(1));
        const auto third_input = as_type<ov::op::v0::Constant>(concat->get_input_node_ptr(2));
        if (!second_input || !third_input)
            return false;

        auto second_value = second_input->cast_vector<int64_t>()[0];
        auto third_value = third_input->cast_vector<int64_t>()[0];
        std::cout << "KVCacheReshapeMatcher | second_value=" << second_value << ", third_value=" << third_value << std::endl;

        return true;
    };
    auto concat_m = wrap_type<ov::op::v0::Concat>(valid_concat_inputs);
    auto reshape_m = wrap_type<ov::op::v1::Reshape>({kv_cache_m, concat_m});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto reshape = std::dynamic_pointer_cast<ov::op::v1::Reshape>(m.get_match_root());

        if (!reshape || transformation_callback(reshape)) {
            return false;
        }

        auto kv_cache = std::dynamic_pointer_cast<ov::intel_gpu::op::KVCache>(pattern_map.at(kv_cache_m).get_node_shared_ptr());
        std::cout << "KVCacheReshapeMatcher::callback | kv_cache=" << kv_cache->get_friendly_name()
                  << ", reshape->get_users()[0]->get_friendly_name()="
                  << reshape->get_users()[0]->get_friendly_name() << std::endl;
        auto kv_cache_new = std::make_shared<ov::intel_gpu::op::KVCache>(kv_cache->get_input_node_shared_ptr(0),
                                                                         kv_cache->get_input_node_shared_ptr(1),
                                                                         kv_cache->get_variable(),
                                                                         true,
                                                                         kv_cache->get_concat_axis(),
                                                                         kv_cache->get_output_element_type(0));
        kv_cache_new->set_friendly_name(kv_cache->get_friendly_name());
        ov::copy_runtime_info(reshape, kv_cache_new);
        ov::replace_node(reshape, kv_cache_new);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(reshape_m, "KVCacheReshapeMatcher");
    this->register_matcher(m, callback);
}

}  // namespace intel_gpu
}  // namespace ov
