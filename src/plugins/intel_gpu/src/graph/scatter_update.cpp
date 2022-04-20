// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "scatter_update_inst.h"

#include "primitive_type_base.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include <string>

namespace cldnn {
primitive_type_id scatter_update::type_id() {
    static primitive_type_base<scatter_update> instance;
    return &instance;
}

static size_t GetNonEmptyDimsNumber(const layout& layout) {
    if (layout.count() != 1) {
        // Count the number of "one size" dimensions starting with X to Batch
        size_t one_size_dims = 0;
        std::vector<int32_t> dims = layout.get_dims();
        for (size_t i = 0; i < dims.size(); i++) {
            if (dims[dims.size() - 1 - i] == 1)
                one_size_dims++;
            else
                break;
        }
        return dims.size() - one_size_dims;
    } else {
        return 1;
    }
}

layout scatter_update_inst::calc_output_layout(scatter_update_node const& node, kernel_impl_params const& impl_param) {
    auto desc = node.get_primitive();

    const int32_t axis = desc->axis;
    const size_t indices_size = impl_param.input_layouts.at(1).count();
    const size_t input_number_of_dims = impl_param.input_layouts.at(0).get_rank();
    const size_t updates_number_of_dims = impl_param.input_layouts.at(2).get_rank();

    auto input_layout = impl_param.input_layouts.at(0);

    auto output_shape = input_layout.size;
    auto input_format = input_layout.format;
    auto output_type = input_layout.data_type;

    if (node.has_fused_primitives()) {
        output_type = node.get_fused_output_layout().data_type;
    }

    if (axis >= input_number_of_dims)
        CLDNN_ERROR_MESSAGE(node.id(), "Incorrect axis value for ScatterUpdate: Axis must be positive and less than the input tensor dimension.");

    if (indices_size > static_cast<size_t>(input_layout.get_dims()[axis])) {
        CLDNN_ERROR_MESSAGE(node.id(),
            "Undefined behavior ScatterUpdate: indices size must not be larger than the output size along the Axis.");
    }

    if (axis > updates_number_of_dims) {
        CLDNN_ERROR_MESSAGE(node.id(),
            "Undefined behavior ScatterUpdate: indices dimention must not be larger than the updates[:Axis] dimentional size.");
    }

    return layout{output_type, input_format, output_shape};
}

std::string scatter_update_inst::to_string(scatter_update_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto& input = node.input();

    std::stringstream primitive_description;

    json_composite scatter_update_info;
    scatter_update_info.add("input id", input.id());
    scatter_update_info.add("axis", desc->axis);
    scatter_update_info.add("output shape", node.input(0).get_output_layout().to_string());

    node_info->add("scatter_update info", scatter_update_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

scatter_update_inst::typed_primitive_inst(network& network, scatter_update_node const& node) : parent(network, node) {}

}  // namespace cldnn
