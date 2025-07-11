// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/port_descriptor.hpp"

#include <cassert>
#include <cstddef>
#include <memory>
#include <snippets/utils/utils.hpp>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_input.hpp"
#include "openvino/core/node_output.hpp"
#include "snippets/emitter.hpp"
#include "snippets/shape_types.hpp"

namespace ov::snippets::lowered {

PortDescriptor::PortDescriptor(const ov::Input<ov::Node>& in, VectorDims subtensor_shape, std::vector<size_t> layout)
    : PortDescriptor(ov::Input<const Node>(in.get_node(), in.get_index()),
                     std::move(subtensor_shape),
                     std::move(layout)) {}

PortDescriptor::PortDescriptor(const ov::Input<const ov::Node>& in,
                               std::vector<size_t> subtensor_shape,
                               std::vector<size_t> layout)
    : PortDescriptor(utils::pshape_to_vdims(in.get_partial_shape()), std::move(subtensor_shape), std::move(layout)) {}

PortDescriptor::PortDescriptor(const ov::Output<ov::Node>& out, VectorDims subtensor_shape, std::vector<size_t> layout)
    : PortDescriptor(ov::Output<const Node>(out.get_node(), out.get_index()),
                     std::move(subtensor_shape),
                     std::move(layout)) {}

PortDescriptor::PortDescriptor(const ov::Output<const ov::Node>& out,
                               std::vector<size_t> subtensor_shape,
                               std::vector<size_t> layout)
    : PortDescriptor(utils::pshape_to_vdims(out.get_partial_shape()), std::move(subtensor_shape), std::move(layout)) {}

PortDescriptor::PortDescriptor(VectorDims shape, VectorDims subtensor_shape, std::vector<size_t> layout, Reg reg)
    : PortDescriptor(std::make_shared<VectorDims>(std::move(shape)),
                     std::move(subtensor_shape),
                     std::move(layout),
                     reg) {}

PortDescriptor::PortDescriptor(VectorDimsPtr shape, VectorDims subtensor_shape, std::vector<size_t> layout, Reg reg)
    : m_tensor_shape(std::move(shape)),
      m_layout(std::move(layout)),
      m_subtensor_shape(std::move(subtensor_shape)),
      m_reg(reg) {
    validate_arguments();
}

PortDescriptor::PortDescriptor() : PortDescriptor(VectorDims(), {}, {}) {}  // to avoid tensor_shape = nullptr

void PortDescriptor::validate_arguments() {
    OPENVINO_ASSERT(m_tensor_shape, "Tensor Shape is nullptr");
    if (!m_tensor_shape->empty() && m_layout.empty()) {
        m_layout = utils::get_planar_layout(m_tensor_shape->size());
    }
    OPENVINO_ASSERT(m_subtensor_shape.size() <= m_tensor_shape->size(),
                    "Snippets tensor descriptor: Subtensor shape must be less than or equal to tensor shape");
    OPENVINO_ASSERT(m_layout.size() == m_tensor_shape->size(),
                    "Snippets tensor descriptor: Layout size must be equal to the shape size");
}

const VectorDims& PortDescriptor::get_shape() const {
    assert(m_tensor_shape && "Failed to get_shape: Tensor Shape is nullptr");
    return *m_tensor_shape;
}

void PortDescriptor::set_shape(const VectorDims& tensor) {
    OPENVINO_ASSERT(m_tensor_shape, "Failed to set_shape: Tensor Shape is nullptr");
    OPENVINO_ASSERT(m_subtensor_shape.size() <= tensor.size(),
                    "Snippets tensor descriptor: Subtensor shape must be less than or equal to tensor shape");
    *m_tensor_shape = tensor;
}

void PortDescriptor::set_layout(const std::vector<size_t>& layout) {
    OPENVINO_ASSERT(layout.size() == m_tensor_shape->size(),
                    "Snippets tensor descriptor: Layout size must be equal to the shape size");
    m_layout = layout;
}
void PortDescriptor::set_subtensor(const VectorDims& subtensor) {
    OPENVINO_ASSERT(subtensor.size() <= m_tensor_shape->size(),
                    "Subtensor shape must be less than or equal to tensor shape");
    m_subtensor_shape = subtensor;
}

void PortDescriptor::set_reg(Reg reg) {
    OPENVINO_ASSERT(m_reg.type != RegType::address, "Failed to set reg: reg with 'address' type mustn't be changed");
    m_reg = reg;
}

void PortDescriptor::set_reg_type(RegType type) {
    OPENVINO_ASSERT(m_reg.type != RegType::address, "Failed to set reg type: address reg type mustn't be changed");
    m_reg.type = type;
}

void PortDescriptor::set_subtensor_dim(size_t idx, VectorDims::value_type value) {
    OPENVINO_ASSERT(idx < m_subtensor_shape.size(), "Failed to set subtensor value: idx should be less than size");
    *(m_subtensor_shape.rbegin() + idx) = value;
}

PortDescriptorPtr PortDescriptor::clone() const {
    auto desc = std::make_shared<PortDescriptor>(*m_tensor_shape, m_subtensor_shape, m_layout);
    desc->set_reg(m_reg);
    return desc;
}

std::string PortDescriptor::serialize() const {
    std::stringstream ss;
    OPENVINO_ASSERT(m_tensor_shape, "TensorShape is nullptr!");
    ss << m_tensor_shape->size() << " ";
    for (auto val : *m_tensor_shape) {
        ss << val << " ";
    }
    ss << m_subtensor_shape.size() << " ";
    for (auto val : m_subtensor_shape) {
        ss << val << " ";
    }
    ss << m_layout.size() << " ";
    for (auto val : m_layout) {
        ss << val << " ";
    }
    ss << m_reg;
    return ss.str();
}
bool operator==(const PortDescriptor& lhs, const PortDescriptor& rhs) {
    return lhs.m_tensor_shape == rhs.m_tensor_shape && lhs.m_layout == rhs.m_layout &&
           lhs.m_subtensor_shape == rhs.m_subtensor_shape && lhs.m_reg == rhs.m_reg;
}

void PortDescriptorUtils::init_default(std::vector<PortDescriptorPtr>& in_descs,
                                       std::vector<PortDescriptorPtr>& out_descs,
                                       const std::shared_ptr<ov::Node>& node) {
    in_descs.resize(node->get_input_size());
    out_descs.resize(node->get_output_size());
    for (size_t i = 0; i < node->get_input_size(); ++i) {
        in_descs[i] = std::make_shared<PortDescriptor>(node->input(i));
    }
    for (size_t i = 0; i < node->get_output_size(); ++i) {
        out_descs[i] = std::make_shared<PortDescriptor>(node->output(i));
    }
}

void PortDescriptorUtils::set_port_descriptor_ptr(const ov::Input<ov::Node>& in, const PortDescriptorPtr& desc) {
    const auto& node = in.get_node()->shared_from_this();
    auto& rt_info = node->get_rt_info();
    const auto& key = PortDescriptorVectorAttribute::get_type_info_static();
    const auto& found = rt_info.find(key);
    if (found == rt_info.end()) {
        std::vector<PortDescriptorPtr> in_descs, out_descs;
        init_default(in_descs, out_descs, node);
        in_descs[in.get_index()] = desc;
        rt_info[key] = PortDescriptorVectorAttribute(in_descs, out_descs);
    } else {
        auto& in_descs = found->second.as<PortDescriptorVectorAttribute>().inputs;
        if (in_descs.size() != node->get_input_size()) {
            OPENVINO_THROW("Set input port descriptor is failed: incorrect count");
        }
        in_descs[in.get_index()] = desc;
    }
}

void PortDescriptorUtils::set_port_descriptor_ptr(const ov::Output<ov::Node>& out, const PortDescriptorPtr& desc) {
    const auto& node = out.get_node_shared_ptr();
    auto& rt_info = node->get_rt_info();
    const auto& key = PortDescriptorVectorAttribute::get_type_info_static();
    const auto& found = rt_info.find(key);
    if (found == rt_info.end()) {
        std::vector<PortDescriptorPtr> in_descs, out_descs;
        init_default(in_descs, out_descs, node);
        out_descs[out.get_index()] = desc;
        rt_info[key] = PortDescriptorVectorAttribute(in_descs, out_descs);
    } else {
        auto& out_descs = found->second.as<PortDescriptorVectorAttribute>().outputs;
        if (out_descs.size() != node->get_output_size()) {
            OPENVINO_THROW("Set output port descriptor is failed: incorrect count");
        }
        out_descs[out.get_index()] = desc;
    }
}

namespace {
template <typename T>
void set_port_desc(const T& port, std::vector<size_t> subtensor, std::vector<size_t> layout) {
    const auto& shape = utils::pshape_to_vdims(port.get_partial_shape());
    for (size_t i = 1; i <= std::min(subtensor.size(), shape.size()); i++) {
        auto& dim = subtensor[subtensor.size() - i];
        if (!utils::is_full_dim_value(dim)) {
            dim = std::min(dim, shape[shape.size() - i]);
        }
    }
    PortDescriptorUtils::set_port_descriptor_ptr(port, std::make_shared<PortDescriptor>(shape, subtensor, layout));
}
}  // namespace

void PortDescriptorUtils::set_port_descriptor(const ov::Input<ov::Node>& in,
                                              std::vector<size_t> subtensor,
                                              std::vector<size_t> layout) {
    set_port_desc(in, std::move(subtensor), std::move(layout));
}
void PortDescriptorUtils::set_port_descriptor(const ov::Output<ov::Node>& out,
                                              std::vector<size_t> subtensor,
                                              std::vector<size_t> layout) {
    set_port_desc(out, std::move(subtensor), std::move(layout));
}

PortDescriptorPtr PortDescriptorUtils::get_port_descriptor_ptr(const ov::Input<ov::Node>& in) {
    return get_port_descriptor_ptr(ov::Input<const Node>(in.get_node(), in.get_index()));
}
PortDescriptorPtr PortDescriptorUtils::get_port_descriptor_ptr(const ov::Input<const ov::Node>& in) {
    const auto& node = in.get_node();
    const auto& rt_info = node->get_rt_info();
    const auto& key = PortDescriptorVectorAttribute::get_type_info_static();
    const auto& found = rt_info.find(key);
    if (found == rt_info.end()) {
        return std::make_shared<PortDescriptor>(in);
    }
    const auto& in_descs = found->second.as<PortDescriptorVectorAttribute>().inputs;
    if (in_descs.size() != node->get_input_size()) {
        OPENVINO_THROW("Get input port descriptor is failed: incorrect count");
    }
    return in_descs[in.get_index()];
}

PortDescriptorPtr PortDescriptorUtils::get_port_descriptor_ptr(const Output<ov::Node>& out) {
    return get_port_descriptor_ptr(ov::Output<const Node>(out.get_node(), out.get_index()));
}
PortDescriptorPtr PortDescriptorUtils::get_port_descriptor_ptr(const Output<const ov::Node>& out) {
    const auto& node = out.get_node();
    const auto& rt_info = node->get_rt_info();
    const auto& key = PortDescriptorVectorAttribute::get_type_info_static();
    const auto& found = rt_info.find(key);
    if (found == rt_info.end()) {
        return std::make_shared<PortDescriptor>(out);
    }
    const auto& out_descs = found->second.as<PortDescriptorVectorAttribute>().outputs;
    if (out_descs.size() != node->get_output_size()) {
        OPENVINO_THROW("Get output port descriptor is failed: incorrect count");
    }
    return out_descs[out.get_index()];
}

void PortDescriptorUtils::set_address_reg_type(const ov::Input<ov::Node>& in) {
    auto desc = get_port_descriptor_ptr(in);
    desc->set_reg_type(RegType::address);
    set_port_descriptor_ptr(in, desc);
}

void PortDescriptorUtils::set_address_reg_type(const ov::Output<ov::Node>& out) {
    auto desc = get_port_descriptor_ptr(out);
    desc->set_reg_type(RegType::address);
    set_port_descriptor_ptr(out, desc);
}

void PortDescriptorUtils::clean(const std::shared_ptr<ov::Node>& node) {
    auto& rt_info = node->get_rt_info();
    rt_info.erase(PortDescriptorVectorAttribute::get_type_info_static());
}
}  // namespace ov::snippets::lowered
