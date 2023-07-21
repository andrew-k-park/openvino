// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "openvino/op/op.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace op {
namespace internal {

///
/// \brief RPE operation.
///
/// \ingroup ov_ops_cpp_api
class TRANSFORMATIONS_API RPE : public ov::op::Op {
public:
    OPENVINO_OP("RPE", "ie_internal_opset", op::Op);

    RPE() = default;
    RPE(const Output<Node>& data, const Output<Node>& sin, const Output<Node>& cos, const int64_t& axis);

    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

private:
    int64_t m_axis{};
};

}  // namespace internal
}  // namespace op
}  // namespace ov
