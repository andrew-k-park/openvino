// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/work_with_devices.hpp"

#include "common/utils.hpp"
#include "intel_npu/config/options.hpp"

namespace {

const std::vector<ov::AnyMap> configs = {
    {{ov::log::level(ov::log::Level::ERR)}},
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest,
                         TestCompiledModelNPU,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::ValuesIn(configs)),
                         TestCompiledModelNPU::getTestCaseName);

}  // namespace
