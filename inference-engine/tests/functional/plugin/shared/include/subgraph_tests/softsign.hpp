// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/softsign.hpp"

namespace SubgraphTestsDefinitions {

TEST_P(SoftsignTest, CompareWithRefImpl) {
    Run();
};

}  // namespace SubgraphTestsDefinitions
