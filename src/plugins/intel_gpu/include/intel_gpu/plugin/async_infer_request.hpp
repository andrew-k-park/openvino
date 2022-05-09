// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <map>
#include <cpp_interfaces/impl/ie_infer_async_request_thread_safe_default.hpp>
#include "intel_gpu/plugin/infer_request_legacy.hpp"
#include "intel_gpu/plugin/infer_request.hpp"

namespace ov {
namespace runtime {
namespace intel_gpu {

class AsyncInferRequest : public InferenceEngine::AsyncInferRequestThreadSafeDefault {
public:
    using Parent = InferenceEngine::AsyncInferRequestThreadSafeDefault;
    AsyncInferRequest(const IInferRequestInternal::Ptr &inferRequest,
                      const InferenceEngine::ITaskExecutor::Ptr& taskExecutor,
                      const InferenceEngine::ITaskExecutor::Ptr& waitExecutor,
                      const InferenceEngine::ITaskExecutor::Ptr& callbackExecutor,
                      const bool isLegacy);

    ~AsyncInferRequest();

    void Infer_ThreadUnsafe() override;
    void StartAsync_ThreadUnsafe() override;

private:
    IInferRequestInternal::Ptr _inferRequest;
    InferenceEngine::ITaskExecutor::Ptr _waitExecutor;
    bool _isLegacy;
};

}  // namespace intel_gpu
}  // namespace runtime
}  // namespace ov
