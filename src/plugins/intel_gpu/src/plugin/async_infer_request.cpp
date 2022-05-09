// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/async_infer_request.hpp"
#include "intel_gpu/plugin/itt.hpp"
#include <memory>

namespace ov {
namespace runtime {
namespace intel_gpu {

AsyncInferRequest::AsyncInferRequest(const IInferRequestInternal::Ptr &inferRequest,
                                     const InferenceEngine::ITaskExecutor::Ptr& taskExecutor,
                                     const InferenceEngine::ITaskExecutor::Ptr& waitExecutor,
                                     const InferenceEngine::ITaskExecutor::Ptr& callbackExecutor,
                                     const bool isLegacy)
    : AsyncInferRequestThreadSafeDefault(inferRequest, taskExecutor, callbackExecutor),
      _inferRequest(inferRequest),
      _waitExecutor(waitExecutor),
      _isLegacy(isLegacy) {
    _pipeline = {};

    if (_isLegacy) {
        if (!std::static_pointer_cast<InferRequestLegacy>(_inferRequest)->use_external_queue()) {
            _pipeline.push_back({taskExecutor,
                        [this] {
                            OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "AsyncInferRequest::PreprocessingAndStartPipeline");
                            std::static_pointer_cast<InferRequestLegacy>(_inferRequest)->setup_stream_graph();
                            std::static_pointer_cast<InferRequestLegacy>(_inferRequest)->preprocess();
                            std::static_pointer_cast<InferRequestLegacy>(_inferRequest)->enqueue();
                            std::static_pointer_cast<InferRequestLegacy>(_inferRequest)->wait();
            } });
        } else {
            _pipeline.push_back({ _waitExecutor,
                            [this] {
                                OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "AsyncInferRequest::WaitPipeline");
                                std::static_pointer_cast<InferRequestLegacy>(_inferRequest)->wait_notify();
                            } });
        }
    } else {
        if (!std::static_pointer_cast<InferRequest>(_inferRequest)->use_external_queue()) {
            _pipeline.push_back({taskExecutor,
                        [this] {
                            OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "AsyncInferRequest::PreprocessingAndStartPipeline");
                            std::static_pointer_cast<InferRequest>(_inferRequest)->setup_stream_graph();
                            std::static_pointer_cast<InferRequest>(_inferRequest)->preprocess();
                            std::static_pointer_cast<InferRequest>(_inferRequest)->enqueue();
                            std::static_pointer_cast<InferRequest>(_inferRequest)->wait();
            } });
        } else {
            _pipeline.push_back({ _waitExecutor,
                            [this] {
                                OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "AsyncInferRequest::WaitPipeline");
                                std::static_pointer_cast<InferRequest>(_inferRequest)->wait_notify();
                            } });
        }
    }
}

void AsyncInferRequest::Infer_ThreadUnsafe() {
    if (_isLegacy) {
        if (std::static_pointer_cast<InferRequestLegacy>(_inferRequest)->use_external_queue()) {
            std::static_pointer_cast<InferRequestLegacy>(_inferRequest)->setup_stream_graph();
            std::static_pointer_cast<InferRequestLegacy>(_inferRequest)->preprocess_notify();
            std::static_pointer_cast<InferRequestLegacy>(_inferRequest)->enqueue_notify();
        }
    } else {
        if (std::static_pointer_cast<InferRequest>(_inferRequest)->use_external_queue()) {
            std::static_pointer_cast<InferRequest>(_inferRequest)->setup_stream_graph();
            std::static_pointer_cast<InferRequest>(_inferRequest)->preprocess_notify();
            std::static_pointer_cast<InferRequest>(_inferRequest)->enqueue_notify();
        }
    }
    Parent::Infer_ThreadUnsafe();
}

void AsyncInferRequest::StartAsync_ThreadUnsafe() {
    if (_isLegacy) {
        if (std::static_pointer_cast<InferRequestLegacy>(_inferRequest)->use_external_queue()) {
            std::static_pointer_cast<InferRequestLegacy>(_inferRequest)->setup_stream_graph();
            std::static_pointer_cast<InferRequestLegacy>(_inferRequest)->preprocess_notify();
            std::static_pointer_cast<InferRequestLegacy>(_inferRequest)->enqueue_notify();
        }
    } else {
        if (std::static_pointer_cast<InferRequest>(_inferRequest)->use_external_queue()) {
            std::static_pointer_cast<InferRequest>(_inferRequest)->setup_stream_graph();
            std::static_pointer_cast<InferRequest>(_inferRequest)->preprocess_notify();
            std::static_pointer_cast<InferRequest>(_inferRequest)->enqueue_notify();
        }
    }
    Parent::StartAsync_ThreadUnsafe();
}

AsyncInferRequest::~AsyncInferRequest() {
    StopAndWait();
}

}  // namespace intel_gpu
}  // namespace runtime
}  // namespace ov
