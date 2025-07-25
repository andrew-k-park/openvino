// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "nodes/executors/dnnl/dnnl_matmul_primitive.hpp"

#include <oneapi/dnnl/dnnl_types.h>

#include <algorithm>
#include <cassert>
#include <common/primitive_attr.hpp>
#include <common/primitive_hashing_utils.hpp>
#include <common/utils.hpp>
#include <cstddef>
#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <utility>
#include <vector>

#include "cpu_memory.h"
#include "cpu_types.h"
#include "dnnl_extension_utils.h"
#include "dnnl_postops_composer.h"
#include "dnnl_utils.hpp"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "memory_desc/dnnl_memory_desc.h"
#include "nodes/executors/dnnl/dnnl_aliases.hpp"
#include "nodes/executors/dnnl/dnnl_shape_agnostic_data.hpp"
#include "nodes/executors/executor.hpp"
#include "nodes/executors/fullyconnected_config.hpp"
#include "nodes/executors/matmul_config.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "onednn/iml_type_mapper.h"
#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"
#include "post_ops.hpp"
#include "utils/cpu_utils.hpp"
#include "utils/debug_capabilities.h"
#include "utils/general_utils.h"

#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
#    include <cpu/x64/cpu_isa_traits.hpp>
#endif

namespace ov::intel_cpu {

using namespace dnnl;
using namespace ov::element;
using namespace executor;

// @todo rewrite using hash_builder
size_t DnnlMatMulPrimitive::Key::hash() const {
    using namespace dnnl::impl;
    using namespace dnnl::impl::primitive_hashing;

    size_t seed = 0;

    for (const auto& ptr : {src, wei, bias, dst}) {
        if (ptr) {
            seed = hash_combine(seed, get_md_hash(*ptr->getDnnlDesc().get()));
        }
    }

    seed = hash_combine(seed, get_attr_hash(*attr.get()));

    return seed;
}

bool DnnlMatMulPrimitive::Key::operator==(const Key& rhs) const {
    bool result = true;

    if (src != rhs.src) {
        result = result && src && rhs.src && src->getDnnlDesc() == rhs.src->getDnnlDesc();
    }
    if (wei != rhs.wei) {
        result = result && wei && rhs.wei && wei->getDnnlDesc() == rhs.wei->getDnnlDesc();
    }
    if (bias != rhs.bias) {
        result = result && bias && rhs.bias && bias->getDnnlDesc() == rhs.bias->getDnnlDesc();
    }
    if (dst != rhs.dst) {
        result = result && dst && rhs.dst && dst->getDnnlDesc() == rhs.dst->getDnnlDesc();
    }

    result = result && *attr.get() == *rhs.attr.get();

    return result;
}

template <typename dimsType>
static dimsType normalizeToRank(const dimsType& vec, size_t rank) {
    if (vec.size() == rank || vec.empty()) {
        return vec;
    }

    dimsType result;
    result.reserve(rank);

    for (size_t i = vec.size(); i < rank; ++i) {
        result.push_back(1);
    }

    result.insert(result.end(), vec.begin(), vec.end());

    return result;
}

std::shared_ptr<DnnlMatMulPrimitive> DnnlMatMulPrimitive::create(const MemoryArgs& memory,
                                                                 [[maybe_unused]] const MatMulAttrs& attrs,
                                                                 const ExecutorContext::CPtr context,
                                                                 const DnnlShapeAgnosticDataPtr& shapeAgnosticData) {
    const auto& srcDesc = MemoryDescUtils::convertToDnnlMemoryDesc(memory.at(ARG_SRC)->getDescPtr());
    const auto& weiDesc = MemoryDescUtils::convertToDnnlMemoryDesc(memory.at(ARG_WEI)->getDescPtr());
    const auto& biaDesc = MemoryDescUtils::convertToDnnlMemoryDesc(memory.at(ARG_BIAS)->getDescPtr());
    const auto& dstDesc = MemoryDescUtils::convertToDnnlMemoryDesc(memory.at(ARG_DST)->getDescPtr());

    Key dnnlMatMulKey{srcDesc, weiDesc, biaDesc, dstDesc, shapeAgnosticData->m_primAttrs.attr};

    auto builder = [&context](const Key& dnnlKey) {
        return std::make_shared<DnnlMatMulPrimitive>(dnnlKey, context->getEngine(), context->getImplPriorities());
    };

    auto runtimeCache = context->getRuntimeCache();
    const auto result = runtimeCache->getOrCreate(dnnlMatMulKey, builder);
    const auto& primitive = result.first;
    assert(primitive);

    return primitive;
}

DnnlMemoryDescPtr DnnlMatMulPrimitive::makeTransposedWeightDescriptor(const DnnlMemoryDescPtr& srcDesc,
                                                                      const DnnlMemoryDescPtr& dstDesc,
                                                                      bool weightsNonTransposed) {
    const auto& weiDesc = srcDesc->getDnnlDesc();
    auto wDims = weiDesc.get_dims();
    auto wDataType = weiDesc.get_data_type();
    std::swap(wDims[wDims.size() - 1], wDims[wDims.size() - 2]);
    dnnl::memory::dims wDims2D = reshapeDownToRank<2>(wDims);

    const auto format = weightsNonTransposed ? dnnl::memory::format_tag::ab : dnnl::memory::format_tag::ba;
    const auto transposedWeiDesc = dnnl::memory::desc{wDims2D, wDataType, format};
    const auto reshapedWeiDesc = transposedWeiDesc.reshape(dstDesc->getDnnlDesc().get_dims());

    return DnnlExtensionUtils::makeDescriptor(reshapedWeiDesc);
}

static DnnlPrimitiveAttrs createPrimitiveAttrs(const PostOps& postOps,
                                               const MemoryArgs& memory,
                                               const ExecutorContext::CPtr& context,
                                               bool useWeightsDecompression,
                                               bool weightsNonTransposed) {
    const auto& srcDesc = memory.at(ARG_SRC)->getDescPtr();
    const auto& weiDesc = memory.at(ARG_WEI)->getDescPtr();
    const auto& dstDesc = memory.at(ARG_DST)->getDescPtr();

    const auto& originalDims = dstDesc->getShape().getMinDims();
    const auto& dims = originalDims;

    auto isINT8 =
        any_of(srcDesc->getPrecision(), ov::element::u8, ov::element::i8) && weiDesc->getPrecision() == ov::element::i8;
    auto outputDataType = DnnlExtensionUtils::ElementTypeToDataType(dstDesc->getPrecision());

    DnnlPostOpsComposer
        dnnlpoc(postOps, context->getEngine(), dims, dims.size() - 1, isINT8, 1 << 0, memory, outputDataType);

    const auto maxRank =
        std::max({srcDesc->getShape().getRank(), weiDesc->getShape().getRank(), dstDesc->getShape().getRank()});
    const auto normWeiDims = normalizeToRank(weiDesc->getShape().getStaticDims(), maxRank);
    if (auto it = memory.find(ARG_WEI | ARG_ATTR_SCALES); it != memory.end()) {
        auto dstPrc = ov::element::f32;
        dnnlpoc.appendDecompressionScales(it->second, !weightsNonTransposed, dstPrc, normWeiDims);
    }
    if (auto it = memory.find(ARG_WEI | ARG_ATTR_ZERO_POINTS); it != memory.end()) {
        // TODO: clarify oneDNN requirements on ZP precision
        auto zp = it->second;
        auto zpPrc = zp->getPrecision();
        auto dstPrc = any_of(zpPrc, i32, i8, u8, i4, u4) ? zpPrc : i32;
        dnnlpoc.appendDecompressionZeroPoints(zp, !weightsNonTransposed, dstPrc, normWeiDims);
    }

    auto primAttrs = dnnlpoc.compose();
    if (useWeightsDecompression) {
        primAttrs.attr.set_fpmath_mode(fpmath_mode::any, true);
    }

    return primAttrs;
}

static dnnl::matmul::primitive_desc createDescriptorInternal(const dnnl::memory::desc& inputDesc,
                                                             const dnnl::memory::desc& weightDesc,
                                                             const dnnl::memory::desc& biasDesc,
                                                             const dnnl::memory::desc& outputDesc,
                                                             const dnnl::primitive_attr& attr,
                                                             const dnnl::engine& engine,
                                                             const bool useWeightsDecompression) {
    auto weiDims = weightDesc.get_dims();
    std::swap(weiDims[weiDims.size() - 1], weiDims[weiDims.size() - 2]);

    const auto maxRank =
        std::max({inputDesc.get_ndims(), weightDesc.get_ndims(), biasDesc.get_ndims(), outputDesc.get_ndims()});
    const auto inpDims = normalizeToRank(inputDesc.get_dims(), maxRank);
    const auto biaDims = normalizeToRank(biasDesc.get_dims(), maxRank);
    const auto outDims = normalizeToRank(outputDesc.get_dims(), maxRank);
    weiDims = normalizeToRank(weiDims, maxRank);

    const dnnl::memory::desc inputsDesc = inputDesc.reshape(inpDims);
    const dnnl::memory::desc outputsDesc = outputDesc.reshape(outDims);
    auto newBiasDesc = !biasDesc.is_zero() ? biasDesc.reshape(biaDims) : biasDesc;

    auto idt = inputDesc.get_data_type();
    auto wdt = idt;
    if (useWeightsDecompression) {
        wdt = weightDesc.get_data_type();
    } else if (any_of(idt, dnnl::memory::data_type::u8, dnnl::memory::data_type::s8)) {
        wdt = memory::data_type::s8;
    }

    const dnnl::memory::desc weightsDesc = dnnl::memory::desc(weiDims, wdt, memory::format_tag::any);

    return {engine, inputsDesc, weightsDesc, newBiasDesc, outputsDesc, attr};
}

static primitive_desc createPrimitiveDesc(const dnnl::memory::desc& inputDesc,
                                          const dnnl::memory::desc& weightDesc,
                                          const dnnl::memory::desc& biasDesc,
                                          const dnnl::memory::desc& outputDesc,
                                          const dnnl::primitive_attr& attr,
                                          const dnnl::engine& engine,
                                          const std::vector<impl_desc_type>& implPriorities,
                                          [[maybe_unused]] const bool useSparseWeights,
                                          const bool useWeightsDecompression) {
    auto prim_desc =
        createDescriptorInternal(inputDesc, weightDesc, biasDesc, outputDesc, attr, engine, useWeightsDecompression);
    OPENVINO_ASSERT(prim_desc, "Failed to create matmul primitive descriptor");
    auto first_desc = dnnl::matmul::primitive_desc(prim_desc.get());

    const bool found = DnnlExtensionUtils::find_implementation(prim_desc, [&](impl_desc_type implType) {
        return any_of_values(implPriorities, implType);
    });

    if (found) {
        return std::move(prim_desc);
    }

    return std::move(first_desc);
}

static VectorDims makeDummyInputDims(const Shape& inShape, const Shape& wShape) {
    const auto& weightDims = wShape.getStaticDims();

    auto inMinDims = inShape.getMinDims();
    auto inMaxDims = inShape.getMaxDims();
    inMinDims.back() = weightDims.back();
    inMaxDims.back() = weightDims.back();

    return MemoryDescUtils::makeDummyShape(Shape(inMinDims, inMaxDims)).getStaticDims();
}

static VectorDims makeDummyOutputDims(const VectorDims& inShape, const VectorDims& wShape, const size_t out_rank) {
    size_t activationRank = inShape.size();
    size_t channelRank = wShape.size() - 1;
    // activation   weight    output_shape
    // NCHW         CoCHW     NCo
    // TNC          CoC       TNCo
    // NC           CoC       NCo
    VectorDims outputShape(out_rank, 1);
    // set Co
    outputShape.back() = wShape[0];
    // set batch dims
    size_t batchRank = activationRank - channelRank;
    size_t startIdx = out_rank - batchRank - 1;
    for (size_t i = 0; i < batchRank; i++) {
        outputShape[i + startIdx] = inShape[i];
    }

    return outputShape;
}

bool DnnlMatMulPrimitive::useWeightsDecompressionImpl(const ov::element::Type inputType,
                                                      const ov::element::Type weightsType) {
#if defined(OPENVINO_ARCH_X86_64)
    if (!dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::avx2)) {
        return false;
    }
#endif

    return (any_of(inputType, f32, bf16, f16) && any_of(weightsType, u8, i8, u4, i4));
}

DnnlShapeAgnosticDataPtr DnnlMatMulPrimitive::createShapeAgnosticData(const FCAttrs& attrs,
                                                                      const MemoryArgs& memory,
                                                                      const ExecutorContext::CPtr& context,
                                                                      const bool cacheWeights) {
    DEBUG_LOG("Creating shape agnostic data");
    auto srcDesc = memory.at(ARG_SRC)->getDescPtr();
    const auto& weiDesc = memory.at(ARG_WEI)->getDescPtr();
    const auto& biasDesc = memory.at(ARG_BIAS)->getDescPtr();
    auto dstDesc = memory.at(ARG_DST)->getDescPtr();

    const auto useWeightsDecompression = useWeightsDecompressionImpl(srcDesc->getPrecision(), weiDesc->getPrecision());
    const auto postOpData =
        createPrimitiveAttrs(attrs.postOps, memory, context, useWeightsDecompression, attrs.weightsNonTransposed);

    if (!cacheWeights) {
        return std::make_shared<DnnlShapeAgnosticData>(postOpData);
    }

    if (srcDesc->getShape().isDynamic()) {
        const auto& inShape = srcDesc->getShape();
        const auto& wShape = weiDesc->getShape();
        const auto& inDymmyDims = makeDummyInputDims(inShape, wShape);
        srcDesc = srcDesc->cloneWithNewDims(inDymmyDims);
        const auto& outDymmyDims =
            makeDummyOutputDims(inDymmyDims, wShape.getStaticDims(), dstDesc->getShape().getRank());
        dstDesc = dstDesc->cloneWithNewDims(outDymmyDims);
    }

    const dnnl::memory::desc srcDnnlDesc = MemoryDescUtils::convertToDnnlMemoryDesc(srcDesc)->getDnnlDesc();
    const dnnl::memory::desc weiDnnlDesc = MemoryDescUtils::convertToDnnlMemoryDesc(weiDesc)->getDnnlDesc();
    const dnnl::memory::desc dstDnnlDesc = MemoryDescUtils::convertToDnnlMemoryDesc(dstDesc)->getDnnlDesc();
    const dnnl::memory::desc biaDnnlDesc = MemoryDescUtils::convertToDnnlMemoryDesc(biasDesc)->getDnnlDesc();

    const auto primDesc = createPrimitiveDesc(srcDnnlDesc,
                                              weiDnnlDesc,
                                              biaDnnlDesc,
                                              dstDnnlDesc,
                                              postOpData.attr,
                                              context->getEngine(),
                                              context->getImplPriorities(),
                                              false,
                                              useWeightsDecompression);

    const auto weightsDesc = DnnlExtensionUtils::makeDescriptor(primDesc.weights_desc());
    auto originalWeightsDesc = MemoryDescUtils::convertToDnnlMemoryDesc(weiDesc);
    originalWeightsDesc = makeTransposedWeightDescriptor(originalWeightsDesc, weightsDesc, attrs.weightsNonTransposed);

    // ignore the result since we just need to put the packed weights into the cache
    (void)utils::prepareWeightsMemory(originalWeightsDesc, weightsDesc, memory.at(ARG_WEI), context);

    return std::make_shared<DnnlShapeAgnosticData>(postOpData);
}

static impl_desc_type implTypeFromPrimDesc(const dnnl::primitive_desc& primDesc) {
    const auto implType = parse_impl_name(primDesc.impl_info_str());
    if (implType == ov::intel_cpu::brgemm_avx512_amx &&
        primDesc.weights_desc().get_format_kind() == memory::format_kind::sparsed) {
        return ov::intel_cpu::brgemm_sparse_avx512_amx;
    }

    return implType;
}

DnnlMatMulPrimitive::DnnlMatMulPrimitive(const Key& key,
                                         const dnnl::engine& engine,
                                         const std::vector<impl_desc_type>& implPriorities)
    : m_stream(dnnl::stream(engine)),
      m_primDesc(createPrimitiveDesc(key.src->getDnnlDesc(),
                                     key.wei->getDnnlDesc(),
                                     key.bias->getDnnlDesc(),
                                     key.dst->getDnnlDesc(),
                                     key.attr,
                                     engine,
                                     implPriorities,
                                     false,
                                     useWeightsDecompressionImpl(key.src->getPrecision(), key.wei->getPrecision()))),
      m_implType(implTypeFromPrimDesc(m_primDesc)),
      m_srcDesc(DnnlExtensionUtils::makeDescriptor(m_primDesc.src_desc())),
      m_weiDesc(DnnlExtensionUtils::makeDescriptor(m_primDesc.weights_desc())),
      m_dstDesc(DnnlExtensionUtils::makeDescriptor(m_primDesc.dst_desc())),
      m_scratchPadDesc(DnnlExtensionUtils::makeDescriptor(m_primDesc.scratchpad_desc())),
      m_prim(primitive(m_primDesc)) {}

void DnnlMatMulPrimitive::execute(const dnnl_primitive_args& primArgs) const {
    m_prim.execute(m_stream, primArgs);
}

}  // namespace ov::intel_cpu
