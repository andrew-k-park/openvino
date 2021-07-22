// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <thread>
#include "primitive_inst.h"
#include "program_impl.h"
#include "cldnn/runtime/debug_configuration.hpp"
#include "cldnn/runtime/error_handler.hpp"
#include "kernel_selector_helper.h"
#include "network_impl.h"
#include "register.hpp"
#include <vector>
#include <list>
#include <utility>
#include <iomanip>
#include <fstream>
#include "to_string_utils.h"


namespace cldnn {
namespace ocl {

// checks if any user in a list is a cpu primitive
bool is_any_user_cpu(const std::list<const program_node*>& users);

/*
Base class for all GPU implementation of specified primitive type.
For example, all gpu convolution implementations should derive from typed_primitive_impl_ocl<convolution>.
*/
static float convert_half_to_float(half_t val, bool flush_denorm_to_zero = false) {
#if defined HALF_HALF_HPP
    return val;
#else
    // FP32 parts extracted from FP16.
    uint32_t sign = (static_cast<uint16_t>(val) & 0x8000U) << 16;
    uint32_t mantissa = (static_cast<uint16_t>(val) & 0x3FFU) << 13;

    uint32_t exp_val_f16 = (static_cast<uint16_t>(val) & 0x7C00U) >> 10;
    uint32_t exp;
    if (exp_val_f16 == 0) {
        // Handling +/-0 and denormals.
        if (mantissa == 0) {
            exp = 0;
        } else if (flush_denorm_to_zero) {
            sign = 0;
            exp = 0;
            mantissa = 0;
        } else {
            // Denorms conversion to normal numbers.
            exp = 127 - 15;
            while (!(mantissa & 0x400000U)) {
                mantissa <<= 1;
                --exp;
            }
            mantissa = (mantissa << 1) & 0x7FFFFFU;
            exp <<= 23;
        }
    } else {
        // Handling +/-infinity, NaN and normal numbers.
        exp = (exp_val_f16 == 0x1FU ? 0xFFU : exp_val_f16 + 127 - 15) << 23;
    }

    float ret;
    reinterpret_cast<uint32_t&>(ret) = sign | exp | mantissa;

    return ret;
#endif
}

static float convert_element(uint32_t u) { return static_cast<float>(u); }

static float convert_element(int32_t i) { return static_cast<float>(i); }

static float convert_element(float f) { return f; }

static float convert_element(half_t h) { return convert_half_to_float(h); }

static size_t get_x_pitch(const layout& layout) {
    try {
        auto tensor_x0 = tensor(batch(0), feature(0), spatial(0, 0, 0, 0));
        auto tensor_x1 = tensor(batch(0), feature(0), spatial(1, 0, 0, 0));
        auto x0 = layout.get_linear_offset(tensor_x0);
        auto x1 = layout.get_linear_offset(tensor_x1);
        return (x1 - x0);
    } catch (...) {
        // When spatial size of x=0, x_pitch is meaningless
        return 0;
    }
}

template <class T>
static void dump(memory::ptr mem, stream& stream, std::ofstream& file_stream) {
    auto&& size = mem->get_layout().size;

    file_stream << "shape: " << size.to_string() << " ";
    file_stream << "(count: " << size.count() << ", original format: " << cldnn::fmt_to_str(mem->get_layout().format) << ")" << std::endl;

    mem_lock<T> lock(mem, stream);
    auto mem_ptr = lock.data();
    auto x_pitch = get_x_pitch(mem->get_layout());
    std::stringstream buffer;

    for (cldnn::tensor::value_type g = 0; g < size.group[0]; ++g) {
        for (cldnn::tensor::value_type b = 0; b < size.batch[0]; ++b) {
            for (cldnn::tensor::value_type f = 0; f < size.feature[0]; ++f) {
                for (cldnn::tensor::value_type w = 0; w < size.spatial[3]; ++w) {
                    for (cldnn::tensor::value_type z = 0; z < size.spatial[2]; ++z) {
                        for (cldnn::tensor::value_type y = 0; y < size.spatial[1]; ++y) {
                            cldnn::tensor t(cldnn::group(g), cldnn::batch(b), cldnn::feature(f), cldnn::spatial(0, y, z, w));
                            size_t input_it = mem->get_layout().get_linear_offset(t);

                            for (cldnn::tensor::value_type x = 0; x < size.spatial[0]; ++x, input_it += x_pitch) {
                                buffer << std::fixed << std::setprecision(6) << convert_element(mem_ptr[input_it]) << std::endl;
                            }
                        }
                    }
                }
            }
        }
    }
    file_stream << buffer.str();
}

template <>
void dump<uint32_t>(memory::ptr mem, stream& stream, std::ofstream& file_stream) {
    auto&& size = mem->get_layout().size;

    file_stream << "shape: ";
    file_stream << size.batch[0] << " ";
    file_stream << size.feature[0] << " ";
    file_stream << size.spatial[1] << " ";
    file_stream << size.spatial[0] << " ";
    file_stream << "(" << size.batch[0] * size.feature[0] * size.spatial[1] * size.spatial[0] << ")" << std::endl;

    mem_lock<uint32_t> lock(mem, stream);
    auto mem_ptr = lock.data();

    for (cldnn::tensor::value_type b = 0; b < size.batch[0]; ++b) {
        for (cldnn::tensor::value_type f = 0; f < (cldnn::tensor::value_type)ceil_div(size.feature[0], 32); ++f) {
            for (cldnn::tensor::value_type z = 0; z < size.spatial[2]; ++z) {
                for (cldnn::tensor::value_type y = 0; y < size.spatial[1]; ++y) {
                    for (cldnn::tensor::value_type x = 0; x < size.spatial[0]; ++x) {
                        cldnn::tensor t(cldnn::batch(b), cldnn::feature(f), cldnn::spatial(x, y, z, 0));
                        size_t input_it = mem->get_layout().get_linear_offset(t);
                        file_stream << mem_ptr[input_it] << std::endl;
                    }
                }
            }
        }
    }
}
static void log_memory_to_file(memory::ptr mem, stream& stream, std::string layerName) {
    GPU_DEBUG_GET_INSTANCE(debug_config);
    std::string filename = layerName;
    std::replace(filename.begin(), filename.end(), '\\', '_');
    std::replace(filename.begin(), filename.end(), '/', '_');
    std::replace(filename.begin(), filename.end(), ' ', '_');
    std::replace(filename.begin(), filename.end(), ':', '_');
    filename = debug_config->dump_layers_path + filename + ".txt";

    std::ofstream file_stream(filename);
    auto mem_dt = mem->get_layout().data_type;
    if (mem_dt == cldnn::data_types::f32)
        dump<float>(mem, stream, file_stream);
    else if (mem_dt == cldnn::data_types::f16)
        dump<half_t>(mem, stream, file_stream);
    else if (mem_dt == cldnn::data_types::bin)
        dump<uint32_t>(mem, stream, file_stream);
    else if (mem_dt == cldnn::data_types::i32)
        dump<int32_t>(mem, stream, file_stream);
    else if (mem_dt == cldnn::data_types::i8)
        dump<int8_t>(mem, stream, file_stream);
    else if (mem_dt == cldnn::data_types::u8)
        dump<uint8_t>(mem, stream, file_stream);
}

template <class PType>
struct typed_primitive_impl_ocl : public typed_primitive_impl<PType> {
    const typed_program_node<PType>& _outer;
    kernel_selector::kernel_data _kernel_data;
    std::vector<kernel_id> _kernel_ids;
    std::vector<kernel::ptr> _kernels;
    // std::vector<memory::cptr> _intermediates_memory;
    std::vector<memory::ptr> _intermediates_memory;

    typed_primitive_impl_ocl(const typed_primitive_impl_ocl<PType>& other)
    : typed_primitive_impl<PType>(other._weights_reorder_params, other._kernel_name)
    , _outer(other._outer)
    , _kernel_data(other._kernel_data)
    , _kernel_ids(other._kernel_ids)
    , _kernels({})
    , _intermediates_memory({}) {
        _kernels.reserve(other._kernels.size());
        for (size_t k = 0; k < other._kernels.size(); ++k) {
            _kernels.emplace_back(other._kernels[k]->clone());
        }
        for (auto& mem : other._intermediates_memory) {
            auto& engine = _outer.get_program().get_engine();
            auto new_mem = engine.allocate_memory(mem->get_layout(), mem->get_allocation_type());
            _intermediates_memory.push_back(new_mem);
        }
    }

    typed_primitive_impl_ocl(const typed_program_node<PType>& arg, const kernel_selector::kernel_data& kd)
        : typed_primitive_impl<PType>(kd.weightsReorderParams, kd.kernelName),
          _outer(arg),
          _kernel_data(kd) {
        // weights reorder params got copied to parent, clear in _kernel_data to release shared ptr
        _kernel_data.weightsReorderParams.engine = kernel_selector::generic_kernel_params::Engine::NONE;
        _kernel_data.weightsReorderParams.cpuKernel = nullptr;
        _kernel_data.weightsReorderParams.clKernel = nullptr;

        _kernel_ids.reserve(kd.kernels.size());
        // Add selected kernels to kernels_cache for the following compilation and save output ids
        for (size_t i = 0; i < kd.kernels.size(); ++i) {
            _kernel_ids.emplace_back(_outer.get_program().add_kernel(kd.kernels[i].code.kernelString));
        }

        for (auto size : kd.internalBufferSizes) {
            auto dtype = from_data_type(kd.internalBufferDataType);
            const auto bpp = data_type_traits::size_of(dtype);
            layout expected_layout = {dtype,
                                      format::bfyx,  // simple linear format (flatten to x channel)
                                      {1, 1, 1, (tensor::value_type)(size / bpp)}};

            auto& eimpl = arg.get_program().get_engine();
            _intermediates_memory.push_back(eimpl.allocate_memory(expected_layout));
        }
    }
    bool is_cpu() const override { return false; }

protected:
    virtual bool optimized_out(typed_primitive_inst<PType>&) const { return false; }

    virtual kernel_arguments_data get_arguments(typed_primitive_inst<PType>& instance, int32_t /*split*/) const {
        kernel_arguments_data args;

        for (size_t i = 0; i < instance.inputs_memory_count(); i++) {
            args.inputs.push_back(instance.input_memory_ptr(i));
        }

        if (instance.has_fused_primitives()) {
            size_t count = instance.get_fused_mem_count();
            for (size_t i = 0; i < count; i++) {
                args.fused_op_inputs.push_back(instance.fused_memory(i));
            }
        }

        args.output = instance.output_memory_ptr();

        return args;
    }

    virtual int32_t get_split() const { return 1; }
    virtual uint32_t get_groups() const { return 1; }
    virtual bool get_depthwise_sep_opt() const { return false; }

    event::ptr aggregate_events(const std::vector<event::ptr>& events, stream& stream, bool group = false, bool is_output = false) const {
        if (events.size() == 1 && !is_output)
            return events[0];

        if (group && !is_output)
            return stream.group_events(events);

        return stream.enqueue_marker(events, is_output);
    }

    void init_kernels() override {
        if (is_cpu()) {
            return;
        }
        _kernels.clear();

        _kernels.reserve(_kernel_ids.size());
        for (size_t k = 0; k < _kernel_ids.size(); ++k) {
            _kernels.emplace_back(std::move(_outer.get_program().get_kernel(_kernel_ids[k])));
        }
    }

    void set_arguments_impl(typed_primitive_inst<PType>& instance) override {
        if (optimized_out(instance) || is_cpu()) {
            return;
        }

        auto split = get_split();

        stream& stream = instance.get_network().get_stream();

        // we iterate over split first in order to be able parallelism with OOOQ mechanism.
        for (size_t k = 0; k < _kernels.size(); ++k) {
            for (decltype(split) i = 0; i < split; i++) {
                auto args = get_arguments(instance, i);
                args.scalars = &_kernel_data.kernels[k].params.scalars;
                args.split = i;

                for (const auto& m : _intermediates_memory) {
                    args.intermediates.push_back(m);
                }


                stream.set_arguments(*_kernels[k], _kernel_data.kernels[k].params, args);
            }
        }
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events,
                            typed_primitive_inst<PType>& instance) override {
        stream& stream = instance.get_network().get_stream();
        if (optimized_out(instance)) {
            return aggregate_events(events, stream, false, instance.is_output());
        }

        std::vector<event::ptr> tmp_events(events);
        std::vector<event::ptr> all_events;

        // TODO - split should be handle in kernel selector by providing multiple kernels.
        auto split = get_split();

        // we iterate over split first in order to be able parallelism with OOOQ mechanism.
        for (size_t k = 0; k < _kernels.size(); ++k) {
            std::vector<event::ptr> new_events;
            for (decltype(split) i = 0; i < split; i++) {
                // is any user of the prim's users is an detecion output, set prim as a output event (event won't be nullptr)
                auto users = instance.node.get_users();
                bool is_output_event = is_any_user_cpu(users) || instance.node.is_output();

                auto args = get_arguments(instance, i);
                args.scalars = &_kernel_data.kernels[k].params.scalars;
                args.split = i;

                for (const auto& m : _intermediates_memory) {
                    args.intermediates.push_back(m);
                }

                auto ev = stream.enqueue_kernel(*_kernels[k], _kernel_data.kernels[k].params, args, tmp_events, is_output_event);
                new_events.push_back(ev);
                all_events.push_back(ev);
            }

            if (_kernels.size() > 1) {
                stream.finish();
                for (decltype(split) i = 0; i < split; i++) {
                    for (size_t m = 0; m < _intermediates_memory.size(); ++m) {
                        log_memory_to_file(_intermediates_memory[m], stream, "dump_K" + std::to_string(k) + "_" + std::to_string(m));
                    }
                }
            }
            tmp_events = new_events;
        }
        if (_kernels.size() > 1) {
            log_memory_to_file(instance.output_memory_ptr(), stream, "dump_output");
        }
        if ((all_events.size() == 0) && (tmp_events.size() > 0))
            return aggregate_events(tmp_events, stream);

        bool group_events = (all_events.size() > 1);
        return aggregate_events(all_events, stream, group_events);
    }
};

}  // namespace ocl
}  // namespace cldnn
