// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "program_wrapper.h"
#include "fully_connected_inst.h"
#include "convolution_inst.h"
#include "gru_seq_inst.h"
#include "permute_inst.h"
#ifdef ENABLE_ONEDNN_FOR_GPU
#include "graph/impls/onednn/utils.hpp"
#endif

using namespace cldnn;
using namespace ::tests;

TEST(post_optimize_weights, dynamic_gru_weights_are_permuted) {
    auto& engine = get_test_engine();

    auto hidden = engine.allocate_memory({ov::PartialShape{1, 1, 3}, data_types::f16, format::bfyx});
    auto weights = engine.allocate_memory({ov::PartialShape{1, 9, 4}, data_types::f16, format::bfyx});
    auto recurrent = engine.allocate_memory({ov::PartialShape{1, 9, 3}, data_types::f16, format::bfyx});
    auto bias = engine.allocate_memory({ov::PartialShape{1, 12}, data_types::f16, format::bfyx});
    auto sequence_lengths = engine.allocate_memory({ov::PartialShape{1}, data_types::i32, format::bfyx});

    gru_seq gru("gru",
                input_info("input"),
                input_info("hidden"),
                input_info(),
                input_info("weights"),
                input_info("recurrent"),
                input_info("bias"),
                input_info("sequence_lengths"),
                0.f,
                false,
                {activation_func::logistic, activation_func::hyperbolic_tan},
                {},
                lstm_weights_order::iofz,
                ov::op::RecurrentSequenceDirection::FORWARD,
                2);
    gru.linear_before_reset = true;

    topology topology(input_layout("input", {ov::PartialShape{1, -1, 4}, data_types::f16, format::bfyx}),
                      data("hidden", hidden),
                      data("weights", weights),
                      data("recurrent", recurrent),
                      data("bias", bias),
                      data("sequence_lengths", sequence_lengths),
                      gru);

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    auto prog = program::build_program(engine, topology, config, false, true);

    reorder_factory rf;
    program_wrapper::apply_opt_pass<compile_graph>(*prog);
    program_wrapper::apply_opt_pass<post_optimize_weights>(*prog, rf);

    auto& gru_node = prog->get_node("gru");
    EXPECT_EQ(gru_node.get_dependency(2).get_output_layout().get_shape(), ov::Shape({1, 4, 9}));
    EXPECT_EQ(gru_node.get_dependency(3).get_output_layout().get_shape(), ov::Shape({1, 3, 9}));
}

TEST(post_optimize_weights, fuse_reorder_to_weights_reorder_test) {
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ { 2, 32 }, data_types::f16, format::bfyx });
    auto weights = engine.allocate_memory({{ 2, 32 }, data_types::f32, format::bfyx });

    topology topology(
        input_layout("input", input->get_layout()),
        input_layout("weights", weights->get_layout()),
        reorder("reorder_dt", input_info("weights"), format::bfyx, data_types::f16),
        fully_connected("fc", input_info("input"), { "reorder_dt" }, "", data_types::f16)
    );

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    auto prog = program::build_program(engine, topology, config, false, true);

    reorder_factory rf;
    program_wrapper::apply_opt_pass<compile_graph>(*prog);
    program_wrapper::apply_opt_pass<post_optimize_weights>(*prog, rf);

    ASSERT_TRUE(has_node(*prog, "reorder_dt"));
    ASSERT_TRUE(format::is_weights_format(prog->get_node("reorder_dt").get_output_layout().format));
}

TEST(post_optimize_weights, fuse_reorder_to_weights_reorder_test_dynamic) {
    auto& engine = get_test_engine();
    if (engine.get_device_info().supports_immad)
        return;

    auto weights = engine.allocate_memory({{ 2, 32 }, data_types::f32, format::bfyx });

    auto in_layout = layout{ ov::PartialShape{ov::Dimension(1), ov::Dimension(-1), ov::Dimension(32)}, data_types::f16, format::bfyx };

    topology topology(
        input_layout("input", in_layout),
        input_layout("weights", weights->get_layout()),
        reorder("reorder_dt", input_info("weights"), format::bfyx, data_types::f16),
        fully_connected("fc", input_info("input"), { "reorder_dt" }, "", data_types::f16, 3)
    );

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    auto prog = program::build_program(engine, topology, config, false, true);

    reorder_factory rf;
    program_wrapper::apply_opt_pass<compile_graph>(*prog);
    program_wrapper::apply_opt_pass<post_optimize_weights>(*prog, rf);

    ASSERT_TRUE(has_node(*prog, "reorder_dt"));
    ASSERT_NE(prog->get_node("fc").get_selected_impl(), nullptr);
    ASSERT_TRUE(format::is_weights_format(prog->get_node("reorder_dt").get_output_layout().format));
}

TEST(post_optimize_weights, weights_reorder_constant_folding_test) {
    auto& engine = get_test_engine();

    ov::Shape pshape = { 4, 16 };
    auto input = engine.allocate_memory({ pshape, data_types::f32, format::bfyx });
    auto weights = engine.allocate_memory({ pshape, data_types::f32, format::bfyx });

    std::vector<float> weights_data(pshape[0] * pshape[1]);
    std::iota(weights_data.begin(), weights_data.end(), 0.f);
    set_values(weights, weights_data);

    topology topology(
        input_layout("input", input->get_layout()),
        data("weights", weights),
        fully_connected("fc", input_info("input"), { "weights" })
    );

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    auto prog = program::build_program(engine, topology, config, false, true);

    reorder_factory rf;
    program_wrapper::apply_opt_pass<compile_graph>(*prog);
    program_wrapper::apply_opt_pass<post_optimize_weights>(*prog, rf);
    program_wrapper::apply_opt_pass<propagate_constants>(*prog);

    ASSERT_TRUE(has_node(*prog, "weights_weights_reorder_0"));
    auto& weights_node = prog->get_node("weights_weights_reorder_0");
    ASSERT_TRUE(weights_node.is_type<data>());

    size_t align = 16; // os_iyx_osv16 format
    size_t aligned_b_size = pshape[0] % align == 0 ? pshape[0]
                                                   : pshape[0] - pshape[0] % align + align;
    std::vector<float> expected(aligned_b_size * pshape[1], 0.f);
    size_t input_idx = 0;
    for (size_t i = 0; i < pshape[0]; ++i) {
        for (size_t j = 0; j < pshape[1]; ++j) {
            expected[j * align + i] = weights_data[input_idx++];
        }
    }

    auto weights_mem_ptr = weights_node.as<data>().get_attached_memory_ptr();
    cldnn::mem_lock<float, mem_lock_type::read> weights_mem(weights_mem_ptr, get_test_stream());

    for (size_t i = 0; i < expected.size(); ++i) {
        ASSERT_EQ(weights_mem[i], expected[i]);
    }
}

TEST(post_optimize_weights, weights_reorder_constant_folding_test_dynamic) {
    auto& engine = get_test_engine();
    if (engine.get_device_info().supports_immad)
        return;
    ov::Shape pshape = { 4, 32 };
    auto in_layout = layout{ ov::PartialShape{ov::Dimension(1), ov::Dimension(-1), ov::Dimension(32)}, data_types::f16, format::bfyx };
    auto weights = engine.allocate_memory({pshape, data_types::f32, format::bfyx });

    std::vector<float> weights_data(pshape[0] * pshape[1]);
    std::iota(weights_data.begin(), weights_data.end(), 0.f);
    set_values(weights, weights_data);

    topology topology(
        input_layout("input", in_layout),
        data("weights", weights),
        fully_connected("fc", input_info("input"), { "weights" }, "", data_types::f16, 3)
    );

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    auto prog = program::build_program(engine, topology, config, false, true);

    reorder_factory rf;
    program_wrapper::apply_opt_pass<compile_graph>(*prog);
    program_wrapper::apply_opt_pass<post_optimize_weights>(*prog, rf);
    program_wrapper::apply_opt_pass<propagate_constants>(*prog);

    ASSERT_TRUE(has_node(*prog, "weights_weights_reorder_0"));
    auto& weights_node = prog->get_node("weights_weights_reorder_0");
    ASSERT_TRUE(weights_node.is_type<data>());

    size_t align = 16; // os_iyx_osv16 format
    size_t aligned_b_size = pshape[0] % align == 0 ? pshape[0]
                                                   : pshape[0] - pshape[0] % align + align;
    std::vector<float> expected(aligned_b_size * pshape[1], 0.f);
    size_t input_idx = 0;
    for (size_t i = 0; i < pshape[0]; ++i) {
        for (size_t j = 0; j < pshape[1]; ++j) {
            expected[j * align + i] = weights_data[input_idx++];
        }
    }

    auto weights_mem_ptr = weights_node.as<data>().get_attached_memory_ptr();
    cldnn::mem_lock<float, mem_lock_type::read> weights_mem(weights_mem_ptr, get_test_stream());

    for (size_t i = 0; i < expected.size(); ++i) {
        ASSERT_EQ(weights_mem[i], expected[i]);
    }
}

TEST(post_optimize_weights, fuse_only_with_supported_weights_layout) {
    // This test case checks that the reorder node is not fused with unsupported weights layout such as b_fs_yx_fsv16.
    //    - Previous : When the weight layout which is a reorder node, is b_fs_yx_fsv16, fusing was attempted but the program terminates.
    //    - Current : If the layout format is not supported, do not fuse and the program won't terminate.
    //
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ { 2, 32 }, data_types::f16, format::bfyx });
    auto weights = engine.allocate_memory({{ 2, 32 }, data_types::f32, format::b_fs_yx_fsv16 });

    topology topology(
        input_layout("input", input->get_layout()),
        input_layout("weights", weights->get_layout()),
        reorder("reorder", input_info("weights"), format::bfyx, data_types::f16),
        fully_connected("fc", input_info("input"), { "reorder" }, "", data_types::f16)
    );

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    auto prog = program::build_program(engine, topology, config, false, true);

    reorder_factory rf;
    program_wrapper::apply_opt_pass<compile_graph>(*prog);
    program_wrapper::apply_opt_pass<post_optimize_weights>(*prog, rf);

    ASSERT_TRUE(has_node(*prog, "reorder"));
}

TEST(post_optimize_weights, fuse_transferred_reorders_of_shared_weights) {
    // Two weights chains start at the same constant: data -> reorder(f32->f16) -> permute -> fc.
    // reorder_transfer moves each reorder below its permute, so the reorder primitive still records the constant
    // as its input. Fusing must use the actual dependency, otherwise the second fc gets the first fc's reorder.
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ { 2, 32 }, data_types::f16, format::bfyx });
    auto weights = engine.allocate_memory({ { 32, 2 }, data_types::f32, format::bfyx });

    topology topology(
        input_layout("input", input->get_layout()),
        data("weights", weights),
        reorder("reorder_dt_0", input_info("weights"), format::bfyx, data_types::f16,
                std::vector<float>(), reorder_mean_mode::subtract, padding(), true),
        permute("permute_0", input_info("reorder_dt_0"), {1, 0}),
        fully_connected("fc_0", input_info("input"), { "permute_0" }, "", data_types::f16),
        reorder("reorder_dt_1", input_info("weights"), format::bfyx, data_types::f16,
                std::vector<float>(), reorder_mean_mode::subtract, padding(), true),
        permute("permute_1", input_info("reorder_dt_1"), {1, 0}),
        fully_connected("fc_1", input_info("input"), { "permute_1" }, "", data_types::f16)
    );

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    auto prog = program::build_program(engine, topology, config, false, true);

    for (auto& node : prog->get_processing_order()) {
        if (!node->is_type<data>())
            node->get_output_layouts();
    }

    reorder_factory rf;
    program_wrapper::apply_opt_pass<reorder_transfer>(*prog);
    program_wrapper::apply_opt_pass<compile_graph>(*prog);
    ASSERT_NO_THROW(program_wrapper::apply_opt_pass<post_optimize_weights>(*prog, rf));

    for (const auto& [fc_id, permute_id] : {std::pair{"fc_0", "permute_0"}, std::pair{"fc_1", "permute_1"}}) {
        auto& fc_weights = prog->get_node(fc_id).get_dependency(1);
        ASSERT_TRUE(fc_weights.is_type<reorder>()) << fc_id;
        EXPECT_TRUE(format::is_weights_format(fc_weights.get_output_layout().format)) << fc_id;
        EXPECT_EQ(fc_weights.get_dependency(0).id(), permute_id);
    }
}

TEST(post_optimize_weights, share_fused_weights_reorder_of_same_input) {
    // Two fully connected layers read the same constant through separate but identical precision reorders.
    // The second fusion finds the first weights reorder in the cache and must reuse it instead of replacing.
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ { 2, 32 }, data_types::f16, format::bfyx });
    auto weights = engine.allocate_memory({ { 2, 32 }, data_types::f32, format::bfyx });

    topology topology(
        input_layout("input", input->get_layout()),
        data("weights", weights),
        reorder("reorder_dt_0", input_info("weights"), format::bfyx, data_types::f16),
        fully_connected("fc_0", input_info("input"), { "reorder_dt_0" }, "", data_types::f16),
        reorder("reorder_dt_1", input_info("weights"), format::bfyx, data_types::f16),
        fully_connected("fc_1", input_info("input"), { "reorder_dt_1" }, "", data_types::f16)
    );

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    auto prog = program::build_program(engine, topology, config, false, true);

    reorder_factory rf;
    program_wrapper::apply_opt_pass<compile_graph>(*prog);
    ASSERT_NO_THROW(program_wrapper::apply_opt_pass<post_optimize_weights>(*prog, rf));

    auto& fc_0_weights = prog->get_node("fc_0").get_dependency(1);
    ASSERT_TRUE(format::is_weights_format(fc_0_weights.get_output_layout().format));
    EXPECT_EQ(&prog->get_node("fc_1").get_dependency(1), &fc_0_weights);
    EXPECT_EQ(fc_0_weights.get_dependency(0).id(), "weights");
    // The shared reorder keeps the id of the reorder it replaced first; the other precision reorder is removed.
    EXPECT_NE(has_node(*prog, "reorder_dt_0"), has_node(*prog, "reorder_dt_1"));
}

TEST(post_optimize_weights, onednn_group_conv_weights_reorder_test) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad)
        return;

    ov::Shape pshape = { 1, 512, 2, 332 };
    ov::Shape weight_pshape = { 64, 1, 8, 5, 5 };
    auto input = engine.allocate_memory({ pshape, data_types::f16, format::bfyx });
    auto weights = engine.allocate_memory({ weight_pshape, data_types::f16, format::bfzyx });

    std::vector<ov::float16> weights_data(weight_pshape[0] * weight_pshape[1] * weight_pshape[2] * weight_pshape[3] * weight_pshape[4]);
    std::iota(weights_data.begin(), weights_data.end(), 0.f);
    set_values(weights, weights_data);

    topology topology(
        input_layout("input", input->get_layout()),
        data("weights", weights),
        convolution("conv", input_info("input"), "weights", "", 64, {1, 1}, {1, 1}, {2, 2}, {2, 2}, true)
    );

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

    if (engine.get_device_info().supports_immad) {
        ov::intel_gpu::ImplementationDesc conv_impl = { format::b_fs_yx_fsv16, std::string(""), impl_types::onednn };
        config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"conv", conv_impl} }));
    }

    auto prog = program::build_program(engine, topology, config);

    auto& conv_node = prog->get_node("conv");
    auto weights_param = conv_node.as<convolution>().get_selected_impl()->get_weights_reorder_params();
    ASSERT_TRUE(format::is_weights_format(prog->get_node("weights_weights_reorder_0").get_output_layout().format));

#ifdef ENABLE_ONEDNN_FOR_GPU
    // Check onednn_weights_params->_out_desc.get_size() with reorder output_layout bytes_count
    auto onednn_weights_params = std::dynamic_pointer_cast<onednn::WeightsReorderParamsOneDNN>(weights_param);
    ASSERT_TRUE(onednn_weights_params->_out_desc->get_size() == prog->get_node("weights_weights_reorder_0").get_output_layout().bytes_count());
#endif
}
