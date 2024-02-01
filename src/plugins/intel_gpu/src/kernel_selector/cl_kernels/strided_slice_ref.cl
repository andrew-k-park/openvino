// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/fetch_utils.cl"

#ifdef STRIDE_TYPE
inline void FUNC(get_slice_step)(OPTIONAL_SHAPE_INFO_ARG
                                 const __global STRIDE_TYPE* stride,
                                 int* step_batch, int* step_feature,
                                 int* step_w, int* step_z, int* step_y, int* step_x)
{
    const uint batch_index = 0;
    const uint feature_index = 1;
#ifdef OUTPUT_LAYOUT_BFYX
    const uint y_index = 2;
    const uint x_index = 3;
#elif OUTPUT_LAYOUT_BFZYX
    const uint z_index = 2;
    const uint y_index = 3;
    const uint x_index = 4;
#elif OUTPUT_LAYOUT_BFWZYX
    const uint w_index = 2;
    const uint z_index = 3;
    const uint y_index = 4;
    const uint x_index = 5;
#endif

    *step_batch = batch_index < STRIDE_DIMS ? stride[batch_index] : 1;
    *step_feature = feature_index < STRIDE_DIMS ? stride[feature_index] : 1;
#ifdef OUTPUT_LAYOUT_BFYX
    *step_w = 0;
    *step_z = 0;
#elif OUTPUT_LAYOUT_BFZYX
    *step_w = 0;
    *step_z = z_index < STRIDE_DIMS ? stride[z_index] : 1;
#elif OUTPUT_LAYOUT_BFWZYX
    *step_w = w_index < STRIDE_DIMS ? stride[w_index] : 1;
    *step_z = z_index < STRIDE_DIMS ? stride[z_index] : 1;
#endif
    *step_y = y_index < STRIDE_DIMS ? stride[y_index] : 1;
    *step_x = x_index < STRIDE_DIMS ? stride[x_index] : 1;
}
#endif // STRIDE_TYPE

#ifdef END_TYPE
inline void FUNC(get_slice_end)(OPTIONAL_SHAPE_INFO_ARG
                                const __global END_TYPE* end,
                                int* end_batch, int* end_feature,
                                int* end_w, int* end_z, int* end_y, int* end_x)
{
    const uint out_batch_num = INPUT0_BATCH_NUM;
    const uint out_feature_num = INPUT0_FEATURE_NUM;
    const uint out_w_num = INPUT0_SIZE_W;
    const uint out_z_num = INPUT0_SIZE_Z;
    const uint out_y_num = INPUT0_SIZE_Y;
    const uint out_x_num = INPUT0_SIZE_X;
    const uint batch_index = 0;
    const uint feature_index = 1;
#ifdef OUTPUT_LAYOUT_BFYX
    const uint y_index = 2;
    const uint x_index = 3;
#elif OUTPUT_LAYOUT_BFZYX
    const uint z_index = 2;
    const uint y_index = 3;
    const uint x_index = 4;
#elif OUTPUT_LAYOUT_BFWZYX
    const uint w_index = 2;
    const uint z_index = 3;
    const uint y_index = 4;
    const uint x_index = 5;
#endif
    END_TYPE batch = batch_index < END_DIMS ? end[batch_index] : 0;
    END_TYPE feature = feature_index < END_DIMS ? end[feature_index] : 0;
#ifdef OUTPUT_LAYOUT_BFWZYX
    END_TYPE w = w_index < END_DIMS ? end[w_index] : 0;
    END_TYPE z = z_index < END_DIMS ? end[z_index] : 0;
#elif OUTPUT_LAYOUT_BFZYX
    END_TYPE z = z_index < END_DIMS ? end[z_index] : 0;
#endif
    END_TYPE y = y_index < END_DIMS ? end[y_index] : 0;
    END_TYPE x = x_index < END_DIMS ? end[x_index] : 0;

    *end_batch = (END_BATCH == 0) ? batch : TO_END_TYPE(out_batch_num);
    *end_feature = (END_FEATURE == 0) ? feature : TO_END_TYPE(out_feature_num);
#ifdef OUTPUT_LAYOUT_BFWZYX
    *end_w = (END_W == 0) ? w: TO_END_TYPE(out_w_num);
    *end_z = (END_Z == 0) ? z: TO_END_TYPE(out_z_num);
#elif OUTPUT_LAYOUT_BFZYX
    *end_z = (END_Z == 0) ? z: TO_END_TYPE(out_z_num);
#endif
    *end_y = (END_Y == 0) ? y : TO_END_TYPE(out_y_num);
    *end_x = (END_X == 0) ? x : TO_END_TYPE(out_x_num);
}
#else
inline void FUNC(get_slice_end)(OPTIONAL_SHAPE_INFO_ARG
                                int* end_batch, int* end_feature,
                                int* end_w, int* end_z, int* end_y, int* end_x)
{
    *end_batch = (END_BATCH == 0) ? SLICE_END_BATCH : INPUT0_BATCH_NUM;
    *end_feature = (END_FEATURE == 0) ? SLICE_END_FEATURE : INPUT0_FEATURE_NUM;
#ifdef OUTPUT_LAYOUT_BFWZYX
    *end_w = (END_W == 0) ? SLICE_END_W: INPUT0_SIZE_W;
    *end_z = (END_Z == 0) ? SLICE_END_Z: INPUT0_SIZE_Z;
#elif OUTPUT_LAYOUT_BFZYX
    *end_z = (END_Z == 0) ? SLICE_END_Z: INPUT0_SIZE_Z;
#endif
    *end_y = (END_Y == 0) ? SLICE_END_Y : INPUT0_SIZE_Y;
    *end_x = (END_X == 0) ? SLICE_END_X : INPUT0_SIZE_X;
}
#endif // END_TYPE

#ifdef BEGIN_TYPE
inline void FUNC(get_slice_begin)(OPTIONAL_SHAPE_INFO_ARG
                                  const __global BEGIN_TYPE* begin,
                                  int* begin_batch, int* begin_feature,
                                  int* begin_w, int* begin_z, int* begin_y, int* begin_x)
{
    const uint batch_index = 0;
    const uint feature_index = 1;
#ifdef OUTPUT_LAYOUT_BFYX
    const uint y_index = 2;
    const uint x_index = 3;
#elif OUTPUT_LAYOUT_BFZYX
    const uint z_index = 2;
    const uint y_index = 3;
    const uint x_index = 4;
#elif OUTPUT_LAYOUT_BFWZYX
    const uint w_index = 2;
    const uint z_index = 3;
    const uint y_index = 4;
    const uint x_index = 5;
#endif

    BEGIN_TYPE batch = batch_index < BEGIN_DIMS ? begin[batch_index] : 0;
    BEGIN_TYPE feature = feature_index < BEGIN_DIMS ? begin[feature_index] : 0;
#ifdef OUTPUT_LAYOUT_BFWZYX
    BEGIN_TYPE w = w_index < BEGIN_DIMS ? begin[w_index] : 0;
    BEGIN_TYPE z = z_index < BEGIN_DIMS ? begin[z_index] : 0;
#elif OUTPUT_LAYOUT_BFZYX
    BEGIN_TYPE z = z_index < BEGIN_DIMS ? begin[z_index] : 0;
#endif
    BEGIN_TYPE y = y_index < BEGIN_DIMS ? begin[y_index] : 0;
    BEGIN_TYPE x = x_index < BEGIN_DIMS ? begin[x_index] : 0;

    *begin_batch = (BEGIN_BATCH == 0) ? batch : 0;
    *begin_feature = (BEGIN_FEATURE == 0) ? feature : 0;
#ifdef OUTPUT_LAYOUT_BFWZYX
    *begin_w = (BEGIN_W == 0) ? w: 0;
    *begin_z = (BEGIN_Z == 0) ? z: 0;
#elif OUTPUT_LAYOUT_BFZYX
    *begin_z = (BEGIN_Z == 0) ? z: 0;
#endif
    *begin_y = (BEGIN_Y == 0) ? y : 0;
    *begin_x = (BEGIN_X == 0) ? x : 0;
}
#else
inline void FUNC(get_slice_begin)(int* begin_batch, int* begin_feature,
                                  int* begin_w, int* begin_z, int* begin_y, int* begin_x)
{
    *begin_batch = (BEGIN_BATCH == 0) ? SLICE_BEGIN_BATCH : 0;
    *begin_feature = (BEGIN_FEATURE == 0) ? SLICE_BEGIN_FEATURE : 0;
#ifdef OUTPUT_LAYOUT_BFWZYX
    *begin_w = (BEGIN_W == 0) ? SLICE_BEGIN_W: 0;
    *begin_z = (BEGIN_Z == 0) ? SLICE_BEGIN_Z: 0;
#elif OUTPUT_LAYOUT_BFZYX
    *begin_z = (BEGIN_Z == 0) ? SLICE_BEGIN_Z: 0;
#endif
    *begin_y = (BEGIN_Y == 0) ? SLICE_BEGIN_Y : 0;
    *begin_x = (BEGIN_X == 0) ? SLICE_BEGIN_X : 0;
}
#endif // BEGIN_TYPE

#ifdef SHRINK_MODE
inline void FUNC(calculate_index)(int* step, int* begin_num, int* end_num, const uint out_num, const int shrink)
{
    if (shrink)
    {
        int real_begin = *begin_num < 0 ? *begin_num + out_num : *begin_num;
        *begin_num = real_begin;
        *end_num = real_begin + 1;
        *step = 1;
    }
    else
#else
inline void FUNC(calculate_index)(int* step, int* begin_num, int* end_num, const uint out_num)
{
#endif
    {
        int real_begin = *begin_num < 0 ? *begin_num + out_num : *begin_num;
        int real_end = *end_num < 0 ? *end_num + out_num : *end_num;
        if (*step < 0) {    
            real_begin = max((int)(0), min((int)(out_num - 1), real_begin));
            real_end = max((int)(-1), min((int)out_num, real_end));
            if (real_begin < real_end) { // for reversing
                const int real_stride = -(*step);
                real_end -= max((int)(0), real_end - 1 - real_begin) % real_stride;
                int tmp = real_begin;
                real_begin = --real_end;
                real_end = --tmp;
            }
        } else {
            real_begin = max((int)(0), min((int)out_num, real_begin));
            real_end = max((int)(0), min((int)out_num, real_end));
            if (real_begin > real_end) { // for reversing
                const int real_stride = *step;
                real_end += max((int)(0), real_begin - real_end - 1) % real_stride;
                int tmp = real_begin;
                real_begin = ++real_end;
                real_end = ++tmp;
                *step = -real_stride;
            }
        }
        *begin_num = real_begin;
        *end_num = real_end;
    }
}

KERNEL(strided_slice_ref)(OPTIONAL_SHAPE_INFO_ARG
                          const __global INPUT0_TYPE* input,
#ifdef BEGIN_TYPE
                          const __global BEGIN_TYPE* begin,
#endif
#ifdef END_TYPE
                          const __global END_TYPE* end,
#endif
#ifdef STRIDE_TYPE
                          const __global STRIDE_TYPE* stride,
#endif
                          __global OUTPUT_TYPE* output)
{
    const uint batch = get_global_id(0);
    const uint feature = get_global_id(1);
    int step_batch, step_feature, step_w, step_z, step_y, step_x;
    int begin_batch, begin_feature, begin_w, begin_z, begin_y, begin_x;
    int end_batch, end_feature, end_w, end_z, end_y, end_x;

#ifdef STRIDE_TYPE
    FUNC_CALL(get_slice_step)(OPTIONAL_SHAPE_INFO_TENSOR stride, &step_batch, &step_feature, &step_w, &step_z, &step_y, &step_x);
#else // STRIDE_TYPE
    step_batch = SLICE_STEPS_BATCH;
    step_feature = SLICE_STEPS_FEATURE;
    step_w = SLICE_STEPS_W;
    step_z = SLICE_STEPS_Z;
    step_y = SLICE_STEPS_Y;
    step_x = SLICE_STEPS_X;
#endif // STRIDE_TYPE

#ifdef BEGIN_TYPE
    FUNC_CALL(get_slice_begin)(OPTIONAL_SHAPE_INFO_TENSOR begin, &begin_batch, &begin_feature, &begin_w, &begin_z, &begin_y, &begin_x);
#else // BEGIN_TYPE
    FUNC_CALL(get_slice_begin)(&begin_batch, &begin_feature, &begin_w, &begin_z, &begin_y, &begin_x);
#endif // BEGIN_TYPE

#ifdef END_TYPE
    FUNC_CALL(get_slice_end)(OPTIONAL_SHAPE_INFO_TENSOR end, &end_batch, &end_feature, &end_w, &end_z, &end_y, &end_x);
#else // END_TYPE
    FUNC_CALL(get_slice_end)(OPTIONAL_SHAPE_INFO_TENSOR &end_batch, &end_feature, &end_w, &end_z, &end_y, &end_x);
#endif // END_TYPE

    // printf("step: b(%u) f(%u) w(%u) z(%u) y(%u) x(%u), begin: b(%u) f(%u) w(%u) z(%u) y(%u) x(%u), end: b(%u) f(%u) w(%u) z(%u) y(%u) x(%u)\n",
    //         step_batch, step_feature, step_w, step_z, step_y, step_x,
    //         begin_batch, begin_feature, begin_w, begin_z, begin_y, begin_x,
    //         end_batch, end_feature, end_w, begin_z, end_y, end_x);

#ifdef SHRINK_MODE
    FUNC_CALL(calculate_index)(&step_batch, &begin_batch, &end_batch, INPUT0_BATCH_NUM, SHRINK_BATCH);
    FUNC_CALL(calculate_index)(&step_feature, &begin_feature, &end_feature, INPUT0_FEATURE_NUM, SHRINK_FEATURE);
#ifdef OUTPUT_LAYOUT_BFYX
    FUNC_CALL(calculate_index)(&step_y, &begin_y, &end_y, INPUT0_SIZE_Y, SHRINK_Y);
    FUNC_CALL(calculate_index)(&step_x, &begin_x, &end_x, INPUT0_SIZE_X, SHRINK_X);
#elif OUTPUT_LAYOUT_BFZYX
    FUNC_CALL(calculate_index)(&step_z, &begin_z, &end_z, INPUT0_SIZE_Z, SHRINK_Z);
    FUNC_CALL(calculate_index)(&step_y, &begin_y, &end_y, INPUT0_SIZE_Y, SHRINK_Y);
    FUNC_CALL(calculate_index)(&step_x, &begin_x, &end_x, INPUT0_SIZE_X, SHRINK_X);
#elif OUTPUT_LAYOUT_BFWZYX
    FUNC_CALL(calculate_index)(&step_w, &begin_w, &end_w, INPUT0_SIZE_W, SHRINK_W);
    FUNC_CALL(calculate_index)(&step_z, &begin_z, &end_z, INPUT0_SIZE_Z, SHRINK_Z);
    FUNC_CALL(calculate_index)(&step_y, &begin_y, &end_y, INPUT0_SIZE_Y, SHRINK_Y);
    FUNC_CALL(calculate_index)(&step_x, &begin_x, &end_x, INPUT0_SIZE_X, SHRINK_X);
#endif // OUTPUT_LAYOUT_BFYX
#else // SHRINK_MODE
    FUNC_CALL(calculate_index)(&step_batch, &begin_batch, &end_batch, INPUT0_BATCH_NUM);
    FUNC_CALL(calculate_index)(&step_feature, &begin_feature, &end_feature, INPUT0_FEATURE_NUM);
#ifdef OUTPUT_LAYOUT_BFYX
    FUNC_CALL(calculate_index)(&step_y, &begin_y, &end_y, INPUT0_SIZE_Y);
    FUNC_CALL(calculate_index)(&step_x, &begin_x, &end_x, INPUT0_SIZE_X);
#elif OUTPUT_LAYOUT_BFZYX
    FUNC_CALL(calculate_index)(&step_z, &begin_z, &end_z, INPUT0_SIZE_Z);
    FUNC_CALL(calculate_index)(&step_y, &begin_y, &end_y, INPUT0_SIZE_Y);
    FUNC_CALL(calculate_index)(&step_x, &begin_x, &end_x, INPUT0_SIZE_X);
#elif OUTPUT_LAYOUT_BFWZYX
    FUNC_CALL(calculate_index)(&step_w, &begin_w, &end_w, INPUT0_SIZE_W);
    FUNC_CALL(calculate_index)(&step_z, &begin_z, &end_z, INPUT0_SIZE_Z);
    FUNC_CALL(calculate_index)(&step_y, &begin_y, &end_y, INPUT0_SIZE_Y);
    FUNC_CALL(calculate_index)(&step_x, &begin_x, &end_x, INPUT0_SIZE_X);
#endif // OUTPUT_LAYOUT_BFYX
#endif // SHRINK_MODE

    const uint slice_begin_batch = (uint)begin_batch;
    const uint slice_begin_feature = (uint)begin_feature;
    const uint slice_begin_w = (uint)begin_w;
    const uint slice_begin_z = (uint)begin_z;
    const uint slice_begin_y = (uint)begin_y;
    const uint slice_begin_x = (uint)begin_x;

    const uint slice_steps_batch = (uint)step_batch;
    const uint slice_steps_feature = (uint)step_feature;
    const uint slice_steps_w = (uint)step_w;
    const uint slice_steps_z = (uint)step_z;
    const uint slice_steps_y = (uint)step_y;
    const uint slice_steps_x = (uint)step_x;

    printf("slice begin: b(%u) f(%u) w(%u) z(%u) y(%u) x(%u), slice step: b(%u) f(%u) w(%u) z(%u) y(%u) x(%u)\n",
            slice_begin_batch, slice_begin_feature, slice_begin_w, slice_begin_z, slice_begin_y, slice_begin_x,
            slice_steps_batch, slice_steps_feature, slice_steps_w, slice_steps_z, slice_steps_y, slice_steps_x);

#if NEW_AXIS_MODE
    // If NEW_AXIS_MODE that just copy input to output
#ifdef OUTPUT_LAYOUT_BFYX
    const uint w_input = 0;
    const uint z_input = 0;
    const uint y_input = (uint)get_global_id(2) / INPUT0_SIZE_X;
    const uint x_input = (uint)get_global_id(2) % INPUT0_SIZE_X;
#elif OUTPUT_LAYOUT_BFZYX
    const uint w_input = 0;
    const uint yx_input = (uint)get_global_id(2) % (INPUT0_SIZE_X * INPUT0_SIZE_Y);
    const uint z_input = (uint)get_global_id(2) / (INPUT0_SIZE_X * INPUT0_SIZE_Y);
    const uint y_input = yx_input / INPUT0_SIZE_X;
    const uint x_input = yx_input % INPUT0_SIZE_X;
#elif OUTPUT_LAYOUT_BFWZYX
    const uint zyx_input = (uint)get_global_id(2) % (INPUT0_SIZE_X * INPUT0_SIZE_Y * INPUT0_SIZE_Z);
    const uint w_input = (uint)get_global_id(2) / (INPUT0_SIZE_X * INPUT0_SIZE_Y * INPUT0_SIZE_Z);
    const uint z_input = zyx_input / (INPUT0_SIZE_X * INPUT0_SIZE_Y);
    const uint yx_input = zyx_input % (INPUT0_SIZE_X * INPUT0_SIZE_Y);
    const uint y_input = yx_input / INPUT0_SIZE_X;
    const uint x_input = yx_input % INPUT0_SIZE_X;
#endif
    const uint input_index = INPUT0_OFFSET +
        batch * INPUT0_BATCH_PITCH +
        feature * INPUT0_FEATURE_PITCH +
        w_input * INPUT0_W_PITCH +
        z_input * INPUT0_Z_PITCH +
        y_input * INPUT0_Y_PITCH +
        x_input * INPUT0_X_PITCH;
    output[input_index] = input[input_index];
#else // NEW_AXIS_MODE
#ifdef OUTPUT_LAYOUT_BFYX
    const uint w = 0;
    const uint z = 0;
    const uint y = get_global_id(2) / OUTPUT_SIZE_X;
    const uint x = get_global_id(2) % OUTPUT_SIZE_X;
#elif OUTPUT_LAYOUT_BFZYX
    const uint w = 0;
    const uint yx = get_global_id(2) % (OUTPUT_SIZE_X * OUTPUT_SIZE_Y);
    const uint z = get_global_id(2) / (OUTPUT_SIZE_X * OUTPUT_SIZE_Y);
    const uint y = yx / OUTPUT_SIZE_X;
    const uint x = yx % OUTPUT_SIZE_X;
#elif OUTPUT_LAYOUT_BFWZYX
    const uint zyx = (uint)get_global_id(2) % (OUTPUT_SIZE_X * OUTPUT_SIZE_Y * OUTPUT_SIZE_Z);
    const uint w = (uint)get_global_id(2) / (OUTPUT_SIZE_X * OUTPUT_SIZE_Y * OUTPUT_SIZE_Z);
    const uint z = zyx / (OUTPUT_SIZE_X * OUTPUT_SIZE_Y);
    const uint yx = zyx % (OUTPUT_SIZE_X * OUTPUT_SIZE_Y);
    const uint y = yx / OUTPUT_SIZE_X;
    const uint x = yx % OUTPUT_SIZE_X;
#endif

    printf("batch(%u) feature(%u) w(%u) z(%u) y(%u) x(%u)\n", batch, feature, w, z, y, x);

#if SHRINK_MODE
    const uint in_indices[] = {INPUT_INDICES_ORDER};
    const uint input_index = INPUT0_OFFSET +
        (slice_begin_batch + in_indices[0] * slice_steps_batch) * INPUT0_BATCH_PITCH +
        (slice_begin_feature + in_indices[1] * slice_steps_feature) * INPUT0_FEATURE_PITCH +
    #if INPUT0_LAYOUT_BFWZYX
        (slice_begin_w + in_indices[2] * slice_steps_w) * INPUT0_W_PITCH +
        (slice_begin_z + in_indices[3] * slice_steps_z) * INPUT0_Z_PITCH +
        (slice_begin_y + in_indices[4] * slice_steps_y) * INPUT0_Y_PITCH +
        (slice_begin_x + in_indices[5] * slice_steps_x) * INPUT0_X_PITCH;
    #elif INPUT0_LAYOUT_BFZYX
        (slice_begin_z + in_indices[2] * slice_steps_z) * INPUT0_Z_PITCH +
        (slice_begin_y + in_indices[3] * slice_steps_y) * INPUT0_Y_PITCH +
        (slice_begin_x + in_indices[4] * slice_steps_x) * INPUT0_X_PITCH;
    #else
        (slice_begin_y + in_indices[2] * slice_steps_y) * INPUT0_Y_PITCH +
        (slice_begin_x + in_indices[3] * slice_steps_x) * INPUT0_X_PITCH;
    #endif
#else // SHRINK_MODE
    const uint in_b = (slice_begin_batch + batch * slice_steps_batch) * INPUT0_BATCH_PITCH;
    const uint in_f = (slice_begin_feature + feature * slice_steps_feature) * INPUT0_FEATURE_PITCH;
    // const uint input_index = INPUT0_OFFSET +
    //         (slice_begin_batch + batch * slice_steps_batch) * INPUT0_BATCH_PITCH +
    //         (slice_begin_feature + feature * slice_steps_feature) * INPUT0_FEATURE_PITCH +
    #if INPUT0_LAYOUT_BFWZYX
            // (slice_begin_w + w * slice_steps_w) * INPUT0_W_PITCH +
            // (slice_begin_z + z * slice_steps_z) * INPUT0_Z_PITCH +
            // (slice_begin_y + y * slice_steps_y) * INPUT0_Y_PITCH +
            // (slice_begin_x + x * slice_steps_x) * INPUT0_X_PITCH;
    const uint in_w = (slice_begin_w + w * slice_steps_w) * INPUT0_W_PITCH;
    const uint in_z = (slice_begin_z + z * slice_steps_z) * INPUT0_Z_PITCH;
    const uint in_y = (slice_begin_y + y * slice_steps_y) * INPUT0_Y_PITCH;
    const uint in_x = (slice_begin_x + x * slice_steps_x) * INPUT0_X_PITCH;
    #elif INPUT0_LAYOUT_BFZYX
            // (slice_begin_z + z * slice_steps_z) * INPUT0_Z_PITCH +
            // (slice_begin_y + y * slice_steps_y) * INPUT0_Y_PITCH +
            // (slice_begin_x + x * slice_steps_x) * INPUT0_X_PITCH;
    const uint in_w = 0;
    const uint in_z = (slice_begin_z + z * slice_steps_z) * INPUT0_Z_PITCH;
    const uint in_y = (slice_begin_y + y * slice_steps_y) * INPUT0_Y_PITCH;
    const uint in_x = (slice_begin_x + x * slice_steps_x) * INPUT0_X_PITCH;
    #else
            // (slice_begin_y + y * slice_steps_y) * INPUT0_Y_PITCH +
            // (slice_begin_x + x * slice_steps_x) * INPUT0_X_PITCH;
    const uint in_w = 0;
    const uint in_z = 0;
    const uint in_y = (slice_begin_y + y * slice_steps_y) * INPUT0_Y_PITCH;
    const uint in_x = (slice_begin_x + x * slice_steps_x) * INPUT0_X_PITCH;
    #endif
    const uint test_b = slice_begin_batch + batch * slice_steps_batch;
    const uint test_f = slice_begin_feature + feature * slice_steps_feature;
    const uint test_w = 0;
    const uint test_z = slice_begin_z + z * slice_steps_z;
    const uint test_y = slice_begin_y + y * slice_steps_y;
    const uint test_x = slice_begin_x + x * slice_steps_x;
    printf("INPUT0_OFFSET(%u) test_b(%u) test_f(%u) test_w(%u) test_z(%u) test_y(%u) test_x(%u)\n", INPUT0_OFFSET,
           test_b, test_f, test_w, test_z, test_y, test_x);
    const uint input_index_test = FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR test_b, test_f, 0, test_z, 0, 0);
    const uint input_index = INPUT0_OFFSET + in_b + in_f + in_w + in_z + in_y + in_x;
#endif // SHRINK_MODE
    printf("input_index(%u)  input_index_test(%u)!!!!!!!!!!! \n", input_index, input_index_test);
    const uint output_index = OUTPUT_OFFSET +
            batch * OUTPUT_BATCH_PITCH +
            feature * OUTPUT_FEATURE_PITCH +
            w * OUTPUT_W_PITCH +
            z * OUTPUT_Z_PITCH +
            y * OUTPUT_Y_PITCH +
            x * OUTPUT_X_PITCH;

    printf("INPUT0_OFFSET(%u), slice_begin: b(%u) f(%u) w(%u) z(%u) y(%u) x(%u), slice_steps: b(%u) f(%u) w(%u) z(%u) y(%u) x(%u)\n",
            INPUT0_OFFSET,
            slice_begin_batch, slice_begin_feature, slice_begin_w, slice_begin_z, slice_begin_y, slice_begin_x,
            slice_steps_batch, slice_steps_feature, slice_steps_w, slice_steps_z, slice_steps_y, slice_steps_x);
    printf("input_index(%u) output_index(%u)\n", input_index, output_index);

#if HAS_FUSED_OPS
    INPUT0_TYPE input_data = input[input_index];
    FUSED_OPS;
    output[output_index] = FUSED_OPS_RESULT;
#else
    output[output_index] = ACTIVATION(input[input_index], ACTIVATION_PARAMS);
#endif
#endif // NEW_AXIS_MODE
}
