// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Optimized 2D bicubic interpolation for bfyx planar format.
//
// The generic resample_ref.cl CUBIC path iterates over all 5 dimensions
// (B, F, Z, Y, X) with unroll_for, producing 4^5 = 1024 loop iterations
// even for a 2D-only resize where B, F, Z are identity axes.  This drives
// high register pressure (REG256 observed) and poor occupancy.
//
// This kernel restricts itself to the common 2D-only case on bfyx layout:
//   - Batch and feature are NOT interpolated (in_B==out_B, in_F==out_F).
//   - Z dimension is absent (4D tensors only).
//   - Only Y and X are interpolated with cubic splines.
//
// Per work-item cost: 4x4 = 16 fma operations vs ~1024 in the ref path.

#include "include/fetch_utils.cl"

#ifdef RTE_OUTPUT
    #define TO_OUTPUT_TYPE(x) CAT(CAT(convert_, OUTPUT_TYPE), _rte)(x)
#else
    #define TO_OUTPUT_TYPE(x) CAT(convert_, OUTPUT_TYPE)(x)
#endif

inline float FUNC(get_original_coordinate)(float num, float scale,
                                           int length_resized, int length_original)
{
    if (scale == 1.0f)
        return num;
#if defined(COORD_TRANS_MODE_HALF_PIXEL)
    return (num + 0.5f) * scale - 0.5f;
#elif defined(COORD_TRANS_MODE_PYTORCH_HALF_PIXEL)
    return (length_resized > 1) ? (num + 0.5f) * scale - 0.5f : 0.f;
#elif defined(COORD_TRANS_MODE_ASYMMETRIC)
    return num * scale;
#elif defined(COORD_TRANS_MODE_TF_HALF_PIXEL_FOR_NN)
    return (num + 0.5f) * scale;
#elif defined(COORD_TRANS_MODE_ALIGN_CORNERS)
    return (length_resized != 1) ? num * (length_original - 1) / (length_resized - 1) : 0.f;
#else
    #error [clDNN resample_bfyx_cubic_opt.cl]: coordinate transformation mode - not supported
#endif
}

inline void FUNC(get_cubic_coeff)(float* cubic_coef, float coord, float coef)
{
    float abs_num = fabs(coord);
    cubic_coef[0] = coef * (abs_num - 1.0f) * (abs_num - 1.0f) * abs_num;
    cubic_coef[1] = ((coef + 2.0f) * abs_num - (coef + 3.0f)) * abs_num * abs_num + 1.0f;
    cubic_coef[2] = (((-coef - 2.0f) * abs_num + (2.0f * coef + 3.0f)) * abs_num - coef) * abs_num;
    cubic_coef[3] = -coef * abs_num * abs_num * (abs_num - 1.0f);
}

KERNEL(resample_bfyx_cubic_opt)(
    OPTIONAL_SHAPE_INFO_ARG
    __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
)
{
    const int ox = (int)get_global_id(0);
    const int oy = (int)get_global_id(1);
    const int f  = (int)get_global_id(2) % OUTPUT_FEATURE_NUM;
    const int b  = (int)get_global_id(2) / OUTPUT_FEATURE_NUM;

    const int in_width  = INPUT0_SIZE_X;
    const int in_height = INPUT0_SIZE_Y;

    // Map output coordinates to fractional input coordinates, accounting for
    // optional spatial padding on the input tensor.
    const float ix = FUNC_CALL(get_original_coordinate)(
                         ox, SCALES[4], OUTPUT_SIZE_X,
                         in_width  + PADS_BEGIN[4] + PADS_END[4]) - (float)PADS_BEGIN[4];
    const float iy = FUNC_CALL(get_original_coordinate)(
                         oy, SCALES[3], OUTPUT_SIZE_Y,
                         in_height + PADS_BEGIN[3] + PADS_END[3]) - (float)PADS_BEGIN[3];

    // Integer base coordinates (floor).
    const int ix0 = (int)floor(ix);
    const int iy0 = (int)floor(iy);

    // Fractional offsets from the base.
    const float fx = ix - (float)ix0;
    const float fy = iy - (float)iy0;

    // Cubic spline coefficients for X and Y axes only.
    float cx[4], cy[4];
    FUNC_CALL(get_cubic_coeff)(cx, fx, CUBE_COEFF);
    FUNC_CALL(get_cubic_coeff)(cy, fy, CUBE_COEFF);

    ACCUMULATOR_TYPE val = ACCUMULATOR_VAL_ZERO;

    // 4×4 = 16 iterations: dy in {-1,0,1,2}, dx in {-1,0,1,2}.
    unroll_for (int dy = 0; dy < 4; ++dy) {
        const int src_y = clamp(iy0 + dy - 1,
                                -PADS_BEGIN[3],
                                in_height + PADS_END[3] - 1);

        unroll_for (int dx = 0; dx < 4; ++dx) {
            const int src_x = clamp(ix0 + dx - 1,
                                    -PADS_BEGIN[4],
                                    in_width + PADS_END[4] - 1);

#if PADDING_USED == 1
            // Only accumulate if the clamped coordinate falls inside the
            // real (non-padded) input region; padding pixels contribute 0.
            if (src_y >= 0 && src_y < in_height &&
                src_x >= 0 && src_x < in_width)
#endif
            {
                val = fma((ACCUMULATOR_TYPE)(cy[dy] * cx[dx]),
                          (ACCUMULATOR_TYPE)input[INPUT0_GET_INDEX(b, f, src_y, src_x)],
                          val);
            }
        }
    }

#if HAS_FUSED_OPS
    #define OF_ID (f)
    FUSED_OPS;
    OUTPUT_TYPE res = FUSED_OPS_RESULT;
    #undef OF_ID
#else
    OUTPUT_TYPE res = ACTIVATION(TO_OUTPUT_TYPE(val), ACTIVATION_PARAMS);
#endif

    output[OUTPUT_GET_INDEX(b, f, oy, ox)] = res;
}

#undef TO_OUTPUT_TYPE
