// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Optimized permute kernel for B <-> F axis swap with innermost axes unchanged.
// Handles order [1, 0, 2, 3] (bfyx -> fbyx) and higher-dim equivalents.
// Since the innermost X dimension is contiguous in memory, no SLM transpose
// is needed — only vectorized load/store along X.
//
// Each work item loops over the entire F dimension so that consecutive
// work items in a SIMD wave share the same (b, y) and stream through
// input memory sequentially. This converts the output-side scatter
// (F output planes separated by B*Y*X) into a small register-held loop,
// dramatically improving DRAM burst utilisation.
//
// GWS: (ceil(X / VEC_WIDTH), Y [* Z [* W ...]], B)

#include "include/batch_headers/fetch_data.cl"

KERNEL(permute_b_f_axes)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
    )
{
    const uint x_t = get_global_id(0);  // X tile index

#if INPUT0_DIMS == 4
    const uint y   = get_global_id(1);
#elif INPUT0_DIMS == 5
    const uint z   = get_global_id(1) / INPUT0_SIZE_Y;
    const uint y   = get_global_id(1) % INPUT0_SIZE_Y;
#elif INPUT0_DIMS == 6
    const uint w   = get_global_id(1) / (INPUT0_SIZE_Z * INPUT0_SIZE_Y);
    const uint z   = get_global_id(1) / INPUT0_SIZE_Y % INPUT0_SIZE_Z;
    const uint y   = get_global_id(1) % INPUT0_SIZE_Y;
#endif

    const uint b   = get_global_id(2);  // F removed from GWS, looped below

    const uint x_base = x_t * VEC_WIDTH;

    // Loop over the full F dimension inside each work item.
    // Input reads are sequential in F (stride = Y*X elements),
    // output writes land in F separate planes — a predictable pattern.
    for (uint f = 0; f < INPUT0_FEATURE_NUM; ++f) {

#if X_REMAINDER_SIZE > 0
        if (x_t == X_TILES) {
            // scalar tail for the last partial tile
            for (uint i = 0; i < X_REMAINDER_SIZE; ++i) {
                const uint x = x_base + i;
#if INPUT0_DIMS == 4
                const uint in_idx  = INPUT0_GET_INDEX(b, f, y, x);
                const uint out_idx = OUTPUT_GET_INDEX(f, b, y, x);
#elif INPUT0_DIMS == 5
                const uint in_idx  = INPUT0_GET_INDEX(b, f, z, y, x);
                const uint out_idx = OUTPUT_GET_INDEX(f, b, z, y, x);
#elif INPUT0_DIMS == 6
                const uint in_idx  = INPUT0_GET_INDEX(b, f, w, z, y, x);
                const uint out_idx = OUTPUT_GET_INDEX(f, b, w, z, y, x);
#endif
                INPUT0_TYPE val = input[in_idx];
#if HAS_FUSED_OPS
                INPUT0_TYPE input_var = val;
                FUSED_OPS;
                output[out_idx] = FUSED_OPS_RESULT;
#else
                output[out_idx] = ACTIVATION(val, ACTIVATION_PARAMS);
#endif
            }
            continue;
        }
#endif  // X_REMAINDER_SIZE > 0

        // Full vectorized tile
#if INPUT0_DIMS == 4
        const uint in_idx  = INPUT0_GET_INDEX(b, f, y, x_base);
        const uint out_idx = OUTPUT_GET_INDEX(f, b, y, x_base);
#elif INPUT0_DIMS == 5
        const uint in_idx  = INPUT0_GET_INDEX(b, f, z, y, x_base);
        const uint out_idx = OUTPUT_GET_INDEX(f, b, z, y, x_base);
#elif INPUT0_DIMS == 6
        const uint in_idx  = INPUT0_GET_INDEX(b, f, w, z, y, x_base);
        const uint out_idx = OUTPUT_GET_INDEX(f, b, w, z, y, x_base);
#endif

        INPUTVTYPE vals = CAT(vload, VEC_WIDTH)(0, input + in_idx);
#if HAS_FUSED_OPS
        OUTPUTVTYPE out_vals;
        __attribute__((opencl_unroll_hint(VEC_WIDTH)))
        for (uint i = 0; i < VEC_WIDTH; ++i) {
            INPUT0_TYPE input_var = vals[i];
            FUSED_OPS;
            out_vals[i] = FUSED_OPS_RESULT;
        }
        CAT(vstore, VEC_WIDTH)(out_vals, 0, output + out_idx);
#else
        OUTPUTVTYPE out_vals = ACTIVATION(CAT(convert_, OUTPUTVTYPE)(vals), ACTIVATION_PARAMS);
        CAT(vstore, VEC_WIDTH)(out_vals, 0, output + out_idx);
#endif
    }  // for f
}
