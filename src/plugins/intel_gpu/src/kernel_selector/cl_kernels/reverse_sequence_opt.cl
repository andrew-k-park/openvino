// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"
#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"

// Subgroup-cooperative reverse_sequence (bfyx, X==1).
// SUB_GROUP_SIZE work-items in dim2 cooperate via intel_sub_group_block_read8
// /write8 to move SUB_GROUP_SIZE * 8 contiguous Y elements per subgroup. The
// kernel assumes:
//   - layout bfyx, 4D
//   - X == 1, no padding on Y (so INPUT0_Y_PITCH == 1)
//   - SEQ_AXIS != Y
//   - Y % (SUB_GROUP_SIZE * 8) == 0

REQD_SUB_GROUP_SIZE(SUB_GROUP_SIZE)
__attribute__((reqd_work_group_size(1, 1, SUB_GROUP_SIZE)))
KERNEL(reverse_sequence_opt)(const __global INPUT0_TYPE* input,
                             const __global INPUT1_TYPE* seq_lengths,
                             __global OUTPUT_TYPE* output)
{
    const uint batch   = get_global_id(0);
    const uint feature = get_global_id(1);
    // dim2 GWS = Y / (SUB_GROUP_SIZE * 8); we lose the per-subgroup id by
    // dividing get_global_id(2) by SUB_GROUP_SIZE.
    const uint y_block = (uint)get_global_id(2) / SUB_GROUP_SIZE;
    const uint y       = y_block * SUB_GROUP_SIZE * 8;
    const uint x       = 0;

    // Input is read at the original (un-reversed) coordinates.
    const uint input_index = INPUT0_GET_INDEX(batch, feature, y, x);

    // Output is written at the same coordinates with SEQ_AXIS reversed.
    uint dimensions[] = { batch, feature, y, x };
    const uint length = (uint)seq_lengths[dimensions[BATCH_AXIS]];
    if (dimensions[SEQ_AXIS] < length)
        dimensions[SEQ_AXIS] = length - dimensions[SEQ_AXIS] - 1;
    const uint output_index = OUTPUT_GET_INDEX(dimensions[0], dimensions[1], dimensions[2], dimensions[3]);

    // Subgroup-cooperative load / store: 16 work-items × 8 elements each
    // = 128 contiguous elements per call.
    MAKE_VECTOR_TYPE(INPUT0_TYPE, 8) v = DT_INPUT_BLOCK_READ8(input, input_index);
    DT_OUTPUT_BLOCK_WRITE8(output, output_index, v);
}
