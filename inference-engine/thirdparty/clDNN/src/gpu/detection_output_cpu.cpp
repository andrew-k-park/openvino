// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "detection_output_inst.h"
#include "kernel.h"
#include "network_impl.h"
#include "implementation_map.h"
#include "math_utils.h"
#include "register_gpu.hpp"
#include "cpu_impl_helpers.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <xmmintrin.h>
#include <vector>
#include <utility>
#include <chrono>
#include <memory>

#ifdef FIX_OPENMP_RELEASE_ISSUE
#ifdef OPENMP_FOUND
#include <omp.h>
#endif
#endif

namespace cldnn {
namespace gpu {

namespace {
    using bounding_box = cldnn::cpu::bounding_box;
}  // namespace

/************************ Detection Output CPU ************************/
struct detection_output_cpu : typed_primitive_impl<detection_output> {
    const detection_output_node& outer;

    explicit detection_output_cpu(const detection_output_node& outer) : outer(outer) {}

    typedef struct {
        int batchId;
        int classId;
        int boxId;
        float score;
    } Scores;

    static void IntersectBBox(const bounding_box& bbox1,
                              const bounding_box& bbox2,
                              bounding_box& intersectBbox) {
        if (bbox2.xmin > bbox1.xmax || bbox2.xmax < bbox1.xmin ||
            bbox2.ymin > bbox1.ymax || bbox2.ymax < bbox1.ymin) {
            intersectBbox.xmin = 0;
            intersectBbox.ymin = 0;
            intersectBbox.xmax = 0;
            intersectBbox.ymax = 0;
        } else {
            intersectBbox.xmin = std::max<float>(bbox1.xmin, bbox2.xmin);
            intersectBbox.ymin = std::max<float>(bbox1.ymin, bbox2.ymin);
            intersectBbox.xmax = std::min<float>(bbox1.xmax, bbox2.xmax);
            intersectBbox.ymax = std::min<float>(bbox1.ymax, bbox2.ymax);
        }
    }

    static float JaccardOverlap(const bounding_box& bbox1, const bounding_box& bbox2) {
        bounding_box intersectBbox;
        IntersectBBox(bbox1, bbox2, intersectBbox);

        float intersectWidth, intersectHeight;
        intersectWidth = intersectBbox.xmax - intersectBbox.xmin;
        intersectHeight = intersectBbox.ymax - intersectBbox.ymin;
        if (intersectWidth > 0 && intersectHeight > 0) {
            float intersect_size = intersectWidth * intersectHeight;
            float bbox1_size = bbox1.area();
            float bbox2_size = bbox2.area();
            return intersect_size / (bbox1_size + bbox2_size - intersect_size);
        } else {
            return 0.0f;
        }
    }

    template <typename T>
    static bool SortScorePairDescend(const std::pair<float, T>& pair1,
                                     const std::pair<float, T>& pair2) {
        return pair1.first > pair2.first;
    }

    template <typename dtype>
    void stage_0_decode_bbox(const detection_output_inst& instance,
                             std::vector<std::vector<bounding_box>>& bboxes_per_image,
                             const int idx_image,
                             const int num_loc_classes,
                             const int prior_info_size,
                             const int prior_coordinates_offset,
                             const prior_box_code_type code_type,
                             const bool variance_encoded_in_target,
                             const bool prior_is_normalized,
                             const size_t image_width,
                             const size_t image_height,
                             const bool clip_before_nms,
                             const int num_of_priors) {
        const auto& args = instance.argument;

        auto& input_location = instance.location_memory();
        mem_lock<dtype> lock_location{input_location};
        auto location_data = lock_location.begin();
        const auto& input_buffer_size = input_location.get_layout().get_buffer_size();
        const int input_buffer_size_f = input_buffer_size.feature[0];
        const int input_buffer_size_x = input_buffer_size.spatial[0];
        const int input_buffer_size_y = input_buffer_size.spatial[1];
        const auto& input_padding = input_location.get_layout().data_padding;
        const int input_padding_lower_x = input_padding.lower_size().spatial[0];
        const int input_padding_lower_y = input_padding.lower_size().spatial[1];
        const int location_size_product = input_buffer_size_y * input_buffer_size_x;
        const int location_padding = input_padding_lower_y * input_buffer_size_x + input_padding_lower_x;

        auto& input_prior_box = instance.prior_box_memory();
        mem_lock<dtype> lock_prior_box{input_prior_box};
        const int num_of_prior_components = num_of_priors * prior_info_size;
        auto prior_data = lock_prior_box.begin() + idx_image * num_of_prior_components * (variance_encoded_in_target ? 1 : 2);

        for (int prior = 0; prior < num_of_priors; ++prior) {
            const int prior_offset = prior * prior_info_size + prior_coordinates_offset;
            const int variance_offset = num_of_prior_components + (prior * PRIOR_BOX_SIZE);
            float prior_bbox_xmin = static_cast<float>(prior_data[prior_offset]);
            float prior_bbox_ymin = static_cast<float>(prior_data[prior_offset + 1]);
            float prior_bbox_xmax = static_cast<float>(prior_data[prior_offset + 2]);
            float prior_bbox_ymax = static_cast<float>(prior_data[prior_offset + 3]);

            if (!prior_is_normalized) {
                prior_bbox_xmin /= image_width;
                prior_bbox_ymin /= image_height;
                prior_bbox_xmax /= image_width;
                prior_bbox_ymax /= image_height;
            }

            for (int cls = 0; cls < num_loc_classes; ++cls) {
                const int label = args.share_location ? 0 : cls;
                if (!args.share_location && label == args.background_label_id) {
                    continue;
                }
                const int locations_offset =
                    (num_loc_classes * (prior * PRIOR_BOX_SIZE) + idx_image * input_buffer_size_f + cls * PRIOR_BOX_SIZE)
                    * location_size_product + location_padding;
                bounding_box decoded_bbox;
                if (code_type == prior_box_code_type::corner) {
                    if (variance_encoded_in_target) {
                        decoded_bbox.xmin = prior_bbox_xmin + static_cast<float>(location_data[locations_offset]);
                        decoded_bbox.ymin = prior_bbox_ymin + static_cast<float>(location_data[locations_offset + location_size_product]);
                        decoded_bbox.xmax = prior_bbox_xmax + static_cast<float>(location_data[locations_offset + 2 * location_size_product]);
                        decoded_bbox.ymax = prior_bbox_ymax + static_cast<float>(location_data[locations_offset + 3 * location_size_product]);
                    } else {
                        decoded_bbox.xmin = prior_bbox_xmin + static_cast<float>(prior_data[variance_offset])
                                            * static_cast<float>(location_data[locations_offset]);
                        decoded_bbox.ymin = prior_bbox_ymin + static_cast<float>(prior_data[variance_offset + 1])
                                            * static_cast<float>(location_data[locations_offset + location_size_product]);
                        decoded_bbox.xmax = prior_bbox_xmax + static_cast<float>(prior_data[variance_offset + 2])
                                            * static_cast<float>(location_data[locations_offset + 2 * location_size_product]);
                        decoded_bbox.ymax = prior_bbox_ymax + static_cast<float>(prior_data[variance_offset + 3])
                                            * static_cast<float>(location_data[locations_offset + 3 * location_size_product]);
                    }
                } else if (code_type == prior_box_code_type::center_size) {
                    const float prior_width = prior_bbox_xmax - prior_bbox_xmin;
                    // assert(prior_width > 0);
                    const float prior_height = prior_bbox_ymax - prior_bbox_ymin;
                    // assert(prior_height > 0);
                    const float prior_center_x = (prior_bbox_xmin + prior_bbox_xmax) / 2.f;
                    const float prior_center_y = (prior_bbox_ymin + prior_bbox_ymax) / 2.f;
                    const float bbox_xmin = static_cast<float>(location_data[locations_offset]);
                    const float bbox_ymin = static_cast<float>(location_data[locations_offset + location_size_product]);
                    const float bbox_xmax = static_cast<float>(location_data[locations_offset + 2 * location_size_product]);
                    const float bbox_ymax = static_cast<float>(location_data[locations_offset + 3 * location_size_product]);
                    float decode_bbox_center_x, decode_bbox_center_y;
                    float decode_bbox_width, decode_bbox_height;
                    if (variance_encoded_in_target) {
                        // variance is encoded in target, we simply need to restore the offset predictions.
                        decode_bbox_center_x = bbox_xmin * prior_width + prior_center_x;
                        decode_bbox_center_y = bbox_ymin * prior_height + prior_center_y;
                        decode_bbox_width = (exp(bbox_xmax) * prior_width);
                        decode_bbox_height = (exp(bbox_ymax) * prior_height);
                    } else {
                        // variance is encoded in bbox, we need to scale the offset accordingly.
                        decode_bbox_center_x = static_cast<float>(prior_data[variance_offset])
                                            * bbox_xmin * prior_width + prior_center_x;
                        decode_bbox_center_y = static_cast<float>(prior_data[variance_offset + 1])
                                            * bbox_ymin * prior_height + prior_center_y;
                        decode_bbox_width = (exp(static_cast<float>(prior_data[variance_offset + 2]) * bbox_xmax)
                                            * prior_width);
                        decode_bbox_height = (exp(static_cast<float>(prior_data[variance_offset + 3]) * bbox_ymax)
                                            * prior_height);
                    }
                    decoded_bbox.xmin = decode_bbox_center_x - decode_bbox_width / 2.0f;
                    decoded_bbox.ymin = decode_bbox_center_y - decode_bbox_height / 2.0f;
                    decoded_bbox.xmax = decode_bbox_center_x + decode_bbox_width / 2.0f;
                    decoded_bbox.ymax = decode_bbox_center_y + decode_bbox_height / 2.0f;
                } else { // prior_box_code_type::corner_size
                    const float prior_width = prior_bbox_xmax - prior_bbox_xmin;
                    assert(prior_width > 0);
                    const float prior_height = prior_bbox_ymax - prior_bbox_ymin;
                    assert(prior_height > 0);
                    const float bbox_xmin = static_cast<float>(location_data[locations_offset]);
                    const float bbox_ymin = static_cast<float>(location_data[locations_offset + location_size_product]);
                    const float bbox_xmax = static_cast<float>(location_data[locations_offset + 2 * location_size_product]);
                    const float bbox_ymax = static_cast<float>(location_data[locations_offset + 3 * location_size_product]);
                    if (variance_encoded_in_target) {
                        // variance is encoded in target, we simply need to add the offset predictions.
                        decoded_bbox.xmin = prior_bbox_xmin + bbox_xmin * prior_width;
                        decoded_bbox.ymin = prior_bbox_ymin + bbox_ymin * prior_height;
                        decoded_bbox.xmax = prior_bbox_xmax + bbox_xmax * prior_width;
                        decoded_bbox.ymax = prior_bbox_ymax + bbox_ymax * prior_height;
                    } else {
                        // variance is encoded in bbox, we need to scale the offset accordingly.
                        decoded_bbox.xmin = prior_bbox_xmin + static_cast<float>(prior_data[variance_offset])
                                            * bbox_xmin * prior_width;
                        decoded_bbox.ymin = prior_bbox_ymin + static_cast<float>(prior_data[variance_offset + 1])
                                            * bbox_ymin * prior_height;
                        decoded_bbox.xmax = prior_bbox_xmax + static_cast<float>(prior_data[variance_offset + 2])
                                            * bbox_xmax * prior_width;
                        decoded_bbox.ymax = prior_bbox_ymax + static_cast<float>(prior_data[variance_offset + 3])
                                            * bbox_ymax * prior_height;
                    }
                }
                if (clip_before_nms) {
                    decoded_bbox.xmin = std::max(0.0f, std::min(1.0f, decoded_bbox.xmin));
                    decoded_bbox.ymin = std::max(0.0f, std::min(1.0f, decoded_bbox.ymin));
                    decoded_bbox.xmax = std::max(0.0f, std::min(1.0f, decoded_bbox.xmax));
                    decoded_bbox.ymax = std::max(0.0f, std::min(1.0f, decoded_bbox.ymax));
                }
                bboxes_per_image[label].emplace_back(decoded_bbox);
            }
        }
    }

    template <typename dtype>
    void stage_0_work_item_with_remainder(float* input_confidence,
                                          unsigned char* buffer1,
                                          unsigned char* buffer2,
                                          int idx_scores,
                                          const int num_classes,
                                          const int num_of_priors,
                                          const float confidence_threshold,
                                          const int remainder) {
        Scores* scores = reinterpret_cast<Scores*>(buffer1);
        int* num_scores = reinterpret_cast<int*>(buffer2);

        const int feature_size = num_classes * num_of_priors;
        const int idx_image = idx_scores / feature_size;

        for (int idx_remainder = idx_scores; idx_remainder < idx_scores + remainder; idx_remainder++) {
            float score = input_confidence[idx_remainder];
            if (score > confidence_threshold) {
                int idx_class = idx_remainder % num_classes;
                int idx_prior = idx_remainder / num_classes;
                int num_scores_offset = idx_image * num_classes + idx_class;
                int acc_num = num_scores[num_scores_offset];
                int scores_offset = (idx_image * feature_size) + idx_class * num_of_priors + acc_num;
                Scores score_info = { idx_image, idx_class, idx_prior, score };
                scores[scores_offset] = score_info;
                num_scores[num_scores_offset] += 1;
            }
        }
    }

    template <typename dtype>
    void stage_0_work_item(float* input_confidence,
                           unsigned char* buffer1,
                           unsigned char* buffer2,
                           const int idx_scores,
                           const int num_of_images,
                           const int num_classes,
                           const int num_of_priors,
                           const float confidence_threshold) {
        Scores* scores = reinterpret_cast<Scores*>(buffer1);
        int* num_scores = reinterpret_cast<int*>(buffer2);
        if (idx_scores == 0) {
            for (int idx_num_scores = 0; idx_num_scores < num_of_images * num_classes; idx_num_scores++) {
                num_scores[idx_num_scores] = 0;
            }
        }

        const int feature_size = num_classes * num_of_priors;
        const int idx_image = idx_scores / feature_size;
        float const* confidence_ptr_float = (float const*)(input_confidence);
        confidence_ptr_float += idx_scores;
        __m128 threshold = _mm_load_ps1(&confidence_threshold);
        __m128 scores_0 = _mm_loadu_ps(confidence_ptr_float);
        confidence_ptr_float += 4;
        __m128 scores_1 = _mm_loadu_ps(confidence_ptr_float);
        __m128i mask128_0 = _mm_castps_si128(_mm_cmpgt_ps(scores_0, threshold));
        __m128i mask128_1 = _mm_castps_si128(_mm_cmpgt_ps(scores_1, threshold));

        int mask = _mm_movemask_ps(_mm_castsi128_ps(mask128_0));
        if (mask & 1) {
            float s = _mm_cvtss_f32(scores_0);
            int idx_class = idx_scores % num_classes;
            int idx_prior = idx_scores / num_classes;
            int num_scores_offset = idx_image * num_classes + idx_class;
            int acc_num = num_scores[num_scores_offset];
            int scores_offset = (idx_image * feature_size) + idx_class * num_of_priors + acc_num;
            Scores score_info = { idx_image, idx_class, idx_prior, s };
            scores[scores_offset] = score_info;
            num_scores[num_scores_offset] += 1;
        }
        if (mask & 2) {
            int score = _mm_extract_ps(scores_0, 1);
            float s = reinterpret_cast<float&>(score);
            int idx_class = (idx_scores + 1) % num_classes;
            int idx_prior = (idx_scores + 1) / num_classes;
            int num_scores_offset = idx_image * num_classes + idx_class;
            int acc_num = num_scores[num_scores_offset];
            int scores_offset = (idx_image * feature_size) + idx_class * num_of_priors + acc_num;
            Scores score_info = { idx_image, idx_class, idx_prior, s };
            scores[scores_offset] = score_info;
            num_scores[num_scores_offset] += 1;
        }
        if (mask & 4) {
            int score = _mm_extract_ps(scores_0, 2);
            float s = reinterpret_cast<float&>(score);
            int idx_class = (idx_scores + 2) % num_classes;
            int idx_prior = (idx_scores + 2) / num_classes;
            int num_scores_offset = idx_image * num_classes + idx_class;
            int acc_num = num_scores[num_scores_offset];
            int scores_offset = (idx_image * feature_size) + idx_class * num_of_priors + acc_num;
            Scores score_info = { idx_image, idx_class, idx_prior, s };
            scores[scores_offset] = score_info;
            num_scores[num_scores_offset] += 1;
        }
        if (mask & 8) {
            int score = _mm_extract_ps(scores_0, 3);
            float s = reinterpret_cast<float&>(score);
            int idx_class = (idx_scores + 3) % num_classes;
            int idx_prior = (idx_scores + 3) / num_classes;
            int num_scores_offset = idx_image * num_classes + idx_class;
            int acc_num = num_scores[num_scores_offset];
            int scores_offset = (idx_image * feature_size) + idx_class * num_of_priors + acc_num;
            Scores score_info = { idx_image, idx_class, idx_prior, s };
            scores[scores_offset] = score_info;
            num_scores[num_scores_offset] += 1;
        }
        mask = _mm_movemask_ps(_mm_castsi128_ps(mask128_1));
        if (mask & 1) {
            float s = _mm_cvtss_f32(scores_1);
            int idx_class = (idx_scores + 4) % num_classes;
            int idx_prior = (idx_scores + 4) / num_classes;
            int num_scores_offset = idx_image * num_classes + idx_class;
            int acc_num = num_scores[num_scores_offset];
            int scores_offset = (idx_image * feature_size) + idx_class * num_of_priors + acc_num;
            Scores score_info = { idx_image, idx_class, idx_prior, s };
            scores[scores_offset] = score_info;
            num_scores[num_scores_offset] += 1;
        }
        if (mask & 2) {
            int score = _mm_extract_ps(scores_1, 1);
            float s = reinterpret_cast<float&>(score);
            int idx_class = (idx_scores + 5) % num_classes;
            int idx_prior = (idx_scores + 5) / num_classes;
            int num_scores_offset = idx_image * num_classes + idx_class;
            int acc_num = num_scores[num_scores_offset];
            int scores_offset = (idx_image * feature_size) + idx_class * num_of_priors + acc_num;
            Scores score_info = { idx_image, idx_class, idx_prior, s };
            scores[scores_offset] = score_info;
            num_scores[num_scores_offset] += 1;
        }
        if (mask & 4) {
            int score = _mm_extract_ps(scores_1, 2);
            float s = reinterpret_cast<float&>(score);
            int idx_class = (idx_scores + 6) % num_classes;
            int idx_prior = (idx_scores + 6) / num_classes;
            int num_scores_offset = idx_image * num_classes + idx_class;
            int acc_num = num_scores[num_scores_offset];
            int scores_offset = (idx_image * feature_size) + idx_class * num_of_priors + acc_num;
            Scores score_info = { idx_image, idx_class, idx_prior, s };
            scores[scores_offset] = score_info;
            num_scores[num_scores_offset] += 1;
        }
        if (mask & 8) {
            int score = _mm_extract_ps(scores_1, 3);
            float s = reinterpret_cast<float&>(score);
            int idx_class = (idx_scores + 7) % num_classes;
            int idx_prior = (idx_scores + 7) / num_classes;
            int num_scores_offset = idx_image * num_classes + idx_class;
            int acc_num = num_scores[num_scores_offset];
            int scores_offset = (idx_image * feature_size) + idx_class * num_of_priors + acc_num;
            Scores score_info = { idx_image, idx_class, idx_prior, s };
            scores[scores_offset] = score_info;
            num_scores[num_scores_offset] += 1;
        }
    }

    template <typename dtype>
    void stage_0_extract_confidence(const detection_output_inst& instance,
                                    std::vector<std::vector<std::pair<float, int>>>& scores_per_image,
                                    const int idx_image,
                                    const int num_of_priors,
                                    const int num_classes,
                                    const float confidence_threshold) {
        auto& input_confidence = instance.confidence_memory();
        mem_lock<dtype> lock_confidence{input_confidence};
        auto confidence_data = lock_confidence.begin();

        const auto& input_buffer_size = input_confidence.get_layout().get_buffer_size();
        const int input_buffer_size_f = input_buffer_size.feature[0];
        const int input_buffer_size_x = input_buffer_size.spatial[0];
        const int input_buffer_size_y = input_buffer_size.spatial[1];
        const auto& input_padding = input_confidence.get_layout().data_padding;
        const int input_padding_lower_x = input_padding.lower_size().spatial[0];
        const int input_padding_lower_y = input_padding.lower_size().spatial[1];
        const int confidence_size_product = input_buffer_size_y * input_buffer_size_x;
        const int confidence_padding = input_padding_lower_y * input_buffer_size_x + input_padding_lower_x;

        int idx = (idx_image * input_buffer_size_f) * confidence_size_product + confidence_padding;
        if (confidence_size_product == 1 && std::is_same<dtype, float>::value) {
            float const* confidence_ptr_float = (float const*)(&(*confidence_data));
            confidence_ptr_float += idx;
            __m128 threshold = _mm_load_ps1(&confidence_threshold);
            for (int prior = 0; prior < num_of_priors; ++prior) {
                int cls = 0;
                for (; cls + 3 < num_classes; cls += 4) {
                    __m128 scores = _mm_loadu_ps(confidence_ptr_float);
                    confidence_ptr_float += 4;
                    __m128i mask128 = _mm_castps_si128(_mm_cmpgt_ps(scores, threshold));
                    if (_mm_testz_si128(mask128, mask128)) {
                        continue;
                    }
                    int mask = _mm_movemask_ps(_mm_castsi128_ps(mask128));
                    if (mask & 1) {
                        scores_per_image[cls + 0].emplace_back(_mm_cvtss_f32(scores), prior);
                    }
                    if (mask & 2) {
                        int score = _mm_extract_ps(scores, 1);
                        float s = reinterpret_cast<float&>(score);
                        scores_per_image[cls + 1].emplace_back(s, prior);
                    }
                    if (mask & 4) {
                        int score = _mm_extract_ps(scores, 2);
                        float s = reinterpret_cast<float&>(score);
                        scores_per_image[cls + 2].emplace_back(s, prior);
                    }
                    if (mask & 8) {
                        int score = _mm_extract_ps(scores, 3);
                        float s = reinterpret_cast<float&>(score);
                        scores_per_image[cls + 3].emplace_back(s, prior);
                    }
                }
                for (; cls < num_classes; ++cls) {
                    float score = *confidence_ptr_float;
                    if (score > confidence_threshold) {
                        scores_per_image[cls].emplace_back(score, prior);
                    }
                    ++confidence_ptr_float;
                }
            }
        } else {
            for (int prior = 0; prior < num_of_priors; ++prior) {
                for (int cls = 0; cls < num_classes; ++cls) {
                    float score = static_cast<float>(confidence_data[idx]);
                    if (score > confidence_threshold) {
                        scores_per_image[cls].emplace_back(score, prior);
                    }
                    idx += confidence_size_product;
                }
            }
        }
    }

    static void sort(std::vector<std::pair<float, int>>& scores,
                     const int top_k) {
        // std::stable_sort(scores.begin(), scores.end(), SortScorePairDescend<int>);
        std::stable_sort(scores.begin(),
                         scores.end(),
                         [](const std::pair<float, int>& p1, const std::pair<float, int>& p2) {
                             return (p1.first > p2.first) || (p1.first == p2.first && p1.second < p2.second);
                         });

        if (top_k > -1 && static_cast<size_t>(top_k) < static_cast<size_t>(scores.size())) {
            scores.resize(top_k);
        }
    }

    static void calc_iou_keep_and_throw(const std::vector<bounding_box>& bboxes,
                                        std::vector<std::pair<float, int>>& scores,
                                        std::vector<int>& indices_per_cls,
                                        const float nms_threshold) {
        for (auto& s : scores) {
            const int idx = s.second;
            bool keep = true;
            for (int k = 0; k < static_cast<int>(indices_per_cls.size()); ++k) {
                const int kept_idx = indices_per_cls[k];
                float overlap = JaccardOverlap(bboxes[idx], bboxes[kept_idx]);
                if (overlap > nms_threshold) {
                    keep = false;
                    break;
                }
            }
            if (keep) {
                indices_per_cls.push_back(idx);
            }
        }
    }

    static void keep_top_k_and_throw(std::vector<std::vector<std::pair<float, int>>>& scores_per_image,
                                     std::vector<std::vector<std::pair<float, int>>>& new_indices,
                                     std::map<int, std::vector<int>>& indices_per_image,
                                     const int num_det,
                                     const int keep_top_k) {
        std::vector<std::pair<float, std::pair<int, int>>> score_index_pairs;
        for (auto it = indices_per_image.begin(); it != indices_per_image.end(); ++it) {
            int label = it->first;
            const std::vector<int>& labelIndices = it->second;
            std::vector<std::pair<float, int>>& scores = scores_per_image[label];
            for (int j = 0; j < static_cast<int>(labelIndices.size()); ++j) {
                int idx = labelIndices[j];
                for (const auto& s : scores) {
                    if (s.second == idx) score_index_pairs.push_back(std::make_pair(s.first, std::make_pair(label, idx)));
                }
            }
        }

        // std::sort(score_index_pairs.begin(),
        //           score_index_pairs.end(),
        //           SortScorePairDescend<std::pair<int, int>>);
        std::sort(score_index_pairs.begin(),
                  score_index_pairs.end(),
                  [](const std::pair<float, std::pair<int, int>>& p1, const std::pair<float, std::pair<int, int>>& p2) {
                      return (p1.first > p2.first) || (p1.first == p2.first && p1.second.second < p2.second.second);
                  });
        if (keep_top_k > -1 && num_det > keep_top_k) {
            score_index_pairs.resize(keep_top_k);
        }

        for (int j = 0; j < static_cast<int>(score_index_pairs.size()); ++j) {
            int label = score_index_pairs[j].second.first;
            int idx = score_index_pairs[j].second.second;
            new_indices[label].emplace_back(score_index_pairs[j].first, idx);
        }
    }

    template <typename dtype>
    void stage_0(const detection_output_inst& instance,
                 std::vector<std::vector<std::vector<bounding_box>>>& bboxes,
                 std::vector<std::vector<std::vector<std::pair<float, int>>>>& confidences) {
        const auto& args = instance.argument;

        const int num_of_images = static_cast<int>(bboxes.size());
        const int num_of_priors = instance.prior_box_memory().get_layout().size.spatial[1] / args.prior_info_size;
        const int num_loc_classes = args.share_location ? 1 : args.num_classes;
        const int num_classes = static_cast<int>(args.num_classes);
        const int block_size = 8;

        auto& input_confidence = instance.confidence_memory();
        mem_lock<dtype> lock_confidence{input_confidence};
        auto confidence_data = lock_confidence.begin();

        const auto& input_buffer_size = input_confidence.get_layout().get_buffer_size();
        const int input_buffer_size_f = input_buffer_size.feature[0];
        const int input_buffer_size_x = input_buffer_size.spatial[0];
        const int input_buffer_size_y = input_buffer_size.spatial[1];
        const auto& input_padding = input_confidence.get_layout().data_padding;
        const int input_padding_lower_x = input_padding.lower_size().spatial[0];
        const int input_padding_lower_y = input_padding.lower_size().spatial[1];
        const int confidence_size_product = input_buffer_size_y * input_buffer_size_x;
        const int confidence_padding = input_padding_lower_y * input_buffer_size_x + input_padding_lower_x;

        constexpr size_t buffer_bytes = 16;
        size_t buffer_stride = num_of_priors * buffer_bytes;
        size_t buffer1_size = num_of_images * num_classes * buffer_stride;
        size_t buffer2_size = num_of_images * num_classes * 4;
        std::unique_ptr<unsigned char[]> intermediate_scores(new unsigned char[buffer1_size]());
        std::unique_ptr<unsigned char[]> intermediate_box_num(new unsigned char[buffer2_size]());

        if (confidence_size_product == 1 && confidence_padding == 0 && std::is_same<dtype, float>::value) {
            const int total_scores = num_of_images * input_buffer_size_f;
            const int remainder = total_scores % block_size;
            float* scores = reinterpret_cast<float*>(confidence_data);
            int idx_scores = 0;
            for (; idx_scores + (block_size - 1) < total_scores; idx_scores += block_size) {
                stage_0_work_item<dtype>(scores, intermediate_scores.get(), intermediate_box_num.get(),
                                         idx_scores, num_of_images, num_classes, num_of_priors, args.confidence_threshold);
            }
            if (remainder > 0) {
                stage_0_work_item_with_remainder<dtype>(scores, intermediate_scores.get(), intermediate_box_num.get(),
                                                        idx_scores, num_classes, num_of_priors,
                                                        args.confidence_threshold, remainder);
            }
            // Debugging
            Scores* inter_scores = reinterpret_cast<Scores*>(intermediate_scores.get());
            int* inter_box_num = reinterpret_cast<int*>(intermediate_box_num.get());
            for (int idx_image = 0; idx_image < num_of_images; idx_image++) {
                for (int idx_class = 0; idx_class < num_classes; idx_class++) {
                    int num_scores_offset = idx_image * num_classes + idx_class;
                    int acc_num = inter_box_num[num_scores_offset];
                    int scores_offset = (idx_image * num_classes * num_of_priors) + idx_class * num_of_priors;
                    for (int idx_inter_scores = 0; idx_inter_scores < acc_num; idx_inter_scores++) {
                        Scores score_info = inter_scores[scores_offset + idx_inter_scores];
                        std::cout << "[batchId:" << score_info.batchId
                                  << ", classId:" << score_info.classId
                                  << ", boxId:" << score_info.boxId
                                  << ", score:" << score_info.score <<"]" << std::endl;
                    }
                }
            }
        }

        for (int image = 0; image < num_of_images; ++image) {
            std::vector<std::vector<bounding_box>>& bboxes_per_image = bboxes[image];
            std::vector<std::vector<std::pair<float, int>>>& scores_per_image = confidences[image];
            bboxes_per_image.resize(num_loc_classes);
            scores_per_image.resize(num_classes);
            stage_0_decode_bbox<dtype>(instance, bboxes_per_image, image, num_loc_classes,
                                       args.prior_info_size, args.prior_coordinates_offset, args.code_type,
                                       args.variance_encoded_in_target, args.prior_is_normalized,
                                       args.input_width, args.input_height, args.clip_before_nms, num_of_priors);
            stage_0_extract_confidence<dtype>(instance, scores_per_image, image, num_of_priors,
                                              num_classes, args.confidence_threshold);
        }
    }

    static void stage_1(const detection_output_inst& instance,
                        std::vector<std::vector<std::vector<std::pair<float, int>>>>& confidences,
                        const int num_of_images) {
        const auto& args = instance.argument;

        const int num_classes = static_cast<int>(args.num_classes);

        for (int image = 0; image < num_of_images; ++image) {
            std::vector<std::vector<std::pair<float, int>>>& scores_per_image = confidences[image];
            for (int cls = 0; cls < num_classes; ++cls) {
                if (cls == args.background_label_id) {
                    scores_per_image[cls].clear();
                    continue;
                }
                std::vector<std::pair<float, int>>& scores = scores_per_image[cls];
                sort(scores, args.top_k);
            }
        }
    }

    static void stage_2(const detection_output_inst& instance,
                        std::vector<std::vector<std::vector<bounding_box>>>& bboxes,
                        std::vector<std::vector<std::vector<std::pair<float, int>>>>& confidences,
                        std::vector<std::map<int, std::vector<int>>>& all_indices,
                        const int num_of_images) {
        const auto& args = instance.argument;

        const int num_classes = static_cast<int>(args.num_classes);

        for (int image = 0; image < num_of_images; ++image) {
            std::vector<std::vector<bounding_box>>& bboxes_per_image = bboxes[image];
            std::vector<std::vector<std::pair<float, int>>>& scores_per_image = confidences[image];
            std::map<int, std::vector<int>>& indices_per_image = all_indices[image];
            for (int cls = 0; cls < num_classes; ++cls) {
                if (cls == args.background_label_id) {
                    continue;
                }
                std::vector<std::pair<float, int>>& scores = scores_per_image[cls];
                const int label = args.share_location ? 0 : cls;
                calc_iou_keep_and_throw(bboxes_per_image[label], scores, indices_per_image[cls], args.nms_threshold);
            }
        }
    }

    template <typename dtype>
    void stage_final(const detection_output_inst& instance,
                     std::vector<std::vector<std::vector<bounding_box>>>& bboxes,
                     std::vector<std::vector<std::vector<std::pair<float, int>>>>& confidences,
                     std::vector<std::map<int, std::vector<int>>>& all_indices,
                     const int num_of_images) {
        const auto& args = instance.argument;

        auto& output = instance.output_memory();
        mem_lock<dtype> lock_output{output};
        auto output_data = lock_output.begin();

        const int num_classes = static_cast<int>(args.num_classes);
        std::vector<std::vector<std::vector<std::pair<float, int>>>> final_detections;

        for (int image = 0; image < num_of_images; ++image) {
            std::map<int, std::vector<int>>& indices_per_image = all_indices[image];
            int num_det = 0;
            for (int cls = 0; cls < num_classes; ++cls) {
                if (cls == args.background_label_id) {
                    continue;
                }
                num_det += indices_per_image[cls].size();
            }
            std::vector<std::vector<std::pair<float, int>>> new_indices(num_classes);
            keep_top_k_and_throw(confidences[image], new_indices, all_indices[image], num_det, args.keep_top_k);
            final_detections.emplace_back(new_indices);
        }

        int count = 0;
        for (int image = 0; image < num_of_images; ++image) {
            const std::vector<std::vector<bounding_box>>& bboxes_per_image = bboxes[image];
            auto& final_detections_per_image = final_detections[image];
            for (int label = 0; label < static_cast<int>(final_detections_per_image.size()); ++label) {
                int loc_label = args.share_location ? 0 : label;
                const std::vector<bounding_box>& bboxes = bboxes_per_image[loc_label];
                const std::vector<std::pair<float, int>>& label_detections = final_detections_per_image[label];
                for (std::pair<float, int> score_prior : label_detections) {
                    output_data[count * DETECTION_OUTPUT_ROW_SIZE] = (dtype)static_cast<float>(image);
                    output_data[count * DETECTION_OUTPUT_ROW_SIZE + 1] =
                        args.decrease_label_id ? ((dtype)(static_cast<float>(label - 1.0f))) : (dtype)static_cast<float>(label);
                    output_data[count * DETECTION_OUTPUT_ROW_SIZE + 2] = (dtype)score_prior.first;
                    const bounding_box& bbox = bboxes[score_prior.second];
                    float xmin = bbox.xmin;
                    float ymin = bbox.ymin;
                    float xmax = bbox.xmax;
                    float ymax = bbox.ymax;

                    if (args.clip_after_nms) {
                        xmin = std::max(0.0f, std::min(1.0f, xmin));
                        ymin = std::max(0.0f, std::min(1.0f, ymin));
                        xmax = std::max(0.0f, std::min(1.0f, xmax));
                        ymax = std::max(0.0f, std::min(1.0f, ymax));
                    }

                    output_data[count * DETECTION_OUTPUT_ROW_SIZE + 3] = (dtype)xmin;
                    output_data[count * DETECTION_OUTPUT_ROW_SIZE + 4] = (dtype)ymin;
                    output_data[count * DETECTION_OUTPUT_ROW_SIZE + 5] = (dtype)xmax;
                    output_data[count * DETECTION_OUTPUT_ROW_SIZE + 6] = (dtype)ymax;
                    ++count;
                }
            }
        }

        // In case number of detections is smaller than keep_top_k fill the rest of the buffer with invalid image id
        // (-1).
        while (count < num_of_images * args.keep_top_k) {
            output_data[count * DETECTION_OUTPUT_ROW_SIZE] = (dtype)-1.f;
            output_data[count * DETECTION_OUTPUT_ROW_SIZE + 1] = (dtype)0.f;
            output_data[count * DETECTION_OUTPUT_ROW_SIZE + 2] = (dtype)0.f;
            output_data[count * DETECTION_OUTPUT_ROW_SIZE + 3] = (dtype)0.f;
            output_data[count * DETECTION_OUTPUT_ROW_SIZE + 4] = (dtype)0.f;
            output_data[count * DETECTION_OUTPUT_ROW_SIZE + 5] = (dtype)0.f;
            output_data[count * DETECTION_OUTPUT_ROW_SIZE + 6] = (dtype)0.f;
            ++count;
        }
    }

    event_impl::ptr execute_impl(const std::vector<event_impl::ptr>& events, detection_output_inst& instance) override {
        for (auto& a : events) {
            a->wait();
        }

        auto ev = instance.get_network().get_engine().create_user_event(instance.get_network().get_id(), false);

        const int num_of_images = instance.location_memory().get_layout().size.batch[0];  // batch size

        std::vector<std::vector<std::vector<bounding_box>>> intermediate_bboxes(num_of_images);
        std::vector<std::vector<std::vector<std::pair<float, int>>>> intermediate_confidences(num_of_images);
        std::vector<std::map<int, std::vector<int>>> intermediate_indices(num_of_images);

        if (instance.location_memory().get_layout().data_type == data_types::f32) {
            stage_0<data_type_to_type<data_types::f32>::type>(instance, intermediate_bboxes, intermediate_confidences);
            stage_1(instance, intermediate_confidences, num_of_images);
            stage_2(instance, intermediate_bboxes, intermediate_confidences, intermediate_indices, num_of_images);
            stage_final<data_type_to_type<data_types::f32>::type>(instance, intermediate_bboxes, intermediate_confidences, intermediate_indices, num_of_images);
        } else {
            stage_0<data_type_to_type<data_types::f16>::type>(instance, intermediate_bboxes, intermediate_confidences);
            stage_1(instance, intermediate_confidences, num_of_images);
            stage_2(instance, intermediate_bboxes, intermediate_confidences, intermediate_indices, num_of_images);
            stage_final<data_type_to_type<data_types::f16>::type>(instance, intermediate_bboxes, intermediate_confidences, intermediate_indices, num_of_images);
        }

        dynamic_cast<cldnn::user_event*>(ev.get())->set();  // set as complete
        // TODO: consider refactoring create_user_event() to return cldnn::user_event*
        return ev;
    }

    static primitive_impl* create(const detection_output_node& arg) { return new detection_output_cpu(arg); }
};

namespace detail {

attach_detection_output_gpu::attach_detection_output_gpu() {
    implementation_map<detection_output>::add(std::make_tuple(engine_types::ocl, data_types::f32, format::bfyx), detection_output_cpu::create);
    implementation_map<detection_output>::add(std::make_tuple(engine_types::ocl, data_types::f16, format::bfyx), detection_output_cpu::create);
}

}  // namespace detail

}  // namespace gpu
}  // namespace cldnn
