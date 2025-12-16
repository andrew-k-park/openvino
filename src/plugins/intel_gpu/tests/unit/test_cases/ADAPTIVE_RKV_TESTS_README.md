# Adaptive R-KV Diversity Calculation Tests

이 디렉토리는 Adaptive R-KV 다양성 계산의 GPU 구현을 검증하는 테스트 케이스들을 포함합니다.

## 테스트 파일 구조

### 1. `adaptive_rkv_diversity_test.cpp`
Reference 구현의 단위 테스트입니다.

**테스트 케이스:**
- `basic_diversity_calculation`: 기본 다양성 계산 검증
- `diversity_with_different_parameters`: 다양한 파라미터 조합 테스트
- `fill_diagonal_correctness`: 대각선 0 채우기 정확성
- `mean_threshold_filtering`: 평균 기반 임계값 필터링
- `block_sum_diversity_values`: 블록별 다양성 합산
- `edge_case_single_block`: 단일 블록 엣지 케이스
- `deterministic_output`: 결정적 출력 검증
- `diversity_reflects_similarity`: 유사도-다양성 관계 검증

**실행 방법:**
```bash
cd build
./bin/intel_gpu_unit_tests --gtest_filter="adaptive_rkv_diversity_reference_test.*"
```

### 2. `paged_attention_gpu_test.cpp` (통합 테스트)
Paged Attention과 통합된 Adaptive R-KV의 end-to-end 테스트입니다.

**테스트 스위트:** `smoke_adaptive_rkv`

**테스트 시나리오:**

#### PREFILL Stage (다양성 계산 활성화)
- 단일 시퀀스: 64, 128, 256, 512 토큰
- 다중 시퀀스: 여러 프롬프트 동시 처리
- 동적 패딩 모드

#### MIXED Stage (다양성 계산 활성화)
- 생성 + 프롬프트 혼합
- Prefix caching 시나리오
- 다중 시퀀스 혼합

#### GENERATE Stage (다양성 계산 스킵)
- 단일 토큰 생성 (인프라 테스트)

#### 추가 기능 조합
- KV 캐시 압축 (BY_TOKEN, BY_CHANNEL)
- GQA (Grouped Query Attention)
- Sliding window attention
- 다양한 헤드/차원 크기

**실행 방법:**
```bash
cd build
./bin/intel_gpu_unit_tests --gtest_filter="adaptive_rkv_test.*"
```

## 테스트 파라미터 설명

### PagedAttentionManager의 Adaptive RKV 파라미터

```cpp
std::vector<int> adaptive_rkv_start_size;           // [1] - 다양성 계산에서 제외할 시작 영역 크기
std::vector<int> adaptive_rkv_evictable_sizes;      // [batch_size] - 시퀀스별 제거 가능 영역 크기
std::vector<int> adaptive_rkv_diversity_block_set_indices;    // [variable] - 유지할 블록 인덱스
std::vector<int> adaptive_rkv_diversity_block_set_begins;     // [batch_size + 1] - 인덱스 시작/끝
```

### 테스트 시 기본 설정
- `start_size`: `block_size * 2` (첫 2개 블록 스킵)
- `evictable_sizes`: `(total_tokens - start_size)` 블록 정렬
- `diversity_block_set_indices`: 빈 벡터 (테스트용)

## 검증 항목

### Reference 구현 테스트 (`adaptive_rkv_diversity_test.cpp`)
1. ✅ L2 정규화 정확성
2. ✅ 코사인 유사도 행렬 계산
3. ✅ 슬라이싱 및 대각선 처리
4. ✅ 평균 기반 필터링
5. ✅ 헤드 집계 (mean across heads)
6. ✅ 블록별 다양성 합산
7. ✅ 출력 형태 검증 ([num_blocks, eviction_size])
8. ✅ 다양성 값 범위 검증

### 통합 테스트 (`paged_attention_gpu_test.cpp`)
1. ✅ Stage별 다양성 계산 활성화/비활성화
2. ✅ 다양한 시퀀스 길이 처리
3. ✅ 다중 시퀀스 배치 처리
4. ✅ KV 캐시 압축과의 호환성
5. ✅ GQA 모드 지원
6. ✅ Sliding window와의 통합
7. ✅ 동적 패딩 모드

## 예상 결과

### 성공 조건
- 모든 테스트 케이스 통과
- Reference 구현과 GPU 구현 간 수치적 일치 (허용 오차 내)
- Stage별 올바른 동작 (GENERATE에서 스킵, PREFILL/MIXED에서 실행)
- 메모리 릭 없음
- 성능 기준 충족

### 실패 가능성
- GENERATE stage에서 다양성 계산이 실행되는 경우
- 출력 형태 불일치
- 수치적 불안정성 (특히 작은 eviction_size)
- KV 캐시 압축 모드에서 정확도 저하

## 디버깅 팁

### 테스트 실패 시
1. **Stage 확인**: 로그에서 PREFILL/MIXED/GENERATE stage 확인
2. **입력 검증**: `adaptive_rkv_start_size`, `evictable_sizes` 값 확인
3. **출력 형태**: diversity 출력이 `[num_blocks, eviction_size]` 형태인지 확인
4. **허용 오차**: FP16/FP32 차이로 인한 작은 오차는 정상

### 상세 로그 활성화
```bash
export CLDNN_VERBOSE=1
./bin/intel_gpu_unit_tests --gtest_filter="adaptive_rkv_test.diversity_calculation"
```

### 개별 테스트 실행
```bash
# 특정 파라미터 조합만 테스트
./bin/intel_gpu_unit_tests --gtest_filter="adaptive_rkv_test.diversity_calculation/5"

# Reference 구현만 테스트
./bin/intel_gpu_unit_tests --gtest_filter="adaptive_rkv_diversity_reference_test.basic_diversity_calculation"
```

## 성능 벤치마크

테스트 실행 시 다음 메트릭을 모니터링하세요:
- 다양성 계산 커널 실행 시간
- 메모리 사용량 (중간 버퍼)
- PREFILL stage 총 실행 시간 증가율
- 배치 크기별 확장성

## Known Issues

1. **GENERATE Stage**: 현재 다양성 계산을 스킵하지만, 향후 주기적 재계산 옵션 추가 예정
2. **큰 Eviction Size**: 매우 큰 eviction_size (>1024)에서 메모리 부족 가능
3. **압축 모드**: BY_CHANNEL 모드에서 약간의 정확도 저하 가능

## 참고 자료

- [Adaptive R-KV 논문](https://arxiv.org/pdf/2505.24133v3)
- [Reference 구현](../../../../src/core/reference/include/openvino/reference/adaptive_rkv_diversity.hpp)
- [GPU 커널 구현](../../src/graph/impls/ocl_v2/adaptive_rkv_diversity.cl)
- [통합 가이드](../../src/graph/impls/ocl_v2/ADAPTIVE_RKV_IMPLEMENTATION_GUIDE.md)
