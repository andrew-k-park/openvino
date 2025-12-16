# PagedAttention과 SDPA 커널의 관계

## 핵심 개념: PagedAttention = SDPA + Paged KV Cache Management

PagedAttention은 **SDPA (Scaled Dot Product Attention)의 특수한 형태**로, KV 캐시를 페이지(블록) 단위로 관리합니다.

## 커널 파일 구조 및 관계

```
OpenCL 커널 파일들:
├── sdpa_opt.cl                    # 일반 SDPA 최적화 커널
│   ├── KERNEL(sdpa_opt)           # 메인 SDPA 계산
│   └── KERNEL(sdpa_opt_finalization_stage)
│
├── paged_attention_opt.cl         # PagedAttention 전용 커널 
│   ├── KERNEL(pa_sdpa_opt)        # Paged KV Cache용 SDPA
│   ├── KERNEL(pa_sdpa_finalization_stage)
│   └── KERNEL(pa_sdpa_scores_calculation)
│
└── sdpa_micro.cl                  # Micro SDPA (oneDNN 기반)
    └── KERNEL(micro_sdpa)         # 작은 블록 최적화 SDPA

C++ 구현 파일들:
└── sdpa/
    ├── paged_attention_opt.cpp    # PagedAttention 구현 선택자
    │   ├── 상황에 따라 다음 중 선택:
    │   ├── → sdpa_opt.cl (IS_PAGED_ATTENTION=1)
    │   ├── → sdpa_micro.cl (IS_PAGED_ATTENTION=1)
    │   └── → paged_attention_opt.cl (전용 커널)
    │
    ├── sdpa_gen_opt.cpp           # sdpa_opt.cl용 JIT 생성
    └── sdpa_gen_micro.cpp         # sdpa_micro.cl용 JIT 생성
```

## 왜 하나의 커널 파일이 아닌가?

### 1. **sdpa_opt.cl** - 범용 SDPA 커널
**용도**: 일반 Attention과 PagedAttention 모두 지원

```c
KERNEL(sdpa_opt)(
#if IS_PAGED_ATTENTION  // JIT 컴파일 시 결정
    // PagedAttention용 파라미터
    const __global int* past_lens,
    const __global int* block_indices,
    const __global int* block_indices_begins,
#else
    // 일반 SDPA 파라미터
    const __global KEY_TYPE* key,
    const __global VALUE_TYPE* value,
#endif
    ...
)
```

**특징**:
- 하나의 소스코드로 두 가지 모드 지원
- `IS_PAGED_ATTENTION` JIT 상수로 컴파일 타임에 분기
- KV 캐시 접근 방식이 다름:
  - 일반: `key[batch, head, seq, dim]` 직접 접근
  - Paged: `key_cache[block_indices[...], head, dim, block_offset]` 간접 접근

### 2. **paged_attention_opt.cl** - PagedAttention 전용 커널
**용도**: PagedAttention만을 위한 최적화된 구현

```c
KERNEL(pa_sdpa_opt)(
    const __global INPUT1_TYPE* key_cache,      // 항상 paged 형태
    const __global INPUT2_TYPE* value_cache,
    const __global INPUT4_TYPE* block_indices,
    ...
)
```

**특징**:
- PagedAttention에 특화된 최적화
- JIT 분기 없이 paged 접근만 구현
- 특수한 메모리 접근 패턴 활용
- GQA (Grouped Query Attention) 전용 최적화

### 3. **sdpa_micro.cl** - Micro 블록 SDPA
**용도**: 작은 블록 크기에 최적화 (oneDNN 기반)

```c
KERNEL(micro_sdpa)(
#if IS_PAGED_ATTENTION
    const __global INPUT3_TYPE* subsequence_begins,
    #if IS_PREFILL == 0
        const __global INPUT3_TYPE* past_lens,
        const __global INPUT3_TYPE* block_indices,
    #endif
#endif
    ...
)
```

**특징**:
- 작은 타일 크기 (8x8, 16x16)에 최적화
- oneDNN의 micro-kernel 전략 사용
- PREFILL/GENERATE 모드 구분
- PagedAttention 지원 추가됨

## 커널 선택 로직 (paged_attention_opt.cpp)

```cpp
// paged_attention_opt.cpp에서 커널 선택
if (can_use_micro_sdpa(params, stage)) {
    // sdpa_micro.cl 사용
    // - 작은 블록 크기
    // - oneDNN 사용 가능
    return SDPAMicroGenerator::create(params);
    
} else if (use_dedicated_pa_kernel(params, stage)) {
    // paged_attention_opt.cl 사용
    // - GQA 최적화
    // - 특수 메모리 패턴
    return PagedAttentionOptKernel::create(params);
    
} else {
    // sdpa_opt.cl 사용 (IS_PAGED_ATTENTION=1)
    // - 범용적인 경우
    return SDPAOptGenerator::create(params, /*is_paged=*/true);
}
```

## PagedAttention에서 SDPA 커널을 재사용하는 이유

### 1. **코드 중복 방지**
- Attention 알고리즘 자체는 동일: `softmax(Q×K^T) × V`
- 차이점은 KV 캐시 접근 방식뿐
- JIT 컴파일로 조건부 코드 생성

### 2. **최적화 공유**
- SDPA에서 개발된 최적화를 PagedAttention에 즉시 적용
- 예: Subgroup shuffle, vectorization, tiling 전략

### 3. **유연성**
- 상황에 따라 최적의 커널 선택
- Micro 커널 ↔ 일반 SDPA ↔ 전용 PA 커널

## Adaptive R-KV와의 통합

모든 SDPA 계열 커널이 Adaptive R-KV를 지원합니다:

```c
// sdpa_opt.cl
#if IS_PAGED_ATTENTION && HAS_ADAPTIVE_RKV
    , const __global int* adaptive_rkv_start_size
    , const __global int* adaptive_rkv_evictable_sizes
    , const __global int* adaptive_rkv_diversity_block_set_indices
    , const __global int* adaptive_rkv_diversity_block_set_indices_begins
#endif

// paged_attention_opt.cl
#if HAS_ADAPTIVE_RKV
    , const __global int* adaptive_rkv_start_size
    , const __global int* adaptive_rkv_evictable_sizes
    , const __global int* adaptive_rkv_diversity_block_set_indices
    , const __global int* adaptive_rkv_diversity_block_set_indices_begins
#endif

// sdpa_micro.cl  
#if IS_PAGED_ATTENTION && HAS_ADAPTIVE_RKV
    , const __global int* adaptive_rkv_start_size
    , const __global int* adaptive_rkv_evictable_sizes
    , const __global int* adaptive_rkv_diversity_block_set_indices
    , const __global int* adaptive_rkv_diversity_block_set_indices_begins
#endif
```

**통합 방식**:
1. SDPA 커널이 Attention 계산 완료
2. GPU 플러그인이 stage를 확인 (PREFILL/MIXED만 계산)
3. `paged_attention_adaptive_rkv.cpp`에서 diversity 커널 실행:
   - CodeBuilder로 JIT 상수 주입
   - kernels_cache.compile()로 커널 컴파일
   - stream.enqueue_kernel()로 GPU 실행
4. 7개 staged 커널 또는 1개 fused 커널 실행
5. 출력: diversity_output [batch, num_blocks, eviction_size]
6. GenAI 레이어에서 마스킹 및 최종 eviction 결정

## 실행 예시

### GENERATE Stage (single token)
```
PagedAttention 요청
  ↓
paged_attention_opt.cpp 커널 선택
  ↓
[선택 1] sdpa_opt.cl (IS_PAGED_ATTENTION=1)
  ├─ Q: [1, heads, 1, dim]          # 1개 토큰
  ├─ K: block_indices → key_cache   # Paged 접근
  ├─ V: block_indices → value_cache
  └─ 출력: [1, heads, 1, dim]
  
Adaptive R-KV:
  └─ ❌ 스킵됨 (should_compute_diversity() = false)
     이유: 1개 토큰 추가로 전체 재계산은 비효율적
```

### PREFILL Stage (prompt processing)
```
PagedAttention 요청
  ↓
paged_attention_opt.cpp 커널 선택
  ↓
[선택 2] sdpa_micro.cl (IS_PAGED_ATTENTION=1, IS_PREFILL=1)
  ├─ Q: [1, heads, seq_len, dim]    # 전체 프롬프트
  ├─ K: subsequence_begins → key    # 연속 접근
  ├─ V: subsequence_begins → value
  └─ 출력: [1, heads, seq_len, dim]

Adaptive R-KV:
  ✅ should_compute_diversity() = true (PREFILL stage)
  ↓
GPU Plugin (paged_attention_adaptive_rkv.cpp):
  ├─ JIT 상수 준비 (NUM_KV_HEADS, HEAD_SIZE, BLOCK_SIZE, ...)
  ├─ CodeBuilder로 커널 코드 생성
  ├─ kernels_cache.compile()
  └─ 7개 staged 커널 실행:
      ├─ adaptive_rkv_normalize_keys
      ├─ adaptive_rkv_compute_similarity
      ├─ adaptive_rkv_slice_and_fill_diagonal
      ├─ adaptive_rkv_threshold_by_mean
      ├─ adaptive_rkv_aggregate_heads
      ├─ adaptive_rkv_block_sum_diversity
      └─ adaptive_rkv_apply_mask_and_reduce
          └─ diversity_output[batch, num_blocks, eviction_size]
```

### MIXED Stage (batch with prompt + generation)
```
PagedAttention 요청
  ↓
paged_attention_opt.cpp 커널 선택
  ↓
[선택 3] paged_attention_opt.cl (전용 커널)
  ├─ 시퀀스별 다른 처리
  ├─ GQA 최적화 적용
  └─ 특수 메모리 접근 패턴

Adaptive R-KV:
  ✅ should_compute_diversity() = true (MIXED stage)
  ↓
GPU Plugin:
  ├─ 현재: 전체 배치에 대해 diversity 계산
  └─ 향후: 프롬프트 시퀀스만 선택적으로 계산 (최적화)
```

## 요약

| 측면 | sdpa_opt.cl | paged_attention_opt.cl | sdpa_micro.cl |
|------|-------------|------------------------|---------------|
| **목적** | 범용 SDPA | PA 전용 최적화 | 작은 블록 최적화 |
| **PagedAttention** | JIT로 지원 | 네이티브 지원 | JIT로 지원 |
| **일반 SDPA** | 네이티브 지원 | 미지원 | JIT로 지원 |
| **최적화 대상** | 범용 | GQA, 특수 메모리 | Micro 블록 |
| **Adaptive RKV** | ✅ 지원 | ✅ 지원 | ✅ 지원 |
| **선택 기준** | 기본 | GQA + 긴 시퀀스 | oneDNN + 작은 블록 |
| **구현 상태** | ✅ 완료 | ✅ 완료 | ✅ 완료 |

**핵심**: PagedAttention은 SDPA의 변형이므로, 기존 SDPA 커널을 재사용하되 KV 캐시 접근 방식만 변경합니다. 이를 통해 코드 중복을 줄이고 최적화를 공유합니다.

**Adaptive R-KV 실행 흐름**:
1. **PREFILL/MIXED Stage**: diversity 계산 활성화 (should_compute_diversity() = true)
2. **GENERATE Stage**: diversity 계산 스킵 (주기적 재계산은 향후 최적화)
3. **GPU 실행**: paged_attention_adaptive_rkv.cpp에서 7개 staged 커널 또는 1개 fused 커널 실행
4. **API**: CodeBuilder + kernels_cache + stream.enqueue_kernel
5. **테스트**: Shape inference 8/8 통과, Unit/Integration 테스트 대기
