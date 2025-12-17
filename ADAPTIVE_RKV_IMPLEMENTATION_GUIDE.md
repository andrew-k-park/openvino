# Adaptive R-KV 구현 가이드

## 목차
1. [구현 순서](#구현-순서)
2. [각 단계별 상세 설명](#각-단계별-상세-설명)
3. [데이터 흐름](#데이터-흐름)
4. [알고리즘 상세](#알고리즘-상세)
5. [통합 방법](#통합-방법)

---

## 구현 순서

### Phase 1: 독립 Diversity 계산 커널 구현
1. **adaptive_rkv_diversity.cl** 생성 (독립 OpenCL 커널)
   - L2 정규화, 코사인 유사도, 필터링, 블록 집계 구현

### Phase 2: Paged Attention 커널 확장
2. **paged_attention_opt.cl** 수정
   - HAS_ADAPTIVE_RKV 조건부 파라미터 추가
   - Diversity 계산 호출 지점 마킹

3. **paged_attention_opt.cpp** 수정
   - JIT 상수 추가: `HAS_ADAPTIVE_RKV`, `PAGED_ATTENTION_BLOCK_SIZE`
   - 인자 디스크립터 확장: 4개 입력 + 1개 출력

### Phase 3: 일반 SDPA 커널 확장
4. **sdpa_gen_opt.cpp** 수정
   - Paged attention 모드에서 JIT 상수 추가

5. **sdpa_opt.cl** 수정
   - IS_PAGED_ATTENTION && HAS_ADAPTIVE_RKV 조건부 파라미터 추가
   - Diversity 계산 통합 지점 표시

### Phase 4: 문서화
6. **ADAPTIVE_RKV_IMPLEMENTATION_SUMMARY.md** 생성
   - 전체 구현 개요 및 파일별 변경사항

7. **SDPA_OPT_ADAPTIVE_RKV_CHANGES.md** 생성
   - SDPA 커널 변경사항 상세 설명

---

## 각 단계별 상세 설명

### 1. adaptive_rkv_diversity.cl - 독립 Diversity 계산 커널

#### 목적
KV-cache eviction을 위한 토큰 다양성(diversity) 계산을 수행하는 독립 실행 가능한 OpenCL 커널

#### 핵심 알고리즘 (6단계)

**Step 1: L2 정규화**
```c
// eviction 영역의 각 key 벡터를 L2 정규화
float l2_norm = 0.0f;
for (int h = 0; h < head_size; h++) {
    float val = key_cache[...];
    l2_norm += val * val;
}
l2_norm = sqrt(l2_norm);

// 정규화된 벡터 저장
normalized_key = original_key / l2_norm;
```

**Step 2: 코사인 유사도 행렬 계산**
```c
// SLM(Shared Local Memory)을 사용한 행렬 계산
__local float cosine_sim_matrix[MAX_EVICTABLE_SIZE * MAX_EVICTABLE_SIZE];

// 각 워크아이템이 행렬의 일부 계산
for (int i = 0; i < evictable_size; i++) {
    for (int j = 0; j < evictable_size; j++) {
        float dot_product = 0.0f;
        for (int h = 0; h < head_size; h++) {
            dot_product += normalized_key[i][h] * normalized_key[j][h];
        }
        cosine_sim_matrix[i * evictable_size + j] = dot_product;
    }
}
```

**Step 3: 대각선 제로화**
```c
// 자기 자신과의 유사도는 0으로 설정
for (int i = 0; i < evictable_size; i++) {
    cosine_sim_matrix[i * evictable_size + i] = 0.0f;
}
```

**Step 4: 평균 기반 필터링**
```c
// 각 행의 평균 계산 후, 평균보다 작은 값은 0으로
for (int i = 0; i < evictable_size; i++) {
    float row_sum = 0.0f;
    for (int j = 0; j < evictable_size; j++) {
        row_sum += cosine_sim_matrix[i * evictable_size + j];
    }
    float row_mean = row_sum / evictable_size;
    
    // 필터링
    for (int j = 0; j < evictable_size; j++) {
        if (cosine_sim_matrix[i * evictable_size + j] < row_mean) {
            cosine_sim_matrix[i * evictable_size + j] = 0.0f;
        }
    }
}
```

**Step 5: 헤드 간 집계**
```c
// 모든 attention head에서 평균 계산
float aggregated_similarity = 0.0f;
for (int head = 0; head < num_heads; head++) {
    aggregated_similarity += cosine_sim_matrix[head][i][j];
}
aggregated_similarity /= num_heads;
```

**Step 6: 블록별 Diversity 집계**
```c
// 각 블록(16개 토큰)의 diversity = -sum(cosine_similarity)
for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
    float block_diversity = 0.0f;
    for (int token = block_idx * 16; token < (block_idx + 1) * 16; token++) {
        for (int j = 0; j < evictable_size; j++) {
            block_diversity -= cosine_sim_matrix[token * evictable_size + j];
        }
    }
    diversity_output[block_idx] = block_diversity;
}
```

#### 메모리 최적화
- **SLM 사용**: 코사인 유사도 행렬을 Shared Local Memory에 저장하여 글로벌 메모리 접근 최소화
- **컴파일 타임 상수**: `MAX_EVICTABLE_SIZE`를 JIT 상수로 제공하여 SLM 크기 결정
- **Atomic 연산**: 멀티 헤드 집계 시 atomic_add 사용 (성능 개선 여지 있음)

#### 제한사항
- MAX_EVICTABLE_SIZE가 SLM 용량 제한 (일반적으로 64KB)
- 큰 eviction 영역은 타일링 필요
- Atomic 연산 오버헤드 (트리 리덕션으로 개선 가능)

---

### 2. paged_attention_opt.cl - Paged Attention 커널 확장

#### 기존 기능
- Paged KV-cache를 사용한 attention 계산
- Block 단위(16 토큰) 메모리 관리
- Multi-stage SDPA (SDPA_STAGE_0, SDPA_STAGE_1)

#### 추가된 기능

**파라미터 확장**
```c
#if HAS_ADAPTIVE_RKV
    , const __global int* adaptive_rkv_start_size          // [1] eviction 시작 위치
    , const __global int* adaptive_rkv_evictable_sizes     // [batch_size]
    , const __global int* adaptive_rkv_evictable_indices   // [total_blocks]
    , const __global int* adaptive_rkv_evictable_begins    // [batch_size + 1]
    , __global OUTPUT_TYPE* diversity_output               // [total_diversity_elements]
#endif
```

**통합 지점**
```c
// SDPA_STAGE_0 끝부분, softmax 계산 후
#if HAS_ADAPTIVE_RKV
    if (partition_idx == 0 && sgid == 0 && sglid == 0) {
        // diversity_output에 시퀀스별 eviction 정보 마킹
        const uint seq_idx = gws_seq_indexes_correspondence[target_seq_dim];
        const uint start_size = adaptive_rkv_start_size[0];
        const uint evictable_size = adaptive_rkv_evictable_sizes[seq_idx];
        
        if (evictable_size > 0) {
            // 실제 diversity 계산은 adaptive_rkv_diversity.cl에서 수행
            // 여기서는 계산이 필요함을 표시만
            diversity_output[seq_idx] = OUTPUT_VAL_ZERO;
        }
    }
#endif
```

#### 데이터 레이아웃
```
adaptive_rkv_evictable_indices 구조:
[batch_0의 블록들... | batch_1의 블록들... | ...]

adaptive_rkv_evictable_begins 구조:
[0, batch_0_block_count, batch_0+1_count, ...]
   ↑                      ↑
   batch_0 시작           batch_1 시작
```

---

### 3. paged_attention_opt.cpp - 인자 디스크립터 관리

#### JIT 상수 추가
```cpp
if (desc->has_adaptive_rkv) {
    jit.make("HAS_ADAPTIVE_RKV", 1);
    jit.make("PAGED_ATTENTION_BLOCK_SIZE", paged_attention::block_size);  // 16
}
```

#### 인자 디스크립터 확장
```cpp
// PagedAttentionInputIdx enum에 추가된 인덱스
// ADAPTIVE_RKV_START_SIZE = 21
// ADAPTIVE_RKV_EVICTABLE_SIZES = 22
// ADAPTIVE_RKV_DIVERSITY_BLOCK_SET_INDICES = 23
// ADAPTIVE_RKV_DIVERSITY_BLOCK_SET_BEGINS = 24

if (has_adaptive_rkv) {
    args.push_back({ArgumentDescriptor::Types::INPUT, 
                    PagedAttentionInputIdx::ADAPTIVE_RKV_START_SIZE});
    args.push_back({ArgumentDescriptor::Types::INPUT, 
                    PagedAttentionInputIdx::ADAPTIVE_RKV_EVICTABLE_SIZES});
    args.push_back({ArgumentDescriptor::Types::INPUT, 
                    PagedAttentionInputIdx::ADAPTIVE_RKV_DIVERSITY_BLOCK_SET_INDICES});
    args.push_back({ArgumentDescriptor::Types::INPUT, 
                    PagedAttentionInputIdx::ADAPTIVE_RKV_DIVERSITY_BLOCK_SET_BEGINS});
    
    // Output port 2: diversity results
    if (params.output_layouts.size() > 2) {
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 2});
    }
}
```

#### 컴파일 플로우
1. Primitive 생성 시 `has_adaptive_rkv` 플래그 확인
2. JIT 상수 설정
3. OpenCL 커널 동적 컴파일
4. 런타임에 추가 버퍼 바인딩

---

### 4. sdpa_gen_opt.cpp - 일반 SDPA JIT 상수

#### Paged Attention 모드 감지
```cpp
const bool is_paged_attention = params.is_type<paged_attention>() ? true : false;

if (is_paged_attention) {
    auto desc = params.typed_desc<paged_attention>();
    
    // 기존 paged attention 설정...
    
    if (desc->has_adaptive_rkv) {
        jit.make("HAS_ADAPTIVE_RKV", 1);
        jit.make("PAGED_ATTENTION_BLOCK_SIZE", paged_attention::block_size);
    }
}
```

#### 역할
- `sdpa_opt.cl` 커널을 paged attention 모드로 컴파일할 때 필요한 상수 제공
- `IS_PAGED_ATTENTION`과 `HAS_ADAPTIVE_RKV` 조합으로 조건부 컴파일

---

### 5. sdpa_opt.cl - 일반 SDPA 커널 확장

#### 조건부 컴파일
```c
#if IS_PAGED_ATTENTION && HAS_ADAPTIVE_RKV
    // Adaptive R-KV 파라미터 추가
    , const __global int* adaptive_rkv_evictable_start_size
    , const __global int* adaptive_rkv_evictable_sizes
    , const __global int* adaptive_rkv_evictable_indices
    , const __global int* adaptive_rkv_evictable_begins
    , __global OUTPUT_TYPE* adaptive_rkv_diversity_output
#endif
```

#### 통합 전략
- **Placeholder 방식**: sdpa_opt.cl은 파라미터만 받고 실제 계산은 하지 않음
- **별도 Dispatch**: Diversity 계산은 `adaptive_rkv_diversity.cl` 커널로 분리
- **이유**: 
  - sdpa_opt는 이미 복잡하여 추가 로직 삽입 시 레지스터 압박
  - Diversity 계산은 독립적으로 최적화 가능
  - 모듈화된 설계로 유지보수 용이

---

## 데이터 흐름

### 전체 파이프라인
```
┌─────────────────────────────────────────────────────────┐
│ 1. openvino.genai Level                                 │
│    - Eviction 영역 결정                                 │
│    - Input 버퍼 준비                                    │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│ 2. paged_attention Primitive                            │
│    - has_adaptive_rkv = true 설정                       │
│    - Input enum 확장 (4개 추가 입력)                    │
│    - Output layout 계산 (diversity 버퍼 크기)           │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│ 3. Kernel Compilation (paged_attention_opt.cpp)         │
│    - JIT 상수 설정: HAS_ADAPTIVE_RKV=1                  │
│    - 인자 디스크립터 생성                               │
│    - OpenCL 커널 컴파일                                 │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│ 4. Kernel Execution (paged_attention_opt.cl)            │
│    - Attention score 계산                               │
│    - Softmax 적용                                       │
│    - Diversity 계산 필요 표시                           │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│ 5. Diversity Kernel Dispatch (adaptive_rkv_diversity.cl)│
│    - L2 정규화                                          │
│    - 코사인 유사도 행렬 계산                            │
│    - 필터링 및 집계                                     │
│    - Diversity 값 출력                                  │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│ 6. openvino.genai Level - Eviction                      │
│    - Diversity 값 기반 블록 선택                        │
│    - KV-cache 업데이트                                  │
└─────────────────────────────────────────────────────────┘
```

### 메모리 맵
```
GPU 메모리 레이아웃:

Input Buffers:
├── key_cache [num_blocks, num_heads, head_size, 16]
├── value_cache [num_blocks, num_heads, head_size, 16]
├── adaptive_rkv_evictable_start_size [batch_size * 2]
│   └── [start_0, size_0, start_1, size_1, ...]
├── adaptive_rkv_evictable_sizes [batch_size]
├── adaptive_rkv_evictable_indices [total_evictable_blocks]
│   └── [block_idx_0, block_idx_1, ...]
└── adaptive_rkv_evictable_begins [batch_size + 1]
    └── [0, offset_1, offset_2, ..., total]

Output Buffers:
├── attention_output [batch, heads, seq_len, head_size]
├── softmax_scores (optional) [batch, heads, seq_len, kv_len]
└── diversity_output [total_diversity_elements]
    └── 각 블록의 diversity 값 (낮을수록 evict 후보)

Intermediate (SLM):
└── cosine_similarity_matrix [MAX_EVICTABLE_SIZE²]
    └── 각 워크그룹이 독립적으로 사용
```

---

## 알고리즘 상세

### Reference Implementation 매핑
원본 C++ 템플릿 → OpenCL 커널 변환

#### C++ Reference (adaptive_rkv_diversity.hpp)
```cpp
template<typename T>
void calculate_diversity(
    const T* key_cache,
    const int* evictable_indices,
    int evictable_size,
    int num_heads,
    int head_size,
    float* diversity_output
) {
    // 1. L2 normalization
    std::vector<float> normalized_keys(evictable_size * num_heads * head_size);
    
    // 2. Cosine similarity
    std::vector<float> cos_sim(evictable_size * evictable_size, 0.0f);
    
    // 3-6. Filtering and aggregation
    // ...
}
```

#### OpenCL Implementation
```c
__kernel void adaptive_rkv_diversity(
    __global const INPUT_TYPE* key_cache,
    __global const int* evictable_indices,
    const int evictable_size,
    const int num_heads,
    const int head_size,
    __global OUTPUT_TYPE* diversity_output,
    __local float* slm_cosine_sim  // SLM 사용!
) {
    // Work-group 기반 병렬화
    const int local_id = get_local_id(0);
    const int local_size = get_local_size(0);
    
    // 1. L2 normalization (각 워크아이템이 일부 담당)
    for (int token = local_id; token < evictable_size; token += local_size) {
        // ...
    }
    
    // 2-6. 나머지 단계도 병렬화
    // ...
}
```

### 성능 최적화 기법

#### 1. Work-Group 크기 선택
```c
// 추천 설정
local_work_size = 256;  // subgroup_size(16)의 배수
global_work_size = evictable_size;  // 토큰 수
```

#### 2. 메모리 접근 패턴
```c
// Coalesced access (연속 메모리 접근)
for (int h = 0; h < head_size; h++) {
    float val = key_cache[block_idx * head_size + h];  // 연속 접근
}

// Strided access 피하기 (성능 저하)
// BAD: key_cache[h * num_blocks + block_idx]
```

#### 3. SLM 뱅크 충돌 회피
```c
// 16-way bank conflict 회피
__local float slm_buffer[EVICT_SIZE * (EVICT_SIZE + 1)];  // +1 padding
```

#### 4. Subgroup 활용
```c
// Subgroup reduction (Intel GPU 최적화)
float local_sum = ...;
float total_sum = sub_group_reduce_add(local_sum);
```

---

## 통합 방법

### 현재 구현 상태

#### 1. 커널 준비 완료
- **adaptive_rkv_diversity.cl**: 독립 실행 가능한 OpenCL 커널 구현 완료
- **paged_attention_opt.cl**: HAS_ADAPTIVE_RKV 파라미터 추가 완료
- **sdpa_opt.cl**: IS_PAGED_ATTENTION && HAS_ADAPTIVE_RKV 파라미터 추가 완료

#### 2. 파라미터 전달 구현 완료
- **paged_attention_opt.cpp**: JIT 상수 및 인자 디스크립터 추가
- **sdpa_gen_opt.cpp**: JIT 상수 추가
- 모든 필요한 입력 버퍼가 커널에 전달되도록 설정

#### 3. 실행 통합 (부분 구현)
- **paged_attention_opt.cpp**: `execute()` 함수에 diversity 커널 호출 지점 추가
- **현재 상태**: Placeholder로 구현되어 있으며, 실제 커널 디스패치는 아직 미구현

### 완전한 통합을 위한 남은 작업

Diversity 커널을 실제로 실행하려면 다음 중 하나의 방법을 선택해야 합니다:

#### 방법 1: 별도 Primitive로 구현 (권장)
```cpp
// openvino.genai 레벨에서 두 개의 primitive를 순차 실행
auto network = std::make_shared<Network>();

// 1. Paged Attention 실행
auto paged_attn_output = network->add(paged_attention_primitive);

// 2. Adaptive R-KV Diversity 계산
auto diversity_primitive = std::make_shared<AdaptiveRKVDiversity>(
    paged_attn_output,
    evictable_sizes,
    evictable_indices,
    // ...
);
auto diversity_output = network->add(diversity_primitive);

// 3. Diversity 기반 Eviction
apply_eviction(diversity_output);
```

**장점**:
- 모듈화되어 테스트 용이
- 각 커널 독립적 최적화 가능
- OpenVINO 표준 primitive 패턴 따름

**단점**:
- 메모리 I/O 오버헤드 (중간 결과 저장)
- 두 번의 커널 디스패치

#### 방법 2: Paged Attention 내부 Stage로 통합
```cpp
class PagedAttentionOptImpl {
    Stage::Ptr pa_diversity_calc = make_stage<PagedAttentionDiversityGenerator>();
    
    event::ptr execute(...) {
        // ... existing stages ...
        
        if (desc->has_adaptive_rkv) {
            res_event = {execute_stage(res_event, instance, pa_diversity_calc)};
        }
        
        return res_event[0];
    }
};
```

**장점**:
- 단일 primitive 내에서 완결
- 메모리 I/O 최소화 가능
- 실행 흐름 간단

**단점**:
- Paged Attention 코드 복잡도 증가
- 독립 테스트 어려움

#### 방법 3: 퓨전 커널 (고급 최적화)
```opencl
// paged_attention_opt.cl 내에서 diversity 계산 직접 수행
#if HAS_ADAPTIVE_RKV
    // Softmax 계산 직후, 값이 SLM에 있을 때
    if (evictable_size > 0) {
        // L2 normalization
        // Cosine similarity 계산
        // Diversity 집계
    }
#endif
```

**장점**:
- 최고 성능 (SLM 재사용)
- 추가 메모리 I/O 없음

**단점**:
- 레지스터 압박 심각
- 코드 복잡도 매우 높음
- 디버깅 어려움

### openvino.genai 레벨 통합 (방법 1 상세)

#### 1. Diversity Primitive 정의
```cpp
// adaptive_rkv_diversity.hpp
namespace cldnn {
struct adaptive_rkv_diversity : public primitive_base<adaptive_rkv_diversity> {
    CLDNN_DECLARE_PRIMITIVE(adaptive_rkv_diversity)
    
    adaptive_rkv_diversity(const primitive_id& id,
                          const input_info& key_cache,
                          const input_info& past_lens,
                          const input_info& block_indices,
                          const input_info& evictable_sizes,
                          // ...
                          )
        : primitive_base(id, {key_cache, past_lens, block_indices, evictable_sizes}) {}
    
    size_t hash() const override { /* ... */ }
    bool operator==(const primitive& rhs) const override { /* ... */ }
};
}
```

#### 2. Diversity Primitive Instance
```cpp
// adaptive_rkv_diversity_inst.h
template<>
class typed_primitive_inst<adaptive_rkv_diversity> : public typed_primitive_inst_base<adaptive_rkv_diversity> {
    using parent = typed_primitive_inst_base<adaptive_rkv_diversity>;

public:
    static layout calc_output_layout(const adaptive_rkv_diversity_node& node, 
                                     kernel_impl_params const& impl_param);
    
    typed_primitive_inst(network& network, const adaptive_rkv_diversity_node& desc);
    
    memory::ptr key_cache_memory_ptr() const { return input_memory_ptr(0); }
    memory::ptr past_lens_memory_ptr() const { return input_memory_ptr(1); }
    // ...
};
```

#### 3. Diversity Kernel Implementation
```cpp
// adaptive_rkv_diversity_impl.cpp (이미 생성됨)
// adaptive_rkv_diversity_kernel.hpp (이미 생성됨)
// 위에서 생성한 파일들을 실제 프로젝트에 통합
```

#### 4. Network 구성
```cpp
// Eviction 영역 결정
std::vector<int> evictable_start_size(batch_size * 2);
std::vector<int> evictable_sizes(batch_size);
std::vector<int> evictable_indices;
std::vector<int> evictable_begins = {0};

for (int b = 0; b < batch_size; b++) {
    int start = compute_eviction_start(b);
    int size = compute_eviction_size(b);
    
    evictable_start_size[b * 2] = start;
    evictable_start_size[b * 2 + 1] = size;
    evictable_sizes[b] = size;
    
    // 해당 시퀀스의 블록 인덱스 수집
    for (int i = start; i < start + size; i++) {
        evictable_indices.push_back(block_table[b][i]);
    }
    evictable_begins.push_back(evictable_indices.size());
}
```

#### 2. Primitive 생성
```cpp
auto paged_attn = std::make_shared<paged_attention>(
    /* 기존 파라미터들... */
);
paged_attn->has_adaptive_rkv = true;
```

#### 3. Network 구성
```cpp
auto network = engine.build_network({
    // ... 기존 레이어들
    paged_attn,
    // ...
});
```

#### 4. Execution
```cpp
// Input 설정
network.set_input_data("evictable_start_size", evictable_start_size_tensor);
network.set_input_data("evictable_sizes", evictable_sizes_tensor);
network.set_input_data("evictable_indices", evictable_indices_tensor);
network.set_input_data("evictable_begins", evictable_begins_tensor);

// 실행
auto outputs = network.execute();

// Diversity 결과 획득
auto diversity = outputs.at("diversity_output").get_memory();
float* diversity_data = diversity->pointer<float>();

// Eviction 결정
std::vector<int> blocks_to_evict = select_blocks_by_diversity(
    diversity_data, 
    evictable_indices, 
    num_blocks_to_evict
);
```

### Diversity 기반 Eviction 로직
```cpp
std::vector<int> select_blocks_by_diversity(
    const float* diversity_values,
    const std::vector<int>& block_indices,
    int num_to_evict
) {
    // Diversity가 낮은 블록부터 evict
    std::vector<std::pair<float, int>> diversity_index_pairs;
    for (int i = 0; i < block_indices.size(); i++) {
        diversity_index_pairs.push_back({diversity_values[i], block_indices[i]});
    }
    
    // Diversity 오름차순 정렬 (낮은 값 = 유사도 높음 = evict 우선)
    std::sort(diversity_index_pairs.begin(), diversity_index_pairs.end());
    
    std::vector<int> evict_blocks;
    for (int i = 0; i < num_to_evict; i++) {
        evict_blocks.push_back(diversity_index_pairs[i].second);
    }
    
    return evict_blocks;
}
```

---

## 테스트 및 검증

### 1. 단위 테스트
```cpp
// adaptive_rkv_diversity.cl 커널 테스트
TEST(AdaptiveRKV, DiversityCalculation) {
    // 1. 간단한 입력 생성 (2개 블록, 1개 헤드)
    std::vector<float> key_cache = {
        // Block 0, Head 0
        1.0f, 0.0f, 0.0f, 0.0f,  // Token 0
        1.0f, 0.0f, 0.0f, 0.0f,  // Token 1 (동일)
        0.0f, 1.0f, 0.0f, 0.0f,  // Token 2 (다름)
        // ...
    };
    
    // 2. Diversity 계산
    auto diversity = calculate_diversity_gpu(key_cache, ...);
    
    // 3. 검증: Token 0,1은 유사하므로 낮은 diversity
    EXPECT_LT(diversity[0], diversity[1]);
}
```

### 2. 통합 테스트
```cpp
TEST(PagedAttention, AdaptiveRKVIntegration) {
    // 전체 파이프라인 테스트
    auto network = create_network_with_adaptive_rkv();
    auto outputs = network.execute(inputs);
    
    ASSERT_TRUE(outputs.count("diversity_output") > 0);
    // Diversity 값 범위 검증
    // Eviction 로직 검증
}
```

### 3. 성능 벤치마크
```cpp
// 다양한 eviction 크기에 대한 성능 측정
for (int evict_size : {32, 64, 128, 256, 512}) {
    auto start = std::chrono::high_resolution_clock::now();
    calculate_diversity(evict_size);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Evict size " << evict_size << ": " << duration.count() << " us\n";
}
```

---

## 디버깅 가이드

### 1. 컴파일 에러
```bash
# JIT 컴파일 로그 확인
export OV_GPU_Verbose=1
export OV_GPU_DumpKernels=1

# 생성된 OpenCL 코드 확인
ls /tmp/ocl_kernels/
cat /tmp/ocl_kernels/adaptive_rkv_diversity.cl
```

### 2. 런타임 에러
```cpp
// 버퍼 크기 검증
OPENVINO_ASSERT(evictable_indices.size() == expected_size,
                "Mismatch in evictable indices size");

// Diversity 값 범위 검증
for (float div : diversity_values) {
    OPENVINO_ASSERT(std::isfinite(div), "NaN or Inf in diversity output");
}
```

### 3. 성능 문제
```bash
# OpenCL profiling
export OV_GPU_EnableProfiling=1

# 커널 실행 시간 확인
# diversity 커널이 전체 시간의 5% 이내여야 함
```

---

## 향후 개선 방향

### 1. 성능 최적화
- [ ] Atomic 연산을 Tree Reduction으로 대체
- [ ] 큰 eviction 영역을 위한 타일링 구현
- [ ] FP16 연산 지원 (메모리 대역폭 절감)
- [ ] Subgroup shuffle을 활용한 최적화

### 2. 기능 확장
- [ ] 동적 eviction 크기 지원 (현재는 컴파일 타임 상수)
- [ ] Multi-query attention 지원
- [ ] Sparse attention 패턴과 결합

### 3. 통합 개선
- [ ] Diversity 커널 자동 dispatch (현재는 수동)
- [ ] genai 레벨 API 단순화
- [ ] 다양한 eviction 정책 지원 (diversity 외 추가 메트릭)

---

## 참고 자료

### 논문
- Adaptive R-KV: https://arxiv.org/pdf/2505.24133v3
- Paged Attention: vLLM paper

### 코드 레퍼런스
- `adaptive_rkv_diversity.hpp`: C++ 참조 구현
- Intel GPU OpenCL 최적화 가이드
- OpenVINO GPU Plugin 문서

### 관련 파일
- `paged_attention.hpp`: Primitive 정의
- `paged_attention_inst.h`: Instance 구현
- `sdpa_base.hpp`: SDPA 기본 클래스

---

## 요약

이 구현은 Adaptive R-KV diversity 계산을 Intel GPU OpenCL 백엔드에 통합합니다:

1. **독립 커널** (`adaptive_rkv_diversity.cl`): 코어 알고리즘 구현
2. **통합 레이어** (`paged_attention_opt.cl/cpp`): 파라미터 전달 및 호출
3. **JIT 컴파일** (`sdpa_gen_opt.cpp`): 조건부 컴파일 지원
4. **확장성**: 일반 SDPA 커널(`sdpa_opt.cl`)에도 통합 가능

핵심은 **모듈화된 설계**로, diversity 계산을 독립 커널로 분리하여:
- 최적화 용이성
- 유지보수성
- 테스트 가능성

을 확보했습니다.
