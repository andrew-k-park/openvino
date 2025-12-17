# Adaptive R-KV 구현 완료 상태 및 다음 단계

## ✅ 완료된 구현

### 1. 핵심 Diversity 계산 커널
파일: `adaptive_rkv_diversity.cl`
- ✅ L2 정규화 구현
- ✅ 코사인 유사도 행렬 계산
- ✅ 대각선 제로화
- ✅ 평균 기반 필터링
- ✅ 헤드 간 집계
- ✅ 블록별 Diversity 집계
- ✅ SLM 활용 메모리 최적화

### 2. Paged Attention 커널 확장
파일: `paged_attention_opt.cl`, `paged_attention_opt.cpp`
- ✅ HAS_ADAPTIVE_RKV 조건부 파라미터 추가
- ✅ JIT 상수 설정 (HAS_ADAPTIVE_RKV, PAGED_ATTENTION_BLOCK_SIZE)
- ✅ 인자 디스크립터 확장 (4개 입력 + 1개 출력)
- ✅ 실행 흐름에 diversity 계산 호출 지점 추가 (placeholder)

### 3. 일반 SDPA 커널 확장
파일: `sdpa_opt.cl`, `sdpa_gen_opt.cpp`
- ✅ IS_PAGED_ATTENTION && HAS_ADAPTIVE_RKV 조건부 파라미터 추가
- ✅ JIT 상수 설정
- ✅ 통합 지점 표시

### 4. 문서화
- ✅ ADAPTIVE_RKV_IMPLEMENTATION_GUIDE.md: 전체 구현 가이드
- ✅ ADAPTIVE_RKV_IMPLEMENTATION_SUMMARY.md: 구현 요약
- ✅ SDPA_OPT_ADAPTIVE_RKV_CHANGES.md: SDPA 커널 변경사항

### 5. 보조 구현 파일 (생성됨, 통합 대기 중)
- ✅ adaptive_rkv_diversity_kernel.hpp: 커널 파라미터 및 JIT 상수
- ✅ adaptive_rkv_diversity_impl.cpp: Primitive 구현 기반 코드

---

## 🔄 부분 구현 (Placeholder)

### Diversity 커널 실행
**위치**: `paged_attention_opt.cpp::execute_diversity_kernel()`
**현재 상태**: 
```cpp
event::ptr execute_diversity_kernel(...) {
    GPU_DEBUG_TRACE_DETAIL << "Adaptive R-KV diversity calculation triggered\n";
    // Placeholder - 실제 커널 디스패치 미구현
    return events[0];
}
```

**이유**: OpenVINO GPU 플러그인의 실제 커널 디스패치 메커니즘을 완전히 파악하지 못해 placeholder로 남겨둠

---

## 🚧 미완성 작업

### 1. 실제 Diversity 커널 디스패치 구현

#### 필요한 작업:
1. **Stage 기반 통합** (권장 방법)
   ```cpp
   class PagedAttentionDiversityGenerator : public KernelGenerator {
   public:
       PagedAttentionDiversityGenerator() 
           : KernelGenerator("adaptive_rkv_diversity", "") {}
       
       JitConstants get_jit_constants(const kernel_impl_params& params) const override;
       Arguments get_arguments_desc(const kernel_impl_params& params) const override;
       DispatchDataFunc get_dispatch_data_func() const override;
   };
   ```

2. **PagedAttentionOptImpl에 Stage 추가**
   ```cpp
   class PagedAttentionOptImpl : public SDPAImplBase {
       // ... existing stages ...
       Stage::Ptr pa_diversity_calc = make_stage<PagedAttentionDiversityGenerator>();
       
       PagedAttentionOptImpl(const kernel_impl_params& params) {
           // ... existing code ...
           if (desc->has_adaptive_rkv) {
               add_stage(pa_diversity_calc, params);
           }
       }
   };
   ```

3. **execute() 함수 수정**
   ```cpp
   event::ptr execute(...) {
       // ... existing stages ...
       
       if (desc->has_adaptive_rkv && params.output_layouts.size() > 2) {
           res_event = {execute_stage(res_event, instance, pa_diversity_calc)};
       }
       
       return res_event[0];
   }
   ```

### 2. 별도 Primitive 구현 (대안)

#### 필요한 작업:
1. **Primitive 정의** (`adaptive_rkv_diversity.hpp`)
2. **Primitive Instance** (`adaptive_rkv_diversity_inst.h`)
3. **Primitive 등록** (factory에 등록)
4. **openvino.genai 레벨 통합**
   - Paged Attention 후 Diversity Primitive 실행
   - Diversity 결과로 Eviction 수행

### 3. Memory Accessor 함수 추가

**위치**: `paged_attention_inst.h`
**필요한 추가**:
```cpp
memory::ptr adaptive_rkv_start_size_memory_ptr() const { 
    return input_memory_ptr(PagedAttentionInputIdx::ADAPTIVE_RKV_START_SIZE); 
}
memory::ptr adaptive_rkv_evictable_sizes_memory_ptr() const { 
    return input_memory_ptr(PagedAttentionInputIdx::ADAPTIVE_RKV_EVICTABLE_SIZES); 
}
memory::ptr adaptive_rkv_diversity_block_set_indices_memory_ptr() const { 
    return input_memory_ptr(PagedAttentionInputIdx::ADAPTIVE_RKV_DIVERSITY_BLOCK_SET_INDICES); 
}
memory::ptr adaptive_rkv_diversity_block_set_begins_memory_ptr() const { 
    return input_memory_ptr(PagedAttentionInputIdx::ADAPTIVE_RKV_DIVERSITY_BLOCK_SET_BEGINS); 
}
```

**현재 상태**: `paged_attention_inst.h`에 이미 선언되어 있음 (81-84줄)
**상태**: ✅ 완료

---

## 📋 구현 우선순위

### High Priority (즉시 필요)
1. ✅ ~Diversity 커널 (.cl) 작성~ → 완료
2. ✅ ~파라미터 전달 구조 구현~ → 완료
3. 🚧 **실제 커널 디스패치 구현** → 진행 필요
4. 🚧 **단위 테스트 작성** → 진행 필요

### Medium Priority (통합 테스트 후)
5. ⬜ 성능 최적화 (Atomic → Tree Reduction)
6. ⬜ FP16 지원
7. ⬜ 동적 eviction 크기 지원

### Low Priority (향후 개선)
8. ⬜ 퓨전 커널 (SDPA + Diversity)
9. ⬜ Multi-query attention 지원
10. ⬜ Sparse attention 통합

---

## 🔧 다음 단계 (구체적 작업)

### Step 1: PagedAttentionDiversityGenerator 클래스 구현
파일: `paged_attention_opt.cpp` (기존 파일에 추가)

```cpp
class PagedAttentionDiversityGenerator : public KernelGenerator {
public:
    PagedAttentionDiversityGenerator() 
        : KernelGenerator("adaptive_rkv_diversity", "") {}
    
    JitConstants get_jit_constants(const kernel_impl_params& params) const override {
        auto jit = make_base_jit_constants(params);
        auto desc = params.typed_desc<paged_attention>();
        
        jit.make("NUM_HEADS", desc->heads_num);
        jit.make("HEAD_SIZE", desc->k_head_size);
        jit.make("PAGED_ATTENTION_BLOCK_SIZE", paged_attention_block_size);
        jit.make("SUBGROUP_SIZE", subgroup_size);
        
        // MAX_EVICTABLE_SIZE를 런타임에 결정 필요
        // 현재는 컴파일 타임 상수로 설정
        jit.make("MAX_EVICTABLE_SIZE", 512); // Conservative estimate
        
        return jit;
    }
    
    Arguments get_arguments_desc(const kernel_impl_params& params) const override {
        Arguments args;
        
        if (params.is_dynamic()) {
            args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
        }
        
        // Inputs: 8개
        args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::KEY_CACHE});
        args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::PAST_LENS});
        args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::BLOCK_INDICES});
        args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::BLOCK_INDICES_BEGINS});
        args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::ADAPTIVE_RKV_START_SIZE});
        args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::ADAPTIVE_RKV_EVICTABLE_SIZES});
        args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::ADAPTIVE_RKV_DIVERSITY_BLOCK_SET_INDICES});
        args.push_back({ArgumentDescriptor::Types::INPUT, PagedAttentionInputIdx::ADAPTIVE_RKV_DIVERSITY_BLOCK_SET_BEGINS});
        
        // Output: Diversity values
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 2});
        
        // Internal buffer: Normalized keys
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        
        return args;
    }
    
    DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& impl_param, KernelData& kd, ImplRuntimeParams* rt_params) {
            auto& wgs = kd.params.workGroups;
            auto params = impl_param;
            
            if (!params.is_dynamic()) {
                auto desc = params.typed_desc<paged_attention>();
                
                // Determine batch size and max evictable size
                const auto& evictable_sizes_layout = params.get_input_layout(
                    PagedAttentionInputIdx::ADAPTIVE_RKV_EVICTABLE_SIZES);
                const size_t batch_size = evictable_sizes_layout.get_partial_shape()[0].get_length();
                
                // For now, use conservative max_evictable_size
                // Ideally, should read from memory to get actual max
                const size_t max_evictable_size = 512;
                
                wgs.global = {batch_size, desc->heads_num, max_evictable_size};
                wgs.local = {1, 1, std::min(max_evictable_size, static_cast<size_t>(256))};
            }
        }};
    }
};
```

### Step 2: PagedAttentionOptImpl에 통합

```cpp
class PagedAttentionOptImpl : public SDPAImplBase {
public:
    // ... existing stages ...
    Stage::Ptr pa_diversity_calc = make_stage<PagedAttentionDiversityGenerator>();
    
    PagedAttentionOptImpl(const kernel_impl_params& params) : PagedAttentionOptImpl() {
        const auto desc = params.typed_desc<paged_attention>();
        
        // ... existing stage additions ...
        
        // Add diversity calculation stage if adaptive R-KV is enabled
        if (desc->has_adaptive_rkv && params.output_layouts.size() > 2) {
            add_stage(pa_diversity_calc, params);
        }
    }
    
    event::ptr execute(...) override {
        // ... existing execution logic ...
        
        if (has_scores_output) {
            res_event = {execute_stage(res_event, instance, pa_scores_calc)};
        }

        // Execute diversity calculation
        if (desc->has_adaptive_rkv && params.output_layouts.size() > 2) {
            res_event = {execute_stage(res_event, instance, pa_diversity_calc)};
        }

        return res_event[0];
    }
};
```

### Step 3: Internal Buffer 계산

`get_internal_buffer_descs()` 함수에 diversity용 버퍼 추가:

```cpp
std::vector<BufferDescriptor> get_internal_buffer_descs(const kernel_impl_params& params) const override {
    std::vector<BufferDescriptor> buffers;
    
    // ... existing buffers ...
    
    // Add normalized keys buffer for diversity calculation
    if (params.typed_desc<paged_attention>()->has_adaptive_rkv) {
        const auto& desc = params.typed_desc<paged_attention>();
        const auto& evictable_sizes_layout = params.get_input_layout(
            PagedAttentionInputIdx::ADAPTIVE_RKV_EVICTABLE_SIZES);
        
        size_t batch_size = evictable_sizes_layout.get_partial_shape()[0].get_length();
        size_t max_evictable_size = 512; // Conservative
        size_t norm_keys_size = batch_size * desc->heads_num * max_evictable_size * desc->k_head_size;
        
        BufferDescriptor norm_keys_buffer;
        norm_keys_buffer.size = norm_keys_size * sizeof(float);
        norm_keys_buffer.type = ov::element::f32;
        buffers.push_back(norm_keys_buffer);
    }
    
    return buffers;
}
```

---

## 🧪 테스트 계획

### 1. 단위 테스트
```cpp
TEST(AdaptiveRKV, DiversityKernelCompilation) {
    // JIT 컴파일 테스트
}

TEST(AdaptiveRKV, SimpleDiversityCalculation) {
    // 2개 토큰, 동일/다른 키 벡터로 diversity 검증
}

TEST(AdaptiveRKV, L2Normalization) {
    // 정규화 정확도 테스트
}

TEST(AdaptiveRKV, CosineSimilarity) {
    // 코사인 유사도 계산 정확도
}
```

### 2. 통합 테스트
```cpp
TEST(PagedAttention, AdaptiveRKVIntegration) {
    // Paged Attention + Diversity 전체 파이프라인
}
```

### 3. 성능 테스트
- Eviction 크기별 실행 시간 측정
- 전체 SDPA 대비 오버헤드 측정 (목표: <5%)

---

## 📊 현재 상태 요약

| 항목 | 상태 | 완성도 |
|------|------|--------|
| Diversity 커널 (.cl) | ✅ 완료 | 100% |
| 파라미터 전달 (JIT, Args) | ✅ 완료 | 100% |
| 커널 디스패치 구조 | 🚧 Placeholder | 20% |
| Memory Accessor | ✅ 완료 | 100% |
| Internal Buffer 관리 | ⬜ 미구현 | 0% |
| 단위 테스트 | ⬜ 미구현 | 0% |
| 통합 테스트 | ⬜ 미구현 | 0% |
| 문서화 | ✅ 완료 | 100% |

**전체 진행률**: 약 **70%**

---

## 💡 추천 다음 작업

1. **Step 1 실행**: PagedAttentionDiversityGenerator 클래스 구현 및 추가
2. **Internal Buffer** 관리 코드 추가
3. **컴파일 테스트**: 전체 프로젝트 빌드 확인
4. **간단한 단위 테스트** 작성 및 실행
5. **디버깅**: GPU 로그 확인하며 커널 실행 검증

---

## 📝 참고 사항

- Diversity 커널은 독립적으로 작동하므로 별도로 테스트 가능
- MAX_EVICTABLE_SIZE는 현재 컴파일 타임 상수 (512)로 설정
- 실제 사용 시 런타임에 메모리를 읽어 최대값 결정 필요
- SLM 크기 제한으로 인해 매우 큰 eviction 영역은 타일링 필요

---

이 문서는 Adaptive R-KV 구현의 현재 상태와 완료를 위해 필요한 구체적인 작업을 정리합니다.
