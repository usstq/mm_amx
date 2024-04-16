


#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#include <stdlib.h>

#include <omp.h>
#include "bf16.hpp"
// #include "kernels_avx512.hpp"
#include "kernels_amx.hpp"
#include "tensor2D.hpp"
#include "timeit.hpp"

timeit timer({
    //{PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, "HW_CYCLES"},
    //{PERF_TYPE_RAW, 0x3c, "CPU_CLK_UNHALTED.THREAD"},
    //{PERF_TYPE_RAW, 0x81d0, "MEM_LOAD_RETIRED.ALL_LOADS"},
    //{PERF_TYPE_HW_CACHE, 0x10002, "LLC_load_misses"},
    
    //{PERF_TYPE_RAW, 0x10d1, "L2_MISS"},
    //{PERF_TYPE_RAW, 0x02d1, "L2_HIT"},  // https://github.com/intel/perfmon/blob/2dfe7d466d46e89899645c094f8a5a2b8ced74f4/SPR/events/sapphirerapids_core.json#L7397
    //{PERF_TYPE_RAW, 0x04d1, "L3_HIT"},

    //{PERF_TYPE_RAW, 0x01d1, "L1_HIT"}, {PERF_TYPE_RAW, 0x08d1, "L1_MISS"},

    //{PERF_TYPE_RAW, 0x01d1, "L1_HIT"}, {PERF_TYPE_RAW, 0x02d1, "L2_HIT"}, {PERF_TYPE_RAW, 0x40d1, "FB_HIT"}, 
    {PERF_TYPE_RAW, 0x01d1, "L1_HIT"}, {PERF_TYPE_RAW, 0x02d1, "L2_HIT"}, {PERF_TYPE_RAW, 0x04d1, "L3_HIT"}, {PERF_TYPE_RAW, 0x20d1, "L3_MISS"},

    //{PERF_TYPE_RAW, 0x81d0, "ALL_LOADS"},        // MEM_INST_RETIRED.ALL_LOADS

    //{PERF_TYPE_RAW, 0x08d1, "L1_MISS"},
    //{PERF_TYPE_RAW, 0x10d1, "L2_MISS"},
    //{PERF_TYPE_RAW, 0x20d1, "L3_MISS"},

    
    //{PERF_TYPE_HW_CACHE, 0x2, "LLC_loads"},
    //{PERF_TYPE_RAW, 0x02b1, "UOPS_EXECUTED.CORE"},
});



void clflush(void* pv, int bytes) {
    auto* p = reinterpret_cast<uint8_t*>(pv);
    for (int i = 0; i < bytes; i += 64) {
        _mm_clflushopt(p + i);
    }
    _mm_mfence();
};

template<typename V>
void clflush(tensor2D<V>& t) {
    clflush(&t[0], t.capacity);
};

void sw_prefetch_L2(void* pv, int bytes) {
    auto* p = reinterpret_cast<uint8_t*>(pv);
    int i;
    for (i = 0; i + 256 <= bytes; i += 64 * 4) {
        _mm_prefetch(p + i, _MM_HINT_T2);
        _mm_prefetch(p + i + 64, _MM_HINT_T2);
        _mm_prefetch(p + i + 64 * 2, _MM_HINT_T2);
        _mm_prefetch(p + i + 64 * 3, _MM_HINT_T2);
    }
    for (; i < bytes; i += 64) {
        _mm_prefetch(p + i, _MM_HINT_T2);
    }
    _mm_mfence();
};

template<typename V>
void sw_prefetch_L2(tensor2D<V>& t) {
    sw_prefetch_L2(&t[0], t.capacity);
};

void load_prefetch_L2(void* pv, int bytes) {
    auto* p = reinterpret_cast<uint8_t*>(pv);
    int i;
    auto sum0 = _mm512_setzero_epi32();
    auto sum1 = _mm512_setzero_epi32();
    auto sum2 = _mm512_setzero_epi32();
    auto sum3 = _mm512_setzero_epi32();
    for (i = 0; i + 256 <= bytes; i += 64 * 4) {
        auto a0 = _mm512_loadu_epi32(p + i);
        auto a1 = _mm512_loadu_epi32(p + i + 64);
        auto a2 = _mm512_loadu_epi32(p + i + 64*2);
        auto a3 = _mm512_loadu_epi32(p + i + 64*3);
        sum0 = _mm512_add_epi32(sum0, a0);
        sum1 = _mm512_add_epi32(sum1, a1);
        sum2 = _mm512_add_epi32(sum2, a2);
        sum3 = _mm512_add_epi32(sum3, a3);
    }
    sum0 = _mm512_add_epi32(sum0, sum1);
    sum2 = _mm512_add_epi32(sum2, sum3);
    sum0 = _mm512_add_epi32(sum0, sum2);
    if (_mm512_cvtsi512_si32(sum0) > 0) {
        std::cout << 1;
    }
};
template<typename V>
void load_prefetch_L2(tensor2D<V>& t) {
    load_prefetch_L2(&t[0], t.capacity);
};

float allsum(const float* src, int count) {
    auto sum0 = _mm512_setzero_ps();
    auto sum1 = _mm512_setzero_ps();
    auto sum2 = _mm512_setzero_ps();
    auto sum3 = _mm512_setzero_ps();
    for (int64_t i = 0; i < count; i += 16 * 4) {
        auto a0 = _mm512_loadu_ps(src + i);
        auto a1 = _mm512_loadu_ps(src + i + 16);
        auto a2 = _mm512_loadu_ps(src + i + 16 * 2);
        auto a3 = _mm512_loadu_ps(src + i + 16 * 3);
        sum0 = _mm512_add_ps(sum0, a0);
        sum1 = _mm512_add_ps(sum1, a1);
        sum2 = _mm512_add_ps(sum2, a2);
        sum3 = _mm512_add_ps(sum3, a3);
    }

    sum0 = _mm512_add_ps(sum0, sum1);
    sum2 = _mm512_add_ps(sum2, sum3);
    sum0 = _mm512_add_ps(sum0, sum2);
    return _mm512_reduce_add_ps(sum0);
}




/*


although :
 - memory foot-print < L2 size
 - kernel `allsum` load avx512 register in one cache line unit

(L1_HIT + L1_MISS) << MEM_INST_RETIRED.ALL_LOADS, due to https://stackoverflow.com/questions/77052435/why-do-mem-load-retired-l1-hit-and-mem-load-retired-l1-miss-not-add-to-the-total
so (L1_HIT + L1_MISS + FB_HIT) = MEM_INST_RETIRED.ALL_LOADS, due to avx512 load from unaligned address hit FillBuffer instead.

by making the tensor memory cache-line aligned (64 bytes), FB_HIT is gone, and (L1_HIT + L1_MISS) = MEM_INST_RETIRED.ALL_LOADS


after clflush the tensor, next kernel execution causes big L3_MISS but also considerable amount of L2_HIT, which means
some prefetcher 

524288_Bytes_8192_CacheLines    : 46.76 us x 1, L1_HIT=49, L2_HIT=2786, L3_HIT=0, L3_MISS=5423

*/

constexpr int N = 1024*1024/8;
float values[N] __attribute__((aligned(64)));  

int main() {
    int nbytes = N*sizeof(float);
    float* base = &values[0];

    for(int i = 0; i < N; i++) values[i] = 1;

    float expected_sum = N;

    MSRConfig _msr0(0x1A0, MASK_1A0_PF);
    MSRConfig _msr1(0x1A4, MASK_1A4_PF);

    printf("base=%p\n", base);

    timer.tag(nbytes, "Bytes", nbytes/64, "CacheLines");

    for(int t = 0; t < 3; t++) {
        float sum = 0;
        clflush(base, nbytes);
        std::cout << "========== clflush ===========" << std::endl;
        if (t == 2) {
            std::cout << "========== sw_prefetch ===========" << std::endl;
            sw_prefetch_L2(base, nbytes);
        }
        if (t == 3) {
            std::cout << "========== load_prefetch_L2 ===========" << std::endl;
            //load_prefetch_L2(t0);
            sum = allsum(base, N);
            ASSERT(expected_sum == sum);
        }
        for(int r = 0; r < 10; r++) {
            timer(1, [&](){ sum = allsum(base, N); });
            ASSERT(expected_sum == sum);
        }
    }
    return 0;
}