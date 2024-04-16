

#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#include <stdlib.h>

#include "bf16.hpp"
#include <omp.h>
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
    {PERF_TYPE_RAW, 0x01d1, "L1_HIT"},
    {PERF_TYPE_RAW, 0x02d1, "L2_HIT"},
    {PERF_TYPE_RAW, 0x04d1, "L3_HIT"},
    {PERF_TYPE_RAW, 0x20d1, "L3_MISS"},

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

template <typename V>
void clflush(tensor2D<V>& t) {
    clflush(&t[0], t.capacity);
};

void sw_prefetch_L2(void* pv, int bytes) {
    auto* p = reinterpret_cast<uint8_t*>(pv);
    int i;
    for (i = 0; i + 256 <= bytes; i += 64 * 4) {
        _mm_prefetch(p + i, _MM_HINT_T1);
        _mm_prefetch(p + i + 64, _MM_HINT_T1);
        _mm_prefetch(p + i + 64 * 2, _MM_HINT_T1);
        _mm_prefetch(p + i + 64 * 3, _MM_HINT_T1);
    }
    for (; i < bytes; i += 64) {
        _mm_prefetch(p + i, _MM_HINT_T1);
    }
    _mm_mfence();
};

template <typename V>
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
        auto a2 = _mm512_loadu_epi32(p + i + 64 * 2);
        auto a3 = _mm512_loadu_epi32(p + i + 64 * 3);
        sum0 = _mm512_add_epi32(sum0, a0);
        sum1 = _mm512_add_epi32(sum1, a1);
        sum2 = _mm512_add_epi32(sum2, a2);
        sum3 = _mm512_add_epi32(sum3, a3);
    }
    sum0 = _mm512_add_epi32(sum0, sum1);
    sum2 = _mm512_add_epi32(sum2, sum3);
    sum0 = _mm512_add_epi32(sum0, sum2);
    if (_mm512_cvtsi512_si32(sum0) < 0) {
        std::cout << 1;
    }
};
template <typename V>
void load_prefetch_L2(tensor2D<V>& t) {
    load_prefetch_L2(&t[0], t.capacity);
};

constexpr double nbytes = 512 * 4096 * sizeof(ov::bfloat16);

int main() {
    EnvVar USE_SAME("USE_SAME", 0);

    // MSRConfig _msr0(0x1A0, MASK_1A0_PF);
    // MSRConfig _msr1(0x1A4, MASK_1A4_PF);

    timer.tag(nbytes, "Bytes", nbytes / 64, "CacheLines");

    int nthr;
    uint8_t* thr_data[128];

#pragma omp parallel
    {
        int ithr = omp_get_thread_num();

        if (0 == ithr) {
            nthr = omp_get_num_threads();
            thr_data[ithr] = reinterpret_cast<uint8_t*>(aligned_alloc(64, nbytes));
            memset(thr_data[ithr], 1, nbytes);
        } else if (!USE_SAME) {
            thr_data[ithr] = reinterpret_cast<uint8_t*>(aligned_alloc(64, nbytes));
            memset(thr_data[ithr], 1, nbytes);
        }
    }

    if (USE_SAME) {
        for (int ithr = 1; ithr < nthr; ithr++) {
            thr_data[ithr] = thr_data[0];
        }
    }

    std::cout << " >>>>>>>>>>>>>>>  nthr = " << nthr << std::endl;

    for (int xx = 0; xx < 3; xx++) {
        std::cout << "========== clflush ===========" << std::endl;
#pragma omp parallel
        {
            int ithr = omp_get_thread_num();
            clflush(thr_data[ithr], nbytes);
        }

        for (int t = 0; t < 3; t++) {
            auto latency = timer(1, [&]() {
#pragma omp parallel
                {
                    int ithr = omp_get_thread_num();
                    sw_prefetch_L2(thr_data[ithr], nbytes);
                }
            });
            std::cout << "\t   sw_prefetch_L2 BW: " << nbytes * 1e-9 / latency << " GB/s x " << nthr << " = " << nbytes * nthr * 1e-9 / latency << " GB/s" << std::endl;

            auto latency2 = timer(1, [&]() {
#pragma omp parallel
                {
                    int ithr = omp_get_thread_num();
                    load_prefetch_L2(thr_data[ithr], nbytes);
                }
            });
            std::cout << "\t   load_prefetch_L2 BW: " << nbytes * 1e-9 / latency2 << " GB/s x " << nthr << " = " << nbytes * nthr * 1e-9 / latency2 << " GB/s" << std::endl;
        }
    }

#if 0
    std::cout << "========== clflush ===========" << std::endl;
#pragma omp parallel
    {
        int ithr = omp_get_thread_num();
        clflush(thr_data[ithr], nbytes);
    }

    for (int ithr = 0; ithr < nthr; ithr++) {
        std::cout << " load on ithr " << ithr << std::endl;
        for (int t = 0; t < 3; t++) {
            auto latency = timer(1, [&]() {
#pragma omp parallel
                {
                    if (ithr == omp_get_thread_num())
                        load_prefetch_L2(thr_data[ithr], nbytes);
                }
            });
        }
    }
#endif
    return 0;
}