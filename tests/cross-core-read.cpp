

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

#if 0
timeit timer({
    {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, "HW_CYCLES"},
    //{PERF_TYPE_RAW, 0x3c, "CPU_CLK_UNHALTED.THREAD"},
    //{PERF_TYPE_RAW, 0x81d0, "MEM_LOAD_RETIRED.ALL_LOADS"},
    //{PERF_TYPE_HW_CACHE, 0x10002, "LLC_load_misses"},

    //{PERF_TYPE_RAW, 0x10d1, "L2_MISS"},
    //{PERF_TYPE_RAW, 0x02d1, "L2_HIT"},  // https://github.com/intel/perfmon/blob/2dfe7d466d46e89899645c094f8a5a2b8ced74f4/SPR/events/sapphirerapids_core.json#L7397
    //{PERF_TYPE_RAW, 0x04d1, "L3_HIT"},

    //{PERF_TYPE_RAW, 0x01d1, "L1_HIT"}, {PERF_TYPE_RAW, 0x08d1, "L1_MISS"},

    //{PERF_TYPE_RAW, 0x01d1, "L1_HIT"}, {PERF_TYPE_RAW, 0x02d1, "L2_HIT"}, {PERF_TYPE_RAW, 0x40d1, "FB_HIT"},


    // https://en.wikipedia.org/wiki/MESI_protocol#Read_For_Ownership
    // A Read For Ownership (RFO) is an operation in cache coherency protocols that combines a read and an invalidate broadcast.
    // The operation is issued by a processor trying to write into a cache line that is in the shared (S) or invalid (I) states
    // of the MESI protocol. The operation causes all other caches to set the state of such a line to I. A read for ownership transaction
    // is a read operation with intent to write to that memory address. Therefore, this operation is exclusive. It brings data to the cache
    // and invalidates all other processor caches that hold this memory line. This is termed "BusRdX" in tables above.
    //
    //{PERF_TYPE_RAW, 0xC224, "RFO_HIT"},
    //{PERF_TYPE_RAW, 0x2224, "RFO_MISS"},
    {PERF_TYPE_RAW, 0x04d2, "XSNP_FWD"},
    //{PERF_TYPE_RAW, 0xe224, "ALL_RFO"},

    //{PERF_TYPE_RAW, 0x412e, "LONGEST_LAT_CACHE.MISS"},
    //{PERF_TYPE_RAW, 0x0426, "USELESS_HWPF"},
    //{PERF_TYPE_RAW, 0x012A, "HWPF_L2"},
    {PERF_TYPE_RAW, 0x2051, "L1D.HWPF_MISS"},
    {PERF_TYPE_RAW, 0x3024, "L2.HWPF_MISS"},

    //{PERF_TYPE_RAW, 0x10d1, "L2_MISS"},
    //{PERF_TYPE_RAW, 0x04d1, "L3_HIT"},
    //{PERF_TYPE_RAW, 0x20d1, "L3_MISS"},

    //{PERF_TYPE_RAW, 0x81d0, "ALL_LOADS"},        // MEM_INST_RETIRED.ALL_LOADS

    //{PERF_TYPE_RAW, 0x08d1, "L1_MISS"},
    //{PERF_TYPE_RAW, 0x10d1, "L2_MISS"},
    //{PERF_TYPE_RAW, 0x20d1, "L3_MISS"},

    //{PERF_TYPE_HW_CACHE, 0x2, "LLC_loads"},
    //{PERF_TYPE_RAW, 0x02b1, "UOPS_EXECUTED.CORE"},
});



void cross_core_L2_read(size_t nbytes = 256 * 256 * 4) {
    int nthr = get_nthr();
    uint8_t* thr_data[128];
#pragma omp parallel
    {
        int ithr = omp_get_thread_num();
        thr_data[ithr] = reinterpret_cast<uint8_t*>(aligned_alloc(64, nbytes));
        memset(thr_data[ithr], 1, nbytes);
    }

    timer.tag(nthr, "threads");

    std::cout << ":::::::::::::::::::::::::::::::\n";
    for (int r = 0; r < 5; r++) {
        for (int k = 0; k < 5; k++) {
            timer(1, [&]() {
#pragma omp parallel
                {
                    int ithr = omp_get_thread_num();
                    load_prefetch_L2(thr_data[ithr], nbytes);
                }
            });
        }

        std::cout << "======= cross-core-cache-read r=" << r << std::endl;
        timer(1, [&]() {
#pragma omp parallel
            {
                int ithr = omp_get_thread_num() + 1;
                if (ithr >= nthr)
                    ithr -= nthr;
                load_prefetch_L2(thr_data[ithr], nbytes);
            }
        });
    }
}
#endif

void my_memset(void* dst_mem, int val, int size) {
    auto* dst0 = reinterpret_cast<uint8_t*>(dst_mem);
    auto vf = _mm256_set1_ps(val);

    // 256bits/32bytes/8floats
    //for (int w = 0; w < W; w += 32) _mm_prefetch(dst0 + stride + w, _MM_HINT_NTA);

    for (int i = 0; i < size; i += 32)
        _mm256_storeu_ps(reinterpret_cast<float*>(dst0 + i), vf);
}


double read_all(void* pv, int bytes, int rounds = 1) {
    auto t0 = __rdtsc();
    auto* p = reinterpret_cast<uint8_t*>(pv);
    int i;
    auto sum0 = _mm512_setzero_epi32();
    auto sum1 = _mm512_setzero_epi32();
    auto sum2 = _mm512_setzero_epi32();
    auto sum3 = _mm512_setzero_epi32();
    for (int r = 0; r < rounds; r++) {
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
    }
    // avoid compiler optimization
    sum0 = _mm512_add_epi32(sum0, sum1);
    sum2 = _mm512_add_epi32(sum2, sum3);
    sum0 = _mm512_add_epi32(sum0, sum2);
    if (_mm512_reduce_add_epi32(sum0) == 0x1234) {
        std::cout << 1;
    }
    return tsc2second(__rdtsc() - t0);
};

void cross_core_L2_read2(size_t nbytes = 128 * 1024) {
    uint8_t* common_src;
    int nthr = get_nthr();
    std::vector<std::stringstream> ss(nthr);
    perf_log plog({
        //{PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, "HW_CYCLES"},
        //{PERF_TYPE_RAW, 0x02d1, "L2_HIT"},
        //{PERF_TYPE_RAW, 0x04d1, "L3_HIT"},
        {PERF_TYPE_RAW, 0x01d2, "XSNP_MISS"},
        {PERF_TYPE_RAW, 0x02d2, "XSNP_NO_FWD"},
        {PERF_TYPE_RAW, 0x04d2, "XSNP_FWD"},
        {PERF_TYPE_RAW, 0x08d2, "XSNP_NONE"},
    });

    printf("========= a common 128KB buffer read by other cores one-by-one ==========\n");
    //
    // first core reads from core0's local `exclusive` copy triggers flush to make the data in `shared`
    // state.
    //
#pragma omp parallel
    {
        int ithr = omp_get_thread_num();
        if (ithr == 0) {
            common_src = reinterpret_cast<uint8_t*>(aligned_alloc(64, nbytes));
            my_memset(common_src, 1, nbytes);
        }


        for (int k = 0; k < 3; k++) {
            //for (int i = nthr-2; i>=0; i--) {
            for (int i = 0; i <= nthr-2; i++) {
                #pragma omp barrier
                // now each core has it's own data in it's own L2 cache
                // in `Exclusive` state. Now we let core 1~(n-1) to read from core0
                if (ithr == i) {
                    for(int r = 0; r < 3; r ++)
                        plog([&](){read_all(common_src, nbytes);});
                }
            }
        }
        //
        #pragma omp barrier
        if (ithr == nthr-1) {
            for(int r = 0; r < 3; r ++)
                plog([&](){read_all(common_src, nbytes);});
        }
    }
}

void clr_cache() {
    static std::vector<uint8_t> big_buffer(1024*1024*2, 0);
    my_memset(&big_buffer[0], 1, big_buffer.size());
    read_all(&big_buffer[0], big_buffer.size());
};

void cross_core_L2_read3(size_t nbytes = 128 * 1024) {
    uint8_t* common_src;
    perf_log plog({
        //{PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, "HW_CYCLES"},
        //{PERF_TYPE_RAW, 0x02d1, "L2_HIT"},
        //{PERF_TYPE_RAW, 0x04d1, "L3_HIT"},
        {PERF_TYPE_RAW, 0x01d2, "XSNP_MISS"},
        {PERF_TYPE_RAW, 0x02d2, "XSNP_NO_FWD"},
        {PERF_TYPE_RAW, 0x04d2, "XSNP_FWD"},
        {PERF_TYPE_RAW, 0x08d2, "XSNP_NONE"},
    });
    int nthr = get_nthr();
    common_src = reinterpret_cast<uint8_t*>(aligned_alloc(64, nbytes));
    my_memset(common_src, 1, nbytes);

    clr_cache();
    printf("======== concurrent multi-threads reading from a common 128K buffer written by thread0 ===========\n");

    std::vector<std::stringstream> ss(nthr);
#pragma omp parallel
    {
        int ithr = omp_get_thread_num();
        // common_src buffer in core0's local L2 cache
        // in `exclusive` state.
        // now following read_all latency test may have 2 results:
        //  1. core0 starts early than other threads
        //     it can read the buffer very quick from local L2 cache.
        //     other cores needs to read from cache-to-cache flush so is slow
        //  2. other cores starts early, core0's data in L2 must transit to `Shared`
        //     and put flush with cached data on bus, this also would block core0's
        //     execution because L2's read bandwidth is taken by cache-to-cache
        //     transfer. so all cores are slow.

        #pragma omp barrier
        plog([&]() {read_all(common_src, nbytes);});

        #pragma omp barrier
        plog([&]() {read_all(common_src, nbytes);});

        #pragma omp barrier
        plog([&]() {read_all(common_src, nbytes);});
    }
}

int main() {
    MSRConfig _msr1;

    printf("%f\n", tsc2second(0));

    clr_cache();
    cross_core_L2_read2(1024*128);
    clr_cache();
    cross_core_L2_read3(1024*128);
    return 0;
}