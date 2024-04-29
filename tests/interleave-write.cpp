

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

constexpr double nbytes = 512 * 4096 * sizeof(ov::bfloat16);

inline void copy_out_32x32(tensor2D<float>& C, int x0, int x1) {
    auto strideC = C.stride;
#define M512_STORE _mm512_storeu_ps

    // #define M512_STORE _mm512_stream_ps

    for (int y = 0; y < C.dims[0]; y += 32) {
        uint8_t* dst0 = reinterpret_cast<uint8_t*>(&C(y, x0));
        for (int x = x0; x < x1; x += 32, dst0 += 32 * sizeof(float)) {
            auto ra0 = _mm512_set1_ps(0.1f);
            auto ra1 = _mm512_set1_ps(0.2f);
            auto rb0 = _mm512_set1_ps(0.3f);
            auto rb1 = _mm512_set1_ps(0.4f);
            auto* dst = dst0;
            for (int i = 0; i < 16; i += 2) {
                M512_STORE(dst, ra0);
                M512_STORE(dst + 64, rb0);
                dst += strideC;
                M512_STORE(dst, ra1);
                M512_STORE(dst + 64, rb1);
                dst += strideC;
            }
            for (int i = 0; i < 16; i += 2) {
                M512_STORE(dst, ra0);
                M512_STORE(dst + 64, rb0);
                dst += strideC;
                M512_STORE(dst, ra1);
                M512_STORE(dst + 64, rb1);
                dst += strideC;
            }
        }
    }
}

inline void copy_out(tensor2D<float>& C, int x0, int x1) {
    auto strideC = C.stride;
    auto ra0 = _mm512_set1_ps(0.1f);

#define M512_STORE _mm512_storeu_ps
    uint8_t* dst0 = reinterpret_cast<uint8_t*>(&C(0, 0));

    for (int y = 0; y < C.dims[0]; y += 16, dst0 += 16 * strideC) {
        for (int x = x0; x < x1; x += 16) {
            auto* dst = dst0 + x * sizeof(float);
            for (int i = 0; i < 16; i++) {
                M512_STORE(dst, ra0);
                dst += strideC;
            }
        }
    }
}

/*
inline void copy_out(tensor2D<float>& C, int x0, int x1) {
    auto strideC = C.stride;
    auto ra0 = _mm256_set1_ps (0.1f);

#define M512_STORE _mm256_storeu_ps
    uint8_t* dst0 = reinterpret_cast<uint8_t*>(&C(0, x0));
    for (int y = 0; y < C.dims[0]; y ++, dst0 += strideC) {
        auto* dst = dst0;
        for (int x = x0; x < x1; x += 8, dst += 8*sizeof(float)) {
            M512_STORE( reinterpret_cast<float*>(dst), ra0);
        }
    }
}
*/

int test_copy_out(int subN = 8) {
    auto nthr = get_nthr();

    tensor2D<float> fullC(256, nthr * subN);
    std::vector<tensor2D<float>> partC(nthr);

#pragma omp parallel
    {
        int ithr = omp_get_thread_num();
        partC[ithr] = tensor2D<float>(256, subN);
    }

    for (int r = 0; r < 4; r++) {
        auto latency = timer.tag("part", partC[0].capacity / 1000, "KBytes")(1, [&]() {
#pragma omp parallel
            {
                int ithr = omp_get_thread_num();
                copy_out(partC[ithr], 0, subN);
            }
        });
        std::cout << "\t   copy_out part BW : " << partC[0].capacity * 1e-9 / latency << " x " << nthr << "=" << fullC.capacity * 1e-9 / latency << " GB/s" << std::endl;

        latency = timer.tag("full", partC[0].capacity / 1000, "KBytes")(1, [&]() {
#pragma omp parallel
            {
                int ithr = omp_get_thread_num();
                copy_out(fullC, ithr * subN, ithr * subN + subN);
            }
        });
        std::cout << "\t   copy_out full BW : " << partC[0].capacity * 1e-9 / latency << " x " << nthr << "=" << fullC.capacity * 1e-9 / latency << " GB/s" << std::endl;
    }
    return 0;
}

void my_memset(void* dst_mem, int stride, int H, int W, float f = 0.1f) {
    auto* dst0 = reinterpret_cast<uint8_t*>(dst_mem);
    auto vf = _mm256_set1_ps(f);
    for (int w = 0; w < W; w += 32) _mm_prefetch(dst0 + w, _MM_HINT_NTA);

    for (int h = 0; h < H; h++, dst0 += stride) {
        // 256bits/32bytes/8floats
        for (int w = 0; w < W; w += 32) _mm_prefetch(dst0 + stride + w, _MM_HINT_NTA);

        for (int w = 0; w < W; w += 32) {
            _mm256_storeu_ps(reinterpret_cast<float*>(dst0 + w), vf);
        }
    }
}


int write_to_Cache(int per_thread_size, bool is_full) {
    auto nthr = get_nthr();
    const int M = 256;
    std::vector<std::shared_ptr<uint8_t>> buff(nthr, nullptr);
    std::vector<uint8_t*> ptrs(nthr, nullptr);

    int stride;
    if (!is_full) {
#pragma omp parallel
        {
            int ithr = omp_get_thread_num();
            buff[ithr] = std::shared_ptr<uint8_t>(reinterpret_cast<uint8_t*>(aligned_alloc(64, M * per_thread_size)), [](void* p) { ::free(p); });
            ptrs[ithr] = buff[ithr].get();
        }
        stride = (per_thread_size);
    } else {
        // use single big buffer
        buff[0] = std::shared_ptr<uint8_t>(reinterpret_cast<uint8_t*>(aligned_alloc(64, M * nthr * per_thread_size)), [](void* p) { ::free(p); });
        for (int ithr = 0; ithr < nthr; ithr++) {
            ptrs[ithr] = buff[0].get() + ithr * per_thread_size;
        }
        stride = (nthr * per_thread_size);
    }

    timer.tag(is_full ? "full" : "part", M, "x", per_thread_size, "Bytes", nthr, "Threads")(
        100,
        [&]() {
#pragma omp parallel
            {
                int ithr = omp_get_thread_num();
                my_memset(ptrs[ithr], stride, M, per_thread_size);
#if 0
                auto v = _mm256_set1_ps(ithr);
                auto* dst0 = reinterpret_cast<float*>(ptrs[ithr]);
                for (int i = 0; i < 256; i++, dst0 += stride) {
                    auto* dst = dst0;
                    for (int r = 0; r < per_thread_size; r += 32, dst += 8) {
                        _mm256_storeu_ps(dst, v);
                    }
                }
                for (int r = 0; r < per_thread_size; r += 32) {
                    auto* dst = reinterpret_cast<float*>(&perThreadArgs[ithr * per_thread_size + r]);
                    for (int i = 0; i < 256; i++, dst += stride) {
                        _mm256_storeu_ps(dst, v);
                    }
                }
#endif
            }
        },
        1.0 * per_thread_size * M * nthr);
    return 1;
}

int read_from_Cache(size_t nbytes, bool USE_SAME) {
    int nthr;
    uint8_t* thr_data[128];

#pragma omp parallel
    {
        int ithr = omp_get_thread_num();

        if (0 == ithr) {
            nthr = omp_get_num_threads();
            thr_data[ithr] = reinterpret_cast<uint8_t*>(aligned_alloc(4096, nbytes));
            memset(thr_data[ithr], 1, nbytes);
        } else if (!USE_SAME) {
            thr_data[ithr] = reinterpret_cast<uint8_t*>(aligned_alloc(4096, nbytes));
            memset(thr_data[ithr], 1, nbytes);
        }
    }

    if (USE_SAME) {
        for (int ithr = 1; ithr < nthr; ithr++) {
            thr_data[ithr] = thr_data[0];
        }
    }
    timer.tag(USE_SAME ? "SAME" : "MULTI", static_cast<uint32_t>(nbytes / 1e3), "KBytes", nbytes / 64, "CacheLines", nthr, "threads");

    for (int xx = 0; xx < 2; xx++) {
        std::cout << "========== clflush " << xx << " ===========" << std::endl;
#pragma omp parallel
        {
            int ithr = omp_get_thread_num();
            clflush(thr_data[ithr], nbytes);
        }

        for (int t = 0; t < 5; t++) {
            timer(
                1,
                [&]() {
#pragma omp parallel
                    {
                        int ithr = omp_get_thread_num();
                        load_prefetch_L2(thr_data[ithr], nbytes);
                    }
                },
                nbytes);
        }
    }
    return 0;
}

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


void cross_core_L2_write(size_t nbytes = 256 * 256 * 4) {
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
                    my_memset(thr_data[ithr], nbytes, 1, nbytes);
                }
            });
        }

        std::cout << "======= cross-core-cache-write r=" << r << std::endl;
        timer(1, [&]() {
#pragma omp parallel
            {
                int ithr = omp_get_thread_num() + 1;
                if (ithr >= nthr)
                    ithr -= nthr;
                my_memset(thr_data[ithr], nbytes, 1, nbytes);
            }
        });
    }
}

int main() {
    // cross_core_L2_write(); cross_core_L2_read();    return 0;
    MSRConfig _msr1;
    for(int r = 0; r < 10; r++) {
        write_to_Cache(1024, true);
        write_to_Cache(1024, false);

        //write_to_Cache(2048, true);
        //write_to_Cache(2048, false);
        //write_to_Cache(4096, true);
        //write_to_Cache(4096, false);
    }
    return 0;

    read_from_Cache(1024 * 1024 * 1, false);
    read_from_Cache(1024 * 1024 * 1.5, false);
    read_from_Cache(1024 * 1024 * 2, false);
    read_from_Cache(1024 * 1024 * 2, true);

    read_from_Cache(1024 * 1024 * 3.8, false);
    read_from_Cache(1024 * 1024 * 3.8, true);

    return test_copy_out(256);

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