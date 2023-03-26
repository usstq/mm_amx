#pragma once

#include <iostream>
#include <omp.h>
#include <cstdlib>
#include <chrono>
#include <immintrin.h>
#include <sstream>
#include <iomanip>



// test BW
double test_bw(double dur, int64_t size) {
    // allocate per-thread buffer
    auto tsc_second = get_tsc_ticks_per_second();

    uint8_t* data[128] = { 0 };
    int failed = 0;
    #pragma omp parallel reduction(+:failed)
    {
        int tid = omp_get_thread_num();
        data[tid] = reinterpret_cast<uint8_t*>(aligned_alloc(64, size));
        if (data[tid] == nullptr) {
            std::cout << "Error, aligned_alloc failed!" << std::endl;
            failed++;
        }
        // memset to 1 ensures physical pages are really allocated
        memset(data[tid], 1, size);
    }
    if (failed) {
        return 0;
    }

    auto doTest = [&](uint8_t* base, int duration_ms, int64_t & actual_reps) {
        auto sum0 = _mm256_setzero_ps();
        auto sum1 = _mm256_setzero_ps();
        auto sum2 = _mm256_setzero_ps();
        auto sum3 = _mm256_setzero_ps();
        auto tsc0 = __rdtsc();
        int64_t r;
        auto tsc_limit = tsc_second * duration_ms/1000l;
        for (r = 0; ; r++) {
            uint8_t* src = base;
            for (int64_t i = 0; i < size; i += 64*4) {
                auto a0 = _mm256_load_ps(reinterpret_cast<const float*>(src));
                auto a1 = _mm256_load_ps(reinterpret_cast<const float*>(src + 64));
                auto a2 = _mm256_load_ps(reinterpret_cast<const float*>(src + 64 * 2));
                auto a3 = _mm256_load_ps(reinterpret_cast<const float*>(src + 64 * 3));
                src+= 32*4;
                sum0 = _mm256_add_ps(sum0, a0);
                sum1 = _mm256_add_ps(sum1, a1);
                sum2 = _mm256_add_ps(sum2, a2);
                sum3 = _mm256_add_ps(sum3, a3);
            }
            if (__rdtsc() - tsc0 > tsc_limit) {
                actual_reps = r+1;
                break;
            }
        }

        sum0 = _mm256_add_ps(sum0, sum1);
        sum2 = _mm256_add_ps(sum2, sum3);
        sum0 = _mm256_add_ps(sum0, sum2);
        auto s7 = _mm256_extract_epi32(_mm256_cvtps_epi32(sum0), 7);
        if (s7 == 0)  return _mm256_extract_epi32(_mm256_cvtps_epi32(sum0), 0);
        if (s7 == 1)  return _mm256_extract_epi32(_mm256_cvtps_epi32(sum0), 1);
        if (s7 == 2)  return _mm256_extract_epi32(_mm256_cvtps_epi32(sum0), 2);
        if (s7 == 3)  return _mm256_extract_epi32(_mm256_cvtps_epi32(sum0), 3);
        if (s7 == 4)  return _mm256_extract_epi32(_mm256_cvtps_epi32(sum0), 4);
        if (s7 == 5)  return _mm256_extract_epi32(_mm256_cvtps_epi32(sum0), 5);
        return _mm256_extract_epi32(_mm256_cvtps_epi32(sum0), 6);
    };

    // warm-up cache 
    int64_t actual_reps[128];
    int prevent_opt = 0;
    #pragma omp parallel reduction(+:prevent_opt)
    {
        int tid = omp_get_thread_num();
        prevent_opt += doTest(data[tid], 100, actual_reps[tid]);
    }

    auto t1 = std::chrono::steady_clock::now();
    #pragma omp parallel reduction(+:prevent_opt)
    {
        int tid = omp_get_thread_num();
        prevent_opt += doTest(data[tid], dur*1000, actual_reps[tid]);
    }
    auto t2 = std::chrono::steady_clock::now();

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        free(data[tid]);
    }

    int64_t total_reps = 0;
    #pragma omp parallel reduction(+:total_reps)
    {
        int tid = omp_get_thread_num();
        total_reps += actual_reps[tid];
    }

    std::chrono::duration<double> dt = t2 - t1;

    auto bytes_per_sec = total_reps / dt.count() * size;

    if (prevent_opt == 123) {
        std::cout << prevent_opt << std::endl;
    }
    return bytes_per_sec;
}

template<int64_t size>
constexpr int64_t next_size() { 
    int64_t nsize = size*2;
    if (nsize > 1024 * 1024 * 1024)
        return 0;
    return nsize;
}

inline int omp_thread_count() {
    int n = 0;
    #pragma omp parallel reduction(+:n)
    n += 1;
    return n;
}

void test_all_bw(double duration){
    auto test = [&](int64_t size) {
        auto OMP_NT = omp_thread_count();
        auto bw = test_bw(duration, size);
        std::cout << "@BufferSize " << pretty_size(size) << " : " << pretty_size(bw) << "B/s  x " << OMP_NT << std::endl;
    };

    test(15*1024);
    test(30*1024);
    test(1*1024*1024);  // 1MB  L2
    test(13*1024*1024); // 13MB L3
    test(56*1024*1024); // 56MB L3
    test(128*1024*1024); // 128MB L3 + DDR
    test(512*1024*1024); // 512MB
    test(1024*1024*1024l); // 1GB DDR
    test(2*1024*1024*1024l); // 1GB DDR
}
