#pragma once

#include <iostream>
#include <omp.h>
#include <cstdlib>
#include <chrono>
#include <immintrin.h>
#include <sstream>
#include <iomanip>



// test BW
template<int64_t size>
double test_bw(double dur) {
    // allocate per-thread buffer
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
    }
    if (failed) {
        return 0;
    }

    auto doTest = [&](uint8_t* base, int64_t repeats) {
        auto sum0 = _mm256_setzero_ps();
        auto sum1 = _mm256_setzero_ps();
        auto sum2 = _mm256_setzero_ps();
        auto sum3 = _mm256_setzero_ps();
        for (int64_t r = 0; r < repeats; r++) {
            uint8_t* src = base;
            for (int64_t i = 0; i < size; i += 32*4) {
                auto a0 = _mm256_load_ps(reinterpret_cast<const float*>(src));
                auto a1 = _mm256_load_ps(reinterpret_cast<const float*>(src + 32));
                auto a2 = _mm256_load_ps(reinterpret_cast<const float*>(src + 32 * 2));
                auto a3 = _mm256_load_ps(reinterpret_cast<const float*>(src + 32 * 3));
                src+= 32*4;
                sum0 = _mm256_add_ps(sum0, a0);
                sum1 = _mm256_add_ps(sum1, a1);
                sum2 = _mm256_add_ps(sum2, a2);
                sum3 = _mm256_add_ps(sum3, a3);
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
    int prevent_opt = 0;
    #pragma omp parallel reduction(+:prevent_opt)
    {
        int tid = omp_get_thread_num();
        prevent_opt += doTest(data[tid], 10);
    }
    // estimate rep_for_dur
    int64_t rep_for_dur = 0;
    for (int64_t rep = 10; ; rep *= 2) {
        // count time
        auto t1 = std::chrono::high_resolution_clock::now();
        #pragma omp parallel reduction(+:prevent_opt)
        {
            int tid = omp_get_thread_num();
            prevent_opt += doTest(data[tid],rep);
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        auto time_us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        //std::cout << __LINE__ << "," << rep << ", " << time_us << std::endl;
        if (time_us > 100*1000) {
            // >100ms, we can estimate how many repeates required to last `dur` second
            rep_for_dur = (dur * 1e6 / time_us) * rep;
            break;
        }
    }


    auto t1 = std::chrono::high_resolution_clock::now();
    #pragma omp parallel reduction(+:prevent_opt)
    {
        int tid = omp_get_thread_num();
        prevent_opt += doTest(data[tid],rep_for_dur);
    }
    auto t2 = std::chrono::high_resolution_clock::now();

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        free(data[tid]);
    }

    std::chrono::duration<double> dt = t2 - t1;

    auto bytes_per_sec = rep_for_dur / dt.count() * size;
    //std::cout << "rep_for_dur=" << rep_for_dur << " dt.count()=" << dt.count() << " size=" << size << " bytes_per_sec=" << bytes_per_sec;

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

template<int64_t size>
void test_all_bw(double duration){
    auto OMP_NT = omp_thread_count();
    auto bw = test_bw<size>(duration);
    std::cout << "@BufferSize " << pretty_size(size) << " : " << pretty_size(bw) << "B/s  x " << OMP_NT << std::endl;
    
    constexpr int64_t nsize = next_size<size>();
    if (nsize)
        test_all_bw<nsize>(duration);
}
