#pragma once

#include <iostream>
#include <chrono>
#include <thread>
#include <pybind11/pybind11.h>
namespace py = pybind11;


//===============================================================================
#include "misc.hpp"
#include "tensor2D.hpp"
#include <cstdlib>

// _rdpmc
#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

inline uint64_t rdtsc_calibrate(int seconds = 1) {
    uint64_t start_ticks;
    start_ticks = __rdtsc();
    std::this_thread::sleep_for(std::chrono::seconds(seconds));
    return (__rdtsc() - start_ticks) / seconds;
}

inline uint64_t get_tsc_ticks_per_second() {
    static auto tsc_ticks_per_second = rdtsc_calibrate();
    return tsc_ticks_per_second;
}
inline double tsc2second(uint64_t diff) {
    return diff * 1.0/get_tsc_ticks_per_second();
}

inline uint64_t second2tsc(double sec) {
    return sec * get_tsc_ticks_per_second();
}

template<typename F>
py::dict benchmark(bool transB, bool constB, int M, int N, int K,
                   F kernel, float duration = 5.0, int cache_MB = 120) {
    tensor2D<ov::bfloat16> A(M, K);
    tensor2D<ov::bfloat16> B(K, N);
    tensor2D<float> C(M, N);
    tensor2D<float> C0(M, N);

    std::vector<char> clr_cache_src(cache_MB*1024*1024, 1);
    std::vector<char> clr_cache_dst(cache_MB*1024*1024, 2);
    
    py::dict ret;

    C0=0;
    matmul(A, B, C0);

    auto clear_cache = [&](){
        memcpy(&clr_cache_dst[0], &clr_cache_src[0], cache_MB*1024*1024);
        return clr_cache_dst[rand() % (cache_MB*1024*1024)];
    };

    const int warm_up = 2;
    for(int i = 0; i < warm_up; i++) {
        clear_cache();
        kernel(false, transB, constB, M, N, K,
                &A[0], A.padded_dim1,
                &B[0], B.padded_dim1,
                &C[0], C.padded_dim1);
    }

    // roughly measure latency
    auto t0 = __rdtsc();
    clear_cache();
    kernel(false, transB, constB, M, N, K,
            &A[0], A.padded_dim1,
            &B[0], B.padded_dim1,
            &C[0], C.padded_dim1);
    auto t1 = __rdtsc();

    auto est_latency = tsc2second(t1 - t0);

    double avg_latency = 0;
    int64_t times = duration/est_latency;
    std::cout << " start test times=" << times << std::flush;
    auto start = std::chrono::high_resolution_clock::now();
    for(int64_t i = 0; i < times; i++) {
        clear_cache();
        auto t0 = __rdtsc();
        kernel(false, transB, constB, M, N, K,
                &A[0], A.padded_dim1,
                &B[0], B.padded_dim1,
                &C[0], C.padded_dim1);
        auto t1 = __rdtsc();
        avg_latency += tsc2second(t1 - t0);
    }
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_latency = finish-start;
    std::cout << " finished in " << total_latency.count() << " seconds" << std::endl;

    avg_latency = avg_latency / times;

    ret[pybind11::str("correct")] = bool(C == C0);
    ret[pybind11::str("latency_ms")] = avg_latency * 1e3;
    ret[pybind11::str("times")] = times;
    ret[pybind11::str("duration")] = total_latency.count();

    return ret;
}
