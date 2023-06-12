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


struct MatmulTask {
    const char * name;
    bool transa;
    bool transb;
    bool constb;
    int64_t m;
    int64_t n;
    int64_t k;
    tensor2D<ov::bfloat16> A;
    tensor2D<ov::bfloat16> B;
    tensor2D<float> C;
    tensor2D<float> C0;
    float duration;
    int cache_MB; 
    std::vector<char> clr_cache_src;
    std::vector<char> clr_cache_dst;
    
    MatmulTask(const char * name,
                bool transa, bool transb, bool constb,
                int64_t m, int64_t n, int64_t k,
                float duration, int cache_MB):
        name(name),
        transa(transa), transb(transb), constb(constb),
        m(m), n(n), k(k),
        duration(duration), cache_MB(cache_MB) {

        // clear cache
        if (cache_MB > 0) {
            clr_cache_src.resize(cache_MB*1024*1024, 1);
            clr_cache_dst.resize(cache_MB*1024*1024, 2);
        }

        // prepare input
        if (transa)
            A.resize(k, m);
        else
            A.resize(m, k);
        
        if (transb)
            B.resize(n, k);
        else
            B.resize(k, n);
        
        A.fill_rnd();
        B.fill_rnd();

        // reference result
        C0.resize(m, n);
        C0=0;
        matmul(A, B, C0);

        // result
        C.resize(m, n);
        C = 0;
    }

    virtual void init() = 0;
    virtual void run() = 0;

    char clear_cache() {
        if (cache_MB > 0) {
            memcpy(&clr_cache_dst[0], &clr_cache_src[0], cache_MB*1024*1024);
            return clr_cache_dst[rand() % (cache_MB*1024*1024)];
        }
        return ' ';
    }

    bool subclass_initialized = false;

    py::dict benchmark() {
        py::dict ret;

        // initialize subclass(cannot call virtual function in constructor)
        if (!subclass_initialized) {
            init();
            subclass_initialized  = true;
        }

        int prevent_opt = 0;
        const int warm_up = 2;
        for(int i = 0; i < warm_up; i++) {
            prevent_opt += clear_cache();
            run();
        }

        // roughly measure latency
        auto t0 = __rdtsc();
        prevent_opt += clear_cache();
        run();
        auto t1 = __rdtsc();

        auto est_latency = tsc2second(t1 - t0);

        double avg_latency = 0;
        int64_t times = duration/est_latency;
        std::cout << "[" << name << "] start test times=" << times << std::flush;
        auto start = std::chrono::high_resolution_clock::now();
        for(int64_t i = 0; i < times; i++) {
            prevent_opt += clear_cache();
            auto t0 = __rdtsc();
            run();
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
        ret[pybind11::str("prevent_opt")] = prevent_opt;

        return ret;
    }
};
