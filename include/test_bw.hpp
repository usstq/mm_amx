#pragma once

#include <chrono>
#include <cstdlib>
#include <immintrin.h>
#include <iomanip>
#include <iostream>
#include <omp.h>
#include <sstream>

#include "jit.hpp"

class TestBWjit : public jit_generator {
public:
    TileConfig m_tile_cfg;
    TestBWjit() {
        create_kernel("TestBWjit");
        m_tile_cfg.reset(1, 0,
                         {
                             {16, 64}, // C:0
                             {16, 64}, // C:1
                             {16, 64}, // C:2
                             {16, 64}, // C:3
                             {16, 64}, // A0:4
                             {16, 64}, // A1:5
                             {16, 64}, // B0:6
                             {16, 64}, // B1:7
                         });
    }

    const TileConfig& tile_config() { return m_tile_cfg; }

    // to save push/pop: do not use `abi_save_gpr_regs`
    // uint8_t* base, int64_t size, uint64_t tsc_limit
    Xbyak::Reg64 reg_base = abi_param1;
    Xbyak::Reg64 reg_size = abi_param2;
    Xbyak::Reg64 reg_tsc_limit = abi_param3; // RDX
    Xbyak::Reg64 reg_tsc_0 = r8;
    Xbyak::Reg64 reg_repeats = r9;
    Xbyak::Reg64 reg_cnt = r10;
    Xbyak::Reg64 reg_tscL = r11;

    void generate() {
        Xbyak::Label loop_begin;
        Xbyak::Label loop_data;

        mov(reg_tscL, abi_param3); // RDX

        rdtsc(); // EDX:EAX
        sal(rdx, 32);
        or_(rax, rdx); // 64bit
        mov(reg_tsc_0, rax);

        xor_(reg_repeats, reg_repeats);

        align(64, false);
        L(loop_begin);

        xor_(reg_cnt, reg_cnt);
        L(loop_data);
        // for (int64_t i = 0; i < size; i += 64*4)
#if 0
            prefetcht0(ptr[reg_base + reg_cnt]);
            prefetcht0(ptr[reg_base + reg_cnt + 64*1]);
            prefetcht0(ptr[reg_base + reg_cnt + 64*2]);
            prefetcht0(ptr[reg_base + reg_cnt + 64*3]);
#else
        vmovaps(zmm0, ptr[reg_base + reg_cnt]);
        vmovaps(zmm1, ptr[reg_base + reg_cnt + 64]);
        vmovaps(zmm2, ptr[reg_base + reg_cnt + 64 * 2]);
        vmovaps(zmm3, ptr[reg_base + reg_cnt + 64 * 3]);
#endif
        add(reg_cnt, 64 * 4);
        cmp(reg_cnt, reg_size);
        jl(loop_data, T_NEAR);

        inc(reg_repeats);
        rdtsc(); // EDX:EAX
        sal(rdx, 32);
        or_(rax, rdx);       // 64bit
        sub(rax, reg_tsc_0); // tsc1 - tsc0
        cmp(rax, reg_tscL);  //
        jl(loop_begin, T_NEAR);

        mov(rax, reg_repeats);
        ret();
    }
};

// test BW
static int64_t doTest(uint8_t* base, int64_t size, uint64_t tsc_limit) {
#if 1
    static TestBWjit jit;
    return jit(base, size, tsc_limit);
#else
    auto sum0 = _mm256_setzero_ps();
    auto sum1 = _mm256_setzero_ps();
    auto sum2 = _mm256_setzero_ps();
    auto sum3 = _mm256_setzero_ps();
    int64_t actual_reps = 0;
    auto tsc0 = __rdtsc();
    int64_t r;
    for (r = 0;; r++) {
        uint8_t* src = base;
        for (int64_t i = 0; i < size; i += 64 * 4) {
            auto a0 = _mm256_load_ps(reinterpret_cast<const float*>(src));
            auto a1 = _mm256_load_ps(reinterpret_cast<const float*>(src + 64));
            auto a2 = _mm256_load_ps(reinterpret_cast<const float*>(src + 64 * 2));
            auto a3 = _mm256_load_ps(reinterpret_cast<const float*>(src + 64 * 3));
            src += 32 * 4;
            sum0 = _mm256_add_ps(sum0, a0);
            sum1 = _mm256_add_ps(sum1, a1);
            sum2 = _mm256_add_ps(sum2, a2);
            sum3 = _mm256_add_ps(sum3, a3);
        }
        if (__rdtsc() - tsc0 > tsc_limit) {
            actual_reps = r + 1;
            break;
        }
    }

    sum0 = _mm256_add_ps(sum0, sum1);
    sum2 = _mm256_add_ps(sum2, sum3);
    sum0 = _mm256_add_ps(sum0, sum2);
    auto s7 = _mm256_extract_epi32(_mm256_cvtps_epi32(sum0), 7);
    if (s7 == 0x12345678) {
        return actual_reps + 1;
    }
    return actual_reps;
#endif
};

double test_bw(double dur, int64_t size) {
    // allocate per-thread buffer
    auto tsc_second = get_tsc_ticks_per_second();

    uint8_t* data[128] = {0};
    int failed = 0;
#pragma omp parallel reduction(+ : failed)
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

    // warm-up cache
    int64_t actual_reps[128];
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        actual_reps[tid] = doTest(data[tid], size, tsc_second / 10);
    }

    // start profile
    auto t1 = std::chrono::steady_clock::now();
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        actual_reps[tid] = doTest(data[tid], size, dur * tsc_second);
    }
    auto t2 = std::chrono::steady_clock::now();

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        free(data[tid]);
    }

    int64_t total_reps = 0;
#pragma omp parallel reduction(+ : total_reps)
    {
        int tid = omp_get_thread_num();
        total_reps += actual_reps[tid];
    }

    std::chrono::duration<double> dt = t2 - t1;
    auto bytes_per_sec = total_reps / dt.count() * size;

    return bytes_per_sec;
}

template <int64_t size>
constexpr int64_t next_size() {
    int64_t nsize = size * 2;
    if (nsize > 1024 * 1024 * 1024)
        return 0;
    return nsize;
}

void test_all_bw(double duration) {
    auto test = [&](int64_t size) {
        auto OMP_NT = omp_thread_count();
        auto bw = test_bw(duration, size);
        std::cout << "@BufferSize " << pretty_size(size) << " : " << pretty_size(bw) << "B/s  x " << OMP_NT << std::endl;
    };

    test(15 * 1024);
    test(30 * 1024);

    for (int64_t KB = 1024; KB < 3072; KB += 256)
        test(KB * 1024); // L2

    test(13 * 1024 * 1024);        // 13MB L3
    test(56 * 1024 * 1024);        // 56MB L3
    test(128 * 1024 * 1024);       // 128MB L3 + DDR
    test(512 * 1024 * 1024);       // 512MB
    test(1024 * 1024 * 1024l);     // 1GB DDR
    test(2 * 1024 * 1024 * 1024l); // 1GB DDR
}
