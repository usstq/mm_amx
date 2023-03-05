#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <vector>
#include <chrono>
#include <iostream>
#include <fstream>
#include <cassert>
#include <thread>
//#include "thread_pool.hpp"
#include "bf16.hpp"

// g++-11 ./test_conv.cpp -O2 -lpthread -march=native && ./a.out

// to use VNNI, we need higher version of compiler:
//    clang-9 ./test_conv.cpp -O2 -lpthread -march=native -lstdc++ && ./a.out

// to use AMX, we need intel compiler 
//   source  ~/intel/oneapi/setvars.sh
//   icx ./mm_amx_bf16.cpp -O2 -lpthread -march=native -lstdc++

// objdump -C -S ./a.out > a.asm

#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

#include "misc.hpp"

//===============================================================
// repeat C=A*B  (bf16_32x32)*(bf16_32x32)=>float_32x32 by R times
// 95% HW usage since data is fetching from L1D
int amx_unit_test_perf() {
    int M = 32;
    int K = 32;
    int N = 32;
    tensor2D<bfloat16> A(M, K);
    tensor2D<bfloat16> B(K, N);
    tensor2D<_Float64x> C(M, N);

    tileconfig_t tfg(1, 0, 8, 16, 64);
    tfg.load();

    int R = 512;
    timeit(-10,[&](){
        const int C00 = 0, C01 = 1, C10 = 2, C11 = 3, A0 = 4, A1 = 5, B0 = 6, B1 = 7;
        auto * pA0 = &A(0,0);
        auto * pA1 = &A(16,0);
        auto * pB0 = &B(0,0);
        auto * pB1 = &B(16,0);
        _tile_zero(C00);
        _tile_zero(C01);
        _tile_zero(C10);
        _tile_zero(C11);

        for(int k = 0; k < R; k++) {
            _tile_loadd(A0, pA0, A.stride);
            _tile_loadd(B0, pB0, B.stride);
            _tile_dpbf16ps(C00, A0, B0);
            _tile_loadd(A1, pA1, A.stride);
            _tile_dpbf16ps(C10, A1, B0);
            _tile_loadd(B1, pB1, B.stride);
            _tile_dpbf16ps(C01, A0, B1);
            _tile_dpbf16ps(C11, A1, B1);
        }
        //_tile_stored(C00, &C(0,0), C.stride);
        //_tile_stored(C01, &C(0,16), C.stride);
        //_tile_stored(C10, &C(16,0), C.stride);
        //_tile_stored(C11, &C(16,16), C.stride);  
    },
    (32 * 32) * 32 * R * 2,
    AMXBf16PeakGops2PerCore * 1e9
    );
    return 0;
}

// A bfloat16_16x32
// B bfloat16_32x16 (layout: 16x16x4)
// C    float_16x16
//
//         B0 B1
//         ...
//         B0 B1
//A0 : A0   C C
//A1 : A1   C C
void TileGemmKernel32x32(bfloat16 * pA, int strideA,     // 32xK
                bfloat16 * pBT, int strideB,    // Kx32 transposed(repacked) as 32xK 
                bfloat16 * pC, int strideC,     // 32x32
                int K) {
    const int C00 = 0, C01 = 1, C10 = 2, C11 = 3, A0 = 4, A1 = 5, B0 = 6, B1 = 7;

    auto * pA0 = reinterpret_cast<int8_t*>(pA);
    auto * pA1 = pA0 + 16*strideA;
    auto * pBT0 = reinterpret_cast<int8_t*>(pBT);
    auto * pBT1 = pBT0 + 16*strideB;
    strideA = 64;
    for (int k=0; k < K; k+=32) {
        _tile_loadd(A0, pA0 + k, strideA);
        _tile_loadd(B0, pBT0 + k, 64);
        _tile_dpbf16ps(C00, A0, B0);
        _tile_loadd(A1, pA1 + k, strideA);
        _tile_dpbf16ps(C10, A1, B0);
        _tile_loadd(B1, pBT1 + k, 64);
        _tile_dpbf16ps(C01, A0, B1);
        _tile_dpbf16ps(C11, A1, B1);
    }
#if 0
    _tile_stored(C00, pbuffC, 16*sizeof(float)*2);
    _tile_stored(C01, pbuffC + 16, 16*sizeof(float)*2);
    auto * psrc = pbuffC;
    auto * pdst = reinterpret_cast<int8_t*>(pC);
    for(int i = 0; i < 16; i ++) {
        auto b = _mm512_loadu_epi16(psrc);
        auto a = _mm512_loadu_epi16(psrc + 16);
        auto c = _mm512_cvtne2ps_pbh(a, b);
        _mm512_storeu_epi16(pdst, c);
        pdst += strideC;
        psrc += 32;
    }

    _tile_stored(C10, pbuffC, 16*sizeof(float)*2);
    _tile_stored(C11, pbuffC + 16, 16*sizeof(float)*2);
    psrc = pbuffC;
    for(int i = 0; i < 16; i ++) {
        auto b = _mm512_loadu_epi16(psrc);
        auto a = _mm512_loadu_epi16(psrc + 16);
        auto c = _mm512_cvtne2ps_pbh(a, b);
        _mm512_storeu_epi16(pdst, c);
        pC += strideC;
        psrc += 32;
    }
#endif
}

// HW usage reduce to 18% even all data are in L1D cache
int amx_unit_test_perf2() {
    int M = 32;
    int K = 24*32;
    int N = 32;
    tensor2D<bfloat16> A(M, K);  // 24KB
    tensor2D<bfloat16> BT(N, K); // 24KB; transposed & relayout
    tensor2D<bfloat16> C(M, N);

    tileconfig_t tfg(1, 0, 8, 16, 64);
    tfg.load();

    // only access a single tile in compact (64-bytes stride) form
    // this gives theoratical peak Gflops given function call overhead
    timeit(-10,[&](){
        const int C00 = 0, C01 = 1, C10 = 2, C11 = 3, A0 = 4, A1 = 5, B0 = 6, B1 = 7;
        auto * pA0 = &A(0,0);
        auto * pA1 = &A(16,0);
        auto * pB0 = &BT(0,0);
        auto * pB1 = &BT(16,0);
        _tile_zero(C00);
        _tile_zero(C01);
        _tile_zero(C10);
        _tile_zero(C11);
        for(int k = 0; k < K; k+=32) {
            _tile_loadd(A0, pA0, 64);
            _tile_loadd(B0, pB0, 64);
            _tile_dpbf16ps(C00, A0, B0);
            _tile_loadd(A1, pA1, 64);
            _tile_dpbf16ps(C10, A1, B0);
            _tile_loadd(B1, pB1, 64);
            _tile_dpbf16ps(C01, A0, B1);
            _tile_dpbf16ps(C11, A1, B1);
        }
    },
    (M * N) * K * 2,
    AMXBf16PeakGops2PerCore * 1e9
    );


    timeit(-10,[&](){
        const int C00 = 0, C01 = 1, C10 = 2, C11 = 3, A0 = 4, A1 = 5, B0 = 6, B1 = 7;
        _tile_zero(C00);
        _tile_zero(C01);
        _tile_zero(C10);
        _tile_zero(C11);
        for (int k=0; k < K; k+=32) {
            _tile_loadd(A0, &A(0, k), A.stride);
            _tile_loadd(B0, &BT(0, k), BT.stride);
            _tile_dpbf16ps(C00, A0, B0);
            _tile_loadd(A1, &A(16, k), A.stride);
            _tile_dpbf16ps(C10, A1, B0);
            _tile_loadd(B1, &BT(16, k), BT.stride);
            _tile_dpbf16ps(C01, A0, B1);
            _tile_dpbf16ps(C11, A1, B1);
        }
    },
    (M * N) * K * 2,
    AMXBf16PeakGops2PerCore * 1e9
    );

    std::cout << C(0,0) << std::endl;
    return 0;
}












// initialize AMX
static bool initAMX = initXTILE();

int main(int argc, const char *argv[]) {
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

    //return amx_unit_test_accuracy();
    amx_unit_test_perf();
    amx_unit_test_perf2();
    return;

    int M = 256;
    int K = 256;
    int N = 256;
    tensor2D<bfloat16> A(M, K);
    tensor2D<bfloat16> B(K, N);
    tensor2D<bfloat16> C0(M, N);
    tensor2D<bfloat16> C1(M, N);

    C0 = 0.0f;
    C1 = 0.0f;
    matmul(A, B, C0);
    //matmul(A, B, C1);


    // warm-up
    matmul_amx(A, B, C1);

    auto avg_latency = timeit(5, [&](){
        matmul_amx(A, B, C1);
    });
    std::cout << "average latency  : " << avg_latency*1e6 << " us" << std::endl;
    std::cout << "AMXBf16 PeakGops : " << AMXBf16PeakGopsPerCore << std::endl;
    std::cout << "Actual Gops      : " << M*N*K*2.0/avg_latency/(1e9) << std::endl;
    std::cout << "  AMX Usage      : " << (M*N*K*2.0/avg_latency)/(1e9*AMXBf16PeakGopsPerCore) << std::endl;

    if(C0 == C1) {
        std::cout << "Correct: C0=\n" << C0 << std::endl;
    } else {
        std::cout << " Wrong C0!=C1" << std::endl;
    }

    return 0;
}
