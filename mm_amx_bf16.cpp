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

//================================================================================
// initialize AMX
static bool initAMX = initXTILE();

//================================================================================
int amx_unit_test_perf() {
    int M = 32;
    // K=12*32, A+B fits L1D, gives 100% HW usage
    // K=80*32  A+B fits L2, gives 70% HW usage
    // K=512*32 A+B fits L2, gives 30% HW usage
    int K = 80*32;
    int N = 32;
    tensor2D<bfloat16> A(M, K);
    tensor2D<bfloat16> BT(N, K);
    tensor2D<bfloat16> C(M, N);
    tileconfig_t tfg(1, 0, 8, 16, 64);

    std::cout << "A & B in L1D (should give theoratical peak Gflops)\n\t";
    timeit(-100,[&](){
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

    std::cout << "TileGemmKernel32x32:\n\t";
    timeit(-100,[&](){
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

    std::cout << "B is transposed and blocked:\n\t";
    timeit(-100,[&](){
        const int C00 = 0, C01 = 1, C10 = 2, C11 = 3, A0 = 4, A1 = 5, B0 = 6, B1 = 7;
        _tile_zero(C00);
        _tile_zero(C01);
        _tile_zero(C10);
        _tile_zero(C11);
        auto *pB = &BT(0, 0);
        for (int k=0; k < K; k+=32) {
            _tile_stream_loadd(A0, &A(0, k), A.stride);
            _tile_stream_loadd(B0, pB, 64);
            _tile_dpbf16ps(C00, A0, B0);
            _tile_stream_loadd(A1, &A(16, k), A.stride);
            _tile_dpbf16ps(C10, A1, B0);
            _tile_stream_loadd(B1, pB + (16*32), 64);
            _tile_dpbf16ps(C01, A0, B1);
            _tile_dpbf16ps(C11, A1, B1);
            pB += 32*32;
        }
    },
    (M * N) * K * 2,
    AMXBf16PeakGops2PerCore * 1e9
    );

    std::cout << "B is transposed and blocked; A is blocked:\n\t";
    timeit(-100,[&](){
        const int C00 = 0, C01 = 1, C10 = 2, C11 = 3, A0 = 4, A1 = 5, B0 = 6, B1 = 7;
        _tile_zero(C00);
        _tile_zero(C01);
        _tile_zero(C10);
        _tile_zero(C11);
        auto *pA = &A(0, 0);
        auto *pB = &BT(0, 0);
        for (int k=0; k < K; k+=32) {
            _tile_loadd(B0, pB + k*(32), 64);
            _tile_loadd(A0, pA + k*(32), 64);
            _tile_dpbf16ps(C00, A0, B0);
            _tile_loadd(A1, pA + k*(32) + (16*32), 64);
            _tile_dpbf16ps(C10, A1, B0);
            _tile_loadd(B1, pB + k*(32) + (16*32), 64);
            _tile_dpbf16ps(C01, A0, B1);
            _tile_dpbf16ps(C11, A1, B1);
        }
    },
    (M * N) * K * 2,
    AMXBf16PeakGops2PerCore * 1e9
    );


    // now we go through real memory area, but assume the layout has been
    // optimized for performance.
    std::cout << "A & B are blocked and sequentially loaded:\n\t";
    tensor2D<bfloat16> AB(M*K + K*N, 32);
    timeit(-100,[&](){
        const int C00 = 0, C01 = 1, C10 = 2, C11 = 3, A0 = 4, A1 = 5, B0 = 6, B1 = 7;
        _tile_zero(C00);
        _tile_zero(C01);
        _tile_zero(C10);
        _tile_zero(C11);
        auto *ptr = &AB(0, 0);
        for (int k=0; k < K; k+=32) {
            _tile_stream_loadd(B0, ptr, 64);
            _tile_stream_loadd(A0, ptr + (16*32), 64);
            _tile_dpbf16ps(C00, A0, B0);
            _tile_stream_loadd(A1, ptr + (2*16*32), 64);
            _tile_dpbf16ps(C10, A1, B0);
            _tile_stream_loadd(B1, ptr + (3*16*32), 64);
            _tile_dpbf16ps(C01, A0, B1);
            _tile_dpbf16ps(C11, A1, B1);
            ptr += (4*16*32);
        }
    },
    (M * N) * K * 2,
    AMXBf16PeakGops2PerCore * 1e9
    );

    std::cout << C(0,0) << std::endl;
    return 0;
}

//================================================================================
// fc layer:  B is const and can be arranged into best sequential format
//------------------
// register blocking:
// A bfloat16_16x32
// B bfloat16_32x16 (layout: 16x16x4)
// C    float_16x16
//
//         B0 B1
//         ...
//         B0 B1
//A0 : A0   C C
//A1 : A1   C C
//------------------
// cache blocking:
//                Bb:     Kx32
//   Ab:  m0*32xK Cb:  m0*32x32
//
// (Ab + Bb) should fit in L2 cache
//    (m0*32xK*elesz + Kx32*elesz) < L2
//     m0 < L2/(32*K*elesz) - 1
//
struct FC_amx_bf16_v1 {
    tensor2D<float> buffC;
    static constexpr int tC00 = 0;
    static constexpr int tC01 = 1;
    static constexpr int tC10 = 2;
    static constexpr int tC11 = 3;
    static constexpr int tA0 = 4;
    static constexpr int tA1 = 5;
    static constexpr int tB0 = 6;
    static constexpr int tB1 = 7;

    FC_amx_bf16_v1() : buffC(16, 2*16) {
    }

    // KpackedB is B matrix in block of 32x32 arranged in column-major
    // each 32x32 block is composed of 2 horizontal neighboring tiles
    // of 32x16(further repacked as 16x16x2)
    // 
    //  +---+---+-----
    //  |B0 |B1 |
    //  |   |   |
    //  +---+---+
    //  |   |   | 
    // 
    struct KpackedB {
        std::shared_ptr<bfloat16> data;
        int K;
        int N;
        int Kblocks;
        int Nblocks;
        KpackedB(tensor2D<bfloat16> & matB) {
            K = matB.dims[0];
            N = matB.dims[1];
            Kblocks = (K + 31)/32;
            Nblocks = (N + 31)/32;
            int total_size = Kblocks * Nblocks * 32 * 32 * sizeof(bfloat16);
            data = std::shared_ptr<bfloat16>(
                        reinterpret_cast<bfloat16*>(aligned_alloc(64, rndup(total_size, 64))),
                        [](void * p){ free(p); });
            for (int k = 0; k < K; k++)
            for (int n = 0; n < N; n++)
                (*this)(k, n) = matB(k, n);
        }

        bfloat16 & operator()(int k, int n) {
            int kb = k/32;
            int nb = n/32;
            int block_offset = (nb*Kblocks + kb)*(32*32);
            int kr = k % 32;
            int nr = n % 32;
            int offset = block_offset;
            
            if (nr >= 16) {
                //B1
                offset += 32*16;
                nr -= 16;
            }
            // (kr,nr) is coordinate in 32x16 submatrix
            // after repack it becomes offset in 16x16x2
            offset += (kr/2)*32 + 2*nr + (kr&1);
            return data.get()[offset];
        }
    };

    // matB has been pre k-packed
    void operator()(tensor2D<bfloat16> & matA,
                    KpackedB & matB,
                    tensor2D<bfloat16> & matC) {
        int M = matC.dims[0];
        int N = matC.dims[1];
        int K = matA.dims[1];
        assert(K == matB.K);
        assert(N == matB.N);

        int elesz = sizeof(uint16_t);
        int L2 = 2048*1024; // 2MB
        int m0 = L2/(32*K*elesz) - 1;
        assert(m0 > 0);
        for (int m = 0; m < M; m += m0*32) { // loop m:
            for(int n = 0; n < N; n+=32) {   // loop n: reuse Ab in L2
                // (m0*32xK) * (Kx32) => m0*32x32
                for (int mi = 0; mi < m0; mi++) { // loop mi: reuse Bb in L2
                    auto curm = m + mi*32;
                    if (curm >= M)
                        break;
                    // load bias or zero
                    _tile_zero(tC00);
                    _tile_zero(tC01);
                    _tile_zero(tC10);
                    _tile_zero(tC11);
                    auto * pA0 = &matA(curm, 0);
                    auto * pA1 = &matA(curm + 16, 0);
                    auto * pB = &matB(0, n);
                    for (int k = 0; k < K; k += 32) {
                        _tile_loadd(tA0, pA0 + k, matA.stride);
                        _tile_loadd(tB0, pB, 64); pB += (16*32);
                        _tile_dpbf16ps(tC00, tA0, tB0);
                        _tile_loadd(tA1, pA1 + k, matA.stride);
                        _tile_dpbf16ps(tC10, tA1, tB0);
                        _tile_loadd(tB1, pB, 64); pB += (16*32);
                        _tile_dpbf16ps(tC01, tA0, tB1);
                        _tile_dpbf16ps(tC11, tA1, tB1);
                    }
                    auto pC = &matC(curm, n);
                    postProcess(pC, matC.stride);
                }
            }
        }
    }

    // post process two C tiles in 16 x 32
    void postProcess16x32(int8_t * pdst, int stride) {
        float * psrc = &buffC(0,0);
        for(int i = 0; i < 16; i ++) {
            auto b = _mm512_loadu_epi16(psrc);
            auto a = _mm512_loadu_epi16(psrc + 16);
            auto c = _mm512_cvtne2ps_pbh(a, b);
            _mm512_storeu_epi16(pdst, c);   // 32 bf16
            pdst += stride;
            psrc += 32;
        }
    }

    void postProcess(bfloat16 * pC, int stride) {
        _tile_stored(tC00, &buffC(0,0), buffC.stride);
        _tile_stored(tC01, &buffC(0,16), buffC.stride);
        postProcess16x32(reinterpret_cast<int8_t*>(pC), stride);
        _tile_stored(tC10, &buffC(0,0), buffC.stride);
        _tile_stored(tC11, &buffC(0,16), buffC.stride);
        postProcess16x32(reinterpret_cast<int8_t*>(pC) + 16*stride, stride);
    }
};


void amx_unit_test_acc() {
    int M = 32*22;  // 896
    int K = 80*32;  // 2560
    int N = 1024;

    tensor2D<bfloat16> A(M, K);
    tensor2D<bfloat16> B(K, N);
    tensor2D<bfloat16> C(M, N);
    tensor2D<bfloat16> C0(M, N);
    FC_amx_bf16_v1 fc;
    FC_amx_bf16_v1::KpackedB Bkpacked(B);

    tileconfig_t tfg(1, 0, 8, 16, 64);
    fc(A, Bkpacked, C);

    C0=0;
    matmul(A, B, C0);
    if (C0 == C) {
        std::cout << "Match!\n";
        std::cout << C << std::endl;
    } else {
        std::cout << "Mismatch!\n";
        std::cout << C0 << std::endl;
        std::cout << C << std::endl;
    }
}

void amx_unit_test_perf2() {
    int M = 32*28;  // 896
    int K = 80*32;  // 2560
    int N = 10240;

    tensor2D<bfloat16> A(M, K);
    tensor2D<bfloat16> B(K, N);
    tensor2D<bfloat16> C(M, N);
    tensor2D<bfloat16> C0(M, N);
    FC_amx_bf16_v1 fc;
    FC_amx_bf16_v1::KpackedB Bkpacked(B);

    tileconfig_t tfg(1, 0, 8, 16, 64);
    timeit(-1000, [&](){
        fc(A, Bkpacked, C);
    },
    double(M * N) * K * 2,
    AMXBf16PeakGops2PerCore * 1e9);
}

int main(int argc, const char *argv[]) {
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

    //amx_unit_test_perf();
    amx_unit_test_acc();
    amx_unit_test_perf2();
    return 0;

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
