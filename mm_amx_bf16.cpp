#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <vector>
#include <chrono>
#include <iostream>
#include <fstream>
#include <cassert>
#include <cstring>
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
    static constexpr int tC00 = 0;
    static constexpr int tC01 = 1;
    static constexpr int tC10 = 2;
    static constexpr int tC11 = 3;
    static constexpr int tA0 = 4;
    static constexpr int tA1 = 5;
    static constexpr int tB0 = 6;
    static constexpr int tB1 = 7;

    // for processing tails
    tensor2D<bfloat16> Atails;

    FC_amx_bf16_v1() {
    }

    // post process kernels, tC00 ~ tC11
    struct PP2bf16 {
        tensor2D<float> buffC;
        PP2bf16() : buffC(16, 2*16) {}
        void postProcess16x32(int8_t * pdst, int stride, int valid_m, int valid_n) {
            float * psrc = &buffC(0,0);
            if (valid_m >= 16 && valid_n >= 32) {
                for(int i = 0; i < 16; i ++) {
                    auto b = _mm512_loadu_epi16(psrc);
                    auto a = _mm512_loadu_epi16(psrc + 16);
                    auto c = _mm512_cvtne2ps_pbh(a, b);
                    _mm512_storeu_epi16(pdst, c);   // 32 bf16
                    pdst += stride;
                    psrc += 32;
                }
            } else {
                __mmask32 k = _cvtu32_mask32(0xFFFFFFFF >> (32-valid_n));
                for(int i = 0; i < valid_m; i ++) {
                    auto b = _mm512_loadu_epi16(psrc);
                    auto a = _mm512_loadu_epi16(psrc + 16);
                    auto c = _mm512_cvtne2ps_pbh(a, b);
                    _mm512_mask_storeu_epi16(pdst, k, c);   // 32 bf16
                    pdst += stride;
                    psrc += 32;
                }
            }
        }
        void operator()(bfloat16 * pC, int stride, int valid_m, int valid_n) {
            _tile_stored(tC00, &buffC(0,0), buffC.stride);
            _tile_stored(tC01, &buffC(0,16), buffC.stride);
            postProcess16x32(reinterpret_cast<int8_t*>(pC), stride, valid_m, valid_n);

            if (valid_m > 16) {
                _tile_stored(tC10, &buffC(0,0), buffC.stride);
                _tile_stored(tC11, &buffC(0,16), buffC.stride);
                postProcess16x32(reinterpret_cast<int8_t*>(pC) + 16*stride, stride, valid_m-16, valid_n);
            }
        }
    };


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
            
            for (int k = 0; k < Kblocks*32; k++)
            for (int n = 0; n < Nblocks*32; n++) {
                if (k < K && n < N)
                    (*this)(k, n) = matB(k, n);
                else
                    (*this)(k, n) = 0; // padding zero
            }
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
    template<typename PP>
    void operator()(tensor2D<bfloat16> & matA,
                    KpackedB & matB,
                    tensor2D<bfloat16> & matC,
                    PP ppkernel) {
        int M = matC.dims[0];
        int N = matC.dims[1];
        int K = matA.dims[1];
        assert(K == matB.K);
        assert(N == matB.N);

        int elesz = sizeof(uint16_t);
        int L2 = 2048*1024; // 2MB
        int slice_size = 32*K*elesz;
        int mc = L2/slice_size - 1;
        assert(mc > 0);

        int mtails = M % 32;

        if (mtails > 0) {
            if (K > Atails.dims[1])
                Atails.resize(32, rndup(K, 32));
            // copy tails into Atails (in unit of 32x32)
            for (int m = 0; m < mtails; m++) {
                memcpy(&Atails(m, 0), &matA(M - mtails + m, 0), matA.stride);
                if (Atails.stride > matA.stride) {
                    memset(reinterpret_cast<int8_t*>(&Atails(m, 0)) + matA.stride,
                           0,
                           Atails.stride - matA.stride);
                }
            }
        }

        for (int m0 = 0; m0 < M; m0 += mc*32) { // loop m:
            int m1 = std::min(m0 + mc*32, M);
            for(int n = 0; n < N; n+=32) {   // loop n: reuse Ab in L2
                // (m0*32xK) * (Kx32) => m0*32x32
                int valid_n = std::min(N - n, 32);
                for (int m = m0; m < m1; m+=32) { // loop mi: reuse Bb in L2
                    int valid_m = std::min(M - m, 32);
                    auto * pA0 = &matA(m, 0);
                    auto * pA1 = &matA(m + 16, 0);
                    auto strideA = matA.stride;
                    auto * pB = &matB(0, n);
                    if (valid_m < 32) {
                        // use Atails buffer to prevent memory read segmentfault
                        pA0 = &Atails(0, 0);
                        pA1 = &Atails(16, 0);
                        strideA = Atails.stride;
                    }
                    _tile_zero(tC00);
                    _tile_zero(tC01);
                    _tile_zero(tC10);
                    _tile_zero(tC11);
                    for (int k = 0; k < K; k += 32) {
                        _tile_loadd(tA0, pA0 + k, strideA);
                        _tile_loadd(tB0, pB, 64); pB += (16*32);
                        _tile_dpbf16ps(tC00, tA0, tB0);
                        _tile_loadd(tA1, pA1 + k, strideA);
                        _tile_dpbf16ps(tC10, tA1, tB0);
                        _tile_loadd(tB1, pB, 64); pB += (16*32);
                        _tile_dpbf16ps(tC01, tA0, tB1);
                        _tile_dpbf16ps(tC11, tA1, tB1);
                    }
                    // post processing the accumulator tiles
                    //  - add bias
                    //  - do activations
                    //  - convert into bfloat16
                    //  - store into C matrix
                    (ppkernel)(&matC(m, n), matC.stride, valid_m, valid_n);
                }
            }
        }
    }
};

void amx_unit_test_acc(int M, int K, int N) {
    tensor2D<bfloat16> A(M, K);
    tensor2D<bfloat16> B(K, N);
    tensor2D<bfloat16> C(M, N);
    tensor2D<bfloat16> C0(M, N);
    FC_amx_bf16_v1 fc;
    FC_amx_bf16_v1::KpackedB Bkpacked(B);
    FC_amx_bf16_v1::PP2bf16 pp;

    tileconfig_t tfg(1, 0, 8, 16, 64);
    fc(A, Bkpacked, C, pp);

    C0=0;
    matmul(A, B, C0);
    std::cout << "[" << M << "," << K << "," << N << "] ";
    if (C0 == C) {
        std::cout << "Match!\n";
        //std::cout << C << std::endl;
    } else {
        std::cout << "Mismatch!\n";
        std::cout << C0 << std::endl;
        std::cout << C << std::endl;
    }
}

void amx_unit_test_perf2(int M, int K, int N, int times = -1000) {
    tensor2D<bfloat16> A(M, K);
    tensor2D<bfloat16> B(K, N);
    tensor2D<bfloat16> C(M, N);
    tensor2D<bfloat16> C0(M, N);
    FC_amx_bf16_v1 fc;
    FC_amx_bf16_v1::KpackedB Bkpacked(B);
    FC_amx_bf16_v1::PP2bf16 pp;

    tileconfig_t tfg(1, 0, 8, 16, 64);
    std::cout << "[" << M << "," << K << "," << N << "] ";
    timeit(times, [&](){
        fc(A, Bkpacked, C, pp);
    },
    double(M * N) * K * 2,
    AMXBf16PeakGops2PerCore * 1e9);
}

int main(int argc, const char *argv[]) {
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

    //amx_unit_test_perf();
    amx_unit_test_acc(32*22, 10*32, 256);
    amx_unit_test_acc(32*22 + 1, 10*32, 256 + 1);
    amx_unit_test_acc(32*22 + 16, 10*32, 256 + 17);
    amx_unit_test_acc(32*22 + 31, 10*32, 256 + 15);
    amx_unit_test_acc(32*22 + 31, 10*32 + 1, 256 + 15);
    amx_unit_test_acc(32*22 + 31, 10*32 + 17, 256 + 15);

    amx_unit_test_perf2(32*28, 32*80, 10240);
    amx_unit_test_perf2(32*28 + 1, 32*80, 10240);
    amx_unit_test_perf2(32*28, 32*80 + 1, 10240);
    amx_unit_test_perf2(32*28, 32*80, 10240 + 1);
    amx_unit_test_perf2(32*28 + 1, 32*80 + 1, 10240 + 1);
    amx_unit_test_perf2(32*28 + 32, 32*80 + 32, 10240 + 32);

    amx_unit_test_perf2(896, 256, 1024, 10000);

    return 0;
}
