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

struct BlockIterator {

    // - total rows M & cols N
    // - block size m & n
    // - blocking scheme in m/n/m/n order, from inner to outer
    // for example:
    //      Blocking(320, 160, 32, 32, {32*4, 160, 320})
    // will first go vertically for 4 32x32 blocks, and then horizontally for 160 elements, then vertially 320
    //
    int M;
    int N;
    int mb;
    int nb;
    BlockIterator(int _M, int _N, int _mb, int _nb,
             const std::vector<int> & blocks) : M(_M), N(_N), mb(_mb), nb(_nb) {
        //
    }

    //Prefix increment
    BlockIterator& operator++() {
        return *this;
    }
};

namespace executor_amx_bf16
{
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
struct FC {
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

    FC() {}

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


//https://stackoverflow.com/questions/29519222/how-to-transpose-a-16x16-matrix-using-simd-instructions
// vector multiply with matrix:
//  mAvB:  A(M, K) * B(K, 1) => C(M, 1)
//  vAmB:  A(1, K) * B(K, N) => C(1, N)
//
// in mAvB form, block of A (16x32) is transposed in register
// in unit of 2 packed bf16, and then vdpbf16ps was used
// to multiply with broadcasted B (2x1) and accumulate into C (16x1)
// 
// B is pre-broadcasted in unit of 2
// 
struct GemAvB {
    tensor2D<bfloat16> Bpadded;
    GemAvB() {
    }

    void operator()(tensor2D<bfloat16> & matA,
                    bfloat16 * vecB,
                    bfloat16 * vecC) {
        int M = matA.dims[0];
        int K = matA.dims[1];

        if (K % 32) {
            if (K > Bpadded.dims[1])
                Bpadded.resize(1, rndup(K, 32));
            auto newB = &Bpadded(0, 0);
            memset(newB, 0, Bpadded.stride);
            memcpy(newB, vecB, K * sizeof(bfloat16));
            vecB = newB;
        }

        auto nstride = matA.stride/sizeof(bfloat16);
        for(int m = 0; m < M; m += 16) {
            auto * pA = &matA(m, 0);
            auto * pBi32 = reinterpret_cast<int32_t*>(vecB);
            __m512 regC0 = _mm512_setzero();
            __m512 regC1 = _mm512_setzero();
            for(int k = 0; k < K; k += 32, pA += 32, pBi32 += 16) {
                // handle Ab: 16x32
                // transposed in register as 16x16x2
                //   r0: (a0,a1)(b0,b1)....
                //   r1: (a2,a3)(b2,b3)....
                //      ...
                //   rf: (a30,a31),(b30,b31)....
                // 
                __m512i t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, ta, tb, tc, td, te, tf;
                __m512i r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, ra, rb, rc, rd, re, rf;
                r0 = _mm512_loadu_epi32(pA + 0*nstride);
                r1 = _mm512_loadu_epi32(pA + 1*nstride);
                r2 = _mm512_loadu_epi32(pA + 2*nstride);
                r3 = _mm512_loadu_epi32(pA + 3*nstride);
                r4 = _mm512_loadu_epi32(pA + 4*nstride);
                r5 = _mm512_loadu_epi32(pA + 5*nstride);
                r6 = _mm512_loadu_epi32(pA + 6*nstride);
                r7 = _mm512_loadu_epi32(pA + 7*nstride);
                r8 = _mm512_loadu_epi32(pA + 8*nstride);
                r9 = _mm512_loadu_epi32(pA + 9*nstride);
                ra = _mm512_loadu_epi32(pA + 10*nstride);
                rb = _mm512_loadu_epi32(pA + 11*nstride);
                rc = _mm512_loadu_epi32(pA + 12*nstride);
                rd = _mm512_loadu_epi32(pA + 13*nstride);
                re = _mm512_loadu_epi32(pA + 14*nstride);
                rf = _mm512_loadu_epi32(pA + 15*nstride);

                t0 = _mm512_unpacklo_epi32(r0,r1); //   0  16   1  17   4  20   5  21   8  24   9  25  12  28  13  29 
                t1 = _mm512_unpackhi_epi32(r0,r1); //   2  18   3  19   6  22   7  23  10  26  11  27  14  30  15  31
                t2 = _mm512_unpacklo_epi32(r2,r3); //  32  48  33  49 ...
                t3 = _mm512_unpackhi_epi32(r2,r3); //  34  50  35  51 ...
                t4 = _mm512_unpacklo_epi32(r4,r5); //  64  80  65  81 ...  
                t5 = _mm512_unpackhi_epi32(r4,r5); //  66  82  67  83 ...
                t6 = _mm512_unpacklo_epi32(r6,r7); //  96 112  97 113 ...
                t7 = _mm512_unpackhi_epi32(r6,r7); //  98 114  99 115 ...
                t8 = _mm512_unpacklo_epi32(r8,r9); // 128 ...
                t9 = _mm512_unpackhi_epi32(r8,r9); // 130 ...
                ta = _mm512_unpacklo_epi32(ra,rb); // 160 ...
                tb = _mm512_unpackhi_epi32(ra,rb); // 162 ...
                tc = _mm512_unpacklo_epi32(rc,rd); // 196 ...
                td = _mm512_unpackhi_epi32(rc,rd); // 198 ...
                te = _mm512_unpacklo_epi32(re,rf); // 228 ...
                tf = _mm512_unpackhi_epi32(re,rf); // 230 ...

                r0 = _mm512_unpacklo_epi64(t0,t2); //   0  16  32  48 ...
                r1 = _mm512_unpackhi_epi64(t0,t2); //   1  17  33  49 ...
                r2 = _mm512_unpacklo_epi64(t1,t3); //   2  18  34  49 ...
                r3 = _mm512_unpackhi_epi64(t1,t3); //   3  19  35  51 ...
                r4 = _mm512_unpacklo_epi64(t4,t6); //  64  80  96 112 ...  
                r5 = _mm512_unpackhi_epi64(t4,t6); //  65  81  97 114 ...
                r6 = _mm512_unpacklo_epi64(t5,t7); //  66  82  98 113 ...
                r7 = _mm512_unpackhi_epi64(t5,t7); //  67  83  99 115 ...
                r8 = _mm512_unpacklo_epi64(t8,ta); // 128 144 160 176 ...  
                r9 = _mm512_unpackhi_epi64(t8,ta); // 129 145 161 178 ...
                ra = _mm512_unpacklo_epi64(t9,tb); // 130 146 162 177 ... 
                rb = _mm512_unpackhi_epi64(t9,tb); // 131 147 163 179 ...
                rc = _mm512_unpacklo_epi64(tc,te); // 192 208 228 240 ... 
                rd = _mm512_unpackhi_epi64(tc,te); // 193 209 229 241 ...
                re = _mm512_unpacklo_epi64(td,tf); // 194 210 230 242 ...
                rf = _mm512_unpackhi_epi64(td,tf); // 195 211 231 243 ...

                t0 = _mm512_shuffle_i32x4(r0, r4, 0x88); //   0  16  32  48   8  24  40  56  64  80  96  112 ...
                t1 = _mm512_shuffle_i32x4(r1, r5, 0x88); //   1  17  33  49 ...
                t2 = _mm512_shuffle_i32x4(r2, r6, 0x88); //   2  18  34  50 ...
                t3 = _mm512_shuffle_i32x4(r3, r7, 0x88); //   3  19  35  51 ...
                t4 = _mm512_shuffle_i32x4(r0, r4, 0xdd); //   4  20  36  52 ...
                t5 = _mm512_shuffle_i32x4(r1, r5, 0xdd); //   5  21  37  53 ...
                t6 = _mm512_shuffle_i32x4(r2, r6, 0xdd); //   6  22  38  54 ...
                t7 = _mm512_shuffle_i32x4(r3, r7, 0xdd); //   7  23  39  55 ...
                t8 = _mm512_shuffle_i32x4(r8, rc, 0x88); // 128 144 160 176 ...
                t9 = _mm512_shuffle_i32x4(r9, rd, 0x88); // 129 145 161 177 ...
                ta = _mm512_shuffle_i32x4(ra, re, 0x88); // 130 146 162 178 ...
                tb = _mm512_shuffle_i32x4(rb, rf, 0x88); // 131 147 163 179 ...
                tc = _mm512_shuffle_i32x4(r8, rc, 0xdd); // 132 148 164 180 ...
                td = _mm512_shuffle_i32x4(r9, rd, 0xdd); // 133 149 165 181 ...
                te = _mm512_shuffle_i32x4(ra, re, 0xdd); // 134 150 166 182 ...
                tf = _mm512_shuffle_i32x4(rb, rf, 0xdd); // 135 151 167 183 ...

                r0 = _mm512_shuffle_i32x4(t0, t8, 0x88); //   0  16  32  48  64  80  96 112 ... 240
                r1 = _mm512_shuffle_i32x4(t1, t9, 0x88); //   1  17  33  49  66  81  97 113 ... 241
                r2 = _mm512_shuffle_i32x4(t2, ta, 0x88); //   2  18  34  50  67  82  98 114 ... 242
                r3 = _mm512_shuffle_i32x4(t3, tb, 0x88); //   3  19  35  51  68  83  99 115 ... 243
                r4 = _mm512_shuffle_i32x4(t4, tc, 0x88); //   4 ...
                r5 = _mm512_shuffle_i32x4(t5, td, 0x88); //   5 ...
                r6 = _mm512_shuffle_i32x4(t6, te, 0x88); //   6 ...
                r7 = _mm512_shuffle_i32x4(t7, tf, 0x88); //   7 ...
                r8 = _mm512_shuffle_i32x4(t0, t8, 0xdd); //   8 ...
                r9 = _mm512_shuffle_i32x4(t1, t9, 0xdd); //   9 ...
                ra = _mm512_shuffle_i32x4(t2, ta, 0xdd); //  10 ...
                rb = _mm512_shuffle_i32x4(t3, tb, 0xdd); //  11 ...
                rc = _mm512_shuffle_i32x4(t4, tc, 0xdd); //  12 ...
                rd = _mm512_shuffle_i32x4(t5, td, 0xdd); //  13 ...
                re = _mm512_shuffle_i32x4(t6, te, 0xdd); //  14 ...
                rf = _mm512_shuffle_i32x4(t7, tf, 0xdd); //  15  31  47  63  79  96 111 127 ... 255

                // vdpbf16ps
                regC0 = _mm512_dpbf16_ps(regC0, r0, _mm512_set1_epi32(pBi32[0]));
                regC1 = _mm512_dpbf16_ps(regC1, r1, _mm512_set1_epi32(pBi32[1]));
                regC0 = _mm512_dpbf16_ps(regC0, r2, _mm512_set1_epi32(pBi32[2]));
                regC1 = _mm512_dpbf16_ps(regC1, r3, _mm512_set1_epi32(pBi32[3]));
                regC0 = _mm512_dpbf16_ps(regC0, r4, _mm512_set1_epi32(pBi32[4]));
                regC1 = _mm512_dpbf16_ps(regC1, r5, _mm512_set1_epi32(pBi32[5]));
                regC0 = _mm512_dpbf16_ps(regC0, r6, _mm512_set1_epi32(pBi32[6]));
                regC1 = _mm512_dpbf16_ps(regC1, r7, _mm512_set1_epi32(pBi32[7]));
                regC0 = _mm512_dpbf16_ps(regC0, r8, _mm512_set1_epi32(pBi32[8]));
                regC1 = _mm512_dpbf16_ps(regC1, r9, _mm512_set1_epi32(pBi32[9]));
                regC0 = _mm512_dpbf16_ps(regC0, ra, _mm512_set1_epi32(pBi32[10]));
                regC1 = _mm512_dpbf16_ps(regC1, rb, _mm512_set1_epi32(pBi32[11]));
                regC0 = _mm512_dpbf16_ps(regC0, rc, _mm512_set1_epi32(pBi32[12]));
                regC1 = _mm512_dpbf16_ps(regC1, rd, _mm512_set1_epi32(pBi32[13]));
                regC0 = _mm512_dpbf16_ps(regC0, re, _mm512_set1_epi32(pBi32[14]));
                regC1 = _mm512_dpbf16_ps(regC1, rf, _mm512_set1_epi32(pBi32[15]));
            }
            regC0 = _mm512_add_ps(regC0, regC1);
            auto regOut = _mm512_cvtne2ps_pbh(regC0, regC0); // only 16 bfloat16 results in lower 256bits 
            _mm256_storeu_si256(reinterpret_cast<__m256i_u *>(vecC + m), _mm512_extracti64x4_epi64(regOut, 0));
        }
    }
};

}

void amx_unit_test_acc(int M, int K, int N) {
    tensor2D<bfloat16> A(M, K);
    tensor2D<bfloat16> B(K, N);
    tensor2D<bfloat16> C(M, N);
    tensor2D<bfloat16> C0(M, N);
    executor_amx_bf16::FC fc;
    executor_amx_bf16::FC::KpackedB Bkpacked(B);
    executor_amx_bf16::FC::PP2bf16 pp;

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
    executor_amx_bf16::FC fc;
    executor_amx_bf16::FC::KpackedB Bkpacked(B);
    executor_amx_bf16::FC::PP2bf16 pp;

    tileconfig_t tfg(1, 0, 8, 16, 64);
    std::cout << "[" << M << "," << K << "," << N << "] ";
    timeit(times, [&](){
        fc(A, Bkpacked, C, pp);
    },
    double(M * N) * K * 2,
    AMXBf16PeakGops2PerCore * 1e9);
}


void amx_unit_test_gemAvB(int M, int K, int times = -1000) {
    int N = 1;
    tensor2D<bfloat16> A(M, K);
    tensor2D<bfloat16> B(1, K);
    tensor2D<bfloat16> B0(K, 1);
    tensor2D<bfloat16> C0(M, 1);
    tensor2D<bfloat16> C(1, M);
    executor_amx_bf16::GemAvB gemAvB;

    // same B, different layout
    std::cout << "[" << M << "," << K << "," << N << "] ";
    B0 = B;
    C0 = 0;
    matmul(A, B0, C0);
    auto C0Tr = C0.Tr();
    gemAvB(A, &B(0,0), &C(0,0));
    if (C0Tr == C) {
        std::cout << "Match!\n";
    } else {
        std::cout << C0Tr << std::endl;
        std::cout << C << std::endl;
        std::cout << "Mismatch!\n";
        return;
    }

    std::cout << "[" << M << "," << K << "," << N << "] ";
    timeit(times, [&](){
        gemAvB(A, &B(0,0), &C(0,0));
    },
    double(M * N) * K * 2,
    256 * 3e9);
}

int main(int argc, const char *argv[]) {
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

    amx_unit_test_gemAvB(901, 80); return 0;
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
