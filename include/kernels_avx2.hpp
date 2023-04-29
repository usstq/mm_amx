
#pragma once

#include "block_iter.hpp"
#include "tensor2D.hpp"
#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

//https://stackoverflow.com/questions/68340319/differences-between-avx-and-avx2
//
//  The number of architectural YMM registers is 16 for 64-bit AVX2
//  thus 
// 
namespace avx2 {
/*
 numactl -m 0 -C 12 ./benchdnn --ip --reset --mode=p --allow-enum-tags-only=0 --engine=cpu --dir=FWD_B \
 --cfg=f32 --stag=ab --wtag=AB16b64a --dtag=ab mb12ic4864oc256

perf,cpu,x64:gemm:jit,,--mode=P --ip --allow-enum-tags-only=false --dir=FWD_I --stag=ab --wtag=ab --dtag=ab --attr-scratchpad=user mb128ic384oc51864,5.09844,9.29395,548.576,9.44089,540.038
tests:1 passed:1 skipped:0 mistrusted:0 unimplemented:0 invalid_arguments:0 failed:0 listed:0
total perf: min(ms):9.29395 avg(ms):9.44089

mb128ic384oc51864

*/

namespace PP {
    struct AddbiasRelu {
        float * bias;
        AddbiasRelu(float * bias) : bias(bias) {
        };

        __m256 bias0;
        __m256 bias1;
        __m256 zero;
        FORCE_INLINE void prepare(int n) {
            // prepare 2x8 biases
            zero = _mm256_setzero_ps();
            bias0 = _mm256_loadu_ps(bias + n);
            bias1 = _mm256_loadu_ps(bias + n + 8);
        }

        FORCE_INLINE void exec(__m256 & v0, __m256 & v1) {
            // bias
            v0 = _mm256_add_ps(v0, bias0);
            v1 = _mm256_add_ps(v1, bias1);
            // relu
            v0 = _mm256_max_ps(v0, zero);
            v1 = _mm256_max_ps(v1, zero);
        }
    };
}

template<int bN, class F>
FORCE_INLINE void loop2D_no_bM(int M, int N, F f) {
    for(int n=0; n<N; n += bN) {
        int valid_n = std::min(N - n, bN);
        f(0, n, M, valid_n);
    }
    return;
}

template<int bM, int bN, class F>
FORCE_INLINE void loop2D(int M, int N, int mc, F f) {
    for(int m0=0; m0<M; m0 += mc*bM) {
        for(int n=0; n<N; n += bN) {
            int valid_n = std::min(N - n, bN);
            int mcnt = std::min(mc, ((M - m0) + bM - 1)/bM);
            for(int m1=0; m1<mcnt; m1++) {
                int m = m0 + m1*bM;
                int valid_m = std::min(M - m, bM);
                f(m, n, valid_m, valid_n);
            }
        }
    }
}

struct Matmul {
    tensor2D<float> internalB;
    Matmul() {};

    // A: 6xK   B:Kx16   C: 6x16
    template<int valid_m, typename PP>
    void kernel_6x16(float * pA, int strideA,
                     float * pB,
                     float * pC, int strideC,
                     int K, int n, int valid_n,
                     PP pp) {
        __m256 c00, c01;
        __m256 c10, c11;
        __m256 c20, c21;
        __m256 c30, c31;
        __m256 c40, c41;
        __m256 c50, c51;
        __m256 b0, b1;
        c00 = c01 = _mm256_setzero_ps();
        if (valid_m > 1) c10 = c11 = _mm256_setzero_ps();
        if (valid_m > 2) c20 = c21 = _mm256_setzero_ps();
        if (valid_m > 3) c30 = c31 = _mm256_setzero_ps();
        if (valid_m > 4) c40 = c41 = _mm256_setzero_ps();
        if (valid_m > 5) c50 = c51 = _mm256_setzero_ps();

        for(int k = 0; k < K; k++, pB += 16) {
            b0 = _mm256_loadu_ps(pB);
            b1 = _mm256_loadu_ps(pB + 8);

            auto a0 = _mm256_set1_ps(pA[k + 0*strideA]);
            c00 = _mm256_fmadd_ps(a0, b0, c00);
            c01 = _mm256_fmadd_ps(a0, b1, c01);

            if (valid_m > 1) {
                auto a1 = _mm256_set1_ps(pA[k + 1*strideA]);
                c10 = _mm256_fmadd_ps(a1, b0, c10);
                c11 = _mm256_fmadd_ps(a1, b1, c11);
            }
            if (valid_m > 2) {
                auto a2 = _mm256_set1_ps(pA[k + 2*strideA]);
                c20 = _mm256_fmadd_ps(a2, b0, c20);
                c21 = _mm256_fmadd_ps(a2, b1, c21);
            }
            if (valid_m > 3) {
                auto a3 = _mm256_set1_ps(pA[k + 3*strideA]);
                c30 = _mm256_fmadd_ps(a3, b0, c30);
                c31 = _mm256_fmadd_ps(a3, b1, c31);
            }
            if (valid_m > 4) {
                auto a4 = _mm256_set1_ps(pA[k + 4*strideA]);
                c40 = _mm256_fmadd_ps(a4, b0, c40);
                c41 = _mm256_fmadd_ps(a4, b1, c41);
            }
            if (valid_m > 5) {
                auto a5 = _mm256_set1_ps(pA[k + 5*strideA]);
                c50 = _mm256_fmadd_ps(a5, b0, c50);
                c51 = _mm256_fmadd_ps(a5, b1, c51);
            }
        }

        pp.prepare(n);

        #define STORE(c0, c1) \
            pp.exec(c0, c1);  \
            _mm256_storeu_ps(pC, c0);  \
            _mm256_storeu_ps(pC + 8, c1);  \
            pC += strideC;

        STORE(c00, c01);
        if (valid_m > 1) { STORE(c10, c11); }
        if (valid_m > 2) { STORE(c20, c21); }
        if (valid_m > 3) { STORE(c30, c31); }
        if (valid_m > 4) { STORE(c40, c41); }
        if (valid_m > 5) { STORE(c50, c51); }
    };

    template<typename P>
    void operator()(tensor2D<float> & matA,
                    tensor2D<float> & matB,
                    tensor2D<float> & matC,
                    int n0, int n1,
                    P pp) {
        int M = matA.dims[0];
        int K = matA.dims[1];
        int N = n1 - n0;

        assert(K == matB.dims[0]);
        assert(N <= matB.dims[1]);
        assert(M == matC.dims[0]);
        assert(N <= matC.dims[1]);

        auto strideA = matA.stride/sizeof(float);
        auto strideB = matB.stride/sizeof(float);
        auto strideC = matC.stride/sizeof(float);

        if (internalB.capacity == 0) {
            // for handling N tails, internalB matrix is padded so read over tails is OK
            internalB.resize((N + 15)/16, K*16);
            loop2D_no_bM<16>(1, N, [&](int m, int n, int valid_m, int valid_n) {
                // pack
                auto * pBdst = &internalB(n/16, 0);
                auto * pBsrc = &matB(0, n0 + n);
                for(int k = 0; k < K; k++) {
                    auto b0 = _mm256_loadu_ps(pBsrc);
                    auto b1 = _mm256_loadu_ps(pBsrc + 8);
                    _mm256_storeu_ps(pBdst, b0);
                    _mm256_storeu_ps(pBdst + 8, b1);
                    pBsrc += strideB;
                    pBdst += 16;
                }
            });
        }

        // do a 6x16 result, use 6x(2x8)=12 256bits register
        auto lambda_kernel_6x16 = [&](int m, int n, int valid_m, int valid_n) {
            auto * pA = &matA(m, 0);
            auto * pB = &internalB(n/16, 0);
            auto * pC = &matC(m, n0 + n);
            switch (valid_m)
            {
                case 6: kernel_6x16<6>(pA, strideA, pB, pC, strideC, K, n0 + n, valid_n, pp); break;
                case 5: kernel_6x16<5>(pA, strideA, pB, pC, strideC, K, n0 + n, valid_n, pp); break;
                case 4: kernel_6x16<4>(pA, strideA, pB, pC, strideC, K, n0 + n, valid_n, pp); break;
                case 3: kernel_6x16<3>(pA, strideA, pB, pC, strideC, K, n0 + n, valid_n, pp); break;
                case 2: kernel_6x16<2>(pA, strideA, pB, pC, strideC, K, n0 + n, valid_n, pp); break;
                case 1: kernel_6x16<1>(pA, strideA, pB, pC, strideC, K, n0 + n, valid_n, pp); break;
            }
        };

        // determine blocking scheme
        int elesz = sizeof(uint16_t);
        int L2 = (256+32)*1024; // Coffee Lake 256K L2/core
        int slice_size = 6*rndup(K, 16)*elesz;
        int mc = L2/slice_size - 1;

        // if 1 32xK slice cannot fit L2, use 1 slice at least
        if (mc == 0)
            mc = 1;

        //std::cout << "mc=" << mc << std::endl;
        loop2D<6, 16>(M, N, mc, lambda_kernel_6x16);
    }
};

}
