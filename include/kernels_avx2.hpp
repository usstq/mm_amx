
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

namespace functional {
    // https://stackoverflow.com/questions/25622745/transpose-an-8x8-float-using-avx-avx2
    inline void transpose8_ps(__m256 &row0, __m256 &row1, __m256 &row2, __m256 &row3, __m256 &row4, __m256 &row5, __m256 &row6, __m256 &row7) {
        __m256 __t0, __t1, __t2, __t3, __t4, __t5, __t6, __t7;
        __m256 __tt0, __tt1, __tt2, __tt3, __tt4, __tt5, __tt6, __tt7;
        __t0 = _mm256_unpacklo_ps(row0, row1);
        __t1 = _mm256_unpackhi_ps(row0, row1);
        __t2 = _mm256_unpacklo_ps(row2, row3);
        __t3 = _mm256_unpackhi_ps(row2, row3);
        __t4 = _mm256_unpacklo_ps(row4, row5);
        __t5 = _mm256_unpackhi_ps(row4, row5);
        __t6 = _mm256_unpacklo_ps(row6, row7);
        __t7 = _mm256_unpackhi_ps(row6, row7);
        __tt0 = _mm256_shuffle_ps(__t0,__t2,_MM_SHUFFLE(1,0,1,0));
        __tt1 = _mm256_shuffle_ps(__t0,__t2,_MM_SHUFFLE(3,2,3,2));
        __tt2 = _mm256_shuffle_ps(__t1,__t3,_MM_SHUFFLE(1,0,1,0));
        __tt3 = _mm256_shuffle_ps(__t1,__t3,_MM_SHUFFLE(3,2,3,2));
        __tt4 = _mm256_shuffle_ps(__t4,__t6,_MM_SHUFFLE(1,0,1,0));
        __tt5 = _mm256_shuffle_ps(__t4,__t6,_MM_SHUFFLE(3,2,3,2));
        __tt6 = _mm256_shuffle_ps(__t5,__t7,_MM_SHUFFLE(1,0,1,0));
        __tt7 = _mm256_shuffle_ps(__t5,__t7,_MM_SHUFFLE(3,2,3,2));
        row0 = _mm256_permute2f128_ps(__tt0, __tt4, 0x20);
        row1 = _mm256_permute2f128_ps(__tt1, __tt5, 0x20);
        row2 = _mm256_permute2f128_ps(__tt2, __tt6, 0x20);
        row3 = _mm256_permute2f128_ps(__tt3, __tt7, 0x20);
        row4 = _mm256_permute2f128_ps(__tt0, __tt4, 0x31);
        row5 = _mm256_permute2f128_ps(__tt1, __tt5, 0x31);
        row6 = _mm256_permute2f128_ps(__tt2, __tt6, 0x31);
        row7 = _mm256_permute2f128_ps(__tt3, __tt7, 0x31);
    }

    inline void transpose_16xK_ps(float * pBdst, float *pBsrc, int strideB, int K) {
        for(int k = 0; k < K; k+=8, pBsrc+=8) {
            {
                auto b0 = _mm256_loadu_ps(pBsrc);
                auto b1 = _mm256_loadu_ps(pBsrc + strideB);
                auto b2 = _mm256_loadu_ps(pBsrc + strideB*2);
                auto b3 = _mm256_loadu_ps(pBsrc + strideB*3);
                auto b4 = _mm256_loadu_ps(pBsrc + strideB*4);
                auto b5 = _mm256_loadu_ps(pBsrc + strideB*5);
                auto b6 = _mm256_loadu_ps(pBsrc + strideB*6);
                auto b7 = _mm256_loadu_ps(pBsrc + strideB*7);
                functional::transpose8_ps(b0, b1, b2, b3, b4, b5, b6, b7);
                _mm256_storeu_ps(pBdst, b0);
                _mm256_storeu_ps(pBdst + 8*2, b1);
                _mm256_storeu_ps(pBdst + 8*4, b2);
                _mm256_storeu_ps(pBdst + 8*6, b3);
                _mm256_storeu_ps(pBdst + 8*8, b4);
                _mm256_storeu_ps(pBdst + 8*10, b5);
                _mm256_storeu_ps(pBdst + 8*12, b6);
                _mm256_storeu_ps(pBdst + 8*14, b7);
            }
            {
                auto b0 = _mm256_loadu_ps(pBsrc + strideB*8);
                auto b1 = _mm256_loadu_ps(pBsrc + strideB*9);
                auto b2 = _mm256_loadu_ps(pBsrc + strideB*10);
                auto b3 = _mm256_loadu_ps(pBsrc + strideB*11);
                auto b4 = _mm256_loadu_ps(pBsrc + strideB*12);
                auto b5 = _mm256_loadu_ps(pBsrc + strideB*13);
                auto b6 = _mm256_loadu_ps(pBsrc + strideB*14);
                auto b7 = _mm256_loadu_ps(pBsrc + strideB*15);
                functional::transpose8_ps(b0, b1, b2, b3, b4, b5, b6, b7);
                _mm256_storeu_ps(pBdst + 8, b0);
                _mm256_storeu_ps(pBdst + 8*3, b1);
                _mm256_storeu_ps(pBdst + 8*5, b2);
                _mm256_storeu_ps(pBdst + 8*7, b3);
                _mm256_storeu_ps(pBdst + 8*9, b4);
                _mm256_storeu_ps(pBdst + 8*11, b5);
                _mm256_storeu_ps(pBdst + 8*13, b6);
                _mm256_storeu_ps(pBdst + 8*15, b7);
            }
            pBdst += 8*16;
        }
    }
}
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

/**************************************
 * loop order: column by column, in case
 * where B needs dynamic transpose, this
 * loop order can keep B slice hot in cache
 */
template<int bM, int bN, class F>
FORCE_INLINE void loop2D_ColumnMajor(int M, int N, F f) {
    for(int n=0; n<N; n += bN) {
        int valid_n = std::min(N - n, bN);
        for(int m=0; m<M; m += bM) {
            int valid_m = std::min(M - m, bM);
            f(m, n, valid_m, valid_n);
        }
    }
}

struct Matmul {
    tensor2D<float> internalB;

    bool constB;
    bool transposeB;
    Matmul(bool constB = false, bool transposeB = false) : constB(constB), transposeB(transposeB) {};

    // A: 6xK   B:Kx16 (no-transpose)  C: 6x16
    template<int valid_m, int valid_n, typename PP>
    void kernel_6x16(float * pA, int strideA,
                     float * pB, int strideB,
                     float * pC, int strideC,
                     int K, int n,
                     PP pp) {
        static_assert(valid_m > 0 && valid_m < 7);
        static_assert(valid_n == 8 || valid_n == 16);
        __m256 c00, c01;
        __m256 c10, c11;
        __m256 c20, c21;
        __m256 c30, c31;
        __m256 c40, c41;
        __m256 c50, c51;
        __m256 b0, b1;

        #define SETZERO(c0, c1) \
            c0 = _mm256_setzero_ps();  \
            if (valid_n == 16) c1 = _mm256_setzero_ps();

        SETZERO(c00, c01);
        if (valid_m > 1) { SETZERO(c10, c11); }
        if (valid_m > 2) { SETZERO(c20, c21); }
        if (valid_m > 3) { SETZERO(c30, c31); }
        if (valid_m > 4) { SETZERO(c40, c41); }
        if (valid_m > 5) { SETZERO(c50, c51); }

        #define FMADD(a, b0, b1, c0, c1) \
            c0 = _mm256_fmadd_ps(a, b0, c0); \
            if (valid_n == 16) c1 = _mm256_fmadd_ps(a, b1, c1);

        for(int k = 0; k < K; k++, pB += strideB) {
            b0 = _mm256_loadu_ps(pB);
            if (valid_n == 16) b1 = _mm256_loadu_ps(pB + 8);

            if (valid_m > 0) {
                auto a0 = _mm256_set1_ps(pA[k + 0*strideA]);
                FMADD(a0, b0, b1, c00, c01);
            }
            if (valid_m > 1) {
                auto a1 = _mm256_set1_ps(pA[k + 1*strideA]);
                FMADD(a1, b0, b1, c10, c11);
            }
            if (valid_m > 2) {
                auto a2 = _mm256_set1_ps(pA[k + 2*strideA]);
                FMADD(a2, b0, b1, c20, c21);
            }
            if (valid_m > 3) {
                auto a3 = _mm256_set1_ps(pA[k + 3*strideA]);
                FMADD(a3, b0, b1, c30, c31);
            }
            if (valid_m > 4) {
                auto a4 = _mm256_set1_ps(pA[k + 4*strideA]);
                FMADD(a4, b0, b1, c40, c41);
            }
            if (valid_m > 5) {
                auto a5 = _mm256_set1_ps(pA[k + 5*strideA]);
                FMADD(a5, b0, b1, c50, c51);
            }
        }

        pp.prepare(n);

        #define STORE(c0, c1) \
            pp.exec(c0, c1);  \
            _mm256_storeu_ps(pC, c0);  \
           if (valid_n == 16) _mm256_storeu_ps(pC + 8, c1);  \
            pC += strideC;

        STORE(c00, c01);
        if (valid_m > 1) { STORE(c10, c11); }
        if (valid_m > 2) { STORE(c20, c21); }
        if (valid_m > 3) { STORE(c30, c31); }
        if (valid_m > 4) { STORE(c40, c41); }
        if (valid_m > 5) { STORE(c50, c51); }
    };


    void reorderB(tensor2D<float> & matB, int n0, int n1) {
        // transposeB : B_NxK
        //
        int K = matB.dims[transposeB ? 1 : 0];
        int N = n1 - n0;
        auto strideB = matB.stride/sizeof(float);
        if (!transposeB) {
            // N tails in internalB matrix is aligned to right border of B
            internalB.resize((N + 15)/16, K*16);
            loop2D_no_bM<16>(1, N, [&](int m, int n, int valid_m, int valid_n) {
                // align to right border of B at N tails
                int nsrc = (valid_n <= 8) ? (n1 - 8) : ((valid_n < 16) ? (n1 - 16) : n0 + n);
                auto * pBdst = &internalB(n/16, 0);
                auto * pBsrc = &matB(0, nsrc);
                for(int k = 0; k < K; k++) {
                    auto b0 = _mm256_loadu_ps(pBsrc);
                    auto b1 = _mm256_loadu_ps(pBsrc + 8);
                    _mm256_storeu_ps(pBdst, b0);
                    _mm256_storeu_ps(pBdst + 8, b1);
                    pBsrc += strideB;
                    pBdst += 16;
                }
            });
        } else {
            // transpose B(NxK) is costly for non-constB:
            //   - it consumes lots of instructions
            //   - it takes more than 8 HW registers (possibly 9 is enough),
            //     so no more register for storing C
            // thus we only want to do it once, due to limited register resource,
            // we cannot archieve that with on-the-fly transpose. so we transpose it
            // into a temp buffer at once
            internalB.resize((N + 15)/16, rndup(K, 8) *16);
            loop2D_no_bM<16>(1, N, [&](int m, int n, int valid_m, int valid_n) {
                // align to right border of B at N tails
                int nsrc = (valid_n <= 8) ? (n1 - 8) : ((valid_n < 16) ? (n1 - 16) : n0 + n);
                auto * pBdst = &internalB(n/16, 0);
                auto * pBsrc = &matB(nsrc, 0);
                functional::transpose_16xK_ps(pBdst, pBsrc, strideB, K);
            });
        }
    }

    bool use_internalB = false;
    bool use_dynTransB = false;
    template<typename P>
    void operator()(tensor2D<float> & matA,
                    tensor2D<float> & matB,
                    tensor2D<float> & matC,
                    int n0, int n1,
                    P pp) {
        int M = matA.dims[0];
        int K = matA.dims[1];
        int N = n1 - n0;
        
        assert(K == matB.dims[transposeB ? 1:0]);
        assert(N <= matB.dims[transposeB ? 0:1]);
        assert(M == matC.dims[0]);
        assert(N <= matC.dims[1]);

        auto strideA = matA.stride/sizeof(float);
        int strideB;
        auto strideC = matC.stride/sizeof(float);

        use_dynTransB = (!constB && transposeB);
        if (use_dynTransB) {
            // dynamically transpose 16 rows of matB into internalB
            internalB.resize(1, rndup(K, 8) * 16);
            use_internalB = true;
        } else if (constB && internalB.capacity == 0) {
            reorderB(matB, n0, n1);
            use_internalB = true;
        } else if (!constB && !transposeB) {
            // use B matrix directly w/o copy it every time, because
            // read B matrix is inevitable, direct access can avoid writting
            // internalB again.
            use_internalB = false;
        }

        if (use_internalB)
            strideB = 16;
        else
            strideB = matB.stride/sizeof(float);

        // do a 6x16 result, use 6x(2x8)=12 256bits register
        auto lambda_kernel_6x16 = [&](int m, int n, int valid_m, int valid_n) {
            auto * pA = &matA(m, 0);
            int ndst = (valid_n <= 8) ? (n1 - 8) : ((valid_n < 16) ? (n1 - 16) : n0 + n);
            auto * pB = use_dynTransB ? &internalB[0] : (use_internalB ? &internalB(n >> 4, 0) : &matB(0, ndst));
            auto * pC = &matC(m, ndst);
            if (use_dynTransB && m == 0) {
                // dynamically transpose 16 rows of matB into internalB
                functional::transpose_16xK_ps(&internalB[0], &matB(ndst, 0), matB.stride/sizeof(float), K);
            }
            if (valid_n <= 8) {
                switch (valid_m)
                {
                    case 6: kernel_6x16<6, 8>(pA, strideA, pB, strideB, pC, strideC, K, ndst, pp); break;
                    case 5: kernel_6x16<5, 8>(pA, strideA, pB, strideB, pC, strideC, K, ndst, pp); break;
                    case 4: kernel_6x16<4, 8>(pA, strideA, pB, strideB, pC, strideC, K, ndst, pp); break;
                    case 3: kernel_6x16<3, 8>(pA, strideA, pB, strideB, pC, strideC, K, ndst, pp); break;
                    case 2: kernel_6x16<2, 8>(pA, strideA, pB, strideB, pC, strideC, K, ndst, pp); break;
                    case 1: kernel_6x16<1, 8>(pA, strideA, pB, strideB, pC, strideC, K, ndst, pp); break;
                }
            } else {
                switch (valid_m)
                {
                    case 6: kernel_6x16<6, 16>(pA, strideA, pB, strideB, pC, strideC, K, ndst, pp); break;
                    case 5: kernel_6x16<5, 16>(pA, strideA, pB, strideB, pC, strideC, K, ndst, pp); break;
                    case 4: kernel_6x16<4, 16>(pA, strideA, pB, strideB, pC, strideC, K, ndst, pp); break;
                    case 3: kernel_6x16<3, 16>(pA, strideA, pB, strideB, pC, strideC, K, ndst, pp); break;
                    case 2: kernel_6x16<2, 16>(pA, strideA, pB, strideB, pC, strideC, K, ndst, pp); break;
                    case 1: kernel_6x16<1, 16>(pA, strideA, pB, strideB, pC, strideC, K, ndst, pp); break;
                }
            }
        };

        /*
        // determine blocking scheme
        int elesz = sizeof(uint16_t);
        int L2 = (256+32)*1024; // Coffee Lake 256K L2/core
        int slice_size = 6 * K *elesz;
        int mc = L2/slice_size - 1;

        // if 1 32xK slice cannot fit L2, use 1 slice at least
        if (mc == 0)
            mc = 1;

        //std::cout << "mc=" << mc << std::endl;
        //loop2D<6, 16>(M, N, mc, lambda_kernel_6x16);
        */
        loop2D_ColumnMajor<6, 16>(M, N, lambda_kernel_6x16);
    }
};

}
