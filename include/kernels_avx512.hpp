
#pragma once

#include "block_iter.hpp"
#include "tensor2D.hpp"
#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

namespace avx512 {
/*
 numactl -m 0 -C 12 ./benchdnn --ip --reset --mode=p --allow-enum-tags-only=0 --engine=cpu --dir=FWD_B \
 --cfg=f32 --stag=ab --wtag=AB16b64a --dtag=ab mb12ic4864oc256

numactl -m 0 -C 12 ./benchdnn --ip --reset --mode=p --allow-enum-tags-only=0 --engine=cpu --dir=FWD_B --cfg=f32 --stag=ab --wtag=AB16b64a --dtag=ab mb1ic160oc256
Output template: perf,%engine%,%impl%,%name%,%prb%,%Gops%,%-time%,%-Gflops%,%0time%,%0Gflops%
perf,cpu,brgemm:avx512_core,,--ip --allow-enum-tags-only=false --mode=P --stag=ab --wtag=AB16b64a --dtag=ab mb1ic160oc256,8.192e-05,0.0012207,67.1089,0.00159982,51.2057
tests:1 passed:1 skipped:0 mistrusted:0 unimplemented:0 invalid_arguments:0 failed:0 listed:0
total perf: min(ms):0.0012207 avg(ms):0.00159982
*/

namespace PP {
struct AddbiasRelu {
    float * bias;
    AddbiasRelu(float * bias) : bias(bias) {};

    __m512 bias0;
    __m512 bias1;
    __m512 bias2;
    __m512 bias3;
    __m512 zero;
    void prepare(int n) {
        // prepare 4x16 biases
        zero = _mm512_setzero_ps();
        bias0 = _mm512_loadu_ps(bias + n);
        bias1 = _mm512_loadu_ps(bias + n + 16);
        bias2 = _mm512_loadu_ps(bias + n + 16*2);
        bias3 = _mm512_loadu_ps(bias + n + 16*3);
    }
    void exec(__m512 & v0, __m512 & v1, __m512 & v2, __m512 & v3) {
        // bias
        v0 = _mm512_add_ps(v0, bias0);
        v1 = _mm512_add_ps(v1, bias1);
        v2 = _mm512_add_ps(v2, bias2);
        v3 = _mm512_add_ps(v3, bias3);

        // relu
        v0 = _mm512_max_ps (v0, zero);
        v1 = _mm512_max_ps (v1, zero);
        v2 = _mm512_max_ps (v2, zero);
        v3 = _mm512_max_ps (v3, zero);
    }
};

}


struct Matmul {
    BlockIterator blk_it;
    tensor2D<float> scratch;
    Matmul() {};

    template<typename P>
    void operator()(tensor2D<float> & matA,
                    tensor2D<float> & matB,
                    tensor2D<float> & matC,
                    P pp) {
        int M = matA.dims[0];
        int K = matA.dims[1];
        assert(K == matB.dims[0]);
        int N = matB.dims[1];

        // determine blocking scheme
        int elesz = sizeof(uint16_t);
        int L2 = 2048*1024; // 2MB
        int slice_size = 6*rndup(K, 6)*elesz;
        int mc = L2/slice_size - 1;

        // if 1 32xK slice cannot fit L2, use 1 slice at least
        if (mc == 0)
            mc = 1;

        auto dmax = std::numeric_limits<int>::max();
        BlockIterator::blkloop bloops[] = {{mc,6,0}, {dmax,0,64}, {dmax,mc*6,0}};
        blk_it.reset(bloops, 3, M, N);

        tensor2D<float> & Atails = scratch;
        int mtails = M % 6;
        if (mtails > 0) {
            Atails.resize(6, rndup(K, 16));
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

        auto strideB = matB.stride/sizeof(float);
        auto strideC = matC.stride/sizeof(float);
        do
        {
            int m = blk_it.m;
            int n = blk_it.n;
            int valid_m = std::min(M - m, 6);
            int valid_n = std::min(N - n, 64);
            auto * pA = &matA(m, 0);
            auto * pB = &matB(0, n);
            auto strideA = matA.stride/sizeof(float);
            if (valid_m < 6) {
                // use Atails buffer to prevent memory read segmentfault
                pA = &Atails(0, 0);
                strideA = Atails.stride/sizeof(float);
            }

            // do a 6x64 result, use 6x(4x16)=24 512bit register
            auto c00 = _mm512_setzero_ps(); auto c01 = _mm512_setzero_ps(); auto c02 = _mm512_setzero_ps(); auto c03 = _mm512_setzero_ps();
            auto c10 = _mm512_setzero_ps(); auto c11 = _mm512_setzero_ps(); auto c12 = _mm512_setzero_ps(); auto c13 = _mm512_setzero_ps();
            auto c20 = _mm512_setzero_ps(); auto c21 = _mm512_setzero_ps(); auto c22 = _mm512_setzero_ps(); auto c23 = _mm512_setzero_ps();
            auto c30 = _mm512_setzero_ps(); auto c31 = _mm512_setzero_ps(); auto c32 = _mm512_setzero_ps(); auto c33 = _mm512_setzero_ps();
            auto c40 = _mm512_setzero_ps(); auto c41 = _mm512_setzero_ps(); auto c42 = _mm512_setzero_ps(); auto c43 = _mm512_setzero_ps();
            auto c50 = _mm512_setzero_ps(); auto c51 = _mm512_setzero_ps(); auto c52 = _mm512_setzero_ps(); auto c53 = _mm512_setzero_ps();

            for(int k = 0; k < K; k++, pB += strideB) {
                auto b0 = _mm512_loadu_ps(pB);
                auto b1 = _mm512_loadu_ps(pB + 16);
                auto b2 = _mm512_loadu_ps(pB + 16*2);
                auto b3 = _mm512_loadu_ps(pB + 16*3);

                auto a0 = _mm512_set1_ps(pA[k + 0*strideA]);
                c00 = _mm512_fmadd_ps(a0, b0, c00);
                c01 = _mm512_fmadd_ps(a0, b1, c01);
                c02 = _mm512_fmadd_ps(a0, b2, c02);
                c03 = _mm512_fmadd_ps(a0, b3, c03);

                auto a1 = _mm512_set1_ps(pA[k + 1*strideA]);
                c10 = _mm512_fmadd_ps(a1, b0, c10);
                c11 = _mm512_fmadd_ps(a1, b1, c11);
                c12 = _mm512_fmadd_ps(a1, b2, c12);
                c13 = _mm512_fmadd_ps(a1, b3, c13);

                auto a2 = _mm512_set1_ps(pA[k + 2*strideA]);
                c20 = _mm512_fmadd_ps(a2, b0, c20);
                c21 = _mm512_fmadd_ps(a2, b1, c21);
                c22 = _mm512_fmadd_ps(a2, b2, c22);
                c23 = _mm512_fmadd_ps(a2, b3, c23);

                auto a3 = _mm512_set1_ps(pA[k + 3*strideA]);
                c30 = _mm512_fmadd_ps(a3, b0, c30);
                c31 = _mm512_fmadd_ps(a3, b1, c31);
                c32 = _mm512_fmadd_ps(a3, b2, c32);
                c33 = _mm512_fmadd_ps(a3, b3, c33);

                auto a4 = _mm512_set1_ps(pA[k + 4*strideA]);
                c40 = _mm512_fmadd_ps(a4, b0, c40);
                c41 = _mm512_fmadd_ps(a4, b1, c41);
                c42 = _mm512_fmadd_ps(a4, b2, c42);
                c43 = _mm512_fmadd_ps(a4, b3, c43);

                auto a5 = _mm512_set1_ps(pA[k + 5*strideA]);
                c50 = _mm512_fmadd_ps(a5, b0, c50);
                c51 = _mm512_fmadd_ps(a5, b1, c51);
                c52 = _mm512_fmadd_ps(a5, b2, c52);
                c53 = _mm512_fmadd_ps(a5, b3, c53);
            }

            //save 6x(4x16) to matC
            pp.prepare(n);

            auto * pC = &matC(m, n);
            __mmask16 mask = _cvtu32_mask16(0xFFFFFFFF >> (32-valid_n));
            auto lppkernel = [&](__m512 & v0,__m512 & v1,__m512 & v2,__m512 & v3){
                pp.exec(v0, v1, v2, v3);
                _mm512_mask_storeu_ps (pC       , mask, v0);
                _mm512_mask_storeu_ps (pC + 16  , mask, v1);
                _mm512_mask_storeu_ps (pC + 16*2, mask, v2);
                _mm512_mask_storeu_ps (pC + 16*3, mask, v3);
                pC += strideC;
                valid_m --;
            };

            if(valid_m) lppkernel(c00, c01, c02, c03);
            if(valid_m) lppkernel(c10, c11, c12, c13);
            if(valid_m) lppkernel(c20, c21, c22, c23);
            if(valid_m) lppkernel(c30, c31, c32, c33);
            if(valid_m) lppkernel(c40, c41, c42, c43);
            if(valid_m) lppkernel(c50, c51, c52, c53);

        }while(blk_it.next());
    }
};

}
