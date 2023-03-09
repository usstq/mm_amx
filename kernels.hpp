#pragma once

#include "misc.hpp"

namespace executor_amx_bf16
{

// BlockIterator: kernels can use this to
//   - quickly go to some sequential index
//   - move to next location

struct BlockIterator {
    struct blkloop {
        int cnt;
        int sz_m;
        int sz_n;
    };

    int idx[16];  // index at each level of blocks
    std::vector<blkloop> bloops;

    int M;
    int N;

    int m;
    int n;
    int seq;
    bool reach_end;

    BlockIterator() = default;
    BlockIterator(const std::vector<blkloop> & _bloops){
        init(_bloops);
    }

    void init(const std::vector<blkloop> & _bloops) {
        bloops = _bloops;
        assert(bloops.size() <= 16);
    }

    void reset(int _M, int _N) {
        M = _M;
        N = _N;
        // reset coordinates to sequence index
        for(int i = 0; i < bloops.size(); i++)
            idx[i] = 0;
        seq = 0;
        m = 0;
        n = 0;
        reach_end = false;
    }
    // update coordinates
    bool next() {
        if (reach_end)
            return false;
        int carry_on = 1;
        for(int i = 0; i < bloops.size(); i++) {
            const auto & bl = bloops[i];
            if (idx[i] == (bl.cnt - 1)) {
                // carry-on on block boundary, no contribution to m/n
                m -= idx[i] * bl.sz_m;
                n -= idx[i] * bl.sz_n;
                idx[i] = 0;
            } else {
                // carry-on on matrix boundary
                if (m + bl.sz_m >= M || n + bl.sz_n >= N) {
                    m -= idx[i] * bl.sz_m;
                    n -= idx[i] * bl.sz_n;
                    idx[i] = 0;
                } else {
                    idx[i]++;
                    m += bl.sz_m;
                    n += bl.sz_n;
                    carry_on = 0;
                    break;
                }
            }
        }
        seq++;
        if (carry_on) {
            // after reach_end
            //  - seq has the number of blocks
            //  - idx are all zeros
            reach_end = true;
            return false;
        }
        return true;
    }
};

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

    BlockIterator bloop;

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



#if 0
// cache blocking 
struct Blocking {
    // reg_m x reg_n can be 1x4 or 2x2(it can be reduced to 1x2/2x1/1x1 at tails)
    int reg_m;  // register blocking scheme, how many tiles in M dimension
    int reg_n;  // register blocking scheme, how many tiles in N dimension

    struct block_loop {
        int axis;    // 0 means vertially along M dimension, 1 means horizontally along N dimension
        int step;   // in unit of basic output element (number of float or bfloat16)
    };
    std::vector<block_loop> bloops;
    // blocking is described in bottom-up order (inner blocking first)
    // upper level bsizes must be interger multiple of lower level. for example
    //   L0 [32x32] : 2x2 tile-register blocking
    //   L1 [0,320] : do 10 L0-blocks vertically
    //   L2 [1,max] : do ? L1-blocks horizontally until all actual columns is done
    //   L3 [0,max] : do ? L2-blocks vertially until all actual rows are done
    //
    // there is no actuall M,N numbers in blocking scheme, so block-scheme is
    // shape-agnostic and it can be hard coded once, executor can apply it to
    // all input shapes.
    //
    // when target C matrix has MxN smaller than L1 blocks, it's handled by executor
    // like this: when executor doing 10 L0-blocks vertially, it stops early when
    // it met M, and it uses tails logic to do last L0-block (by copy Atail)
    //

    // all L0 blocks can be sequentially indexed
    int total_blocks (int M, int N) {
        //
    }
};

// Matmul support B [KxN] matrix in few forms:
//    1. already transposed in KxN, only re-pack is needed
//    2. not transposed yet, it's NxK, need transpose and re-pack 
//
//     w=q*k': [S,80]*[80,S] => [S,S]   TBD
//     x=w*v : [S,S]*[S,80] => [S,80]   TBD
// standard implementation (not optimized):
//  a reorder routine that can prepare B-type tiles by:
//      1. transpose + re-pack
//      2. re-pack only
//  after which each B-type tile memory is placed sequentially
//  acording to blocking scheme's accessing order, it should be
//  done in single thread order, for examplem if whole matmul
//  is splitted among several cores, each core is responsible
//  for a few submatrixes result, then each core should do
//  this reorder on their own for their part of the B matrix.
//
//  if B matrix is constant, then it can be done only once
//  if not, it needs to be done on every execution.
//  
//  the memory of the prepared part of the B matrix will be kept
//  in the executor as a state(to save memory allocation overhead)
//  across calls
//
//  blocking scheme is determined outside, once it's done, executor's
//  job is just to execute it:
//   - using amx intrinsic.
//   - handling B matrix repack.
//   - handling tails.
// hard assumption:
//   - there is no blocking done on dimension K, and it only done on M,N
//   - two register/tile blockings on C matrix: 2x2 (32x32) or 1x4 (16x64)
//     determined by outside logic, executor has no strategy at all

// using Blocking described above, C matrix can be decomposed into a lot of
// blocks in different level of sizes, and it multi-threading scheme can tell
// executor to do only part of the job by giving range of indexes on some block level
// for example, for C matrix of 320x320, 10x10 L0 blocks are sequentially numbered
// as 0-99 (the order is determined by bloops), and split them among 5 threads
// each will only do 20 L0-blocks, in the order determined by Blocking struct.
//

void Matmul(tensor2D<bfloat16> & matA,
            tensor2D<bfloat16> & matB,
            tensor2D<bfloat16> & matC,
            tensor2D<bfloat16> & scratch, // scratch tensor for packing B
            bool transposeB,
            const Blocking & blk,
            int start, int end) {
    // check the range of B used by this kernel
    // (transpose) repack it into scratch

    // loop according to blk
}
#endif
} // namespace executor_amx_bf16
