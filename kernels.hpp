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
    const blkloop *bloops;
    int num_bloops;

    int M;
    int N;

    int m;
    int n;
    int seq;
    bool reach_end;

    BlockIterator() = default;

    void reset(const blkloop * _bloops, int _num_bloops, int _M, int _N) {
        assert(_num_bloops <= 16);
        bloops = _bloops;
        num_bloops = _num_bloops;
        M = _M;
        N = _N;
        // reset coordinates to sequence index
        for(int i = 0; i < num_bloops; i++)
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
        for(int i = 0; i < num_bloops; i++) {
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
// need to support:
//   - optimized single core performance
//   - support transpose
//   - blocking scheme
//
struct KpackedB {
    std::shared_ptr<bfloat16> data;
    int64_t capacity;
    int K;
    int N;
    int Kblocks;
    int Nblocks;
    KpackedB() {
        capacity = 0;
        K = N = 0;
        Kblocks = Nblocks = 0;
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

    void resize(int _K, int _N) {
        K = _K;
        N = _N;
        Kblocks = rndup(K, 32)/32;
        Nblocks = rndup(N, 32)/32;
        int64_t need_capacity = Kblocks * Nblocks * 32 * 32 * sizeof(bfloat16);
        need_capacity = rndup(need_capacity, 64);
        if (capacity < need_capacity) {
            capacity = need_capacity;
            data = std::shared_ptr<bfloat16>(
                        reinterpret_cast<bfloat16*>(aligned_alloc(64, capacity)),
                        [](void * p){ free(p); });
        }
    }

    void operator=(const float & v) {
        for(int k = 0; k<capacity/sizeof(bfloat16); k++)
            data.get()[k] = v;
    }
};

constexpr int tC00 = 0;
constexpr int tC01 = 1;
constexpr int tC10 = 2;
constexpr int tC11 = 3;
constexpr int tA0 = 4;
constexpr int tA1 = 5;
constexpr int tB0 = 6;
constexpr int tB1 = 7;

namespace functional {

    void transpose_m512i_16x16(__m512i &r0, __m512i &r1, __m512i &r2, __m512i &r3,
                               __m512i &r4, __m512i &r5, __m512i &r6, __m512i &r7,
                               __m512i &r8, __m512i &r9, __m512i &ra, __m512i &rb,
                               __m512i &rc, __m512i &rd, __m512i &re, __m512i &rf) {
        __m512i t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, ta, tb, tc, td, te, tf;

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
    }

    void transpose_epi32_16x16(void * _dst, const void * src, int stride) {
        auto * dst = reinterpret_cast<uint32_t*>(_dst);
        __m512i r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, ra, rb, rc, rd, re, rf;
        auto * pA = reinterpret_cast<const uint8_t*>(src);
        r0 = _mm512_loadu_epi32(pA);
        r1 = _mm512_loadu_epi32(pA + stride);
        r2 = _mm512_loadu_epi32(pA + 2*stride);
        r3 = _mm512_loadu_epi32(pA + 3*stride);
        r4 = _mm512_loadu_epi32(pA + 4*stride);
        r5 = _mm512_loadu_epi32(pA + 5*stride);
        r6 = _mm512_loadu_epi32(pA + 6*stride);
        r7 = _mm512_loadu_epi32(pA + 7*stride);
        r8 = _mm512_loadu_epi32(pA + 8*stride);
        r9 = _mm512_loadu_epi32(pA + 9*stride);
        ra = _mm512_loadu_epi32(pA + 10*stride);
        rb = _mm512_loadu_epi32(pA + 11*stride);
        rc = _mm512_loadu_epi32(pA + 12*stride);
        rd = _mm512_loadu_epi32(pA + 13*stride);
        re = _mm512_loadu_epi32(pA + 14*stride);
        rf = _mm512_loadu_epi32(pA + 15*stride);

        transpose_m512i_16x16(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, ra, rb, rc, rd, re, rf);

        _mm512_storeu_epi32(dst, r0);
        _mm512_storeu_epi32(dst + 16, r1);
        _mm512_storeu_epi32(dst + 2*16, r2);
        _mm512_storeu_epi32(dst + 3*16, r3);
        _mm512_storeu_epi32(dst + 4*16, r4);
        _mm512_storeu_epi32(dst + 5*16, r5);
        _mm512_storeu_epi32(dst + 6*16, r6);
        _mm512_storeu_epi32(dst + 7*16, r7);
        _mm512_storeu_epi32(dst + 8*16, r8);
        _mm512_storeu_epi32(dst + 9*16, r9);
        _mm512_storeu_epi32(dst + 10*16, ra);
        _mm512_storeu_epi32(dst + 11*16, rb);
        _mm512_storeu_epi32(dst + 12*16, rc);
        _mm512_storeu_epi32(dst + 13*16, rd);
        _mm512_storeu_epi32(dst + 14*16, re);
        _mm512_storeu_epi32(dst + 15*16, rf);
    }

    // 16xN, N<=16, non-valid part is filled with zeros
    void transpose_epi32_16xN(void * _dst, const void * src, int stride, int valid_n) {
        auto * dst = reinterpret_cast<uint32_t*>(_dst);
        __m512i r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, ra, rb, rc, rd, re, rf;
        auto * pA = reinterpret_cast<const uint8_t*>(src);
        __mmask16 k = _cvtu32_mask32(0xFFFFFFFF >> (32-valid_n));
        r0 = _mm512_maskz_loadu_epi16 (k, pA);
        r1 = _mm512_maskz_loadu_epi16 (k, pA + stride);
        r2 = _mm512_maskz_loadu_epi16 (k, pA + 2*stride);
        r3 = _mm512_maskz_loadu_epi16 (k, pA + 3*stride);
        r4 = _mm512_maskz_loadu_epi16 (k, pA + 4*stride);
        r5 = _mm512_maskz_loadu_epi16 (k, pA + 5*stride);
        r6 = _mm512_maskz_loadu_epi16 (k, pA + 6*stride);
        r7 = _mm512_maskz_loadu_epi16 (k, pA + 7*stride);
        r8 = _mm512_maskz_loadu_epi16 (k, pA + 8*stride);
        r9 = _mm512_maskz_loadu_epi16 (k, pA + 9*stride);
        ra = _mm512_maskz_loadu_epi16 (k, pA + 10*stride);
        rb = _mm512_maskz_loadu_epi16 (k, pA + 11*stride);
        rc = _mm512_maskz_loadu_epi16 (k, pA + 12*stride);
        rd = _mm512_maskz_loadu_epi16 (k, pA + 13*stride);
        re = _mm512_maskz_loadu_epi16 (k, pA + 14*stride);
        rf = _mm512_maskz_loadu_epi16 (k, pA + 15*stride);
        transpose_m512i_16x16(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, ra, rb, rc, rd, re, rf);
        _mm512_storeu_epi32(dst, r0);
        _mm512_storeu_epi32(dst + 16, r1);
        _mm512_storeu_epi32(dst + 2*16, r2);
        _mm512_storeu_epi32(dst + 3*16, r3);
        _mm512_storeu_epi32(dst + 4*16, r4);
        _mm512_storeu_epi32(dst + 5*16, r5);
        _mm512_storeu_epi32(dst + 6*16, r6);
        _mm512_storeu_epi32(dst + 7*16, r7);
        _mm512_storeu_epi32(dst + 8*16, r8);
        _mm512_storeu_epi32(dst + 9*16, r9);
        _mm512_storeu_epi32(dst + 10*16, ra);
        _mm512_storeu_epi32(dst + 11*16, rb);
        _mm512_storeu_epi32(dst + 12*16, rc);
        _mm512_storeu_epi32(dst + 13*16, rd);
        _mm512_storeu_epi32(dst + 14*16, re);
        _mm512_storeu_epi32(dst + 15*16, rf);
    }

    void gelu_erf_minmax_approx() {
        /*
```c++
// gelu_erf(x) polynomial for direct erf approximation (formula defined)
static const table_t gelu_erf_minimax_polynomial {
        {gelu_erf_minimax_pol, {0x3f4c4228, true}}, // p0 = 0x1.98845p-1
        {gelu_erf_minimax_pol, {0xbe082bc7, true}}, // p1 = -0x1.10578ep-3
        {gelu_erf_minimax_pol, {0x3ca3621f, true}}, // p2 = 0x1.46c43ep-6
        {gelu_erf_minimax_pol, {0xbb1b7399, true}}, // p3 = -0x1.36e732p-9
        {gelu_erf_minimax_pol, {0x3970b255, true}}, // p4 = 0x1.e164aap-13
        {gelu_erf_minimax_pol, {0xb79b0914, true}}, // p5 = -0x1.361228p-16
        {gelu_erf_minimax_pol, {0x35a776e9, true}}, // p6 = 0x1.4eedd2p-20
        {gelu_erf_minimax_pol, {0xb3969b11, true}}, // p7 = -0x1.2d3622p-24
        {gelu_erf_minimax_pol, {0x315d4a4f, true}}, // p8 = 0x1.ba949ep-29
        {gelu_erf_minimax_pol, {0xaf013b2c, true}}, // p9 = -0x1.027658p-33
        {gelu_erf_minimax_pol, {0x2c67ddb2, true}}, // p10 = 0x1.cfbb64p-39
        {gelu_erf_minimax_pol, {0xa998c963, true}}, // p11 = -0x1.3192c6p-44
        {gelu_erf_minimax_pol, {0x268a7927, true}}, // p12 = 0x1.14f24ep-50
        {gelu_erf_minimax_pol, {0xa3198977, true}}, // p13 = -0x1.3312eep-57
        {gelu_erf_minimax_pol, {0x1f1c83fd, true}}, // p14 = 0x1.3907fap-65
};

template <cpu_isa_t isa, typename Wmm>
void jit_uni_eltwise_injector_f32<isa,
        Wmm>::gelu_erf_minimax_approx_compute_vector_fwd(const Vmm &vmm_src) {
    using namespace Xbyak::util;

    // TODO: consider enabling for lower ISA
    if (!is_superset(isa, avx512_core)) return;

    // register mapping
    Vmm vmm_pol = vmm_aux1, vmm_src_square = vmm_aux2, vmm_src_half = vmm_aux3,
        vmm_src_positive = vmm_aux4;

    

    h->uni_vmulps(vmm_src_square, vmm_src, vmm_src);    //vmm_src_square = (x*x)
    h->uni_vmovups(vmm_src_positive, vmm_src);          //vmm_src_positive = x
    h->uni_vandps(vmm_src_positive, vmm_src_positive, table_val(positive_mask)); // vmm_src_positive = x & 0x7FFFFFFFF

    h->uni_vmulps(vmm_src_half, vmm_src, table_val(half));                  // vmm_src_half = x * (1/2)

    // compute P(x^2)
    h->uni_vmovups(vmm_pol, table_val(gelu_erf_minimax_pol, 14));           // 
    // TODO: consider reducing latency by spitting into parital sums, for
    // example by using x^4 polynomial
    for (int deg = 13; deg >= 0; --deg) {
        h->uni_vfmadd213ps(
                vmm_pol, vmm_src_square, table_val(gelu_erf_minimax_pol, deg));
    }

    // 1.0f + erf(x * inv_sqrt2) = 1.0f + x * P(x^2)
    h->uni_vfmadd213ps(vmm_pol, vmm_src, table_val(one));
    // move instead first blend_with_mask?
    h->uni_vmulps(vmm_pol, vmm_pol, vmm_src_half);
    // Now we blend the results
    // [saturation_ubound; +inf] : we return x
    // [-inf; neg_saturation_ubound] : we return 0.0f
    h->uni_vmovups(vmm_mask, table_val(gelu_erf_minimax_neg_saturation_ubound));
    compute_cmp_mask(vmm_mask, vmm_src, _cmp_ge_os);
    blend_with_mask(vmm_src, table_val(zero));
    // [neg_saturation_ubound; -linear_ubound] or
    // [linear_ubound; saturation_lbound] : we return P(x)
    h->uni_vmovups(vmm_mask, table_val(gelu_erf_minimax_saturation_lbound));
    compute_cmp_mask(vmm_mask, vmm_src_positive, _cmp_gt_os);
    blend_with_mask(vmm_src, vmm_pol);
    // [-linear_ubound; linear_ubound] : we return 0.5f * x
    h->uni_vmovups(vmm_mask, table_val(gelu_erf_minimax_linear_ubound));
    compute_cmp_mask(vmm_mask, vmm_src_positive, _cmp_gt_os);
    blend_with_mask(vmm_src, vmm_src_half);
}
    */
    }
    

    void kpack_tile_B0B1(void * _dst0, void * _dst1, const void * _src, int stride, int src_rows) {
        // in 64unit
        //
        //  [a1 a2 a3 a4 | a5 a6 a7 a8]
        //  [b1 b2 b3 b4 | b5 b6 b7 b8]
        // _mm512_permutexvar_epi64
        //  [a1 a5 a2 a6 a3 a7 a4 a8]
        //  [b1 b5 b2 b6 b3 b7 b4 b8]
        // _mm512_unpacklo_epi16 works in 128 lanes, & means interleave
        //  [a1&b1 a2&b2 a3&b3 a4&b4]
        // _mm512_unpackhi_epi16 
        //  [a5&b5 a6&b6 a7&b7 a8&b8]
        //
        //
        static const uint64_t idx[8] = {0,4,1,5,2,6,3,7};
        auto midx = _mm512_loadu_epi64(idx);
        const auto * src = reinterpret_cast<const int8_t *>(_src);
        auto * dst0 = reinterpret_cast<int8_t *>(_dst0);
        auto * dst1 = reinterpret_cast<int8_t *>(_dst1);
        __m512i a,b,rowB0, rowB1;
        if (src_rows == 32) {
            for (int row = 0; row < 16; row++) {
                a = _mm512_loadu_epi16(src);
                b = _mm512_loadu_epi16(src + stride);
                a = _mm512_permutexvar_epi64(midx, a);
                b = _mm512_permutexvar_epi64(midx, b);
                rowB0 = _mm512_unpacklo_epi16(a, b);
                rowB1 = _mm512_unpackhi_epi16(a, b);
                _mm512_storeu_epi16(dst0, rowB0);
                _mm512_storeu_epi16(dst1, rowB1);
                src += 2*stride;
                dst0 += 64;
                dst1 += 64;
            }
        } else {
            int row = 0;
            for (; row < (src_rows/2); row++) {
                a = _mm512_loadu_epi16(src);
                b = _mm512_loadu_epi16(src + stride);
                a = _mm512_permutexvar_epi64(midx, a);
                b = _mm512_permutexvar_epi64(midx, b);
                rowB0 = _mm512_unpacklo_epi16(a, b);
                rowB1 = _mm512_unpackhi_epi16(a, b);
                _mm512_storeu_epi16(dst0, rowB0);
                _mm512_storeu_epi16(dst1, rowB1);
                src += 2*stride;
                dst0 += 64;
                dst1 += 64;
            }
            // the rest rows contains zeros
            if (src_rows & 1) {
                a = _mm512_loadu_epi16(src);
                b = _mm512_setzero_si512();

                a = _mm512_permutexvar_epi64(midx, a);
                b = _mm512_permutexvar_epi64(midx, b);
                auto rowB0 = _mm512_unpacklo_epi16(a, b);
                auto rowB1 = _mm512_unpackhi_epi16(a, b);
                _mm512_storeu_epi16(dst0, rowB0);
                _mm512_storeu_epi16(dst1, rowB1);
                src += 2*stride;
                dst0 += 64;
                dst1 += 64;
                row ++;
            }
            rowB0 = _mm512_setzero_si512();
            rowB1 = _mm512_setzero_si512();
            for(; row < 16; row++) {
                _mm512_storeu_epi16(dst0, rowB0);
                _mm512_storeu_epi16(dst1, rowB1);
                src += 2*stride;
                dst0 += 64;
                dst1 += 64;
            }
        }
    }

    // prepare B matrix for C matrix 2x2 blocking (B matrix
    // will be accessed in 1x2)
    // given 2x2 blocking scheme, Kx32 blocks are always
    // accessed sequentially:
    // transpose/repack each 32xK bfloat16 submatrix
    // into Kx32 slices (each number is a 16x32 bf16-block):
    //   0 2 4 6 ... ...
    //   1 3 5 7 ... ...
    // 
    void prepareB(KpackedB & B, tensor2D<bfloat16> & matB, bool transpose = false) {
        int K = matB.dims[transpose?1:0];
        int N = matB.dims[transpose?0:1];
        B.resize(K, N);
        if (1) {
            if (transpose) {
                assert(N == matB.dims[0]);
                assert(K == matB.dims[1]);
                for(int n = 0; n < N; n+=32) {
                    auto * dst = &B(0, n);
                    auto * src0 = &matB(n, 0);
                    auto * src1 = &matB(n + 16, 0);
                    int k;
                    for(k = 0; k + 32 <= K; k+=32, src0+=32, src1+=32) {
                        // B0 (16x32) => transpose+repack as 32x16 => 16x16x2
                        functional::transpose_epi32_16x16(dst, src0, matB.stride);
                        dst += (16*32);
                        // B1
                        functional::transpose_epi32_16x16(dst, src1, matB.stride);
                        dst += (16*32);
                    }
                    if (k < K) {
                        functional::transpose_epi32_16xN(dst, src0, matB.stride, K-k);
                        dst += (16*32);
                        functional::transpose_epi32_16xN(dst, src1, matB.stride, K-k);
                        dst += (16*32);
                    }
                }
            } else {
                assert(K == matB.dims[0]);
                assert(N == matB.dims[1]);
                // pack only then layout sequentially
                for(int n = 0; n < N; n+=32) {
                    auto * dst = &B(0, n);
                    for(int k = 0; k < K; k+=32) {
                        // B0 B1 32x(16+16) => repack as two 16x16x2
                        int src_rows = std::min(K - k, 32);
                        functional::kpack_tile_B0B1(dst, dst + (16*32), &matB(k, n), matB.stride, src_rows);
                        dst += (16*32)*2;
                    }
                }
            }
        } else {
            for (int k = 0; k < B.Kblocks*32; k++)
                for (int n = 0; n < B.Nblocks*32; n++) {
                if (k < K && n < N)
                    B(k, n) = transpose ? matB(n, k) : matB(k, n);
                else
                    B(k, n) = 0; // padding zero
            }
        }
    }
};

// 2x2 tiles post process kernels
namespace PostProcess {

    // 4 tiles located at C matrix (m,n) of size (valid_m, valid_n)
    //   tC00/tC01
    //   tC10/tC11
    void func_save2bf16(tensor2D<bfloat16> & matBF16, tensor2D<float> & buffC, int m, int n, int valid_m, int valid_n) {
        auto * psrc = &buffC(0,0);
        int8_t * pdst = reinterpret_cast<int8_t*>(&(matBF16(m, n)));
        int stride = matBF16.stride;
        __mmask32 k = _cvtu32_mask32(0xFFFFFFFF >> (32-valid_n));
        while(valid_m >= 16) {
            for(int i = 0; i < 16; i ++) {
                auto b = _mm512_loadu_epi16(psrc);
                auto a = _mm512_loadu_epi16(psrc + 16);
                auto c = _mm512_cvtne2ps_pbh(a, b);
                _mm512_mask_storeu_epi16(pdst, k, c);   // 32 bf16
                pdst += stride;
                psrc += 32;
            }
            valid_m -= 16;
        }
        for(int i = 0; i < valid_m; i ++) {
            auto b = _mm512_loadu_epi16(psrc);
            auto a = _mm512_loadu_epi16(psrc + 16);
            auto c = _mm512_cvtne2ps_pbh(a, b);
            _mm512_mask_storeu_epi16(pdst, k, c);   // 32 bf16
            pdst += stride;
            psrc += 32;
        }
    }

    // a helper callable for most frequenctly used pp kernels
    struct save2bf16 {
        tensor2D<bfloat16> & C;
        save2bf16(tensor2D<bfloat16> & C) : C(C) {}
        void operator()(tensor2D<float> & buffC, int m, int n, int valid_m, int valid_n) {
            func_save2bf16(C, buffC, m, n, valid_m, valid_n);
        }
    };
};

// matmul (FC)
//
// constB constrols if it's FC or not 
//
// multi-thread caller can split the whole C matrix
// into grid (better in unit with size of multiple of 32x32)
// each grid is a considered as a independent matmul on
// submatrix of A,B and C.

struct Matmul {
    KpackedB internalB;
    tensor2D<bfloat16> scratch;
    BlockIterator blk_it;
    bool constB;
    bool transposeB;
    // 2x2 C tiles buffer
    tensor2D<float> buffC;

    Matmul(bool constB = false, bool transposeB = false) : 
        constB(constB), transposeB(transposeB), buffC(32, 32) {}

    
    // different ppkernel has difference runtime args
    // which is set by caller, since only caller knows
    // what setter methods to use for specific ppkernel
    template<typename PP>
    void operator()(tensor2D<bfloat16> & matA,
                    tensor2D<bfloat16> & matB,
                    PP ppkernel) {
        int M = matA.dims[0];
        int K = matA.dims[1];
        int N = matB.dims[transposeB ? 0 : 1];
        assert(K == matB.dims[transposeB ? 1 : 0]);

        // determine blocking scheme
        int elesz = sizeof(uint16_t);
        int L2 = 2048*1024; // 2MB
        int slice_size = 32*K*elesz;
        int mc = L2/slice_size - 1;
        assert(mc > 0);

        auto dmax = std::numeric_limits<int>::max();
        BlockIterator::blkloop bloops[] = {
            {mc,32,0}, {dmax,0,32}, {dmax,mc*32,0}
        };
        blk_it.reset(bloops, 3, M, N);

        // for non-constB, internalB is updated every time
        // for constB, internalB is updated once
        if (!constB || (internalB.capacity == 0)) {
            functional::prepareB(internalB, matB, transposeB);
        }

        // prepare tails buffer
        tensor2D<bfloat16> & Atails = scratch;
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
        // main loop
        do
        {
            int m = blk_it.m;
            int n = blk_it.n;
            int valid_m = std::min(M - m, 32);
            int valid_n = std::min(N - n, 32);
            auto * pA0 = &matA(m, 0);
            auto * pA1 = &matA(m + 16, 0);
            auto strideA = matA.stride;
            auto * pB = &internalB(0, n);
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
            if (valid_m <= 16) {
                // 1x2 is enough
                for (int k = 0; k < K; k += 32) {
                    _tile_loadd(tA0, pA0 + k, strideA);
                    _tile_loadd(tB0, pB, 64); pB += (16*32);
                    _tile_dpbf16ps(tC00, tA0, tB0);
                    _tile_loadd(tB1, pB, 64); pB += (16*32);
                    _tile_dpbf16ps(tC01, tA0, tB1);
                }
                _tile_stored(tC00, &buffC(0,0), buffC.stride);
                _tile_stored(tC01, &buffC(0,16), buffC.stride);
            } else {
                // 2x2
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
                _tile_stored(tC00, &buffC(0,0), buffC.stride);
                _tile_stored(tC01, &buffC(0,16), buffC.stride);
                _tile_stored(tC10, &buffC(16,0), buffC.stride);
                _tile_stored(tC11, &buffC(16,16), buffC.stride);
            }
            // post processing the accumulator tiles
            //  - add bias
            //  - do activations
            //  - convert into bfloat16
            //  - store into C matrix
            (ppkernel)(buffC, m, n, valid_m, valid_n);
        } while(blk_it.next());
    }
};


#if 0
// using only 1 tile in C matrix:
//
//      B0
//      B1
//      ...
//      B5
//   A0 C0
//
// when K < (32*6)=192, Kx16 B sub-matrix can repack dynamically once and
// all hold in titles, then we can go vertially until A-sub matrix is fit
// in L2 cache, then we go next
// 
template<typename PP, int K>
void matmul1x1(tensor2D<bfloat16> & matA,
                tensor2D<bfloat16> & matB,
                tensor2D<bfloat16> & matC,
                tensor2D<bfloat16> & scratch,
                bool transB,
                PP ppkernel) {
    constexpr int tB0 = 0;
    constexpr int tB1 = 1;
    constexpr int tB2 = 2;
    constexpr int tB3 = 3;
    constexpr int tB4 = 4;
    constexpr int tB5 = 5;
    constexpr int tC = 6;
    constexpr int tA = 7;
    tensor2D<bfloat16> & Atails = scratch;
    int M = matC.dims[0];
    int N = matC.dims[1];
    int _K = matA.dims[1];
    assert(_K == matB.K);
    assert(N == matB.N);
    assert(_K == K);
    assert (K < 32*6);

    // determine blocking scheme
    int elesz = sizeof(uint16_t);
    int L2 = 2048*1024; // 2MB
    int slice_size = 16*K*elesz;
    int mc = L2/slice_size - 1;
    assert(mc > 0);

    auto dmax = std::numeric_limits<int>::max();
    BlockIterator::blkloop bloops[] = {
        {1,mc*16,0}, {dmax,0,16}, {dmax,mc*16,0}
    };
    bloop.reset(bloops, 3, M, N);

    int mtails = M % 16;
    if (mtails > 0) {
        Atails.resize(16, rndup(K, 32));
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

    do
    {
        // k loop is unrolled, so we handle mc*16 blocks
        int m = blk_it.m;
        int n = blk_it.n;

        // load all B tiles
        for (int k = 0; k < K; k += 32) {

        }
        int valid_m = std::min(M - m, 16);
        int valid_n = std::min(N - n, 16);


        auto * pA0 = &matA(m, 0);
        auto strideA = matA.stride;
        auto * pB = &matB(0, n);
        if (valid_m < 16) {
            // use Atails buffer to prevent memory read segmentfault
            pA0 = &Atails(0, 0);
            strideA = Atails.stride;
        }
        // load all B tiles

        // k loop is unrolled, 
        _tile_zero(tC);
        _tile_loadd(tA, pA0 + 0, strideA);
        _tile_dpbf16ps(tC, tA, tB0);

        if (K > 32) {
            _tile_loadd(tA, pA0 + 32, strideA);
            _tile_dpbf16ps(tC, tA, tB1);
        }
        if (K > 32*2) {
            _tile_loadd(tA, pA0 + 32*2, strideA);
            _tile_dpbf16ps(tC, tA, tB2);
        }
        if (K > 32*3) {
            _tile_loadd(tA, pA0 + 32*3, strideA);
            _tile_dpbf16ps(tC, tA, tB3);
        }
        if (K > 32*4) {
            _tile_loadd(tA, pA0 + 32*4, strideA);
            _tile_dpbf16ps(tC, tA, tB4);
        }
        if (K > 32*5) {
            _tile_loadd(tA, pA0 + 32*5, strideA);
            _tile_dpbf16ps(tC, tA, tB5);
        }

        // post processing the accumulator tiles
        //  - add bias
        //  - do activations
        //  - convert into bfloat16
        //  - store into C matrix
        (ppkernel)(&matC(m, n), matC.stride, valid_m, valid_n);
    } while(blk_it.next());
}
#endif
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

        for(int m = 0; m < M; m += 16) {
            auto * pA = reinterpret_cast<uint8_t*>(&matA(m, 0));
            auto * pBi32 = reinterpret_cast<int32_t*>(vecB);
            __m512 regC0 = _mm512_setzero();
            __m512 regC1 = _mm512_setzero();
            for(int k = 0; k < K; k += 32, pA += 64, pBi32 += 16) {
                // handle Ab: 16x32
                // transposed in register as 16x16x2
                //   r0: (a0,a1)(b0,b1)....
                //   r1: (a2,a3)(b2,b3)....
                //      ...
                //   rf: (a30,a31),(b30,b31)....
                //
                __m512i r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, ra, rb, rc, rd, re, rf;
                auto stride = matA.stride;
                r0 = _mm512_loadu_epi32(pA);
                r1 = _mm512_loadu_epi32(pA + stride);
                r2 = _mm512_loadu_epi32(pA + 2*stride);
                r3 = _mm512_loadu_epi32(pA + 3*stride);
                r4 = _mm512_loadu_epi32(pA + 4*stride);
                r5 = _mm512_loadu_epi32(pA + 5*stride);
                r6 = _mm512_loadu_epi32(pA + 6*stride);
                r7 = _mm512_loadu_epi32(pA + 7*stride);
                r8 = _mm512_loadu_epi32(pA + 8*stride);
                r9 = _mm512_loadu_epi32(pA + 9*stride);
                ra = _mm512_loadu_epi32(pA + 10*stride);
                rb = _mm512_loadu_epi32(pA + 11*stride);
                rc = _mm512_loadu_epi32(pA + 12*stride);
                rd = _mm512_loadu_epi32(pA + 13*stride);
                re = _mm512_loadu_epi32(pA + 14*stride);
                rf = _mm512_loadu_epi32(pA + 15*stride);
                
                functional::transpose_m512i_16x16(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, ra, rb, rc, rd, re, rf);

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
