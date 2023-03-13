#pragma once

#include "misc.hpp"


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


namespace amx_bf16
{
struct tileconfig_t {
    uint8_t palette_id;
    uint8_t startRow;
    uint8_t reserved[14];
    uint16_t cols[16];
    uint8_t rows[16];
    tileconfig_t() = default;
    tileconfig_t(int palette, int _startRow, int numTiles, int _rows, int columnsBytes) {
        palette_id = palette;
        startRow = _startRow;
        for(int i = 0; i < 14; i++) {
            reserved[i] = 0;
        }
        for(int i = 0; i < numTiles; i++) {
            cols[i] = columnsBytes;
            rows[i] = _rows;
        }
        for(int i = numTiles; i < 16; i++) {
            cols[i] = 0;
            rows[i] = 0;
        }
        load();
    }
    ~tileconfig_t() {
        _tile_release();
    }
    void load() {
        //std::cout << "\ttile load config ... " << std::flush;
        _tile_loadconfig(this);
        //std::cout << *this << std::flush << std::endl;
    }
    void store() {
        _tile_storeconfig(this);
    }
    friend std::ostream& operator<<(std::ostream& out, const tileconfig_t& cfg) {
        out << " palette_id=" << static_cast<int>(cfg.palette_id);
        out << " startRow=" << static_cast<int>(cfg.startRow);
        out << " row x colsb=(";
        for (int i = 0; i < 16;i++) {
            if (cfg.rows[i] == 0 && cfg.cols[i] == 0)
                continue;
            if (i > 0) out << ",";
            out << static_cast<int>(cfg.rows[i]) << "x" << static_cast<int>(cfg.cols[i]);
        }
        out << ")";
        return out;
    }
} __attribute__ ((__packed__));


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

    inline void transpose_m512i_16x16(__m512i &r0, __m512i &r1, __m512i &r2, __m512i &r3,
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

    inline void transpose_epi32_16x16(void * _dst, const void * src, int stride) {
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
    inline void transpose_epi32_16xN(void * _dst, const void * src, int stride, int valid_n) {
        auto * dst = reinterpret_cast<uint32_t*>(_dst);
        __m512i r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, ra, rb, rc, rd, re, rf;
        auto * pA = reinterpret_cast<const uint8_t*>(src);
        uint32_t mask_value = 0xFFFFFFFF >> (32-valid_n);
        __mmask32 mask = _cvtu32_mask32(mask_value);
        r0 = _mm512_maskz_loadu_epi16 (mask, pA);
        r1 = _mm512_maskz_loadu_epi16 (mask, pA + stride);
        r2 = _mm512_maskz_loadu_epi16 (mask, pA + 2*stride);
        r3 = _mm512_maskz_loadu_epi16 (mask, pA + 3*stride);
        r4 = _mm512_maskz_loadu_epi16 (mask, pA + 4*stride);
        r5 = _mm512_maskz_loadu_epi16 (mask, pA + 5*stride);
        r6 = _mm512_maskz_loadu_epi16 (mask, pA + 6*stride);
        r7 = _mm512_maskz_loadu_epi16 (mask, pA + 7*stride);
        r8 = _mm512_maskz_loadu_epi16 (mask, pA + 8*stride);
        r9 = _mm512_maskz_loadu_epi16 (mask, pA + 9*stride);
        ra = _mm512_maskz_loadu_epi16 (mask, pA + 10*stride);
        rb = _mm512_maskz_loadu_epi16 (mask, pA + 11*stride);
        rc = _mm512_maskz_loadu_epi16 (mask, pA + 12*stride);
        rd = _mm512_maskz_loadu_epi16 (mask, pA + 13*stride);
        re = _mm512_maskz_loadu_epi16 (mask, pA + 14*stride);
        rf = _mm512_maskz_loadu_epi16 (mask, pA + 15*stride);
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

    // gelu_erf_minimax_approx_compute_vector_fwd in oneDNN
    //   x*0.5*(1+erf(x/sqrt(2))) = x*0.5*(1 + x*Polynomial(x^2))
    inline __m512 gelu_erf_minmax_approx(__m512 & x) {
        auto x2 = _mm512_mul_ps(x, x); // x^2
        
        auto x_positive = _mm512_castsi512_ps(_mm512_and_epi32(_mm512_castps_si512(x), _mm512_set1_epi32(0x7FFFFFFF)));    // clear sign mask
        auto x_half = _mm512_mul_ps(x, _mm512_set1_ps(0.5f));

        auto poly = _mm512_castsi512_ps(_mm512_set1_epi32(0x1f1c83fd));
        poly = _mm512_fmadd_ps(poly, x2, _mm512_castsi512_ps(_mm512_set1_epi32(0xa3198977))); // poly * x^2 + xxx
        poly = _mm512_fmadd_ps(poly, x2, _mm512_castsi512_ps(_mm512_set1_epi32(0x268a7927)));
        poly = _mm512_fmadd_ps(poly, x2, _mm512_castsi512_ps(_mm512_set1_epi32(0xa998c963)));
        poly = _mm512_fmadd_ps(poly, x2, _mm512_castsi512_ps(_mm512_set1_epi32(0x2c67ddb2)));
        poly = _mm512_fmadd_ps(poly, x2, _mm512_castsi512_ps(_mm512_set1_epi32(0xaf013b2c)));
        poly = _mm512_fmadd_ps(poly, x2, _mm512_castsi512_ps(_mm512_set1_epi32(0x315d4a4f)));
        poly = _mm512_fmadd_ps(poly, x2, _mm512_castsi512_ps(_mm512_set1_epi32(0xb3969b11)));
        poly = _mm512_fmadd_ps(poly, x2, _mm512_castsi512_ps(_mm512_set1_epi32(0x35a776e9)));
        poly = _mm512_fmadd_ps(poly, x2, _mm512_castsi512_ps(_mm512_set1_epi32(0xb79b0914)));
        poly = _mm512_fmadd_ps(poly, x2, _mm512_castsi512_ps(_mm512_set1_epi32(0x3970b255)));
        poly = _mm512_fmadd_ps(poly, x2, _mm512_castsi512_ps(_mm512_set1_epi32(0xbb1b7399)));
        poly = _mm512_fmadd_ps(poly, x2, _mm512_castsi512_ps(_mm512_set1_epi32(0x3ca3621f)));
        poly = _mm512_fmadd_ps(poly, x2, _mm512_castsi512_ps(_mm512_set1_epi32(0xbe082bc7)));
        poly = _mm512_fmadd_ps(poly, x2, _mm512_castsi512_ps(_mm512_set1_epi32(0x3f4c4228)));

        // 1.0f + erf(x * inv_sqrt2) = 1.0f + x * P(x^2)
        poly = _mm512_fmadd_ps(poly, x, _mm512_set1_ps(1.0f));
        // x*0.5*(1 + x*Polynomial(x^2))
        poly = _mm512_mul_ps(poly, x_half);

        // combine:
        // zone_id
        //  1 -inf; -saturation_lbound           : 0.0f
        //  2 -saturation_lbound; -linear_ubound : x*0.5*(1 + x*Polynomial(x^2))
        //  3 -linear_ubound, linear_ubound         : x*0.5
        //  4 linear_ubound : saturation_lbound     : x*0.5*(1 + x*Polynomial(x^2))
        //  5 saturation_lbound: +inf               : x
        constexpr int neg_saturation_lbound = 0xc0a00000;
        constexpr int linear_ubound = 0x33800000;
        constexpr int saturation_lbound = 0x40a00000;

        auto mask_x_not_zone1 = _mm512_cmpnlt_ps_mask(x, _mm512_castsi512_ps(_mm512_set1_epi32(neg_saturation_lbound)));
        x = _mm512_maskz_mov_ps(mask_x_not_zone1, x);

        auto mask_x_in_zone5 = _mm512_cmpnle_ps_mask(x_positive, _mm512_castsi512_ps(_mm512_set1_epi32(saturation_lbound)));
        poly = _mm512_mask_mov_ps(poly, mask_x_in_zone5, x);

        auto mask_x_in_zone3 = _mm512_cmple_ps_mask(x_positive, _mm512_castsi512_ps(_mm512_set1_epi32(linear_ubound)));
        poly = _mm512_mask_mov_ps(poly, mask_x_in_zone3, x_half);
        return poly;
    }

    inline void kpack_tile_B0B1(void * _dst0, void * _dst1, const void * _src, int stride, int src_rows) {
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
    inline void prepareB(KpackedB & B, tensor2D<bfloat16> & matB, bool transpose = false) {
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
                    for(k = 0; (k + 32) <= K; k+=32, src0+=32, src1+=32) {
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

// 4 tiles located at C matrix (m,n) of size (valid_m, valid_n)
//   tC00/tC01
//   tC10/tC11

namespace PP {

    // PP kernel has infinite combinations, the generation
    // of specific PP kernel with high-efficiency means
    // it should be combined in source code level and
    // let compiler to optimize it. extra syntax sugar
    // through more complex meta programing is not so
    // worthy. hard code all combinations needed is not
    // that difficult and it also ensures most efficient
    // implementation.

    // a helper callable for most frequenctly used pp kernels
    struct Store2bf16 {
        tensor2D<bfloat16> & C;
        Store2bf16(tensor2D<bfloat16> & C) : C(C) {}
        void operator()(tensor2D<float> & buffC, int m, int n, int valid_m, int valid_n) {
            auto * psrc = &buffC(0,0);
            int8_t * pdst = reinterpret_cast<int8_t*>(&(C(m, n)));
            int stride = C.stride;
            __mmask32 k = _cvtu32_mask32(0xFFFFFFFF >> (32-valid_n));
            for(int i = 0; i < valid_m; i ++) {
                auto r0 = _mm512_loadu_ps(psrc);
                auto r1 = _mm512_loadu_ps(psrc + 16);
                auto c = _mm512_cvtne2ps_pbh(r1, r0);
                _mm512_mask_storeu_epi16(pdst, k, c);   // 32 bf16
                pdst += stride;
                psrc += 32;
                m++;
            }
        }
    };

    struct Addbias_Store2bf16 {
        tensor2D<bfloat16> & C;
        float * bias;
        Addbias_Store2bf16(tensor2D<bfloat16> & C, float * bias) : C(C), bias(bias) {}
        void operator()(tensor2D<float> & buffC, int m, int n, int valid_m, int valid_n) {
            auto * psrc = &buffC(0,0);
            int8_t * pdst = reinterpret_cast<int8_t*>(&(C(m, n)));
            int stride = C.stride;
            auto bias0 = _mm512_loadu_ps(bias + n);
            auto bias1 = _mm512_loadu_ps(bias + n + 16);
            __mmask32 k = _cvtu32_mask32(0xFFFFFFFF >> (32-valid_n));
            for(int i = 0; i < valid_m; i ++) {
                auto r0 = _mm512_loadu_ps(psrc);
                auto r1 = _mm512_loadu_ps(psrc + 16);
                // add bias
                r0 = _mm512_add_ps(r0, bias0);
                r1 = _mm512_add_ps(r1, bias1);
                // cvt2 bf16
                auto c = _mm512_cvtne2ps_pbh(r1, r0);
                //store
                _mm512_mask_storeu_epi16(pdst, k, c);   // 32 bf16
                pdst += stride;
                psrc += 32;
            }
        }
    };


    struct Addbias_Gelu_Store2bf16 {
        tensor2D<bfloat16> & C;
        float * bias;
        Addbias_Gelu_Store2bf16(tensor2D<bfloat16> & C, float * bias) : C(C), bias(bias) {}
        void operator()(tensor2D<float> & buffC, int m, int n, int valid_m, int valid_n) {
            auto * psrc = &buffC(0,0);
            int8_t * pdst = reinterpret_cast<int8_t*>(&(C(m, n)));
            int stride = C.stride;
            auto bias0 = _mm512_loadu_ps(bias + n);
            auto bias1 = _mm512_loadu_ps(bias + n + 16);
            __mmask32 k = _cvtu32_mask32(0xFFFFFFFF >> (32-valid_n));
            for(int i = 0; i < valid_m; i ++) {
                auto r0 = _mm512_loadu_ps(psrc);
                auto r1 = _mm512_loadu_ps(psrc + 16);
                // add bias
                r0 = _mm512_add_ps(r0, bias0);
                r1 = _mm512_add_ps(r1, bias1);
                r0 = functional::gelu_erf_minmax_approx(r0);
                r1 = functional::gelu_erf_minmax_approx(r1);
                // cvt2 bf16
                auto c = _mm512_cvtne2ps_pbh(r1, r0);
                //store
                _mm512_mask_storeu_epi16(pdst, k, c);   // 32 bf16
                pdst += stride;
                psrc += 32;
            }
        }
    };

    struct Store2float {
        tensor2D<float> & C;
        Store2float(tensor2D<float> & C) : C(C) {}
        void operator()(tensor2D<float> & buffC, int m, int n, int valid_m, int valid_n) {
            auto * psrc = &buffC(0,0);
            int8_t * pdst = reinterpret_cast<int8_t*>(&(C(m, n)));
            int stride = C.stride;
            uint32_t mask = 0xFFFFFFFF >> (32-valid_n);
            __mmask32 k0 = _cvtu32_mask32(mask & 0xFFFF);
            __mmask32 k1 = _cvtu32_mask32(mask >> 16);
            for(int i = 0; i < valid_m; i ++) {
                auto r0 = _mm512_loadu_ps(psrc);
                auto r1 = _mm512_loadu_ps(psrc + 16);
                _mm512_mask_storeu_ps(pdst, k0, r0);
                _mm512_mask_storeu_ps(pdst + 64, k1, r1);
                pdst += stride;
                psrc += 32;
                m++;
            }
        }
    };
}

// matmul (FC)
//
// constB constrols whether it's FC or not 
struct Matmul {
    KpackedB internalB;
    tensor2D<bfloat16> scratch;
    BlockIterator blk_it;
    bool constB;
    bool transposeB;
    // 2x2 C tiles buffer
    // most usecase requires post-processing with AVX, thus buffC
    // is used to transfer data to AVX register
    tensor2D<float> buffC;

    Matmul(bool constB = false, bool transposeB = false) : 
        constB(constB), transposeB(transposeB), buffC(32, 32) {}

    // empty PP (for test purpose only, uncommon in real use case)
    void operator()(tensor2D<bfloat16> & matA,
                    tensor2D<bfloat16> & matB,
                    tensor2D<bfloat16> & matC) {
        PP::Store2bf16 ppkernel(matC);
        (*this)(matA, matB, ppkernel);
    }

    void operator()(tensor2D<bfloat16> & matA,
                    tensor2D<bfloat16> & matB,
                    tensor2D<float> & matC) {
        PP::Store2float ppkernel(matC);
        (*this)(matA, matB, ppkernel);
    }

    template<typename PP>
    void operator()(tensor2D<bfloat16> & matA,
                    tensor2D<bfloat16> & matB,
                    PP ppkernel) {
        int N = matB.dims[transposeB ? 0 : 1];
        (*this)(matA, matB, 0, N, ppkernel);
    }

    // ppkernel is a callable which captures the runtime args
    // by itself, so no need to pass in any post-process related
    // runtime args through this API
    //
    // n0/n1 allows us for calculating only partial results, so it
    // can be used to run on multi-cores in parallel  
    //
    // ppkernel will be invoked with true (m,n) with n0-offset added
    // so ppkernel don't need to know which sub-matrix it's working on.
    //
    // for most ppkernels w/o runtime state, a single ppkernel can be
    // shared among all threads.
    //
    // but for ppkernels doing reductions, it needs separate instance
    // for each thread, also a post-merging process to combine the results.
    //
    // ppkernels are simple to write, further wrapping or structurelize only
    // makes the design more complex, so we stop doing that.
    template<typename PP>
    void operator()(tensor2D<bfloat16> & matA,
                    tensor2D<bfloat16> & _matB,
                    int n0, int n1,
                    PP ppkernel) {
        int M = matA.dims[0];
        int K = matA.dims[1];

        //build sub-matrix B
        int Bd0 = transposeB ? (n1-n0) : _matB.dims[0];
        int Bd1 = transposeB ? _matB.dims[1] : (n1-n0);
        bfloat16 * pbase = transposeB ? (&_matB(n0, 0)):(&_matB(0, n0));
        tensor2D<bfloat16> matB(Bd0, Bd1, pbase, _matB.stride);

        assert(K == matB.dims[transposeB ? 1 : 0]);
        int N = matB.dims[transposeB ? 0 : 1];

        // determine blocking scheme
        int elesz = sizeof(uint16_t);
        int L2 = 2048*1024; // 2MB
        int slice_size = 32*rndup(K, 32)*elesz;
        int mc = L2/slice_size - 1;
        
        // if 1 32xK slice cannot fit L2, use 1 slice at least
        if (mc == 0)
            mc = 1;

        int L1 = 48*1024; // 48K
        int mcL1 = L1/slice_size - 1;

        //std::cout << "mc=" << mc << std::endl;
        //if (mcL1 > 2) {
        //mc = 2;
        //}


        auto dmax = std::numeric_limits<int>::max();
        BlockIterator::blkloop bloops[] = {{mc,32,0}, {dmax,0,32}, {dmax,mc*32,0}};
        blk_it.reset(bloops, 3, M, N);
        //BlockIterator::blkloop bloops[] = {{2,32,0}, {2,0,32}, {dmax, 64, 0}, {dmax, 0, 64}};
        //blk_it.reset(bloops, 4, M, N);

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
        tileconfig_t tfg(1, 0, 8, 16, 64);
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
                _tile_loadd(tA0, pA0 + 0, strideA);
                for (int k = 0; k < K; k += 32) {
                    _tile_loadd(tB0, pB, 64); pB += (16*32);
                    _tile_dpbf16ps(tC00, tA0, tB0);
                    _tile_loadd(tA1, pA1 + k, strideA);
                    _tile_dpbf16ps(tC10, tA1, tB0);
                    _tile_loadd(tB1, pB, 64); pB += (16*32);
                    _tile_dpbf16ps(tC01, tA0, tB1);
                    _tile_loadd(tA0, pA0 + k + 32, strideA);    // balance load & dp. load next
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
            (ppkernel)(buffC, m, n + n0, valid_m, valid_n);
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
} // namespace amx_bf16


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

enum Activation {
    RELU,
};

template<Activation act>
struct Matmul {
    BlockIterator blk_it;
    tensor2D<float> scratch;
    Matmul() {};

    void operator()(tensor2D<float> & matA,
                    tensor2D<float> & matB,
                    tensor2D<float> & matC,
                    float * bias) {
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
            auto bias0 = bias ? _mm512_loadu_ps(bias + n) : _mm512_setzero_ps();
            auto bias1 = bias ? _mm512_loadu_ps(bias + n + 16) : _mm512_setzero_ps();
            auto bias2 = bias ? _mm512_loadu_ps(bias + n + 16*2) : _mm512_setzero_ps();
            auto bias3 = bias ? _mm512_loadu_ps(bias + n + 16*3) : _mm512_setzero_ps();

            auto * pC = &matC(m, n);
            auto zero = _mm512_setzero_ps();
            __mmask16 mask = _cvtu32_mask16(0xFFFFFFFF >> (32-valid_n));

            // this lambda function is easier to optimize by compiler
            auto lppkernel = [&](__m512 & v0,__m512 & v1,__m512 & v2,__m512 & v3){
                v0 = _mm512_add_ps(v0, bias0);
                v1 = _mm512_add_ps(v1, bias1);
                v2 = _mm512_add_ps(v2, bias2);
                v3 = _mm512_add_ps(v3, bias3);
                if (act == RELU) {
                    v0 = _mm512_mask_mov_ps(v0, _mm512_cmplt_ps_mask(v0, zero), zero);
                    v1 = _mm512_mask_mov_ps(v1, _mm512_cmplt_ps_mask(v1, zero), zero);
                    v2 = _mm512_mask_mov_ps(v2, _mm512_cmplt_ps_mask(v2, zero), zero);
                    v3 = _mm512_mask_mov_ps(v3, _mm512_cmplt_ps_mask(v3, zero), zero);
                }
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
