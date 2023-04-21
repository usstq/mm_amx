#pragma once

/*
https://www.intel.com/content/www/us/en/develop/documentation/cpp-compiler-developer-guide-and-reference/top/compiler-reference/intrinsics/intrinsics-for-amx-instructions/intrinsics-for-amx-tile-instructions/tile-loadconfig.html

void _tile_loadconfig (const void * mem_addr)
	format of memory payload. each field is a byte.
		 0: palette_id
		 1: startRow (8b)
	 2-15: reserved (must be zero)
	16-17: tile0.colsb -- bytes_per_row
	18-19: tile1.colsb
	20-21: tile2.colsb
			...
	46-47: tile15.colsb
		48: tile0.rows
		49: tile1.rows
		50: tile2.rows
			 ...
		63: tile15.rows

void _tile_storeconfig (void * mem_addr)
    Stores the current tile configuration to a 64-byte memory location specified by "mem_addr".
    The tile configuration format is specified below, and includes the tile type pallette,
    the number of bytes per row, and the number of rows. If tiles are not configured,
    all zeroes will be stored to memory.
*/

/*
void _tile_loadd (__tile dst, const void * base, int stride)
    Load tile rows from memory specifieid by "base" address and "stride"
    into destination tile "dst" using the tile configuration previously
    configured via "_tile_loadconfig".
    Operation:
        start := tileconfig.startRow
        IF start == 0 // not restarting, zero incoming state
            tilezero(dst)
        FI
        nbytes := dst.colsb
        DO WHILE start < dst.rows
            memptr := base + start * stride
            write_row_and_zero(dst, start, read_memory(memptr, nbytes), nbytes)
            start := start + 1
        OD
        zero_upper_rows(dst, dst.rows)
        zero_tileconfig_start()

void _tile_stored (__tile src, void * base, int stride)
    Store the tile specified by "src" to memory specifieid by "base" address and "stride"
    using the tile configuration previously configured via "_tile_loadconfig".
    Operation:
        start := tileconfig.startRow
        DO WHILE start < src.rows
            memptr := base + start * stride
            write_memory(memptr, src.colsb, src.row[start])
            start := start + 1
        OD
        zero_tileconfig_start()

void _tile_stream_loadd (__tile dst, const void * base, int stride)
    Load tile rows from memory specifieid by "base" address and "stride"
    into destination tile "dst" using the tile configuration previously
    configured via "_tile_loadconfig". This intrinsic provides a hint to
    the implementation that the data will likely not be reused in the near
    future and the data caching can be optimized accordingly.

void _tile_zero (__tile tdest)
    Zero the tile specified by "tdest".
    Operation:
        nbytes := palette_table[tileconfig.palette_id].bytes_per_row
        FOR i := 0 TO palette_table[tileconfig.palette_id].max_rows-1
            FOR j := 0 TO nbytes-1
                tdest.row[i].byte[j] := 0
            ENDFOR
        ENDFOR
	

void _tile_release ()
    Release the tile configuration to return to the init state, which releases all storage it currently holds.


Instruction Throughput Latency
LDTILECFG                204
STTILECFG                19
TILETRELEASE             13
TDP / *          16      52
TILELOADD         8      45
TILELOADDT1      33      48
TILESTORED       16
TILEZERO          0      16

Due to the high latency of the LDTILECFG instruction we recommend issuing a single pair
of LDTILECFG and TILERELEASE operations per Intel AMX-based DL layer implementation.


• A-tiles can have between 1-16 rows and 1-MAX_TILE_K columns.
• B-tiles can have between 1-MAX_TILE_K rows and 1–16 columns.
• C-tiles can have between 1-16 rows and 1–16 columns.

MAX_TILE_K=64/sizeof(type_t)
          = 32 BF16
          = 64 INT8

A tiles and B tiles contain data of type_t, which can be (u)int8 or bfloat16.
• C tiles contain data of type res_type_t:
• int32 if type_t=(u)int8
• float if type_t=bfloat16

Like the Intel® DL Boost use case, the B matrix must undergo a re-layout before it can be used within the
corresponding Intel AMX multiply instruction.

BF16    C_float_16x16 = A_bfloat16_16x32 * B_bfloat16_32x16 (re-layout as 16x16x2 Ab2a)
INT8    C_int32_16x16 = A_int8_16x64 * B_int8_64x16 (re-layout as 16x16x4 Ab4a)



FC:
    (2,1~900,2560)x(2560,7680)
    (2,1~900,2560)x(2560,2560)
    (2,1~900,2560)x(2560,10240) GELU
    (2,1~900,10240)x(10240,2560)

matmul:
    (1~990,80) * (80,1~990)
    (1~990,1~990) * (1~990,80)

    // (2,32,1~990,80)*(2,32,80,1~990)
    // (2,32,1~990,1~990)(2,32,1~990,80)
*/

#include "misc.hpp"
#include "block_iter.hpp"
#include "tensor2D.hpp"

#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

#include "bf16.hpp"
#ifdef ENABLE_NUMA
#include "numa.h"
#endif

// https://renatocunha.com/2015/12/msync-pointer-validity/
#include <sys/mman.h>
#include <unistd.h>
bool is_pointer_valid(void *p) {
    /* get the page size */
    size_t page_size = sysconf(_SC_PAGESIZE);
    /* find the address of the page that contains p */
    void *base = (void *)((((size_t)p) / page_size) * page_size);
    /* call msync, if it returns non-zero, return false */
    return msync(base, page_size, MS_ASYNC) == 0;
}

//#define FORCE_INLINE inline __attribute__((always_inline))
#define FORCE_INLINE 

using ov::bfloat16;

namespace amx_bf16 {

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
template<typename T>
struct KpackedB {
    std::shared_ptr<T> data;
    int64_t capacity;
    int K;
    int N;
    int Kblocks;
    int Nblocks;
    float quant_scale;
    float dequant_scale;
    KpackedB() {
        capacity = 0;
        K = N = 0;
        Kblocks = Nblocks = 0;
    }

    T & operator()(int k, int n) {
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
        int64_t need_capacity = (Kblocks * Nblocks + 1) * 32 * 32 * sizeof(T);
        need_capacity = rndup(need_capacity, 64);
        if (capacity < need_capacity) {
            capacity = need_capacity;
#ifdef ENABLE_NUMA
            if (USE_NUMA) {
                data = std::shared_ptr<T>(
                            reinterpret_cast<T*>(numa_alloc_local(capacity)),
                            [need_capacity](void * p){ numa_free(p, need_capacity); });
            } else {
#else
            {
#endif
                data = std::shared_ptr<T>(
                            reinterpret_cast<T*>(aligned_alloc(64, capacity)),
                            [](void * p){ ::free(p); });
            }
            memset(data.get(), 0, capacity);
            if (reinterpret_cast<uintptr_t>(data.get()) % 64)
                std::cout << "WARNING: data is not cache-line aligned!" << std::endl;

        }
    }

    __m512 m512_dq_scale;

    void set_scale(float quant, float dequant) {
        quant_scale = quant;
        dequant_scale = dequant;
        m512_dq_scale = _mm512_set1_ps(dequant_scale);
    }

    void operator=(const float & v) {
        for(int k = 0; k<capacity/sizeof(T); k++)
            data.get()[k] = v;
    }

    // quant from src weight and store all in data
    // quant: bf16->fp32->int8:
    //  1, q_scale = 127 / max(abs(w))
    //  2, w_i8 = round(w * q_scale)
    void quant_from(KpackedB<bfloat16>& src) {
        resize(src.K, src.N);
        auto scale = _mm512_set1_ps(quant_scale);
        auto p_src = &src(0, 0);
        auto p_dst = data.get();
        for (int k = 0; k < src.Kblocks*32; k++) {
            for (int n = 0; n < src.Nblocks*32; n += 16, p_src += 16, p_dst += 16) {
                auto a = _mm512_cvtepi16_epi32(_mm256_loadu_epi16(p_src));
                a = _mm512_slli_epi32(a, 16);
                auto a_f = _mm512_mul_ps((__m512)a, scale);
                a = _mm512_cvtps_epi32(a_f);
                auto a_256 = _mm512_cvtsepi32_epi16(a);
                auto a_128 = _mm256_cvtsepi16_epi8(a_256);
                _mm_store_si128((__m128i*)(p_dst), a_128);
            }
        }
    }

    template<int K>
    void deq_Kx32_full(int8_t *&src, bfloat16 *dst)
    {
        for (int k = 0; k < K; k++)
        {
            auto a = _mm_load_si128((__m128i *)src);        // 16 int8
            auto b = _mm_load_si128((__m128i *)(src + 16)); // 16 int8
            auto a_512 = _mm512_cvtepi8_epi32(a);           // 16 int32
            auto b_512 = _mm512_cvtepi8_epi32(b);           // 16 int32
            auto a_f = _mm512_cvtepi32_ps(a_512);           // 16 ps
            auto b_f = _mm512_cvtepi32_ps(b_512);           // 16 ps

            // dequantize, can be moved to post-process of C matrix
            a_f = _mm512_mul_ps(a_f, m512_dq_scale);   // dequantize
            b_f = _mm512_mul_ps(b_f, m512_dq_scale);   // dequantize

            auto reg_out = _mm512_cvtne2ps_pbh(b_f, a_f); // 32 packed bf16
            _mm512_store_epi32(dst, (__m512i)reg_out);    //
            src += 32;                                    // 32 int8_t dequantized into 32 bf16
            dst += 32;
        }
    };


    // dequantize is moved to post-process of C matrix
    template<int K>
    void i8_to_bf16_Kx32(int8_t *&src, bfloat16 *dst)
    {
        for (int k = 0; k < K; k++)
        {
            auto a = _mm_load_si128((__m128i *)src);        // 16 int8
            auto b = _mm_load_si128((__m128i *)(src + 16)); // 16 int8
            auto a_512 = _mm512_cvtepi8_epi32(a);           // 16 int32
            auto b_512 = _mm512_cvtepi8_epi32(b);           // 16 int32
            auto a_f = _mm512_cvtepi32_ps(a_512);           // 16 ps
            auto b_f = _mm512_cvtepi32_ps(b_512);           // 16 ps
            auto reg_out = _mm512_cvtne2ps_pbh(b_f, a_f); // 32 packed bf16
            _mm512_store_epi32(dst, (__m512i)reg_out);    //
            src += 32;                                    // 32 int8_t dequantized into 32 bf16
            dst += 32;
        }
    };

    // dequant one block 16x32 i8->bf16
    // dequant: int8->fp32->bf16
    //  1, dq_scale = max(abs(w)) / 127
    //  2, w = w_i8 * dq_scale
    void dequant16x32_to(int8_t*& src, bfloat16* dst) {
        auto dq_scale = _mm512_set1_ps(dequant_scale);
        for (int k = 0; k < 16; k++) {
            auto a = _mm_load_si128((__m128i*)src);
            auto b = _mm_load_si128((__m128i*)(src + 16));
            auto a_512 = _mm512_cvtepi8_epi32(a);
            auto b_512 = _mm512_cvtepi8_epi32(b);
            auto a_f = _mm512_cvtepi32_ps(a_512);
            auto b_f = _mm512_cvtepi32_ps(b_512);
            a_f = _mm512_mul_ps(a_f, dq_scale);
            b_f = _mm512_mul_ps(b_f, dq_scale);
            auto reg_out = _mm512_cvtne2ps_pbh(b_f, a_f);
            _mm512_store_epi32(dst, (__m512i)reg_out);
            src += 32;
            dst += 32;
        }
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
    inline void prepareB(KpackedB<bfloat16> & B, tensor2D<bfloat16> & matB, bool transpose = false) {
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

    inline void get_min_max(tensor2D<bfloat16> & matB, float& min, float& max) {
        int K = matB.dims[0];
        int N = matB.dims[1];
        auto m_max = _mm512_set1_ps(-__FLT_MAX__);
        auto m_min = _mm512_set1_ps(__FLT_MAX__);
        for (int k = 0; k < K; k++) {
            int n = 0;
            for (; n < N / 16 * 16; n += 16) {
                auto a = _mm512_cvtepi16_epi32(_mm256_loadu_epi16(&matB(k, n)));
                a = _mm512_slli_epi32(a, 16);
                m_max = _mm512_max_ps((__m512)a, m_max);
                m_min = _mm512_min_ps((__m512)a, m_min);
            }
            if (n != N) {
                __mmask16 msk = _cvtu32_mask16(0xFFFFu >> (16 - (N - n)));
                auto a = _mm512_cvtepi16_epi32(_mm256_maskz_loadu_epi16(msk, &matB(k, n)));
                a = _mm512_slli_epi32(a, 16);
                m_max = _mm512_mask_max_ps(m_max, msk, (__m512)a, m_max);
                m_min = _mm512_mask_min_ps(m_min, msk, (__m512)a, m_min);
            }
        }
        max = _mm512_reduce_max_ps(m_max);
        min = _mm512_reduce_min_ps(m_min);
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
    template<bool support_deq>
    struct Dequantizable {
        float deq_scale;
        bool do_deq = false;
        void set_deq_scale(float scale = 1.0f) {
            assert (support_deq);
            deq_scale = scale;
            do_deq = (deq_scale != 1.0f);
        }
    };

    struct Store2bf16 : Dequantizable<true> {
        tensor2D<bfloat16> & C;

        Store2bf16(tensor2D<bfloat16> & C) : C(C) {}

        FORCE_INLINE void operator()(tensor2D<float> & buffC, int m, int n, int valid_m, int valid_n) {
            // fast dynamic dispatch to polymorphic implementations
            if (do_deq)
                exec<true>(buffC, m, n, valid_m, valid_n);
            else
                exec<false>(buffC, m, n, valid_m, valid_n);
        }

        template<bool deq>
        FORCE_INLINE void exec(tensor2D<float> & buffC, int m, int n, int valid_m, int valid_n) {
            auto * psrc = &buffC(0,0);
            int8_t * pdst = reinterpret_cast<int8_t*>(&(C(m, n)));
            int stride = C.stride;
            __mmask32 k = _cvtu32_mask32(0xFFFFFFFF >> (32-valid_n));
            auto m512_deq_scale = _mm512_set1_ps(deq_scale);
            for(int i = 0; i < valid_m; i ++) {
                auto r0 = _mm512_loadu_ps(psrc);
                auto r1 = _mm512_loadu_ps(psrc + 16);
                if (deq) {
                    r0 = _mm512_mul_ps(r0, m512_deq_scale);   // dequantize
                    r1 = _mm512_mul_ps(r1, m512_deq_scale);   // dequantize
                }
                auto c = _mm512_cvtne2ps_pbh(r1, r0);
                _mm512_mask_storeu_epi16(pdst, k, c);   // 32 bf16
                pdst += stride;
                psrc += 32;
                m++;
            }
        }
    };

    struct Addbias_Store2bf16 : Dequantizable<false> {
        tensor2D<bfloat16> & C;
        float * bias;
        Addbias_Store2bf16(tensor2D<bfloat16> & C, float * bias) : C(C), bias(bias) {}

        FORCE_INLINE void operator()(tensor2D<float> & buffC, int m, int n, int valid_m, int valid_n) {
            auto * psrc = &buffC(0,0);
            int8_t * pdst = reinterpret_cast<int8_t*>(&(C(m, n)));
            int stride = C.stride;
            auto bias0 = _mm512_loadu_ps(bias + n);
            auto bias1 = _mm512_loadu_ps(bias + n + 16);
            __mmask32 k = _cvtu32_mask32(0xFFFFFFFF >> (32-valid_n));
            for(int i = 0; i < valid_m; i ++) {
                auto r0 = _mm512_loadu_ps(psrc);
                auto r1 = _mm512_loadu_ps(psrc + 16);
                r0 = _mm512_add_ps(r0, bias0);          // add bias
                r1 = _mm512_add_ps(r1, bias1);          // add bias
                auto c = _mm512_cvtne2ps_pbh(r1, r0);   // cvt2 bf16
                _mm512_mask_storeu_epi16(pdst, k, c);   // store 32 x bf16
                pdst += stride;
                psrc += 32;
            }
        }
    };

    struct Addbias_Gelu_Store2bf16 : Dequantizable<true> {
        tensor2D<bfloat16> & C;
        float * bias;

        Addbias_Gelu_Store2bf16(tensor2D<bfloat16> & C, float * bias) : C(C), bias(bias) {}

        FORCE_INLINE void operator()(tensor2D<float> & buffC, int m, int n, int valid_m, int valid_n) {
            // fast dynamic dispatch to polymorphic implementations
            if (do_deq)
                exec<true>(buffC, m, n, valid_m, valid_n);
            else
                exec<false>(buffC, m, n, valid_m, valid_n);
        }

        template<bool deq>
        FORCE_INLINE void exec(tensor2D<float> & buffC, int m, int n, int valid_m, int valid_n) {
            auto * psrc = &buffC(0,0);
            int8_t * pdst = reinterpret_cast<int8_t*>(&(C(m, n)));
            int stride = C.stride;
            auto bias0 = _mm512_loadu_ps(bias + n);
            auto bias1 = _mm512_loadu_ps(bias + n + 16);
            __mmask32 k = _cvtu32_mask32(0xFFFFFFFF >> (32-valid_n));
            auto m512_deq_scale = _mm512_set1_ps(deq_scale);
            for(int i = 0; i < valid_m; i ++) {
                auto r0 = _mm512_loadu_ps(psrc);
                auto r1 = _mm512_loadu_ps(psrc + 16);
                if (deq) {
                    r0 = _mm512_mul_ps(r0, m512_deq_scale);   // dequantize
                    r1 = _mm512_mul_ps(r1, m512_deq_scale);   // dequantize
                }
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

    struct Store2float : Dequantizable<false> {
        tensor2D<float> & C;
        Store2float(tensor2D<float> & C) : C(C) {}
        FORCE_INLINE void operator()(tensor2D<float> & buffC, int m, int n, int valid_m, int valid_n) {
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

template <int bytes, int sel=_MM_HINT_T0, int advance = 4096>
void prefetch_bytes(void *src)
{
    int8_t *p = reinterpret_cast<int8_t *>(src);
    for (int i = 0; i < bytes; i+=64)
        _mm_prefetch(p + i + advance, sel);
}
template <int... tmm>
void zero_tiles() { int dummy[sizeof...(tmm)] = {(_tile_zero(tmm), 0)...}; }

// matmul (FC)
//
// constB constrols whether it's FC or not 
// store precision for weight compression, only for BF16 AMX

template<typename T>
tensor2D<T> getSubMatB(tensor2D<T> & _matB, int n0, int n1, bool transposeB) {
    int Bd0 = transposeB ? (n1-n0) : _matB.dims[0];
    int Bd1 = transposeB ? _matB.dims[1] : (n1-n0);
    T * pbase = transposeB ? (&_matB(n0, 0)):(&_matB(0, n0));
    return tensor2D<T>(Bd0, Bd1, pbase, _matB.stride);
}

template<int bN, class F>
void loop2D_no_bM(int M, int N, F f) {
    for(int n=0; n<N; n += bN) {
        int valid_n = std::min(N - n, bN);
        f(0, n, M, valid_n);
    }
    return;
}

template<int bM, int bN, class F>
void loop2D(int M, int N, int mc, F f) {
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

// avoid M tails by shift last tile window up
// a little, with some overlapped/redundant computation
// but it works only when (M >= bM)
template<int bM, int bN, class F>
void loop2D_opt_Mtail(int M, int N, int mc, F f) {
    int tailM = (M % (mc*bM)) % bM;
    assert(M > bM);
    for(int m0=0; m0<M; m0 += mc*bM) {
        for(int n=0; n<N; n += bN) {
            int valid_n = std::min(N - n, bN);
            int mcnt = std::min(mc, ((M - m0) + bM - 1)/bM);
            for(int m1=0; m1<mcnt; m1++) {
                int m = m0 + m1*bM;
                if (M - m < bM) {
                    // shift kernel window up to make valid_m still in whole bM,
                    // avoid tail window with (valid_m < bM)
                    m = M - bM;
                }
                f(m, n, bM, valid_n);
            }
        }
    }
}

template<class T = void>
struct acc_type {};
template<>
struct acc_type <bfloat16> { typedef float type; };
template<>
struct acc_type <int8_t> { typedef int32_t type; };
template<>
struct acc_type <uint8_t> { typedef int32_t type; };

template<typename TA, typename TB, typename TC = typename acc_type<TA>::type>
struct Matmul {
    // B matrix is orgnized as tensor2D of shape axb where a=round_up_div(N, 32), b=round_up(K,32/64)*32
    // so b is size of submatrix of Kx32 composed of two columns of B0/B1 tiles.
    //tensor2D<TB> internalB;
    KpackedB<TB> internalB;

    bool constB;
    bool transposeB;

    constexpr static bool is_bf16s8 = std::is_same<TA,bfloat16>::value && std::is_same<TB,int8_t>::value;
    constexpr static bool is_bf16bf16 = std::is_same<TA,bfloat16>::value && std::is_same<TB,bfloat16>::value;
    constexpr static bool is_s8s8 = std::is_same<TA,int8_t>::value && std::is_same<TB,int8_t>::value;
    constexpr static bool is_s8u8 = std::is_same<TA,int8_t>::value && std::is_same<TB,uint8_t>::value;
    constexpr static bool is_u8s8 = std::is_same<TA,uint8_t>::value && std::is_same<TB,int8_t>::value;
    constexpr static bool is_u8u8 = std::is_same<TA,uint8_t>::value && std::is_same<TB,uint8_t>::value;
    constexpr static bool is_i8_mode = is_s8s8 || is_s8u8 || is_u8s8 || is_u8u8;

    // AMX bf16 & int8 has same M(=16) in A,C tile and same N(=16) in B tile
    // but only different K(32 vs 64) in A,C & B tiles
    constexpr static int kStep = is_i8_mode ? 64 : 32;

    // 2x2 C tiles buffer
    // most usecase requires post-processing with AVX, thus buffC
    // is used to transfer data to AVX register
    tensor2D<TC> buffC;

    Matmul(bool constB = false, bool transposeB = false) : 
        constB(constB), transposeB(transposeB), buffC(32, 32) {}

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

    // I cannot find a way to call TDP intrinsic polymophically using overload or template.
    // have to use old-macro-tricks, hopefully these compile-time checks can be optimized
    // by compiler.
#define TILE_DP(dst, a, b) \
    if (is_bf16bf16) _tile_dpbf16ps(dst, a, b); \
    if (is_s8s8) _tile_dpbssd(dst, a, b); \
    if (is_s8u8) _tile_dpbsud(dst, a, b); \
    if (is_u8s8) _tile_dpbusd(dst, a, b); \
    if (is_u8u8) _tile_dpbuud(dst, a, b);

    template<int tmmN, typename PP>
    void kernel_slimB(int M, int N,
                    tensor2D<TA> & A,
                    void * B,
                    tensor2D<TC> & buffC,
                    PP ppkernel) {
        auto * pB0 = reinterpret_cast<int8_t*>(B);
        auto * pC0 = &buffC[0];
        int8_t * pA0 = reinterpret_cast<int8_t*>(&A[0]);
        int strideA = A.stride;
        // load B tiles outside of loop
        if (tmmN > 0) _tile_loadd(2, pB0, 64); pB0 += 1024;
        if (tmmN > 1) _tile_loadd(3, pB0, 64); pB0 += 1024;
        if (tmmN > 2) _tile_loadd(4, pB0, 64); pB0 += 1024;
        if (tmmN > 3) _tile_loadd(5, pB0, 64); pB0 += 1024;
        if (tmmN > 4) _tile_loadd(6, pB0, 64); pB0 += 1024;
        if (tmmN > 5) _tile_loadd(7, pB0, 64); pB0 += 1024;
        //asm("int3");
        for(int m0 = 0; m0 < M; m0+=16) {
            int m = m0;
            pA0 += 16*A.stride;
            if (M - m0 < 16) {
                // shift up to prevent M-tails
                pA0 -= (M - m0)*A.stride;
                m = M - 16;
            }
            auto * pA = pA0;
            if (tmmN > 0) {
                _tile_loadd(1, pA, strideA); pA += 64;
                TILE_DP(0, 1, 2);
            }
            if (tmmN > 1) {
                _tile_loadd(1, pA, strideA); pA += 64;
                TILE_DP(0, 1, 3);
            }
            if (tmmN > 2) {
                _tile_loadd(1, pA, strideA); pA += 64;
                TILE_DP(0, 1, 4);
            }
            if (tmmN > 3) {
                _tile_loadd(1, pA, strideA); pA += 64;
                TILE_DP(0, 1, 5);
            }
            if (tmmN > 4) {
                _tile_loadd(1, pA, strideA); pA += 64;
                TILE_DP(0, 1, 6);
            }
            if (tmmN > 5) {
                _tile_loadd(1, pA, strideA); pA += 64;
                TILE_DP(0, 1, 7);
            }
            _tile_stored(0, pC0, buffC.stride);
            (ppkernel)(buffC, m, 0, 16, N);
        }
    }

    template<typename PP>
    void operator()(tensor2D<TA> & matA,
                    tensor2D<TB> & _matB,
                    int n0, int n1,
                    PP ppkernel) {
        auto matB = getSubMatB(_matB, n0, n1, transposeB);
        int M = matA.dims[0];
        int K = matA.dims[1];
        int N = matB.dims[transposeB ? 0 : 1];
        assert(K == matB.dims[transposeB ? 1 : 0]);

        // for non-constB, internalB is updated every time
        // for constB, internalB is updated once
        if (!constB || (internalB.capacity == 0)) {
            functional::prepareB(internalB, matB, transposeB);
        }

        // Due to the fact that we load a full tile at tails of K dimension
        // we may access memory address beyond the limit of A matrix
        // we do an preliminary check to ensure this will not cause segmentfalut.
        int Ktails = K % kStep;
        if (Ktails) {
            // the last tileload will load 64 bytes with only Ktails*sizeof(bfloat16) bytes
            // are inside A matrix, the rest is going to exceed A's range, if that's inside
            // the same page, it's OK, otherwise, we requires next page following it is valid
            // to access.
            auto * pAlast = reinterpret_cast<int8_t*>(&matA(M-1, K-1));
            auto * plast = pAlast - Ktails * sizeof(TA) + 64;
            if ((reinterpret_cast<uintptr_t>(plast) & (~4095)) != (reinterpret_cast<uintptr_t>(pAlast) & (~4095))) {
                assert(is_pointer_valid(plast)); // we cannot over-access A, implementation fails.
            }
        }

        // special case when whole B matrix can fit in 6 tiles
        // we can load B only once
        if (M >= 16 && N <= 16 && K <= 6*kStep) {
            // B is zero-padded
            // C:0
            // A:1
            // B:2,3,4,5,6,7
            auto * pB0 = reinterpret_cast<int8_t*>(internalB.data.get());
            tileconfig_t tfg(1, 0, 8, 16, 64);
            switch((K + kStep - 1)/kStep) {
                case 1: kernel_slimB<1>(M, N, matA, pB0, buffC, ppkernel); break;
                case 2: kernel_slimB<2>(M, N, matA, pB0, buffC, ppkernel); break;
                case 3: kernel_slimB<3>(M, N, matA, pB0, buffC, ppkernel); break;
                case 4: kernel_slimB<4>(M, N, matA, pB0, buffC, ppkernel); break;
                case 5: kernel_slimB<5>(M, N, matA, pB0, buffC, ppkernel); break;
                case 6: kernel_slimB<6>(M, N, matA, pB0, buffC, ppkernel); break;
                default:
                    assert(false); // impossible since (K <= 6*kStep)
            }
            return;
        }

        if (M <= 16) {
            // register/cache blocking scheme is simplified when M <= 16
            // C_MxN: 0,1
            // A_MxK: 2,
            // B_KxN: 3, 4
            tileconfig_t tfg(1, 0, {M,M,M,16,16}, 64);
            auto * pB0 = reinterpret_cast<int8_t*>(internalB.data.get());
            auto * const pC0 = &buffC[0];
            int k;
            const auto strideA = matA.stride;

            loop2D_no_bM<32>(M, N, [&](int m, int n, int valid_m, int valid_n) {
                zero_tiles<0, 1>();
                int8_t * pA0 = reinterpret_cast<int8_t*>(&matA[0]);
                for(k=0; k<K; k+=kStep) {
                    _tile_loadd(2, pA0, strideA); pA0 += 64;  // tile A Mx32/Mx64, cols is always 64
                    prefetch_bytes<1024, _MM_HINT_T1, 4096*48>(pB0);
                    _tile_loadd(3, pB0, 64); pB0 += 1024;     // tile B0 32x16(16x16x2)/64x16(16x16x4) is always 1KB
                    prefetch_bytes<1024, _MM_HINT_T1, 4096*48>(pB0);
                    _tile_loadd(4, pB0, 64); pB0 += 1024;     // tile B1 32x16(16x16x2)/64x16(16x16x4) is always 1KB
                    TILE_DP(0, 2, 3); // C0 += A*B0
                    TILE_DP(1, 2, 4); // C1 += A*B1
                }
                _tile_stored(0, pC0, buffC.stride);
                _tile_stored(1, pC0 + 16, buffC.stride);
                //int valid_n = std::min(N - n, 32);
                (ppkernel)(buffC, 0, n + n0, M, valid_n);
            });
            return;
        }

        auto kernel_2x2 = [&](int m, int n, int valid_m, int valid_n) {
            auto * pA0 = reinterpret_cast<int8_t*>(&matA(m, 0));
            auto * pA1 = reinterpret_cast<int8_t*>(&matA(m + 16, 0));
            auto strideA = matA.stride;
            auto * pB = reinterpret_cast<int8_t*>(&internalB(0, n));
            zero_tiles<0, 1, 2, 3>();
            // 2x2
            for (int k = 0; k < K; k += kStep) {
                _tile_loadd(4, pA0, strideA); pA0 += 64;
                _tile_loadd(6, pB, 64); pB += 1024;
                prefetch_bytes<1024>(pB);
                TILE_DP(0, 4, 6);

                _tile_loadd(5, pA1, strideA); pA1 += 64;
                TILE_DP(2, 5, 6);
                _tile_loadd(7, pB, 64); pB += 1024;
                prefetch_bytes<1024>(pB);
                TILE_DP(1, 4, 7);

                TILE_DP(3, 5, 7);
            }
            _tile_stored(0, &buffC(0,0), buffC.stride);
            _tile_stored(1, &buffC(0,16), buffC.stride);
            _tile_stored(2, &buffC(16,0), buffC.stride);
            _tile_stored(3, &buffC(16,16), buffC.stride);
            (ppkernel)(buffC, m, n + n0, valid_m, valid_n);
        };

        if (M <= 32 && M >16) {
            // 2x2 tile, C:0/1/2/3 A:4/5 B:6/7 no blocking along M dimension
            tileconfig_t tfg(1, 0, {16,16,M-16,M-16,16,M-16,16,16}, 64);
            loop2D_no_bM<32>(M, N, kernel_2x2);
            return;
        }

        // generic input shapes with M > 32
        // determine cache blocking scheme
        int elesz = sizeof(TA);
        int L2 = 2048*1024; // 2MB
        int slice_size = 32*rndup(K, 32)*elesz;
        int mc = std::max(1, L2/slice_size - 1);

        // M > bM
        tileconfig_t tfg(1, 0, 8, 16, 64);
        loop2D_opt_Mtail<32, 32>(M, N, mc, kernel_2x2);
    }
#if 0
    // for M < 16, use 1x2 tiles
    template<typename PP>
    void exec_Wint8_m16(tensor2D<bfloat16> & matA,
                        tensor2D<bfloat16> & _matB,
                        int n0, int n1,
                        PP ppkernel) {
        auto matB = getSubMatB(_matB, n0, n1);
        int M = matA.dims[0];
        int K = matA.dims[1];
        assert(K == matB.dims[transposeB ? 1 : 0]);
        int N = matB.dims[transposeB ? 0 : 1];

        if (!constB || (internalBI8.capacity == 0)) {
            // this dynamic quantization of weight matrix using minmax
            // is time-consuming, should be used only for constB
            if (!constB)
                std::cout << "\t WANING: dynamic quantization of weight matrix for non-constB is time-consuming " << std::endl;

            float min, max;
            amx_bf16::functional::get_min_max(_matB, min, max);
            float q, dq;
            max = std::max(std::abs(max), std::abs(min));
            q = 127 / max;
            dq = max / 127;
            internalBI8.set_scale(q, dq);

            KpackedB<bfloat16> internalTmpB;
            functional::prepareB(internalTmpB, matB, transposeB);
            internalBI8.quant_from(internalTmpB);
        }

        // dequantize scale is moved into ppkernel
        ppkernel.set_deq_scale(internalBI8.dequant_scale);

        // register/cache blocking scheme is simplified when M <= 16
        // C_MxN: 0,1
        // A_MxK: 2,
        // B_KxN: 3, 4
        auto * pBint = internalBI8.data.get();
        auto & B2buff = scratch;
        B2buff.resize(32*2, 32);
        auto * const pB = &B2buff[0];
        auto * pBsrc = pB + (32*32) * 0;
        auto * pBdst = pB + (32*32) * 1;
        internalBI8.i8_to_bf16_Kx32<32>(pBint, pBsrc);

        tileconfig_t tfg(1, 0, {M,M,M,16, 16, 16}, 64);
        auto * const pC0 = &buffC[0];
        const auto strideA = matA.stride;
        int Kbody = AKtailBuff.prepare(K);
        int k;
        constexpr int prefetch_ahead = 64*1024;
/*
        B2buff.resize((N + 31)/32*Kbody, 32);
        auto * pBcache = &B2buff[0];
        pBsrc = pBdst = pBcache;
        loop2D_no_bM(M, N, [&](int m, int n, int valid_m, int valid_n){
            for(k=0; k<Kbody; k+=64) {
                //prefetch_bytes<1024, _MM_HINT_T1, prefetch_ahead>(pBint);
                internalBI8.i8_to_bf16_Kx32<32>(pBint, pBdst);

                prefetch_bytes<1024, _MM_HINT_T1, prefetch_ahead>(pBint);
                internalBI8.i8_to_bf16_Kx32<32>(pBint, pBdst + 32*32);
            }
        });
        return;
*/
        for(int n = 0; n < N; n += 2*16) {
            // C:Mx32 = A:Mx32 x B:32x32
            zero_tiles<0, 1>();
            auto * pA0 = &matA[0];
#if 0
            for(k=0; k<Kbody; k+=32) {
                // 1x2
                _tile_loadd(2, pA0, strideA); pA0 += 32;   // tile A Mx32
                prefetch_bytes<512, _MM_HINT_T0, 4096>(pBsrc);

                _tile_loadd(3, pBsrc, 64);
                _tile_dpbf16ps(0, 2, 3); // C0 += A*B0

                prefetch_bytes<512, _MM_HINT_T0, 4096>(pBsrc);
                _tile_loadd(4, pBsrc + 16*32, 64);
                _tile_dpbf16ps(1, 2, 4); // C1 += A*B1
                pBsrc += 32*32;
            }
#endif
#if 1
            for(k=0; k<Kbody; k+=32) {
                // 1x2
                //_tile_loadd(5, pBint + 4096*12, 64); // as prefetch
                _tile_loadd(2, pA0, strideA); pA0 += 32;   // tile A Mx32
                prefetch_bytes<1024, _MM_HINT_T1, prefetch_ahead>(pBint);

                internalBI8.i8_to_bf16_Kx32<8>(pBint, pBdst);
                _tile_loadd(3, pBsrc, 64);
                internalBI8.i8_to_bf16_Kx32<8>(pBint, pBdst + 8*32);
                _tile_dpbf16ps(0, 2, 3); // C0 += A*B0

                prefetch_bytes<1024, _MM_HINT_T1, prefetch_ahead>(pBint);
                internalBI8.i8_to_bf16_Kx32<8>(pBint, pBdst + 16*32);
                _tile_loadd(4, pBsrc + 16*32, 64);
                internalBI8.i8_to_bf16_Kx32<8>(pBint, pBdst + 24*32);
                _tile_dpbf16ps(1, 2, 4); // C1 += A*B1
                std::swap(pBsrc, pBdst);
            }
            // ktail
            if (k < K) {
                auto * src = AKtailBuff.load(pA0, M, strideA);
                _tile_loadd(2, src, 64);

                prefetch_bytes<1024, _MM_HINT_T1, prefetch_ahead>(pBint);
                internalBI8.i8_to_bf16_Kx32<16>(pBint, pBdst);
                _tile_loadd(3, pBsrc, 64);
                _tile_dpbf16ps(0, 2, 3);

                prefetch_bytes<1024, _MM_HINT_T1, prefetch_ahead>(pBint);
                internalBI8.i8_to_bf16_Kx32<16>(pBint, pBdst + 16*32);
                _tile_loadd(4, pBsrc + 16*32, 64);
                _tile_dpbf16ps(1, 2, 4);
                std::swap(pBsrc, pBdst);
            }
#endif
#if 0
            for(k=0; k<Kbody; k+=32) {
                // 1x2
                _tile_loadd(2, pA0, strideA); pA0 += 32;   // tile A Mx32

                prefetch_bytes<512, _MM_HINT_T1, 4096*12>(pBint);
                internalBI8.i8_to_bf16_Kx32<16>(pBint, pB);
                _tile_loadd(3, pB, 64);
                _tile_dpbf16ps(0, 2, 3); // C0 += A*B0

                prefetch_bytes<512, _MM_HINT_T1, 4096*12>(pBint);
                internalBI8.i8_to_bf16_Kx32<16>(pBint, pB + 16*32);
                _tile_loadd(4, pB + 16*32, 64);
                _tile_dpbf16ps(1, 2, 4); // C1 += A*B1
            }
            // ktail
            if (k < K) {
                auto * src = AKtailBuff.load(pA0, M, strideA);
                _tile_loadd(2, src, 64);

                prefetch_bytes<512, _MM_HINT_T1, 4096*12>(pBint);
                internalBI8.i8_to_bf16_Kx32<16>(pBint, pB);
                _tile_loadd(3, pB, 64);
                _tile_dpbf16ps(0, 2, 3);

                prefetch_bytes<512, _MM_HINT_T1, 4096*12>(pBint);
                internalBI8.i8_to_bf16_Kx32<16>(pBint, pB + 16*32);
                _tile_loadd(4, pB + 16*32, 64);
                _tile_dpbf16ps(1, 2, 4);
            }
#endif
            //prefetch_bytes<2048, _MM_HINT_T1, prefetch_ahead>(pBint);
            _tile_stored(0, pC0, buffC.stride);
            _tile_stored(1, pC0 + 16, buffC.stride);
            //prefetch_bytes<2048, _MM_HINT_T1, prefetch_ahead>(pBint + 2048);
            int valid_n = std::min(N - n, 32);
            (ppkernel)(buffC, 0, n + n0, M, valid_n);
        }
    }
#endif
};

// specialization:
//  TA is bfloat16 and TB is int8_t, decompressed on the fly into bfloat16 by simply convert
template<>
struct Matmul<bfloat16, int8_t, float> {
    KpackedB<int8_t> internalBI8;

    // wei_buff is ping-pong buffer containing bfloat16 weights decompressed on the fly.
    tensor2D<bfloat16> weiBuff;

    bool constB;
    bool transposeB;

    constexpr static int kStep = 32;

    // 2x2 C tiles buffer
    // most usecase requires post-processing with AVX, thus buffC
    // is used to transfer data to AVX register
    tensor2D<float> buffC;

    Matmul(bool constB = false, bool transposeB = false) : 
        constB(constB), transposeB(transposeB), buffC(32, 32) {}

    template<typename PP>
    void operator()(tensor2D<bfloat16> & matA,
                    tensor2D<bfloat16> & _matB,
                    int n0, int n1,
                    PP ppkernel) {
        auto matB = getSubMatB(_matB, n0, n1, transposeB);
        int M = matA.dims[0];
        int K = matA.dims[1];
        int N = matB.dims[transposeB ? 0 : 1];
        assert(K == matB.dims[transposeB ? 1 : 0]);

        // for non-constB, internalB is updated every time
        // for constB, internalB is updated once
        if (!constB || (internalBI8.capacity == 0)) {
            // this dynamic quantization of weight matrix using minmax
            // is time-consuming, should be used only for constB
            if (!constB) {
                std::cout << "\t WANING: dynamic quantization of weight matrix for non-constB is time-consuming " << std::endl;
            }
            float min, max;
            amx_bf16::functional::get_min_max(_matB, min, max);
            float q, dq;
            max = std::max(std::abs(max), std::abs(min));
            q = 127 / max;
            dq = max / 127;
            internalBI8.set_scale(q, dq);

            KpackedB<bfloat16> internalTmpB;
            functional::prepareB(internalTmpB, matB, transposeB);
            internalBI8.quant_from(internalTmpB);
        }

        ppkernel.set_deq_scale(internalBI8.dequant_scale);

        if (M <= 16) {
            // C:0/1  A:2  B:3/4
            // dequantize scale is moved into ppkernel
            constexpr int prefetch_ahead = 64*1024;
            tileconfig_t tfg(1, 0, {M,M,M,16,16}, 64);
            auto * pBint = internalBI8.data.get();
            auto & B2buff = weiBuff;
            B2buff.resize(32*2, 32);
            auto * const pB = &B2buff[0];
            auto * pBsrc = pB + (32*32) * 0;
            auto * pBdst = pB + (32*32) * 1;
            internalBI8.i8_to_bf16_Kx32<32>(pBint, pBsrc);

            auto * const pC0 = &buffC[0];
            const auto strideA = matA.stride;
            loop2D_no_bM<32>(M, N, [&](int m, int n, int valid_m, int valid_n) {
                // C:Mx32 = A:Mx32 x B:32x32
                zero_tiles<0, 1>();
                auto * pA0 = &matA[0];
                for(int k=0; k<K; k+=32) {
                    // 1x2
                    _tile_loadd(2, pA0, strideA); pA0 += 32;   // tile A Mx32
                    prefetch_bytes<512, _MM_HINT_T1, prefetch_ahead>(pBint);

                    internalBI8.i8_to_bf16_Kx32<8>(pBint, pBdst);
                    _tile_loadd(3, pBsrc, 64);
                    internalBI8.i8_to_bf16_Kx32<8>(pBint, pBdst + 8*32);
                    _tile_dpbf16ps(0, 2, 3); // C0 += A*B0

                    prefetch_bytes<512, _MM_HINT_T1, prefetch_ahead>(pBint);
                    internalBI8.i8_to_bf16_Kx32<8>(pBint, pBdst + 16*32);
                    _tile_loadd(4, pBsrc + 16*32, 64);
                    internalBI8.i8_to_bf16_Kx32<8>(pBint, pBdst + 24*32);
                    _tile_dpbf16ps(1, 2, 4); // C1 += A*B1
                    std::swap(pBsrc, pBdst);
                }
                //prefetch_bytes<2048, _MM_HINT_T1, prefetch_ahead>(pBint);
                _tile_stored(0, pC0, buffC.stride);
                _tile_stored(1, pC0 + 16, buffC.stride);
                //prefetch_bytes<2048, _MM_HINT_T1, prefetch_ahead>(pBint + 2048);
                //int valid_n = std::min(N - n, 32);
                (ppkernel)(buffC, 0, n + n0, M, valid_n);
            });
            return;
        }

        // 4 tiles buffC is reused as decompressed bf16 weights 
        constexpr int prefetch_ahead = 16*1024;
        bfloat16 * pBa = reinterpret_cast<bfloat16*>(&buffC(0,0));
        bfloat16 * pBb = pBa + (16*32)*2;
        auto kernel_2x2 = [&](int m, int n, int valid_m, int valid_n) {
            auto strideA = matA.stride;
            auto * pA0 = &matA(m, 0);
            auto * pA1 = &matA(m + 16, 0);
            int8_t *pBint = &internalBI8(0, n);

            internalBI8.i8_to_bf16_Kx32<32>(pBint, pBb);

            zero_tiles<0, 1, 2, 3>();
            for (int k = 0; k < K; k += 32) {
                internalBI8.i8_to_bf16_Kx32<16>(pBint, pBa);

                _tile_loadd(4, pA0 + k, strideA);
                _tile_loadd(6, pBb, 64);
                _tile_dpbf16ps(0, 4, 6);

                _tile_loadd(5, pA1 + k, strideA);
                _tile_dpbf16ps(2, 5, 6);

                internalBI8.i8_to_bf16_Kx32<16>(pBint, pBa + 16*32);

                _tile_loadd(7, pBb + 16*32, 64);
                _tile_dpbf16ps(1, 4, 7);
                _tile_dpbf16ps(3, 5, 7);

                std::swap(pBa, pBb);
            }
            _tile_stored(0, &buffC(0,0), buffC.stride);
            _tile_stored(1, &buffC(0,16), buffC.stride);
            _tile_stored(2, &buffC(16,0), buffC.stride);
            _tile_stored(3, &buffC(16,16), buffC.stride);
            (ppkernel)(buffC, m, n + n0, valid_m, valid_n);
        };

        if (M <= 32 && M > 16) {
            // 2x2 C:0/1/2/3 A:4/5  B:6/7
            tileconfig_t tfg(1, 0, {16, 16, M-16, M-16, 16, M-16, 16, 16}, 64);
            loop2D_no_bM<32>(M, N, kernel_2x2);
            return;
        }

        // determine blocking scheme
        int elesz = sizeof(uint16_t);
        int L2 = 2048*1024; // 2MB
        int slice_size = 32*rndup(K, 32)*elesz;
        int mc = std::max(1, L2/slice_size - 1); // if 1 32xK slice cannot fit L2, use 1 slice at least

        // main loop
        tileconfig_t tfg(1, 0, 8, 16, 64);
        loop2D_opt_Mtail<32, 32>(M, N, mc, kernel_2x2);
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

