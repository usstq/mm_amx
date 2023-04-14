#pragma once

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

using ov::bfloat16;


// A non-type template parameter pack named as tmm
template <int... tmm>
void check_tiles()
{
    tensor2D<float> C(16, 16);
    auto check_C = [&](int i)
    {
        if (!C.is_normal())
            std::cout << ANSIcolor("1;31") << "Error!" << ANSIcolor() << std::endl;
        return 0;
    };
    int dummy[sizeof...(tmm)] = {
        (
            _tile_stored(tmm, &C[0], C.stride), // store to C
            check_C(tmm),                       // check C
            0)...};
}

template <int... tmm>
void zero_tiles() { int dummy[sizeof...(tmm)] = {(_tile_zero(tmm), 0)...}; }
template <int... tmm>
void load_tiles_with_random_bf16()
{
    tensor2D<bfloat16> A(16, 32);
    int dummy[sizeof...(tmm)] = {(A.fill_rnd(), _tile_loadd(tmm, &A[0], 64), 0)...};
}

template <int bytes, int advance = 4096*12>
void prefetch_bytes(void *src)
{
    int8_t *p = reinterpret_cast<int8_t *>(src);
    int cachelines = bytes / 64;
    for (int i = 0; i < cachelines; i++)
        _mm_prefetch(p + i * 64 + advance, _MM_HINT_T1);
}

static auto dequant_16x32_dq_scale = _mm512_set1_ps(0.2f);
// 2 bf16 tiles buffer for B matrix dequntized on-the-fly

// CPU_CLK_UNHALTED.THREAD measure shows
// throughput of this loop body is:
//   4.25 cycles when 2 _mm512_mul_ps are removed
//   5.25 cycles when 2 _mm512_mul_ps are not removed
// 68/84 cycles per call
template<int K>
void dequant_Kx32(int8_t *&src, bfloat16 *dst)
{
    for (int k = 0; k < K; k++)
    {
        auto a = _mm_load_si128((__m128i *)src);        // 16 int8
        auto b = _mm_load_si128((__m128i *)(src + 16)); // 16 int8
        auto a_512 = _mm512_cvtepi8_epi32(a);           // 16 int32
        auto b_512 = _mm512_cvtepi8_epi32(b);           // 16 int32
        //auto a_512 = _mm512_loadu_ps((__m512 *)src);
        //auto b_512 = _mm512_loadu_ps((__m512 *)src + 64);

        auto a_f = _mm512_cvtepi32_ps(a_512);           // 16 ps
        auto b_f = _mm512_cvtepi32_ps(b_512);           // 16 ps

        //a_f = _mm512_mul_ps(a_f, dequant_16x32_dq_scale);   // dequantize
        //b_f = _mm512_mul_ps(b_f, dequant_16x32_dq_scale);   // dequantize
        auto reg_out = _mm512_cvtne2ps_pbh(b_f, a_f); // 32 packed bf16
        _mm512_store_epi32(dst, (__m512i)reg_out);    //
        src += 32;                                    // 32 int8_t dequantized into 32 bf16
        dst += 32;
    }
};
inline void dequant_16x32(int8_t *&src, bfloat16 *dst)
{
    dequant_Kx32<16>(src, dst);
};

inline void dequant_i4_16x32(int8_t *&src, bfloat16 *dst)
{
    for (int k = 0; k < 16; k++)
    {
        auto a = _mm_load_si128((__m128i *)src);        // 16 int8
        auto b = _mm_load_si128((__m128i *)(src + 16)); // 16 int8
        auto a_512 = _mm512_cvtepi8_epi32(a);           // 16 int32
        auto b_512 = _mm512_cvtepi8_epi32(b);           // 16 int32
        auto a_f = _mm512_cvtepi32_ps(a_512);           // 16 ps
        auto b_f = _mm512_cvtepi32_ps(b_512);           // 16 ps
        // a_f = _mm512_mul_ps(a_f, dequant_16x32_dq_scale);   // dequantize
        // b_f = _mm512_mul_ps(b_f, dequant_16x32_dq_scale);   // dequantize

        auto reg_out = _mm512_cvtne2ps_pbh(b_f, a_f); // 32 packed bf16
        _mm512_store_epi32(dst, (__m512i)reg_out);    //
        src += 32;                                    // 32 int8_t dequantized into 32 bf16
        dst += 32;
    }
};

inline void fake_dequant_i8_16x32(int8_t *&src, bfloat16 *dst)
{
    for (int k = 0; k < 16; k += 2)
    {
        auto a = _mm512_load_si512((__m512i *)src); // read 32 bf16
        _mm512_store_si512(dst, a);
        _mm512_store_si512(dst + 32, a);
        src += 64;
        dst += 32 * 2;
    }
};

inline void fake_dequant_i4_16x32(int8_t *&src, bfloat16 *dst)
{
    for (int k = 0; k < 16; k += 4)
    {
        auto a = _mm512_load_si512((__m512i *)src); // read 32 bf16
        _mm512_store_si512(dst, a);
        _mm512_store_si512(dst + 32, a);
        _mm512_store_si512(dst + 32 * 2, a);
        _mm512_store_si512(dst + 32 * 3, a);
        src += 64;
        dst += 32 * 4;
    }
};
