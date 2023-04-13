#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <vector>
#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>
#include <cstring>
#include <thread>

#include "kernels_amxbf16.hpp"
#include "kernels_avx512.hpp"
#include "thread_pool.hpp"
#include "timeit.hpp"
#include "misc.hpp"
#include "test_bw.hpp"

#include <omp.h>
timeit benchmark;

int OMP_NT = omp_thread_count();
static bool initAMX = initXTILE();

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

template <int bytes, int advance = 4096>
void prefetch_bytes(void *src)
{
    int8_t *p = reinterpret_cast<int8_t *>(src);
    int cachelines = bytes / 64;
    for (int i = 0; i < cachelines; i++)
        _mm_prefetch(p + i * 64 + advance, _MM_HINT_T0);
}

static auto dequant_16x32_dq_scale = _mm512_set1_ps(0.2f);
// 2 bf16 tiles buffer for B matrix dequntized on-the-fly

inline void dequant_16x32(int8_t *&src, bfloat16 *dst)
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

void unittest_base(int K)
{
    tensor2D<bfloat16> B(K, 32); // assume each tile of B is already packed as 16x(16x2)
    zero_tiles<0, 1, 2, 3>();
    load_tiles_with_random_bf16<4, 5>();
    benchmark.tag(__func__)([&]()
                            {
        auto * pB0 = &B[0];
        for(int k = 0; k < K; k+=32) {
            // the order of load & tdp is not so important
            _tile_loadd(6, pB0, 64); pB0 += 16*32;     // 1KB tile
            prefetch_bytes<1024>(pB0);
            _tile_dpbf16ps(0, 4, 6);
            _tile_dpbf16ps(1, 5, 6);
            _tile_loadd(7, pB0, 64);  pB0 += 16*32;    // 1KB tile
            prefetch_bytes<1024>(pB0);
            _tile_dpbf16ps(2, 4, 7);
            _tile_dpbf16ps(3, 5, 7);
        } },
                            2048.0 * K / 32);
    check_tiles<0, 1, 2, 3>();
}

void unittest_halfB(int K)
{
    tensor2D<bfloat16> B(K, 32); // assume each tile of B is already packed as 16x(16x2)
    zero_tiles<0, 1, 2, 3>();
    load_tiles_with_random_bf16<4, 5>();
    benchmark.tag(__func__)([&]()
                            {
        auto * pB0 = &B[0];
        for(int k = 0; k < K; k+=32) {
            _tile_loadd(6, pB0, 64); pB0 += 16*32;
            prefetch_bytes<1024>(pB0); // prefetch 1KB tile from next 4KB page
            //_tile_loadd(B1, pB0, 64);  pB0 += 16*32;
            _tile_dpbf16ps(0, 4, 6);
            _tile_dpbf16ps(1, 5, 6);
            _tile_dpbf16ps(2, 4, 6);
            _tile_dpbf16ps(3, 5, 6);
        } },
                            1024.0 * K / 32);
    check_tiles<0, 1, 2, 3>();
}

void unittest_Wint8(int K)
{
    tensor2D<int8_t> B(K, 32, true); // assume each tile of B is already packed as 16x(16x2)
    tensor2D<bfloat16> B2buff(16 * 2, 32);
    auto *pB0 = &B2buff(0, 0);
    auto *pB1 = &B2buff(16, 0);
    zero_tiles<0, 1, 2, 3>();
    load_tiles_with_random_bf16<4, 5>();
    benchmark.tag(__func__)([&]()
                            {
        auto * pBint = &B[0];
        for(int k = 0; k < K; k+=32) {
            dequant_16x32(pBint, pB0); // 512 bytes int8 => 1KB tile bf16
            prefetch_bytes<512>(pBint); // prefetch tile from next 4KB page
            _tile_loadd(6, pB0, 64);
            _tile_dpbf16ps(0, 4, 6);
            _tile_dpbf16ps(1, 5, 6);

            dequant_16x32(pBint, pB1);
            prefetch_bytes<512>(pBint);
            _tile_loadd(7, pB1, 64);
            _tile_dpbf16ps(2, 4, 7);
            _tile_dpbf16ps(3, 5, 7);
        } },
                            1024.0 * K / 32);
    check_tiles<0, 1, 2, 3>();
}

void unittest_Wint8B(int K)
{
    tensor2D<int8_t> B(K + 32, 32, true); // assume each tile of B is already packed as 16x(16x2)
    tensor2D<bfloat16> B2buff(16 * 4, 32);

    auto *pB0 = &B2buff(0, 0);
    auto *pB1 = &B2buff(16 * 2, 0);

    zero_tiles<0, 1, 2, 3>();
    load_tiles_with_random_bf16<4, 5>();
    benchmark.tag(__func__)([&]()
                            {
        auto * pBint = &B[0];

        dequant_16x32(pBint, pB1);
        dequant_16x32(pBint, pB1 + 16*32);

        for(int k = 0; k < K; k+=32) {
            dequant_16x32(pBint, pB0); // 512 bytes int8 => 1KB tile bf16
            prefetch_bytes<512>(pBint); // prefetch tile from next 4KB page
            _tile_loadd(6, pB1, 64);
            _tile_dpbf16ps(0, 4, 6);
            _tile_dpbf16ps(1, 5, 6);

            dequant_16x32(pBint, pB0 + 16*32);
            prefetch_bytes<512>(pBint);
            _tile_loadd(7, pB1 + 16*32, 64);
            _tile_dpbf16ps(2, 4, 7);
            _tile_dpbf16ps(3, 5, 7);
            std::swap(pB0, pB1);
        } },
                            1024.0 * K / 32);
    check_tiles<0, 1, 2, 3>();
}

void unittest_Wint8C(int K)
{
    tensor2D<int8_t> B(K, 32, true);         // assume each tile of B is already packed as 16x(16x2)
    tensor2D<bfloat16> B4buff(64, 32, true); // assume each tile of B is already packed as 16x(16x2)

    auto *pB0 = &B4buff(0, 0);
    auto *pB1 = &B4buff(16, 0);
    auto *pB2 = &B4buff(16 * 2, 0);
    auto *pB3 = &B4buff(16 * 3, 0);

    zero_tiles<0, 1, 2, 3>();
    load_tiles_with_random_bf16<4, 5>();
    benchmark.tag(__func__)([&]()
                            {
        auto * pBint = &B[0];
        for(int k = 0; k < K; k+=32) {
            dequant_16x32(pBint, pB0); // 512 bytes int8 => 1KB tile bf16
            dequant_16x32(pBint, pB1);
            prefetch_bytes<1024, 4096>(pBint);

            _tile_loadd(6, pB0, 64);
            _tile_dpbf16ps(0, 4, 6);
            _tile_dpbf16ps(1, 5, 6);

            _tile_loadd(7, pB1, 64);
            _tile_dpbf16ps(2, 4, 7);
            _tile_dpbf16ps(3, 5, 7);
        } },
                            1024.0 * K / 32);
    check_tiles<0, 1, 2, 3>();
}

void unittest_WFakeint8(int K)
{
    tensor2D<bfloat16> B(K, 32, true);
    tensor2D<bfloat16> B2buff(16 * 2, 32);
    auto *pB0 = &B2buff(0, 0);
    auto *pB1 = &B2buff(16, 0);
    zero_tiles<0, 1, 2, 3>();
    load_tiles_with_random_bf16<4, 5>();
    benchmark.tag(__func__)(
        [&]()
        {
        auto * pBint = reinterpret_cast<int8_t*>(&B[0]);
        for(int k = 0; k < K; k+=32) {
            fake_dequant_i8_16x32(pBint, pB0);  // 512 bytes => 1KB tile
            prefetch_bytes<512>(pBint);
            _tile_loadd(6, pB0, 64);
            _tile_dpbf16ps(0, 4, 6);
            _tile_dpbf16ps(1, 5, 6);

            fake_dequant_i8_16x32(pBint, pB1);  // 512 bytes => 1KB tile
            prefetch_bytes<512>(pBint);
            _tile_loadd(7, pB1, 64);
            _tile_dpbf16ps(2, 4, 7);
            _tile_dpbf16ps(3, 5, 7);
        } },
        1024.0 * K / 32);
    check_tiles<0, 1, 2, 3>();
}


void unittest_WFakeint8B(int K)
{
    tensor2D<bfloat16> B(K+32, 32, true);
    tensor2D<bfloat16> B4buff(16 * 4, 32);
    auto *pB0 = &B4buff(0, 0);
    auto *pB1 = &B4buff(16*2, 0);
    zero_tiles<0, 1, 2, 3>();
    load_tiles_with_random_bf16<4, 5>();
    benchmark.tag(__func__)(
        [&]()
        {
        auto * pBint = reinterpret_cast<int8_t*>(&B[0]);
        fake_dequant_i8_16x32(pBint, pB1);
        fake_dequant_i8_16x32(pBint, pB1 + 16*32);
        for(int k = 0; k < K; k+=32) {
            fake_dequant_i8_16x32(pBint, pB0);  // 512 bytes => 1KB tile
            prefetch_bytes<512>(pBint);
            _tile_loadd(6, pB1, 64);
            _tile_dpbf16ps(0, 4, 6);
            _tile_dpbf16ps(1, 5, 6);

            fake_dequant_i8_16x32(pBint, pB0 + 16*32);  // 512 bytes => 1KB tile
            prefetch_bytes<512>(pBint);
            _tile_loadd(7, pB1 + 16*32, 64);
            _tile_dpbf16ps(2, 4, 7);
            _tile_dpbf16ps(3, 5, 7);
            std::swap(pB0, pB1);
        } },
        1024.0 * K / 32);
    check_tiles<0, 1, 2, 3>();
}

void unittest_WFakeint4(int K)
{
    tensor2D<bfloat16> B(K, 32, true);
    tensor2D<bfloat16> B2buff(16 * 2, 32);
    auto *pB0 = &B2buff(0, 0);
    auto *pB1 = &B2buff(16, 0);
    zero_tiles<0, 1, 2, 3>();
    load_tiles_with_random_bf16<4, 5>();
    benchmark.tag(__func__)([&]()
                            {
        auto * pBint = reinterpret_cast<int8_t*>(&B[0]);
        for(int k = 0; k < K; k+=32) {
            fake_dequant_i4_16x32(pBint, pB0); // 256Bytes=>1KB
            prefetch_bytes<256>(pBint);
            _tile_loadd(6, pB0, 64);
            _tile_dpbf16ps(0, 4, 6);
            _tile_dpbf16ps(1, 5, 6);

            fake_dequant_i4_16x32(pBint, pB1); // 256Bytes=>1KB
            prefetch_bytes<256>(pBint);
            _tile_loadd(7, pB1, 64);
            _tile_dpbf16ps(2, 4, 7);
            _tile_dpbf16ps(3, 5, 7);
        } },
                            512.0 * K / 32);
    check_tiles<0, 1, 2, 3>();
}

void unittest_WFakeint4B(int K)
{
    tensor2D<bfloat16> B(K, 32, true);
    tensor2D<bfloat16> B2buff(16 * 4, 32);
    auto *pB0 = &B2buff(0, 0);
    auto *pB1 = &B2buff(16*2, 0);
    zero_tiles<0, 1, 2, 3>();
    load_tiles_with_random_bf16<4, 5>();
    benchmark.tag(__func__)([&]() {
        auto * pBint = reinterpret_cast<int8_t*>(&B[0]);
        fake_dequant_i4_16x32(pBint, pB0); 
        fake_dequant_i4_16x32(pBint, pB0 + 16*32);
        for(int k = 0; k < K; k+=32) {
            fake_dequant_i4_16x32(pBint, pB0); // 256Bytes=>1KB
            prefetch_bytes<256>(pBint);
            _tile_loadd(6, pB1, 64);
            _tile_dpbf16ps(0, 4, 6);
            _tile_dpbf16ps(1, 5, 6);

            fake_dequant_i4_16x32(pBint, pB0 + 16*32); // 256Bytes=>1KB
            prefetch_bytes<256>(pBint);
            _tile_loadd(7, pB1 + 16*32, 64);
            _tile_dpbf16ps(2, 4, 7);
            _tile_dpbf16ps(3, 5, 7);
            std::swap(pB0, pB1);
        }}, 512.0 * K / 32);
    check_tiles<0, 1, 2, 3>();
}

void unittest_WFakeint4C(int K)
{
    tensor2D<bfloat16> B(K, 32, true);
    tensor2D<bfloat16> B3buff(32, 32, true);
    tensor2D<bfloat16> B4buff(16 * 4, 32, true);
    auto *pB0 = &B4buff(0, 0);
    auto *pB1 = &B4buff(16, 0);
    auto *pB2 = &B4buff(16 * 2, 0);
    auto *pB3 = &B4buff(16 * 3, 0);

    auto *pB30 = &B3buff(0, 0);
    auto *pB31 = &B3buff(16, 0);

    zero_tiles<0, 1, 2, 3>();
    load_tiles_with_random_bf16<4, 5>();
    benchmark.tag(__func__)([&]()
                            {
        auto * pBint = reinterpret_cast<int8_t*>(&B[0]);
        for(int k = 0; k < K; k+=64) {
            fake_dequant_i4_16x32(pBint, pB0); // 256Bytes=>1KB
            fake_dequant_i4_16x32(pBint, pB1); // 256Bytes=>1KB
            fake_dequant_i4_16x32(pBint, pB2); // 256Bytes=>1KB
            fake_dequant_i4_16x32(pBint, pB3); // 256Bytes=>1KB
            prefetch_bytes<256*4>(pBint);
            _tile_loadd(6, pB0, 64);
            _tile_dpbf16ps(0, 4, 6);
            _tile_dpbf16ps(1, 5, 6);

            _tile_loadd(7, pB1, 64);
            _tile_dpbf16ps(2, 4, 7);
            _tile_dpbf16ps(3, 5, 7);

            _tile_loadd(6, pB2, 64);
            _tile_dpbf16ps(0, 4, 6);
            _tile_dpbf16ps(1, 5, 6);

            _tile_loadd(7, pB3, 64);
            _tile_dpbf16ps(2, 4, 7);
            _tile_dpbf16ps(3, 5, 7);

        } },
                            512.0 * K / 32);
    check_tiles<0, 1, 2, 3>();
}

void unittest_WFakeint4D(int K)
{
    tensor2D<bfloat16> B(K, 32, true);
    tensor2D<bfloat16> B2buff(16 * 2, 32);
    auto *pB0 = &B2buff(0, 0);
    auto *pB1 = &B2buff(16, 0);
    zero_tiles<0, 1, 2, 3>();
    load_tiles_with_random_bf16<4, 5>();
    benchmark.tag(__func__)([&]()
                            {
        auto * pBint = reinterpret_cast<int8_t*>(&B[0]);
        for(int k = 0; k < K; k+=32) {
            fake_dequant_i4_16x32(pBint, pB0); // 256Bytes=>1KB
            for (int k = 0; k < 4; k++) _mm_prefetch(pBint + k*64 + 4096, _MM_HINT_T0);
            _tile_loadd(6, pB0, 64);
            _tile_dpbf16ps(0, 4, 6);
            _tile_dpbf16ps(1, 5, 6);

            fake_dequant_i4_16x32(pBint, pB1); // 256Bytes=>1KB
            for (int k = 0; k < 4; k++) _mm_prefetch(pBint + k*64 + 4096, _MM_HINT_T0);
            _tile_loadd(7, pB1, 64);
            _tile_dpbf16ps(2, 4, 7);
            _tile_dpbf16ps(3, 5, 7);
        } },
                            512.0 * K / 32);
    check_tiles<0, 1, 2, 3>();
}

void unittest_avx512_base(int K)
{
    tensor2D<bfloat16> A(32, 32);
    tensor2D<bfloat16> B(K, 32);

    auto *pA0 = &A(0, 0);
    auto *pA1 = &A(16, 0);

    __m512 rC[16];
    for (int i = 0; i < 16; i++)
        rC[i] = _mm512_setzero();

    auto avx512_bf16_dp = [&rC](bfloat16 *srcA, bfloat16 *&srcB)
    {
        for (int i = 0; i < 16; i++)
        {
            // prefetch next 4KB page, w/o this we cannot reach max BW when
            // memory footprint is bigger than L2
            _mm_prefetch(srcB + (64 + i) * 32, _MM_HINT_T0);
            auto rA = _mm512_load_si512(srcA);
            auto rB = _mm512_load_si512(srcB);
            rC[i] = _mm512_dpbf16_ps(rC[i], (__m512bh)rA, (__m512bh)rB);
            srcA += 32;
            srcB += 32;
        }
    };

    benchmark.tag(__func__)([&]()
                            {
        // this is not equivalent to _tile_dpbf16ps, just go over whole B matrix
        // with avx512 instructions
        auto * pB = &B[0];
        for(int k = 0; k < K; k+=32) {
            avx512_bf16_dp(pA0, pB);
            avx512_bf16_dp(pA1, pB);
        } },
                            K * 64);

    tensor2D<float> C(16, 16);
    for (int i = 0; i < 16; i++)
        _mm512_store_ps(&C(i, 0), rC[i]);
    if (!C.is_normal())
        std::cout << ANSIcolor("1;31") << "Error!" << ANSIcolor() << std::endl;
}

void unittest_avx512_fakedq(int K)
{
    tensor2D<bfloat16> A(32, 32);
    tensor2D<bfloat16> B(K, 32);
    tensor2D<bfloat16> B2buff(16 * 2, 32);

    auto *pA0 = &A(0, 0);
    auto *pA1 = &A(16, 0);
    auto *pB0 = &B2buff(0, 0);
    auto *pB1 = &B2buff(16, 0);

    __m512 rC[16];
    for (int i = 0; i < 16; i++)
        rC[i] = _mm512_setzero();

    auto avx512_bf16_dp = [&rC](bfloat16 *srcA, bfloat16 *srcB)
    {
        for (int i = 0; i < 16; i++)
        {
            // load A, B
            //_mm_prefetch(srcB + (16 + i)*32, _MM_HINT_T0);
            auto rA = _mm512_load_si512(srcA);
            auto rB = _mm512_load_si512(srcB);
            rC[i] = _mm512_dpbf16_ps(rC[i], (__m512bh)rA, (__m512bh)rB);
            srcA += 32;
            srcB += 32;
        }
    };

    benchmark.tag(__func__)([&]()
                            {
        auto * pBint = reinterpret_cast<int8_t*>(&B[0]);
        for(int k = 0; k < K; k+=32) {
            fake_dequant_i4_16x32(pBint, pB0);  // 1/4 KB => 1KB tile
            avx512_bf16_dp(pA0, pB0);

            fake_dequant_i4_16x32(pBint, pB1);  // 1/4 KB => 1KB tile
            avx512_bf16_dp(pA1, pB1);
        } },
                            (1024 / 4) * 2 * K / 32);

    tensor2D<float> C(16, 16);
    for (int i = 0; i < 16; i++)
        _mm512_store_ps(&C(i, 0), rC[i]);
    if (!C.is_normal())
        std::cout << ANSIcolor("1;31") << "Error!" << ANSIcolor() << std::endl;
}

int amx_unit_test_int8(int K)
{
    tileconfig_t tfg(1, 0, 8, 16, 64);
    double BmatrixSize = K * 32 * sizeof(bfloat16);
    std::cout << "# K = " << K / 32 << "*32 = " << K << ", sizeof(B)=" << pretty_size(BmatrixSize, "B") << std::endl;
    unittest_base(K);
    unittest_halfB(K);
    unittest_Wint8(K);
    unittest_Wint8B(K);
    unittest_Wint8C(K);
    unittest_WFakeint8(K);
    unittest_WFakeint8B(K);
    unittest_WFakeint4(K);
    unittest_WFakeint4B(K);
    unittest_WFakeint4C(K);
    unittest_WFakeint4D(K);
    unittest_avx512_base(K);
    unittest_avx512_fakedq(K);
    return 0;
}

int main()
{
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    std::cout << ANSIcolor("31") << "omp_get_num_threads() = " << omp_get_num_threads() << std::endl
              << ANSIcolor();
    std::cout << ANSIcolor("31") << "OMP_NT = " << OMP_NT << std::endl
              << ANSIcolor();

    // test 1000 ms
    benchmark.set_time_ms(-1000);
    benchmark.set_unit("B/s");
    benchmark.set_peak_metric_per_second(18e9); // 18 GB/s

    amx_unit_test_int8(102400 * 32);
    // amx_unit_test_int8(80*32);
    amx_unit_test_int8(16 * 32);
}