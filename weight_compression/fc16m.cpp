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


#include "timeit.hpp"
#include "misc.hpp"
#include "test_bw.hpp"

#include <omp.h>

#include "wcompress.hpp"

int OMP_NT = omp_thread_count();
auto &___x = std::cout << ANSIcolor("31") << "OMP_NT = " << OMP_NT << ANSIcolor() << std::endl; 
static bool initAMX = initXTILE();


void cache_flush(void * p, int sz) {
    int8_t * data = reinterpret_cast<int8_t*>(p);
    for(int i = 0; i<sz; i+=64) {
        _mm_clflush(data + i);
    }
}

// https://raw.githubusercontent.com/intel/perfmon/main/SPR/events/sapphirerapids_core.json
timeit benchmark(
    {
        {PERF_TYPE_RAW, 0x3c, "CPU_CLK_UNHALTED.THREAD"},
        //{PERF_TYPE_RAW, 0x81d0, "MEM_LOAD_RETIRED.ALL_LOADS"},
        //{PERF_TYPE_HW_CACHE, 0x10002, "LLC_load_misses"},
        //{PERF_TYPE_HW_CACHE, 0x2, "LLC_loads"},
        //{PERF_TYPE_RAW, 0x02b1, "UOPS_EXECUTED.CORE"},
    }
);


/*===========================================
  for problem  C_MxN = A_MxK * B_KxN with M<16
  we can setup tileconfig to make A,C tiles with less rows
===========================================*/
float quantize_i8_scale = 2;
tensor2D<bfloat16> A;
tensor2D<bfloat16> B;
tensor2D<bfloat16> Bpacked1x2;
tensor2D<bfloat16> Bpacked1x4;
tensor2D<int8_t> Bi8packed1x2;
tensor2D<float> C_ref;
tensor2D<float> C;

// L = 1 ... 4
template<class T, int L>
tensor2D<T> repackB_1xL(const tensor2D<bfloat16> &Bi, float scale=1.0f) {
    int K = Bi.dims[0];
    int N = Bi.dims[1];

    assert(K % 32 == 0);
    assert(N % (L*16) == 0);

    tensor2D<T> Bo;
    // repacked as (?*16)x32
    Bo.resize(K*N/32, 32, true);
    T * dst = &Bo[0];

    auto repack_1tile = [&](int k0, int n0) {
        //std::cout << "k0,n0="<<k0 << "," << n0 << std::endl;
        auto * src0 = &Bi(k0, n0);
        auto * src1 = &Bi(k0+1, n0);
        for(int k = 0; k<32; k+=2) {
            for(int n = 0; n<16; n++) {
                *dst++ = src0[n] * scale;
                *dst++ = src1[n] * scale;
            }
            src0 += Bi.stride;
            src1 += Bi.stride;
        }
    };

    // loop in computation order
    for(int n0 = 0; n0 < N; n0 += L*16) {
        // reduce on dim K
        for(int k=0; k<K; k+=32) {
            // 1xL
            for(int l=0; l<L; l++) {
                repack_1tile(k, n0 + l*16);
            }
        }
    }
    return Bo;
}

void prepare_data(int M, int K, int N) {
    B.resize(K, N);
    B.fill_rnd();
    A.resize(M, K);
    A.fill_rnd();
    C.resize(M, N);

    // reference result
    C_ref.resize(M, N);
    for(int m = 0; m<M; m++) {
        for(int n=0; n<N; n++) {
            float sum = 0;
            for(int k=0; k<K; k++) {
                sum += A(m,k) * B(k,n);
            }
            C_ref(m,n) = sum;
        }
    }

    // prepack B into 1x2 1x4 ...
    Bpacked1x2 = repackB_1xL<bfloat16,2>(B);
    Bpacked1x4 = repackB_1xL<bfloat16,4>(B);
    Bi8packed1x2 = repackB_1xL<int8_t,2>(B, quantize_i8_scale);
}

void compare_with_ref() {
    int M = A.dims[0];
    int K = A.dims[1];
    int N = C_ref.dims[1];
    assert(C.is_normal());
    if (C_ref == C) {
        std::cout << ANSIcolor("1;32") << "Match!\n" << ANSIcolor();
    } else {
        std::cout << ANSIcolor("1;31") << "Mismatch!\n" << ANSIcolor();
        //std::cout << C_ref << std::endl;
        //std::cout << C << std::endl;
    }
}

void fc16m_base_1x2() {
     // assume each tile of B is already packed as 16x(16x2)
    int M = A.dims[0];
    int K = A.dims[1];
    int N = C_ref.dims[1];
    C.resize(M, N);

    // tiles allocation
    // C: 0,1,2,3
    // A: 4,
    // B: 5,6,7
    tileconfig_t tfg(1, 0, {M,M,M,M,M,16,16,16}, 64);

    zero_tiles<0, 1, 2, 3>();
    load_tiles_with_random_bf16<4, 5>();

    // CPU_CLK_UNHALTED.THREAD = 4508 for M,K,N = 2,16*32,192
    // so inner-most loop body took 4508/(192/32)/16 = 47 cycles
    // on average, theoratical throughput needs simulation to tell
    // exactly(since throughput of instructions may hide each other)
    // but we can know:
    //     if AMX ALU is fully utilized, the throughput should be 16*2 = 32
    //     if AMX load is fully utilized, the throughput should be 8*3 = 24
    //
    //
    benchmark.tag(__func__, M, K, N)([&]()  {
        auto * pB0 = &Bpacked1x2[0];
        auto * pC0 = &C[0];
        for(int n0 = 0; n0 < N; n0 += 2*16) {
            // reduce on dim K
            zero_tiles<0, 1>();
            auto * pA0 = &A[0];
            auto Astride = A.stride;
            for(int k=0; k<K; k+=32) {
                // 1x2
                _tile_loadd(4, pA0, Astride); pA0 += 32;   // tile A Mx32
                prefetch_bytes<1024, _MM_HINT_T0, 4096>(pB0);
                _tile_loadd(6, pB0, 64); pB0 += 16*32;     // tile B 16x32
                prefetch_bytes<1024, _MM_HINT_T0, 4096>(pB0);
                _tile_loadd(7, pB0, 64); pB0 += 16*32;     // tile B 16x32
                _tile_dpbf16ps(0, 4, 6); // C0 += A*B0
                _tile_dpbf16ps(1, 4, 7); // C1 += A*B1
            }
            // no post ops, just store
            _tile_stored(0, pC0 + n0, C.stride);
            _tile_stored(1, pC0 + n0 + 16, C.stride);
        }
        }, Bpacked1x2.capacity);
    
    compare_with_ref();
}

void fc16m_base_1x4() {
     // assume each tile of B is already packed as 16x(16x2)
    int M = A.dims[0];
    int K = A.dims[1];
    int N = C_ref.dims[1];
    C.resize(M, N);

    // tiles allocation
    // C: 0,1,2,3
    // A: 4,
    // B: 5,6,7
    tileconfig_t tfg(1, 0, {M,M,M,M,M,16,16,16}, 64);

    zero_tiles<0, 1, 2, 3>();
    load_tiles_with_random_bf16<4, 5>();
    benchmark.tag(__func__, M, K, N)([&]()
                            {
        auto * pB0 = &Bpacked1x4[0];
        auto * pC0 = &C[0];
        
        for(int n0 = 0; n0 < N; n0 += 4*16) {
            // reduce on dim K
            zero_tiles<0, 1, 2, 3>();
            auto * pA0 = &A[0];
            auto Astride = A.stride;
            for(int k=0; k<K; k+=32) {
                // 1x2
                _tile_loadd(4, pA0, Astride); pA0 += 32;   // tile A Mx32
                prefetch_bytes<1024, _MM_HINT_T0, 4096>(pB0);
                _tile_loadd(6, pB0, 64); pB0 += 16*32;     // tile B 16x32
                prefetch_bytes<1024, _MM_HINT_T0, 4096>(pB0);
                _tile_loadd(7, pB0, 64); pB0 += 16*32;     // tile B 16x32
                _tile_dpbf16ps(0, 4, 6); // C0 += A*B0
                _tile_dpbf16ps(1, 4, 7); // C1 += A*B1
                prefetch_bytes<1024, _MM_HINT_T0, 4096>(pB0);
                _tile_loadd(6, pB0, 64); pB0 += 16*32;     // tile B 16x32
                prefetch_bytes<1024, _MM_HINT_T0, 4096>(pB0);
                _tile_loadd(7, pB0, 64); pB0 += 16*32;     // tile B 16x32
                _tile_dpbf16ps(2, 4, 6); // C1 += A*B1
                _tile_dpbf16ps(3, 4, 7); // C1 += A*B1
            }
            // no post ops, just store
            _tile_stored(0, pC0 + n0, C.stride);
            _tile_stored(1, pC0 + n0 + 16, C.stride);
            _tile_stored(2, pC0 + n0 + 32, C.stride);
            _tile_stored(3, pC0 + n0 + 48, C.stride);
        }
        }, Bpacked1x4.capacity);
    
    compare_with_ref();
}
//===========================================================================================
void matmul16m_base(tensor2D<bfloat16> & A,
                    tensor2D<bfloat16> & _Bpacked1x2,
                    tensor2D<float> & C) {
    int M = A.dims[0];
    int K = A.dims[1];
    int N = C.dims[1];
    // tiles allocation
    // C: 0,1
    // A: 2
    // B: 3,4
    tileconfig_t tfg(1, 0, {M,M,M,16,16}, 64);
    auto * pB0 = &_Bpacked1x2[0];
    auto * pC0 = &C[0];

    for(int n0 = 0; n0 < N; n0 += 2*16) {
        // C:Mx32 = A:Mx32 x B:32x32
        zero_tiles<0, 1>();
        auto * pA0 = &A[0];
        auto Astride = A.stride;
        for(int k=0; k<K; k+=32) {
            // 1x2
            _tile_loadd(2, pA0, Astride); pA0 += 32;   // tile A Mx32
            prefetch_bytes<1024, _MM_HINT_T1, 4096*48>(pB0);
            _tile_loadd(3, pB0, 64); pB0 += 16*32;     // tile B 16x32
            prefetch_bytes<1024, _MM_HINT_T1, 4096*48>(pB0);
            _tile_loadd(4, pB0, 64); pB0 += 16*32;     // tile B 16x32
            _tile_dpbf16ps(0, 2, 3); // C0 += A*B0
            _tile_dpbf16ps(1, 2, 4); // C1 += A*B1
        }
        _tile_stored(0, pC0 + n0, C.stride);
        _tile_stored(1, pC0 + n0 + 16, C.stride);
    }
}



static auto deq_packed_dq_scale = _mm512_set1_ps(1.0f/quantize_i8_scale);


template<int K>
void deq_Kx32_full(int8_t *&src, bfloat16 *dst)
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

        // dequantize, moved to post-process of C matrix
        a_f = _mm512_mul_ps(a_f, deq_packed_dq_scale);   // dequantize
        b_f = _mm512_mul_ps(b_f, deq_packed_dq_scale);   // dequantize

        auto reg_out = _mm512_cvtne2ps_pbh(b_f, a_f); // 32 packed bf16
        _mm512_store_epi32(dst, (__m512i)reg_out);    //
        src += 32;                                    // 32 int8_t dequantized into 32 bf16
        dst += 32;
    }
};

template<int K>
void deq_Kx32(int8_t *&src, bfloat16 *dst)
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

        // dequantize, moved to post-process of C matrix
        //a_f = _mm512_mul_ps(a_f, deq_packed_dq_scale);   // dequantize
        //b_f = _mm512_mul_ps(b_f, deq_packed_dq_scale);   // dequantize

        auto reg_out = _mm512_cvtne2ps_pbh(b_f, a_f); // 32 packed bf16
        _mm512_store_epi32(dst, (__m512i)reg_out);    //
        src += 32;                                    // 32 int8_t dequantized into 32 bf16
        dst += 32;
    }
};

template<int K>
inline void fake_deq_Kx32(int8_t *&src, bfloat16 *dst)
{
    for (int k = 0; k < K; k += 2)
    {
        auto a = _mm512_load_si512((__m512i *)src); // read 32 bf16
        _mm512_store_si512(dst, a);
        _mm512_store_si512(dst + 32, a);
        src += 64;
        dst += 32 * 2;
    }
};


void scale_Kx16(int K, void * dst, int stride, __m512 scale)
{
    auto * p = reinterpret_cast<int8_t*>(dst);
    for (int k = 0; k < K; k++)
    {
        auto a_f = _mm512_loadu_ps(p);      // 16 float
        a_f = _mm512_mul_ps(a_f, scale);    // dequantize
        _mm512_storeu_ps(p, a_f);
        p += stride;
    }
};


void matmul16m_Wint80(tensor2D<bfloat16> & A,
                    tensor2D<int8_t> & Bpacked1x2,
                    tensor2D<float> & C) {
    static thread_local tensor2D<bfloat16> B2buff(16*2, 32);

    int M = A.dims[0];
    int K = A.dims[1];
    int N = C.dims[1];
    
    // tiles allocation
    // C: 0,1
    // A: 2
    // B: 3,4
    tileconfig_t tfg(1, 0, {M,M,M,16,16}, 64);
    auto * pBint = &Bpacked1x2[0];
    auto * pC0 = &C[0];
    auto Astride = A.stride;

    for(int n0 = 0; n0 < N; n0 += 2*16) {
        // C:Mx32 = A:Mx32 x B:32x32
        zero_tiles<0, 1>();
        auto * pA0 = &A[0];
        auto * pB = &B2buff[0];
        
        for(int k=0; k<K; k+=32) {
            // 1x2
            _tile_loadd(2, pA0, Astride); pA0 += 32;   // tile A Mx32

            prefetch_bytes<512, _MM_HINT_T1, 4096*12>(pBint);
            deq_Kx32_full<16>(pBint, pB);
            _tile_loadd(3, pB, 64);
            _tile_dpbf16ps(0, 2, 3); // C0 += A*B0

            prefetch_bytes<512, _MM_HINT_T1, 4096*12>(pBint);
            deq_Kx32_full<16>(pBint, pB + 16*32);
            _tile_loadd(4, pB + 16*32, 64);
            _tile_dpbf16ps(1, 2, 4); // C1 += A*B1
        }
        _tile_stored(0, pC0 + n0, C.stride);
        _tile_stored(1, pC0 + n0 + 16, C.stride);
    }
}

void matmul16m_Wint8(tensor2D<bfloat16> & A,
                    tensor2D<int8_t> & Bpacked1x2,
                    tensor2D<float> & C) {
    static thread_local tensor2D<bfloat16> B2buff(16*2, 32);

    int M = A.dims[0];
    int K = A.dims[1];
    int N = C.dims[1];
    
    // tiles allocation
    // C: 0,1
    // A: 2
    // B: 3,4
    tileconfig_t tfg(1, 0, {M,M,M,16,16}, 64);
    auto * pBint = &Bpacked1x2[0];
    auto * pC0 = &C[0];
    auto Astride = A.stride;

    for(int n0 = 0; n0 < N; n0 += 2*16) {
        // C:Mx32 = A:Mx32 x B:32x32
        zero_tiles<0, 1>();
        auto * pA0 = &A[0];
        auto * pB = &B2buff[0];
        
        for(int k=0; k<K; k+=32) {
            // 1x2
            _tile_loadd(2, pA0, Astride); pA0 += 32;   // tile A Mx32

            prefetch_bytes<512, _MM_HINT_T1, 4096*12>(pBint);
            deq_Kx32<16>(pBint, pB);
            _tile_loadd(3, pB, 64);
            _tile_dpbf16ps(0, 2, 3); // C0 += A*B0

            prefetch_bytes<512, _MM_HINT_T1, 4096*12>(pBint);
            deq_Kx32<16>(pBint, pB + 16*32);
            _tile_loadd(4, pB + 16*32, 64);
            _tile_dpbf16ps(1, 2, 4); // C1 += A*B1
        }
        _tile_stored(0, pC0 + n0, C.stride);
        scale_Kx16(M, pC0 + n0, C.stride, deq_packed_dq_scale);

        _tile_stored(1, pC0 + n0 + 16, C.stride);
        scale_Kx16(M, pC0 + n0 + 16, C.stride, deq_packed_dq_scale);        
    }
}

void matmul16m_Wint8B(tensor2D<bfloat16> & A,
                      tensor2D<int8_t> & Bpacked1x2,
                      tensor2D<float> & C) {
    static thread_local tensor2D<bfloat16> B2buff(32*2, 32);

    int M = A.dims[0];
    int K = A.dims[1];
    int N = C.dims[1];
    
    // tiles allocation
    // C: 0,1,2,3
    // A: 4
    // B: 5,6,7
    tileconfig_t tfg(1, 0, {M,M,M,M,M,16,16,16}, 64);
    auto * pBint = &Bpacked1x2[0];
    auto * pC0 = &C[0];
    auto Astride = A.stride;
    auto * pB = &B2buff[0];
    int off = 0;
    
    auto * pBsrc = pB + (32*32) * 0;
    auto * pBdst = pB + (32*32) * 1;
    deq_Kx32<32*1>(pBint, pBsrc);

    for(int n0 = 0; n0 < N; n0 += 2*16) {
        // C:Mx32 = A:Mx32 x B:32x32
        zero_tiles<0, 1>();
        auto * pA0 = &A[0];
        for(int k=0; k<K; k+=32) {
            // 1x2
            prefetch_bytes<512, _MM_HINT_T1, 4096*48>(pBint);
            _tile_loadd(4, pA0, Astride); pA0 += 32;   // tile A Mx32

            deq_Kx32<8>(pBint, pBdst);
            _tile_loadd(5, pBsrc, 64);
            deq_Kx32<8>(pBint, pBdst + 8*32);
            _tile_dpbf16ps(0, 4, 5); // C0 += A*B0

            prefetch_bytes<512, _MM_HINT_T1, 4096*48>(pBint);
            deq_Kx32<8>(pBint, pBdst + 16*32);
            _tile_loadd(6, pBsrc + 16*32, 64);
            deq_Kx32<8>(pBint, pBdst + 24*32);
            _tile_dpbf16ps(1, 4, 6); // C1 += A*B1
            std::swap(pBsrc, pBdst);
        }
        _tile_stored(0, pC0 + n0, C.stride);
        scale_Kx16(M, pC0 + n0, C.stride, deq_packed_dq_scale);

        _tile_stored(1, pC0 + n0 + 16, C.stride);
        scale_Kx16(M, pC0 + n0 + 16, C.stride, deq_packed_dq_scale);
    }
}

void fc16m_test_all(int M, int K, int N) {
    double BmatrixSize = K * N * sizeof(bfloat16);
    std::cout << "# K = " << K / 32 << "*32 = " << K << ", sizeof(B)=" << pretty_size(BmatrixSize, "B") << std::endl;
    std::cout << " preparing data ..." << std::flush;
    prepare_data(M, K, N);
    std::cout << "\r                 \r" << std::flush;

    // 1x4 is only very little faster than 1x2 (since A matrix is in cache)
    fc16m_base_1x2();
    fc16m_base_1x4();
 
    // matmul16m_base is only little slower than 1x2 due to more initializations
    benchmark.tag("matmul16m_base", M, K, N)([&]() { matmul16m_base(A, Bpacked1x2, C);}, Bpacked1x2.capacity); compare_with_ref();

    benchmark.tag("matmul16m_Wint8", M, K, N)([&]() { matmul16m_Wint8(A, Bi8packed1x2, C);}, Bi8packed1x2.capacity); compare_with_ref();
    benchmark.tag("matmul16m_Wint8B", M, K, N)([&]() { matmul16m_Wint8B(A, Bi8packed1x2, C);}, Bi8packed1x2.capacity); compare_with_ref();
}


struct mt_context {
    tensor2D<bfloat16> A;
    tensor2D<bfloat16> Bpacked1x2;
    tensor2D<int8_t> Bi8packed1x2;
    tensor2D<float> C;
};


// parallel w/o dependency
void fc16m_test_mt0(int M, int K, int N) {
    double BmatrixSize = K * N * sizeof(bfloat16);
    std::cout << "# K = " << K / 32 << "*32 = " << K << ", sizeof(B)=" << pretty_size(BmatrixSize, "B") << std::endl;
    std::cout << " preparing data ..." << std::flush;
    prepare_data(M, K, N);
    std::cout << "\r                 \r" << std::flush;

    std::vector<mt_context> ctx(OMP_NT);

    auto check_results = [&]() {
        int errors = 0;
        #pragma omp parallel reduction(+:errors)
        {
            int i = omp_get_thread_num();
            if (C_ref == ctx[i].C) {
            } else {
                errors++;
                //std::cout << C_ref << std::endl;
                //std::cout << C << std::endl;
            }
        }
        if (errors == 0) {
            std::cout << ANSIcolor("1;32") << "Match!\n" << ANSIcolor();
        } else {
            std::cout << ANSIcolor("1;31") << "Mismatch!\n" << ANSIcolor();
        }
    };

    #pragma omp parallel
    {
        int i = omp_get_thread_num();
        ctx[i].A = A.clone();
        ctx[i].Bpacked1x2 = Bpacked1x2.clone();
        ctx[i].Bi8packed1x2 = Bi8packed1x2.clone();
        ctx[i].C = C.clone();
    }

    //benchmark.set_time_ms(456434);
    benchmark.tag("matmul16m_base", "ST", OMP_NT, M, K, N)([&]() {
            int i = omp_get_thread_num();
            matmul16m_base(ctx[i].A, ctx[i].Bpacked1x2, ctx[i].C);
        },
        Bpacked1x2.capacity);

    benchmark.tag("matmul16m_base", "MT", OMP_NT, M, K, N)([&]() {
            #pragma omp parallel
            {
                int i = omp_get_thread_num();
                matmul16m_base(ctx[i].A, ctx[i].Bpacked1x2, ctx[i].C);
            }
        },
        Bpacked1x2.capacity * OMP_NT); check_results();

    benchmark.tag("matmul16m_Wint8B", "ST", OMP_NT, M, K, N)([&]() {
            int i = omp_get_thread_num();
            matmul16m_Wint8B(ctx[i].A, ctx[i].Bi8packed1x2, ctx[i].C);
        },
        Bi8packed1x2.capacity); check_results();

    benchmark.tag("matmul16m_Wint8B", "MT", OMP_NT, M, K, N)([&]() {
            #pragma omp parallel
            {
                int i = omp_get_thread_num();
                matmul16m_Wint8B(ctx[i].A, ctx[i].Bi8packed1x2, ctx[i].C);
            }
        },
        Bi8packed1x2.capacity * OMP_NT); check_results();
}

//================================================================
// multi-threaded version, split B matrix along N dimensions
// this has state which stores splitted B matrix for each worker thread
template <typename T, typename Q>
inline void splitter(const T& n, const Q& team, const Q& tid, T& n_start, T& n_end) {
    if (team <= 1 || n == 0) {
        n_start = 0;
        n_end = n;
    } else {
        T n1 = (n + (T)team - 1) / (T)team;
        T n2 = n1 - 1;
        T T1 = n - n2 * (T)team;
        n_end = (T)tid < T1 ? n1 : n2;
        n_start = (T)tid <= T1 ? tid * n1 : T1 * n1 + ((T)tid - T1) * n2;
    }

    n_end += n_start;
}

template<class WType>
struct MultiThreaded {

    struct context {
        tensor2D<WType> Wpacked1x2;
        int n0;
        int n1;
    };
    std::vector<context> Ctx;

    MultiThreaded(tensor2D<bfloat16> & W, float scale = 1.0f) {
        // split W along N dims, in uint of 32
        int K = W.dims[0];
        int N = W.dims[1];
        int work_amount = (N + 31)/32;
        Ctx.resize(OMP_NT);
        // for each splitted W, packed into 1x2 and store continously
        #pragma omp parallel
        {
            int t = omp_get_thread_num();
            int start, end;
            splitter(work_amount, OMP_NT, t, start, end);
            int n0 = start*32;
            int n1 = end*32;
            if (n1 > N) n1 = N;
            Ctx[t].Wpacked1x2 = repackB_1xL<WType,2>(tensor2D<bfloat16>(K, n1-n0, &W(0, n0), W.stride), scale);
            Ctx[t].n0 = n0;
            Ctx[t].n1 = n1;
        }
    }

    template<typename Kernel>
    void operator()(tensor2D<bfloat16> & A, tensor2D<float> & C, Kernel & kern) {
        #pragma omp parallel
        {
            int t = omp_get_thread_num();
            int M = A.dims[0];
            int n0 = Ctx[t].n0;
            int n1 = Ctx[t].n1;
            tensor2D<float> subC(M, n1-n0, &C(0, n0), C.stride);
            kern(A, Ctx[t].Wpacked1x2, subC);
        }
    }
};

void fc16m_test_mt1(int M, int K, int N) {
    double BmatrixSize = K * N * sizeof(bfloat16);
    std::cout << "# K = " << K / 32 << "*32 = " << K << ", sizeof(B)=" << pretty_size(BmatrixSize, "B") << std::endl;
    std::cout << " preparing data ..." << std::flush;
    prepare_data(M, K, N);
    std::cout << "\r                 \r" << std::flush;

    MultiThreaded<bfloat16> mmmt_bf16(B);
    MultiThreaded<int8_t> mmmt_i8(B, quantize_i8_scale);

    benchmark.tag("mmmt_bf16", OMP_NT, M, K, N)([&]() {
            mmmt_bf16(A, C, matmul16m_base);
        },
        Bpacked1x2.capacity); compare_with_ref();

    benchmark.tag("mmmt_int8", OMP_NT, M, K, N)([&]() {
            mmmt_i8(A, C, matmul16m_Wint8B);
        },
        Bi8packed1x2.capacity); compare_with_ref();
}

// multilayer
void fc16m_test_mt2(int M, int K, int N, int L) {
    double BmatrixSize = K * N * sizeof(bfloat16);
    std::cout << "# K = " << K / 32 << "*32 = " << K << ", sizeof(B) * " << L << " =" << pretty_size(BmatrixSize * L, "B") << std::endl;
    std::cout << " preparing data ..." << std::flush;
    prepare_data(M, K, N);
    std::cout << "\r                 \r" << std::flush;

    for(int i=0;i<2;i++)
    {
        std::vector<std::shared_ptr<MultiThreaded<bfloat16>>> mmmt;
        for(int l=0; l<L; l++) mmmt.emplace_back(new MultiThreaded<bfloat16>(B));
        benchmark.tag("mmmt_bf16", OMP_NT, M, K, N, L)([&]() {
            for(int l=0; l<L; l++)
                (*mmmt[l])(A, C, matmul16m_base);
        },
        Bpacked1x2.capacity * L);
    }
    for(int i=0;i<2;i++)
    {
        std::vector<std::shared_ptr<MultiThreaded<int8_t>>> mmmt;
        for(int l=0; l<L; l++) mmmt.emplace_back(new MultiThreaded<int8_t>(B, quantize_i8_scale));
        benchmark.tag("mmmt_int80", OMP_NT, M, K, N, L)([&]() {
            for(int l=0; l<L; l++)
                (*mmmt[l])(A, C, matmul16m_Wint80);
        },
        Bi8packed1x2.capacity * L);
    }
    for(int i=0;i<2;i++)
    {
        std::vector<std::shared_ptr<MultiThreaded<int8_t>>> mmmt;
        for(int l=0; l<L; l++) mmmt.emplace_back(new MultiThreaded<int8_t>(B, quantize_i8_scale));
        benchmark.tag("mmmt_int8", OMP_NT, M, K, N, L)([&]() {
            for(int l=0; l<L; l++)
                (*mmmt[l])(A, C, matmul16m_Wint8);
        },
        Bi8packed1x2.capacity * L);
    }
    for(int i=0;i<2;i++)
    {
        std::vector<std::shared_ptr<MultiThreaded<int8_t>>> mmmt;
        for(int l=0; l<L; l++) mmmt.emplace_back(new MultiThreaded<int8_t>(B, quantize_i8_scale));
        benchmark.tag("mmmt_i8B", OMP_NT, M, K, N, L)([&]() {
            for(int l=0; l<L; l++)
                (*mmmt[l])(A, C, matmul16m_Wint8B);
        },
        Bi8packed1x2.capacity * L);
    }
}

int main()
{
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

    // test 1000 ms
    benchmark.set_time_ms(-10000);
    benchmark.set_unit("B/s");
    benchmark.set_peak_metric_per_second(19e9); // 19 GB/s

    // accuracy test
    //fc16m_test_all(7, 64, 128);

    // L1D
    //fc16m_test_all(2, 80 * 32, 192);
    //fc16m_test_mt0(2, 80 * 32, 192);

    // sizeof(B)=50MB    memory bandwidth can reach 24GB(L3)
    // sizeof(B)=100MB+, with prefetch, memory bandwidth can reach 19GB
    // for INT8 weight compression also access ext-DDR, we need 200MB+
    //fc16m_test_all(2, 80 * 32, 81920);
    //fc16m_test_mt0(2, 80 * 32, 81920);

    //fc16m_test_mt1(2, 80*32, 81920);
    fc16m_test_mt2(2, 80*32, 10752, 40);
}
