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

//#include "kernels_amx.hpp"
//#include "kernels_avx512.hpp"
#include "tensor2D.hpp"
#include "timeit.hpp"

#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#include <stdlib.h>

#include <omp.h>


timeit timer(
    {
        {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, "HW_CYCLES"},
        //{PERF_TYPE_RAW, 0x3c, "CPU_CLK_UNHALTED.THREAD"},
        //{PERF_TYPE_RAW, 0x81d0, "MEM_LOAD_RETIRED.ALL_LOADS"},
        //{PERF_TYPE_HW_CACHE, 0x10002, "LLC_load_misses"},
        //{PERF_TYPE_HW_CACHE, 0x2, "LLC_loads"},
        //{PERF_TYPE_RAW, 0x02b1, "UOPS_EXECUTED.CORE"},
    }
);

//================================================================================
// initialize AMX
static bool initAMX = initXTILE();


template<class T = void>
struct acc_type {};
template<>
struct acc_type <ov::bfloat16> { typedef float type; };
template<>
struct acc_type <float> { typedef float type; };
template<>
struct acc_type <int8_t> { typedef int32_t type; };
template<>
struct acc_type <uint8_t> { typedef int32_t type; };

template<class T>
using acc_type_t = typename acc_type<T>::type;

template<typename C=void>
void zero_tiles() {
}

template <int t0, int... tmm>
void zero_tiles() {
    _tile_zero(t0);
    zero_tiles<tmm...>();
}

template<typename TA, typename TB, typename TC = acc_type_t<TA>>
struct MatmulVector {
    MatmulVector() {}
    constexpr static bool is_bf16s8 = std::is_same<TA,ov::bfloat16>::value && std::is_same<TB,int8_t>::value;
    constexpr static bool is_bf16bf16 = std::is_same<TA,ov::bfloat16>::value && std::is_same<TB,ov::bfloat16>::value;
    constexpr static bool is_s8s8 = std::is_same<TA,int8_t>::value && std::is_same<TB,int8_t>::value;
    constexpr static bool is_s8u8 = std::is_same<TA,int8_t>::value && std::is_same<TB,uint8_t>::value;
    constexpr static bool is_u8s8 = std::is_same<TA,uint8_t>::value && std::is_same<TB,int8_t>::value;
    constexpr static bool is_u8u8 = std::is_same<TA,uint8_t>::value && std::is_same<TB,uint8_t>::value;
    constexpr static bool is_i8_mode = is_s8s8 || is_s8u8 || is_u8s8 || is_u8u8;
    constexpr static int kStep = is_i8_mode ? 64 : 32;

#define TILE_DP(dst, a, b) \
    if (is_bf16bf16) _tile_dpbf16ps(dst, a, b); \
    if (is_s8s8) _tile_dpbssd(dst, a, b); \
    if (is_s8u8) _tile_dpbsud(dst, a, b); \
    if (is_u8s8) _tile_dpbusd(dst, a, b); \
    if (is_u8u8) _tile_dpbuud(dst, a, b);

    alignas(64) int8_t KtailBuff[64];

    template<int tmmN, bool bFallbackKtails>
    void kernel(int M, int K, const void * pA, int strideA, const void * vB, void * vC) {
        static_assert(tmmN >= 1 && tmmN <= 6, "tmmN must be within [1-6] range");
        const auto * pA0 = reinterpret_cast<const int8_t*>(pA);
        int KLastOffBytes = (K - kStep) * sizeof(TA); 
        const auto * pB0 = reinterpret_cast<const int8_t*>(vB);
        auto * pC0 = reinterpret_cast<int8_t*>(vC);

        const auto * pBLast = pB0 + 64*(tmmN - 1);
        int Ktail = K & (kStep - 1);
        if (Ktail) {
            if (bFallbackKtails) {
                // if bContainMtails, the last submatrix needs to use special to prevent A matrix read overflow
                // K tails is handled by:
                //  - zero-padding the last tile of vector B, at the top
                //  - right-align last tile load from matA
                __mmask64 kmask = _cvtu64_mask64(0xFFFFFFFFFFFFFFFFull << (kStep - Ktail)*sizeof(TB));
                auto r = _mm512_maskz_loadu_epi8(kmask, pB0 + KLastOffBytes);
                _mm512_storeu_epi8(KtailBuff, r);
            } else {
                // each row of A can be read overflow w/o worrying NaN numbers
                // zero-padding the last tile of vector B as bottom is enough 
                __mmask64 kmask = _cvtu64_mask64(0xFFFFFFFFFFFFFFFFull >> (kStep - Ktail)*sizeof(TB));
                KLastOffBytes = (K - Ktail)*sizeof(TA);
                auto r = _mm512_maskz_loadu_epi8(kmask, pB0 + KLastOffBytes);
                _mm512_storeu_epi8(KtailBuff, r);
            }
            pBLast = KtailBuff;
        }

        // load B tiles outside of loop
        if (tmmN == 1) {
            _tile_loadd(2, pB0, 4);
        }
        if (tmmN == 2) {
            _tile_loadd(2, pB0, 4);
            _tile_loadd(3, pBLast, 4);
        }
        if (tmmN == 3) {
            _tile_loadd(2, pB0, 4);
            _tile_loadd(3, pB0 + 64, 4);
            _tile_loadd(4, pBLast, 4);
        }
        if (tmmN == 4) {
            _tile_loadd(2, pB0, 4);
            _tile_loadd(3, pB0 + 64, 4);
            _tile_loadd(4, pB0 + 64*2, 4);
            _tile_loadd(5, pBLast, 4);
        }
        if (tmmN == 5) {
            _tile_loadd(2, pB0, 4);
            _tile_loadd(3, pB0 + 64, 4);
            _tile_loadd(4, pB0 + 64*2, 4);
            _tile_loadd(5, pB0 + 64*3, 4);
            _tile_loadd(6, pBLast, 4);
        }
        if (tmmN == 6) {
            _tile_loadd(2, pB0, 4);
            _tile_loadd(3, pB0 + 64, 4);
            _tile_loadd(4, pB0 + 64*2, 4);
            _tile_loadd(5, pB0 + 64*3, 4);
            _tile_loadd(6, pB0 + 64*4, 4);
            _tile_loadd(7, pBLast, 4);
        }
        //asm("int3");
        for(int m = 0; m < M; m+=16) {
            zero_tiles<0>();
            if (tmmN == 1) {
                _tile_loadd(1, pA0, strideA); TILE_DP(0, 1, 2);
            }
            if (tmmN == 2) {
                _tile_loadd(1, pA0, strideA); TILE_DP(0, 1, 2);
                _tile_loadd(1, pA0 + KLastOffBytes, strideA); TILE_DP(0, 1, 3);
            }
            if (tmmN == 3) {
                _tile_loadd(1, pA0, strideA); TILE_DP(0, 1, 2);
                _tile_loadd(1, pA0 + 64, strideA); TILE_DP(0, 1, 3);
                _tile_loadd(1, pA0 + KLastOffBytes, strideA);  TILE_DP(0, 1, 4);
            }
            if (tmmN == 4) {
                _tile_loadd(1, pA0, strideA); TILE_DP(0, 1, 2);
                _tile_loadd(1, pA0 + 64, strideA); TILE_DP(0, 1, 3);
                _tile_loadd(1, pA0 + 128, strideA);  TILE_DP(0, 1, 4);
                _tile_loadd(1, pA0 + KLastOffBytes, strideA); TILE_DP(0, 1, 5);
            }
            if (tmmN == 6) {
                _tile_loadd(1, pA0, strideA); TILE_DP(0, 1, 2);
                _tile_loadd(1, pA0 + 64, strideA); TILE_DP(0, 1, 3);
                _tile_loadd(1, pA0 + 128, strideA);  TILE_DP(0, 1, 4);
                _tile_loadd(1, pA0 + 192, strideA); TILE_DP(0, 1, 5);
                _tile_loadd(1, pA0 + KLastOffBytes, strideA); TILE_DP(0, 1, 6);
            }
            if (tmmN == 7) {
                _tile_loadd(1, pA0, strideA); TILE_DP(0, 1, 2);
                _tile_loadd(1, pA0 + 64, strideA); TILE_DP(0, 1, 3);
                _tile_loadd(1, pA0 + 128, strideA);  TILE_DP(0, 1, 4);
                _tile_loadd(1, pA0 + 192, strideA); TILE_DP(0, 1, 5);
                _tile_loadd(1, pA0 + 256, strideA); TILE_DP(0, 1, 6);
                _tile_loadd(1, pA0 + KLastOffBytes, strideA); TILE_DP(0, 1, 7);
            }
            _tile_stored(0, pC0, 4); pC0 += 16*4;   // C is single column, always take 4 bytes
            pA0 += 16 * strideA;
        }
    }

    void operator()(tensor2D<TA> & matA, const TB * vB, TC * vC) {
        int M = matA.dims[0];
        int K = matA.dims[1];
        TA * pA = &matA[0];
        int strideA = matA.stride;

        // M tails is handled
        assert(K >= kStep && K <= 6*kStep);

        int Ktail = K & (kStep - 1);
        int Mtail = M & (16 - 1);
        int Mbody = M - Mtail;
        int numBtiles = (K + kStep - 1)/kStep;

        // if we have Ktails, then it will always be handled in Mtail, so we split
        // Mtail out even if it's zero
        if (Ktail) {
            if (Mtail == 0) {
                Mtail = 16;
                Mbody -= 16;
            }
        }

        if (Mbody) {
            tileconfig_t tfg(1, 0, {
                {16, 4},  // C:0   M x 1     (4b)
                {16, 64}, // A:1   M x 32/64 (64b)
                {16, 4}, // B:2   32/64 x 1 (4b)
                {16, 4}, // B:3
                {16, 4}, // B:4
                {16, 4}, // B:5
                {16, 4}, // B:6
                {16, 4}, // B:7
            });
            // Ktail fallback will always be done at Mtails loop
            switch(numBtiles) {
                case 1: kernel<1, false>(Mbody, K, pA, strideA, vB, vC); break;
                case 2: kernel<2, false>(Mbody, K, pA, strideA, vB, vC); break;
                case 3: kernel<3, false>(Mbody, K, pA, strideA, vB, vC); break;
                case 4: kernel<4, false>(Mbody, K, pA, strideA, vB, vC); break;
                case 5: kernel<5, false>(Mbody, K, pA, strideA, vB, vC); break;
                case 6: kernel<6, false>(Mbody, K, pA, strideA, vB, vC); break;
                default:
                    assert(false); // impossible since (K <= 6*kStep)
            }
        }

        if (Mtail) {
            pA = &matA(Mbody, 0);
            tileconfig_t tfg(1, 0, {
                {Mtail, 4},   // C:0   M x 1     (4b)
                {Mtail, 64},  // A:1   M x 32/64 (64b)
                {16, 4}, // B:2   32/64 x 1 (4b)
                {16, 4}, // B:3
                {16, 4}, // B:4
                {16, 4}, // B:5
                {16, 4}, // B:6
                {16, 4}, // B:7
            });
            if (Ktail) {
                switch(numBtiles) {
                    case 1: kernel<1, true>(Mtail, K, pA, strideA, vB, vC + Mbody); break;
                    case 2: kernel<2, true>(Mtail, K, pA, strideA, vB, vC + Mbody); break;
                    case 3: kernel<3, true>(Mtail, K, pA, strideA, vB, vC + Mbody); break;
                    case 4: kernel<4, true>(Mtail, K, pA, strideA, vB, vC + Mbody); break;
                    case 5: kernel<5, true>(Mtail, K, pA, strideA, vB, vC + Mbody); break;
                    case 6: kernel<6, true>(Mtail, K, pA, strideA, vB, vC + Mbody); break;
                    default:
                        assert(false); // impossible since (K <= 6*kStep)
                }
            } else {
                switch(numBtiles) {
                    case 1: kernel<1, false>(Mtail, K, pA, strideA, vB, vC + Mbody); break;
                    case 2: kernel<2, false>(Mtail, K, pA, strideA, vB, vC + Mbody); break;
                    case 3: kernel<3, false>(Mtail, K, pA, strideA, vB, vC + Mbody); break;
                    case 4: kernel<4, false>(Mtail, K, pA, strideA, vB, vC + Mbody); break;
                    case 5: kernel<5, false>(Mtail, K, pA, strideA, vB, vC + Mbody); break;
                    case 6: kernel<6, false>(Mtail, K, pA, strideA, vB, vC + Mbody); break;
                    default:
                        assert(false); // impossible since (K <= 6*kStep)
                }
            }
        }
    }
};

using ov::bfloat16;

inline __m512 cvt_bf16_to_fp32(const __m256i src) {
    __m512i y = _mm512_cvtepu16_epi32(src);
    return _mm512_castsi512_ps(_mm512_slli_epi32(y, 16));
}

inline __m512 mm512_uni_loadu_ps(const ov::bfloat16* a) {
    auto vec_bf16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
    return cvt_bf16_to_fp32(vec_bf16);
}

inline __m512 mm512_uni_loadu_ps(const float* a) {
    return _mm512_loadu_ps(a);
}

#define vec_len_f32_avx512 16

template<typename TA, typename TB>
static float dot_product(TA* a, TB* b, size_t n) {
    size_t i = 0;
    float sum = 0.0f;
    auto vsum0 = _mm512_setzero_ps();
    auto vsum1 = _mm512_setzero_ps();
    auto vsum2 = _mm512_setzero_ps();
    auto vsum3 = _mm512_setzero_ps();
    for (; i + 4 * vec_len_f32_avx512 <= n; i += 4 * vec_len_f32_avx512) {
        auto va0 = mm512_uni_loadu_ps(a + i);
        auto va1 = mm512_uni_loadu_ps(a + i + vec_len_f32_avx512);
        auto va2 = mm512_uni_loadu_ps(a + i + vec_len_f32_avx512 * 2);
        auto va3 = mm512_uni_loadu_ps(a + i + vec_len_f32_avx512 * 3);

        auto vb0 = mm512_uni_loadu_ps(b + i);
        auto vb1 = mm512_uni_loadu_ps(b + i + vec_len_f32_avx512);
        auto vb2 = mm512_uni_loadu_ps(b + i + vec_len_f32_avx512 * 2);
        auto vb3 = mm512_uni_loadu_ps(b + i + vec_len_f32_avx512 * 3);

        vsum0 = _mm512_fmadd_ps(va0, vb0, vsum0);
        vsum1 = _mm512_fmadd_ps(va1, vb1, vsum1);
        vsum2 = _mm512_fmadd_ps(va2, vb2, vsum2);
        vsum3 = _mm512_fmadd_ps(va3, vb3, vsum3);
    }
    if (i + 2 * vec_len_f32_avx512 <= n) {
        auto va0 = mm512_uni_loadu_ps(a + i);
        auto va1 = mm512_uni_loadu_ps(a + i + vec_len_f32_avx512);

        auto vb0 = mm512_uni_loadu_ps(b + i);
        auto vb1 = mm512_uni_loadu_ps(b + i + vec_len_f32_avx512);

        vsum0 = _mm512_fmadd_ps(va0, vb0, vsum0);
        vsum1 = _mm512_fmadd_ps(va1, vb1, vsum1);
        i += 2 * vec_len_f32_avx512;
    }
    if (i + vec_len_f32_avx512 <= n) {
        auto va0 = mm512_uni_loadu_ps(a + i);
        auto vb0 = mm512_uni_loadu_ps(b + i);
        vsum0 = _mm512_fmadd_ps(va0, vb0, vsum0);
        i += vec_len_f32_avx512;
    }
    vsum0 = _mm512_add_ps(vsum0, vsum1);
    vsum2 = _mm512_add_ps(vsum2, vsum3);
    vsum0 = _mm512_add_ps(vsum0, vsum2);
    sum = _mm512_reduce_add_ps(vsum0);

    for (; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

template<typename T>
struct MatmulVectorAVX512_DotP {
    MatmulVectorAVX512_DotP() {}
    void operator()(tensor2D<T> & matA, const T * vB, float * vC) {
        int M = matA.dims[0];
        int K = matA.dims[1];
        T * pA = &matA[0];
        int strideA = matA.stride/sizeof(T);
        for(int m = 0; m < M; m++, pA+=strideA) {
            vC[m] = dot_product(pA, vB, K);
        }
    }
};

template<typename T>
struct MatmulVectorAVX512_BRG {
    MatmulVectorAVX512_BRG() {}
    // vA   :   1xK
    // matB :   KxN
    void operator()(const T * pA, tensor2D<T> & matB,  float * vC) {
        int K = matB.dims[0];
        int N = matB.dims[1];
        T* baseB = &matB[0];
        int strideB = matB.stride/sizeof(T);
        for (int n = 0; n < N; n+= 16*8) {
            auto vsum0 = _mm512_setzero_ps();
            auto vsum1 = _mm512_setzero_ps();
            auto vsum2 = _mm512_setzero_ps();
            auto vsum3 = _mm512_setzero_ps();
            auto vsum4 = _mm512_setzero_ps();
            auto vsum5 = _mm512_setzero_ps();
            auto vsum6 = _mm512_setzero_ps();
            auto vsum7 = _mm512_setzero_ps();
            auto* pB = baseB + n;
            for(int k = 0; k < K; k++, pB+=strideB) {
                // broadcast pA[k]
                // load pB
                auto va0 = _mm512_set1_ps(pA[k]);

                vsum0 = _mm512_fmadd_ps(va0, mm512_uni_loadu_ps(pB + 0), vsum0);
                vsum1 = _mm512_fmadd_ps(va0, mm512_uni_loadu_ps(pB + 16*1), vsum1);
                vsum2 = _mm512_fmadd_ps(va0, mm512_uni_loadu_ps(pB + 16*2), vsum2);
                vsum3 = _mm512_fmadd_ps(va0, mm512_uni_loadu_ps(pB + 16*3), vsum3);

                vsum4 = _mm512_fmadd_ps(va0, mm512_uni_loadu_ps(pB + 16*4), vsum4);
                vsum5 = _mm512_fmadd_ps(va0, mm512_uni_loadu_ps(pB + 16*5), vsum5);
                vsum6 = _mm512_fmadd_ps(va0, mm512_uni_loadu_ps(pB + 16*6), vsum6);
                vsum7 = _mm512_fmadd_ps(va0, mm512_uni_loadu_ps(pB + 16*7), vsum7);
            }
            _mm512_storeu_ps(vC + n + 0, vsum0);
            _mm512_storeu_ps(vC + n + 16*1, vsum1);
            _mm512_storeu_ps(vC + n + 16*2, vsum2);
            _mm512_storeu_ps(vC + n + 16*3, vsum3);
            _mm512_storeu_ps(vC + n + 16*4, vsum4);
            _mm512_storeu_ps(vC + n + 16*5, vsum5);
            _mm512_storeu_ps(vC + n + 16*6, vsum6);
            _mm512_storeu_ps(vC + n + 16*7, vsum7);
        }
    }
};


int OMP_NT = omp_thread_count();

template<typename T>
int amx_unit_test_gemAvB(int M, int K, int times = -1000) {
    int N = 1;
    tensor2D<T> A(M, K, true); // ensure stride of A matrix is multiple of cache line, which is vital to performance.
    tensor2D<T> At = A.Tr();
    tensor2D<T> B(K, 1, true);
    tensor2D<float> C0(M, 1, true);    // reference result
    tensor2D<float> C1(M, 1, true);    // actual result

    MatmulVector<T, T> matxvec;
    MatmulVectorAVX512_DotP<T> matxvec_dotp;
    MatmulVectorAVX512_BRG<T> matxvec_brg;
    // same B, different layout
    std::cout << __func__ << "(" << M << "," << K << ")\n";
    C0 = 0;
    matmul(A, B, C0);

    matxvec(A, &B[0], &C1[0]);
    if (C0 == C1) {
        std::cout << ANSIcolor("1;32") << "amx Match!\n" << ANSIcolor();
    } else {
        logger() << C0 << std::endl;
        logger() << C1 << std::endl;
        std::cout << ANSIcolor("1;31") << "amx Mismatch!\n" << ANSIcolor();
    }

    matxvec_dotp(A, &B[0], &C1[0]);
    if (C0 == C1) {
        std::cout << ANSIcolor("1;32") << "dotp Match!\n" << ANSIcolor();
    } else {
        logger() << C0 << std::endl;
        logger() << C1 << std::endl;
        std::cout << ANSIcolor("1;31") << "dotp Mismatch!\n" << ANSIcolor();
    }

    matxvec_brg(&B[0], At, &C1[0]);
    if (C0 == C1) {
        std::cout << ANSIcolor("1;32") << "brg Match!\n" << ANSIcolor();
    } else {
        logger() << C0 << std::endl;
        logger() << C1 << std::endl;
        std::cout << ANSIcolor("1;31") << "brg Mismatch!\n" << ANSIcolor();
    }

    timer.tag(__func__, M, K, N, "q*K_AMX")(times, [&](){
        matxvec(A, &B[0], &C1[0]);
    });
    timer.tag(__func__, M, K, N, "q*K_dotp")(times, [&](){
        matxvec_dotp(A, &B[0], &C1[0]);
    });
    timer.tag(__func__, M, K, N, "q*K_brg")(times, [&](){
        matxvec_brg(&B[0], At, &C1[0]);
    });
    return 0;
}

int main(int argc, const char *argv[]) {
    timer.set_app(argv[0]);
    //thp.Start();
    //test_all_bw(3.0); return 0;
    //test_parallel_FC();

    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    std::cout << ANSIcolor("31") << "omp_get_num_threads() = " << omp_get_num_threads() << std::endl << ANSIcolor();
    std::cout << ANSIcolor("31") << "OMP_NT = " << OMP_NT << std::endl << ANSIcolor();

    //amx_unit_test_gemAvB<float>(901, 96);
    std::cout << "===============================FP32========================\n";
    //amx_unit_test_gemAvB<float>(256, 128);
    //amx_unit_test_gemAvB<float>(256, 128);
    std::cout << "===============================BF16========================\n";
    amx_unit_test_gemAvB<ov::bfloat16>(256, 128);
    amx_unit_test_gemAvB<ov::bfloat16>(256, 128);

}
