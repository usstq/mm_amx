/*
<torch/extension.h> is the one-stop header to include all the necessary PyTorch bits to write C++ extensions. It includes:
 - The ATen library, which is our primary API for tensor computation,
 - pybind11, which is how we create Python bindings for our C++ code,
 - Headers that manage the details of interaction between ATen and pybind11.
*/
#include <omp.h>
#include <torch/extension.h>
// #include "misc.hpp"
// #include "timeit.hpp"
#include "profiler.hpp"

// define PARALLEL_NT_STATIC
template <typename F>
void parallel_nt_static_omp(const F& func) {
#pragma omp parallel
    { func(omp_get_thread_num(), omp_get_num_threads()); }
}
#define PARALLEL_NT_STATIC(...) parallel_nt_static_omp(__VA_ARGS__)

#define stringify(a) xstr(a)
#define xstr(a) #a
#define ASSERT(x)                                                                                                                                                                                                                                                                      \
    if (!(x))                                                                                                                                                                                                                                                                          \
    throw std::runtime_error(__FILE__ ":" stringify(__LINE__) " (" #x ") failed!")

#include "tensor2D.hpp"

#include "jit.hpp"

namespace helper {

inline void transpose_m512i_16x16(__m512i& r0, __m512i& r1, __m512i& r2, __m512i& r3, __m512i& r4, __m512i& r5, __m512i& r6, __m512i& r7, __m512i& r8, __m512i& r9, __m512i& ra, __m512i& rb, __m512i& rc, __m512i& rd, __m512i& re, __m512i& rf) {
    __m512i t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, ta, tb, tc, td, te, tf;

    t0 = _mm512_unpacklo_epi32(r0, r1); //   0  16   1  17   4  20   5  21   8  24   9  25  12  28  13  29
    t1 = _mm512_unpackhi_epi32(r0, r1); //   2  18   3  19   6  22   7  23  10  26  11  27  14  30  15  31
    t2 = _mm512_unpacklo_epi32(r2, r3); //  32  48  33  49 ...
    t3 = _mm512_unpackhi_epi32(r2, r3); //  34  50  35  51 ...
    t4 = _mm512_unpacklo_epi32(r4, r5); //  64  80  65  81 ...
    t5 = _mm512_unpackhi_epi32(r4, r5); //  66  82  67  83 ...
    t6 = _mm512_unpacklo_epi32(r6, r7); //  96 112  97 113 ...
    t7 = _mm512_unpackhi_epi32(r6, r7); //  98 114  99 115 ...
    t8 = _mm512_unpacklo_epi32(r8, r9); // 128 ...
    t9 = _mm512_unpackhi_epi32(r8, r9); // 130 ...
    ta = _mm512_unpacklo_epi32(ra, rb); // 160 ...
    tb = _mm512_unpackhi_epi32(ra, rb); // 162 ...
    tc = _mm512_unpacklo_epi32(rc, rd); // 196 ...
    td = _mm512_unpackhi_epi32(rc, rd); // 198 ...
    te = _mm512_unpacklo_epi32(re, rf); // 228 ...
    tf = _mm512_unpackhi_epi32(re, rf); // 230 ...

    r0 = _mm512_unpacklo_epi64(t0, t2); //   0  16  32  48 ...
    r1 = _mm512_unpackhi_epi64(t0, t2); //   1  17  33  49 ...
    r2 = _mm512_unpacklo_epi64(t1, t3); //   2  18  34  49 ...
    r3 = _mm512_unpackhi_epi64(t1, t3); //   3  19  35  51 ...
    r4 = _mm512_unpacklo_epi64(t4, t6); //  64  80  96 112 ...
    r5 = _mm512_unpackhi_epi64(t4, t6); //  65  81  97 114 ...
    r6 = _mm512_unpacklo_epi64(t5, t7); //  66  82  98 113 ...
    r7 = _mm512_unpackhi_epi64(t5, t7); //  67  83  99 115 ...
    r8 = _mm512_unpacklo_epi64(t8, ta); // 128 144 160 176 ...
    r9 = _mm512_unpackhi_epi64(t8, ta); // 129 145 161 178 ...
    ra = _mm512_unpacklo_epi64(t9, tb); // 130 146 162 177 ...
    rb = _mm512_unpackhi_epi64(t9, tb); // 131 147 163 179 ...
    rc = _mm512_unpacklo_epi64(tc, te); // 192 208 228 240 ...
    rd = _mm512_unpackhi_epi64(tc, te); // 193 209 229 241 ...
    re = _mm512_unpacklo_epi64(td, tf); // 194 210 230 242 ...
    rf = _mm512_unpackhi_epi64(td, tf); // 195 211 231 243 ...

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

inline void transpose_epi32_16x16(void* _dst, const void* src, int stride) {
    auto* dst = reinterpret_cast<uint32_t*>(_dst);
    __m512i r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, ra, rb, rc, rd, re, rf;
    auto* pA = reinterpret_cast<const uint8_t*>(src);
    r0 = _mm512_loadu_epi32(pA);
    r1 = _mm512_loadu_epi32(pA + stride);
    r2 = _mm512_loadu_epi32(pA + 2 * stride);
    r3 = _mm512_loadu_epi32(pA + 3 * stride);
    r4 = _mm512_loadu_epi32(pA + 4 * stride);
    r5 = _mm512_loadu_epi32(pA + 5 * stride);
    r6 = _mm512_loadu_epi32(pA + 6 * stride);
    r7 = _mm512_loadu_epi32(pA + 7 * stride);
    r8 = _mm512_loadu_epi32(pA + 8 * stride);
    r9 = _mm512_loadu_epi32(pA + 9 * stride);
    ra = _mm512_loadu_epi32(pA + 10 * stride);
    rb = _mm512_loadu_epi32(pA + 11 * stride);
    rc = _mm512_loadu_epi32(pA + 12 * stride);
    rd = _mm512_loadu_epi32(pA + 13 * stride);
    re = _mm512_loadu_epi32(pA + 14 * stride);
    rf = _mm512_loadu_epi32(pA + 15 * stride);

    transpose_m512i_16x16(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, ra, rb, rc, rd, re, rf);

    _mm512_storeu_epi32(dst, r0);
    _mm512_storeu_epi32(dst + 16, r1);
    _mm512_storeu_epi32(dst + 2 * 16, r2);
    _mm512_storeu_epi32(dst + 3 * 16, r3);
    _mm512_storeu_epi32(dst + 4 * 16, r4);
    _mm512_storeu_epi32(dst + 5 * 16, r5);
    _mm512_storeu_epi32(dst + 6 * 16, r6);
    _mm512_storeu_epi32(dst + 7 * 16, r7);
    _mm512_storeu_epi32(dst + 8 * 16, r8);
    _mm512_storeu_epi32(dst + 9 * 16, r9);
    _mm512_storeu_epi32(dst + 10 * 16, ra);
    _mm512_storeu_epi32(dst + 11 * 16, rb);
    _mm512_storeu_epi32(dst + 12 * 16, rc);
    _mm512_storeu_epi32(dst + 13 * 16, rd);
    _mm512_storeu_epi32(dst + 14 * 16, re);
    _mm512_storeu_epi32(dst + 15 * 16, rf);
}

inline void exp_ps_avx512(__m512& src) {
    static __m512 exp_ln_flt_min_f = _mm512_castsi512_ps(_mm512_set1_epi32(0xc2aeac50)); // log(FLT_MIN)
    static __m512 exp_ln_flt_max_f = _mm512_castsi512_ps(_mm512_set1_epi32(0x42b17218)); // log(FLT_MAX)
    static __m512 exp_log2ef = _mm512_castsi512_ps(_mm512_set1_epi32(0x3fb8aa3b));       // log2(e)
    static __m512 half = _mm512_castsi512_ps(_mm512_set1_epi32(0x3f000000));             // 0.5f
    static __m512 ln2f = _mm512_castsi512_ps(_mm512_set1_epi32(0x3f317218));             // ln(2)
    static __m512 one = _mm512_castsi512_ps(_mm512_set1_epi32(0x3f800000));              // 1.0f
    static __m512i exponent_bias = _mm512_set1_epi32(0x0000007f);                        // 127
    static constexpr int n_mantissa_bits = 23;
    static __m512 exp_pol1 = _mm512_castsi512_ps(_mm512_set1_epi32(0x3f7ffffb)); // p1 = 0.999999701f
    static __m512 exp_pol2 = _mm512_castsi512_ps(_mm512_set1_epi32(0x3efffee3)); // p2 = 0.499991506f
    static __m512 exp_pol3 = _mm512_castsi512_ps(_mm512_set1_epi32(0x3e2aad40)); // p3 = 0.166676521f
    static __m512 exp_pol4 = _mm512_castsi512_ps(_mm512_set1_epi32(0x3d2b9d0d)); // p4 = 0.0418978221f
    static __m512 exp_pol5 = _mm512_castsi512_ps(_mm512_set1_epi32(0x3c07cfce)); // p5 = 0.00828929059f
    static __m512 two = _mm512_castsi512_ps(_mm512_set1_epi32(0x40000000));      // 2
    // exp(x) =
    // = exp(n * ln(2) + r) // divide x by ln(2) and get quot and rem
    // = 2^n * exp(r)       // simplify the exp(n*ln(2)) expression

    // get mask of values lower than log(FLT_MIN) to zero them in the output
    auto zero_mask = _mm512_cmp_ps_mask(src, exp_ln_flt_min_f, _CMP_LT_OS);

    // clip src
    src = _mm512_min_ps(src, exp_ln_flt_max_f);
    src = _mm512_max_ps(src, exp_ln_flt_min_f);

    // aux1 : r
    auto aux1 = src;

    // calculate exp(x)
    // fx = x * log2(e) + 0.5
    src = _mm512_mul_ps(src, exp_log2ef);
    src = _mm512_add_ps(src, half);

    // tmp = floorf(fx)
    src = _mm512_floor_ps(src);

    // aux1 = x - fx * ln2
    aux1 = _mm512_fnmadd_ps(src, ln2f, aux1);
    // We do not count 2^n here, because n can reach 128 and 2^128 is not
    // representable by fp32, so to get around this problem, instead of computing
    // 2^n * exp(r) will be counted 2*2^(n-1)*exp(r), because 2^127
    // and 2 are numbers representable in fp32.

    // compute 2^(n-1)
    src = _mm512_sub_ps(src, one);
    auto aux2_i = _mm512_cvtps_epi32(src);
    aux2_i = _mm512_add_epi32(aux2_i, exponent_bias);
    aux2_i = _mm512_slli_epi32(aux2_i, n_mantissa_bits);

    // set zeroes at those points which were < log(FLT_MIN)
    auto zero = _mm512_setzero_ps();
    auto aux2 = _mm512_mask_blend_ps(zero_mask, _mm512_castsi512_ps(aux2_i), zero);

    // compute polynomial
    src = exp_pol5;
    src = _mm512_fmadd_ps(src, aux1, exp_pol4);
    src = _mm512_fmadd_ps(src, aux1, exp_pol3);
    src = _mm512_fmadd_ps(src, aux1, exp_pol2);
    src = _mm512_fmadd_ps(src, aux1, exp_pol1);
    src = _mm512_fmadd_ps(src, aux1, one);

    // y = y * 2^n
    src = _mm512_mul_ps(src, aux2);
    src = _mm512_mul_ps(src, two);
}

inline __m512 silu_ps_avx512(const __m512 x) {
    static __m512 one = _mm512_castsi512_ps(_mm512_set1_epi32(0x3f800000)); // 1.0f
    auto negx = _mm512_sub_ps(_mm512_setzero_ps(), x);                      // -x
    exp_ps_avx512(negx);                                                    // exp(-x)
    auto sigmoidx = _mm512_rcp14_ps(_mm512_add_ps(one, negx));              // 1/[1+exp(-x)]
    return _mm512_mul_ps(x, sigmoidx);
}

}; // namespace helper

static tensor2D<ov::bfloat16> repack_weights(tensor2D<ov::bfloat16>& Bt) {
    int N = Bt.dims[0];
    int K = Bt.dims[1];
    tensor2D<ov::bfloat16> BPacked(K * N, 1, true);
    for (int n = 0, i = 0; n < N; n += 32) {
        for (int k = 0; k < K; k += 32) {
            helper::transpose_epi32_16x16(&BPacked[i * 16 * 32], &Bt(n, k), Bt.stride);
            i++;
            helper::transpose_epi32_16x16(&BPacked[i * 16 * 32], &Bt(n + 16, k), Bt.stride);
            i++;
        }
    }
    return BPacked;
}

#if 0

class Linear32x32_AMX : public jit_generator {
public:
    TileConfig m_tile_cfg;
    bool m_do_accumulation;

    Linear32x32_AMX(bool do_accumulation) : m_do_accumulation(do_accumulation) {
        create_kernel("Linear32x32_AMX");
        m_tile_cfg.reset(1, 0,
                         {
                             {16, 64}, // C:0
                             {16, 64}, // C:1
                             {16, 64}, // C:2
                             {16, 64}, // C:3
                             {16, 64}, // A0:4
                             {16, 64}, // A1:5
                             {16, 64}, // B0:6
                             {16, 64}, // B1:7
                         });
    }

    const TileConfig& tile_config() { return m_tile_cfg; }

    int64_t m_ktiles;
    void* prefetch_ptrA;
    // to save push/pop: do not use `abi_save_gpr_regs`
    Xbyak::Reg64 reg_A_addr = abi_param1;
    Xbyak::Reg64 reg_A_stride = abi_param2;
    Xbyak::Reg64 reg_B_addr = abi_param3;
    Xbyak::Reg64 reg_C_addr = abi_param4;
    Xbyak::Reg64 reg_C_stride = abi_param5;
    Xbyak::Reg64 reg_prefetchA = abi_param6;
    Xbyak::Reg64 reg_ktiles = rax;
    Xbyak::Reg64 reg_B_stride = r10;
    Xbyak::Reg64 reg_A1_addr = r11;

    Xbyak::Tmm tmmC00 = tmm0;
    Xbyak::Tmm tmmC10 = tmm1;
    Xbyak::Tmm tmmC01 = tmm2;
    Xbyak::Tmm tmmC11 = tmm3;
    Xbyak::Tmm tmmA0 = tmm4;
    Xbyak::Tmm tmmA1 = tmm5;
    Xbyak::Tmm tmmB0 = tmm6;
    Xbyak::Tmm tmmB1 = tmm7;

    void generate() {
        /*
                       B: 1x2 tiles
        A : 2x1 tiles  C: 2x2 tiles
        */
        Xbyak::Label loop_over_ktiles;

        if (m_do_accumulation) {
            auto reg_C1_addr = reg_A1_addr; // reuse reg_A1_addr
            tileloadd(tmmC00, ptr[reg_C_addr + reg_C_stride]);
            tileloadd(tmmC01, ptr[reg_C_addr + reg_C_stride + 64]);
            lea(reg_C1_addr, ptr[reg_C_addr + reg_C_stride * 8]);
            lea(reg_C1_addr, ptr[reg_C1_addr + reg_C_stride * 8]);
            tileloadd(tmmC10, ptr[reg_C1_addr + reg_C_stride]);
            tileloadd(tmmC11, ptr[reg_C1_addr + reg_C_stride + 64]);
        } else {
            tilezero(tmmC00);
            tilezero(tmmC01);
            tilezero(tmmC10);
            tilezero(tmmC11);
        }
        mov(reg_B_stride, reinterpret_cast<uintptr_t>(&m_ktiles));
        mov(reg_ktiles, ptr[reg_B_stride + 0]);

        mov(reg_B_stride, reinterpret_cast<uintptr_t>(&prefetch_ptrA));
        mov(reg_prefetchA, ptr[reg_B_stride + 0]);
        
        lea(reg_A1_addr, ptr[reg_A_addr + reg_A_stride * 8]);
        lea(reg_A1_addr, ptr[reg_A1_addr + reg_A_stride * 8]);
        mov(reg_B_stride, 64);

        auto const_A_steps = 64;

        align(64, false);
        L(loop_over_ktiles);
        // for (int k = 0; k < Ktiles; k++) {
        tileloadd(tmmA0, ptr[reg_A_addr + reg_A_stride]);
        tileloadd(tmmB0, ptr[reg_B_addr + reg_B_stride]);
        lea(reg_B_addr, ptr[reg_B_addr + 1024]);

        // prefetch next 32xK A-sub matrix
        prefetcht1(ptr[reg_prefetchA + 0]);
        prefetcht1(ptr[reg_prefetchA + 64]);
        prefetcht1(ptr[reg_prefetchA + 64*2]);
        prefetcht1(ptr[reg_prefetchA + 64*3]);
        lea(reg_prefetchA, ptr[reg_prefetchA + 64*4]);

        tdpbf16ps(tmmC00, tmmA0, tmmB0);

        tileloadd(tmmA1, ptr[reg_A1_addr + reg_A_stride]);
        tdpbf16ps(tmmC10, tmmA1, tmmB0);

        tileloadd(tmmB1, ptr[reg_B_addr + reg_B_stride]);

        tdpbf16ps(tmmC01, tmmA0, tmmB1);
        tdpbf16ps(tmmC11, tmmA1, tmmB1);
        //}
        lea(reg_A_addr, ptr[reg_A_addr + const_A_steps]);
        lea(reg_A1_addr, ptr[reg_A1_addr + const_A_steps]);
        lea(reg_B_addr, ptr[reg_B_addr + 1024]);
        dec(reg_ktiles);
        jnz(loop_over_ktiles, T_NEAR);

        tilestored(ptr[reg_C_addr + reg_C_stride], tmmC00);
        tilestored(ptr[reg_C_addr + reg_C_stride + 64], tmmC01);
        lea(reg_C_addr, ptr[reg_C_addr + reg_C_stride * 8]);
        lea(reg_C_addr, ptr[reg_C_addr + reg_C_stride * 8]);
        tilestored(ptr[reg_C_addr + reg_C_stride], tmmC10);
        tilestored(ptr[reg_C_addr + reg_C_stride + 64], tmmC11);
        ret();
    }
};


class LinearNxN {
public:
    int m_Ktiles;
    int m_K;
    int m_M;
    int m_N;
    Linear32x32_AMX m_kernel_0;
    Linear32x32_AMX m_kernel_1;
    tensor2D<ov::bfloat16> m_B0;
    tensor2D<ov::bfloat16> m_B1;

    uint8_t fake_buff[256];
    /*
    LinearNxN(int K, int M, int N, ov::bfloat16* weight, int w_stride) : m_K(K), m_M(M), m_N(N), m_kernel_0(false), m_kernel_1(true) {
        m_Ktiles = m_K / 32;
        m_kernel_0.m_ktiles = m_Ktiles;
        m_kernel_1.m_ktiles = m_Ktiles;
        m_kernel_0.prefetch_ptrA = fake_buff;
        m_kernel_1.prefetch_ptrA = fake_buff;
        ASSERT((m_K % 32) == 0);
        set_weight(weight, w_stride);
    }
    */
    
    const TileConfig& tile_config() { return m_kernel_0.m_tile_cfg; }

    // Bt: [N, K]
    template <int kernel_idx = 0>
    void call_kernel(int x, int y, ov::bfloat16* A0, int strideA, ov::bfloat16* B0, float* C0, int strideC, int ktiles) {
        // clang-format off
        if (kernel_idx == 0)
            m_kernel_0(reinterpret_cast<uint8_t*>(A0) + y * strideA, strideA,
                    reinterpret_cast<uint8_t*>(B0) + ((x / 32) * (ktiles)) * 2048,
                    reinterpret_cast<uint8_t*>(C0 + x) + y * strideC, strideC,
                    ktiles);
        else
            m_kernel_1(reinterpret_cast<uint8_t*>(A0) + y * strideA, strideA,
                    reinterpret_cast<uint8_t*>(B0) + ((x / 32) * (ktiles)) * 2048,
                    reinterpret_cast<uint8_t*>(C0 + x) + y * strideC, strideC,
                    ktiles);
        // clang-format on
    }

    //float Cx[32*32];

    void call_general(int x0, int x1, int y0, int y1, ov::bfloat16* A0, int strideA, ov::bfloat16* B0, float* C0, int strideC) {
        // 128x128 : 560
        // ECOUT(x0, "->", x1,",",  y0,"->",  y1,", strideA=", strideA, ", strideC=", strideC);
        if (y1 - y0 >= x1 - x0) {
            auto* ptrA0 = reinterpret_cast<uint8_t*>(A0);
            auto* ptrC0 = reinterpret_cast<uint8_t*>(C0);

            auto prefetch_blk_bytes = 32 * m_K * sizeof(ov::bfloat16) / ((x1-x0)/32);
            for (int y = y0; y < y1; y += 32, ptrA0 += 32 * strideA, ptrC0 += 32 * strideC) {
                auto* ptrB0 = reinterpret_cast<uint8_t*>(B0);

                // for prefetching next 32xK subA
                auto* ptrA1 = ptrA0 + (32) * strideA;
                for (int x = x0; x < x1; x += 32, ptrB0 += m_Ktiles * 2048) {

                    // too many SW prefetch would also block CPU HW pipeline, so it must be mixed into kernel
                    //for(int i = 0; i < prefetch_blk_bytes; i += 64) _mm_prefetch(ptrA1 + i, _MM_HINT_T2);

                    m_kernel_0.prefetch_ptrA = ptrA1;
                    m_kernel_0(ptrA0, strideA, ptrB0, ptrC0 + x * sizeof(float), strideC, m_Ktiles);
                    //m_kernel_0(ptrA0, strideA, ptrB0, Cx, 32*sizeof(float), m_Ktiles);

                    // prefetch next 32xK subA
                    ptrA1 += prefetch_blk_bytes;
                }
            }
        } else {
            bool downward = true;
            for (int x = x0; x < x1; x += 32, downward = !downward) {
                if (downward) {
                    for (int y = y0; y < y1; y += 32) {
                        call_kernel(x, y, A0, strideA, B0, C0, strideC, m_Ktiles);
                    }
                } else {
                    for (int y = y1 - 32; y >= y0; y -= 32) {
                        call_kernel(x, y, A0, strideA, B0, C0, strideC, m_Ktiles);
                    }
                }
            }
        }
    }

    // B0: repacked
    void operator()(ov::bfloat16* A0, int strideA, float* C0, int strideC) {
        ov::bfloat16* B0 = &m_B0[0];
        call_general(0, m_N, 0, m_M, A0, strideA, B0, C0, strideC);
        return;
    }

    void forward(torch::Tensor x, torch::Tensor y) {
        PROFILE(_prof, m_name);
        TileConfigScope _tcfg(tile_config());
        ASSERT(x.dense_dim() == 2);
        m_M = x.size(0);
        ASSERT(x.size(1) == m_K);
        call_general(0, m_N, 0, m_M,
            reinterpret_cast<ov::bfloat16*>(x.data_ptr()), x.stride(0)*sizeof(ov::bfloat16),
            &m_B0[0],
            reinterpret_cast<float*>(y.data_ptr()), y.stride(0)*sizeof(float));
    }

    void _set_weight(ov::bfloat16* weight, int w_stride) {
        tensor2D<ov::bfloat16> Bt(m_N, m_K, weight, w_stride);
        m_B0 = repack_weights(Bt);
    }

    void set_weight(torch::Tensor weight) {
        ASSERT(weight.dtype() == torch::kBFloat16);
        ASSERT(weight.dense_dim() == 2);
        m_N = weight.size(0);
        m_K = weight.size(1);
        ASSERT((m_N % 32) == 0);
        ASSERT((m_K % 32) == 0);
        m_M = 0;// unknown

        _set_weight(reinterpret_cast<ov::bfloat16*>(weight.data_ptr()), weight.stride(0)*sizeof(ov::bfloat16));

        m_Ktiles = m_K / 32;
        m_kernel_0.m_ktiles = m_Ktiles;
        m_kernel_1.m_ktiles = m_Ktiles;
        m_kernel_0.prefetch_ptrA = fake_buff;
        m_kernel_1.prefetch_ptrA = fake_buff;
    }

    LinearNxN() : m_kernel_0(false), m_kernel_1(true) {
    }

    std::string m_name;

    const std::string& get_name() {
        return m_name;
    }
    void set_name(std::string& name) {
        m_name = name;
    }

    LinearNxN(const LinearNxN& other) : m_kernel_0(false), m_kernel_1(true) {
        m_name = other.m_name;
        m_M = other.m_M;
        m_N = other.m_N;
        m_K = other.m_K;
        m_Ktiles = other.m_Ktiles;
        m_kernel_0.m_ktiles = m_Ktiles;
        m_kernel_1.m_ktiles = m_Ktiles;        
        m_kernel_0.prefetch_ptrA = fake_buff;
        m_kernel_1.prefetch_ptrA = fake_buff;        
        m_B0 = other.m_B0.clone();
    }
};


void clflush(torch::Tensor x) {
    ASSERT(x.is_contiguous());
    auto* p = reinterpret_cast<uint8_t*>(x.data_ptr());
    auto nbytes = x.nbytes();
    for (size_t i = 0; i < nbytes; i += 64) {
        _mm_clflushopt(p + i);
    }
    _mm_mfence();
}

#endif

//===============================================================

/**

 Silu(Gate) + Up

  silu(x)=x∗σ(x)
  σ(x) = 1/(1+exp(-x))

 y1 = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
 y2 = self.down_proj(y)
*/
EnvVar NOSWPF("NOSWPF", 0);
EnvVar SWPFA("SWPFA", 0); // prefetch A is slower, so disable it

class LinearGateUp : public jit_generator {
public:
    TileConfig m_tile_cfg;

    int64_t m_ktiles;

    int m_BM; // blockM: L2-cache kernel
    int m_BK; // blockK: L2-cache kernel
    int m_BN; // blockN: L2-cache kernel

    bool m_do_accumulation;

    int m_prefetch_Blines;

    // both A & B data will be prefetched from memory for next kernel invokation
    // and the prefetches are evenly distributed into each kernel.
    //
    // we first tackle the prefetching of B, because each time
    // we will call it with a new B, and run() will have to prefetch new B
    // for next round, so next B is also of size (KxN) elements
    //    distributes into (BM/32)*(BN/32) kernels:
    //    each kernel has (BK/32) iterations, thus each kernel iteration
    //    need to prefetch (BKxBN)/(BMxBNxBK/32768) = 32768/BM bfloat16-elements
    //    which is 1024/BM cache lines, this has to be determined at
    //    code-generation time. with BM=256, this is only 4.
    //
    // prefetch A can be done in unit of 32xBK elements, which must be evenly distributed
    // into (BN/32)*(BK/32) kernel iterations, each iteration prefetch/copy 32xBK/(BN*BK/1024) = 32768/BN bfloat16-elements
    // or 1024/BN cache lines. with BM=256, this is only 4 too.
    //
    // prefetch or copy?
    //   prefetch strided sub-matrix of A is tricky, consider each 32x32 AMX jit kernel has [BK/32] iterations
    //   and it's called (BN/32) times, each kernel must prefetch 32*BK/(BN/32) = (1024/BN)*BK elements
    //   since each kernel has [BK/32] loop iterations, each iteration fetch (1024/BN)*BK/(BK/32) = 1024*32/BN
    //   bytes.
    //
    //   when 1024 is not divisible by BN, it's fine, just prefetch more
    //
    // copy data from A to a ping-pong buffer has advantage:
    //    - read can be done in continous way most suitable for HW prefetcher
    //    - write to ping-pong buffer is within L2 cache, which should be fast
    //    - data transfer rate is small comparing to L2-bandwidth, shouldn't be a big issue for interleaved write to L2.
    //    - read from ping-pong buffer is much faster and free of odd-multiple-cache-line restriction.
    // so we prefer distribute the repacking of A sub-matrix into ping-pong buffer into kernel.
    // for BN=256, each kernel read 4*BK elements into ping-pong, each iteration read 4*BK*sizeof(bfloat16)/(BK/32)=256bytes = 4-512bits zmm registers
    //
    //
    Linear32x32_AMX_mkernel(int BM = 256, int BK = 256, int BN = 256, bool do_accumulation = false) : m_BM(BM), m_BK(BK), m_BN(BN), m_do_accumulation(do_accumulation) {

        // B: [K, N]
        m_ktiles = m_BK / 32;

        // prefetch block B requires
        m_prefetch_Blines = 32768 * sizeof(ov::bfloat16) / 64 / m_BM;

        if (NOSWPF) {
            m_prefetch_Blines = 0;
        }

        create_kernel("Linear32x32_AMX_mkernel");
        m_tile_cfg.reset(1, 0,
                         {
                             {16, 64}, // C:0
                             {16, 64}, // C:1
                             {16, 64}, // C:2
                             {16, 64}, // C:3
                             {16, 64}, // A0:4
                             {16, 64}, // A1:5
                             {16, 64}, // B0:6
                             {16, 64}, // B1:7
                         });
    }

    // to save push/pop: do not use `abi_save_gpr_regs`
    Xbyak::Reg64 reg_A_addr = abi_param1;
    Xbyak::Reg64 reg_A_stride = abi_param2;
    Xbyak::Reg64 reg_B_addr = abi_param3;
    Xbyak::Reg64 reg_C_addr = abi_param4;
    Xbyak::Reg64 reg_C_stride = abi_param5;
    Xbyak::Reg64 reg_prefetch = abi_param6; // prefetch B

    Xbyak::Reg64 reg_ktiles = rax;
    Xbyak::Reg64 reg_prefetch_A = r9;
    Xbyak::Reg64 reg_prefetch_A1 = r12;
    Xbyak::Reg64 reg_B_stride = r10;
    Xbyak::Reg64 reg_A1_addr = r11;

    Xbyak::Tmm tmmC00 = tmm0;
    Xbyak::Tmm tmmC10 = tmm1;
    Xbyak::Tmm tmmC01 = tmm2;
    Xbyak::Tmm tmmC11 = tmm3;
    Xbyak::Tmm tmmA0 = tmm4;
    Xbyak::Tmm tmmA1 = tmm5;
    Xbyak::Tmm tmmB0 = tmm6;
    Xbyak::Tmm tmmB1 = tmm7;

    uint8_t* prefetch_next_A_addr;

    void generate() {
        auto num_PFB = m_prefetch_Blines;
        int cur_PFB = 0;
        /*
                       B: 1x2 tiles
        A : 2x1 tiles  C: 2x2 tiles
        */
        Xbyak::Label loop_over_ktiles;

        if (m_do_accumulation) {
            auto reg_C1_addr = reg_A1_addr; // reuse reg_A1_addr
#if 0
            mov(reg_B_stride, 64);
            tileloadd(tmmC00, ptr[reg_C_addr + reg_B_stride]);
            tileloadd(tmmC01, ptr[reg_C_addr + reg_B_stride + 1024]);
            tileloadd(tmmC10, ptr[reg_C_addr + reg_B_stride + 1024 * 2]);
            tileloadd(tmmC11, ptr[reg_C_addr + reg_B_stride + 1024 * 3]);
#else
            tileloadd(tmmC00, ptr[reg_C_addr + reg_C_stride]);
            tileloadd(tmmC01, ptr[reg_C_addr + reg_C_stride + 64]);
            lea(reg_C1_addr, ptr[reg_C_addr + reg_C_stride * 8]);
            lea(reg_C1_addr, ptr[reg_C1_addr + reg_C_stride * 8]);
            tileloadd(tmmC10, ptr[reg_C1_addr + reg_C_stride]);
            tileloadd(tmmC11, ptr[reg_C1_addr + reg_C_stride + 64]);
#endif
        } else {
            tilezero(tmmC00);
            tilezero(tmmC01);
            tilezero(tmmC10);
            tilezero(tmmC11);
        }
        mov(reg_B_stride, reinterpret_cast<uintptr_t>(&m_ktiles));
        mov(reg_ktiles, ptr[reg_B_stride + 0]);

        if (SWPFA) {
            push(reg_prefetch_A1);
            mov(reg_B_stride, reinterpret_cast<uintptr_t>(&prefetch_next_A_addr));
            mov(reg_prefetch_A, ptr[reg_B_stride + 0]);
            mov(reg_prefetch_A1, ptr[reg_prefetch_A + 2 * reg_A_stride]);
        }

        lea(reg_A1_addr, ptr[reg_A_addr + reg_A_stride * 8]);
        lea(reg_A1_addr, ptr[reg_A1_addr + reg_A_stride * 8]);
        mov(reg_B_stride, 64);

        auto const_A_steps = 64;

        align(64, false);
        L(loop_over_ktiles);
        // for (int k = 0; k < Ktiles; k++) {
        tileloadd(tmmA0, ptr[reg_A_addr + reg_A_stride]);
        tileloadd(tmmB0, ptr[reg_B_addr + reg_B_stride]);
        lea(reg_B_addr, ptr[reg_B_addr + 1024]);

        tdpbf16ps(tmmC00, tmmA0, tmmB0);
        if (cur_PFB < num_PFB) {
            prefetcht2(ptr[reg_prefetch + cur_PFB * 64]);
            cur_PFB++;
        }

        if (SWPFA)
            prefetcht2(ptr[reg_prefetch_A]);

        tileloadd(tmmA1, ptr[reg_A1_addr + reg_A_stride]);
        tdpbf16ps(tmmC10, tmmA1, tmmB0);
        if (cur_PFB < num_PFB) {
            prefetcht2(ptr[reg_prefetch + cur_PFB * 64]);
            cur_PFB++;
        }
        if (SWPFA)
            prefetcht2(ptr[reg_prefetch_A + reg_A_stride]);

        tileloadd(tmmB1, ptr[reg_B_addr + reg_B_stride]);

        // prefetch [num_Ktiles X 256] bytes

        tdpbf16ps(tmmC01, tmmA0, tmmB1);
        if (cur_PFB < num_PFB) {
            prefetcht2(ptr[reg_prefetch + cur_PFB * 64]);
            cur_PFB++;
        }
        if (SWPFA)
            prefetcht2(ptr[reg_prefetch_A1]);

        tdpbf16ps(tmmC11, tmmA1, tmmB1);
        // prefetch next sub-block B matrix
        if (cur_PFB < num_PFB) {
            for (int pi = cur_PFB; pi < num_PFB; pi++) {
                prefetcht2(ptr[reg_prefetch + pi * 64]);
            }
        }
        if (SWPFA)
            prefetcht2(ptr[reg_prefetch_A1 + reg_A_stride]);

        lea(reg_prefetch, ptr[reg_prefetch + 64 * num_PFB]);

        if (SWPFA)
            lea(reg_prefetch_A, ptr[reg_prefetch_A + 64]);
        if (SWPFA)
            lea(reg_prefetch_A1, ptr[reg_prefetch_A1 + 64]);

        //}
        lea(reg_A_addr, ptr[reg_A_addr + const_A_steps]);
        lea(reg_A1_addr, ptr[reg_A1_addr + const_A_steps]);
        lea(reg_B_addr, ptr[reg_B_addr + 1024]);
        dec(reg_ktiles);
        jnz(loop_over_ktiles, T_NEAR);

#if 0
        tilestored(ptr[reg_C_addr + reg_B_stride], tmmC00);
        tilestored(ptr[reg_C_addr + reg_B_stride + 1024], tmmC01);
        tilestored(ptr[reg_C_addr + reg_B_stride + 1024 * 2], tmmC10);
        tilestored(ptr[reg_C_addr + reg_B_stride + 1024 * 3], tmmC11);
#else
        tilestored(ptr[reg_C_addr + reg_C_stride], tmmC00);
        tilestored(ptr[reg_C_addr + reg_C_stride + 64], tmmC01);
        lea(reg_C_addr, ptr[reg_C_addr + reg_C_stride * 8]);
        lea(reg_C_addr, ptr[reg_C_addr + reg_C_stride * 8]);
        tilestored(ptr[reg_C_addr + reg_C_stride], tmmC10);
        tilestored(ptr[reg_C_addr + reg_C_stride + 64], tmmC11);
#endif
        if (SWPFA)
            pop(reg_prefetch_A1);
        ret();
    }

    // run L2 cache blocking kernel with size:
    //    [BM, BK]*[BK, BN] => [BM, BN]
    //
    // prefetch of A can be done inside of this level of kernel
    // since it's done in unit of 32-rows
    // but prefetch of next B must be specified by caller.
    //
    void run(uint8_t* pA, int strideA, // A
             uint8_t* pB, int strideB, // strideB is special
             uint8_t* pC, int strideC, // C
             uint8_t* prefetch_B       // prefetch B
    ) {
        auto prefetch_step = m_prefetch_Blines * 64 * m_ktiles;

        for (int m = 0; m < m_BM; m += 32, pA += 32 * strideA, pC += 32 * strideC) {
            auto* pB1 = pB;
            prefetch_next_A_addr = pA + 32 * strideA;
            if (m + 32 >= m_BM)
                prefetch_next_A_addr = pA;
            for (int n = 0; n < m_BN; n += 32, pB1 += strideB, prefetch_B += prefetch_step) {
                (*this)(pA, strideA, pB1, pC + n * sizeof(float), strideC, prefetch_B);
                prefetch_next_A_addr += 4 * strideA;
            }
        }
    }
};

//===============================================================
static bool _init_xtile = initXTILE();

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<LinearNxN>(m, "LinearNxN", py::dynamic_attr())
        .def(py::init<>())
        .def("set_weight", static_cast<void (LinearNxN::*)(torch::Tensor)>(&LinearNxN::set_weight))
        .def("forward", &LinearNxN::forward)
        .def_property("name", &LinearNxN::get_name, &LinearNxN::set_name)
        // https://docs.python.org/3/library/copy.html
        .def("__deepcopy__", [](const LinearNxN& self, py::dict) { return LinearNxN(self); });
    m.def("clflush", &clflush, "Clear cache");
}
