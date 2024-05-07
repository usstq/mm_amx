#include "jit.hpp"
#include <vector>

#include "dnnl_kernels.hpp"

#if !defined(XBYAK64_GCC)
#error NOT SUPPORTED
#endif

#include "timeit.hpp"

#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#include <stdlib.h>

#include <omp.h>

#include "bf16.hpp"
// #include "kernels_avx512.hpp"
#include "kernels_amx.hpp"
#include "tensor2D.hpp"

namespace AMX_MLP {

EnvVar SWPFA("SWPFA", 0); // prefetch A is slower, so disable it
EnvVar MHINT("MHINT", 0);

//
// MKernel make linear a compute-bound kernel (as much as possible)
//
class MKernel : public jit_generator {
public:
    TileConfig m_tile_cfg;

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
    MKernel() = default;

    MKernel(int M_hint) { setup(M_hint); }

    //  M_hint is only a hint for prefetching, set to 0 to avoid prefetch
    void setup(int M_hint = 0) {
        if ((int)MHINT > 0)
            M_hint = MHINT;

        if (M_hint == 0) {
            m_prefetch_Blines = 0;
        } else {
            m_prefetch_Blines = 32768 * sizeof(ov::bfloat16) / 64 / M_hint;
        }

        create_kernel("MKernel");
        tile_config_M(m_tile_cfg, M_hint);
    }

    // M can change w/o code-regeneration
    // with the help of :
    //  - m_BM_hint controls dynamic behaviour of the kernel
    //  - tile config controls behaviour of tileload & TMUL
    void tile_config_M(TileConfig& tile_cfg, int M) {
        auto rows0 = 16;
        auto rows1 = 16;
        if (M < 32) {
            // kernel is for processing Mtails
            if (M > 16) {
                rows0 = 16;
                rows1 = M - 16;
            } else {
                //  both A0 & A1 load from same memory, to avoid code-regeneration
                rows0 = rows1 = M;
            }
        }
        tile_cfg.reset(1, 0,
                         {
                             {rows0, 64}, // C00:0
                             {rows0, 64}, // C01:1
                             {rows1, 64}, // C10:2
                             {rows1, 64}, // C11:3
                             {rows0, 64}, // A0:4
                             {rows1, 64}, // A1:5
                             {16, 64},    // B0:6
                             {16, 64},    // B1:7
                         });
    }

    // row data is in layout [N, K], maybe smaller than [32, 16]
    template <typename T>
    void repackB(ov::bfloat16* dst, T* src, int N_stride, int N, int K) {

        if (N == 16 && K == 32 && std::is_same<T, ov::bfloat16>::value) {
            // SIMD optimized version
            // std::cout << "." << std::flush;
            amx_kernel::functional::transpose_epi32_16x16(dst, src, N_stride * sizeof(T));
            return;
        }

        assert(K <= 32);
        assert(N <= 16);
        int k = 0;
        ov::bfloat16 bf16zero(0.0f);
        for (; k < 32; k += 2) {
            int n = 0;
            bool is_k0_valid = (k) < K;
            bool is_k1_valid = (k + 1) < K;
            auto* psrc = src + k;
            for (; n < 16 && n < N; n++, psrc += N_stride) {
                *dst++ = is_k0_valid ? ov::bfloat16(psrc[0]) : bf16zero;
                *dst++ = is_k1_valid ? ov::bfloat16(psrc[1]) : bf16zero;
            }
            for (; n < 16; n++) {
                *dst++ = 0;
                *dst++ = 0;
            }
        }
    }

    // weight is supposed to be of shape[N, K], stride in unit of bytes
    // N should be m_BN
    // K should be m_BK
    template <typename T>
    tensor2D<ov::bfloat16> prepareB(T* p_weight, int stride, int N, int K) {
        tensor2D<ov::bfloat16> ret;
        ASSERT((N % 32) == 0);
        ASSERT((K % 32) == 0);
        // weight matrix is in unit of [N/32, Kx32]
        ret.resize(N / 32, K * 32, true);

        auto N_stride = stride / sizeof(T);
        for (int n = 0, blkn = 0; n < N; n += 32, blkn++) {
            for (int k = 0, blkk = 0; k < K; k += 32, blkk++) {
                // two adjacent 32x16 (512) block of weight: dst0 & dst1
                auto* dst0 = &ret(blkn, blkk * 1024);
                auto* dst1 = dst0 + 16 * 32;
                auto valid_k = (K - k) < 32 ? (K - k) : 32;

                auto* src0 = p_weight + n * N_stride + k;
                auto valid_n0 = (N - n) < 16 ? (N - n) : 16;
                repackB<T>(dst0, src0, N_stride, valid_n0, valid_k);

                auto* src1 = p_weight + (n + 16) * N_stride + k;
                auto valid_n1 = (N - (n + 16)) < 16 ? (N - (n + 16)) : 16;
                repackB<T>(dst1, src1, N_stride, valid_n1, valid_k);
            }
        }
        return ret;
    }

    // to save push/pop: do not use `abi_save_gpr_regs`
    uint8_t* prefetch_next_A_addr;

    struct call_args {
        const uint8_t* pA;  // bfloat16
        int64_t strideA;    // in bytes
        const uint8_t* pB;  // bfloat16
        const uint8_t* pC;  // float32
        int64_t strideC;    // in bytes
        const uint8_t* prefetch;
        int64_t k_tiles;    // K / 32
        int64_t do_accumulation;
        int64_t M;
    };

    void generate() {
        Xbyak::Reg64 reg_A_addr = abi_param2;
        Xbyak::Reg64 reg_A_stride = abi_param3;
        Xbyak::Reg64 reg_B_addr = abi_param4;
        Xbyak::Reg64 reg_C_addr = abi_param5;
        Xbyak::Reg64 reg_C_stride = abi_param6;

        Xbyak::Reg64 reg_ktiles = rax;
        Xbyak::Reg64 reg_B_stride = r10;
        Xbyak::Reg64 reg_A1_addr = r11;
        Xbyak::Reg64 reg_prefetch = r12;

        Xbyak::Tmm tmmC00 = tmm0;
        Xbyak::Tmm tmmC01 = tmm1;
        Xbyak::Tmm tmmC10 = tmm2;
        Xbyak::Tmm tmmC11 = tmm3;
        Xbyak::Tmm tmmA0 = tmm4;
        Xbyak::Tmm tmmA1 = tmm5;
        Xbyak::Tmm tmmB0 = tmm6;
        Xbyak::Tmm tmmB1 = tmm7;

        auto num_PFB = m_prefetch_Blines;
        int cur_PFB = 0;
        /*
                       B: 1x2 tiles
        A : 2x1 tiles  C: 2x2 tiles
        */
        Xbyak::Label loop_over_ktiles;
        Xbyak::Label skip_load;

        push(reg_prefetch);
        {
            auto reg_tmp = reg_B_stride;
            tilezero(tmmC00);
            tilezero(tmmC01);
            tilezero(tmmC10);
            tilezero(tmmC11);

            mov(reg_A_addr, ptr[abi_param1 + offsetof(call_args, pA)]);
            mov(reg_A_stride, ptr[abi_param1 + offsetof(call_args, strideA)]);
            mov(reg_B_addr, ptr[abi_param1 + offsetof(call_args, pB)]);
            mov(reg_C_addr, ptr[abi_param1 + offsetof(call_args, pC)]);
            mov(reg_C_stride, ptr[abi_param1 + offsetof(call_args, strideC)]);
            mov(reg_prefetch, ptr[abi_param1 + offsetof(call_args, prefetch)]);
            mov(reg_ktiles, ptr[abi_param1 + offsetof(call_args, k_tiles)]);

            lea(reg_A1_addr, ptr[reg_A_addr + reg_A_stride * 8]);
            lea(reg_A1_addr, ptr[reg_A1_addr + reg_A_stride * 8]);

            // reg_A1_addr = reg_A_addr if M <= 16 (to avoid tileloadd segmentfault)
            mov(reg_tmp, ptr[abi_param1 + offsetof(call_args, M)]);
            cmp(reg_tmp, 16);
            cmovle(reg_A1_addr, reg_A_addr);

            mov(reg_tmp, ptr[abi_param1 + offsetof(call_args, do_accumulation)]);
            and_(reg_tmp, 1);
            jz(skip_load);
            {
                auto reg_C1_addr = reg_tmp;
                tileloadd(tmmC00, ptr[reg_C_addr + reg_C_stride]);
                tileloadd(tmmC01, ptr[reg_C_addr + reg_C_stride + 64]);
                lea(reg_C1_addr, ptr[reg_C_addr + reg_C_stride * 8]);
                lea(reg_C1_addr, ptr[reg_C1_addr + reg_C_stride * 8]);
                tileloadd(tmmC10, ptr[reg_C1_addr + reg_C_stride]);
                tileloadd(tmmC11, ptr[reg_C1_addr + reg_C_stride + 64]);
            }
            L(skip_load);
        }

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

        tileloadd(tmmA1, ptr[reg_A1_addr + reg_A_stride]);
        tdpbf16ps(tmmC10, tmmA1, tmmB0);
        if (cur_PFB < num_PFB) {
            prefetcht2(ptr[reg_prefetch + cur_PFB * 64]);
            cur_PFB++;
        }

        tileloadd(tmmB1, ptr[reg_B_addr + reg_B_stride]);

        // prefetch [K_tiles X 256] bytes

        tdpbf16ps(tmmC01, tmmA0, tmmB1);
        if (cur_PFB < num_PFB) {
            prefetcht2(ptr[reg_prefetch + cur_PFB * 64]);
            cur_PFB++;
        }

        tdpbf16ps(tmmC11, tmmA1, tmmB1);
        // prefetch next sub-block B matrix
        if (cur_PFB < num_PFB) {
            for (int pi = cur_PFB; pi < num_PFB; pi++) {
                prefetcht2(ptr[reg_prefetch + pi * 64]);
            }
        }

        lea(reg_prefetch, ptr[reg_prefetch + 64 * num_PFB]);

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
        pop(reg_prefetch);
        ret();
    }

    // run L2 cache blocking kernel with size:
    //    [BM, BK]*[BK, BN] => [BM, BN]
    //
    // prefetch of A can be done inside of this level of kernel
    // since it's done in unit of 32-rows
    // but prefetch of next B must be specified by caller.
    //
    void run(int M,                              // actual M
             uint8_t* pA, int strideA,           // A [M, K]
             tensor2D<ov::bfloat16>& repacked_B, // B [N/32, K*32]
             uint8_t* pC, int strideC,           // C [M, N]
             uint8_t* prefetch_B,                // prefetch B
             bool do_accumulation) {
        call_args args;
        // number of blocks in N dimension (in unit of 32 columns)
        auto num_blkN = repacked_B.dims[0];
        auto K = repacked_B.dims[1] / 32;
        auto* pB = reinterpret_cast<uint8_t*>(&repacked_B[0]);
        auto strideB = repacked_B.stride;

        args.do_accumulation = do_accumulation;
        args.k_tiles = K / 32;
        args.strideA = strideA;
        args.strideC = strideC;
        args.prefetch = prefetch_B;
        assert((K % 32) == 0);

        auto prefetch_step = m_prefetch_Blines * 64 * args.k_tiles;

        // if (BM != m_BM_hint) it only effect prefetch of B which is not vital to function
        for (int m = 0; m < M; m += 32, pA += 32 * strideA, pC += 32 * strideC) {
            args.pB = pB;
            // prefetch_next_A_addr = pA + 32 * strideA;
            // if (m + 32 >= BM)
            //     prefetch_next_A_addr = pA;
            args.M = std::min(M - m, 32);
            args.pA = pA;
            for (int ni = 0; ni < num_blkN; ni++, args.pB += strideB, args.prefetch += prefetch_step) {
                args.pC = pC + ni * 32 * sizeof(float);
                (*this)(&args);
                //(*this)(pA, strideA, pB1, pC + ni * 32 * sizeof(float), strideC, prefetch_B);
                // prefetch_next_A_addr += 4 * strideA;
            }
        }
    }
};

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

class Linear {
public:
    struct Work {
        MKernel* p_jit_amx0 = nullptr;

        std::vector<tensor2D<ov::bfloat16>> weights;
        tensor2D<float> C;
        std::shared_ptr<std::atomic_int> sync_flag;
        int n0 = 0;
        int n1 = 0;
        int k0 = 0;
        int k1 = 0;
        int BN = 0;
        int blk_K_size = 0;
        operator bool() { return BN > 0; }

        // input : weight [N, K], setup repacks range of N [n_start, n_end)
        template <typename T>
        void setup(T* p_weight, int stride) {
            auto num_blk_K = (k1 - k0) / blk_K_size;
            auto* pw = p_weight + n0 * stride / sizeof(T) + k0;
            weights.resize(num_blk_K);
            for (int k = 0; k < num_blk_K; k++) {
                weights[k] = p_jit_amx0->prepareB(pw + k * blk_K_size, stride, BN, blk_K_size);
            }
        }

        void run(int M, uint8_t* pA, int strideA) {
            int num_blk_K = weights.size();

            TileConfig tile_cfg0;
            TileConfig tile_cfg1;

            auto Mtails = M % 32;
            auto Mbody = M - Mtails;
            p_jit_amx0->tile_config_M(tile_cfg0, 32);
            if (Mtails) {
                p_jit_amx0->tile_config_M(tile_cfg1, Mtails);
            }

            bool cur_tile_config = false;
            TileConfigScope tcfg(tile_cfg0);
            auto set_tile_config = [&](bool to_tail) {
                if (cur_tile_config != to_tail) {
                    if (to_tail)
                        tcfg.update(tile_cfg1);
                    else
                        tcfg.update(tile_cfg0);
                    cur_tile_config = to_tail;
                }
            };

            C.resize(Mbody + (Mtails? 32:0), BN);

            pA += k0 * sizeof(ov::bfloat16);
            auto pC = reinterpret_cast<uint8_t*>(&C[0]);
            bool do_accumulation = false;
            for (int ki = 0; ki < num_blk_K; ki++) {
                tensor2D<ov::bfloat16>& blockB = weights[ki];
                tensor2D<ov::bfloat16>& blockB1 = weights[(ki + 1) < num_blk_K ? (ki + 1) : ki];

                if (Mbody) {
                    set_tile_config(false);
                    p_jit_amx0->run(Mbody, pA + ki * blk_K_size * sizeof(ov::bfloat16), strideA, blockB, pC, C.stride,
                                    reinterpret_cast<uint8_t*>(&blockB1[0]), do_accumulation);
                }
                
                if (Mtails) {
                    set_tile_config(true);
                    p_jit_amx0->run(Mtails, pA + ki * blk_K_size * sizeof(ov::bfloat16) + Mbody*strideA, strideA, blockB, pC + Mbody*C.stride, C.stride,
                                reinterpret_cast<uint8_t*>(&blockB1[0]), do_accumulation);
                }
                do_accumulation = true;
            }
        }
    };
    std::vector<Work> works;

    int used_nthr = 0;
    bool do_splitK = false;

    Linear() {}

    // weight [N, K]
    // Gate & Up are interleaved in N dimension: 16-gate / 16-up
    // and post-ops will compute  silu(gate)*up in unit of 16 elements
    // and store out as bfloat16.
    template <typename T, int BM = 256>
    void setup(T* p_weight, int stride, int N, int K, bool _do_splitK = false) {
        static MKernel jit_amx0(BM);

        const int blk_K_size = 256;
        // prepare weights, split N among threads
        // in unit of 32
        ASSERT((N % 32) == 0);
        ASSERT((K % blk_K_size) == 0);
        auto nthr = get_nthr();
        auto num_blk_N = N / 32;
        auto num_blk_K = K / blk_K_size;
        works.resize(nthr);

        do_splitK = _do_splitK;
        auto K_splits = do_splitK ? 2 : 1;
        // every thread should do same amount of work, and some cores can be idle
        auto valid_nthr = nthr / K_splits;
        auto blkN_per_thread = (num_blk_N + valid_nthr - 1) / valid_nthr;
        auto start_blkN = 0;
        used_nthr = 0;
        auto blkK_per_thread = (num_blk_K + K_splits - 1) / K_splits;
        for (int ithr = 0; ithr < nthr; ithr += K_splits) {
            auto blkN = std::min(num_blk_N - start_blkN, blkN_per_thread);

            if (blkN) {
                auto shared_atomic = std::make_shared<std::atomic_int>(0);
                auto start_blkK = 0;
                for (int ik = 0; ik < K_splits; ik++) {
                    auto blk_K = std::min(num_blk_K - start_blkK, blkK_per_thread);

                    auto& work = works[ithr + ik];

                    work.p_jit_amx0 = &jit_amx0;
                    work.sync_flag = shared_atomic;
                    work.blk_K_size = blk_K_size;

                    work.n0 = (start_blkN) * 32;
                    work.n1 = (start_blkN + blkN) * 32;
                    work.BN = blkN * 32;
                    work.k0 = start_blkK * blk_K_size;
                    work.k1 = (start_blkK + blk_K) * blk_K_size;
                    start_blkK += blk_K;
                    used_nthr++;
                }
            }

            start_blkN += blkN;
        }

        ECOUT("Linear N,K=", N, ",", K, " used_nthr=", used_nthr, "  do_splitK=", do_splitK);

#pragma omp parallel
        {
            int ithr = omp_get_thread_num();
            auto& work = works[ithr];
            if (work) {
                work.setup(p_weight, stride);
            }
        }
    }

    // A bfloat16 [256,  num_blk_K * 256]
    void run(uint8_t* pA, int strideA, int M) {
#pragma omp parallel
        {
            int ithr = omp_get_thread_num();
            auto& work = works[ithr];
            if (work) {
                work.run(M, pA, strideA);
            }
        }
    }
    /*
        void run(uint8_t* pA, int strideA, int M, float* dstC, int strideC) {
    #pragma omp parallel
            {
                int ithr = omp_get_thread_num();
                auto& work = works[ithr];
                if (work) {
                    work.run(M, pA, strideA);
                    auto* src = &work.C[0];
                    auto* dst = dstC + work.n0;
                    auto strideS = work.C.stride / sizeof(*src);
                    auto strideD = strideC / sizeof(*dst);
                    for (int m = 0; m < M; m++, src += strideS, dst += strideD) {
                        // the prefetch distance is increased to ensure by the time store happens
                        // prefetch has done and no HW prefetcher is triggered
                        auto* prefetch_dst = (m + 2 < M) ? (dst + 2 * strideD) : (dst);
                        for (int n = 0; n < work.BN; n += 32) {
                            auto d0 = _mm512_loadu_ps(src + n);
                            auto d1 = _mm512_loadu_ps(src + n + 16);
                            _mm_prefetch(prefetch_dst + n, _MM_HINT_ET1);
                            _mm_prefetch(prefetch_dst + n + 16, _MM_HINT_ET1);
                            _mm512_storeu_ps(dst + n, d0);
                            _mm512_storeu_ps(dst + n + 16, d1);
                        }
                    }
                }
            }
        }
    */

    void run(uint8_t* pA, int strideA, int M, ov::bfloat16* dstC, int strideC) {
#pragma omp parallel
        {
            int ithr = omp_get_thread_num();
            auto& work = works[ithr];
            if (work) {
                work.run(M, pA, strideA);

                if (do_splitK) {
                    auto sync_id = work.sync_flag->fetch_add(1);
                    // (0,1) (2,3)
                    if (sync_id & 1) {
                        auto& peer = works[(ithr & 1) ? (ithr - 1) : (ithr + 1)];
                        // the other one has finished, we can do the reduce sum
                        auto* src0 = &work.C[0];
                        auto* src1 = &peer.C[0];
                        auto* dst = dstC + work.n0;
                        auto strideS = work.C.stride / sizeof(*src0);
                        auto strideD = strideC / sizeof(*dst);
                        for (int m = 0; m < M; m++, src0 += strideS, src1 += strideS, dst += strideD) {
                            // the prefetch distance is increased to ensure by the time store happens
                            // prefetch has done and no HW prefetcher is triggered
                            auto* prefetch_dst = (m + 2 < M) ? (dst + 2 * strideD) : (dst);
                            for (int n = 0; n < work.BN; n += 32) {
                                auto d0 = _mm512_loadu_ps(src0 + n);
                                auto d0b = _mm512_loadu_ps(src1 + n);
                                auto d1 = _mm512_loadu_ps(src0 + n + 16);
                                auto d1b = _mm512_loadu_ps(src1 + n + 16);
                                d0 = _mm512_add_ps(d0, d0b);
                                d1 = _mm512_add_ps(d1, d1b);
                                auto v_bh = _mm512_cvtne2ps_pbh(d1, d0);
                                _mm_prefetch(prefetch_dst + n, _MM_HINT_ET1);
                                _mm512_storeu_ps(dst + n, reinterpret_cast<__m512&>(v_bh));
                            }
                        }
                    }
                } else {
                    auto* src = &work.C[0];
                    auto* dst = dstC + work.n0;
                    auto strideS = work.C.stride / sizeof(*src);
                    auto strideD = strideC / sizeof(*dst);
                    for (int m = 0; m < M; m++, src += strideS, dst += strideD) {
                        // the prefetch distance is increased to ensure by the time store happens
                        // prefetch has done and no HW prefetcher is triggered
                        auto* prefetch_dst = (m + 2 < M) ? (dst + 2 * strideD) : (dst);
                        for (int n = 0; n < work.BN; n += 32) {
                            auto d0 = _mm512_loadu_ps(src + n);
                            auto d1 = _mm512_loadu_ps(src + n + 16);
                            auto v_bh = _mm512_cvtne2ps_pbh(d1, d0);
                            _mm_prefetch(prefetch_dst + n, _MM_HINT_ET1);
                            _mm512_storeu_ps(dst + n, reinterpret_cast<__m512&>(v_bh));
                        }
                    }
                }
            }
        }
    }

    // gate & up are interleaved: 16 gates + 16 up
    void runGateUp(uint8_t* pA, int strideA, int M, ov::bfloat16* dstC, int strideC) {
#pragma omp parallel
        {
            int ithr = omp_get_thread_num();
            auto& work = works[ithr];
            if (work.BN > 0) {
                work.run(M, pA, strideA);
                // K reduce is done, results of [M, BN] sub-block is ready in L2.
                // combine Gate & Up
                auto* src = &work.C[0];
                auto strideS = work.C.stride / sizeof(*src);
                auto* dst = dstC + (work.n0 / 2); // important output is only half of the total N
                auto strideD = strideC / sizeof(*dst);
                for (int m = 0; m < M; m++, src += strideS, dst += strideD) {
                    auto* prefetch_dst = (m + 1 < M) ? (dst + strideD) : (dst);
                    for (int n = 0, i = 0; n < work.BN; n += 32, i += 16) {
                        auto v_gate = _mm512_loadu_ps(src + n);
                        auto v_up = _mm512_loadu_ps(src + n + 16);
                        v_gate = silu_ps_avx512(v_gate);
                        v_up = _mm512_mul_ps(v_gate, v_up);
                        auto v_bh = _mm512_cvtneps_pbh(v_up);
                        // Greate Optimization:
                        //  following prefetchnta prevents L2 HW prefetcher prefetch interleaved
                        //  channels belonging to other cores which will causes too much cross-core cache coherent cost.
                        _mm_prefetch(prefetch_dst + i, _MM_HINT_ET1);
                        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), reinterpret_cast<__m256i&>(v_bh));
                    }
                }
            }
        }
    }
};

struct MLP {
    Linear gate_up;
    Linear down;
    MLP() {}

    tensor2D<ov::bfloat16> actUp;
    int m_N;
    int m_M = 0;

    // [M, K] x [N, K] => [M, N] x [K, N] => [M, K]
    // w_gate/w_up : [N, K]
    //     w_down  : [K, N]
    void setup(ov::bfloat16* pw_gate, ov::bfloat16* pw_up, ov::bfloat16* pw_down, int K, int N) {
        // [N, K] [N, K] interleave (16-16-...) into [2*N, K]
        tensor2D<ov::bfloat16> w_gate(N, K, pw_gate, K * sizeof(ov::bfloat16));
        tensor2D<ov::bfloat16> w_up(N, K, pw_up, K * sizeof(ov::bfloat16));
        tensor2D<ov::bfloat16> w_down(K, N, pw_down, N * sizeof(ov::bfloat16));

        tensor2D<ov::bfloat16> w_gate_up;
        w_gate_up.resize(2 * N, K, true);
        for (int n = 0; n < N; n += 16) {
            for (int i = 0; i < 16; i++)
                memcpy(&w_gate_up(2 * n + i, 0), &w_gate(n + i, 0), K * sizeof(ov::bfloat16));
            for (int i = 0; i < 16; i++)
                memcpy(&w_gate_up(2 * n + 16 + i, 0), &w_up(n + i, 0), K * sizeof(ov::bfloat16));
        }
        gate_up.setup(&w_gate_up[0], w_gate_up.stride, N * 2, K);
        down.setup(&w_down[0], w_down.stride, K, N, true);
        m_N = N;
    }

    void setM(int M) {
        if (m_M < M) {
            actUp.resize(M, m_N, false);
            m_M = M;
        }
    }

    void run(uint8_t* pA, int strideA, int M, ov::bfloat16* dstC, int strideC) {
        setM(M);

        gate_up.runGateUp(pA, strideA, M, &actUp[0], actUp.stride);
        down.run(reinterpret_cast<uint8_t*>(&actUp[0]), actUp.stride, M, dstC, strideC);
    }
};

}; // namespace AMX_MLP

void mlp_ref(tensor2D<ov::bfloat16>& A, tensor2D<ov::bfloat16>& gate, tensor2D<ov::bfloat16>& up, tensor2D<ov::bfloat16>& down,
             tensor2D<ov::bfloat16>& result) {
    auto M = A.dims[0];
    auto K = gate.dims[0];
    auto N = gate.dims[1];
    ASSERT(A.dims[1] == K);
    ASSERT(up.dims[0] == K);
    ASSERT(up.dims[1] == N);
    ASSERT(down.dims[0] == N);
    ASSERT(down.dims[1] == K);

    tensor2D<float> Agate(M, N, true);
    tensor2D<float> Aup(M, N, true);
    tensor2D<ov::bfloat16> act2(M, N, true);

    Agate = 0;
    Aup = 0;
    matmul(A, gate, Agate);
    matmul(A, up, Aup);

    // SiLu(gate) * up
    auto silu = [&](float x) { return x * 1.0f / (1.0f + std::exp(-x)); };

    auto silu_gate_up = [](float g, float u) {
        auto v_gate = _mm512_set1_ps(g);
        auto v_up = _mm512_set1_ps(u);
        auto vsilu = AMX_MLP::silu_ps_avx512(v_gate);
        v_up = _mm512_mul_ps(vsilu, v_up);
        auto v_bh = _mm512_cvtneps_pbh(v_up);
        int result = _mm512_cvtsi512_si32(reinterpret_cast<__m512i&>(v_bh));
        return ov::bfloat16::from_bits(result & 0xFFFF);
    };

    int num_error = 0;
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            auto gate = Agate(m, n);
            auto up = Aup(m, n);
            float f = silu_gate_up(gate, up);
            if (num_error < 10) {
                // f_reference
                float f_reference = silu(gate) * up;
                if (f_reference != f) {
                    ECOUT("\t  SiLU_diff @(", m, ", ", n, ") gate,up=", std::fixed, std::setprecision(3), gate, ",", up, "  ref:", f_reference, "!=", f);
                    num_error++;
                }
            }
            act2(m, n) = f;
        }
    }

    result = 0;
    matmul(act2, down, result);
}

EnvVar envM("M", 256);

void test1() {
    int M = envM;
    int K = 4096;
    int N = 11008;
    int num_layers = 32;
    std::vector<AMX_MLP::MLP> mlps(num_layers);
    tensor2D<ov::bfloat16> gate(K, N, true);
    tensor2D<ov::bfloat16> up(K, N, true);
    tensor2D<ov::bfloat16> down(N, K, true);
    tensor2D<ov::bfloat16> A(M, K, true);
    tensor2D<ov::bfloat16> C1(M, K, true);
    tensor2D<ov::bfloat16> C0(M, K, true);

    auto nthr = get_nthr();

    ECOUT("");
    mlp_ref(A, gate, up, down, C0);

    ECOUT("");
    auto gateT = gate.Tr();
    auto upT = up.Tr();
    auto downT = down.Tr();
    ECOUT("");

    for (auto& mlp : mlps)
        mlp.setup(&gateT[0], &upT[0], &downT[0], K, N);

    mlps[0].run(reinterpret_cast<uint8_t*>(&A[0]), A.stride, M, &C1[0], C1.stride);

    std::string acc;
    const char* acc_color = nullptr;
    if (C1.allclose(C0, 0.05, 0.01)) {
        acc = "[PASS]";
    } else {
        acc = "[FAIL]";
        acc_color = "1;31";
    }
    perf_log plog({
        {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, "HW_CYCLES"},
        {PERF_TYPE_RAW, 0x21a6, "BOUND_ON_LOADS"},
    });
    plog.reserve(512);
    plog.tag("MLP", M, K, N, acc);
    plog.color(acc_color);

    for (int r = 0; r < 4; r++) {
        plog();
        bool flag = true;
        for (auto& mlp : mlps) {
            auto& in = (flag) ? (A) : (C1);
            auto& out = (!flag) ? (A) : (C1);
            flag = !flag;
            plog([&]() { mlp.run(reinterpret_cast<uint8_t*>(&in[0]), in.stride, M, &out[0], out.stride); });
        }
        plog();
        flag = true;
        for (auto& mlp : mlps) {
            auto& in = (flag) ? (A) : (C1);
            auto& out = (!flag) ? (A) : (C1);
            flag = !flag;
            mlp.setM(M);
            plog([&]() { mlp.gate_up.runGateUp(reinterpret_cast<uint8_t*>(&in[0]), in.stride, M, &mlp.actUp[0], mlp.actUp.stride); },
                 4.0 * M * N * K / mlp.gate_up.used_nthr);
            plog([&]() { mlp.down.run(reinterpret_cast<uint8_t*>(&mlp.actUp[0]), mlp.actUp.stride, M, &out[0], out.stride); },
                 2.0 * M * N * K / mlp.down.used_nthr);
        }
    }
}

int main() {
    MSRConfig _msr;
    bool initAMX = initXTILE();
#pragma omp parallel
    { srand(0); }
    test1();
}
