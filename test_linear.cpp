#include "jit.hpp"
#include <vector>

#include "dnnl_kernels.hpp"

#if !defined(XBYAK64_GCC)
#error NOT SUPPORTED
#endif

/*
C = A @ B

               B: 1x2 tiles
A : 2x1 tiles  C: 2x2 tiles

A : [32, K]
B : [K, 32] repacked
C : [32, 32]
*/

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
    void* prefetch_ptr;
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

        mov(reg_B_stride, reinterpret_cast<uintptr_t>(&prefetch_ptr));
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

#include "bf16.hpp"
// #include "kernels_avx512.hpp"
#include "kernels_amx.hpp"
#include "tensor2D.hpp"

static tensor2D<ov::bfloat16> repack_weights(tensor2D<ov::bfloat16>& Bt) {
    int N = Bt.dims[0];
    int K = Bt.dims[1];
    tensor2D<ov::bfloat16> BPacked(K * N, 1, true);
    for (int n = 0, i = 0; n < N; n += 32) {
        for (int k = 0; k < K; k += 32) {
            amx_kernel::functional::transpose_epi32_16x16(&BPacked[i * 16 * 32], &Bt(n, k), Bt.stride);
            i++;
            amx_kernel::functional::transpose_epi32_16x16(&BPacked[i * 16 * 32], &Bt(n + 16, k), Bt.stride);
            i++;
        }
    }
    return BPacked;
}


class LinearNxN {
public:
    Linear32x32_AMX m_kernel_0;
    Linear32x32_AMX m_kernel_1;
    tensor2D<ov::bfloat16> m_B0;
    tensor2D<ov::bfloat16> m_B1;
    int m_Ktiles;
    int m_K;
    int m_M;
    int m_N;
    int m_flag;

    uint8_t fake_buff[256];
    LinearNxN(int K, int M, int N, ov::bfloat16* weight, int w_stride) : m_K(K), m_M(M), m_N(N), m_kernel_0(false), m_kernel_1(true) {
        m_flag = std::getenv("LIFLAGS") ? atoi(std::getenv("LIFLAGS")) : 0;
        m_Ktiles = m_K / 32;
        m_kernel_0.m_ktiles = m_Ktiles;
        m_kernel_1.m_ktiles = m_Ktiles;
        m_kernel_0.prefetch_ptr = fake_buff;
        m_kernel_1.prefetch_ptr = fake_buff;
        assert((m_K % 32) == 0);
        set_weight(weight, w_stride);
    }
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

    template <int kernel_idx = 0>
    void call_64x64(int x, int y, ov::bfloat16* A0, int strideA, ov::bfloat16* B0, float* C0, int strideC, int ktiles) {
        call_kernel<kernel_idx>(x + 0, y + 0, A0, strideA, B0, C0, strideC, ktiles);
        call_kernel<kernel_idx>(x + 32, y + 0, A0, strideA, B0, C0, strideC, ktiles);
        call_kernel<kernel_idx>(x + 32, y + 32, A0, strideA, B0, C0, strideC, ktiles);
        call_kernel<kernel_idx>(x + 0, y + 32, A0, strideA, B0, C0, strideC, ktiles);
    }

    template <int kernel_idx = 0>
    void call_128x128(int x0, int y0, ov::bfloat16* A0, int strideA, ov::bfloat16* B0, float* C0, int strideC, int ktiles) {
        // 520 Ops/cycle
        call_64x64<kernel_idx>(x0 + 0, y0 + 0, A0, strideA, B0, C0, strideC, ktiles);
        call_64x64<kernel_idx>(x0 + 64, y0 + 0, A0, strideA, B0, C0, strideC, ktiles);
        call_64x64<kernel_idx>(x0 + 64, y0 + 64, A0, strideA, B0, C0, strideC, ktiles);
        call_64x64<kernel_idx>(x0 + 0, y0 + 64, A0, strideA, B0, C0, strideC, ktiles);
    }

    void call_256x256(int x0, int y0, ov::bfloat16* A0, int strideA, ov::bfloat16* B0, float* C0, int strideC, int ktiles) {
        // 520 Ops/cycle
        call_128x128(x0 + 0, y0 + 0, A0, strideA, B0, C0, strideC, ktiles);
        call_128x128(x0 + 128, y0 + 0, A0, strideA, B0, C0, strideC, ktiles);
        call_128x128(x0 + 128, y0 + 128, A0, strideA, B0, C0, strideC, ktiles);
        call_128x128(x0 + 0, y0 + 128, A0, strideA, B0, C0, strideC, ktiles);
    }
    void call_512x512(int x0, int y0, ov::bfloat16* A0, int strideA, ov::bfloat16* B0, float* C0, int strideC, int ktiles) {
        call_256x256(x0 + 0, y0 + 0, A0, strideA, B0, C0, strideC, ktiles);
        call_256x256(x0 + 256, y0 + 0, A0, strideA, B0, C0, strideC, ktiles);
        call_256x256(x0 + 256, y0 + 256, A0, strideA, B0, C0, strideC, ktiles);
        call_256x256(x0 + 0, y0 + 256, A0, strideA, B0, C0, strideC, ktiles);
    }

    //float Cx[32*32];

    void call_general(int x0, int x1, int y0, int y1, ov::bfloat16* A0, int strideA, ov::bfloat16* B0, float* C0, int strideC) {
        // 128x128 : 560
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

                    m_kernel_0.prefetch_ptr = ptrA1;
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

        if (m_M == 32 && m_N == 32)
            call_kernel(0, 0, A0, strideA, B0, C0, strideC, m_Ktiles);
        else if (m_M == 64 && m_N == 64)
            call_64x64(0, 0, A0, strideA, B0, C0, strideC, m_Ktiles);
        else if (m_M == 128 && m_N == 128 && false)
            call_128x128(0, 0, A0, strideA, B0, C0, strideC, m_Ktiles);
        else if (m_M == 256 && m_N == 256 && false)
            call_256x256(0, 0, A0, strideA, B0, C0, strideC, m_Ktiles);
        else if (m_M == 512 && m_N == 512 && false)
            call_512x512(0, 0, A0, strideA, B0, C0, strideC, m_Ktiles);
        else
            call_general(0, m_N, 0, m_M, A0, strideA, B0, C0, strideC);
    }

    void call_128x128_splitK(int x0, int y0, ov::bfloat16* A0, int strideA, ov::bfloat16* B0, float* C0, int strideC, int ktiles) {
        // split along K dimension
        call_128x128<0>(x0, y0, A0, strideA, &m_B0[0], C0, strideC, ktiles / 2);
        call_128x128<1>(x0, y0, A0 + (ktiles / 2) * 32, strideA, &m_B1[0], C0, strideC, ktiles / 2);
    }

    void set_weight(ov::bfloat16* weight, int w_stride) {
        if (m_M == 128 && m_N == 128 && m_K == 4096 && false) {
            // split K
            tensor2D<ov::bfloat16> Bt0(m_N, m_K / 2, weight, w_stride);
            tensor2D<ov::bfloat16> Bt1(m_N, m_K / 2, weight + (m_K / 2), w_stride);
            m_B0 = repack_weights(Bt0);
            m_B1 = repack_weights(Bt1);
        } else {
            //
            tensor2D<ov::bfloat16> Bt(m_N, m_K, weight, w_stride);
            m_B0 = repack_weights(Bt);
        }
    }
};


class LinearReduceK {
public:
    Linear32x32_AMX m_kernel_0;
    Linear32x32_AMX m_kernel_1;
    tensor2D<ov::bfloat16> m_B;
    int m_Ktiles;
    int m_subKtiles = 8;
    int m_K;
    int m_M;
    int m_N;
    int m_flag;

    uint8_t fake_buff[256];
    LinearReduceK(int K, int M, int N, ov::bfloat16* weight, int w_stride) : m_K(K), m_M(M), m_N(N), m_kernel_0(false), m_kernel_1(true) {
        m_flag = std::getenv("LIFLAGS") ? atoi(std::getenv("LIFLAGS")) : 0;
        m_Ktiles = m_K / 32;
        m_kernel_0.m_ktiles = m_Ktiles;
        m_kernel_1.m_ktiles = m_Ktiles;
        m_kernel_0.prefetch_ptr = fake_buff;
        m_kernel_1.prefetch_ptr = fake_buff;
        assert((m_K % 32) == 0);
        set_weight(weight, w_stride);
    }
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

    void call_general(int x0, int x1, int y0, int y1, ov::bfloat16* A0, int strideA, ov::bfloat16* B0, float* C0, int strideC) {
        for (int kt = 0; kt < m_Ktiles; kt += m_subKtiles) {
            // 256=8x32, reduce on 8Ktile
            //
            auto curKtiles = (m_Ktiles - kt);
            if (curKtiles > m_subKtiles)
                curKtiles = m_subKtiles;
            
            auto curK = curKtiles * 32;
            auto* ptrA0 = reinterpret_cast<uint8_t*>(A0) + kt*32*sizeof(ov::bfloat16);
            auto* ptrC0 = reinterpret_cast<uint8_t*>(C0);

            // call 32x32 kernels (assume both M & N are integer multiple of 32)
            // 
            // prefetch for next subKtiles:
            //    M*K element of A
            //    N*K element of B
            // current subKtile will call 32x32 kernels (M/32)*(N/32) = M*N/1024 times
            // thus each 32x32 kernel needs to prefetch:
            //     1024*K/N element of A
            //     1024*K/M element of B
            // evenly prefetch A & B for next subKtiles

            // 32x32 kernel will loop (m_subKtiles) times, so in each iteration
            // we need to load:
            //    sub-block A:  [sizeof(bf16)*1024*(m_subKtiles*32)/N]/m_subKtiles = 2048*32/N bytes (64x4 bytes for N=256)
            //    sub-block B:  [sizeof(bf16)*1024*(m_subKtiles*32)/M]/m_subKtiles = 2048*32/M bytes (64x4 bytes for M=256)
            //
            // since 32x32 kernel has only one register-pointer for prefetch, we can prefetch A & B interleaved
            // 32x32 kernel will prefetch A or B for 64x8=512 bytes on each iteration.
            //
            // next sub-block A matrix is not contigours, it inner dimension of (m_subKtiles*32) = 256 (512bytes for bf16)
            // thus prefetch pointer has to jump over A's stride on each 512bytes.
            // that's not an issue
            //

            auto* prefetch_B = reinterpret_cast<uint8_t*>(B0) + (kt + m_subKtiles) * (m_N/32) * 2048;
            auto prefetch_blkB_bytes = sizeof(ov::bfloat16) * 1024*(m_subKtiles*32) / m_M;

            auto* prefetch_A = reinterpret_cast<uint8_t*>(A0) + (kt + m_subKtiles) * 32 * sizeof(ov::bfloat16);
            
            for (int y = y0; y < y1; y += 32, ptrA0 += 32 * strideA, ptrC0 += 32 * strideC) {
                auto* ptrB0 = reinterpret_cast<uint8_t*>(B0) + kt * (m_N/32) * 2048;
                // for prefetching next 32xK subA
                for (int x = x0; x < x1; x += 32, ptrB0 += curKtiles * 2048) {

                    // too many SW prefetch would also block CPU HW pipeline, so it must be mixed into kernel
                    //for(int i = 0; i < prefetch_blk_bytes; i += 64) _mm_prefetch(ptrA1 + i, _MM_HINT_T2);

                    if (kt == 0) {
                        m_kernel_0.prefetch_ptr = prefetch_B;
                        m_kernel_0.m_ktiles = curKtiles;
                        m_kernel_0(ptrA0, strideA, ptrB0, ptrC0 + x * sizeof(float), strideC, curKtiles);
                    } else {
                        m_kernel_1.prefetch_ptr = prefetch_B;
                        m_kernel_1.m_ktiles = curKtiles;
                        m_kernel_1(ptrA0, strideA, ptrB0, ptrC0 + x * sizeof(float), strideC, curKtiles);
                    }

                    // prefetch next 32xK subA
                    prefetch_B += prefetch_blkB_bytes;
                }
            }
        }
    }

    // B0: repacked
    void operator()(ov::bfloat16* A0, int strideA, float* C0, int strideC) {
        ov::bfloat16* B0 = &m_B[0];
        call_general(0, m_N, 0, m_M, A0, strideA, B0, C0, strideC);
        return;
    }

    void set_weight(ov::bfloat16* weight, int w_stride) {
        m_B = tensor2D<ov::bfloat16>(m_N * m_K, 1, true);

        int i = 0;
        for (int kt = 0; kt < m_Ktiles; kt += m_subKtiles) {
            auto curKtiles = (m_Ktiles - kt);
            if (curKtiles > m_subKtiles)
                curKtiles = m_subKtiles;
            tensor2D<ov::bfloat16> Bt(m_N, curKtiles * 32, weight + kt*32, w_stride);
            for (int n = 0; n < Bt.dims[0]; n += 32) {
                for (int k = 0; k < Bt.dims[1]; k += 32) {
                    amx_kernel::functional::transpose_epi32_16x16(&m_B[i * 16 * 32], &Bt(n, k), Bt.stride);
                    i++;
                    amx_kernel::functional::transpose_epi32_16x16(&m_B[i * 16 * 32], &Bt(n + 16, k), Bt.stride);
                    i++;
                }
            }
        }
    }
};

#include "timeit.hpp"

#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#include <stdlib.h>

#include <omp.h>

timeit timer({
    {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, "HW_CYCLES"},
    //{PERF_TYPE_RAW, 0x3c, "CPU_CLK_UNHALTED.THREAD"},
    //{PERF_TYPE_RAW, 0x81d0, "MEM_LOAD_RETIRED.ALL_LOADS"},
    //{PERF_TYPE_HW_CACHE, 0x10002, "LLC_load_misses"},
    //{PERF_TYPE_HW_CACHE, 0x2, "LLC_loads"},
    //{PERF_TYPE_RAW, 0x02b1, "UOPS_EXECUTED.CORE"},
});

template<typename LinearKernel = LinearNxN>
int amx_jit(const int M, const int N, const int K, int times = -1000) {
    tensor2D<ov::bfloat16> A(M, K,
                             true); // ensure stride of A matrix is multiple of
                                    // cache line, which is vital to performance.
    tensor2D<ov::bfloat16> B(K, N, true);
    auto Bt = B.Tr();

    tensor2D<float> C0(M, N, true); // reference result
    tensor2D<float> C1(M, N, true); // actual result
    LinearKernel mm_jit(K, M, N, &Bt[0], Bt.stride);

    C0 = 0;
    matmul(A, B, C0);

    std::string acc;
    std::string acc_color;
    {
        TileConfigScope tcfg(mm_jit.tile_config());
        mm_jit(&A[0], A.stride, &C1[0], C1.stride);
    }

    if (C0 == C1) {
        acc = "[PASS]";
    } else {
        if (std::getenv("SHOW_ERR")) {
            std::cout << "============= A ================ " << std::endl;
            std::cout << A << std::endl;
            std::cout << "============= B ================ " << std::endl;
            std::cout << B << std::endl;
            logger() << C0 << std::endl;
            logger() << C1 << std::endl;
        }
        acc = "[FAIL]";
        acc_color = "1;31";
    }

    timer.tag(__func__, " (M=", M, ",N=", N, ",K=", K, ")", acc)
        .color(acc_color)(
            times,
            [&]() {
                TileConfigScope tcfg(mm_jit.tile_config());
                mm_jit(&A[0], A.stride, &C1[0], C1.stride);
            },
            2.0 * M * N * K // OPS per call
        );

    return 0;
}


void clflush(void* pv, int bytes) {
    auto* p = reinterpret_cast<uint8_t*>(pv);
    for (int i = 0; i < bytes; i += 64) {
        _mm_clflushopt(p + i);
    }
    _mm_mfence();
};

void clflush(tensor2D<ov::bfloat16>& t) {
    clflush(&t[0], t.capacity);
};

void sw_prefetch_L2(void* pv, int bytes) {
    auto* p = reinterpret_cast<uint8_t*>(pv);
    int i;
    for (i = 0; i + 256 <= bytes; i += 64 * 4) {
        _mm_prefetch(p + i, _MM_HINT_T2);
        _mm_prefetch(p + i + 64, _MM_HINT_T2);
        _mm_prefetch(p + i + 64 * 2, _MM_HINT_T2);
        _mm_prefetch(p + i + 64 * 3, _MM_HINT_T2);
    }
    for (; i < bytes; i += 64) {
        _mm_prefetch(p + i, _MM_HINT_T2);
    }
    _mm_mfence();
};

void load_prefetch_L2(void* pv, int bytes) {
    auto* p = reinterpret_cast<uint8_t*>(pv);
    int i;
    auto sum0 = _mm512_setzero_epi32();
    auto sum1 = _mm512_setzero_epi32();
    auto sum2 = _mm512_setzero_epi32();
    auto sum3 = _mm512_setzero_epi32();
    for (i = 0; i + 256 <= bytes; i += 64 * 4) {
        auto a0 = _mm512_loadu_epi32(p + i);
        auto a1 = _mm512_loadu_epi32(p + i + 64);
        auto a2 = _mm512_loadu_epi32(p + i + 64*2);
        auto a3 = _mm512_loadu_epi32(p + i + 64*3);
        sum0 = _mm512_add_epi32(sum0, a0);
        sum1 = _mm512_add_epi32(sum1, a1);
        sum2 = _mm512_add_epi32(sum2, a2);
        sum3 = _mm512_add_epi32(sum3, a3);
    }
    sum0 = _mm512_add_epi32(sum0, sum1);
    sum2 = _mm512_add_epi32(sum2, sum3);
    sum0 = _mm512_add_epi32(sum0, sum2);
    if (_mm512_cvtsi512_si32(sum0) > 0) {
        std::cout << 1;
    }
};

int amx_jit_special_reduceK(const int M, const int N, const int K) {
    tensor2D<ov::bfloat16> A(M, K, true);
    tensor2D<ov::bfloat16> B(K, N, true);
    auto Bt = B.Tr();

    tensor2D<float> C0(M, N, true); // reference result
    tensor2D<float> C1(M, N, true); // actual result
    LinearNxN       mm_jit(K, M, N, &Bt[0], Bt.stride);
    LinearReduceK   mm_rk(K, M, N, &Bt[0], Bt.stride);
    timer._clflush = 0;

    auto do_benchmarks_jit = [&]() {
        for (int i = 0; i < 3; i++) {
            timer.tag("NosplitK (M=", M, ",N=", N, ",K=", K, ")")(
                1,
                [&]() {
                    TileConfigScope tcfg(mm_jit.tile_config());
                    mm_jit(&A[0], A.stride, &C1[0], C1.stride);
                },
                2.0 * M * N * K);
        }
    };
    auto do_benchmarks_rk = [&]() {
        for (int i = 0; i < 3; i++) {
            timer.tag("ReduceK  (M=", M, ",N=", N, ",K=", K, ")")(
                1,
                [&]() {
                    TileConfigScope tcfg(mm_rk.tile_config());
                    mm_rk(&A[0], A.stride, &C1[0], C1.stride);
                },
                2.0 * M * N * K);
        }
    };

    ECOUT("-------------");
    do_benchmarks_jit();
    do_benchmarks_rk();

    ECOUT("-------------");
    do_benchmarks_jit();
    do_benchmarks_rk();

    ECOUT("----- clflush B matrix for jit --------");
    clflush(mm_jit.m_B0);
    do_benchmarks_jit();

    ECOUT("----- clflush B matrix for rk --------");
    clflush(mm_rk.m_B);
    do_benchmarks_rk();

    ECOUT("----- clflush A matrix for jit --------");
    clflush(A);
    do_benchmarks_jit();

    ECOUT("----- clflush A matrix for rk --------");
    clflush(A);
    do_benchmarks_rk();


    ECOUT("----- clflush A&B matrix for jit --------");
    clflush(A);
    clflush(mm_jit.m_B0);
    do_benchmarks_jit();

    ECOUT("----- clflush A&B matrix for rk --------");
    clflush(A);
    clflush(mm_rk.m_B);
    do_benchmarks_rk();

    return 0;
}

int amx_jit_special(const int M, const int N, const int K) {
    tensor2D<ov::bfloat16> A(M, K, true);
    tensor2D<ov::bfloat16> B(K, N, true);
    auto Bt = B.Tr();

    tensor2D<float> C0(M, N, true); // reference result
    tensor2D<float> C1(M, N, true); // actual result
    LinearNxN mm_jit(K, M, N, &Bt[0], Bt.stride);

    C0 = 0;
    matmul(A, B, C0);

    std::string acc;
    std::string acc_color;
    {
        TileConfigScope tcfg(mm_jit.tile_config());
        mm_jit(&A[0], A.stride, &C1[0], C1.stride);
    }

    if (C0 == C1) {
        acc = "[PASS]";
    } else {
        if (std::getenv("SHOW_ERR")) {
            std::cout << "============= A ================ " << std::endl;
            std::cout << A << std::endl;
            std::cout << "============= B ================ " << std::endl;
            std::cout << B << std::endl;
            logger() << C0 << std::endl;
            logger() << C1 << std::endl;
        }
        acc = "[FAIL]";
        acc_color = "1;31";
    }

    timer.tag(__func__, " (M=", M, ",N=", N, ",K=", K, ")", acc).color(acc_color);


    {
        tensor2D<uint8_t> D(512 * 1024 * 1024, 1, true);
        std::cout << "------- clear-cache " << D.capacity << " bytes then SW prefetch them back -------\n";
        for (int i = 0; i < 4; i++) {
            // D is too large to fit any level of cache, thus no need to flush it out of the cache
            auto latency = timer(1, [&]() { sw_prefetch_L2(&D[0], D.capacity); });
            std::cout << "    DDR=>L2 bandwidth: " << D.capacity * 1e-9 / latency << " GB/s" << std::endl;
        }
    }

    timer._clflush = 0;

    auto do_benchmarks = [&]() {
        for (int i = 0; i < 3; i++) {
            timer(
                1,
                [&]() {
                    TileConfigScope tcfg(mm_jit.tile_config());
                    mm_jit(&A[0], A.stride, &C1[0], C1.stride);
                },
                M * N * K * 2);
        }
    };

    ECOUT("------- memcpy w/o any prefetching-------");
    timer.clear_cache();
    do_benchmarks();

    ECOUT("------- clflush B w/o any prefetching------- +", mm_jit.m_B0.capacity);
    //clflush(&A[0], A.capacity);
    clflush(&mm_jit.m_B0[0], mm_jit.m_B0.capacity);
    //clflush(&C1[0], C1.capacity);
    do_benchmarks();

    ECOUT("------- clflush A w/o any prefetching------- +", A.capacity);
    clflush(&A[0], A.capacity);
    //clflush(&mm_jit.m_B0[0], mm_jit.m_B0.capacity);
    //clflush(&C1[0], C1.capacity);
    do_benchmarks();

    ECOUT("------- clflush C w/o any prefetching------- +", C1.capacity);
    clflush(&C1[0], C1.capacity);
    do_benchmarks();

    ECOUT("------- clflush A/B w/o any prefetching------- +", A.capacity);
    clflush(&A[0], A.capacity);
    clflush(&mm_jit.m_B0[0], mm_jit.m_B0.capacity);
    //clflush(&C1[0], C1.capacity);
    do_benchmarks();

    ECOUT("------- clflush A/B/C w/o any prefetching------- +", C1.capacity);
    clflush(&A[0], A.capacity);
    clflush(&mm_jit.m_B0[0], mm_jit.m_B0.capacity);
    clflush(&C1[0], C1.capacity);
    do_benchmarks();

    ECOUT("------- memcpy + clflush w/o any prefetching-------");
    timer.clear_cache();
    clflush(&A[0], A.capacity);
    clflush(&mm_jit.m_B0[0], mm_jit.m_B0.capacity);
    clflush(&C1[0], C1.capacity);
    do_benchmarks();

    ECOUT("------- clear-cache & tileload : whole L2 is cleared, including code & stack? -------");
    timer.clear_cache();
    auto swpf_latency = timer(1, [&]() {
        sw_prefetch_L2(&A[0], A.capacity);
        sw_prefetch_L2(&mm_jit.m_B0[0], mm_jit.m_B0.capacity);
        sw_prefetch_L2(&C1[0], C1.capacity);
    });
    ECOUT("    prefetch ", A.capacity + mm_jit.m_B0.capacity + C1.capacity, " bytes, DDR=>L2 bandwidth = ", (A.capacity + mm_jit.m_B0.capacity + C1.capacity) * 1e-9 / swpf_latency, " GB/s");
    do_benchmarks();

    ECOUT("------- clear-cache & prefetch : only A/B/C is cleared, SW prefetch to L2 can recover them -------");
    clflush(&A[0], A.capacity);                     // 512K data flushed + 30us
    clflush(&mm_jit.m_B0[0], mm_jit.m_B0.capacity); // 512K data flushed +28us
    //clflush(&C1[0], C1.capacity);                   //  101->91 = 10
    swpf_latency = timer(1, [&]() {
        sw_prefetch_L2(&A[0], A.capacity);                     // 37us to fetch / 26us overhead
        sw_prefetch_L2(&mm_jit.m_B0[0], mm_jit.m_B0.capacity); // 34us to fetch / 30us overhead
        //sw_prefetch_L2(&C1[0], C1.capacity);                   // 5us to fetch / 10us overhead
    });
    ECOUT("    prefetch ", A.capacity + mm_jit.m_B0.capacity + C1.capacity, " bytes, DDR=>L2 bandwidth = ", (A.capacity + mm_jit.m_B0.capacity + C1.capacity) * 1e-9 / swpf_latency, " GB/s");
    do_benchmarks();

    ECOUT("------- SW prefetch to L2 when data is already there -------");
    swpf_latency = timer(1, [&]() {
        sw_prefetch_L2(&A[0], A.capacity);
        sw_prefetch_L2(&mm_jit.m_B0[0], mm_jit.m_B0.capacity);
        sw_prefetch_L2(&C1[0], C1.capacity);
    });
    do_benchmarks();

    ECOUT("------- only A is flushed, prefetch A before kernel -------");
    clflush(&A[0], A.capacity);
    swpf_latency = timer(1, [&]() {
        sw_prefetch_L2(&A[0], A.capacity);
    });
    do_benchmarks();

    ECOUT("------- only A is flushed, prefetch A row by row -------");
    clflush(&A[0], A.capacity);
    do_benchmarks();

    ECOUT("------- only A is flushed, prefetch A row by row + 32 rows prelog-------");
    clflush(&A[0], A.capacity);
    swpf_latency = timer(1, [&]() {
        //sw_prefetch_L2(&A[0], A.stride * (32));
        //sw_prefetch_L2(&mm_jit.m_B0[0], mm_jit.m_B0.capacity);
        load_prefetch_L2(&A[0], A.stride * (32));
    });
    do_benchmarks();

    return 0;
}


int amx_mm(const int M, const int N, int K, int times = -1000) {
    tensor2D<ov::bfloat16> A(M, K, true); // ensure stride of A matrix is multiple of cache line, which is vital to performance.
    tensor2D<ov::bfloat16> B(K, N, true);
    auto Bt = B.Tr();
    std::vector<ov::bfloat16> BPacked(K * N, 0);
    tensor2D<float> C0(M, N, true); // reference result
    tensor2D<float> C1(M, N, true); // actual result
    amx_kernel::Matmul<ov::bfloat16, ov::bfloat16> mm32x32(true, true);
    amx_kernel::PP::BiasGeluStore<float, amx_kernel::PP::Steps::NONE> pp(C1);

    std::string acc;
    std::string acc_color;
    C0 = 0;
    matmul(A, B, C0);

    mm32x32(A, Bt, 0, N, pp);
    if (C0 == C1) {
        acc = "[PASS]";
    } else {
        acc_color = "1;31";
        acc = "[FAIL]";
    }

    timer.tag(__func__, "  (M=", M, ",N=", N, ",K=", K, ")", acc)
        .color(acc_color)(
            times, [&]() { mm32x32(A, Bt, 0, N, pp); },
            2.0 * M * N * K // OPS per call
        );

    return 0;
}

int amx_dnnl(const int M, const int N, int K, int times = -1000) {
    tensor2D<ov::bfloat16> A(M, K, true); // ensure stride of A matrix is multiple of cache line, which is vital to performance.
    tensor2D<ov::bfloat16> B(K, N, true); // [IC, OC]
    tensor2D<float> C0(M, N, true);       // reference result
    tensor2D<float> C1(M, N, true);       // actual result
    DNNLInnerProduct mmdnnl(M, N, K, &B[0], true);

    mmdnnl.set_A(&A[0]);
    mmdnnl.set_C(&C1[0]);

    std::string acc;
    std::string acc_color;
    C0 = 0;
    matmul(A, B, C0);

    mmdnnl.run();
    if (C0 == C1) {
        acc = "[PASS]";
    } else {
        acc_color = "1;31";
        acc = "[FAIL]";
    }

    timer.tag(__func__, "(M=", M, ",N=", N, ",K=", K, ")", acc)
        .color(acc_color)(
            times, [&]() { mmdnnl.run(); },
            2.0 * M * N * K // OPS per call
        );

    return 0;
}

#include "test_bw.hpp"

int main(int argc, const char* argv[]) {
    srand(0);
    bool initAMX = initXTILE();

    /*
    // using big AMX matmul as cache cleaner to avoid AMX being open/close Port5
    Linear32x32_AMX clr_cache_amx(false);
    const int ABsize = 64 * 1024 * 1024;
    std::vector<ov::bfloat16> clr_cache_A(ABsize, 1.0f);
    std::vector<ov::bfloat16> clr_cache_B(ABsize, 2.0f);
    float clr_cache_C[32 * 32];

    timer.hook_clear_cache = [&]() {
        // std::cout << "clr_cache_amx\n";
        TileConfigScope tcfg(clr_cache_amx.tile_config());
        int K = ABsize / 32;
        clr_cache_amx(&clr_cache_A[0], K * sizeof(ov::bfloat16), &clr_cache_B[0], clr_cache_C, 32 * sizeof(float), K / 32);
        clr_cache_amx(&clr_cache_A[0], K * sizeof(ov::bfloat16), &clr_cache_B[0], clr_cache_C, 32 * sizeof(float), K / 32);
    };
    */

    timer.set_app(argv[0]);

    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    std::cout << ANSIcolor("31") << "omp_get_num_threads() = " << omp_get_num_threads() << std::endl << ANSIcolor();

    //test_all_bw(3);

    std::cout << "===============================128x128 (==L2)========================\n";
    if (0) {
        amx_mm(128, 128, 2048);
        amx_dnnl(128, 128, 2048);
        amx_jit(128, 128, 2048);
    }

    if (0) {
        std::cout << "===============================256x256x4096 (>>L2)========================\n";
        for (int i = 0; i < 2; i++) {
            amx_mm(256, 256, 4096);
            amx_dnnl(256, 256, 4096);
            amx_jit(256, 256, 4096);
            amx_jit<LinearReduceK>(256, 256, 4096);
        }
        std::cout << "===============================256x256x11008 (>>L2)========================\n";
        for (int i = 0; i < 2; i++) {
            amx_mm(256, 256, 11008);
            amx_dnnl(256, 256, 11008);
            amx_jit(256, 256, 11008);
            amx_jit<LinearReduceK>(256, 256, 11008);
        }
        std::cout << "===============================256x256x25600 (>>L2)========================\n";
        for (int i = 0; i < 2; i++) {
            amx_mm(256, 256, 25600);
            amx_dnnl(256, 256, 25600);
            amx_jit(256, 256, 25600);
            amx_jit<LinearReduceK>(256, 256, 25600);
        }
        return 0;
    }

    //amx_jit_special_reduceK(256, 256, 4096); return 0;
    //amx_jit_special_reduceK(256, 256, 2560);
    amx_jit_special_reduceK(256, 256, 25600); return 0;

    if (0) {
        for(int i =0; i < 4; i++){
        //timer._clflush = 0;
        std::cout << "=============================== no split on K ========================\n";
        amx_jit(256, 256, 256);
        amx_jit(256, 256, 512);
        amx_jit(256, 256, 1024);
        amx_jit(256, 256, 2048);
        //amx_jit(512, 512, 128);
        //amx_jit(512, 512, 256);
        //amx_jit(512, 512, 512);
        std::cout << "=============================== split on K ========================\n";
        amx_jit<LinearReduceK>(256, 256, 256);
        amx_jit<LinearReduceK>(256, 256, 512);
        amx_jit<LinearReduceK>(256, 256, 1024);
        amx_jit<LinearReduceK>(256, 256, 2048);
        }
        return 0;
    }

    amx_jit_special(128, 128, 2048);
    std::cout << "===============================256x256 (==L2)========================\n";
    amx_jit_special(256, 256, 1024);
    //amx_jit_special(256, 256, 512);

    return 0;

    timer._clflush = 1;
    amx_jit(256, 256, 512);
    amx_jit(256, 256, 1024);
    amx_jit(256, 256, 2048);
    amx_jit(512, 512, 1024);
    amx_jit(256, 256, 4096);
    
    return 0;


    std::cout << "===============================BF16========================\n";
    amx_mm(32, 32, 128);
    amx_jit(32, 32, 128);
    amx_mm(32, 32, 128);
    amx_jit(32, 32, 128);

    std::cout << "===============================32x32 (L2)========================\n";
    for (int i = 0; i < 2; i++) {
        amx_mm(32, 32, 4096);
        amx_jit(32, 32, 4096);
    }
    std::cout << "===============================64x64 (L2)========================\n";
    for (int i = 0; i < 2; i++) {
        amx_mm(64, 64, 4096);
        amx_dnnl(64, 64, 4096);
        amx_jit(64, 64, 4096);
    }

    std::cout << "===============================128x128 (==L2)========================\n";
    for (int i = 0; i < 2; i++) {
        amx_mm(128, 128, 4096);
        amx_dnnl(128, 128, 4096);
        amx_jit(128, 128, 4096);
    }
    std::cout << "===============================128x128 (==L2)========================\n";
    for (int i = 0; i < 2; i++) {
        amx_mm(128, 128, 2048);
        amx_dnnl(128, 128, 2048);
        amx_jit(128, 128, 2048);
    }
    std::cout << "===============================192x192 (>L2)========================\n";
    for (int i = 0; i < 2; i++) {
        amx_mm(192, 192, 4096);
        amx_dnnl(192, 192, 4096);
        amx_jit(192, 192, 4096);
    }

    std::cout << "===============================256x256 (>L2)========================\n";
    for (int i = 0; i < 2; i++) {
        amx_mm(256, 256, 4096);
        amx_dnnl(256, 256, 4096);
        amx_jit(256, 256, 4096);
    }
    std::cout << "===============================512x512 (>L2)========================\n";
    for (int i = 0; i < 2; i++) {
        amx_mm(512, 512, 1024);
        amx_dnnl(512, 512, 1024);
        amx_jit(512, 512, 1024);
    }
    std::cout << "===============================512x512 (>L2)========================\n";
    for (int i = 0; i < 2; i++) {
        amx_mm(512, 512, 2048);
        amx_dnnl(512, 512, 2048);
        amx_jit(512, 512, 2048);
    }
    std::cout << "===============================256x160 (>L2)========================\n";
    for (int i = 0; i < 2; i++) {
        amx_mm(256, 160, 4096);
        amx_dnnl(256, 160, 4096);
        amx_jit(256, 160, 4096);
    }
    std::cout << "===============================256x320 (>L2)========================\n";
    for (int i = 0; i < 2; i++) {
        amx_mm(256, 320, 4096);
        amx_dnnl(256, 320, 4096);
        amx_jit(256, 320, 4096);
    }

    std::cout << "===============================256x4096 (>L2)========================\n";
    for (int i = 0; i < 2; i++) {
        amx_mm(256, 4096, 4096);
        amx_dnnl(256, 4096, 4096);
        amx_jit(256, 4096, 4096);
    }
#if 0
    std::cout << "===============================32x32 (LLC)========================\n";
    for (int i = 0; i < 2; i++) {
        amx_mm(32, 32, 4096 * 16);
        amx_jit<Linear32x32_AMX>(32, 32, 4096 * 16);
    }
    std::cout << "===============================64x64 (LLC)========================\n";
    for (int i = 0; i < 2; i++) {
        amx_mm(64, 64, 4096 * 16);
        amx_jit<Linear64x64>(64, 64, 4096 * 16);
    }
#endif
}