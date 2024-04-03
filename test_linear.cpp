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

    // to save push/pop: do not use `abi_save_gpr_regs`
    Xbyak::Reg64 reg_A_addr = abi_param1;
    Xbyak::Reg64 reg_A_stride = abi_param2;
    Xbyak::Reg64 reg_B_addr = abi_param3;
    Xbyak::Reg64 reg_C_addr = abi_param4;
    Xbyak::Reg64 reg_C_stride = abi_param5;
    Xbyak::Reg64 reg_ktiles = abi_param6;
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
        lea(reg_A1_addr, ptr[reg_A_addr + reg_A_stride * 8]);
        lea(reg_A1_addr, ptr[reg_A1_addr + reg_A_stride * 8]);
        mov(reg_B_stride, 64);

        auto const_A_steps = 64;

        bool is_matrix_A_blocked = std::getenv("ABLK") != nullptr;
        if (is_matrix_A_blocked) {
            // if matrix is blocked in 16x32, ops/cycle 630=>700
            mov(reg_A_stride, 64);
            const_A_steps = 1024;
        }

        bool do_sw_prefetch = std::getenv("SWPF") != nullptr;

        align(64, false);
        L(loop_over_ktiles);
        // for (int k = 0; k < Ktiles; k++) {
        tileloadd(tmmA0, ptr[reg_A_addr + reg_A_stride]);
        if (is_matrix_A_blocked && do_sw_prefetch) {
            for (int i = 0; i < 1024; i += 64)
                prefetcht0(ptr[reg_A_addr + 4096 + i]);
        }
        tileloadd(tmmB0, ptr[reg_B_addr + reg_B_stride]);
        if (do_sw_prefetch) {
            for (int i = 0; i < 1024; i += 64)
                prefetcht0(ptr[reg_B_addr + 4096 + i]);
        }
        lea(reg_B_addr, ptr[reg_B_addr + 1024]);

        tdpbf16ps(tmmC00, tmmA0, tmmB0);

        tileloadd(tmmA1, ptr[reg_A1_addr + reg_A_stride]);
        if (is_matrix_A_blocked && do_sw_prefetch) {
            for (int i = 0; i < 1024; i += 64)
                prefetcht0(ptr[reg_A1_addr + 4096 + i]);
        }

        tdpbf16ps(tmmC10, tmmA1, tmmB0);

        tileloadd(tmmB1, ptr[reg_B_addr + reg_B_stride]);
        if (do_sw_prefetch) {
            for (int i = 0; i < 1024; i += 64)
                prefetcht0(ptr[reg_B_addr + 4096 + i]);
        }

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
    LinearNxN(int K, int M, int N, ov::bfloat16* weight, int w_stride) : m_K(K), m_M(M), m_N(N), m_kernel_0(false), m_kernel_1(true) {
        m_flag = std::getenv("LIFLAGS") ? atoi(std::getenv("LIFLAGS")) : 0;
        m_Ktiles = m_K / 32;
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

    EnvVar PFAB{"PFAB", 8};

    void call_general(int x0, int x1, int y0, int y1, ov::bfloat16* A0, int strideA, ov::bfloat16* B0, float* C0, int strideC) {
        // 128x128 : 560
        if (y1 - y0 >= x1 - x0) {
            auto* ptrA0 = reinterpret_cast<uint8_t*>(A0);
            auto* ptrC0 = reinterpret_cast<uint8_t*>(C0);
            for (int y = y0; y < y1; y += 32, ptrA0 += 32 * strideA, ptrC0 += 32 * strideC) {
                auto* ptrB0 = reinterpret_cast<uint8_t*>(B0);
                for (int x = x0; x < x1; x += 32, ptrB0 += m_Ktiles * 2048) {
                    m_kernel_0(ptrA0, strideA, ptrB0, ptrC0 + x * sizeof(float), strideC, m_Ktiles);

                    // call_kernel(x, y, A0, strideA, B0, C0, strideC, m_Ktiles);
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
        if (m_M == 32 && m_N == 32)
            call_kernel(0, 0, A0, strideA, B0, C0, strideC, m_Ktiles);
        else if (m_M == 64 && m_N == 64)
            call_64x64(0, 0, A0, strideA, B0, C0, strideC, m_Ktiles);
        else if (m_M == 128 && m_N == 128 && false)
            call_128x128(0, 0, A0, strideA, B0, C0, strideC, m_Ktiles);
        else if (m_M == 256 && m_N == 256)
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

int amx_jit(const int M, const int N, const int K, int times = -1000) {
    tensor2D<ov::bfloat16> A(M, K,
                             true); // ensure stride of A matrix is multiple of
                                    // cache line, which is vital to performance.
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

    timer.tag(__func__, " (M=", M, ",N=", N, ",K=", K, ")", acc)
        .color(acc_color)(
            times,
            [&]() {
                TileConfigScope tcfg(mm_jit.tile_config());
                mm_jit(&A[0], A.stride, &C1[0], C1.stride);
            },
            M * N * K * 2 // OPS per call
        );

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

    auto clflush = [](void * pv, int bytes) {
        auto* p = reinterpret_cast<uint8_t*>(pv);
        for(int i = 0; i < bytes; i+=64) {
            _mm_clflush(p + i);
        }
    };

    auto sw_prefetch_L2 = [](void * pv, int bytes) {
        auto* p = reinterpret_cast<uint8_t*>(pv);
        for(int i = 0; i < bytes; i+=64) {
            _mm_prefetch(p + i, _MM_HINT_T1);
        }
    };

    timer._clflush = 0;
    for (int i = 0; i < 15; i++) {
        if (i == 2) {
            std::cout << "------- clear-cache -------\n";
            timer.clear_cache();
        }
        if (i == 5) {
            std::cout << "------- clear-cache & tileload : whole L2 is cleared, including code & stack? -------\n";
            timer.clear_cache();
            timer(1, [&](){
                sw_prefetch_L2(&A[0], A.capacity);
                sw_prefetch_L2(&mm_jit.m_B0[0], mm_jit.m_B0.capacity);
                sw_prefetch_L2(&C1[0], C1.capacity);
            });
        }
        if (i == 8) {
            std::cout << "------- clear-cache & prefetch : only A/B/C is cleared, SW prefetch to L2 can recover them -------\n";
            //timer.clear_cache();
            clflush(&A[0], A.capacity);  // 512K data flushed + 30us
            clflush(&mm_jit.m_B0[0], mm_jit.m_B0.capacity); // 512K data flushed +28us 
            clflush(&C1[0], C1.capacity); //  101->91 = 10

            timer(1, [&](){
                sw_prefetch_L2(&A[0], A.capacity);
                sw_prefetch_L2(&mm_jit.m_B0[0], mm_jit.m_B0.capacity);
                sw_prefetch_L2(&C1[0], C1.capacity);
            });
        }
        timer(
            1,
            [&]() {
                TileConfigScope tcfg(mm_jit.tile_config());
                mm_jit(&A[0], A.stride, &C1[0], C1.stride);
            },
            M * N * K * 2);
    }

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
            M * N * K * 2 // OPS per call
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
            M * N * K * 2 // OPS per call
        );

    return 0;
}

int main(int argc, const char* argv[]) {
    srand(0);
    bool initAMX = initXTILE();

    // using big AMX matmul as cache cleaner to avoid AMX being open/close Port5
    Linear32x32_AMX clr_cache_amx(false);
    const int ABsize = 64 * 1024 * 1024;
    std::vector<ov::bfloat16> clr_cache_A(ABsize, 1.0f);
    std::vector<ov::bfloat16> clr_cache_B(ABsize, 2.0f);
    float clr_cache_C[32 * 32];
    /*
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

    std::cout << "===============================128x128 (==L2)========================\n";
    if (0) {
        amx_mm(128, 128, 2048);
        amx_dnnl(128, 128, 2048);
        amx_jit(128, 128, 2048);
    }
    std::cout << "===============================128x128 (==L2)========================\n";
    amx_jit_special(128, 128, 2048);
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