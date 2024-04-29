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

//================================================================================================================

class Linear32x32_AMX : public jit_generator {
public:
    TileConfig m_tile_cfg;
    bool m_do_accumulation;

    tensor2D<ov::bfloat16> m_Weight;
    int64_t m_ktiles;

    int m_K;
    int m_N;

    Linear32x32_AMX(tensor2D<ov::bfloat16>& B, bool do_accumulation = false) : m_do_accumulation(do_accumulation) {

        // B: [K, N]
        m_K = B.dims[0];
        m_N = B.dims[1];

        ASSERT((m_K % 32) == 0);
        auto Bt = B.Tr();
        m_Weight = repack_weights(Bt);
        m_ktiles = m_K / 32;

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

    // to save push/pop: do not use `abi_save_gpr_regs`
    Xbyak::Reg64 reg_A_addr = abi_param1;
    Xbyak::Reg64 reg_A_stride = abi_param2;
    Xbyak::Reg64 reg_B_addr = abi_param3;
    Xbyak::Reg64 reg_C_addr = abi_param4;
    Xbyak::Reg64 reg_C_stride = abi_param5;

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

        tileloadd(tmmA1, ptr[reg_A1_addr + reg_A_stride]);
        tdpbf16ps(tmmC10, tmmA1, tmmB0);

        tileloadd(tmmB1, ptr[reg_B_addr + reg_B_stride]);

        // prefetch [num_Ktiles X 256] bytes

        tdpbf16ps(tmmC01, tmmA0, tmmB1);
        tdpbf16ps(tmmC11, tmmA1, tmmB1);
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
        ret();
    }

    void run(tensor2D<ov::bfloat16>& A, tensor2D<float>& C) {
        TileConfigScope tcfg(m_tile_cfg);
        // loop order
        int M = A.dims[0];
        int K = A.dims[1];
        // ASSERT((M%32) == 0); ASSERT(K == m_K); ASSERT(C.dims[0] == M); ASSERT(C.dims[1] == m_N);

        auto strideA = A.stride;
        auto strideC = C.stride;
        auto* pA = reinterpret_cast<uint8_t*>(&A[0]);
        auto* pC = reinterpret_cast<uint8_t*>(&C[0]);
        for (int m = 0; m < M; m += 32, pA += 32 * strideA, pC += 32 * strideC) {
            auto* pB = reinterpret_cast<uint8_t*>(&m_Weight[0]);
            for (int n = 0; n < m_N; n += 32, pB += m_ktiles * 2048) {
                (*this)(pA, strideA, pB, pC + n * sizeof(float), strideC);
            }
        }
    }
};

int fix_stride(int stride) {
    // according to [Tip6](https://www.intel.com/content/www/us/en/developer/articles/technical/a-simple-example-to-measure-the-performance-of-an-intel-mkl-function.html)
    int best_stride_cache_lines = (stride + 63) / 64;
    if ((best_stride_cache_lines % 1) == 0)
        best_stride_cache_lines++;
    return best_stride_cache_lines * 64;
}

void test_L2(int M, int K, int N, bool do_padK = true) {

    int best_padded_K = K;
    if (do_padK) {
        best_padded_K = fix_stride(K * sizeof(ov::bfloat16)) / sizeof(ov::bfloat16);
        std::cout << "[WARNING] K padded from " << K << " to " << best_padded_K << std::endl;
    }

    tensor2D<ov::bfloat16> A_padded(M, best_padded_K, true);
    tensor2D<ov::bfloat16> A(M, K, &A_padded[0], A_padded.stride);

    tensor2D<ov::bfloat16> B(K, N, true);
    tensor2D<float> C0(M, N, true); // reference result
    tensor2D<float> C1(M, N, true); // reference result
    Linear32x32_AMX jit_amx(B);

    C0 = 0;
    matmul(A, B, C0);
    jit_amx.run(A, C1);

    std::string acc;
    const char* acc_color = nullptr;
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

    perf_log plog({
        {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, "HW_CYCLES"},
        //{PERF_TYPE_RAW, 0x08d1, "L1_MISS"},
        //{PERF_TYPE_RAW, 0x04b2, "PORT_2_3_10"},
        //{PERF_TYPE_RAW, 0x01b2, "PORT_0"},
        //{PERF_TYPE_RAW, 0x02b2, "PORT_1"},
        //{PERF_TYPE_RAW, 0x10b2, "PORT_4_9"},
        //{PERF_TYPE_RAW, 0x20b2, "PORT_5_11"},
        //{PERF_TYPE_RAW, 0x40b2, "PORT_6"},
        //{PERF_TYPE_RAW, 0x80b2, "PORT_7_8"},
        {PERF_TYPE_RAW, 0x21a6, "BOUND_ON_LOADS"},
        {PERF_TYPE_RAW, 0x019c,"IDQ_UOPS_NOT_DELIVERED"},
        {PERF_TYPE_RAW, 0x10d1, "L2_MISS"},

    });

    plog.tag(__func__, M, K, N, acc, "padK", best_padded_K);
    plog.color(acc_color);

#pragma omp parallel
    {
        tensor2D<ov::bfloat16> A2_padded = A_padded.clone();
        tensor2D<ov::bfloat16> A2(M, K, &A2_padded[0], A2_padded.stride);
        tensor2D<float> C2(M, N, true); // reference result
        Linear32x32_AMX jit_amx2(B);
        jit_amx2.run(A2, C2);
        jit_amx2.run(A2, C2);

#pragma omp barrier
        plog(
            [&]() {
                for (int r = 0; r < 10; r++) {
                    jit_amx2.run(A2, C2);
                }
            },
            10 * 2.0 * M * N * K // OPS per call per core);
        );
    }
}

//================================================================================================================

class Linear32x32_AMX_blockedA : public jit_generator {
public:
    TileConfig m_tile_cfg;
    bool m_do_accumulation;

    tensor2D<ov::bfloat16> m_Weight;
    int64_t m_ktiles;

    int m_K;
    int m_N;

    Linear32x32_AMX_blockedA(tensor2D<ov::bfloat16>& B, bool do_accumulation = false) : m_do_accumulation(do_accumulation) {
        // B: [K, N]
        m_K = B.dims[0];
        m_N = B.dims[1];

        ASSERT((m_K % 32) == 0);
        auto Bt = B.Tr();
        m_Weight = repack_weights(Bt);
        m_ktiles = m_K / 32;

        create_kernel("Linear32x32_AMX_blockedA");
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

        mov(reg_B_stride, 64);
        mov(reg_A_stride, 64);

        auto const_A_steps = 64;

        align(64, false);
        L(loop_over_ktiles);
        // for (int k = 0; k < Ktiles; k++) {
        tileloadd(tmmA0, ptr[reg_A_addr + reg_A_stride]);
        lea(reg_A_addr, ptr[reg_A_addr + 1024]);
        tileloadd(tmmB0, ptr[reg_B_addr + reg_B_stride]);
        lea(reg_B_addr, ptr[reg_B_addr + 1024]);

        tdpbf16ps(tmmC00, tmmA0, tmmB0);

        tileloadd(tmmA1, ptr[reg_A_addr + reg_A_stride]);
        tdpbf16ps(tmmC10, tmmA1, tmmB0);

        tileloadd(tmmB1, ptr[reg_B_addr + reg_B_stride]);

        // prefetch [num_Ktiles X 256] bytes

        tdpbf16ps(tmmC01, tmmA0, tmmB1);
        tdpbf16ps(tmmC11, tmmA1, tmmB1);
        //}
        lea(reg_A_addr, ptr[reg_A_addr + 1024]);
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
        ret();
    }

    void prepareA(tensor2D<ov::bfloat16>& A, tensor2D<ov::bfloat16>& A2) {
        // make A blocked in
        int M = A.dims[0];
        int K = A.dims[1];
        auto strideA = A.stride;
        // ASSERT((M % 32) == 0); ASSERT(K == m_K);
        A2.resize(M, K, true);
        auto* src = reinterpret_cast<uint8_t*>(&A[0]);
        auto* dst = reinterpret_cast<uint8_t*>(&A2[0]);

        auto make_tile = [](uint8_t* src, int stride, uint8_t* dst) {
            for (int m = 0; m < 16; m++, src += stride, dst += 64) {
                memcpy(dst, src, 64);
            }
        };

        for (int m = 0; m < M; m += 32, src += 32 * strideA) {
            for (int k = 0; k < K; k += 32) {
                // 16x32 tile1 tile3
                // 16x32 tile2 tile4
                make_tile(src + k * sizeof(ov::bfloat16), strideA, dst);
                dst += 1024;
                make_tile(src + k * sizeof(ov::bfloat16) + 16 * strideA, strideA, dst);
                dst += 1024;
            }
        }
    }

    void run(tensor2D<ov::bfloat16>& A, tensor2D<float>& C) {
        TileConfigScope tcfg(m_tile_cfg);
        // loop order
        int M = A.dims[0];
        int K = A.dims[1];
        // ASSERT((M%32) == 0); ASSERT(K == m_K); ASSERT(C.dims[0] == M); ASSERT(C.dims[1] == m_N);

        auto strideA = A.stride;
        auto strideC = C.stride;
        auto* pA = reinterpret_cast<uint8_t*>(&A[0]);
        auto* pC = reinterpret_cast<uint8_t*>(&C[0]);
        for (int m = 0; m < M; m += 32, pA += m_ktiles * 2048, pC += 32 * strideC) {
            auto* pB = reinterpret_cast<uint8_t*>(&m_Weight[0]);
            for (int n = 0; n < m_N; n += 32, pB += m_ktiles * 2048) {
                (*this)(pA, strideA, pB, pC + n * sizeof(float), strideC);
            }
        }
    }
};

void test_L2_blocked(int M, int K, int N) {
    tensor2D<ov::bfloat16> A(M, K,
                             true); // ensure stride of A matrix is multiple of
                                    // cache line, which is vital to performance.
    tensor2D<ov::bfloat16> B(K, N, true);
    tensor2D<float> C0(M, N, true); // reference result
    tensor2D<float> C1(M, N, true); // reference result

    Linear32x32_AMX_blockedA jit_max(B);
    tensor2D<ov::bfloat16> A1;

    {
        perf_log plog;
        plog.tag("prepareA", M, K, N);
        for (int r = 0; r < 10; r++)
            plog([&]() { jit_max.prepareA(A, A1); });
    }

    C0 = 0;
    matmul(A, B, C0);
    jit_max.run(A1, C1);

    std::string acc;
    const char* acc_color = nullptr;
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

    perf_log plog({
        {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, "HW_CYCLES"},
        //{PERF_TYPE_RAW, 0x08d1, "L1_MISS"},
        //{PERF_TYPE_RAW, 0x04b2, "PORT_2_3_10"},
        //{PERF_TYPE_RAW, 0x01b2, "PORT_0"},
        //{PERF_TYPE_RAW, 0x02b2, "PORT_1"},
        //{PERF_TYPE_RAW, 0x10b2, "PORT_4_9"},
        //{PERF_TYPE_RAW, 0x20b2, "PORT_5_11"},
        //{PERF_TYPE_RAW, 0x40b2, "PORT_6"},
        //{PERF_TYPE_RAW, 0x80b2, "PORT_7_8"},
        {PERF_TYPE_RAW, 0x21a6, "BOUND_ON_LOADS"},
        {PERF_TYPE_RAW, 0x019c,"IDQ_UOPS_NOT_DELIVERED"},
        {PERF_TYPE_RAW, 0x10d1, "L2_MISS"},

    });
    plog.tag(__func__, M, K, N, acc);
    plog.color(acc_color);
#pragma omp parallel
    {
        tensor2D<ov::bfloat16> A2 = A1.clone();
        tensor2D<float> C2(M, N, true); // reference result
        Linear32x32_AMX_blockedA jit_amx2(B);
        jit_amx2.run(A2, C2);
        jit_amx2.run(A2, C2);

#pragma omp barrier
        plog(
            [&]() {
                for (int r = 0; r < 10; r++) {
                    jit_amx2.run(A2, C2);
                }
            },
            10 * 2.0 * M * N * K // OPS per call per core);
        );
    }
}

int main() {
    MSRConfig _msr;
    bool initAMX = initXTILE();

    test_L2(256, 256, 256, false);
    test_L2(256, 256, 256, true);
    test_L2_blocked(256, 256, 256);
    return 0;

    printf(":::::::::: AMX Usage on different M_K_N config ::::::::::\n");
    test_L2(128, 256, 128); // GOps/sec 1192.21  Ops/cycle 685
    test_L2(128, 256, 256); // GOps/sec 1218.87  Ops/cycle 710 <======== best
    test_L2(128, 256, 512); // GOps/sec 1286.36  Ops/cycle 711 <======== best

    test_L2(256, 256, 128); // GOps/sec 1200.93  Ops/cycle 697
    test_L2(256, 256, 256); // GOps/sec 1299.47  Ops/cycle 719 <======== best
    test_L2(256, 256, 512); // GOps/sec 1262.48  Ops/cycle 702

    test_L2(512, 256, 256); // GOps/sec 1293.38  Ops/cycle 713 <======== best
    test_L2(512, 256, 512); // GOps/sec 1136.08  Ops/cycle 628
    return 0;
}
