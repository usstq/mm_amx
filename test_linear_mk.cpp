#include "jit.hpp"
#include <vector>

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
    int m_K;
    TileConfig m_tile_cfg;
    Linear32x32_AMX(int K) : m_K(K) {
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
    Xbyak::Reg64 reg_B_stride = r10;
    Xbyak::Reg64 reg_A1_addr = r11;
    Xbyak::Reg64 reg_ktiles = r9;

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
        lea(reg_A1_addr, ptr[reg_A_addr + reg_A_stride * 8]);
        lea(reg_A1_addr, ptr[reg_A1_addr + reg_A_stride * 8]);
        auto Ktiles = m_K / 32;
        assert(m_K % 32 == 0);
        mov(reg_B_stride, 64);
        tilezero(tmmC00);
        tilezero(tmmC01);
        tilezero(tmmC10);
        tilezero(tmmC11);
        mov(reg_ktiles, Ktiles);

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

class Linear16x96_AMX : public jit_generator {
public:
    int m_K;
    TileConfig m_tile_cfg;
    Linear16x96_AMX(int K) : m_K(K) {
        create_kernel("Linear16x96_AMX");
        m_tile_cfg.reset(1, 0,
                         {
                             {16, 64}, // C:0
                             {16, 64}, // C:1
                             {16, 64}, // C:2
                             {16, 64}, // C:3
                             {16, 64}, // C:4
                             {16, 64}, // C:5
                             {16, 64}, // A:6
                             {16, 64}, // B:7
                         });
    }

    const TileConfig& tile_config() { return m_tile_cfg; }

    // to save push/pop: do not use `abi_save_gpr_regs`
    Xbyak::Reg64 reg_A_addr = abi_param1;
    Xbyak::Reg64 reg_A_stride = abi_param2;
    Xbyak::Reg64 reg_B_addr = abi_param3;
    Xbyak::Reg64 reg_C_addr = abi_param4;
    Xbyak::Reg64 reg_C_stride = abi_param5;
    Xbyak::Reg64 reg_B_stride = r10;
    Xbyak::Reg64 reg_ktiles = r9;

    Xbyak::Tmm tmmC0 = tmm0;
    Xbyak::Tmm tmmC1 = tmm1;
    Xbyak::Tmm tmmC2 = tmm2;
    Xbyak::Tmm tmmC3 = tmm3;
    Xbyak::Tmm tmmC4 = tmm4;
    Xbyak::Tmm tmmC5 = tmm5;
    Xbyak::Tmm tmmA = tmm6;
    Xbyak::Tmm tmmB = tmm7;

    void generate() {
        /*
                       B: 1x2 tiles
        A : 2x1 tiles  C: 2x2 tiles
        */
        Xbyak::Label loop_over_ktiles;
        auto Ktiles = m_K / 32;
        assert(m_K % 32 == 0);
        mov(reg_B_stride, 64);
        tilezero(tmmC0);
        tilezero(tmmC1);
        tilezero(tmmC2);
        tilezero(tmmC3);
        tilezero(tmmC4);
        tilezero(tmmC5);
        mov(reg_ktiles, Ktiles);

        align(64, false);
        L(loop_over_ktiles);
        // for (int k = 0; k < Ktiles; k++) {
        tileloadd(tmmA, ptr[reg_A_addr + reg_A_stride]);

        // reuse tmmA
        for (int c = 0; c < 6; c++) {
            tileloadd(tmmB, ptr[reg_B_addr + reg_B_stride + c * 1024]);
            for (int i = 0; i < 1024; i += 64)
                prefetcht0(ptr[reg_B_addr + 4096 + i + c * 1024]);
            tdpbf16ps(Xbyak::Tmm(tmmC0.getIdx() + c), tmmA, tmmB);
        }
        //}
        lea(reg_A_addr, ptr[reg_A_addr + 64]);
        lea(reg_B_addr, ptr[reg_B_addr + 6 * 1024]);
        dec(reg_ktiles);
        jnz(loop_over_ktiles, T_NEAR);

        for (int c = 0; c < 6; c++)
            tilestored(ptr[reg_C_addr + reg_C_stride + 64 * c], Xbyak::Tmm(tmmC0.getIdx() + c));
        ret();
    }
};

class Linear16x64_AMX : public jit_generator {
public:
    int m_K;
    TileConfig m_tile_cfg;
    Linear16x64_AMX(int K) : m_K(K) {
        create_kernel("Linear16x64_AMX");
        m_tile_cfg.reset(1, 0,
                         {
                             {16, 64}, // C:0
                             {16, 64}, // C:1
                             {16, 64}, // C:2
                             {16, 64}, // C:3
                             {16, 64}, // C:4
                             {16, 64}, // C:5
                             {16, 64}, // A:6
                             {16, 64}, // B:7
                         });
    }

    const TileConfig& tile_config() { return m_tile_cfg; }

    // to save push/pop: do not use `abi_save_gpr_regs`
    Xbyak::Reg64 reg_A_addr = abi_param1;
    Xbyak::Reg64 reg_A_stride = abi_param2;
    Xbyak::Reg64 reg_B_addr = abi_param3;
    Xbyak::Reg64 reg_C_addr = abi_param4;
    Xbyak::Reg64 reg_C_stride = abi_param5;
    Xbyak::Reg64 reg_B_stride = r10;
    Xbyak::Reg64 reg_ktiles = r9;

    Xbyak::Tmm tmmC0 = tmm0;
    Xbyak::Tmm tmmC1 = tmm1;
    Xbyak::Tmm tmmC2 = tmm2;
    Xbyak::Tmm tmmC3 = tmm3;
    Xbyak::Tmm tmmA = tmm4;
    Xbyak::Tmm tmmB0 = tmm5;
    Xbyak::Tmm tmmB1 = tmm6;

    void generate() {
        /*
                       B: 1x2 tiles
        A : 2x1 tiles  C: 2x2 tiles
        */
        Xbyak::Label loop_over_ktiles;
        auto Ktiles = m_K / 32;
        assert(m_K % 32 == 0);
        mov(reg_B_stride, 64);
        tilezero(tmmC0);
        tilezero(tmmC1);
        tilezero(tmmC2);
        tilezero(tmmC3);
        mov(reg_ktiles, Ktiles);
        tileloadd(tmmB0, ptr[reg_B_addr + reg_B_stride]);
        lea(reg_B_addr, ptr[reg_B_addr + 1024]);

        align(64, false);
        L(loop_over_ktiles);
        // for (int k = 0; k < Ktiles; k++) {
        tileloadd(tmmA, ptr[reg_A_addr + reg_A_stride]);

        // reuse tmmA
        tileloadd(tmmB1, ptr[reg_B_addr + reg_B_stride]);
        for (int i = 0; i < 1024; i += 64)
            prefetcht0(ptr[reg_B_addr + 4096 + i]);
        tdpbf16ps(tmmC0, tmmA, tmmB0);
        lea(reg_B_addr, ptr[reg_B_addr + 1024]);

        tileloadd(tmmB0, ptr[reg_B_addr + reg_B_stride]);
        for (int i = 0; i < 1024; i += 64)
            prefetcht0(ptr[reg_B_addr + 4096 + i]);
        tdpbf16ps(tmmC1, tmmA, tmmB1);
        lea(reg_B_addr, ptr[reg_B_addr + 1024]);

        tileloadd(tmmB1, ptr[reg_B_addr + reg_B_stride]);
        for (int i = 0; i < 1024; i += 64)
            prefetcht0(ptr[reg_B_addr + 4096 + i]);
        tdpbf16ps(tmmC2, tmmA, tmmB0);
        lea(reg_B_addr, ptr[reg_B_addr + 1024]);

        tileloadd(tmmB0, ptr[reg_B_addr + reg_B_stride]);
        for (int i = 0; i < 1024; i += 64)
            prefetcht0(ptr[reg_B_addr + 4096 + i]);
        tdpbf16ps(tmmC3, tmmA, tmmB1);
        lea(reg_B_addr, ptr[reg_B_addr + 1024]);

        //}
        lea(reg_A_addr, ptr[reg_A_addr + 64]);
        dec(reg_ktiles);
        jnz(loop_over_ktiles, T_NEAR);

        for (int c = 0; c < 4; c++)
            tilestored(ptr[reg_C_addr + reg_C_stride + 64 * c], Xbyak::Tmm(tmmC0.getIdx() + c));
        ret();
    }
};

#include "kernels_amx.hpp"
// #include "kernels_avx512.hpp"
#include "tensor2D.hpp"
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

template <typename LinearAMX>
int amx_jit(const int M, const int N, const int K, int times = -1000) {
    tensor2D<ov::bfloat16> A(M, K,
                             true); // ensure stride of A matrix is multiple of
                                    // cache line, which is vital to performance.
    tensor2D<ov::bfloat16> B(K, N, true);
    auto Bt = B.Tr();
    tensor2D<ov::bfloat16> BPacked(K * N, 1, true);
    tensor2D<float> C0(M, N, true); // reference result
    tensor2D<float> C1(M, N, true); // actual result
    LinearAMX mm_jit(K);
    TileConfigScope tcfg(mm_jit.tile_config());

    for (int k = 0, i = 0; k < K; k += 32) {
        for (int n = 0; n < N; n += 16) {
            amx_kernel::functional::transpose_epi32_16x16(&BPacked[i * 16 * 32], &Bt(n, k), Bt.stride);
            i++;
        }
    }

    C0 = 0;
    matmul(A, B, C0);

    std::string acc;
    std::string acc_color;
    mm_jit(&A[0], A.stride, &BPacked[0], &C1[0], C1.stride);

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

    timer.tag(__func__, "(M=", M, ",N=", N, ",K=", K, ")", acc)
        .color(acc_color)(
            times, [&]() { mm_jit(&A[0], A.stride, &BPacked[0], &C1[0], C1.stride); },
            M * N * K * 2 // OPS per call
        );

    return 0;
}

int amx_mm(const int M, const int N, int K, int times = -1000) {
    tensor2D<ov::bfloat16> A(M, K,
                             true); // ensure stride of A matrix is multiple of
                                    // cache line, which is vital to performance.
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

    timer.tag(__func__, " (M=", M, ",N=", N, ",K=", K, ")", acc)
        .color(acc_color)(
            times, [&]() { mm32x32(A, Bt, 0, N, pp); },
            M * N * K * 2 // OPS per call
        );

    return 0;
}

class InstProfiler : public jit_generator {
public:
    InstProfiler() { create_kernel("InstProfiler"); }
    TileConfig m_tile_cfg;
    const TileConfig& tile_config() { return m_tile_cfg; }

    Xbyak::Reg64 reg_addrA = abi_param1;
    Xbyak::Reg64 reg_strideA = abi_param2;
    Xbyak::Reg64 reg_stepA = abi_param3;
    Xbyak::Reg64 reg_addrB = abi_param4;
    Xbyak::Reg64 reg_cnt = abi_param5;
    Xbyak::Reg64 reg_strideB = r10;

    void generate() {
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
        Xbyak::Label loop;
        mov(reg_strideB, 64);
        align(64, false);
        L(loop);
        tileloadd(tmm0, ptr[reg_addrA + reg_strideA]);
        lea(reg_addrA, ptr[reg_addrA + reg_stepA]);
        tileloadd(tmm1, ptr[reg_addrB + reg_strideB]);
        lea(reg_addrB, ptr[reg_addrB + 1024]);

        tileloadd(tmm2, ptr[reg_addrA + reg_strideA]);
        lea(reg_addrA, ptr[reg_addrA + reg_stepA]);
        tileloadd(tmm3, ptr[reg_addrB + reg_strideB]);
        lea(reg_addrB, ptr[reg_addrB + 1024]);

        tileloadd(tmm4, ptr[reg_addrA + reg_strideA]);
        lea(reg_addrA, ptr[reg_addrA + reg_stepA]);
        tileloadd(tmm5, ptr[reg_addrB + reg_strideB]);
        lea(reg_addrB, ptr[reg_addrB + 1024]);

        tileloadd(tmm6, ptr[reg_addrA + reg_strideA]);
        lea(reg_addrA, ptr[reg_addrA + reg_stepA]);
        tileloadd(tmm7, ptr[reg_addrB + reg_strideB]);
        lea(reg_addrB, ptr[reg_addrB + 1024]);

        dec(reg_cnt);
        jnz(loop);
        ret();
    }
};
void profile_tileload() {
    const int K = 32 * 8 * 20;
    tensor2D<ov::bfloat16> A(16, K, true);
    tensor2D<ov::bfloat16> B(K, 16, true);
    InstProfiler p;
    TileConfigScope tcfg(p.tile_config());

    auto count = K / (32 * 8);
    timer.tag(__func__, "A(K=", K, ")")(100, [&]() { p(&A[0], A.stride,  64, &B[0], count); });
    std::cout << "\t" << timer.perf_counters["HW_CYCLES"] / count / 8 << " cycles/tileLoad\n";
    timer.tag(__func__, "B(K=", K, ")")(100, [&]() { p(&A[0], 64, 1024, &B[0], count); });
    std::cout << "\t" << timer.perf_counters["HW_CYCLES"] / count / 8 << " cycles/tileLoad\n";
}

int main(int argc, const char* argv[]) {
    srand(0);
    bool initAMX = initXTILE();

    timer.set_app(argv[0]);

    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    std::cout << ANSIcolor("31") << "omp_get_num_threads() = " << omp_get_num_threads() << std::endl << ANSIcolor();

    std::cout << "===============================Strided load is slightly slower========================\n";
    profile_tileload();
    profile_tileload();
    profile_tileload();
    std::cout << "===============================BF16========================\n";
    amx_mm(32, 32, 128);
    amx_jit<Linear32x32_AMX>(32, 32, 128);
    amx_mm(32, 32, 128);
    amx_jit<Linear32x32_AMX>(32, 32, 128);

    std::cout << "===============================32x32 (L2)========================\n";
    for (int i = 0; i < 2; i++) {
        amx_mm(32, 32, 4096);
        amx_jit<Linear32x32_AMX>(32, 32, 4096);
    }
    std::cout << "===============================32x32 (LLC)========================\n";
    for (int i = 0; i < 2; i++) {
        amx_mm(32, 32, 4096 * 16);
        amx_jit<Linear32x32_AMX>(32, 32, 4096 * 16);
    }
    std::cout << "===============================16x96========================\n";
    for (int i = 0; i < 2; i++) {
        amx_mm(16, 96, 4096);
        amx_jit<Linear16x96_AMX>(16, 96, 4096);
    }
    std::cout << "===============================16x64========================\n";
    for (int i = 0; i < 2; i++) {
        amx_mm(16, 64, 4096);
        amx_jit<Linear16x64_AMX>(16, 64, 4096);
    }
}