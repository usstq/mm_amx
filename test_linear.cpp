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
    int m_K;
    TileConfig m_tile_cfg;
    Linear32x32_AMX(int K, int M, int N) : m_K(K) {
        assert(M == 32);
        assert(N == 32);
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

#include "bf16.hpp"

class LinearNxN {
public:
    Linear32x32_AMX m_kernel;
    int m_K;
    int m_M;
    int m_N;
    int m_flag;
    LinearNxN(int K, int M, int N) : m_K(K), m_M(M), m_N(N), m_kernel(K, 32, 32) { m_flag = std::getenv("LIFLAGS") ? atoi(std::getenv("LIFLAGS")) : 0; }
    const TileConfig& tile_config() { return m_kernel.m_tile_cfg; }

    // Bt: [N, K]

    void call_kernel(int x, int y, ov::bfloat16* A0, int strideA, ov::bfloat16* B0, float* C0, int strideC) {
        m_kernel(reinterpret_cast<uint8_t*>(A0) + y * strideA, strideA, reinterpret_cast<uint8_t*>(B0) + ((x / 32) * (m_K / 32)) * 2048, reinterpret_cast<uint8_t*>(C0 + x) + y * strideC, strideC);
    }

    void call_64x64(int x, int y, ov::bfloat16* A0, int strideA, ov::bfloat16* B0, float* C0, int strideC) {
        call_kernel(x + 0, y + 0, A0, strideA, B0, C0, strideC);
        call_kernel(x + 32, y + 0, A0, strideA, B0, C0, strideC);
        call_kernel(x + 32, y + 32, A0, strideA, B0, C0, strideC);
        call_kernel(x + 0, y + 32, A0, strideA, B0, C0, strideC);
    }

    void call_128x128(int x0, int y0, ov::bfloat16* A0, int strideA, ov::bfloat16* B0, float* C0, int strideC) {
        // 520 Ops/cycle  
        call_64x64(x0+0, y0+0, A0, strideA, B0, C0, strideC);
        call_64x64(x0+64, y0+0, A0, strideA, B0, C0, strideC);
        call_64x64(x0+64, y0+64, A0, strideA, B0, C0, strideC);
        call_64x64(x0+0, y0+64, A0, strideA, B0, C0, strideC);
    }

    void call_256x256(int x0, int y0, ov::bfloat16* A0, int strideA, ov::bfloat16* B0, float* C0, int strideC) {
        // 520 Ops/cycle  
        call_128x128(x0+0, y0+0, A0, strideA, B0, C0, strideC);
        call_128x128(x0+128, y0+0, A0, strideA, B0, C0, strideC);
        call_128x128(x0+128, y0+128, A0, strideA, B0, C0, strideC);
        call_128x128(x0+0, y0+128, A0, strideA, B0, C0, strideC);
    }

    void call_general(int x0, int x1, int y0, int y1, ov::bfloat16* A0, int strideA, ov::bfloat16* B0, float* C0, int strideC) {
        // 128x128 : 560
        for (int y = y0; y < y1; y+=64) {
            for (int x = x0; x < x1; x += 32) {
                call_kernel(x, y, A0, strideA, B0, C0, strideC);
            }
            for (int x = x1 - 32; x >= x0; x -= 32) {
                call_kernel(x, y+32, A0, strideA, B0, C0, strideC);
            }
        }
    }

    // B0: repacked
    void operator()(ov::bfloat16* A0, int strideA, ov::bfloat16* B0, float* C0, int strideC) {
        if (m_M == 64 && m_N == 64)
            call_64x64(0, 0, A0, strideA, B0, C0, strideC);
        else if (m_M == 256 && m_N == 256)
            call_256x256(0, 0, A0, strideA, B0, C0, strideC);
        else
            call_general(0, m_N, 0, m_M, A0, strideA, B0, C0, strideC);
    }
};

// #include "kernels_avx512.hpp"
#include "kernels_amx.hpp"
#include "tensor2D.hpp"

#include "timeit.hpp"

#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#include <stdlib.h>

#include <omp.h>

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

    tensor2D<float> C0(M, N, true); // reference result
    tensor2D<float> C1(M, N, true); // actual result
    LinearAMX mm_jit(K, M, N);
    TileConfigScope tcfg(mm_jit.tile_config());

    auto BPacked = repack_weights(Bt);

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

    timer.tag(__func__, " (M=", M, ",N=", N, ",K=", K, ")", acc)
        .color(acc_color)(
            times, [&]() { mm32x32(A, Bt, 0, N, pp); },
            M * N * K * 2 // OPS per call
        );

    return 0;
}

int amx_dnnl(const int M, const int N, int K, int times = -1000) {
    tensor2D<ov::bfloat16> A(M, K, true); // ensure stride of A matrix is multiple of cache line, which is vital to performance.
    tensor2D<ov::bfloat16> B(K, N, true); // [IC, OC]
    tensor2D<float> C0(M, N, true); // reference result
    tensor2D<float> C1(M, N, true); // actual result
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

    timer.set_app(argv[0]);

    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    std::cout << ANSIcolor("31") << "omp_get_num_threads() = " << omp_get_num_threads() << std::endl << ANSIcolor();

#if 0
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
    std::cout << "===============================64x64 (L2)========================\n";
    for (int i = 0; i < 2; i++) {
        amx_mm(64, 64, 4096);
        amx_jit<LinearNxN>(64, 64, 4096);
    }
#endif
    std::cout << "===============================128x128 (==L2)========================\n";
    for (int i = 0; i < 2; i++) {
        amx_mm(128, 128, 4096);
        amx_dnnl(128, 128, 4096);
        amx_jit<LinearNxN>(128, 128, 4096);
    }
    std::cout << "===============================256x256 (>L2)========================\n";
    for (int i = 0; i < 2; i++) {
        amx_mm(256, 256, 4096);
        amx_dnnl(256, 256, 4096);
        amx_jit<LinearNxN>(256, 256, 4096);
    }
    std::cout << "===============================256x320 (>L2)========================\n";
    for (int i = 0; i < 2; i++) {
        amx_mm(256, 320, 4096);
        amx_dnnl(256, 320, 4096);
        amx_jit<LinearNxN>(256, 320, 4096);
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