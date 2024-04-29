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


class TileLoad : public jit_generator {
public:
    TileConfig m_tile_cfg;
    bool m_is_A_blocked;
    TileLoad(bool is_A_blocked) : m_is_A_blocked(is_A_blocked) {
        create_kernel("TileLoad");
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
    Xbyak::Reg64 reg_A_step = abi_param3;
    Xbyak::Reg64 reg_B_addr = abi_param4;
    Xbyak::Reg64 reg_tiles = abi_param5;

    Xbyak::Reg64 reg_A1_addr = r11;
    Xbyak::Reg64 reg_B_stride = r10;

    Xbyak::Tmm tmmC00 = tmm0;
    Xbyak::Tmm tmmC10 = tmm1;
    Xbyak::Tmm tmmC01 = tmm2;
    Xbyak::Tmm tmmC11 = tmm3;
    Xbyak::Tmm tmmA0 = tmm4;
    Xbyak::Tmm tmmA1 = tmm5;
    Xbyak::Tmm tmmB0 = tmm6;
    Xbyak::Tmm tmmB1 = tmm7;

    void generate() {
        Xbyak::Label loop_over_ktiles;

        mov(reg_B_stride, 64);

        lea(reg_A1_addr, ptr[reg_A_addr + 8*reg_A_stride]);
        lea(reg_A1_addr, ptr[reg_A1_addr + 8*reg_A_stride]);

        align(64, false);
        L(loop_over_ktiles);
        // for (int k = 0; k < Ktiles; k++) {
        if (m_is_A_blocked) {
            tileloadd(tmmA0, ptr[reg_A_addr + reg_B_stride]);
            lea(reg_A_addr, ptr[reg_A_addr + 1024]);

            tileloadd(tmmA1, ptr[reg_A_addr + reg_B_stride]);
            lea(reg_A_addr, ptr[reg_A_addr + 1024]);
        } else {
            tileloadd(tmmA0, ptr[reg_A_addr + reg_A_stride]);
            lea(reg_A_addr, ptr[reg_A_addr + reg_A_step]);

            tileloadd(tmmA1, ptr[reg_A1_addr + reg_A_stride]);
            lea(reg_A1_addr, ptr[reg_A1_addr + reg_A_step]);
        }

        tileloadd(tmmB0, ptr[reg_B_addr + reg_B_stride]);
        lea(reg_B_addr, ptr[reg_B_addr + 1024]);

        tileloadd(tmmB1, ptr[reg_B_addr + reg_B_stride]);
        lea(reg_B_addr, ptr[reg_B_addr + 1024]);

        dec(reg_tiles);
        jnz(loop_over_ktiles, T_NEAR);

        ret();
    }
};

void test_load() {
    TileLoad tload0(false);
    TileLoad tload1(true);


    EnvVar NCLS("NCLS", 64);
    int num_tiles = ((int)NCLS)*64/sizeof(ov::bfloat16)/32;
    tensor2D<ov::bfloat16> A(32, num_tiles*32, true);
    tensor2D<ov::bfloat16> B(32, num_tiles*32, true);

    perf_log plog({
        {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, "HW_CYCLES"},
        {PERF_TYPE_RAW, 0x21a6, "BOUND_ON_LOADS"},
        {PERF_TYPE_RAW, 0x10d1, "L2_MISS"},
        {PERF_TYPE_RAW, 0x08d1, "L1_MISS"},
    });
    plog.tag("Stride", A.stride);
    plog.reserve(512);

    TileConfigScope tcfg(tload0.m_tile_cfg);
    for (int i=0; i < 5; i++) {
        plog([&]() {
            tload0(&A[0], A.stride, 64, &B[0], num_tiles);
        });
        plog([&]() {
            tload1(&A[0], 64, 1024, &B[0], num_tiles);
        });
    }
}

int main() {
    bool initAMX = initXTILE();
    test_load();
    return 0;
}