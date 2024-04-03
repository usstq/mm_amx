#include "jit.hpp"
#include <vector>

#if !defined(XBYAK64_GCC)
#error NOT SUPPORTED
#endif

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


class InstProfiler : public jit_generator {
public:
    InstProfiler() { create_kernel("InstProfiler"); }
    TileConfig m_tile_cfg;
    const TileConfig& tile_config() { return m_tile_cfg; }

    Xbyak::Reg64 reg_addrA = abi_param1;
    Xbyak::Reg64 reg_strideA = abi_param2;
    Xbyak::Reg64 reg_stepA = abi_param3;
    Xbyak::Reg64 reg_cnt = abi_param4;

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

        static int do_prefetch = readenv("do_prefetch");
        static int do_move = readenv("do_move");

        // prefetcht0 also took time
        // tileloadd can also do prefetcht0's job
        // tileloadd & prefetcht0 are similar, but prefetcht0 is
        // little faster when data is already in cache.
        align(64, false);
        L(loop);
        for (int i = 0; i < 8; i++) {
            if (do_prefetch) {
                for (int i = 0; i < 1024; i += 64)
                    prefetcht0(ptr[reg_addrA + i]);
            } else {
                tileloadd(Xbyak::Tmm(0), ptr[reg_addrA + reg_strideA]);
            }
            
            if (do_move)
                lea(reg_addrA, ptr[reg_addrA + reg_stepA]);
        }

        dec(reg_cnt);
        jnz(loop);
        ret();
    }
};

void profile_tileload() {
    const int K = 32 * 8 * 2000;
    tensor2D<ov::bfloat16> A(16, K, true);
    InstProfiler p;
    TileConfigScope tcfg(p.tile_config());

    std::cout << "A matrix size: " << 16*K*2/1024.0/1024.0 << " MB\n";
    auto count = K / (32 * 8);
    timer.tag(__func__, "strided (K=", K, ")")(100, [&]() { p(&A[0], A.stride,  64, count); });
    std::cout << "\t" << timer.perf_counters["HW_CYCLES"] / count / 8 << " cycles/tileLoad\n";

    timer.tag(__func__, "compcat (K=", K, ")")(100, [&]() { p(&A[0], 64,  1024, count); });
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
}