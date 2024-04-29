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
EnvVar NOSWPF("NOSWPF", 0);

class Linear32x32_AMXL2_swpfB : public jit_generator {
public:
    TileConfig m_tile_cfg;

    int64_t m_ktiles;

    int m_BM; // blockM: L2-cache kernel
    int m_BK; // blockK: L2-cache kernel
    int m_BN; // blockN: L2-cache kernel

    bool m_do_accumulation;
    bool m_is_C_blocked;
    int m_prefetch_Blines;

    // both A & B data will be prefetched from memory for next kernel invokation
    // and the prefetches are evenly distributed into each kernel.
    //
    // we first tackle the prefetching of B, because each time
    // we will call it with a new B, and run() will have to prefetch new B
    // for next round, so next B is also of size (KxN) elements
    //    distributes into (BM/32)*(BN/32) kernels:
    //    each kernel has (BK/32) iterations, thus each kernel iteration
    //    need to prefetch (BKxBN)/(BMxBNxBK/32768) = 32768/BM elements
    //    which is 1024/BM cache lines, this has to be determined at
    //    code-generation time. with BM=256, this is only 4.
    //
    Linear32x32_AMXL2_swpfB(int BM = 256, int BK = 256, int BN = 256, bool do_accumulation = false, bool is_C_blocked = false) : m_BM(BM), m_BK(BK), m_BN(BN), m_do_accumulation(do_accumulation), m_is_C_blocked(is_C_blocked) {

        // B: [K, N]
        m_ktiles = m_BK / 32;

        // prefetch block B requires
        m_prefetch_Blines = 32768 * sizeof(ov::bfloat16) / 64 / m_BM;

        if (NOSWPF) {
            m_prefetch_Blines = 0;
        }

        create_kernel("Linear32x32_AMXL2_swpfB");
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

        // prefetch next sub-block B matrix
        if (m_prefetch_Blines > 0) {
            for (int pi = 0; pi < m_prefetch_Blines; pi++) {
                prefetcht2(ptr[reg_prefetch + pi * 64]);
            }
            lea(reg_prefetch, ptr[reg_prefetch + 64 * m_prefetch_Blines]);
        }

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

    // run L2 cache blocking kernel with size:
    //    [BM, BK]*[BK, BN] => [BM, BN]
    //
    void run(uint8_t* pA, int strideA, // A
             uint8_t* pB, int strideB, // strideB is special
             uint8_t* pC, int strideC, // C
             uint8_t* prefetch_B       // prefetch B
    ) {
        auto prefetch_step = m_prefetch_Blines * 64 * m_ktiles;

        for (int m = 0; m < m_BM; m += 32, pA += 32 * strideA, pC += 32 * strideC) {
            auto* pB1 = pB;
            for (int n = 0; n < m_BN; n += 32, pB1 += strideB, prefetch_B += prefetch_step) {
                (*this)(pA, strideA, pB1, pC + n * sizeof(float), strideC, prefetch_B);
            }
        }
    }
};

void clr_cache() {
    thread_local std::vector<uint8_t> big_buffer(1024 * 1024 * 2, 0);
    memset(&big_buffer[0], 1, big_buffer.size());
    load_prefetch_L2(&big_buffer[0], big_buffer.size());
};

int fix_stride(int stride) {
    // according to [Tip6](https://www.intel.com/content/www/us/en/developer/articles/technical/a-simple-example-to-measure-the-performance-of-an-intel-mkl-function.html)
    int best_stride_cache_lines = (stride + 63) / 64;
    if ((best_stride_cache_lines % 1) == 0)
        best_stride_cache_lines++;
    return best_stride_cache_lines * 64;
}

EnvVar FLAGX("FLAGX",0);

void test_prefetch_B(int M, int K, int N, int num_blkB = 16) {

    int nthr = get_nthr();
    int best_pad_K = fix_stride(K * sizeof(ov::bfloat16)) / sizeof(ov::bfloat16);
    tensor2D<ov::bfloat16> A_padded(M, best_pad_K, true);

    tensor2D<ov::bfloat16> A(M, K, &A_padded[0], A_padded.stride);
    tensor2D<ov::bfloat16> B(K, N, true);
    tensor2D<float> C0(M, N, true); // reference result
    tensor2D<float> C1(M, N, true); // reference result

    Linear32x32_AMXL2_swpfB jit_amx(M, K, N, true);

    auto Bt = B.Tr();
    auto B1 = repack_weights(Bt);

    C0 = 0;
    C1 = 0;
    matmul(A, B, C0);

    auto strideB = (K / 32) * 2048;
    {
        TileConfigScope tcfg(jit_amx.m_tile_cfg);
        jit_amx.run(reinterpret_cast<uint8_t*>(&A[0]), A.stride,   //
                    reinterpret_cast<uint8_t*>(&B1[0]), strideB,   //
                    reinterpret_cast<uint8_t*>(&C1[0]), C1.stride, //
                    reinterpret_cast<uint8_t*>(&B1[0]));
    }

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
        {PERF_TYPE_RAW, 0x21a6, "BOUND_ON_LOADS"},
        {PERF_TYPE_RAW, 0x019c,"IDQ_UOPS_NOT_DELIVERED"},
        {PERF_TYPE_RAW, 0x10d1, "L2_MISS"},
        //{PERF_TYPE_RAW, 0x0480, "ICACHE_DATA.STALLS"},
        //{PERF_TYPE_RAW, 0x0483, "ICACHE_TAG.STALLS"},
    });
    plog.reserve(512);
    plog.tag("cache-COLD", M, K, N, acc);
    plog.color(acc_color);

#pragma omp parallel
    {
        tensor2D<ov::bfloat16> A2_padded(M, best_pad_K, true);
        tensor2D<ov::bfloat16> A2(M, K, &A2_padded[0], A2_padded.stride);

        Linear32x32_AMXL2_swpfB jit_amx(M, K, N, true);
        int ithr = omp_get_thread_num();

        tensor2D<float> C2(M, N, true); // reference result
        TileConfigScope tcfg(jit_amx.m_tile_cfg);
        // we will clone many B1 to test if prefetch works
        std::vector<tensor2D<ov::bfloat16>> B1s;
        for (int i = 0; i < num_blkB; i++) {
            B1s.emplace_back(B1.clone());
        }

#pragma omp barrier

        if (nthr == 1) {
            tensor2D<ov::bfloat16>& blockB = B1s[0];
            for(int i = 0; i < (int)(FLAGX); i++) {
                plog([&]() {
                            jit_amx.run(reinterpret_cast<uint8_t*>(&A2[0]), A2.stride,     //
                                        reinterpret_cast<uint8_t*>(&blockB[0]), strideB, //
                                        reinterpret_cast<uint8_t*>(&C2[0]), C2.stride,   //
                                        reinterpret_cast<uint8_t*>(&blockB[0]));
                        },
                        2.0 * M * N * K // OPS per call per core
                    );
            }
        }
        if (nthr == 1) {
            clr_cache();
            clr_cache();
            clr_cache();
            clr_cache();

            for (int r = 0; r < (int)(FLAGX); r++) {
                tensor2D<ov::bfloat16>& blockB = B1s[0];
                auto r1 = r + 1;
                tensor2D<ov::bfloat16>& blockB1 = B1s[r1 < B1s.size() ? r1 : r];
                plog(
                    [&]() {
                        jit_amx.run(reinterpret_cast<uint8_t*>(&A2[0]), A2.stride,     //
                                    reinterpret_cast<uint8_t*>(&blockB[0]), strideB, //
                                    reinterpret_cast<uint8_t*>(&C2[0]), C2.stride,   //
                                    reinterpret_cast<uint8_t*>(&blockB1[0]));
                    },
                    2.0 * M * N * K // OPS per call per core
                );
            }
        }

        clr_cache();
        clr_cache();
        clr_cache();
        clr_cache();

#pragma omp barrier

        for(int warmup = 0; warmup < 4; warmup++) {
            tensor2D<ov::bfloat16>& blockB = B1s[0];
            tensor2D<ov::bfloat16>& blockB1 = B1s[0];
            jit_amx.run(reinterpret_cast<uint8_t*>(&A2[0]), A2.stride,     //
                        reinterpret_cast<uint8_t*>(&blockB[0]), strideB, //
                        reinterpret_cast<uint8_t*>(&C2[0]), C2.stride,   //
                        reinterpret_cast<uint8_t*>(&blockB1[0]));

        }
#pragma omp barrier
        plog(
            [&]() {
                for (int r = 0; r < B1s.size(); r++) {
                    tensor2D<ov::bfloat16>& blockB = B1s[r];
                    auto r1 = r + 1;
                    tensor2D<ov::bfloat16>& blockB1 = B1s[r1 < B1s.size() ? r1 : r];
                    jit_amx.run(reinterpret_cast<uint8_t*>(&A2[0]), A2.stride,     //
                                reinterpret_cast<uint8_t*>(&blockB[0]), strideB, //
                                reinterpret_cast<uint8_t*>(&C2[0]), C2.stride,   //
                                reinterpret_cast<uint8_t*>(&blockB1[0]));
                }
            },
            (B1s.size()) * 2.0 * M * N * K // OPS per call per core
        );
    }
}

#if 0
class Linear32x32_AMXL2_swpfAB : public jit_generator {
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
    Linear32x32_AMXL2_swpfAB(int BM = 256, int BK = 256, int BN = 256, bool do_accumulation = false) : m_BM(BM), m_BK(BK), m_BN(BN), m_do_accumulation(do_accumulation) {

        // B: [K, N]
        m_ktiles = m_BK / 32;

        // prefetch block B requires
        m_prefetch_Blines = 32768 * sizeof(ov::bfloat16) / 64 / m_BM;

        if (NOSWPF) {
            m_prefetch_Blines = 0;
        }

        create_kernel("Linear32x32_AMXL2_swpfB");
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

        // prefetch next sub-block B matrix
        if (m_prefetch_Blines > 0) {
            for (int pi = 0; pi < m_prefetch_Blines; pi++) {
                prefetcht2(ptr[reg_prefetch + pi * 64]);
            }
            lea(reg_prefetch, ptr[reg_prefetch + 64 * m_prefetch_Blines]);
        }

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

    // run L2 cache blocking kernel with size:
    //    [BM, BK]*[BK, BN] => [BM, BN]
    //
    void run(uint8_t* pA, int strideA, // A
             uint8_t* pB, int strideB, // strideB is special
             uint8_t* pC, int strideC, // C
             uint8_t* prefetch_B       // prefetch B
    ) {
        auto prefetch_step = m_prefetch_Blines * 64 * m_ktiles;

        for (int m = 0; m < m_BM; m += 32, pA += 32 * strideA, pC += 32 * strideC) {
            auto* pB1 = pB;
            for (int n = 0; n < m_BN; n += 32, pB1 += strideB, prefetch_B += prefetch_step) {
                (*this)(pA, strideA, pB1, pC + n * sizeof(float), strideC, prefetch_B);
            }
        }
    }
};

void test_prefetch_AB(int M, int K, int N) {
    tensor2D<ov::bfloat16> A(M, K,
                             true); // ensure stride of A matrix is multiple of
                                    // cache line, which is vital to performance.
    tensor2D<ov::bfloat16> B(K, N, true);
    tensor2D<float> C0(M, N, true); // reference result
    tensor2D<float> C1(M, N, true); // reference result

    Linear32x32_AMXL2_swpfAB jit_amx(M, K, N, false);

    auto Bt = B.Tr();
    auto B1 = repack_weights(Bt);

    C0 = 0;
    matmul(A, B, C0);

    auto strideB = (K / 32) * 2048;
    {
        TileConfigScope tcfg(jit_amx.m_tile_cfg);
        jit_amx.run(reinterpret_cast<uint8_t*>(&A[0]), A.stride,   //
                    reinterpret_cast<uint8_t*>(&B1[0]), strideB,   //
                    reinterpret_cast<uint8_t*>(&C1[0]), C1.stride, //
                    reinterpret_cast<uint8_t*>(&B1[0]));
    }

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
        {PERF_TYPE_RAW, 0x21a6, "BOUND_ON_LOADS"},
    });
    plog.reserve(512);
    plog.tag("cache-COLD", M, K, N, acc);
    plog.color(acc_color);

#pragma omp parallel
    {
        int ithr = omp_get_thread_num();
        tensor2D<float> C2(M, N, true); // reference result
        TileConfigScope tcfg(jit_amx.m_tile_cfg);
        // we will clone many B1 to test if prefetch works
        std::vector<tensor2D<ov::bfloat16>> B1s;
        tensor2D<ov::bfloat16> A1(M, K*1280, true);
        for (int i = 0; i < 1280; i++) {
            B1s.emplace_back(B1.clone());
        }
#if 0
#pragma omp barrier
        {
            // plog.tag("cache-HOT", M, K, N, acc);
            // plog.color(acc_color);
            for (int r = 0; r < 10; r++) {
                tensor2D<ov::bfloat16>& blockB = B1s[0];
                tensor2D<ov::bfloat16>& blockB1 = B1s[0];
                plog(
                    [&]() {
                        jit_amx.run(reinterpret_cast<uint8_t*>(&A[0]), A.stride,     //
                                    reinterpret_cast<uint8_t*>(&blockB[0]), strideB, //
                                    reinterpret_cast<uint8_t*>(&C2[0]), C2.stride,   //
                                    reinterpret_cast<uint8_t*>(&blockB1[0]));
                    },
                    2.0 * M * N * K // OPS per call per core
                );
            };
        }
#endif
        clr_cache();
        // plog.tag("cache-COLD", M, K, N, acc);
        // plog.color(acc_color);

#pragma omp barrier
        plog(
            [&]() {
                for (int r = 0; r < B1s.size(); r++) {
                    tensor2D<ov::bfloat16>& blockB = B1s[r];
                    auto r1 = r + 1;
                    tensor2D<ov::bfloat16>& blockB1 = B1s[r1 < B1s.size() ? r1 : r];
                    jit_amx.run(reinterpret_cast<uint8_t*>(&A[0]), A.stride,     //
                                reinterpret_cast<uint8_t*>(&blockB[0]), strideB, //
                                reinterpret_cast<uint8_t*>(&C2[0]), C2.stride,   //
                                reinterpret_cast<uint8_t*>(&blockB1[0]));
                }
            },
            (B1s.size()) * 2.0 * M * N * K // OPS per call per core
        );
    }
}
#endif

int main() {
    MSRConfig _msr;
    bool initAMX = initXTILE();
    // running 56 cores
    test_prefetch_B(256, 256, 256);
    test_prefetch_B(256, 256, 256); // 570 1000
    //test_prefetch_B(256, 256, 384); // 579 1040
    // test_prefetch_B(256, 256, 512); // 535 1000
    return 0;
}
