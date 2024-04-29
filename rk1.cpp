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

timeit timer({
    {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, "HW_CYCLES"},
    //{PERF_TYPE_RAW, 0x01d1, "L1_HIT"}, {PERF_TYPE_RAW, 0x02d1, "L2_HIT"},  {PERF_TYPE_RAW, 0x40d1, "FB_HIT"},  {PERF_TYPE_RAW, 0x10d1, "L2_MISS"}, // {PERF_TYPE_RAW, 0x04d1, "L3_HIT"}, {PERF_TYPE_RAW, 0x20d1, "L3_MISS"},
    //{PERF_TYPE_RAW, 0x01d1, "L1_HIT"}, {PERF_TYPE_RAW, 0x02d1, "L2_HIT"}, {PERF_TYPE_RAW, 0x40d1, "FB_HIT"}, {PERF_TYPE_RAW, 0x04d1, "L3_HIT"}, //{PERF_TYPE_RAW, 0x20d1, "L3_MISS"}, /

    // {PERF_TYPE_RAW, 0x81d0, "ALL_LOADS"},        // MEM_INST_RETIRED.ALL_LOADS
    //{PERF_TYPE_RAW, 0x3c, "CPU_CLK_UNHALTED.THREAD"},
    //{PERF_TYPE_RAW, 0x81d0, "MEM_LOAD_RETIRED.ALL_LOADS"},
    //{PERF_TYPE_HW_CACHE, 0x10002, "LLC_load_misses"},
    //{PERF_TYPE_HW_CACHE, 0x2, "LLC_loads"},
    //{PERF_TYPE_RAW, 0x02b1, "UOPS_EXECUTED.CORE"},
});

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
    void* prefetch_Aptr;
    // to save push/pop: do not use `abi_save_gpr_regs`
    Xbyak::Reg64 reg_A_addr = abi_param1;
    Xbyak::Reg64 reg_A_stride = abi_param2;
    Xbyak::Reg64 reg_B_addr = abi_param3;
    Xbyak::Reg64 reg_C_addr = abi_param4;
    Xbyak::Reg64 reg_C_stride = abi_param5;
    Xbyak::Reg64 reg_prefetch = abi_param6;
    Xbyak::Reg64 reg_prefetchA = r12;
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

        push(reg_prefetchA);

        if (m_do_accumulation) {
            auto reg_C1_addr = reg_A1_addr; // reuse reg_A1_addr
#if 1
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

        mov(reg_B_stride, reinterpret_cast<uintptr_t>(&prefetch_ptr));
        mov(reg_prefetch, ptr[reg_B_stride + 0]);

        mov(reg_B_stride, reinterpret_cast<uintptr_t>(&prefetch_Aptr));
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
#if 1
        prefetcht2(ptr[reg_prefetch + 0]);
        prefetcht2(ptr[reg_prefetch + 64]);
        prefetcht2(ptr[reg_prefetch + 64 * 2]);
        prefetcht2(ptr[reg_prefetch + 64 * 3]);
        lea(reg_prefetch, ptr[reg_prefetch + 64 * 4]);
#endif

        tdpbf16ps(tmmC00, tmmA0, tmmB0);

        tileloadd(tmmA1, ptr[reg_A1_addr + reg_A_stride]);
        tdpbf16ps(tmmC10, tmmA1, tmmB0);

        tileloadd(tmmB1, ptr[reg_B_addr + reg_B_stride]);

// prefetch [num_Ktiles X 256] bytes
#if 1
        prefetcht2(ptr[reg_prefetchA + 0]);
        prefetcht2(ptr[reg_prefetchA + 64]);
        prefetcht2(ptr[reg_prefetchA + 64 * 2]);
        prefetcht2(ptr[reg_prefetchA + 64 * 3]);
        lea(reg_prefetchA, ptr[reg_prefetchA + reg_A_stride]);
#endif
        tdpbf16ps(tmmC01, tmmA0, tmmB1);
        tdpbf16ps(tmmC11, tmmA1, tmmB1);
        //}
        lea(reg_A_addr, ptr[reg_A_addr + const_A_steps]);
        lea(reg_A1_addr, ptr[reg_A1_addr + const_A_steps]);
        lea(reg_B_addr, ptr[reg_B_addr + 1024]);
        dec(reg_ktiles);
        jnz(loop_over_ktiles, T_NEAR);

#if 1
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
        pop(reg_prefetchA);
        ret();
    }
};

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

class LinearReduceK {
public:
    Linear32x32_AMX m_kernel_0;
    Linear32x32_AMX m_kernel_1;
    tensor2D<ov::bfloat16> m_B;
    tensor2D<float> m_C; // tempory C buffer
    int m_Ktiles;
    int m_subKtiles = 8;
    int m_K;
    int m_M;
    int m_N;
    int m_subN;

    int m_nthr;
    int m_Nblock_size;

    uint8_t fake_buff[256];
    LinearReduceK(int K, int M, int N, ov::bfloat16* weight, int w_stride) : m_K(K), m_M(M), m_N(N), m_kernel_0(false), m_kernel_1(true) { setup(K, M, N, weight, w_stride); }

    LinearReduceK() : m_kernel_0(false), m_kernel_1(true) {}

    void setup(int K, int M, int N, ov::bfloat16* weight, int w_stride) {
        m_K = K;
        m_M = M;
        m_N = N;
        m_Ktiles = m_K / 32;
        m_kernel_0.m_ktiles = m_Ktiles;
        m_kernel_1.m_ktiles = m_Ktiles;
        m_kernel_0.prefetch_ptr = fake_buff;
        m_kernel_1.prefetch_ptr = fake_buff;
        assert((m_K % 32) == 0);
        /*
#pragma omp parallel
        {
            int ithr = omp_get_thread_num();
            if (0 == ithr)
                m_nthr = omp_get_num_threads();
        }
        */
        m_nthr = 1;
        // parallel along m_N dimension
        ASSERT((m_N % m_nthr) == 0);
        m_Nblock_size = (m_N / m_nthr); // N blocks per thread

        // each thread prcoess m_Nblock_size, with size of multiple of 32
        ASSERT((m_Nblock_size % 32) == 0);

        // printf(" M,K,N=%d,%d,%d m_Nblock_size = %d\n", m_M, m_K, m_N, m_Nblock_size);
        //  i'th thread execute range : start=(m_Nblock_size*i/m_nthr), end=(start + m_Nblock_size)

        set_weight(weight, w_stride);

        m_C.resize(m_M, m_N);
    }

    const TileConfig& tile_config() { return m_kernel_0.m_tile_cfg; }

    void call_block(int m0, int m1, int n0, int n1, uint8_t* A0, int strideA, uint8_t* B0, uint8_t* C0, int strideC0) {
        auto N = n1 - n0;
        auto kernel = [&](int kt) {
            // 256=8x32, reduce on 8Ktile
            //
            bool is_last_subk = false;
            auto curKtiles = (m_Ktiles - kt);
            if (curKtiles >= m_subKtiles) {
                curKtiles = m_subKtiles;
                is_last_subk = true;
            }

            auto curK = curKtiles * 32;
            auto* ptrA0 = reinterpret_cast<uint8_t*>(A0) + kt * 32 * sizeof(ov::bfloat16);
            auto* ptrC0 = reinterpret_cast<uint8_t*>(&m_C[0]);

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

            auto* prefetch_B = reinterpret_cast<uint8_t*>(B0) + (kt + m_subKtiles) * (sizeof(ov::bfloat16) * 32 * m_Nblock_size);

            // sizeof(ov::bfloat16) * (m_subKtiles * 32) * (n1-n0) / ((m1 - m0) * (n1 - n0) / 32 / 32) / m_subKtiles;
            auto prefetch_blkB_bytes = sizeof(ov::bfloat16) * (32) * (32 * 32) / (m1 - m0);

            // printf("======== prefetch_blkB_bytes=%zu\n", prefetch_blkB_bytes);
            for (int y = m0; y < m1; y += 32, ptrA0 += 32 * strideA) {
                auto* ptrB0 = reinterpret_cast<uint8_t*>(B0) + kt * (sizeof(ov::bfloat16) * 32 * m_Nblock_size);
                auto* prefetch_A = ptrA0 + 32 * strideA;
                if (y + 32 >= m1) {
                    // prefetch A from next K sub-block instead
                    prefetch_A = reinterpret_cast<uint8_t*>(A0) + (kt + m_subKtiles) * 32 * sizeof(ov::bfloat16);
                }
                for (int x = n0, flag = 0; x < n1; x += 32, flag++, ptrB0 += curKtiles * 2048, ptrC0 += 1024 * 4) {

                    // too many SW prefetch would also block CPU HW pipeline, so it must be mixed into kernel
                    // for(int i = 0; i < prefetch_blk_bytes; i += 64) _mm_prefetch(ptrA1 + i, _MM_HINT_T2);
                    if (kt == 0) {
                        m_kernel_0.prefetch_ptr = prefetch_B;
                        m_kernel_0.m_ktiles = curKtiles;
                        m_kernel_0.prefetch_Aptr = prefetch_A;
                        m_kernel_0(ptrA0, strideA, ptrB0, ptrC0, 64, curKtiles);
                    } else {
                        m_kernel_1.prefetch_ptr = prefetch_B;
                        m_kernel_1.m_ktiles = curKtiles;
                        m_kernel_1.prefetch_Aptr = prefetch_A;
                        m_kernel_1(ptrA0, strideA, ptrB0, ptrC0, 64, curKtiles);
                    }
                    if ((flag & 1) == 0) {
                        prefetch_A += 256;
                    } else {
                        prefetch_A += 8 * strideA - 256;
                    }
                    prefetch_B += prefetch_blkB_bytes;
#if 0
                    if (is_last_subk) {
                        // last K split, we can copy m_C out to C0 sefely
                        uint8_t* dst = (C0 + y * strideC0 + x * sizeof(float));
                        uint8_t* src0 = (ptrC0);
                        uint8_t* src1 = (ptrC0 + 1024);
                        for (int i = 0; i < 16; i+=2) {
                            auto ra0 = _mm512_loadu_ps(src0);
                            auto ra1 = _mm512_loadu_ps(src0 + 64); src0 += 64*2;
                            auto rb0 = _mm512_loadu_ps(src1);
                            auto rb1 = _mm512_loadu_ps(src1 + 64); src1 += 64*2;
                            _mm512_stream_ps(dst, ra0); _mm512_stream_ps(dst + 64, rb0); dst += strideC0;
                            _mm512_stream_ps(dst, ra1); _mm512_stream_ps(dst + 64, rb1); dst += strideC0;
                        }
                        src0 = (ptrC0 + 1024*2);
                        src1 = (ptrC0 + 1024*3);
                        for (int i = 0; i < 16; i+=2) {
                            auto ra0 = _mm512_loadu_ps(src0);
                            auto ra1 = _mm512_loadu_ps(src0 + 64); src0 += 64*2;
                            auto rb0 = _mm512_loadu_ps(src1);
                            auto rb1 = _mm512_loadu_ps(src1 + 64); src1 += 64*2;
                            _mm512_stream_ps(dst, ra0); _mm512_stream_ps(dst + 64, rb0); dst += strideC0;
                            _mm512_stream_ps(dst, ra1); _mm512_stream_ps(dst + 64, rb1); dst += strideC0;
                        }
                    }
#endif
                }
            }
        };
        for (int kt = 0; kt < m_Ktiles; kt += m_subKtiles) {
            kernel(kt);
#if 0
            std::cout << "\t kt=" << kt << std::endl;
            for(int xx = 0; xx < 10; xx++)
                timer(
                    1, [&]() { kernel(kt); }, 2.0 * 256 * 256 * 256);
#endif
        }
    }

    void operator()(ov::bfloat16* A0, int strideA, float* C0, int strideC) {
        {
            TileConfigScope tcfg(tile_config());
            int ithr = 0; // omp_get_thread_num();
            auto n0 = m_Nblock_size * ithr;
            auto n1 = n0 + m_Nblock_size;
            // printf("m_Nblock_size=%d, ithr,n0,n1=%d,%d,%d\n", m_Nblock_size, ithr, n0, n1);
            auto* ptrA = reinterpret_cast<uint8_t*>(A0);
            auto* ptrC = reinterpret_cast<uint8_t*>(C0);
            auto* ptrB = reinterpret_cast<uint8_t*>(&m_B[0]) + n0 * m_K * sizeof(ov::bfloat16);
            // for (int m = 0; m < m_M; m += 256, ptrA += 256*strideA, ptrC += 256*strideC) {
            //     for (int n = 0; n < m_Nblock_size; n += 256) {
            call_block(0, m_M, n0, n1, ptrA, strideA, ptrB, ptrC, strideC);
        }
    }

    void set_weight(ov::bfloat16* weight, int w_stride) {
        m_B = tensor2D<ov::bfloat16>(m_N * m_K, 1, true);

        int i = 0;

        for (int ithr = 0; ithr < m_nthr; ithr++) {
            auto n0 = m_Nblock_size * ithr;
            auto n1 = n0 + m_Nblock_size;
            for (int kt = 0; kt < m_Ktiles; kt += m_subKtiles) {
                auto curKtiles = (m_Ktiles - kt);
                if (curKtiles > m_subKtiles)
                    curKtiles = m_subKtiles;
                tensor2D<ov::bfloat16> Bt(m_N, curKtiles * 32, weight + kt * 32, w_stride);
                for (int n = n0; n < n1; n += 32) {
                    for (int k = 0; k < Bt.dims[1]; k += 32) {
                        amx_kernel::functional::transpose_epi32_16x16(&m_B[i * 16 * 32], &Bt(n, k), Bt.stride);
                        i++;
                        amx_kernel::functional::transpose_epi32_16x16(&m_B[i * 16 * 32], &Bt(n + 16, k), Bt.stride);
                        i++;
                    }
                }
            }
        }
    }
};

int amx_jit2(const int M, const int N, const int K, int times = 100, bool clear_cache = true) {
    tensor2D<ov::bfloat16> A(M, K,
                             true); // ensure stride of A matrix is multiple of
                                    // cache line, which is vital to performance.
    tensor2D<ov::bfloat16> B(K, N, true);

    tensor2D<ov::bfloat16> A2 = A.clone();
    auto Bt = B.Tr();

    tensor2D<float> C0(M, N, true); // reference result
    tensor2D<float> C1(M, N, true); // actual result

    int nthr = get_nthr();
    ASSERT((N % nthr) == 0);
    // ASSERT((N / nthr) == 256);

    std::vector<tensor2D<float>> partC(nthr);
    std::vector<LinearReduceK> mm_jits(nthr);
#pragma omp parallel
    {
        int ithr = omp_get_thread_num();
        mm_jits[ithr].setup(K, M, N / nthr, &Bt(N * ithr / nthr, 0), Bt.stride);
        partC[ithr] = tensor2D<float>(M, N / nthr, true);
    }

    C0 = 0;
    matmul(A, B, C0);

    std::string acc;
    std::string acc_color;

#pragma omp parallel
    {
        int ithr = omp_get_thread_num();
        mm_jits[ithr](&A[0], A.stride, &C1(0, N * ithr / nthr), C1.stride);
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

    if (clear_cache) {
        timer.hook_clear_cache = [&]() {
// std::cout << "clr_cache_amx\n";
#pragma omp parallel
            {
                int ithr = omp_get_thread_num();
                A.clflush();
                mm_jits[ithr].m_B.clflush();
                partC[ithr].clflush();
            }
        };
    }

    timer.tag(__func__, M, K, N, acc)
        .color(acc_color)(
            times,
            [&]() {
#pragma omp parallel
                {
                    int ithr = omp_get_thread_num();
                    mm_jits[ithr](&A[0], A.stride, &partC[ithr](0, 0), partC[ithr].stride);
                    // mm_jits[ithr](&A[0], A.stride, &C1(0, N * ithr / nthr), C1.stride);
                }
            },
            2.0 * M * N * K / nthr // OPS per call per core
        );

    if (clear_cache) {
        timer.hook_clear_cache = nullptr;
    }
    return 0;
}

int amx_jit_special_reduceK(const int M, const int N, const int K) {
    tensor2D<ov::bfloat16> A(M, K, true);
    tensor2D<ov::bfloat16> B(K, N, true);
    auto Bt = B.Tr();

    tensor2D<float> C0(M, N, true); // reference result
    tensor2D<float> C1(M, N, true); // actual result
    LinearReduceK mm_rk(K, M, N, &Bt[0], Bt.stride);

    timer._clflush = 0;
    timer.tag("ReduceK  (M=", M, ",N=", N, ",K=", K, ")");

    auto do_benchmarks_rk = [&]() {
        for (int i = 0; i < 3; i++) {
            timer(
                1,
                [&]() {
                    TileConfigScope tcfg(mm_rk.tile_config());
                    mm_rk(&A[0], A.stride, &C1[0], C1.stride);
                },
                2.0 * M * N * K);
        }
    };

    ECOUT("-------------");
    do_benchmarks_rk();

    ECOUT("-------------");
    do_benchmarks_rk();

    ECOUT("----- clflush B matrix for rk --------");
    mm_rk.m_B.clflush();
    do_benchmarks_rk();

    ECOUT("----- clflush A matrix for rk --------");
    A.clflush();
    do_benchmarks_rk();

    ECOUT("----- clflush A&B matrix for rk --------");
    A.clflush();
    mm_rk.m_B.clflush();
    do_benchmarks_rk();

    return 0;
}

int amx_dnnl(const int M, const int N, int K, int times = 100) {
    tensor2D<ov::bfloat16> A(M, K, true); // ensure stride of A matrix is multiple of cache line, which is vital to performance.
    tensor2D<ov::bfloat16> B(K, N, true); // [IC, OC]
    tensor2D<float> C0(M, N, true);       // reference result
    tensor2D<float> C1(M, N, true);       // actual result
    DNNLInnerProduct mmdnnl(M, N, K, &B[0], true);

    mmdnnl.set_A(&A[0]);
    mmdnnl.set_C(&C1[0]);

    int nthr = get_nthr();

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
    size_t w_size;
    void* wptr = mmdnnl.get_weight(&w_size);
    timer.hook_clear_cache = [&]() {
// std::cout << "clr_cache_amx\n";
#pragma omp parallel
        {
            int ithr = omp_get_thread_num();
            A.clflush();
            clflush(wptr, w_size);
            C1.clflush();
        }
    };

    timer.tag(__func__, M, K, N, acc)
        .color(acc_color)(
            times, [&]() { mmdnnl.run(); },
            2.0 * M * N * K / nthr // OPS per call
        );

    timer.hook_clear_cache = nullptr;
    return 0;
}

int main(int argc, const char* argv[]) {
    int nthr = get_nthr();

    MSRConfig _msr;
    srand(0);
    bool initAMX = initXTILE();
    timer.set_app(argv[0]);

    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    std::cout << ANSIcolor("31") << "nthr = " << nthr << std::endl << ANSIcolor();

    if (1) {
        /*
        if (nthr == 1) {
            for (int i = 0; i < 3; i++) {
                amx_jit2(256, 256, 11008, 100, false); // 4096);
                amx_jit2(320, 320, 11008, 100, false); // 4096);
                amx_jit2(384, 384, 11008, 100, false); // 4096);
                amx_jit2(448, 448, 11008, 100, false); // 4096);
                amx_jit2(512, 512, 11008, 100, false); // 4096);
            }
        }
        */
        std::cout << "==================== nthr = " << nthr << "==================================\n";
        for (int i = 0; i < 3; i++) {
            amx_dnnl(256, 256 * nthr, 11008); // 4096);
            amx_jit2(256, 256 * nthr, 11008); // 4096);
        }
        return 0;
        std::cout << "==================== nthr = " << nthr << "==================================\n";
        for (int i = 0; i < 3; i++) {
            amx_dnnl(320, 320 * nthr, 11008); // 4096);
            amx_jit2(320, 320 * nthr, 11008); // 4096);
        }
        std::cout << "==================== nthr = " << nthr << "==================================\n";
        for (int i = 0; i < 3; i++) {
            amx_dnnl(512, 512 * nthr, 11008); // 4096);
            amx_jit2(512, 512 * nthr, 11008); // 4096);
        }
        return 0;
    }

    // amx_jit_special_reduceK(256, 256, 4096); return 0;
    // amx_jit_special_reduceK(256, 256, 2560);
    // MSRConfig _msr0(0x1A4, MSR_BIT(0)|MSR_BIT(1)|MSR_BIT(2));
    // amx_jit_special_reduceK(256, 256, 256); return 0;
    amx_jit_special_reduceK(256, 256, 4096);
    return 0;
}

#if 0
template <typename LinearKernel = LinearReduceK>
int amx_jit(const int M, const int N, const int K, int times = 100) {
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
    { mm_jit(&A[0], A.stride, &C1[0], C1.stride); }

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
            times, [&]() { mm_jit(&A[0], A.stride, &C1[0], C1.stride); },
            2.0 * M * N * K // OPS per call
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
    // clflush(&A[0], A.capacity);
    clflush(&mm_jit.m_B0[0], mm_jit.m_B0.capacity);
    // clflush(&C1[0], C1.capacity);
    do_benchmarks();

    ECOUT("------- clflush A w/o any prefetching------- +", A.capacity);
    clflush(&A[0], A.capacity);
    // clflush(&mm_jit.m_B0[0], mm_jit.m_B0.capacity);
    // clflush(&C1[0], C1.capacity);
    do_benchmarks();

    ECOUT("------- clflush C w/o any prefetching------- +", C1.capacity);
    clflush(&C1[0], C1.capacity);
    do_benchmarks();

    ECOUT("------- clflush A/B w/o any prefetching------- +", A.capacity);
    clflush(&A[0], A.capacity);
    clflush(&mm_jit.m_B0[0], mm_jit.m_B0.capacity);
    // clflush(&C1[0], C1.capacity);
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
    // clflush(&C1[0], C1.capacity);                   //  101->91 = 10
    swpf_latency = timer(1, [&]() {
        sw_prefetch_L2(&A[0], A.capacity);                     // 37us to fetch / 26us overhead
        sw_prefetch_L2(&mm_jit.m_B0[0], mm_jit.m_B0.capacity); // 34us to fetch / 30us overhead
        // sw_prefetch_L2(&C1[0], C1.capacity);                   // 5us to fetch / 10us overhead
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
    swpf_latency = timer(1, [&]() { sw_prefetch_L2(&A[0], A.capacity); });
    do_benchmarks();

    ECOUT("------- only A is flushed, prefetch A row by row -------");
    clflush(&A[0], A.capacity);
    do_benchmarks();

    ECOUT("------- only A is flushed, prefetch A row by row + 32 rows prelog-------");
    clflush(&A[0], A.capacity);
    swpf_latency = timer(1, [&]() {
        // sw_prefetch_L2(&A[0], A.stride * (32));
        // sw_prefetch_L2(&mm_jit.m_B0[0], mm_jit.m_B0.capacity);
        load_prefetch_L2(&A[0], A.stride * (32));
    });
    do_benchmarks();

    return 0;
}
#endif