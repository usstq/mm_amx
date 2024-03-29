#include <vector>
#include "jit.hpp"

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
                         {16, 64},  // C:0
                         {16, 64},  // C:1
                         {16, 64},  // C:2
                         {16, 64},  // C:3
                         {16, 64},  // A0:4
                         {16, 64},  // A1:5
                         {16, 64},  // B0:6
                         {16, 64},  // B1:7
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
    align(64, false);
    L(loop_over_ktiles);
    //for (int k = 0; k < Ktiles; k++) {
      tileloadd(tmmA0, ptr[reg_A_addr + reg_A_stride]);
      tileloadd(tmmB0, ptr[reg_B_addr + reg_B_stride]);
      for(int i = 0; i < 1024; i+=64) prefetcht0(ptr[reg_B_addr + 4096 + 2048 + i]);
      tdpbf16ps(tmmC00, tmmA0, tmmB0);

      tileloadd(tmmA1, ptr[reg_A1_addr + reg_A_stride]);
      tdpbf16ps(tmmC10, tmmA1, tmmB0);

      lea(reg_B_addr, ptr[reg_B_addr + 1024]);
      tileloadd(tmmB1, ptr[reg_B_addr + reg_B_stride]);
      for(int i = 0; i < 1024; i+=64) prefetcht0(ptr[reg_B_addr + 4096 + 2048 + i]);
      tdpbf16ps(tmmC01, tmmA0, tmmB1);
      tdpbf16ps(tmmC11, tmmA1, tmmB1);
    //}
    add(reg_A_addr, 64);
    add(reg_A1_addr, 64);
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


int amx_jit32x32(int K, int times = -1000) {
  const int M = 32;
  const int N = 32;
  tensor2D<ov::bfloat16> A(M, K,
                           true);  // ensure stride of A matrix is multiple of
                                   // cache line, which is vital to performance.
  tensor2D<ov::bfloat16> B(K, N, true);
  auto Bt = B.Tr();
  std::vector<ov::bfloat16> BPacked(K * N, 0);
  tensor2D<float> C0(M, N, true);  // reference result
  tensor2D<float> C1(M, N, true);  // actual result
  Linear32x32_AMX mm32x32(K);

  TileConfigScope tcfg(mm32x32.tile_config());

  for (int k = 0, i = 0; k < K; k += 32) {
    amx_kernel::functional::transpose_epi32_16x16(&BPacked[i * 16 * 32],
                                                  &Bt(0, k), Bt.stride);
    i++;
    amx_kernel::functional::transpose_epi32_16x16(&BPacked[i * 16 * 32],
                                                  &Bt(16, k), Bt.stride);
    i++;
  }

  C0 = 0;
  matmul(A, B, C0);
  
  std::string acc;

  mm32x32(&A[0], A.stride, &BPacked[0], &C1[0], C1.stride);
  if (C0 == C1) {
    acc = "[PASS]";
    //std::cout << ANSIcolor("1;32") << "amx Match!\n" << ANSIcolor();
  } else {
    std::cout << "============= A ================ " << std::endl;
    std::cout << A << std::endl;
    std::cout << "============= B ================ " << std::endl;
    std::cout << B << std::endl;
    logger() << C0 << std::endl;
    logger() << C1 << std::endl;
    acc = "[FAIL]";
    //std::cout << ANSIcolor("1;31") << "amx Mismatch!\n" << ANSIcolor();
  }

  timer.tag(__func__, "(M=", M, ",N=", N, ",K=", K, ")", acc)(times, [&]() {
    mm32x32(&A[0], A.stride, &BPacked[0], &C1[0], C1.stride);
  },
  M*N*K*2 // OPS per call
  );

  return 0;
}

int amx_mm32x32(int K, int times = -1000) {
  const int M = 32;
  const int N = 32;
  tensor2D<ov::bfloat16> A(M, K,
                           true);  // ensure stride of A matrix is multiple of
                                   // cache line, which is vital to performance.
  tensor2D<ov::bfloat16> B(K, N, true);
  auto Bt = B.Tr();
  std::vector<ov::bfloat16> BPacked(K * N, 0);
  tensor2D<float> C0(M, N, true);  // reference result
  tensor2D<float> C1(M, N, true);  // actual result
  amx_kernel::Matmul<ov::bfloat16, ov::bfloat16> mm32x32(true, true);
  amx_kernel::PP::BiasGeluStore<float, amx_kernel::PP::Steps::NONE> pp(C1);

  std::string acc;
  C0 = 0;
  matmul(A, B, C0);

  mm32x32(A, Bt, 0, N, pp);
  if (C0 == C1) {
    acc = "[PASS]";
    //ss_name << ANSIcolor("1;32").str << "[PASS]" << ANSIcolor();
  } else {
    std::cout << "============= A ================ " << std::endl;
    std::cout << A << std::endl;
    std::cout << "============= B ================ " << std::endl;
    std::cout << B << std::endl;
    logger() << C0 << std::endl;
    logger() << C1 << std::endl;
    //ss_name << ANSIcolor("1;31") << "[FAIL]" << ANSIcolor();
    acc = "[FAIL]";
  }

  timer.tag(__func__, "(M=", M, ",N=", N, ",K=", K, ")", acc)(times, [&]() {
    mm32x32(A, Bt, 0, N, pp);
  },
  M*N*K*2 // OPS per call
  );

  return 0;
}

int main(int argc, const char* argv[]) {
  srand(0);
  bool initAMX = initXTILE();

  timer.set_app(argv[0]);

  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  std::cout << ANSIcolor("31")
            << "omp_get_num_threads() = " << omp_get_num_threads() << std::endl
            << ANSIcolor();

  std::cout << "===============================BF16========================\n";
  for(int i = 0; i<10; i++) {
    amx_mm32x32(4096);
    amx_jit32x32(4096);
  }
}