#include <vector>
#include "jit.hpp"

#if !defined(XBYAK64_GCC)
#error NOT SUPPORTED
#endif

/*
w = query * Key

query: [1,      head_size]
Key  : [block_size, head_size]
w    : [1, block_size]

head_size is known at compile time
*/
class MatMulVec_AMX : public jit_generator {
  void operator=(const MatMulVec_AMX&);

 public:
  int m_head_size;
  int m_block_size;
  TileConfig m_tile_cfg;
  MatMulVec_AMX(int head_size, int block_size)
      : m_head_size(head_size), m_block_size(block_size) {
    create_kernel("MatMulVec_AMX");
    m_tile_cfg.reset(1, 0,
                     {
                         {16, 4},   // C:0   M x 1     (4b)
                         {16, 64},  // A:1   M x 32/64 (64b)
                         {16, 4},   // B:2   32/64 x 1 (4b)
                         {16, 4},   // B:3
                         {16, 4},   // B:4
                         {16, 4},   // B:5
                         {16, 4},   // B:6
                         {16, 4},   // B:7
                     });
  }

  const TileConfig& tile_config() { return m_tile_cfg; }

  // to save push/pop: do not use `abi_save_gpr_regs`
  Xbyak::Reg64 reg_q_addr = abi_param1;
  Xbyak::Reg64 reg_k_addr = abi_param2;
  Xbyak::Reg64 reg_dst_addr = abi_param3;
  Xbyak::Reg64 reg_stride_A = r8;
  Xbyak::Reg64 reg_stride_BC = r9;

  Xbyak::Tmm tmmC = tmm0;
  Xbyak::Tmm tmmA = tmm1;
  Xbyak::Tmm tmmB0 = tmm2;
  Xbyak::Tmm tmmB1 = tmm3;
  Xbyak::Tmm tmmB2 = tmm4;
  Xbyak::Tmm tmmB3 = tmm5;
  Xbyak::Tmm tmmB4 = tmm6;
  Xbyak::Tmm tmmB5 = tmm7;

  void generate() {
    mov(reg_stride_A, m_head_size * 2);
    mov(reg_stride_BC, 4);
    const int kStep = 32;
    if ((m_head_size % 32) != 0)
      throw std::runtime_error("head size is not multiple of 32");
    if ((m_block_size % 16) != 0)
      throw std::runtime_error("block size is not multiple of 16");
    auto num_B_tiles = m_head_size / kStep;
    if (num_B_tiles > 6)
      throw std::runtime_error("number of B tiles is bigger than 6");

    /*
                                B(query)    head_size x 1
    A(key) matrix : block_size x head_size  C(dst) block_size x 1
    */
    // load query into B tiles
    for (int i = 0; i < num_B_tiles; i++) {
      tileloadd(Xbyak::Tmm(tmmB0.getIdx() + i),
                ptr[reg_q_addr + reg_stride_BC + i * 64]);
    }

    for (int m = 0; m < m_block_size; m += 16) {
      tilezero(tmmC);
      for (int i = 0; i < num_B_tiles; i++) {
        tileloadd(tmmA, ptr[reg_k_addr + reg_stride_A + i * 64]);
        tdpbf16ps(tmmC, tmmA, Xbyak::Tmm(tmmB0.getIdx() + i));
      }
      tilestored(ptr[reg_dst_addr + reg_stride_BC + m * sizeof(float)], tmmC);
      // add(reg_dst_addr, 16*sizeof(float));
      add(reg_k_addr, m_head_size * 2 * 16);
    }
    ret();
  }
};

// #include "kernels_amx.hpp"
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
template <typename T>
int amx_unit_test_gemAvB(int M, int K, int times = -1000) {
  int N = 1;
  tensor2D<T> A(M, K, true);  // ensure stride of A matrix is multiple of cache
                              // line, which is vital to performance.
  tensor2D<T> B(K, 1, true);
  tensor2D<float> C0(M, 1, true);  // reference result
  tensor2D<float> C1(M, 1, true);  // actual result
  MatMulVec_AMX matxvec(K, M);
  TileConfigScope tilecfg(matxvec.tile_config());

  // same B, different layout
  std::cout << __func__ << "(" << M << "," << K << ")\n";

  C0 = 0;
  matmul(A, B, C0);

  matxvec(&B[0], &A[0], &C1[0]);
  if (C0 == C1) {
    std::cout << ANSIcolor("1;32") << "amx Match!\n" << ANSIcolor();
  } else {
    std::cout << "============= A ================ " << std::endl;
    std::cout << A << std::endl;
    std::cout << "============= B ================ " << std::endl;
    std::cout << B << std::endl;
    logger() << C0 << std::endl;
    logger() << C1 << std::endl;
    std::cout << ANSIcolor("1;31") << "amx Mismatch!\n" << ANSIcolor();
  }

  timer.tag(__func__, M, K, N, "q*K_AMX")(
      times, [&]() { matxvec(&B[0], &A[0], &C1[0]); });

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
  amx_unit_test_gemAvB<ov::bfloat16>(256, 128);
  amx_unit_test_gemAvB<ov::bfloat16>(256, 128);
}