#include <vector>
#include "jit.hpp"

#if !defined(XBYAK64_GCC)
#error NOT SUPPORTED
#endif

class Sample : public jit_generator {
  void operator=(const Sample&);

 public:
  Sample() { create_kernel(); }

  void generate() {
    preamble();
    // inLocalLabel(); // use local label for multiple instance
    mov(reg_n, reg_params);  // n
    xor_(reg_sum, reg_sum);  // sum
    test(reg_n, reg_n);
    jz(".exit");
    xor_(reg_i, reg_i);  // i
    L(".lp");
    add(reg_sum, reg_i);
    inc(reg_i);

    cmp(reg_i, reg_n);
    jbe(".lp");  // jmp to previous @@
    L(".exit");  // <B>

    // outLocalLabel(); // end of local label
    postamble();
  }

  Xbyak::Reg32 reg_n = r8d;
  Xbyak::Reg32 reg_sum = eax;
  Xbyak::Reg32 reg_i = r10d;
  Xbyak::Reg64 reg_params = abi_param1;
};

void test_jit0() {
  Sample s;
  for (int i = 0; i <= 10; i++) {
    std::cout << "0+1+...+" << i << "=" << s(i) << std::endl;
  }
}

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
  bool is_i8_mode = false;

  struct tileconfig_t {
    uint8_t palette_id;
    uint8_t startRow;
    uint8_t reserved[14];
    uint16_t cols[16];
    uint8_t rows[16];
    void reset(int palette,
               int _startRow,
               const std::vector<std::pair<int, int>>& _rows_columnsBytes) {
      palette_id = palette;
      startRow = _startRow;
      unsigned long i;
      for (i = 0; i < 14; i++) {
        reserved[i] = 0;
      }
      for (i = 0; i < _rows_columnsBytes.size(); i++) {
        rows[i] = _rows_columnsBytes[i].first;
        cols[i] = _rows_columnsBytes[i].second;
      }
      for (; i < 16; i++) {
        cols[i] = 0;
        rows[i] = 0;
      }
    }
  } m_tile_cfg;

  MatMulVec_AMX(int head_size, int block_size)
      : m_head_size(head_size), m_block_size(block_size) {
    create_kernel();
  }

  struct CallArgs {
    void* query;
    void* key;
    float* dst;
  };

  Xbyak::Reg64 reg_q_addr = r8;
  Xbyak::Reg64 reg_k_addr = r9;
  Xbyak::Reg64 reg_dst_addr = r10;
  Xbyak::Reg64 reg_i = r11;
  Xbyak::Reg64 reg_BlockSize = r12;
  
  Xbyak::Reg64 reg_params = abi_param1;

  Xbyak::Tmm tmmC = tmm0;
  Xbyak::Tmm tmmA = tmm1;
  Xbyak::Tmm tmmB0 = tmm2;
  Xbyak::Tmm tmmB1 = tmm3;
  Xbyak::Tmm tmmB2 = tmm4;
  Xbyak::Tmm tmmB3 = tmm5;
  Xbyak::Tmm tmmB4 = tmm6;
  Xbyak::Tmm tmmB5 = tmm7;

  void generate() {
    // generate tile config
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

    preamble();
    // inLocalLabel(); // use local label for multiple instance
#define GET_OFF(field) offsetof(CallArgs, field)
    mov(reg_q_addr, ptr[reg_params + GET_OFF(query)]);
    mov(reg_k_addr, ptr[reg_params + GET_OFF(key)]);
    mov(reg_dst_addr, ptr[reg_params + GET_OFF(dst)]);

    // load tile config
    ldtilecfg(ptr[&m_tile_cfg]);

    constexpr static int kStep = is_i8_mode ? 64 : 32;
    int hs_tail = m_head_size & (kStep - 1);
    assert(hs_tail == 0);
    /*
                                B(query)    head_size x 1

    A(key) matrix : block_size x head_size  C(dst) block_size x 1
    */
    // load query into B tiles
    for (int i = 0, offset = 0, tmm_idx = tmmB0.getIdx(); i < m_head_size; i += kStep, offset += 64, tmm_idx++) {
        tileloadd(Xbyak::Tmm(tmm_idx), ptr[reg_q_addr + offset]);
    }

    // unroll by m_block_size
    for (int m = 0; m < m_block_size; m+=16) {
        for(int i = 0; i < m_head_size/kStep; i++) {
            tileloadd(tmmA, ptr[reg_k_addr + i*64]);
            tdpbf16ps(tmmC, tmmA, Xbyak::Tmm(tmmB0.getIdx() + i));
        }
    }

    xor_(reg_i, reg_i);
    L(".lp");
    add(reg_i, 16);

    cmp(reg_i, reg_BlockSize);
    jbe(".lp");  // jmp to previous @@

    mov(reg_n, reg_params);  // n
    xor_(reg_sum, reg_sum);  // sum
    test(reg_n, reg_n);
    jz(".exit");
    xor_(reg_i, reg_i);  // i
    L(".lp");
    add(reg_sum, reg_i);
    inc(reg_i);

    cmp(reg_i, reg_n);
    jbe(".lp");  // jmp to previous @@
    L(".exit");  // <B>

    tilerelease();
    // outLocalLabel(); // end of local label
    postamble();
  }
};

int main() {
  return 0;
}