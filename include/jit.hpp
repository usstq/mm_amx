
#include "xbyak.h"

#include <cstdlib>
#include <fstream>
#include <vector>

#include "misc.hpp"

#ifdef XBYAK64
constexpr Xbyak::Operand::Code abi_save_gpr_regs[] = {
    Xbyak::Operand::RBP, Xbyak::Operand::RBX, Xbyak::Operand::R12,
    Xbyak::Operand::R13, Xbyak::Operand::R14, Xbyak::Operand::R15,
#ifdef _WIN32
    Xbyak::Operand::RDI, Xbyak::Operand::RSI,
#endif
};

constexpr Xbyak::Operand::Code abi_param_regs[] = {
#ifdef _WIN32
    Xbyak::Operand::RCX, Xbyak::Operand::RDX, Xbyak::Operand::R8,
    Xbyak::Operand::R9
#else
    Xbyak::Operand::RDI,
    Xbyak::Operand::RSI,
    Xbyak::Operand::RDX,
    Xbyak::Operand::RCX,
    Xbyak::Operand::R8,
    Xbyak::Operand::R9
#endif
};

constexpr Xbyak::Operand::Code abi_not_param_reg =
#ifdef _WIN32
    Xbyak::Operand::RDI;
#else
    Xbyak::Operand::RCX;
#endif

#define abi_param1 Xbyak::Reg64(abi_param_regs[0])
#define abi_param2 Xbyak::Reg64(abi_param_regs[1])
#define abi_param3 Xbyak::Reg64(abi_param_regs[2])
#define abi_param4 Xbyak::Reg64(abi_param_regs[3])
#define abi_param5 Xbyak::Reg64(abi_param_regs[4])
#define abi_param6 Xbyak::Reg64(abi_param_regs[5])
#define abi_not_param1 Xbyak::Reg64(abi_not_param_reg)
#endif

class jit_generator : public Xbyak::CodeGenerator {
 public:
  
  static std::string& jit_debug() {
    static EnvVar v("JIT_DEBUG");
    return v.v_str;
  }

  jit_generator() : Xbyak::CodeGenerator(Xbyak::DEFAULT_MAX_CODE_SIZE*4, (void*)0) {
  }

 protected:
  const size_t num_abi_save_gpr_regs =
      sizeof(abi_save_gpr_regs) / sizeof(abi_save_gpr_regs[0]);

  void preamble() {
    for (size_t i = 0; i < num_abi_save_gpr_regs; ++i) {
      push(Xbyak::Reg64(abi_save_gpr_regs[i]));
      // Stack magic: save rsp into rbp state to be able to unwind stack.
      if (i == 0)
        mov(rbp, rsp);
    }
  }
  void uni_vzeroupper() {
    // if (mayiuse(avx))
    vzeroupper();
  }
  void postamble() {
    for (size_t i = 0; i < num_abi_save_gpr_regs; ++i)
      pop(Xbyak::Reg64(abi_save_gpr_regs[num_abi_save_gpr_regs - 1 - i]));
    uni_vzeroupper();
    ret();
  }

  virtual void generate() {
    ret();
  }

  const Xbyak::uint8* jit_ker_ = nullptr;
  const char * ker_name = "?";
  virtual int create_kernel(const char* name = "?") {
    int err_code = Xbyak::GetError();
    if (err_code != Xbyak::ERR_NONE)
      return err_code;
    generate();
    ker_name = name;
#ifdef JIT_DEBUG    
    if (!jit_debug().empty()) {
        std::cout << "jit_generator generate() is done: " << name << std::endl;
        if (jit_debug() == name || jit_debug() == "*") {
            dump();
        }
    }
#endif
    jit_ker_ = getCode();
    return (jit_ker_) ? 0 : -1;
  }

 public:
  template <typename... kernel_args_t>
  int operator()(kernel_args_t... args) const {
    using jit_kernel_func_t = int (*)(const kernel_args_t... args);
    auto* fptr = (jit_kernel_func_t)jit_ker_;
#ifdef JIT_DEBUG
    if (!jit_debug().empty()) {
      if (jit_debug() == ker_name || jit_debug() == "*") {
        std::cout << "jit kernel " << ker_name << " @ 0x" << std::hex
                << reinterpret_cast<uintptr_t>(jit_ker_)
                << " is being called.\n";
        asm("int3");
      }
    }
#endif
    return (*fptr)(std::forward<kernel_args_t>(args)...);
  }

  void dump() {
    std::ofstream outfile;
    outfile.open("temp.bin", std::ios_base::binary);
    outfile.write(reinterpret_cast<const char*>(getCode()), getSize());
    outfile.close();
    system("objdump -D -b binary -mi386:x86-64 -M intel temp.bin");
  }

  std::vector<uint8_t> log_buffer;
  uint8_t* m_log_addr;
  Xbyak::Reg64 reg_scratch = r9;
  int log_tile_count = 0;

  void log_tile(Xbyak::Tmm tmm, Xbyak::Reg64 reg_stride) {
    auto offset = log_buffer.size();
    log_buffer.resize(offset + 1024, 0xFF);
    m_log_addr = log_buffer.data();
    log_tile_count++;
    // reload base
    mov(reg_scratch, reinterpret_cast<uintptr_t>(&m_log_addr));
    mov(reg_scratch, ptr[reg_scratch]);
    tilestored(ptr[reg_scratch + reg_stride + offset], tmm);
  }

  template <typename T>
  void show_log() {
    T* pdata = reinterpret_cast<T*>(m_log_addr);
    for (int log = 0; log < log_tile_count; log++) {
      std::cout << "========== log " << log << std::endl;
      for (int y = 0; y < 16; y++, pdata += 32) {
        std::cout << "[" << y << "]: ";
        for (int x = 0; x < 32; x++) {
          std::cout << pdata[x] << ",";
        }
        std::cout << "\n";
      }
    }
  }
};

struct TileConfig {
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
} __attribute__((__packed__));

class TileConfiger : public jit_generator {
 public:
  TileConfiger() { create_kernel(); }
  void generate() override {
    Xbyak::Label release;
    test(abi_param1, abi_param1);
    jz(release);
    ldtilecfg(ptr[abi_param1]);
    ret();
    L(release);
    tilerelease();
    ret();
  }
};

// https://stackoverflow.com/questions/23690416/c-template-singleton-static-pointer-initialization-in-header-file
template <typename T>
class Singleton {
 public:
  static T& get() {
    static T instance;
    return instance;
  }
};

class TileConfigScope {
 public:
  TileConfigScope(const TileConfig& cfg) {
    (Singleton<TileConfiger>::get())(&cfg);
  };
  ~TileConfigScope() { (Singleton<TileConfiger>::get())(nullptr); }
};
