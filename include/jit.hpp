
#include "/home/tingqian/openvino/src/plugins/intel_cpu/thirdparty/onednn/src/cpu/x64/xbyak/xbyak.h"

#ifdef XBYAK64
constexpr Xbyak::Operand::Code abi_save_gpr_regs[] = {
        Xbyak::Operand::RBP,
        Xbyak::Operand::RBX,
        Xbyak::Operand::R12,
        Xbyak::Operand::R13,
        Xbyak::Operand::R14,
        Xbyak::Operand::R15,
#ifdef _WIN32
        Xbyak::Operand::RDI,
        Xbyak::Operand::RSI,
#endif
};

constexpr Xbyak::Operand::Code abi_param_regs[] = {
#ifdef _WIN32
        Xbyak::Operand::RCX, Xbyak::Operand::RDX, Xbyak::Operand::R8,
        Xbyak::Operand::R9
#else
        Xbyak::Operand::RDI, Xbyak::Operand::RSI, Xbyak::Operand::RDX,
        Xbyak::Operand::RCX, Xbyak::Operand::R8, Xbyak::Operand::R9
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
	jit_generator() : Xbyak::CodeGenerator(Xbyak::DEFAULT_MAX_CODE_SIZE, 0) {
    }

protected:
    const size_t num_abi_save_gpr_regs
        = sizeof(abi_save_gpr_regs) / sizeof(abi_save_gpr_regs[0]);

    void preamble() {
        for (size_t i = 0; i < num_abi_save_gpr_regs; ++i) {
            push(Xbyak::Reg64(abi_save_gpr_regs[i]));
            // Stack magic: save rsp into rbp state to be able to unwind stack.
            if (i == 0) mov(rbp, rsp);
        }
    }
    void uni_vzeroupper() {
        //if (mayiuse(avx))
        vzeroupper();
    }
    void postamble() {
        for (size_t i = 0; i < num_abi_save_gpr_regs; ++i)
            pop(Xbyak::Reg64(abi_save_gpr_regs[num_abi_save_gpr_regs - 1 - i]));
        uni_vzeroupper();
        ret();
    }

    virtual void generate() = 0;

    const Xbyak::uint8 *jit_ker_ = nullptr;
    virtual int create_kernel() {
        int err_code = Xbyak::GetError();
        if (err_code != Xbyak::ERR_NONE) return err_code;
        generate();
        jit_ker_ = getCode();
        return (jit_ker_) ? 0 : -1;
    }

public:
    template <typename... kernel_args_t>
    int operator()(kernel_args_t... args) const {
        using jit_kernel_func_t = int (*)(const kernel_args_t... args);
        auto *fptr = (jit_kernel_func_t)jit_ker_;
        return (*fptr)(std::forward<kernel_args_t>(args)...);
    }
};

