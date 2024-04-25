/*

 https://abertschi.ch/blog/2022/prefetching/

*/


#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#include <stdlib.h>

#include <omp.h>
#include "misc.hpp"
#include "timeit.hpp"

#include "jit.hpp"

timeit timer({
    //{PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, "HW_CYCLES"},
    //{PERF_TYPE_RAW, 0x3c, "CPU_CLK_UNHALTED.THREAD"},
    //{PERF_TYPE_RAW, 0x81d0, "MEM_LOAD_RETIRED.ALL_LOADS"},
    //{PERF_TYPE_HW_CACHE, 0x10002, "LLC_load_misses"},
    
    //{PERF_TYPE_RAW, 0x10d1, "L2_MISS"},
    //{PERF_TYPE_RAW, 0x02d1, "L2_HIT"},  // https://github.com/intel/perfmon/blob/2dfe7d466d46e89899645c094f8a5a2b8ced74f4/SPR/events/sapphirerapids_core.json#L7397
    //{PERF_TYPE_RAW, 0x04d1, "L3_HIT"},

    //{PERF_TYPE_RAW, 0x01d1, "L1_HIT"}, {PERF_TYPE_RAW, 0x08d1, "L1_MISS"},

    //{PERF_TYPE_RAW, 0x01d1, "L1_HIT"}, {PERF_TYPE_RAW, 0x02d1, "L2_HIT"}, {PERF_TYPE_RAW, 0x40d1, "FB_HIT"}, 
    {PERF_TYPE_RAW, 0x01d1, "L1_HIT"}, {PERF_TYPE_RAW, 0x02d1, "L2_HIT"}, {PERF_TYPE_RAW, 0x04d1, "L3_HIT"}, {PERF_TYPE_RAW, 0x20d1, "L3_MISS"},

    //{PERF_TYPE_RAW, 0x81d0, "ALL_LOADS"},        // MEM_INST_RETIRED.ALL_LOADS

    //{PERF_TYPE_RAW, 0x08d1, "L1_MISS"},
    //{PERF_TYPE_RAW, 0x10d1, "L2_MISS"},
    //{PERF_TYPE_RAW, 0x20d1, "L3_MISS"},

    
    //{PERF_TYPE_HW_CACHE, 0x2, "LLC_loads"},
    //{PERF_TYPE_RAW, 0x02b1, "UOPS_EXECUTED.CORE"},
});


class MemAccessPattern : public jit_generator {
 public:
  std::vector<int> m_cache_lines;
  bool m_is_write;
  bool m_use_prefetch_nta;
  MemAccessPattern(const std::vector<int>& cache_lines, bool is_write, bool use_prefetch_nta) : m_cache_lines(cache_lines), m_is_write(is_write), m_use_prefetch_nta(use_prefetch_nta) {
    create_kernel("MemAccessPattern");
  }
  // to save push/pop: do not use `abi_save_gpr_regs`
  Xbyak::Reg64 reg_addr = abi_param1;
  void generate() {
    mfence();
    if (!m_is_write) {
        vpxorq(zmm0, zmm0, zmm0);
        for(auto& i : m_cache_lines) {
            if (m_use_prefetch_nta) {
                // SW prefetch also triggers HW prefetch
                //prefetchw(ptr[reg_addr + i*64]);
                //prefetcht2(ptr[reg_addr + i*64]); 
                prefetchnta(ptr[reg_addr + i*64]);
            } else {
                prefetchnta(ptr[reg_addr + i*64]);
                vmovups(zmm1, ptr[reg_addr + i*64]);
                vaddps(zmm0, zmm0, zmm1);
            }
        }
    } else {
        // write
        vpxorq(zmm0, zmm0, zmm0);
        for(auto& i : m_cache_lines) {
            if (m_use_prefetch_nta) {
                prefetchnta(ptr[reg_addr + i*64]);
            } else {
                vmovups(ptr[reg_addr + i*64], zmm0);
            }
        }
    }
    mfence();
    ret();
  }
};

class MeasureAccess : public jit_generator {
 public:
  
  MeasureAccess() {
    create_kernel("MeasureAccess");
  }
  // to save push/pop: do not use `abi_save_gpr_regs`
  Xbyak::Reg64 reg_addr = abi_param1;
  Xbyak::Reg64 reg_cycles = rax;
  Xbyak::Reg64 reg_tsc_0 = r9;
  Xbyak::Reg64 reg_dummy = r10;

  void generate() {

    mfence();

    // reg_tsc_0
    rdtsc(); // EDX:EAX
    sal(rdx, 32);
    or_(rax, rdx); // 64bit
    mov(reg_tsc_0, rax);

    mfence();

    // dummy access
    vmovups(zmm0, ptr[reg_addr]);

    prefetchwt1
    mfence();

    // delta tsc
    rdtsc(); // EDX:EAX
    sal(rdx, 32);
    or_(rax, rdx); // 64bit
    sub(rax, reg_tsc_0);

    ret();
  }
};


void spin_wait(double seconds){
    auto wait_tsc = second2tsc(seconds);
    auto t0 = __rdtsc();
    while(__rdtsc() - t0 < wait_tsc);
}



// when NTA prefetched data is evicted from L1, it was dropped siliently
// not into L2 or L3.
void test_prefetch_nta() {
    MeasureAccess measure_access;

    // 1MB
    uint64_t nbytes = 1024*128;
    auto* data = reinterpret_cast<uint8_t*>(aligned_alloc(4096, nbytes));
    for(int i = 0; i < nbytes; i++) data[i] = 1;

    // clear cache by memset & load
    std::vector<uint8_t> big_buffer(1024*1024*128, 0);
    for(int cache_line = 0; cache_line < nbytes/64; cache_line ++) {
        
        memset(&big_buffer[0], cache_line, big_buffer.size());
        load_prefetch_L2(&big_buffer[0], big_buffer.size());

        // prefetch nta
        for(int i = 0; i < nbytes; i+=64) {
            _mm_prefetch(data + i, _MM_HINT_NTA);
            //_mm_prefetch(data + i, _MM_HINT_T2);
        }

        spin_wait(0.001);

        // are data still in L2?
        auto access_time = measure_access(data + cache_line*64);

        // https://en.wikipedia.org/wiki/ANSI_escape_code#3-bit_and_4-bit
        const char * fg_color = "90";
        //if (std::find(read_pattern.begin(), read_pattern.end(), cache_line) != read_pattern.end()) {
        //    fg_color = "32";
        //}

        auto nbars = static_cast<int>(tsc2second(access_time)*1e9 * 100/256);
        std::string progress_bar;
        // https://github.com/Changaco/unicode-progress-bars/blob/master/generator.html
        for(int i = 0; i < nbars; i++) progress_bar += "▅";// "█";
        printf(" cache_line[%3d] : %6.2f ns : \033[1;%sm %s \033[0m\n", cache_line, tsc2second(access_time)*1e9, fg_color, progress_bar.c_str());
    }
}

int main() {
    test_prefetch_nta();
    return 0;
}