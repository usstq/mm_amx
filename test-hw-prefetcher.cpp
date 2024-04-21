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

    mfence();

    // delta tsc
    rdtsc(); // EDX:EAX
    sal(rdx, 32);
    or_(rax, rdx); // 64bit
    sub(rax, reg_tsc_0);

    ret();
  }
};

void test_read_prefetch() {
    MeasureAccess measure_access;

    MSRConfig _msr1(0x1A4, MASK_1A4_PF);

    uint64_t nbytes = 8192;
    auto* data = reinterpret_cast<uint8_t*>(aligned_alloc(4096, nbytes));
    for(int i = 0; i < nbytes; i++) data[i] = 1;

    auto data_access = [&]() {
        auto sum = _mm512_setzero_ps();
        for(int i = 6; i < 16; i++) sum += _mm512_loadu_ps(data + i*64);
        if(_mm512_reduce_add_ps(sum) == 1) {
            abort();
        }
    };

    auto wait_tsc = second2tsc(1e-3);
    auto wait = [&](){
        auto t0 = __rdtsc();
        while(__rdtsc() - t0 < wait_tsc);
    };

    // 8192 = 128 x 64 = 128 cacheline
    std::vector<int64_t> access_times(nbytes/64, 0);

    const int repeats = 1;

    for(int r = 0; r < repeats; r ++) {
        for(int cache_line = 0; cache_line < access_times.size(); cache_line++) {
            // flush the probing array
            // clflush(data, nbytes);
            {
                for (int i = 0; i < nbytes; i += 64) _mm_clflush(data + i);
                _mm_mfence();
            }

            wait();

            // access pattern triggers HW prefetch
            data_access();
            _mm_mfence();

            wait();

            // check which elements have been prefetched
            access_times[cache_line] += measure_access(data + cache_line*64);
        }
    }
    ::free(data);

    // show access_times
    for(int cache_line = 0; cache_line < access_times.size(); cache_line++) {
        
        auto nbars = static_cast<int>(tsc2second(access_times[cache_line]/repeats)*1e9 * 100/256);
        std::string progress_bar;
        for(int i = 0; i < nbars; i++) progress_bar += "█";
        printf(" cache_line[%3d] : %6.2f ns : \033[1;90m %s \033[0m\n", cache_line, tsc2second(access_times[cache_line]/repeats)*1e9, progress_bar.c_str());
    }
}

int main() {
    test_read_prefetch();
    return 0;
}