#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <vector>
#include <chrono>
#include <iostream>
#include <fstream>
#include <cassert>
#include <thread>
#include <map>
#include <limits>
#include <functional>

//#include "thread_pool.hpp"
#include "bf16.hpp"


// g++-11 ./test_conv.cpp -O2 -lpthread -march=native && ./a.out

// to use VNNI, we need higher version of compiler:
//    clang-9 ./test_conv.cpp -O2 -lpthread -march=native -lstdc++ && ./a.out

// to use AMX, we need intel compiler 
//   source  ~/intel/oneapi/setvars.sh
//   icx ./mm_amx_bf16.cpp -O2 -lpthread -march=native -lstdc++

// objdump -C -S ./a.out > a.asm

#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#include <stdlib.h>

//=============================================================
// BF16-amx Peak (Gops)
// c += a*b is counted as 2 Ops
// 
const double AMXBf16OpsPerTDP = (16*16*32)*2;
const double AMXBf16TDPThrouput = 16;
const double AMXBf16OpsPerCycleCore = AMXBf16OpsPerTDP/AMXBf16TDPThrouput;
const double AMXBf16FreqGHz = 2.05;
const double AMXBf16Freq2GHz = 3;//2.32;
const double AMXBf16PeakGopsPerCore = AMXBf16OpsPerCycleCore * AMXBf16FreqGHz;
const double AMXBf16PeakGops2PerCore = AMXBf16OpsPerCycleCore * AMXBf16Freq2GHz;

const double AVX512FreqGHz = 3;//2.32;
const double FP32OpsPerCycleCore = 64; // 2 AVX512_FMAs/cycle/core = 2*(16+16) Ops/cycle/core
const double FP32PeakGopsPerCore = FP32OpsPerCycleCore * AVX512FreqGHz;

//===============================================================
using ov::bfloat16;

#define rndup(x, n) (((x + n - 1)/n)*n)

template<typename T>
void show(const T * data, int rows, int cols) {
    std::ostream& out = std::cout;
    out << "==============\n";
    for(int i0=0; i0 < rows; i0++) {
        out << "[" << i0 << "," << 0 << "]: ";
        for(int i1=0; i1<cols; i1++)
            out << data[i0 * cols + i1] << ",";
        out << std::endl;
    }
}

template<typename T, int tile>
void tshow() {
    if (std::is_same<bfloat16,T>::value) {
        bfloat16 data[16*32];
        _tile_stored(tile, data, 64);
        show(data, 16, 32);
    }
    if (std::is_same<float,T>::value) {
        float data[16*16];
        _tile_stored(tile, data, 64);
        show(data, 16, 16);
    }
}

void vshow_bf16(__m512i v) {
    bfloat16 values[32];
    _mm512_storeu_epi16(values, v);
    show(values, 1, 32);
}

void vshow(__m512 v) {
    float values[16];
    _mm512_storeu_ps(values, v);
    show(values, 1, 16);
}

struct ANSIcolor {
    const char * code;
    ANSIcolor(const char * code = "0") : code(code){
    }
    friend std::ostream& operator<<(std::ostream& out, const ANSIcolor& obj) {
        out << "\033[" << obj.code << "m";
        return out;
    }
};

template<typename T>
struct tensor2D {
    int dims[2];
    std::shared_ptr<T> data;
    int capacity;
    int stride;
    int padded_dim1;
    tensor2D() {
        dims[0] = 0;
        dims[1] = 0;
        capacity = 0;
    }

    operator bool() {
        return dims[0] * dims[1] > 0;
    }

    tensor2D(int d0, int d1) {
        capacity = 0;
        resize(d0, d1);
        fill_rnd();
    }

    tensor2D(int d0, int d1, T * ext, int _stride) {
        capacity = 1;
        data = std::shared_ptr<T>(ext, [](void *) {});
        dims[0] = d0;
        dims[1] = d1;
        stride = _stride;
        padded_dim1 = stride / sizeof(T);
    }

    tensor2D<T> Tr() {
        tensor2D<T> ret(dims[1], dims[0]);
        for(int c0=0; c0 < dims[0]; ++c0) {
            for(int c1=0; c1 < dims[1]; ++c1) {
                ret(c1, c0) = (*this)(c0, c1);
            }
        }
        return ret;
    }

    void resize(int d0, int d1) {
        dims[0] = d0;
        dims[1] = d1;
        stride = d1 * sizeof(T);
        if (stride % 64) {
            auto stride_fix = rndup(stride, 64);
            std::cout << ANSIcolor("0;34") << "\tWarnning: stride " << stride << " is not aligned to cache line, will increase to " << stride_fix
                      << " (" << stride_fix/64 << " cache lines)\n" << ANSIcolor();
            stride = stride_fix;
        }
        padded_dim1 = stride / sizeof(T);

        // resize method never shrink capacity
        auto need_capacity = dims[0] * stride;
        if (capacity < need_capacity) {
            capacity = need_capacity;
            // align begin address to cache line is vital, so tile load can
            // use all bandwidth (L1D/L2 only deliver data in unit of 64-byte aligned cache-line)
            data = std::shared_ptr<T>(
                        reinterpret_cast<T*>(aligned_alloc(64, capacity)),
                        [](void * p) { free(p); });
        }
    }

    T & operator[](int i) {
        return data.get()[i];
    }

    const T & operator[](int i) const {
        return data.get()[i];
    }

    //https://stackoverflow.com/questions/1936399/c-array-operator-with-multiple-arguments
    T & operator()(int i0, int i1) {
        return (*this)[i0 * padded_dim1 + i1];
    }

    const T & operator()(int i0, int i1) const {
        return (*this)[i0 * padded_dim1 + i1];
    }

    void fill_rnd() {
        for(int i = 0; i<dims[0]*padded_dim1; i++) {
            // lower mantissa can help to avoid small errors in accuracy comparison
            (*this)[i] = (rand() & 1) - 0.5;
        }
    }

    void operator=(const T & v) {
        for(int k = 0; k<dims[0]*padded_dim1; k++)
            (*this)[k] = v;
    }

    void operator=(const tensor2D<T> & t2) {
        assert(dims[0]*dims[1] == t2.dims[0] * t2.dims[1]);
        for(int c0 = 0; c0 < dims[0]; c0++)
        for(int c1 = 0; c1 < dims[1]; c1++) {
            int k = c0*dims[1] + c1;
            auto c2 = k / t2.dims[1];
            auto c3 = k % t2.dims[1];
            (*this)(c0, c1) = t2(c2, c3);
        }
    }

    bool operator==(const tensor2D<T> & rhs) {
        bool ok = true;
        if (dims[0] != rhs.dims[0] || dims[1] != rhs.dims[1])
            return false;
        for(int i0=0; i0<dims[0]; i0++)
        for(int i1=0; i1<dims[1]; i1++) {
            if ((*this)(i0,i1) != rhs(i0,i1)) {
                std::cout << " operator== failed at (" << i0 << ", " << i1 << ")  value "
                          << (*this)(i0,i1) << "!=" << rhs(i0,i1) << std::endl;
                ok = false;
                return ok;
            }
        }
        return ok;
    }
    bool compare(const tensor2D<T> & rhs, float tolerance) {
        float max_abs_diff = 0;
        float max_rel_diff = 0;
        if (dims[0] != rhs.dims[0] || dims[1] != rhs.dims[1])
            return false;
        for(int i0=0; i0<dims[0]; i0++)
        for(int i1=0; i1<dims[1]; i1++) {
            auto diff = std::fabs((*this)(i0,i1) - rhs(i0,i1));
            auto rel_diff = diff/std::fabs((*this)(i0,i1));
            max_abs_diff = std::max(max_abs_diff, diff);
            if (std::fabs((*this)(i0,i1) > 0) && diff > 0)
                max_rel_diff = std::max(max_rel_diff, rel_diff);
        }
        std::cout << "max_abs_diff=" << max_abs_diff << " max_rel_diff=" << max_rel_diff;
        return tolerance > max_abs_diff;
    }
    friend std::ostream& operator<<(std::ostream& out, const tensor2D<T>& obj) {
        int i0;
        auto showline = [&](int i) {
            out << "[" << i << "," << 0 << "]: ";
            int i1;
            for(i1=0; i1<obj.dims[1] && i1 < 8; i1++) {
                out << obj(i0,i1) << ",";
            }
            if (i1 < obj.dims[1]) out << "...";
            out << std::endl;
        };
        for(i0=0; i0 < obj.dims[0] && i0 < 32; i0++) {
            showline(i0);
        }
        if (i0 < obj.dims[0]) {
            out << "... ... ... ..." << std::endl;
            showline(obj.dims[0] - 1);
        }
        return out;
    }
};

using func_act = std::function<float(float)>;

template<typename T>
void matmul(tensor2D<T> & A,
            tensor2D<T> & B,
            tensor2D<T> & C,
            float * bias = nullptr,
            func_act act = func_act()) {
    int M = C.dims[0];
    int N = C.dims[1];
    int K = A.dims[1];
    assert(B.dims[0] == K);
    assert(B.dims[1] == N);
    for(int m = 0; m < M; m++) {
        for(int n = 0; n < N; n++) {
            float sum = C(m,n);
            int k;
            for (k = 0; (k + 32) <= K; k += 32) {
                float psum0 = 0;
                float psum1 = 0;
                for(int p = 0; p < 32; p+=2) {
                    psum0 += static_cast<float>(A(m,k+p)) * static_cast<float>(B(k+p,n));
                    psum1 += static_cast<float>(A(m,k+p+1)) * static_cast<float>(B(k+p+1,n));
                }
                sum += (psum0 + psum1);
            }
            for(; k < K; k++) {
                sum += static_cast<float>(A(m,k)) * static_cast<float>(B(k,n));
            }
            if (bias) {
                sum += bias[n];
            }
            if (act) {
                sum = act(sum);
            }
            C(m,n) = sum;
        }
    }
}

//===============================================================
#ifndef _GNU_SOURCE
#define _GNU_SOURCE         /* See feature_test_macros(7) */
#endif
#include <unistd.h>
#include <sys/syscall.h>   /* For SYS_xxx definitions */

#define XFEATURE_XTILECFG 17
#define XFEATURE_XTILEDATA 18
#define XFEATURE_MASK_XTILECFG (1 << XFEATURE_XTILECFG)
#define XFEATURE_MASK_XTILEDATA (1 << XFEATURE_XTILEDATA)
#define XFEATURE_MASK_XTILE (XFEATURE_MASK_XTILECFG | XFEATURE_MASK_XTILEDATA)
#define ARCH_GET_XCOMP_PERM 0x1022
#define ARCH_REQ_XCOMP_PERM 0x1023

bool initXTILE() {
    unsigned long bitmask = 0;
    long status = syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &bitmask);
    if (0 != status) return false;
    if (bitmask & XFEATURE_MASK_XTILEDATA) return true;

    status = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
    if (0 != status)
        return false; // XFEATURE_XTILEDATA setup is failed, TMUL usage is not allowed
    status = syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &bitmask);

    // XFEATURE_XTILEDATA setup is failed, can't use TMUL
    if (0 != status || !(bitmask & XFEATURE_MASK_XTILEDATA)) return false;

    // XFEATURE_XTILEDATA set successfully, TMUL usage is allowed
    return true;
}

//===============================================================
uint64_t rdtsc_calibrate(int seconds = 1) {
    uint64_t start_ticks;
    start_ticks = __rdtsc();
    std::this_thread::sleep_for(std::chrono::seconds(seconds));
    return (__rdtsc() - start_ticks) / seconds;
}

struct RDTSC {
    uint64_t tsc_ticks_per_second;
    RDTSC() {
        tsc_ticks_per_second = rdtsc_calibrate();
        name = nullptr;
    }

    uint64_t st;
    const char * name;
    void start(const char * _name = nullptr) {
        if (name) {
            double dt = (__rdtsc() - st) * 1.0 / tsc_ticks_per_second;
            std::cout << " [RDTSC] : " << name << " took " << dt*1e6 << " us" << std::endl;
            name = nullptr;
        }
        name = _name;
        st = __rdtsc();
    }
    void end() {
        start(nullptr);
    }
};

uint64_t get_tsc_ticks_per_second() {
    static auto tsc_ticks_per_second = rdtsc_calibrate();
    return tsc_ticks_per_second;
}
double tsc2second(uint64_t diff) {
    return diff * 1.0/get_tsc_ticks_per_second();
}

uint64_t second2tsc(double sec) {
    return sec * get_tsc_ticks_per_second();
}

// timeit will record best latency for each problem in a csv log file
// and it will also show hint about whether it's improved or descreased
// over changes
struct timeit {
    const char * app_version;
    timeit() {
    }

    void set_app(const char * _app_version) {
        app_version = _app_version;
    }
    std::map<std::string, double> records;

    template<typename Callable>
    double operator()(
                      int expect_times_milliseconds,
                      const Callable & c,
                      double opsPerCall = 0,
                      double peakOpsPerSecond = 0,
                      const char * prob = nullptr) {
        int times;

        // cache warm-up
        c();
        c();

        // determine times
        if (expect_times_milliseconds > 0) {
            times = expect_times_milliseconds;
        } else {
            double expect_duration = -expect_times_milliseconds * 0.001;
            // estimate how many times required to reach the duration
            auto start = __rdtsc();
            c();
            auto oneshot = __rdtsc() - start;
            times = second2tsc(expect_duration)/oneshot;
        }

        // profiling
        auto start = std::chrono::high_resolution_clock::now();
        for(int i = 0; i < times; i++) {
            c();
        }
        auto finish = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> total_latency = finish-start;
        auto avg_latency = total_latency.count()/times;
        std::cout << ANSIcolor("0;33") << "Average latency : " << avg_latency*1e6 << " us x " << times;
        if (opsPerCall > 0 && peakOpsPerSecond > 0) {
            std::cout << "  HW Usage : " << static_cast<int>(100*(opsPerCall/avg_latency)/(peakOpsPerSecond)) << "% ("
                    << opsPerCall/avg_latency/(1e9) << " Gops /"
                    << peakOpsPerSecond/1e9 << " Gops)";
        }
        std::cout << ANSIcolor() << std::endl;
        return avg_latency;
    }
};

/*
https://www.intel.com/content/www/us/en/develop/documentation/cpp-compiler-developer-guide-and-reference/top/compiler-reference/intrinsics/intrinsics-for-amx-instructions/intrinsics-for-amx-tile-instructions/tile-loadconfig.html

void _tile_loadconfig (const void * mem_addr)
	format of memory payload. each field is a byte.
		 0: palette_id
		 1: startRow (8b)
	 2-15: reserved (must be zero)
	16-17: tile0.colsb -- bytes_per_row
	18-19: tile1.colsb
	20-21: tile2.colsb
			...
	46-47: tile15.colsb
		48: tile0.rows
		49: tile1.rows
		50: tile2.rows
			 ...
		63: tile15.rows

void _tile_storeconfig (void * mem_addr)
    Stores the current tile configuration to a 64-byte memory location specified by "mem_addr".
    The tile configuration format is specified below, and includes the tile type pallette,
    the number of bytes per row, and the number of rows. If tiles are not configured,
    all zeroes will be stored to memory.
*/

/*
void _tile_loadd (__tile dst, const void * base, int stride)
    Load tile rows from memory specifieid by "base" address and "stride"
    into destination tile "dst" using the tile configuration previously
    configured via "_tile_loadconfig".
    Operation:
        start := tileconfig.startRow
        IF start == 0 // not restarting, zero incoming state
            tilezero(dst)
        FI
        nbytes := dst.colsb
        DO WHILE start < dst.rows
            memptr := base + start * stride
            write_row_and_zero(dst, start, read_memory(memptr, nbytes), nbytes)
            start := start + 1
        OD
        zero_upper_rows(dst, dst.rows)
        zero_tileconfig_start()

void _tile_stored (__tile src, void * base, int stride)
    Store the tile specified by "src" to memory specifieid by "base" address and "stride"
    using the tile configuration previously configured via "_tile_loadconfig".
    Operation:
        start := tileconfig.startRow
        DO WHILE start < src.rows
            memptr := base + start * stride
            write_memory(memptr, src.colsb, src.row[start])
            start := start + 1
        OD
        zero_tileconfig_start()

void _tile_stream_loadd (__tile dst, const void * base, int stride)
    Load tile rows from memory specifieid by "base" address and "stride"
    into destination tile "dst" using the tile configuration previously
    configured via "_tile_loadconfig". This intrinsic provides a hint to
    the implementation that the data will likely not be reused in the near
    future and the data caching can be optimized accordingly.

void _tile_zero (__tile tdest)
    Zero the tile specified by "tdest".
    Operation:
        nbytes := palette_table[tileconfig.palette_id].bytes_per_row
        FOR i := 0 TO palette_table[tileconfig.palette_id].max_rows-1
            FOR j := 0 TO nbytes-1
                tdest.row[i].byte[j] := 0
            ENDFOR
        ENDFOR
	

void _tile_release ()
    Release the tile configuration to return to the init state, which releases all storage it currently holds.


Instruction Throughput Latency
LDTILECFG                204
STTILECFG                19
TILETRELEASE             13
TDP / *          16      52
TILELOADD         8      45
TILELOADDT1      33      48
TILESTORED       16
TILEZERO          0      16

Due to the high latency of the LDTILECFG instruction we recommend issuing a single pair
of LDTILECFG and TILERELEASE operations per Intel AMX-based DL layer implementation.


• A-tiles can have between 1-16 rows and 1-MAX_TILE_K columns.
• B-tiles can have between 1-MAX_TILE_K rows and 1–16 columns.
• C-tiles can have between 1-16 rows and 1–16 columns.

MAX_TILE_K=64/sizeof(type_t)
          = 32 BF16
          = 64 INT8

A tiles and B tiles contain data of type_t, which can be (u)int8 or bfloat16.
• C tiles contain data of type res_type_t:
• int32 if type_t=(u)int8
• float if type_t=bfloat16

Like the Intel® DL Boost use case, the B matrix must undergo a re-layout before it can be used within the
corresponding Intel AMX multiply instruction.

BF16    C_float_16x16 = A_bfloat16_16x32 * B_bfloat16_32x16 (re-layout as 16x16x2 Ab2a)
INT8    C_int32_16x16 = A_int8_16x64 * B_int8_64x16 (re-layout as 16x16x4 Ab4a)



FC:
    (2,1~900,2560)x(2560,7680)
    (2,1~900,2560)x(2560,2560)
    (2,1~900,2560)x(2560,10240) GELU
    (2,1~900,10240)x(10240,2560)

matmul:
    (1~990,80) * (80,1~990)
    (1~990,1~990) * (1~990,80)

    // (2,32,1~990,80)*(2,32,80,1~990)
    // (2,32,1~990,1~990)(2,32,1~990,80)
*/
