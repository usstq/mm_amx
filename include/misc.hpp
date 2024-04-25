#pragma once

#include <cassert>
#include <chrono>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <thread>
#include <vector>

// #include "thread_pool.hpp"
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

#define rndup(x, n) (((x + n - 1) / n) * n)

template <typename T>
inline void show(const T* data, int rows, int cols) {
    std::ostream& out = std::cout;
    out << "==============\n";
    for (int i0 = 0; i0 < rows; i0++) {
        out << "[" << i0 << "," << 0 << "]: ";
        for (int i1 = 0; i1 < cols; i1++)
            // https://stackoverflow.com/questions/14644716/how-to-output-a-character-as-an-integer-through-cout/28414758#28414758
            out << +data[i0 * cols + i1] << ",";
        out << std::endl;
    }
}

template <typename T>
inline void vshow(__m512i v) {
    T values[512 / 8 / sizeof(T)];
    _mm512_storeu_si512(values, v);
    show(values, 1, 512 / 8 / sizeof(T));
}

template <typename T>
inline void vshow(__m512 v) {
    T values[512 / 8 / sizeof(T)];
    _mm512_storeu_ps(values, v);
    show(values, 1, 512 / 8 / sizeof(T));
}

struct ANSIcolor {
    const char* code;
    std::string str;
    ANSIcolor(const char* code = "0") : code(code) {
        std::stringstream ss;
        ss << "\033[" << code << "m";
        str = ss.str();
    }

    friend std::ostream& operator<<(std::ostream& out, const ANSIcolor& obj) {
        out << "\033[" << obj.code << "m";
        return out;
    }
};

struct pretty_size {
    double sz;
    std::string txt;
    pretty_size(double sz, const char* unit = "") : sz(sz) {
        std::stringstream ss;
        ss << std::setprecision(5) << std::setw(5);
        if (sz < 1024)
            ss << sz;
        else if (sz < 1024 * 1024)
            ss << (sz / 1024) << " K";
        else if (sz < 1024 * 1024 * 1024)
            ss << (sz / 1024 / 1024) << " M";
        else
            ss << (sz / 1024 / 1024 / 1024) << " G";
        ss << unit;
        txt = ss.str();
    }
    friend std::ostream& operator<<(std::ostream& os, const pretty_size& ps) {
        os << ps.txt;
        return os;
    }
};

inline int readenv(const char* name) {
    int v = 0;
    auto* p = std::getenv(name);
    if (p)
        v = std::atoi(p);
    std::cout << ANSIcolor("32") << "ENV: " << name << " = " << v << std::endl << ANSIcolor();
    return v;
}

struct EnvVar {
    std::string v_str;
    int v_int;
    const char* name;
    void init(const char* _name, bool with_int_default, int int_default = 0) {
        name = _name;
        auto* p = std::getenv(name);
        bool is_int = false;
        if (p) {
            v_str = p;
            char* end;
            v_int = strtol(p, &end, 0);
            is_int = (end != p) || with_int_default;
        } else if (with_int_default) {
            v_int = int_default;
            is_int = true;
        }
        std::cout << ANSIcolor("32") << "ENV: " << name << " = ";
        if (is_int)
            std::cout << v_int;
        else
            std::cout << "\"" << v_str << "\"";
        std::cout << std::endl << ANSIcolor();
    }
    EnvVar(const char* name) { init(name, false); }
    EnvVar(const char* name, int int_default) { init(name, true, int_default); }

    int operator=(int v_new) {
        v_int = v_new;
        std::cout << ANSIcolor("32") << "ENV: " << name << " = " << v_int << ANSIcolor() << std::endl;
        return v_int;
    }

    operator std::string() { return v_str; }
    operator int() { return v_int; }
    operator bool() { return v_int; }
};

/*
sdm-vol-4.pdf

0x1A0: IA32_MISC_ENABLE

    bit 9 (1<<9  0x200)
        Hardware Prefetcher Disable (R/W)
        When set, disables the hardware prefetcher operation on streams of data.
        When clear (default), enables the prefetch queue.
        Disabling of the hardware prefetcher may impact processor performance.

    bit 19 (1<<19 0x80000)
        Adjacent Cache Line Prefetch Disable (R/W)
        When set to 1, the processor fetches the cache line that contains data currently required by the processor.
        When set to 0, the processor fetches cache lines that comprise a cache line pair (128 bytes).
        Single processor platforms should not set this bit. Server platforms should set
        or clear this bit based on platform performance observed in validation and
        testing.
        BIOS may contain a setup option that controls the setting of this bit.

    bit 37:
        DCU Prefetcher Disable (R/W)
        When set to 1, the DCU L1 data cache prefetcher is disabled. The default
        value after reset is 0. BIOS may write ‘1’ to disable this feature.
        The DCU prefetcher is an L1 data cache prefetcher. When the DCU prefetcher
        detects multiple loads from the same line done within a time limit, the DCU
        prefetcher assumes the next line will be required. The next line is prefetched
        in to the L1 data cache from memory or L2.

0x1A4: MSR_PREFETCH_CONTROL

bit-0 L2 Hardware Prefetcher Disable (R/W)
        If 1, disables the L2 hardware prefetcher, which fetches additional lines of
        code or data into the L2 cache.

bit-1 L2 Adjacent Cache Line Prefetcher Disable (R/W)
        If 1, disables the adjacent cache line prefetcher, which fetches the cache
        line that comprises a cache line pair (128 bytes).

bit-2 DCU Hardware Prefetcher Disable (R/W)
        If 1, disables the L1 data cache prefetcher, which fetches the next cache
        line into L1 data cache.
*/
#define MSR_BIT(x) (uint64_t(1) << x)

#define MASK_1A0_PF (MSR_BIT(9) | MSR_BIT(19) | MSR_BIT(37))
#define MASK_1A4_PF (MSR_BIT(0) | MSR_BIT(1) | MSR_BIT(2))

#include <fcntl.h>
#include <sched.h>
#include <unistd.h>

struct MSRConfig {
    int offset = -1;
    uint32_t mask = 0;
    EnvVar MSRCONFIG{"MSRCONFIG", 0};

    std::vector<uint64_t> v_old;
    std::vector<int> cpus;

    MSRConfig() {
        std::string msr_cfg = MSRCONFIG;
        if (sscanf(msr_cfg.c_str(), "%x,%x", &offset, &mask) < 2) {
            offset = -1;
            return;
        }

        cpu_set_t cpu_set;        /* Define your cpu_set bit mask. */
        CPU_ZERO(&cpu_set); 
        if(sched_getaffinity(0, sizeof(cpu_set), &cpu_set)) {
            perror("sched_getaffinity");
            abort();
        }
        auto nproc = sysconf(_SC_NPROCESSORS_ONLN);
        for (int i = 0; i < nproc; i++) {
            if (CPU_ISSET(i, &cpu_set)) {
                cpus.push_back(i);
            }
        }

        v_old.resize(cpus.size(), 0);
        for (int i=0; i < cpus.size(); i++) {
            v_old[i] = rdmsr<uint64_t>(cpus[i], offset);
            auto v_new = v_old[i] | mask;
            wrmsr(cpus[i], offset, v_new);
            auto v_now = rdmsr<uint64_t>(cpus[i], offset);
            printf("[cpu %d] MSR[%x] setup  %lx -> %lx  ...\n", cpus[i], offset, v_old[i], v_now);
        }
    }

    ~MSRConfig() {
        if (offset >= 0) {
            for (int i=0; i < cpus.size(); i++) {
                wrmsr(cpus[i], offset, v_old[i]);
                auto v_now = rdmsr<uint64_t>(cpus[i], offset);
                printf("[cpu %d] MSR[%x] recover to %lx  ...\n", cpus[i], offset, v_now);
            }
        }
    }

    template <typename T>
    T rdmsr(int cpu, int offset) {
        std::string fname = "/dev/cpu/";
        fname += std::to_string(cpu);
        fname += "/msr";
        auto fd = open(fname.c_str(), O_RDONLY);
        lseek(fd, offset, SEEK_SET);
        T rv = 0;
        auto nbytes = read(fd, &rv, sizeof(rv));
        if (nbytes != sizeof(rv)) {
            perror("read MSR file failed");
            abort();
        }
        return rv;
    }

    template <typename T>
    T wrmsr(int cpu, int offset, T rv) {
        std::string fname = "/dev/cpu/";
        fname += std::to_string(cpu);
        fname += "/msr";
        auto fd = open(fname.c_str(), O_RDWR);
        lseek(fd, offset, SEEK_SET);
        auto nbytes = write(fd, &rv, sizeof(rv));
        if (nbytes != sizeof(rv)) {
            printf("write cpu [%d] MSR [0x%X] with [0x%lX] failed: nbytes=%zd\n", cpu, offset, rv, nbytes);
            perror("");
            abort();
        }
        return rv;
    }
};

#define stringify(a) xstr(a)
#define xstr(a) #a
#define ASSERT(cond)                                                                                                                                                                                                                                                                   \
    if (!(cond)) {                                                                                                                                                                                                                                                                     \
        std::cout << "ASSERT(" << stringify(cond) << ") failed at " << __LINE__ << "\n" << std::endl;                                                                                                                                                                                  \
        abort();                                                                                                                                                                                                                                                                       \
    }

void clflush(void* pv, int bytes) {
    auto* p = reinterpret_cast<uint8_t*>(pv);
    for (int i = 0; i < bytes; i += 64) {
        _mm_clflushopt(p + i);
    }
    _mm_mfence();
};

void sw_prefetch_L2(void* pv, int bytes) {
    auto* p = reinterpret_cast<uint8_t*>(pv);
    int i;
    for (i = 0; i + 256 <= bytes; i += 64 * 4) {
        _mm_prefetch(p + i, _MM_HINT_T2);
        _mm_prefetch(p + i + 64, _MM_HINT_T2);
        _mm_prefetch(p + i + 64 * 2, _MM_HINT_T2);
        _mm_prefetch(p + i + 64 * 3, _MM_HINT_T2);
    }
    for (; i < bytes; i += 64) {
        _mm_prefetch(p + i, _MM_HINT_T2);
    }
    _mm_mfence();
};

void load_prefetch_L2(void* pv, int bytes, int rounds = 1) {
    auto* p = reinterpret_cast<uint8_t*>(pv);
    int i;
    auto sum0 = _mm512_setzero_epi32();
    auto sum1 = _mm512_setzero_epi32();
    auto sum2 = _mm512_setzero_epi32();
    auto sum3 = _mm512_setzero_epi32();
    for (int r = 0; r < rounds; r++) {
        for (i = 0; i + 256 <= bytes; i += 64 * 4) {
            auto a0 = _mm512_loadu_epi32(p + i);
            auto a1 = _mm512_loadu_epi32(p + i + 64);
            auto a2 = _mm512_loadu_epi32(p + i + 64 * 2);
            auto a3 = _mm512_loadu_epi32(p + i + 64 * 3);
            sum0 = _mm512_add_epi32(sum0, a0);
            sum1 = _mm512_add_epi32(sum1, a1);
            sum2 = _mm512_add_epi32(sum2, a2);
            sum3 = _mm512_add_epi32(sum3, a3);
        }
    }
    sum0 = _mm512_add_epi32(sum0, sum1);
    sum2 = _mm512_add_epi32(sum2, sum3);
    sum0 = _mm512_add_epi32(sum0, sum2);
    if (_mm512_cvtsi512_si32(sum0) == 0x1234) {
        std::cout << 1;
    }
};

#include <omp.h>

inline int get_nthr() {
    static int _nthr = []() {
        int nthr;
#pragma omp parallel
        {
            if (0 == omp_get_thread_num())
                nthr = omp_get_num_threads();
        }
        return nthr;
    }();
    return _nthr;
}

//===============================================================
#ifndef _GNU_SOURCE
#define _GNU_SOURCE /* See feature_test_macros(7) */
#endif
#include <sys/syscall.h> /* For SYS_xxx definitions */
#include <unistd.h>

#define XFEATURE_XTILECFG 17
#define XFEATURE_XTILEDATA 18
#define XFEATURE_MASK_XTILECFG (1 << XFEATURE_XTILECFG)
#define XFEATURE_MASK_XTILEDATA (1 << XFEATURE_XTILEDATA)
#define XFEATURE_MASK_XTILE (XFEATURE_MASK_XTILECFG | XFEATURE_MASK_XTILEDATA)
#define ARCH_GET_XCOMP_PERM 0x1022
#define ARCH_REQ_XCOMP_PERM 0x1023

inline bool initXTILE() {
    unsigned long bitmask = 0;
    long status = syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &bitmask);
    if (0 != status)
        return false;
    if (bitmask & XFEATURE_MASK_XTILEDATA)
        return true;

    status = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
    if (0 != status)
        return false; // XFEATURE_XTILEDATA setup is failed, TMUL usage is not allowed
    status = syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &bitmask);

    // XFEATURE_XTILEDATA setup is failed, can't use TMUL
    if (0 != status || !(bitmask & XFEATURE_MASK_XTILEDATA))
        return false;

    // XFEATURE_XTILEDATA set successfully, TMUL usage is allowed
    std::cout << "initXTILE success!\n";
    return true;
}
//===============================================================
#ifdef _tile_stored
template <typename T, int tile>
inline void tshow() {
    if (std::is_same<ov::bfloat16, T>::value) {
        ov::bfloat16 data[16 * 32];
        _tile_stored(tile, data, 64);
        show(data, 16, 32);
    }
    if (std::is_same<float, T>::value) {
        float data[16 * 16];
        _tile_stored(tile, data, 64);
        show(data, 16, 16);
    }
    if (std::is_same<int8_t, T>::value) {
        int8_t data[16 * 64];
        _tile_stored(tile, data, 64);
        show(data, 16, 64);
    }
    if (std::is_same<uint8_t, T>::value) {
        uint8_t data[16 * 64];
        _tile_stored(tile, data, 64);
        show(data, 16, 64);
    }
}
#endif

template <typename T0, typename T1>
std::vector<std::pair<T0, T1>> zip_vector(const std::vector<T0>& v0, const std::vector<T1>& v1) {
    std::vector<std::pair<T0, T1>> ret;
    auto sz = v0.size();
    for (decltype(sz) i = 0; i < sz; i++)
        ret.push_back(std::make_pair(v0[i], v1[i]));
    return ret;
}

struct tileconfig_t {
    uint8_t palette_id;
    uint8_t startRow;
    uint8_t reserved[14];
    uint16_t cols[16];
    uint8_t rows[16];
    tileconfig_t() = default;

    tileconfig_t(int palette, int _startRow, const std::vector<std::pair<int, int>>& _rows_columnsBytes) {
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
        load();
    }

    tileconfig_t(int palette, int _startRow, const std::vector<int>& _rows, int columnsBytes) : tileconfig_t(palette, _startRow, zip_vector(_rows, std::vector<int>(_rows.size(), columnsBytes))) {}
    tileconfig_t(int palette, int _startRow, int numTiles, int _rows, int columnsBytes) : tileconfig_t(palette, _startRow, std::vector<std::pair<int, int>>(numTiles, {_rows, columnsBytes})) {}

    ~tileconfig_t() { _tile_release(); }
    void load() {
        // std::cout << "\ttile load config ... " << std::flush;
        _tile_loadconfig(this);
        // std::cout << *this << std::flush << std::endl;
    }
    void store() { _tile_storeconfig(this); }
    friend std::ostream& operator<<(std::ostream& out, const tileconfig_t& cfg) {
        out << " palette_id=" << static_cast<int>(cfg.palette_id);
        out << " startRow=" << static_cast<int>(cfg.startRow);
        out << " row x colsb=(";
        for (int i = 0; i < 16; i++) {
            if (cfg.rows[i] == 0 && cfg.cols[i] == 0)
                continue;
            if (i > 0)
                out << ",";
            out << static_cast<int>(cfg.rows[i]) << "x" << static_cast<int>(cfg.cols[i]);
        }
        out << ")";
        return out;
    }
} __attribute__((__packed__));

// default implementation
template <typename T>
struct TypeName {
    static const char* get() { return typeid(T).name(); }
};

// a specialization for each type of those you want to support
// and don't like the string returned by typeid
template <>
struct TypeName<int32_t> {
    static const char* get() { return "int32_t"; }
};
template <>
struct TypeName<float> {
    static const char* get() { return "foat"; }
};
template <>
struct TypeName<ov::bfloat16> {
    static const char* get() { return "bfloat16"; }
};
template <>
struct TypeName<int8_t> {
    static const char* get() { return "int8_t"; }
};

std::ostream& logger() {
    // https://stackoverflow.com/questions/11826554/standard-no-op-output-stream
    static class NullBuffer : public std::streambuf {
    public:
        int overflow(int c) { return c; }
    } null_buffer;
    static std::ostream null_stream(&null_buffer);
    static int log_level = std::getenv("LOGL") ? atoi(std::getenv("LOGL")) : 0;
    return log_level == 0 ? null_stream : std::cout;
}

template <typename T, typename Q>
inline void splitter(const T& n, const Q& team, const Q& tid, T& n_start, T& n_end) {
    if (team <= 1 || n == 0) {
        n_start = 0;
        n_end = n;
    } else {
        T n1 = (n + (T)team - 1) / (T)team;
        T n2 = n1 - 1;
        T T1 = n - n2 * (T)team;
        n_end = (T)tid < T1 ? n1 : n2;
        n_start = (T)tid <= T1 ? tid * n1 : T1 * n1 + ((T)tid - T1) * n2;
    }

    n_end += n_start;
}

template <typename... Ts>
void easy_cout(const char* file, const char* func, int line, Ts... args) {
    std::string file_path(file);
    std::string file_name(file);
    auto last_sep = file_path.find_last_of('/');
    if (last_sep == std::string::npos)
        last_sep = file_path.find_last_of('\\');
    if (last_sep != std::string::npos)
        file_name = file_path.substr(last_sep + 1);

    std::string file_name_with_line = file_name + ":" + std::to_string(line);
    auto tag = file_name_with_line + " " + func + "()";

    std::stringstream ss;
    int dummy[sizeof...(Ts)] = {(ss << args, 0)...};
    std::cout << tag << " " << ss.str() << std::endl;
}

#define ECOUT(...) easy_cout(__FILE__, __func__, __LINE__, __VA_ARGS__)

inline int omp_thread_count() {
    int n = 0;
#pragma omp parallel reduction(+ : n)
    n += 1;
    return n;
}

#define FORCE_INLINE inline __attribute__((always_inline))
// #define FORCE_INLINE

#ifdef ENABLE_NUMA
static auto USE_NUMA = readenv("USE_NUMA");
#endif
