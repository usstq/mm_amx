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
    const char * name;
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
};

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
