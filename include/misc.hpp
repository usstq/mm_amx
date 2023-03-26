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
#include <iomanip>

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

template<typename T>
void vshow(__m512i v) {
    T values[512/8/sizeof(T)];
    _mm512_storeu_si512(values, v);
    show(values, 1, 512/8/sizeof(T));
}

template<typename T>
void vshow(__m512 v) {
    T values[512/8/sizeof(T)];
    _mm512_storeu_ps(values, v);
    show(values, 1, 512/8/sizeof(T));
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


struct pretty_size {
    double sz;
    std::string txt;
    pretty_size(double sz) : sz(sz) {
        std::stringstream ss;
        ss << std::setprecision(5) << std::setw(5);
        if (sz < 1024)
            ss << sz;
        else if (sz < 1024 * 1024)
            ss << (sz / 1024) << " K";
        else if (sz < 1024 * 1024 * 1024)
            ss << (sz / 1024/1024) << " M";
        else
            ss << (sz / 1024 / 1024/1024) << " G";
        txt = ss.str();
    }
    friend std::ostream& operator<<(std::ostream& os, const pretty_size& ps) {
        os << ps.txt;
        return os;
    }
};

inline int readenv(const char * name) {
    int v = 0;
    auto * p = std::getenv(name);
    if (p)
        v = std::atoi(p);
    std::cout << ANSIcolor("32") << "ENV: " << name << " = " << v << std::endl << ANSIcolor();
    return v;
}

static auto USE_NUMA = readenv("USE_NUMA");
