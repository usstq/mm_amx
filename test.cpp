#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <vector>
#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <cassert>
#include <cstring>
#include <thread>

#include "bf16.hpp"

// g++ ./test.cpp -O2 -lpthread -march=native -lstdc++

// to use VNNI, we need higher version of compiler:
//    clang-9 ./test_conv.cpp -O2 -lpthread -march=native -lstdc++ && ./a.out

// to use AMX, we need intel compiler 
//   source  ~/intel/oneapi/setvars.sh
//   icx ./mm_amx_bf16.cpp -O2 -lpthread -march=native -lstdc++

// objdump -C -S ./a.out > a.asm

//#include "kernels_amxbf16.hpp"
#include "kernels_avx512.hpp"
#include "thread_pool.hpp"
#include "timeit.hpp"
#include "misc.hpp"
timeit timer;

void amx_Matmul_perf_float(int M, int K, int N, int times = -1000) {
    tensor2D<float> A(M, K);
    tensor2D<float> B(K, N);
    tensor2D<float> C(M, N);
    tensor2D<float> C0(M, N);
    tensor2D<float> Bias(1, N);
    avx512::Matmul<avx512::RELU> mm;
    std::cout << __func__ << " [" << M << "," << K << "," << N << "] ";

    C0=0;
    matmul(A, B, C0, &Bias(0,0), [](float x){
        return std::max(x, 0.0f);
    });
    mm(A, B, C, &Bias(0,0));
    if (C0 == C) {
        std::cout << ANSIcolor("1;32") << "Match!\n" << ANSIcolor();
        //std::cout << C << std::endl;
    } else {
        std::cout << ANSIcolor("1;31") << "Mismatch!\n" << ANSIcolor();
        std::cout << C0 << std::endl;
        std::cout << C << std::endl;
    }

    timer(times, [&](){
        mm(A, B, C, &Bias(0,0));
    },
    double(M * N) * K * 2,
    FP32PeakGopsPerCore * 1e9);
}

int main(int argc, const char *argv[]) {
    timer.set_app(argv[0]);

    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

    amx_Matmul_perf_float(512, 256, 256);
    //amx_Matmul_perf_float(16, 256, 256);
    //amx_Matmul_perf_float(224, 256, 256);

    return 0;
}
