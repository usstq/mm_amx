#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <vector>
#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>
#include <cstring>
#include <thread>


#include "timeit.hpp"
#include "misc.hpp"
#include "test_bw.hpp"

#include <omp.h>

#include "wcompress.hpp"

int OMP_NT = omp_thread_count();
auto &___x = std::cout << ANSIcolor("31") << "OMP_NT = " << OMP_NT << ANSIcolor() << std::endl; 
static bool initAMX = initXTILE();


// https://raw.githubusercontent.com/intel/perfmon/main/SPR/events/sapphirerapids_core.json
timeit benchmark(
    {
        {PERF_TYPE_RAW, 0x3c, "CPU_CLK_UNHALTED.THREAD"},
        //{PERF_TYPE_RAW, 0x81d0, "MEM_LOAD_RETIRED.ALL_LOADS"},
        //{PERF_TYPE_HW_CACHE, 0x10002, "LLC_load_misses"},
        //{PERF_TYPE_HW_CACHE, 0x2, "LLC_loads"},
        //{PERF_TYPE_RAW, 0x02b1, "UOPS_EXECUTED.CORE"},
    }
);


/*===========================================
  for problem  C_MxN = A_MxK * B_KxN with M<16
  we can setup tileconfig to make A,C tiles with less rows
===========================================*/
tensor2D<bfloat16> A;
tensor2D<bfloat16> B;
tensor2D<bfloat16> Bpacked;
tensor2D<float> C_ref;
tensor2D<float> C;

// L = 1 ... 4
template<int L>
tensor2D<bfloat16> repackB_1xL(tensor2D<bfloat16> &Bi) {
    int K = Bi.dims[0];
    int N = Bi.dims[1];

    assert(K % 32 == 0);
    assert(N % (L*16) == 0);

    tensor2D<bfloat16> Bo;
    // repacked as (?*16)x32
    Bo.resize(K*N/32, 32);
    bfloat16 * dst = &Bo[0];

    auto repack_1tile = [&](int k0, int n0) {
        //std::cout << "k0,n0="<<k0 << "," << n0 << std::endl;
        auto * src0 = &Bi(k0, n0);
        auto * src1 = &Bi(k0+1, n0);
        for(int k = 0; k<32; k+=2) {
            for(int n = 0; n<16; n++) {
                *dst++ = src0[n];
                *dst++ = src1[n];
            }
            src0 += Bi.stride;
            src1 += Bi.stride;
        }
    };

    // loop in computation order
    for(int n0 = 0; n0 < N; n0 += L*16) {
        // reduce on dim K
        for(int k=0; k<K; k+=32) {
            // 1xL
            for(int l=0; l<L; l++) {
                repack_1tile(k, n0 + l*16);
            }
        }
    }
    return Bo;
}

void prepare_data(int M, int K, int N) {
    B.resize(K, N);
    B.fill_rnd();
    A.resize(M, K);
    A.fill_rnd();

    // reference result
    C_ref.resize(M, N);
    for(int m = 0; m<M; m++) {
        for(int n=0; n<N; n++) {
            float sum = 0;
            for(int k=0; k<K; k++) {
                sum += A(m,k) * B(k,n);
            }
            C_ref(m,n) = sum;
        }
    }

    // prepack B into 1x2 1x4 ...
    Bpacked = repackB_1xL<2>(B);
}

void compare_with_ref() {
    int M = A.dims[0];
    int K = A.dims[1];
    int N = C_ref.dims[1];
    assert(C.is_normal());
    if (C_ref == C) {
        std::cout << ANSIcolor("1;32") << "Match!\n" << ANSIcolor();
    } else {
        std::cout << ANSIcolor("1;31") << "Mismatch!\n" << ANSIcolor();
        //std::cout << C_ref << std::endl;
        //std::cout << C << std::endl;
    }
}

void fc16m_base_1x2() {
     // assume each tile of B is already packed as 16x(16x2)
    int M = A.dims[0];
    int K = A.dims[1];
    int N = C_ref.dims[1];
    C.resize(M, N);

    // tiles allocation
    // C: 0,1,2,3
    // A: 4,
    // B: 5,6,7
    tileconfig_t tfg(1, 0, {M,M,M,M,M,16,16,16}, 64);

    zero_tiles<0, 1, 2, 3>();
    load_tiles_with_random_bf16<4, 5>();
    benchmark.tag(__func__, M, K, N)([&]()
                            {
        auto * pB0 = &Bpacked[0];
        auto * pC0 = &C[0];
        
        for(int n0 = 0; n0 < N; n0 += 2*16) {
            // reduce on dim K
            zero_tiles<0, 1>();
            auto * pA0 = &A[0];
            auto Astride = A.stride;
            for(int k=0; k<K; k+=32) {
                // 1x2
                _tile_loadd(4, pA0, Astride); pA0 += 32;   // tile A Mx32
                _tile_loadd(6, pB0, 64); pB0 += 16*32;     // tile B 16x32
                _tile_loadd(7, pB0, 64); pB0 += 16*32;     // tile B 16x32
                _tile_dpbf16ps(0, 4, 6); // C0 += A*B0
                _tile_dpbf16ps(1, 4, 7); // C1 += A*B1
            }
            // no post ops, just store
            _tile_stored(0, pC0 + n0, C.stride);
            _tile_stored(1, pC0 + n0 + 16, C.stride);
        }
        }, Bpacked.capacity);
    
    compare_with_ref();
}


void fc16m_test_all(int M, int K, int N) {
    
    double BmatrixSize = K * N * sizeof(bfloat16);
    std::cout << "# K = " << K / 32 << "*32 = " << K << ", sizeof(B)=" << pretty_size(BmatrixSize, "B") << std::endl;
    std::cout << " preparing data ..." << std::flush;
    prepare_data(M, K, N);
    std::cout << "\r                 \r" << std::flush;

    fc16m_base_1x2();
}

int main()
{
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

    // test 1000 ms
    benchmark.set_time_ms(-2000);
    benchmark.set_unit("B/s");
    benchmark.set_peak_metric_per_second(18e9); // 18 GB/s

    // accuracy test
    fc16m_test_all(7, 64, 128);

    // L1D
    fc16m_test_all(2, 16 * 32, 192);
    //fc16m_test_all(102400 * 32);
    // fc16m_test_all(80*32);
}
