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

#include "misc.hpp"
#include "kernels_avx2.hpp"
#include "timeit.hpp"
#include <omp.h>
#include "test_bw.hpp"
// https://raw.githubusercontent.com/intel/perfmon/main/SPR/events/sapphirerapids_core.json
timeit benchmark;
/*
(
    {
        {PERF_TYPE_RAW, 0x3c, "CPU_CLK_UNHALTED.THREAD"},
        //{PERF_TYPE_RAW, 0x81d0, "MEM_LOAD_RETIRED.ALL_LOADS"},
        //{PERF_TYPE_HW_CACHE, 0x10002, "LLC_load_misses"},
        //{PERF_TYPE_HW_CACHE, 0x2, "LLC_loads"},
        //{PERF_TYPE_RAW, 0x02b1, "UOPS_EXECUTED.CORE"},
    }
);
*/

// vfmadd132ps ymm(8 floats)  Throughput (CPI)=0.5
const double vfmaddOpsPerCycle = 16;


int OMP_NT = omp_thread_count();

struct MatmulMTOMP {
    std::vector<std::shared_ptr<avx2::Matmul>> ops;
    bool transposeB = false;
    MatmulMTOMP(bool constB = false, bool transposeB = false) : transposeB(transposeB) {
        for(int i = 0; i < OMP_NT; i++)
            ops.push_back(std::make_shared<avx2::Matmul>(constB, transposeB));
    }

    template<typename P>
    void operator()(tensor2D<float> & matA,
                    tensor2D<float> & matB,
                    tensor2D<float> & matC,
                    P ppkernel) {
        int M = matA.dims[0];
        int K = matA.dims[1];
        int N = matB.dims[transposeB ? 0:1];
        // split along N dimension
        int work_amount = rndup(N, 16)/16;

        auto kernel = [&](int tid, int cnt) {
            int start, end;
            splitter(work_amount, cnt, tid, start, end);
            int n0 = start*16;
            int n1 = end*16;
            if (n1 > N) n1 = N;
            //tensor2D<bfloat16> copyA = matA.clone();
            // C[:, N0:N1] = A * B[:, N0:N1]
            (*ops[tid].get())(matA, matB, matC, n0, n1, ppkernel);
        };

        #pragma omp parallel for
        for(int i = 0; i<OMP_NT; i++) {
            kernel(i, OMP_NT);
        }
    }
};

void amx_Matmul_perf_float(int M, int K, int N, int times = -1000) {
    tensor2D<float> A(M, K);
    tensor2D<float> B(K, N);
    tensor2D<float> Br = B.Tr();
    tensor2D<float> C(M, N);
    tensor2D<float> C0(M, N);
    tensor2D<float> Bias(1, N);
    avx2::PP::AddbiasRelu pp(&Bias[0]);
    MatmulMTOMP fc(true, false);
    MatmulMTOMP mm(false, false);
    MatmulMTOMP mmTr(false, true);
    std::cout << __func__ << " [" << M << "," << K << "," << N << "] ";

    C0=0;
    matmul(A, B, C0, &Bias[0], [](float x){        return std::max(x, 0.0f);    });
    //matmul(A, B, C0);
    C = 0;
    fc(A, B, C, pp);
    if (C0 == C) {
        std::cout << ANSIcolor("1;32") << "fc-Match!" << ANSIcolor();
        //std::cout << C << std::endl;
    } else {
        std::cout << ANSIcolor("1;31") << "fc-Mismatch!" << ANSIcolor();
        logger() << C0 << std::endl;
        logger() << C << std::endl;
    }

    C = 0;
    mm(A, B, C, pp);
    if (C0 == C) {
        std::cout << ANSIcolor("1;32") << "mm-Match!" << ANSIcolor();
        //std::cout << C << std::endl;
    } else {
        std::cout << ANSIcolor("1;31") << "mm-Mismatch!" << ANSIcolor();
        logger() << C0 << std::endl;
        logger() << C << std::endl;
    }

    C = 0;
    mmTr(A, Br, C, pp);
    if (C0 == C) {
        std::cout << ANSIcolor("1;32") << "mmTr-Match!" << ANSIcolor();
        //std::cout << C << std::endl;
    } else {
        std::cout << ANSIcolor("1;31") << "mmTr-Mismatch!" << ANSIcolor();
        logger() << C0 << std::endl;
        logger() << C << std::endl;
    }

    std::cout << std::endl;

    //benchmark.set_peak_metric_per_second(vfmaddOpsPerCycle * 4.3e9); // 4.3GHz

    benchmark.tag("fc")(times, [&](){
        fc(A, B, C, pp);
    },
    double(M * N) * K * 2, vfmaddOpsPerCycle * 4.3e9);

    benchmark.tag("mm")(times, [&](){
        mm(A, B, C, pp);
    },
    double(M * N) * K * 2);

    benchmark.tag("mmTr")(times, [&](){
        mmTr(A, Br, C, pp);
    },
    double(M * N) * K * 2);
}

int main(int argc, const char *argv[]) {
    benchmark.set_app(argv[0]);

    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    std::cout << ANSIcolor("31") << "omp_get_num_threads() = " << omp_get_num_threads() << std::endl << ANSIcolor();
    std::cout << ANSIcolor("31") << "OMP_NT = " << OMP_NT << std::endl << ANSIcolor();

    //test_all_bw(3);

    // amx_Matmul_perf_float(128, 384, 51864);
    amx_Matmul_perf_float(128, 384, 51864, -1000);

    amx_Matmul_perf_float(128, 385, 51864, -1000);

    amx_Matmul_perf_float(126, 384, 51872, -1000);
    amx_Matmul_perf_float(126+6, 384, 51872, -1000);
    amx_Matmul_perf_float(128, 384, 51872, -1000);
    

    //[1,64,384] x [384, 384]
    amx_Matmul_perf_float(66, 384, 384, -1000);
    
    //amx_Matmul_perf_float(16, 256, 256);
    //amx_Matmul_perf_float(224, 256, 256);

    return 0;
}
