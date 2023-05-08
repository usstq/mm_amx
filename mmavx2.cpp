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
#include <cmath>

#include "misc.hpp"
#include "kernels_avx2.hpp"
#include "timeit.hpp"
#include <omp.h>
#include "test_bw.hpp"
// https://raw.githubusercontent.com/intel/perfmon/main/SPR/events/sapphirerapids_core.json
timeit benchmark
(
    {
        {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, "HW_CYCLES"},
        //{PERF_TYPE_RAW, 0x3c, "CPU_CLK_UNHALTED.THREAD"},
        //{PERF_TYPE_RAW, 0x81d0, "MEM_LOAD_RETIRED.ALL_LOADS"},
        //{PERF_TYPE_HW_CACHE, 0x10002, "LLC_load_misses"},
        //{PERF_TYPE_HW_CACHE, 0x2, "LLC_loads"},
        //{PERF_TYPE_RAW, 0x02b1, "UOPS_EXECUTED.CORE"},
    }
);


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

    // projection based on memory bandwidth
    double CacheBW_BpS = (710.0/6)  *1024*1024*1024; // L2 cache bandwidth ()
    double DDRBW_BpS = (63.0/6)*1024*1024*1024;

    // access A/B/C at lease once, in DDR bandwidth
    auto latDDR = (A.capacity + double(B.capacity + C.capacity)/OMP_NT)/DDRBW_BpS;
    // for each output point, 2*K reads from cache
    auto latL2 = (M*N)*(2*K)*sizeof(float)/OMP_NT/CacheBW_BpS;
    auto latL1 = (M*N)*(2*K)*sizeof(float)/OMP_NT/CacheBW_BpS;
    auto latALU = (M*N/OMP_NT)*(K/8)/(2 * (OMP_NT > 1 ? 4.229e9 : 4.677e9)); // fmadd tput:0.5 @ 4GHz
    std::cout << " Proj: MEM=" << (latDDR + latL2) * 1e3 << " ms (" << latDDR*1e3 << " + " << latL2*1e3 << ")"
              << " ALU=" << latALU*1e3 << " ms" << std::endl;

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

int test_exp() {
    float x[8] = {
        0.1f, 12.2f, 3.4f, 80.0f,
        -0.1f, -12.2f, -3.4f, -80.0f,
    };
    float y[8];
    auto m256x = _mm256_loadu_ps(x);
    avx2::functional::exp_ps(m256x);
    _mm256_storeu_ps(y, m256x);
    int errors = 0;
    for(int i=0;i<8;i++) {
        auto ref = expf(x[i]);
        if (std::abs((y[i] - ref)/ref) > 0.0001f) {
            errors++;
            std::cout << ANSIcolor("31") << "error: " << y[i] << "   ref:" << expf(x[i]) << ANSIcolor() << std::endl;
        }
    }
    if (errors == 0) {
        std::cout << ANSIcolor("32") << __func__ << " Pass" << ANSIcolor() << std::endl;
    }
    return 0;
}

int test_hmax() {
    float x[8] = {
        0.1f, 12.2f, 3.4f, 80.0f,
        -0.1f, -12.2f, -3.4f, -80.0f,
    };
    float y[8];
    int errors = 0;
    for(int i=0; i<8; i++) {
        x[i] = 654.123f;
        auto m256x = _mm256_loadu_ps(x);
        avx2::functional::hmax(m256x);
        _mm256_storeu_ps(y, m256x);
        for(int k=0;k<8;k++) {
            if (y[k] != x[i]) {
                errors++;
                std::cout << ANSIcolor("31") << "error: " << y[k] << " != max " << x[i] << ANSIcolor() << std::endl;
            }
        }
        x[i] = i;
    }
    if (errors == 0) {
        std::cout << ANSIcolor("32") << __func__ << " Pass" << ANSIcolor() << std::endl;
    }
    return 0;
}

int test_softmax() {
    tensor2D<float> x;
    tensor2D<float> y0;
    tensor2D<float> y1;

    auto softmax_ref = [&](tensor2D<float>& x, tensor2D<float>& y) {
        float x_max = std::numeric_limits<float>::lowest();
        for(int i = 0; i < x.dims[1]; i++) {
            x_max = std::max(x_max, x[i]);
        }
        y = x.clone();
        float sum = 0;
        for(int i = 0; i < x.dims[1]; i++) {
            y[i] = expf(x[i]-x_max);
            sum += y[i];
        }
        for(int i = 0; i < x.dims[1]; i++) {
            y[i] = y[i]/sum;
        }
    };
    int errors = 0;
    for(int N = 1; N < 129; N++) {
        x.resize(1, N);
        x.fill_rnd();
        softmax_ref(x, y0);
        y1 = x.clone();
        avx2::functional::softmax(&y1[0], N);
        for(int i=0;i<N;i++) {
            if (abs((y0[i] - y1[i])/y0[i]) > 0.0001f) {
                errors ++;
                std::cout << "#" << i << "/" << N << ":  " <<y0[i] << " vs " << y1[i] << " diff " << (y0[i] - y1[i]) << std::endl;
            }
        }
    }
    if (errors == 0) {
        std::cout << ANSIcolor("32") << __func__ << " Pass" << ANSIcolor() << std::endl;
    }
    return 0;
}

// confirmed, _mm256_maskstore_ps support dest address unaligned to 256bits
int test_vmaskmovps_alignment () {
    float data[8] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
    int32_t mask[8] = {-1, 0, -1, 0, -1, 0, -1, 0};
    float data_masked[8] = {0.1f, 0, 0.3f, 0, 0.5f, 0, 0.7f, 0};
    float data_cmp[8];

    float a[64];

    auto ymm = _mm256_loadu_ps(data);
    auto ymm_mask = _mm256_loadu_si256((__m256i const *)mask);
    int errors = 0;
    for(int i=0;i<64-8;i++) {
        for(int k=0;k<64;k++) {
            a[k] = k;
        }
        _mm256_maskstore_ps(a + i, ymm_mask, ymm);
        
        for(int k=0;k<64;k++) {
            int offset = k-i;
            float expect = k;
            if (offset >=0 && offset <8 && ((offset & 1) == 0))  expect = data[offset];
            if (a[k] != expect) {
                errors++;
                std::cout << "when i==" << i << " a[" << k << "] is " << a[k] << ", expect " << expect << std::endl; 
            }
        }

        // _mm256_maskload_ps also support
        auto ymm_load = _mm256_maskload_ps(a + i, ymm_mask);
        _mm256_storeu_ps(data_cmp, ymm_load);
        for(int k=0;k<8;k++) {
            if (data_cmp[k] != data_masked[k]) {
                errors++;
            }
        }
    }
    if (errors == 0) {
        std::cout << ANSIcolor("32") << __func__ << " Pass" << ANSIcolor() << std::endl;
    }else{
        std::cout << ANSIcolor("31") << __func__ << " Failed" << ANSIcolor() << std::endl;
    }
    return 0;
}
int main(int argc, const char *argv[]) {
    benchmark.set_app(argv[0]);

    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    std::cout << ANSIcolor("31") << "omp_get_num_threads() = " << omp_get_num_threads() << std::endl << ANSIcolor();
    std::cout << ANSIcolor("31") << "OMP_NT = " << OMP_NT << std::endl << ANSIcolor();

    //return test_vmaskmovps_alignment();
    //return test_exp();
    //return test_hmax();
    //return test_softmax();
    //test_all_bw(3);

    if (0) {
        avx2::PP::None nonepp;
        constexpr int M = 6;
        constexpr int N = 16;
        constexpr bool Ngt8 = N>8;
        int K = 1920*8;
        tensor2D<float> A(6, K);
        tensor2D<float> B(K, N, true);
        tensor2D<float> C(6, N, true);
        auto * pA = &A[0];
        auto * pB = &B[0];
        auto * pC = &C[0];
        auto strideA = A.stride/sizeof(float);
        auto strideB = B.stride/sizeof(float);
        auto strideC = C.stride/sizeof(float);
        auto latALU = (M*N)*(K/8)/(2 * 4.677e9);
        auto latAVG = benchmark.tag("fc")(-10000, [&](){
            avx2::Matmul::kernel_6x16<M, Ngt8>(pA, strideA, pB, strideB, pC, strideC, K, 0, nonepp);
            //avx2::kernel_4x24<M, N>(pA, strideA, pB, strideB, pC, strideC, K, 0, nonepp);
            //avx2::kernel_14x8<M, N>(pA, strideA, pB, strideB, pC, strideC, K, 0, nonepp);
        });
        std::cout << "Proj: ALU=" << latALU * 1e6 << " us " << latAVG*100/latALU << " %" << std::endl;
        
        if (benchmark.perf_counters.size())
            std::cout << "Cycles per iteration in kernel: " << (double)benchmark.perf_counters[0]/K  << std::endl;
        return 0;
    }

    // amx_Matmul_perf_float(128, 384, 51864);
    amx_Matmul_perf_float(128, 384, 1024, -1000);

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
