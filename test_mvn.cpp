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
#include "mvn_avx2.hpp"
#include "tensor2D.hpp"
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

int test_mvn() {
    tensor2D<float> x;
    tensor2D<float> y0;
    tensor2D<float> y1;
    tensor2D<float> bias;
    tensor2D<float> scale;
    float eps = 1e-5;
    bool inside_sqrt = true;

    auto mvn_ref = [&](tensor2D<float>& x, tensor2D<float>& y, tensor2D<float>& scale, tensor2D<float>& bias) {
        float x_max = std::numeric_limits<float>::lowest();
        float sum = 0;
        auto ele_num = x.dims[1];
        for(int i = 0; i < x.dims[1]; i++) {
            sum += x[i];
        }
        float mean = sum / ele_num;
        float sum_power2 = 0;
        for(int i = 0; i < x.dims[1]; i++) {
            sum_power2 += (x[i] - mean) * (x[i] - mean);
        }
        float var = sum_power2 / ele_num;
        var = 1.0f / (inside_sqrt ? std::sqrt(var + eps) : std::sqrt(var) + eps);
        for(int i = 0; i < x.dims[1]; i++) {
            y[i] = (x[i] - mean) * var * scale[i] + bias[i];
        }
    };
    int errors = 0;
    for(int N = 1; N < 129; N++) {
        x.resize(1, N);
        x.fill_rnd();
        bias.resize(1, N);
        bias.fill_rnd();
        scale.resize(1, N);
        scale.fill_rnd();
        y0 = x.clone();
        y1 = x.clone();
        mvn_ref(x, y0, scale, bias);
        mvn_line_scale_bias(&x[0], N, eps, inside_sqrt, &y1[0], &scale[0], &bias[0]);
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
    test_mvn();

    return 0;
}
