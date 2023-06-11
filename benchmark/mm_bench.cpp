#include <iostream>
#include <chrono>
#include <pybind11/pybind11.h>
namespace py = pybind11;


//===============================================================================
#include "mkl.h"

void matmul_bf16_mkl(
    bool transa, bool transb,
    int64_t m, int64_t n, int64_t k,
    void *a, int64_t lda,
    void *b, int64_t ldb,
    void *c, int64_t ldc)
{
    float alpha = 1.0f;
    float beta = 0.0f;
    MKL_BF16 *a_mat = reinterpret_cast<MKL_BF16 *>(a);
    MKL_BF16 *b_mat = reinterpret_cast<MKL_BF16 *>(b);
    float *c_mat = reinterpret_cast<float *>(c);
    CBLAS_TRANSPOSE transa_ = transa ? CblasTrans : CblasNoTrans;
    CBLAS_TRANSPOSE transb_ = transb ? CblasTrans : CblasNoTrans;

    cblas_gemm_bf16bf16f32(CblasRowMajor,
                           transa_, transb_,
                           m, n, k,
                           alpha, a_mat, lda, b_mat, ldb,
                           beta, c_mat, ldc);
}

//===============================================================================
#include "oneapi/dnnl/dnnl.hpp"
// MatMul_onednn<dnnl::memory::data_type::bf16>
// MatMul_onednn<dnnl::memory::data_type::f32>
struct MatMul_onednn
{
    dnnl::engine eng;
    dnnl::matmul::primitive_desc matmul_pd;
    dnnl::matmul matmul_p;
    dnnl::stream stream;
    dnnl::memory::data_type dt;

    // whether use dynamic shape support
    const bool dyn_mode;

    MatMul_onednn(dnnl::memory::data_type dt, bool dyn_mode) : eng(dnnl::engine::kind::cpu, 0),
                                                               stream(eng),
                                                               dt(dt),
                                                               dyn_mode(dyn_mode)
    {
        if (dyn_mode)
            init_dynamic_primitive();
    }

    void init_dynamic_primitive()
    {
        dnnl::memory::dims a_shape = {DNNL_RUNTIME_DIM_VAL, DNNL_RUNTIME_DIM_VAL};
        dnnl::memory::dims b_shape = {DNNL_RUNTIME_DIM_VAL, DNNL_RUNTIME_DIM_VAL};
        dnnl::memory::dims c_shape = {DNNL_RUNTIME_DIM_VAL, DNNL_RUNTIME_DIM_VAL};

        // we don't know if A,B is transposed, so both dimensions are dynamic
        dnnl::memory::dims a_strides_ = {DNNL_RUNTIME_DIM_VAL, DNNL_RUNTIME_DIM_VAL};
        dnnl::memory::dims b_strides_ = {DNNL_RUNTIME_DIM_VAL, DNNL_RUNTIME_DIM_VAL};

        // we known the row-major C matrix is not transposed, so inner strides is 1
        dnnl::memory::dims c_strides_ = {DNNL_RUNTIME_DIM_VAL, 1};

        dnnl::memory::desc a_md(a_shape, dt, a_strides_);
        dnnl::memory::desc b_md(b_shape, dt, b_strides_);
        dnnl::memory::desc c_md(c_shape, dnnl::memory::data_type::f32, c_strides_);

        // Create attributes (to handle alpha dynamically and beta if necessary)
        dnnl::primitive_attr attr;
        attr.set_scales_mask(DNNL_ARG_WEIGHTS, /* mask */ 0);

        // Create a MatMul primitive
        matmul_pd = dnnl::matmul::primitive_desc(eng, a_md, b_md, c_md, attr);
        matmul_p = dnnl::matmul(matmul_pd);
    }

    bool m_transa = false;
    bool m_transb = false;
    int64_t m_M = 0;
    int64_t m_N = 0;
    int64_t m_K = 0;
    int64_t m_lda = 0;
    int64_t m_ldb = 0;
    int64_t m_ldc = 0;
    std::unordered_map<int, dnnl::memory> m_args;

    // static shape
    void update_static_shape(bool transa, bool transb,
                             int64_t M, int64_t N, int64_t K,
                             void *A, int64_t lda,
                             void *B, int64_t ldb,
                             void *C, int64_t ldc)
    {
        if (m_transa == transa && m_transb == transb &&
            m_M == M && m_N == N && m_K == K &&
            m_lda == lda && m_ldb == ldb && m_ldc == ldc)
        {
            // no shape/strides config changes, just reset pointer
            m_args[DNNL_ARG_SRC].set_data_handle(A);
            m_args[DNNL_ARG_WEIGHTS].set_data_handle(B);
            m_args[DNNL_ARG_DST].set_data_handle(C);
            return;
        }
        dnnl::memory::dims a_shape = {M, K};
        dnnl::memory::dims b_shape = {K, N};
        dnnl::memory::dims c_shape = {M, N};

        dnnl::memory::dims a_strides = (!transa) ? dnnl::memory::dims{lda, 1} : dnnl::memory::dims{1, lda};
        dnnl::memory::dims b_strides = (!transb) ? dnnl::memory::dims{ldb, 1} : dnnl::memory::dims{1, ldb};
        dnnl::memory::dims c_strides = dnnl::memory::dims{ldc, 1};

        dnnl::memory::desc a_md(a_shape, dt, a_strides);
        dnnl::memory::desc b_md(b_shape, dt, b_strides);
        dnnl::memory::desc c_md(c_shape, dnnl::memory::data_type::f32, c_strides);

        // Prepare oneDNN memory for alpha
        // memory alpha_m({{1}, memory::data_type::f32, {1}}, eng, &alpha);

        // Create attributes (to handle alpha dynamically and beta if necessary)
        dnnl::primitive_attr attr;
        //attr.set_scales_mask(DNNL_ARG_WEIGHTS, /* mask */ 0);

        if (!dyn_mode)
        {
            // Create static shape MatMul primitive
            matmul_pd = dnnl::matmul::primitive_desc(eng, a_md, b_md, c_md, attr);
            matmul_p = dnnl::matmul(matmul_pd);
        }
        else
        {
            // dynamic mode MatMul primitive only create once
        }

        std::cout << matmul_pd.impl_info_str() << std::endl;

        dnnl::memory A_m(a_md, eng, A);
        dnnl::memory B_m(b_md, eng, B);
        dnnl::memory C_m(c_md, eng, C);

        m_args.clear();
        m_args[DNNL_ARG_SRC] = A_m;
        m_args[DNNL_ARG_WEIGHTS] = B_m;
        m_args[DNNL_ARG_DST] = C_m;
        // m_args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS] = alpha_m;

        m_transa = transa;
        m_transb = transb;
        m_M = M;
        m_N = N;
        m_K = K;
        m_lda = lda;
        m_ldb = ldb;
        m_ldc = ldc;
    }

    void run(bool transa, bool transb,
             int64_t M, int64_t N, int64_t K,
             void *A, int64_t lda,
             void *B, int64_t ldb,
             void *C, int64_t ldc)
    {
        // update static shape related resources:
        //   - primitive (in static shape mode)
        //   - runtime args for primitive
        update_static_shape(transa, transb, M, N, K, A, lda, B, ldb, C, ldc);

        // Execute the MatMul primitive
        matmul_p.execute(stream, m_args);
        stream.wait();
    }
};

void matmul_bf16_onednn_dyn(
    bool transa, bool transb,
    int64_t m, int64_t n, int64_t k,
    void *a, int64_t lda,
    void *b, int64_t ldb,
    void *c, int64_t ldc)
{
    static MatMul_onednn mm(dnnl::memory::data_type::bf16, true);
    mm.run(transa, transb, m, n, k, a, lda, b, ldb, c, ldc);
}

void matmul_bf16_onednn_static(
    bool transa, bool transb,
    int64_t m, int64_t n, int64_t k,
    void *a, int64_t lda,
    void *b, int64_t ldb,
    void *c, int64_t ldc)
{
    static MatMul_onednn mm(dnnl::memory::data_type::bf16, false);
    mm.run(transa, transb, m, n, k, a, lda, b, ldb, c, ldc);
}

//===============================================================================
#include "misc.hpp"
#include "tensor2D.hpp"
#include <cstdlib>


// _rdpmc
#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif


uint64_t rdtsc_calibrate(int seconds = 1) {
    uint64_t start_ticks;
    start_ticks = __rdtsc();
    std::this_thread::sleep_for(std::chrono::seconds(seconds));
    return (__rdtsc() - start_ticks) / seconds;
}
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

static float g_duration = 5.0;
static int g_cache_MB = 120;

template<typename F>
py::dict benchmark(int M, int N, int K,
                   F kernel) {
    tensor2D<ov::bfloat16> A(M, K);
    tensor2D<ov::bfloat16> B(K, N);
    tensor2D<float> C(M, N);
    tensor2D<float> C0(M, N);

    std::vector<char> clr_cache_src(g_cache_MB*1024*1024, 1);
    std::vector<char> clr_cache_dst(g_cache_MB*1024*1024, 2);
    
    py::dict ret;

    C0=0;
    matmul(A, B, C0);

    auto clear_cache = [&](){
        memcpy(&clr_cache_dst[0], &clr_cache_src[0], g_cache_MB*1024*1024);
        return clr_cache_dst[rand() % (g_cache_MB*1024*1024)];
    };

    const int warm_up = 2;
    for(int i = 0; i < warm_up; i++) {
        clear_cache();
        kernel(false, false, M, N, K,
                &A[0], A.padded_dim1,
                &B[0], B.padded_dim1,
                &C[0], C.padded_dim1);
    }

    // roughly measure latency
    auto t0 = __rdtsc();
    clear_cache();
    kernel(false, false, M, N, K,
            &A[0], A.padded_dim1,
            &B[0], B.padded_dim1,
            &C[0], C.padded_dim1);
    auto t1 = __rdtsc();

    auto est_latency = tsc2second(t1 - t0);

    double avg_latency = 0;
    int64_t times = g_duration/est_latency;
    std::cout << " start test times=" << times << std::flush;
    auto start = std::chrono::high_resolution_clock::now();
    for(int64_t i = 0; i < times; i++) {
        clear_cache();
        auto t0 = __rdtsc();
        kernel(false, false, M, N, K,
                &A[0], A.padded_dim1,
                &B[0], B.padded_dim1,
                &C[0], C.padded_dim1);
        auto t1 = __rdtsc();
        avg_latency += tsc2second(t1 - t0);
    }
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_latency = finish-start;
    std::cout << " finished in " << total_latency.count() << " seconds" << std::endl;

    avg_latency = avg_latency / times;

    ret[pybind11::str("correct")] = bool(C == C0);
    ret[pybind11::str("latency_ms")] = avg_latency * 1e3;
    ret[pybind11::str("times")] = times;
    ret[pybind11::str("duration")] = total_latency.count();

    return ret;
}

int add(int i, int j)
{
    return i + j;
}


PYBIND11_MODULE(mm_bench, m)
{

    m.def("add", &add, R"pbdoc(
        Add two numbers

        Some other explanation about the add function.
    )pbdoc");

    m.def(
        "subtract", [](int i, int j)
        { return i - j; },
        R"pbdoc(
        Subtract two numbers

        Some other explanation about the subtract function.
    )pbdoc");
    m.attr("duration") = pybind11::float_(g_duration);
    m.attr("cache_MB") = pybind11::int_(g_cache_MB);

    m.def("benchmark_mkl", [](int M, int N, int K){
        return benchmark(M, N, K, matmul_bf16_mkl);
    });
    m.def("benchmark_dnnl", [](int M, int N, int K){
        return benchmark(M, N, K, matmul_bf16_onednn_static);
    });
    m.def("benchmark_dnnl2", [](int M, int N, int K){
        return benchmark(M, N, K, matmul_bf16_onednn_dyn);
    });
}
