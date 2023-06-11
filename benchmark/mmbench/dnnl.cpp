#include "utils.hpp"
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
    bool transa, bool transb, bool constB,
    int64_t m, int64_t n, int64_t k,
    void *a, int64_t lda,
    void *b, int64_t ldb,
    void *c, int64_t ldc)
{
    static MatMul_onednn mm(dnnl::memory::data_type::bf16, true);
    mm.run(transa, transb, m, n, k, a, lda, b, ldb, c, ldc);
}

void matmul_bf16_onednn_static(
    bool transa, bool transb, bool constB,
    int64_t m, int64_t n, int64_t k,
    void *a, int64_t lda,
    void *b, int64_t ldb,
    void *c, int64_t ldc)
{
    static MatMul_onednn mm(dnnl::memory::data_type::bf16, false);
    mm.run(transa, transb, m, n, k, a, lda, b, ldb, c, ldc);
}

PYBIND11_MODULE(dnnl, m)
{
    m.def("benchmark", [](bool transB, bool constB, int M, int N, int K, float duration, int cache_MB){
        return benchmark(transB, constB, M, N, K, matmul_bf16_onednn_static, duration, cache_MB);
    });
    m.def("benchmark2", [](bool transB, bool constB, int M, int N, int K, float duration, int cache_MB){
        return benchmark(transB, constB, M, N, K, matmul_bf16_onednn_dyn, duration, cache_MB);
    });
}
