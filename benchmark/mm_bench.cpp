#include <pybind11/pybind11.h>

//===============================================================================
#include "mkl.h"

void matmul_bf16_mkl(
    bool transa, bool transb,
    int64_t m, int64_t n, int64_t k,
    void * a, int64_t lda,
    void * b, int64_t ldb,
    void * c, int64_t ldc
) {
    float alpha = 1.0f;
    float beta = 0.0f;
    MKL_BF16 * a_mat = reinterpret_cast<MKL_BF16 *>(a);
    MKL_BF16 * b_mat = reinterpret_cast<MKL_BF16 *>(b);
    float * c_mat = reinterpret_cast<float *>(c);
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
struct MatMul_bf16_onednn {
    dnnl::engine eng;
    dnnl::matmul::primitive_desc matmul_pd;
    dnnl::matmul matmul_p;

    MatMul_bf16_onednn() :
        eng(dnnl::engine::kind::cpu, 0) {
        dnnl::memory::dims a_shape = {DNNL_RUNTIME_DIM_VAL, DNNL_RUNTIME_DIM_VAL};
        dnnl::memory::dims b_shape = {DNNL_RUNTIME_DIM_VAL, DNNL_RUNTIME_DIM_VAL};
        dnnl::memory::dims c_shape = {DNNL_RUNTIME_DIM_VAL, DNNL_RUNTIME_DIM_VAL};

        dnnl::memory::dims a_strides_ = {DNNL_RUNTIME_DIM_VAL, DNNL_RUNTIME_DIM_VAL};
        dnnl::memory::dims b_strides_ = {DNNL_RUNTIME_DIM_VAL, DNNL_RUNTIME_DIM_VAL};
        dnnl::memory::dims c_strides_ = {DNNL_RUNTIME_DIM_VAL, 1};

        dnnl::memory::desc a_md(a_shape, dnnl::memory::data_type::f32, a_strides_);
        dnnl::memory::desc b_md(b_shape, dnnl::memory::data_type::f32, b_strides_);
        dnnl::memory::desc c_md(c_shape, dnnl::memory::data_type::f32, c_strides_);

        // Create attributes (to handle alpha dynamically and beta if necessary)
        dnnl::primitive_attr attr;
        attr.set_scales_mask(DNNL_ARG_WEIGHTS, /* mask */ 0);

        // Create a MatMul primitive
        matmul_pd = dnnl::matmul::primitive_desc(eng, a_md, b_md, c_md, attr);
        matmul_p = dnnl::matmul(matmul_pd);
    }

    void run(
        bool transa, bool transb,
        int64_t M, int64_t N, int64_t K,
        void * A, int64_t lda,
        void * B, int64_t ldb,
        void * C, int64_t ldc) {
        // Translate transA and transB
        dnnl::memory::dims a_strides = (!transa) ? dnnl::memory::dims{lda, 1} : dnnl::memory::dims{1, lda};
        dnnl::memory::dims b_strides = (!transb) ? dnnl::memory::dims{ldb, 1} : dnnl::memory::dims{1, ldb};

        // Wrap raw pointers into oneDNN memories (with proper shapes)
        dnnl::memory A_m({{M, K}, dnnl::memory::data_type::bf16, a_strides}, eng, (void *)A);
        dnnl::memory B_m({{K, N}, dnnl::memory::data_type::bf16, b_strides}, eng, (void *)B);
        dnnl::memory C_m({{M, N}, dnnl::memory::data_type::f32, {ldc, 1}}, eng, (void *)C);

        // Prepare oneDNN memory for alpha
        //memory alpha_m({{1}, memory::data_type::f32, {1}}, eng, &alpha);

        // Execute the MatMul primitive
        dnnl::stream s(eng);
        matmul_p.execute(s,
                        {
                            {DNNL_ARG_SRC, A_m},
                            {DNNL_ARG_WEIGHTS, B_m},
                            {DNNL_ARG_DST, C_m},
                            //{DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, alpha_m}
                        });
        s.wait();
    }
};

//===============================================================================
int add(int i, int j) {
    return i + j;
}

namespace py = pybind11;

PYBIND11_MODULE(python_example, m) {

    m.def("add", &add, R"pbdoc(
        Add two numbers

        Some other explanation about the add function.
    )pbdoc");

    m.def("subtract", [](int i, int j) { return i - j; }, R"pbdoc(
        Subtract two numbers

        Some other explanation about the subtract function.
    )pbdoc");
}