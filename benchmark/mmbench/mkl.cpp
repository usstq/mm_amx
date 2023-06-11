#include "utils.hpp"
#include "mkl.h"

void matmul_bf16_mkl(
    bool transa, bool transb, bool constB,
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

PYBIND11_MODULE(mkl, m)
{
    m.def("benchmark", [](bool transB, bool constB, int M, int N, int K,float duration, int cache_MB){
        return benchmark(transB, constB, M, N, K, matmul_bf16_mkl, duration, cache_MB);
    });
}