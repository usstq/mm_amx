#include "utils.hpp"
#include "mkl.h"

struct MatmulTaskMKL : public MatmulTask {
    using MatmulTask::MatmulTask;

    tensor2D<ov::bfloat16> Bpacked;

    void init() override {
        if (constb) {
            auto sizeB = cblas_gemm_bf16bf16f32_pack_get_size(CblasBMatrix, m, n, k);

            Bpacked.resize(1, sizeB/sizeof(ov::bfloat16));
            MKL_BF16 *b_mat = reinterpret_cast<MKL_BF16 *>(&B[0]);
            MKL_BF16 *p_mat = reinterpret_cast<MKL_BF16 *>(&Bpacked[0]);
            cblas_gemm_bf16bf16f32_pack(
                    CblasRowMajor,
                    CblasBMatrix,
                    transb ? CblasTrans : CblasNoTrans,
                    m, n, k,
                    b_mat, B.padded_dim1,
                    p_mat
                    );
        }
    }

    void run() override {
        float alpha = 1.0f;
        float beta = 0.0f;

        if (constb) {
            MKL_BF16 *a_mat = reinterpret_cast<MKL_BF16 *>(&A[0]);
            MKL_BF16 *b_mat = reinterpret_cast<MKL_BF16 *>(&Bpacked[0]);
            float *c_mat = reinterpret_cast<float *>(&C[0]);
            CBLAS_TRANSPOSE transa_ = transa ? CblasTrans : CblasNoTrans;

            cblas_gemm_bf16bf16f32_compute(CblasRowMajor,
                                            transa_, CblasPacked,
                                            m, n, k,
                                            alpha, a_mat, A.padded_dim1,
                                            b_mat, B.padded_dim1,
                                            beta, c_mat, C.padded_dim1);
        } else {
            MKL_BF16 *a_mat = reinterpret_cast<MKL_BF16 *>(&A[0]);
            MKL_BF16 *b_mat = reinterpret_cast<MKL_BF16 *>(&B[0]);
            float *c_mat = reinterpret_cast<float *>(&C[0]);
            CBLAS_TRANSPOSE transa_ = transa ? CblasTrans : CblasNoTrans;
            CBLAS_TRANSPOSE transb_ = transb ? CblasTrans : CblasNoTrans;

            cblas_gemm_bf16bf16f32(CblasRowMajor,
                                    transa_, transb_,
                                    m, n, k,
                                    alpha, a_mat, A.padded_dim1,
                                    b_mat, B.padded_dim1,
                                    beta, c_mat, C.padded_dim1);
        }
    }
};

PYBIND11_MODULE(mkl, m)
{
    m.def("benchmark", [](bool transB, bool constb, int M, int N, int K,float duration, int cache_MB, bool check_correct){
        MatmulTaskMKL task("mkl", false, transB, constb, M, N, K, duration, cache_MB, check_correct);
        return task.benchmark();
    });
}
