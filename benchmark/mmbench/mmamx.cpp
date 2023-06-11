#include "utils.hpp"
#include "mkl.h"
#include "bf16.hpp"
#include "misc.hpp"
#include "kernels_amx.hpp"


static bool initAMX = initXTILE();

// this is a non-template class dynamic type dispatching
struct Matmul {
    enum WeightPrecision {
        Weight_BF16,
        Weight_INT8,
        Weight_INT4
    };
    amx_kernel::Matmul<ov::bfloat16, ov::bfloat16> mbf16bf16;
    amx_kernel::Matmul<ov::bfloat16, int8_t> mbf16s8;
    amx_kernel::Matmul<int8_t, int8_t> ms8s8;
    tensor2D<int8_t> compressedB;
    WeightPrecision wei_prec;
    bool transposeB;

    Matmul(bool constB = false, bool transposeB = false, WeightPrecision wei_prec = Weight_BF16) :
        mbf16bf16(constB, transposeB), mbf16s8(constB, transposeB), ms8s8(constB, transposeB), transposeB(transposeB), wei_prec(wei_prec) {
    }
    template<typename T, typename PP, typename std::enable_if<std::is_same<T, ov::bfloat16>::value || std::is_same<T, int8_t>::value, bool>::type = true>
    void operator()(tensor2D<T> & A,
                    tensor2D<T> & B,
                    PP ppkernel) {
        int N = B.dims[transposeB?0:1];
        (*this)(A, B, 0, N, ppkernel);
    }

    // ov::bfloat16 overload, wei_prec specifies whether we do internal weight-compression
    // by quantization
    template<typename PP>
    void operator()(tensor2D<ov::bfloat16> & A,
                    tensor2D<ov::bfloat16> & B,
                    int n0, int n1,
                    PP ppkernel) {
        if (wei_prec == Weight_BF16)
            mbf16bf16(A, B, n0, n1, ppkernel);
        if (wei_prec == Weight_INT8) {
            // dynamically quantize weight B matrix into int8_t before pass to
            // mbf16s8
            mbf16s8(A, B, n0, n1, ppkernel);
        }
    }

    // int8_t overload
    template<typename PP>
    void operator()(tensor2D<int8_t> & A,
                    tensor2D<int8_t> & B,
                    int n0, int n1,
                    PP ppkernel) {
        ms8s8(A, B, n0, n1, ppkernel);
    }
};



struct MatmulMTOMP {
    Matmul::WeightPrecision rt_precision;
    std::vector<std::shared_ptr<Matmul>> ops;
    bool transposeB;
    int OMP_NT;
    MatmulMTOMP(bool transposeB = false,
                bool constB = false,
                Matmul::WeightPrecision precision=Matmul::Weight_BF16) : transposeB(transposeB), rt_precision(precision) {
        OMP_NT = omp_thread_count();
        for(int i = 0; i < OMP_NT; i++)
            ops.push_back(std::make_shared<Matmul>(constB, transposeB, rt_precision));
    }

    template<typename T, typename P>
    void operator()(tensor2D<T> & matA,
                    tensor2D<T> & matB,
                    P ppkernel) {
        int M = matA.dims[0];
        int K = matA.dims[1];
        int N = matB.dims[transposeB ? 0:1];
        // split along N dimension
        int work_amount = rndup(N, 32)/32;

        auto kernel = [&](int tid, int cnt) {
            int start, end;
            splitter(work_amount, cnt, tid, start, end);
            int n0 = start*32;
            int n1 = end*32;
            if (n1 > N) n1 = N;
            //tensor2D<bfloat16> copyA = matA.clone();
            // C[:, N0:N1] = A * B[:, N0:N1]
            (*ops[tid].get())(matA, matB, n0, n1, ppkernel);
        };

        #pragma omp parallel for
        for(int i = 0; i<OMP_NT; i++) {
            kernel(i, OMP_NT);
        }
    }
};

void matmul_bf16_mmamx(
    bool transa, bool transb, bool constB,
    int64_t M, int64_t N, int64_t K,
    void *a, int64_t lda,
    void *b, int64_t ldb,
    void *c, int64_t ldc)
{
    tensor2D<ov::bfloat16> A(M, K, reinterpret_cast<ov::bfloat16*>(a), lda*sizeof(ov::bfloat16));
    tensor2D<ov::bfloat16> B(K, N, reinterpret_cast<ov::bfloat16*>(b), ldb*sizeof(ov::bfloat16));
    tensor2D<float> C(M, N, reinterpret_cast<float*>(c), ldc*sizeof(float));
    amx_kernel::PP::BiasGeluStore<float, amx_kernel::PP::Steps::NONE> pp0(C);
    MatmulMTOMP mm(transb, constB);
    mm(A, B, pp0);
}

PYBIND11_MODULE(mmamx, m)
{
    m.def("benchmark", [](bool transB, bool constB, int M, int N, int K, float duration, int cache_MB){
        return benchmark(transB, constB, M, N, K, matmul_bf16_mmamx, duration, cache_MB);
    });
}