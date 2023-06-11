#include "utils.hpp"
#include "mkl.h"
#include "bf16.hpp"
#include "misc.hpp"
#include "kernels_amx.hpp"



// this is a non-template class dynamic type dispatching
struct Matmul {
    enum WeightPrecision {
        Weight_BF16,
        Weight_INT8,
        Weight_INT4
    };
    amx_kernel::Matmul<ov::bfloat16, ov::bfloat16> mbf16bf16;
    tensor2D<int8_t> compressedB;
    WeightPrecision wei_prec;
    bool transposeB;

    Matmul(bool _constB = false, bool _transposeB = false, WeightPrecision _wei_prec = Weight_BF16) :
        mbf16bf16(_constB, _transposeB) {
        transposeB = _transposeB;
        wei_prec = _wei_prec;
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
        else
            assert(false);
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
    MatmulMTOMP() = default;

    void init(bool _transposeB = false,
                bool _constB = false,
                Matmul::WeightPrecision _precision=Matmul::Weight_BF16) {
        transposeB = _transposeB;
        rt_precision = _precision;
        OMP_NT = omp_thread_count();
        for(int i = 0; i < OMP_NT; i++)
            ops.push_back(std::make_shared<Matmul>(_constB, _transposeB, rt_precision));
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


struct MatmulTaskMMAMX : public MatmulTask {
    using MatmulTask::MatmulTask;

    MatmulMTOMP mm;

    void init() override {
        mm.init(transb, constB);
    }
    void run() override {
        amx_kernel::PP::BiasGeluStore<float, amx_kernel::PP::Steps::NONE> pp0(C);
        mm(A, B, pp0);
    }
};

PYBIND11_MODULE(mmamx, m)
{
    static bool initAMX = initXTILE();
    std::cout << "initAMX=" << initAMX << std::endl;

    m.def("benchmark", [](bool transB, bool constB, int M, int N, int K,float duration, int cache_MB){
        MatmulTaskMMAMX task(false, transB, constB, M, N, K, duration, cache_MB);
        return task.benchmark();
    });
}