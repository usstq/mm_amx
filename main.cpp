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

#include "kernels_amx.hpp"
#include "kernels_avx512.hpp"
#include "thread_pool.hpp"
#include "timeit.hpp"
#include "misc.hpp"
#include "test_bw.hpp"

#include "thread_pool.hpp"
#include <omp.h>
timeit timer;

//================================================================================
// initialize AMX
static bool initAMX = initXTILE();

struct Matmul {
    enum WeightPrecision {
        Weight_BF16,
        Weight_INT8,
        Weight_INT4
    };
    amx::Matmul<bfloat16, bfloat16> mbf16bf16;
    amx::Matmul<bfloat16, int8_t> mbf16s8;
    amx::Matmul<int8_t, int8_t> ms8s8;
    tensor2D<int8_t> compressedB;
    WeightPrecision wei_prec;
    bool transposeB;

    Matmul(bool constB = false, bool transposeB = false, WeightPrecision wei_prec = Weight_BF16) :
        mbf16bf16(constB, transposeB), mbf16s8(constB, transposeB), ms8s8(constB, transposeB), transposeB(transposeB), wei_prec(wei_prec) {
    }
    template<typename T, typename PP, std::enable_if_t<std::is_same<T, bfloat16>::value || std::is_same<T, int8_t>::value, bool> = true>
    void operator()(tensor2D<T> & A,
                    tensor2D<T> & B,
                    PP ppkernel) {
        int N = B.dims[transposeB?0:1];
        (*this)(A, B, 0, N, ppkernel);
    }

    // bfloat16 overload
    template<typename PP>
    void operator()(tensor2D<bfloat16> & A,
                    tensor2D<bfloat16> & B,
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

std::ostream & operator<<(std::ostream & os, Matmul::WeightPrecision & prec) {
    static const char* names_prec[] = {
    "bf16",
    "int8",
    "int4"
    };
    os << names_prec[(int)prec];
    return os;
}

static Matmul::WeightPrecision precision = Matmul::Weight_BF16;

//================================================================================
int amx_unit_test_perf() {
    int M = 32;
    // K=12*32, A+B fits L1D, gives 100% HW usage
    // K=80*32  A+B fits L2, gives 70% HW usage
    // K=512*32 A+B fits L2, gives 30% HW usage
    int K = 80*32;
    int N = 32;
    tensor2D<bfloat16> A(M, K);
    tensor2D<bfloat16> BT(N, K);
    tensor2D<bfloat16> C(M, N);
    tileconfig_t tfg(1, 0, 8, 16, 64);

    std::cout << "A & B in L1D (should give theoratical peak Gflops)\n\t";
    timer(-100,[&](){
        const int C00 = 0, C01 = 1, C10 = 2, C11 = 3, A0 = 4, A1 = 5, B0 = 6, B1 = 7;
        auto * pA0 = &A(0,0);
        auto * pA1 = &A(16,0);
        auto * pB0 = &BT(0,0);
        auto * pB1 = &BT(16,0);
        _tile_zero(C00);
        _tile_zero(C01);
        _tile_zero(C10);
        _tile_zero(C11);
        for(int k = 0; k < K; k+=32) {
            _tile_loadd(A0, pA0, 64);
            _tile_loadd(B0, pB0, 64);
            _tile_dpbf16ps(C00, A0, B0);
            _tile_loadd(A1, pA1, 64);
            _tile_dpbf16ps(C10, A1, B0);
            _tile_loadd(B1, pB1, 64);
            _tile_dpbf16ps(C01, A0, B1);
            _tile_dpbf16ps(C11, A1, B1);
        }
    },
    (M * N) * K * 2,
    AMXBf16PeakGops2PerCore * 1e9
    );

    std::cout << "TileGemmKernel32x32:\n\t";
    timer(-100,[&](){
        const int C00 = 0, C01 = 1, C10 = 2, C11 = 3, A0 = 4, A1 = 5, B0 = 6, B1 = 7;
        _tile_zero(C00);
        _tile_zero(C01);
        _tile_zero(C10);
        _tile_zero(C11);
        for (int k=0; k < K; k+=32) {
            _tile_loadd(A0, &A(0, k), A.stride);
            _tile_loadd(B0, &BT(0, k), BT.stride);
            _tile_dpbf16ps(C00, A0, B0);
            _tile_loadd(A1, &A(16, k), A.stride);
            _tile_dpbf16ps(C10, A1, B0);
            _tile_loadd(B1, &BT(16, k), BT.stride);
            _tile_dpbf16ps(C01, A0, B1);
            _tile_dpbf16ps(C11, A1, B1);
        }
    },
    (M * N) * K * 2,
    AMXBf16PeakGops2PerCore * 1e9
    );

    std::cout << "B is transposed and blocked:\n\t";
    timer(-100,[&](){
        const int C00 = 0, C01 = 1, C10 = 2, C11 = 3, A0 = 4, A1 = 5, B0 = 6, B1 = 7;
        _tile_zero(C00);
        _tile_zero(C01);
        _tile_zero(C10);
        _tile_zero(C11);
        auto *pB = &BT(0, 0);
        for (int k=0; k < K; k+=32) {
            _tile_stream_loadd(A0, &A(0, k), A.stride);
            _tile_stream_loadd(B0, pB, 64);
            _tile_dpbf16ps(C00, A0, B0);
            _tile_stream_loadd(A1, &A(16, k), A.stride);
            _tile_dpbf16ps(C10, A1, B0);
            _tile_stream_loadd(B1, pB + (16*32), 64);
            _tile_dpbf16ps(C01, A0, B1);
            _tile_dpbf16ps(C11, A1, B1);
            pB += 32*32;
        }
    },
    (M * N) * K * 2,
    AMXBf16PeakGops2PerCore * 1e9
    );

    std::cout << "B is transposed and blocked; A is blocked:\n\t";
    timer(-100,[&](){
        const int C00 = 0, C01 = 1, C10 = 2, C11 = 3, A0 = 4, A1 = 5, B0 = 6, B1 = 7;
        _tile_zero(C00);
        _tile_zero(C01);
        _tile_zero(C10);
        _tile_zero(C11);
        auto *pA = &A(0, 0);
        auto *pB = &BT(0, 0);
        for (int k=0; k < K; k+=32) {
            _tile_loadd(B0, pB + k*(32), 64);
            _tile_loadd(A0, pA + k*(32), 64);
            _tile_dpbf16ps(C00, A0, B0);
            _tile_loadd(A1, pA + k*(32) + (16*32), 64);
            _tile_dpbf16ps(C10, A1, B0);
            _tile_loadd(B1, pB + k*(32) + (16*32), 64);
            _tile_dpbf16ps(C01, A0, B1);
            _tile_dpbf16ps(C11, A1, B1);
        }
    },
    (M * N) * K * 2,
    AMXBf16PeakGops2PerCore * 1e9
    );


    // now we go through real memory area, but assume the layout has been
    // optimized for performance.
    std::cout << "A & B are blocked and sequentially loaded:\n\t";
    tensor2D<bfloat16> AB(M*K + K*N, 32);
    timer(-100,[&](){
        const int C00 = 0, C01 = 1, C10 = 2, C11 = 3, A0 = 4, A1 = 5, B0 = 6, B1 = 7;
        _tile_zero(C00);
        _tile_zero(C01);
        _tile_zero(C10);
        _tile_zero(C11);
        auto *ptr = &AB(0, 0);
        for (int k=0; k < K; k+=32) {
            _tile_stream_loadd(B0, ptr, 64);
            _tile_stream_loadd(A0, ptr + (16*32), 64);
            _tile_dpbf16ps(C00, A0, B0);
            _tile_stream_loadd(A1, ptr + (2*16*32), 64);
            _tile_dpbf16ps(C10, A1, B0);
            _tile_stream_loadd(B1, ptr + (3*16*32), 64);
            _tile_dpbf16ps(C01, A0, B1);
            _tile_dpbf16ps(C11, A1, B1);
            ptr += (4*16*32);
        }
    },
    (M * N) * K * 2,
    AMXBf16PeakGops2PerCore * 1e9
    );

    std::cout << C(0,0) << std::endl;
    return 0;
}

template<typename T, amx::PP::Steps ppsteps>
void amx_FC_acc(int M, int K, int N) {
    tensor2D<T> A(M, K);
    tensor2D<T> B(K, N);
    tensor2D<T> BT = B.Tr();
    tensor2D<bfloat16> C(M, N);
    tensor2D<bfloat16> C0(M, N);
    Matmul fc(true, false, precision);
    Matmul fcTr(true, true, precision);
    amx::PP::BiasGeluStore<bfloat16, ppsteps> pp(C);
    std::stringstream ss;
    fc(A, B, pp);
    C0=0;
    matmul(A, B, C0);
    std::cout << __func__ << " [" << M << "," << K << "," << N << "," << TypeName<T>::get() << "(" << precision << ")] ";
    if (C0 == C) {
        ss << ANSIcolor("1;32") << "no_trans: Match!     " << ANSIcolor();
    } else {
        ss << ANSIcolor("1;31") << "no_trans: Mismatch!  " << ANSIcolor();
        //std::cout << C0 << std::endl;
        //std::cout << C << std::endl;
    }
    C = 0;
    fcTr(A, BT, pp);
    if (C0 == C) {
        ss << ANSIcolor("1;32") << "trans:  Match!" << ANSIcolor();
    } else {
        ss << ANSIcolor("1;31") << "trans:  Mismatch!" << ANSIcolor();
        //std::cout << C0 << std::endl;
        //std::cout << C << std::endl;
    }
    std::cout << ss.str() << std::endl;
}

template<typename T, amx::PP::Steps ppsteps>
void amx_FC_perf(int M, int K, int N, int times = -1000) {
    tensor2D<T> A(M, K);
    tensor2D<T> B(K, N);
    tensor2D<bfloat16> C(M, N);
    Matmul mm(true, false, precision);
    amx::PP::BiasGeluStore<bfloat16, ppsteps> pp(C);
    timer.tag(__func__, M, K, N, TypeName<T>::get(), precision)(times, [&](){
        mm(A, B, pp);
    },
    double(M * N) * K * 2,
    AMXBf16PeakGops2PerCore * 1e9);
}

void amx_Matmul_perf(int M, int K, int N, bool transB, int times = -1000) {
    tensor2D<bfloat16> A(M, K);
    tensor2D<bfloat16> B(K, N);
    tensor2D<bfloat16> BT = B.Tr();
    tensor2D<bfloat16> C(M, N);
    tensor2D<bfloat16> C0(M, N);
    Matmul mm(false, transB);
    amx::PP::BiasGeluStore<bfloat16, amx::PP::Steps::NONE> pp(C);
    std::cout << __func__ << " [" << M << "," << K << "," << N << "] ";

    C0=0;
    matmul(A, B, C0);
    mm(A, transB?BT:B, pp);
    if (C0 == C) {
        std::cout << ANSIcolor("1;32") << "Match!\n" << ANSIcolor();
        //std::cout << C << std::endl;
    } else {
        std::cout << ANSIcolor("1;31") << "Mismatch!\n" << ANSIcolor();
        std::cout << C0 << std::endl;
        std::cout << C << std::endl;
    }
    std::cout << C0 << std::endl;
    std::cout << C << std::endl;
    timer(times, [&](){
        mm(A, transB?BT:B, pp);
    },
    double(M * N) * K * 2,
    AMXBf16PeakGops2PerCore * 1e9);
}

void amx_unit_test_gemAvB(int M, int K, int times = -1000) {
    int N = 1;
    tensor2D<bfloat16> A(M, K);
    tensor2D<bfloat16> B(1, K);
    tensor2D<bfloat16> B0(K, 1);
    tensor2D<bfloat16> C0(M, 1);
    tensor2D<bfloat16> C(1, M);
    amx::GemAvB gemAvB;

    // same B, different layout
    std::cout << __func__ << " [" << M << "," << K << "," << N << "] ";
    B0 = B;
    C0 = 0;
    matmul(A, B0, C0);
    auto C0Tr = C0.Tr();
    gemAvB(A, &B(0,0), &C(0,0));
    if (C0Tr == C) {
        std::cout << ANSIcolor("1;32") << "Match!\n" << ANSIcolor();
    } else {
        std::cout << C0Tr << std::endl;
        std::cout << C << std::endl;
        std::cout << ANSIcolor("1;31") << "Mismatch!\n" << ANSIcolor();
        return;
    }

    std::cout << __func__ << " [" << M << "," << K << "," << N << "] ";
    timer(times, [&](){
        gemAvB(A, &B(0,0), &C(0,0));
    },
    double(M * N) * K * 2,
    256 * 3e9);
}

void test_blk_loops() {
    int max = 9999;
    BlockIterator loc;
    BlockIterator::blkloop bloops[] = {
        {10,32,0},{max,0,32},{max,320,0}
    };
    //loc.reset(bloops, 896,10240);
    //do {
    //    std::cout << "[" << loc.seq << "]   " << loc.m << "," << loc.n
    //              << "  idx =  " << loc.idx[0] << "," << loc.idx[1] << "," << loc.idx[2] << std::endl;
    //}while(loc.next());

    loc.reset(bloops, 3, 896, 10240);
    do {
    }while(loc.next());
    std::cout << loc.seq << std::endl;
    
    std::cout << __func__;
    timer(-1000, [&](){
        loc.reset(bloops, 3, 10240, 10240);
        do {
        }while(loc.next());
    });
}

#if 0

// ThreadPool has much lower performance than OMP

ThreadPool thp;

// multi-threaded matmul
struct MatmulMT {
    Matmul::WeightPrecision rt_precision;
    std::vector<std::shared_ptr<Matmul>> ops;
    bool transposeB;
    MatmulMT(bool constB = false,
             bool transposeB = false,
             Matmul::WeightPrecision precision=Matmul::Weight_BF16) : transposeB(transposeB), rt_precision(precision) {
        for(int i = 0; i < thp.num_threads; i++)
            ops.push_back(std::make_shared<Matmul>(constB, transposeB));
    }

    template<typename P>
    void operator()(tensor2D<bfloat16> & matA,
                    tensor2D<bfloat16> & matB,
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
            tensor2D<bfloat16> copyA = matA.clone();
            // C[:, N0:N1] = A * B[:, N0:N1]
            (*ops[tid].get())(copyA, matB, n0, n1, ppkernel);
        };
        thp.Paralell_NT(kernel);
    }
};
#endif

int OMP_NT = omp_thread_count();

struct MatmulMTOMP {
    Matmul::WeightPrecision rt_precision;
    std::vector<std::shared_ptr<Matmul>> ops;
    bool transposeB;
    MatmulMTOMP(bool constB = false,
                bool transposeB = false,
                Matmul::WeightPrecision precision=Matmul::Weight_BF16) : transposeB(transposeB), rt_precision(precision) {
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

template<typename T, amx::PP::Steps steps>
void amx_MatmulMT_perf(int M, int K, int N, bool transB, int times = -1000) {
    tensor2D<T> A(M, K);
    tensor2D<T> B(K, N);
    tensor2D<T> BT = B.Tr();
    tensor2D<T> C(M, N);
    tensor2D<T> C0(M, N);
    Matmul mm(true, transB, precision);
    MatmulMTOMP      mmMT(true, transB, precision);
    tensor2D<float> Bias0(1, N);
    amx::PP::BiasGeluStore<T, steps> pp0(C0, &Bias0(0,0));
    amx::PP::BiasGeluStore<T, steps> pp(C, &Bias0(0,0));

    if (steps & amx::PP::Steps::DEQUANT) {
        pp0.set_deq_scale(0.25f);
        pp.set_deq_scale(0.25f);
    }
    if (steps & amx::PP::Steps::QUANT) {
        pp0.set_q_scale(4.0f);
        pp.set_q_scale(4.0f);
    }
    timer.tag(__func__, "ST", M, K, N, TypeName<T>::get(), precision)(times, [&](){
        mm(A, transB?BT:B, pp0);
    },
    double(M * N) * K * 2,
    AMXBf16PeakGops2PerCore * 1e9);

    // only test perf on MultiThread case when
    // N is big enough to be divided into OMP_NT cores
    if (N >= 32*OMP_NT) {
        timer.tag(__func__, "MT", M, K, N, TypeName<T>::get(), precision)(times, [&](){
            mmMT(A, transB?BT:B, pp);
        },
        double(M * N) * K * 2,
        AMXBf16PeakGops2PerCore * 1e9);

        if (C0 == C) {
            std::cout << ANSIcolor("1;32") << "Match!\n" << ANSIcolor();
            //std::cout << C << std::endl;
        } else {
            std::cout << ANSIcolor("1;31") << "Mismatch!\n" << ANSIcolor();
            std::cout << C0 << std::endl;
            std::cout << C << std::endl;
            return;
        }
    }
}

void amx_MatmulMT_BiasGelu_acc(int M, int K, int N, bool transB) {
    tensor2D<bfloat16> A(M, K);
    tensor2D<bfloat16> B(K, N);
    tensor2D<bfloat16> BT = B.Tr();
    tensor2D<bfloat16> C(M, N);
    tensor2D<bfloat16> C0(M, N);
    tensor2D<float> Bias(1, N);
    Matmul mm(true, transB);
    amx::PP::BiasGeluStore<bfloat16, amx::PP::Steps::BIAS_GELU> pp0(C, &Bias(0,0));

    std::cout << __func__ << " [" << M << "," << K << "," << N << "] ";
    C0 = 0;
    matmul(A, B, C0, &Bias(0,0), [](float x){
        return x*0.5*(1 + std::erf(x/std::sqrt(2)));
    });
    {
        mm(A, transB?BT:B, pp0);
    }

    if (C0.compare(C, 0.001f)) {
        std::cout << ANSIcolor("1;32") << "Match!\n" << ANSIcolor();
    } else {
        std::cout << C0 << std::endl;
        std::cout << C << std::endl;
        std::cout << ANSIcolor("1;31") << "Mismatch!\n" << ANSIcolor();
    }
}

void amx_MatmulMT_BiasGelu_perf(int M, int K, int N, bool transB, int times = -1000) {
    tensor2D<bfloat16> A(M, K);
    tensor2D<bfloat16> B(K, N);
    tensor2D<bfloat16> BT = B.Tr();
    tensor2D<bfloat16> C(M, N);
    tensor2D<bfloat16> C0(M, N);
    tensor2D<float> Bias(1, N);
    Matmul mm(true, transB);
    MatmulMTOMP      mmMT(true, transB);
    amx::PP::BiasGeluStore<bfloat16, amx::PP::Steps::BIAS_GELU> pp0(C0, &Bias(0,0));
    amx::PP::BiasGeluStore<bfloat16, amx::PP::Steps::BIAS_GELU> pp(C, &Bias(0,0));

    std::cout << __func__ << " [" << M << "," << K << "," << N << "] ";

    timer(times, [&](){
        mm(A, transB?BT:B, pp0);
    },
    double(M * N) * K * 2,
    AMXBf16PeakGops2PerCore * 1e9);

    timer(times, [&](){
        mmMT(A, transB?BT:B, pp);
    },
    double(M * N) * K * 2,
    AMXBf16PeakGops2PerCore * 1e9);
    if (C0 == C) {
        std::cout << ANSIcolor("1;32") << "Match!\n" << ANSIcolor();
        //std::cout << C << std::endl;
    } else {
        std::cout << ANSIcolor("1;31") << "Mismatch!\n" << ANSIcolor();
        std::cout << C0 << std::endl;
        std::cout << C << std::endl;
        return;
    }
}



void amx_Matmul_perf_float(int M, int K, int N, int times = -1000) {
    tensor2D<float> A(M, K);
    tensor2D<float> B(K, N);
    tensor2D<float> C(M, N);
    tensor2D<float> C0(M, N);
    tensor2D<float> Bias(1, N);
    avx512::Matmul mm;
    avx512::PP::AddbiasRelu pp(&Bias(0,0));
    std::cout << __func__ << " [" << M << "," << K << "," << N << "] ";

    C0=0;
    matmul(A, B, C0, &Bias(0,0), [](float x){
        return std::max(x, 0.0f);
    });
    mm(A, B, C, pp);
    if (C0 == C) {
        std::cout << ANSIcolor("1;32") << "Match!\n" << ANSIcolor();
        //std::cout << C << std::endl;
    } else {
        std::cout << ANSIcolor("1;31") << "Mismatch!\n" << ANSIcolor();
        std::cout << C0 << std::endl;
        std::cout << C << std::endl;
    }

    timer(times, [&](){
        mm(A, B, C, pp);
    },
    double(M * N) * K * 2,
    FP32PeakGopsPerCore * 1e9);
}

void test_bf16() {
    for(int i=0;i<65536;i++) {
        auto bf16i = bfloat16(i);
        auto bf16i2i = static_cast<int>(static_cast<float>(bf16i));
        if (bf16i2i != i) {
            std::cout << "bfloat16 cannot represent int " << i << std::endl;
            break;
        }
    }
    {
        auto a = bfloat16(std::nan("1"));
        auto b = bfloat16(0.0f);
        auto c = a*b;
        std::cout << c << std::endl;
    }
    {
        tensor2D<bfloat16> A(16, 32);
        tensor2D<bfloat16> BT(16, 32);
        tensor2D<bfloat16> C(16, 16);
        tileconfig_t tfg(1, 0, 8, 16, 64);
        const int tileA = 0;
        const int tileB = 1;
        const int tileC = 2;
        A = bfloat16(std::nan("0"));
        BT = bfloat16(0.0f);
        _tile_loadd(tileA, &A(0,0), 64);
        _tile_loadd(tileB, &BT(0,0), 64);
        _tile_dpbf16ps(tileC, tileA, tileB);
        tshow<float, 2>();
    }

}

/*
repeate following topology 
   [M,K][K,N] => [M,N]   A1*B1=>A2
   [M,N][N,K] => [M,K]   A2*B2=>A1
   ...

*/
template<typename T, amx::PP::Steps ppsteps>
void amx_FC_MTML_perf(int M, int K, int N, int repeates, int times = -1000) {
    tensor2D<T> A1(M, K);
    tensor2D<T> A2(M, N);

    std::vector<tensor2D<T>> B1s;
    std::vector<tensor2D<T>> B2s;
    std::vector<tensor2D<float>> biasA1;
    std::vector<tensor2D<float>> biasA2;
    std::vector<MatmulMTOMP> FC1;
    std::vector<MatmulMTOMP> FC2;
    //MatmulMTOMP               mmMT(true, false, precision);

    for(int i = 0; i<repeates; i++) {
        B1s.emplace_back(K, N);
        B2s.emplace_back(N, K);
        biasA1.emplace_back(1, K);
        biasA2.emplace_back(1, N);
        // MatmulMTOMP internally will cache B matrix, so we need
        // multiple instances, one for each FC layer.
        FC1.emplace_back(true, false, precision);
        FC2.emplace_back(true, false, precision);
    }

    double elesize = (precision == Matmul::Weight_BF16)? sizeof(bfloat16) : sizeof(int8_t);

    timer.tag(__func__, M, K, N, precision, repeates)(times, [&](){
        for(int i = 0; i<repeates; i++) {
            amx::PP::BiasGeluStore<T, ppsteps> ppToA2(A2, &biasA2[i](0,0));
            amx::PP::BiasGeluStore<T, ppsteps> ppToA1(A1, &biasA1[i](0,0));
            if (ppsteps & amx::PP::Steps::DEQUANT) {
                ppToA2.set_deq_scale(0.2);
                ppToA1.set_deq_scale(0.2);
            }
            if (ppsteps & amx::PP::Steps::QUANT) {
                ppToA2.set_q_scale(128);
                ppToA1.set_q_scale(128);
            }
            FC1[i](A1, B1s[i], ppToA2);
            FC2[i](A2, B2s[i], ppToA1);
        }
    },
    (double(N) * K * elesize) * 2 * repeates,
    1e12,
    "Byte/s");
}


template<typename T, amx::PP::Steps ppsteps>
void test_FC_acc() {
    amx_FC_acc<T, ppsteps>(128, 96, 16);
    amx_FC_acc<T, ppsteps>(2, 2560, 10752);
    amx_FC_acc<T, ppsteps>(2, 10*32 + 17, 256 + 15);
    amx_FC_acc<T, ppsteps>(22, 2560, 10752);
    amx_FC_acc<T, ppsteps>(32*22, 10*32, 256);
    amx_FC_acc<T, ppsteps>(32*22 + 1, 10*32, 256 + 1);
    amx_FC_acc<T, ppsteps>(32*22 + 16, 10*32, 256 + 17);
    amx_FC_acc<T, ppsteps>(32*22 + 31, 10*32, 256 + 15);
    amx_FC_acc<T, ppsteps>(32*22 + 31, 10*32 + 1, 256 + 15);
    amx_FC_acc<T, ppsteps>(32*22 + 31, 10*32 + 17, 256 + 15);
    amx_FC_acc<T, ppsteps>(2, 10*32, 256);
}

int test_acc() {
    test_FC_acc<int8_t, amx::PP::Steps::DEQUANT_BIAS_GELU>();

    precision = Matmul::Weight_BF16;
    test_FC_acc<bfloat16, amx::PP::Steps::BIAS_GELU>();
    precision = Matmul::Weight_INT8;
    test_FC_acc<bfloat16, amx::PP::Steps::DEQUANT_BIAS_GELU>();
    return 0;
}

template<typename T, amx::PP::Steps ppsteps>
void test_FC_perf() {
    amx_FC_perf<T, ppsteps>(2, 2560, 10752);
    amx_FC_perf<T, ppsteps>(22, 2560, 10752);
    amx_FC_perf<T, ppsteps>(32*28, 32*80, 10240);
    amx_FC_perf<T, ppsteps>(32*28 + 1, 32*80, 10240);
    amx_FC_perf<T, ppsteps>(32*28 + 16, 32*80, 10240);
    amx_FC_perf<T, ppsteps>(32*28 + 17, 32*80, 10240);
    amx_FC_perf<T, ppsteps>(32*28 + 31, 32*80, 10240);
    amx_FC_perf<T, ppsteps>(32*28, 32*80, 10240);
    amx_FC_perf<T, ppsteps>(32*28 + 1, 32*80, 10240);
    amx_FC_perf<T, ppsteps>(32*28, 32*80 + 1, 10240);
    amx_FC_perf<T, ppsteps>(32*28, 32*80, 10240 + 1);
    amx_FC_perf<T, ppsteps>(32*28 + 1, 32*80 + 1, 10240 + 1);
    amx_FC_perf<T, ppsteps>(32*28 + 32, 32*80 + 32, 10240 + 32);
    amx_FC_perf<T, ppsteps>(896, 256, 1024, 10000);
    amx_FC_perf<T, ppsteps>(896, 256, 1024, 10000);
}

void test_perf() {
    precision = Matmul::Weight_BF16;
    test_FC_perf<int8_t, amx::PP::Steps::DEQUANT>();
    precision = Matmul::Weight_BF16;
    test_FC_perf<bfloat16, amx::PP::Steps::NONE>();
    precision = Matmul::Weight_INT8;
    test_FC_perf<bfloat16, amx::PP::Steps::DEQUANT>();
}

/*
 B matrix is 50MB, 56-cores took 2.8GB, so it can use almost full HBM bandwidth 600GB+
    test_parallel_FC_2_2560_10240_bf16 Avg latency:
        4420.87 us x 221  HW Usage : 66% (664.125 GByte/s /1000 GByte/s)
*/
void test_parallel_FC(int L, int M, int K, int N, int times = -5000) {
    tensor2D<bfloat16> A0(M, K);
    tensor2D<bfloat16> B0(K, N);
    tensor2D<bfloat16> C0(M, N);
    tensor2D<float> Bias0(1, N);

    struct mm_single_layer {
        tensor2D<bfloat16> A;
        tensor2D<bfloat16> B;
        tensor2D<bfloat16> C;
        tensor2D<float> Bias;
        int _N;
        std::shared_ptr<Matmul> mm;
        void create(tensor2D<bfloat16> & Atemplate,
                    tensor2D<bfloat16> & Btemplate,
                    tensor2D<bfloat16> & Ctemplate,
                    tensor2D<float> & BiasTemplate) {
            A = Atemplate.clone();
            B = Btemplate.clone();
            C = Ctemplate.clone();
            Bias = BiasTemplate.clone();
            _N = B.dims[1];
            mm.reset(new Matmul(true, false, precision));
        }
        void run() {
            // post-ops do nothing
            //amx::PP::Dummy ppkernel(C);
            amx::PP::BiasGeluStore<bfloat16, amx::PP::Steps::BIAS_GELU> ppkernel(C, &Bias(0,0));
            (*mm.get())(A, B, 0, _N, ppkernel);
        }
    };

    struct mm_multi_layer {
        std::vector<mm_single_layer> mms;
        void create(int layers,
                    tensor2D<bfloat16> & Atemplate,
                    tensor2D<bfloat16> & Btemplate,
                    tensor2D<bfloat16> & Ctemplate,
                    tensor2D<float> & BiasTemplate) {
            mms.resize(layers);
            for(int i = 0; i < layers; i++) {
                mms[i].create(Atemplate, Btemplate, Ctemplate, BiasTemplate);
            }
        }
        void run() {
            for(auto & layer : mms) {
                layer.run();
            }
        }
    };

    std::vector<mm_multi_layer> mms(OMP_NT);

    #pragma omp parallel
    {
        int i = omp_get_thread_num();
        mms[i].create(L, A0, B0, C0, Bias0);
    }

    double elesize = (precision == Matmul::Weight_BF16)? sizeof(bfloat16) : sizeof(int8_t);
    timer.tag(__func__, L, M, K, N, precision)(times, [&](){
        #pragma omp parallel
        {
            int i = omp_get_thread_num();
            mms[i].run();
        }
    },
    (double(N) * K * elesize * L) * OMP_NT,
    1e12,
    "Byte/s");
}

void test_parallel_FC() {
    precision = Matmul::Weight_BF16;
    // K*N is same, but K is bigger, bandwidth usage is high & more stable
    while(1) {
        std::cout << "=========================\n";
        test_parallel_FC(1, 2, 25600, 1024);
        test_parallel_FC(1, 2, 25600, 1024);
        test_parallel_FC(1, 2, 25600, 1024);
        std::cout << "=========================\n";
        test_parallel_FC(1, 2, 2560, 10240);
        test_parallel_FC(1, 2, 2560, 10240);
        test_parallel_FC(1, 2, 2560, 10240);
        std::cout << "=========================\n";
        // multi-layer, bandwidth usage is very unstable
        test_parallel_FC(40, 2, 2560, 256);
        test_parallel_FC(40, 2, 2560, 256);
        test_parallel_FC(40, 2, 2560, 256);
    }
}

using amx::PP::Steps;

//=====================================================================================================
int main(int argc, const char *argv[]) {
    timer.set_app(argv[0]);
    //thp.Start();
    //test_all_bw(3.0); return 0;
    //test_parallel_FC();

    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    std::cout << ANSIcolor("31") << "omp_get_num_threads() = " << omp_get_num_threads() << std::endl << ANSIcolor();
    std::cout << ANSIcolor("31") << "OMP_NT = " << OMP_NT << std::endl << ANSIcolor();

    //test_acc();    test_perf();    return 0;

    precision = Matmul::Weight_BF16;
    amx_MatmulMT_perf<bfloat16, Steps::BIAS_GELU>(2, 2560, 10752, false, -1000);
    amx_MatmulMT_perf<bfloat16, Steps::BIAS_GELU>(2, 2560, 10752, false, -1000);
    //amx_MatmulMT_perf(2, 2560, 10752, false, -1000);
    precision = Matmul::Weight_INT8;
    amx_MatmulMT_perf<bfloat16, Steps::DEQUANT_BIAS_GELU>(2, 2560, 10752, false, -1000);
    amx_MatmulMT_perf<bfloat16, Steps::DEQUANT_BIAS_GELU>(2, 2560, 10752, false, -1000);

    amx_MatmulMT_perf<int8_t, Steps::DEQUANT_BIAS_GELU_QUANT>(2, 2560, 10752, false, -1000);
    amx_MatmulMT_perf<int8_t, Steps::DEQUANT_BIAS_GELU_QUANT>(2, 2560, 10752, false, -1000);

    for(int i=0;i<10;i++) {
        precision = Matmul::Weight_BF16;
        amx_FC_MTML_perf<bfloat16, Steps::BIAS_GELU>(2, 2560, 10752, 20, -10000);
        amx_FC_MTML_perf<bfloat16, Steps::BIAS_GELU>(2, 2560, 10752, 20, -10000);
        precision = Matmul::Weight_INT8;
        amx_FC_MTML_perf<bfloat16, Steps::DEQUANT_BIAS_GELU>(2, 2560, 10752, 20, -10000);
        amx_FC_MTML_perf<bfloat16, Steps::DEQUANT_BIAS_GELU>(2, 2560, 10752, 20, -10000);

        amx_FC_MTML_perf<int8_t, Steps::DEQUANT_BIAS_GELU_QUANT>(2, 2560, 10752, 20, -10000);
        amx_FC_MTML_perf<int8_t, Steps::DEQUANT_BIAS_GELU_QUANT>(2, 2560, 10752, 20, -10000);
    }
    return 0;
    // return 0;

    //test_bf16(); return 0;
    //amx_Matmul_perf(12, 256, 32, true); return 0;

    amx_Matmul_perf_float(16, 256, 256);
    amx_Matmul_perf_float(224, 256, 256);
    amx_Matmul_perf_float(512, 256, 256);
    
    //amx_Matmul_perf(32, 120, 5, true); return 0;
    //amx_Matmul_perf(32, 18, 5, true); return 0;

    //amx_FC_perf(32, 5120, 32, -1000); return 0;
    //amx_Matmul_perf(928, 96, 928, true); return 0;

    amx_MatmulMT_BiasGelu_acc(88, 77, 66, false);
    amx_MatmulMT_perf<bfloat16, Steps::BIAS_GELU>(2*901, 2560, 7680, false);
    amx_MatmulMT_BiasGelu_perf(2*901, 2560, 7680, false);


    amx_Matmul_perf(928, 96, 928, true);
    amx_Matmul_perf(901, 80, 901, true);
    amx_Matmul_perf(901, 901, 80, false); 

    test_blk_loops();

    amx_unit_test_gemAvB(901, 80);
    return 0;
}
