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

#include "kernels_amxbf16.hpp"
#include "kernels_avx512.hpp"
#include "thread_pool.hpp"
#include "timeit.hpp"
#include "misc.hpp"

#include "thread_pool.hpp"

timeit timer;

//================================================================================
// initialize AMX
static bool initAMX = initXTILE();

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
    amx_bf16::tileconfig_t tfg(1, 0, 8, 16, 64);

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

void amx_FC_acc(int M, int K, int N) {
    tensor2D<bfloat16> A(M, K);
    tensor2D<bfloat16> B(K, N);
    tensor2D<bfloat16> C(M, N);
    tensor2D<bfloat16> C0(M, N);
    amx_bf16::Matmul fc(true);

    fc(A, B, C);

    C0=0;
    matmul(A, B, C0);
    std::cout << __func__ << " [" << M << "," << K << "," << N << "] ";
    if (C0 == C) {
        std::cout << ANSIcolor("1;32") << "Match!\n" << ANSIcolor();
        //std::cout << C << std::endl;
    } else {
        std::cout << ANSIcolor("1;31") << "Mismatch!\n" << ANSIcolor();
        std::cout << C0 << std::endl;
        std::cout << C << std::endl;
    }
}

void amx_FC_perf(int M, int K, int N, int times = -1000) {
    tensor2D<bfloat16> A(M, K);
    tensor2D<bfloat16> B(K, N);
    tensor2D<bfloat16> C(M, N);
    tensor2D<bfloat16> C0(M, N);
    amx_bf16::Matmul mm(true);

    std::cout << __func__ << " [" << M << "," << K << "," << N << "] ";
    timer(times, [&](){
        mm(A, B, C);
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
    amx_bf16::Matmul mm(false, transB);

    std::cout << __func__ << " [" << M << "," << K << "," << N << "] ";

    C0=0;
    matmul(A, B, C0);
    mm(A, transB?BT:B, C);
    if (C0 == C) {
        std::cout << ANSIcolor("1;32") << "Match!\n" << ANSIcolor();
        //std::cout << C << std::endl;
    } else {
        std::cout << ANSIcolor("1;31") << "Mismatch!\n" << ANSIcolor();
        std::cout << C0 << std::endl;
        std::cout << C << std::endl;
    }

    timer(times, [&](){
        mm(A, transB?BT:B, C);
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
    amx_bf16::GemAvB gemAvB;

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


ThreadPool thp;

// multi-threaded matmul
struct MatmulMT {
    std::vector<std::shared_ptr<amx_bf16::Matmul>> ops;
    bool transposeB;
    MatmulMT(bool constB = false, bool transposeB = false) : transposeB(transposeB) {
        for(int i = 0; i < thp.num_threads; i++)
            ops.push_back(std::make_shared<amx_bf16::Matmul>(constB, transposeB));
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
            // C[:, N0:N1] = A * B[:, N0:N1]
            (*ops[tid].get())(matA, matB, n0, n1, ppkernel);
        };
        thp.Paralell_NT(kernel);
    }
};

void amx_MatmulMT_perf(int M, int K, int N, bool transB, int times = -1000) {
    tensor2D<bfloat16> A(M, K);
    tensor2D<bfloat16> B(K, N);
    tensor2D<bfloat16> BT = B.Tr();
    tensor2D<bfloat16> C(M, N);
    tensor2D<bfloat16> C0(M, N);
    amx_bf16::Matmul mm(true, transB);
    MatmulMT                  mmMT(true, transB);
    amx_bf16::PP::Store2bf16 pp0(C0);
    amx_bf16::PP::Store2bf16 pp(C);

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


void amx_MatmulMT_BiasGelu_acc(int M, int K, int N, bool transB) {
    tensor2D<bfloat16> A(M, K);
    tensor2D<bfloat16> B(K, N);
    tensor2D<bfloat16> BT = B.Tr();
    tensor2D<bfloat16> C(M, N);
    tensor2D<bfloat16> C0(M, N);
    tensor2D<float> Bias(1, N);
    amx_bf16::Matmul mm(true, transB);
    amx_bf16::PP::Addbias_Gelu_Store2bf16 pp0(C, &Bias(0,0));

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
    amx_bf16::Matmul mm(true, transB);
    MatmulMT                  mmMT(true, transB);
    amx_bf16::PP::Addbias_Gelu_Store2bf16 pp0(C0, &Bias(0,0));
    amx_bf16::PP::Addbias_Gelu_Store2bf16 pp(C, &Bias(0,0));

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
    avx512::Matmul<avx512::RELU> mm;
    std::cout << __func__ << " [" << M << "," << K << "," << N << "] ";

    C0=0;
    matmul(A, B, C0, &Bias(0,0), [](float x){
        return std::max(x, 0.0f);
    });
    mm(A, B, C, &Bias(0,0));
    if (C0 == C) {
        std::cout << ANSIcolor("1;32") << "Match!\n" << ANSIcolor();
        //std::cout << C << std::endl;
    } else {
        std::cout << ANSIcolor("1;31") << "Mismatch!\n" << ANSIcolor();
        std::cout << C0 << std::endl;
        std::cout << C << std::endl;
    }

    timer(times, [&](){
        mm(A, B, C, &Bias(0,0));
    },
    double(M * N) * K * 2,
    FP32PeakGopsPerCore * 1e9);
}

int main(int argc, const char *argv[]) {
    timer.set_app(argv[0]);
    thp.Start();

    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

    amx_Matmul_perf_float(16, 256, 256);
    amx_Matmul_perf_float(224, 256, 256);
    //amx_Matmul_perf(32, 120, 5, true); return 0;
    //amx_Matmul_perf(32, 18, 5, true); return 0;

    //amx_FC_perf(32, 5120, 32, -1000); return 0;
    //amx_Matmul_perf(928, 96, 928, true); return 0;

    amx_MatmulMT_BiasGelu_acc(88, 77, 66, false);
    amx_MatmulMT_perf(2*901, 2560, 7680, false);
    amx_MatmulMT_BiasGelu_perf(2*901, 2560, 7680, false);

    amx_FC_perf(32*28, 32*80, 10240);
    amx_FC_perf(32*28 + 1, 32*80, 10240);
    amx_FC_perf(32*28 + 16, 32*80, 10240);
    amx_FC_perf(32*28 + 17, 32*80, 10240);
    amx_FC_perf(32*28 + 31, 32*80, 10240);

    amx_Matmul_perf(928, 96, 928, true);
    amx_Matmul_perf(901, 80, 901, true);
    amx_Matmul_perf(901, 901, 80, false); 

    test_blk_loops();

    amx_unit_test_gemAvB(901, 80);
    //amx_unit_test_perf();
    amx_FC_acc(32*22, 10*32, 256);
    amx_FC_acc(32*22 + 1, 10*32, 256 + 1);
    amx_FC_acc(32*22 + 16, 10*32, 256 + 17);
    amx_FC_acc(32*22 + 31, 10*32, 256 + 15);
    amx_FC_acc(32*22 + 31, 10*32 + 1, 256 + 15);
    amx_FC_acc(32*22 + 31, 10*32 + 17, 256 + 15);

    amx_FC_perf(32*28, 32*80, 10240);
    amx_FC_perf(32*28 + 1, 32*80, 10240);
    amx_FC_perf(32*28, 32*80 + 1, 10240);
    amx_FC_perf(32*28, 32*80, 10240 + 1);
    amx_FC_perf(32*28 + 1, 32*80 + 1, 10240 + 1);
    amx_FC_perf(32*28 + 32, 32*80 + 32, 10240 + 32);

    amx_FC_perf(896, 256, 1024, 10000);


    return 0;
}
