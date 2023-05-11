/*
<torch/extension.h> is the one-stop header to include all the necessary PyTorch bits to write C++ extensions. It includes:
 - The ATen library, which is our primary API for tensor computation,
 - pybind11, which is how we create Python bindings for our C++ code,
 - Headers that manage the details of interaction between ATen and pybind11.
*/
#include "misc.hpp"
#include "kernels_avx2.hpp"
//#include "timeit.hpp"
#include "profiler.hpp"

#include <omp.h>

#include <torch/extension.h>

#include <iostream>
#include <vector>


/*
    q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) 
    k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1)
    v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

    qk = q @ k
    if mask is not None:
        qk = qk + mask[:n_ctx, :n_ctx]
    qk = qk.float()

    w = F.softmax(qk, dim=-1).to(q.dtype)
    wv = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)

 B: beam-width, can be 1-5
 M: number of tokens in query, can be 1-448, 1500
 N: number of tokens in past_kv, can be 1-448, 1500
 H: head
 K: 64
  
 q: [B, M, H*K]
 k: [B, N, H*K]
 v: [B, N, H*K]

 w = q @ k' =>  [B, H, M*N]
 w = softmax_per_row (w)
 wv = w @ v =>  [B, M, H*K]
*/

struct MHA {
    std::vector<std::shared_ptr<avx2::Matmul>> ops_qk;
    std::vector<std::shared_ptr<avx2::Matmul>> ops_wv;
    int OMP_NT;
    std::vector<tensor2D<float>> all_qk;
    avx2::PP::None pp_none;

    MHA() {
        OMP_NT = omp_thread_count();
        for(int i = 0; i < OMP_NT; i++) {
            ops_qk.push_back(std::make_shared<avx2::Matmul>(false, true));
            ops_wv.push_back(std::make_shared<avx2::Matmul>(false, false));
            all_qk.emplace_back();
        }
    }

    /*
        q: M x K
        k: N x K (need transpose)
        v: N x K
    */
    void one_head_attention(tensor2D<float> & q, tensor2D<float> & k, tensor2D<float> & v, tensor2D<float> & wv, bool causal_mask) {
        auto M = q.dims[0];
        auto N = k.dims[0];
        auto K = v.dims[1];
        
        int ompi = omp_get_thread_num();
        auto & qk = all_qk[ompi];
        qk.resize(M, N);
        (*ops_qk[ompi])(q, k, qk, 0, N, pp_none);

        // softmax per row
        if (causal_mask && M > 1) {
            for(int m = 0; m<M; m++) {
                int valid_n = std::min(N, m+1);
                avx2::functional::softmax(&qk(m,0), valid_n);
                // the rest part is set as zero
                memset(&qk(m, valid_n), 0, sizeof(float)*(N - valid_n));
            }
        } else {
            for(int m = 0; m<M; m++) {
                avx2::functional::softmax(&qk(m,0), N);
            }
        }
        // combine
        (*ops_wv[ompi])(qk, v, wv, 0, K, pp_none);
    }
};

void qkv_attention(float * pQ, float * pK, float * pV, float * pWV,
                   int B, int M, int N, int H, int K, bool causal_mask)
{
    static MHA mha;

    int stride_b_q = M*H*K;
    int stride_b_kv = N*H*K;
    int stride_bytes_hk = H*K*sizeof(float);
    int stride_h = K;

    #pragma omp parallel for collapse(2)
    for(int b = 0; b < B; b++) {
        for(int h = 0; h < H; h++) {
            // M can also run in parallel, but since
            // it's range is small, if we run it in one core, it can share k
            // q[b, 0:M, h, K] => MxK
            // k[b, 0:N, h, K] => NxK
            // v[b, 0:N, h, K] => NxK
            tensor2D<float> q(M, K, pQ + b*stride_b_q + h*stride_h, stride_bytes_hk);
            tensor2D<float> k(N, K, pK + b*stride_b_kv + h*stride_h, stride_bytes_hk);
            tensor2D<float> v(N, K, pV + b*stride_b_kv + h*stride_h, stride_bytes_hk);
            tensor2D<float> wv(M, K, pWV + b*stride_b_q + h*stride_h, stride_bytes_hk);
            mha.one_head_attention(q, k, v, wv, causal_mask);
        }
    }
}

static ProfilerManager profiler;

torch::Tensor mha_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    bool causal_mask)
{
    int B = q.size(0);
    int M = q.size(1);
    int HK = q.size(2);
    int N = k.size(1);

    auto prof = profiler.Profile("mha", B, M, N, HK);
    int ndims = q.dim();
    AT_ASSERT(ndims == 3);
    AT_ASSERT(k.dim() == ndims);
    AT_ASSERT(v.dim() == ndims);

    float * pQ = q.data_ptr<float>();
    float * pK = k.data_ptr<float>();
    float * pV = v.data_ptr<float>();

    AT_ASSERT(B == k.size(0));
    AT_ASSERT(HK == k.size(2));

    AT_ASSERT(B == v.size(0));
    AT_ASSERT(N == v.size(1));
    AT_ASSERT(HK == v.size(2));

    int K = 64;
    int H = HK / K;
    AT_ASSERT(HK == H*K);

    auto wv = q.new_empty({B, M, HK});
    float * pWV =  wv.data_ptr<float>();
    qkv_attention(pQ, pK, pV, pWV,
                  B, M, N, H, K, causal_mask);
    return wv;
}

bool opt_mha = false;
bool opt_mlp = false;

/*
 x : [B, M, H*K]
 w0: [1536, 384]
 x1: [B, M, 1536]
  GELU
 w2: [384, 1536]
 x2: [B, M, H*K]
*/
static int OMP_NT = omp_thread_count();

struct MLP {
    std::vector<std::shared_ptr<avx2::Matmul>> ops_fc1;
    std::vector<std::shared_ptr<avx2::Matmul>> ops_fc2;

    tensor2D<float> mTemp;
    MLP() {
        for(int i = 0; i < OMP_NT; i++) {
            ops_fc1.push_back(std::make_shared<avx2::Matmul>(true, true));
            ops_fc2.push_back(std::make_shared<avx2::Matmul>(true, true));
        }
    }

    void exec(float * x, int M, int K0,
              float * w1, float * b1, int K1,
              float * w2, float * b2, int K2,
              float * y) {
        // w1: K1xK0, need transpose, split on K1
        // w2: K2xK1, need transpose, split on K2
        tensor2D<float> mX(M, K0, x, K0*sizeof(float));
        tensor2D<float> mY(M, K2, y, K2*sizeof(float));
        tensor2D<float> mW1(K1, K0, w1, K0*sizeof(float));
        tensor2D<float> mW2(K2, K1, w2, K1*sizeof(float));

        mTemp.resize(M, K1);

        int work_amount1 = rndup(K1, 16)/16;
        avx2::PP::AddbiasAct<avx2::PP::Act_GELU> pp1(b1);
        auto kernel1 = [&](int tid, int cnt) {
            int start, end;
            splitter(work_amount1, cnt, tid, start, end);
            int n0 = start*16;
            int n1 = end*16;
            if (n1 > K1) n1 = K1;
            (*ops_fc1[tid].get())(mX, mW1, mTemp, n0, n1, pp1);
        };

        int work_amount2 = rndup(K2, 16)/16;
        avx2::PP::AddbiasAct<avx2::PP::Act_NONE> pp2(b2);
        auto kernel2 = [&](int tid, int cnt) {
            int start, end;
            splitter(work_amount2, cnt, tid, start, end);
            int n0 = start*16;
            int n1 = end*16;
            if (n1 > K2) n1 = K2;
            (*ops_fc2[tid].get())(mTemp, mW2, mY, n0, n1, pp2);
        };

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            kernel1(tid, OMP_NT);
        }
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            kernel2(tid, OMP_NT);
        }
    }

    // x + GELU(x*w0 + bias0) * w1 + bias1
    torch::Tensor forward(
        torch::Tensor x,
        torch::Tensor w0,
        torch::Tensor bias0,
        torch::Tensor w1,
        torch::Tensor bias1) {
        
        auto prof = profiler.Profile("MLP");
        int ndims = x.dim();
        
        AT_ASSERT(x.dim() == 3);
        AT_ASSERT(w0.dim() == 2);
        AT_ASSERT(w1.dim() == 2);
        AT_ASSERT(bias0.dim() == 1);
        AT_ASSERT(bias1.dim() == 1);
        AT_ASSERT(bias1.size(0) == x.size(2));

        auto y = x.new_empty({x.size(0), x.size(1), x.size(2)});
        exec(x.data_ptr<float>(), x.size(0) * x.size(1), x.size(2),
                w0.data_ptr<float>(), bias0.data_ptr<float>(), bias0.size(0),
                w1.data_ptr<float>(), bias1.data_ptr<float>(), bias1.size(0),
                y.data_ptr<float>());
        return y;
    }
};

struct event {
    ProfilerManager::ProfileDataWrapper d;
    event(const std::string & name) {
        d = std::move(profiler.Profile(name));
    }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mha_forward", &mha_forward, "MHA forward");
    py::class_<MLP>(m, "MLP")
        .def(py::init<>())
        .def("forward", &MLP::forward);

    py::class_<event>(m, "event")
        .def(py::init<const std::string &>());

    m.attr("opt_mha") = pybind11::bool_(opt_mha);
    m.attr("opt_mlp") = pybind11::bool_(opt_mlp);
}
