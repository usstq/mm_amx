/*
<torch/extension.h> is the one-stop header to include all the necessary PyTorch bits to write C++ extensions. It includes:
 - The ATen library, which is our primary API for tensor computation,
 - pybind11, which is how we create Python bindings for our C++ code,
 - Headers that manage the details of interaction between ATen and pybind11.
*/
#include <omp.h>

#include "misc.hpp"
#include "kernels_avx2.hpp"

// define PARALLEL_NT_STATIC
template<typename F>
void parallel_nt_static_omp(const F& func) {
    #pragma omp parallel
    { 
        func(omp_get_thread_num(), omp_get_num_threads());
    }
}

#define PARALLEL_NT_STATIC(...) parallel_nt_static_omp(__VA_ARGS__)
#include "kernels_mha.hpp"


//#include "timeit.hpp"
#include "profiler.hpp"


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

MHA2Kernels mha2;

static ProfilerManager profiler;
/*
#================================
#           q : [B, M, H*K]
#         k&v : [B, H, N, K]
# attn_output : [B, M, H, K]
#
# output_attentions: False
# attention_mask   : False
#================================
*/
torch::Tensor mha_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    bool causal_mask)
{
    AT_ASSERT(q.dim() == 3 && k.dim() == 4 && v.dim() == 4);

    int B = q.size(0);
    int M = q.size(1);
    int HK = q.size(2);
    int N = k.size(2);
    int K = k.size(3);
    int H = HK / K;

    AT_ASSERT(HK == H*K &&
              B == k.size(0) && B == v.size(0) &&
              H == k.size(1) && H == v.size(1) &&
              N == k.size(2) && N == v.size(2) &&
              K == k.size(3) && K == v.size(3));

    auto wv = q.new_empty({B, M, H*K});
    //auto prof = profiler.Profile("mha", B, M, N, HK);
    tensorND<float> tq(q.data_ptr<float>(), {B, M, H, K});
    tensorND<float> tk(k.data_ptr<float>(), {B, H, N, K});
    tensorND<float> tv(v.data_ptr<float>(), {B, H, N, K});
    tensorND<float> twv(wv.data_ptr<float>(), {B, M, H, K});

    mha2(tq, tk, tv, twv, true, causal_mask);
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
#if 0

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
        tensorND<float> mX(M, K0, x, K0*sizeof(float));
        tensorND<float> mY(M, K2, y, K2*sizeof(float));
        tensorND<float> mW1(K1, K0, w1, K0*sizeof(float));
        tensorND<float> mW2(K2, K1, w2, K1*sizeof(float));

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
#endif
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mha_forward", &mha_forward, "MHA forward");
/*
    py::class_<MLP>(m, "MLP")
        .def(py::init<>())
        .def("forward", &MLP::forward);
*/
    //py::class_<event>(m, "event").def(py::init<const std::string &>());

    m.attr("opt_mha") = pybind11::bool_(opt_mha);
    m.attr("opt_mlp") = pybind11::bool_(opt_mlp);
}
