/*
<torch/extension.h> is the one-stop header to include all the necessary PyTorch bits to write C++ extensions. It includes:
 - The ATen library, which is our primary API for tensor computation,
 - pybind11, which is how we create Python bindings for our C++ code,
 - Headers that manage the details of interaction between ATen and pybind11.
*/
#include "misc.hpp"
#include "kernels_avx2.hpp"
#include "timeit.hpp"
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

torch::Tensor mha_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    bool causal_mask)
{
    int ndims = q.dim();
    AT_ASSERT(ndims == 3);
    AT_ASSERT(k.dim() == ndims);
    AT_ASSERT(v.dim() == ndims);

    float * pQ = q.data_ptr<float>();
    float * pK = k.data_ptr<float>();
    float * pV = v.data_ptr<float>();
    
    int B = q.size(0);
    int M = q.size(1);
    int HK = q.size(2);

    AT_ASSERT(B == k.size(0));
    int N = k.size(1);
    AT_ASSERT(HK == k.size(2));

    AT_ASSERT(B == v.size(0));
    AT_ASSERT(N == v.size(1));
    AT_ASSERT(HK == v.size(2));

    int K = 64;
    int H = HK / K;
    AT_ASSERT(HK == H*K);

    auto wv = q.new_empty({B, M, HK});
    float * pWV =  wv.data_ptr<float>();
    //tensor2D<float> wv(B*M, HK, true);
    //float * pWV = &wv[0];
    qkv_attention(pQ, pK, pV, pWV,
                  B, M, N, H, K, causal_mask);
    return wv;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &mha_forward, "MHA forward");
}
