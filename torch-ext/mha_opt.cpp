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

*/

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
    int OMP_NT;
    tensor2D<float> qk;
    avx2::PP::None pp_none;

    MHA() {
        OMP_NT = omp_thread_count();
        for(int i = 0; i < OMP_NT; i++) ops_qk.push_back(std::make_shared<avx2::Matmul>(false, true));
    }

    /*
        q: M x K
        k: N x K (need transpose)
        v: N x K
    */
    void one_head_attention(tensor2D<float> & q, tensor2D<float> & k, tensor2D<float> & v, bool causal_mask) {
        auto M = q.dims[0];
        auto N = k.dims[0];
        int ompi = omp_get_thread_num();
        qk.resize(M, N);
        (*ops_qk[ompi])(q, k, qk, 0, N, pp_none);

        // softmax per row
        
    }
};

void qkv_attention(float * pQ, float * pK, float * pV, float * pWV,
                   int B, int M, int N, int H, int K, bool causal_mask)
{
    static MHA mha;

    int stride_b_q = M*H*K;
    int stride_b_kv = N*H*K;
    int stride_bytes_hk = H*K*sizeof(float);

    #pragma omp parallel for collapse(2)
    for(int b = 0; b < B; b++) {
        for(int h = 0; h < H; h++) {
            // M can also run in parallel, but since
            // it's range is small, if we run it in one core, it can share k
            // q[b, 0:M, h, K] => MxK
            // k[b, 0:N, h, K] => NxK
            tensor2D<float> q(M, K, pQ + b*stride_b_q + h*stride_h, stride_bytes_hk);
            tensor2D<float> k(N, K, pK + b*stride_b_kv + h*stride_h, stride_bytes_hk);
            tensor2D<float> v(N, K, pV + b*stride_b_kv + h*stride_h, stride_bytes_hk);
            mha.one_head_attention(q, k, v, causal_mask);
        }
    }
}

at::Tensor mha_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    bool causal_mask)
{
    int ndims = q.dim();
    assert(ndims == 3);
    assert(k.dim() == ndims);
    assert(v.dim() == ndims);

    float * pQ = q.data_ptr<float>();
    float * pK = k.data_ptr<float>();
    float * pV = v.data_ptr<float>();
    
    int B = q.size(0);
    int M = q.size(1);
    int HK = q.size(2);

    assert(B == k.size(0));
    int N = k.size(1);
    assert(HK == k.size(2));

    assert(B == v.size(0));
    assert(N == v.size(1));
    assert(HK == v.size(2));

    int K = 64;
    int H = HK / K;
    assert(HK == H*K);

    auto wv = q.new_empty({B, M, HK});
    float * pWV =  wv.data_ptr<float>();

    qkv_attention(pQ, pK, pV, pWV,
                  B, M, N, H, K, causal_mask);
    return wv;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &mha_forward, "MHA forward");
}
