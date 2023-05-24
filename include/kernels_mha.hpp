#include "kernels_avx2.hpp"
#include <vector>

#ifndef PARALLEL_NT_STATIC
template <typename F>
void parallel_nt_static_dummy(const F& func) {
    func(0, 1);
}
#define PARALLEL_NT_STATIC(...) parallel_nt_static_dummy(__VA_ARGS__)
#endif

template<typename ... Args>
static inline void log(Args&& ... args) {
    std::stringstream ss;
    int dummy[] = {(ss << std::forward<Args>(args), 0)...};
    UNUSED(dummy);
    ss << std::endl;
    std::cout << ss.str();
}

static inline size_t offset2coord(size_t off, size_t D, size_t &d) {
    auto next_off = off/D;
    d = off - next_off*D;
    return next_off;
}

template<typename ... Args>
static inline size_t offset2coord(size_t off, size_t D, size_t &d, Args&& ... args) {
    off = offset2coord(off, std::forward<Args>(args)...);
    auto next_off = off/D;
    d = off - next_off*D;
    return next_off;
}

template <typename T, typename Q>
inline void splitter1d(const T& n, const Q& team, const Q& tid, T& n_start, T& n_end) {
    if (team <= 1 || n == 0) {
        n_start = 0;
        n_end = n;
    } else {
        T n1 = (n + (T)team - 1) / (T)team;
        T n2 = n1 - 1;
        T T1 = n - n2 * (T)team;
        n_end = (T)tid < T1 ? n1 : n2;
        n_start = (T)tid <= T1 ? tid * n1 : T1 * n1 + ((T)tid - T1) * n2;
    }

    n_end += n_start;
}

struct MHA2Kernels {
    std::vector<std::shared_ptr<avx2::Matmul>> ops_qk;
    std::vector<std::shared_ptr<avx2::Matmul>> ops_wv;
    std::vector<tensorND<float>> all_qk;
    tensorND<float> sub_states;
    tensorND<float> qk_max;
    tensorND<float> qk_sum;
    avx2::PP::None pp_none;

    int bN;                 // blocking on N dimension
    MHA2Kernels() {
        bN = std::getenv("bN") ? atoi(std::getenv("bN")) : 256;
        int NT = 0;
        PARALLEL_NT_STATIC([&](int i, int n){
            NT = n;
        });
        for(int i = 0; i < NT; i++) {
            ops_qk.push_back(std::make_shared<avx2::Matmul>(false, true));
            ops_wv.push_back(std::make_shared<avx2::Matmul>(false, false));
            all_qk.emplace_back();
        }
    }

    void operator()(tensorND<float>& q0,
                    tensorND<float>& k0,
                    tensorND<float>& v0,
                    tensorND<float>& wv0,
                    bool kv_head_transposed,
                    bool with_causal_mask) {
        auto B = q0.shape[0];
        auto M = q0.shape[1];
        auto H = q0.shape[2];
        auto K = q0.shape[3];
        auto N = k0.shape[kv_head_transposed ? 2:1];
        if (kv_head_transposed) {
            //q0  {B, M, H, K}
            //k0  {B, H, N, K}
            //v0  {B, H, N, K}
            //wv0 {B, M, H, K}
            static int sss = 0;
            // if (N > bN) sss ++;
            if (with_causal_mask || (N < bN) || (sss & 3)==2) {
                const size_t work_amount = size_t(B)*H;
                PARALLEL_NT_STATIC([&](int ithr, int nthr) {
                    size_t start{0}, end{0}, h;
                    splitter1d(work_amount, nthr, ithr, start, end);
                    for(size_t bh = start; bh < end; bh++) {
                        auto b = offset2coord(bh, H, h);
                        //  q[b, 0:M, h, K] => MxK
                        //  k[b, h, 0:N, K] => NxK
                        //  v[b, h, 0:N, K] => NxK
                        // wv[b, 0:M, h, K] => MxK
                        tensorND<float> q = q0.Slice(b, fullslice(), h, fullslice());
                        tensorND<float> k = k0.Slice(b, h, fullslice(), fullslice());
                        tensorND<float> v = v0.Slice(b, h, fullslice(), fullslice());
                        tensorND<float> wv = wv0.Slice(b, fullslice(), h, fullslice());
                        one_head_attention(ithr, q, k, v, wv, 0, with_causal_mask);
                    }
                });
            } else {
                // no with_kv_cache, no with_causal_mask
                //
                // kernel register blocking on 6x16, so N is split in unit of 256 columns
                // that means each token will be encoded by 256 key/values and finally combined
                // if N is smaller than 256, we don't split them
                int num_sub_states = (N + bN - 1) / bN;
                const size_t work_amount = (size_t)(B * H) * num_sub_states;

                // s[B, M, H, nb, K]
                sub_states.resize({B, M, H,num_sub_states, K}, false);
                qk_max.resize({B, H, num_sub_states, M}, false);
                qk_sum.resize({B, H, num_sub_states, M}, false);

                //auto _prof1 = Profile("substate");
                PARALLEL_NT_STATIC([&](int ithr, int nthr) {
                    // each work item is doing  M x bN sub-states encoding
                    // and finally, main thread will combine sub-states into one
                    size_t start{0}, end{0};
                    splitter1d(work_amount, nthr, ithr, start, end);
                    if (start == end) return;
                    //std::stringstream ss; ss << ithr << "/" << nthr << std::endl;   std::cout << ss.str();
                    // encoding sub-states one by one
                    for (auto cur = start; cur < end; cur++) {
                        size_t h;
                        size_t nb;
                        auto b = offset2coord(cur, H, h, num_sub_states, nb);
                        auto n0 = nb * bN;
                        auto n1 = std::min(size_t(N), n0 + bN);

                        //  q[b, 0:M, h, K]   => M x K
                        //  k[b, h, n0:n1, K] => bN x K
                        //  v[b, h, n0:n1, K] => bN x K
                        //  s[b, 0:M, h, nb, K] => M x K
                        // wv[b, 0:M, h, K] => M x K
                        tensorND<float> q = q0.Slice(b, fullslice(), h, fullslice());
                        tensorND<float> k = k0.Slice(b, h, slice(n0, n1), fullslice());
                        tensorND<float> v = v0.Slice(b, h, slice(n0, n1), fullslice());
                        tensorND<float> s = sub_states.Slice(b, fullslice(), h, nb, fullslice());

                        one_head_attention(ithr, q, k, v, s, 0, with_causal_mask,
                                        &qk_max(b, h, nb, 0),
                                        &qk_sum(b, h, nb, 0));
                    }
                });

                // combine sub-states
                //auto _prof2 = Profile("combine");
                for(int b = 0; b<B; b++) {
                    for(int h = 0; h<H; h++) {
                        //  s[b, 0:M, h, nb, K] => M x nb x K
                        // wv[b, 0:M, h, K] => M x K
                        // qk_max [b,h,nb,M]
                        tensorND<float> wv = wv0.Slice(b, fullslice(), h, fullslice());

                        // qk_max: [b, h, 0:num_sub_states, 0:M]
                        // qk_sum: [b, h, 0:num_sub_states, 0:M]
                        auto p_qk_max = qk_max.Slice(b, h, fullslice(), fullslice());  // num_sub_states x M
                        auto p_qk_sum = qk_sum.Slice(b, h, fullslice(), fullslice());  // num_sub_states x M
                        for (int m = 0; m < M; m++) {
                            float * p_wv = &wv(m, 0);
                            // get weights of sub-states :
                            //    tmax = max_i(qk_max_i)
                            //    
                            //    tsum_i = sum_i * exp(qk_max_i - tmax)
                            //    weight_i = tsum_i/sum_i(tsum_i)
                            //float tmax = std::numeric_limits<float>::lowest();
                            auto tmax = _mm256_set1_ps(std::numeric_limits<float>::lowest());
                            for (int nb = 0; nb < num_sub_states; nb++) {
                                auto sub_max = _mm256_broadcast_ss(&p_qk_max(nb,m));
                                tmax = _mm256_max_ps(tmax, sub_max);
                            }

                            auto tsum = _mm256_setzero_ps();
                            for (int nb = 0; nb < num_sub_states; nb++) {
                                auto sub_max = _mm256_broadcast_ss(&p_qk_max(nb,m));
                                sub_max = _mm256_sub_ps(sub_max, tmax);
                                auto sub_sum = _mm256_broadcast_ss(&p_qk_sum(nb,m));
                                avx2::functional::exp_ps(sub_max);
                                sub_sum = _mm256_mul_ps(sub_sum, sub_max);
                                p_qk_sum(nb,m) = _mm256_cvtss_f32(sub_sum);
                                tsum = _mm256_add_ps(tsum, sub_sum);
                                //p_qk_sum[nb*M + m] *= std::exp(p_qk_max[nb*M + m] - tmax);
                                //tsum += p_qk_sum[nb*M + m];
                            }

                            static __m256 one = _mm256_castsi256_ps(_mm256_set1_epi32(0x3f800000)); // 1.0f
                            auto tweight_recip = _mm256_div_ps(one, tsum);                          // 1/sum_exp

                            // linear combine sub-states, 
                            __m256i wv_mask = _mm256_setzero_si256();
                            for (int nb = 0; nb < num_sub_states; nb++) {
                                //
                                float* p_sub = &sub_states(b, 0, h, nb, 0);

                                auto x_weight = _mm256_broadcast_ss(&p_qk_sum(nb, m));
                                x_weight = _mm256_mul_ps(x_weight, tweight_recip);
                                //auto x_weight = _mm256_set1_ps(p_qk_sum[nb*M + m] * tweight_recip);
                                // wv = substates * weight    for nb=0
                                // wv += substates * weight   otherwise
                                if (nb == 1) wv_mask = avx2::functional::get_mask(8);
                                int k;
                                for(k = 0; (k+8) <= K; k += 8) {
                                    auto x_sub = _mm256_loadu_ps(p_sub + k);
                                    auto x_new = _mm256_maskload_ps(p_wv + k, wv_mask);
                                    x_new = _mm256_fmadd_ps(x_sub, x_weight, x_new);
                                    _mm256_storeu_ps(p_wv + k, x_new);
                                }
                                if (k < K) {
                                    auto mask = avx2::functional::get_mask(K&7);
                                    auto x_sub = _mm256_maskload_ps(p_sub + k, mask);
                                    auto x_new = _mm256_maskload_ps(p_wv + k, wv_mask);
                                    x_new = _mm256_fmadd_ps(x_sub, x_weight, x_new);
                                    _mm256_maskstore_ps(p_wv + k, mask, x_new);
                                }
                            }
                        }
                    }
                }
            }
        } else {
            // with_causal_mask:0   with_kv_cache:0   kv_head_transposed:0
            // parallel in B/H/M dimensions
            const size_t work_amount = (size_t) B * H * M;
            //q0  {B, M, H, K}
            //k0  {B, N, H, K}
            //v0  {B, N, H, K}
            //wv0 {B, M, H, K}
            PARALLEL_NT_STATIC([&](int ithr, int nthr) {
                size_t start{0}, end{0};
                splitter1d(work_amount, nthr, ithr, start, end);
                size_t bh0, mb0;
                size_t bh1, mb1;
                if (start == end) return;
                bh0 = offset2coord(start, M, mb0);
                bh1 = offset2coord(end, M, mb1);
                auto m_start = mb0;
                auto m_end = std::min(size_t(M), mb1);

                // first head
                auto m0 = m_start;
                auto m1 = m0;
                // bh = b*H + h
                for(auto bh = bh0; bh <= bh1; bh++) {
                    // determine m1 for current head
                    m1 = (bh == bh1) ? m_end : M;
                    if (m1 <= m0) break;

                    //  q[b, m0:m1, h, K] => (m1-m0)xK
                    //  k[b, 0:N,   h, K] => NxK
                    //  v[b, 0:N,   h, K] => NxK
                    // wv[b, m0:m1, h, K] => (m1-m0)xK
                    auto b = bh/H;
                    auto h = bh - b*H;
                    tensorND<float> q = q0.Slice(b, slice(m0, m1), h, fullslice());
                    tensorND<float> k = k0.Slice(b, fullslice(), h, fullslice());
                    tensorND<float> v = v0.Slice(b, fullslice(), h, fullslice());
                    tensorND<float> wv = wv0.Slice(b, slice(m0, m1), h, fullslice());
                    one_head_attention(ithr, q, k, v, wv, m0, with_causal_mask);

                    // m0 for next head is always 0
                    m0 = 0;
                }
            });
        }
    }

    /*
        q: M x K
        k: N x K (need transpose)
        v: N x K
    */
    void one_head_attention(int tid,
                            tensorND<float>& q,
                            tensorND<float>& k,
                            tensorND<float>& v,
                            tensorND<float>& wv,
                            int causal_m0,
                            bool causal_mask,
                            float * qk_max,
                            float * qk_sum) {
        auto M = q.shape[0];
        auto N = k.shape[0];
        auto K = v.shape[1];

        auto & qk = all_qk[tid];
        qk.resize({M, N}, false);
        (*ops_qk[tid])(q, k, qk, 0, N, pp_none);

        // softmax per row
        if (causal_mask && M > 1) {
            for(int m = 0; m<M; m++) {
                int valid_n = std::min(N, m + causal_m0 + 1);
                avx2::functional::softmax(&qk(m,0), valid_n, qk_max + m, qk_sum + m);
                // the rest part is set as zero
                memset(&qk(m, valid_n), 0, sizeof(float)*(N - valid_n));
            }
        } else {
            for(int m = 0; m<M; m++) {
                avx2::functional::softmax(&qk(m,0), N, qk_max + m, qk_sum + m);
            }
        }
        // combine
        (*ops_wv[tid])(qk, v, wv, 0, K, pp_none);
    }


    void one_head_attention(int tid,
                            tensorND<float>& q,
                            tensorND<float>& k,
                            tensorND<float>& v,
                            tensorND<float>& wv,
                            int causal_m0,
                            bool causal_mask) {
        auto M = q.shape[0];
        auto N = k.shape[0];
        auto K = v.shape[1];

        auto & qk = all_qk[tid];
        qk.resize({M, N}, false);
        (*ops_qk[tid])(q, k, qk, 0, N, pp_none);

        // softmax per row
        if (causal_mask && M > 1) {
            for(int m = 0; m<M; m++) {
                int valid_n = std::min(N, m + causal_m0 + 1);
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
        (*ops_wv[tid])(qk, v, wv, 0, K, pp_none);
    }
};
