#include "oneapi/dnnl/dnnl.hpp"

// MatMul_onednn<dnnl::memory::data_type::bf16>
// MatMul_onednn<dnnl::memory::data_type::f32>
struct DNNLInnerProduct {
    dnnl::engine eng;
    dnnl::stream stream;
    dnnl::matmul::primitive_desc matmul_pd;
    dnnl::matmul matmul_p;
    dnnl::inner_product_forward::primitive_desc inner_product_pd;
    dnnl::inner_product_forward inner_product_p;
    dnnl::memory::data_type dt;

    int m;
    int OC;
    int IC;

    void * B0;
    bool constb;
    bool transa = false;
    bool transb = false;

    DNNLInnerProduct(int M, int N, int K, void * B_ptr, bool transb) : m(M), OC(N), IC(K), B0(B_ptr), transb(transb) {
        eng = dnnl::engine(dnnl::engine::kind::cpu, 0);
        if (!eng)
            throw std::runtime_error("dnnl::engine::kind::cpu failed to create!");
        stream = dnnl::stream(eng);
        dt = dnnl::memory::data_type::bf16;
        constb = (B0 != nullptr);
        initialize_args();
    }

    dnnl::memory::desc a_md;
    dnnl::memory::desc b_md;
    dnnl::memory::desc c_md;
    void set_A(void * ptr) {
        m_args[DNNL_ARG_SRC] = dnnl::memory(a_md, eng, ptr);
    }
    void set_C(void * ptr) {
        m_args[DNNL_ARG_DST] = dnnl::memory(c_md, eng, ptr);
    }
    void set_B(void * ptr) {
        m_args[DNNL_ARG_WEIGHTS] = dnnl::memory(b_md, eng, ptr);
    }

    std::unordered_map<int, dnnl::memory> m_args;
    // static shape
    void initialize_args()
    {
        dnnl::memory::dims a_shape = {m, IC};
        dnnl::memory::dims b_shape = {OC, IC};
        dnnl::memory::dims c_shape = {m, OC};

        dnnl::memory::dims a_strides = (!transa) ? dnnl::memory::dims{IC, 1} : dnnl::memory::dims{1, IC};
        dnnl::memory::dims b_strides = (!transb) ? dnnl::memory::dims{IC, 1} : dnnl::memory::dims{1, OC};
        dnnl::memory::dims c_strides = dnnl::memory::dims{OC, 1};

        a_md = dnnl::memory::desc(a_shape, dt, a_strides);
        b_md = dnnl::memory::desc(b_shape, dt, b_strides);
        c_md = dnnl::memory::desc(c_shape, dnnl::memory::data_type::f32, c_strides);

        // Prepare oneDNN memory for alpha
        // memory alpha_m({{1}, memory::data_type::f32, {1}}, eng, &alpha);

        // Create attributes (to handle alpha dynamically and beta if necessary)
        dnnl::primitive_attr attr;
        //attr.set_scales_mask(DNNL_ARG_WEIGHTS, /* mask */ 0);

        dnnl::memory B_m(b_md, eng, B0);

        m_args.clear();
        m_args[DNNL_ARG_WEIGHTS] = B_m;

        if (constb) {
            // create inner_product_forward
            auto inner_product_weights_md = dnnl::memory::desc(b_shape, dt, dnnl::memory::format_tag::any);
            inner_product_pd = dnnl::inner_product_forward::primitive_desc(eng,
                                dnnl::prop_kind::forward_inference,
                                a_md, inner_product_weights_md,
                                c_md, attr);
            if (inner_product_pd.weights_desc() != b_md) {
                auto inner_product_weights_mem = dnnl::memory(inner_product_pd.weights_desc(), eng);
                dnnl::reorder(B_m, inner_product_weights_mem)
                        .execute(stream, B_m, inner_product_weights_mem);
                stream.wait();
                b_md = inner_product_pd.weights_desc();
                m_args[DNNL_ARG_WEIGHTS] = inner_product_weights_mem;
            }

            inner_product_p = dnnl::inner_product_forward(inner_product_pd);
            //std::cout << "inner_product: " << inner_product_pd.impl_info_str() << std::endl;
        } else {
            // Create static shape MatMul primitive
            matmul_pd = dnnl::matmul::primitive_desc(eng, a_md, b_md, c_md, attr);
            matmul_p = dnnl::matmul(matmul_pd);
            std::cout << "matmul: " << matmul_pd.impl_info_str() << std::endl;
        }
        // m_args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS] = alpha_m;
    }

    void run() {
        // Execute the MatMul primitive
        if (constb) {
            inner_product_p.execute(stream, m_args);
        } else {
            matmul_p.execute(stream, m_args);
        }
        stream.wait();
    }
};
