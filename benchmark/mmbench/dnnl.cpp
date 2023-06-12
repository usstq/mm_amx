#include "utils.hpp"
#include "oneapi/dnnl/dnnl.hpp"

// MatMul_onednn<dnnl::memory::data_type::bf16>
// MatMul_onednn<dnnl::memory::data_type::f32>
struct MatmulTaskDNNL : public MatmulTask {
    using MatmulTask::MatmulTask;

    dnnl::engine eng;
    dnnl::matmul::primitive_desc matmul_pd;
    dnnl::matmul matmul_p;
    dnnl::stream stream;
    dnnl::memory::data_type dt;

    // whether use dynamic shape support
    bool dyn_mode;

    void init() override {
        eng = dnnl::engine(dnnl::engine::kind::cpu, 0);
        stream = dnnl::stream(eng);
        dt = dnnl::memory::data_type::bf16;
        dyn_mode = false;

        initialize_args();
    }

    void init_dynamic_primitive()
    {
        dnnl::memory::dims a_shape = {DNNL_RUNTIME_DIM_VAL, DNNL_RUNTIME_DIM_VAL};
        dnnl::memory::dims b_shape = {DNNL_RUNTIME_DIM_VAL, DNNL_RUNTIME_DIM_VAL};
        dnnl::memory::dims c_shape = {DNNL_RUNTIME_DIM_VAL, DNNL_RUNTIME_DIM_VAL};

        // we don't know if A,B is transposed, so both dimensions are dynamic
        dnnl::memory::dims a_strides_ = {DNNL_RUNTIME_DIM_VAL, DNNL_RUNTIME_DIM_VAL};
        dnnl::memory::dims b_strides_ = {DNNL_RUNTIME_DIM_VAL, DNNL_RUNTIME_DIM_VAL};

        // we known the row-major C matrix is not transposed, so inner strides is 1
        dnnl::memory::dims c_strides_ = {DNNL_RUNTIME_DIM_VAL, 1};

        dnnl::memory::desc a_md(a_shape, dt, a_strides_);
        dnnl::memory::desc b_md(b_shape, dt, b_strides_);
        dnnl::memory::desc c_md(c_shape, dnnl::memory::data_type::f32, c_strides_);

        // Create attributes (to handle alpha dynamically and beta if necessary)
        dnnl::primitive_attr attr;
        attr.set_scales_mask(DNNL_ARG_WEIGHTS, /* mask */ 0);

        // Create a MatMul primitive
        matmul_pd = dnnl::matmul::primitive_desc(eng, a_md, b_md, c_md, attr);
        matmul_p = dnnl::matmul(matmul_pd);
    }

    std::unordered_map<int, dnnl::memory> m_args;
    // static shape
    void initialize_args()
    {
        dnnl::memory::dims a_shape = {m, k};
        dnnl::memory::dims b_shape = {k, n};
        dnnl::memory::dims c_shape = {m, n};

        dnnl::memory::dims a_strides = (!transa) ? dnnl::memory::dims{A.padded_dim1, 1} : dnnl::memory::dims{1, A.padded_dim1};
        dnnl::memory::dims b_strides = (!transb) ? dnnl::memory::dims{B.padded_dim1, 1} : dnnl::memory::dims{1, B.padded_dim1};
        dnnl::memory::dims c_strides = dnnl::memory::dims{C.padded_dim1, 1};

        dnnl::memory::desc a_md(a_shape, dt, a_strides);
        dnnl::memory::desc b_md(b_shape, dt, b_strides);
        dnnl::memory::desc c_md(c_shape, dnnl::memory::data_type::f32, c_strides);

        // Prepare oneDNN memory for alpha
        // memory alpha_m({{1}, memory::data_type::f32, {1}}, eng, &alpha);

        // Create attributes (to handle alpha dynamically and beta if necessary)
        dnnl::primitive_attr attr;
        //attr.set_scales_mask(DNNL_ARG_WEIGHTS, /* mask */ 0);

        if (!dyn_mode)
        {
            // Create static shape MatMul primitive
            matmul_pd = dnnl::matmul::primitive_desc(eng, a_md, b_md, c_md, attr);
            matmul_p = dnnl::matmul(matmul_pd);
        }
        else
        {
            // dynamic mode MatMul primitive only create once
            init_dynamic_primitive();
        }

        std::cout << matmul_pd.impl_info_str() << std::endl;

        dnnl::memory A_m(a_md, eng, &A[0]);
        dnnl::memory B_m(b_md, eng, &B[0]);
        dnnl::memory C_m(c_md, eng, &C[0]);

        m_args.clear();
        m_args[DNNL_ARG_SRC] = A_m;
        m_args[DNNL_ARG_WEIGHTS] = B_m;
        m_args[DNNL_ARG_DST] = C_m;
        // m_args[DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS] = alpha_m;
    }

    void run() override {
        // Execute the MatMul primitive
        matmul_p.execute(stream, m_args);
        stream.wait();
    }
};

PYBIND11_MODULE(dnnl, m)
{
    m.def("benchmark", [](bool transB, bool constB, int M, int N, int K,float duration, int cache_MB){
        MatmulTaskDNNL task("dnnl", false, transB, constB, M, N, K, duration, cache_MB);
        return task.benchmark();
    });
}
