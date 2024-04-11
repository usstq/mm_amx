import torch
from tqdm import tqdm
import copy
import time

# need import torch before this ext, otherwise:
# ImportError: libc10.so: cannot open shared object file: No such file or directory
import mlp_opt

print(dir(mlp_opt))

torch.manual_seed(0)

def check_acc(M=32, N=32, K=32):
    m = torch.nn.Linear(K, N, bias=False)
    m.weight.data = torch.randint(-10, 11, (N, K), dtype=torch.float32)

    n = mlp_opt.LinearNxN()

    x = torch.randint(-10, 11, (M, K), dtype=torch.float32)
    xbf16 = x.to(dtype=torch.bfloat16)

    y0 = m(x)
    y1 = torch.empty((M, N), dtype=torch.float)

    n.set_weight(m.weight.data.to(dtype=torch.bfloat16))
    n.forward(xbf16, y1)

    allclose = torch.allclose(y0, y1)
    print(f"M={M},N={N},K={K} ", "pass" if allclose else "failed")
    if not allclose:
        print("y0=", y0)
        print("y1=", y1)

check_acc(32, 32, 32)
check_acc(32, 32, 4096)
check_acc(512, 128, 4096)

class Config(object):
    opt = False
    clflush = False
    hidden_size = 4096
    intermediate_size = 11008
    pass

class LlamaMLP(torch.nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()

        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = torch.nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = torch.nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = torch.nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

        def init_int_float(linear):
            return
            linear.weight.data = torch.randint(-10, 11, linear.weight.data.shape, dtype=torch.float32)

        init_int_float(self.gate_proj)
        init_int_float(self.up_proj)
        init_int_float(self.down_proj)
        self.act_fn = torch.nn.SiLU()

        def new_LinearNxN(org_linear):
            opt_linear = mlp_opt.LinearNxN()
            opt_linear.set_weight(org_linear.weight.data.to(dtype=torch.bfloat16))
            return opt_linear

        self.opt_gate_proj = new_LinearNxN(self.gate_proj)
        self.opt_up_proj = new_LinearNxN(self.up_proj)
        self.opt_down_proj = new_LinearNxN(self.down_proj)

        self.opt_gate_proj.name = f"gate_proj_{layer_id}"
        self.opt_up_proj.name = f"up_proj_{layer_id}"
        self.opt_down_proj.name = f"down_proj_{layer_id}"
        self.opt = False

    def flops(self, M):
        x = 0
        x += M * self.hidden_size * self.intermediate_size * 2 # gate_proj
        x += M * self.hidden_size * self.intermediate_size * 2 # up_proj
        x += M * self.hidden_size * self.intermediate_size * 2 # down_proj
        return x

    def numel(self, M):
        x = 0
        x += M * self.hidden_size + self.gate_proj.weight.numel() + self.up_proj.weight.numel()
        x += M * self.intermediate_size + self.down_proj.weight.numel()
        x += M * self.hidden_size
        return x

    def set_id(self, layer_id):
        self.opt_gate_proj.name = f"gate_proj_{layer_id}"
        self.opt_up_proj.name = f"up_proj_{layer_id}"
        self.opt_down_proj.name = f"down_proj_{layer_id}"

    def forward(self, x):

        if self.config.clflush:
            mlp_opt.clflush(x)

        if not self.config.opt:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
            return down_proj

        batch_size = x.shape[0]

        xbf16 = x.to(torch.bfloat16)

        y0 = torch.empty((batch_size, self.intermediate_size), dtype=torch.float)
        y1 = torch.empty((batch_size, self.intermediate_size), dtype=torch.float)
        self.opt_gate_proj.forward(xbf16, y0)
        self.opt_up_proj.forward(xbf16, y1)

        down_proj = torch.empty((batch_size, self.hidden_size), dtype=torch.float)
        self.opt_down_proj.forward((self.act_fn(y0) * y1).to(torch.bfloat16), down_proj)
        return down_proj


def check_acc_llama7b_MLP(M=256):
    llama7b_config = Config()

    mlp = LlamaMLP(llama7b_config)

    K = llama7b_config.hidden_size
    x = torch.randint(-1, 2, (M, K), dtype=torch.float32)

    llama7b_config.do_opt = False
    with torch.no_grad():
        y00 = mlp.forward(x)

    with torch.backends.mkldnn.verbose(torch.backends.mkldnn.VERBOSE_ON):
        with torch.cpu.amp.autocast():
            with torch.no_grad():
                y0 = mlp.forward(x)
    y1 = y0.to(torch.float)

    llama7b_config.do_opt = True
    y2 = mlp.forward(x)

    rtol=1e-05
    atol=1e-08

    atol=1e-02

    print(" allclose(y1, y00) = ", torch.allclose(y1, y00, atol=atol, rtol=rtol))

    allclose = torch.allclose(y2, y00, atol=atol, rtol=rtol)
    if not allclose:
        print("y00=", y00)
        print("y1=", y1)
        print("y2=", y2)
    print(f"llama7b_MLP M={M} ", "pass" if allclose else "failed")

check_acc_llama7b_MLP(M=32)



class LlamaRMSNorm(torch.nn.Module):
    def __init__(self, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return hidden_states.to(input_dtype)

def perf_llama7b_MLP(M=256, num_layers=32, rounds=10, clflush = False):
    llama7b_config = Config()
    mlp = LlamaMLP(llama7b_config)

    seqNet = torch.nn.Sequential()
    for i in tqdm(range(num_layers)):
        seqNet.append(LlamaRMSNorm())
        m = copy.deepcopy(mlp)
        m.set_id(i)
        m.config = llama7b_config
        seqNet.append(m)

    x = torch.randint(-1, 2, (M, llama7b_config.hidden_size), dtype=torch.float32)

    llama7b_config.clflush = clflush

    llama7b_config.opt = False
    t0 = time.time()
    with torch.cpu.amp.autocast():
        with torch.no_grad():
            for i in tqdm(range(rounds)):
                y0 = seqNet(x)
    dt = (time.time() - t0)/rounds/num_layers
    print(f"    per-layer dt: {dt}")
    print(f"    GFlops: {mlp.flops(M) * 1e-9/ dt}")
    print(f"    GB/s  : {mlp.numel(M) * 2 * 1e-9/ dt}")
    
    y0 = y0.to(torch.float)

    llama7b_config.opt = True
    t0 = time.time()
    for i in tqdm(range(rounds)):
        y1 = seqNet(x)
    dt = (time.time() - t0)/rounds/num_layers
    print(f"    per-layer dt: {dt}")
    print(f"    GFlops: {mlp.flops(M) * 1e-9/ dt}")
    print(f"    GB/s  : {mlp.numel(M) * 2 * 1e-9/ dt}")


    print(torch.allclose(y1, y0))
    print("y0=", y0)
    print("y1=", y1)

perf_llama7b_MLP(M=256, num_layers=32, clflush = False)