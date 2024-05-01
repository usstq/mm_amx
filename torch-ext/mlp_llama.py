import torch
from torch import nn
from collections import OrderedDict
import time

class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU() #ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

class config:
    def __init__(self, hidden_size, intermediate_size):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

cfg = config(4096, 11008)

m = OrderedDict()
for i in range(32):
    m[f"mlp{i}"] = LlamaMLP(cfg)
model = nn.Sequential(m)

device = "cpu"
model = model.to(device)
#print(model)
print(torch.__version__)


N=256
X = torch.rand(N, cfg.hidden_size, device=device)

print("========== bf16 =============")
with torch.cpu.amp.autocast(cache_enabled=True):
    # warm-up
    Y = model(X)
    Y = model(X)
    t0 = time.time()
    Y = model(X)
    dt = (time.time() - t0)
    print(f"{dt*1e3:.3f} ms")

print("========== fp32 =============")
for i in range(3):
    t0 = time.time()
    Y = model(X)
    dt = time.time() - t0
    print(f"{i}: {dt*1e3:.3f} ms")

print("========== bf16 =============")
with torch.cpu.amp.autocast(cache_enabled=True):
    # warm-up
    Y = model(X)
    Y = model(X)
    t0 = time.time()
    Y = model(X)
    dt = (time.time() - t0)
    print(f"{dt*1e3:.3f} ms")



'''

# Are oneDNN performance reasonable?

`ONEDNN_VERBOSE=2` shows torch 2.2.2+cpu's oneDNN version info:

```bash
$ ONEDNN_VERBOSE=2 numactl -C56-111 -l python ./mlp_llama.py

onednn_verbose,info,oneDNN v3.3.2 (commit 2dc95a2ad0841e29db8b22fbccaf3e5da7992b01)
onednn_verbose,info,cpu,runtime:OpenMP,nthr:56
onednn_verbose,info,cpu,isa:Intel AVX-512 with float16, Intel DL Boost and bfloat16 support and Intel AMX with bfloat16 and 8-bit integer support
```

for a fake 32-layers LLama7b style MLP only model, we saw best performance of bfloat16 inference of 121 ms happens after a float32 precesion inference warm-up,
for which no explainations found. per layer latency is around 1.1ms~1.3ms (can be found in verbose log):

```bash
$ ONEDNN_VERBOSE=2 numactl -C56-111 -l python ./mlp_llama.py
# 121/32=3.78 ms per layer
# 1.25+1.34+1.1 = 3.69ms
   ... ...
onednn_verbose,primitive,exec,cpu,matmul,brg:avx512_core_amx,undef,src_bf16::blocked:ab::f0 wei_bf16::blocked:ba::f0 dst_bf16::blocked:ab::f0,attr-scratchpad:user ,,256x11008:11008x4096,1.21704
onednn_verbose,primitive,exec,cpu,matmul,brg:avx512_core_amx,undef,src_bf16::blocked:ab::f0 wei_bf16::blocked:ba::f0 dst_bf16::blocked:ab::f0,attr-scratchpad:user ,,256x4096:4096x11008,1.25293
onednn_verbose,primitive,exec,cpu,matmul,brg:avx512_core_amx,undef,src_bf16::blocked:ab::f0 wei_bf16::blocked:ba::f0 dst_bf16::blocked:ab::f0,attr-scratchpad:user ,,256x4096:4096x11008,1.34912
   ... ...
onednn_verbose,primitive,exec,cpu,matmul,brg:avx512_core_amx,undef,src_bf16::blocked:ab::f0 wei_bf16::blocked:ba::f0 dst_bf16::blocked:ab::f0,attr-scratchpad:user ,,256x11008:11008x4096,1.10205
onednn_verbose,primitive,exec,cpu,matmul,brg:avx512_core_amx,undef,src_bf16::blocked:ab::f0 wei_bf16::blocked:ba::f0 dst_bf16::blocked:ab::f0,attr-scratchpad:user ,,256x4096:4096x11008,1.31396
onednn_verbose,primitive,exec,cpu,matmul,brg:avx512_core_amx,undef,src_bf16::blocked:ab::f0 wei_bf16::blocked:ba::f0 dst_bf16::blocked:ab::f0,attr-scratchpad:user ,,256x4096:4096x11008,1.28687
   ... ...
```

all 3 Linear layers in MLP has same FLOPS: `256*4096*11008*2`, according to what we saw from ONEDNN_VERBOSE, the actual GFLOPS/second/core is low:

 - 256*4096*11008*2/1.25e-3/1e9/56 = 330 GFLOps/second/core
 - 256*4096*11008*2/1.34e-3/1e9/56 = 307 GFLOps/second/core
 - 256*4096*11008*2/1.0e-3/1e9/56  = 412 GFLOps/second/core


if we only use single core, 2629.163/32/3 = 27ms per layer, `256*4096*11008*2/27e-3/1e9 = 855 GFLOps/second/core`

So multi-core parallel is not doing well, which is the key

for the particular problem: N=256, IC=4096, OC=11008,

we will combine/interleave weights of up_proj & gate_proj into pairs, and post-process can be done in unit of 32x32 (2x2 tiles)
for example, first 32x16 column is up_proj, second 32x16 columns are gate_proj, then we can do `self.act_fn(self.gate_proj(x)) * self.up_proj(x)`
in block level, after this combination, result can be converted into a 32x16 bfloat16.

to avoid interleaved cross-core store, we just save result of each core into it's own per-thread local buffer, this makes down_proj's input
activation a special format (src is a linked-list, each block is 256x256), this is not a problem, down_proj


down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
'''
