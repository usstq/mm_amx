from . import dnnl
from . import mkl
from . import mmamx

import argparse

import time
from hwcounter import Timer, count, count_end
tsc_cycles_per_sec = 0
start = count()
time.sleep(1)
tsc_cycles_per_sec = count_end() - start
print(f"tsc_cycles_per_sec={tsc_cycles_per_sec/1e9} G")

def test_torch(M, N, K, duration, cacheMB, checkCorrect):
    import torch
    with torch.no_grad():
        if cacheMB > 0:
            clrc0 = torch.randn(int(cacheMB*1024*1024/4)).float()
            clrc1 = torch.randn(int(cacheMB*1024*1024/4)).float()
        
        tensor1 = torch.randn(M, K).bfloat16()
        tensor2 = torch.randn(K, N).bfloat16()
        tensor3 = torch.randn(M, N).bfloat16()
        # warmup
        torch.matmul(tensor1, tensor2, out = tensor3)
        torch.matmul(tensor1, tensor2, out = tensor3)
        
        # estimate repeat times
        start = count()
        if cacheMB > 0:
            clrc1 = torch.clone(clrc0)
        torch.matmul(tensor1, tensor2, out = tensor3)
        est_lat = count_end() - start
        times = int(duration * tsc_cycles_per_sec / est_lat)

        total_cycles = 0
        t0 = time.time()
        for i in range(times):
            # clear cache
            if cacheMB > 0:
                clrc1 = torch.clone(clrc0)

            start = count()
            torch.matmul(tensor1, tensor2, out = tensor3)
            total_cycles += float(count_end() - start)/tsc_cycles_per_sec
        latency = total_cycles/times
    return {'latency_ms' : latency * 1e3, 'times' : times, 'duration': time.time() - t0}

def main(args):
    TransB = False
    constb = args.constb
    if (len(args.mnk) == 0):
        M, N, K = 32, 10240, 10240
    else:
        M, N, K = args.mnk[0],args.mnk[1],args.mnk[2]

    duration = args.duration
    cacheMB = args.cacheMB
    checkCorrect = args.check
    print(f"benchmark on M,N,K=[{M},{N},{K}] constb={constb} duration={duration} cacheMB={cacheMB}")

    for i in range(2):
        a = mkl.benchmark(TransB, constb, M, N, K , duration, cacheMB, checkCorrect)
        b = dnnl.benchmark(TransB, constb, M, N, K , duration, cacheMB, checkCorrect)
        c = mmamx.benchmark(TransB, constb, M, N, K , duration, cacheMB, checkCorrect)
        d = test_torch(M, N, K, duration, cacheMB, checkCorrect)
        print(f"  mkl: {a}")
        print(f" dnnl: {b}")
        print(f"mmaxm: {c}")
        print(f"torch: {d}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
    parser.add_argument('-c', '--cacheMB', type=int, default=0)
    parser.add_argument('-d', '--duration', type=float, default=10)
    parser.add_argument('--constb', action="store_true")
    parser.add_argument('--check', action="store_true")
    parser.add_argument('--mnk', nargs="+", type=int)
    
    args = parser.parse_args()
    
    main(args)
