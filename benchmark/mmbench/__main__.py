from . import dnnl
from . import mkl
from . import mmamx

def run():
    TransB = False
    ConstB = False
    M, N, K = 32, 256, 2560
    duration = 10
    cacheMB = 120
    for i in range(2):
        a = mkl.benchmark(TransB, ConstB, M, N, K , duration, cacheMB)
        b = dnnl.benchmark(TransB, ConstB, M, N, K , duration, cacheMB)
        c = mmamx.benchmark(TransB, ConstB, M, N, K , duration, cacheMB)
        print(f"  mkl: {a}")
        print(f" dnnl: {b}")
        print(f"mmaxm: {c}")

if __name__ == "__main__":
    run()