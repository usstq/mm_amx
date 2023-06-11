from . import dnnl
from . import mkl
from . import mmamx

def run():
    TransB = False
    ConstB = False
    for i in range(2):
        a = mkl.benchmark(TransB, ConstB, 256,256,2560,10,120)
        b = dnnl.benchmark(TransB, ConstB, 32,256,2560,10,120)
        c = mmamx.benchmark(TransB, ConstB, 32,256,2560,10,120)
        print(f"  mkl: {a}")
        print(f" dnnl: {b}")
        print(f"mmaxm: {c}")

if __name__ == "__main__":
    run()