# Why benchmark?

Competitions lead to progress

# Why yet another library for matmul?

Intrinsic based, simple to understand, simple to reuse, refactor & redesign.

# oneMKL

install through pip

```bash
pip install mkl-include>=2021.0.0
pip install mkl-static>=2021.0.0
# all headers
pip show mkl-include -f
```

# oneDNN

install through pip

```bash
pip install onednn-cpu-gomp
pip install onednn-devel-cpu-gomp
# all headers
pip show onednn-devel-cpu-gomp -f
```

benchmark is triggered through python, test result is collected back by python, and charts are generated.

benchmark is run following steps:
 - loop:
    - clear cache by memcpy
    - read TSC clock into t0
    - run matmul
    - read TSC clock into t1
    - record t1-t0
 - compare result with reference for correctness

all latency & correctness are returned back to python

