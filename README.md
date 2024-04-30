# GEMM(Matmul) using AMX(BF16)

## GEMM Formula

$$
C_{ij} = \sum_{k} A_{ik} \cdot B_{kj}
$$

With:

 - A matrix as MxK
 - B matrix as KxN
 - C matrix as MxN

Note above formula is usually generalized to matrix blocking case where:

 - element $A_{ik}$ becomes a sub-matrix of A with size m⨯k
 - element $B_{kj}$ becomes a sub-matrix of B with size k⨯n
 - element $C_{ij}$ becomes a sub-matrix of C with size m⨯n
 - the multiply $\cdot$ becomes matrix multiplcation $A_{ij}^{m⨯k} \cdot B^{k⨯n}_{kj}$

with this generalization, orginal definition becomes a special case where :

 - m=n=k=1
 - sub-matrix is degenerated into scalar
 - multiply $\cdot$ becomes a scalar multiplication.

![mm_basic.svg](./mm_basic.svg)

consider each dot in above graph as an element, which is sub-matrix in general.

## Baisc calculation scheme

As shown in Intel manual[^1], AMX can do sub-matrix multiplication & accumulation with tile registers (which has 2D layout instead 1D/vector registers used in SIMD).

As brgemm paper [^brgemm] suggested, we should choose the size of sub-matrix of C so it can fit tile register(s), and load & store it only once, before & after all accumulations into it were done.

Thus the basic flow is, for each sub-matrix $C_{ij}$, we do what specified in GEMM Formula:

 1. load $C_{ij}$ into tile registers
 2. go over the dimension K:
    - load sub-matrix $A_{ik}$, $B_{kj}$
    - do sub-matrix multiplication & accumulation of $C_{ij} += A_{ik} \cdot B_{kj}$
 3. store $C_{ij}$ into memory

Step 2 is a reduce-procedure which is usually the computational heavy part or hot-spot, and it's heavily memory bounded, given that the throughput of TDP* instruction (*the AMX matmul & accumulation instruction*) is 16 cycles, during which:

 - L1D can load (64x2)x16 = 2KB which is 2 tiles

   there is almost no such use case which can guarantee tiles are always loaded from L1D
 
 - L2 can load (48)x16 = 768 Bytes which is 75% tile [^2]
  
   according to what we saw in extreamly simplified sample `Linear32x32_AMX`:
   - strided tile load is slower than dense/compat tile load
   - 74.8% tile load can be archieved if both A&B are loaded in compact
   - 65.8% tile load can be archieved if only B is loaded in compact
   - when A & B matrix are small enough to hold inside L2 cache, SW prefetching into L1 cache hurts performance.
 - LLC can load (21)x16 = 336 Bytes which is 32.8% tile [^2]
   
   according to what we saw in extreamly simplified sample `Linear32x32_AMX`:
   - only 11%~15% tile loads per 16 cycles from LLC (thus AMX usage is only 11%~15%)

 - 8-channel 4800MT/s DDR can load (4.8e9 x 8 x 8/2e9)/56 x 16 = 43.8 Bytes/core @2GHz CPU frequency on chip with 56-cores, which is 4.2% tile

so we should load less tiles in order to perform a single TDP* instruction, which can be done by register blocking.

## Register blocking

consider using single tile for C submatrix, then we need load one tile A and one tile B for each TDP* computation `C+=A*B`, this gives 2 tile-loads per TDP*, similarly:

 - 1x2 blocking: `1A*2B=>2C`, 3/2 tile loads per TDP*
 - 1x4 blocking: `1A*4B=>4C`, 5/4 tile loads per TDP*
 - 2x2 blocking: `2A*2B=>4C`, 4/4 tile loads per TDP*

given limited number of HW tile registers available, 2x2 blocking is best in terms of number of tile loads per TDP*.

so in bf16 case:

 - 4 tiles arranged in 2x2 represent 32x32 float32 sub-matrix of C
 - 2 tiles in 2x1 represents 32x16 bf16 sub-matrix of A
 - 2 tiles in 1x2 represents 16x32 bf16 sub-matrix of B

### [amx-L2](./tests/amx-l2.cpp)

this test shows that when blocked A matrix can archieve higher AMX usage while normal layout A matrix cannot.
since the test didn't flush cache and all A/B/C can be resident in L2, so L1D (DCU) Hardware Prefetcher (Stream Prefetcher)
will has major impact on the performance.

to explain why blocked A is faster, we use [tile-load.cpp](./tests/tile-load.cpp) and we found it's related to the stride of A, as
[Tip6](https://www.intel.com/content/www/us/en/developer/articles/technical/a-simple-example-to-measure-the-performance-of-an-intel-mkl-function.html)
pointed out, there are 32 cache lines to be accessed by 2 tileloads of A tiles, we can get highest bandwidth if they are not located in same cache-way
and according to the test, this can only happen if stride contains odd number of cache-line (not just avoid power of 2, but any multiple of 2).

we can see :
 - padding stride of A matrix to odd number of cache line size can boost performance to level of blocked A matrix!
 - this is important only when your algorithm consumed a lot of L2 bandwidth, if only A matrix is load, it's OK, if we add B matrix, it's getting worse.
 - performance is much more stable (and fast) if we disable L1D HW prefetcher by `MSRCONFIG=0x1a4,4 sudo -E numactl -C56 -l ./a.out`
   so here the prefetcher is somehow interfere the performance a little.

this odd-number of cache line stride is not easy to satisfy, consider Llama's MLP, the feature dimension do not satisfy this: `4096*2/64=128`,`11008*2/64=344`.

```bash
$ numactl -C56 -l ./a.out 
ENV: USE_NUMA = 0
ENV: MSRCONFIG = 0
initXTILE success!
rdtsc is calibrating ... done.
================[ perf_log : test_L2_256_256_256_[PASS]_padK_256 ]================
   #  thr    latency, HW_CYCLES,  CPU(GHz), Ops/cycle,  GOps/sec,BOUND_ON_LOADS
   0     0   73.86us,    187615,      2.54,       178,    454.29,    764707
   1     0   40.53us,     80160,      1.98,       418,    827.88,    358720
   2     0   30.38us,     51812,      1.71,       647,   1104.41,    228481
   3     0   28.40us,     48441,      1.71,       692,   1181.39,    213362
   4     0   28.25us,     48203,      1.71,       696,   1187.71,    212530
[WARNING] K padded from 256 to 288
================[ perf_log : test_L2_256_256_256_[PASS]_padK_288 ]================
   #  thr    latency, HW_CYCLES,  CPU(GHz), Ops/cycle,  GOps/sec,BOUND_ON_LOADS
   0     0   67.42us,    172852,      2.56,       194,    497.67,    689249
   1     0   31.55us,     61087,      1.94,       549,   1063.57,    270380
   2     0   26.36us,     44969,      1.71,       746,   1272.86,    195132
   3     0   26.43us,     45090,      1.71,       744,   1269.65,    195626
   4     0   26.46us,     45167,      1.71,       742,   1268.00,    195793

================[ perf_log : prepareA_256_256_256 ]================
   #  thr    latency
   0     0   54.61us
   1     0    3.47us
   2     0    3.60us
   3     0    3.49us
================[ perf_log : test_L2_blocked_256_256_256_[PASS] ]================
   #  thr    latency, HW_CYCLES,  CPU(GHz), Ops/cycle,  GOps/sec,BOUND_ON_LOADS
   0     0   96.36us,    239086,      2.48,       140,    348.20,   1011280
   1     0   26.68us,     46479,      1.74,       721,   1257.80,    201350
   2     0   26.36us,     44966,      1.71,       746,   1272.86,    195108
   3     0   26.64us,     45461,      1.71,       738,   1259.48,    197757
   4     0   26.47us,     45157,      1.71,       743,   1267.41,    196012
```
so as long as A&B&C can fit in L2, AMX usage is good, and padding is helpful (by 10%).

with K fixed, M & N has to be big enough to make compute-bound possible (see below), but not too big
to oveflow the L2 cache size.


| M, N (K=256)      |   GOps/sec | Ops/cycle | with padK       |
| :---------------- | :--------: | --------: | :-------------: |
| 128, 128          |    1192.21 |     685   | 1221.74/710     |
| 128, 256          |    1218.87 |   **710** | **1272.50/740** |
| 128, 512          |    1286.36 |   **711** | **1292.71/715** |
|  |  |  | |
| 256, 128          |    1200.93 |     697   | 1217.02/711     |
| 256, 256          |    1299.47 |   **719** | **1315.57/728** |
| 256, 512          |    1262.48 |     702   | 1292.02/716     |
|  |  |  | |
| 512, 256          |    1293.38 |   **713** | 1285.31/712     |
| 512, 512          |    1136.08 |     628   | 1186.56/658     |

when run on 1 SPR socket with 56 cores, we should focus on `GOps/second` since CPU frequency is changing (while L2 bandwidth is not?):
 - each core has it's own copy of A/B/C & jit kernel.
 - all cores run same 256x256x256 kernels for 2 times for warm-up.
 - all cores run same 256x256x256 kernels for 10 times for measurement, (L2-miss is almost zero).
 - only 1136/1127/1090 `GOps/sec` can be reached for each core in blocked/odd-CL-stride/normal-stride case
 - thus we can say L2 AMX kernel's peak-Glops upper-compute-bound (per 56-cores socket) is 1136*56/1e3 = 63 `Tflops/sec`
   which is bounded by L2 bandwidth.

## Cache blocking

Throughput of Memory hierarchy:

```bash
@BufferSize    15 K : 380.22 GB/s  x 1
@BufferSize    30 K : 457.55 GB/s  x 1
@BufferSize     1 M : 181.53 GB/s  x 1
@BufferSize  1.25 M : 158.51 GB/s  x 1
@BufferSize   1.5 M : 127.43 GB/s  x 1
@BufferSize  1.75 M : 91.819 GB/s  x 1
@BufferSize     2 M : 62.778 GB/s  x 1
@BufferSize  2.25 M : 50.531 GB/s  x 1
@BufferSize   2.5 M : 41.677 GB/s  x 1
@BufferSize  2.75 M : 37.173 GB/s  x 1
@BufferSize    13 M : 31.123 GB/s  x 1
@BufferSize    56 M : 31.116 GB/s  x 1
@BufferSize   128 M : 23.965 GB/s  x 1
@BufferSize   512 M : 14.299 GB/s  x 1
@BufferSize     1 G : 13.363 GB/s  x 1
@BufferSize     2 G :    13 GB/s  x 1
```

Fist let's think about compute-memory ratio of a general matmul problem, to see how can a MatMul become compute-bounded:

 - we need to access a new `(MxK)` A matrix for `(2*MxKxN)` Flops, so memory access vs Flops is 1:2N = 1/(2N):1
 - we need to access whole `(NxK)` B matrix for `(2*MxKxN)` Flops, so memory access vs Flops is 1:2M = 1/(2M):1
 - in total 1 Flops needs `[1/(2N) + 1/(2M)]` elements, or `(1/N + 1/M)` bytes in BF16 format.
 - AMX peak MAdds per cycle is 1024 Flops/cycle, with CPU frequency 1.8 GHz (when all cores are busy), 1843.2 GFlops
 - due to limitation of L2 bandwidth, 60~70% of peak GFlops can be archieved, if A is in normal layout, only 1242 GFlops can be reached (67%)
 - as a comparison, AVX512 FP32 peak Gflops is 64 Flops/cycle with CPU frequency 2.6 GHz, 166 GFlops,
   so in practice we should expect AMX's thoughput to be 1242/166 ~ **7.5X** of AVX512.
 - per core DDR bandwidth is BW/cores, which generate `BW/cores/(1/N + 1/M)` GFlops computations for each core.
 - so AMX L2-bounded ALU usage is `BW/cores/(1/N + 1/M)/1242`

| M, N              | total BW (GB/s) |  GFlops | AMX Usage(L2) |
| :---------------- | :-------------: | ------: | ------------: |
| 256, 256          |         260     |   594   |     48%       |
| 512, 512          |         260     |  1188   |     95%       |
| 256, 256          |         520     |  1188   |     95%       |

Thus we need to keep M/N big to get better AMX usage, so for cache-blocking in single core, we shouldn't parallel by splitting along M & N dimension,
we should split slong K dimension in single core when doing cache blocking, so in each sub-block [M x BK] [BK x N] => [M, N] we can get better AMX Usage.

sub-matrix A can be prefetched row by row. but sub-matrix B must be prefetched in whole (because it's being reused/accessed as a whole).

### With B sub-block prefetched [amx-ddr](./tests/amx-ddr.cpp)

with prefetch of B matrix added, we have 602 Ops/cycle in cache-COLD which is slower than L2 cache-HOT case (660)
which means prefetch is not completely hidden by computation.

```bash
# ============================
   #  thr    latency, HW_CYCLES,  CPU(GHz), Ops/cycle,  GOps/sec,BOUND_ON_LOADS,ICACHE_DATA.STALLS,ICACHE_TAG.STALLS
   0     0   65.18us,    161405,      2.48,       207,    514.78,    640797,       259,       184
   1     0   30.86us,     52677,      1.71,       636,   1087.15,    230414,         0,        71
   2     0   49.96us,     67786,      1.36,       495,    671.64,    254693,       104,        77
   3     0   30.87us,     52688,      1.71,       636,   1086.96,    230482,         0,         0
   4     0   30.84us,     52606,      1.71,       637,   1088.09,    230884,         0,         0
   5     0   30.95us,     52809,      1.71,       635,   1084.19,    230935,         0,         0
   6     0   30.45us,     52637,      1.73,       637,   1101.79,    229837,         0,         0
   7     0   29.21us,     52760,      1.81,       635,   1148.55,    230968,         0,         0
   8     0   29.22us,     52758,      1.81,       636,   1148.33,    231265,         0,         0
   9     0   29.12us,     52595,      1.81,       637,   1152.14,    230490,         0,         0
#================================
  10     0   51.47us,     96009,      1.87,       349,    651.98,    402421,       125,      2245
  11     0   30.39us,     54912,      1.81,       611,   1104.12,    241408,        24,       122
  12     0   30.47us,     55065,      1.81,       609,   1101.26,    242417,         0,         0
  13     0   30.32us,     54791,      1.81,       612,   1106.67,    241386,         0,         0
  14     0   30.37us,     54869,      1.81,       611,   1104.89,    241890,         0,         0
  15     0   30.32us,     54785,      1.81,       612,   1106.55,    241155,         0,         0
  16     0   30.62us,     55323,      1.81,       606,   1095.95,    243948,         0,         0
  17     0   39.34us,     60638,      1.54,       553,    852.86,    250099,       155,         0
  18     0   31.09us,     56152,      1.81,       597,   1079.19,    247694,         0,         0
  19     0   30.64us,     55361,      1.81,       606,   1095.00,    243810,         0,         0
# ============================
  20     0  507.68us,    917373,      1.81,       585,   1057.49,   4035017,         0,        56
```
we can see the first (256x256) kernel execution take 70us (doubled) latency, due to ICACHE miss & DCACHE miss.
and this explains the overall average Ops/cycle is only `585` (`(600*15+348)/16 ~ 584`).
 - all cache hit: `636` Ops/cycle
 - prefetch miss: `610` Ops/cycle
 - average      : `585` Ops/cycle


On multi-core case, all 56 cores perform:
 - same kernel 
 - same A
 - different set of (16) B sub-blocks
 - different C
 - no prefetch of A since A is reused for each B and it should be in L2 cache.

we need to care about `GOps/sec` instead of `Ops/cycle` since CPU frequency changes a lot,
but DDR bandwidth is stable, so it limits the `Ops/cycle`, thus we focus on `GOps/sec` which
is directly sensible by user.

| M, N, num_of_B    | Cores |  GOps/sec/core |
| :---------------- | :---: | -------------: |
| 256, 256 16 1st   | 56    |    643         |
| 256, 256 16       | 56    |    959         |

we can see first time execution (w/o warmup) took significantly more time, `BOUND_ON_LOADS` & `STALLS_L2_MISS` is also higher.
It's due to that we only use 16 256x256 bf16 B matrix for the experiment above which is only 2MB per core, the data is located
in DDR for first time, and it's been loaded into L3 after that.

if we increase the number of 256x256 B matrix to 160, this gap is much lower:

| M, N  num_of_B    | Cores |  GOps/sec/core |
| :---------------- | :---: | -------------: |
| 256, 256 160 1st  | 56    |    916         |
| 256, 256 160      | 56    |    992         |

we also see the Gops was higher, since the penalty of the first cache-cold B matrix was amortized over much more number of following B matrixes.
(but in reality or practical problem maybe we don't have so many B matrixes to amortize the first cache-cold cost).

the average latency is 5.4ms, consider the total size of B matrixes loaded, effective DDR bandwidth consumed is `160*256*256*2/5.4e-3/1e9*56 ~= 217 GB/s`,
which didn't reach the peak,  if we remove the computation instruction `tdpbf16ps` out of the jit kernel, we can get much better DDR bandwidth `160*256*256*2/4.7e-3/1e9*56 ~= 250 GB/s`,
but we also got much higher CPU frequency (~2.8GHz) in this case since the power-consuming AMX ALU is not working. so prefetch & ALU is not 100% in parallel.

 - ALU usage didn't reach L2-bound `700 Ops/cycle`;
 - DDR bandwidth didn't reach `260 GB/s` peak;

the prefetch instructions have been distributed into the inner loop evenly, what can be done more?

### With A&B sub-block prefetched [amx-mm](./tests/amx-mm.cpp)

| M, N  num_of_B    | Cores |  GOps/sec/core               |
| :---------------- | :---: | ---------------------------: |
| 256, 256 43 1st   | 56    |  769 / 463 / 645 / 705 / 623 | 
| 256, 256 43       | 56    |  953 / 549 / 867 / 864 / 695 |

> GOps/sec/core : `common 1x256x256 A` / `per-thread 43x256x256 A` / `common 43x256x256 A` / `common 256x11008 A` / `+prefetcA`
so prefetch of A actually not working well. we disable it by default.

## Multicore parallelism

suppose there are enough output channels which can be split evenly among all cores (after splitting, each core still got a N which is big enough to reach high AMX Usage).
we prefer split along output channels.

Reading A matrix can be perfectly shared by all cores, which means, when all cores are reading the same A matrix, it will be read into L3 cache only once and shared by all cores.

```bash
# test_bw
========== clflush 1 ===========
MULTI_2097_KBytes_32768_CacheLines_56_threads   : 449.78 us x 1, HW_CYCLES=1259880 CPU~2.80GHz 1.66(Ops/cycle), L2_HIT=5843, L3_HIT=12, L3_MISS=26981 4.66(GOps/s)
MULTI_2097_KBytes_32768_CacheLines_56_threads   : 82.51 us x 1, HW_CYCLES=231293 CPU~2.80GHz 9.07(Ops/cycle), L2_HIT=18016, L3_HIT=14759, L3_MISS=36 25.42(GOps/s)
MULTI_2097_KBytes_32768_CacheLines_56_threads   : 85.79 us x 1, HW_CYCLES=235733 CPU~2.75GHz 8.90(Ops/cycle), L2_HIT=18459, L3_HIT=14288, L3_MISS=2 24.45(GOps/s)
MULTI_2097_KBytes_32768_CacheLines_56_threads   : 75.00 us x 1, HW_CYCLES=210435 CPU~2.81GHz 9.97(Ops/cycle), L2_HIT=19465, L3_HIT=13337, L3_MISS=0 27.96(GOps/s)
MULTI_2097_KBytes_32768_CacheLines_56_threads   : 68.22 us x 1, HW_CYCLES=191388 CPU~2.81GHz 10.96(Ops/cycle), L2_HIT=20214, L3_HIT=12578, L3_MISS=1 30.74(GOps/s)
========== clflush 0 ===========
SAME_2097_KBytes_32768_CacheLines_56_threads    : 220.97 us x 1, HW_CYCLES=619189 CPU~2.80GHz 3.39(Ops/cycle), L2_HIT=6067, L3_HIT=17622, L3_MISS=6332 9.49(GOps/s)
SAME_2097_KBytes_32768_CacheLines_56_threads    : 64.63 us x 1, HW_CYCLES=175747 CPU~2.72GHz 11.93(Ops/cycle), L2_HIT=17491, L3_HIT=15219, L3_MISS=0 32.45(GOps/s)
SAME_2097_KBytes_32768_CacheLines_56_threads    : 48.08 us x 1, HW_CYCLES=134953 CPU~2.81GHz 15.54(Ops/cycle), L2_HIT=19206, L3_HIT=13627, L3_MISS=0 43.62(GOps/s)
SAME_2097_KBytes_32768_CacheLines_56_threads    : 44.57 us x 1, HW_CYCLES=125638 CPU~2.82GHz 16.69(Ops/cycle), L2_HIT=20503, L3_HIT=12332, L3_MISS=0 47.05(GOps/s)
SAME_2097_KBytes_32768_CacheLines_56_threads    : 42.80 us x 1, HW_CYCLES=120191 CPU~2.81GHz 17.45(Ops/cycle), L2_HIT=21429, L3_HIT=11391, L3_MISS=0 49.00(GOps/s)
```

we can see:
 - after clflush 1 & 0, 56-threads read from DDR is much faster when they are reading SAME 2MB memory since only copy of it was required to load into LLC.
 - when they are cached in LLC, `SAME` case is still faster than `MULTI` case by factor of two, although L3_MISS is zero in both cases.
   this means LLC ring topology has some "broadcast" capability?

Reading B matrix cannot be shared since they are not the same block, so whole DDR bandwidth is divided amoung cores.

C matrix is written

# [cross-core-read.cpp](./tests/cross-core-read.cpp)

cross-core data access involves many concepts:
 - [MESI protocol](https://en.wikipedia.org/wiki/MESI_protocol)
 - [MESIF protocol](https://www.realworldtech.com/common-system-interface/5/)
 - [intel Uncore programming guide](323535-intel-xeon-processor-7500-series-uncore.pdf)
   - CPU core is composed of ALU/FPU,L1/L2 cache;
   - all access to shared LLC is directed to a C-Box(LLC coherent engine) via ring interconnet;
   - there is a proprietary hashing algorithm maps target physical addresses into target C-Box slice
     to keep the traffic across the C-Box instances relatively uniform for a wide range of possible
     address patterns.
   - C-Box is responsible for maintaining coherence between:
     - cores within same socket sharing same LLC
     - generating snoops & collecting snoop responses when MESI protocol requires
     - cross-socket coherent through QPI

 - [L3 slice/cache](https://repositories.lib.utexas.edu/server/api/core/bitstreams/15430f7d-4595-4669-9473-21c0706a08a9/content)
 - [Snoop filter events](https://hadibrais.wordpress.com/2019/04/25/considering-snoops-when-counting-cache-hit-and-miss-events-on-client-processors/)
   - XSNP_MISS   : Retired load instructions whose data sources were L3 hit and cross-core snoop missed in on-pkg core cache
   - XSNP_NO_FWD : Retired load instructions whose data sources were L3 and cross-core snoop hits in on-pkg core cache
                   (data was `shared`/`clean` in another core's local cache)
   - XSNP_FWD    : Retired load instructions whose data sources were HitM responses from shared L3
                   (data was `exclusivly`/`dirty` owned by another core's local cache)
   - XSNP_NONE   : Retired load instructions whose data sources were hits in L3 without snoops required

 - [Data_Sharing](https://github.com/andikleen/pmu-tools/blob/4b5d2e4f677317e00cbbee47f48c3d590e8db42b/spr_server_ratios.py#L1832)
   - L2 to L2 data transfer is even slower than L3_HIT (near DDR latency), XSNP_FWD/XSNP_NO_FWD events can measure it.
   - from experiments, the 1st/2nd core access/share same cache-line with 0th core will trigger this penalty.
     and also this penalty brings data into L3 cache (as an optimization attemp) so the rest cores will load from L3 w/o suffering the same penalty.
   - case 1:
      - when core0 read it's own data, no snoop overhead incurs at all. (`Exclusive` in core0)
      - when core1 read the same data as core0 just read: XSNP_FWD happens and the data is filled into L3.  (`Shared` in core0 & core1)
   - case 2:
      - when core0 generate/write it's own data, no snoop overhead. (`Modified` in core0)
      - when core1 read the same data that core0 has just produced: XSNP_FWD happens (`Eclusive` in core1, `Invalid` in core0)
      - when core2 read the same data again, XSNP_NO_FWD happens and data is filled into L3 (`Shared` in core1&2)

```bash
$ numactl -C0-4 -l ./a.out
========= a common 128KB buffer read by other cores one-by-one ==========
 thread  id  : latency, XSNP_MISS, XSNP_NO_FWD, XSNP_FWD, XSNP_NONE
 thread[  0] :   1.96us,        0,        0,        0,        8
 thread[  0] :   1.71us,        0,        0,        0,        0
 thread[  0] :   1.33us,        0,        0,        0,        0
 thread[  1] :  11.89us,        0,        0,     1413,        0
 thread[  1] :   1.23us,        0,        0,        0,        0
 thread[  1] :   1.25us,        0,        0,        0,        0
 thread[  2] :  11.06us,        0,     1331,        0,        3
 thread[  2] :   1.23us,        0,        0,        0,        4
 thread[  2] :   1.23us,        0,        0,        0,        0
 thread[  3] :   4.99us,        0,        0,        0,     1261
 thread[  3] :   1.71us,        0,        0,        0,        1
 thread[  3] :   1.72us,        0,        0,        0,        0
 thread[  0] :   4.60us,        0,        0,        0,     1329
 thread[  0] :   1.22us,        0,        0,        0,        1
 thread[  0] :   1.13us,        0,        0,        0,        0
 thread[  1] :   1.33us,        0,        0,        0,        0
 thread[  1] :   1.24us,        0,        0,        0,        0
 thread[  1] :   1.19us,        0,        0,        0,        0
 thread[  2] :   1.33us,        0,        0,        0,        0
 thread[  2] :   1.15us,        0,        0,        0,        0
 thread[  2] :   1.26us,        0,        0,        0,        0
 thread[  3] :   1.72us,        0,        1,        0,        0
 thread[  3] :   1.19us,        0,        0,        0,        0
 thread[  3] :   1.23us,        0,        0,        0,        0
 thread[  0] :   1.31us,        0,        0,        0,        0
 thread[  0] :   1.22us,        0,        0,        0,        0
 thread[  0] :   1.14us,        0,        0,        0,        0
 thread[  1] :   1.30us,        0,        0,        0,        0
 thread[  1] :   1.20us,        0,        0,        0,        0
 thread[  1] :   1.24us,        0,        0,        0,        0
 thread[  2] :   1.28us,        0,        0,        0,        0
 thread[  2] :   1.24us,        0,        0,        0,        0
 thread[  2] :   1.19us,        0,        0,        0,        0
 thread[  3] :   1.77us,        0,        0,        0,        0
 thread[  3] :   1.20us,        0,        0,        0,        0
 thread[  3] :   1.20us,        0,        0,        0,        0
 thread[  4] :   5.17us,        0,        0,        0,     1273
 thread[  4] :   1.23us,        0,        0,        0,        2
 thread[  4] :   1.24us,        0,        0,        0,        0
======== concurrent multi-threads reading from a common 128K buffer written by thread0 ===========
 thread  id  : latency, XSNP_MISS, XSNP_NO_FWD, XSNP_FWD, XSNP_NONE
 thread[  0] :  10.43us,        0,       31,       48,      922
 thread[  3] :  11.27us,        0,       65,      135,      709
 thread[  2] :  12.41us,        0,       40,      219,      499
 thread[  4] :  12.97us,        0,       40,       80,      375
 thread[  1] :  10.69us,        0,       36,       89,      821
 thread[  4] :   1.27us,        0,        0,        0,       22
 thread[  3] :   1.26us,        0,        0,        0,       10
 thread[  0] :   1.24us,        0,        0,        0,        9
 thread[  2] :   1.29us,        0,        0,        0,        6
 thread[  1] :   1.44us,        0,        0,        0,        0
 thread[  4] :   1.24us,        0,        0,        1,        0
 thread[  3] :   1.29us,        0,        0,        0,        1
 thread[  0] :   1.20us,        0,        2,        0,        0
 thread[  1] :   1.24us,        0,        1,        0,        1
 thread[  2] :   1.31us,        0,        0,        0,        1
```

# [interleave-write.cpp](./tests/interleave-write.cpp)

multi-cores write to same big buffer in interleaving style is slow
it's not false-sharing since each core writes in unit of cache-line size (64B) aligned at cache-line boundary.
when interleaving step is bigger than 4KB, the speed is recovered most.

```bash
#
full_256_x_1024_Bytes_2_Threads : 22.78 us x 100, HW_CYCLES=88188 CPU~3.87GHz 5.95(Ops/cycle), L2_HIT=8, L3_HIT=0, L3_MISS=0 23.01(GOps/s)
part_256_x_1024_Bytes_2_Threads : 5.24 us x 100, HW_CYCLES=19801 CPU~3.78GHz 26.48(Ops/cycle), L2_HIT=18, L3_HIT=0, L3_MISS=0 100.14(GOps/s)
full_256_x_2048_Bytes_2_Threads : 20.38 us x 100, HW_CYCLES=78522 CPU~3.85GHz 13.35(Ops/cycle), L2_HIT=3, L3_HIT=0, L3_MISS=0 51.46(GOps/s)
part_256_x_2048_Bytes_2_Threads : 9.43 us x 100, HW_CYCLES=36178 CPU~3.84GHz 28.98(Ops/cycle), L2_HIT=12, L3_HIT=0, L3_MISS=0 111.25(GOps/s)
full_256_x_4096_Bytes_2_Threads : 19.46 us x 100, HW_CYCLES=75123 CPU~3.86GHz 27.92(Ops/cycle), L2_HIT=11, L3_HIT=0, L3_MISS=0 107.78(GOps/s)
part_256_x_4096_Bytes_2_Threads : 17.86 us x 100, HW_CYCLES=68997 CPU~3.86GHz 30.39(Ops/cycle), L2_HIT=12, L3_HIT=0, L3_MISS=0 117.43(GOps/s)
```

## Refernces

[^1]: chap 20 - "Intel® 64 and IA-32 Architectures Optimization Reference Manual"

[^2]: Table 2-7. Cache Parameters of the Ice Lake Client Microarchitecture - "Intel® 64 and IA-32 Architectures Optimization Reference Manual"

[^brgemm]: High-Performance Deep Learning via a Single Building Block