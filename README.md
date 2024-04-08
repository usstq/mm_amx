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

## Cache blocking

Tile registers are loaded from cache, reuse cached memories as much as possible can further increase load bandwidth.

Since we want to keep sub-matrix C in tile registers all the time w/o store/load into temporary buffers, the cache blocking is basically done in 2 dimensions of C matrix, the loop order of sub-matrix of C determines how we reuse the required memories of A & B sub0-matrixies.

If we only considering L2 cache, we have following cache blocking scheme which divide M dimension into smaller pieces with size m, so sub-row of A with size m⨯K can fit into L2 cache totally, to be reused to generate a m⨯N sub-row of result matrix C.

![cache_blk.png](./cache_blk.png)


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


```bash
amx_jit_ (M=_256_,N=_256_,K=_512_)_[PASS]       : 117.97 us x 11, HW_CYCLES=396832 169.11(Ops/cycle)
amx_jit_ (M=_256_,N=_256_,K=_1024_)_[PASS]      : 189.19 us x 11, HW_CYCLES=599602 223.84(Ops/cycle)
amx_jit_ (M=_256_,N=_256_,K=_2048_)_[PASS]      : 273.17 us x 11, HW_CYCLES=835525 321.28(Ops/cycle) <======= 
amx_jit_ (M=_512_,N=_512_,K=_1024_)_[PASS]      : 501.86 us x 11, HW_CYCLES=1549072 346.58(Ops/cycle) <======= 
amx_jit_ (M=_256_,N=_256_,K=_4096_)_[PASS]      : 685.72 us x 11, HW_CYCLES=2115576 253.77(Ops/cycle)
```

 - when A/B fits L2 cache, prefer bigger M/N than K, to get higher Ops/byte.
 - when B matrix cannot fit into L2, Ops/cycle would reduce.

thus we prefer to split along K dimension to fit L2 (to preserve M/N dimension size for higher Ops/byte).

using 32x32 kernel (2x2 tiles):

 - we need to access a new (32xK) A sub-rows for each (32xKxN) Multiply-Adds, so memory access vs MAdds is 1:N.
 - we need to access whole (N*K) B matrix for (MxKxN) MAdds, so memory access vs MAdds is 1:M.

A can be prefetched row by row. but B must be prefetched in whole (because it's being reused/accessed as a whole).

SW prefetch instructions can also block CPU pipeline if there are too many of them, so we have to evenly distribute them into kernel.
but how many prefetches is required in each (32x32) AMX kernel is determined by: `32*K*sizeof(bf16)/(K/32)/P = (2048/P)`, here P is the number of (32x32) kernel invocations which maybe a variable rather than a compile time constant.


## Multicore parallelism



## Refernces

[^1]: chap 20 - "Intel® 64 and IA-32 Architectures Optimization Reference Manual"

[^2]: Table 2-7. Cache Parameters of the Ice Lake Client Microarchitecture - "Intel® 64 and IA-32 Architectures Optimization Reference Manual"

[^brgemm]: High-Performance Deep Learning via a Single Building Block