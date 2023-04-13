# Weight compression

AMX's high throughput makes matrix multiplication a memory bounded operation. to reduce memory bound, we can use compressed format for weight matrix and decompress it on-the-fly into the cache. the decompression takes additional computations, thus it's a trade-off between cpu & memory bound. Ideally decompression should just use-up the superfluous/extra computational power wasted on waitting on memory sub-system.

## INT8 compression

On cases where weight matrix is overwhelming bigger than activations, one method is that we can use off-line pre-quantized weight matrix, and dequantize it into bfloat16 format on-the-fly before feeding into AMX BF16 ALU, unlike full quantization algorithm like SmoothQuant[^1], this optimization still uses bfloat16 as runtime precision, the quantization/dequantization of weight marix is more like a simple form of compression/decompression algorithm.

To answer the question - what implementation of this idea can give best performance, we built a extreamly simplified single core benchmark (*which is not an valid matrix multiplication, but it helps to rule out complex factors*)

```c++
void unittest_base(int K)
{
    tensor2D<bfloat16> B(K, 32); // assume each tile of B is already packed as 16x(16x2)
    zero_tiles<0, 1, 2, 3>();
    load_tiles_with_random_bf16<4, 5>();
    benchmark.tag(__func__)([&]() {
        auto * pB0 = &B[0];
        for(int k = 0; k < K; k+=32) {
            _tile_loadd(6, pB0, 64); pB0 += 16*32;     // 1KB tile
            prefetch_bytes<1024>(pB0);
            _tile_dpbf16ps(0, 4, 6);
            _tile_dpbf16ps(1, 5, 6);
            _tile_loadd(7, pB0, 64);  pB0 += 16*32;    // 1KB tile
            prefetch_bytes<1024>(pB0);
            _tile_dpbf16ps(2, 4, 7);
            _tile_dpbf16ps(3, 5, 7);
        } }, 2048.0 * K / 32);
    check_tiles<0, 1, 2, 3>();
}
```

here B matrix is of size Kx32, and we assume each 16 subrows of B is a tile that has been repacked into 16x16x2 layout for being used by `tdpbf16ps` instruction as source operand b.

With quantized/compressed version matrix B, it element type becomes int8_t instead of bfloat16, its shape is remained:


```cpp

inline void dequant_16x32(int8_t *&src, bfloat16 *dst)
{
    for (int k = 0; k < 16; k++) // unrolled by compiler
    {
        auto a = _mm_load_si128((__m128i *)src);        // 16 int8
        auto b = _mm_load_si128((__m128i *)(src + 16)); // 16 int8
        auto a_512 = _mm512_cvtepi8_epi32(a);           // 16 int32
        auto b_512 = _mm512_cvtepi8_epi32(b);           // 16 int32
        auto a_f = _mm512_cvtepi32_ps(a_512);           // 16 ps
        auto b_f = _mm512_cvtepi32_ps(b_512);           // 16 ps
        // a_f = _mm512_mul_ps(a_f, dequant_16x32_dq_scale);   // dequantize
        // b_f = _mm512_mul_ps(b_f, dequant_16x32_dq_scale);   // dequantize
        auto reg_out = _mm512_cvtne2ps_pbh(b_f, a_f); // 32 packed bf16
        _mm512_store_epi32(dst, (__m512i)reg_out);    //
        src += 32;                                    // 32 int8_t dequantized into 32 bf16
        dst += 32;
    }
};

void unittest_Wint8(int K)
{
    tensor2D<int8_t> B(K, 32, true); // assume each tile of B is already packed as 16x(16x2)
    tensor2D<bfloat16> B2buff(16 * 2, 32);
    auto *pB0 = &B2buff(0, 0);  // temp buffer for tile B0 (tmm6)
    auto *pB1 = &B2buff(16, 0);  // temp buffer for tile B1 (tmm7)
    zero_tiles<0, 1, 2, 3>();
    load_tiles_with_random_bf16<4, 5>();
    benchmark.tag(__func__)([&]() {
        auto * pBint = &B[0];
        for(int k = 0; k < K; k+=32) {
            dequant_16x32(pBint, pB0); // 512 bytes int8 => 1KB tile bf16
            prefetch_bytes<512>(pBint); // prefetch tile from next 4KB page
            _tile_loadd(6, pB0, 64);
            _tile_dpbf16ps(0, 4, 6);
            _tile_dpbf16ps(1, 5, 6);

            dequant_16x32(pBint, pB1);
            prefetch_bytes<512>(pBint);
            _tile_loadd(7, pB1, 64);
            _tile_dpbf16ps(2, 4, 7);
            _tile_dpbf16ps(3, 5, 7);
        } }, 1024.0 * K / 32);
    check_tiles<0, 1, 2, 3>();
}
```

We know that `dequant_16x32` increases computations but reduces memory accesses (by half), so it should be fater in memory-bound case and slower in compute-boud case, we can control the value of K to test out these 2 cases:

```bash
# K = 102400*32 = 3276800, sizeof(B)=  200 MB
unittest_base  : 11060 us x 89  HW Usage : 105% (18.9616 GB/s /18 GB/s)
unittest_Wint8  : 6852.85 us x 145  HW Usage : 85% (15.3013 GB/s /18 GB/s)
# K = 16*32 = 512, sizeof(B)=   32 KB
unittest_base  : 0.330489 us x 1010171  HW Usage : 550% (99.1501 GB/s /18 GB/s)
unittest_Wint8  : 0.950541 us x 732268  HW Usage : 95% (17.2365 GB/s /18 GB/s)
```

In `200MB` case, we can see `unittest_Wint8` didn't reach 18~19GB/s peak memory bandwidth as `unittest_base` did, thus it's latency was higher than half of the latency of `unittest_base`, this should be improved.

In `32KB` case, we can see `unittest_Wint8` is significantly slower, it cannot enjoy the bandwidth of L1D due additional computaions in `dequant_16x32`, the computational bound limits the effective memory bandwidth usage to 17GB only.

Notice that `unittest_Wint8()` has similar mmeory bandwidth on both cases which leads me suspecting that `unittest_Wint8()` didn't exploit ALU fully and maybe both two cases are still computational bounded.

To confirm my guess I replaced `dequant_16x32` with `fake_dequant_i8_16x32` which is super-light-weighted, although it breaks correctness & functionality, but we only focus on the performance in this test:

```cpp
inline void fake_dequant_i8_16x32(int8_t *&src, bfloat16 *dst)
{
    for (int k = 0; k < 16; k += 2)  // unrolled by compiler
    {
        auto a = _mm512_load_si512((__m512i *)src); // read 32 bf16
        _mm512_store_si512(dst, a);
        _mm512_store_si512(dst + 32, a);
        src += 64;
        dst += 32 * 2;
    }
};
void unittest_WFakeint8(int K)
{
    tensor2D<bfloat16> B(K, 32, true);
    tensor2D<bfloat16> B2buff(16 * 2, 32);
    auto *pB0 = &B2buff(0, 0);
    auto *pB1 = &B2buff(16, 0);
    zero_tiles<0, 1, 2, 3>();
    load_tiles_with_random_bf16<4, 5>();
    benchmark.tag(__func__)( [&]() {
        auto * pBint = reinterpret_cast<int8_t*>(&B[0]);
        for(int k = 0; k < K; k+=32) {
            fake_dequant_i8_16x32(pBint, pB0);  // 512 bytes => 1KB tile
            prefetch_bytes<512>(pBint);
            _tile_loadd(6, pB0, 64);
            _tile_dpbf16ps(0, 4, 6);
            _tile_dpbf16ps(1, 5, 6);

            fake_dequant_i8_16x32(pBint, pB1);  // 512 bytes => 1KB tile
            prefetch_bytes<512>(pBint);
            _tile_loadd(7, pB1, 64);
            _tile_dpbf16ps(2, 4, 7);
            _tile_dpbf16ps(3, 5, 7);
        } },
        1024.0 * K / 32);
    check_tiles<0, 1, 2, 3>();
}
```

here is the test result:

```bash
# K = 102400*32 = 3276800, sizeof(B)=  200 MB
unittest_base       : 11060 us x 89  HW Usage : 105% (18.9616 GB/s /18 GB/s)
unittest_Wint8      : 6852.85 us x 145  HW Usage : 85% (15.3013 GB/s /18 GB/s)
unittest_WFakeint8  : 6007.1 us x 165  HW Usage : 96% (17.4556 GB/s /18 GB/s)
# K = 16*32 = 512, sizeof(B)=   32 KB
unittest_base       : 0.330489 us x 1010171  HW Usage : 550% (99.1501 GB/s /18 GB/s)
unittest_Wint8      : 0.950541 us x 732268  HW Usage : 95% (17.2365 GB/s /18 GB/s)
unittest_WFakeint8  : 0.885301 us x 653030  HW Usage : 102% (18.5067 GB/s /18 GB/s)
```

Suprisingly, `unittest_WFakeint8` is still so slow in `32KB` case, which suggest that ALU isn't fully exploited.

After some investigation & discussion with Luo,Cheng , we found that our naive implementation has data dependency between neighbouring instruction blocks `fake_dequant_i8_16x32` and `_tile_loadd/_tile_dpbf16ps` which prevents them from beging executed in parallel, so we fixed it with a ping-pong buffer in `unittest_WFakeint8B`:

```cpp

void unittest_WFakeint8B(int K)
{
    tensor2D<bfloat16> B(K+32, 32, true);
    tensor2D<bfloat16> B4buff(16 * 4, 32);
    auto *pB0 = &B4buff(0, 0);
    auto *pB1 = &B4buff(16*2, 0);
    zero_tiles<0, 1, 2, 3>();
    load_tiles_with_random_bf16<4, 5>();
    benchmark.tag(__func__)( [&]() {
        auto * pBint = reinterpret_cast<int8_t*>(&B[0]);
        fake_dequant_i8_16x32(pBint, pB1);
        fake_dequant_i8_16x32(pBint, pB1 + 16*32);
        for(int k = 0; k < K; k+=32) {
            fake_dequant_i8_16x32(pBint, pB0);  // 512 bytes => 1KB tile
            prefetch_bytes<512>(pBint);
            _tile_loadd(6, pB1, 64);
            _tile_dpbf16ps(0, 4, 6);
            _tile_dpbf16ps(1, 5, 6);

            fake_dequant_i8_16x32(pBint, pB0 + 16*32);  // 512 bytes => 1KB tile
            prefetch_bytes<512>(pBint);
            _tile_loadd(7, pB1 + 16*32, 64);
            _tile_dpbf16ps(2, 4, 7);
            _tile_dpbf16ps(3, 5, 7);
            std::swap(pB0, pB1);            //swap ping-pong buffer
        } },
        1024.0 * K / 32);
    check_tiles<0, 1, 2, 3>();
}
```

And we can see `unittest_WFakeint8B` now can enjoy the L2/L1D bandwidth in `32KB` case, which means in `200MB` case it's more likely to be bounded by memory. and we also applied the same optimization to `unittest_Wint8B`, although with extra computations doing real de-quantization, we can still see it uses more bandwidth in `32KB` case, and in `200MB` case it also uses higher bandwidth now and latency reduced by ~10%.

```bash
# K = 102400*32 = 3276800, sizeof(B)=  200 MB
unittest_base       : 11060 us x 89  HW Usage : 105% (18.9616 GB/s /18 GB/s)
unittest_Wint8      : 6852.85 us x 145  HW Usage : 85% (15.3013 GB/s /18 GB/s)
unittest_Wint8B     : 6294.46 us x 159  HW Usage : 92% (16.6587 GB/s /18 GB/s)
unittest_WFakeint8  : 6007.1 us x 165  HW Usage : 96% (17.4556 GB/s /18 GB/s)
unittest_WFakeint8B : 5245.57 us x 178  HW Usage : 111% (19.9897 GB/s /18 GB/s)
# K = 16*32 = 512, sizeof(B)=   32 KB
unittest_base       : 0.330489 us x 1010171  HW Usage : 550% (99.1501 GB/s /18 GB/s)
unittest_Wint8      : 0.950541 us x 732268  HW Usage : 95% (17.2365 GB/s /18 GB/s)
unittest_Wint8B     : 0.73128 us x 408192  HW Usage : 124% (22.4046 GB/s /18 GB/s)
unittest_WFakeint8  : 0.885301 us x 653030  HW Usage : 102% (18.5067 GB/s /18 GB/s)
unittest_WFakeint8B : 0.348609 us x 788344  HW Usage : 261% (46.9982 GB/s /18 GB/s)
```

In `200MB` case, `unittest_Wint8B`'s bandwidth usage is still 10% lower than `unittest_base`, so latency is only reduced by 43% rather than ideally 50%, this is possibly due to non-ideal concurrency of prefetch & computation, because `unittest_WFakeint8B` archives ~50% reduction in latency. need to further analysis on it.

## INT4 compression

INT4 compression should further reduce memory footprint & bandwidth requirement per inference, with possibly more computations in dequantization. We haven't found efficient instructions to implement a fully-functional `dequant()` for INT4, but as what's been done in `unittest_WFakeint8B` we can use a fake dequant implementation (pls check source code for details) to measure a performance upper bound for it:

```bash
# K = 102400*32 = 3276800, sizeof(B)=  200 MB
unittest_base           : 10921.2 us x 91  HW Usage : 106% (19.2026 GB/s /18 GB/s)
unittest_Wint8          : 6979.36 us x 144  HW Usage : 83% (15.0239 GB/s /18 GB/s)
unittest_Wint8B         : 6159.34 us x 162  HW Usage : 94% (17.0242 GB/s /18 GB/s)
unittest_WFakeint8      : 6214.16 us x 166  HW Usage : 93% (16.874 GB/s /18 GB/s)
unittest_WFakeint8B     : 5299.02 us x 187  HW Usage : 109% (19.7881 GB/s /18 GB/s)
unittest_WFakeint4      : 5901.96 us x 174  HW Usage : 49% (8.88329 GB/s /18 GB/s)
unittest_WFakeint4B     : 2752.39 us x 353  HW Usage : 105% (19.0484 GB/s /18 GB/s)
# K = 16*32 = 512, sizeof(B)=   32 KB
unittest_basee          : 0.331108 us x 235865  HW Usage : 549% (98.9647 GB/s /18 GB/s)
unittest_Wint8          : 0.974215 us x 800055  HW Usage : 93% (16.8176 GB/s /18 GB/s)
unittest_Wint8B         : 0.755867 us x 623373  HW Usage : 120% (21.6758 GB/s /18 GB/s)
unittest_WFakeint8      : 0.905821 us x 868547  HW Usage : 100% (18.0875 GB/s /18 GB/s)
unittest_WFakeint8B     : 0.348046 us x 845469  HW Usage : 261% (47.0742 GB/s /18 GB/s)
unittest_WFakeint4      : 0.890886 us x 770130  HW Usage : 51% (9.19534 GB/s /18 GB/s)
unittest_WFakeint4B     : 0.353482 us x 946012  HW Usage : 128% (23.1752 GB/s /18 GB/s)
```

Note that `unittest_WFakeint4B` uses ping-pong buffer while `unittest_WFakeint4` doesn't, we conclude that:

 1. in `200MB` case, INT4 compression can almost remain the same bandwidth as `unittest_base`, thus latency is reduced by ~75%.
 2. `ping-pong` buffer is vital to performance in on-the-fly de-compression implementations.

But as memory bandwidth requirement is futher reduced in INT4 cases, if we cannot find an efficient dequantization algorithm to keep extra computations low, it would unlikely to archieve 75% latency reduction.


## Why ping-pong buffer is important?

Ping-pong buffer is widely used for algorithm optimizations on DSP where DMA is programed to copy data into one of the buffer and ALU is programed to calculate on another buffer filled by DMA in previous iteration.

On CPU it's seldom used because CPU has powerful/smart HW designed to do the trick for you automatically:

 - Reorder execution engine automatically seeks for parallelism between computation instructions from current iteration and data load instructions from next iterations.
 - HW prefetcher predict the data being used for next iteration and start prefetch it early before next iteration begins

These HW designed removed the needs for DMA & Ping-pong buffer for most algorithm to archieve reasonable performance, but in this particular case, we found:

 - due to 1KB tile register capacity and the fact that HW prefetcher cannot prefetch across 4KB page boundary, manually issue SW prefetch instructions is very important, it can increase the performance consistently & obviously in 200MB case.
 - ping-pong buffer works well to solve data dependency issue, so concurrency is increased from 2 to 4:
    - `Dequantize(B0) => AMX_ALU(B0)`
    - `Dequantize(B1) => AMX_ALU(B1)`

   to
    - `Dequantize(B0a)` can start once `AMX_ALU(B0a)` from t-1 is finished
    - `Dequantize(B1a)` can start once `AMX_ALU(B1a)` from t-1 is finished
    - `AMX_ALU(B0b)`    can start once `Dequantize(B0b)` from t-1 is finished
    - `AMX_ALU(B1b)`    can start once `Dequantize(B1b)` from t-1 is finished


[^1]: https://arxiv.org/abs/2211.10438