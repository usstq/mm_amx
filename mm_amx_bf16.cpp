#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <vector>
#include <chrono>
#include <iostream>
#include <fstream>
#include <cassert>

//#include "thread_pool.hpp"
#include "bf16.hpp"

// g++-11 ./test_conv.cpp -O2 -lpthread -march=native && ./a.out

// to use VNNI, we need higher version of compiler:
//    clang-9 ./test_conv.cpp -O2 -lpthread -march=native -lstdc++ && ./a.out

// to use AMX, we need intel compiler 
//   source  ~/intel/oneapi/setvars.sh
//   icx ./mm_amx_bf16.cpp -O2 -lpthread -march=native -lstdc++

// objdump -C -S ./a.out > a.asm

#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

//===============================================================
using ov::bfloat16;

template<typename T>
struct tensor2D {
    int dims[2];
    T * data;
    int stride;
    tensor2D(int d0, int d1) {
        dims[0] = d0;
        dims[1] = d1;
        data = new T[d0*d1]();
        for(int i = 0; i<d0*d1; i++) {
            // lower mantissa can help to avoid small errors in accuracy comparison
            data[i] = (rand() % 4)/4.0 - 0.5;
        }
        stride = d1 * sizeof(T);
    }
    T & at(int i0, int i1) {
        return data[i0 * dims[1] + i1];
    }
    const T & at(int i0, int i1) const {
        return data[i0 * dims[1] + i1];
    }
    void operator=(const T & v) {
        for(int k = 0; k<dims[0]*dims[1]; k++) {
            data[k] = v;
        }
    }
    bool operator==(const tensor2D<T> & rhs) {
        bool ok = true;
        if (dims[0] != rhs.dims[0] || dims[1] != rhs.dims[1])
            return false;
        for(int i0=0; i0<dims[0]; i0++)
        for(int i1=0; i1<dims[1]; i1++) {
            if (at(i0,i1) != rhs.at(i0,i1)) {
                std::cout << " operator== failed at (" << i0 << ", " << i1 << ")  value "
                          << at(i0,i1) << "!=" << rhs.at(i0,i1) << std::endl;
                ok = false;
            }
        }
        return ok;
    }
    friend std::ostream& operator<<(std::ostream& out, const tensor2D<T>& obj) {
        int i0;
        auto showline = [&](int i) {
            out << "[" << i << "," << 0 << "]: ";
            int i1;
            for(i1=0; i1<obj.dims[1] && i1 < 8; i1++) {
                out << obj.at(i0,i1) << ",";
            }
            if (i1 < obj.dims[1]) out << "...";
            out << std::endl;
        };
        for(i0=0; i0 < obj.dims[0] && i0 < 8; i0++) {
            showline(i0);
        }
        if (i0 < obj.dims[0]) {
            out << "... ... ... ..." << std::endl;
            showline(obj.dims[0] - 1);
        }
        return out;
    }
};

void matmul(tensor2D<bfloat16> & A, tensor2D<bfloat16> & B, tensor2D<bfloat16> & C) {
    int M = C.dims[0];
    int N = C.dims[1];
    int K = A.dims[1];
    assert(B.dims[0] == K);
    assert(B.dims[1] == N);
    for(int m = 0; m < M; m++) {
        for(int n = 0; n < N; n++) {
            float sum = C.at(m,n);
            int k;
            for (k = 0; (k + 32) <= K; k += 32) {
                float psum0 = 0;
                float psum1 = 0;
                for(int p = 0; p < 32; p+=2) {
                    psum0 += static_cast<float>(A.at(m,k+p)) * static_cast<float>(B.at(k+p,n));
                    psum1 += static_cast<float>(A.at(m,k+p+1)) * static_cast<float>(B.at(k+p+1,n));
                }
                sum += (psum0 + psum1);
            }
            for(; k < K; k++) {
                sum += static_cast<float>(A.at(m,k)) * static_cast<float>(B.at(k,n));
            }
            C.at(m,n) = sum;
        }
    }
}

//===============================================================
#ifndef _GNU_SOURCE
#define _GNU_SOURCE         /* See feature_test_macros(7) */
#endif
#include <unistd.h>
#include <sys/syscall.h>   /* For SYS_xxx definitions */

#define XFEATURE_XTILECFG 17
#define XFEATURE_XTILEDATA 18
#define XFEATURE_MASK_XTILECFG (1 << XFEATURE_XTILECFG)
#define XFEATURE_MASK_XTILEDATA (1 << XFEATURE_XTILEDATA)
#define XFEATURE_MASK_XTILE (XFEATURE_MASK_XTILECFG | XFEATURE_MASK_XTILEDATA)
#define ARCH_GET_XCOMP_PERM 0x1022
#define ARCH_REQ_XCOMP_PERM 0x1023

bool initXTILE() {
    unsigned long bitmask = 0;
    long status = syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &bitmask);
    if (0 != status) return false;
    if (bitmask & XFEATURE_MASK_XTILEDATA) return true;

    status = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
    if (0 != status)
        return false; // XFEATURE_XTILEDATA setup is failed, TMUL usage is not allowed
    status = syscall(SYS_arch_prctl, ARCH_GET_XCOMP_PERM, &bitmask);

    // XFEATURE_XTILEDATA setup is failed, can't use TMUL
    if (0 != status || !(bitmask & XFEATURE_MASK_XTILEDATA)) return false;

    // XFEATURE_XTILEDATA set successfully, TMUL usage is allowed
    return true;
}

//===============================================================
template<typename Callable>
double timeit(int times, const Callable & c) {
    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < times; i++) {
        c();
    }
    auto finish = std::chrono::high_resolution_clock::now();
    auto avg_useconds = std::chrono::duration_cast<std::chrono::microseconds>(finish-start).count()/times;
    std::cout << "latency per iterattion  = " << avg_useconds << " us\n";
    return avg_useconds;
}

//===============================================================
/*
https://www.intel.com/content/www/us/en/develop/documentation/cpp-compiler-developer-guide-and-reference/top/compiler-reference/intrinsics/intrinsics-for-amx-instructions/intrinsics-for-amx-tile-instructions/tile-loadconfig.html

void _tile_loadconfig (const void * mem_addr)
	format of memory payload. each field is a byte.
		 0: palette_id
		 1: startRow (8b)
	 2-15: reserved (must be zero)
	16-17: tile0.colsb -- bytes_per_row
	18-19: tile1.colsb
	20-21: tile2.colsb
			...
	46-47: tile15.colsb
		48: tile0.rows
		49: tile1.rows
		50: tile2.rows
			 ...
		63: tile15.rows

void _tile_storeconfig (void * mem_addr)
    Stores the current tile configuration to a 64-byte memory location specified by "mem_addr".
    The tile configuration format is specified below, and includes the tile type pallette,
    the number of bytes per row, and the number of rows. If tiles are not configured,
    all zeroes will be stored to memory.
*/
struct tileconfig_t {
    uint8_t palette_id;
    uint8_t startRow;
    uint8_t reserved[14];
    uint16_t cols[16];
    uint8_t rows[16];
    tileconfig_t() = default;
    tileconfig_t(int palette, int _startRow, int numTiles, int _rows, int columnsBytes) {
        palette_id = palette;
        startRow = _startRow;
        for(int i = 0; i < 14; i++) {
            reserved[i] = 0;
        }
        for(int i = 0; i < numTiles; i++) {
            cols[i] = columnsBytes;
            rows[i] = _rows;
        }
        for(int i = numTiles; i < 16; i++) {
            cols[i] = 0;
            rows[i] = 0;
        }
    }

    void load() {
        _tile_loadconfig(this);
    }
    void store() {
        _tile_storeconfig(this);
    }
    friend std::ostream& operator<<(std::ostream& out, const tileconfig_t& cfg) {
        out << "tile config" << std::endl;
        out << "    palette_id :" << static_cast<int>(cfg.palette_id) << std::endl;
        out << "    startRow   :" << static_cast<int>(cfg.startRow) << std::endl;
        for (int i = 0;i < 16;i++) {
            if (cfg.rows[i] == 0 && cfg.cols[i] == 0)
                continue;
            out << "    tile" << i << " : " << static_cast<int>(cfg.rows[i]) << " x " << static_cast<int>(cfg.cols[i]) << std::endl;
        }
        return out;
    }
} __attribute__ ((__packed__));

/*
void _tile_loadd (__tile dst, const void * base, int stride)
    Load tile rows from memory specifieid by "base" address and "stride"
    into destination tile "dst" using the tile configuration previously
    configured via "_tile_loadconfig".
    Operation:
        start := tileconfig.startRow
        IF start == 0 // not restarting, zero incoming state
            tilezero(dst)
        FI
        nbytes := dst.colsb
        DO WHILE start < dst.rows
            memptr := base + start * stride
            write_row_and_zero(dst, start, read_memory(memptr, nbytes), nbytes)
            start := start + 1
        OD
        zero_upper_rows(dst, dst.rows)
        zero_tileconfig_start()

void _tile_stored (__tile src, void * base, int stride)
    Store the tile specified by "src" to memory specifieid by "base" address and "stride"
    using the tile configuration previously configured via "_tile_loadconfig".
    Operation:
        start := tileconfig.startRow
        DO WHILE start < src.rows
            memptr := base + start * stride
            write_memory(memptr, src.colsb, src.row[start])
            start := start + 1
        OD
        zero_tileconfig_start()

void _tile_stream_loadd (__tile dst, const void * base, int stride)
    Load tile rows from memory specifieid by "base" address and "stride"
    into destination tile "dst" using the tile configuration previously
    configured via "_tile_loadconfig". This intrinsic provides a hint to
    the implementation that the data will likely not be reused in the near
    future and the data caching can be optimized accordingly.

void _tile_zero (__tile tdest)
    Zero the tile specified by "tdest".
    Operation:
        nbytes := palette_table[tileconfig.palette_id].bytes_per_row
        FOR i := 0 TO palette_table[tileconfig.palette_id].max_rows-1
            FOR j := 0 TO nbytes-1
                tdest.row[i].byte[j] := 0
            ENDFOR
        ENDFOR
	

void _tile_release ()
    Release the tile configuration to return to the init state, which releases all storage it currently holds.


Instruction Throughput Latency
LDTILECFG                204
STTILECFG                19
TILETRELEASE             13
TDP / *          16      52
TILELOADD         8      45
TILELOADDT1      33      48
TILESTORED       16
TILEZERO          0      16

Due to the high latency of the LDTILECFG instruction we recommend issuing a single pair
of LDTILECFG and TILERELEASE operations per Intel AMX-based DL layer implementation.


• A-tiles can have between 1-16 rows and 1-MAX_TILE_K columns.
• B-tiles can have between 1-MAX_TILE_K rows and 1–16 columns.
• C-tiles can have between 1-16 rows and 1–16 columns.

MAX_TILE_K=64/sizeof(type_t)
          = 32 BF16
          = 64 INT8

A tiles and B tiles contain data of type_t, which can be (u)int8 or bfloat16.
• C tiles contain data of type res_type_t:
• int32 if type_t=(u)int8
• float if type_t=bfloat16

Like the Intel® DL Boost use case, the B matrix must undergo a re-layout before it can be used within the
corresponding Intel AMX multiply instruction.

BF16    C_float_16x16 = A_bfloat16_16x32 * B_bfloat16_32x16 (re-layout as 16x16x2 Ab2a)
INT8    C_int32_16x16 = A_int8_16x64 * B_int8_64x16 (re-layout as 16x16x4 Ab4a)



FC:
    (2,1~900,2560)x(2560,7680)
    (2,1~900,2560)x(2560,2560)
    (2,1~900,2560)x(2560,10240) GELU
    (2,1~900,10240)x(10240,2560)

matmul:
    (1~990,80) * (80,1~990)
    (1~990,1~990) * (1~990,80)

    // (2,32,1~990,80)*(2,32,80,1~990)
    // (2,32,1~990,1~990)(2,32,1~990,80)
*/

template<typename T, int N>
struct PPBuffer {
    static const int tile_ele_cnt = 16*64/(sizeof(T));
    T buffer[tile_ele_cnt*N];

    __m512i midx;
    PPBuffer() {
        static const uint64_t idx[8] = {0,4,1,5,2,6,3,7};
        midx = _mm512_loadu_epi64(idx);
    }

    T * tile(int i) {
        return buffer + i*tile_ele_cnt;
    }

    void relayout_2B(const T * _src, int stride) {
        // in 64unit
        //
        //  [a1 a2 a3 a4 | a5 a6 a7 a8]
        //  [b1 b2 b3 b4 | b5 b6 b7 b8]
        // _mm512_permutexvar_epi64
        //  [a1 a5 a2 a6 a3 a7 a4 a8]
        //  [b1 b5 b2 b6 b3 b7 b4 b8]
        // _mm512_unpacklo_epi16 works in 128 lanes, & means interleave
        //  [a1&b1 a2&b2 a3&b3 a4&b4]
        // _mm512_unpackhi_epi16 
        //  [a5&b5 a6&b6 a7&b7 a8&b8]
        //
        //
        const auto * src = reinterpret_cast<const int8_t *>(_src);
        auto * pB0 = reinterpret_cast<int8_t *>(tile(0));
        auto * pB1 = reinterpret_cast<int8_t *>(tile(1));
        for (int row = 0; row < 16; row ++) {
            auto a = _mm512_loadu_epi16(src);
            auto b = _mm512_loadu_epi16(src + stride);
            a = _mm512_permutexvar_epi64(midx, a);
            b = _mm512_permutexvar_epi64(midx, b);
            auto rowB0 = _mm512_unpacklo_epi16(a, b);
            auto rowB1 = _mm512_unpackhi_epi16(a, b);
            _mm512_storeu_epi16(pB0, rowB0);
            _mm512_storeu_epi16(pB1, rowB1);
            pB0 += 64;
            pB1 += 64;
            src += 2*stride;
        }
    }
};

void matmul_amx(tensor2D<bfloat16> & A, tensor2D<bfloat16> & B, tensor2D<bfloat16> & C) {
    int M = C.dims[0];
    int N = C.dims[1];
    int K = A.dims[1];
    assert(B.dims[0] == K);
    assert(B.dims[1] == N);

    tileconfig_t tfg(1, 0, 8, 16, 64);
    tfg.load();

    // use 2x2 register blocking to generate 32 x 32 float accumulation results:
    //
    //     B0  B1
    //  A0 C00  C01
    //  A1 C10  C11
    //
    // cache blocking: loop order is M,N,K
    //
    //
    PPBuffer<bfloat16, 2> buffB;

    const int C00 = 0, C01 = 1, C10 = 2, C11 = 3, A0 = 4, A1 = 5, B0 = 6, B1 = 7;
    for(int m = 0; m < M; m+=32) {
        for(int n = 0; n < N; n+=32) {
            _tile_zero(C00);
            _tile_zero(C01);
            _tile_zero(C10);
            _tile_zero(C11);
            for (int k=0; k < K; k+=32) {
                _tile_loadd(A0, &A.at(m, k), A.stride);
                buffB.relayout_2B(&B.at(k, n), B.stride);
                _tile_loadd(B0, buffB.tile(0), 64);
                _tile_dpbf16ps(C00, A0, B0);
                _tile_loadd(A1, &A.at(m + 16, k), A.stride);
                _tile_dpbf16ps(C10, A1, B0);
                _tile_loadd(B1, buffB.tile(1), 64);
                _tile_dpbf16ps(C01, A0, B1);
                _tile_dpbf16ps(C11, A1, B1);
            }

            auto * pC = &C.at(m, n);

            // C is float, save to buffer and do post processing
            // then store to memory
            auto * pbuffC = reinterpret_cast<float *>(buffB.tile(0));
            _tile_stored(C00, pbuffC, 64*2);
            _tile_stored(C01, pbuffC + 16, 64*2);
            auto * psrc = pbuffC;
            for(int i = 0; i < 16; i ++) {
                auto b = _mm512_loadu_epi16(psrc);
                auto a = _mm512_loadu_epi16(psrc + 16);
                auto c = _mm512_cvtne2ps_pbh(a, b);
                _mm512_storeu_epi16(pC, c);
                pC += C.dims[1];
                psrc += 32;
            }

            _tile_stored(C10, pbuffC, 64*2);
            _tile_stored(C11, pbuffC + 16, 64*2);
            psrc = pbuffC;
            for(int i = 0; i < 16; i ++) {
                auto b = _mm512_loadu_epi16(psrc);
                auto a = _mm512_loadu_epi16(psrc + 16);
                auto c = _mm512_cvtne2ps_pbh(a, b);
                _mm512_storeu_epi16(pC, c);
                pC += C.dims[1];
                psrc += 32;
            }
        }
    }
    _tile_release();
}

int amx_unit_test() {
    int M = 16;
    int K = 32;
    int N = 16;
    tensor2D<bfloat16> A(M, K);
    tensor2D<bfloat16> B(K, N);
    tensor2D<bfloat16> C0(M, N);
    tensor2D<bfloat16> C1(M, N);
    //B = 1.0f;
    matmul(A, B, C0);
    
    initXTILE();
    const int C00 = 0, C01 = 1, C10 = 2, C11 = 3, A0 = 4, A1 = 5, B0 = 6, B1 = 7;
    tileconfig_t tfg(1, 0, 8, 16, 64);
    tfg.load();
    PPBuffer<bfloat16, 2> buffB;

    _tile_zero(C00);
    _tile_loadd(A0, &A.at(0, 0), A.stride);
    buffB.relayout_2B(&B.at(0, 0), B.stride);
    _tile_loadd(B0, buffB.tile(0), 64);
    _tile_dpbf16ps(C00, A0, B0);
    _tile_stored(C00, buffB.tile(0), 64);
    auto * pbuffC = reinterpret_cast<float *>(buffB.tile(0));
    for(int m = 0; m < M; m++)
    for(int n = 0; n < N; n++) {
        std::cout << "[" << m << "," << n << "]: " << C0.at(m, n) << " vs " << pbuffC[m*16 + n] << std::endl;
    }
    _tile_release();
    return 0;
}

int main(int argc, const char *argv[]) {
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);

    //return amx_unit_test();

    int M = 320;
    int K = 320;
    int N = 320;
    tensor2D<bfloat16> A(M, K);
    tensor2D<bfloat16> B(K, N);
    tensor2D<bfloat16> C0(M, N);
    tensor2D<bfloat16> C1(M, N);

    C0 = 0.0f;
    C1 = 0.0f;
    matmul(A, B, C0);
    //matmul(A, B, C1);

    // AMX
    initXTILE();

    timeit(1000, [&](){
        matmul_amx(A, B, C1);
    });

    if(C0 == C1) {
        std::cout << "Correct: C0=\n" << C0 << std::endl;
    } else {
        std::cout << " Wrong C0!=C1" << std::endl;
    }

    return 0;
}
