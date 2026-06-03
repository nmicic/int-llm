/*
 * bench_int64.cu — Optimized INT64 GPU GEMM spike for TinyLlama decode shapes
 *
 * Tests whether an optimized INT64 kernel can close the gap against FP16/INT8
 * tensor-core paths measured in bench_gemm.cu (Phase 1).
 *
 * Arithmetic semantics tested (all kernels use the same grid/block config):
 *   1. "naive":     int64 × int64 → int64 (lower 64 bits, Phase 1 reference)
 *   2. "fpmul":     (int128(a*b)) >> 48, accumulated in int64
 *                   Uses PTX mul.hi.s64. Matches CPU fp_mul() exactly.
 *   3. "exact128":  Full int128 accumulation via PTX carry-chain, >>48 at end.
 *                   Gold-standard exact Q16.48 matmul arithmetic.
 *
 * Block-size sweep: tries {16, 32, 64, 128} 1D threads per block.
 * Also tests split-K with 3D grid for small-N shapes.
 *
 * Build:
 *   NVCC=/usr/local/cuda-13.0/bin/nvcc
 *   $NVCC -O3 -arch=sm_120 bench_int64.cu -o bench_int64 -lcublas
 *
 * Run:
 *   ./bench_int64
 *   ./bench_int64 --json /path/to/report.json
 *   ./bench_int64 --with-lmhead
 *
 * Copyright (c) 2026 — MicroGPT-C GPU feasibility R11
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

/* ---- Error checking ---- */

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t st = (call); \
    if (st != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error at %s:%d: status=%d\n", __FILE__, __LINE__, (int)st); \
        exit(1); \
    } \
} while(0)

/* ---- Timer ---- */

typedef struct { cudaEvent_t start, stop; } GpuTimer;
static void timer_create(GpuTimer *t) {
    CUDA_CHECK(cudaEventCreate(&t->start));
    CUDA_CHECK(cudaEventCreate(&t->stop));
}
static void timer_destroy(GpuTimer *t) {
    cudaEventDestroy(t->start); cudaEventDestroy(t->stop);
}
static void timer_start(GpuTimer *t) { CUDA_CHECK(cudaEventRecord(t->start, 0)); }
static void timer_stop(GpuTimer *t) {
    CUDA_CHECK(cudaEventRecord(t->stop, 0));
    CUDA_CHECK(cudaEventSynchronize(t->stop));
}
static float timer_ms(GpuTimer *t) {
    float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, t->start, t->stop)); return ms;
}

/* ---- Shape definitions ---- */

typedef struct { const char *name; int K; int N; } ShapeClass;

static const ShapeClass DECODE_SHAPES[] = {
    {"q_o_proj",  2048,  2048},
    {"kv_proj",   2048,   256},
    {"gate_up",   2048,  5632},
    {"down_proj", 5632,  2048},
};
#define N_DECODE_SHAPES 4
static const ShapeClass LMHEAD_SHAPE = {"lm_head", 2048, 32000};
static const int M_VALUES[] = {1, 4, 8};
#define N_M_VALUES 3

#define WARMUP_ITERS 10
#define BENCH_ITERS  50

/* ================================================================
 * INT64 Kernels — all use 1D blocks over columns, 2D grid (N, M)
 *
 * For M=1: gridDim.y = 1, each thread handles one output column,
 * serial loop over K. Coalesced B reads across threads.
 * ================================================================ */

/* --- Naive: int64 × int64 → int64 (lower bits) --- */
__global__ void kern_naive(const int64_t * __restrict__ A,
                           const int64_t * __restrict__ B,
                           int64_t * __restrict__ C,
                           int K, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y;
    if (col >= N) return;
    const int64_t *Arow = A + (int64_t)row * K;
    int64_t acc = 0;
    for (int k = 0; k < K; k++)
        acc += Arow[k] * B[(int64_t)k * N + col];
    C[(int64_t)row * N + col] = acc;
}

/* --- fpmul: (int128(a*b)) >> 48 per product, int64 accumulation --- */
__global__ void kern_fpmul(const int64_t * __restrict__ A,
                           const int64_t * __restrict__ B,
                           int64_t * __restrict__ C,
                           int K, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y;
    if (col >= N) return;
    const int64_t *Arow = A + (int64_t)row * K;
    int64_t acc = 0;
    for (int k = 0; k < K; k++) {
        int64_t a = Arow[k], b = B[(int64_t)k * N + col];
        uint64_t lo = (uint64_t)a * (uint64_t)b;
        int64_t hi;
        asm("mul.hi.s64 %0, %1, %2;" : "=l"(hi) : "l"(a), "l"(b));
        acc += (hi << 16) | (lo >> 48);
    }
    C[(int64_t)row * N + col] = acc;
}

/* --- exact128: full int128 accumulation, >>48 at end --- */
__global__ void kern_exact128(const int64_t * __restrict__ A,
                              const int64_t * __restrict__ B,
                              int64_t * __restrict__ C,
                              int K, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y;
    if (col >= N) return;
    const int64_t *Arow = A + (int64_t)row * K;
    uint64_t acc_lo = 0;
    int64_t  acc_hi = 0;
    for (int k = 0; k < K; k++) {
        int64_t a = Arow[k], b = B[(int64_t)k * N + col];
        uint64_t p_lo = (uint64_t)a * (uint64_t)b;
        int64_t  p_hi;
        asm("mul.hi.s64 %0, %1, %2;" : "=l"(p_hi) : "l"(a), "l"(b));
        asm("add.cc.u64  %0, %1, %2;" : "=l"(acc_lo) : "l"(acc_lo), "l"(p_lo));
        asm("addc.s64    %0, %1, %2;" : "=l"(acc_hi) : "l"(acc_hi), "l"(p_hi));
    }
    C[(int64_t)row * N + col] = ((uint64_t)acc_hi << 16) | (acc_lo >> 48);
}

/* --- Split-K fpmul: 3D grid (col_blocks, M, k_splits), atomicAdd --- */
__global__ void kern_splitk_fpmul(const int64_t * __restrict__ A,
                                   const int64_t * __restrict__ B,
                                   int64_t * __restrict__ C,
                                   int K, int N, int k_chunk) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y;
    if (col >= N) return;
    int k_start = blockIdx.z * k_chunk;
    int k_end = k_start + k_chunk;
    if (k_end > K) k_end = K;
    const int64_t *Arow = A + (int64_t)row * K;
    int64_t acc = 0;
    for (int k = k_start; k < k_end; k++) {
        int64_t a = Arow[k], b = B[(int64_t)k * N + col];
        uint64_t lo = (uint64_t)a * (uint64_t)b;
        int64_t hi;
        asm("mul.hi.s64 %0, %1, %2;" : "=l"(hi) : "l"(a), "l"(b));
        acc += (hi << 16) | (lo >> 48);
    }
    atomicAdd((unsigned long long *)&C[(int64_t)row * N + col],
              (unsigned long long)acc);
}

/* ================================================================
 * Kernel type enum and unified runner
 * ================================================================ */

typedef void (*KernFn)(const int64_t*, const int64_t*, int64_t*, int, int);

static float run_kernel(KernFn kern, int64_t *d_A, int64_t *d_B, int64_t *d_C,
                        int M, int K, int N, int block_x) {
    dim3 block(block_x);
    dim3 grid((N + block_x - 1) / block_x, M);

    GpuTimer timer;
    timer_create(&timer);

    for (int i = 0; i < WARMUP_ITERS; i++)
        kern<<<grid, block>>>(d_A, d_B, d_C, K, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    timer_start(&timer);
    for (int i = 0; i < BENCH_ITERS; i++)
        kern<<<grid, block>>>(d_A, d_B, d_C, K, N);
    timer_stop(&timer);

    float avg = timer_ms(&timer) / BENCH_ITERS;
    timer_destroy(&timer);
    return avg;
}

static float run_splitk(int64_t *d_A, int64_t *d_B, int64_t *d_C,
                         int M, int K, int N, int block_x, int n_splits,
                         size_t sC) {
    int k_chunk = (K + n_splits - 1) / n_splits;
    dim3 block(block_x);
    dim3 grid((N + block_x - 1) / block_x, M, n_splits);

    GpuTimer timer;
    timer_create(&timer);

    for (int i = 0; i < WARMUP_ITERS; i++) {
        CUDA_CHECK(cudaMemsetAsync(d_C, 0, sC, 0));
        kern_splitk_fpmul<<<grid, block>>>(d_A, d_B, d_C, K, N, k_chunk);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    timer_start(&timer);
    for (int i = 0; i < BENCH_ITERS; i++) {
        CUDA_CHECK(cudaMemsetAsync(d_C, 0, sC, 0));
        kern_splitk_fpmul<<<grid, block>>>(d_A, d_B, d_C, K, N, k_chunk);
    }
    timer_stop(&timer);

    float avg = timer_ms(&timer) / BENCH_ITERS;
    timer_destroy(&timer);
    return avg;
}

/* ================================================================
 * FP16 baseline
 * ================================================================ */

static float bench_fp16(cublasHandle_t handle, int M, int K, int N) {
    half *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, (size_t)M * K * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B, (size_t)K * N * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C, (size_t)M * N * sizeof(half)));
    CUDA_CHECK(cudaMemset(d_A, 0x3C, (size_t)M * K * sizeof(half)));
    CUDA_CHECK(cudaMemset(d_B, 0x3C, (size_t)K * N * sizeof(half)));
    CUDA_CHECK(cudaMemset(d_C, 0, (size_t)M * N * sizeof(half)));

    __half alpha = __float2half(1.0f), beta = __float2half(0.0f);
    GpuTimer timer; timer_create(&timer);

    for (int i = 0; i < WARMUP_ITERS; i++)
        CUBLAS_CHECK(cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                     N, M, K, &alpha, d_B, K, d_A, K, &beta, d_C, N));
    CUDA_CHECK(cudaDeviceSynchronize());

    timer_start(&timer);
    for (int i = 0; i < BENCH_ITERS; i++)
        CUBLAS_CHECK(cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                     N, M, K, &alpha, d_B, K, d_A, K, &beta, d_C, N));
    timer_stop(&timer);

    float avg = timer_ms(&timer) / BENCH_ITERS;
    timer_destroy(&timer);
    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));
    return avg;
}

/* ================================================================
 * Main
 * ================================================================ */

int main(int argc, char **argv) {
    const char *json_path = NULL;
    int with_lmhead = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--json") == 0 && i + 1 < argc) json_path = argv[++i];
        else if (strcmp(argv[i], "--with-lmhead") == 0) with_lmhead = 1;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s  (%d SMs, compute %d.%d)\n",
           prop.name, prop.multiProcessorCount, prop.major, prop.minor);
    printf("VRAM: %zu MiB, CUDA: %d.%d\n\n",
           prop.totalGlobalMem / (1024*1024),
           CUDART_VERSION/1000, (CUDART_VERSION%1000)/10);

    cublasHandle_t cublasH;
    CUBLAS_CHECK(cublasCreate(&cublasH));
    CUBLAS_CHECK(cublasSetMathMode(cublasH, CUBLAS_TENSOR_OP_MATH));

    printf("=== INT64 Optimized Spike — TinyLlama Decode Shapes ===\n");
    printf("Warmup: %d, Bench: %d iters per measurement\n", WARMUP_ITERS, BENCH_ITERS);
    printf("Arithmetic:\n");
    printf("  naive:    int64*int64 -> int64 (lower bits)\n");
    printf("  fpmul:    (int128(a*b))>>48 per product, int64 accum [PTX mul.hi.s64]\n");
    printf("  exact128: int128 accum via PTX carry-chain, >>48 at end\n");
    printf("  splitk:   fpmul + split-K (3D grid, atomicAdd)\n\n");

    int n_shapes = N_DECODE_SHAPES + (with_lmhead ? 1 : 0);
    ShapeClass shapes[N_DECODE_SHAPES + 1];
    for (int i = 0; i < N_DECODE_SHAPES; i++) shapes[i] = DECODE_SHAPES[i];
    if (with_lmhead) shapes[N_DECODE_SHAPES] = LMHEAD_SHAPE;

    /* Block sizes to sweep */
    int block_sizes[] = {16, 32, 64, 128, 256};
    int n_block_sizes = 5;

    /* Split-K factors to try */
    int splitk_factors[] = {4, 8, 16, 32};
    int n_splitk = 4;

    /* ========== Phase 1: Block-size sweep for naive/fpmul/exact128 ========== */
    printf("=== Phase 1: Block-size sweep (M=1 only) ===\n\n");

    for (int si = 0; si < n_shapes; si++) {
        int M = 1, K = shapes[si].K, N = shapes[si].N;
        size_t sA = (size_t)M*K*8, sB = (size_t)K*N*8, sC = (size_t)M*N*8;

        if (sA + sB + sC > 30ULL*1024*1024*1024) {
            printf("%-10s SKIPPED (too large for INT64)\n\n", shapes[si].name);
            continue;
        }

        int64_t *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, sA));
        CUDA_CHECK(cudaMalloc(&d_B, sB));
        CUDA_CHECK(cudaMalloc(&d_C, sC));
        CUDA_CHECK(cudaMemset(d_A, 0x01, sA));
        CUDA_CHECK(cudaMemset(d_B, 0x01, sB));

        printf("%-10s (M=%d, K=%d, N=%d)  B=%.1f MB\n",
               shapes[si].name, M, K, N, sB / (1024.0*1024.0));
        printf("  %6s | %8s %6s | %8s %6s | %8s %6s\n",
               "BlkSz", "naive", "blocks", "fpmul", "blocks", "exact128", "blocks");

        for (int bi = 0; bi < n_block_sizes; bi++) {
            int bs = block_sizes[bi];
            int n_blocks = ((N + bs - 1) / bs) * M;

            CUDA_CHECK(cudaMemset(d_C, 0, sC));
            float t_naive = run_kernel(kern_naive, d_A, d_B, d_C, M, K, N, bs);

            CUDA_CHECK(cudaMemset(d_C, 0, sC));
            float t_fpmul = run_kernel(kern_fpmul, d_A, d_B, d_C, M, K, N, bs);

            CUDA_CHECK(cudaMemset(d_C, 0, sC));
            float t_exact = run_kernel(kern_exact128, d_A, d_B, d_C, M, K, N, bs);

            printf("  %6d | %7.1fµs %6d | %7.1fµs %6d | %7.1fµs %6d\n",
                   bs, t_naive*1000, n_blocks, t_fpmul*1000, n_blocks,
                   t_exact*1000, n_blocks);
        }

        /* Split-K sweep (fpmul only, best block size) */
        printf("  Split-K (fpmul, block=32):\n");
        for (int ki = 0; ki < n_splitk; ki++) {
            int ks = splitk_factors[ki];
            int k_chunk = (K + ks - 1) / ks;
            int total_blocks = ((N + 31) / 32) * M * ks;

            CUDA_CHECK(cudaMemset(d_C, 0, sC));
            float t = run_splitk(d_A, d_B, d_C, M, K, N, 32, ks, sC);

            printf("    splits=%2d  k_chunk=%4d  blocks=%5d  %.1fµs\n",
                   ks, k_chunk, total_blocks, t * 1000);
        }

        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
        printf("\n");
    }

    /* ========== Phase 2: Best-of comparison vs FP16 ========== */
    printf("=== Phase 2: Best INT64 vs FP16 — all shapes × M values ===\n\n");

    printf("%-10s %3s %5s %6s | %9s %9s %9s %9s | %9s | %s  %s\n",
           "Shape", "M", "K", "N",
           "naive_b", "fpmul_b", "exact_b", "splitk_b",
           "FP16",
           "fpmul/FP16", "speedup");
    printf("%-10s %3s %5s %6s | %9s %9s %9s %9s | %9s | %s  %s\n",
           "----------", "---", "-----", "------",
           "---------", "---------", "---------", "---------",
           "---------",
           "-----------", "--------");

    /* Result storage for JSON */
    typedef struct {
        const char *shape; int M, K, N;
        float naive_ms, fpmul_ms, exact128_ms, splitk_ms, fp16_ms;
        int best_block;
    } Result;
    Result results[64];
    int n_results = 0;

    for (int si = 0; si < n_shapes; si++) {
        for (int mi = 0; mi < N_M_VALUES; mi++) {
            int M = M_VALUES[mi];
            int K = shapes[si].K;
            int N = shapes[si].N;
            size_t sA = (size_t)M*K*8, sB = (size_t)K*N*8, sC = (size_t)M*N*8;

            if (sA + sB + sC > 30ULL*1024*1024*1024) continue;

            int64_t *d_A, *d_B, *d_C;
            CUDA_CHECK(cudaMalloc(&d_A, sA));
            CUDA_CHECK(cudaMalloc(&d_B, sB));
            CUDA_CHECK(cudaMalloc(&d_C, sC));
            CUDA_CHECK(cudaMemset(d_A, 0x01, sA));
            CUDA_CHECK(cudaMemset(d_B, 0x01, sB));

            /* Find best block size for each kernel */
            float best_naive = 1e9, best_fpmul = 1e9, best_exact = 1e9;
            int best_blk = 32;

            for (int bi = 0; bi < n_block_sizes; bi++) {
                int bs = block_sizes[bi];

                CUDA_CHECK(cudaMemset(d_C, 0, sC));
                float t = run_kernel(kern_naive, d_A, d_B, d_C, M, K, N, bs);
                if (t < best_naive) best_naive = t;

                CUDA_CHECK(cudaMemset(d_C, 0, sC));
                t = run_kernel(kern_fpmul, d_A, d_B, d_C, M, K, N, bs);
                if (t < best_fpmul) { best_fpmul = t; best_blk = bs; }

                CUDA_CHECK(cudaMemset(d_C, 0, sC));
                t = run_kernel(kern_exact128, d_A, d_B, d_C, M, K, N, bs);
                if (t < best_exact) best_exact = t;
            }

            /* Best split-K */
            float best_splitk = 1e9;
            for (int ki = 0; ki < n_splitk; ki++) {
                for (int bi = 0; bi < 3; bi++) { /* try block 16,32,64 */
                    int bs = block_sizes[bi];
                    CUDA_CHECK(cudaMemset(d_C, 0, sC));
                    float t = run_splitk(d_A, d_B, d_C, M, K, N, bs,
                                          splitk_factors[ki], sC);
                    if (t > 0 && t < best_splitk) best_splitk = t;
                }
            }
            if (best_splitk > 1e8) best_splitk = -1.0f;

            CUDA_CHECK(cudaFree(d_A));
            CUDA_CHECK(cudaFree(d_B));
            CUDA_CHECK(cudaFree(d_C));

            /* FP16 baseline */
            float fp16_ms = bench_fp16(cublasH, M, K, N);

            /* Best fpmul time (minimum of non-split and split) */
            float best_int64 = best_fpmul;
            if (best_splitk > 0 && best_splitk < best_int64) best_int64 = best_splitk;

            char ratio[16], speedup[16];
            if (best_int64 > 0 && fp16_ms > 0)
                snprintf(ratio, sizeof(ratio), "%.1fx", best_int64 / fp16_ms);
            else snprintf(ratio, sizeof(ratio), "N/A");

            /* naive->best improvement */
            if (best_int64 > 0 && best_naive > 0)
                snprintf(speedup, sizeof(speedup), "%.2fx",
                         best_naive / best_int64);
            else snprintf(speedup, sizeof(speedup), "N/A");

            printf("%-10s %3d %5d %6d | %8.1fµs %8.1fµs %8.1fµs %8.1fµs | %8.1fµs | %10s  %s\n",
                   shapes[si].name, M, K, N,
                   best_naive*1000, best_fpmul*1000, best_exact*1000,
                   best_splitk > 0 ? best_splitk*1000 : 0.0f,
                   fp16_ms*1000,
                   ratio, speedup);

            if (n_results < 64) {
                Result *r = &results[n_results++];
                r->shape = shapes[si].name; r->M = M; r->K = K; r->N = N;
                r->naive_ms = best_naive; r->fpmul_ms = best_fpmul;
                r->exact128_ms = best_exact; r->splitk_ms = best_splitk;
                r->fp16_ms = fp16_ms; r->best_block = best_blk;
            }
        }
        printf("\n");
    }

    /* ========== M=1 decode summary ========== */
    printf("=== M=1 Decode Layer Cost Summary ===\n\n");

    int proj_mult[] = {2, 2, 2, 1};  /* q/o:2, k/v:2, gate/up:2, down:1 */
    double lyr_naive=0, lyr_fpmul=0, lyr_exact=0, lyr_splitk=0, lyr_fp16=0;

    for (int si = 0; si < N_DECODE_SHAPES; si++) {
        for (int i = 0; i < n_results; i++) {
            Result *r = &results[i];
            if (r->M == 1 && r->K == DECODE_SHAPES[si].K
                && r->N == DECODE_SHAPES[si].N
                && strcmp(r->shape, DECODE_SHAPES[si].name) == 0) {
                int m = proj_mult[si];
                lyr_naive  += r->naive_ms * m;
                lyr_fpmul  += r->fpmul_ms * m;
                lyr_exact  += r->exact128_ms * m;
                if (r->splitk_ms > 0) lyr_splitk += r->splitk_ms * m;
                lyr_fp16   += r->fp16_ms * m;
                break;
            }
        }
    }

    printf("  Per-layer M=1 decode (7 projections, best block per shape):\n");
    printf("    naive:    %.3f ms\n", lyr_naive);
    printf("    fpmul:    %.3f ms\n", lyr_fpmul);
    printf("    exact128: %.3f ms\n", lyr_exact);
    printf("    splitk:   %.3f ms\n", lyr_splitk);
    printf("    FP16:     %.3f ms\n\n", lyr_fp16);

    double full_fpmul = lyr_fpmul * 22;
    double full_fp16  = lyr_fp16 * 22;
    /* Use minimum of fpmul and splitk for best INT64 */
    double best_i64_lyr = lyr_fpmul;
    if (lyr_splitk > 0 && lyr_splitk < best_i64_lyr) best_i64_lyr = lyr_splitk;
    double full_best = best_i64_lyr * 22;

    printf("  Full-model (22 layers, no lm_head, GEMM-only):\n");
    printf("    best INT64: %.2f ms  (%.0f tok/s)\n", full_best, 1000.0/full_best);
    printf("    FP16:       %.2f ms  (%.0f tok/s)\n", full_fp16, 1000.0/full_fp16);
    printf("    Gap:        %.1fx slower\n\n", full_best / full_fp16);

    /* Bandwidth analysis */
    printf("=== Bandwidth Analysis (M=1) ===\n\n");
    printf("%-10s | %8s %8s | %8s | %8s %8s\n",
           "Shape", "B(MB)", "BW_min", "best_i64", "FP16", "i64_eff%%");
    for (int si = 0; si < N_DECODE_SHAPES; si++) {
        int K = DECODE_SHAPES[si].K, N = DECODE_SHAPES[si].N;
        double b_mb = (double)K * N * 8 / (1024.0*1024.0);
        double bw_min_us = (double)K * N * 8 / 1792e9 * 1e6;
        float best_t = 0;
        float fp16_t = 0;
        for (int i = 0; i < n_results; i++) {
            if (results[i].M == 1
                && strcmp(results[i].shape, DECODE_SHAPES[si].name) == 0) {
                best_t = results[i].fpmul_ms;
                if (results[i].splitk_ms > 0 && results[i].splitk_ms < best_t)
                    best_t = results[i].splitk_ms;
                fp16_t = results[i].fp16_ms;
                break;
            }
        }
        double eff = bw_min_us / (best_t * 1000) * 100;
        printf("%-10s | %7.1f %7.1fµs | %7.1fµs | %7.1fµs   %5.1f%%\n",
               DECODE_SHAPES[si].name, b_mb, bw_min_us,
               best_t * 1000, fp16_t * 1000, eff);
    }
    printf("  (BW_min = theoretical floor at 1792 GB/s for INT64 B-matrix read)\n");
    printf("  (i64_eff%% = how close to peak bandwidth the best INT64 kernel gets)\n");

    /* JSON */
    if (json_path) {
        FILE *fp = fopen(json_path, "w");
        if (fp) {
            fprintf(fp, "{\n");
            fprintf(fp, "  \"gpu\": \"%s\", \"sms\": %d, \"compute\": \"%d.%d\",\n",
                    prop.name, prop.multiProcessorCount, prop.major, prop.minor);
            fprintf(fp, "  \"warmup\": %d, \"iters\": %d,\n", WARMUP_ITERS, BENCH_ITERS);
            fprintf(fp, "  \"results\": [\n");
            for (int i = 0; i < n_results; i++) {
                Result *r = &results[i];
                fprintf(fp, "    {\"shape\":\"%s\",\"M\":%d,\"K\":%d,\"N\":%d,"
                        "\"naive_ms\":%.6f,\"fpmul_ms\":%.6f,\"exact128_ms\":%.6f,"
                        "\"splitk_ms\":%.6f,\"fp16_ms\":%.6f,\"best_block\":%d}%s\n",
                        r->shape, r->M, r->K, r->N,
                        r->naive_ms, r->fpmul_ms, r->exact128_ms,
                        r->splitk_ms, r->fp16_ms, r->best_block,
                        i < n_results-1 ? "," : "");
            }
            fprintf(fp, "  ],\n");
            fprintf(fp, "  \"m1_layer\": {\"naive\":%.6f,\"fpmul\":%.6f,"
                    "\"exact128\":%.6f,\"splitk\":%.6f,\"fp16\":%.6f}\n",
                    lyr_naive, lyr_fpmul, lyr_exact, lyr_splitk, lyr_fp16);
            fprintf(fp, "}\n");
            fclose(fp);
            printf("\nJSON: %s\n", json_path);
        }
    }

    cublasDestroy(cublasH);
    printf("\nDone.\n");
    return 0;
}
