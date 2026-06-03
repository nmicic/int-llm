/*
 * bench_gemm.cu — GPU GEMM feasibility benchmark for TinyLlama linear-layer shapes
 *
 * Benchmarks representative TinyLlama/Llama-family linear-layer shapes on the
 * RTX 5090 to determine whether FP16 or INT8 is the better first GPU compute path.
 *
 * Shapes tested (M tokens × K input × N output):
 *   Q/O projection:   M × 2048 × 2048
 *   K/V projection:   M × 2048 × 256
 *   Gate/Up proj:      M × 2048 × 5632
 *   Down proj:         M × 5632 × 2048
 *   lm_head:           M × 2048 × 32000
 *
 * Regimes:
 *   Decode-like:  M = 1
 *   Prefill-like: M = 4, 8, 16, 32, 64, 128
 *
 * Paths:
 *   1. FP16 Tensor Core via cublasHgemm
 *   2. INT8 Tensor Core via cublasLtMatmul (INT8 in, INT32 out)
 *   3. Optional INT64 negative-control via naive kernel (one shape only)
 *
 * Build:
 *   NVCC=/usr/local/cuda-13.0/bin/nvcc
 *   $NVCC -O3 -arch=sm_120 bench_gemm.cu -o bench_gemm -lcublas -lcublasLt
 *
 * Run:
 *   ./bench_gemm
 *   ./bench_gemm --json report.json
 *
 * Copyright (c) 2026 — MicroGPT-C GPU feasibility phase
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cublasLt.h>

/* ---------- error helpers ---------- */

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

/* ---------- timing ---------- */

typedef struct {
    cudaEvent_t start, stop;
} GpuTimer;

static void timer_create(GpuTimer *t) {
    CUDA_CHECK(cudaEventCreate(&t->start));
    CUDA_CHECK(cudaEventCreate(&t->stop));
}

static void timer_destroy(GpuTimer *t) {
    cudaEventDestroy(t->start);
    cudaEventDestroy(t->stop);
}

static void timer_start(GpuTimer *t, cudaStream_t s) {
    CUDA_CHECK(cudaEventRecord(t->start, s));
}

static void timer_stop(GpuTimer *t, cudaStream_t s) {
    CUDA_CHECK(cudaEventRecord(t->stop, s));
    CUDA_CHECK(cudaEventSynchronize(t->stop));
}

static float timer_ms(GpuTimer *t) {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t->start, t->stop));
    return ms;
}

/* ---------- shape definitions ---------- */

typedef struct {
    const char *name;
    int M;   /* tokens / batch dim */
    int K;   /* input dim (weight rows) */
    int N;   /* output dim (weight cols) */
} GemmShape;

/* TinyLlama projection classes */
static const int SHAPE_K[] = {2048, 2048, 2048, 5632, 2048};
static const int SHAPE_N[] = {2048,  256, 5632, 2048, 32000};
static const char *SHAPE_NAMES[] = {
    "q_o_proj", "kv_proj", "gate_up", "down_proj", "lm_head"
};
#define N_SHAPE_CLASSES 5

/* Token batch sizes: decode-like (1) through prefill-like */
static const int BATCH_SIZES[] = {1, 4, 8, 16, 32, 64, 128};
#define N_BATCH_SIZES 7

/* ---------- benchmark config ---------- */

#define WARMUP_ITERS  5
#define BENCH_ITERS  20

/* ---------- FP16 benchmark ---------- */

static float bench_fp16(cublasHandle_t handle, int M, int K, int N) {
    /* Allocate device memory */
    half *d_A, *d_B, *d_C;
    size_t size_A = (size_t)M * K * sizeof(half);
    size_t size_B = (size_t)K * N * sizeof(half);
    size_t size_C = (size_t)M * N * sizeof(half);

    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));

    /* Fill with small random-ish values to avoid degenerate cases */
    /* Using curand would be cleaner, but memset is fine for perf measurement */
    CUDA_CHECK(cudaMemset(d_A, 0x3C, size_A)); /* ~1.0 in fp16 */
    CUDA_CHECK(cudaMemset(d_B, 0x3C, size_B));
    CUDA_CHECK(cudaMemset(d_C, 0, size_C));

    /*
     * cuBLAS is column-major. For row-major C = A * B where:
     *   A is M×K, B is K×N, C is M×N
     * We compute: C^T = B^T * A^T
     * So call: cublas(N, M, K, B, N, A, K, C, N) with no-transpose
     *
     * Actually the standard trick: pass transB=T, transA=N for row-major
     * C(M,N) = A(M,K) * B(K,N)
     * cublas col-major: C_col(N,M) = B^T_col(N,K) * A^T_col(K,M)
     * cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, B, K, A, K, &beta, C, N)
     *
     * But for performance benchmarking, the exact layout doesn't matter much —
     * we just need correct dimensions. Let's use the simplest correct call.
     */

    __half alpha_h = __float2half(1.0f);
    __half beta_h  = __float2half(0.0f);

    GpuTimer timer;
    timer_create(&timer);

    /* Warmup */
    for (int i = 0; i < WARMUP_ITERS; i++) {
        CUBLAS_CHECK(cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                 N, M, K,
                                 &alpha_h,
                                 d_B, K,
                                 d_A, K,
                                 &beta_h,
                                 d_C, N));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Timed iterations */
    timer_start(&timer, 0);
    for (int i = 0; i < BENCH_ITERS; i++) {
        CUBLAS_CHECK(cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                 N, M, K,
                                 &alpha_h,
                                 d_B, K,
                                 d_A, K,
                                 &beta_h,
                                 d_C, N));
    }
    timer_stop(&timer, 0);

    float total_ms = timer_ms(&timer);
    float avg_ms = total_ms / BENCH_ITERS;

    timer_destroy(&timer);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return avg_ms;
}

/* ---------- INT8 benchmark via cuBLASLt ---------- */

static float bench_int8(cublasLtHandle_t ltHandle, int M, int K, int N) {
    /* INT8 inputs, INT32 output accumulation */
    int8_t *d_A, *d_B;
    int32_t *d_C;
    size_t size_A = (size_t)M * K;
    size_t size_B = (size_t)K * N;
    size_t size_C = (size_t)M * N * sizeof(int32_t);

    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));

    CUDA_CHECK(cudaMemset(d_A, 1, size_A));
    CUDA_CHECK(cudaMemset(d_B, 1, size_B));
    CUDA_CHECK(cudaMemset(d_C, 0, size_C));

    /* cuBLASLt matmul descriptor */
    cublasLtMatmulDesc_t matmulDesc;
    cublasLtMatrixLayout_t layoutA, layoutB, layoutC;

    /* Create matmul descriptor: INT8 compute */
    CUBLAS_CHECK(cublasLtMatmulDescCreate(&matmulDesc,
                                           CUBLAS_COMPUTE_32I,
                                           CUDA_R_32I));

    /*
     * For row-major INT8 GEMM: C(M,N) = A(M,K) * B(K,N)
     * cuBLASLt col-major: C_col(N,M) = B^T_col(N,K) * A^T_col(K,M)
     */
    cublasOperation_t transA = CUBLAS_OP_T;
    cublasOperation_t transB = CUBLAS_OP_N;

    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc,
                                                 CUBLASLT_MATMUL_DESC_TRANSA,
                                                 &transA, sizeof(transA)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc,
                                                 CUBLASLT_MATMUL_DESC_TRANSB,
                                                 &transB, sizeof(transB)));

    /* Layout: B (used as "A" in cuBLASLt col-major) is K×N stored row-major → col N×K */
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&layoutA, CUDA_R_8I, K, N, K));
    /* Layout: A (used as "B" in cuBLASLt col-major) is M×K stored row-major → col K×M */
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&layoutB, CUDA_R_8I, K, M, K));
    /* Layout: C is M×N stored row-major → col N×M */
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&layoutC, CUDA_R_32I, N, M, N));

    int32_t alpha_i = 1;
    int32_t beta_i  = 0;

    /* Heuristic search for best algo */
    cublasLtMatmulPreference_t pref;
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    size_t workspaceSize = 4 * 1024 * 1024; /* 4 MB workspace */
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(pref,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspaceSize, sizeof(workspaceSize)));

    cublasLtMatmulHeuristicResult_t heuResult;
    int returnedResults = 0;
    cublasStatus_t heuStatus = cublasLtMatmulAlgoGetHeuristic(
        ltHandle, matmulDesc, layoutA, layoutB, layoutC, layoutC,
        pref, 1, &heuResult, &returnedResults);

    if (heuStatus != CUBLAS_STATUS_SUCCESS || returnedResults == 0) {
        /* INT8 not supported for this shape — return -1 to signal */
        cublasLtMatmulPreferenceDestroy(pref);
        cublasLtMatrixLayoutDestroy(layoutA);
        cublasLtMatrixLayoutDestroy(layoutB);
        cublasLtMatrixLayoutDestroy(layoutC);
        cublasLtMatmulDescDestroy(matmulDesc);
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
        return -1.0f;
    }

    void *workspace = NULL;
    if (heuResult.workspaceSize > 0) {
        CUDA_CHECK(cudaMalloc(&workspace, heuResult.workspaceSize));
    }

    GpuTimer timer;
    timer_create(&timer);

    /* Warmup */
    for (int i = 0; i < WARMUP_ITERS; i++) {
        CUBLAS_CHECK(cublasLtMatmul(ltHandle, matmulDesc,
                                     &alpha_i,
                                     d_B, layoutA,   /* B^T as "A" in col-major */
                                     d_A, layoutB,   /* A^T as "B" in col-major */
                                     &beta_i,
                                     d_C, layoutC,
                                     d_C, layoutC,
                                     &heuResult.algo,
                                     workspace, heuResult.workspaceSize,
                                     0));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Timed iterations */
    timer_start(&timer, 0);
    for (int i = 0; i < BENCH_ITERS; i++) {
        CUBLAS_CHECK(cublasLtMatmul(ltHandle, matmulDesc,
                                     &alpha_i,
                                     d_B, layoutA,
                                     d_A, layoutB,
                                     &beta_i,
                                     d_C, layoutC,
                                     d_C, layoutC,
                                     &heuResult.algo,
                                     workspace, heuResult.workspaceSize,
                                     0));
    }
    timer_stop(&timer, 0);

    float total_ms = timer_ms(&timer);
    float avg_ms = total_ms / BENCH_ITERS;

    timer_destroy(&timer);
    if (workspace) CUDA_CHECK(cudaFree(workspace));
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(layoutA);
    cublasLtMatrixLayoutDestroy(layoutB);
    cublasLtMatrixLayoutDestroy(layoutC);
    cublasLtMatmulDescDestroy(matmulDesc);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return avg_ms;
}

/* ---------- INT64 negative control (naive, one shape only) ---------- */

__global__ void gemm_int64_naive(const int64_t *A, const int64_t *B,
                                  int64_t *C, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        int64_t sum = 0;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

static float bench_int64_naive(int M, int K, int N) {
    int64_t *d_A, *d_B, *d_C;
    size_t size_A = (size_t)M * K * sizeof(int64_t);
    size_t size_B = (size_t)K * N * sizeof(int64_t);
    size_t size_C = (size_t)M * N * sizeof(int64_t);

    /* Check if this fits in VRAM */
    size_t total = size_A + size_B + size_C;
    if (total > 30ULL * 1024 * 1024 * 1024) {
        return -2.0f; /* too large */
    }

    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));
    CUDA_CHECK(cudaMemset(d_A, 1, size_A));
    CUDA_CHECK(cudaMemset(d_B, 1, size_B));
    CUDA_CHECK(cudaMemset(d_C, 0, size_C));

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    GpuTimer timer;
    timer_create(&timer);

    /* Warmup */
    for (int i = 0; i < 2; i++) {
        gemm_int64_naive<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Only 3 iterations — this will be slow */
    int iters = 3;
    timer_start(&timer, 0);
    for (int i = 0; i < iters; i++) {
        gemm_int64_naive<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    }
    timer_stop(&timer, 0);

    float total_ms = timer_ms(&timer);
    float avg_ms = total_ms / iters;

    timer_destroy(&timer);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return avg_ms;
}

/* ---------- result storage ---------- */

typedef struct {
    const char *shape_name;
    int M, K, N;
    float fp16_ms;
    float int8_ms;
    float int64_ms;  /* -1 = not tested, -2 = too large */
    double fp16_tflops;
    double int8_tops;
} BenchResult;

static double compute_tflops(int M, int K, int N, float ms) {
    if (ms <= 0) return 0;
    double flops = 2.0 * M * K * N;  /* multiply-accumulate = 2 ops */
    return flops / (ms * 1e-3) / 1e12;
}

/* ---------- main ---------- */

int main(int argc, char **argv) {
    const char *json_path = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--json") == 0 && i + 1 < argc) {
            json_path = argv[++i];
        }
    }

    /* Print GPU info */
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s\n", prop.name);
    printf("SM count: %d\n", prop.multiProcessorCount);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Memory: %zu MiB\n", prop.totalGlobalMem / (1024 * 1024));
    printf("CUDA version (runtime): %d.%d\n",
           CUDART_VERSION / 1000, (CUDART_VERSION % 1000) / 10);
    printf("\n");

    /* Create handles */
    cublasHandle_t cublasH;
    cublasLtHandle_t cublasLtH;
    CUBLAS_CHECK(cublasCreate(&cublasH));
    CUBLAS_CHECK(cublasLtCreate(&cublasLtH));

    /* Enable tensor core math */
    CUBLAS_CHECK(cublasSetMathMode(cublasH, CUBLAS_TENSOR_OP_MATH));

    printf("=== TinyLlama GPU GEMM Feasibility Benchmark ===\n");
    printf("Warmup: %d iters, Bench: %d iters per measurement\n\n", WARMUP_ITERS, BENCH_ITERS);

    /* Run all shape × batch combinations */
    int n_results = 0;
    BenchResult results[N_SHAPE_CLASSES * N_BATCH_SIZES + 4]; /* extra for INT64 */

    printf("%-12s %5s %5s %6s | %10s %8s | %10s %8s | %s\n",
           "Shape", "M", "K", "N",
           "FP16(ms)", "TFLOPS",
           "INT8(ms)", "TOPS",
           "INT8/FP16");
    printf("%-12s %5s %5s %6s | %10s %8s | %10s %8s | %s\n",
           "------------", "-----", "-----", "------",
           "----------", "--------",
           "----------", "--------",
           "--------");

    for (int si = 0; si < N_SHAPE_CLASSES; si++) {
        for (int bi = 0; bi < N_BATCH_SIZES; bi++) {
            int M = BATCH_SIZES[bi];
            int K = SHAPE_K[si];
            int N = SHAPE_N[si];

            float fp16_ms = bench_fp16(cublasH, M, K, N);
            float int8_ms = bench_int8(cublasLtH, M, K, N);

            double fp16_tflops = compute_tflops(M, K, N, fp16_ms);
            double int8_tops = compute_tflops(M, K, N, int8_ms);

            BenchResult *r = &results[n_results++];
            r->shape_name = SHAPE_NAMES[si];
            r->M = M;
            r->K = K;
            r->N = N;
            r->fp16_ms = fp16_ms;
            r->int8_ms = int8_ms;
            r->int64_ms = -1.0f;
            r->fp16_tflops = fp16_tflops;
            r->int8_tops = int8_tops;

            char ratio_str[32];
            if (int8_ms > 0 && fp16_ms > 0) {
                snprintf(ratio_str, sizeof(ratio_str), "%.2fx", fp16_ms / int8_ms);
            } else if (int8_ms < 0) {
                snprintf(ratio_str, sizeof(ratio_str), "N/A");
            } else {
                snprintf(ratio_str, sizeof(ratio_str), "?");
            }

            printf("%-12s %5d %5d %6d | %10.4f %7.2fT | %10.4f %7.2fT | %s\n",
                   SHAPE_NAMES[si], M, K, N,
                   fp16_ms, fp16_tflops,
                   int8_ms > 0 ? int8_ms : 0.0f,
                   int8_ms > 0 ? int8_tops : 0.0,
                   ratio_str);
        }
        printf("\n"); /* visual separator between shape classes */
    }

    /* INT64 negative control — just q_o_proj M=1 and M=16 */
    printf("\n=== INT64 Negative Control (naive CUDA kernel) ===\n");
    printf("%-12s %5s %5s %6s | %10s\n", "Shape", "M", "K", "N", "INT64(ms)");

    int int64_shapes[][3] = {{1, 2048, 2048}, {16, 2048, 2048}};
    for (int i = 0; i < 2; i++) {
        int M = int64_shapes[i][0];
        int K = int64_shapes[i][1];
        int N = int64_shapes[i][2];
        float ms = bench_int64_naive(M, K, N);

        BenchResult *r = &results[n_results++];
        r->shape_name = "int64_ctrl";
        r->M = M; r->K = K; r->N = N;
        r->fp16_ms = -1; r->int8_ms = -1;
        r->int64_ms = ms;
        r->fp16_tflops = 0; r->int8_tops = 0;

        if (ms >= 0) {
            printf("%-12s %5d %5d %6d | %10.4f\n", "q_o_proj", M, K, N, ms);
        } else {
            printf("%-12s %5d %5d %6d | %10s\n", "q_o_proj", M, K, N,
                   ms == -2 ? "TOO_LARGE" : "FAILED");
        }
    }

    /* Summary analysis */
    printf("\n=== Analysis Summary ===\n");

    /* Find decode-like (M=1) averages */
    double fp16_decode_total = 0, int8_decode_total = 0;
    int fp16_decode_n = 0, int8_decode_n = 0;
    double fp16_prefill_total = 0, int8_prefill_total = 0;
    int fp16_prefill_n = 0, int8_prefill_n = 0;

    for (int i = 0; i < n_results; i++) {
        BenchResult *r = &results[i];
        if (r->fp16_ms <= 0 || strcmp(r->shape_name, "int64_ctrl") == 0) continue;

        if (r->M == 1) {
            fp16_decode_total += r->fp16_ms;
            fp16_decode_n++;
            if (r->int8_ms > 0) {
                int8_decode_total += r->int8_ms;
                int8_decode_n++;
            }
        } else if (r->M >= 16) {
            fp16_prefill_total += r->fp16_ms;
            fp16_prefill_n++;
            if (r->int8_ms > 0) {
                int8_prefill_total += r->int8_ms;
                int8_prefill_n++;
            }
        }
    }

    /* Estimate single-layer decode cost = sum of all projections at M=1 */
    double fp16_layer_decode = 0, int8_layer_decode = 0;
    int int8_layer_ok = 1;
    for (int si = 0; si < N_SHAPE_CLASSES; si++) {
        for (int i = 0; i < n_results; i++) {
            if (results[i].M == 1 && results[i].K == SHAPE_K[si]
                && results[i].N == SHAPE_N[si]
                && strcmp(results[i].shape_name, SHAPE_NAMES[si]) == 0) {
                /* For gate/up we need 2x, but they're already separate entries */
                /* Actually gate and up are both in gate_up class. In real inference
                   there are: Q, K, V, O, gate, up, down projections.
                   Our shape classes are: q_o (covers Q and O, so 2x),
                   kv (covers K and V, so 2x), gate_up (covers gate and up, so 2x),
                   down (1x), lm_head (1x per token). */
                int mult = 1;
                if (si == 0) mult = 2;  /* Q + O */
                if (si == 1) mult = 2;  /* K + V */
                if (si == 2) mult = 2;  /* gate + up */

                fp16_layer_decode += results[i].fp16_ms * mult;
                if (results[i].int8_ms > 0) {
                    int8_layer_decode += results[i].int8_ms * mult;
                } else {
                    int8_layer_ok = 0;
                }
                break;
            }
        }
    }

    /* Subtract lm_head from layer cost (it's per-token, not per-layer per-position) */
    /* Actually lm_head IS per decode step, just once not per-layer. Keep it separate. */
    double fp16_lm_head = 0, int8_lm_head = 0;
    for (int i = 0; i < n_results; i++) {
        if (results[i].M == 1 && strcmp(results[i].shape_name, "lm_head") == 0) {
            fp16_lm_head = results[i].fp16_ms;
            int8_lm_head = results[i].int8_ms > 0 ? results[i].int8_ms : 0;
            /* Remove from layer sum */
            fp16_layer_decode -= results[i].fp16_ms;
            if (results[i].int8_ms > 0) {
                int8_layer_decode -= results[i].int8_ms;
            }
            break;
        }
    }

    printf("\nEstimated single-token decode cost (one TinyLlama layer, GEMM only):\n");
    printf("  FP16:  %.4f ms\n", fp16_layer_decode);
    if (int8_layer_ok) {
        printf("  INT8:  %.4f ms\n", int8_layer_decode);
        printf("  Ratio: %.2fx\n", fp16_layer_decode / int8_layer_decode);
    }

    printf("\nEstimated full-model decode cost (22 layers + lm_head, GEMM only):\n");
    double fp16_full = fp16_layer_decode * 22 + fp16_lm_head;
    printf("  FP16:  %.3f ms  (%.1f tok/s)\n", fp16_full, 1000.0 / fp16_full);
    if (int8_layer_ok) {
        double int8_full = int8_layer_decode * 22 + int8_lm_head;
        printf("  INT8:  %.3f ms  (%.1f tok/s)\n", int8_full, 1000.0 / int8_full);
    }

    printf("\nNote: these are GEMM-only estimates. Real decode also includes:\n");
    printf("  - RMSNorm, RoPE, softmax, SiLU (likely FP16/FP32, small cost)\n");
    printf("  - KV-cache memory traffic\n");
    printf("  - Weight loading from VRAM\n");
    printf("  - Kernel launch overhead (significant at M=1)\n");

    printf("\nCPU Q16.48 reference: ~851 ms/tok (native mmap, R9 pre-R7 baseline; post-R7 is ~13%% faster)\n");

    /* JSON output */
    if (json_path) {
        FILE *fp = fopen(json_path, "w");
        if (!fp) {
            fprintf(stderr, "Warning: could not open %s for writing\n", json_path);
        } else {
            fprintf(fp, "{\n");
            fprintf(fp, "  \"gpu\": \"%s\",\n", prop.name);
            fprintf(fp, "  \"sm_count\": %d,\n", prop.multiProcessorCount);
            fprintf(fp, "  \"compute_capability\": \"%d.%d\",\n", prop.major, prop.minor);
            fprintf(fp, "  \"vram_mib\": %zu,\n", prop.totalGlobalMem / (1024 * 1024));
            fprintf(fp, "  \"cuda_runtime\": \"%d.%d\",\n",
                    CUDART_VERSION / 1000, (CUDART_VERSION % 1000) / 10);
            fprintf(fp, "  \"warmup_iters\": %d,\n", WARMUP_ITERS);
            fprintf(fp, "  \"bench_iters\": %d,\n", BENCH_ITERS);
            fprintf(fp, "  \"results\": [\n");
            for (int i = 0; i < n_results; i++) {
                BenchResult *r = &results[i];
                fprintf(fp, "    {\"shape\": \"%s\", \"M\": %d, \"K\": %d, \"N\": %d, "
                        "\"fp16_ms\": %.6f, \"int8_ms\": %.6f, \"int64_ms\": %.6f, "
                        "\"fp16_tflops\": %.4f, \"int8_tops\": %.4f}%s\n",
                        r->shape_name, r->M, r->K, r->N,
                        r->fp16_ms, r->int8_ms, r->int64_ms,
                        r->fp16_tflops, r->int8_tops,
                        (i < n_results - 1) ? "," : "");
            }
            fprintf(fp, "  ],\n");
            fprintf(fp, "  \"decode_estimate\": {\n");
            fprintf(fp, "    \"fp16_layer_ms\": %.6f,\n", fp16_layer_decode);
            fprintf(fp, "    \"fp16_full_model_ms\": %.6f,\n", fp16_full);
            if (int8_layer_ok) {
                double int8_full = int8_layer_decode * 22 + int8_lm_head;
                fprintf(fp, "    \"int8_layer_ms\": %.6f,\n", int8_layer_decode);
                fprintf(fp, "    \"int8_full_model_ms\": %.6f\n", int8_full);
            } else {
                fprintf(fp, "    \"int8_layer_ms\": null,\n");
                fprintf(fp, "    \"int8_full_model_ms\": null\n");
            }
            fprintf(fp, "  }\n");
            fprintf(fp, "}\n");
            fclose(fp);
            printf("\nJSON report written to: %s\n", json_path);
        }
    }

    /* Cleanup */
    cublasDestroy(cublasH);
    cublasLtDestroy(cublasLtH);

    printf("\nBenchmark complete.\n");
    return 0;
}
