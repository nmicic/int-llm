/*
 * bench_cudacore_p13b.cu — P13B: M-sweep for CUDA-core FP32 vs INT32
 *
 * Extends P13 to larger M, testing whether the bandwidth-bound equality
 * persists or a compute-bound gap appears as M grows.
 *
 * Kernel: 2D grid, one thread per output element C[m,n], serial K loop.
 *   C[M,N] = A[M,K] * W[N,K]^T  (W stored row-major as [N,K])
 * At M=1 this degenerates to the P13 GEMV. At larger M, compute load
 * grows proportionally.
 *
 * No cuBLAS, no Tensor Cores, no WMMA. Custom CUDA-core-only kernels.
 * Both paths identical structure, same tile/block/access pattern.
 *
 * Build:
 *   NVCC=/usr/local/cuda-13.0/bin/nvcc
 *   $NVCC -O3 -arch=sm_120 bench_cudacore_p13b.cu -o bench_cudacore_p13b -lm
 *
 * Run:
 *   ./bench_cudacore_p13b
 *
 * Copyright (c) 2026 — MicroGPT-C P13B CUDA-core M-sweep
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <cuda_runtime.h>

static void ck(cudaError_t e, int l) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA L%d: %s\n", l, cudaGetErrorString(e)); exit(1);
    }
}
#define CK(x) ck((x), __LINE__)

/* ================================================================
 * Q16.16 fixed-point (same as P13)
 * ================================================================ */

#define Q16_SHIFT 16
#define Q16_SCALE 65536.0f

__global__ void kern_float_to_q16(int32_t *out, const float *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = (int32_t)rintf(in[i] * Q16_SCALE);
}

/* ================================================================
 * CUDA-core-only GEMM kernels
 *
 * C[M,N] = A[M,K] × W[N,K]^T
 * Both use 2D blocks (BX × BY), one thread per output element.
 * Serial K loop. No shared memory. Identical structure.
 *
 * Memory access per thread: A[m, 0..K-1] (row stride K, broadcast across col),
 * W[n, 0..K-1] (row stride K, broadcast across row).
 * Adjacent threads in x read adjacent W rows (stride K between elements
 * within the K loop — not perfectly coalesced, but EQUAL for both paths).
 * ================================================================ */

#define BX 16   /* threads in N direction */
#define BY 16   /* threads in M direction */

__global__ void kern_gemm_fp32(float *C, const float *A, const float *W,
                                int M, int N, int K) {
    int n = blockIdx.x * BX + threadIdx.x;
    int m = blockIdx.y * BY + threadIdx.y;
    if (m >= M || n >= N) return;

    const float *a_row = A + (int64_t)m * K;
    const float *w_row = W + (int64_t)n * K;
    float acc = 0.0f;
    for (int k = 0; k < K; k++)
        acc += a_row[k] * w_row[k];   /* FP32 FMA */
    C[(int64_t)m * N + n] = acc;
}

__global__ void kern_gemm_int32(int32_t *C, const int32_t *A, const int32_t *W,
                                 int M, int N, int K) {
    int n = blockIdx.x * BX + threadIdx.x;
    int m = blockIdx.y * BY + threadIdx.y;
    if (m >= M || n >= N) return;

    const int32_t *a_row = A + (int64_t)m * K;
    const int32_t *w_row = W + (int64_t)n * K;
    int64_t acc = 0;
    for (int k = 0; k < K; k++)
        acc += (int64_t)a_row[k] * (int64_t)w_row[k];   /* INT64 mul+add */
    C[(int64_t)m * N + n] = (int32_t)(acc >> Q16_SHIFT);
}

/* ================================================================
 * Benchmark
 * ================================================================ */

static const int M_SWEEP[] = {1, 2, 4, 8, 16, 32, 64};
#define N_M_SWEEP 7

#define WARMUP 10
#define BENCH  50

static void run_sweep(const char *name, int N_out, int K) {
    printf("\n=== %s (N=%d, K=%d) ===\n", name, N_out, K);
    printf("%-6s | %10s %10s | %10s %10s | %7s | %s\n",
           "M", "FP32(µs)", "GB/s", "INT32(µs)", "GB/s", "ratio", "drift");
    printf("%-6s | %10s %10s | %10s %10s | %7s | %s\n",
           "------", "----------", "----------",
           "----------", "----------", "-------", "-----------");

    /* Allocate weight matrix (same for all M) */
    float *h_W = (float*)malloc((int64_t)N_out * K * sizeof(float));
    for (int64_t i = 0; i < (int64_t)N_out * K; i++)
        h_W[i] = sinf((float)(i * 7 + 3)) * 0.5f;

    float *d_Wf;
    int32_t *d_Wi;
    CK(cudaMalloc(&d_Wf, (int64_t)N_out * K * sizeof(float)));
    CK(cudaMalloc(&d_Wi, (int64_t)N_out * K * sizeof(int32_t)));
    CK(cudaMemcpy(d_Wf, h_W, (int64_t)N_out * K * sizeof(float), cudaMemcpyHostToDevice));
    kern_float_to_q16<<<((int64_t)N_out * K + 255) / 256, 256>>>(d_Wi, d_Wf, N_out * K);
    CK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));

    for (int mi = 0; mi < N_M_SWEEP; mi++) {
        int M = M_SWEEP[mi];

        /* Allocate and fill A[M,K] and C[M,N] */
        float *h_A = (float*)malloc((int64_t)M * K * sizeof(float));
        for (int64_t i = 0; i < (int64_t)M * K; i++)
            h_A[i] = sinf((float)(i * 13 + 7)) * 0.3f;

        float *d_Af, *d_Cf;
        int32_t *d_Ai, *d_Ci;
        CK(cudaMalloc(&d_Af, (int64_t)M * K * sizeof(float)));
        CK(cudaMalloc(&d_Cf, (int64_t)M * N_out * sizeof(float)));
        CK(cudaMalloc(&d_Ai, (int64_t)M * K * sizeof(int32_t)));
        CK(cudaMalloc(&d_Ci, (int64_t)M * N_out * sizeof(int32_t)));
        CK(cudaMemcpy(d_Af, h_A, (int64_t)M * K * sizeof(float), cudaMemcpyHostToDevice));
        kern_float_to_q16<<<((int64_t)M * K + 255) / 256, 256>>>(d_Ai, d_Af, M * K);
        CK(cudaDeviceSynchronize());

        dim3 block(BX, BY);
        dim3 grid((N_out + BX - 1) / BX, (M + BY - 1) / BY);

        /* FP32 */
        for (int i = 0; i < WARMUP; i++)
            kern_gemm_fp32<<<grid, block>>>(d_Cf, d_Af, d_Wf, M, N_out, K);
        CK(cudaDeviceSynchronize());
        CK(cudaEventRecord(t0));
        for (int i = 0; i < BENCH; i++)
            kern_gemm_fp32<<<grid, block>>>(d_Cf, d_Af, d_Wf, M, N_out, K);
        CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
        float fp32_ms; CK(cudaEventElapsedTime(&fp32_ms, t0, t1));
        float fp32_us = fp32_ms * 1000.0f / BENCH;

        /* INT32 */
        for (int i = 0; i < WARMUP; i++)
            kern_gemm_int32<<<grid, block>>>(d_Ci, d_Ai, d_Wi, M, N_out, K);
        CK(cudaDeviceSynchronize());
        CK(cudaEventRecord(t0));
        for (int i = 0; i < BENCH; i++)
            kern_gemm_int32<<<grid, block>>>(d_Ci, d_Ai, d_Wi, M, N_out, K);
        CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
        float int32_ms; CK(cudaEventElapsedTime(&int32_ms, t0, t1));
        float int32_us = int32_ms * 1000.0f / BENCH;

        /* Drift check */
        float *h_Cf = (float*)malloc((int64_t)M * N_out * sizeof(float));
        int32_t *h_Ci = (int32_t*)malloc((int64_t)M * N_out * sizeof(int32_t));
        CK(cudaMemcpy(h_Cf, d_Cf, (int64_t)M * N_out * sizeof(float), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(h_Ci, d_Ci, (int64_t)M * N_out * sizeof(int32_t), cudaMemcpyDeviceToHost));
        double max_err = 0, sum_err = 0;
        for (int64_t i = 0; i < (int64_t)M * N_out; i++) {
            double err = fabs(h_Cf[i] - (double)h_Ci[i] / Q16_SCALE);
            if (err > max_err) max_err = err;
            sum_err += err;
        }
        double mean_err = sum_err / ((int64_t)M * N_out);

        /* Effective bandwidth: read W[N,K] + A[M,K], write C[M,N], all 4B/elem */
        double bytes = ((double)N_out * K + (double)M * K + (double)M * N_out) * 4.0;
        double fp32_bw = bytes / (fp32_us * 1e-6) / 1e9;
        double int32_bw = bytes / (int32_us * 1e-6) / 1e9;

        float ratio = fp32_us / int32_us;

        printf("%-6d | %9.1f %9.0f | %9.1f %9.0f | %6.3f | max=%.1e mean=%.1e\n",
               M, fp32_us, fp32_bw, int32_us, int32_bw, ratio, max_err, mean_err);

        free(h_A); free(h_Cf); free(h_Ci);
        cudaFree(d_Af); cudaFree(d_Cf); cudaFree(d_Ai); cudaFree(d_Ci);
    }

    free(h_W);
    cudaFree(d_Wf); cudaFree(d_Wi);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
}

int main(void) {
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (%d SMs, compute %d.%d)\n",
           prop.name, prop.multiProcessorCount, prop.major, prop.minor);

    printf("\n=== P13B: CUDA-Core-Only M-Sweep ===\n");
    printf("FP32 vs INT32 Q16.16, no Tensor Cores, matched 2D GEMM kernels.\n");
    printf("Block: %dx%d, one thread per output element, serial K loop.\n", BX, BY);
    printf("INT32: INT64 accumulation, >>16 rescale at end.\n");
    printf("ratio > 1: INT32 faster. ratio < 1: FP32 faster.\n");

    run_sweep("q_o_proj", 2048, 2048);
    run_sweep("down_proj", 2048, 5632);

    printf("\nDone.\n");
    return 0;
}
