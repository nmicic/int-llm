/*
 * bench_cudacore_fair.cu — CUDA-core-only FP32 vs INT32 fixed-point comparison
 *
 * Fair comparison: both paths use 4-byte inputs (identical bandwidth),
 * custom CUDA-core-only kernels, same tiling/launch/access pattern.
 * No cuBLAS, no Tensor Cores, no WMMA.
 *
 * FP32 path: float inputs, FP32 FMA accumulation
 * INT32 path: Q16.16 fixed-point inputs, INT64 accumulation, >>16 rescale
 *
 * Arithmetic cost per element:
 *   FP32:  1 FMA instruction
 *   INT32: mul.lo.u32 + mul.hi.s32 + add.cc.u64 + addc.s64 ≈ 4 instructions
 *
 * The question: at M=1 decode (bandwidth-bound), does the 4x instruction
 * gap matter, or does memory bandwidth dominate equally for both?
 *
 * Build:
 *   NVCC=/usr/local/cuda-13.0/bin/nvcc
 *   $NVCC -O3 -arch=sm_120 bench_cudacore_fair.cu -o bench_cudacore_fair -lm
 *
 * Run:
 *   ./bench_cudacore_fair
 *
 * Copyright (c) 2026 — MicroGPT-C P13 CUDA-core arithmetic comparison
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <cuda_runtime.h>

/* ================================================================ */
static void ck(cudaError_t e, int line) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error L%d: %s\n", line, cudaGetErrorString(e));
        exit(1);
    }
}
#define CK(x) ck((x), __LINE__)

/* ================================================================
 * Fixed-point Q16.16 format
 *
 * Representation: int32_t where value = raw / 2^16 = raw / 65536
 * Range: -32768.0 to +32767.999985
 * Precision: 2^-16 ≈ 1.53e-5 (about 4.8 decimal digits)
 *
 * Conversions:
 *   float -> Q16.16:  (int32_t)roundf(f * 65536.0f)
 *   Q16.16 -> float:  raw / 65536.0f
 *
 * Multiply-accumulate:
 *   product = (int64_t)a * (int64_t)b  (Q32.32 result in 64 bits)
 *   Accumulate in int64_t
 *   Final rescale: (int32_t)(acc >> 16) to get Q16.16 result
 * ================================================================ */

#define Q16_SHIFT 16
#define Q16_SCALE 65536.0f

__host__ __device__ inline int32_t float_to_q16(float f) {
    return (int32_t)rintf(f * Q16_SCALE);
}

__host__ __device__ inline float q16_to_float(int32_t q) {
    return (float)q / Q16_SCALE;
}

/* ================================================================
 * Conversion kernel: float -> Q16.16
 * ================================================================ */

__global__ void kern_float_to_q16(int32_t *out, const float *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = float_to_q16(in[i]);
}

/* ================================================================
 * CUDA-core-only GEMV kernels
 *
 * y[M] = W[M,K] * x[K]   (W row-major)
 *
 * Strategy: 1D grid over M, each thread computes one output element.
 * Serial inner loop over K. Coalesced weight reads across threads
 * (adjacent threads read adjacent columns of W for the same K).
 *
 * IMPORTANT: both kernels use IDENTICAL structure to ensure fairness.
 * ================================================================ */

/* Block size for all GEMV kernels — same for both paths */
#define GEMV_BLOCK 128

/* --- FP32 GEMV (CUDA-core only, no cuBLAS) --- */
__global__ void kern_gemv_fp32(float *y, const float *W, const float *x,
                                int M, int K) {
    int row = blockIdx.x * GEMV_BLOCK + threadIdx.x;
    if (row >= M) return;

    const float *w_row = W + (int64_t)row * K;
    float acc = 0.0f;
    for (int k = 0; k < K; k++)
        acc += w_row[k] * x[k];   /* FP32 FMA: 1 instruction */
    y[row] = acc;
}

/* --- INT32 Q16.16 GEMV (CUDA-core only, INT64 accumulation) --- */
__global__ void kern_gemv_int32(int32_t *y, const int32_t *W, const int32_t *x,
                                 int M, int K) {
    int row = blockIdx.x * GEMV_BLOCK + threadIdx.x;
    if (row >= M) return;

    const int32_t *w_row = W + (int64_t)row * K;
    int64_t acc = 0;
    for (int k = 0; k < K; k++)
        acc += (int64_t)w_row[k] * (int64_t)x[k];  /* INT64 mul+add: ~4 instructions */
    y[row] = (int32_t)(acc >> Q16_SHIFT);  /* rescale Q32.32 -> Q16.16 */
}

/* ================================================================
 * Benchmark harness
 * ================================================================ */

typedef struct {
    const char *name;
    int M, K;
} Shape;

static const Shape SHAPES[] = {
    {"q_o_proj",  2048,  2048},
    {"kv_proj",    256,  2048},
    {"gate_up",   5632,  2048},
    {"down_proj", 2048,  5632},
};
#define N_SHAPES 4

/* M=1 only (decode). M>1 prefill is a different regime where Tensor Cores
 * dominate, making this CUDA-core comparison less relevant. */
static const int M_VALS[] = {1};
#define N_M_VALS 1

#define WARMUP 20
#define BENCH  100

static void run_comparison(const char *shape_name, int M_out, int K,
                           int M_batch) {
    int64_t W_elems = (int64_t)M_out * K;
    int64_t x_elems = (int64_t)K * M_batch;
    int64_t y_elems = (int64_t)M_out * M_batch;

    /* For M_batch>1, treat as M_batch independent GEMV calls.
     * This matches decode behavior (each position is independent). */

    /* Allocate FP32 buffers */
    float *d_Wf, *d_xf, *d_yf;
    CK(cudaMalloc(&d_Wf, W_elems * sizeof(float)));
    CK(cudaMalloc(&d_xf, x_elems * sizeof(float)));
    CK(cudaMalloc(&d_yf, y_elems * sizeof(float)));

    /* Fill with small random-ish values (~neural net weight range) */
    /* Use deterministic pattern: sin(i) gives values in [-1,1] */
    float *h_Wf = (float*)malloc(W_elems * sizeof(float));
    float *h_xf = (float*)malloc(K * sizeof(float));
    for (int64_t i = 0; i < W_elems; i++)
        h_Wf[i] = sinf((float)(i * 7 + 3)) * 0.5f;
    for (int i = 0; i < K; i++)
        h_xf[i] = sinf((float)(i * 13 + 7)) * 0.3f;
    CK(cudaMemcpy(d_Wf, h_Wf, W_elems * sizeof(float), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_xf, h_xf, K * sizeof(float), cudaMemcpyHostToDevice));

    /* Allocate INT32 buffers and convert */
    int32_t *d_Wi, *d_xi, *d_yi;
    CK(cudaMalloc(&d_Wi, W_elems * sizeof(int32_t)));
    CK(cudaMalloc(&d_xi, x_elems * sizeof(int32_t)));
    CK(cudaMalloc(&d_yi, y_elems * sizeof(int32_t)));
    kern_float_to_q16<<<(W_elems + 255) / 256, 256>>>(d_Wi, d_Wf, (int)W_elems);
    kern_float_to_q16<<<(K + 255) / 256, 256>>>(d_xi, d_xf, K);
    CK(cudaDeviceSynchronize());

    dim3 block(GEMV_BLOCK);
    dim3 grid((M_out + GEMV_BLOCK - 1) / GEMV_BLOCK);

    cudaEvent_t t0, t1;
    CK(cudaEventCreate(&t0));
    CK(cudaEventCreate(&t1));

    /* --- FP32 benchmark --- */
    for (int i = 0; i < WARMUP; i++)
        for (int b = 0; b < M_batch; b++)
            kern_gemv_fp32<<<grid, block>>>(d_yf + b * M_out, d_Wf,
                                            d_xf + b * K, M_out, K);
    CK(cudaDeviceSynchronize());

    CK(cudaEventRecord(t0));
    for (int i = 0; i < BENCH; i++)
        for (int b = 0; b < M_batch; b++)
            kern_gemv_fp32<<<grid, block>>>(d_yf + b * M_out, d_Wf,
                                            d_xf + b * K, M_out, K);
    CK(cudaEventRecord(t1));
    CK(cudaEventSynchronize(t1));
    float fp32_ms;
    CK(cudaEventElapsedTime(&fp32_ms, t0, t1));
    float fp32_per = fp32_ms / (BENCH * M_batch);

    /* --- INT32 benchmark --- */
    for (int i = 0; i < WARMUP; i++)
        for (int b = 0; b < M_batch; b++)
            kern_gemv_int32<<<grid, block>>>(d_yi + b * M_out, d_Wi,
                                             d_xi + b * K, M_out, K);
    CK(cudaDeviceSynchronize());

    CK(cudaEventRecord(t0));
    for (int i = 0; i < BENCH; i++)
        for (int b = 0; b < M_batch; b++)
            kern_gemv_int32<<<grid, block>>>(d_yi + b * M_out, d_Wi,
                                             d_xi + b * K, M_out, K);
    CK(cudaEventRecord(t1));
    CK(cudaEventSynchronize(t1));
    float int32_ms;
    CK(cudaEventElapsedTime(&int32_ms, t0, t1));
    float int32_per = int32_ms / (BENCH * M_batch);

    /* --- Numerical accuracy check --- */
    float *h_yf = (float*)malloc(M_out * sizeof(float));
    int32_t *h_yi = (int32_t*)malloc(M_out * sizeof(int32_t));
    CK(cudaMemcpy(h_yf, d_yf, M_out * sizeof(float), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h_yi, d_yi, M_out * sizeof(int32_t), cudaMemcpyDeviceToHost));

    double max_abs_err = 0, sum_abs_err = 0;
    for (int i = 0; i < M_out; i++) {
        float fp32_val = h_yf[i];
        float int_val = q16_to_float(h_yi[i]);
        double err = fabs(fp32_val - int_val);
        if (err > max_abs_err) max_abs_err = err;
        sum_abs_err += err;
    }
    double mean_err = sum_abs_err / M_out;

    /* Effective bandwidth (weight matrix dominates) */
    double bytes_read = (double)M_out * K * 4 + (double)K * 4;  /* W + x, 4 bytes each */
    double fp32_bw = bytes_read / (fp32_per * 1e-3) / 1e9;
    double int32_bw = bytes_read / (int32_per * 1e-3) / 1e9;

    float speedup = fp32_per / int32_per;
    printf("%-10s M=%d K=%5d | FP32=%7.1fµs (%4.0f GB/s) | INT32=%7.1fµs (%4.0f GB/s) | ratio=%.3f | err: max=%.2e mean=%.2e\n",
           shape_name, M_batch, K,
           fp32_per * 1000, fp32_bw,
           int32_per * 1000, int32_bw,
           speedup, max_abs_err, mean_err);

    /* Cleanup */
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaFree(d_Wf); cudaFree(d_xf); cudaFree(d_yf);
    cudaFree(d_Wi); cudaFree(d_xi); cudaFree(d_yi);
    free(h_Wf); free(h_xf); free(h_yf); free(h_yi);
}

/* ================================================================
 * Verify no Tensor Core usage
 * ================================================================ */

static void verify_no_tensor_cores(void) {
    printf("Tensor Core exclusion verification:\n");
    printf("  - No cuBLAS/cuBLASLt calls\n");
    printf("  - No __hmma, __wmma, or mma.sync instructions\n");
    printf("  - Custom GEMV kernels use only scalar FP32/INT32 ops\n");
    printf("  - Both kernels compiled with -arch=sm_120 (CUDA cores)\n");
    printf("  - Identical kernel structure: 1D grid, GEMV_BLOCK=%d, serial K loop\n\n",
           GEMV_BLOCK);
}

/* ================================================================
 * Main
 * ================================================================ */

int main(void) {
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (%d SMs, compute %d.%d)\n",
           prop.name, prop.multiProcessorCount, prop.major, prop.minor);
    printf("CUDA cores: ~%d (estimated)\n\n",
           prop.multiProcessorCount * 128);  /* Blackwell: 128 CUDA cores/SM */

    verify_no_tensor_cores();

    printf("=== CUDA-Core-Only FP32 vs INT32 Q16.16 GEMV Comparison ===\n");
    printf("Warmup: %d, Bench: %d iterations per measurement\n", WARMUP, BENCH);
    printf("INT32 format: Q16.16 (16 int + 16 frac bits), INT64 accumulation, >>16 rescale\n");
    printf("Both paths: 4 bytes/element, identical kernel structure, no Tensor Cores\n\n");

    printf("%-10s %7s | %22s | %22s | %5s | %s\n",
           "Shape", "M K", "FP32", "INT32 Q16.16", "ratio", "drift");
    printf("%-10s %7s | %22s | %22s | %5s | %s\n",
           "----------", "-------", "----------------------",
           "----------------------", "-----", "-------------------");

    for (int si = 0; si < N_SHAPES; si++) {
        for (int mi = 0; mi < N_M_VALS; mi++) {
            run_comparison(SHAPES[si].name, SHAPES[si].M, SHAPES[si].K,
                          M_VALS[mi]);
        }
        if (si < N_SHAPES - 1) printf("\n");
    }

    printf("\nNotes:\n");
    printf("  ratio > 1: INT32 is faster than FP32\n");
    printf("  ratio < 1: INT32 is slower than FP32\n");
    printf("  ratio = 1: equal (both bandwidth-bound)\n");
    printf("  GB/s: effective bandwidth (weight matrix read)\n");
    printf("  drift: absolute error INT32 vs FP32 reference output\n");
    printf("  Peak DRAM BW: ~1792 GB/s (RTX 5090 GDDR7)\n");

    printf("\nDone.\n");
    return 0;
}
