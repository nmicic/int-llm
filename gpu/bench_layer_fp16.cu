/*
 * bench_layer_fp16.cu — Standalone FP16 layer-path benchmark for TinyLlama
 *
 * Loads one transformer layer's weights directly from safetensors (BF16),
 * converts to FP16, runs the full layer forward on GPU, and compares
 * against existing float32 reference tensors for drift measurement.
 *
 * Forward pass implemented:
 *   RMSNorm -> Q/K/V proj -> RoPE -> KV cache -> Attention (GQA) ->
 *   O proj -> Residual -> RMSNorm -> Gate/Up proj -> SiLU*Up ->
 *   Down proj -> Residual
 *
 * Build:
 *   NVCC=/usr/local/cuda-13.0/bin/nvcc
 *   $NVCC -O3 -arch=sm_120 bench_layer_fp16.cu -o bench_layer_fp16 -lcublas -lm
 *
 * Run:
 *   ./bench_layer_fp16 /path/to/TinyLlama-model-dir
 *
 * Copyright (c) 2026 — MicroGPT-C R11 GPU feasibility
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

/* ================================================================
 * Error helpers
 * ================================================================ */

#define CUDA_CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(e)); exit(1); \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t s = (call); \
    if (s != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, (int)s); \
        exit(1); \
    } \
} while(0)

/* ================================================================
 * GPU timer
 * ================================================================ */

typedef struct { cudaEvent_t a, b; } GTimer;
static void gt_create(GTimer *t) {
    CUDA_CHECK(cudaEventCreate(&t->a)); CUDA_CHECK(cudaEventCreate(&t->b));
}
static void gt_destroy(GTimer *t) { cudaEventDestroy(t->a); cudaEventDestroy(t->b); }
static void gt_start(GTimer *t) { CUDA_CHECK(cudaEventRecord(t->a, 0)); }
static void gt_stop(GTimer *t) {
    CUDA_CHECK(cudaEventRecord(t->b, 0)); CUDA_CHECK(cudaEventSynchronize(t->b));
}
static float gt_ms(GTimer *t) {
    float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, t->a, t->b)); return ms;
}

/* ================================================================
 * TinyLlama config
 * ================================================================ */

#define DIM          2048
#define N_HEADS      32
#define N_KV_HEADS   4
#define HEAD_DIM     64
#define INTER_DIM    5632
#define KV_DIM       (N_KV_HEADS * HEAD_DIM)   /* 256 */
#define GQA_GROUP    (N_HEADS / N_KV_HEADS)    /* 8   */
#define MAX_SEQ      256                        /* enough for benchmark */
#define ROPE_THETA   10000.0f
#define RMSNORM_EPS  1e-5f
#define BOS_TOKEN    1

/* ================================================================
 * Minimal safetensors loader
 * ================================================================ */

typedef struct {
    uint8_t *data;        /* mmap'd file */
    size_t   file_size;
    char    *json_hdr;    /* parsed JSON header string */
    size_t   hdr_len;
    size_t   data_start;  /* offset where tensor data begins */
} SafeTensors;

static int st_open(SafeTensors *st, const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror(path); return -1; }
    struct stat sb;
    fstat(fd, &sb);
    st->file_size = sb.st_size;
    st->data = (uint8_t*)mmap(NULL, st->file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (st->data == MAP_FAILED) { perror("mmap"); return -1; }

    /* Header: 8-byte LE uint64 length, then JSON */
    memcpy(&st->hdr_len, st->data, 8);
    st->json_hdr = (char*)(st->data + 8);
    st->data_start = 8 + st->hdr_len;
    return 0;
}

static void st_close(SafeTensors *st) {
    if (st->data) munmap(st->data, st->file_size);
}

/* Find a tensor's data_offsets in the JSON header.
 * Returns pointer to raw BF16 data and sets *count to number of elements. */
static const uint16_t *st_find_bf16(SafeTensors *st, const char *name,
                                     int64_t *count, int *shape0, int *shape1) {
    /* Search for "name": {..., "data_offsets": [start, end]} */
    char key[256];
    snprintf(key, sizeof(key), "\"%s\"", name);
    const char *p = strstr(st->json_hdr, key);
    if (!p) { fprintf(stderr, "tensor not found: %s\n", name); return NULL; }

    /* Find data_offsets */
    const char *ofs = strstr(p, "\"data_offsets\"");
    if (!ofs) return NULL;
    ofs = strchr(ofs, '[');
    if (!ofs) return NULL;
    int64_t start, end;
    sscanf(ofs + 1, "%ld, %ld", &start, &end);
    *count = (end - start) / 2;  /* BF16 = 2 bytes each */

    /* Find shape */
    const char *sh = strstr(p, "\"shape\"");
    if (sh) {
        sh = strchr(sh, '[');
        if (sh) {
            int s0 = 0, s1 = 0;
            if (sscanf(sh + 1, "%d, %d", &s0, &s1) == 2) {
                if (shape0) *shape0 = s0;
                if (shape1) *shape1 = s1;
            } else {
                sscanf(sh + 1, "%d", &s0);
                if (shape0) *shape0 = s0;
                if (shape1) *shape1 = 1;
            }
        }
    }

    return (const uint16_t*)(st->data + st->data_start + start);
}

/* ================================================================
 * BF16 -> FP16 conversion
 * ================================================================ */

/* Host-side: convert one BF16 to float */
static float bf16_to_float(uint16_t v) {
    uint32_t bits = (uint32_t)v << 16;
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

/* GPU kernel: convert BF16 array to FP16 array */
__global__ void kern_bf16_to_fp16(half *out, const uint16_t *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    /* BF16: [1 sign][8 exp][7 mantissa] -> float32 -> FP16 */
    uint32_t bits = (uint32_t)in[i] << 16;
    float f;
    memcpy(&f, &bits, 4);
    out[i] = __float2half(f);
}

static void convert_bf16_to_fp16(half *d_out, const uint16_t *h_bf16, int n) {
    uint16_t *d_bf16;
    CUDA_CHECK(cudaMalloc(&d_bf16, n * 2));
    CUDA_CHECK(cudaMemcpy(d_bf16, h_bf16, n * 2, cudaMemcpyHostToDevice));
    kern_bf16_to_fp16<<<(n + 255) / 256, 256>>>(d_out, d_bf16, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_bf16));
}

/* ================================================================
 * CUDA kernels for transformer ops
 * ================================================================ */

/* RMSNorm: out = (x / sqrt(mean(x^2) + eps)) * weight
 * FP32 accumulation for numerical stability.
 * Uses 1024 threads with stride to handle dim > 1024 (CUDA max threads/block). */
#define NORM_THREADS 1024
__global__ void kern_rmsnorm(half *out, const half *x, const half *weight, int dim) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;

    /* Phase 1: each thread accumulates sum-of-squares for its strided elements */
    float partial = 0.0f;
    for (int i = tid; i < dim; i += NORM_THREADS)
        { float v = __half2float(x[i]); partial += v * v; }
    smem[tid] = partial;
    __syncthreads();

    /* Phase 2: parallel reduction across 1024 threads */
    for (int s = NORM_THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }

    /* Phase 3: normalize with stride */
    float inv_rms = rsqrtf(smem[0] / dim + RMSNORM_EPS);
    for (int i = tid; i < dim; i += NORM_THREADS) {
        float v = __half2float(x[i]);
        float w = __half2float(weight[i]);
        out[i] = __float2half(v * inv_rms * w);
    }
}

/* RoPE: apply rotary embedding to a head vector [head_dim] at position pos.
 * cos_table, sin_table: [max_seq * half_dim], precomputed. */
__global__ void kern_rope(half *vec, const float *cos_tbl, const float *sin_tbl,
                           int pos, int n_heads_to_rope, int half_dim) {
    /* One thread per dimension pair, across all heads */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_heads_to_rope * half_dim;
    if (idx >= total) return;

    int head = idx / half_dim;
    int i    = idx % half_dim;
    int base = head * (half_dim * 2);  /* start of this head's data */
    int tbl  = pos * half_dim + i;

    float x0 = __half2float(vec[base + i]);
    float x1 = __half2float(vec[base + i + half_dim]);
    float c  = cos_tbl[tbl];
    float s  = sin_tbl[tbl];

    vec[base + i]            = __float2half(x0 * c - x1 * s);
    vec[base + i + half_dim] = __float2half(x1 * c + x0 * s);
}

/* Attention: single-position decode with GQA.
 * For each of n_heads Q heads, compute:
 *   scores[t] = Q_h · K_cache[t, kv_h] / sqrt(head_dim)
 *   attn_out[h] = softmax(scores) @ V_cache[:, kv_h]
 *
 * Grid: n_heads blocks, HEAD_DIM threads per block.
 */
__global__ void kern_attention(
    half *attn_out,           /* [n_heads * head_dim] */
    const half *q,            /* [n_heads * head_dim] */
    const half *kv_cache_k,   /* [seq_len * n_kv_heads * head_dim] */
    const half *kv_cache_v,   /* [seq_len * n_kv_heads * head_dim] */
    int seq_len, int n_heads, int n_kv_heads, int head_dim, int gqa_group)
{
    int h = blockIdx.x;
    int d = threadIdx.x;
    if (h >= n_heads || d >= head_dim) return;

    int kv_h = h / gqa_group;
    const half *q_h = q + h * head_dim;

    extern __shared__ float sh[];
    float *scores = sh;                    /* [seq_len] */
    float *vsum   = sh + seq_len;          /* [head_dim] (reuse after softmax) */

    /* --- Compute attention scores --- */
    /* Each thread handles one dimension d, cooperatively computes dot products */
    for (int t = 0; t < seq_len; t++) {
        const half *k_t = kv_cache_k + t * n_kv_heads * head_dim + kv_h * head_dim;
        float prod = __half2float(q_h[d]) * __half2float(k_t[d]);

        /* Warp-level reduction across dimensions */
        /* head_dim=64, so 2 warps. Use shared memory reduction. */
        sh[seq_len + d] = prod;  /* temporary storage */
        __syncthreads();

        if (d == 0) {
            float dot = 0.0f;
            for (int dd = 0; dd < head_dim; dd++)
                dot += sh[seq_len + dd];
            scores[t] = dot * rsqrtf((float)head_dim);  /* scale by 1/sqrt(D) */
        }
        __syncthreads();
    }

    /* --- Softmax (thread 0 only) --- */
    if (d == 0) {
        float max_s = scores[0];
        for (int t = 1; t < seq_len; t++)
            if (scores[t] > max_s) max_s = scores[t];
        float sum_e = 0.0f;
        for (int t = 0; t < seq_len; t++) {
            scores[t] = expf(scores[t] - max_s);
            sum_e += scores[t];
        }
        float inv = 1.0f / sum_e;
        for (int t = 0; t < seq_len; t++)
            scores[t] *= inv;
    }
    __syncthreads();

    /* --- Weighted sum of V --- */
    float acc = 0.0f;
    for (int t = 0; t < seq_len; t++) {
        const half *v_t = kv_cache_v + t * n_kv_heads * head_dim + kv_h * head_dim;
        acc += scores[t] * __half2float(v_t[d]);
    }
    attn_out[h * head_dim + d] = __float2half(acc);
}

/* SiLU(gate) * up, element-wise. SiLU(x) = x * sigmoid(x). */
__global__ void kern_silu_mul(half *out, const half *gate, const half *up, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = __half2float(gate[i]);
    float u = __half2float(up[i]);
    float sig = 1.0f / (1.0f + expf(-g));
    out[i] = __float2half(g * sig * u);
}

/* Residual: hidden += addition */
__global__ void kern_residual_add(half *hidden, const half *addition, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float h = __half2float(hidden[i]);
    float a = __half2float(addition[i]);
    hidden[i] = __float2half(h + a);
}

/* ================================================================
 * cuBLAS matrix-vector multiply wrapper
 * y[M] = W[M,K] * x[K]  (W stored row-major)
 * ================================================================ */

static void matmul_fp16(cublasHandle_t handle,
                         half *d_y, const half *d_W, const half *d_x,
                         int M, int K) {
    __half alpha = __float2half(1.0f);
    __half beta  = __float2half(0.0f);
    /* Row-major W[M,K] is K×M in col-major. Transpose to get M×K. */
    CUBLAS_CHECK(cublasHgemm(handle,
                             CUBLAS_OP_T, CUBLAS_OP_N,
                             M, 1, K,
                             &alpha,
                             d_W, K,    /* A: K×M col-major, transposed to M×K */
                             d_x, K,    /* B: K×1 col-major */
                             &beta,
                             d_y, M));  /* C: M×1 col-major */
}

/* ================================================================
 * Layer weight structure
 * ================================================================ */

typedef struct {
    half *input_norm;      /* [DIM] */
    half *q_proj;          /* [DIM * DIM] */
    half *k_proj;          /* [KV_DIM * DIM] */
    half *v_proj;          /* [KV_DIM * DIM] */
    half *o_proj;          /* [DIM * DIM] */
    half *post_attn_norm;  /* [DIM] */
    half *gate_proj;       /* [INTER_DIM * DIM] */
    half *up_proj;         /* [INTER_DIM * DIM] */
    half *down_proj;       /* [DIM * INTER_DIM] */
} LayerWeights;

/* Scratch buffers for one forward pass */
typedef struct {
    half *normed;       /* [DIM] */
    half *q;            /* [DIM] = n_heads * head_dim */
    half *k;            /* [KV_DIM] */
    half *v;            /* [KV_DIM] */
    half *attn_out;     /* [DIM] */
    half *o_out;        /* [DIM] */
    half *normed2;      /* [DIM] */
    half *gate;         /* [INTER_DIM] */
    half *up;           /* [INTER_DIM] */
    half *silu_up;      /* [INTER_DIM] */
    half *mlp_out;      /* [DIM] */
    /* KV cache */
    half *kv_cache_k;   /* [MAX_SEQ * KV_DIM] */
    half *kv_cache_v;   /* [MAX_SEQ * KV_DIM] */
    /* RoPE tables (FP32 for accuracy) */
    float *rope_cos;    /* [MAX_SEQ * HEAD_DIM/2] */
    float *rope_sin;    /* [MAX_SEQ * HEAD_DIM/2] */
} Scratch;

static void alloc_scratch(Scratch *s) {
    CUDA_CHECK(cudaMalloc(&s->normed,   DIM * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&s->q,        DIM * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&s->k,        KV_DIM * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&s->v,        KV_DIM * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&s->attn_out, DIM * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&s->o_out,    DIM * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&s->normed2,  DIM * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&s->gate,     INTER_DIM * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&s->up,       INTER_DIM * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&s->silu_up,  INTER_DIM * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&s->mlp_out,  DIM * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&s->kv_cache_k, MAX_SEQ * KV_DIM * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&s->kv_cache_v, MAX_SEQ * KV_DIM * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&s->rope_cos, MAX_SEQ * (HEAD_DIM / 2) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s->rope_sin, MAX_SEQ * (HEAD_DIM / 2) * sizeof(float)));
    /* Zero KV cache */
    CUDA_CHECK(cudaMemset(s->kv_cache_k, 0, MAX_SEQ * KV_DIM * sizeof(half)));
    CUDA_CHECK(cudaMemset(s->kv_cache_v, 0, MAX_SEQ * KV_DIM * sizeof(half)));
}

static void free_scratch(Scratch *s) {
    cudaFree(s->normed); cudaFree(s->q); cudaFree(s->k); cudaFree(s->v);
    cudaFree(s->attn_out); cudaFree(s->o_out); cudaFree(s->normed2);
    cudaFree(s->gate); cudaFree(s->up); cudaFree(s->silu_up);
    cudaFree(s->mlp_out); cudaFree(s->kv_cache_k); cudaFree(s->kv_cache_v);
    cudaFree(s->rope_cos); cudaFree(s->rope_sin);
}

/* ================================================================
 * RoPE table initialization
 * ================================================================ */

static void init_rope_tables(Scratch *s) {
    int half_dim = HEAD_DIM / 2;
    float *h_cos = (float*)malloc(MAX_SEQ * half_dim * sizeof(float));
    float *h_sin = (float*)malloc(MAX_SEQ * half_dim * sizeof(float));

    for (int i = 0; i < half_dim; i++) {
        float theta_i = powf(ROPE_THETA, -2.0f * i / HEAD_DIM);
        for (int pos = 0; pos < MAX_SEQ; pos++) {
            float angle = pos * theta_i;
            h_cos[pos * half_dim + i] = cosf(angle);
            h_sin[pos * half_dim + i] = sinf(angle);
        }
    }

    CUDA_CHECK(cudaMemcpy(s->rope_cos, h_cos, MAX_SEQ * half_dim * sizeof(float),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s->rope_sin, h_sin, MAX_SEQ * half_dim * sizeof(float),
                           cudaMemcpyHostToDevice));
    free(h_cos);
    free(h_sin);
}

/* ================================================================
 * Full layer forward pass
 * ================================================================ */

static void layer_forward(cublasHandle_t handle, half *d_hidden,
                            LayerWeights *w, Scratch *s, int pos) {
    int seq_len = pos + 1;
    int half_dim = HEAD_DIM / 2;
    int blk256 = 256;

    /* 1. Pre-attention RMSNorm */
    kern_rmsnorm<<<1, NORM_THREADS, NORM_THREADS * sizeof(float)>>>(
        s->normed, d_hidden, w->input_norm, DIM);

    /* 2. Q/K/V projections */
    matmul_fp16(handle, s->q, w->q_proj, s->normed, DIM, DIM);
    matmul_fp16(handle, s->k, w->k_proj, s->normed, KV_DIM, DIM);
    matmul_fp16(handle, s->v, w->v_proj, s->normed, KV_DIM, DIM);

    /* 3. RoPE */
    int rope_q_total = N_HEADS * half_dim;
    kern_rope<<<(rope_q_total + blk256 - 1) / blk256, blk256>>>(
        s->q, s->rope_cos, s->rope_sin, pos, N_HEADS, half_dim);
    int rope_k_total = N_KV_HEADS * half_dim;
    kern_rope<<<(rope_k_total + blk256 - 1) / blk256, blk256>>>(
        s->k, s->rope_cos, s->rope_sin, pos, N_KV_HEADS, half_dim);

    /* 4. Store K/V in cache at position pos */
    CUDA_CHECK(cudaMemcpy(s->kv_cache_k + pos * KV_DIM, s->k,
                           KV_DIM * sizeof(half), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(s->kv_cache_v + pos * KV_DIM, s->v,
                           KV_DIM * sizeof(half), cudaMemcpyDeviceToDevice));

    /* 5. Attention (GQA) */
    int attn_smem = (seq_len + HEAD_DIM) * sizeof(float);
    kern_attention<<<N_HEADS, HEAD_DIM, attn_smem>>>(
        s->attn_out, s->q, s->kv_cache_k, s->kv_cache_v,
        seq_len, N_HEADS, N_KV_HEADS, HEAD_DIM, GQA_GROUP);

    /* 6. O projection */
    matmul_fp16(handle, s->o_out, w->o_proj, s->attn_out, DIM, DIM);

    /* 7. Residual add */
    kern_residual_add<<<(DIM + blk256 - 1) / blk256, blk256>>>(
        d_hidden, s->o_out, DIM);

    /* 8. Post-attention RMSNorm */
    kern_rmsnorm<<<1, NORM_THREADS, NORM_THREADS * sizeof(float)>>>(
        s->normed2, d_hidden, w->post_attn_norm, DIM);

    /* 9. MLP: gate/up projections */
    matmul_fp16(handle, s->gate, w->gate_proj, s->normed2, INTER_DIM, DIM);
    matmul_fp16(handle, s->up,   w->up_proj,   s->normed2, INTER_DIM, DIM);

    /* 10. SiLU(gate) * up */
    kern_silu_mul<<<(INTER_DIM + blk256 - 1) / blk256, blk256>>>(
        s->silu_up, s->gate, s->up, INTER_DIM);

    /* 11. Down projection */
    matmul_fp16(handle, s->mlp_out, w->down_proj, s->silu_up, DIM, INTER_DIM);

    /* 12. Residual add */
    kern_residual_add<<<(DIM + blk256 - 1) / blk256, blk256>>>(
        d_hidden, s->mlp_out, DIM);
}

/* ================================================================
 * Load reference tensor from float32 binary file
 * ================================================================ */

static float *load_ref(const char *dir, const char *name) {
    char path[512];
    snprintf(path, sizeof(path), "%s/%s", dir, name);
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return NULL; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    float *buf = (float*)malloc(sz);
    if ((long)fread(buf, 1, sz, f) != sz) {
        fprintf(stderr, "Short read: %s\n", path);
        free(buf); fclose(f); return NULL;
    }
    fclose(f);
    return buf;
}

/* ================================================================
 * Drift comparison: FP16 GPU output vs float32 reference
 * ================================================================ */

static void compare_drift(const char *label, const half *d_gpu, const float *ref,
                           int n) {
    /* Copy GPU output to host */
    half *h_gpu = (half*)malloc(n * sizeof(half));
    CUDA_CHECK(cudaMemcpy(h_gpu, d_gpu, n * sizeof(half), cudaMemcpyDeviceToHost));

    double max_abs = 0, sum_abs = 0, sum_sq = 0;
    double ref_max_abs = 0;
    for (int i = 0; i < n; i++) {
        float gpu_val = __half2float(h_gpu[i]);
        float ref_val = ref[i];
        double diff = fabs(gpu_val - ref_val);
        if (diff > max_abs) max_abs = diff;
        sum_abs += diff;
        sum_sq += diff * diff;
        if (fabs(ref_val) > ref_max_abs) ref_max_abs = fabs(ref_val);
    }
    double mean_abs = sum_abs / n;
    double rmse = sqrt(sum_sq / n);
    double rel_max = (ref_max_abs > 0) ? max_abs / ref_max_abs : 0;

    printf("  %-22s max_abs=%.6f  mean_abs=%.6f  rmse=%.6f  rel_max=%.4f%%\n",
           label, max_abs, mean_abs, rmse, rel_max * 100);
    free(h_gpu);
}

/* ================================================================
 * Main
 * ================================================================ */

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model_dir> [layer_idx]\n", argv[0]);
        return 1;
    }
    const char *model_dir = argv[1];
    int layer_idx = (argc > 2) ? atoi(argv[2]) : 0;

    /* GPU info */
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (%d SMs, compute %d.%d)\n",
           prop.name, prop.multiProcessorCount, prop.major, prop.minor);

    /* Open safetensors */
    char st_path[512];
    snprintf(st_path, sizeof(st_path), "%s/model.safetensors", model_dir);
    SafeTensors st;
    if (st_open(&st, st_path) != 0) return 1;
    printf("Loaded safetensors: header=%zu bytes, data_start=%zu\n",
           st.hdr_len, st.data_start);

    /* Load layer weights */
    printf("\nLoading layer %d weights (BF16 -> FP16)...\n", layer_idx);
    LayerWeights w;
    char tname[128];
    int64_t cnt;
    int s0, s1;

    #define LOAD_WEIGHT(field, name, expected_count) do { \
        snprintf(tname, sizeof(tname), "model.layers.%d.%s", layer_idx, name); \
        const uint16_t *raw = st_find_bf16(&st, tname, &cnt, &s0, &s1); \
        if (!raw) { fprintf(stderr, "FATAL: %s not found\n", tname); return 1; } \
        if (cnt != (expected_count)) { \
            fprintf(stderr, "Shape mismatch: %s got %ld expected %d\n", \
                    tname, (long)cnt, (expected_count)); return 1; } \
        CUDA_CHECK(cudaMalloc(&w.field, cnt * sizeof(half))); \
        convert_bf16_to_fp16(w.field, raw, (int)cnt); \
        printf("  %-40s [%d x %d] %ld elements\n", tname, s0, s1, (long)cnt); \
    } while(0)

    LOAD_WEIGHT(input_norm,    "input_layernorm.weight",           DIM);
    LOAD_WEIGHT(q_proj,        "self_attn.q_proj.weight",          DIM * DIM);
    LOAD_WEIGHT(k_proj,        "self_attn.k_proj.weight",          KV_DIM * DIM);
    LOAD_WEIGHT(v_proj,        "self_attn.v_proj.weight",          KV_DIM * DIM);
    LOAD_WEIGHT(o_proj,        "self_attn.o_proj.weight",          DIM * DIM);
    LOAD_WEIGHT(post_attn_norm,"post_attention_layernorm.weight",  DIM);
    LOAD_WEIGHT(gate_proj,     "mlp.gate_proj.weight",             INTER_DIM * DIM);
    LOAD_WEIGHT(up_proj,       "mlp.up_proj.weight",               INTER_DIM * DIM);
    LOAD_WEIGHT(down_proj,     "mlp.down_proj.weight",             DIM * INTER_DIM);
    #undef LOAD_WEIGHT

    /* Load BOS embedding */
    const uint16_t *embed_raw = st_find_bf16(&st, "model.embed_tokens.weight",
                                               &cnt, &s0, &s1);
    if (!embed_raw) { fprintf(stderr, "embed_tokens not found\n"); return 1; }
    printf("  embed_tokens: [%d, %d] -> extracting token %d\n", s0, s1, BOS_TOKEN);
    half *d_hidden;
    CUDA_CHECK(cudaMalloc(&d_hidden, DIM * sizeof(half)));
    convert_bf16_to_fp16(d_hidden, embed_raw + BOS_TOKEN * DIM, DIM);

    /* cuBLAS handle */
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    /* Allocate scratch and init RoPE */
    Scratch scr;
    alloc_scratch(&scr);
    init_rope_tables(&scr);

    /* ============================================================
     * Correctness pass: layer N, pos=0 (BOS token)
     * Reference tensors only exist for layer 0.
     * ============================================================ */
    printf("\n=== Correctness Pass: layer %d, pos=0 (BOS) ===\n", layer_idx);

    float *ref_embed = NULL, *ref_norm = NULL, *ref_q = NULL, *ref_k = NULL;
    float *ref_v = NULL, *ref_q_rope = NULL, *ref_attn = NULL, *ref_post = NULL;
    float *ref_postnm = NULL, *ref_mlp = NULL, *ref_out = NULL;

    if (layer_idx == 0) {
        ref_embed  = load_ref(model_dir, "layer0_ref_embedding.bin");
        ref_norm   = load_ref(model_dir, "layer0_ref_rmsnorm.bin");
        ref_q      = load_ref(model_dir, "layer0_ref_q_proj.bin");
        ref_k      = load_ref(model_dir, "layer0_ref_k_proj.bin");
        ref_v      = load_ref(model_dir, "layer0_ref_v_proj.bin");
        ref_q_rope = load_ref(model_dir, "layer0_ref_q_rope.bin");
        ref_attn   = load_ref(model_dir, "layer0_ref_attn_out.bin");
        ref_post   = load_ref(model_dir, "layer0_ref_post_attn.bin");
        ref_postnm = load_ref(model_dir, "layer0_ref_post_attn_norm.bin");
        ref_mlp    = load_ref(model_dir, "layer0_ref_mlp_out.bin");
        ref_out    = load_ref(model_dir, "layer0_ref_layer_out.bin");
    } else {
        printf("  (reference tensors only available for layer 0; skipping drift comparison)\n");
    }

    /* Compare BOS embedding */
    if (ref_embed) compare_drift("embedding", d_hidden, ref_embed, DIM);

    /* Run layer with intermediate checks */

    int blk256 = 256;
    int half_dim = HEAD_DIM / 2;

    /* 1. RMSNorm */
    kern_rmsnorm<<<1, NORM_THREADS, NORM_THREADS * sizeof(float)>>>(
        scr.normed, d_hidden, w.input_norm, DIM);
    CUDA_CHECK(cudaDeviceSynchronize());
    if (ref_norm) compare_drift("rmsnorm", scr.normed, ref_norm, DIM);

    /* 2. Q/K/V projections */
    matmul_fp16(handle, scr.q, w.q_proj, scr.normed, DIM, DIM);
    matmul_fp16(handle, scr.k, w.k_proj, scr.normed, KV_DIM, DIM);
    matmul_fp16(handle, scr.v, w.v_proj, scr.normed, KV_DIM, DIM);
    CUDA_CHECK(cudaDeviceSynchronize());
    if (ref_q) compare_drift("q_proj", scr.q, ref_q, DIM);
    if (ref_k) compare_drift("k_proj", scr.k, ref_k, KV_DIM);
    if (ref_v) compare_drift("v_proj", scr.v, ref_v, KV_DIM);

    /* 3. RoPE */
    kern_rope<<<(N_HEADS * half_dim + blk256 - 1) / blk256, blk256>>>(
        scr.q, scr.rope_cos, scr.rope_sin, 0, N_HEADS, half_dim);
    kern_rope<<<(N_KV_HEADS * half_dim + blk256 - 1) / blk256, blk256>>>(
        scr.k, scr.rope_cos, scr.rope_sin, 0, N_KV_HEADS, half_dim);
    CUDA_CHECK(cudaDeviceSynchronize());
    if (ref_q_rope) compare_drift("q_rope", scr.q, ref_q_rope, DIM);

    /* 4. Store KV */
    CUDA_CHECK(cudaMemcpy(scr.kv_cache_k, scr.k, KV_DIM * sizeof(half),
                           cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(scr.kv_cache_v, scr.v, KV_DIM * sizeof(half),
                           cudaMemcpyDeviceToDevice));

    /* 5. Attention */
    int attn_smem = (1 + HEAD_DIM) * sizeof(float);
    kern_attention<<<N_HEADS, HEAD_DIM, attn_smem>>>(
        scr.attn_out, scr.q, scr.kv_cache_k, scr.kv_cache_v,
        1, N_HEADS, N_KV_HEADS, HEAD_DIM, GQA_GROUP);
    CUDA_CHECK(cudaDeviceSynchronize());
    if (ref_attn) compare_drift("attn_out", scr.attn_out, ref_attn, DIM);

    /* 6. O projection + residual */
    matmul_fp16(handle, scr.o_out, w.o_proj, scr.attn_out, DIM, DIM);
    kern_residual_add<<<(DIM + blk256 - 1) / blk256, blk256>>>(
        d_hidden, scr.o_out, DIM);
    CUDA_CHECK(cudaDeviceSynchronize());
    if (ref_post) compare_drift("post_attn+res", d_hidden, ref_post, DIM);

    /* 8. Post-attention RMSNorm */
    kern_rmsnorm<<<1, NORM_THREADS, NORM_THREADS * sizeof(float)>>>(
        scr.normed2, d_hidden, w.post_attn_norm, DIM);
    CUDA_CHECK(cudaDeviceSynchronize());
    if (ref_postnm) compare_drift("post_attn_norm", scr.normed2, ref_postnm, DIM);

    /* 9. MLP */
    matmul_fp16(handle, scr.gate, w.gate_proj, scr.normed2, INTER_DIM, DIM);
    matmul_fp16(handle, scr.up,   w.up_proj,   scr.normed2, INTER_DIM, DIM);
    kern_silu_mul<<<(INTER_DIM + blk256 - 1) / blk256, blk256>>>(
        scr.silu_up, scr.gate, scr.up, INTER_DIM);
    matmul_fp16(handle, scr.mlp_out, w.down_proj, scr.silu_up, DIM, INTER_DIM);
    CUDA_CHECK(cudaDeviceSynchronize());
    if (ref_mlp) compare_drift("mlp_out", scr.mlp_out, ref_mlp, DIM);

    /* 10. Final residual */
    kern_residual_add<<<(DIM + blk256 - 1) / blk256, blk256>>>(
        d_hidden, scr.mlp_out, DIM);
    CUDA_CHECK(cudaDeviceSynchronize());
    if (ref_out) compare_drift("layer_out", d_hidden, ref_out, DIM);

    /* ============================================================
     * Timing pass: measure full layer forward
     * ============================================================ */
    printf("\n=== Timing Pass ===\n");

    /* Re-init hidden to BOS embedding */
    convert_bf16_to_fp16(d_hidden, embed_raw + BOS_TOKEN * DIM, DIM);
    CUDA_CHECK(cudaMemset(scr.kv_cache_k, 0, MAX_SEQ * KV_DIM * sizeof(half)));
    CUDA_CHECK(cudaMemset(scr.kv_cache_v, 0, MAX_SEQ * KV_DIM * sizeof(half)));

    /* Warmup */
    for (int i = 0; i < 5; i++) {
        convert_bf16_to_fp16(d_hidden, embed_raw + BOS_TOKEN * DIM, DIM);
        CUDA_CHECK(cudaMemset(scr.kv_cache_k, 0, MAX_SEQ * KV_DIM * sizeof(half)));
        CUDA_CHECK(cudaMemset(scr.kv_cache_v, 0, MAX_SEQ * KV_DIM * sizeof(half)));
        layer_forward(handle, d_hidden, &w, &scr, 0);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Timed: single layer at pos=0 (decode-like, trivial attention) */
    int N_ITERS = 100;
    GTimer timer;
    gt_create(&timer);

    gt_start(&timer);
    for (int i = 0; i < N_ITERS; i++) {
        convert_bf16_to_fp16(d_hidden, embed_raw + BOS_TOKEN * DIM, DIM);
        CUDA_CHECK(cudaMemset(scr.kv_cache_k, 0, MAX_SEQ * KV_DIM * sizeof(half)));
        CUDA_CHECK(cudaMemset(scr.kv_cache_v, 0, MAX_SEQ * KV_DIM * sizeof(half)));
        layer_forward(handle, d_hidden, &w, &scr, 0);
    }
    gt_stop(&timer);
    float pos0_ms = gt_ms(&timer) / N_ITERS;
    printf("  Layer forward (pos=0, seq_len=1):  %.4f ms\n", pos0_ms);
    printf("  Implied 22-layer decode:           %.3f ms  (%.0f tok/s)\n",
           pos0_ms * 22, 1000.0 / (pos0_ms * 22));

    /* Timed: layer at pos=127 with random KV cache (realistic attention) */
    /* Fill KV cache with small random values */
    int fill_len = 127;
    half *h_rand = (half*)malloc(fill_len * KV_DIM * sizeof(half));
    for (int i = 0; i < fill_len * KV_DIM; i++)
        h_rand[i] = __float2half(0.01f * (i % 100 - 50));
    CUDA_CHECK(cudaMemcpy(scr.kv_cache_k, h_rand, fill_len * KV_DIM * sizeof(half),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(scr.kv_cache_v, h_rand, fill_len * KV_DIM * sizeof(half),
                           cudaMemcpyHostToDevice));
    free(h_rand);

    gt_start(&timer);
    for (int i = 0; i < N_ITERS; i++) {
        convert_bf16_to_fp16(d_hidden, embed_raw + BOS_TOKEN * DIM, DIM);
        layer_forward(handle, d_hidden, &w, &scr, fill_len);
    }
    gt_stop(&timer);
    float pos127_ms = gt_ms(&timer) / N_ITERS;
    printf("  Layer forward (pos=127, seq_len=128): %.4f ms\n", pos127_ms);
    printf("  Implied 22-layer decode:              %.3f ms  (%.0f tok/s)\n",
           pos127_ms * 22, 1000.0 / (pos127_ms * 22));

    /* Component timing breakdown (pos=0, single iteration with per-step timing) */
    printf("\n=== Component Timing (single iteration, pos=0) ===\n");

    convert_bf16_to_fp16(d_hidden, embed_raw + BOS_TOKEN * DIM, DIM);
    CUDA_CHECK(cudaMemset(scr.kv_cache_k, 0, MAX_SEQ * KV_DIM * sizeof(half)));
    CUDA_CHECK(cudaMemset(scr.kv_cache_v, 0, MAX_SEQ * KV_DIM * sizeof(half)));
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Per-component timing (explicit, avoids macro comma issues) */
    float ct;

    gt_start(&timer);
    kern_rmsnorm<<<1, NORM_THREADS, NORM_THREADS * sizeof(float)>>>(
        scr.normed, d_hidden, w.input_norm, DIM);
    gt_stop(&timer); ct = gt_ms(&timer);
    printf("  %-25s %.4f ms\n", "RMSNorm (pre-attn)", ct);

    gt_start(&timer);
    matmul_fp16(handle, scr.q, w.q_proj, scr.normed, DIM, DIM);
    gt_stop(&timer); ct = gt_ms(&timer);
    printf("  %-25s %.4f ms\n", "Q projection", ct);

    gt_start(&timer);
    matmul_fp16(handle, scr.k, w.k_proj, scr.normed, KV_DIM, DIM);
    gt_stop(&timer); ct = gt_ms(&timer);
    printf("  %-25s %.4f ms\n", "K projection", ct);

    gt_start(&timer);
    matmul_fp16(handle, scr.v, w.v_proj, scr.normed, KV_DIM, DIM);
    gt_stop(&timer); ct = gt_ms(&timer);
    printf("  %-25s %.4f ms\n", "V projection", ct);

    gt_start(&timer);
    kern_rope<<<(N_HEADS * half_dim + blk256 - 1) / blk256, blk256>>>(
        scr.q, scr.rope_cos, scr.rope_sin, 0, N_HEADS, half_dim);
    kern_rope<<<(N_KV_HEADS * half_dim + blk256 - 1) / blk256, blk256>>>(
        scr.k, scr.rope_cos, scr.rope_sin, 0, N_KV_HEADS, half_dim);
    gt_stop(&timer); ct = gt_ms(&timer);
    printf("  %-25s %.4f ms\n", "RoPE (Q+K)", ct);

    gt_start(&timer);
    CUDA_CHECK(cudaMemcpy(scr.kv_cache_k, scr.k,
               KV_DIM * sizeof(half), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(scr.kv_cache_v, scr.v,
               KV_DIM * sizeof(half), cudaMemcpyDeviceToDevice));
    gt_stop(&timer); ct = gt_ms(&timer);
    printf("  %-25s %.4f ms\n", "KV cache store", ct);

    gt_start(&timer);
    kern_attention<<<N_HEADS, HEAD_DIM, (1 + HEAD_DIM) * sizeof(float)>>>(
        scr.attn_out, scr.q, scr.kv_cache_k, scr.kv_cache_v,
        1, N_HEADS, N_KV_HEADS, HEAD_DIM, GQA_GROUP);
    gt_stop(&timer); ct = gt_ms(&timer);
    printf("  %-25s %.4f ms\n", "Attention (seq=1)", ct);

    gt_start(&timer);
    matmul_fp16(handle, scr.o_out, w.o_proj, scr.attn_out, DIM, DIM);
    gt_stop(&timer); ct = gt_ms(&timer);
    printf("  %-25s %.4f ms\n", "O projection", ct);

    gt_start(&timer);
    kern_residual_add<<<(DIM + blk256 - 1) / blk256, blk256>>>(
        d_hidden, scr.o_out, DIM);
    gt_stop(&timer); ct = gt_ms(&timer);
    printf("  %-25s %.4f ms\n", "Residual add (attn)", ct);

    gt_start(&timer);
    kern_rmsnorm<<<1, NORM_THREADS, NORM_THREADS * sizeof(float)>>>(
        scr.normed2, d_hidden, w.post_attn_norm, DIM);
    gt_stop(&timer); ct = gt_ms(&timer);
    printf("  %-25s %.4f ms\n", "RMSNorm (post-attn)", ct);

    gt_start(&timer);
    matmul_fp16(handle, scr.gate, w.gate_proj, scr.normed2, INTER_DIM, DIM);
    gt_stop(&timer); ct = gt_ms(&timer);
    printf("  %-25s %.4f ms\n", "Gate projection", ct);

    gt_start(&timer);
    matmul_fp16(handle, scr.up, w.up_proj, scr.normed2, INTER_DIM, DIM);
    gt_stop(&timer); ct = gt_ms(&timer);
    printf("  %-25s %.4f ms\n", "Up projection", ct);

    gt_start(&timer);
    kern_silu_mul<<<(INTER_DIM + blk256 - 1) / blk256, blk256>>>(
        scr.silu_up, scr.gate, scr.up, INTER_DIM);
    gt_stop(&timer); ct = gt_ms(&timer);
    printf("  %-25s %.4f ms\n", "SiLU * Up", ct);

    gt_start(&timer);
    matmul_fp16(handle, scr.mlp_out, w.down_proj, scr.silu_up, DIM, INTER_DIM);
    gt_stop(&timer); ct = gt_ms(&timer);
    printf("  %-25s %.4f ms\n", "Down projection", ct);

    gt_start(&timer);
    kern_residual_add<<<(DIM + blk256 - 1) / blk256, blk256>>>(
        d_hidden, scr.mlp_out, DIM);
    gt_stop(&timer); ct = gt_ms(&timer);
    printf("  %-25s %.4f ms\n", "Residual add (MLP)", ct);

    /* Summary */
    printf("\n=== Summary ===\n");
    printf("  Layer timing (pos=0):   %.4f ms\n", pos0_ms);
    printf("  Layer timing (pos=127): %.4f ms\n", pos127_ms);
    printf("  22-layer decode (pos=0):   %.3f ms  (%.0f tok/s, GEMM+kernel only)\n",
           pos0_ms * 22, 1000.0 / (pos0_ms * 22));
    printf("  22-layer decode (pos=127): %.3f ms  (%.0f tok/s, GEMM+kernel only)\n",
           pos127_ms * 22, 1000.0 / (pos127_ms * 22));
    printf("  Note: no lm_head, no weight loading, no KV management overhead.\n");
    printf("  Note: timing includes embedding re-init per iteration.\n");

    /* Cleanup */
    gt_destroy(&timer);
    free_scratch(&scr);
    cudaFree(d_hidden);
    cudaFree(w.input_norm); cudaFree(w.q_proj); cudaFree(w.k_proj);
    cudaFree(w.v_proj); cudaFree(w.o_proj); cudaFree(w.post_attn_norm);
    cudaFree(w.gate_proj); cudaFree(w.up_proj); cudaFree(w.down_proj);
    cublasDestroy(handle);
    st_close(&st);

    if (ref_embed)  free(ref_embed);
    if (ref_norm)   free(ref_norm);
    if (ref_q)      free(ref_q);
    if (ref_k)      free(ref_k);
    if (ref_v)      free(ref_v);
    if (ref_q_rope) free(ref_q_rope);
    if (ref_attn)   free(ref_attn);
    if (ref_post)   free(ref_post);
    if (ref_postnm) free(ref_postnm);
    if (ref_mlp)    free(ref_mlp);
    if (ref_out)    free(ref_out);

    printf("\nDone.\n");
    return 0;
}
