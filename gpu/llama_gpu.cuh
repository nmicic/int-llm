/*
 * llama_gpu.cuh — Reusable FP16 GPU runtime for TinyLlama inference
 *
 * Header-only CUDA runtime: safetensors loading, BF16->FP16 conversion,
 * all transformer kernels, full forward pass, and greedy generation.
 *
 * Include from a .cu file and call:
 *   llama_gpu_init()      — load model from safetensors directory
 *   llama_gpu_generate()  — prefill + greedy decode
 *   llama_gpu_free()      — release all GPU/host resources
 *
 * Config is currently hardcoded to TinyLlama-1.1B. Future work may
 * read config.json dynamically.
 *
 * Copyright (c) 2026 — MicroGPT-C R11 GPU runtime
 */

#ifndef LLAMA_GPU_CUH
#define LLAMA_GPU_CUH

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
 * Error macros
 * ================================================================ */

#define CUDA_CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(e)); exit(1); } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t s = (call); \
    if (s != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, (int)s); \
        exit(1); } \
} while(0)

/* ================================================================
 * TinyLlama config (hardcoded for now)
 * ================================================================ */

#define LGPU_DIM          2048
#define LGPU_N_HEADS      32
#define LGPU_N_KV_HEADS   4
#define LGPU_HEAD_DIM     64
#define LGPU_INTER_DIM    5632
#define LGPU_KV_DIM       (LGPU_N_KV_HEADS * LGPU_HEAD_DIM)  /* 256 */
#define LGPU_GQA_GROUP    (LGPU_N_HEADS / LGPU_N_KV_HEADS)   /* 8   */
#define LGPU_N_LAYERS     22
#define LGPU_VOCAB_SIZE   32000
#define LGPU_MAX_SEQ      256
#define LGPU_ROPE_THETA   10000.0f
#define LGPU_RMSNORM_EPS  1e-5f
#define LGPU_NORM_THREADS 1024

/* ================================================================
 * Minimal safetensors loader
 * ================================================================ */

typedef struct {
    uint8_t *data; size_t file_size;
    char *json_hdr; size_t hdr_len, data_start;
} LGpuSafeTensors;

static int lgpu_st_open(LGpuSafeTensors *st, const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror(path); return -1; }
    struct stat sb; fstat(fd, &sb);
    st->file_size = sb.st_size;
    st->data = (uint8_t*)mmap(NULL, st->file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (st->data == MAP_FAILED) { perror("mmap"); return -1; }
    memcpy(&st->hdr_len, st->data, 8);
    st->json_hdr = (char*)(st->data + 8);
    st->data_start = 8 + st->hdr_len;
    return 0;
}

static void lgpu_st_close(LGpuSafeTensors *st) {
    if (st->data) munmap(st->data, st->file_size);
    st->data = NULL;
}

static const uint16_t *lgpu_st_find_bf16(LGpuSafeTensors *st, const char *name,
                                          int64_t *count) {
    char key[256]; snprintf(key, sizeof(key), "\"%s\"", name);
    const char *p = strstr(st->json_hdr, key);
    if (!p) { fprintf(stderr, "tensor not found: %s\n", name); return NULL; }
    const char *ofs = strstr(p, "\"data_offsets\"");
    if (!ofs) return NULL;
    ofs = strchr(ofs, '['); if (!ofs) return NULL;
    int64_t start, end;
    sscanf(ofs + 1, "%ld, %ld", &start, &end);
    *count = (end - start) / 2;
    return (const uint16_t*)(st->data + st->data_start + start);
}

/* ================================================================
 * BF16 -> FP16 conversion
 * ================================================================ */

__global__ void lgpu_kern_bf16_to_fp16(half *out, const uint16_t *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t bits = (uint32_t)in[i] << 16;
    float f; memcpy(&f, &bits, 4);
    out[i] = __float2half(f);
}

static void lgpu_convert_bf16(half *d_out, const uint16_t *h_bf16, int64_t n) {
    uint16_t *d_bf16;
    CUDA_CHECK(cudaMalloc(&d_bf16, n * 2));
    CUDA_CHECK(cudaMemcpy(d_bf16, h_bf16, n * 2, cudaMemcpyHostToDevice));
    lgpu_kern_bf16_to_fp16<<<((int)n + 255) / 256, 256>>>(d_out, d_bf16, (int)n);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_bf16));
}

/* ================================================================
 * CUDA kernels
 * ================================================================ */

__global__ void lgpu_kern_rmsnorm(half *out, const half *x, const half *weight, int dim) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    float partial = 0.0f;
    for (int i = tid; i < dim; i += LGPU_NORM_THREADS)
        { float v = __half2float(x[i]); partial += v * v; }
    smem[tid] = partial;
    __syncthreads();
    for (int s = LGPU_NORM_THREADS / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float inv_rms = rsqrtf(smem[0] / dim + LGPU_RMSNORM_EPS);
    for (int i = tid; i < dim; i += LGPU_NORM_THREADS) {
        float v = __half2float(x[i]);
        float w = __half2float(weight[i]);
        out[i] = __float2half(v * inv_rms * w);
    }
}

__global__ void lgpu_kern_rope(half *vec, const float *cos_tbl, const float *sin_tbl,
                                int pos, int n_heads_to_rope, int half_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_heads_to_rope * half_dim) return;
    int head = idx / half_dim, i = idx % half_dim;
    int base = head * (half_dim * 2), tbl = pos * half_dim + i;
    float x0 = __half2float(vec[base + i]);
    float x1 = __half2float(vec[base + i + half_dim]);
    float c = cos_tbl[tbl], s = sin_tbl[tbl];
    vec[base + i]            = __float2half(x0 * c - x1 * s);
    vec[base + i + half_dim] = __float2half(x1 * c + x0 * s);
}

__global__ void lgpu_kern_attention(
    half *attn_out, const half *q,
    const half *kv_cache_k, const half *kv_cache_v,
    int seq_len, int n_heads, int n_kv_heads, int head_dim, int gqa_group)
{
    int h = blockIdx.x, d = threadIdx.x;
    if (h >= n_heads || d >= head_dim) return;
    int kv_h = h / gqa_group;
    const half *q_h = q + h * head_dim;
    extern __shared__ float sh[];
    float *scores = sh;
    for (int t = 0; t < seq_len; t++) {
        const half *k_t = kv_cache_k + t * n_kv_heads * head_dim + kv_h * head_dim;
        sh[seq_len + d] = __half2float(q_h[d]) * __half2float(k_t[d]);
        __syncthreads();
        if (d == 0) {
            float dot = 0.0f;
            for (int dd = 0; dd < head_dim; dd++) dot += sh[seq_len + dd];
            scores[t] = dot * rsqrtf((float)head_dim);
        }
        __syncthreads();
    }
    if (d == 0) {
        float max_s = scores[0];
        for (int t = 1; t < seq_len; t++) if (scores[t] > max_s) max_s = scores[t];
        float sum_e = 0.0f;
        for (int t = 0; t < seq_len; t++) { scores[t] = expf(scores[t] - max_s); sum_e += scores[t]; }
        for (int t = 0; t < seq_len; t++) scores[t] /= sum_e;
    }
    __syncthreads();
    float acc = 0.0f;
    for (int t = 0; t < seq_len; t++) {
        const half *v_t = kv_cache_v + t * n_kv_heads * head_dim + kv_h * head_dim;
        acc += scores[t] * __half2float(v_t[d]);
    }
    attn_out[h * head_dim + d] = __float2half(acc);
}

__global__ void lgpu_kern_silu_mul(half *out, const half *gate, const half *up, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = __half2float(gate[i]), u = __half2float(up[i]);
    out[i] = __float2half(g / (1.0f + expf(-g)) * u);
}

__global__ void lgpu_kern_residual_add(half *hidden, const half *addition, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    hidden[i] = __float2half(__half2float(hidden[i]) + __half2float(addition[i]));
}

__global__ void lgpu_kern_embed_lookup(half *out, const half *table, int token_id, int dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;
    out[i] = table[(int64_t)token_id * dim + i];
}

__global__ void lgpu_kern_argmax(int *result, const half *logits, int n) {
    extern __shared__ char argmax_smem[];
    float *vals = (float*)argmax_smem;
    int   *idxs = (int*)(argmax_smem + blockDim.x * sizeof(float));
    int tid = threadIdx.x;
    float best_val = -1e30f; int best_idx = 0;
    for (int i = tid; i < n; i += blockDim.x) {
        float v = __half2float(logits[i]);
        if (v > best_val) { best_val = v; best_idx = i; }
    }
    vals[tid] = best_val; idxs[tid] = best_idx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && vals[tid + s] > vals[tid])
            { vals[tid] = vals[tid + s]; idxs[tid] = idxs[tid + s]; }
        __syncthreads();
    }
    if (tid == 0) *result = idxs[0];
}

/* ================================================================
 * cuBLAS wrapper: y[M] = W[M,K] * x[K]  (W row-major)
 * ================================================================ */

static void lgpu_matmul(cublasHandle_t h, half *y, const half *W, const half *x,
                         int M, int K) {
    __half a = __float2half(1.0f), b = __float2half(0.0f);
    CUBLAS_CHECK(cublasHgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, M, 1, K,
                             &a, W, K, x, K, &b, y, M));
}

/* ================================================================
 * Weight and scratch types
 * ================================================================ */

typedef struct {
    half *input_norm, *q_proj, *k_proj, *v_proj, *o_proj;
    half *post_attn_norm, *gate_proj, *up_proj, *down_proj;
} LGpuLayerWeights;

typedef struct {
    half *embed_tokens;     /* [VOCAB * DIM] */
    half *final_norm;       /* [DIM] */
    half *lm_head;          /* [VOCAB * DIM] */
    LGpuLayerWeights layers[LGPU_N_LAYERS];
} LGpuModelWeights;

typedef struct {
    half *normed, *q, *k, *v, *attn_out, *o_out;
    half *normed2, *gate, *up, *silu_up, *mlp_out;
    half *logits;           /* [VOCAB_SIZE] */
    half *kv_cache_k, *kv_cache_v;  /* [N_LAYERS * MAX_SEQ * KV_DIM] */
    float *rope_cos, *rope_sin;
    int *d_argmax_result;
} LGpuScratch;

/* ================================================================
 * Runtime context — owns all GPU state
 * ================================================================ */

typedef struct {
    LGpuModelWeights weights;
    LGpuScratch      scratch;
    cublasHandle_t    cublas;
    half             *d_hidden;
    LGpuSafeTensors   st;      /* keeps mmap alive during load */
    int               cublas_ok; /* 1 if cublas handle was created */
    int               loaded;    /* 1 if full init succeeded */
} LlamaGPU;

/* ================================================================
 * Internal helpers
 * ================================================================ */

static void lgpu_alloc_scratch(LGpuScratch *s) {
    CUDA_CHECK(cudaMalloc(&s->normed,   LGPU_DIM * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&s->q,        LGPU_DIM * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&s->k,        LGPU_KV_DIM * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&s->v,        LGPU_KV_DIM * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&s->attn_out, LGPU_DIM * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&s->o_out,    LGPU_DIM * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&s->normed2,  LGPU_DIM * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&s->gate,     LGPU_INTER_DIM * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&s->up,       LGPU_INTER_DIM * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&s->silu_up,  LGPU_INTER_DIM * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&s->mlp_out,  LGPU_DIM * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&s->logits,   LGPU_VOCAB_SIZE * sizeof(half)));
    size_t kv_bytes = (int64_t)LGPU_N_LAYERS * LGPU_MAX_SEQ * LGPU_KV_DIM * sizeof(half);
    CUDA_CHECK(cudaMalloc(&s->kv_cache_k, kv_bytes));
    CUDA_CHECK(cudaMalloc(&s->kv_cache_v, kv_bytes));
    CUDA_CHECK(cudaMalloc(&s->rope_cos, LGPU_MAX_SEQ * (LGPU_HEAD_DIM/2) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s->rope_sin, LGPU_MAX_SEQ * (LGPU_HEAD_DIM/2) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s->d_argmax_result, sizeof(int)));
}

static void lgpu_free_scratch(LGpuScratch *s) {
    cudaFree(s->normed); cudaFree(s->q); cudaFree(s->k); cudaFree(s->v);
    cudaFree(s->attn_out); cudaFree(s->o_out); cudaFree(s->normed2);
    cudaFree(s->gate); cudaFree(s->up); cudaFree(s->silu_up);
    cudaFree(s->mlp_out); cudaFree(s->logits);
    cudaFree(s->kv_cache_k); cudaFree(s->kv_cache_v);
    cudaFree(s->rope_cos); cudaFree(s->rope_sin);
    cudaFree(s->d_argmax_result);
    memset(s, 0, sizeof(*s));
}

static void lgpu_init_rope(LGpuScratch *s) {
    int hd = LGPU_HEAD_DIM / 2;
    float *hc = (float*)malloc(LGPU_MAX_SEQ * hd * sizeof(float));
    float *hs = (float*)malloc(LGPU_MAX_SEQ * hd * sizeof(float));
    for (int i = 0; i < hd; i++) {
        float theta = powf(LGPU_ROPE_THETA, -2.0f * i / LGPU_HEAD_DIM);
        for (int p = 0; p < LGPU_MAX_SEQ; p++) {
            float a = p * theta;
            hc[p * hd + i] = cosf(a);
            hs[p * hd + i] = sinf(a);
        }
    }
    CUDA_CHECK(cudaMemcpy(s->rope_cos, hc, LGPU_MAX_SEQ * hd * sizeof(float),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(s->rope_sin, hs, LGPU_MAX_SEQ * hd * sizeof(float),
                           cudaMemcpyHostToDevice));
    free(hc); free(hs);
}

/* ================================================================
 * Layer forward
 * ================================================================ */

static void lgpu_layer_forward(cublasHandle_t h, half *d_hidden,
                                LGpuLayerWeights *w, LGpuScratch *s,
                                int layer_idx, int pos) {
    int seq_len = pos + 1, hd = LGPU_HEAD_DIM / 2, b = 256;
    int64_t kv_off = (int64_t)layer_idx * LGPU_MAX_SEQ * LGPU_KV_DIM;

    lgpu_kern_rmsnorm<<<1, LGPU_NORM_THREADS, LGPU_NORM_THREADS * sizeof(float)>>>(
        s->normed, d_hidden, w->input_norm, LGPU_DIM);

    lgpu_matmul(h, s->q, w->q_proj, s->normed, LGPU_DIM, LGPU_DIM);
    lgpu_matmul(h, s->k, w->k_proj, s->normed, LGPU_KV_DIM, LGPU_DIM);
    lgpu_matmul(h, s->v, w->v_proj, s->normed, LGPU_KV_DIM, LGPU_DIM);

    lgpu_kern_rope<<<(LGPU_N_HEADS * hd + b-1)/b, b>>>(
        s->q, s->rope_cos, s->rope_sin, pos, LGPU_N_HEADS, hd);
    lgpu_kern_rope<<<(LGPU_N_KV_HEADS * hd + b-1)/b, b>>>(
        s->k, s->rope_cos, s->rope_sin, pos, LGPU_N_KV_HEADS, hd);

    CUDA_CHECK(cudaMemcpy(s->kv_cache_k + kv_off + pos * LGPU_KV_DIM,
                           s->k, LGPU_KV_DIM * sizeof(half), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(s->kv_cache_v + kv_off + pos * LGPU_KV_DIM,
                           s->v, LGPU_KV_DIM * sizeof(half), cudaMemcpyDeviceToDevice));

    lgpu_kern_attention<<<LGPU_N_HEADS, LGPU_HEAD_DIM,
                          (seq_len + LGPU_HEAD_DIM) * sizeof(float)>>>(
        s->attn_out, s->q,
        s->kv_cache_k + kv_off, s->kv_cache_v + kv_off,
        seq_len, LGPU_N_HEADS, LGPU_N_KV_HEADS, LGPU_HEAD_DIM, LGPU_GQA_GROUP);

    lgpu_matmul(h, s->o_out, w->o_proj, s->attn_out, LGPU_DIM, LGPU_DIM);
    lgpu_kern_residual_add<<<(LGPU_DIM+b-1)/b, b>>>(d_hidden, s->o_out, LGPU_DIM);

    lgpu_kern_rmsnorm<<<1, LGPU_NORM_THREADS, LGPU_NORM_THREADS * sizeof(float)>>>(
        s->normed2, d_hidden, w->post_attn_norm, LGPU_DIM);

    lgpu_matmul(h, s->gate, w->gate_proj, s->normed2, LGPU_INTER_DIM, LGPU_DIM);
    lgpu_matmul(h, s->up,   w->up_proj,   s->normed2, LGPU_INTER_DIM, LGPU_DIM);
    lgpu_kern_silu_mul<<<(LGPU_INTER_DIM+b-1)/b, b>>>(
        s->silu_up, s->gate, s->up, LGPU_INTER_DIM);
    lgpu_matmul(h, s->mlp_out, w->down_proj, s->silu_up, LGPU_DIM, LGPU_INTER_DIM);
    lgpu_kern_residual_add<<<(LGPU_DIM+b-1)/b, b>>>(d_hidden, s->mlp_out, LGPU_DIM);
}

/* ================================================================
 * Full forward: embed -> N layers -> final norm -> lm_head -> argmax
 * Returns next token ID.
 * ================================================================ */

static int lgpu_forward_token(LlamaGPU *ctx, int token, int pos) {
    LGpuScratch *s = &ctx->scratch;

    lgpu_kern_embed_lookup<<<(LGPU_DIM+255)/256, 256>>>(
        ctx->d_hidden, ctx->weights.embed_tokens, token, LGPU_DIM);

    for (int L = 0; L < LGPU_N_LAYERS; L++)
        lgpu_layer_forward(ctx->cublas, ctx->d_hidden,
                           &ctx->weights.layers[L], s, L, pos);

    lgpu_kern_rmsnorm<<<1, LGPU_NORM_THREADS, LGPU_NORM_THREADS * sizeof(float)>>>(
        s->normed, ctx->d_hidden, ctx->weights.final_norm, LGPU_DIM);

    lgpu_matmul(ctx->cublas, s->logits, ctx->weights.lm_head, s->normed,
                LGPU_VOCAB_SIZE, LGPU_DIM);

    int smem = LGPU_NORM_THREADS * (sizeof(float) + sizeof(int));
    lgpu_kern_argmax<<<1, LGPU_NORM_THREADS, smem>>>(
        s->d_argmax_result, s->logits, LGPU_VOCAB_SIZE);

    int result;
    CUDA_CHECK(cudaMemcpy(&result, s->d_argmax_result, sizeof(int),
                           cudaMemcpyDeviceToHost));
    return result;
}

/* ================================================================
 * Public API
 * ================================================================ */

/* Forward declaration (init calls free on failure). */
static void llama_gpu_free(LlamaGPU *ctx);

/* Load model from a HuggingFace-style directory containing model.safetensors.
 * Returns 0 on success.  On failure, cleans up any partial state so the
 * caller does not need to call llama_gpu_free(). */
static int llama_gpu_init(LlamaGPU *ctx, const char *model_dir) {
    memset(ctx, 0, sizeof(*ctx));

    char path[512];
    snprintf(path, sizeof(path), "%s/model.safetensors", model_dir);
    if (lgpu_st_open(&ctx->st, path) != 0) return -1;

    int64_t cnt;
    /* On tensor-load failure: close mmap, free any partial GPU allocs, return. */
    #define LD(dest, name, expected) do { \
        const uint16_t *raw = lgpu_st_find_bf16(&ctx->st, name, &cnt); \
        if (!raw || cnt != (int64_t)(expected)) { \
            fprintf(stderr, "FATAL: %s: count=%ld expected=%ld\n", \
                    name, raw ? (long)cnt : -1L, (long)(expected)); \
            lgpu_st_close(&ctx->st); llama_gpu_free(ctx); return -1; } \
        CUDA_CHECK(cudaMalloc(&dest, cnt * sizeof(half))); \
        lgpu_convert_bf16(dest, raw, cnt); \
    } while(0)

    LD(ctx->weights.embed_tokens, "model.embed_tokens.weight",
       (int64_t)LGPU_VOCAB_SIZE * LGPU_DIM);
    LD(ctx->weights.final_norm,   "model.norm.weight", LGPU_DIM);
    LD(ctx->weights.lm_head,      "lm_head.weight",
       (int64_t)LGPU_VOCAB_SIZE * LGPU_DIM);

    for (int L = 0; L < LGPU_N_LAYERS; L++) {
        char tn[128];
        #define LL(field, suffix, sz) do { \
            snprintf(tn, sizeof(tn), "model.layers.%d.%s", L, suffix); \
            LD(ctx->weights.layers[L].field, tn, sz); \
        } while(0)
        LL(input_norm,    "input_layernorm.weight",          LGPU_DIM);
        LL(q_proj,        "self_attn.q_proj.weight",         (int64_t)LGPU_DIM * LGPU_DIM);
        LL(k_proj,        "self_attn.k_proj.weight",         (int64_t)LGPU_KV_DIM * LGPU_DIM);
        LL(v_proj,        "self_attn.v_proj.weight",         (int64_t)LGPU_KV_DIM * LGPU_DIM);
        LL(o_proj,        "self_attn.o_proj.weight",         (int64_t)LGPU_DIM * LGPU_DIM);
        LL(post_attn_norm,"post_attention_layernorm.weight",  LGPU_DIM);
        LL(gate_proj,     "mlp.gate_proj.weight",            (int64_t)LGPU_INTER_DIM * LGPU_DIM);
        LL(up_proj,       "mlp.up_proj.weight",              (int64_t)LGPU_INTER_DIM * LGPU_DIM);
        LL(down_proj,     "mlp.down_proj.weight",            (int64_t)LGPU_DIM * LGPU_INTER_DIM);
        #undef LL
    }
    #undef LD

    lgpu_st_close(&ctx->st);  /* weights are now on GPU; mmap no longer needed */

    CUBLAS_CHECK(cublasCreate(&ctx->cublas));
    CUBLAS_CHECK(cublasSetMathMode(ctx->cublas, CUBLAS_TENSOR_OP_MATH));
    ctx->cublas_ok = 1;

    lgpu_alloc_scratch(&ctx->scratch);
    lgpu_init_rope(&ctx->scratch);
    CUDA_CHECK(cudaMalloc(&ctx->d_hidden, LGPU_DIM * sizeof(half)));

    ctx->loaded = 1;
    return 0;
}

/* Reset KV cache (call before a new prompt). */
static void llama_gpu_reset(LlamaGPU *ctx) {
    size_t kv_bytes = (int64_t)LGPU_N_LAYERS * LGPU_MAX_SEQ * LGPU_KV_DIM * sizeof(half);
    CUDA_CHECK(cudaMemset(ctx->scratch.kv_cache_k, 0, kv_bytes));
    CUDA_CHECK(cudaMemset(ctx->scratch.kv_cache_v, 0, kv_bytes));
}

/* Run prefill + greedy decode.
 * prompt: array of prompt_len token IDs (must be >= 1).
 * output: caller-allocated array for max_new_tokens generated IDs.
 * Returns number of tokens actually generated (0 on invalid input). */
static int llama_gpu_generate(LlamaGPU *ctx,
                               const int *prompt, int prompt_len,
                               int *output, int max_new_tokens) {
    if (prompt_len <= 0 || max_new_tokens <= 0 || !prompt || !output)
        return 0;

    llama_gpu_reset(ctx);

    /* Prefill: process all prompt tokens */
    int next_token = 0;
    for (int p = 0; p < prompt_len; p++)
        next_token = lgpu_forward_token(ctx, prompt[p], p);

    output[0] = next_token;

    /* Decode: generate remaining tokens */
    int generated = 1;
    for (int g = 1; g < max_new_tokens; g++) {
        int pos = prompt_len + g - 1;
        if (pos >= LGPU_MAX_SEQ) break;
        output[g] = lgpu_forward_token(ctx, output[g - 1], pos);
        generated = g + 1;
    }

    return generated;
}

/* Release all resources.  Safe to call on partially-initialized or
 * zeroed state (cudaFree(NULL) is a no-op). */
static void llama_gpu_free(LlamaGPU *ctx) {
    lgpu_free_scratch(&ctx->scratch);
    cudaFree(ctx->d_hidden);       ctx->d_hidden = NULL;
    cudaFree(ctx->weights.embed_tokens); ctx->weights.embed_tokens = NULL;
    cudaFree(ctx->weights.final_norm);   ctx->weights.final_norm = NULL;
    cudaFree(ctx->weights.lm_head);      ctx->weights.lm_head = NULL;
    for (int L = 0; L < LGPU_N_LAYERS; L++) {
        LGpuLayerWeights *lw = &ctx->weights.layers[L];
        cudaFree(lw->input_norm); cudaFree(lw->q_proj); cudaFree(lw->k_proj);
        cudaFree(lw->v_proj); cudaFree(lw->o_proj); cudaFree(lw->post_attn_norm);
        cudaFree(lw->gate_proj); cudaFree(lw->up_proj); cudaFree(lw->down_proj);
        memset(lw, 0, sizeof(*lw));
    }
    if (ctx->cublas_ok) { cublasDestroy(ctx->cublas); ctx->cublas_ok = 0; }
    lgpu_st_close(&ctx->st);
    ctx->loaded = 0;
}

#endif /* LLAMA_GPU_CUH */
