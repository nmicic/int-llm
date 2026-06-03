/*
 * bench_int8_compare.cu — INT8 vs FP16 comparison on TinyLlama
 *
 * Runs both FP16 (accepted baseline) and experimental INT8 matmul paths
 * on the same prompts, compares token output against CPU oracle.
 *
 * INT8 scheme: per-tensor absmax symmetric quantization.
 *   - Weights quantized once at init: scale_w = max(|W|)/127, W_i8 = round(W/scale_w)
 *   - Activations quantized per-call: scale_x = max(|x|)/127, x_i8 = round(x/scale_x)
 *   - Matmul via cuBLASLt: y_i32 = W_i8 * x_i8 (INT32 accumulation)
 *   - Dequantize: y_fp16 = y_i32 * (scale_w * scale_x)
 *   - Non-matmul ops (RMSNorm, RoPE, attention, SiLU, residual) stay FP16/FP32.
 *
 * Build:
 *   NVCC=/usr/local/cuda-13.0/bin/nvcc
 *   $NVCC -O3 -arch=sm_120 bench_int8_compare.cu -o bench_int8_compare -lcublas -lcublasLt -lm
 *
 * Run:
 *   ./bench_int8_compare /path/to/TinyLlama-model-dir
 *
 * Copyright (c) 2026 — MicroGPT-C R11 GPU INT8 comparison
 */

#include "llama_gpu.cuh"
#include <cublasLt.h>

/* ================================================================
 * INT8 quantization kernels
 * ================================================================ */

/* Compute absmax of FP16 data, store scale = absmax/127 on device */
__global__ void kern_absmax_scale(float *scale_out, const half *data, int n) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    float local_max = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float v = fabsf(__half2float(data[i]));
        if (v > local_max) local_max = v;
    }
    smem[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && smem[tid + s] > smem[tid]) smem[tid] = smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        float mx = smem[0];
        *scale_out = (mx > 0.0f) ? mx / 127.0f : 1.0f;
    }
}

/* Quantize FP16 → INT8 with given scale (on device) */
__global__ void kern_quantize(int8_t *out, const half *in, const float *scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float s = *scale;
    float v = __half2float(in[i]) / s;
    v = fminf(fmaxf(v, -127.0f), 127.0f);
    out[i] = (int8_t)rintf(v);
}

/* Dequantize INT32 → FP16: out = in * combined_scale */
__global__ void kern_dequant(half *out, const int32_t *in, float combined_scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = __float2half((float)in[i] * combined_scale);
}

/* ================================================================
 * INT8 weight + per-tensor quantized matmul
 * ================================================================ */

typedef struct {
    int8_t *data;     /* quantized weight [M*K] */
    float   h_scale;  /* host-side scale for dequant */
    int     M, K;     /* shape */
} I8Weight;

/* Quantize one FP16 weight tensor to INT8 on GPU */
static void i8_quantize_weight(I8Weight *iw, const half *d_fp16, int M, int K) {
    int64_t n = (int64_t)M * K;
    iw->M = M; iw->K = K;
    CUDA_CHECK(cudaMalloc(&iw->data, n));

    /* Compute scale on device, copy to host */
    float *d_scale;
    CUDA_CHECK(cudaMalloc(&d_scale, sizeof(float)));
    kern_absmax_scale<<<1, 1024, 1024 * sizeof(float)>>>(d_scale, d_fp16, (int)n);
    CUDA_CHECK(cudaMemcpy(&iw->h_scale, d_scale, sizeof(float), cudaMemcpyDeviceToHost));

    /* Quantize */
    kern_quantize<<<((int)n + 255) / 256, 256>>>(iw->data, d_fp16, d_scale, (int)n);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_scale));
}

/* INT8 matmul: y_fp16[M] = W_i8[M,K] * x_fp16[K]
 * Quantizes x on the fly, uses cuBLASLt, dequantizes result. */
static void i8_matmul(cublasLtHandle_t lt, half *d_y, I8Weight *w, const half *d_x,
                       int8_t *d_x_i8, int32_t *d_y_i32, float *d_scale_x,
                       void *workspace, size_t ws_size) {
    int M = w->M, K = w->K, b = 256;

    /* 1. Quantize activation x → INT8 */
    kern_absmax_scale<<<1, 1024, 1024 * sizeof(float)>>>(d_scale_x, d_x, K);
    kern_quantize<<<(K + b-1)/b, b>>>(d_x_i8, d_x, d_scale_x, K);

    /* 2. cuBLASLt INT8 matmul: y_i32[M] = W_i8[M,K] * x_i8[K] */
    cublasLtMatmulDesc_t matmulDesc;
    cublasLtMatrixLayout_t layoutA, layoutB, layoutC;

    CUBLAS_CHECK(cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32I, CUDA_R_32I));

    cublasOperation_t opT = CUBLAS_OP_T, opN = CUBLAS_OP_N;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc,
        CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(matmulDesc,
        CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));

    /* W row-major [M,K] = col-major K×M. With TRANSA=T → M×K. */
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&layoutA, CUDA_R_8I, K, M, K));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&layoutB, CUDA_R_8I, K, 1, K));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&layoutC, CUDA_R_32I, M, 1, M));

    int32_t alpha = 1, beta = 0;

    cublasLtMatmulPreference_t pref;
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(pref,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_size, sizeof(ws_size)));

    cublasLtMatmulHeuristicResult_t heur;
    int nResult = 0;
    CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(lt, matmulDesc,
        layoutA, layoutB, layoutC, layoutC, pref, 1, &heur, &nResult));

    if (nResult > 0) {
        CUBLAS_CHECK(cublasLtMatmul(lt, matmulDesc, &alpha,
            w->data, layoutA, d_x_i8, layoutB,
            &beta, d_y_i32, layoutC, d_y_i32, layoutC,
            &heur.algo, workspace, heur.workspaceSize, 0));
    }

    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(layoutA);
    cublasLtMatrixLayoutDestroy(layoutB);
    cublasLtMatrixLayoutDestroy(layoutC);
    cublasLtMatmulDescDestroy(matmulDesc);

    /* 3. Get activation scale from device */
    float h_scale_x;
    CUDA_CHECK(cudaMemcpy(&h_scale_x, d_scale_x, sizeof(float), cudaMemcpyDeviceToHost));

    /* 4. Dequantize: y_fp16 = y_i32 * (scale_w * scale_x) */
    float combined = w->h_scale * h_scale_x;
    kern_dequant<<<(M + b-1)/b, b>>>(d_y, d_y_i32, combined, M);
}

/* ================================================================
 * INT8 layer + model structures
 * ================================================================ */

typedef struct {
    I8Weight q_proj, k_proj, v_proj, o_proj;
    I8Weight gate_proj, up_proj, down_proj;
    /* norms stay FP16 — shared from the FP16 model */
} I8LayerWeights;

typedef struct {
    I8LayerWeights layers[LGPU_N_LAYERS];
    I8Weight lm_head;
} I8ModelWeights;

/* Scratch for INT8 path (shared with FP16 scratch for non-matmul ops) */
typedef struct {
    int8_t  *x_i8;        /* activation quantization buffer [max(K)] */
    int32_t *y_i32;       /* matmul output buffer [max(M)] */
    float   *d_scale_x;   /* device scalar for activation scale */
    void    *workspace;    /* cuBLASLt workspace */
    size_t   ws_size;
} I8Scratch;

static void i8_alloc_scratch(I8Scratch *s) {
    int max_k = LGPU_INTER_DIM;  /* largest input dim (down_proj) */
    int max_m = LGPU_VOCAB_SIZE; /* largest output dim (lm_head) */
    s->ws_size = 4 * 1024 * 1024;
    CUDA_CHECK(cudaMalloc(&s->x_i8, max_k));
    CUDA_CHECK(cudaMalloc(&s->y_i32, max_m * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&s->d_scale_x, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&s->workspace, s->ws_size));
}

static void i8_free_scratch(I8Scratch *s) {
    cudaFree(s->x_i8); cudaFree(s->y_i32);
    cudaFree(s->d_scale_x); cudaFree(s->workspace);
    memset(s, 0, sizeof(*s));
}

/* Quantize all weight matrices from the loaded FP16 model */
static void i8_quantize_model(LGpuModelWeights *fp16, I8ModelWeights *i8) {
    for (int L = 0; L < LGPU_N_LAYERS; L++) {
        LGpuLayerWeights *f = &fp16->layers[L];
        I8LayerWeights *q = &i8->layers[L];
        i8_quantize_weight(&q->q_proj,    f->q_proj,    LGPU_DIM,       LGPU_DIM);
        i8_quantize_weight(&q->k_proj,    f->k_proj,    LGPU_KV_DIM,    LGPU_DIM);
        i8_quantize_weight(&q->v_proj,    f->v_proj,    LGPU_KV_DIM,    LGPU_DIM);
        i8_quantize_weight(&q->o_proj,    f->o_proj,    LGPU_DIM,       LGPU_DIM);
        i8_quantize_weight(&q->gate_proj, f->gate_proj, LGPU_INTER_DIM, LGPU_DIM);
        i8_quantize_weight(&q->up_proj,   f->up_proj,   LGPU_INTER_DIM, LGPU_DIM);
        i8_quantize_weight(&q->down_proj, f->down_proj, LGPU_DIM,       LGPU_INTER_DIM);
    }
    i8_quantize_weight(&i8->lm_head, fp16->lm_head, LGPU_VOCAB_SIZE, LGPU_DIM);
}

static void i8_free_model(I8ModelWeights *m) {
    for (int L = 0; L < LGPU_N_LAYERS; L++) {
        I8LayerWeights *q = &m->layers[L];
        cudaFree(q->q_proj.data); cudaFree(q->k_proj.data);
        cudaFree(q->v_proj.data); cudaFree(q->o_proj.data);
        cudaFree(q->gate_proj.data); cudaFree(q->up_proj.data);
        cudaFree(q->down_proj.data);
    }
    cudaFree(m->lm_head.data);
}

/* ================================================================
 * INT8 layer forward (non-matmul ops reuse FP16 kernels from llama_gpu.cuh)
 * ================================================================ */

static void i8_layer_forward(cublasLtHandle_t lt, half *d_hidden,
                              LGpuLayerWeights *fw,  /* FP16 norms */
                              I8LayerWeights *iw,     /* INT8 projections */
                              LGpuScratch *s, I8Scratch *is,
                              int layer_idx, int pos) {
    int seq_len = pos + 1, hd = LGPU_HEAD_DIM / 2, b = 256;
    int64_t kv_off = (int64_t)layer_idx * LGPU_MAX_SEQ * LGPU_KV_DIM;

    /* RMSNorm (FP16) */
    lgpu_kern_rmsnorm<<<1, LGPU_NORM_THREADS, LGPU_NORM_THREADS * sizeof(float)>>>(
        s->normed, d_hidden, fw->input_norm, LGPU_DIM);

    /* Q/K/V projections (INT8) */
    i8_matmul(lt, s->q, &iw->q_proj, s->normed, is->x_i8, is->y_i32,
              is->d_scale_x, is->workspace, is->ws_size);
    i8_matmul(lt, s->k, &iw->k_proj, s->normed, is->x_i8, is->y_i32,
              is->d_scale_x, is->workspace, is->ws_size);
    i8_matmul(lt, s->v, &iw->v_proj, s->normed, is->x_i8, is->y_i32,
              is->d_scale_x, is->workspace, is->ws_size);

    /* RoPE (FP16) */
    lgpu_kern_rope<<<(LGPU_N_HEADS * hd + b-1)/b, b>>>(
        s->q, s->rope_cos, s->rope_sin, pos, LGPU_N_HEADS, hd);
    lgpu_kern_rope<<<(LGPU_N_KV_HEADS * hd + b-1)/b, b>>>(
        s->k, s->rope_cos, s->rope_sin, pos, LGPU_N_KV_HEADS, hd);

    /* KV cache store */
    CUDA_CHECK(cudaMemcpy(s->kv_cache_k + kv_off + pos * LGPU_KV_DIM,
                           s->k, LGPU_KV_DIM * sizeof(half), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(s->kv_cache_v + kv_off + pos * LGPU_KV_DIM,
                           s->v, LGPU_KV_DIM * sizeof(half), cudaMemcpyDeviceToDevice));

    /* Attention (FP16/FP32) */
    lgpu_kern_attention<<<LGPU_N_HEADS, LGPU_HEAD_DIM,
                          (seq_len + LGPU_HEAD_DIM) * sizeof(float)>>>(
        s->attn_out, s->q,
        s->kv_cache_k + kv_off, s->kv_cache_v + kv_off,
        seq_len, LGPU_N_HEADS, LGPU_N_KV_HEADS, LGPU_HEAD_DIM, LGPU_GQA_GROUP);

    /* O projection (INT8) */
    i8_matmul(lt, s->o_out, &iw->o_proj, s->attn_out, is->x_i8, is->y_i32,
              is->d_scale_x, is->workspace, is->ws_size);

    /* Residual (FP16) */
    lgpu_kern_residual_add<<<(LGPU_DIM+b-1)/b, b>>>(d_hidden, s->o_out, LGPU_DIM);

    /* Post-attn RMSNorm (FP16) */
    lgpu_kern_rmsnorm<<<1, LGPU_NORM_THREADS, LGPU_NORM_THREADS * sizeof(float)>>>(
        s->normed2, d_hidden, fw->post_attn_norm, LGPU_DIM);

    /* MLP gate/up/down (INT8) */
    i8_matmul(lt, s->gate, &iw->gate_proj, s->normed2, is->x_i8, is->y_i32,
              is->d_scale_x, is->workspace, is->ws_size);
    i8_matmul(lt, s->up, &iw->up_proj, s->normed2, is->x_i8, is->y_i32,
              is->d_scale_x, is->workspace, is->ws_size);

    /* SiLU * up (FP16) */
    lgpu_kern_silu_mul<<<(LGPU_INTER_DIM+b-1)/b, b>>>(
        s->silu_up, s->gate, s->up, LGPU_INTER_DIM);

    /* Down projection (INT8) */
    i8_matmul(lt, s->mlp_out, &iw->down_proj, s->silu_up, is->x_i8, is->y_i32,
              is->d_scale_x, is->workspace, is->ws_size);

    /* Residual (FP16) */
    lgpu_kern_residual_add<<<(LGPU_DIM+b-1)/b, b>>>(d_hidden, s->mlp_out, LGPU_DIM);
}

/* INT8 full forward + argmax */
static int i8_forward_token(LlamaGPU *ctx, cublasLtHandle_t lt,
                             I8ModelWeights *i8m, I8Scratch *is,
                             int token, int pos) {
    LGpuScratch *s = &ctx->scratch;

    lgpu_kern_embed_lookup<<<(LGPU_DIM+255)/256, 256>>>(
        ctx->d_hidden, ctx->weights.embed_tokens, token, LGPU_DIM);

    for (int L = 0; L < LGPU_N_LAYERS; L++)
        i8_layer_forward(lt, ctx->d_hidden, &ctx->weights.layers[L],
                         &i8m->layers[L], s, is, L, pos);

    lgpu_kern_rmsnorm<<<1, LGPU_NORM_THREADS, LGPU_NORM_THREADS * sizeof(float)>>>(
        s->normed, ctx->d_hidden, ctx->weights.final_norm, LGPU_DIM);

    /* lm_head (INT8) */
    i8_matmul(lt, s->logits, &i8m->lm_head, s->normed, is->x_i8, is->y_i32,
              is->d_scale_x, is->workspace, is->ws_size);

    int smem = LGPU_NORM_THREADS * (sizeof(float) + sizeof(int));
    lgpu_kern_argmax<<<1, LGPU_NORM_THREADS, smem>>>(
        s->d_argmax_result, s->logits, LGPU_VOCAB_SIZE);

    int result;
    CUDA_CHECK(cudaMemcpy(&result, s->d_argmax_result, sizeof(int),
                           cudaMemcpyDeviceToHost));
    return result;
}

/* ================================================================
 * Reference prompts (same as bench_full_fp16.cu)
 * ================================================================ */

#define GEN_TOKENS 20
typedef struct { const char *name; const int *prompt; int len; const int *ref; } Prompt;

static const int P_FR[] = {1,450,7483,310,3444,338};
static const int R_FR[] = {3681,29889,13,13,29906,29889,350,29889,450,7483,
                           310,9556,338,5115,29889,13,13,29941,29889,315};
static const int P_ST[] = {1,9038,2501,263,931};
static const int R_ST[] = {29892,727,471,263,4123,6114,4257,365,2354,29889,
                           2296,10600,297,263,2319,4726,29892,988,14332,6363};
static const int P_MA[] = {1,29896,29974,29896,29922};
static const int R_MA[] = {29896,13,13,6295,278,2533,310,278,937,302,
                           4958,310,278,383,747,265,21566,5665,338,29871};
static const int P_LI[] = {1,450,6593,310,2834,338};
static const int R_LI[] = {304,1284,596,15935,29892,12359,434,372,411,599,
                           596,1795,29892,322,5967,263,25000,29889,13,13};
static const Prompt PROMPTS[] = {
    {"france_capital",P_FR,6,R_FR}, {"story_beginning",P_ST,5,R_ST},
    {"simple_math",P_MA,5,R_MA}, {"meaning_of_life",P_LI,6,R_LI},
};
#define N_PROMPTS 4

/* ================================================================
 * Main
 * ================================================================ */

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s <model_dir>\n", argv[0]); return 1; }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (%d SMs, compute %d.%d)\n",
           prop.name, prop.multiProcessorCount, prop.major, prop.minor);

    /* Load FP16 model */
    LlamaGPU ctx;
    printf("Loading FP16 model...\n");
    if (llama_gpu_init(&ctx, argv[1]) != 0) return 1;

    /* Quantize to INT8 */
    printf("Quantizing weights to INT8 (per-tensor absmax symmetric)...\n");
    I8ModelWeights i8m;
    i8_quantize_model(&ctx.weights, &i8m);

    cublasLtHandle_t lt;
    CUBLAS_CHECK(cublasLtCreate(&lt));

    I8Scratch is;
    i8_alloc_scratch(&is);
    printf("Ready.\n");

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    /* ============================================================
     * Run both FP16 and INT8 on all 4 prompts
     * ============================================================ */

    printf("\n=== FP16 vs INT8 Comparison (4 prompts × 20 tokens) ===\n");
    printf("INT8: per-tensor absmax symmetric, matmuls only. Non-matmul ops in FP16/FP32.\n\n");

    int fp16_total = 0, i8_total = 0;
    float fp16_decode_ms = 0, i8_decode_ms = 0;
    int fp16_decode_steps = 0, i8_decode_steps = 0;

    for (int pi = 0; pi < N_PROMPTS; pi++) {
        const Prompt *p = &PROMPTS[pi];
        printf("--- %s ---\n", p->name);

        /* ---- FP16 ---- */
        llama_gpu_reset(&ctx);
        CUDA_CHECK(cudaEventRecord(t0, 0));
        int fp16_first = 0;
        for (int i = 0; i < p->len; i++)
            fp16_first = lgpu_forward_token(&ctx, p->prompt[i], i);
        CUDA_CHECK(cudaEventRecord(t1, 0));
        CUDA_CHECK(cudaEventSynchronize(t1));

        int fp16_out[GEN_TOKENS];
        fp16_out[0] = fp16_first;

        float fp16_pre;
        CUDA_CHECK(cudaEventElapsedTime(&fp16_pre, t0, t1));

        CUDA_CHECK(cudaEventRecord(t0, 0));
        for (int g = 1; g < GEN_TOKENS; g++) {
            int pos = p->len + g - 1;
            fp16_out[g] = lgpu_forward_token(&ctx, fp16_out[g-1], pos);
        }
        CUDA_CHECK(cudaEventRecord(t1, 0));
        CUDA_CHECK(cudaEventSynchronize(t1));
        float fp16_dec;
        CUDA_CHECK(cudaEventElapsedTime(&fp16_dec, t0, t1));

        int fp16_match = 0;
        for (int g = 0; g < GEN_TOKENS; g++)
            if (fp16_out[g] == p->ref[g]) fp16_match++;
        fp16_total += fp16_match;
        fp16_decode_ms += fp16_dec;
        fp16_decode_steps += GEN_TOKENS - 1;

        /* ---- INT8 ---- */
        llama_gpu_reset(&ctx);  /* reuse same KV cache / scratch */
        CUDA_CHECK(cudaEventRecord(t0, 0));
        int i8_first = 0;
        for (int i = 0; i < p->len; i++)
            i8_first = i8_forward_token(&ctx, lt, &i8m, &is, p->prompt[i], i);
        CUDA_CHECK(cudaEventRecord(t1, 0));
        CUDA_CHECK(cudaEventSynchronize(t1));

        int i8_out[GEN_TOKENS];
        i8_out[0] = i8_first;

        float i8_pre;
        CUDA_CHECK(cudaEventElapsedTime(&i8_pre, t0, t1));

        CUDA_CHECK(cudaEventRecord(t0, 0));
        for (int g = 1; g < GEN_TOKENS; g++) {
            int pos = p->len + g - 1;
            i8_out[g] = i8_forward_token(&ctx, lt, &i8m, &is, i8_out[g-1], pos);
        }
        CUDA_CHECK(cudaEventRecord(t1, 0));
        CUDA_CHECK(cudaEventSynchronize(t1));
        float i8_dec;
        CUDA_CHECK(cudaEventElapsedTime(&i8_dec, t0, t1));

        int i8_match = 0, i8_div = -1;
        for (int g = 0; g < GEN_TOKENS; g++) {
            if (i8_out[g] == p->ref[g]) i8_match++;
            else if (i8_div < 0) i8_div = g;
        }
        i8_total += i8_match;
        i8_decode_ms += i8_dec;
        i8_decode_steps += GEN_TOKENS - 1;

        /* Token-level comparison */
        int agree = 0;
        for (int g = 0; g < GEN_TOKENS; g++)
            if (fp16_out[g] == i8_out[g]) agree++;

        float fp16_tps = 1000.0f * (GEN_TOKENS-1) / fp16_dec;
        float i8_tps   = 1000.0f * (GEN_TOKENS-1) / i8_dec;

        printf("  FP16: %d/%d oracle  %.1f ms decode  %.0f tok/s\n",
               fp16_match, GEN_TOKENS, fp16_dec, fp16_tps);
        printf("  INT8: %d/%d oracle  %.1f ms decode  %.0f tok/s",
               i8_match, GEN_TOKENS, i8_dec, i8_tps);
        if (i8_div >= 0)
            printf("  first_div=%d (got %d, ref %d)", i8_div, i8_out[i8_div], p->ref[i8_div]);
        printf("\n  FP16↔INT8 agree: %d/%d  speedup: %.2fx\n\n",
               agree, GEN_TOKENS, fp16_dec / i8_dec);
    }

    /* Aggregate */
    float fp16_avg = fp16_decode_ms / fp16_decode_steps;
    float i8_avg   = i8_decode_ms / i8_decode_steps;

    printf("========================================\n");
    printf("FP16 oracle: %d/80  Decode: %.2f ms/tok (%.0f tok/s)\n",
           fp16_total, fp16_avg, 1000.0 / fp16_avg);
    printf("INT8 oracle: %d/80  Decode: %.2f ms/tok (%.0f tok/s)\n",
           i8_total, i8_avg, 1000.0 / i8_avg);
    printf("Speedup: %.2fx\n", fp16_avg / i8_avg);
    printf("========================================\n");

    /* Cleanup */
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    i8_free_scratch(&is);
    i8_free_model(&i8m);
    cublasLtDestroy(lt);
    llama_gpu_free(&ctx);
    printf("\nDone.\n");
    return 0;
}
