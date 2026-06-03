/*
 * bench_fp8_compare.cu — FP8 E4M3 vs FP16 comparison on TinyLlama
 *
 * Runs both FP16 (accepted baseline) and experimental FP8 matmul paths
 * on the same prompts, compares token output against CPU oracle.
 *
 * FP8 scheme: E4M3 weights (converted from FP16 at init), E4M3 activations
 * (converted per matmul), FP32 accumulation, FP16 output via cuBLASLt.
 * Non-matmul ops stay FP16/FP32.
 *
 * Build:
 *   NVCC=/usr/local/cuda-13.0/bin/nvcc
 *   $NVCC -O3 -arch=sm_120 bench_fp8_compare.cu -o bench_fp8_compare -lcublas -lcublasLt -lm
 *
 * Copyright (c) 2026 — MicroGPT-C R11 GPU FP8 comparison
 */

#include "llama_gpu.cuh"
#include <cuda_fp8.h>
#include <cublasLt.h>

/* ================================================================
 * FP16 -> E4M3 conversion kernel
 * ================================================================ */

__global__ void kern_fp16_to_e4m3(__nv_fp8_e4m3 *out, const half *in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = __nv_fp8_e4m3(__half2float(in[i]));
}

/* ================================================================
 * FP8 matmul plan — pre-cached per shape
 * ================================================================ */

typedef struct {
    cublasLtMatmulDesc_t     desc;
    cublasLtMatrixLayout_t   layoutA, layoutB, layoutC;
    cublasLtMatmulAlgo_t     algo;
    size_t                   ws_needed;
    int M, K;
} FP8Plan;

static int fp8_plan_create(FP8Plan *p, cublasLtHandle_t lt, int M, int K) {
    p->M = M; p->K = K;
    CUBLAS_CHECK(cublasLtMatmulDescCreate(&p->desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t opT = CUBLAS_OP_T, opN = CUBLAS_OP_N;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(p->desc,
        CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(p->desc,
        CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));

    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&p->layoutA, CUDA_R_8F_E4M3, K, M, K));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&p->layoutB, CUDA_R_8F_E4M3, K, 1, K));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&p->layoutC, CUDA_R_16F, M, 1, M));

    cublasLtMatmulPreference_t pref;
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    size_t ws = 4 * 1024 * 1024;
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(pref,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));

    cublasLtMatmulHeuristicResult_t heur;
    int nR = 0;
    cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(
        lt, p->desc, p->layoutA, p->layoutB, p->layoutC, p->layoutC,
        pref, 1, &heur, &nR);
    cublasLtMatmulPreferenceDestroy(pref);

    if (st != CUBLAS_STATUS_SUCCESS || nR == 0) return -1;
    p->algo = heur.algo;
    p->ws_needed = heur.workspaceSize;
    return 0;
}

static void fp8_plan_destroy(FP8Plan *p) {
    cublasLtMatrixLayoutDestroy(p->layoutA);
    cublasLtMatrixLayoutDestroy(p->layoutB);
    cublasLtMatrixLayoutDestroy(p->layoutC);
    cublasLtMatmulDescDestroy(p->desc);
}

/* ================================================================
 * FP8 weight + matmul execution
 * ================================================================ */

typedef struct {
    __nv_fp8_e4m3 *data;  /* [M*K] E4M3 weights */
    int M, K;
} FP8Weight;

static void fp8_convert_weight(FP8Weight *fw, const half *d_fp16, int M, int K) {
    fw->M = M; fw->K = K;
    CUDA_CHECK(cudaMalloc(&fw->data, (size_t)M * K));
    kern_fp16_to_e4m3<<<((int64_t)M*K + 255)/256, 256>>>(fw->data, d_fp16, M*K);
    CUDA_CHECK(cudaDeviceSynchronize());
}

/* FP8 matmul: y_fp16[M] = W_e4m3[M,K] * x_fp16[K]
 * Converts x to E4M3 on the fly, uses cached plan. */
static void fp8_matmul(cublasLtHandle_t lt, FP8Plan *plan,
                        half *d_y, FP8Weight *w, const half *d_x,
                        __nv_fp8_e4m3 *d_x8, void *workspace) {
    kern_fp16_to_e4m3<<<(w->K + 255)/256, 256>>>(d_x8, d_x, w->K);
    float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasLtMatmul(lt, plan->desc, &alpha,
        w->data, plan->layoutA, d_x8, plan->layoutB,
        &beta, d_y, plan->layoutC, d_y, plan->layoutC,
        &plan->algo, workspace, plan->ws_needed, 0));
}

/* ================================================================
 * FP8 model structures
 * ================================================================ */

typedef struct {
    FP8Weight q_proj, k_proj, v_proj, o_proj;
    FP8Weight gate_proj, up_proj, down_proj;
} FP8LayerWeights;

typedef struct {
    FP8LayerWeights layers[LGPU_N_LAYERS];
    FP8Weight lm_head;
    /* Plans for each distinct shape */
    FP8Plan plan_dim_dim;    /* 2048×2048 (Q,O) */
    FP8Plan plan_kv_dim;     /* 256×2048  (K,V) */
    FP8Plan plan_inter_dim;  /* 5632×2048 (gate,up) */
    FP8Plan plan_dim_inter;  /* 2048×5632 (down) */
    FP8Plan plan_vocab_dim;  /* 32000×2048 (lm_head) */
    __nv_fp8_e4m3 *d_x8;    /* activation conversion buffer */
    void *workspace;
} FP8Model;

static int fp8_model_init(FP8Model *m, LGpuModelWeights *fp16, cublasLtHandle_t lt) {
    /* Create plans */
    if (fp8_plan_create(&m->plan_dim_dim, lt, LGPU_DIM, LGPU_DIM) ||
        fp8_plan_create(&m->plan_kv_dim, lt, LGPU_KV_DIM, LGPU_DIM) ||
        fp8_plan_create(&m->plan_inter_dim, lt, LGPU_INTER_DIM, LGPU_DIM) ||
        fp8_plan_create(&m->plan_dim_inter, lt, LGPU_DIM, LGPU_INTER_DIM) ||
        fp8_plan_create(&m->plan_vocab_dim, lt, LGPU_VOCAB_SIZE, LGPU_DIM))
        return -1;

    /* Allocate scratch */
    CUDA_CHECK(cudaMalloc(&m->d_x8, LGPU_INTER_DIM));  /* largest K */
    size_t ws = 4 * 1024 * 1024;
    CUDA_CHECK(cudaMalloc(&m->workspace, ws));

    /* Convert weights */
    for (int L = 0; L < LGPU_N_LAYERS; L++) {
        LGpuLayerWeights *f = &fp16->layers[L];
        FP8LayerWeights *q = &m->layers[L];
        fp8_convert_weight(&q->q_proj, f->q_proj, LGPU_DIM, LGPU_DIM);
        fp8_convert_weight(&q->k_proj, f->k_proj, LGPU_KV_DIM, LGPU_DIM);
        fp8_convert_weight(&q->v_proj, f->v_proj, LGPU_KV_DIM, LGPU_DIM);
        fp8_convert_weight(&q->o_proj, f->o_proj, LGPU_DIM, LGPU_DIM);
        fp8_convert_weight(&q->gate_proj, f->gate_proj, LGPU_INTER_DIM, LGPU_DIM);
        fp8_convert_weight(&q->up_proj, f->up_proj, LGPU_INTER_DIM, LGPU_DIM);
        fp8_convert_weight(&q->down_proj, f->down_proj, LGPU_DIM, LGPU_INTER_DIM);
    }
    fp8_convert_weight(&m->lm_head, fp16->lm_head, LGPU_VOCAB_SIZE, LGPU_DIM);
    return 0;
}

static void fp8_model_free(FP8Model *m) {
    for (int L = 0; L < LGPU_N_LAYERS; L++) {
        FP8LayerWeights *q = &m->layers[L];
        cudaFree(q->q_proj.data); cudaFree(q->k_proj.data);
        cudaFree(q->v_proj.data); cudaFree(q->o_proj.data);
        cudaFree(q->gate_proj.data); cudaFree(q->up_proj.data);
        cudaFree(q->down_proj.data);
    }
    cudaFree(m->lm_head.data);
    cudaFree(m->d_x8); cudaFree(m->workspace);
    fp8_plan_destroy(&m->plan_dim_dim);
    fp8_plan_destroy(&m->plan_kv_dim);
    fp8_plan_destroy(&m->plan_inter_dim);
    fp8_plan_destroy(&m->plan_dim_inter);
    fp8_plan_destroy(&m->plan_vocab_dim);
}

/* ================================================================
 * FP8 layer forward (non-matmul ops reuse FP16 kernels)
 * ================================================================ */

static void fp8_layer_forward(cublasLtHandle_t lt, half *d_hidden,
                               LGpuLayerWeights *fw, FP8LayerWeights *f8w,
                               FP8Model *f8m, LGpuScratch *s,
                               int layer_idx, int pos) {
    int seq_len = pos + 1, hd = LGPU_HEAD_DIM / 2, b = 256;
    int64_t kv_off = (int64_t)layer_idx * LGPU_MAX_SEQ * LGPU_KV_DIM;

    lgpu_kern_rmsnorm<<<1, LGPU_NORM_THREADS, LGPU_NORM_THREADS * sizeof(float)>>>(
        s->normed, d_hidden, fw->input_norm, LGPU_DIM);

    fp8_matmul(lt, &f8m->plan_dim_dim, s->q, &f8w->q_proj, s->normed, f8m->d_x8, f8m->workspace);
    fp8_matmul(lt, &f8m->plan_kv_dim, s->k, &f8w->k_proj, s->normed, f8m->d_x8, f8m->workspace);
    fp8_matmul(lt, &f8m->plan_kv_dim, s->v, &f8w->v_proj, s->normed, f8m->d_x8, f8m->workspace);

    lgpu_kern_rope<<<(LGPU_N_HEADS * hd + b-1)/b, b>>>(
        s->q, s->rope_cos, s->rope_sin, pos, LGPU_N_HEADS, hd);
    lgpu_kern_rope<<<(LGPU_N_KV_HEADS * hd + b-1)/b, b>>>(
        s->k, s->rope_cos, s->rope_sin, pos, LGPU_N_KV_HEADS, hd);

    cudaMemcpyAsync(s->kv_cache_k + kv_off + pos * LGPU_KV_DIM,
                    s->k, LGPU_KV_DIM * sizeof(half), cudaMemcpyDeviceToDevice, 0);
    cudaMemcpyAsync(s->kv_cache_v + kv_off + pos * LGPU_KV_DIM,
                    s->v, LGPU_KV_DIM * sizeof(half), cudaMemcpyDeviceToDevice, 0);

    int attn_smem = (LGPU_MAX_SEQ + LGPU_HEAD_DIM) * sizeof(float);
    lgpu_kern_attention<<<LGPU_N_HEADS, LGPU_HEAD_DIM, attn_smem>>>(
        s->attn_out, s->q,
        s->kv_cache_k + kv_off, s->kv_cache_v + kv_off,
        seq_len, LGPU_N_HEADS, LGPU_N_KV_HEADS, LGPU_HEAD_DIM, LGPU_GQA_GROUP);

    fp8_matmul(lt, &f8m->plan_dim_dim, s->o_out, &f8w->o_proj, s->attn_out, f8m->d_x8, f8m->workspace);
    lgpu_kern_residual_add<<<(LGPU_DIM+b-1)/b, b>>>(d_hidden, s->o_out, LGPU_DIM);

    lgpu_kern_rmsnorm<<<1, LGPU_NORM_THREADS, LGPU_NORM_THREADS * sizeof(float)>>>(
        s->normed2, d_hidden, fw->post_attn_norm, LGPU_DIM);

    fp8_matmul(lt, &f8m->plan_inter_dim, s->gate, &f8w->gate_proj, s->normed2, f8m->d_x8, f8m->workspace);
    fp8_matmul(lt, &f8m->plan_inter_dim, s->up, &f8w->up_proj, s->normed2, f8m->d_x8, f8m->workspace);
    lgpu_kern_silu_mul<<<(LGPU_INTER_DIM+b-1)/b, b>>>(s->silu_up, s->gate, s->up, LGPU_INTER_DIM);
    fp8_matmul(lt, &f8m->plan_dim_inter, s->mlp_out, &f8w->down_proj, s->silu_up, f8m->d_x8, f8m->workspace);
    lgpu_kern_residual_add<<<(LGPU_DIM+b-1)/b, b>>>(d_hidden, s->mlp_out, LGPU_DIM);
}

static int fp8_forward_token(LlamaGPU *ctx, cublasLtHandle_t lt,
                              FP8Model *f8m, int token, int pos) {
    LGpuScratch *s = &ctx->scratch;
    lgpu_kern_embed_lookup<<<(LGPU_DIM+255)/256, 256>>>(
        ctx->d_hidden, ctx->weights.embed_tokens, token, LGPU_DIM);
    for (int L = 0; L < LGPU_N_LAYERS; L++)
        fp8_layer_forward(lt, ctx->d_hidden, &ctx->weights.layers[L],
                          &f8m->layers[L], f8m, s, L, pos);
    lgpu_kern_rmsnorm<<<1, LGPU_NORM_THREADS, LGPU_NORM_THREADS * sizeof(float)>>>(
        s->normed, ctx->d_hidden, ctx->weights.final_norm, LGPU_DIM);
    fp8_matmul(lt, &f8m->plan_vocab_dim, s->logits, &f8m->lm_head, s->normed,
               f8m->d_x8, f8m->workspace);
    int smem = LGPU_NORM_THREADS * (sizeof(float) + sizeof(int));
    lgpu_kern_argmax<<<1, LGPU_NORM_THREADS, smem>>>(
        s->d_argmax_result, s->logits, LGPU_VOCAB_SIZE);
    int result;
    CUDA_CHECK(cudaMemcpy(&result, s->d_argmax_result, sizeof(int), cudaMemcpyDeviceToHost));
    return result;
}

/* ================================================================
 * Reference tokens
 * ================================================================ */

#define GEN_TOKENS 20
typedef struct { const char *name; const int *p; int len; const int *r; } Prompt;
static const int P_FR[]={1,450,7483,310,3444,338};
static const int R_FR[]={3681,29889,13,13,29906,29889,350,29889,450,7483,310,9556,338,5115,29889,13,13,29941,29889,315};
static const int P_ST[]={1,9038,2501,263,931};
static const int R_ST[]={29892,727,471,263,4123,6114,4257,365,2354,29889,2296,10600,297,263,2319,4726,29892,988,14332,6363};
static const int P_MA[]={1,29896,29974,29896,29922};
static const int R_MA[]={29896,13,13,6295,278,2533,310,278,937,302,4958,310,278,383,747,265,21566,5665,338,29871};
static const int P_LI[]={1,450,6593,310,2834,338};
static const int R_LI[]={304,1284,596,15935,29892,12359,434,372,411,599,596,1795,29892,322,5967,263,25000,29889,13,13};
static const Prompt PROMPTS[]={
    {"france",P_FR,6,R_FR},{"story",P_ST,5,R_ST},
    {"math",P_MA,5,R_MA},{"life",P_LI,6,R_LI}};

/* ================================================================
 * Main
 * ================================================================ */

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s <model_dir>\n", argv[0]); return 1; }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (%d SMs, compute %d.%d)\n",
           prop.name, prop.multiProcessorCount, prop.major, prop.minor);

    LlamaGPU ctx;
    printf("Loading FP16 model...\n");
    if (llama_gpu_init(&ctx, argv[1]) != 0) return 1;

    cublasLtHandle_t lt;
    CUBLAS_CHECK(cublasLtCreate(&lt));

    printf("Converting weights to FP8 E4M3 (one-time, from FP16)...\n");
    FP8Model f8m;
    if (fp8_model_init(&f8m, &ctx.weights, lt) != 0) {
        fprintf(stderr, "FP8 plan creation failed\n"); return 1;
    }
    printf("Ready.\n");

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));

    printf("\n=== FP8 E4M3 vs FP16 Comparison (4 prompts x 20 tokens) ===\n");
    printf("FP8: E4M3 weights+activations, FP32 accumulation, FP16 output.\n");
    printf("Non-matmul ops: FP16/FP32. cuBLASLt with cached descriptors.\n\n");

    int fp16_total=0, fp8_total=0;
    float fp16_dec_ms=0, fp8_dec_ms=0;
    int dec_steps=0;

    for (int pi = 0; pi < 4; pi++) {
        const Prompt *p = &PROMPTS[pi];
        printf("--- %s ---\n", p->name);

        /* FP16 */
        llama_gpu_reset(&ctx);
        int fp16_out[GEN_TOKENS];
        CUDA_CHECK(cudaEventRecord(t0));
        fp16_out[0] = 0;
        for (int i = 0; i < p->len; i++)
            fp16_out[0] = lgpu_forward_token(&ctx, p->p[i], i);
        CUDA_CHECK(cudaEventRecord(t1)); CUDA_CHECK(cudaEventSynchronize(t1));
        float fp16_pre; CUDA_CHECK(cudaEventElapsedTime(&fp16_pre, t0, t1));

        CUDA_CHECK(cudaEventRecord(t0));
        for (int g = 1; g < GEN_TOKENS; g++)
            fp16_out[g] = lgpu_forward_token(&ctx, fp16_out[g-1], p->len+g-1);
        CUDA_CHECK(cudaEventRecord(t1)); CUDA_CHECK(cudaEventSynchronize(t1));
        float fp16_d; CUDA_CHECK(cudaEventElapsedTime(&fp16_d, t0, t1));

        int fp16_m=0;
        for (int g=0;g<GEN_TOKENS;g++) if(fp16_out[g]==p->r[g]) fp16_m++;
        fp16_total += fp16_m;
        fp16_dec_ms += fp16_d;

        /* FP8 */
        llama_gpu_reset(&ctx);
        int fp8_out[GEN_TOKENS];
        CUDA_CHECK(cudaEventRecord(t0));
        fp8_out[0] = 0;
        for (int i = 0; i < p->len; i++)
            fp8_out[0] = fp8_forward_token(&ctx, lt, &f8m, p->p[i], i);
        CUDA_CHECK(cudaEventRecord(t1)); CUDA_CHECK(cudaEventSynchronize(t1));
        float fp8_pre; CUDA_CHECK(cudaEventElapsedTime(&fp8_pre, t0, t1));

        CUDA_CHECK(cudaEventRecord(t0));
        for (int g = 1; g < GEN_TOKENS; g++)
            fp8_out[g] = fp8_forward_token(&ctx, lt, &f8m, fp8_out[g-1], p->len+g-1);
        CUDA_CHECK(cudaEventRecord(t1)); CUDA_CHECK(cudaEventSynchronize(t1));
        float fp8_d; CUDA_CHECK(cudaEventElapsedTime(&fp8_d, t0, t1));

        int fp8_m=0, fp8_div=-1;
        for (int g=0;g<GEN_TOKENS;g++) {
            if(fp8_out[g]==p->r[g]) fp8_m++;
            else if(fp8_div<0) fp8_div=g;
        }
        fp8_total += fp8_m;
        fp8_dec_ms += fp8_d;
        dec_steps += GEN_TOKENS - 1;

        int agree=0;
        for(int g=0;g<GEN_TOKENS;g++) if(fp16_out[g]==fp8_out[g]) agree++;

        float fp16_tps = 1000.f*(GEN_TOKENS-1)/fp16_d;
        float fp8_tps  = 1000.f*(GEN_TOKENS-1)/fp8_d;

        printf("  FP16: %d/%d oracle  %.1fms decode  %.0f tok/s\n", fp16_m, GEN_TOKENS, fp16_d, fp16_tps);
        printf("  FP8:  %d/%d oracle  %.1fms decode  %.0f tok/s", fp8_m, GEN_TOKENS, fp8_d, fp8_tps);
        if(fp8_div>=0) printf("  first_div=%d", fp8_div);
        printf("\n  agree: %d/%d  speedup: %.2fx\n\n", agree, GEN_TOKENS, fp16_d/fp8_d);
    }

    float fp16_avg = fp16_dec_ms / dec_steps;
    float fp8_avg  = fp8_dec_ms / dec_steps;
    printf("========================================\n");
    printf("FP16: %d/80 oracle  %.2f ms/tok (%.0f tok/s)\n", fp16_total, fp16_avg, 1000/fp16_avg);
    printf("FP8:  %d/80 oracle  %.2f ms/tok (%.0f tok/s)\n", fp8_total, fp8_avg, 1000/fp8_avg);
    printf("Speedup: %.2fx\n", fp16_avg/fp8_avg);
    printf("========================================\n");

    cudaEventDestroy(t0); cudaEventDestroy(t1);
    fp8_model_free(&f8m);
    cublasLtDestroy(lt);
    llama_gpu_free(&ctx);
    printf("\nDone.\n");
    return 0;
}
