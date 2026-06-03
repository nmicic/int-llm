/*
 * bench_graph_fp16.cu — CUDA graph capture test for FP16 decode
 *
 * Measures launch overhead by comparing:
 *   1. Baseline: N repeated lgpu_forward_token calls at a fixed position
 *   2. Graph:    N replays of a captured forward pass at the same position
 *
 * The difference is the launch overhead that graphs eliminate.
 * Real generation (autoregressive) is NOT graphed — each step has different
 * pos/seq_len parameters. This benchmark measures the ceiling.
 *
 * Also runs the correctness gates (non-graphed) to confirm FP16 baseline.
 *
 * Build:
 *   NVCC=/usr/local/cuda-13.0/bin/nvcc
 *   $NVCC -O3 -arch=sm_120 bench_graph_fp16.cu -o bench_graph_fp16 -lcublas -lm
 *
 * Run:
 *   ./bench_graph_fp16 /path/to/TinyLlama-model-dir
 *
 * Copyright (c) 2026 — MicroGPT-C R11 CUDA graph evaluation
 */

#include "llama_gpu.cuh"

/* ================================================================
 * Forward pass without D2H (capturable by CUDA graph)
 * All ops on given stream. Uses cudaMemcpyAsync for D2D.
 * Attention shared mem allocated for worst case (MAX_SEQ).
 * ================================================================ */

static void lgpu_layer_forward_async(cublasHandle_t h, half *d_hidden,
                                      LGpuLayerWeights *w, LGpuScratch *s,
                                      int layer_idx, int pos, cudaStream_t stream) {
    int seq_len = pos + 1, hd = LGPU_HEAD_DIM / 2, b = 256;
    int64_t kv_off = (int64_t)layer_idx * LGPU_MAX_SEQ * LGPU_KV_DIM;

    lgpu_kern_rmsnorm<<<1, LGPU_NORM_THREADS, LGPU_NORM_THREADS * sizeof(float), stream>>>(
        s->normed, d_hidden, w->input_norm, LGPU_DIM);

    /* cuBLAS uses the stream set on the handle */
    lgpu_matmul(h, s->q, w->q_proj, s->normed, LGPU_DIM, LGPU_DIM);
    lgpu_matmul(h, s->k, w->k_proj, s->normed, LGPU_KV_DIM, LGPU_DIM);
    lgpu_matmul(h, s->v, w->v_proj, s->normed, LGPU_KV_DIM, LGPU_DIM);

    lgpu_kern_rope<<<(LGPU_N_HEADS * hd + b-1)/b, b, 0, stream>>>(
        s->q, s->rope_cos, s->rope_sin, pos, LGPU_N_HEADS, hd);
    lgpu_kern_rope<<<(LGPU_N_KV_HEADS * hd + b-1)/b, b, 0, stream>>>(
        s->k, s->rope_cos, s->rope_sin, pos, LGPU_N_KV_HEADS, hd);

    /* Async D2D for KV store (required for graph capture) */
    cudaMemcpyAsync(s->kv_cache_k + kv_off + pos * LGPU_KV_DIM,
                    s->k, LGPU_KV_DIM * sizeof(half),
                    cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(s->kv_cache_v + kv_off + pos * LGPU_KV_DIM,
                    s->v, LGPU_KV_DIM * sizeof(half),
                    cudaMemcpyDeviceToDevice, stream);

    /* Attention: use max shared mem so launch config is position-invariant */
    int attn_smem = (LGPU_MAX_SEQ + LGPU_HEAD_DIM) * sizeof(float);
    lgpu_kern_attention<<<LGPU_N_HEADS, LGPU_HEAD_DIM, attn_smem, stream>>>(
        s->attn_out, s->q,
        s->kv_cache_k + kv_off, s->kv_cache_v + kv_off,
        seq_len, LGPU_N_HEADS, LGPU_N_KV_HEADS, LGPU_HEAD_DIM, LGPU_GQA_GROUP);

    lgpu_matmul(h, s->o_out, w->o_proj, s->attn_out, LGPU_DIM, LGPU_DIM);
    lgpu_kern_residual_add<<<(LGPU_DIM+b-1)/b, b, 0, stream>>>(
        d_hidden, s->o_out, LGPU_DIM);

    lgpu_kern_rmsnorm<<<1, LGPU_NORM_THREADS, LGPU_NORM_THREADS * sizeof(float), stream>>>(
        s->normed2, d_hidden, w->post_attn_norm, LGPU_DIM);

    lgpu_matmul(h, s->gate, w->gate_proj, s->normed2, LGPU_INTER_DIM, LGPU_DIM);
    lgpu_matmul(h, s->up,   w->up_proj,   s->normed2, LGPU_INTER_DIM, LGPU_DIM);
    lgpu_kern_silu_mul<<<(LGPU_INTER_DIM+b-1)/b, b, 0, stream>>>(
        s->silu_up, s->gate, s->up, LGPU_INTER_DIM);
    lgpu_matmul(h, s->mlp_out, w->down_proj, s->silu_up, LGPU_DIM, LGPU_INTER_DIM);
    lgpu_kern_residual_add<<<(LGPU_DIM+b-1)/b, b, 0, stream>>>(
        d_hidden, s->mlp_out, LGPU_DIM);
}

static void lgpu_forward_nohost(LlamaGPU *ctx, int token, int pos,
                                 cudaStream_t stream) {
    LGpuScratch *s = &ctx->scratch;

    lgpu_kern_embed_lookup<<<(LGPU_DIM+255)/256, 256, 0, stream>>>(
        ctx->d_hidden, ctx->weights.embed_tokens, token, LGPU_DIM);

    for (int L = 0; L < LGPU_N_LAYERS; L++)
        lgpu_layer_forward_async(ctx->cublas, ctx->d_hidden,
                                  &ctx->weights.layers[L], s, L, pos, stream);

    lgpu_kern_rmsnorm<<<1, LGPU_NORM_THREADS,
                        LGPU_NORM_THREADS * sizeof(float), stream>>>(
        s->normed, ctx->d_hidden, ctx->weights.final_norm, LGPU_DIM);

    lgpu_matmul(ctx->cublas, s->logits, ctx->weights.lm_head, s->normed,
                LGPU_VOCAB_SIZE, LGPU_DIM);

    int smem = LGPU_NORM_THREADS * (sizeof(float) + sizeof(int));
    lgpu_kern_argmax<<<1, LGPU_NORM_THREADS, smem, stream>>>(
        s->d_argmax_result, s->logits, LGPU_VOCAB_SIZE);
}

/* ================================================================
 * Reference tokens for correctness gate
 * ================================================================ */

#define GEN_TOKENS 20
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
typedef struct { const char *name; const int *p; int len; const int *r; } PCase;
static const PCase PCASES[] = {
    {"france",P_FR,6,R_FR}, {"story",P_ST,5,R_ST},
    {"math",P_MA,5,R_MA}, {"life",P_LI,6,R_LI},
};

/* 251-token long reference (france) for spot check */
static const int REF_251[] = {
    3681,29889,13,13,29906,29889,350,29889,450,7483,
    310,9556,338,5115,29889,13,13,29941,29889,315,
    29889,450,7483,310,278,3303,3900,338,7660,29892,
    360,29889,29907,29889,13,13,29946,29889,319,29889,
    450,7483,310,7400,338,13476,10011,29889,13,13,
    29945,29889,360,29889,450,7483,310,8314,338,1815,
    495,336,29889,13,13,29953,29889,350,29889,450,
    7483,310,278,3303,12626,338,4517,29889,13,13,
    29955,29889,315,29889,450,7483,310,1570,13450,338,
    5674,4885,29889,13,13,29947,29889,319,29889,450,
    7483,310,4275,10557,338,349,2267,4108,29889,13,
    13,29929,29889,360,29889,450,7483,310,7513,338,
    1570,5556,2918,29889,13,13,29896,29900,29889,350,
    29889,450,7483,310,5546,338,20377,29889,13,13,
    29896,29896,29889,315,29889,450,7483,310,7551,338,
    1522,823,292,29889,13,13,29896,29906,29889,319,
    29889,450,7483,310,16078,338,11638,423,29889,13,
    13,29896,29941,29889,360,29889,450,7483,310,12568,
    338,12568,4412,29889,13,13,29896,29946,29889,350,
    29889,450,7483,310,4275,19109,338,922,5059,29889,
    13,13,29896,29945,29889,315,29889,450,7483,310,
    498,26517,338,14320,29895,554,29889,13,13,29896,
    29953,29889,319,29889,450,7483,310,16704,423,338,
    13669,16979,29889,13,13,29896,29955,29889,360,29889,
    450,7483,310,278,26260,338,2315,4233,29889,13,
    13
};

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
    printf("Loading model...\n");
    if (llama_gpu_init(&ctx, argv[1]) != 0) return 1;
    printf("Model loaded.\n");

    /* Create a non-default stream for capture */
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUBLAS_CHECK(cublasSetStream(ctx.cublas, stream));

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    /* Fixed position for overhead measurement */
    int test_token = 3681;  /* first generated token from france_capital */
    int test_pos   = 50;    /* mid-range position */
    int N_ITERS    = 200;

    /* Fill KV cache up to test_pos with real data first */
    printf("\nFilling KV cache to pos=%d...\n", test_pos);
    CUBLAS_CHECK(cublasSetStream(ctx.cublas, 0));  /* default stream for prefill */
    llama_gpu_reset(&ctx);
    for (int p = 0; p < 6; p++)
        lgpu_forward_token(&ctx, P_FR[p], p);
    for (int g = 0; g < test_pos - 6; g++)
        lgpu_forward_token(&ctx, test_token, 6 + g);
    CUBLAS_CHECK(cublasSetStream(ctx.cublas, stream));

    /* ============================================================
     * Baseline: repeated forward calls (no graph)
     * ============================================================ */
    printf("\n=== Overhead Measurement (pos=%d, %d iterations) ===\n", test_pos, N_ITERS);

    /* Warmup */
    for (int i = 0; i < 10; i++)
        lgpu_forward_nohost(&ctx, test_token, test_pos, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    /* Timed baseline */
    CUDA_CHECK(cudaEventRecord(t0, stream));
    for (int i = 0; i < N_ITERS; i++)
        lgpu_forward_nohost(&ctx, test_token, test_pos, stream);
    CUDA_CHECK(cudaEventRecord(t1, stream));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float baseline_ms;
    CUDA_CHECK(cudaEventElapsedTime(&baseline_ms, t0, t1));
    float baseline_per = baseline_ms / N_ITERS;
    printf("  Baseline (no graph):  %.3f ms total, %.4f ms/iter (%.0f tok/s)\n",
           baseline_ms, baseline_per, 1000.0 / baseline_per);

    /* ============================================================
     * Graph capture attempt
     * ============================================================ */
    printf("\n=== CUDA Graph Capture ===\n");

    cudaGraph_t graph;
    cudaGraphExec_t graphExec = NULL;

    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    lgpu_forward_nohost(&ctx, test_token, test_pos, stream);
    cudaError_t cap_err = cudaStreamEndCapture(stream, &graph);

    if (cap_err != cudaSuccess) {
        printf("  Graph capture FAILED: %s\n", cudaGetErrorString(cap_err));
        printf("  Skipping graph replay measurement.\n");
    } else {
        size_t numNodes;
        CUDA_CHECK(cudaGraphGetNodes(graph, NULL, &numNodes));
        printf("  Graph captured: %zu nodes\n", numNodes);

        CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, 0));

        /* Warmup replays */
        for (int i = 0; i < 10; i++)
            CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        /* Timed replays */
        CUDA_CHECK(cudaEventRecord(t0, stream));
        for (int i = 0; i < N_ITERS; i++)
            CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
        CUDA_CHECK(cudaEventRecord(t1, stream));
        CUDA_CHECK(cudaEventSynchronize(t1));
        float graph_ms;
        CUDA_CHECK(cudaEventElapsedTime(&graph_ms, t0, t1));
        float graph_per = graph_ms / N_ITERS;

        float speedup = baseline_per / graph_per;
        float saved_us = (baseline_per - graph_per) * 1000;

        printf("  Graph replay:         %.3f ms total, %.4f ms/iter (%.0f tok/s)\n",
               graph_ms, graph_per, 1000.0 / graph_per);
        printf("  Speedup: %.2fx  (saved %.0f µs/iter = %.0f%% of baseline)\n",
               speedup, saved_us, (1.0 - graph_per/baseline_per) * 100);

        printf("\n  NOTE: This is same-position replay (NOT autoregressive generation).\n");
        printf("  It measures the CPU launch overhead ceiling that graphs eliminate.\n");
        printf("  Real autoregressive use requires per-step node updates for pos/seq_len.\n");

        cudaGraphExecDestroy(graphExec);
        cudaGraphDestroy(graph);
    }

    /* ============================================================
     * Correctness verification (non-graphed, standard path)
     * ============================================================ */
    printf("\n=== Correctness Verification (non-graphed FP16) ===\n");
    CUBLAS_CHECK(cublasSetStream(ctx.cublas, 0));  /* back to default stream */

    /* 80/80 short gate */
    int total_match = 0;
    for (int pi = 0; pi < 4; pi++) {
        const PCase *pc = &PCASES[pi];
        int output[GEN_TOKENS];

        llama_gpu_reset(&ctx);
        output[0] = 0;
        for (int p = 0; p < pc->len; p++)
            output[0] = lgpu_forward_token(&ctx, pc->p[p], p);
        for (int g = 1; g < GEN_TOKENS; g++)
            output[g] = lgpu_forward_token(&ctx, output[g-1], pc->len + g - 1);

        int match = 0;
        for (int g = 0; g < GEN_TOKENS; g++)
            if (output[g] == pc->r[g]) match++;
        total_match += match;
        printf("  %s: %d/%d %s\n", pc->name, match, GEN_TOKENS,
               match == GEN_TOKENS ? "PERFECT" : "DIVERGED");
    }
    printf("  80-token gate: %d/80\n", total_match);

    /* 251-token long spot check */
    llama_gpu_reset(&ctx);
    int long_out[251];
    long_out[0] = 0;
    for (int p = 0; p < 6; p++)
        long_out[0] = lgpu_forward_token(&ctx, P_FR[p], p);
    for (int g = 1; g < 251; g++) {
        int pos = 6 + g - 1;
        if (pos >= LGPU_MAX_SEQ) break;
        long_out[g] = lgpu_forward_token(&ctx, long_out[g-1], pos);
    }
    int long_match = 0;
    for (int g = 0; g < 251; g++)
        if (long_out[g] == REF_251[g]) long_match++;
    printf("  251-token gate: %d/251 %s\n", long_match,
           long_match == 251 ? "PERFECT" : "DIVERGED");

    /* Cleanup */
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaStreamDestroy(stream);
    llama_gpu_free(&ctx);
    printf("\nDone.\n");
    return 0;
}
