/*
 * bench_full_fp16.cu — 80/80 oracle gate benchmark for FP16 GPU TinyLlama
 *
 * Uses the reusable runtime from llama_gpu.cuh. Runs all 4 reference prompts,
 * compares generated tokens against CPU oracle, reports timing.
 *
 * Build:
 *   NVCC=/usr/local/cuda-13.0/bin/nvcc
 *   $NVCC -O3 -arch=sm_120 bench_full_fp16.cu -o bench_full_fp16 -lcublas -lm
 *
 * Run:
 *   ./bench_full_fp16 /path/to/TinyLlama-model-dir
 *
 * Copyright (c) 2026 — MicroGPT-C R11 GPU benchmark
 */

#include "llama_gpu.cuh"

/* ================================================================
 * Reference prompts and expected tokens (from llama_int.c + reference_tokens.txt)
 * ================================================================ */

#define GEN_TOKENS 20

typedef struct {
    const char *name;
    const int  *prompt;
    int         prompt_len;
    const int  *ref;
} PromptCase;

static const int P_FRANCE[] = {1, 450, 7483, 310, 3444, 338};
static const int R_FRANCE[] = {
    3681, 29889, 13, 13, 29906, 29889, 350, 29889, 450, 7483,
    310, 9556, 338, 5115, 29889, 13, 13, 29941, 29889, 315};

static const int P_STORY[] = {1, 9038, 2501, 263, 931};
static const int R_STORY[] = {
    29892, 727, 471, 263, 4123, 6114, 4257, 365, 2354, 29889,
    2296, 10600, 297, 263, 2319, 4726, 29892, 988, 14332, 6363};

static const int P_MATH[] = {1, 29896, 29974, 29896, 29922};
static const int R_MATH[] = {
    29896, 13, 13, 6295, 278, 2533, 310, 278, 937, 302,
    4958, 310, 278, 383, 747, 265, 21566, 5665, 338, 29871};

static const int P_LIFE[] = {1, 450, 6593, 310, 2834, 338};
static const int R_LIFE[] = {
    304, 1284, 596, 15935, 29892, 12359, 434, 372, 411, 599,
    596, 1795, 29892, 322, 5967, 263, 25000, 29889, 13, 13};

static const PromptCase ALL_PROMPTS[] = {
    {"france_capital",  P_FRANCE, 6, R_FRANCE},
    {"story_beginning", P_STORY,  5, R_STORY},
    {"simple_math",     P_MATH,   5, R_MATH},
    {"meaning_of_life", P_LIFE,   6, R_LIFE},
};
#define N_PROMPTS 4

/* ================================================================
 * Main
 * ================================================================ */

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model_dir>\n", argv[0]); return 1;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (%d SMs, compute %d.%d)\n",
           prop.name, prop.multiProcessorCount, prop.major, prop.minor);

    LlamaGPU ctx;
    printf("Loading model...\n");
    if (llama_gpu_init(&ctx, argv[1]) != 0) return 1;
    printf("Model loaded.\n");

    /* Timer */
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    int total_match = 0, total_tokens = 0;
    float total_decode_ms = 0;
    int total_decode_steps = 0;

    for (int pi = 0; pi < N_PROMPTS; pi++) {
        const PromptCase *pc = &ALL_PROMPTS[pi];
        printf("\n=== %s (%d prefill + %d decode) ===\n",
               pc->name, pc->prompt_len, GEN_TOKENS);

        int output[GEN_TOKENS];
        llama_gpu_reset(&ctx);

        /* --- Prefill (timed separately) --- */
        CUDA_CHECK(cudaEventRecord(t0, 0));
        int first_tok = 0;
        for (int p = 0; p < pc->prompt_len; p++)
            first_tok = lgpu_forward_token(&ctx, pc->prompt[p], p);
        CUDA_CHECK(cudaEventRecord(t1, 0));
        CUDA_CHECK(cudaEventSynchronize(t1));
        float prefill_ms;
        CUDA_CHECK(cudaEventElapsedTime(&prefill_ms, t0, t1));

        output[0] = first_tok;

        /* --- Decode (timed separately) --- */
        CUDA_CHECK(cudaEventRecord(t0, 0));
        for (int g = 1; g < GEN_TOKENS; g++) {
            int pos = pc->prompt_len + g - 1;
            output[g] = lgpu_forward_token(&ctx, output[g-1], pos);
        }
        CUDA_CHECK(cudaEventRecord(t1, 0));
        CUDA_CHECK(cudaEventSynchronize(t1));
        float decode_ms;
        CUDA_CHECK(cudaEventElapsedTime(&decode_ms, t0, t1));
        int decode_steps = GEN_TOKENS - 1;

        /* Compare */
        int match = 0, first_div = -1;
        for (int g = 0; g < GEN_TOKENS; g++) {
            if (output[g] == pc->ref[g]) match++;
            else if (first_div < 0) first_div = g;
        }

        printf("  Generated: [");
        for (int g = 0; g < GEN_TOKENS; g++)
            printf("%s%d%s", g ? "," : "", output[g],
                   output[g] == pc->ref[g] ? "" : "*");
        printf("]\n");

        printf("  Match: %d/%d", match, GEN_TOKENS);
        if (first_div >= 0)
            printf("  first_div=%d (got %d, ref %d)",
                   first_div, output[first_div], pc->ref[first_div]);
        else
            printf("  PERFECT");

        float dpt = decode_ms / decode_steps;
        printf("\n  Prefill: %.1f ms  Decode: %.1f ms (%.2f ms/tok, %.0f tok/s)\n",
               prefill_ms, decode_ms, dpt, 1000.0 / dpt);

        total_match += match;
        total_tokens += GEN_TOKENS;
        total_decode_ms += decode_ms;
        total_decode_steps += decode_steps;
    }

    printf("\n========================================\n");
    printf("=== TOTAL: %d/%d tokens match (%.1f%%) ===\n",
           total_match, total_tokens,
           100.0 * total_match / total_tokens);
    if (total_match == total_tokens)
        printf("=== PERFECT MATCH — all tokens identical to CPU oracle ===\n");
    printf("========================================\n");

    float avg_decode = total_decode_ms / total_decode_steps;
    printf("\nDecode average: %.2f ms/tok (%.0f tok/s)\n",
           avg_decode, 1000.0 / avg_decode);

    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    llama_gpu_free(&ctx);
    printf("Done.\n");
    return 0;
}
