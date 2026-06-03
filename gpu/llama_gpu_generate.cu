/*
 * llama_gpu_generate.cu — Standalone FP16 GPU inference CLI for TinyLlama
 *
 * Loads model from safetensors, runs greedy decode, prints generated tokens.
 * Uses the reusable runtime from llama_gpu.cuh.
 *
 * Build:
 *   NVCC=/usr/local/cuda-13.0/bin/nvcc
 *   $NVCC -O3 -arch=sm_120 llama_gpu_generate.cu -o llama_gpu_generate -lcublas -lm
 *
 * Run:
 *   ./llama_gpu_generate <model_dir>
 *   ./llama_gpu_generate <model_dir> --tokens 1,450,7483,310,3444,338 --max-new 20
 *
 * Copyright (c) 2026 — MicroGPT-C R11 GPU runtime
 */

#include "llama_gpu.cuh"
#include <cuda_runtime.h>

/* Default prompt: PROMPT_FRANCE from llama_int.c */
static const int DEFAULT_PROMPT[] = {1, 450, 7483, 310, 3444, 338};
#define DEFAULT_PROMPT_LEN 6
#define DEFAULT_MAX_NEW    20

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model_dir> [--tokens T1,T2,...] [--max-new N]\n",
                argv[0]);
        return 1;
    }
    const char *model_dir = argv[1];

    /* Parse optional args */
    const int *prompt = DEFAULT_PROMPT;
    int prompt_len = DEFAULT_PROMPT_LEN;
    int max_new = DEFAULT_MAX_NEW;
    int custom_tokens[LGPU_MAX_SEQ];
    int custom_len = 0;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) {
            /* Parse comma-separated token IDs */
            const char *s = argv[++i];
            while (*s && custom_len < LGPU_MAX_SEQ) {
                custom_tokens[custom_len++] = atoi(s);
                while (*s && *s != ',') s++;
                if (*s == ',') s++;
            }
            prompt = custom_tokens;
            prompt_len = custom_len;
        } else if (strcmp(argv[i], "--max-new") == 0 && i + 1 < argc) {
            max_new = atoi(argv[++i]);
        }
    }

    /* Validate input */
    if (prompt_len <= 0) {
        fprintf(stderr, "Error: prompt must have at least one token.\n"); return 1;
    }
    if (max_new <= 0) {
        fprintf(stderr, "Error: --max-new must be >= 1.\n"); return 1;
    }

    /* GPU info */
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (%d SMs)\n", prop.name, prop.multiProcessorCount);

    /* Init model */
    printf("Loading model from %s ...\n", model_dir);
    LlamaGPU ctx;
    if (llama_gpu_init(&ctx, model_dir) != 0) {
        fprintf(stderr, "Failed to load model.\n");
        return 1;
    }
    printf("Model loaded.\n\n");

    /* Print prompt */
    printf("Prompt (%d tokens): [", prompt_len);
    for (int i = 0; i < prompt_len; i++)
        printf("%s%d", i ? ", " : "", prompt[i]);
    printf("]\n");

    /* Generate with separate prefill/decode timing */
    int *output = (int*)malloc(max_new * sizeof(int));

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    llama_gpu_reset(&ctx);

    /* Prefill */
    CUDA_CHECK(cudaEventRecord(t0, 0));
    int first_tok = 0;
    for (int p = 0; p < prompt_len; p++)
        first_tok = lgpu_forward_token(&ctx, prompt[p], p);
    CUDA_CHECK(cudaEventRecord(t1, 0));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float prefill_ms;
    CUDA_CHECK(cudaEventElapsedTime(&prefill_ms, t0, t1));

    output[0] = first_tok;
    int n_gen = 1;

    /* Decode */
    CUDA_CHECK(cudaEventRecord(t0, 0));
    for (int g = 1; g < max_new; g++) {
        int pos = prompt_len + g - 1;
        if (pos >= LGPU_MAX_SEQ) break;
        output[g] = lgpu_forward_token(&ctx, output[g-1], pos);
        n_gen = g + 1;
    }
    CUDA_CHECK(cudaEventRecord(t1, 0));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float decode_ms;
    CUDA_CHECK(cudaEventElapsedTime(&decode_ms, t0, t1));

    /* Print results */
    printf("Generated (%d tokens): [", n_gen);
    for (int i = 0; i < n_gen; i++)
        printf("%s%d", i ? ", " : "", output[i]);
    printf("]\n\n");

    int decode_steps = n_gen > 1 ? n_gen - 1 : 0;
    float dpt = decode_steps > 0 ? decode_ms / decode_steps : 0;
    printf("Prefill: %.1f ms (%d tokens)  ", prefill_ms, prompt_len);
    printf("Decode: %.1f ms (%.2f ms/tok, %.0f tok/s)\n",
           decode_ms, dpt, dpt > 0 ? 1000.0 / dpt : 0);

    /* Cleanup */
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    free(output);
    llama_gpu_free(&ctx);
    printf("Done.\n");
    return 0;
}
