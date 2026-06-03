/*
 * Author: Nenad Mićić
 * LinkedIn: https://be.linkedin.com/in/nenadmicic
 *
 * Copyright (c) 2026 Nenad Mićić
 * SPDX-License-Identifier: Apache-2.0
 *
 * llama_int.c — Integer-Only Llama-Family Inference Engine
 * ========================================================
 *
 * Llama-family inference using ZERO floating-point operations.
 * All computation in Q16.48 fixed-point via fp_math.h.
 *
 * Supports any Llama-architecture model via config.json:
 *   - TinyLlama-1.1B (22 layers, 2048 dim, GQA 32Q/4KV heads)
 *   - Llama-2-7B     (32 layers, 4096 dim, MHA 32Q/32KV heads)
 *   - Other Llama variants with GQA or MHA
 *
 * Architecture:
 *   - Activation: SwiGLU (silu(gate) * up)
 *   - Normalization: RMSNorm (not LayerNorm)
 *   - Position encoding: RoPE (Rotary Position Embeddings)
 *   - Attention: MHA or GQA (Grouped Query Attention)
 *
 * Weight loading: two modes —
 *   --cache-layers: convert all layers at startup (high RAM).
 *   streaming (default): convert one layer at a time, free after use.
 *
 * Compile: gcc -O3 -march=native -o llama_int llama_int.c
 * Run:     ./llama_int <model_dir> [--verify | --generate | --benchmark]
 */

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <limits.h>
#include <sys/wait.h>
#include <unistd.h>

#include "fp_math.h"
#include "safetensors.h"
#include "tokenizer.h"

/* ================================================================== */
/*  Profiling / Observability (R2)                                      */
/* ================================================================== */

/*
 * Lightweight wall-clock profiling for the integer inference runtime.
 * Accumulates timings and memory sizes in a single struct, emitted
 * at exit as human-readable summary + machine-readable KPI lines.
 * Optional JSON output via --profile-json <path>.
 *
 * Instrumentation overhead: one clock_gettime pair per measured stage.
 * No allocation, no threads, no global state beyond the struct.
 */

static double ts_diff(struct timespec a, struct timespec b) {
    return (b.tv_sec - a.tv_sec) + (b.tv_nsec - a.tv_nsec) / 1e9;
}

typedef struct {
    /* Wall-clock timings (seconds) */
    double t_safetensors_load;
    double t_global_weights;
    double t_rope_init;
    double t_kv_alloc;
    double t_layer_cache;       /* 0 if streaming */
    double t_prefill_total;
    double t_decode_total;
    double t_generation_total;

    /* Per-token streaming breakdown (accumulated across generate()) */
    double t_layer_load_total;  /* streaming: safetensors→Q16.48 conversion */
    double t_layer_compute_total; /* transformer_layer_forward time */

    /* Counts */
    int    prefill_tokens;
    int    decode_tokens;
    int    forward_calls;       /* total llama_forward() calls */

    /* Memory accounting (bytes) */
    size_t mem_embed_tokens;
    size_t mem_lm_head;         /* 0 if tied */
    size_t mem_final_norm;
    size_t mem_kv_cache;
    size_t mem_rope_tables;
    size_t mem_cached_layers;   /* 0 if streaming */
    size_t mem_one_layer;       /* estimated single-layer working set */

    /* Per-layer stage timing (R3-lite), accumulated across all forward calls. */
    double *t_per_layer_attn;   /* norm1 -> residual1, one slot per layer */
    double *t_per_layer_mlp;    /* norm2 -> residual2, one slot per layer */
    int    num_layers;          /* array length for per-layer timing */

    /* Flags */
    int    enabled;
    int    layers_cached;       /* 1 = cached mode, 0 = streaming */
    int    native_loaded;       /* 1 = native .mgw mmap path */
    int    native_stream;       /* 1 = native .mgw streaming path (R9B) */
    int    native_compressed;   /* 1 = compressed .mgwc path (R9C-A) */
} profile_t;

/* Global profile — zero-initialized by default (disabled) */
static profile_t g_profile;

static void *checked_calloc(size_t count, size_t size, const char *what);

static void profile_init_layers(int num_layers) {
    if (!g_profile.enabled || g_profile.t_per_layer_attn || num_layers <= 0) return;
    g_profile.num_layers = num_layers;
    g_profile.t_per_layer_attn = (double *)checked_calloc(
        (size_t)num_layers, sizeof(double), "profile_layer_attn");
    g_profile.t_per_layer_mlp = (double *)checked_calloc(
        (size_t)num_layers, sizeof(double), "profile_layer_mlp");
}

static void profile_free_layers(void) {
    free(g_profile.t_per_layer_attn);
    free(g_profile.t_per_layer_mlp);
    g_profile.t_per_layer_attn = NULL;
    g_profile.t_per_layer_mlp = NULL;
    g_profile.num_layers = 0;
}

/* ================================================================== */
/*  Runtime KV Quantize-in-Place (R9C-B3)                               */
/* ================================================================== */

/*
 * Experimental lossy KV compression: after writing K/V to the cache,
 * immediately quantize each element to b bits using per-position
 * L-inf magnitude scaling, then dequantize back to Q16.48 in place.
 *
 * This simulates the effect of compressed KV storage on token output
 * without changing data structures. The attention path reads the same
 * cache pointers — it just sees lossy-reconstructed values.
 *
 * Per-position scaling: each KV vector (per head) gets its own magnitude
 * M = max(|v[i]|). Elements are quantized into [-M, M] with b bits.
 * This avoids the running-min/max problem of per-dim-across-sequence.
 */

typedef struct {
    int enabled;     /* 1 = quantize K/V in place after cache write */
    int k_bits;      /* 0 = exact K, >0 = quantize to this many bits */
    int v_bits;      /* 0 = exact V, >0 = quantize to this many bits */
    int packed;      /* 1 = use packed storage (--kv-packed), 0 = in-place only */
} kv_quant_config_t;

static kv_quant_config_t g_kv_quant = {0};

/* Quantize-then-dequantize a fixed_t vector in place.
 * Uses per-vector L-inf magnitude with b-bit uniform quantization. */
static void kv_quantize_inplace(fixed_t *vec, int D, int b) {
    if (b <= 0 || b >= 64) return;  /* 0 = exact, >= 64 = no-op */
    uint64_t Q = (uint64_t)1 << b;

    /* Find magnitude */
    int64_t M = 0;
    for (int i = 0; i < D; i++) {
        int64_t a = vec[i] < 0 ? -vec[i] : vec[i];
        if (a > M) M = a;
    }
    if (M == 0) return;  /* all zeros — nothing to do */

    /* Quantize and dequantize each element */
    for (int i = 0; i < D; i++) {
        /* q = floor((v[i] + M) * Q / (2*M)), clamped to [0, Q-1] */
        __int128 q = ((__int128)(vec[i] + M) * (int64_t)Q) / (2 * M);
        if (q < 0) q = 0;
        if (q >= (int64_t)Q) q = (int64_t)Q - 1;
        /* Dequantize: midpoint of bin → v'[i] = (2*q+1)*M/Q - M */
        vec[i] = (fixed_t)(((__int128)(2 * (int64_t)q + 1) * M) / (int64_t)Q) - M;
    }
}

static void profile_print_summary(const profile_t *p, const char *model_name) {
    if (!p->enabled) return;

    size_t total_mem = p->mem_embed_tokens + p->mem_lm_head + p->mem_final_norm
                     + p->mem_kv_cache + p->mem_rope_tables + p->mem_cached_layers;
    if (!p->layers_cached)
        total_mem += p->mem_one_layer;  /* streaming: one layer always resident */
    double total_init = p->t_safetensors_load + p->t_global_weights
                      + p->t_rope_init + p->t_kv_alloc + p->t_layer_cache;

    printf("\n");
    printf("================================================================\n");
    printf("  PROFILE SUMMARY\n");
    printf("================================================================\n");

    printf("\n  Timing (seconds):\n");
    if (p->native_compressed)
        printf("    compressed_load      %8.3f\n", p->t_safetensors_load);
    else if (p->native_loaded || p->native_stream)
        printf("    native_load          %8.3f\n", p->t_safetensors_load);
    else {
        printf("    safetensors_load     %8.3f\n", p->t_safetensors_load);
        printf("    global_weights       %8.3f\n", p->t_global_weights);
    }
    printf("    rope_init            %8.3f\n", p->t_rope_init);
    printf("    kv_alloc             %8.3f\n", p->t_kv_alloc);
    if (p->layers_cached)
        printf("    layer_cache          %8.3f\n", p->t_layer_cache);
    printf("    init_total           %8.3f\n", total_init);
    printf("    ---\n");
    if (p->prefill_tokens > 0)
        printf("    prefill              %8.3f  (%d tokens, %.1f ms/tok)\n",
               p->t_prefill_total, p->prefill_tokens,
               p->prefill_tokens > 0 ? 1000.0 * p->t_prefill_total / p->prefill_tokens : 0);
    if (p->decode_tokens > 0)
        printf("    decode               %8.3f  (%d tokens, %.1f ms/tok)\n",
               p->t_decode_total, p->decode_tokens,
               p->decode_tokens > 0 ? 1000.0 * p->t_decode_total / p->decode_tokens : 0);
    if (p->t_generation_total > 0)
        printf("    generation_total     %8.3f\n", p->t_generation_total);
    if (!p->layers_cached && p->t_layer_load_total > 0) {
        double compute_frac = 0;
        double total_fwd = p->t_layer_load_total + p->t_layer_compute_total;
        if (total_fwd > 0) compute_frac = 100.0 * p->t_layer_compute_total / total_fwd;
        printf("    ---\n");
        printf("    layer_load (stream)  %8.3f  (%.0f%% of layer time)\n",
               p->t_layer_load_total,
               total_fwd > 0 ? 100.0 * p->t_layer_load_total / total_fwd : 0);
        printf("    layer_compute        %8.3f  (%.0f%% of layer time)\n",
               p->t_layer_compute_total, compute_frac);
    }

    printf("\n  Memory (MiB):\n");
    printf("    embed_tokens         %8.1f\n", p->mem_embed_tokens / (1024.0 * 1024.0));
    if (p->mem_lm_head > 0)
        printf("    lm_head              %8.1f\n", p->mem_lm_head / (1024.0 * 1024.0));
    else
        printf("    lm_head              (tied to embed_tokens)\n");
    printf("    final_norm           %8.1f\n", p->mem_final_norm / (1024.0 * 1024.0));
    printf("    kv_cache             %8.1f\n", p->mem_kv_cache / (1024.0 * 1024.0));
    printf("    rope_tables          %8.3f\n", p->mem_rope_tables / (1024.0 * 1024.0));
    if (p->layers_cached)
        printf("    cached_layers        %8.1f\n", p->mem_cached_layers / (1024.0 * 1024.0));
    else
        printf("    one_layer_working    %8.1f\n", p->mem_one_layer / (1024.0 * 1024.0));
    printf("    peak_resident        %8.1f\n", total_mem / (1024.0 * 1024.0));

    /* Machine-readable KPI lines */
    printf("\n  Machine-readable KPIs:\n");
    printf("KPI:model=%s\n", model_name);
    const char *load_path_str = p->native_compressed ? "native-compressed" :
                                (p->native_loaded || p->native_stream) ? "native" : "safetensors";
    const char *mode_str = p->native_compressed ? "native-compressed" :
                           p->native_loaded ? "native" :
                           p->native_stream ? "native-stream" :
                           p->layers_cached ? "cached" : "streaming";
    printf("KPI:load_path=%s\n", load_path_str);
    printf("KPI:mode=%s\n", mode_str);
    if (p->native_compressed) {
        printf("KPI:t_native_compressed_load=%.4f\n", p->t_safetensors_load);
    } else if (p->native_loaded || p->native_stream) {
        printf("KPI:t_native_load=%.4f\n", p->t_safetensors_load);
    } else {
        printf("KPI:t_safetensors_load=%.4f\n", p->t_safetensors_load);
        printf("KPI:t_global_weights=%.4f\n", p->t_global_weights);
    }
    printf("KPI:t_rope_init=%.4f\n", p->t_rope_init);
    printf("KPI:t_kv_alloc=%.4f\n", p->t_kv_alloc);
    printf("KPI:t_layer_cache=%.4f\n", p->t_layer_cache);
    printf("KPI:t_init_total=%.4f\n", total_init);
    printf("KPI:t_prefill=%.4f\n", p->t_prefill_total);
    printf("KPI:t_decode=%.4f\n", p->t_decode_total);
    printf("KPI:t_generation=%.4f\n", p->t_generation_total);
    printf("KPI:prefill_tokens=%d\n", p->prefill_tokens);
    printf("KPI:decode_tokens=%d\n", p->decode_tokens);
    printf("KPI:forward_calls=%d\n", p->forward_calls);
    if (p->prefill_tokens > 0)
        printf("KPI:prefill_ms_per_tok=%.2f\n",
               1000.0 * p->t_prefill_total / p->prefill_tokens);
    if (p->decode_tokens > 0)
        printf("KPI:decode_ms_per_tok=%.2f\n",
               1000.0 * p->t_decode_total / p->decode_tokens);
    if (p->t_generation_total > 0 && (p->prefill_tokens + p->decode_tokens) > 0)
        printf("KPI:tok_per_sec=%.2f\n",
               (p->prefill_tokens + p->decode_tokens) / p->t_generation_total);
    if (!p->layers_cached) {
        printf("KPI:t_layer_load=%.4f\n", p->t_layer_load_total);
        printf("KPI:t_layer_compute=%.4f\n", p->t_layer_compute_total);
    }
    printf("KPI:mem_embed_bytes=%zu\n", p->mem_embed_tokens);
    printf("KPI:mem_lm_head_bytes=%zu\n", p->mem_lm_head);
    printf("KPI:mem_final_norm_bytes=%zu\n", p->mem_final_norm);
    printf("KPI:mem_kv_cache_bytes=%zu\n", p->mem_kv_cache);
    printf("KPI:mem_rope_bytes=%zu\n", p->mem_rope_tables);
    printf("KPI:mem_cached_layers_bytes=%zu\n", p->mem_cached_layers);
    printf("KPI:mem_peak_bytes=%zu\n", total_mem);
    printf("KPI:mem_one_layer_bytes=%zu\n", p->mem_one_layer);
}

static void profile_write_json(const profile_t *p, const char *path,
                                const char *model_name) {
    if (!p->enabled) return;
    FILE *f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "WARNING: cannot write profile to %s\n", path);
        return;
    }

    size_t total_mem = p->mem_embed_tokens + p->mem_lm_head + p->mem_final_norm
                     + p->mem_kv_cache + p->mem_rope_tables + p->mem_cached_layers;
    if (!p->layers_cached)
        total_mem += p->mem_one_layer;  /* streaming: one layer always resident */
    double total_init = p->t_safetensors_load + p->t_global_weights
                      + p->t_rope_init + p->t_kv_alloc + p->t_layer_cache;

    fprintf(f, "{\n");
    fprintf(f, "  \"model\": \"%s\",\n", model_name);
    const char *json_load_path = p->native_compressed ? "native-compressed" :
                                (p->native_loaded || p->native_stream) ? "native" : "safetensors";
    const char *json_mode = p->native_compressed ? "native-compressed" :
                            p->native_loaded ? "native" :
                            p->native_stream ? "native-stream" :
                            p->layers_cached ? "cached" : "streaming";
    fprintf(f, "  \"load_path\": \"%s\",\n", json_load_path);
    fprintf(f, "  \"mode\": \"%s\",\n", json_mode);
    fprintf(f, "  \"timing_seconds\": {\n");
    fprintf(f, "    \"safetensors_load\": %.4f,\n", p->t_safetensors_load);
    fprintf(f, "    \"global_weights\": %.4f,\n", p->t_global_weights);
    fprintf(f, "    \"rope_init\": %.4f,\n", p->t_rope_init);
    fprintf(f, "    \"kv_alloc\": %.4f,\n", p->t_kv_alloc);
    fprintf(f, "    \"layer_cache\": %.4f,\n", p->t_layer_cache);
    fprintf(f, "    \"init_total\": %.4f,\n", total_init);
    fprintf(f, "    \"prefill\": %.4f,\n", p->t_prefill_total);
    fprintf(f, "    \"decode\": %.4f,\n", p->t_decode_total);
    fprintf(f, "    \"generation\": %.4f,\n", p->t_generation_total);
    fprintf(f, "    \"layer_load\": %.4f,\n", p->t_layer_load_total);
    fprintf(f, "    \"layer_compute\": %.4f\n", p->t_layer_compute_total);
    fprintf(f, "  },\n");
    fprintf(f, "  \"counts\": {\n");
    fprintf(f, "    \"prefill_tokens\": %d,\n", p->prefill_tokens);
    fprintf(f, "    \"decode_tokens\": %d,\n", p->decode_tokens);
    fprintf(f, "    \"forward_calls\": %d\n", p->forward_calls);
    fprintf(f, "  },\n");
    fprintf(f, "  \"throughput\": {\n");
    if (p->prefill_tokens > 0)
        fprintf(f, "    \"prefill_ms_per_tok\": %.2f,\n",
                1000.0 * p->t_prefill_total / p->prefill_tokens);
    if (p->decode_tokens > 0)
        fprintf(f, "    \"decode_ms_per_tok\": %.2f,\n",
                1000.0 * p->t_decode_total / p->decode_tokens);
    if (p->t_generation_total > 0 && (p->prefill_tokens + p->decode_tokens) > 0)
        fprintf(f, "    \"tok_per_sec\": %.2f\n",
                (p->prefill_tokens + p->decode_tokens) / p->t_generation_total);
    else
        fprintf(f, "    \"tok_per_sec\": 0\n");
    fprintf(f, "  },\n");
    fprintf(f, "  \"memory_bytes\": {\n");
    fprintf(f, "    \"embed_tokens\": %zu,\n", p->mem_embed_tokens);
    fprintf(f, "    \"lm_head\": %zu,\n", p->mem_lm_head);
    fprintf(f, "    \"final_norm\": %zu,\n", p->mem_final_norm);
    fprintf(f, "    \"kv_cache\": %zu,\n", p->mem_kv_cache);
    fprintf(f, "    \"rope_tables\": %zu,\n", p->mem_rope_tables);
    fprintf(f, "    \"cached_layers\": %zu,\n", p->mem_cached_layers);
    fprintf(f, "    \"one_layer_working\": %zu,\n", p->mem_one_layer);
    fprintf(f, "    \"peak_resident\": %zu\n", total_mem);
    fprintf(f, "  }");
    if (p->t_per_layer_attn && p->num_layers > 0) {
        fprintf(f, ",\n  \"per_layer\": {\n");
        fprintf(f, "    \"num_layers\": %d,\n", p->num_layers);
        fprintf(f, "    \"attn_seconds\": [");
        for (int i = 0; i < p->num_layers; i++)
            fprintf(f, "%s%.6f", i ? ", " : "", p->t_per_layer_attn[i]);
        fprintf(f, "],\n");
        fprintf(f, "    \"mlp_seconds\": [");
        for (int i = 0; i < p->num_layers; i++)
            fprintf(f, "%s%.6f", i ? ", " : "", p->t_per_layer_mlp[i]);
        fprintf(f, "]\n");
        fprintf(f, "  }");
    }
    fprintf(f, "\n}\n");
    fclose(f);
    printf("Profile written to %s\n", path);
}

/* ================================================================== */
/*  Native Weight Format (R9)                                           */
/* ================================================================== */

/*
 * .mgw — MicroGPT native weight container
 *
 * Stores Q16.48 weights in a flat binary file that can be memory-mapped
 * for direct use, eliminating the repeated float→fixed conversion that
 * dominates startup and streaming-mode runtime.
 *
 * Layout:
 *   [File Header : 64 bytes]
 *   [Config Block: 64 bytes]
 *   [Tensor Index: num_tensors × 96 bytes]
 *   [Tensor Data : packed int64_t arrays, 8-byte aligned]
 *
 * All multi-byte values are host-native byte order.
 * An endian marker in the header allows rejection on cross-endian load.
 */

#define MGW_MAGIC       "MGW\0"       /* 4 bytes */
#define MGW_MAGIC_SIZE  4
#define MGW_VERSION     1
#define MGW_ENDIAN_TAG  0x01020304u   /* host-native endian check */

#define MGW_HEADER_SIZE 64
#define MGW_CONFIG_SIZE 64
#define MGW_INDEX_ENTRY_SIZE 96

/* File header — 64 bytes */
typedef struct {
    char     magic[4];        /*  0: "MGW\0"                 */
    uint32_t version;         /*  4: format version (1)      */
    uint32_t endian_tag;      /*  8: 0x01020304              */
    uint32_t num_tensors;     /* 12: number of tensors       */
    uint64_t index_offset;    /* 16: absolute offset of index */
    uint64_t data_offset;     /* 24: absolute offset of data  */
    uint8_t  reserved[32];    /* 32: zeros (forward compat)   */
} mgw_header_t;

/* Config block — 64 bytes */
typedef struct {
    int32_t hidden_dim;       /*  0 */
    int32_t num_heads;        /*  4 */
    int32_t num_kv_heads;     /*  8 */
    int32_t head_dim;         /* 12 */
    int32_t num_layers;       /* 16 */
    int32_t intermediate_dim; /* 20 */
    int32_t vocab_size;       /* 24 */
    int32_t max_seq_len;      /* 28 */
    int32_t rope_theta;       /* 32 */
    int32_t lm_head_tied;     /* 36: 1 = lm_head aliases embed_tokens */
    int32_t reserved[6];      /* 40: zeros */
} mgw_config_t;

/* Index entry — 96 bytes per tensor (fields ordered for natural alignment) */
typedef struct {
    char     name[64];        /*  0: null-terminated tensor name */
    uint64_t num_elements;    /* 64: total element count         */
    uint64_t data_offset;     /* 72: absolute file offset        */
    uint32_t ndims;           /* 80: 1 or 2                     */
    uint32_t shape[2];        /* 84: shape[1]=0 for 1D          */
    uint32_t reserved;        /* 92: zero                        */
} mgw_index_entry_t;

/* Compile-time layout checks */
_Static_assert(sizeof(mgw_header_t) == MGW_HEADER_SIZE, "mgw_header_t must be 64 bytes");
_Static_assert(sizeof(mgw_config_t) == MGW_CONFIG_SIZE, "mgw_config_t must be 64 bytes");
_Static_assert(sizeof(mgw_index_entry_t) == MGW_INDEX_ENTRY_SIZE, "mgw_index_entry_t must be 96 bytes");

/* ================================================================== */
/*  Checked allocation — abort on OOM instead of silent NULL deref      */
/* ================================================================== */

static void *checked_malloc(size_t n, const char *what) {
    void *p = malloc(n);
    if (!p && n > 0) {
        fprintf(stderr, "FATAL: OOM allocating %zu bytes for %s\n", n, what);
        exit(1);
    }
    return p;
}

static void *checked_calloc(size_t count, size_t size, const char *what) {
    void *p = calloc(count, size);
    if (!p && count > 0) {
        fprintf(stderr, "FATAL: OOM allocating %zu × %zu bytes for %s\n",
                count, size, what);
        exit(1);
    }
    return p;
}

static char *checked_strdup(const char *s, const char *what) {
    size_t n = strlen(s) + 1;
    char *out = (char *)checked_malloc(n, what);
    memcpy(out, s, n);
    return out;
}

/* ================================================================== */
/*  Tokenizer Bridge Helpers                                            */
/* ================================================================== */

static void resolve_exec_dir(const char *argv0, char *out, size_t out_sz) {
    char resolved[PATH_MAX];
    if (realpath(argv0, resolved)) {
        char *slash = strrchr(resolved, '/');
        if (slash) *slash = '\0';
        snprintf(out, out_sz, "%s", resolved);
        return;
    }
    if (getcwd(out, out_sz))
        return;
    snprintf(out, out_sz, ".");
}

static void resolve_python_bin(const char *exec_dir, char *out, size_t out_sz) {
    snprintf(out, out_sz, "%s/venv/bin/python", exec_dir);
    if (access(out, X_OK) == 0)
        return;
    snprintf(out, out_sz, "python3");
}

static int resolve_tokenizer_helper(const char *exec_dir, char *out, size_t out_sz) {
    snprintf(out, out_sz, "%s/hf_tokenizer_bridge.py", exec_dir);
    return access(out, R_OK) == 0 ? 0 : -1;
}

static int write_string_to_tempfile(char *tmpl, const char *contents) {
    int fd = mkstemp(tmpl);
    if (fd < 0) {
        perror("mkstemp");
        return -1;
    }
    FILE *f = fdopen(fd, "w");
    if (!f) {
        perror("fdopen");
        close(fd);
        unlink(tmpl);
        return -1;
    }
    if (fputs(contents, f) == EOF) {
        perror("fputs");
        fclose(f);
        unlink(tmpl);
        return -1;
    }
    if (fclose(f) != 0) {
        perror("fclose");
        unlink(tmpl);
        return -1;
    }
    return 0;
}

static int write_tokens_to_tempfile(char *tmpl, const int *tokens, int n) {
    int fd = mkstemp(tmpl);
    if (fd < 0) {
        perror("mkstemp");
        return -1;
    }
    FILE *f = fdopen(fd, "w");
    if (!f) {
        perror("fdopen");
        close(fd);
        unlink(tmpl);
        return -1;
    }
    for (int i = 0; i < n; i++) {
        if (i > 0) fputc(' ', f);
        fprintf(f, "%d", tokens[i]);
    }
    fputc('\n', f);
    if (fclose(f) != 0) {
        perror("fclose");
        unlink(tmpl);
        return -1;
    }
    return 0;
}

static char *read_pipe_all(FILE *pipe) {
    size_t cap = 4096;
    size_t len = 0;
    char *buf = (char *)checked_malloc(cap, "pipe_output");
    while (!feof(pipe)) {
        if (len + 1024 + 1 > cap) {
            cap *= 2;
            char *grown = (char *)realloc(buf, cap);
            if (!grown) {
                free(buf);
                fprintf(stderr, "FATAL: OOM growing pipe buffer\n");
                exit(1);
            }
            buf = grown;
        }
        size_t n = fread(buf + len, 1, 1024, pipe);
        len += n;
        if (ferror(pipe)) {
            free(buf);
            return NULL;
        }
    }
    buf[len] = '\0';
    return buf;
}

static int run_capture_command(const char *cmd, char **output) {
    /* TODO(release hardening): this fallback command is shell-built by callers.
     * Normal model paths work, but embedded quotes can break shell quoting.
     * Replace with fork/exec or posix_spawn before accepting untrusted paths. */
    FILE *pipe = popen(cmd, "r");
    if (!pipe) {
        perror("popen");
        return -1;
    }
    *output = read_pipe_all(pipe);
    int status = pclose(pipe);
    if (status != 0)
        return -1;
    return 0;
}

static int parse_token_list(const char *text, int **tokens_out, int *len_out) {
    char *copy = checked_strdup(text, "token_parse_copy");
    int cap = 128;
    int len = 0;
    int *tokens = (int *)checked_malloc((size_t)cap * sizeof(int), "prompt_tokens");

    char *save = NULL;
    for (char *tok = strtok_r(copy, " \t\r\n,", &save);
         tok;
         tok = strtok_r(NULL, " \t\r\n,", &save)) {
        char *end = NULL;
        long v = strtol(tok, &end, 10);
        if (end == tok || *end != '\0')
            continue;
        if (len == cap) {
            cap *= 2;
            int *grown = (int *)realloc(tokens, (size_t)cap * sizeof(int));
            if (!grown) {
                free(copy);
                free(tokens);
                fprintf(stderr, "FATAL: OOM growing token buffer\n");
                exit(1);
            }
            tokens = grown;
        }
        tokens[len++] = (int)v;
    }

    free(copy);
    if (len == 0) {
        free(tokens);
        return -1;
    }

    *tokens_out = tokens;
    *len_out = len;
    return 0;
}

/* Global C-native tokenizer (R6). Loaded lazily on first use. */
static tokenizer_t g_tokenizer;
static int g_tokenizer_loaded = 0;
static const char *g_tokenizer_model_dir = NULL;

static int ensure_c_tokenizer(const char *model_dir) {
    if (g_tokenizer_loaded) {
        /* Check if model_dir changed since last load */
        if (g_tokenizer_model_dir && strcmp(g_tokenizer_model_dir, model_dir) != 0) {
            tokenizer_free(&g_tokenizer);
            g_tokenizer_loaded = 0;
            free((void *)g_tokenizer_model_dir);
            g_tokenizer_model_dir = NULL;
        } else {
            return 0;
        }
    }
    tokenizer_init(&g_tokenizer);
    if (tokenizer_load(&g_tokenizer, model_dir) != 0) return -1;
    g_tokenizer_model_dir = strdup(model_dir);
    if (!g_tokenizer_model_dir) {
        tokenizer_free(&g_tokenizer);
        return -1;  /* OOM — fall back to Python bridge */
    }
    g_tokenizer_loaded = 1;
    return 0;
}

static int tokenize_text_prompt(const char *exec_dir, const char *model_dir,
                                 const char *prompt_text,
                                 int **tokens_out, int *len_out) {
    /* Try C-native tokenizer first (R6) */
    if (ensure_c_tokenizer(model_dir) == 0) {
        int *ids = (int *)malloc(4096 * sizeof(int));
        if (!ids) return -1;
        int n = 0;
        if (tokenizer_encode(&g_tokenizer, prompt_text, ids, 4096, &n) == 0 && n > 0) {
            *tokens_out = ids;
            *len_out = n;
            return 0;
        }
        free(ids);
        fprintf(stderr, "C-native tokenizer failed, falling back to Python bridge\n");
    }

    /* Fallback: Python bridge */
    char python_bin[PATH_MAX];
    char helper_path[PATH_MAX];
    char tmp_path[] = "/tmp/llama_prompt_XXXXXX";
    char cmd[PATH_MAX * 3 + 128];
    char *out = NULL;

    if (resolve_tokenizer_helper(exec_dir, helper_path, sizeof(helper_path)) != 0) {
        fprintf(stderr,
                "Tokenizer helper not found: %s/hf_tokenizer_bridge.py\n"
                "  --prompt needs a tokenizer. Either provide a model dir containing\n"
                "  tokenizer.json, or keep hf_tokenizer_bridge.py beside the binary and\n"
                "  install transformers (e.g. `python3 -m venv venv && "
                "venv/bin/pip install transformers`).\n", exec_dir);
        return -1;
    }

    if (write_string_to_tempfile(tmp_path, prompt_text) != 0)
        return -1;

    resolve_python_bin(exec_dir, python_bin, sizeof(python_bin));
    snprintf(cmd, sizeof(cmd), "%s '%s' encode '%s' '%s'",
             python_bin, helper_path, model_dir, tmp_path);

    int rc = run_capture_command(cmd, &out);
    unlink(tmp_path);
    if (rc != 0 || !out) {
        free(out);
        fprintf(stderr,
                "Failed to tokenize prompt via %s.\n"
                "  Ensure that interpreter can import transformers "
                "(`%s -c 'import transformers'`),\n"
                "  or pass a model dir containing tokenizer.json for the C-native tokenizer.\n",
                python_bin, python_bin);
        return -1;
    }

    rc = parse_token_list(out, tokens_out, len_out);
    free(out);
    if (rc != 0) {
        fprintf(stderr, "Tokenizer helper returned no token IDs\n");
        return -1;
    }
    return 0;
}

static char *decode_generated_tokens(const char *exec_dir, const char *model_dir,
                                      const int *tokens, int n) {
    if (n <= 0)
        return checked_strdup("", "decoded_empty");

    /* Try C-native tokenizer first (R6) */
    if (ensure_c_tokenizer(model_dir) == 0) {
        char *buf = (char *)malloc(n * TOK_MAX_TOKEN_LEN + 16);
        if (buf) {
            int len = tokenizer_decode(&g_tokenizer, tokens, n, buf, n * TOK_MAX_TOKEN_LEN);
            if (len >= 0) return buf;
            free(buf);
        }
    }

    /* Fallback: Python bridge */
    char python_bin[PATH_MAX];
    char helper_path[PATH_MAX];
    char tmp_path[] = "/tmp/llama_tokens_XXXXXX";
    char cmd[PATH_MAX * 3 + 128];
    char *out = NULL;

    if (resolve_tokenizer_helper(exec_dir, helper_path, sizeof(helper_path)) != 0)
        return NULL;
    if (write_tokens_to_tempfile(tmp_path, tokens, n) != 0)
        return NULL;

    resolve_python_bin(exec_dir, python_bin, sizeof(python_bin));
    snprintf(cmd, sizeof(cmd), "%s '%s' decode '%s' '%s'",
             python_bin, helper_path, model_dir, tmp_path);

    int rc = run_capture_command(cmd, &out);
    unlink(tmp_path);
    if (rc != 0 || !out) {
        free(out);
        return NULL;
    }

    size_t len = strlen(out);
    while (len > 0 && (out[len - 1] == '\n' || out[len - 1] == '\r'))
        out[--len] = '\0';
    return out;
}

/* ================================================================== */
/*  Model Configuration                                                  */
/* ================================================================== */

typedef struct {
    int hidden_dim;       /* hidden_size from config.json          */
    int num_heads;        /* num_attention_heads (Q heads)         */
    int num_kv_heads;     /* num_key_value_heads (K/V heads, GQA) */
    int head_dim;         /* hidden_dim / num_heads (derived)      */
    int num_layers;       /* num_hidden_layers                     */
    int intermediate_dim; /* intermediate_size                     */
    int vocab_size;       /* vocab_size                            */
    int max_seq_len;      /* max_position_embeddings               */
    int rope_theta;       /* RoPE base frequency (default 10000)   */
} llama_config_t;

/* Fallback for models without config.json (legacy 7B compatibility) */
static const llama_config_t LLAMA_7B_CONFIG = {
    .hidden_dim       = 4096,
    .num_heads        = 32,
    .num_kv_heads     = 32,
    .head_dim         = 128,
    .num_layers       = 32,
    .intermediate_dim = 11008,
    .vocab_size       = 32000,
    .max_seq_len      = 2048,
    .rope_theta       = 10000,
};

/* ================================================================== */
/*  config.json Parser                                                   */
/* ================================================================== */

/* Extract an integer value for a given key from JSON text.
 * Handles both integer literals (32) and float-like (10000.0) by truncation. */
static int json_extract_int(const char *json, const char *key, int default_val) {
    char pattern[256];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *p = strstr(json, pattern);
    if (!p) return default_val;
    p += strlen(pattern);
    while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r' || *p == ':') p++;
    if (*p == 'n') return default_val; /* null */
    char *end;
    long val = strtol(p, &end, 10);
    return (int)val;
}

/* Load model configuration from config.json.
 * Returns 0 on success, -1 if file not found, -2 on validation failure. */
static int load_config_json(const char *model_dir, llama_config_t *cfg) {
    char path[512];
    snprintf(path, sizeof(path), "%s/config.json", model_dir);
    FILE *f = fopen(path, "r");
    if (!f) return -1;

    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    if (len <= 0 || len > 1024 * 1024) { /* sanity: max 1 MiB */
        fclose(f);
        return -1;
    }
    fseek(f, 0, SEEK_SET);

    char *buf = (char *)malloc(len + 1);
    if (!buf) { fclose(f); return -1; }
    size_t nread = fread(buf, 1, len, f);
    fclose(f);
    buf[nread] = '\0';

    /* Verify it's a Llama model */
    if (!strstr(buf, "\"LlamaForCausalLM\"") && !strstr(buf, "\"llama\"")) {
        fprintf(stderr, "WARNING: config.json does not appear to be a Llama model\n");
    }

    cfg->hidden_dim       = json_extract_int(buf, "hidden_size", -1);
    cfg->num_heads        = json_extract_int(buf, "num_attention_heads", -1);
    cfg->num_kv_heads     = json_extract_int(buf, "num_key_value_heads", cfg->num_heads);
    cfg->num_layers       = json_extract_int(buf, "num_hidden_layers", -1);
    cfg->intermediate_dim = json_extract_int(buf, "intermediate_size", -1);
    cfg->vocab_size       = json_extract_int(buf, "vocab_size", -1);
    cfg->max_seq_len      = json_extract_int(buf, "max_position_embeddings", 2048);
    cfg->rope_theta       = json_extract_int(buf, "rope_theta", 10000);

    free(buf);

    /* Validate required fields */
    if (cfg->hidden_dim <= 0 || cfg->num_heads <= 0 || cfg->num_layers <= 0 ||
        cfg->intermediate_dim <= 0 || cfg->vocab_size <= 0) {
        fprintf(stderr, "ERROR: config.json missing required fields\n");
        fprintf(stderr, "  hidden_size=%d num_attention_heads=%d num_hidden_layers=%d\n",
                cfg->hidden_dim, cfg->num_heads, cfg->num_layers);
        fprintf(stderr, "  intermediate_size=%d vocab_size=%d\n",
                cfg->intermediate_dim, cfg->vocab_size);
        return -2;
    }

    /* Derive head_dim */
    if (cfg->hidden_dim % cfg->num_heads != 0) {
        fprintf(stderr, "ERROR: hidden_size (%d) not divisible by num_attention_heads (%d)\n",
                cfg->hidden_dim, cfg->num_heads);
        return -2;
    }
    cfg->head_dim = cfg->hidden_dim / cfg->num_heads;

    /* Validate GQA consistency */
    if (cfg->num_kv_heads <= 0 || cfg->num_heads % cfg->num_kv_heads != 0) {
        fprintf(stderr, "ERROR: num_attention_heads (%d) not divisible by num_key_value_heads (%d)\n",
                cfg->num_heads, cfg->num_kv_heads);
        return -2;
    }

    return 0;
}

static void print_config(const llama_config_t *cfg) {
    int kv_dim = cfg->num_kv_heads * cfg->head_dim;
    int gqa_group = cfg->num_heads / cfg->num_kv_heads;
    printf("  hidden_dim       = %d\n", cfg->hidden_dim);
    printf("  num_heads        = %d (Q heads)\n", cfg->num_heads);
    printf("  num_kv_heads     = %d (KV heads, %d Q per KV = %s)\n",
           cfg->num_kv_heads, gqa_group,
           gqa_group == 1 ? "MHA" : "GQA");
    printf("  head_dim         = %d\n", cfg->head_dim);
    printf("  kv_dim           = %d\n", kv_dim);
    printf("  num_layers       = %d\n", cfg->num_layers);
    printf("  intermediate_dim = %d\n", cfg->intermediate_dim);
    printf("  vocab_size       = %d\n", cfg->vocab_size);
    printf("  max_seq_len      = %d\n", cfg->max_seq_len);
    printf("  rope_theta       = %d\n", cfg->rope_theta);
}

/* ================================================================== */
/*  Tensor Shape Validation (Finding 2)                                 */
/* ================================================================== */

/* Validate that a tensor has the expected shape. Returns 0 on match, -1 on mismatch. */
static int expect_shape_1d(st_tensor_t *t, int64_t d0) {
    if (t->ndim != 1 || t->shape[0] != d0) {
        fprintf(stderr, "Shape mismatch for '%s': expected [%lld], got [",
                t->name, (long long)d0);
        for (int i = 0; i < t->ndim; i++) {
            if (i) fprintf(stderr, ",");
            fprintf(stderr, "%lld", (long long)t->shape[i]);
        }
        fprintf(stderr, "]\n");
        return -1;
    }
    return 0;
}

static int expect_shape_2d(st_tensor_t *t, int64_t d0, int64_t d1) {
    if (t->ndim != 2 || t->shape[0] != d0 || t->shape[1] != d1) {
        fprintf(stderr, "Shape mismatch for '%s': expected [%lld,%lld], got [",
                t->name, (long long)d0, (long long)d1);
        for (int i = 0; i < t->ndim; i++) {
            if (i) fprintf(stderr, ",");
            fprintf(stderr, "%lld", (long long)t->shape[i]);
        }
        fprintf(stderr, "]\n");
        return -1;
    }
    return 0;
}

/* ================================================================== */
/*  RoPE Precomputed Tables                                             */
/* ================================================================== */

typedef struct {
    fixed_t *cos_table;   /* [max_seq_len * head_dim/2] */
    fixed_t *sin_table;   /* [max_seq_len * head_dim/2] */
    int max_seq_len;
    int half_dim;         /* head_dim / 2 */
} rope_table_t;

static void rope_init(rope_table_t *rope, const llama_config_t *cfg) {
    fp_math_init();

    rope->max_seq_len = cfg->max_seq_len;
    rope->half_dim = cfg->head_dim / 2;
    int n = rope->max_seq_len * rope->half_dim;

    rope->cos_table = (fixed_t *)checked_malloc(n * sizeof(fixed_t), "RoPE cos");
    rope->sin_table = (fixed_t *)checked_malloc(n * sizeof(fixed_t), "RoPE sin");

    fixed_t log_theta_base = fp_log(fp_from_int(cfg->rope_theta));

    for (int i = 0; i < rope->half_dim; i++) {
        fixed_t neg_frac = -fp_div(fp_from_int(2 * i), fp_from_int(cfg->head_dim));
        fixed_t theta_i = fp_exp(fp_mul(neg_frac, log_theta_base));

        for (int pos = 0; pos < rope->max_seq_len; pos++) {
            fixed_t angle = fp_mul(fp_from_int(pos), theta_i);
            fixed_t cos_val, sin_val;
            fp_sincos(angle, &cos_val, &sin_val);

            int idx = pos * rope->half_dim + i;
            rope->cos_table[idx] = cos_val;
            rope->sin_table[idx] = sin_val;
        }
    }
}

static void rope_free(rope_table_t *rope) {
    free(rope->cos_table);
    free(rope->sin_table);
}

/* Apply RoPE rotation to a single head's vector.
 * Uses half-split pairing: pairs (vec[i], vec[i+half]) for i = 0..half-1.
 * This matches HuggingFace's rotate_half convention used by all
 * HuggingFace-distributed Llama models (TinyLlama, Llama-2, etc.). */
static void rope_apply(const rope_table_t *rope, fixed_t *vec, int pos) {
    int half = rope->half_dim;
    int base = pos * half;
    for (int i = 0; i < half; i++) {
        fixed_t cos_val = rope->cos_table[base + i];
        fixed_t sin_val = rope->sin_table[base + i];
        fixed_t x0 = vec[i];
        fixed_t x1 = vec[i + half];
        vec[i]        = fp_mul(x0, cos_val) - fp_mul(x1, sin_val);
        vec[i + half] = fp_mul(x1, cos_val) + fp_mul(x0, sin_val);
    }
}

/* ================================================================== */
/*  Integer Matrix Operations                                           */
/* ================================================================== */

static void matmul_vec(fixed_t *out, const int64_t *weight,
                        const fixed_t *in, int M, int N) {
    /* Process 4 rows at a time: each in[j] load is shared across 4 accumulators,
     * reducing input-vector memory traffic by ~4x (R7).
     * Tested: 2-row (+7%), 4-row (+13% — current winner), 8-row (-8.6% — register spills). */
    int i = 0;
    for (; i + 3 < M; i += 4) {
        int128_t acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
        const int64_t *r0 = weight + (int64_t)(i + 0) * N;
        const int64_t *r1 = weight + (int64_t)(i + 1) * N;
        const int64_t *r2 = weight + (int64_t)(i + 2) * N;
        const int64_t *r3 = weight + (int64_t)(i + 3) * N;
        for (int j = 0; j < N; j++) {
            int128_t v = (int128_t)in[j];
            acc0 += v * r0[j];
            acc1 += v * r1[j];
            acc2 += v * r2[j];
            acc3 += v * r3[j];
        }
        out[i + 0] = (fixed_t)(acc0 >> FP_PRECISION);
        out[i + 1] = (fixed_t)(acc1 >> FP_PRECISION);
        out[i + 2] = (fixed_t)(acc2 >> FP_PRECISION);
        out[i + 3] = (fixed_t)(acc3 >> FP_PRECISION);
    }
    /* Handle remaining rows */
    for (; i < M; i++) {
        int128_t acc = 0;
        const int64_t *row = weight + (int64_t)i * N;
        for (int j = 0; j < N; j++)
            acc += (int128_t)row[j] * in[j];
        out[i] = (fixed_t)(acc >> FP_PRECISION);
    }
}

static void rmsnorm(fixed_t *out, const fixed_t *x, const int64_t *weight, int dim) {
    int128_t sum_sq = 0;
    for (int i = 0; i < dim; i++)
        sum_sq += (int128_t)x[i] * x[i];
    fixed_t mean_sq = (fixed_t)((sum_sq >> FP_PRECISION) / dim);
    fixed_t eps = (fixed_t)2814749767LL; /* 1e-5 in Q16.48 */
    mean_sq += eps;
    fixed_t inv_rms = fp_inv_sqrt(mean_sq);
    for (int i = 0; i < dim; i++)
        out[i] = fp_mul(fp_mul(x[i], inv_rms), weight[i]);
}

static void softmax(fixed_t *x, int n) {
    fixed_t max_val = x[0];
    for (int i = 1; i < n; i++)
        if (x[i] > max_val) max_val = x[i];
    fixed_t sum = 0;
    for (int i = 0; i < n; i++) {
        x[i] = fp_safe_exp(x[i] - max_val);
        sum += x[i];
    }
    if (sum > 0)
        for (int i = 0; i < n; i++)
            x[i] = fp_div(x[i], sum);
}

/* ================================================================== */
/*  KV-Cache                                                            */
/* ================================================================== */

typedef struct {
    fixed_t *key_cache;    /* [num_layers][max_seq_len][num_kv_heads][head_dim] */
    fixed_t *value_cache;

    /* Packed KV cache (R9C-B4): NULL unless --kv-packed is active */
    uint8_t  *pk_key_codes;   /* [num_layers][max_seq_len][num_kv_heads][head_dim] uint8 */
    uint16_t *pk_val_codes;   /* [num_layers][max_seq_len][num_kv_heads][head_dim] uint16 */
    int64_t  *pk_key_mag;     /* [num_layers][max_seq_len][num_kv_heads] — per-vector L-inf */
    int64_t  *pk_val_mag;     /* same layout */
    int       pk_k_bits;      /* 0 = not packed */
    int       pk_v_bits;

    int max_seq_len;
    int num_kv_heads;      /* KV heads (may differ from Q heads in GQA) */
    int head_dim;
    int num_layers;
    int cur_len;           /* current sequence length (for reset) */
} kv_cache_t;

static int kv_cache_init(kv_cache_t *kv, const llama_config_t *cfg) {
    kv->max_seq_len = cfg->max_seq_len;
    kv->num_kv_heads = cfg->num_kv_heads;
    kv->head_dim = cfg->head_dim;
    kv->num_layers = cfg->num_layers;
    kv->cur_len = 0;

    size_t per_layer = (size_t)cfg->max_seq_len * cfg->num_kv_heads * cfg->head_dim;
    size_t total_elems = per_layer * cfg->num_layers;
    size_t total_bytes = total_elems * sizeof(fixed_t);

    /* In packed mode, skip exact cache — packed arrays are the real storage */
    if (g_kv_quant.packed) {
        kv->key_cache = NULL;
        kv->value_cache = NULL;
        printf("KV-cache: packed mode — skipping exact cache allocation (%.1f MiB saved)\n",
               2.0 * total_bytes / (1024.0 * 1024.0));
    } else {
        printf("KV-cache: allocating %.1f MiB (%zu elements × 2)...\n",
               2.0 * total_bytes / (1024.0 * 1024.0), total_elems);
        kv->key_cache = (fixed_t *)calloc(total_elems, sizeof(fixed_t));
        kv->value_cache = (fixed_t *)calloc(total_elems, sizeof(fixed_t));
        if (!kv->key_cache || !kv->value_cache) {
            fprintf(stderr, "FATAL: OOM allocating KV-cache (%.1f MiB needed)\n",
                    2.0 * total_bytes / (1024.0 * 1024.0));
            free(kv->key_cache);
            free(kv->value_cache);
            kv->key_cache = kv->value_cache = NULL;
            return -1;
        }
    }

    /* Packed KV cache (R9C-B4): allocate narrow-code + magnitude arrays */
    kv->pk_key_codes = NULL;
    kv->pk_val_codes = NULL;
    kv->pk_key_mag = NULL;
    kv->pk_val_mag = NULL;
    kv->pk_k_bits = 0;
    kv->pk_v_bits = 0;
    size_t pk_extra = 0;
    if (g_kv_quant.packed) {
        kv->pk_k_bits = g_kv_quant.k_bits;
        kv->pk_v_bits = g_kv_quant.v_bits;
        /* Code arrays: 1 byte per K element, 2 bytes per V element */
        kv->pk_key_codes = (uint8_t *)calloc(total_elems, sizeof(uint8_t));
        kv->pk_val_codes = (uint16_t *)calloc(total_elems, sizeof(uint16_t));
        /* Magnitude arrays: 1 int64_t per vector (per layer×pos×head) */
        size_t mag_elems = (size_t)cfg->num_layers * cfg->max_seq_len * cfg->num_kv_heads;
        kv->pk_key_mag = (int64_t *)calloc(mag_elems, sizeof(int64_t));
        kv->pk_val_mag = (int64_t *)calloc(mag_elems, sizeof(int64_t));
        if (!kv->pk_key_codes || !kv->pk_val_codes || !kv->pk_key_mag || !kv->pk_val_mag) {
            fprintf(stderr, "FATAL: OOM allocating packed KV cache\n");
            free(kv->pk_key_codes); free(kv->pk_val_codes);
            free(kv->pk_key_mag); free(kv->pk_val_mag);
            kv->pk_key_codes = NULL; kv->pk_val_codes = NULL;
            kv->pk_key_mag = NULL; kv->pk_val_mag = NULL;
            kv->pk_k_bits = 0; kv->pk_v_bits = 0;
            return -1;  /* exact cache was not allocated in packed mode */
        } else {
            pk_extra = total_elems * 1 + total_elems * 2 + mag_elems * 8 * 2;
            printf("KV-cache: packed K%dV%d allocated (%.1f MiB codes + %.1f MiB magnitudes)\n",
                   kv->pk_k_bits, kv->pk_v_bits,
                   (total_elems * 1 + total_elems * 2) / (1024.0 * 1024.0),
                   (mag_elems * 8 * 2) / (1024.0 * 1024.0));
        }
    }

    printf("KV-cache: %.1f MiB exact (+ %.2f MiB packed)\n",
           2.0 * total_bytes / (1024.0 * 1024.0),
           (double)pk_extra / (1024.0 * 1024.0));
    return 0;
}

static void kv_cache_reset(kv_cache_t *kv) {
    size_t per_layer = (size_t)kv->max_seq_len * kv->num_kv_heads * kv->head_dim;
    size_t total = per_layer * kv->num_layers;
    if (kv->key_cache) memset(kv->key_cache, 0, total * sizeof(fixed_t));
    if (kv->value_cache) memset(kv->value_cache, 0, total * sizeof(fixed_t));
    if (kv->pk_key_codes) {
        memset(kv->pk_key_codes, 0, total * sizeof(uint8_t));
        memset(kv->pk_val_codes, 0, total * sizeof(uint16_t));
        size_t mag_total = (size_t)kv->num_layers * kv->max_seq_len * kv->num_kv_heads;
        memset(kv->pk_key_mag, 0, mag_total * sizeof(int64_t));
        memset(kv->pk_val_mag, 0, mag_total * sizeof(int64_t));
    }
    kv->cur_len = 0;
}

static fixed_t *kv_key_at(kv_cache_t *kv, int layer, int pos) {
    size_t offset = ((size_t)layer * kv->max_seq_len + pos) *
                    kv->num_kv_heads * kv->head_dim;
    return kv->key_cache + offset;
}

static fixed_t *kv_value_at(kv_cache_t *kv, int layer, int pos) {
    size_t offset = ((size_t)layer * kv->max_seq_len + pos) *
                    kv->num_kv_heads * kv->head_dim;
    return kv->value_cache + offset;
}

/* Packed KV: quantize one head's K or V vector into packed storage.
 * src: exact Q16.48 vector [head_dim], b: bit width, Q: 1<<b.
 * Writes codes to dst_codes and magnitude to *dst_mag. */
static void kv_pack_vector(const fixed_t *src, int D, int b,
                            uint8_t *dst_k_codes, uint16_t *dst_v_codes,
                            int64_t *dst_mag, int is_v) {
    uint64_t Q = (uint64_t)1 << b;
    int64_t M = 0;
    for (int i = 0; i < D; i++) {
        int64_t a = src[i] < 0 ? -src[i] : src[i];
        if (a > M) M = a;
    }
    *dst_mag = M;
    if (M == 0) {
        if (is_v) { for (int i = 0; i < D; i++) dst_v_codes[i] = 0; }
        else      { for (int i = 0; i < D; i++) dst_k_codes[i] = 0; }
        return;
    }
    for (int i = 0; i < D; i++) {
        __int128 q = ((__int128)(src[i] + M) * (int64_t)Q) / (2 * M);
        if (q < 0) q = 0;
        if (q >= (int64_t)Q) q = (int64_t)Q - 1;
        if (is_v) dst_v_codes[i] = (uint16_t)q;
        else      dst_k_codes[i] = (uint8_t)q;
    }
}

/* Packed KV: dequantize one head's K or V vector from packed storage into dst.
 * dst: output Q16.48 vector [head_dim]. */
static void kv_unpack_vector(fixed_t *dst, int D, int b,
                              const uint8_t *src_k_codes, const uint16_t *src_v_codes,
                              int64_t mag, int is_v) {
    if (mag == 0) { for (int i = 0; i < D; i++) dst[i] = 0; return; }
    uint64_t Q = (uint64_t)1 << b;
    for (int i = 0; i < D; i++) {
        int64_t q = is_v ? src_v_codes[i] : src_k_codes[i];
        dst[i] = (fixed_t)(((__int128)(2 * q + 1) * mag) / (int64_t)Q) - mag;
    }
}

static void kv_cache_free(kv_cache_t *kv) {
    free(kv->key_cache);
    free(kv->value_cache);
    free(kv->pk_key_codes);
    free(kv->pk_val_codes);
    free(kv->pk_key_mag);
    free(kv->pk_val_mag);
}

/* ================================================================== */
/*  Per-Layer Weight Streaming                                          */
/* ================================================================== */

typedef struct {
    int64_t *input_layernorm;   /* [hidden_dim] */
    int64_t *q_proj;            /* [hidden_dim][hidden_dim] */
    int64_t *k_proj;            /* [kv_dim][hidden_dim]  (kv_dim = num_kv_heads * head_dim) */
    int64_t *v_proj;            /* [kv_dim][hidden_dim] */
    int64_t *o_proj;            /* [hidden_dim][hidden_dim] */
    int64_t *post_attn_norm;    /* [hidden_dim] */
    int64_t *gate_proj;         /* [intermediate_dim][hidden_dim] */
    int64_t *up_proj;           /* [intermediate_dim][hidden_dim] */
    int64_t *down_proj;         /* [hidden_dim][intermediate_dim] */
} layer_weights_t;

/* Load and shape-validate weights for one transformer layer.
 * Returns 0 on success, -1 on missing/misshapen/OOM tensor. */
static int load_layer_weights(st_model_t *st, int layer_idx,
                               layer_weights_t *w, const llama_config_t *cfg) {
    char name[256];
    st_tensor_t *t;
    int dim = cfg->hidden_dim;
    int kv_dim = cfg->num_kv_heads * cfg->head_dim;
    int inter = cfg->intermediate_dim;

    #define LOAD_1D(field, fmt, expected_d0) do { \
        snprintf(name, sizeof(name), fmt, layer_idx); \
        t = st_model_find(st, name); \
        if (!t) { fprintf(stderr, "Missing tensor: %s\n", name); return -1; } \
        if (expect_shape_1d(t, expected_d0) != 0) return -1; \
        w->field = st_tensor_to_q1648(st, t); \
        if (!w->field) return -1; \
    } while(0)

    #define LOAD_2D(field, fmt, d0, d1) do { \
        snprintf(name, sizeof(name), fmt, layer_idx); \
        t = st_model_find(st, name); \
        if (!t) { fprintf(stderr, "Missing tensor: %s\n", name); return -1; } \
        if (expect_shape_2d(t, d0, d1) != 0) return -1; \
        w->field = st_tensor_to_q1648(st, t); \
        if (!w->field) return -1; \
    } while(0)

    LOAD_1D(input_layernorm, "model.layers.%d.input_layernorm.weight", dim);
    LOAD_2D(q_proj,          "model.layers.%d.self_attn.q_proj.weight", dim, dim);
    LOAD_2D(k_proj,          "model.layers.%d.self_attn.k_proj.weight", kv_dim, dim);
    LOAD_2D(v_proj,          "model.layers.%d.self_attn.v_proj.weight", kv_dim, dim);
    LOAD_2D(o_proj,          "model.layers.%d.self_attn.o_proj.weight", dim, dim);
    LOAD_1D(post_attn_norm,  "model.layers.%d.post_attention_layernorm.weight", dim);
    LOAD_2D(gate_proj,       "model.layers.%d.mlp.gate_proj.weight", inter, dim);
    LOAD_2D(up_proj,         "model.layers.%d.mlp.up_proj.weight",   inter, dim);
    LOAD_2D(down_proj,       "model.layers.%d.mlp.down_proj.weight", dim, inter);

    #undef LOAD_1D
    #undef LOAD_2D
    return 0;
}

static void free_layer_weights(layer_weights_t *w) {
    free(w->input_layernorm);
    free(w->q_proj);
    free(w->k_proj);
    free(w->v_proj);
    free(w->o_proj);
    free(w->post_attn_norm);
    free(w->gate_proj);
    free(w->up_proj);
    free(w->down_proj);
    memset(w, 0, sizeof(*w));
}

/* ================================================================== */
/*  Single Transformer Layer Forward Pass                               */
/* ================================================================== */

/* File-scoped scratch pointer — set by llama_forward before layer loop,
 * cleared after. NULL means transformer_layer_forward uses per-call malloc. */
static fixed_t *g_layer_scratch = NULL;

static void transformer_layer_forward(
    fixed_t *hidden,           /* [hidden_dim] — input/output */
    const layer_weights_t *w,
    const llama_config_t *cfg,
    const rope_table_t *rope,
    kv_cache_t *kv,
    int layer_idx,
    int pos
) {
    int dim = cfg->hidden_dim;
    int nh = cfg->num_heads;
    int nkv = cfg->num_kv_heads;
    int hd = cfg->head_dim;
    int kv_dim = nkv * hd;
    int inter = cfg->intermediate_dim;
    int gqa_group = nh / nkv;  /* Q heads per KV head (1 = MHA, >1 = GQA) */

    /* Use pre-allocated scratch (R7) if available, else malloc per call.
     * g_layer_scratch is set by llama_forward before the layer loop. */
    fixed_t *normed, *q, *k, *v, *attn_out, *o_out;
    fixed_t *gate, *up, *mlp_out, *normed2, *silu_gate, *gate_up;
    int _scratch_local = 0;

    if (g_layer_scratch) {
        fixed_t *_p = g_layer_scratch;
        normed    = _p; _p += dim;
        q         = _p; _p += dim;
        k         = _p; _p += kv_dim;
        v         = _p; _p += kv_dim;
        attn_out  = _p; _p += dim;
        o_out     = _p; _p += dim;
        gate      = _p; _p += inter;
        up        = _p; _p += inter;
        mlp_out   = _p; _p += dim;
        normed2   = _p; _p += dim;
        silu_gate = _p; _p += inter;
        gate_up   = _p;
    } else {
        normed    = (fixed_t *)checked_malloc(dim * sizeof(fixed_t), "normed");
        q         = (fixed_t *)checked_malloc(dim * sizeof(fixed_t), "q");
        k         = (fixed_t *)checked_malloc(kv_dim * sizeof(fixed_t), "k");
        v         = (fixed_t *)checked_malloc(kv_dim * sizeof(fixed_t), "v");
        attn_out  = (fixed_t *)checked_malloc(dim * sizeof(fixed_t), "attn_out");
        o_out     = (fixed_t *)checked_malloc(dim * sizeof(fixed_t), "o_out");
        gate      = (fixed_t *)checked_malloc(inter * sizeof(fixed_t), "gate");
        up        = (fixed_t *)checked_malloc(inter * sizeof(fixed_t), "up");
        mlp_out   = (fixed_t *)checked_malloc(dim * sizeof(fixed_t), "mlp_out");
        normed2   = (fixed_t *)checked_malloc(dim * sizeof(fixed_t), "normed2");
        silu_gate = (fixed_t *)checked_malloc(inter * sizeof(fixed_t), "silu_gate");
        gate_up   = (fixed_t *)checked_malloc(inter * sizeof(fixed_t), "gate_up");
        _scratch_local = 1;
    }
    struct timespec pl_t0, pl_t1;
    int pl_enabled = g_profile.enabled && g_profile.t_per_layer_attn;

    if (pl_enabled) clock_gettime(CLOCK_MONOTONIC, &pl_t0);

    /* 1. RMSNorm */
    rmsnorm(normed, hidden, w->input_layernorm, dim);

    /* 2. Q/K/V projections (K/V may be smaller with GQA) */
    matmul_vec(q, w->q_proj, normed, dim, dim);
    matmul_vec(k, w->k_proj, normed, kv_dim, dim);
    matmul_vec(v, w->v_proj, normed, kv_dim, dim);

    /* 3. RoPE — apply to all Q heads, but only num_kv_heads K heads */
    for (int h = 0; h < nh; h++)
        rope_apply(rope, q + h * hd, pos);
    for (int h = 0; h < nkv; h++)
        rope_apply(rope, k + h * hd, pos);

    /* Store K/V in cache */
    if (kv->pk_key_codes) {
        /* Packed path (R9C-B4): quantize directly into narrow storage */
        size_t elem_off = ((size_t)layer_idx * kv->max_seq_len + pos) *
                          kv->num_kv_heads * kv->head_dim;
        size_t mag_off  = ((size_t)layer_idx * kv->max_seq_len + pos) *
                          kv->num_kv_heads;
        for (int h = 0; h < nkv; h++) {
            kv_pack_vector(k + h * hd, hd, kv->pk_k_bits,
                           kv->pk_key_codes + elem_off + h * hd, NULL,
                           kv->pk_key_mag + mag_off + h, 0);
            kv_pack_vector(v + h * hd, hd, kv->pk_v_bits,
                           NULL, kv->pk_val_codes + elem_off + h * hd,
                           kv->pk_val_mag + mag_off + h, 1);
        }
        /* No exact shadow write — packed arrays are the sole KV storage */
    } else {
        /* Exact path (default) */
        memcpy(kv_key_at(kv, layer_idx, pos), k, kv_dim * sizeof(fixed_t));
        memcpy(kv_value_at(kv, layer_idx, pos), v, kv_dim * sizeof(fixed_t));

        /* Runtime KV quantize-in-place (R9C-B3): lossy compress each head's
         * K/V vector immediately after writing to the cache. */
        if (g_kv_quant.enabled && !g_kv_quant.packed) {
            fixed_t *k_cached = kv_key_at(kv, layer_idx, pos);
            fixed_t *v_cached = kv_value_at(kv, layer_idx, pos);
            for (int h = 0; h < nkv; h++) {
                if (g_kv_quant.k_bits > 0)
                    kv_quantize_inplace(k_cached + h * hd, hd, g_kv_quant.k_bits);
                if (g_kv_quant.v_bits > 0)
                    kv_quantize_inplace(v_cached + h * hd, hd, g_kv_quant.v_bits);
            }
        }
    }

    /* 4. Multi-Head Attention with GQA */
    fixed_t scale = fp_inv_sqrt(fp_from_int(hd));

    if (kv->pk_key_codes) {
        /* ---- Packed attention path with GQA reuse (R9C-B4 + R7) ----
         * Iterate by KV-head group. Each K/V vector is unpacked once
         * and reused by all Q heads sharing that KV head (8:1 in GQA).
         * At seq_len=200+, this eliminates ~87% of dequantize work. */
        fixed_t *k_scratch = (fixed_t *)checked_malloc(hd * sizeof(fixed_t), "pk_k_scratch");
        fixed_t *v_scratch = (fixed_t *)checked_malloc(hd * sizeof(fixed_t), "pk_v_scratch");
        int seq_len = pos + 1;

        /* Pre-allocate scores for all Q heads in one GQA group */
        fixed_t **group_scores = (fixed_t **)checked_malloc(
            gqa_group * sizeof(fixed_t *), "gqa_scores_ptrs");
        for (int g = 0; g < gqa_group; g++)
            group_scores[g] = (fixed_t *)checked_malloc(
                seq_len * sizeof(fixed_t), "gqa_scores");

        for (int kv_h = 0; kv_h < nkv; kv_h++) {
            int h_start = kv_h * gqa_group;

            /* K pass: unpack each K once, compute dots for all Q heads */
            for (int t = 0; t < seq_len; t++) {
                size_t elem_off = ((size_t)layer_idx * kv->max_seq_len + t) *
                                  kv->num_kv_heads * hd + kv_h * hd;
                size_t mag_off  = ((size_t)layer_idx * kv->max_seq_len + t) *
                                  kv->num_kv_heads + kv_h;
                kv_unpack_vector(k_scratch, hd, kv->pk_k_bits,
                                 kv->pk_key_codes + elem_off, NULL,
                                 kv->pk_key_mag[mag_off], 0);
                for (int g = 0; g < gqa_group; g++) {
                    fixed_t *q_h = q + (h_start + g) * hd;
                    int128_t dot = 0;
                    for (int d = 0; d < hd; d++)
                        dot += (int128_t)q_h[d] * k_scratch[d];
                    group_scores[g][t] = fp_mul((fixed_t)(dot >> FP_PRECISION), scale);
                }
            }

            for (int g = 0; g < gqa_group; g++)
                softmax(group_scores[g], seq_len);

            /* V pass: unpack each V once, accumulate for all Q heads */
            for (int g = 0; g < gqa_group; g++) {
                fixed_t *attn_h = attn_out + (h_start + g) * hd;
                memset(attn_h, 0, hd * sizeof(fixed_t));
            }
            for (int t = 0; t < seq_len; t++) {
                size_t elem_off = ((size_t)layer_idx * kv->max_seq_len + t) *
                                  kv->num_kv_heads * hd + kv_h * hd;
                size_t mag_off  = ((size_t)layer_idx * kv->max_seq_len + t) *
                                  kv->num_kv_heads + kv_h;
                kv_unpack_vector(v_scratch, hd, kv->pk_v_bits,
                                 NULL, kv->pk_val_codes + elem_off,
                                 kv->pk_val_mag[mag_off], 1);
                for (int g = 0; g < gqa_group; g++) {
                    fixed_t *attn_h = attn_out + (h_start + g) * hd;
                    fixed_t s = group_scores[g][t];
                    for (int d = 0; d < hd; d++)
                        attn_h[d] += fp_mul(s, v_scratch[d]);
                }
            }
        }

        for (int g = 0; g < gqa_group; g++)
            free(group_scores[g]);
        free(group_scores);
        free(k_scratch);
        free(v_scratch);
    } else {
        /* ---- Exact attention path (unchanged) ---- */
        for (int h = 0; h < nh; h++) {
            int kv_h = h / gqa_group;
            fixed_t *q_h = q + h * hd;
            int seq_len = pos + 1;
            fixed_t *scores = (fixed_t *)checked_malloc(seq_len * sizeof(fixed_t), "attn_scores");

            for (int t = 0; t < seq_len; t++) {
                fixed_t *k_cached = kv_key_at(kv, layer_idx, t) + kv_h * hd;
                int128_t dot = 0;
                for (int d = 0; d < hd; d++)
                    dot += (int128_t)q_h[d] * k_cached[d];
                scores[t] = fp_mul((fixed_t)(dot >> FP_PRECISION), scale);
            }

            softmax(scores, seq_len);

            fixed_t *attn_h = attn_out + h * hd;
            memset(attn_h, 0, hd * sizeof(fixed_t));
            for (int t = 0; t < seq_len; t++) {
                fixed_t *v_cached = kv_value_at(kv, layer_idx, t) + kv_h * hd;
                for (int d = 0; d < hd; d++)
                    attn_h[d] += fp_mul(scores[t], v_cached[d]);
            }
            free(scores);
        }
    }

    /* 5. Output projection */
    matmul_vec(o_out, w->o_proj, attn_out, dim, dim);

    /* 6. Residual add */
    for (int i = 0; i < dim; i++) hidden[i] += o_out[i];

    if (pl_enabled) {
        clock_gettime(CLOCK_MONOTONIC, &pl_t1);
        g_profile.t_per_layer_attn[layer_idx] += ts_diff(pl_t0, pl_t1);
        clock_gettime(CLOCK_MONOTONIC, &pl_t0);
    }

    /* 7. Post-attention RMSNorm */
    rmsnorm(normed2, hidden, w->post_attn_norm, dim);

    /* 8. SwiGLU MLP */
    matmul_vec(gate, w->gate_proj, normed2, inter, dim);
    matmul_vec(up, w->up_proj, normed2, inter, dim);
    for (int i = 0; i < inter; i++)
        silu_gate[i] = fp_silu(gate[i]);
    for (int i = 0; i < inter; i++)
        gate_up[i] = fp_mul(silu_gate[i], up[i]);
    matmul_vec(mlp_out, w->down_proj, gate_up, dim, inter);

    /* 9. Residual add */
    for (int i = 0; i < dim; i++) hidden[i] += mlp_out[i];

    if (pl_enabled) {
        clock_gettime(CLOCK_MONOTONIC, &pl_t1);
        g_profile.t_per_layer_mlp[layer_idx] += ts_diff(pl_t0, pl_t1);
    }

    if (_scratch_local) {
        free(normed); free(q); free(k); free(v);
        free(attn_out); free(o_out);
        free(gate); free(up); free(mlp_out);
        free(normed2); free(silu_gate); free(gate_up);
    }
}

/* ================================================================== */
/*  Model                                                               */
/* ================================================================== */

typedef struct {
    st_model_t      st_model;
    llama_config_t  cfg;
    rope_table_t    rope;
    kv_cache_t      kv;
    int64_t        *embed_tokens;    /* [vocab_size][hidden_dim] */
    int64_t        *final_norm;      /* [hidden_dim] */
    int64_t        *lm_head;         /* [vocab_size][hidden_dim] */
    int             lm_head_tied;    /* 1 if lm_head == embed_tokens */

    /* Cached layer weights (NULL in streaming mode) */
    layer_weights_t *cached_layers;  /* [num_layers] or NULL */
    int              layers_cached;

    /* Native weight format (R9) — mmap backing */
    void            *native_mmap;       /* mmap base (NULL if safetensors path) */
    size_t           native_mmap_size;  /* mmap region size */
    int              native_loaded;     /* 1 = weights backed by mmap, don't free */

    /* Native streaming (R9B) — read layers on demand from .mgw fd */
    int              native_stream;     /* 1 = native-stream mode active */
    int              native_stream_fd;  /* open fd for .mgw file (-1 if unused) */
    uint32_t         ns_num_tensors;    /* tensor count from header */
    mgw_index_entry_t *ns_index;        /* malloc'd copy of tensor index */
    /* Pre-allocated layer scratch buffers (R7) — avoids malloc/free per forward call */
    fixed_t         *scratch_buf;       /* single allocation for all 12 per-layer buffers */
    size_t           scratch_size;      /* total byte count */
} llama_model_t;

/* Allocate pre-allocated scratch workspace for transformer_layer_forward (R7).
 * Layout: [normed | q | k | v | attn_out | o_out | gate | up | mlp_out | normed2 | silu_gate | gate_up]
 * All 12 buffers in one contiguous allocation. */
static int llama_init_scratch(llama_model_t *m) {
    const llama_config_t *c = &m->cfg;
    int dim = c->hidden_dim;
    int kv_dim = c->num_kv_heads * c->head_dim;
    int inter = c->intermediate_dim;
    /* 4 buffers of dim, 2 of kv_dim, 6 of inter (or dim) */
    size_t sz = (size_t)(
        dim +       /* normed */
        dim +       /* q */
        kv_dim +    /* k */
        kv_dim +    /* v */
        dim +       /* attn_out */
        dim +       /* o_out */
        inter +     /* gate */
        inter +     /* up */
        dim +       /* mlp_out */
        dim +       /* normed2 */
        inter +     /* silu_gate */
        inter       /* gate_up */
    ) * sizeof(fixed_t);
    m->scratch_buf = (fixed_t *)calloc(1, sz);
    if (!m->scratch_buf) {
        fprintf(stderr, "FATAL: OOM allocating layer scratch (%.1f MiB)\n",
                sz / (1024.0 * 1024.0));
        return -1;
    }
    m->scratch_size = sz;
    printf("Layer scratch: %.1f MiB pre-allocated (eliminates per-call malloc)\n",
           sz / (1024.0 * 1024.0));
    return 0;
}

/* Load and shape-validate global weights (Finding 2) */
static int llama_load_global_weights(llama_model_t *m) {
    st_tensor_t *t;
    int dim = m->cfg.hidden_dim;
    int vocab = m->cfg.vocab_size;

    t = st_model_find(&m->st_model, "model.embed_tokens.weight");
    if (!t) { fprintf(stderr, "Missing embed_tokens.weight\n"); return -1; }
    if (expect_shape_2d(t, vocab, dim) != 0) return -1;
    printf("  embed_tokens: converting %lld elements (%.1f GiB)...\n",
           (long long)t->num_elements,
           (double)t->num_elements * 8 / (1024.0 * 1024.0 * 1024.0));
    m->embed_tokens = st_tensor_to_q1648(&m->st_model, t);
    if (!m->embed_tokens) return -1;

    t = st_model_find(&m->st_model, "model.norm.weight");
    if (!t) { fprintf(stderr, "Missing model.norm.weight\n"); return -1; }
    if (expect_shape_1d(t, dim) != 0) return -1;
    m->final_norm = st_tensor_to_q1648(&m->st_model, t);
    if (!m->final_norm) return -1;

    t = st_model_find(&m->st_model, "lm_head.weight");
    if (!t) {
        printf("  lm_head not found, using tied embed_tokens\n");
        m->lm_head = m->embed_tokens;
        m->lm_head_tied = 1;
    } else {
        if (expect_shape_2d(t, vocab, dim) != 0) return -1;
        printf("  lm_head: converting %lld elements (%.1f GiB)...\n",
               (long long)t->num_elements,
               (double)t->num_elements * 8 / (1024.0 * 1024.0 * 1024.0));
        m->lm_head = st_tensor_to_q1648(&m->st_model, t);
        if (!m->lm_head) return -1;
        m->lm_head_tied = 0;
    }

    return 0;
}

/* Pre-cache all layer weights at startup (Finding 4 — avoids per-token reload) */
static int llama_cache_all_layers(llama_model_t *m) {
    int nl = m->cfg.num_layers;
    m->cached_layers = (layer_weights_t *)checked_calloc(nl, sizeof(layer_weights_t),
                                                          "cached_layers");
    printf("Pre-caching all %d layers...\n", nl);
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int i = 0; i < nl; i++) {
        if (load_layer_weights(&m->st_model, i, &m->cached_layers[i], &m->cfg) != 0) {
            fprintf(stderr, "Failed to load layer %d\n", i);
            /* Free already-loaded layers */
            for (int j = 0; j < i; j++)
                free_layer_weights(&m->cached_layers[j]);
            free(m->cached_layers);
            m->cached_layers = NULL;
            return -1;
        }
        if ((i + 1) % 8 == 0 || i == nl - 1)
            printf("  cached layer %d/%d\n", i + 1, nl);
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    printf("All layers cached in %.1f seconds\n",
           (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9);
    m->layers_cached = 1;
    return 0;
}

/* ================================================================== */
/*  Native Weight Export (R9)                                           */
/* ================================================================== */

/*
 * Export all model weights to the native .mgw format.
 * The model must be loaded from safetensors first (global weights resident).
 * Layer weights are streamed: load one layer from safetensors, write to
 * the output file, free, then load the next.  Peak RAM stays low.
 */
static int mgw_export(llama_model_t *m, const char *output_path) {
    const llama_config_t *cfg = &m->cfg;
    int kv_dim = cfg->num_kv_heads * cfg->head_dim;

    /* Count tensors: 3 global (or 2 if tied) + 9 per layer */
    int num_global = m->lm_head_tied ? 2 : 3;
    int num_tensors = num_global + cfg->num_layers * 9;

    /* Build tensor info array */
    typedef struct { char name[64]; uint32_t ndims; uint32_t shape[2]; uint64_t num_elements; } tinfo_t;
    tinfo_t *info = (tinfo_t *)checked_calloc(num_tensors, sizeof(tinfo_t), "mgw_export_info");
    int ti = 0;

    #define ADD_TENSOR(n, d, s0, s1) do { \
        snprintf(info[ti].name, 64, "%s", (n)); \
        info[ti].ndims = (d); info[ti].shape[0] = (s0); info[ti].shape[1] = (s1); \
        info[ti].num_elements = (uint64_t)(s0) * ((d) == 2 ? (uint64_t)(s1) : 1ULL); \
        ti++; \
    } while(0)

    ADD_TENSOR("model.embed_tokens.weight", 2, cfg->vocab_size, cfg->hidden_dim);
    ADD_TENSOR("model.norm.weight",         1, cfg->hidden_dim, 0);
    if (!m->lm_head_tied)
        ADD_TENSOR("lm_head.weight",        2, cfg->vocab_size, cfg->hidden_dim);

    for (int L = 0; L < cfg->num_layers; L++) {
        char buf[64];
        #define ADD_LT(suffix, s0, s1, nd) do { \
            snprintf(buf, 64, "model.layers.%d." suffix, L); \
            ADD_TENSOR(buf, nd, s0, s1); \
        } while(0)
        ADD_LT("input_layernorm.weight",            cfg->hidden_dim, 0, 1);
        ADD_LT("self_attn.q_proj.weight",           cfg->hidden_dim, cfg->hidden_dim, 2);
        ADD_LT("self_attn.k_proj.weight",           kv_dim, cfg->hidden_dim, 2);
        ADD_LT("self_attn.v_proj.weight",           kv_dim, cfg->hidden_dim, 2);
        ADD_LT("self_attn.o_proj.weight",           cfg->hidden_dim, cfg->hidden_dim, 2);
        ADD_LT("post_attention_layernorm.weight",   cfg->hidden_dim, 0, 1);
        ADD_LT("mlp.gate_proj.weight",              cfg->intermediate_dim, cfg->hidden_dim, 2);
        ADD_LT("mlp.up_proj.weight",                cfg->intermediate_dim, cfg->hidden_dim, 2);
        ADD_LT("mlp.down_proj.weight",              cfg->hidden_dim, cfg->intermediate_dim, 2);
        #undef ADD_LT
    }
    #undef ADD_TENSOR

    if (ti != num_tensors) {
        fprintf(stderr, "BUG: tensor count mismatch (%d vs %d)\n", ti, num_tensors);
        free(info); return -1;
    }

    /* Compute layout offsets */
    uint64_t index_off = MGW_HEADER_SIZE + MGW_CONFIG_SIZE;
    uint64_t data_off  = index_off + (uint64_t)num_tensors * MGW_INDEX_ENTRY_SIZE;
    data_off = (data_off + 7) & ~7ULL;  /* 8-byte align */

    uint64_t *tensor_offsets = (uint64_t *)checked_calloc(num_tensors, sizeof(uint64_t), "mgw_offsets");
    uint64_t cur = data_off;
    for (int i = 0; i < num_tensors; i++) {
        tensor_offsets[i] = cur;
        cur += info[i].num_elements * sizeof(int64_t);
    }
    uint64_t total_file_size = cur;

    FILE *f = fopen(output_path, "wb");
    if (!f) {
        fprintf(stderr, "Cannot open %s for writing\n", output_path);
        free(info); free(tensor_offsets); return -1;
    }

    /* --- Write header --- */
    mgw_header_t hdr;
    memset(&hdr, 0, sizeof(hdr));
    memcpy(hdr.magic, MGW_MAGIC, MGW_MAGIC_SIZE);
    hdr.version     = MGW_VERSION;
    hdr.endian_tag  = MGW_ENDIAN_TAG;
    hdr.num_tensors = (uint32_t)num_tensors;
    hdr.index_offset = index_off;
    hdr.data_offset  = data_off;
    if (fwrite(&hdr, sizeof(hdr), 1, f) != 1) goto write_err;

    /* --- Write config --- */
    mgw_config_t mcfg;
    memset(&mcfg, 0, sizeof(mcfg));
    mcfg.hidden_dim       = cfg->hidden_dim;
    mcfg.num_heads        = cfg->num_heads;
    mcfg.num_kv_heads     = cfg->num_kv_heads;
    mcfg.head_dim         = cfg->head_dim;
    mcfg.num_layers       = cfg->num_layers;
    mcfg.intermediate_dim = cfg->intermediate_dim;
    mcfg.vocab_size       = cfg->vocab_size;
    mcfg.max_seq_len      = cfg->max_seq_len;
    mcfg.rope_theta       = cfg->rope_theta;
    mcfg.lm_head_tied     = m->lm_head_tied;
    if (fwrite(&mcfg, sizeof(mcfg), 1, f) != 1) goto write_err;

    /* --- Write index --- */
    for (int i = 0; i < num_tensors; i++) {
        mgw_index_entry_t entry;
        memset(&entry, 0, sizeof(entry));
        strncpy(entry.name, info[i].name, 63);
        entry.num_elements = info[i].num_elements;
        entry.data_offset  = tensor_offsets[i];
        entry.ndims        = info[i].ndims;
        entry.shape[0]     = info[i].shape[0];
        entry.shape[1]     = info[i].shape[1];
        if (fwrite(&entry, sizeof(entry), 1, f) != 1) goto write_err;
    }

    /* Pad to data_offset if needed */
    long pos = ftell(f);
    if (pos < 0) goto write_err;
    while ((uint64_t)pos < data_off) {
        uint8_t zero = 0;
        if (fwrite(&zero, 1, 1, f) != 1) goto write_err;
        pos++;
    }

    /* --- Write tensor data --- */
    printf("Exporting %d tensors to %s...\n", num_tensors, output_path);

    /* Global tensors (already in memory as Q16.48) */
    ti = 0;

    /* embed_tokens */
    printf("  [%d/%d] embed_tokens (%llu elements)\n", ti+1, num_tensors,
           (unsigned long long)info[ti].num_elements);
    if (fwrite(m->embed_tokens, sizeof(int64_t), info[ti].num_elements, f)
        != info[ti].num_elements) goto write_err;
    ti++;

    /* final_norm */
    printf("  [%d/%d] final_norm\n", ti+1, num_tensors);
    if (fwrite(m->final_norm, sizeof(int64_t), info[ti].num_elements, f)
        != info[ti].num_elements) goto write_err;
    ti++;

    /* lm_head (if not tied) */
    if (!m->lm_head_tied) {
        printf("  [%d/%d] lm_head (%llu elements)\n", ti+1, num_tensors,
               (unsigned long long)info[ti].num_elements);
        if (fwrite(m->lm_head, sizeof(int64_t), info[ti].num_elements, f)
            != info[ti].num_elements) goto write_err;
        ti++;
    }

    /* Per-layer: load from safetensors → write → free */
    for (int L = 0; L < cfg->num_layers; L++) {
        layer_weights_t w;
        memset(&w, 0, sizeof(w));
        if (load_layer_weights(&m->st_model, L, &w, cfg) != 0) {
            fprintf(stderr, "Failed to load layer %d for export\n", L);
            free_layer_weights(&w);
            goto write_err;
        }

        int64_t *fields[] = {
            w.input_layernorm, w.q_proj, w.k_proj, w.v_proj, w.o_proj,
            w.post_attn_norm, w.gate_proj, w.up_proj, w.down_proj
        };
        for (int fi = 0; fi < 9; fi++) {
            if (fwrite(fields[fi], sizeof(int64_t), info[ti].num_elements, f)
                != info[ti].num_elements) {
                free_layer_weights(&w);
                goto write_err;
            }
            ti++;
        }
        free_layer_weights(&w);

        if ((L + 1) % 4 == 0 || L == cfg->num_layers - 1)
            printf("  exported layer %d/%d\n", L + 1, cfg->num_layers);
    }

    fclose(f);
    free(info);
    free(tensor_offsets);

    printf("Export complete: %s (%.1f GiB, %d tensors)\n",
           output_path, total_file_size / (1024.0 * 1024.0 * 1024.0), num_tensors);
    return 0;

write_err:
    fprintf(stderr, "Write error during export to %s\n", output_path);
    fclose(f);
    free(info);
    free(tensor_offsets);
    return -1;
}

/* ================================================================== */
/*  Compressed Native Format (R9C-A)                                    */
/* ================================================================== */

/*
 * .mgwc — Compressed Native Weight Format
 *
 * Stores tensor data in the original float16/bfloat16 representation
 * (2 bytes per element) instead of Q16.48 (8 bytes per element).
 * This gives exactly 4:1 compression on tensor data.
 *
 * The format is structurally identical to .mgw but:
 *   - Magic is "MGWC" instead of "MGW\0"
 *   - Index entries use the reserved field as a dtype tag
 *   - Data section stores uint16_t dtype values, not int64_t Q16.48
 *   - Data alignment is 2-byte (not 8-byte)
 *
 * The compression is EXACT relative to the exported source dtype stream:
 * loading .mgwc produces bit-identical Q16.48 values to loading .mgw because
 * both paths apply the same deterministic float16/bfloat16 → Q16.48 conversion.
 *
 * Export path:  safetensors → .mgwc  (--export-native-compressed)
 * Load path:    .mgwc → Q16.48 in memory  (--native-compressed)
 */

#define MGWC_MAGIC       "MGWC"
#define MGWC_VERSION     1

/* Dtype tags stored in mgw_index_entry_t.reserved field */
#define MGWC_DTYPE_F16   1
#define MGWC_DTYPE_BF16  2

/* Export safetensors → .mgwc compressed native format.
 * Writes raw float16/bfloat16 bytes (2B per element) instead of Q16.48 (8B).
 * The safetensors model must already be loaded (shards mmap'd). */
static int mgwc_export(llama_model_t *m, const char *output_path) {
    const llama_config_t *cfg = &m->cfg;
    int kv_dim = cfg->num_kv_heads * cfg->head_dim;

    /* Count tensors: 3 global (or 2 if tied) + 9 per layer */
    int num_global = m->lm_head_tied ? 2 : 3;
    int num_tensors = num_global + cfg->num_layers * 9;

    /* Build tensor info array (name, dims, shape, dtype) */
    typedef struct {
        char name[64]; uint32_t ndims; uint32_t shape[2];
        uint64_t num_elements; uint32_t dtype;
    } tinfo_t;
    tinfo_t *info = (tinfo_t *)checked_calloc(num_tensors, sizeof(tinfo_t), "mgwc_export_info");
    int ti = 0;

    /* Look up tensors in safetensors to get dtype */
    #define ADD_CTENSOR(n, d, s0, s1) do { \
        snprintf(info[ti].name, 64, "%s", (n)); \
        info[ti].ndims = (d); info[ti].shape[0] = (s0); info[ti].shape[1] = (s1); \
        info[ti].num_elements = (uint64_t)(s0) * ((d) == 2 ? (uint64_t)(s1) : 1ULL); \
        st_tensor_t *_st = st_model_find(&m->st_model, (n)); \
        if (!_st) { fprintf(stderr, "mgwc: tensor '%s' not in safetensors\n", (n)); \
                     free(info); return -1; } \
        if (_st->dtype == ST_DTYPE_BF16) { \
            info[ti].dtype = MGWC_DTYPE_BF16; \
        } else if (_st->dtype == ST_DTYPE_F16) { \
            info[ti].dtype = MGWC_DTYPE_F16; \
        } else { \
            fprintf(stderr, "mgwc: unsupported dtype for tensor '%s' (dtype=%d, " \
                    "only F16 and BF16 are supported)\n", (n), (int)_st->dtype); \
            free(info); return -1; \
        } \
        ti++; \
    } while(0)

    ADD_CTENSOR("model.embed_tokens.weight", 2, cfg->vocab_size, cfg->hidden_dim);
    ADD_CTENSOR("model.norm.weight",         1, cfg->hidden_dim, 0);
    if (!m->lm_head_tied)
        ADD_CTENSOR("lm_head.weight",        2, cfg->vocab_size, cfg->hidden_dim);

    for (int L = 0; L < cfg->num_layers; L++) {
        char buf[64];
        #define ADD_CLT(suffix, s0, s1, nd) do { \
            snprintf(buf, 64, "model.layers.%d." suffix, L); \
            ADD_CTENSOR(buf, nd, s0, s1); \
        } while(0)
        ADD_CLT("input_layernorm.weight",            cfg->hidden_dim, 0, 1);
        ADD_CLT("self_attn.q_proj.weight",           cfg->hidden_dim, cfg->hidden_dim, 2);
        ADD_CLT("self_attn.k_proj.weight",           kv_dim, cfg->hidden_dim, 2);
        ADD_CLT("self_attn.v_proj.weight",           kv_dim, cfg->hidden_dim, 2);
        ADD_CLT("self_attn.o_proj.weight",           cfg->hidden_dim, cfg->hidden_dim, 2);
        ADD_CLT("post_attention_layernorm.weight",   cfg->hidden_dim, 0, 1);
        ADD_CLT("mlp.gate_proj.weight",              cfg->intermediate_dim, cfg->hidden_dim, 2);
        ADD_CLT("mlp.up_proj.weight",                cfg->intermediate_dim, cfg->hidden_dim, 2);
        ADD_CLT("mlp.down_proj.weight",              cfg->hidden_dim, cfg->intermediate_dim, 2);
        #undef ADD_CLT
    }
    #undef ADD_CTENSOR

    if (ti != num_tensors) {
        fprintf(stderr, "BUG: tensor count mismatch (%d vs %d)\n", ti, num_tensors);
        free(info); return -1;
    }

    /* Compute layout offsets — 2 bytes per element */
    uint64_t index_off = MGW_HEADER_SIZE + MGW_CONFIG_SIZE;
    uint64_t data_off  = index_off + (uint64_t)num_tensors * MGW_INDEX_ENTRY_SIZE;
    data_off = (data_off + 1) & ~1ULL;  /* 2-byte align */

    uint64_t *tensor_offsets = (uint64_t *)checked_calloc(num_tensors, sizeof(uint64_t), "mgwc_offsets");
    uint64_t cur = data_off;
    for (int i = 0; i < num_tensors; i++) {
        tensor_offsets[i] = cur;
        cur += info[i].num_elements * sizeof(uint16_t);  /* 2 bytes per element */
    }
    uint64_t total_file_size = cur;

    FILE *f = fopen(output_path, "wb");
    if (!f) {
        fprintf(stderr, "Cannot open %s for writing\n", output_path);
        free(info); free(tensor_offsets); return -1;
    }

    /* --- Write header --- */
    mgw_header_t hdr;
    memset(&hdr, 0, sizeof(hdr));
    memcpy(hdr.magic, MGWC_MAGIC, 4);
    hdr.version     = MGWC_VERSION;
    hdr.endian_tag  = MGW_ENDIAN_TAG;
    hdr.num_tensors = (uint32_t)num_tensors;
    hdr.index_offset = index_off;
    hdr.data_offset  = data_off;
    if (fwrite(&hdr, sizeof(hdr), 1, f) != 1) goto mgwc_write_err;

    /* --- Write config --- */
    mgw_config_t mcfg;
    memset(&mcfg, 0, sizeof(mcfg));
    mcfg.hidden_dim       = cfg->hidden_dim;
    mcfg.num_heads        = cfg->num_heads;
    mcfg.num_kv_heads     = cfg->num_kv_heads;
    mcfg.head_dim         = cfg->head_dim;
    mcfg.num_layers       = cfg->num_layers;
    mcfg.intermediate_dim = cfg->intermediate_dim;
    mcfg.vocab_size       = cfg->vocab_size;
    mcfg.max_seq_len      = cfg->max_seq_len;
    mcfg.rope_theta       = cfg->rope_theta;
    mcfg.lm_head_tied     = m->lm_head_tied;
    if (fwrite(&mcfg, sizeof(mcfg), 1, f) != 1) goto mgwc_write_err;

    /* --- Write index (dtype in reserved field) --- */
    for (int i = 0; i < num_tensors; i++) {
        mgw_index_entry_t entry;
        memset(&entry, 0, sizeof(entry));
        strncpy(entry.name, info[i].name, 63);
        entry.num_elements = info[i].num_elements;
        entry.data_offset  = tensor_offsets[i];
        entry.ndims        = info[i].ndims;
        entry.shape[0]     = info[i].shape[0];
        entry.shape[1]     = info[i].shape[1];
        entry.reserved     = info[i].dtype;  /* MGWC_DTYPE_F16 or MGWC_DTYPE_BF16 */
        if (fwrite(&entry, sizeof(entry), 1, f) != 1) goto mgwc_write_err;
    }

    /* Pad to data_offset */
    long pos = ftell(f);
    if (pos < 0) goto mgwc_write_err;
    while ((uint64_t)pos < data_off) {
        uint8_t zero = 0;
        if (fwrite(&zero, 1, 1, f) != 1) goto mgwc_write_err;
        pos++;
    }

    /* --- Write tensor data as raw dtype bytes --- */
    printf("Exporting %d tensors to compressed format %s...\n", num_tensors, output_path);

    for (int i = 0; i < num_tensors; i++) {
        st_tensor_t *st = st_model_find(&m->st_model, info[i].name);
        if (!st) {
            fprintf(stderr, "mgwc: tensor '%s' not found in safetensors\n", info[i].name);
            goto mgwc_write_err;
        }
        if (st_tensor_validate_bounds(&m->st_model, st) != 0) {
            fprintf(stderr, "mgwc: bounds check failed for '%s'\n", info[i].name);
            goto mgwc_write_err;
        }

        const uint8_t *raw = st_tensor_data(&m->st_model, st);
        size_t nbytes = info[i].num_elements * sizeof(uint16_t);
        if (fwrite(raw, 1, nbytes, f) != nbytes) goto mgwc_write_err;

        if (i < num_global || (i - num_global) % 9 == 0)
            printf("  [%d/%d] %s (%llu elements)\n", i+1, num_tensors,
                   info[i].name, (unsigned long long)info[i].num_elements);
    }

    fclose(f);
    free(info);
    free(tensor_offsets);

    printf("Export complete: %s (%.2f GiB, %d tensors)\n",
           output_path, total_file_size / (1024.0 * 1024.0 * 1024.0), num_tensors);
    return 0;

mgwc_write_err:
    fprintf(stderr, "Write error during compressed export to %s\n", output_path);
    fclose(f);
    free(info);
    free(tensor_offsets);
    return -1;
}

/* Load model from a .mgwc file.
 * Reads raw dtype bytes and converts to Q16.48 in memory.
 * All layers are loaded (cached mode). */
static int mgwc_load(llama_model_t *m, const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "Cannot open compressed native weight file: %s\n", path);
        return -1;
    }

    struct stat sb;
    if (fstat(fd, &sb) < 0) { close(fd); return -1; }

    if ((size_t)sb.st_size < MGW_HEADER_SIZE + MGW_CONFIG_SIZE) {
        fprintf(stderr, "mgwc: file too small: %s\n", path);
        close(fd); return -1;
    }

    /* mmap the entire file for sequential reading */
    uint8_t *map = (uint8_t *)mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (map == MAP_FAILED) {
        fprintf(stderr, "mgwc: mmap failed for %s\n", path);
        return -1;
    }

    /* Validate header */
    const mgw_header_t *hdr = (const mgw_header_t *)map;
    if (memcmp(hdr->magic, MGWC_MAGIC, 4) != 0) {
        fprintf(stderr, "mgwc: bad magic in %s\n", path);
        munmap(map, sb.st_size); return -1;
    }
    if (hdr->version != MGWC_VERSION) {
        fprintf(stderr, "mgwc: unsupported version %u in %s\n", hdr->version, path);
        munmap(map, sb.st_size); return -1;
    }
    if (hdr->endian_tag != MGW_ENDIAN_TAG) {
        fprintf(stderr, "mgwc: endian mismatch in %s\n", path);
        munmap(map, sb.st_size); return -1;
    }

    uint32_t num_tensors = hdr->num_tensors;
    uint64_t index_off = hdr->index_offset;
    uint64_t index_end = index_off + (uint64_t)num_tensors * MGW_INDEX_ENTRY_SIZE;
    if (index_end > (uint64_t)sb.st_size) {
        fprintf(stderr, "mgwc: index exceeds file size in %s\n", path);
        munmap(map, sb.st_size); return -1;
    }

    /* Read config */
    const mgw_config_t *mcfg = (const mgw_config_t *)(map + MGW_HEADER_SIZE);
    m->cfg.hidden_dim       = mcfg->hidden_dim;
    m->cfg.num_heads        = mcfg->num_heads;
    m->cfg.num_kv_heads     = mcfg->num_kv_heads;
    m->cfg.head_dim         = mcfg->head_dim;
    m->cfg.num_layers       = mcfg->num_layers;
    m->cfg.intermediate_dim = mcfg->intermediate_dim;
    m->cfg.vocab_size       = mcfg->vocab_size;
    m->cfg.max_seq_len      = mcfg->max_seq_len;
    m->cfg.rope_theta       = mcfg->rope_theta;
    m->lm_head_tied         = mcfg->lm_head_tied;

    if (m->cfg.hidden_dim <= 0 || m->cfg.num_heads <= 0 || m->cfg.num_layers <= 0
        || m->cfg.vocab_size <= 0) {
        fprintf(stderr, "mgwc: invalid config in %s\n", path);
        munmap(map, sb.st_size); return -1;
    }

    printf("Loaded compressed native config from %s:\n", path);
    printf("  hidden_dim       = %d\n", m->cfg.hidden_dim);
    printf("  num_heads        = %d (Q heads)\n", m->cfg.num_heads);
    printf("  num_kv_heads     = %d (KV heads, %d Q per KV = %s)\n",
           m->cfg.num_kv_heads,
           m->cfg.num_heads / m->cfg.num_kv_heads,
           m->cfg.num_heads == m->cfg.num_kv_heads ? "MHA" : "GQA");
    printf("  head_dim         = %d\n", m->cfg.head_dim);
    printf("  kv_dim           = %d\n", m->cfg.num_kv_heads * m->cfg.head_dim);
    printf("  num_layers       = %d\n", m->cfg.num_layers);
    printf("  intermediate_dim = %d\n", m->cfg.intermediate_dim);
    printf("  vocab_size       = %d\n", m->cfg.vocab_size);
    printf("  max_seq_len      = %d\n", m->cfg.max_seq_len);
    printf("  rope_theta       = %d\n", m->cfg.rope_theta);

    /* Allocate cached layer array */
    m->cached_layers = (layer_weights_t *)checked_calloc(
        m->cfg.num_layers, sizeof(layer_weights_t), "mgwc_cached_layers");

    /* Helper: convert uint16_t dtype data → Q16.48 malloc'd array */
    #define MGWC_CONVERT(entry_ptr, out_ptr) do { \
        const mgw_index_entry_t *_e = (entry_ptr); \
        uint64_t _n = _e->num_elements; \
        if (_n > UINT64_MAX / sizeof(int64_t)) { \
            fprintf(stderr, "mgwc: element count overflow for '%s'\n", name); \
            goto mgwc_load_err; \
        } \
        size_t _src_bytes = _n * sizeof(uint16_t); \
        if (_e->data_offset + _src_bytes > (uint64_t)sb.st_size) { \
            fprintf(stderr, "mgwc: data overflows file for '%s'\n", name); \
            goto mgwc_load_err; \
        } \
        if (_e->reserved != MGWC_DTYPE_F16 && _e->reserved != MGWC_DTYPE_BF16) { \
            fprintf(stderr, "mgwc: unknown dtype %u for tensor '%s'\n", \
                    _e->reserved, name); \
            goto mgwc_load_err; \
        } \
        const uint16_t *_src = (const uint16_t *)(map + _e->data_offset); \
        int64_t *_dst = (int64_t *)checked_malloc(_n * sizeof(int64_t), name); \
        if (_e->reserved == MGWC_DTYPE_BF16) { \
            for (uint64_t _i = 0; _i < _n; _i++) \
                _dst[_i] = bfloat16_to_q1648(_src[_i]); \
        } else { \
            for (uint64_t _i = 0; _i < _n; _i++) \
                _dst[_i] = float16_to_q1648(_src[_i]); \
        } \
        (out_ptr) = _dst; \
    } while(0)

    /* Walk index — got_layer declared here so mgwc_load_err can free it */
    int *got_layer = (int *)checked_calloc(m->cfg.num_layers, sizeof(int), "mgwc_layer_check");
    const mgw_index_entry_t *idx = (const mgw_index_entry_t *)(map + index_off);

    for (uint32_t i = 0; i < num_tensors; i++) {
        const mgw_index_entry_t *e = &idx[i];

        /* Defensive copy: NUL-terminate the mmapped name */
        char name[64];
        memcpy(name, e->name, 63);
        name[63] = '\0';

        if (strcmp(name, "model.embed_tokens.weight") == 0) {
            MGWC_CONVERT(e, m->embed_tokens);
        } else if (strcmp(name, "model.norm.weight") == 0) {
            MGWC_CONVERT(e, m->final_norm);
        } else if (strcmp(name, "lm_head.weight") == 0) {
            MGWC_CONVERT(e, m->lm_head);
        } else {
            /* Per-layer tensor */
            int layer_idx = -1;
            if (sscanf(name, "model.layers.%d.", &layer_idx) == 1
                && layer_idx >= 0 && layer_idx < m->cfg.num_layers) {
                layer_weights_t *lw = &m->cached_layers[layer_idx];
                int64_t *converted = NULL;
                MGWC_CONVERT(e, converted);

                const char *suffix = name;
                /* Find the suffix after "model.layers.N." */
                {
                    char prefix[32];
                    snprintf(prefix, sizeof(prefix), "model.layers.%d.", layer_idx);
                    size_t plen = strlen(prefix);
                    if (strlen(name) > plen)
                        suffix = name + plen;
                }

                if (strstr(suffix, "input_layernorm"))    lw->input_layernorm = converted;
                else if (strstr(suffix, "q_proj"))         lw->q_proj = converted;
                else if (strstr(suffix, "k_proj"))         lw->k_proj = converted;
                else if (strstr(suffix, "v_proj"))         lw->v_proj = converted;
                else if (strstr(suffix, "o_proj"))         lw->o_proj = converted;
                else if (strstr(suffix, "post_attention")) lw->post_attn_norm = converted;
                else if (strstr(suffix, "gate_proj"))      lw->gate_proj = converted;
                else if (strstr(suffix, "up_proj"))        lw->up_proj = converted;
                else if (strstr(suffix, "down_proj"))      lw->down_proj = converted;
                else { free(converted); continue; }

                got_layer[layer_idx]++;
            }
        }
    }

    /* Validate completeness */
    if (!m->embed_tokens || !m->final_norm) {
        fprintf(stderr, "mgwc: missing global tensors\n");
        goto mgwc_load_err;
    }
    if (!m->lm_head_tied && !m->lm_head) {
        fprintf(stderr, "mgwc: missing lm_head and not tied\n");
        goto mgwc_load_err;
    }
    if (m->lm_head_tied) m->lm_head = m->embed_tokens;

    for (int L = 0; L < m->cfg.num_layers; L++) {
        if (got_layer[L] != 9) {
            fprintf(stderr, "mgwc: layer %d has %d/9 tensors\n", L, got_layer[L]);
            goto mgwc_load_err;
        }
    }
    free(got_layer);
    got_layer = NULL;

    /* Mark model state — weights are malloc'd (not mmap'd) but all cached */
    m->layers_cached = 1;
    m->native_loaded = 0;  /* weights are malloc'd, need freeing */

    printf("Compressed native weights loaded: %u tensors, %.2f GiB file "
           "(decompressed to Q16.48 in memory)\n",
           num_tensors, (double)sb.st_size / (1024.0 * 1024.0 * 1024.0));

    munmap(map, sb.st_size);
    return 0;

mgwc_load_err:
    /* Clean up any already-allocated weight buffers and tracking arrays */
    free(got_layer);
    if (m->embed_tokens) { free(m->embed_tokens); m->embed_tokens = NULL; }
    if (m->final_norm)   { free(m->final_norm);   m->final_norm = NULL; }
    if (m->lm_head && !m->lm_head_tied) { free(m->lm_head); m->lm_head = NULL; }
    if (m->cached_layers) {
        for (int _L = 0; _L < m->cfg.num_layers; _L++)
            free_layer_weights(&m->cached_layers[_L]);
        free(m->cached_layers);
        m->cached_layers = NULL;
    }
    munmap(map, sb.st_size);
    return -1;

    #undef MGWC_CONVERT
}

/* ================================================================== */
/*  Native Weight Loader (R9)                                           */
/* ================================================================== */

/*
 * Load model from a .mgw file via mmap.
 * All weight pointers reference the mmap'd region directly.
 * Layers are always "cached" (the OS handles virtual memory paging).
 *
 * Config is read from the .mgw file, not from config.json.
 * This makes the native file self-contained.
 */
static int mgw_load(llama_model_t *m, const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "Cannot open native weight file: %s\n", path);
        return -1;
    }

    struct stat sb;
    if (fstat(fd, &sb) != 0 || sb.st_size < (off_t)(MGW_HEADER_SIZE + MGW_CONFIG_SIZE)) {
        fprintf(stderr, "Native weight file too small or stat failed: %s\n", path);
        close(fd);
        return -1;
    }

    void *map = mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (map == MAP_FAILED) {
        fprintf(stderr, "mmap failed for %s\n", path);
        return -1;
    }

    const uint8_t *base = (const uint8_t *)map;

    /* Validate header */
    const mgw_header_t *hdr = (const mgw_header_t *)base;
    if (memcmp(hdr->magic, MGW_MAGIC, MGW_MAGIC_SIZE) != 0) {
        fprintf(stderr, "Bad magic in %s (not a .mgw file)\n", path);
        munmap(map, sb.st_size);
        return -1;
    }
    if (hdr->version != MGW_VERSION) {
        fprintf(stderr, "Unsupported .mgw version %u (expected %u)\n",
                hdr->version, MGW_VERSION);
        munmap(map, sb.st_size);
        return -1;
    }
    if (hdr->endian_tag != MGW_ENDIAN_TAG) {
        fprintf(stderr, "Endian mismatch in %s (file from different architecture)\n", path);
        munmap(map, sb.st_size);
        return -1;
    }

    uint32_t num_tensors = hdr->num_tensors;
    uint64_t index_off   = hdr->index_offset;

    /* Bounds check: index must fit in file */
    uint64_t index_end = index_off + (uint64_t)num_tensors * MGW_INDEX_ENTRY_SIZE;
    if (index_end > (uint64_t)sb.st_size) {
        fprintf(stderr, "Index extends beyond file in %s\n", path);
        munmap(map, sb.st_size);
        return -1;
    }

    /* Read config */
    const mgw_config_t *mcfg = (const mgw_config_t *)(base + MGW_HEADER_SIZE);
    m->cfg.hidden_dim       = mcfg->hidden_dim;
    m->cfg.num_heads        = mcfg->num_heads;
    m->cfg.num_kv_heads     = mcfg->num_kv_heads;
    m->cfg.head_dim         = mcfg->head_dim;
    m->cfg.num_layers       = mcfg->num_layers;
    m->cfg.intermediate_dim = mcfg->intermediate_dim;
    m->cfg.vocab_size       = mcfg->vocab_size;
    m->cfg.max_seq_len      = mcfg->max_seq_len;
    m->cfg.rope_theta       = mcfg->rope_theta;
    int lm_head_tied        = mcfg->lm_head_tied;

    /* Basic config validation */
    if (m->cfg.hidden_dim <= 0 || m->cfg.num_heads <= 0 || m->cfg.num_layers <= 0
        || m->cfg.vocab_size <= 0) {
        fprintf(stderr, "Invalid config in %s\n", path);
        munmap(map, sb.st_size);
        return -1;
    }

    printf("Loaded native config from %s:\n", path);
    print_config(&m->cfg);

    /* Allocate cached_layers array (pointers will reference mmap) */
    m->cached_layers = (layer_weights_t *)checked_calloc(
        m->cfg.num_layers, sizeof(layer_weights_t), "native_cached_layers");

    /* Walk index and assign weight pointers */
    int got_embed = 0, got_norm = 0, got_lm_head = 0;
    int *got_layer = (int *)checked_calloc(m->cfg.num_layers, sizeof(int), "native_layer_check");

    for (uint32_t i = 0; i < num_tensors; i++) {
        const mgw_index_entry_t *e =
            (const mgw_index_entry_t *)(base + index_off + (uint64_t)i * MGW_INDEX_ENTRY_SIZE);

        /* Safe name copy — force null termination (defense against crafted files) */
        char tname[64];
        memcpy(tname, e->name, 63);
        tname[63] = '\0';

        /* Overflow check on element count before computing data end */
        uint64_t doff = e->data_offset;
        if (e->num_elements > UINT64_MAX / sizeof(int64_t)) {
            fprintf(stderr, "Tensor '%s' element count overflow\n", tname);
            free(got_layer); free(m->cached_layers); m->cached_layers = NULL;
            munmap(map, sb.st_size);
            return -1;
        }
        uint64_t data_bytes = e->num_elements * sizeof(int64_t);

        /* Bounds check: tensor data must fit in file */
        if (doff > (uint64_t)sb.st_size || data_bytes > (uint64_t)sb.st_size - doff) {
            fprintf(stderr, "Tensor '%s' data extends beyond file\n", tname);
            free(got_layer); free(m->cached_layers); m->cached_layers = NULL;
            munmap(map, sb.st_size);
            return -1;
        }

        /* Alignment check — int64_t requires 8-byte alignment */
        if (doff % 8 != 0) {
            fprintf(stderr, "Tensor '%s' data offset not 8-byte aligned\n", tname);
            free(got_layer); free(m->cached_layers); m->cached_layers = NULL;
            munmap(map, sb.st_size);
            return -1;
        }

        int64_t *data_ptr = (int64_t *)(base + doff);

        /* Match global tensors */
        if (strcmp(tname, "model.embed_tokens.weight") == 0) {
            m->embed_tokens = data_ptr;
            got_embed = 1;
        } else if (strcmp(tname, "model.norm.weight") == 0) {
            m->final_norm = data_ptr;
            got_norm = 1;
        } else if (strcmp(tname, "lm_head.weight") == 0) {
            m->lm_head = data_ptr;
            got_lm_head = 1;
        } else {
            /* Parse layer tensor: "model.layers.N.<suffix>" */
            int layer_idx = -1;
            char suffix[128];
            if (sscanf(tname, "model.layers.%d.%127s", &layer_idx, suffix) == 2
                && layer_idx >= 0 && layer_idx < m->cfg.num_layers) {
                layer_weights_t *lw = &m->cached_layers[layer_idx];
                if (strcmp(suffix, "input_layernorm.weight") == 0)
                    lw->input_layernorm = data_ptr;
                else if (strcmp(suffix, "self_attn.q_proj.weight") == 0)
                    lw->q_proj = data_ptr;
                else if (strcmp(suffix, "self_attn.k_proj.weight") == 0)
                    lw->k_proj = data_ptr;
                else if (strcmp(suffix, "self_attn.v_proj.weight") == 0)
                    lw->v_proj = data_ptr;
                else if (strcmp(suffix, "self_attn.o_proj.weight") == 0)
                    lw->o_proj = data_ptr;
                else if (strcmp(suffix, "post_attention_layernorm.weight") == 0)
                    lw->post_attn_norm = data_ptr;
                else if (strcmp(suffix, "mlp.gate_proj.weight") == 0)
                    lw->gate_proj = data_ptr;
                else if (strcmp(suffix, "mlp.up_proj.weight") == 0)
                    lw->up_proj = data_ptr;
                else if (strcmp(suffix, "mlp.down_proj.weight") == 0)
                    lw->down_proj = data_ptr;
                else {
                    fprintf(stderr, "WARNING: unknown layer tensor '%s' in %s\n",
                            tname, path);
                }
                got_layer[layer_idx]++;
            } else {
                fprintf(stderr, "WARNING: unrecognized tensor '%s' in %s\n",
                        tname, path);
            }
        }
    }

    /* Validate completeness */
    if (!got_embed) {
        fprintf(stderr, "Missing embed_tokens in %s\n", path);
        free(got_layer); free(m->cached_layers); m->cached_layers = NULL;
        munmap(map, sb.st_size);
        return -1;
    }
    if (!got_norm) {
        fprintf(stderr, "Missing final_norm in %s\n", path);
        free(got_layer); free(m->cached_layers); m->cached_layers = NULL;
        munmap(map, sb.st_size);
        return -1;
    }

    if (lm_head_tied) {
        m->lm_head = m->embed_tokens;
        m->lm_head_tied = 1;
        printf("  lm_head: tied to embed_tokens (from native config)\n");
    } else if (!got_lm_head) {
        fprintf(stderr, "Missing lm_head in %s (not marked as tied)\n", path);
        free(got_layer); free(m->cached_layers); m->cached_layers = NULL;
        munmap(map, sb.st_size);
        return -1;
    } else {
        m->lm_head_tied = 0;
    }

    for (int L = 0; L < m->cfg.num_layers; L++) {
        if (got_layer[L] != 9) {
            fprintf(stderr, "Layer %d incomplete (%d/9 tensors) in %s\n",
                    L, got_layer[L], path);
            free(got_layer); free(m->cached_layers); m->cached_layers = NULL;
            munmap(map, sb.st_size);
            return -1;
        }
    }
    free(got_layer);

    /* Mark model as native-loaded (mmap backed) */
    m->native_mmap      = map;
    m->native_mmap_size = sb.st_size;
    m->native_loaded    = 1;
    m->layers_cached    = 1;

    printf("Native weights loaded: %u tensors, %.1f GiB mmap'd\n",
           num_tensors, (double)sb.st_size / (1024.0 * 1024.0 * 1024.0));
    return 0;
}

/* ================================================================== */
/*  Native Streaming Load (R9B)                                         */
/* ================================================================== */

/*
 * Load model from a .mgw file in streaming mode.
 * Only global weights (embed_tokens, final_norm, lm_head) are read into
 * malloc'd buffers at startup.  Layer weights are NOT loaded here —
 * they are read on demand during forward passes via mgw_stream_load_layer().
 *
 * This gives the storage-format speed benefit of native Q16.48 (no float
 * conversion) with memory footprint similar to safetensors streaming:
 * global weights + one layer working set + KV-cache.
 *
 * The fd is kept open for the lifetime of the model.
 */
static int mgw_load_streaming(llama_model_t *m, const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "Cannot open native weight file: %s\n", path);
        return -1;
    }

    /* Read header */
    mgw_header_t hdr;
    ssize_t hdr_rd = pread(fd, &hdr, sizeof(hdr), 0);
    if (hdr_rd < 0 || (size_t)hdr_rd != sizeof(hdr)) {
        fprintf(stderr, "Cannot read header from %s\n", path);
        close(fd);
        return -1;
    }
    if (memcmp(hdr.magic, MGW_MAGIC, MGW_MAGIC_SIZE) != 0) {
        fprintf(stderr, "Bad magic in %s (not a .mgw file)\n", path);
        close(fd);
        return -1;
    }
    if (hdr.version != MGW_VERSION) {
        fprintf(stderr, "Unsupported .mgw version %u (expected %u)\n",
                hdr.version, MGW_VERSION);
        close(fd);
        return -1;
    }
    if (hdr.endian_tag != MGW_ENDIAN_TAG) {
        fprintf(stderr, "Endian mismatch in %s\n", path);
        close(fd);
        return -1;
    }

    /* Read config */
    mgw_config_t mcfg;
    ssize_t cfg_rd = pread(fd, &mcfg, sizeof(mcfg), MGW_HEADER_SIZE);
    if (cfg_rd < 0 || (size_t)cfg_rd != sizeof(mcfg)) {
        fprintf(stderr, "Cannot read config from %s\n", path);
        close(fd);
        return -1;
    }

    m->cfg.hidden_dim       = mcfg.hidden_dim;
    m->cfg.num_heads        = mcfg.num_heads;
    m->cfg.num_kv_heads     = mcfg.num_kv_heads;
    m->cfg.head_dim         = mcfg.head_dim;
    m->cfg.num_layers       = mcfg.num_layers;
    m->cfg.intermediate_dim = mcfg.intermediate_dim;
    m->cfg.vocab_size       = mcfg.vocab_size;
    m->cfg.max_seq_len      = mcfg.max_seq_len;
    m->cfg.rope_theta       = mcfg.rope_theta;
    int lm_head_tied        = mcfg.lm_head_tied;

    if (m->cfg.hidden_dim <= 0 || m->cfg.num_heads <= 0 || m->cfg.num_layers <= 0
        || m->cfg.vocab_size <= 0) {
        fprintf(stderr, "Invalid config in %s\n", path);
        close(fd);
        return -1;
    }

    printf("Loaded native config from %s (stream mode):\n", path);
    print_config(&m->cfg);

    /* Read entire index into memory */
    uint32_t num_tensors = hdr.num_tensors;
    uint64_t index_off   = hdr.index_offset;
    size_t index_bytes   = (size_t)num_tensors * MGW_INDEX_ENTRY_SIZE;
    mgw_index_entry_t *index = (mgw_index_entry_t *)checked_malloc(
        index_bytes, "native_stream_index");

    ssize_t rd = pread(fd, index, index_bytes, (off_t)index_off);
    if (rd < 0 || (size_t)rd != index_bytes) {
        fprintf(stderr, "Cannot read tensor index from %s\n", path);
        free(index);
        close(fd);
        return -1;
    }

    /* Walk index: read global tensors, record layer tensor locations */
    int got_embed = 0, got_norm = 0, got_lm_head = 0;

    for (uint32_t i = 0; i < num_tensors; i++) {
        mgw_index_entry_t *e = &index[i];
        char tname[64];
        memcpy(tname, e->name, 63);
        tname[63] = '\0';

        /* Only load global tensors now */
        int is_global = 0;
        int64_t **dest = NULL;

        if (strcmp(tname, "model.embed_tokens.weight") == 0) {
            dest = &m->embed_tokens;
            got_embed = 1;
            is_global = 1;
        } else if (strcmp(tname, "model.norm.weight") == 0) {
            dest = &m->final_norm;
            got_norm = 1;
            is_global = 1;
        } else if (strcmp(tname, "lm_head.weight") == 0) {
            dest = &m->lm_head;
            got_lm_head = 1;
            is_global = 1;
        }

        if (is_global) {
            size_t nbytes = (size_t)e->num_elements * sizeof(int64_t);
            int64_t *buf = (int64_t *)checked_malloc(nbytes, tname);
            ssize_t r = pread(fd, buf, nbytes, (off_t)e->data_offset);
            if (r < 0 || (size_t)r != nbytes) {
                fprintf(stderr, "Failed to read tensor '%s' from %s\n", tname, path);
                free(buf);
                free(index);
                close(fd);
                return -1;
            }
            *dest = buf;
        }
    }

    if (!got_embed || !got_norm) {
        fprintf(stderr, "Missing required global tensors in %s\n", path);
        free(m->embed_tokens); free(m->final_norm);
        free(index);
        close(fd);
        return -1;
    }

    if (lm_head_tied) {
        m->lm_head = m->embed_tokens;
        m->lm_head_tied = 1;
        printf("  lm_head: tied to embed_tokens (from native config)\n");
    } else if (!got_lm_head) {
        fprintf(stderr, "Missing lm_head in %s (not marked as tied)\n", path);
        free(m->embed_tokens); free(m->final_norm);
        free(index);
        close(fd);
        return -1;
    } else {
        m->lm_head_tied = 0;
    }

    /* Store streaming state */
    m->native_stream    = 1;
    m->native_stream_fd = fd;
    m->ns_num_tensors   = num_tensors;
    m->ns_index         = index;
    m->native_loaded    = 0;  /* NOT mmap — weight data is malloc'd */
    m->layers_cached    = 0;  /* layers loaded on demand */
    m->cached_layers    = NULL;

    printf("Native stream mode: %u tensors indexed, global weights loaded, "
           "layers read on demand\n", num_tensors);
    return 0;
}

/*
 * Read one layer's weights from the .mgw file into a layer_weights_t struct.
 * Each tensor is read via pread() from the stored index.
 * The caller must free the layer with free_layer_weights() after use.
 *
 * Returns 0 on success, -1 on error.
 */
static int mgw_stream_load_layer(llama_model_t *m, int layer_idx,
                                  layer_weights_t *w) {
    memset(w, 0, sizeof(*w));

    int fd = m->native_stream_fd;
    char prefix[64];
    snprintf(prefix, sizeof(prefix), "model.layers.%d.", layer_idx);
    size_t prefix_len = strlen(prefix);

    int found = 0;
    for (uint32_t i = 0; i < m->ns_num_tensors; i++) {
        const mgw_index_entry_t *e = &m->ns_index[i];
        char tname[64];
        memcpy(tname, e->name, 63);
        tname[63] = '\0';

        if (strncmp(tname, prefix, prefix_len) != 0)
            continue;

        const char *suffix = tname + prefix_len;
        int64_t **dest = NULL;

        if (strcmp(suffix, "input_layernorm.weight") == 0)
            dest = &w->input_layernorm;
        else if (strcmp(suffix, "self_attn.q_proj.weight") == 0)
            dest = &w->q_proj;
        else if (strcmp(suffix, "self_attn.k_proj.weight") == 0)
            dest = &w->k_proj;
        else if (strcmp(suffix, "self_attn.v_proj.weight") == 0)
            dest = &w->v_proj;
        else if (strcmp(suffix, "self_attn.o_proj.weight") == 0)
            dest = &w->o_proj;
        else if (strcmp(suffix, "post_attention_layernorm.weight") == 0)
            dest = &w->post_attn_norm;
        else if (strcmp(suffix, "mlp.gate_proj.weight") == 0)
            dest = &w->gate_proj;
        else if (strcmp(suffix, "mlp.up_proj.weight") == 0)
            dest = &w->up_proj;
        else if (strcmp(suffix, "mlp.down_proj.weight") == 0)
            dest = &w->down_proj;

        if (!dest) continue;

        size_t nbytes = (size_t)e->num_elements * sizeof(int64_t);
        *dest = (int64_t *)checked_malloc(nbytes, tname);
        ssize_t r = pread(fd, *dest, nbytes, (off_t)e->data_offset);
        if (r < 0 || (size_t)r != nbytes) {
            fprintf(stderr, "Failed to read layer %d tensor '%s'\n",
                    layer_idx, suffix);
            free_layer_weights(w);
            return -1;
        }
        found++;
    }

    if (found != 9) {
        fprintf(stderr, "Layer %d incomplete: got %d/9 tensors from native stream\n",
                layer_idx, found);
        free_layer_weights(w);
        return -1;
    }

    return 0;
}

/* Forward pass for one token at given position.
 * Returns logits[vocab_size]. Caller must free(). Returns NULL on error. */
static fixed_t *llama_forward(llama_model_t *m, int token, int pos) {
    if (g_profile.enabled) g_profile.forward_calls++;
    const llama_config_t *cfg = &m->cfg;
    int dim = cfg->hidden_dim;

    /* Bounds-check token ID (Finding 2) */
    if (token < 0 || token >= cfg->vocab_size) {
        fprintf(stderr, "Token ID %d out of range [0, %d)\n", token, cfg->vocab_size);
        return NULL;
    }
    if (pos < 0 || pos >= cfg->max_seq_len) {
        fprintf(stderr, "Position %d out of range [0, %d)\n", pos, cfg->max_seq_len);
        return NULL;
    }

    fixed_t *hidden = (fixed_t *)checked_malloc(dim * sizeof(fixed_t), "hidden");
    for (int i = 0; i < dim; i++)
        hidden[i] = m->embed_tokens[(int64_t)token * dim + i];

    /* Set scratch pointer for transformer_layer_forward (R7) */
    g_layer_scratch = m->scratch_buf;

    for (int layer = 0; layer < cfg->num_layers; layer++) {
        if (m->layers_cached) {
            /* Use pre-cached weights (no re-conversion) */
            struct timespec lc0, lc1;
            if (g_profile.enabled) clock_gettime(CLOCK_MONOTONIC, &lc0);
            transformer_layer_forward(hidden, &m->cached_layers[layer],
                                       cfg, &m->rope, &m->kv, layer, pos);
            if (g_profile.enabled) {
                clock_gettime(CLOCK_MONOTONIC, &lc1);
                g_profile.t_layer_compute_total += ts_diff(lc0, lc1);
            }
        } else {
            /* Streaming: load, run, free */
            layer_weights_t w;
            memset(&w, 0, sizeof(w));
            struct timespec ll0, ll1, lc0, lc1;
            if (g_profile.enabled) clock_gettime(CLOCK_MONOTONIC, &ll0);
            int load_ok;
            if (m->native_stream) {
                /* Native stream (R9B): read Q16.48 layer directly from .mgw */
                load_ok = mgw_stream_load_layer(m, layer, &w);
            } else {
                /* Safetensors stream: convert float→Q16.48 per layer */
                load_ok = load_layer_weights(&m->st_model, layer, &w, cfg);
            }
            if (load_ok != 0) {
                fprintf(stderr, "Failed to load layer %d weights\n", layer);
                free_layer_weights(&w); /* free any partially loaded tensors */
                g_layer_scratch = NULL;
                free(hidden);
                return NULL;
            }
            if (g_profile.enabled) {
                clock_gettime(CLOCK_MONOTONIC, &ll1);
                g_profile.t_layer_load_total += ts_diff(ll0, ll1);
                clock_gettime(CLOCK_MONOTONIC, &lc0);
            }
            transformer_layer_forward(hidden, &w, cfg, &m->rope, &m->kv,
                                       layer, pos);
            if (g_profile.enabled) {
                clock_gettime(CLOCK_MONOTONIC, &lc1);
                g_profile.t_layer_compute_total += ts_diff(lc0, lc1);
            }
            free_layer_weights(&w);
        }
    }

    fixed_t *normed = (fixed_t *)checked_malloc(dim * sizeof(fixed_t), "final_norm");
    rmsnorm(normed, hidden, m->final_norm, dim);

    fixed_t *logits = (fixed_t *)checked_malloc(cfg->vocab_size * sizeof(fixed_t), "logits");
    matmul_vec(logits, m->lm_head, normed, cfg->vocab_size, dim);

    g_layer_scratch = NULL;  /* clear scratch pointer */
    free(hidden);
    free(normed);
    return logits;
}

static int argmax(const fixed_t *logits, int n) {
    int best = 0;
    for (int i = 1; i < n; i++)
        if (logits[i] > logits[best]) best = i;
    return best;
}

/* ================================================================== */
/*  Sampling Support (R5)                                               */
/* ================================================================== */

/*
 * Deterministic PRNG: xoroshiro128+ with SplitMix64 seeding.
 * Used exclusively for token sampling in --generate / --prompt paths.
 * Verification and benchmark paths always use greedy (argmax).
 */

typedef struct {
    uint64_t s[2];
} rng_state_t;

static void rng_seed(rng_state_t *rng, uint64_t seed) {
    /* SplitMix64 to initialize both state words */
    uint64_t z = seed;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    rng->s[0] = z ^ (z >> 31);
    z = seed + 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    rng->s[1] = z ^ (z >> 31);
    if (rng->s[0] == 0 && rng->s[1] == 0) rng->s[0] = 1;
}

static uint64_t rng_next(rng_state_t *rng) {
    uint64_t s0 = rng->s[0], s1 = rng->s[1];
    uint64_t result = s0 + s1;
    s1 ^= s0;
    rng->s[0] = ((s0 << 55) | (s0 >> 9)) ^ s1 ^ (s1 << 14);
    rng->s[1] = (s1 << 36) | (s1 >> 28);
    return result;
}

/*
 * Sampling parameters. Constructed once from CLI flags at startup.
 * Passed as NULL pointer for greedy paths (verify, benchmark).
 */
typedef struct {
    fixed_t temperature;  /* FP_ZERO → greedy (default) */
    int top_k;            /* 0 → disabled */
    fixed_t top_p;        /* FP_ONE → disabled */
    rng_state_t rng;
} sample_params_t;

/* Comparator for descending sort of fixed_t values */
static int cmp_fixed_desc(const void *a, const void *b) {
    fixed_t va = *(const fixed_t *)a;
    fixed_t vb = *(const fixed_t *)b;
    return (vb > va) - (vb < va);
}

/*
 * Select a token from logits using temperature scaling, top-k, top-p,
 * and random sampling. If sp is NULL or temperature <= 0, returns argmax.
 *
 * All arithmetic is in Q16.48 fixed-point. The only float→fixed conversion
 * happens once at CLI parsing time (main), not here.
 *
 * Flow: temperature → top-k mask → softmax → top-p nucleus → sample
 */
static int sample_token(const fixed_t *logits, int vocab_size,
                        sample_params_t *sp) {
    /* Near-zero temperature (< 0.01) is effectively greedy — avoid
     * blowing up logits via fp_div with a tiny divisor. */
    fixed_t temp_floor = FP_ONE / 100; /* 0.01 in Q16.48 */
    if (!sp || sp->temperature <= 0 || sp->temperature < temp_floor)
        return argmax(logits, vocab_size);

    fixed_t *work = (fixed_t *)checked_malloc(
        (size_t)vocab_size * sizeof(fixed_t), "sample_work");

    /* 1. Temperature scaling: logit_i / T */
    for (int i = 0; i < vocab_size; i++)
        work[i] = fp_div(logits[i], sp->temperature);

    /* 2. Top-k: keep only the K highest logits, mask rest to -inf */
    if (sp->top_k > 0 && sp->top_k < vocab_size) {
        fixed_t *sorted = (fixed_t *)checked_malloc(
            (size_t)vocab_size * sizeof(fixed_t), "topk_sort");
        memcpy(sorted, work, (size_t)vocab_size * sizeof(fixed_t));
        qsort(sorted, (size_t)vocab_size, sizeof(fixed_t), cmp_fixed_desc);
        fixed_t threshold = sorted[sp->top_k - 1];
        free(sorted);

        fixed_t neg_inf = INT64_MIN / 2;
        for (int i = 0; i < vocab_size; i++)
            if (work[i] < threshold) work[i] = neg_inf;
    }

    /* 3. Softmax → probabilities (in-place, reuses existing function) */
    softmax(work, vocab_size);

    /* 4. Top-p (nucleus): keep smallest set with cumulative prob >= p */
    if (sp->top_p > 0 && sp->top_p < FP_ONE) {
        /*
         * Mark kept tokens by negating their probability. This avoids
         * a separate boolean array. Works because all probs are >= 0
         * after softmax, and the search skips values <= 0.
         */
        fixed_t cumsum = 0;
        while (cumsum < sp->top_p) {
            int best = -1;
            for (int i = 0; i < vocab_size; i++) {
                if (work[i] <= 0) continue;
                if (best < 0 || work[i] > work[best]) best = i;
            }
            if (best < 0) break;
            cumsum += work[best];
            work[best] = -work[best]; /* negate to mark as kept */
        }
        /* Restore kept (negative → positive), zero out the rest */
        for (int i = 0; i < vocab_size; i++) {
            if (work[i] < 0) work[i] = -work[i];
            else work[i] = 0;
        }
        /* Renormalize */
        fixed_t sum = 0;
        for (int i = 0; i < vocab_size; i++) sum += work[i];
        if (sum > 0)
            for (int i = 0; i < vocab_size; i++)
                if (work[i] > 0) work[i] = fp_div(work[i], sum);
    }

    /* 5. Sample: random threshold in [0, FP_ONE), scan cumulative.
     * Fallback to highest-probability surviving token if rounding
     * leaves total mass below the random threshold. */
    fixed_t r = (fixed_t)(rng_next(&sp->rng) >> (64 - FP_PRECISION));
    fixed_t cumsum = 0;
    int selected = -1;
    for (int i = 0; i < vocab_size; i++) {
        cumsum += work[i];
        if (cumsum > r) {
            selected = i;
            break;
        }
    }
    if (selected < 0)
        selected = argmax(work, vocab_size);

    free(work);
    return selected;
}

/* ================================================================== */
/*  Pre-tokenized Prompts (Quick Path)                                  */
/* ================================================================== */

#define LLAMA_BOS 1
#define LLAMA_EOS 2

static const int PROMPT_FRANCE[] = {1, 450, 7483, 310, 3444, 338};
static const int PROMPT_FRANCE_LEN = 6;

static const int PROMPT_STORY[] = {1, 9038, 2501, 263, 931};
static const int PROMPT_STORY_LEN = 5;

static const int PROMPT_MATH[] = {1, 29896, 29974, 29896, 29922};
static const int PROMPT_MATH_LEN = 5;

static const int PROMPT_LIFE[] = {1, 450, 6593, 310, 2834, 338};
static const int PROMPT_LIFE_LEN = 6;

/* ================================================================== */
/*  Autoregressive Generation                                           */
/* ================================================================== */

/* Generate returns the generated token IDs and count, for comparison.
 * out_tokens must be pre-allocated with room for prompt_len + max_new_tokens.
 * sp == NULL → greedy decoding (argmax). sp != NULL → sampling. */
static int generate(llama_model_t *m, const int *prompt, int prompt_len,
                    int max_new_tokens, int *out_tokens,
                    sample_params_t *sp) {
    kv_cache_reset(&m->kv);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    memcpy(out_tokens, prompt, prompt_len * sizeof(int));
    int total_len = prompt_len;

    /* Prefill */
    struct timespec pf0, pf1;
    if (g_profile.enabled) clock_gettime(CLOCK_MONOTONIC, &pf0);
    for (int pos = 0; pos < prompt_len; pos++) {
        fixed_t *logits = llama_forward(m, out_tokens[pos], pos);
        if (!logits) {
            fprintf(stderr, "Forward pass failed at position %d\n", pos);
            return -1;
        }
        if (pos == prompt_len - 1) {
            int next = sample_token(logits, m->cfg.vocab_size, sp);
            out_tokens[total_len++] = next;
        }
        free(logits);
    }
    if (g_profile.enabled) {
        clock_gettime(CLOCK_MONOTONIC, &pf1);
        g_profile.t_prefill_total += ts_diff(pf0, pf1);
        g_profile.prefill_tokens += prompt_len;
    }

    /* Decode */
    struct timespec dc0, dc1;
    int decode_start_len = total_len;
    if (g_profile.enabled) clock_gettime(CLOCK_MONOTONIC, &dc0);
    for (int i = 1; i < max_new_tokens; i++) {
        int pos = prompt_len + i - 1;
        fixed_t *logits = llama_forward(m, out_tokens[pos], pos);
        if (!logits) break;

        int next = sample_token(logits, m->cfg.vocab_size, sp);
        free(logits);

        if (next == LLAMA_EOS) break;
        out_tokens[total_len++] = next;
    }
    if (g_profile.enabled) {
        clock_gettime(CLOCK_MONOTONIC, &dc1);
        g_profile.t_decode_total += ts_diff(dc0, dc1);
        g_profile.decode_tokens += total_len - decode_start_len;
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = ts_diff(t0, t1);
    int generated = total_len - prompt_len;
    if (g_profile.enabled) g_profile.t_generation_total += elapsed;

    printf("  Generated %d tokens in %.2f seconds", generated, elapsed);
    if (elapsed > 0) printf(" (%.2f tok/s)", generated / elapsed);
    printf("\n  Tokens: [");
    for (int i = 0; i < total_len; i++) {
        if (i > 0) printf(", ");
        printf("%d", out_tokens[i]);
    }
    printf("]\n");

    return total_len;
}

/* ================================================================== */
/*  Reference-Based Verification (Finding 1)                            */
/* ================================================================== */

/*
 * --verify mode now does two things:
 *   1. Runs layer 0 on BOS embedding and prints output for manual comparison
 *      with a Python reference.
 *   2. Checks structural invariants:
 *      - RMSNorm output has unit RMS
 *      - Softmax output sums to 1
 *      - Residual connection preserves magnitude order
 *   3. If a reference file exists (<model_dir>/layer0_reference.bin), compares
 *      the first 4096 output values against it and reports max absolute error.
 */

/* Compute RMS of a vector */
static double compute_rms(const fixed_t *x, int n) {
    int128_t sum_sq = 0;
    for (int i = 0; i < n; i++)
        sum_sq += (int128_t)x[i] * x[i];
    double mean_sq = (double)(sum_sq >> FP_PRECISION) / n / (double)FP_ONE;
    if (mean_sq < 0) mean_sq = 0;
    /* Compute sqrt using fp_sqrt for consistency */
    return fp_to_double(fp_sqrt((fixed_t)((sum_sq >> FP_PRECISION) / n)));
}

/* ================================================================== */
/*  Stage-by-Stage Reference Comparison                                 */
/* ================================================================== */

typedef struct {
    double max_err;
    double mean_err;
    int    worst_idx;
    int    found;       /* 1 if reference file existed */
} stage_result_t;

/* Compare integer values against a float32 reference file.
 * File name: <model_dir>/layer0_ref_<stage>.bin
 * Returns {found=0} if file doesn't exist. */
static stage_result_t compare_stage(const char *model_dir, const char *stage,
                                     const fixed_t *vals, int n) {
    stage_result_t r = {0, 0, 0, 0};
    char path[512];
    snprintf(path, sizeof(path), "%s/layer0_ref_%s.bin", model_dir, stage);
    FILE *f = fopen(path, "rb");
    if (!f) return r;
    r.found = 1;

    float *ref = (float *)checked_malloc(n * sizeof(float), "ref_stage");
    size_t nread = fread(ref, sizeof(float), n, f);
    fclose(f);

    if ((int)nread != n) {
        printf("    %-20s TRUNCATED (%zu/%d)\n", stage, nread, n);
        free(ref);
        r.max_err = 1e9;  /* treat truncated as hard failure */
        return r;
    }

    for (int i = 0; i < n; i++) {
        double ours = fp_to_double(vals[i]);
        double theirs = (double)ref[i];
        double err = ours > theirs ? ours - theirs : theirs - ours;
        if (err > r.max_err) { r.max_err = err; r.worst_idx = i; }
        r.mean_err += err;
    }
    r.mean_err /= n;

    const char *status = r.max_err < 0.005 ? "OK" : "DRIFT";
    printf("    %-20s %s  max=%.8f  mean=%.8f  worst[%d] ours=%.8f ref=%.8f\n",
           stage, status, r.max_err, r.mean_err, r.worst_idx,
           fp_to_double(vals[r.worst_idx]), (double)ref[r.worst_idx]);

    free(ref);
    return r;
}

static int verify_single_layer(llama_model_t *m, const char *model_dir) {
    printf("\n=== Single Layer Verification (Layer 0) ===\n");

    const llama_config_t *cfg = &m->cfg;
    int dim = cfg->hidden_dim;
    int hd = cfg->head_dim;
    int pass_count = 0, fail_count = 0;

    #define VERIFY(name, cond) do { \
        if (cond) { pass_count++; printf("  PASS: %s\n", name); } \
        else      { fail_count++; printf("  FAIL: %s\n", name); } \
    } while(0)

    /* Embedding lookup */
    fixed_t *hidden = (fixed_t *)checked_malloc(dim * sizeof(fixed_t), "hidden");
    for (int i = 0; i < dim; i++)
        hidden[i] = m->embed_tokens[(int64_t)LLAMA_BOS * dim + i];

    /* Input statistics */
    fixed_t min_h = hidden[0], max_h = hidden[0];
    int128_t sum_h = 0;
    for (int i = 0; i < dim; i++) {
        if (hidden[i] < min_h) min_h = hidden[i];
        if (hidden[i] > max_h) max_h = hidden[i];
        sum_h += hidden[i];
    }
    printf("  Input embedding (BOS): min=%.6f max=%.6f mean=%.6f\n",
           fp_to_double(min_h), fp_to_double(max_h),
           fp_to_double((fixed_t)(sum_h / dim)));

    /* Check embedding is non-trivial */
    VERIFY("embedding non-zero", max_h > 0 || min_h < 0);
    VERIFY("embedding has variation", max_h != min_h);

    /* Load layer 0 */
    layer_weights_t w;
    memset(&w, 0, sizeof(w));
    if (m->layers_cached) {
        w = m->cached_layers[0]; /* borrow, don't free */
    } else {
        if (load_layer_weights(&m->st_model, 0, &w, cfg) != 0) {
            fprintf(stderr, "Failed to load layer 0\n");
            free(hidden);
            return 1;
        }
    }

    /* -- Verify RMSNorm output has ~unit RMS -- */
    fixed_t *normed = (fixed_t *)checked_malloc(dim * sizeof(fixed_t), "normed_verify");
    rmsnorm(normed, hidden, w.input_layernorm, dim);
    double rms_normed = compute_rms(normed, dim);
    /* After RMSNorm with weights, RMS should be roughly in range [0.1, 10] */
    VERIFY("RMSNorm output RMS in [0.01, 100]", rms_normed > 0.01 && rms_normed < 100);
    printf("    RMSNorm output RMS = %.6f\n", rms_normed);

    /* -- Verify Q/K/V projections produce non-zero output -- */
    int kv_dim = cfg->num_kv_heads * hd;
    fixed_t *q_vec = (fixed_t *)checked_malloc(dim * sizeof(fixed_t), "q_verify");
    matmul_vec(q_vec, w.q_proj, normed, dim, dim);
    double rms_q = compute_rms(q_vec, dim);
    VERIFY("Q projection non-zero", rms_q > 0);
    printf("    Q projection RMS = %.6f\n", rms_q);

    /* -- Verify attention scores sum to 1 via softmax -- */
    fixed_t *k_vec = (fixed_t *)checked_malloc(kv_dim * sizeof(fixed_t), "k_verify");
    matmul_vec(k_vec, w.k_proj, normed, kv_dim, dim);
    rope_apply(&m->rope, q_vec, 0);    /* first Q head */
    rope_apply(&m->rope, k_vec, 0);    /* first KV head */

    /* For position 0, attention is trivially [1.0] (single key), but verify softmax */
    {
        int128_t dot = 0;
        for (int d = 0; d < hd; d++)
            dot += (int128_t)q_vec[d] * k_vec[d];
        fixed_t scale = fp_inv_sqrt(fp_from_int(hd));
        fixed_t score = fp_mul((fixed_t)(dot >> FP_PRECISION), scale);
        VERIFY("attention score is finite", fp_abs(score) < fp_from_int(10000));
    }
    fixed_t attn_prob = fp_safe_exp(FP_ZERO); /* softmax of single element = 1.0 */
    VERIFY("single-position softmax = 1.0",
           fp_abs(attn_prob - FP_ONE) < FP_ONE / 1000);

    free(q_vec);
    free(k_vec);
    free(normed);

    /* ================================================================
     * Stage-by-stage reference comparison.
     * Manually executes each step of layer 0 and compares with
     * float32 reference files generated by --dump-reference.
     * This isolates the first point of divergence.
     * ================================================================ */

    printf("\n--- Stage-by-Stage Reference Comparison ---\n");

    /* Reference files are required for the stage-by-stage gate.
     * Missing refs are a hard failure — not a silent skip. */
    {
        char probe[512];
        snprintf(probe, sizeof(probe), "%s/layer0_ref_embedding.bin", model_dir);
        FILE *fp = fopen(probe, "rb");
        if (!fp) {
            printf("  FAIL: No intermediate reference files found.\n");
            printf("  Generate with:\n");
            printf("    ./%s %s --dump-reference > gen_ref.py && python3 gen_ref.py\n",
                   "llama_int", model_dir);
            fail_count++;
            goto verify_done;
        }
        fclose(fp);
    }

    int nkv_heads = cfg->num_kv_heads;
    int gqa_group = cfg->num_heads / nkv_heads;
    int kv_dim_v = nkv_heads * hd;
    int inter = cfg->intermediate_dim;

    int stages_checked = 0, stages_ok = 0, stages_missing = 0;
    const char *first_drift = NULL;
    double first_drift_err = 0;

    #define CHECK_STAGE(name, buf, count) do { \
        stage_result_t _r = compare_stage(model_dir, name, buf, count); \
        if (_r.found) { \
            stages_checked++; \
            if (_r.max_err < 0.005) stages_ok++; \
            else if (!first_drift) { first_drift = name; first_drift_err = _r.max_err; } \
        } else { \
            stages_missing++; \
            printf("    %-20s MISSING\n", name); \
        } \
    } while(0)

    kv_cache_reset(&m->kv);

    /* Stage 1: Embedding */
    /* hidden already holds BOS embedding from above */
    CHECK_STAGE("embedding", hidden, dim);

    /* Stage 2: Input RMSNorm */
    fixed_t *s_normed = (fixed_t *)checked_malloc(dim * sizeof(fixed_t), "s_normed");
    rmsnorm(s_normed, hidden, w.input_layernorm, dim);
    CHECK_STAGE("rmsnorm", s_normed, dim);

    /* Stage 3-5: Q/K/V projections */
    fixed_t *s_q = (fixed_t *)checked_malloc(dim * sizeof(fixed_t), "s_q");
    fixed_t *s_k = (fixed_t *)checked_malloc(kv_dim_v * sizeof(fixed_t), "s_k");
    fixed_t *s_v = (fixed_t *)checked_malloc(kv_dim_v * sizeof(fixed_t), "s_v");
    matmul_vec(s_q, w.q_proj, s_normed, dim, dim);
    matmul_vec(s_k, w.k_proj, s_normed, kv_dim_v, dim);
    matmul_vec(s_v, w.v_proj, s_normed, kv_dim_v, dim);
    CHECK_STAGE("q_proj", s_q, dim);
    CHECK_STAGE("k_proj", s_k, kv_dim_v);
    CHECK_STAGE("v_proj", s_v, kv_dim_v);

    /* Stage 6-7: RoPE */
    for (int h = 0; h < cfg->num_heads; h++)
        rope_apply(&m->rope, s_q + h * hd, 0);
    for (int h = 0; h < nkv_heads; h++)
        rope_apply(&m->rope, s_k + h * hd, 0);
    CHECK_STAGE("q_rope", s_q, dim);
    CHECK_STAGE("k_rope", s_k, kv_dim_v);

    /* Stage 8: Attention output (position 0 → single-token attention)
     * Single-position softmax is always 1.0, so attn_out = V.
     * In packed mode, exact KV cache may be NULL — use local s_k/s_v directly. */
    fixed_t *s_attn = (fixed_t *)checked_malloc(dim * sizeof(fixed_t), "s_attn");
    fixed_t scale = fp_inv_sqrt(fp_from_int(hd));
    if (m->kv.key_cache) {
        memcpy(kv_key_at(&m->kv, 0, 0), s_k, kv_dim_v * sizeof(fixed_t));
        memcpy(kv_value_at(&m->kv, 0, 0), s_v, kv_dim_v * sizeof(fixed_t));
    }
    for (int h = 0; h < cfg->num_heads; h++) {
        int kv_h = h / gqa_group;
        fixed_t *q_h = s_q + h * hd;
        fixed_t *k_local = s_k + kv_h * hd; /* use local buffer, not cache */
        int128_t dot = 0;
        for (int d = 0; d < hd; d++)
            dot += (int128_t)q_h[d] * k_local[d];
        (void)scale; (void)dot;
        fixed_t *v_local = s_v + kv_h * hd;
        memcpy(s_attn + h * hd, v_local, hd * sizeof(fixed_t));
    }
    CHECK_STAGE("attn_out", s_attn, dim);

    /* Stage 9: Output projection */
    fixed_t *s_o = (fixed_t *)checked_malloc(dim * sizeof(fixed_t), "s_o");
    matmul_vec(s_o, w.o_proj, s_attn, dim, dim);
    CHECK_STAGE("o_proj", s_o, dim);

    /* Stage 10: First residual add */
    fixed_t *s_post_attn = (fixed_t *)checked_malloc(dim * sizeof(fixed_t), "s_post_attn");
    for (int i = 0; i < dim; i++)
        s_post_attn[i] = hidden[i] + s_o[i];
    CHECK_STAGE("post_attn", s_post_attn, dim);

    /* Stage 11: Post-attention RMSNorm */
    fixed_t *s_post_norm = (fixed_t *)checked_malloc(dim * sizeof(fixed_t), "s_post_norm");
    rmsnorm(s_post_norm, s_post_attn, w.post_attn_norm, dim);
    CHECK_STAGE("post_attn_norm", s_post_norm, dim);

    /* Stage 12: MLP output */
    fixed_t *s_gate = (fixed_t *)checked_malloc(inter * sizeof(fixed_t), "s_gate");
    fixed_t *s_up   = (fixed_t *)checked_malloc(inter * sizeof(fixed_t), "s_up");
    fixed_t *s_silu = (fixed_t *)checked_malloc(inter * sizeof(fixed_t), "s_silu");
    fixed_t *s_gu   = (fixed_t *)checked_malloc(inter * sizeof(fixed_t), "s_gu");
    fixed_t *s_mlp  = (fixed_t *)checked_malloc(dim * sizeof(fixed_t), "s_mlp");
    matmul_vec(s_gate, w.gate_proj, s_post_norm, inter, dim);
    matmul_vec(s_up, w.up_proj, s_post_norm, inter, dim);
    for (int i = 0; i < inter; i++)
        s_silu[i] = fp_silu(s_gate[i]);
    for (int i = 0; i < inter; i++)
        s_gu[i] = fp_mul(s_silu[i], s_up[i]);
    matmul_vec(s_mlp, w.down_proj, s_gu, dim, inter);
    CHECK_STAGE("mlp_out", s_mlp, dim);

    /* Stage 13: Final layer output (second residual) */
    fixed_t *s_layer_out = (fixed_t *)checked_malloc(dim * sizeof(fixed_t), "s_layer_out");
    for (int i = 0; i < dim; i++)
        s_layer_out[i] = s_post_attn[i] + s_mlp[i];
    CHECK_STAGE("layer_out", s_layer_out, dim);

    /* Print first 16 values of layer output */
    printf("  Layer 0 output (first 16):\n    ");
    for (int i = 0; i < 16 && i < dim; i++) {
        printf("%.6f ", fp_to_double(s_layer_out[i]));
        if ((i + 1) % 8 == 0) printf("\n    ");
    }

    /* Summary */
    int total_expected = stages_checked + stages_missing;
    printf("\n  Stages: %d checked, %d OK, %d drifted, %d missing (of %d)\n",
           stages_checked, stages_ok, stages_checked - stages_ok,
           stages_missing, total_expected);
    if (first_drift)
        printf("  FIRST DIVERGENCE: %s (max_err=%.8f)\n", first_drift, first_drift_err);
    else if (stages_checked > 0 && stages_missing == 0)
        printf("  ALL STAGES WITHIN THRESHOLD\n");

    VERIFY("all 13 stage reference files present",
           stages_checked == 13 && stages_missing == 0);
    VERIFY("all 13 stages within threshold",
           stages_ok == 13);

    free(s_normed); free(s_q); free(s_k); free(s_v);
    free(s_attn); free(s_o); free(s_post_attn); free(s_post_norm);
    free(s_gate); free(s_up); free(s_silu); free(s_gu);
    free(s_mlp); free(s_layer_out);

verify_done:

    /* -- Summary -- */
    printf("\n  Verification: %d passed, %d failed\n", pass_count, fail_count);
    if (fail_count > 0)
        printf("  *** VERIFICATION FAILURES — output may be incorrect ***\n");

    if (!m->layers_cached)
        free_layer_weights(&w);
    free(hidden);

    #undef VERIFY
    #undef CHECK_STAGE

    return fail_count;
}

/* ================================================================== */
/*  Full-Forward Verification (R1.2)                                    */
/* ================================================================== */

/*
 * Runs BOS through all layers, compares final hidden state and top
 * logits against float reference files:
 *   <model_dir>/full_ref_final_hidden.bin   — float32[hidden_dim]
 *   <model_dir>/full_ref_logits_top5.bin    — 5 x (int32 index, float32 value)
 *
 * This validates that error does not blow up across the full layer stack.
 */

static int verify_full_forward(llama_model_t *m, const char *model_dir) {
    printf("\n=== Full-Forward Verification (All %d Layers) ===\n",
           m->cfg.num_layers);

    const llama_config_t *cfg = &m->cfg;
    int dim = cfg->hidden_dim;
    int pass_count = 0, fail_count = 0;

    #define FF_VERIFY(name, cond) do { \
        if (cond) { pass_count++; printf("  PASS: %s\n", name); } \
        else      { fail_count++; printf("  FAIL: %s\n", name); } \
    } while(0)

    /* Run full forward pass on BOS */
    kv_cache_reset(&m->kv);
    fixed_t *logits = llama_forward(m, LLAMA_BOS, 0);
    if (!logits) {
        printf("  FAIL: forward pass returned NULL\n");
        printf("\n  Full-Forward: 0 passed, 1 failed\n");
        return 1;
    }

    /* Find top-5 tokens */
    int top5_idx[5] = {0, 0, 0, 0, 0};
    fixed_t top5_val[5] = {0, 0, 0, 0, 0};
    for (int k = 0; k < 5; k++) {
        int best = -1;
        for (int i = 0; i < cfg->vocab_size; i++) {
            int skip = 0;
            for (int j = 0; j < k; j++)
                if (i == top5_idx[j]) { skip = 1; break; }
            if (skip) continue;
            if (best < 0 || logits[i] > logits[best]) best = i;
        }
        top5_idx[k] = best;
        top5_val[k] = logits[best];
    }

    printf("  Top-5 predicted tokens (BOS):\n");
    for (int k = 0; k < 5; k++)
        printf("    #%d: token %5d  logit=%.6f\n",
               k + 1, top5_idx[k], fp_to_double(top5_val[k]));

    /* Compare final hidden state against reference */
    {
        char path[512];
        snprintf(path, sizeof(path), "%s/full_ref_final_hidden.bin", model_dir);
        FILE *f = fopen(path, "rb");
        if (f) {
            float *ref = (float *)checked_malloc(dim * sizeof(float), "ref_hidden");
            size_t nread = fread(ref, sizeof(float), dim, f);
            fclose(f);
            if ((int)nread == dim) {
                /* We need to recompute final hidden to compare
                 * (llama_forward frees it internally). Re-run embedding
                 * and layers to get the pre-lm_head hidden state. */
                kv_cache_reset(&m->kv);
                fixed_t *hidden = (fixed_t *)checked_malloc(dim * sizeof(fixed_t), "ff_hidden");
                for (int i = 0; i < dim; i++)
                    hidden[i] = m->embed_tokens[(int64_t)LLAMA_BOS * dim + i];
                int all_layers_ok = 1;
                for (int layer = 0; layer < cfg->num_layers; layer++) {
                    if (m->layers_cached) {
                        transformer_layer_forward(hidden, &m->cached_layers[layer],
                                                   cfg, &m->rope, &m->kv, layer, 0);
                    } else {
                        layer_weights_t w;
                        memset(&w, 0, sizeof(w));
                        if (load_layer_weights(&m->st_model, layer, &w, cfg) != 0) {
                            fprintf(stderr, "  Failed to load layer %d\n", layer);
                            free_layer_weights(&w);
                            all_layers_ok = 0;
                            break;
                        }
                        transformer_layer_forward(hidden, &w, cfg, &m->rope, &m->kv,
                                                   layer, 0);
                        free_layer_weights(&w);
                    }
                }
                if (!all_layers_ok) {
                    FF_VERIFY("full-forward layer replay completed", 0);
                } else {
                    /* Apply final RMSNorm */
                    fixed_t *normed = (fixed_t *)checked_malloc(dim * sizeof(fixed_t), "ff_norm");
                    rmsnorm(normed, hidden, m->final_norm, dim);

                    double max_err = 0, mean_err = 0;
                    int worst_idx = 0;
                    for (int i = 0; i < dim; i++) {
                        double ours = fp_to_double(normed[i]);
                        double theirs = (double)ref[i];
                        double err = ours > theirs ? ours - theirs : theirs - ours;
                        if (err > max_err) { max_err = err; worst_idx = i; }
                        mean_err += err;
                    }
                    mean_err /= dim;

                    printf("  Final hidden (%d layers + norm):\n", cfg->num_layers);
                    printf("    max_err=%.8f  mean_err=%.8f  worst[%d] ours=%.8f ref=%.8f\n",
                           max_err, mean_err, worst_idx,
                           fp_to_double(normed[worst_idx]), (double)ref[worst_idx]);
                    /* Known-good baseline: max_err ~0.014 on TinyLlama.
                     * Threshold at 0.05 gives ~3.5x margin — tight enough
                     * to catch meaningful regressions. */
                    FF_VERIFY("final hidden max_err < 0.05", max_err < 0.05);
                    free(normed);
                }
                free(hidden);
            } else {
                printf("  full_ref_final_hidden.bin: truncated (%zu/%d)\n", nread, dim);
                FF_VERIFY("full_ref_final_hidden.bin not truncated", 0);
            }
            free(ref);
        } else {
            printf("  FAIL: full_ref_final_hidden.bin missing\n");
            printf("  Generate with:\n");
            printf("    ./%s %s --dump-reference > gen_ref.py && python3 gen_ref.py\n",
                   "llama_int", model_dir);
            FF_VERIFY("full_ref_final_hidden.bin present", 0);
        }
    }

    /* Compare top-5 logits against reference */
    {
        char path[512];
        snprintf(path, sizeof(path), "%s/full_ref_logits_top5.bin", model_dir);
        FILE *f = fopen(path, "rb");
        if (f) {
            int32_t ref_idx[5];
            float ref_val[5];
            int ok = 1;
            for (int k = 0; k < 5; k++) {
                if (fread(&ref_idx[k], sizeof(int32_t), 1, f) != 1 ||
                    fread(&ref_val[k], sizeof(float), 1, f) != 1) {
                    ok = 0;
                    break;
                }
            }
            fclose(f);

            if (ok) {
                printf("  Top-5 logit comparison:\n");
                int top1_match = (top5_idx[0] == ref_idx[0]);
                int top5_match = 0;
                for (int k = 0; k < 5; k++) {
                    double err = fp_to_double(top5_val[k]) - (double)ref_val[k];
                    if (err < 0) err = -err;
                    int idx_match = (top5_idx[k] == ref_idx[k]);
                    if (idx_match) top5_match++;
                    printf("    #%d: ours=%5d(%.4f)  ref=%5d(%.4f)  err=%.6f  %s\n",
                           k + 1, top5_idx[k], fp_to_double(top5_val[k]),
                           ref_idx[k], ref_val[k], err,
                           idx_match ? "MATCH" : "MISMATCH");
                }
                double max_logit_err = 0;
                for (int k = 0; k < 5; k++) {
                    double e = fp_to_double(top5_val[k]) - (double)ref_val[k];
                    if (e < 0) e = -e;
                    if (e > max_logit_err) max_logit_err = e;
                }
                FF_VERIFY("top-1 token matches float reference", top1_match);
                FF_VERIFY("top-5 tokens all match float reference", top5_match == 5);
                /* Known-good baseline: max logit err ~0.001 on TinyLlama.
                 * Threshold at 0.005 gives ~5x margin. */
                FF_VERIFY("max logit error < 0.005", max_logit_err < 0.005);
                printf("  Top-5 agreement: %d/5, max logit err: %.6f\n",
                       top5_match, max_logit_err);
            }
            if (!ok) {
                printf("  full_ref_logits_top5.bin: truncated\n");
                FF_VERIFY("full_ref_logits_top5.bin not truncated", 0);
            }
        } else {
            printf("  FAIL: full_ref_logits_top5.bin missing\n");
            printf("  Generate with:\n");
            printf("    ./%s %s --dump-reference > gen_ref.py && python3 gen_ref.py\n",
                   "llama_int", model_dir);
            FF_VERIFY("full_ref_logits_top5.bin present", 0);
        }
    }

    free(logits);

    printf("\n  Full-Forward: %d passed, %d failed\n", pass_count, fail_count);
    if (fail_count > 0)
        printf("  *** FULL-FORWARD FAILURES ***\n");

    #undef FF_VERIFY

    return fail_count;
}

/* ================================================================== */
/*  Multi-Position Verification (R1.3)                                  */
/* ================================================================== */

/*
 * Runs a short fixed prompt (PROMPT_FRANCE, 6 tokens) through all layers
 * at each position, accumulating the KV-cache across positions.
 * Compares per-position: final hidden state, top-1 token, top-5 agreement,
 * max logit error against float reference files:
 *   <model_dir>/multipos_ref_pos<i>_hidden.bin  — float32[hidden_dim]
 *   <model_dir>/multipos_ref_pos<i>_top5.bin    — 5 x (int32 index, float32 value)
 *
 * This validates that the integer path stays aligned with the float reference
 * across multiple causal positions with real KV-cache accumulation, non-trivial
 * softmax, and GQA mapping.
 */

static int verify_multi_position(llama_model_t *m, const char *model_dir) {
    printf("\n=== Multi-Position Verification (R1.3) ===\n");

    const llama_config_t *cfg = &m->cfg;
    int dim = cfg->hidden_dim;
    int pass_count = 0, fail_count = 0;

    #define MP_VERIFY(name, cond) do { \
        if (cond) { pass_count++; printf("  PASS: %s\n", name); } \
        else      { fail_count++; printf("  FAIL: %s\n", name); } \
    } while(0)

    const int *prompt = PROMPT_FRANCE;
    int prompt_len = PROMPT_FRANCE_LEN;

    printf("  Prompt: PROMPT_FRANCE (%d tokens): [", prompt_len);
    for (int i = 0; i < prompt_len; i++)
        printf("%s%d", i ? ", " : "", prompt[i]);
    printf("]\n\n");

    kv_cache_reset(&m->kv);

    for (int pos = 0; pos < prompt_len; pos++) {
        printf("  --- Position %d (token %d) ---\n", pos, prompt[pos]);

        /* Embedding */
        fixed_t *hidden = (fixed_t *)checked_malloc(
            dim * sizeof(fixed_t), "mp_hidden");
        for (int i = 0; i < dim; i++)
            hidden[i] = m->embed_tokens[(int64_t)prompt[pos] * dim + i];

        /* Run through all layers (KV-cache accumulates across positions) */
        int layers_ok = 1;
        for (int layer = 0; layer < cfg->num_layers; layer++) {
            if (m->layers_cached) {
                transformer_layer_forward(hidden, &m->cached_layers[layer],
                                           cfg, &m->rope, &m->kv, layer, pos);
            } else {
                layer_weights_t w;
                memset(&w, 0, sizeof(w));
                if (load_layer_weights(&m->st_model, layer, &w, cfg) != 0) {
                    fprintf(stderr, "  Failed to load layer %d at pos %d\n",
                            layer, pos);
                    free_layer_weights(&w);
                    layers_ok = 0;
                    break;
                }
                transformer_layer_forward(hidden, &w, cfg, &m->rope, &m->kv,
                                           layer, pos);
                free_layer_weights(&w);
            }
        }

        if (!layers_ok) {
            char msg[80];
            snprintf(msg, sizeof(msg), "pos %d layer replay completed", pos);
            MP_VERIFY(msg, 0);
            free(hidden);
            break;
        }

        /* Final RMSNorm */
        fixed_t *normed = (fixed_t *)checked_malloc(
            dim * sizeof(fixed_t), "mp_norm");
        rmsnorm(normed, hidden, m->final_norm, dim);

        /* Compute logits */
        fixed_t *logits = (fixed_t *)checked_malloc(
            cfg->vocab_size * sizeof(fixed_t), "mp_logits");
        matmul_vec(logits, m->lm_head, normed, cfg->vocab_size, dim);

        /* Find top-5 tokens */
        int top5_idx[5] = {0, 0, 0, 0, 0};
        fixed_t top5_val[5] = {0, 0, 0, 0, 0};
        for (int k = 0; k < 5; k++) {
            int best = -1;
            for (int i = 0; i < cfg->vocab_size; i++) {
                int skip = 0;
                for (int j = 0; j < k; j++)
                    if (i == top5_idx[j]) { skip = 1; break; }
                if (skip) continue;
                if (best < 0 || logits[i] > logits[best]) best = i;
            }
            top5_idx[k] = best;
            top5_val[k] = logits[best];
        }

        /* Compare hidden state against reference */
        {
            char path[512];
            snprintf(path, sizeof(path), "%s/multipos_ref_pos%d_hidden.bin",
                     model_dir, pos);
            FILE *f = fopen(path, "rb");
            if (f) {
                float *ref = (float *)checked_malloc(
                    dim * sizeof(float), "mp_ref_h");
                size_t nread = fread(ref, sizeof(float), dim, f);
                fclose(f);
                if ((int)nread == dim) {
                    double max_err = 0, mean_err = 0;
                    int worst = 0;
                    for (int i = 0; i < dim; i++) {
                        double ours = fp_to_double(normed[i]);
                        double theirs = (double)ref[i];
                        double err = ours > theirs ? ours - theirs : theirs - ours;
                        if (err > max_err) { max_err = err; worst = i; }
                        mean_err += err;
                    }
                    mean_err /= dim;
                    printf("    hidden: max_err=%.8f mean_err=%.8f worst[%d]\n",
                           max_err, mean_err, worst);
                    char msg[80];
                    snprintf(msg, sizeof(msg),
                             "pos %d hidden max_err < 0.05", pos);
                    MP_VERIFY(msg, max_err < 0.05);
                } else {
                    printf("    multipos_ref_pos%d_hidden.bin: truncated "
                           "(%zu/%d)\n", pos, nread, dim);
                    char msg[80];
                    snprintf(msg, sizeof(msg),
                             "pos %d hidden ref not truncated", pos);
                    MP_VERIFY(msg, 0);
                }
                free(ref);
            } else {
                char msg[80];
                snprintf(msg, sizeof(msg), "pos %d hidden ref present", pos);
                MP_VERIFY(msg, 0);
                printf("    Generate with: ./llama_int %s --dump-reference"
                       " > gen_ref.py && python3 gen_ref.py\n", model_dir);
            }
        }

        /* Compare top-5 logits against reference */
        {
            char path[512];
            snprintf(path, sizeof(path), "%s/multipos_ref_pos%d_top5.bin",
                     model_dir, pos);
            FILE *f = fopen(path, "rb");
            if (f) {
                int32_t ref_idx[5];
                float ref_val[5];
                int ok = 1;
                for (int k = 0; k < 5; k++) {
                    if (fread(&ref_idx[k], sizeof(int32_t), 1, f) != 1 ||
                        fread(&ref_val[k], sizeof(float), 1, f) != 1) {
                        ok = 0;
                        break;
                    }
                }
                fclose(f);

                if (ok) {
                    int top1_match = (top5_idx[0] == ref_idx[0]);
                    int top5_match = 0;
                    double max_logit_err = 0;
                    for (int k = 0; k < 5; k++) {
                        int idx_match = (top5_idx[k] == ref_idx[k]);
                        if (idx_match) top5_match++;
                        double e = fp_to_double(top5_val[k]) -
                                   (double)ref_val[k];
                        if (e < 0) e = -e;
                        if (e > max_logit_err) max_logit_err = e;
                        printf("    #%d: ours=%5d(%.4f) ref=%5d(%.4f) "
                               "err=%.6f %s\n",
                               k + 1, top5_idx[k],
                               fp_to_double(top5_val[k]),
                               ref_idx[k], ref_val[k], e,
                               idx_match ? "MATCH" : "MISMATCH");
                    }
                    char msg[80];
                    snprintf(msg, sizeof(msg),
                             "pos %d top-1 matches", pos);
                    MP_VERIFY(msg, top1_match);
                    snprintf(msg, sizeof(msg),
                             "pos %d top-5 all match", pos);
                    MP_VERIFY(msg, top5_match == 5);
                    snprintf(msg, sizeof(msg),
                             "pos %d max logit err < 0.005", pos);
                    MP_VERIFY(msg, max_logit_err < 0.005);
                    printf("    top-5: %d/5, max logit err: %.6f\n",
                           top5_match, max_logit_err);
                } else {
                    printf("    multipos_ref_pos%d_top5.bin: truncated\n",
                           pos);
                    char msg[80];
                    snprintf(msg, sizeof(msg),
                             "pos %d top5 ref not truncated", pos);
                    MP_VERIFY(msg, 0);
                }
            } else {
                char msg[80];
                snprintf(msg, sizeof(msg), "pos %d top5 ref present", pos);
                MP_VERIFY(msg, 0);
            }
        }

        free(logits);
        free(normed);
        free(hidden);
    }

    printf("\n  Multi-Position: %d passed, %d failed\n", pass_count, fail_count);
    if (fail_count > 0)
        printf("  *** MULTI-POSITION FAILURES ***\n");

    #undef MP_VERIFY

    return fail_count;
}

/* ================================================================== */
/*  Token-Level Agreement Gate (R1.4)                                   */
/* ================================================================== */

/*
 * Generates continuations for all 4 fixed prompts using greedy decoding
 * (argmax — same path as --generate) and compares each generated token
 * against the float reference in <model_dir>/reference_tokens.txt.
 *
 * Hard gate: every generated token must exactly match the float reference.
 * Missing or malformed reference files fail hard.
 *
 * Since both sides use greedy decoding, exact token match == top-1 agreement
 * at every generation step.
 *
 * Reference file format (one line per prompt):
 *   <prompt_name> <tok1> <tok2> ... <tokN>
 *
 * Generated by: --dump-reference > gen_ref.py && python3 gen_ref.py
 */

#define TOKEN_GATE_MAX_NEW      20  /* must match --dump-reference script */
#define TOKEN_GATE_MAX_REF     128  /* max reference tokens per line */
#define TOKEN_GATE_NUM_PROMPTS   4

static int verify_token_agreement(llama_model_t *m, const char *model_dir) {
    printf("\n=== Token-Level Agreement Gate (R1.4) ===\n");

    int pass_count = 0, fail_count = 0;

    #define TG_VERIFY(name, cond) do { \
        if (cond) { pass_count++; printf("  PASS: %s\n", name); } \
        else      { fail_count++; printf("  FAIL: %s\n", name); } \
    } while(0)

    struct {
        const char *name;
        const int *prompt;
        int len;
    } prompts[] = {
        {"france_capital",  PROMPT_FRANCE, PROMPT_FRANCE_LEN},
        {"story_beginning", PROMPT_STORY,  PROMPT_STORY_LEN},
        {"simple_math",     PROMPT_MATH,   PROMPT_MATH_LEN},
        {"meaning_of_life", PROMPT_LIFE,   PROMPT_LIFE_LEN},
    };

    /* Load reference_tokens.txt from model_dir — hard fail if missing */
    char ref_path[512];
    snprintf(ref_path, sizeof(ref_path), "%s/reference_tokens.txt", model_dir);
    FILE *ref_fp = fopen(ref_path, "r");
    if (!ref_fp) {
        printf("  %s not found\n", ref_path);
        printf("  Generate with:\n");
        printf("    ./llama_int %s --dump-reference > gen_ref.py && "
               "python3 gen_ref.py\n", model_dir);
        fail_count++;
        printf("  FAIL: reference_tokens.txt present\n");
        printf("\n  Token Agreement: %d passed, %d failed\n",
               pass_count, fail_count);
        printf("  *** TOKEN AGREEMENT FAILURES ***\n");
        return fail_count;
    }
    pass_count++;
    printf("  PASS: reference_tokens.txt present\n");

    /* Parse reference file into per-prompt arrays */
    struct {
        char name[256];
        int tokens[TOKEN_GATE_MAX_REF];
        int count;
    } refs[16];
    int num_refs = 0;

    {
        char line[4096];
        int parse_errors = 0;
        while (fgets(line, sizeof(line), ref_fp) && num_refs < 16) {
            char *saveptr = NULL;
            char *tok = strtok_r(line, " \t\n", &saveptr);
            if (!tok) continue;
            strncpy(refs[num_refs].name, tok, sizeof(refs[num_refs].name) - 1);
            refs[num_refs].name[sizeof(refs[num_refs].name) - 1] = '\0';
            refs[num_refs].count = 0;
            while ((tok = strtok_r(NULL, " \t\n", &saveptr)) &&
                   refs[num_refs].count < TOKEN_GATE_MAX_REF) {
                char *end = NULL;
                long v = strtol(tok, &end, 10);
                if (end == tok || *end != '\0') {
                    printf("  WARNING: non-integer token '%s' in %s "
                           "line '%s'\n", tok, ref_path, refs[num_refs].name);
                    parse_errors++;
                    break;
                }
                refs[num_refs].tokens[refs[num_refs].count++] = (int)v;
            }
            num_refs++;
        }
        fclose(ref_fp);
        printf("  Loaded %d reference entries from %s\n", num_refs, ref_path);

        if (parse_errors > 0) {
            fail_count++;
            printf("  FAIL: reference_tokens.txt contains non-integer tokens\n");
            printf("\n  Token Agreement: %d passed, %d failed\n",
                   pass_count, fail_count);
            printf("  *** TOKEN AGREEMENT FAILURES ***\n");
            return fail_count;
        }
    }

    /* Every prompt must have a reference entry with exactly
     * TOKEN_GATE_MAX_NEW tokens. A short reference would lower the
     * comparison bar and allow a truncated/corrupted file to pass. */
    int all_refs_valid = 1;
    for (int p = 0; p < TOKEN_GATE_NUM_PROMPTS; p++) {
        int found = 0;
        int count = 0;
        for (int r = 0; r < num_refs; r++) {
            if (strcmp(prompts[p].name, refs[r].name) == 0) {
                found = 1;
                count = refs[r].count;
                break;
            }
        }
        if (!found) {
            char msg[80];
            snprintf(msg, sizeof(msg), "%s reference entry found",
                     prompts[p].name);
            TG_VERIFY(msg, 0);
            all_refs_valid = 0;
        } else if (count != TOKEN_GATE_MAX_NEW) {
            char msg[80];
            snprintf(msg, sizeof(msg), "%s reference has %d tokens "
                     "(expected %d)", prompts[p].name, count,
                     TOKEN_GATE_MAX_NEW);
            TG_VERIFY(msg, 0);
            all_refs_valid = 0;
        }
    }
    if (!all_refs_valid) {
        printf("\n  Token Agreement: %d passed, %d failed\n",
               pass_count, fail_count);
        printf("  *** TOKEN AGREEMENT FAILURES ***\n");
        return fail_count;
    }

    /* Generate and compare for each prompt */
    int total_compared = 0, total_matched = 0;

    for (int p = 0; p < TOKEN_GATE_NUM_PROMPTS; p++) {
        printf("  --- %s (%d prompt tokens + %d max generation steps) ---\n",
               prompts[p].name, prompts[p].len, TOKEN_GATE_MAX_NEW);

        /* Find first matching reference */
        int ref_count = 0;
        int *ref_tokens = NULL;
        for (int r = 0; r < num_refs; r++) {
            if (strcmp(prompts[p].name, refs[r].name) == 0) {
                ref_tokens = refs[r].tokens;
                ref_count = refs[r].count;
                break;
            }
        }

        /* Generate via greedy decoding (same path as --generate) */
        int max_buf = prompts[p].len + TOKEN_GATE_MAX_NEW;
        int *gen_buf = (int *)checked_malloc(
            (size_t)max_buf * sizeof(int), "tg_gen");
        int total_len = generate(m, prompts[p].prompt, prompts[p].len,
                                  TOKEN_GATE_MAX_NEW, gen_buf, NULL);
        int gen_new = (total_len > prompts[p].len)
                      ? total_len - prompts[p].len : 0;

        /* Token-by-token comparison */
        int cmp_len = ref_count < gen_new ? ref_count : gen_new;
        int matches = 0;
        int first_diff = -1;

        for (int i = 0; i < cmp_len; i++) {
            int gen_tok = gen_buf[prompts[p].len + i];
            int ref_tok = ref_tokens[i];
            if (gen_tok == ref_tok) {
                matches++;
            } else {
                if (first_diff < 0) first_diff = i;
                printf("    step %2d: got %5d, expected %5d  MISMATCH\n",
                       i, gen_tok, ref_tok);
            }
        }

        /* Treat un-generated positions as mismatches */
        if (gen_new < ref_count) {
            printf("    Generated %d tokens but reference has %d "
                   "(early EOS or failure)\n", gen_new, ref_count);
            if (first_diff < 0) first_diff = gen_new;
        }

        int prompt_perfect = (matches == ref_count);
        double rate = ref_count > 0 ?
                      100.0 * matches / ref_count : 0.0;

        total_compared += ref_count;
        total_matched += matches;

        printf("    Result: %d/%d match (%.1f%%)", matches, ref_count, rate);
        if (first_diff >= 0)
            printf(", first divergence at step %d", first_diff);
        else
            printf(", PERFECT");
        printf("\n");

        char msg[80];
        snprintf(msg, sizeof(msg), "%s %d/%d tokens match",
                 prompts[p].name, matches, ref_count);
        TG_VERIFY(msg, prompt_perfect);

        free(gen_buf);
    }

    /* Defensive: fail if no tokens were actually compared */
    if (total_compared == 0) {
        fail_count++;
        printf("  FAIL: no tokens were compared (empty or broken references)\n");
    }

    /* Overall summary */
    double overall_rate = total_compared > 0
                          ? 100.0 * total_matched / total_compared : 0.0;
    printf("\n  Overall: %d/%d tokens match (%.1f%%)\n",
           total_matched, total_compared, overall_rate);

    if (total_matched == total_compared && total_compared > 0)
        printf("  INTEGER GENERATION MATCHES FLOAT REFERENCE\n");

    printf("\n  Token Agreement: %d passed, %d failed\n", pass_count, fail_count);
    if (fail_count > 0)
        printf("  *** TOKEN AGREEMENT FAILURES ***\n");

    #undef TG_VERIFY
    return fail_count;
}

/* ================================================================== */
/*  Quality Benchmark with Actual Comparison (Finding 1)                */
/* ================================================================== */

/*
 * --benchmark mode now:
 *   1. Runs generation on each prompt and records output token IDs
 *   2. If <model_dir>/reference_tokens.txt exists, compares token-by-token
 *   3. Reports: token match rate, first divergence position
 *   4. Computes top-1 logit magnitude stats as a sanity check
 *
 * Reference file format (one line per prompt):
 *   <prompt_name> <tok1> <tok2> ... <tokN>
 *
 * Generate with:
 *   python3 generate_reference.py <model_dir>
 */

static int quality_benchmark(llama_model_t *m, const char *model_dir) {
    printf("\n================================================================\n");
    printf("  Quality Benchmark: Integer-Only Llama-2-7B\n");
    printf("================================================================\n");

    struct {
        const char *name;
        const int *prompt;
        int len;
    } prompts[] = {
        {"france_capital",  PROMPT_FRANCE, PROMPT_FRANCE_LEN},
        {"story_beginning", PROMPT_STORY,  PROMPT_STORY_LEN},
        {"simple_math",     PROMPT_MATH,   PROMPT_MATH_LEN},
        {"meaning_of_life", PROMPT_LIFE,   PROMPT_LIFE_LEN},
    };
    int num_prompts = 4;
    int max_new = 20;

    /* Storage for generated tokens */
    int bench_failed = 0;
    int **gen_tokens = (int **)checked_malloc(num_prompts * sizeof(int *), "gen_tokens");
    int *gen_lens = (int *)checked_malloc(num_prompts * sizeof(int), "gen_lens");

    for (int p = 0; p < num_prompts; p++) {
        printf("\n--- Prompt: %s ---\n", prompts[p].name);
        gen_tokens[p] = (int *)checked_malloc(
            (prompts[p].len + max_new) * sizeof(int), "gen_tok_buf");
        gen_lens[p] = generate(m, prompts[p].prompt, prompts[p].len,
                                max_new, gen_tokens[p], NULL);
        if (gen_lens[p] < 0) gen_lens[p] = prompts[p].len; /* generation failed */

        /* Top-1 logit sanity: run one more forward pass and check logit stats */
        fixed_t *logits = llama_forward(m, gen_tokens[p][prompts[p].len - 1],
                                         prompts[p].len - 1);
        if (logits) {
            int top = argmax(logits, m->cfg.vocab_size);
            printf("  Top-1 token at end of prompt: %d (logit=%.4f)\n",
                   top, fp_to_double(logits[top]));

            /* Check logit distribution sanity */
            fixed_t max_l = logits[0], min_l = logits[0];
            int128_t sum_l = 0;
            for (int i = 0; i < m->cfg.vocab_size; i++) {
                if (logits[i] > max_l) max_l = logits[i];
                if (logits[i] < min_l) min_l = logits[i];
                sum_l += logits[i];
            }
            printf("  Logit stats: min=%.4f max=%.4f mean=%.6f spread=%.4f\n",
                   fp_to_double(min_l), fp_to_double(max_l),
                   fp_to_double((fixed_t)(sum_l / m->cfg.vocab_size)),
                   fp_to_double(max_l - min_l));

            /* Sanity: logits should have nonzero spread */
            if (max_l == min_l)
                printf("  WARNING: all logits identical — model output is degenerate\n");

            free(logits);
        }
    }

    /* Try loading reference tokens */
    char ref_path[512];
    snprintf(ref_path, sizeof(ref_path), "%s/reference_tokens.txt", model_dir);
    FILE *ref_fp = fopen(ref_path, "r");
    if (!ref_fp) {
        snprintf(ref_path, sizeof(ref_path), "reference_tokens.txt");
        ref_fp = fopen(ref_path, "r");
    }

    if (ref_fp) {
        printf("\n--- Reference Comparison (%s) ---\n", ref_path);
        char line[4096];
        int total_compared = 0, total_match = 0;

        while (fgets(line, sizeof(line), ref_fp)) {
            char ref_name[256];
            int ref_toks[128];
            int ref_count = 0;

            /* Parse: name tok1 tok2 ... */
            char *saveptr = NULL;
            char *tok = strtok_r(line, " \t\n", &saveptr);
            if (!tok) continue;
            strncpy(ref_name, tok, sizeof(ref_name) - 1);
            ref_name[sizeof(ref_name) - 1] = '\0';

            while ((tok = strtok_r(NULL, " \t\n", &saveptr)) && ref_count < 128) {
                char *endp;
                long val = strtol(tok, &endp, 10);
                if (endp == tok || *endp != '\0') break;  /* invalid token */
                ref_toks[ref_count++] = (int)val;
            }

            /* Find matching prompt */
            for (int p = 0; p < num_prompts; p++) {
                if (strcmp(ref_name, prompts[p].name) != 0) continue;

                int gen_new = gen_lens[p] - prompts[p].len;
                int cmp_len = gen_new < ref_count ? gen_new : ref_count;
                if (ref_count == 0) {
                    printf("  %s: WARNING — 0 reference tokens parsed (skipped)\n", ref_name);
                    bench_failed = 1;
                    break;
                }
                int matches = 0;
                int first_diff = -1;

                for (int i = 0; i < cmp_len; i++) {
                    int gen_tok = gen_tokens[p][prompts[p].len + i];
                    if (gen_tok == ref_toks[i]) {
                        matches++;
                    } else if (first_diff < 0) {
                        first_diff = i;
                    }
                }

                total_compared += cmp_len;
                total_match += matches;

                printf("  %s: %d/%d tokens match (%.1f%%)",
                       ref_name, matches, cmp_len,
                       cmp_len > 0 ? 100.0 * matches / cmp_len : 0.0);
                if (first_diff >= 0)
                    printf(", first diff at position %d (got %d, expected %d)",
                           first_diff,
                           gen_tokens[p][prompts[p].len + first_diff],
                           ref_toks[first_diff]);
                printf("\n");
                break;
            }
        }
        fclose(ref_fp);

        printf("\n  TOTAL: %d/%d tokens match (%.1f%%)\n",
               total_match, total_compared,
               total_compared > 0 ? 100.0 * total_match / total_compared : 0.0);
        if (total_match == total_compared && total_compared > 0)
            printf("  PERFECT MATCH — integer output identical to float reference\n");
        else if (total_compared > 0 && 100.0 * total_match / total_compared > 95.0)
            printf("  EXCELLENT — >95%% token agreement with float reference\n");
        if (total_compared > 0 && total_match != total_compared)
            bench_failed = 1;
        if (total_compared == 0) {
            printf("  WARNING: 0 tokens compared — reference file may be empty or malformed\n");
            bench_failed = 1;
        }
    } else {
        printf("\n  No reference_tokens.txt found — skipping token comparison.\n");
        printf("  Generate one with: ./llama_int %s --dump-reference > gen_ref.py && python3 gen_ref.py\n",
               model_dir);
    }

    for (int p = 0; p < num_prompts; p++)
        free(gen_tokens[p]);
    free(gen_tokens);
    free(gen_lens);
    return bench_failed;
}

/* ================================================================== */
/*  Dump Reference Mode                                                 */
/* ================================================================== */

/* --dump-reference: prints a Python script that generates float32 reference
 * data for every intermediate stage in layer 0, plus token-level references.
 * model_path is a local HuggingFace model directory. */
static void dump_reference_script(const char *model_path) {
    printf("#!/usr/bin/env python3\n");
    printf("\"\"\"Generate stage-by-stage reference data for llama_int --verify.\"\"\"\n");
    printf("import torch, struct, math, os, sys\n");
    printf("import torch.nn.functional as F\n");
    printf("from transformers import AutoModelForCausalLM\n\n");

    printf("MODEL = '%s'\n", model_path);
    printf("OUT = os.path.dirname(os.path.abspath(MODEL)) if os.path.isabs(MODEL) else '.'\n");
    printf("OUT = MODEL if os.path.isdir(MODEL) else '.'\n\n");

    printf("def dump(name, tensor):\n");
    printf("    vals = tensor.detach().float().flatten().tolist()\n");
    printf("    path = os.path.join(OUT, f'layer0_ref_{name}.bin')\n");
    printf("    with open(path, 'wb') as f:\n");
    printf("        for v in vals:\n");
    printf("            f.write(struct.pack('<f', v))\n");
    printf("    print(f'  {name}: {len(vals)} floats -> {path}')\n\n");

    printf("try:\n");
    printf("    m = AutoModelForCausalLM.from_pretrained(\n");
    printf("        MODEL, torch_dtype=torch.float32, local_files_only=True)\n");
    printf("except TypeError:\n");
    printf("    m = AutoModelForCausalLM.from_pretrained(\n");
    printf("        MODEL, dtype=torch.float32, local_files_only=True)\n");
    printf("m.eval()\n\n");

    printf("try:\n");
    printf("    from transformers.models.llama.modeling_llama import (\n");
    printf("        apply_rotary_pos_emb, repeat_kv)\n");
    printf("except ImportError:\n");
    printf("    print('ERROR: cannot import Llama internals', file=sys.stderr)\n");
    printf("    sys.exit(1)\n\n");

    printf("nh  = m.config.num_attention_heads\n");
    printf("nkv = m.config.num_key_value_heads\n");
    printf("hd  = m.config.hidden_size // nh\n");
    printf("gqa = nh // nkv\n");
    printf("print(f'Model: {m.config._name_or_path}')\n");
    printf("print(f'  heads={nh} kv_heads={nkv} head_dim={hd} gqa_group={gqa}')\n\n");

    printf("layer = m.model.layers[0]\n");
    printf("attn  = layer.self_attn\n\n");

    printf("with torch.no_grad():\n");
    printf("    # 1. Embedding (BOS = token 1)\n");
    printf("    x = m.model.embed_tokens(torch.tensor([[1]]))  # [1,1,dim]\n");
    printf("    dump('embedding', x[0, 0])\n\n");

    printf("    # 2. Input RMSNorm\n");
    printf("    normed = layer.input_layernorm(x)\n");
    printf("    dump('rmsnorm', normed[0, 0])\n\n");

    printf("    # 3-5. Q/K/V projections (flat, pre-RoPE)\n");
    printf("    q = attn.q_proj(normed)  # [1,1,dim]\n");
    printf("    k = attn.k_proj(normed)  # [1,1,kv_dim]\n");
    printf("    v = attn.v_proj(normed)  # [1,1,kv_dim]\n");
    printf("    dump('q_proj', q[0, 0])\n");
    printf("    dump('k_proj', k[0, 0])\n");
    printf("    dump('v_proj', v[0, 0])\n\n");

    printf("    # 6-7. RoPE\n");
    printf("    q_r = q.view(1, 1, nh, hd).transpose(1, 2)\n");
    printf("    k_r = k.view(1, 1, nkv, hd).transpose(1, 2)\n");
    printf("    pos_ids = torch.tensor([[0]])\n");
    printf("    # Get cos/sin — try rotary_emb on layer's attn, fall back to model\n");
    printf("    try:\n");
    printf("        cos, sin = attn.rotary_emb(v, position_ids=pos_ids)\n");
    printf("    except (AttributeError, TypeError):\n");
    printf("        try:\n");
    printf("            cos, sin = m.model.rotary_emb(v, position_ids=pos_ids)\n");
    printf("        except TypeError:\n");
    printf("            cos, sin = m.model.rotary_emb(v, seq_len=1)\n");
    printf("    q_rope, k_rope = apply_rotary_pos_emb(q_r, k_r, cos, sin)\n");
    printf("    dump('q_rope', q_rope.transpose(1, 2).reshape(1, 1, -1)[0, 0])\n");
    printf("    dump('k_rope', k_rope.transpose(1, 2).reshape(1, 1, -1)[0, 0])\n\n");

    printf("    # 8. Attention output (pos=0 → single token)\n");
    printf("    k_exp = repeat_kv(k_rope, gqa)\n");
    printf("    v_r = v.view(1, 1, nkv, hd).transpose(1, 2)\n");
    printf("    v_exp = repeat_kv(v_r, gqa)\n");
    printf("    aw = torch.matmul(q_rope, k_exp.transpose(2, 3)) / math.sqrt(hd)\n");
    printf("    aw = F.softmax(aw, dim=-1)\n");
    printf("    ao = torch.matmul(aw, v_exp)\n");
    printf("    ao_flat = ao.transpose(1, 2).reshape(1, 1, -1)\n");
    printf("    dump('attn_out', ao_flat[0, 0])\n\n");

    printf("    # 9. Output projection\n");
    printf("    o_out = attn.o_proj(ao_flat)\n");
    printf("    dump('o_proj', o_out[0, 0])\n\n");

    printf("    # 10. First residual\n");
    printf("    post_attn = x + o_out\n");
    printf("    dump('post_attn', post_attn[0, 0])\n\n");

    printf("    # 11. Post-attention norm\n");
    printf("    pan = layer.post_attention_layernorm(post_attn)\n");
    printf("    dump('post_attn_norm', pan[0, 0])\n\n");

    printf("    # 12. MLP\n");
    printf("    mlp_out = layer.mlp(pan)\n");
    printf("    dump('mlp_out', mlp_out[0, 0])\n\n");

    printf("    # 13. Final layer output\n");
    printf("    layer_out = post_attn + mlp_out\n");
    printf("    dump('layer_out', layer_out[0, 0])\n\n");

    printf("    # Backward compat: also write layer0_reference.bin\n");
    printf("    dump('reference', layer_out[0, 0])\n");
    printf("    import shutil\n");
    printf("    shutil.copy(os.path.join(OUT, 'layer0_ref_reference.bin'),\n");
    printf("                os.path.join(OUT, 'layer0_reference.bin'))\n\n");

    printf("print(f'\\nAll layer-0 intermediates written to {OUT}/')\n\n");

    /* Full-forward reference: run all layers on BOS, save final hidden + top logits */
    printf("# === Full-Forward Reference (R1.2) ===\n");
    printf("print('\\nRunning full forward pass on BOS...')\n");
    printf("with torch.no_grad():\n");
    printf("    bos = torch.tensor([[1]])\n");
    printf("    out = m(bos, output_hidden_states=True)\n");
    printf("    # out.hidden_states[-1] is already post-final-RMSNorm in\n");
    printf("    # HuggingFace Transformers (LlamaModel applies model.norm\n");
    printf("    # before returning, then appends to hidden_states list).\n");
    printf("    # Do NOT apply m.model.norm() again — that would double-norm.\n");
    printf("    normed = out.hidden_states[-1]  # [1, 1, dim] — already normed\n");
    printf("    vals = normed[0, 0].float().tolist()\n");
    printf("    path = os.path.join(OUT, 'full_ref_final_hidden.bin')\n");
    printf("    with open(path, 'wb') as f:\n");
    printf("        for v in vals:\n");
    printf("            f.write(struct.pack('<f', v))\n");
    printf("    print(f'  final_hidden: {len(vals)} floats -> {path}')\n\n");

    printf("    # Top-5 logits\n");
    printf("    logits = out.logits[0, 0]  # [vocab_size]\n");
    printf("    top5 = torch.topk(logits, 5)\n");
    printf("    path = os.path.join(OUT, 'full_ref_logits_top5.bin')\n");
    printf("    with open(path, 'wb') as f:\n");
    printf("        for idx, val in zip(top5.indices.tolist(), top5.values.tolist()):\n");
    printf("            f.write(struct.pack('<i', idx))\n");
    printf("            f.write(struct.pack('<f', val))\n");
    printf("    print(f'  top5 logits -> {path}')\n");
    printf("    for i, (idx, val) in enumerate(zip(top5.indices.tolist(), top5.values.tolist())):\n");
    printf("        print(f'    #{i+1}: token {idx:5d}  logit={val:.6f}')\n\n");

    /* Multi-position reference: run PROMPT_FRANCE through all layers at each position */
    printf("# === Multi-Position Reference (R1.3) ===\n");
    printf("print('\\nRunning multi-position forward pass on PROMPT_FRANCE...')\n");
    printf("prompt_france = [1, 450, 7483, 310, 3444, 338]\n");
    printf("with torch.no_grad():\n");
    printf("    inp = torch.tensor([prompt_france])\n");
    printf("    out = m(inp, output_hidden_states=True)\n");
    printf("    for pos in range(len(prompt_france)):\n");
    printf("        hidden = out.hidden_states[-1][0, pos].float().tolist()\n");
    printf("        path = os.path.join(OUT, f'multipos_ref_pos{pos}_hidden.bin')\n");
    printf("        with open(path, 'wb') as f:\n");
    printf("            for v in hidden:\n");
    printf("                f.write(struct.pack('<f', v))\n");
    printf("        print(f'  pos {pos}: {len(hidden)} floats -> {path}')\n");
    printf("        logits_pos = out.logits[0, pos].float()\n");
    printf("        top5 = torch.topk(logits_pos, 5)\n");
    printf("        path = os.path.join(OUT, f'multipos_ref_pos{pos}_top5.bin')\n");
    printf("        with open(path, 'wb') as f:\n");
    printf("            for idx, val in zip(top5.indices.tolist(), top5.values.tolist()):\n");
    printf("                f.write(struct.pack('<i', idx))\n");
    printf("                f.write(struct.pack('<f', val))\n");
    printf("        print(f'  pos {pos}: top5 -> {path}')\n");
    printf("        for i, (idx, val) in enumerate(zip(top5.indices.tolist(), top5.values.tolist())):\n");
    printf("            print(f'    #{i+1}: token {idx:5d}  logit={val:.6f}')\n");
    printf("print('Multi-position references written.')\n\n");

    printf("# Token reference\n");
    printf("prompts = {\n");
    printf("    'france_capital': [1,450,7483,310,3444,338],\n");
    printf("    'story_beginning': [1,9038,2501,263,931],\n");
    printf("    'simple_math': [1,29896,29974,29896,29922],\n");
    printf("    'meaning_of_life': [1,450,6593,310,2834,338],\n");
    printf("}\n");
    printf("with open(os.path.join(OUT, 'reference_tokens.txt'), 'w') as f:\n");
    printf("    for name, toks in prompts.items():\n");
    printf("        inp = torch.tensor([toks])\n");
    printf("        out = m.generate(inp, max_new_tokens=20, do_sample=False)\n");
    printf("        new_toks = out[0][len(toks):].tolist()\n");
    printf("        f.write(name + ' ' + ' '.join(map(str, new_toks)) + '\\n')\n");
    printf("        print(f'{name}: {new_toks}')\n");
    printf("print('Wrote reference_tokens.txt')\n");
}

/* ================================================================== */
/*  Main                                                                */
/* ================================================================== */

static void usage(const char *prog) {
    printf("Usage: %s <model_dir> [options]\n", prog);
    printf("\n");
    printf("  <model_dir>       Directory with safetensors shards (or .mgw file with --native)\n");
    printf("\n");
    printf("Options:\n");
    printf("  --verify          Layer-0 structural verification + reference comparison (default)\n");
    printf("  --generate        Generate text from preset prompts\n");
    printf("  --prompt TEXT     Tokenize TEXT with local tokenizer and generate a reply\n");
    printf("  --max-new-tokens N  Number of tokens to generate (default 20)\n");
    printf("  --benchmark       Full quality benchmark with token comparison\n");
    printf("  --dump-reference  Print Python script to generate reference data\n");
    printf("  --cache-layers    Pre-load all layers into RAM (high memory)\n");
    printf("\n");
    printf("Native weight format (R9/R9B):\n");
    printf("  --export-native PATH  Export model weights to native .mgw format\n");
    printf("  --native              Load from native .mgw file via mmap (all layers resident)\n");
    printf("  --native-stream       Load from native .mgw file, streaming layers on demand (low RAM)\n");
    printf("  --ref-dir DIR         Reference file directory for --verify/--benchmark with --native/--native-stream\n");
    printf("\n");
    printf("Compressed native format (R9C-A):\n");
    printf("  --export-native-compressed PATH  Export to compressed .mgwc format (dtype-native, ~4:1)\n");
    printf("  --native-compressed              Load from .mgwc file (decompresses dtype→Q16.48)\n");
    printf("\n");
    printf("KV compression (R9C-B):\n");
    printf("  --kv-quantize KnVn               Runtime lossy KV quantize-in-place (e.g. K16V4, K0V4)\n");
    printf("                                   K0 = exact K, V0 = exact V. Experimental — changes output!\n");
    printf("  --kv-packed KnVn                 Packed KV cache: uint8 K + uint16 V codes (e.g. K8V10)\n");
    printf("                                   Real memory savings. Dequantize on read. Experimental!\n");
    printf("\n");
    printf("Profiling (R2):\n");
    printf("  --profile             Print timing and memory summary with KPI lines\n");
    printf("  --profile-json PATH   Write JSON profile to PATH\n");
    printf("\n");
    printf("Sampling (--generate only, greedy by default):\n");
    printf("  --temperature F   Temperature for logit scaling (e.g. 0.8)\n");
    printf("  --top-k N         Keep only top-K tokens before sampling (e.g. 40)\n");
    printf("  --top-p F         Nucleus sampling threshold (e.g. 0.9)\n");
    printf("  --seed N          PRNG seed for reproducible sampling (e.g. 42)\n");
    printf("\n");
    printf("Memory requirements (depend on model size):\n");
    printf("  Streaming (default):   one layer + embeddings + KV-cache (safetensors I/O)\n");
    printf("  --cache-layers:        all layers + embeddings + KV-cache\n");
    printf("  --native:              all layers via mmap + embeddings + KV-cache\n");
    printf("  --native-stream:       one layer + embeddings + KV-cache (native I/O, no conversion)\n");
    printf("\n");
    printf("Reference-based validation:\n");
    printf("  1. Run: %s <model_dir> --dump-reference > gen_ref.py\n", prog);
    printf("  2. Run: python3 gen_ref.py  (generates layer0_reference.bin, reference_tokens.txt)\n");
    printf("  3. Run: %s <model_dir> --verify     (compares against reference)\n", prog);
    printf("  4. Run: %s <model_dir> --benchmark  (compares generated tokens)\n", prog);
    printf("\n");
    printf("Text prompt mode:\n");
    printf("  %s <model_dir> --generate --prompt \"What is the capital of France?\"\n", prog);
    printf("  Uses local tokenizer assets plus venv/python tokenizer bridge.\n");
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }

    const char *model_dir = argv[1];
    char exec_dir[PATH_MAX];
    resolve_exec_dir(argv[0], exec_dir, sizeof(exec_dir));

    /* Check for --dump-reference early (no model loading needed) */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--dump-reference") == 0) {
            dump_reference_script(model_dir);
            return 0;
        }
    }
    const char *mode = "--verify";
    const char *prompt_text = NULL;
    int cache_layers = 0;
    int max_new_tokens = 20;

    /* Native weight format (R9/R9B) */
    const char *export_native_path = NULL;
    const char *ref_dir = NULL;   /* override for reference file directory */
    int use_native = 0;
    int use_native_stream = 0;

    /* Compressed native format (R9C-A) */
    const char *export_compressed_path = NULL;
    int use_native_compressed = 0;
    int kv_quantize_flag_set = 0;  /* track if --kv-quantize was passed */
    int kv_packed_flag_set = 0;    /* track if --kv-packed was passed */

    /* Profiling (R2) */
    const char *profile_json_path = NULL;

    /* Sampling parameters (--generate only) */
    double cli_temperature = 0.0;  /* 0 → greedy */
    int    cli_top_k       = 0;    /* 0 → disabled */
    double cli_top_p       = 1.0;  /* 1.0 → disabled */
    int    cli_seed_set    = 0;
    uint64_t cli_seed      = 0;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--cache-layers") == 0) {
            cache_layers = 1;
        } else if (strcmp(argv[i], "--prompt") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "--prompt requires a text argument\n");
                usage(argv[0]);
                return 1;
            }
            prompt_text = argv[++i];
            mode = "--generate";
        } else if (strcmp(argv[i], "--max-new-tokens") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "--max-new-tokens requires an integer argument\n");
                usage(argv[0]);
                return 1;
            }
            char *end;
            max_new_tokens = (int)strtol(argv[++i], &end, 10);
            if (end == argv[i] || *end != '\0' || max_new_tokens <= 0) {
                fprintf(stderr, "--max-new-tokens requires a positive integer\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--temperature") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "--temperature requires a float argument\n");
                return 1;
            }
            char *end;
            cli_temperature = strtod(argv[++i], &end);
            if (end == argv[i] || *end != '\0' || cli_temperature < 0.0) {
                fprintf(stderr, "--temperature requires a non-negative number\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--top-k") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "--top-k requires an integer argument\n");
                return 1;
            }
            char *end;
            cli_top_k = (int)strtol(argv[++i], &end, 10);
            if (end == argv[i] || *end != '\0' || cli_top_k < 0) {
                fprintf(stderr, "--top-k requires a non-negative integer\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--top-p") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "--top-p requires a float argument\n");
                return 1;
            }
            char *end;
            cli_top_p = strtod(argv[++i], &end);
            if (end == argv[i] || *end != '\0' || cli_top_p <= 0.0 || cli_top_p > 1.0) {
                fprintf(stderr, "--top-p requires a number in (0, 1.0]\n");
                return 1;
            }
        } else if (strcmp(argv[i], "--seed") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "--seed requires an integer argument\n");
                return 1;
            }
            char *end;
            cli_seed = (uint64_t)strtoull(argv[++i], &end, 10);
            if (end == argv[i] || *end != '\0') {
                fprintf(stderr, "--seed requires a non-negative integer\n");
                return 1;
            }
            cli_seed_set = 1;
        } else if (strcmp(argv[i], "--export-native") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "--export-native requires an output file path\n");
                return 1;
            }
            export_native_path = argv[++i];
        } else if (strcmp(argv[i], "--native") == 0) {
            use_native = 1;
        } else if (strcmp(argv[i], "--native-stream") == 0) {
            use_native_stream = 1;
        } else if (strcmp(argv[i], "--export-native-compressed") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "--export-native-compressed requires an output file path\n");
                return 1;
            }
            export_compressed_path = argv[++i];
        } else if (strcmp(argv[i], "--native-compressed") == 0) {
            use_native_compressed = 1;
        } else if (strcmp(argv[i], "--kv-quantize") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "--kv-quantize requires format KbitsVbits (e.g. K16V4)\n");
                return 1;
            }
            const char *spec = argv[++i];
            int kb = 0, vb = 0;
            if (sscanf(spec, "K%dV%d", &kb, &vb) != 2 || kb < 0 || vb < 0 || kb > 63 || vb > 63) {
                fprintf(stderr, "--kv-quantize: invalid format '%s', expected KnVn (e.g. K16V4, K0V4)\n", spec);
                return 1;
            }
            g_kv_quant.enabled = 1;
            g_kv_quant.k_bits = kb;
            g_kv_quant.v_bits = vb;
            kv_quantize_flag_set = 1;
        } else if (strcmp(argv[i], "--kv-packed") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "--kv-packed requires format KbitsVbits (e.g. K8V10)\n");
                return 1;
            }
            const char *spec = argv[++i];
            int kb = 0, vb = 0;
            if (sscanf(spec, "K%dV%d", &kb, &vb) != 2 || kb < 1 || vb < 1
                || kb > 8 || vb > 16) {
                fprintf(stderr, "--kv-packed: invalid '%s', expected KnVn "
                        "(K: 1-8 bits, V: 1-16 bits, e.g. K8V10)\n", spec);
                return 1;
            }
            g_kv_quant.enabled = 1;
            g_kv_quant.packed = 1;
            g_kv_quant.k_bits = kb;
            g_kv_quant.v_bits = vb;
            kv_packed_flag_set = 1;
        } else if (strcmp(argv[i], "--ref-dir") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "--ref-dir requires a directory path\n");
                return 1;
            }
            ref_dir = argv[++i];
        } else if (strcmp(argv[i], "--profile") == 0) {
            g_profile.enabled = 1;
        } else if (strcmp(argv[i], "--profile-json") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "--profile-json requires a file path\n");
                return 1;
            }
            profile_json_path = argv[++i];
            g_profile.enabled = 1;
        } else if (argv[i][0] == '-') {
            mode = argv[i];
        }
    }

    /* Build sampling params if any sampling option was set */
    sample_params_t sampling;
    sample_params_t *sp = NULL;
    if (cli_temperature > 0.0) {
        memset(&sampling, 0, sizeof(sampling));
        /* CLI boundary: float→fixed conversion isolated here */
        sampling.temperature = (fixed_t)(cli_temperature * (double)FP_ONE);
        sampling.top_k       = cli_top_k;
        sampling.top_p       = (fixed_t)(cli_top_p * (double)FP_ONE);
        if (cli_seed_set) {
            rng_seed(&sampling.rng, cli_seed);
        } else {
            /* Use time-based seed, print it for reproducibility */
            struct timespec ts;
            clock_gettime(CLOCK_MONOTONIC, &ts);
            cli_seed = (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
            rng_seed(&sampling.rng, cli_seed);
        }
        sp = &sampling;
    } else if (cli_top_k > 0 || cli_top_p < 1.0 || cli_seed_set) {
        fprintf(stderr, "WARNING: --top-k, --top-p, and --seed require --temperature > 0 to take effect\n");
    }

    printf("================================================================\n");
    printf("  Integer-Only Llama Inference\n");
    printf("  Q16.48 fixed-point — ZERO floating-point operations\n");
    printf("================================================================\n\n");

    if (sp) {
        printf("Sampling: temperature=%.2f top_k=%d top_p=%.2f seed=%llu\n\n",
               fp_to_double(sp->temperature), sp->top_k,
               fp_to_double(sp->top_p), (unsigned long long)cli_seed);
    }

    llama_model_t model;
    memset(&model, 0, sizeof(model));

    fp_math_init();
    printf("fp_math initialized (pi = "); fp_print(FP_PI, 12); printf(")\n");

    struct timespec tp0, tp1; /* profiling timestamps */

    /* Mutually exclusive native modes */
    if (use_native && use_native_stream) {
        fprintf(stderr, "Cannot use both --native and --native-stream\n");
        return 1;
    }
    if (use_native_stream && cache_layers) {
        fprintf(stderr, "Cannot use --cache-layers with --native-stream "
                "(streaming is the point)\n");
        return 1;
    }
    if (use_native_stream && export_native_path) {
        fprintf(stderr, "Cannot use --export-native with --native-stream "
                "(already native)\n");
        return 1;
    }
    if (use_native_compressed && (use_native || use_native_stream)) {
        fprintf(stderr, "Cannot combine --native-compressed with --native or --native-stream\n");
        return 1;
    }
    if (use_native_compressed && export_native_path) {
        fprintf(stderr, "Cannot use --export-native with --native-compressed\n");
        return 1;
    }
    if (export_compressed_path && (use_native || use_native_stream || use_native_compressed)) {
        fprintf(stderr, "Cannot use --export-native-compressed with --native/--native-stream/--native-compressed\n");
        return 1;
    }
    if (export_compressed_path && export_native_path) {
        fprintf(stderr, "Cannot use both --export-native and --export-native-compressed\n");
        return 1;
    }
    if (kv_packed_flag_set && kv_quantize_flag_set) {
        fprintf(stderr, "Cannot use both --kv-packed and --kv-quantize\n");
        return 1;
    }

    model.native_stream_fd = -1;  /* sentinel: no fd open */

    if (use_native_compressed) {
        /* ============================================================ */
        /*  Compressed native path (R9C-A) — dtype→Q16.48 on load      */
        /* ============================================================ */
        printf("Loading compressed native weight file: %s\n", model_dir);
        if (g_profile.enabled) clock_gettime(CLOCK_MONOTONIC, &tp0);
        if (mgwc_load(&model, model_dir) != 0) {
            fprintf(stderr, "Failed to load compressed native weights from %s\n",
                    model_dir);
            return 1;
        }
        if (g_profile.enabled) {
            clock_gettime(CLOCK_MONOTONIC, &tp1);
            g_profile.t_safetensors_load = ts_diff(tp0, tp1);
            g_profile.native_loaded = 0;
            g_profile.layers_cached = 1;
            g_profile.native_compressed = 1;
        }

        printf("Compressed native weights loaded: %d tensors\n",
               model.lm_head_tied
               ? 2 + model.cfg.num_layers * 9
               : 3 + model.cfg.num_layers * 9);

    } else if (use_native_stream) {
        /* ============================================================ */
        /*  Native streaming path (R9B) — low-memory .mgw access        */
        /* ============================================================ */
        printf("Loading native weight file (stream mode): %s\n", model_dir);
        if (g_profile.enabled) clock_gettime(CLOCK_MONOTONIC, &tp0);
        if (mgw_load_streaming(&model, model_dir) != 0) {
            fprintf(stderr, "Failed to load native weights (stream) from %s\n",
                    model_dir);
            return 1;
        }
        if (g_profile.enabled) {
            clock_gettime(CLOCK_MONOTONIC, &tp1);
            g_profile.t_safetensors_load = ts_diff(tp0, tp1);
            g_profile.t_global_weights = 0;
            g_profile.native_stream = 1;
        }
        printf("\n");
    } else if (use_native) {
        /* ============================================================ */
        /*  Native weight path (R9) — load from .mgw file               */
        /* ============================================================ */
        printf("Loading native weight file: %s\n", model_dir);
        if (g_profile.enabled) clock_gettime(CLOCK_MONOTONIC, &tp0);
        if (mgw_load(&model, model_dir) != 0) {
            fprintf(stderr, "Failed to load native weights from %s\n", model_dir);
            return 1;
        }
        if (g_profile.enabled) {
            clock_gettime(CLOCK_MONOTONIC, &tp1);
            /* Native load replaces both safetensors_load and global_weights */
            g_profile.t_safetensors_load = ts_diff(tp0, tp1);
            g_profile.t_global_weights = 0;
            g_profile.native_loaded = 1;
        }
        printf("\n");

        /* --export-native is incompatible with --native */
        if (export_native_path) {
            fprintf(stderr, "Cannot use --export-native with --native (already native)\n");
            munmap(model.native_mmap, model.native_mmap_size);
            free(model.cached_layers);
            return 1;
        }
    } else {
        /* ============================================================ */
        /*  Safetensors path (baseline/reference)                        */
        /* ============================================================ */

        /* Load config.json if available, fall back to 7B defaults */
        int cfg_result = load_config_json(model_dir, &model.cfg);
        if (cfg_result == 0) {
            printf("Loaded config.json:\n");
            print_config(&model.cfg);
        } else if (cfg_result == -2) {
            return 1;
        } else {
            printf("WARNING: No config.json in %s — using Llama-2-7B defaults\n", model_dir);
            model.cfg = LLAMA_7B_CONFIG;
            print_config(&model.cfg);
        }
        printf("\n");

        /* Load safetensors FIRST — validate model before allocating RAM */
        st_model_init(&model.st_model);
        char path[512];
        int shards_loaded = 0;

        if (g_profile.enabled) clock_gettime(CLOCK_MONOTONIC, &tp0);

        /* Try single-file format first (most common for small models) */
        snprintf(path, sizeof(path), "%s/model.safetensors", model_dir);
        if (st_model_load_shard(&model.st_model, path) == 0)
            shards_loaded++;

        /* Try multi-shard format: model-00001-of-NNNNN.safetensors */
        if (shards_loaded == 0) {
            for (int total = 2; total <= 8 && shards_loaded == 0; total++) {
                snprintf(path, sizeof(path), "%s/model-00001-of-%05d.safetensors",
                         model_dir, total);
                if (st_model_load_shard(&model.st_model, path) != 0)
                    continue;
                shards_loaded = 1;
                for (int s = 2; s <= total; s++) {
                    snprintf(path, sizeof(path), "%s/model-%05d-of-%05d.safetensors",
                             model_dir, s, total);
                    if (st_model_load_shard(&model.st_model, path) == 0)
                        shards_loaded++;
                    else
                        fprintf(stderr, "WARNING: missing shard %d of %d\n", s, total);
                }
                break;
            }
        }

        if (shards_loaded == 0) {
            fprintf(stderr, "Error: No safetensors files found in %s\n", model_dir);
            return 1;
        }
        if (g_profile.enabled) {
            clock_gettime(CLOCK_MONOTONIC, &tp1);
            g_profile.t_safetensors_load = ts_diff(tp0, tp1);
        }
        printf("Loaded %d shard(s), %d tensors total\n",
               model.st_model.num_shards, model.st_model.num_tensors);

        /* Load global weights (with shape validation) */
        printf("Loading global weights...\n");
        if (g_profile.enabled) clock_gettime(CLOCK_MONOTONIC, &tp0);
        if (llama_load_global_weights(&model) != 0) {
            fprintf(stderr, "Failed to load global weights\n");
            st_model_free(&model.st_model);
            return 1;
        }
        if (g_profile.enabled) {
            clock_gettime(CLOCK_MONOTONIC, &tp1);
            g_profile.t_global_weights = ts_diff(tp0, tp1);
        }
        printf("Global weights loaded\n");

        /* Handle --export-native (export then exit) */
        if (export_native_path) {
            printf("\n--- Exporting to native format ---\n");
            int rc = mgw_export(&model, export_native_path);
            /* Cleanup safetensors path */
            if (!model.lm_head_tied) free(model.lm_head);
            free(model.embed_tokens);
            free(model.final_norm);
            st_model_free(&model.st_model);
            printf("\nDone.\n");
            return rc != 0 ? 1 : 0;
        }

        /* Handle --export-native-compressed (export then exit) */
        if (export_compressed_path) {
            printf("\n--- Exporting to compressed native format ---\n");
            int rc = mgwc_export(&model, export_compressed_path);
            /* Cleanup safetensors path */
            if (!model.lm_head_tied) free(model.lm_head);
            free(model.embed_tokens);
            free(model.final_norm);
            st_model_free(&model.st_model);
            printf("\nDone.\n");
            return rc != 0 ? 1 : 0;
        }
    }

    /* Compute memory accounting (only when profiling) */
    if (g_profile.enabled) {
        const llama_config_t *c = &model.cfg;
        g_profile.mem_embed_tokens = (size_t)c->vocab_size * c->hidden_dim * sizeof(fixed_t);
        g_profile.mem_lm_head = model.lm_head_tied ? 0
            : (size_t)c->vocab_size * c->hidden_dim * sizeof(fixed_t);
        g_profile.mem_final_norm = (size_t)c->hidden_dim * sizeof(fixed_t);
        int kv_dim = c->num_kv_heads * c->head_dim;
        size_t one_layer = ((size_t)c->hidden_dim
                          + (size_t)c->hidden_dim * c->hidden_dim
                          + (size_t)kv_dim * c->hidden_dim
                          + (size_t)kv_dim * c->hidden_dim
                          + (size_t)c->hidden_dim * c->hidden_dim
                          + (size_t)c->hidden_dim
                          + (size_t)c->intermediate_dim * c->hidden_dim
                          + (size_t)c->intermediate_dim * c->hidden_dim
                          + (size_t)c->hidden_dim * c->intermediate_dim
                          ) * sizeof(fixed_t);
        g_profile.mem_one_layer = one_layer;
    }

    /* Allocate RoPE tables and KV-cache (shared by both paths) */
    printf("Initializing RoPE tables...\n");
    clock_gettime(CLOCK_MONOTONIC, &tp0);
    rope_init(&model.rope, &model.cfg);
    clock_gettime(CLOCK_MONOTONIC, &tp1);
    {
        double rope_sec = ts_diff(tp0, tp1);
        printf("  RoPE tables ready (%.2f seconds)\n", rope_sec);
        if (g_profile.enabled) {
            g_profile.t_rope_init = rope_sec;
            g_profile.mem_rope_tables = 2 * (size_t)model.cfg.max_seq_len
                                          * (model.cfg.head_dim / 2)
                                          * sizeof(fixed_t);
        }
    }

    /* KV-cache */
    if (g_profile.enabled) clock_gettime(CLOCK_MONOTONIC, &tp0);
    if (kv_cache_init(&model.kv, &model.cfg) != 0) {
        rope_free(&model.rope);
        /* Error cleanup depends on load path */
        if (model.native_loaded) {
            free(model.cached_layers);
            munmap(model.native_mmap, model.native_mmap_size);
        } else if (model.native_stream) {
            if (!model.lm_head_tied) free(model.lm_head);
            free(model.embed_tokens);
            free(model.final_norm);
            free(model.ns_index);
            if (model.native_stream_fd >= 0) close(model.native_stream_fd);
        } else {
            if (!model.lm_head_tied) free(model.lm_head);
            free(model.embed_tokens);
            free(model.final_norm);
            st_model_free(&model.st_model);
        }
        return 1;
    }
    /* Pre-allocate layer scratch buffers (R7) — non-fatal if this fails */
    llama_init_scratch(&model);

    if (g_profile.enabled) {
        clock_gettime(CLOCK_MONOTONIC, &tp1);
        g_profile.t_kv_alloc = ts_diff(tp0, tp1);
        size_t per_layer = (size_t)model.cfg.max_seq_len * model.cfg.num_kv_heads
                         * model.cfg.head_dim;
        if (model.kv.pk_key_codes) {
            /* Packed mode: report actual packed storage, not exact cache */
            size_t mag_elems = (size_t)model.cfg.num_layers * model.cfg.max_seq_len
                             * model.cfg.num_kv_heads;
            g_profile.mem_kv_cache = per_layer * model.cfg.num_layers * sizeof(uint8_t)  /* K codes */
                                   + per_layer * model.cfg.num_layers * sizeof(uint16_t) /* V codes */
                                   + mag_elems * sizeof(int64_t) * 2;                    /* K+V magnitudes */
        } else {
            g_profile.mem_kv_cache = 2 * per_layer * model.cfg.num_layers * sizeof(fixed_t);
        }
    }

    /* Layer caching strategy */
    if (model.native_stream) {
        /* Native stream (R9B): layers read on demand from .mgw fd */
        printf("Native stream path: layers read on demand from .mgw (low memory)\n");
        model.layers_cached = 0;
        model.cached_layers = NULL;
        /* profiling: no cached layers, one-layer working set only */
    } else if (model.native_loaded) {
        /* Native mmap path: layers already "cached" via mmap — nothing to do */
        printf("Native mmap path: all layers available via mmap (OS-managed paging)\n");
        if (g_profile.enabled) {
            g_profile.mem_cached_layers = g_profile.mem_one_layer * model.cfg.num_layers;
            g_profile.layers_cached = 1;
        }
    } else if (use_native_compressed && model.layers_cached) {
        /* Compressed native (R9C-A): layers already decompressed and cached */
        printf("Compressed native path: all layers decompressed to Q16.48 in memory\n");
        if (g_profile.enabled) {
            g_profile.mem_cached_layers = g_profile.mem_one_layer * model.cfg.num_layers;
            g_profile.layers_cached = 1;
        }
    } else if (cache_layers) {
        if (g_profile.enabled) clock_gettime(CLOCK_MONOTONIC, &tp0);
        if (llama_cache_all_layers(&model) != 0) {
            fprintf(stderr, "Failed to cache layers (not enough RAM)\n");
            fprintf(stderr, "Try running without --cache-layers (streaming mode)\n");
            kv_cache_free(&model.kv);
            rope_free(&model.rope);
            if (!model.lm_head_tied) free(model.lm_head);
            free(model.embed_tokens);
            free(model.final_norm);
            st_model_free(&model.st_model);
            return 1;
        }
        if (g_profile.enabled) {
            clock_gettime(CLOCK_MONOTONIC, &tp1);
            g_profile.t_layer_cache = ts_diff(tp0, tp1);
            g_profile.mem_cached_layers = g_profile.mem_one_layer * model.cfg.num_layers;
            g_profile.layers_cached = 1;
        }
    } else {
        printf("Running in streaming mode (per-layer load/convert/free)\n");
        model.layers_cached = 0;
        model.cached_layers = NULL;
    }

    if (g_kv_quant.enabled) {
        if (g_kv_quant.packed) {
            printf("\n*** EXPERIMENTAL: Packed KV cache (R9C-B4) ***\n");
            printf("  K: %d-bit codes (uint8_t) + int64_t magnitude per vector\n",
                   g_kv_quant.k_bits);
            printf("  V: %d-bit codes (uint16_t) + int64_t magnitude per vector\n",
                   g_kv_quant.v_bits);
            printf("  Attention reads dequantize from packed storage.\n");
            printf("  WARNING: Experimental — output may differ from exact baseline!\n\n");
        } else {
            printf("\n*** EXPERIMENTAL: Runtime KV quantize-in-place (R9C-B3) ***\n");
            printf("  K bits: %d (%s)\n", g_kv_quant.k_bits,
                   g_kv_quant.k_bits == 0 ? "exact" : "lossy");
            printf("  V bits: %d (%s)\n", g_kv_quant.v_bits,
                   g_kv_quant.v_bits == 0 ? "exact" : "lossy");
            printf("  WARNING: This modifies KV cache values — output may differ from exact baseline!\n\n");
        }
    }

    profile_init_layers(model.cfg.num_layers);

    int run_status = 0;

    /* Determine reference file directory.
     * For safetensors path: same as model_dir (HuggingFace snapshot).
     * For native path: must be specified via --ref-dir since model_dir is a .mgw file. */
    const char *effective_ref_dir = ref_dir ? ref_dir : model_dir;
    if ((use_native || use_native_stream || use_native_compressed) && !ref_dir) {
        if (strcmp(mode, "--verify") == 0 || strcmp(mode, "--benchmark") == 0)
            printf("NOTE: --native without --ref-dir — looking for reference files alongside %s\n",
                   model_dir);
    }

    /* Run requested mode */
    if (strcmp(mode, "--verify") == 0) {
        int vf = verify_single_layer(&model, effective_ref_dir);
        int vff = verify_full_forward(&model, effective_ref_dir);
        int vmp = verify_multi_position(&model, effective_ref_dir);
        int vtg = verify_token_agreement(&model, effective_ref_dir);
        if (vf > 0 || vff > 0 || vmp > 0 || vtg > 0)
            run_status = 1;
    } else if (strcmp(mode, "--generate") == 0) {
        if (prompt_text) {
            int *prompt_tokens = NULL;
            int prompt_len = 0;
            if (tokenize_text_prompt(exec_dir, effective_ref_dir, prompt_text,
                                     &prompt_tokens, &prompt_len) != 0) {
                fprintf(stderr, "Failed to tokenize custom prompt\n");
                run_status = 1;
            } else {
                int *tokens = (int *)checked_malloc(
                    (size_t)(prompt_len + max_new_tokens) * sizeof(int), "custom_gen_tokens");
                printf("\n--- Prompt ---\n%s\n", prompt_text);
                printf("  Prompt tokenized to %d tokens\n\n", prompt_len);
                int total_len = generate(&model, prompt_tokens, prompt_len,
                                         max_new_tokens, tokens, sp);
                if (total_len > prompt_len) {
                    char *decoded = decode_generated_tokens(exec_dir, effective_ref_dir,
                                                            tokens + prompt_len,
                                                            total_len - prompt_len);
                    if (decoded) {
                        printf("  Text: %s\n", decoded);
                        free(decoded);
                    } else {
                        printf("  Text decode unavailable (tokenizer helper failed)\n");
                    }
                }
                free(tokens);
                free(prompt_tokens);
            }
        } else {
            int *tokens = (int *)checked_malloc(
                (PROMPT_FRANCE_LEN + max_new_tokens) * sizeof(int), "default_gen_tokens");
            printf("\n--- \"The capital of France is\" ---\n");
            generate(&model, PROMPT_FRANCE, PROMPT_FRANCE_LEN, max_new_tokens, tokens, sp);
            free(tokens);
        }
    } else if (strcmp(mode, "--benchmark") == 0) {
        if (quality_benchmark(&model, effective_ref_dir) != 0)
            run_status = 1;
    } else {
        fprintf(stderr, "Unknown mode: %s\n", mode);
        usage(argv[0]);
        run_status = 1;
    }

    /* Profile output (before cleanup) */
    if (g_profile.enabled) {
        profile_print_summary(&g_profile, model_dir);
        if (profile_json_path)
            profile_write_json(&g_profile, profile_json_path, model_dir);
    }

    /* Cleanup */
    free(model.scratch_buf);
    kv_cache_free(&model.kv);
    rope_free(&model.rope);
    if (model.native_loaded) {
        /* Native mmap path: weight pointers reference mmap — only free the
         * index array and munmap the backing region.  Do NOT free weight data. */
        free(model.cached_layers);
        munmap(model.native_mmap, model.native_mmap_size);
    } else if (model.native_stream) {
        /* Native stream path (R9B): global weights are malloc'd, layers are
         * freed after each forward call. Close fd and free index. */
        if (!model.lm_head_tied)
            free(model.lm_head);
        free(model.embed_tokens);
        free(model.final_norm);
        free(model.ns_index);
        if (model.native_stream_fd >= 0)
            close(model.native_stream_fd);
    } else {
        /* Safetensors path: weight data was malloc'd — free each tensor */
        if (!model.lm_head_tied)
            free(model.lm_head);
        free(model.embed_tokens);
        free(model.final_norm);
        if (model.cached_layers) {
            for (int i = 0; i < model.cfg.num_layers; i++)
                free_layer_weights(&model.cached_layers[i]);
            free(model.cached_layers);
        }
        st_model_free(&model.st_model);
    }

    profile_free_layers();
    if (g_tokenizer_loaded) {
        tokenizer_free(&g_tokenizer);
        free((void *)g_tokenizer_model_dir);
    }

    printf("\nDone.\n");
    return run_status;
}
