/*
 * Author: Nenad Mićić
 * LinkedIn: https://be.linkedin.com/in/nenadmicic
 *
 * Copyright (c) 2026 Nenad Mićić
 * SPDX-License-Identifier: Apache-2.0
 *
 * safetensors.h — Header-Only Safetensors Parser for Llama-2 Weights
 * ====================================================================
 *
 * Parses .safetensors files (HuggingFace format) for loading Llama-2-7B
 * weights into integer-only inference engine.
 *
 * Features:
 *   - Minimal JSON parser for safetensors header (flat key→object)
 *   - Memory-mapped file access via mmap()
 *   - Float16/BFloat16 → Q16.48 conversion for model weights
 *   - Multi-shard support (Llama-7B uses 2 shards)
 *
 * ZERO external dependencies beyond POSIX (mmap, open, stat).
 *
 * Usage:
 *   st_model_t model;
 *   st_model_init(&model);
 *   st_model_load_shard(&model, "model-00001-of-00002.safetensors");
 *   st_model_load_shard(&model, "model-00002-of-00002.safetensors");
 *   st_tensor_t *t = st_model_find(&model, "model.layers.0.self_attn.q_proj.weight");
 *   int64_t *q1648 = st_tensor_to_q1648(t);  // converted to Q16.48
 *   // ... use weights ...
 *   free(q1648);
 *   st_model_free(&model);
 */

#ifndef SAFETENSORS_H
#define SAFETENSORS_H

#if !defined(_POSIX_C_SOURCE) || _POSIX_C_SOURCE < 200809L
#undef  _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

/* ================================================================== */
/*  Constants                                                           */
/* ================================================================== */

#define ST_MAX_TENSORS   512
#define ST_MAX_SHARDS    4
#define ST_MAX_NAME_LEN  256
#define ST_MAX_DIMS      4

/* Supported dtypes */
typedef enum {
    ST_DTYPE_F16 = 0,
    ST_DTYPE_BF16,
    ST_DTYPE_F32,
    ST_DTYPE_UNKNOWN
} st_dtype_t;

/* ================================================================== */
/*  Float16/BFloat16 → Q16.48 Conversion                               */
/* ================================================================== */

/* IEEE 754 half-precision (float16): 1 sign + 5 exponent + 10 mantissa.
 * Within Q16.48 range, normal model weights preserve their source mantissa. */
static inline int64_t float16_to_q1648(uint16_t bits) {
    int sign = (bits >> 15) & 1;
    int exponent = (bits >> 10) & 0x1F;
    int mantissa = bits & 0x3FF;

    if (exponent == 0 && mantissa == 0) return 0;  /* ±zero */

    if (exponent == 0x1F) {
        /* Inf/NaN → clamp to max representable */
        int64_t val = (int64_t)32000 << 48;
        return sign ? -val : val;
    }

    int64_t result;
    if (exponent == 0) {
        /* Denormalized: value = (-1)^s × 2^(-14) × (0.mantissa)
         * = mantissa × 2^(-14-10) = mantissa × 2^(-24)
         * In Q16.48: mantissa × 2^(48-24) = mantissa × 2^24 */
        result = (int64_t)mantissa << 24;
    } else {
        /* Normalized: value = (-1)^s × 2^(e-15) × (1.mantissa)
         * = (1024 + mantissa) × 2^(e-15-10)
         * = (1024 + mantissa) × 2^(e-25)
         * In Q16.48: (1024 + mantissa) × 2^(e-25+48) = (1024 + mantissa) × 2^(e+23) */
        int shift = exponent + 23;  /* e-25+48 = e+23 */
        int64_t significand = 1024 + mantissa;

        if (shift >= 0 && shift < 64) {
            result = significand << shift;
        } else if (shift < 0 && shift > -64) {
            result = significand >> (-shift);
        } else {
            result = 0;
        }
    }

    return sign ? -result : result;
}

/* BFloat16: 1 sign + 8 exponent + 7 mantissa.
 * Within Q16.48 range, normal model weights preserve their source mantissa. */
static inline int64_t bfloat16_to_q1648(uint16_t bits) {
    int sign = (bits >> 15) & 1;
    int exponent = (bits >> 7) & 0xFF;
    int mantissa = bits & 0x7F;

    if (exponent == 0 && mantissa == 0) return 0;  /* ±zero */

    if (exponent == 0xFF) {
        /* Inf/NaN → clamp */
        int64_t val = (int64_t)32000 << 48;
        return sign ? -val : val;
    }

    int64_t result;
    if (exponent == 0) {
        /* Denormalized: value = (-1)^s × 2^(-126) × (0.mantissa)
         * = mantissa × 2^(-126-7) = mantissa × 2^(-133)
         * In Q16.48: mantissa × 2^(48-133) = mantissa × 2^(-85) → 0 (subnormal) */
        result = 0;
    } else {
        /* Normalized: value = (-1)^s × 2^(e-127) × (1.mantissa)
         * = (128 + mantissa) × 2^(e-127-7)
         * = (128 + mantissa) × 2^(e-134)
         * In Q16.48: (128 + mantissa) × 2^(e-134+48) = (128 + mantissa) × 2^(e-86) */
        int shift = exponent - 86;  /* e-134+48 */
        int64_t significand = 128 + mantissa;

        if (shift >= 0 && shift < 64) {
            result = significand << shift;
        } else if (shift < 0 && shift > -64) {
            result = significand >> (-shift);
        } else {
            result = 0;
        }
    }

    return sign ? -result : result;
}

/* Float32 → Q16.48 for verification/reference tensors. Values outside Q16.48
 * range or below one fixed-point ULP cannot be represented exactly. */
static inline int64_t float32_to_q1648(float f) {
    /* Direct conversion via multiplication */
    double d = (double)f;
    return (int64_t)(d * (double)((int64_t)1 << 48));
}

/* Q16.48 → float32 (for verification only) */
static inline float q1648_to_float32(int64_t q) {
    return (float)((double)q / (double)((int64_t)1 << 48));
}

/* ================================================================== */
/*  Tensor Descriptor                                                   */
/* ================================================================== */

typedef struct {
    char     name[ST_MAX_NAME_LEN];
    st_dtype_t dtype;
    int      ndim;
    int64_t  shape[ST_MAX_DIMS];
    int64_t  num_elements;       /* product of shape dims */
    size_t   data_offset;        /* byte offset within data section */
    size_t   data_size;          /* byte size of raw data */
    int      shard_idx;          /* which shard file contains this tensor */
} st_tensor_t;

/* ================================================================== */
/*  Shard (mmap'd file)                                                 */
/* ================================================================== */

typedef struct {
    int       fd;
    uint8_t  *map;               /* mmap base */
    size_t    file_size;
    uint8_t  *data_start;        /* pointer to start of tensor data */
    size_t    header_size;       /* JSON header size (from 8-byte prefix) */
} st_shard_t;

/* ================================================================== */
/*  Model (collection of shards + tensors)                              */
/* ================================================================== */

typedef struct {
    st_shard_t  shards[ST_MAX_SHARDS];
    int         num_shards;
    st_tensor_t tensors[ST_MAX_TENSORS];
    int         num_tensors;
} st_model_t;

static void st_model_init(st_model_t *m) {
    memset(m, 0, sizeof(*m));
}

/* ================================================================== */
/*  Minimal JSON Parser for Safetensors Header                          */
/* ================================================================== */

/* Skip whitespace (bounded by end pointer) */
static const char *st_skip_ws(const char *p, const char *end) {
    while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
    return p;
}

/* Parse a JSON string, write to buf, return pointer past closing quote */
static const char *st_parse_string(const char *p, const char *end, char *buf, int max_len) {
    if (p >= end || *p != '"') return NULL;
    p++;
    int i = 0;
    while (p < end && *p != '"') {
        if (*p == '\\') {
            p++;
            if (p >= end) return NULL;
        }
        if (i < max_len - 1) buf[i++] = *p;
        p++;
    }
    buf[i] = '\0';
    if (p < end && *p == '"') p++;
    return p;
}

/* Parse an integer from JSON (bounded) */
static const char *st_parse_int(const char *p, const char *end, int64_t *out) {
    int64_t val = 0;
    int neg = 0;
    if (p < end && *p == '-') { neg = 1; p++; }
    while (p < end && *p >= '0' && *p <= '9') {
        val = val * 10 + (*p - '0');
        p++;
    }
    *out = neg ? -val : val;
    return p;
}

/* Parse dtype string → enum */
static st_dtype_t st_parse_dtype(const char *s) {
    if (strcmp(s, "F16") == 0) return ST_DTYPE_F16;
    if (strcmp(s, "BF16") == 0) return ST_DTYPE_BF16;
    if (strcmp(s, "F32") == 0) return ST_DTYPE_F32;
    return ST_DTYPE_UNKNOWN;
}

/* Bytes per element for dtype */
static int st_dtype_size(st_dtype_t dt) {
    switch (dt) {
        case ST_DTYPE_F16:  return 2;
        case ST_DTYPE_BF16: return 2;
        case ST_DTYPE_F32:  return 4;
        default: return 0;
    }
}

/* Parse one tensor entry: "name": {"dtype":"F16","shape":[d1,d2],"data_offsets":[start,end]}
 * p points to the opening quote of the tensor name. All reads bounded by end. */
static const char *st_parse_tensor_entry(const char *p, const char *end,
                                          st_tensor_t *t, int shard_idx) {
    /* Parse name */
    p = st_parse_string(p, end, t->name, ST_MAX_NAME_LEN);
    if (!p) return NULL;

    p = st_skip_ws(p, end);
    if (p >= end || *p != ':') return NULL;
    p = st_skip_ws(p + 1, end);

    if (p >= end || *p != '{') return NULL;
    p = st_skip_ws(p + 1, end);

    t->shard_idx = shard_idx;
    t->ndim = 0;
    t->num_elements = 1;

    /* Parse object fields */
    while (p < end && *p != '}') {
        char key[64];
        p = st_parse_string(p, end, key, sizeof(key));
        if (!p) return NULL;
        p = st_skip_ws(p, end);
        if (p >= end || *p != ':') return NULL;
        p = st_skip_ws(p + 1, end);

        if (strcmp(key, "dtype") == 0) {
            char dtype_str[16];
            p = st_parse_string(p, end, dtype_str, sizeof(dtype_str));
            if (!p) return NULL;
            t->dtype = st_parse_dtype(dtype_str);
        } else if (strcmp(key, "shape") == 0) {
            /* Parse array [d1, d2, ...] */
            if (p >= end || *p != '[') return NULL;
            p = st_skip_ws(p + 1, end);
            t->ndim = 0;
            while (p < end && *p != ']') {
                int64_t dim;
                p = st_parse_int(p, end, &dim);
                if (t->ndim < ST_MAX_DIMS) {
                    t->shape[t->ndim] = dim;
                    t->num_elements *= dim;
                }
                t->ndim++;
                p = st_skip_ws(p, end);
                if (p < end && *p == ',') p = st_skip_ws(p + 1, end);
            }
            if (p < end && *p == ']') p++;
        } else if (strcmp(key, "data_offsets") == 0) {
            /* Parse [start, end] */
            if (p >= end || *p != '[') return NULL;
            p = st_skip_ws(p + 1, end);
            int64_t start_off, end_off;
            p = st_parse_int(p, end, &start_off);
            p = st_skip_ws(p, end);
            if (p < end && *p == ',') p = st_skip_ws(p + 1, end);
            p = st_parse_int(p, end, &end_off);
            p = st_skip_ws(p, end);
            if (p < end && *p == ']') p++;
            t->data_offset = (size_t)start_off;
            t->data_size = (size_t)(end_off - start_off);
        } else {
            /* Skip unknown value — could be string, number, array, object */
            if (p >= end) return NULL;
            if (*p == '"') {
                char dummy[256];
                p = st_parse_string(p, end, dummy, sizeof(dummy));
            } else if (*p == '[') {
                int depth = 1;
                p++;
                while (p < end && depth > 0) {
                    if (*p == '[') depth++;
                    else if (*p == ']') depth--;
                    p++;
                }
            } else if (*p == '{') {
                int depth = 1;
                p++;
                while (p < end && depth > 0) {
                    if (*p == '{') depth++;
                    else if (*p == '}') depth--;
                    p++;
                }
            } else {
                while (p < end && *p != ',' && *p != '}') p++;
            }
        }

        p = st_skip_ws(p, end);
        if (p < end && *p == ',') p = st_skip_ws(p + 1, end);
    }

    if (p < end && *p == '}') p++;
    return p;
}

/* ================================================================== */
/*  Shard Loading (mmap + header parse)                                 */
/* ================================================================== */

static int st_model_load_shard(st_model_t *m, const char *path) {
    if (m->num_shards >= ST_MAX_SHARDS) {
        fprintf(stderr, "safetensors: too many shards (max %d)\n", ST_MAX_SHARDS);
        return -1;
    }

    int shard_idx = m->num_shards;
    st_shard_t *s = &m->shards[shard_idx];

    /* Open file */
    s->fd = open(path, O_RDONLY);
    if (s->fd < 0) {
        fprintf(stderr, "safetensors: cannot open %s\n", path);
        return -1;
    }

    /* Get file size */
    struct stat sb;
    if (fstat(s->fd, &sb) < 0) {
        close(s->fd);
        fprintf(stderr, "safetensors: cannot stat %s\n", path);
        return -1;
    }
    s->file_size = (size_t)sb.st_size;

    /* mmap the entire file read-only */
    s->map = (uint8_t *)mmap(NULL, s->file_size, PROT_READ, MAP_PRIVATE, s->fd, 0);
    if (s->map == MAP_FAILED) {
        close(s->fd);
        fprintf(stderr, "safetensors: mmap failed for %s\n", path);
        return -1;
    }

    /* File must be at least 8 bytes for the header length prefix */
    if (s->file_size < 8) {
        fprintf(stderr, "safetensors: file too small (%zu bytes): %s\n",
                s->file_size, path);
        munmap(s->map, s->file_size);
        close(s->fd);
        return -1;
    }

    /* First 8 bytes: little-endian u64 = header JSON length */
    uint64_t header_len = 0;
    for (int i = 0; i < 8; i++)
        header_len |= ((uint64_t)s->map[i]) << (i * 8);

    /* Bounds check: header must fit within file */
    if (header_len > s->file_size - 8) {
        fprintf(stderr, "safetensors: header length %llu exceeds file size %zu: %s\n",
                (unsigned long long)header_len, s->file_size, path);
        munmap(s->map, s->file_size);
        close(s->fd);
        return -1;
    }

    s->header_size = (size_t)header_len;
    s->data_start = s->map + 8 + s->header_size;

    /* Parse header JSON — all reads bounded by header end */
    const char *json = (const char *)(s->map + 8);
    const char *json_end = json + s->header_size;
    const char *p = json;
    p = st_skip_ws(p, json_end);

    if (p >= json_end || *p != '{') {
        fprintf(stderr, "safetensors: invalid JSON header in %s\n", path);
        munmap(s->map, s->file_size);
        close(s->fd);
        return -1;
    }
    p = st_skip_ws(p + 1, json_end);

    while (p < json_end && *p != '}' && m->num_tensors < ST_MAX_TENSORS) {
        /* Check if this is the "__metadata__" key — skip it */
        if (*p == '"') {
            /* Peek at key name */
            const char *peek = p + 1;
            if (peek + 12 <= json_end && strncmp(peek, "__metadata__", 12) == 0) {
                /* Skip metadata entry */
                char dummy_name[ST_MAX_NAME_LEN];
                p = st_parse_string(p, json_end, dummy_name, sizeof(dummy_name));
                if (!p) break;
                p = st_skip_ws(p, json_end);
                if (p < json_end && *p == ':') p = st_skip_ws(p + 1, json_end);
                /* Skip the value object */
                if (p < json_end && *p == '{') {
                    int depth = 1;
                    p++;
                    while (p < json_end && depth > 0) {
                        if (*p == '{') depth++;
                        else if (*p == '}') depth--;
                        p++;
                    }
                }
                p = st_skip_ws(p, json_end);
                if (p < json_end && *p == ',') p = st_skip_ws(p + 1, json_end);
                continue;
            }

            /* Parse tensor entry */
            st_tensor_t *t = &m->tensors[m->num_tensors];
            const char *next = st_parse_tensor_entry(p, json_end, t, shard_idx);
            if (next) {
                m->num_tensors++;
                p = st_skip_ws(next, json_end);
                if (p < json_end && *p == ',') p = st_skip_ws(p + 1, json_end);
            } else {
                /* Parse error — skip to next entry */
                while (p < json_end && *p != ',' && *p != '}') p++;
                if (p < json_end && *p == ',') p = st_skip_ws(p + 1, json_end);
            }
        } else {
            p++;
        }
    }

    m->num_shards++;
    printf("safetensors: loaded shard %s (%zu bytes, %d tensors total)\n",
           path, s->file_size, m->num_tensors);
    return 0;
}

/* ================================================================== */
/*  Tensor Lookup & Data Access                                         */
/* ================================================================== */

/* Find tensor by name. Returns NULL if not found. */
static st_tensor_t *st_model_find(st_model_t *m, const char *name) {
    for (int i = 0; i < m->num_tensors; i++) {
        if (strcmp(m->tensors[i].name, name) == 0)
            return &m->tensors[i];
    }
    return NULL;
}

/* Validate that a tensor's data region falls within its shard's mmap.
 * Returns 0 on success, -1 on out-of-bounds. */
static int st_tensor_validate_bounds(st_model_t *m, st_tensor_t *t) {
    if (t->shard_idx < 0 || t->shard_idx >= m->num_shards) {
        fprintf(stderr, "safetensors: tensor '%s' references invalid shard %d\n",
                t->name, t->shard_idx);
        return -1;
    }
    st_shard_t *s = &m->shards[t->shard_idx];
    size_t data_region = s->file_size - 8 - s->header_size;
    if (t->data_offset + t->data_size > data_region) {
        fprintf(stderr, "safetensors: tensor '%s' data [%zu..%zu) exceeds shard data region (%zu bytes)\n",
                t->name, t->data_offset, t->data_offset + t->data_size, data_region);
        return -1;
    }
    /* Cross-check data_size against dtype + num_elements */
    int elem_sz = st_dtype_size(t->dtype);
    if (elem_sz > 0) {
        size_t expected = (size_t)t->num_elements * elem_sz;
        if (t->data_size < expected) {
            fprintf(stderr, "safetensors: tensor '%s' data_size %zu < expected %zu (%lld elems × %d bytes)\n",
                    t->name, t->data_size, expected, (long long)t->num_elements, elem_sz);
            return -1;
        }
    }
    return 0;
}

/* Get raw data pointer for a tensor. Caller should call
 * st_tensor_validate_bounds() first for safety. */
static const uint8_t *st_tensor_data(st_model_t *m, st_tensor_t *t) {
    st_shard_t *s = &m->shards[t->shard_idx];
    return s->data_start + t->data_offset;
}

/* Convert tensor data to Q16.48 array. Caller must free() result.
 * Returns NULL on error (bounds violation, OOM, unsupported dtype). */
static int64_t *st_tensor_to_q1648(st_model_t *m, st_tensor_t *t) {
    if (st_tensor_validate_bounds(m, t) != 0) return NULL;

    const uint8_t *raw = st_tensor_data(m, t);
    int64_t *out = (int64_t *)malloc((size_t)t->num_elements * sizeof(int64_t));
    if (!out) {
        fprintf(stderr, "safetensors: OOM allocating %lld × 8 bytes for tensor '%s'\n",
                (long long)t->num_elements, t->name);
        return NULL;
    }

    switch (t->dtype) {
    case ST_DTYPE_F16: {
        const uint16_t *fp16 = (const uint16_t *)raw;
        for (int64_t i = 0; i < t->num_elements; i++)
            out[i] = float16_to_q1648(fp16[i]);
        break;
    }
    case ST_DTYPE_BF16: {
        const uint16_t *bf16 = (const uint16_t *)raw;
        for (int64_t i = 0; i < t->num_elements; i++)
            out[i] = bfloat16_to_q1648(bf16[i]);
        break;
    }
    case ST_DTYPE_F32: {
        const float *fp32 = (const float *)raw;
        for (int64_t i = 0; i < t->num_elements; i++)
            out[i] = float32_to_q1648(fp32[i]);
        break;
    }
    default:
        free(out);
        return NULL;
    }

    return out;
}

/* Convert a sub-range of tensor to Q16.48. For per-layer streaming.
 * start/count are in elements (not bytes). */
static int64_t *st_tensor_to_q1648_range(st_model_t *m, st_tensor_t *t,
                                           int64_t start, int64_t count) {
    if (st_tensor_validate_bounds(m, t) != 0) return NULL;
    if (start < 0 || count <= 0 || start + count > t->num_elements) {
        fprintf(stderr, "safetensors: range [%lld..%lld) out of bounds for tensor '%s' (%lld elems)\n",
                (long long)start, (long long)(start + count), t->name, (long long)t->num_elements);
        return NULL;
    }

    int64_t *out = (int64_t *)malloc((size_t)count * sizeof(int64_t));
    if (!out) {
        fprintf(stderr, "safetensors: OOM allocating %lld × 8 bytes for range of '%s'\n",
                (long long)count, t->name);
        return NULL;
    }

    const uint8_t *raw = st_tensor_data(m, t);
    int elem_size = st_dtype_size(t->dtype);

    switch (t->dtype) {
    case ST_DTYPE_F16: {
        const uint16_t *fp16 = (const uint16_t *)raw + start;
        for (int64_t i = 0; i < count; i++)
            out[i] = float16_to_q1648(fp16[i]);
        break;
    }
    case ST_DTYPE_BF16: {
        const uint16_t *bf16 = (const uint16_t *)raw + start;
        for (int64_t i = 0; i < count; i++)
            out[i] = bfloat16_to_q1648(bf16[i]);
        break;
    }
    case ST_DTYPE_F32: {
        const float *fp32 = (const float *)raw + start;
        for (int64_t i = 0; i < count; i++)
            out[i] = float32_to_q1648(fp32[i]);
        break;
    }
    default:
        free(out);
        return NULL;
    }
    (void)elem_size;

    return out;
}

/* ================================================================== */
/*  Cleanup                                                             */
/* ================================================================== */

static void st_model_free(st_model_t *m) {
    for (int i = 0; i < m->num_shards; i++) {
        if (m->shards[i].map && m->shards[i].map != MAP_FAILED)
            munmap(m->shards[i].map, m->shards[i].file_size);
        if (m->shards[i].fd >= 0)
            close(m->shards[i].fd);
    }
    memset(m, 0, sizeof(*m));
}

/* ================================================================== */
/*  Debug: Print all tensors                                            */
/* ================================================================== */

static void st_model_print(st_model_t *m) {
    printf("safetensors model: %d shards, %d tensors\n",
           m->num_shards, m->num_tensors);
    for (int i = 0; i < m->num_tensors; i++) {
        st_tensor_t *t = &m->tensors[i];
        printf("  [%3d] %-60s dtype=%d shape=[", i, t->name, t->dtype);
        for (int d = 0; d < t->ndim; d++) {
            if (d > 0) printf(",");
            printf("%lld", (long long)t->shape[d]);
        }
        printf("] elems=%lld offset=%zu size=%zu shard=%d\n",
               (long long)t->num_elements, t->data_offset, t->data_size, t->shard_idx);
    }
}

#endif /* SAFETENSORS_H */
