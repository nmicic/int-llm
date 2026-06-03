/*
 * Author: Nenad Mićić
 * LinkedIn: https://be.linkedin.com/in/nenadmicic
 *
 * Copyright (c) 2026 Nenad Mićić
 * SPDX-License-Identifier: Apache-2.0
 *
 * tokenizer.h — C-Native BPE Tokenizer for Llama-Family Models (R6)
 * ==================================================================
 *
 * Loads HuggingFace tokenizer.json and implements encode/decode in pure C.
 * Supports the SentencePiece-style BPE format used by TinyLlama/Llama-2:
 *   - ▁ (U+2581) as word boundary marker
 *   - byte_fallback for unknown characters
 *   - 32000 vocab, 61249 merges
 *
 * ZERO external dependencies beyond POSIX.
 *
 * Usage:
 *   tokenizer_t tok;
 *   tokenizer_init(&tok);
 *   tokenizer_load(&tok, "path/to/model_dir");
 *   int ids[256]; int n;
 *   tokenizer_encode(&tok, "Hello world", ids, 256, &n);
 *   char text[1024];
 *   tokenizer_decode(&tok, ids, n, text, 1024);
 *   tokenizer_free(&tok);
 */

#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* ================================================================== */
/*  Constants                                                           */
/* ================================================================== */

#define TOK_MAX_VOCAB     64000
#define TOK_MAX_MERGES    128000
#define TOK_MAX_TOKEN_LEN 256
#define TOK_BOS_ID        1
#define TOK_EOS_ID        2
#define TOK_UNK_ID        0

/* UTF-8 encoding of ▁ (U+2581) = 0xE2 0x96 0x81 */
#define TOK_SPIECE_UNDERLINE "\xe2\x96\x81"
#define TOK_SPIECE_UNDERLINE_LEN 3

/* ================================================================== */
/*  Data Structures                                                     */
/* ================================================================== */

typedef struct {
    char *text;      /* token string (heap-allocated) */
    int   id;        /* token ID */
    float score;     /* merge priority (lower = higher priority) */
} tok_vocab_entry_t;

typedef struct {
    int left;        /* vocab ID of left part */
    int right;       /* vocab ID of right part */
    int result;      /* vocab ID of merged result */
    int rank;        /* merge rank (0 = highest priority) */
} tok_merge_t;

typedef struct {
    tok_vocab_entry_t *vocab;       /* [vocab_size] */
    int                vocab_size;
    int               *str_to_id;   /* hash map: string → vocab ID (open addressing) */
    char             **id_to_str;   /* [vocab_size]: ID → string */
    uint32_t          *hash_keys;   /* hash of the string */
    int                hash_cap;    /* hash table capacity */

    tok_merge_t       *merges;      /* [num_merges] */
    int                num_merges;

    int                bos_id;
    int                eos_id;
    int                unk_id;
    int                add_bos;     /* prepend BOS on encode */
    int                byte_fallback; /* use <0xHH> for unknown bytes */
} tokenizer_t;

/* ================================================================== */
/*  Minimal JSON Helpers (bounded, no recursion)                        */
/* ================================================================== */

static const char *tok_skip_ws(const char *p, const char *end) {
    while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
    return p;
}

/* Parse a JSON string value. Handles escapes. Returns pointer past closing quote.
 * Writes unescaped string to buf, NUL-terminated. */
static const char *tok_parse_json_string(const char *p, const char *end,
                                          char *buf, int buf_sz) {
    if (p >= end || *p != '"') return NULL;
    p++;
    int i = 0;
    while (p < end && *p != '"') {
        if (*p == '\\') {
            p++;
            if (p >= end) return NULL;
            char c = *p;
            if (c == 'n') c = '\n';
            else if (c == 't') c = '\t';
            else if (c == 'r') c = '\r';
            else if (c == '\\') c = '\\';
            else if (c == '"') c = '"';
            else if (c == '/') c = '/';
            else if (c == 'u') {
                /* Parse \uXXXX */
                if (p + 4 >= end) return NULL;
                unsigned int cp = 0;
                for (int k = 1; k <= 4; k++) {
                    char h = p[k];
                    cp <<= 4;
                    if (h >= '0' && h <= '9') cp |= h - '0';
                    else if (h >= 'a' && h <= 'f') cp |= 10 + h - 'a';
                    else if (h >= 'A' && h <= 'F') cp |= 10 + h - 'A';
                    else return NULL;
                }
                p += 4;
                /* Encode as UTF-8 */
                if (cp < 0x80) {
                    if (i < buf_sz - 1) buf[i++] = (char)cp;
                } else if (cp < 0x800) {
                    if (i < buf_sz - 2) {
                        buf[i++] = (char)(0xC0 | (cp >> 6));
                        buf[i++] = (char)(0x80 | (cp & 0x3F));
                    }
                } else {
                    if (i < buf_sz - 3) {
                        buf[i++] = (char)(0xE0 | (cp >> 12));
                        buf[i++] = (char)(0x80 | ((cp >> 6) & 0x3F));
                        buf[i++] = (char)(0x80 | (cp & 0x3F));
                    }
                }
                p++;
                continue;
            }
            if (i < buf_sz - 1) buf[i++] = c;
        } else {
            if (i < buf_sz - 1) buf[i++] = *p;
        }
        p++;
    }
    buf[i] = '\0';
    if (p < end && *p == '"') p++;
    return p;
}

/* Skip a JSON value (string, number, object, array, bool, null) */
static const char *tok_skip_json_value(const char *p, const char *end) {
    p = tok_skip_ws(p, end);
    if (p >= end) return NULL;
    if (*p == '"') {
        p++;
        while (p < end && *p != '"') {
            if (*p == '\\') { p++; if (p >= end) break; }
            p++;
        }
        if (p < end) p++;
        return p;
    }
    if (*p == '{') {
        int depth = 1; p++;
        while (p < end && depth > 0) {
            if (*p == '{') depth++;
            else if (*p == '}') depth--;
            else if (*p == '"') {
                p++;
                while (p < end && *p != '"') {
                    if (*p == '\\') { p++; if (p >= end) break; }
                    p++;
                }
            }
            p++;
        }
        return p;
    }
    if (*p == '[') {
        int depth = 1; p++;
        while (p < end && depth > 0) {
            if (*p == '[') depth++;
            else if (*p == ']') depth--;
            else if (*p == '"') {
                p++;
                while (p < end && *p != '"') {
                    if (*p == '\\') { p++; if (p >= end) break; }
                    p++;
                }
            }
            p++;
        }
        return p;
    }
    /* number, true, false, null */
    while (p < end && *p != ',' && *p != '}' && *p != ']' && *p != ' ' && *p != '\n' && *p != '\r' && *p != '\t') p++;
    return p;
}

/* ================================================================== */
/*  Hash Table for Vocab Lookup                                         */
/* ================================================================== */

static uint32_t tok_hash_str(const char *s) {
    uint32_t h = 5381;
    while (*s) h = ((h << 5) + h) + (uint8_t)*s++;
    return h;
}

static void tok_hash_insert(tokenizer_t *t, const char *key, int id) {
    uint32_t h = tok_hash_str(key);
    int idx = (int)(h % (uint32_t)t->hash_cap);
    while (t->str_to_id[idx] >= 0) {
        idx = (idx + 1) % t->hash_cap;
    }
    t->str_to_id[idx] = id;
    t->hash_keys[idx] = h;
}

static int tok_hash_lookup(const tokenizer_t *t, const char *key) {
    uint32_t h = tok_hash_str(key);
    int idx = (int)(h % (uint32_t)t->hash_cap);
    for (int probe = 0; probe < t->hash_cap; probe++) {
        if (t->str_to_id[idx] < 0) return -1;  /* empty slot */
        if (t->hash_keys[idx] == h &&
            strcmp(t->id_to_str[t->str_to_id[idx]], key) == 0) {
            return t->str_to_id[idx];
        }
        idx = (idx + 1) % t->hash_cap;
    }
    return -1;
}

/* ================================================================== */
/*  Load from tokenizer.json                                            */
/* ================================================================== */

/* Forward declaration (used in tokenizer_load error paths) */
static void tokenizer_free(tokenizer_t *t);

static void tokenizer_init(tokenizer_t *t) {
    memset(t, 0, sizeof(*t));
    t->bos_id = TOK_BOS_ID;
    t->eos_id = TOK_EOS_ID;
    t->unk_id = TOK_UNK_ID;
    t->add_bos = 1;
    t->byte_fallback = 1;
}

static int tokenizer_load(tokenizer_t *t, const char *model_dir) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/tokenizer.json", model_dir);
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "tokenizer: cannot open %s\n", path);
        return -1;
    }
    fseek(f, 0, SEEK_END);
    long fsz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *json = (char *)malloc(fsz + 1);
    if (!json) { fclose(f); return -1; }
    size_t nread = fread(json, 1, fsz, f);
    fclose(f);
    if ((long)nread != fsz) {
        fprintf(stderr, "tokenizer: short read on %s (%zu/%ld)\n", path, nread, fsz);
        free(json);
        return -1;
    }
    json[fsz] = '\0';

    const char *p = json;
    const char *end = json + fsz;

    /* Allocate vocab and hash table */
    t->vocab = (tok_vocab_entry_t *)calloc(TOK_MAX_VOCAB, sizeof(tok_vocab_entry_t));
    t->id_to_str = (char **)calloc(TOK_MAX_VOCAB, sizeof(char *));
    t->hash_cap = TOK_MAX_VOCAB * 4;  /* load factor ~25% */
    t->str_to_id = (int *)malloc(t->hash_cap * sizeof(int));
    t->hash_keys = (uint32_t *)calloc(t->hash_cap, sizeof(uint32_t));
    t->merges = (tok_merge_t *)calloc(TOK_MAX_MERGES, sizeof(tok_merge_t));

    if (!t->vocab || !t->id_to_str || !t->str_to_id || !t->hash_keys || !t->merges) {
        fprintf(stderr, "tokenizer: OOM allocating tables\n");
        tokenizer_free(t);
        free(json);
        return -1;
    }
    memset(t->str_to_id, 0xFF, t->hash_cap * sizeof(int));  /* -1 = empty */

    /* Find "model" → "vocab" section */
    const char *vocab_start = strstr(p, "\"vocab\"");
    if (!vocab_start) { tokenizer_free(t); free(json); return -1; }
    /* Skip to the opening { of the vocab object */
    vocab_start = strchr(vocab_start + 7, '{');
    if (!vocab_start) { tokenizer_free(t); free(json); return -1; }
    vocab_start++;

    /* Parse vocab entries: "token_string": id */
    char tok_buf[TOK_MAX_TOKEN_LEN];
    const char *vp = vocab_start;
    t->vocab_size = 0;
    while (vp < end) {
        vp = tok_skip_ws(vp, end);
        if (vp >= end || *vp == '}') break;
        if (*vp == ',') { vp++; continue; }
        vp = tok_parse_json_string(vp, end, tok_buf, sizeof(tok_buf));
        if (!vp) break;
        vp = tok_skip_ws(vp, end);
        if (vp >= end || *vp != ':') break;
        vp++;
        vp = tok_skip_ws(vp, end);
        /* Parse integer ID */
        int id = 0;
        int neg = 0;
        if (*vp == '-') { neg = 1; vp++; }
        while (vp < end && *vp >= '0' && *vp <= '9') {
            id = id * 10 + (*vp - '0');
            vp++;
        }
        if (neg) id = -id;

        if (id >= 0 && id < TOK_MAX_VOCAB) {
            char *dup = strdup(tok_buf);
            if (!dup) { tokenizer_free(t); free(json); return -1; }
            t->vocab[id].text = dup;
            t->vocab[id].id = id;
            t->id_to_str[id] = t->vocab[id].text;
            tok_hash_insert(t, tok_buf, id);
            if (id >= t->vocab_size) t->vocab_size = id + 1;
        }
    }

    /* Find "merges" array */
    const char *merges_start = strstr(p, "\"merges\"");
    if (!merges_start) { free(json); return 0; } /* merges optional — vocab-only still works for decode */
    merges_start = strchr(merges_start + 8, '[');
    if (!merges_start) { free(json); return 0; }
    merges_start++;

    /* Parse merges: each is a string "left right" */
    const char *mp = merges_start;
    t->num_merges = 0;
    while (mp < end && t->num_merges < TOK_MAX_MERGES) {
        mp = tok_skip_ws(mp, end);
        if (mp >= end || *mp == ']') break;
        if (*mp == ',') { mp++; continue; }
        mp = tok_parse_json_string(mp, end, tok_buf, sizeof(tok_buf));
        if (!mp) break;

        /* Split "left right" */
        char *space = strchr(tok_buf, ' ');
        if (!space) continue;
        *space = '\0';
        const char *left_str = tok_buf;
        const char *right_str = space + 1;

        int left_id = tok_hash_lookup(t, left_str);
        int right_id = tok_hash_lookup(t, right_str);
        if (left_id < 0 || right_id < 0) continue;

        /* Find or create the merged token */
        *space = '\0'; /* already done, but be explicit */
        char merged[TOK_MAX_TOKEN_LEN * 2];
        snprintf(merged, sizeof(merged), "%s%s", left_str, right_str);
        int merged_id = tok_hash_lookup(t, merged);
        if (merged_id < 0) continue; /* merged token not in vocab — skip */

        tok_merge_t *m = &t->merges[t->num_merges];
        m->left = left_id;
        m->right = right_id;
        m->result = merged_id;
        m->rank = t->num_merges;
        t->num_merges++;
    }

    free(json);
    if (t->vocab_size == 0) {
        fprintf(stderr, "tokenizer: 0 vocab entries loaded from %s — file may be malformed\n", path);
        tokenizer_free(t);
        return -1;
    }
    printf("tokenizer: loaded %d vocab entries, %d merges from %s\n",
           t->vocab_size, t->num_merges, path);
    return 0;
}

/* ================================================================== */
/*  Encode: text → token IDs                                            */
/* ================================================================== */

/* Normalize: prepend ▁, replace spaces with ▁ */
static void tok_normalize(const char *input, char *output, int out_sz) {
    int i = 0;
    /* Prepend ▁ */
    if (i + TOK_SPIECE_UNDERLINE_LEN < out_sz) {
        memcpy(output + i, TOK_SPIECE_UNDERLINE, TOK_SPIECE_UNDERLINE_LEN);
        i += TOK_SPIECE_UNDERLINE_LEN;
    }
    while (*input && i < out_sz - 4) {
        if (*input == ' ') {
            memcpy(output + i, TOK_SPIECE_UNDERLINE, TOK_SPIECE_UNDERLINE_LEN);
            i += TOK_SPIECE_UNDERLINE_LEN;
            input++;
        } else {
            output[i++] = *input++;
        }
    }
    output[i] = '\0';
}

/* BPE encode: start with character-level tokens, iteratively merge */
static int tokenizer_encode(const tokenizer_t *t, const char *text,
                             int *out_ids, int max_ids, int *out_len) {
    /* Step 1: normalize */
    int norm_sz = (int)strlen(text) * 4 + 16;
    char *norm = (char *)malloc(norm_sz);
    if (!norm) return -1;
    tok_normalize(text, norm, norm_sz);

    /* Step 2: split into initial tokens (try to match longest vocab entries
     * character-by-character, with byte fallback for unknowns) */
    int *ids = (int *)malloc(norm_sz * sizeof(int));
    if (!ids) { free(norm); return -1; }
    int n = 0;
    const char *s = norm;
    while (*s && n < norm_sz - 1) {
        /* Try longest match in vocab */
        int best_len = 0;
        int best_id = -1;
        /* Try up to TOK_MAX_TOKEN_LEN bytes */
        for (int len = 1; len <= TOK_MAX_TOKEN_LEN && s[len-1]; len++) {
            char tmp[TOK_MAX_TOKEN_LEN + 1];
            memcpy(tmp, s, len);
            tmp[len] = '\0';
            int id = tok_hash_lookup(t, tmp);
            if (id >= 0) {
                best_len = len;
                best_id = id;
            }
        }
        if (best_len > 0) {
            ids[n++] = best_id;
            s += best_len;
        } else {
            /* Byte fallback: encode as <0xHH> */
            if (t->byte_fallback) {
                char byte_tok[8];
                snprintf(byte_tok, sizeof(byte_tok), "<0x%02X>",
                         (unsigned char)*s);
                int byte_id = tok_hash_lookup(t, byte_tok);
                if (byte_id >= 0) {
                    ids[n++] = byte_id;
                } else {
                    ids[n++] = t->unk_id;
                }
            } else {
                ids[n++] = t->unk_id;
            }
            s++;
        }
    }

    /* Step 3: BPE merge loop — repeatedly merge the highest-priority pair */
    /* Build a merge rank lookup: for each pair (left, right), find its rank.
     * Since merges are ordered by priority, earlier merges have lower rank. */
    int changed = 1;
    while (changed && n > 1) {
        changed = 0;
        /* Find the best merge (lowest rank) among all adjacent pairs */
        int best_rank = t->num_merges;  /* sentinel: no merge found */
        int best_pos = -1;
        for (int i = 0; i < n - 1; i++) {
            /* Linear scan over merges — acceptable for small n */
            for (int m = 0; m < t->num_merges && m < best_rank; m++) {
                if (t->merges[m].left == ids[i] && t->merges[m].right == ids[i+1]) {
                    best_rank = m;
                    best_pos = i;
                    break;
                }
            }
        }
        if (best_pos >= 0) {
            ids[best_pos] = t->merges[best_rank].result;
            /* Remove ids[best_pos+1] by shifting */
            memmove(&ids[best_pos + 1], &ids[best_pos + 2],
                    (n - best_pos - 2) * sizeof(int));
            n--;
            changed = 1;
        }
    }

    /* Step 4: prepend BOS if configured */
    int out_n = 0;
    if (t->add_bos && out_n < max_ids) {
        out_ids[out_n++] = t->bos_id;
    }
    for (int i = 0; i < n && out_n < max_ids; i++) {
        out_ids[out_n++] = ids[i];
    }

    free(ids);
    free(norm);
    *out_len = out_n;
    return 0;
}

/* ================================================================== */
/*  Decode: token IDs → text                                            */
/* ================================================================== */

static int tokenizer_decode(const tokenizer_t *t, const int *ids, int n,
                             char *out, int out_sz) {
    int pos = 0;
    for (int i = 0; i < n; i++) {
        int id = ids[i];
        /* Skip special tokens */
        if (id == t->bos_id || id == t->eos_id) continue;

        const char *tok_str = NULL;
        if (id >= 0 && id < t->vocab_size && t->id_to_str[id]) {
            tok_str = t->id_to_str[id];
        }
        if (!tok_str) continue;

        /* Check for byte token: <0xHH> */
        if (tok_str[0] == '<' && tok_str[1] == '0' && tok_str[2] == 'x' &&
            tok_str[5] == '>' && tok_str[6] == '\0') {
            unsigned int byte_val = 0;
            for (int k = 3; k <= 4; k++) {
                char h = tok_str[k];
                byte_val <<= 4;
                if (h >= '0' && h <= '9') byte_val |= h - '0';
                else if (h >= 'a' && h <= 'f') byte_val |= 10 + h - 'a';
                else if (h >= 'A' && h <= 'F') byte_val |= 10 + h - 'A';
            }
            if (pos < out_sz - 1) out[pos++] = (char)byte_val;
            continue;
        }

        /* Copy token string, replacing ▁ with space */
        const char *s = tok_str;
        while (*s && pos < out_sz - 1) {
            if ((uint8_t)s[0] == 0xE2 && s[1] && s[2] &&
                (uint8_t)s[1] == 0x96 && (uint8_t)s[2] == 0x81) {
                out[pos++] = ' ';
                s += 3;
            } else {
                out[pos++] = *s++;
            }
        }
    }
    out[pos] = '\0';

    /* Strip leading space (SentencePiece convention) */
    if (pos > 0 && out[0] == ' ') {
        memmove(out, out + 1, pos);
        pos--;
    }

    return pos;
}

static void tokenizer_free(tokenizer_t *t) {
    if (t->vocab) {
        for (int i = 0; i < t->vocab_size; i++)
            free(t->vocab[i].text);
        free(t->vocab);
    }
    free(t->id_to_str);
    free(t->str_to_id);
    free(t->hash_keys);
    free(t->merges);
    memset(t, 0, sizeof(*t));
}

#endif /* TOKENIZER_H */
