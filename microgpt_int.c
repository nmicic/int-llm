/*
 * Author: Nenad Mićić
 * LinkedIn: https://be.linkedin.com/in/nenadmicic
 *
 * Copyright (c) 2026 Nenad Mićić
 * SPDX-License-Identifier: Apache-2.0
 *
 * microgpt_int.c — Integer/Bitwise-Only GPT
 * ==========================================
 *
 * Complete GPT training + inference using ZERO floating-point operations.
 * All math via fixed-point Q16.48 integer arithmetic.
 *
 * Derived from microgpt.c, rewritten to use fp_math.h primitives:
 *   - exp/log/sqrt/sin/cos → integer refinement invariants
 *   - CORDIC for trigonometry (shifts and adds only)
 *   - Pi from Machin's formula (integer Taylor series)
 *   - Gaussian RNG via CLT (sum of 12 uniforms)
 *   - Adam optimizer with running beta products (no powf)
 *
 * Compile: gcc -O3 -march=native -o gpt_int microgpt_int.c
 *          (NO -lm needed!)
 *
 * Note: stdio.h/stdlib.h/string.h are for I/O and memory only.
 *       ALL math is in fp_math.h (integer-only).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "fp_math.h"

/* ------------------------------------------------------------------ */
/*  Dataset loading (unchanged — already pure integer)                 */
/* ------------------------------------------------------------------ */
#define MAX_DOCS 85000
#define MAX_DOC_LEN 512
#define MAX_CHARS 128

static char docs[MAX_DOCS][MAX_DOC_LEN];
static int num_docs = 0;

static void load_dataset(const char *filename) {
    FILE *f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "Cannot open %s (run `make input` to download the demo dataset)\n", filename);
        exit(1);
    }
    char line[256];
    while (fgets(line, sizeof(line), f) && num_docs < MAX_DOCS) {
        int len = (int)strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r'))
            line[--len] = 0;
        if (len > 0) {
            strncpy(docs[num_docs], line, MAX_DOC_LEN - 1);
            docs[num_docs][MAX_DOC_LEN - 1] = 0;
            num_docs++;
        }
    }
    fclose(f);
}

/* ------------------------------------------------------------------ */
/*  Tokenizer (unchanged — already pure integer)                       */
/* ------------------------------------------------------------------ */
static char uchars_arr[MAX_CHARS];
static int vocab_size, BOS, num_uchars = 0;

static int char_to_id(char c) {
    for (int i = 0; i < num_uchars; i++)
        if (uchars_arr[i] == c) return i;
    return -1;
}

static int cmp_char(const void *a, const void *b) {
    return *(const char *)a - *(const char *)b;
}

static void build_tokenizer(void) {
    int seen[256] = {0};
    for (int d = 0; d < num_docs; d++)
        for (int i = 0; docs[d][i]; i++)
            seen[(unsigned char)docs[d][i]] = 1;
    for (int i = 0; i < 256; i++)
        if (seen[i]) uchars_arr[num_uchars++] = (char)i;
    qsort(uchars_arr, num_uchars, sizeof(char), cmp_char);
    BOS = num_uchars;
    vocab_size = num_uchars + 1;
}

/* ------------------------------------------------------------------ */
/*  Model hyper-parameters                                             */
/* ------------------------------------------------------------------ */
#define N_EMBD 32
#define N_HEAD 4
#define N_LAYER 1
#define BLOCK_SIZE 8
#define HEAD_DIM (N_EMBD / N_HEAD)
#define MLP_DIM (4 * N_EMBD)

/* ------------------------------------------------------------------ */
/*  Parameters & gradients (fixed_t = int64_t arrays)                  */
/* ------------------------------------------------------------------ */
static fixed_t *wte, *d_wte;
static fixed_t *wpe, *d_wpe;
static fixed_t *lm_head, *d_lm_head;

static fixed_t *attn_wq[N_LAYER], *d_attn_wq[N_LAYER];
static fixed_t *attn_wk[N_LAYER], *d_attn_wk[N_LAYER];
static fixed_t *attn_wv[N_LAYER], *d_attn_wv[N_LAYER];
static fixed_t *attn_wo[N_LAYER], *d_attn_wo[N_LAYER];
static fixed_t *mlp_fc1[N_LAYER], *d_mlp_fc1[N_LAYER];
static fixed_t *mlp_fc2[N_LAYER], *d_mlp_fc2[N_LAYER];

/* Adam optimizer buffers */
static fixed_t *adam_m_wte, *adam_v_wte;
static fixed_t *adam_m_wpe, *adam_v_wpe;
static fixed_t *adam_m_lm, *adam_v_lm;
static fixed_t *adam_m_wq[N_LAYER], *adam_v_wq[N_LAYER];
static fixed_t *adam_m_wk[N_LAYER], *adam_v_wk[N_LAYER];
static fixed_t *adam_m_wv[N_LAYER], *adam_v_wv[N_LAYER];
static fixed_t *adam_m_wo[N_LAYER], *adam_v_wo[N_LAYER];
static fixed_t *adam_m_fc1[N_LAYER], *adam_v_fc1[N_LAYER];
static fixed_t *adam_m_fc2[N_LAYER], *adam_v_fc2[N_LAYER];

static int num_params = 0;
static fixed_t ATTN_SCALE; /* cached 1/sqrt(HEAD_DIM) */

static fixed_t *make_param(int size, fixed_t std) {
    fixed_t *p = (fixed_t *)calloc(size, sizeof(fixed_t));
    for (int i = 0; i < size; i++)
        p[i] = fp_gaussian(FP_ZERO, std);
    num_params += size;
    return p;
}

static fixed_t *make_zero(int size) {
    return (fixed_t *)calloc(size, sizeof(fixed_t));
}

static void init_params(void) {
    /* 0.02 in Q16.48 */
    fixed_t std_02 = FP_ONE / 50;
    int es = vocab_size * N_EMBD, ps = BLOCK_SIZE * N_EMBD;
    int as = N_EMBD * N_EMBD, ms = MLP_DIM * N_EMBD;

    wte = make_param(es, std_02);  d_wte = make_zero(es);
    adam_m_wte = make_zero(es);    adam_v_wte = make_zero(es);
    wpe = make_param(ps, std_02);  d_wpe = make_zero(ps);
    adam_m_wpe = make_zero(ps);    adam_v_wpe = make_zero(ps);
    lm_head = make_param(es, std_02); d_lm_head = make_zero(es);
    adam_m_lm = make_zero(es);     adam_v_lm = make_zero(es);

    for (int i = 0; i < N_LAYER; i++) {
        attn_wq[i] = make_param(as, std_02);   d_attn_wq[i] = make_zero(as);
        adam_m_wq[i] = make_zero(as);           adam_v_wq[i] = make_zero(as);
        attn_wk[i] = make_param(as, std_02);   d_attn_wk[i] = make_zero(as);
        adam_m_wk[i] = make_zero(as);           adam_v_wk[i] = make_zero(as);
        attn_wv[i] = make_param(as, std_02);   d_attn_wv[i] = make_zero(as);
        adam_m_wv[i] = make_zero(as);           adam_v_wv[i] = make_zero(as);
        attn_wo[i] = make_param(as, FP_ZERO);  d_attn_wo[i] = make_zero(as);
        adam_m_wo[i] = make_zero(as);           adam_v_wo[i] = make_zero(as);
        mlp_fc1[i] = make_param(ms, std_02);   d_mlp_fc1[i] = make_zero(ms);
        adam_m_fc1[i] = make_zero(ms);          adam_v_fc1[i] = make_zero(ms);
        mlp_fc2[i] = make_param(ms, FP_ZERO);  d_mlp_fc2[i] = make_zero(ms);
        adam_m_fc2[i] = make_zero(ms);          adam_v_fc2[i] = make_zero(ms);
    }
    printf("num params: %d\n", num_params);
    ATTN_SCALE = fp_inv_sqrt(fp_from_int(HEAD_DIM));
}

/* ------------------------------------------------------------------ */
/*  Forward-pass activation storage (whole-sequence, flat 2D arrays)   */
/*  NOTE: these arrays have no layer dimension — they are reused per   */
/*  layer iteration. Safe for N_LAYER==1; for multi-layer, each array  */
/*  below (and the backward accumulators) would need a layer index.    */
/* ------------------------------------------------------------------ */
static fixed_t fwd_emb[BLOCK_SIZE][N_EMBD];
static fixed_t fwd_rms_init[BLOCK_SIZE];
static fixed_t fwd_rms_ms_init[BLOCK_SIZE];    /* int-specific: ms_eps for rmsnorm_bwd */
static fixed_t fwd_x_in[BLOCK_SIZE][N_EMBD];
static fixed_t fwd_rms_attn[BLOCK_SIZE];
static fixed_t fwd_rms_ms_attn[BLOCK_SIZE];    /* int-specific */
static fixed_t fwd_xn_attn[BLOCK_SIZE][N_EMBD];
static fixed_t fwd_q[BLOCK_SIZE][N_EMBD];
static fixed_t fwd_k[BLOCK_SIZE][N_EMBD];
static fixed_t fwd_v[BLOCK_SIZE][N_EMBD];
static fixed_t fwd_attn_w[BLOCK_SIZE][N_HEAD][BLOCK_SIZE];
static fixed_t fwd_attn_out[BLOCK_SIZE][N_EMBD];
static fixed_t fwd_x_mid[BLOCK_SIZE][N_EMBD];
static fixed_t fwd_rms_mlp[BLOCK_SIZE];
static fixed_t fwd_rms_ms_mlp[BLOCK_SIZE];     /* int-specific */
static fixed_t fwd_xn_mlp[BLOCK_SIZE][N_EMBD];
static fixed_t fwd_mlp_pre[BLOCK_SIZE][MLP_DIM];
static fixed_t fwd_mlp_post[BLOCK_SIZE][MLP_DIM];
static fixed_t fwd_x_out[BLOCK_SIZE][N_EMBD];
static fixed_t fwd_probs[BLOCK_SIZE][MAX_CHARS + 1];

/* Backward accumulators for two-phase gradient computation */
static fixed_t bwd_dk[BLOCK_SIZE][N_EMBD];
static fixed_t bwd_dv[BLOCK_SIZE][N_EMBD];
static fixed_t bwd_dq[BLOCK_SIZE][N_EMBD];
static fixed_t bwd_d_res[BLOCK_SIZE][N_EMBD];

/* ------------------------------------------------------------------ */
/*  Forward building blocks (integer-only)                             */
/* ------------------------------------------------------------------ */

static inline void linear_fwd(const fixed_t *restrict x,
                               const fixed_t *restrict w,
                               int nout, int nin, fixed_t *restrict out) {
    for (int r = 0; r < nout; r++) {
        fixed_t s = 0;
        const fixed_t *wr = w + r * nin;
        for (int c = 0; c < nin; c++)
            s += fp_mul(wr[c], x[c]);
        out[r] = s;
    }
}

/* RMSNorm: returns scale = 1/sqrt(mean_sq + eps)
 * Also stores ms_plus_eps for overflow-safe backward */
static inline fixed_t rmsnorm_fwd(const fixed_t *x, int n, fixed_t *out,
                                   fixed_t *ms_out) {
    fixed_t ms = 0;
    for (int i = 0; i < n; i++)
        ms += fp_mul(x[i], x[i]);
    ms = ms / n; /* integer divide — avoids 128-bit fp_div */
    fixed_t eps = FP_ONE / 100000; /* 1e-5 */
    fixed_t ms_eps = ms + eps;
    if (ms_out) *ms_out = ms_eps;
    fixed_t scale = fp_inv_sqrt(ms_eps);
    for (int i = 0; i < n; i++)
        out[i] = fp_mul(x[i], scale);
    return scale;
}

static inline void softmax_fwd(const fixed_t *logits, int n, fixed_t *probs) {
    fixed_t mx = logits[0];
    for (int i = 1; i < n; i++)
        if (logits[i] > mx) mx = logits[i];
    fixed_t sum = 0;
    for (int i = 0; i < n; i++) {
        probs[i] = fp_safe_exp(logits[i] - mx);
        sum += probs[i];
    }
    if (sum == 0) sum = 1; /* safety */
    fixed_t inv = fp_div(FP_ONE, sum);
    for (int i = 0; i < n; i++)
        probs[i] = fp_mul(probs[i], inv);
}

/* ------------------------------------------------------------------ */
/*  Backward building blocks (integer-only)                            */
/* ------------------------------------------------------------------ */

static inline void linear_bwd_x(const fixed_t *restrict w,
                                 const fixed_t *restrict dout,
                                 int nout, int nin, fixed_t *restrict dx) {
    for (int c = 0; c < nin; c++) {
        fixed_t s = 0;
        for (int r = 0; r < nout; r++)
            s += fp_mul(dout[r], w[r * nin + c]);
        dx[c] += s;
    }
}

static inline void linear_bwd_w(const fixed_t *restrict x,
                                 const fixed_t *restrict dout,
                                 int nout, int nin, fixed_t *restrict dw) {
    for (int r = 0; r < nout; r++) {
        fixed_t dr = dout[r];
        fixed_t *dwr = dw + r * nin;
        for (int c = 0; c < nin; c++)
            dwr[c] += fp_mul(dr, x[c]);
    }
}

/* RMSNorm backward — reformulated to avoid scale^3 overflow.
 * Uses: dx += scale*dout - (x * dot) / (n * ms_eps)
 * where dot = sum(dout[i] * x[i]), ms_eps = mean_sq + eps */
static inline void rmsnorm_bwd(const fixed_t *x, fixed_t scale,
                                fixed_t ms_eps, const fixed_t *dout,
                                int n, fixed_t *dx) {
    fixed_t dot = 0;
    for (int i = 0; i < n; i++)
        dot += fp_mul(dout[i], x[i]);

    /* inv_nms = 1 / (n * ms_eps) = scale^2 / n
     * This avoids computing scale^3 which can overflow */
    fixed_t nms = fp_mul(fp_from_int(n), ms_eps);
    fixed_t inv_nms = fp_div(FP_ONE, nms);

    for (int i = 0; i < n; i++) {
        fixed_t term = fp_mul(inv_nms, fp_mul(x[i], dot));
        dx[i] += fp_mul(scale, dout[i]) - term;
    }
}

/* ------------------------------------------------------------------ */
/*  Training forward pass (whole-sequence, fills activation arrays)    */
/* ------------------------------------------------------------------ */
static fixed_t gpt_forward(int n, const int *tokens) {
    fixed_t logits[MAX_CHARS + 1];

    /* Embedding + initial RMSNorm */
    for (int t = 0; t < n; t++) {
        for (int i = 0; i < N_EMBD; i++)
            fwd_emb[t][i] = wte[tokens[t] * N_EMBD + i] + wpe[t * N_EMBD + i];
        fwd_rms_init[t] = rmsnorm_fwd(fwd_emb[t], N_EMBD, fwd_x_in[t],
                                       &fwd_rms_ms_init[t]);
    }

    /* Transformer layer(s) */
    for (int li = 0; li < N_LAYER; li++) {
        /* Pre-attention RMSNorm + QKV projections for all positions */
        for (int t = 0; t < n; t++) {
            fwd_rms_attn[t] = rmsnorm_fwd(fwd_x_in[t], N_EMBD, fwd_xn_attn[t],
                                           &fwd_rms_ms_attn[t]);
            linear_fwd(fwd_xn_attn[t], attn_wq[li], N_EMBD, N_EMBD, fwd_q[t]);
            linear_fwd(fwd_xn_attn[t], attn_wk[li], N_EMBD, N_EMBD, fwd_k[t]);
            linear_fwd(fwd_xn_attn[t], attn_wv[li], N_EMBD, N_EMBD, fwd_v[t]);
        }

        /* Causal self-attention */
        fixed_t attn_scale = ATTN_SCALE;
        for (int t = 0; t < n; t++) {
            for (int h = 0; h < N_HEAD; h++) {
                int hs = h * HEAD_DIM;
                fixed_t al[BLOCK_SIZE];
                for (int s = 0; s <= t; s++) {
                    fixed_t dot = 0;
                    for (int j = 0; j < HEAD_DIM; j++)
                        dot += fp_mul(fwd_q[t][hs + j], fwd_k[s][hs + j]);
                    al[s] = fp_mul(dot, attn_scale);
                }
                /* softmax over attention weights */
                fixed_t mx = al[0];
                for (int s = 1; s <= t; s++)
                    if (al[s] > mx) mx = al[s];
                fixed_t sm = 0;
                for (int s = 0; s <= t; s++) {
                    al[s] = fp_safe_exp(al[s] - mx);
                    sm += al[s];
                }
                if (sm == 0) sm = 1;
                fixed_t inv_sm = fp_div(FP_ONE, sm);
                for (int s = 0; s <= t; s++) {
                    al[s] = fp_mul(al[s], inv_sm);
                    fwd_attn_w[t][h][s] = al[s];
                }
                for (int j = 0; j < HEAD_DIM; j++) {
                    fixed_t acc = 0;
                    for (int s = 0; s <= t; s++)
                        acc += fp_mul(al[s], fwd_v[s][hs + j]);
                    fwd_attn_out[t][hs + j] = acc;
                }
            }
        }

        /* Wo projection + attention residual */
        for (int t = 0; t < n; t++) {
            fixed_t wo_out[N_EMBD];
            linear_fwd(fwd_attn_out[t], attn_wo[li], N_EMBD, N_EMBD, wo_out);
            for (int i = 0; i < N_EMBD; i++)
                fwd_x_mid[t][i] = wo_out[i] + fwd_x_in[t][i];
        }

        /* MLP: RMSNorm -> fc1 -> ReLU -> fc2 + residual */
        for (int t = 0; t < n; t++) {
            fwd_rms_mlp[t] = rmsnorm_fwd(fwd_x_mid[t], N_EMBD, fwd_xn_mlp[t],
                                          &fwd_rms_ms_mlp[t]);
            linear_fwd(fwd_xn_mlp[t], mlp_fc1[li], MLP_DIM, N_EMBD,
                       fwd_mlp_pre[t]);
            for (int i = 0; i < MLP_DIM; i++)
                fwd_mlp_post[t][i] =
                    fwd_mlp_pre[t][i] > 0 ? fwd_mlp_pre[t][i] : 0; /* ReLU */
            fixed_t fc2_out[N_EMBD];
            linear_fwd(fwd_mlp_post[t], mlp_fc2[li], N_EMBD, MLP_DIM, fc2_out);
            for (int i = 0; i < N_EMBD; i++)
                fwd_x_out[t][i] = fc2_out[i] + fwd_x_mid[t][i];
        }
    }

    /* LM head -> softmax -> cross-entropy loss */
    fixed_t total_loss = 0;
    for (int t = 0; t < n; t++) {
        linear_fwd(fwd_x_out[t], lm_head, vocab_size, N_EMBD, logits);
        softmax_fwd(logits, vocab_size, fwd_probs[t]);
        fixed_t prob = fwd_probs[t][tokens[t + 1]];
        if (prob < 4) prob = 4; /* floor to avoid log(0) */
        total_loss += -fp_safe_log(prob);
    }
    return fp_div(total_loss, fp_from_int(n));
}

/* ------------------------------------------------------------------ */
/*  Backward pass (two-phase, integer-only)                            */
/*  Phase 1 (reverse): loss -> lm_head -> MLP -> Wo -> attn scores    */
/*  Phase 2 (forward): QKV weight grads -> rmsnorm -> embedding grads  */
/* ------------------------------------------------------------------ */
static void gpt_backward(int n, const int *tokens) {
    fixed_t inv_n = fp_div(FP_ONE, fp_from_int(n));
    memset(bwd_dk, 0, sizeof(bwd_dk));
    memset(bwd_dv, 0, sizeof(bwd_dv));

    /* Phase 1: reverse over positions */
    for (int t = n - 1; t >= 0; t--) {
        int target = tokens[t + 1];

        /* dL/dlogits = softmax - one_hot, averaged */
        fixed_t dl[MAX_CHARS + 1];
        for (int i = 0; i < vocab_size; i++)
            dl[i] = fp_mul(fwd_probs[t][i] -
                           (i == target ? FP_ONE : FP_ZERO), inv_n);

        fixed_t dx[N_EMBD];
        memset(dx, 0, sizeof(dx));
        linear_bwd_x(lm_head, dl, vocab_size, N_EMBD, dx);
        linear_bwd_w(fwd_x_out[t], dl, vocab_size, N_EMBD, d_lm_head);

        for (int li = N_LAYER - 1; li >= 0; li--) {
            /* MLP backward */
            fixed_t d_h2[MLP_DIM];
            memset(d_h2, 0, sizeof(d_h2));
            linear_bwd_x(mlp_fc2[li], dx, N_EMBD, MLP_DIM, d_h2);
            linear_bwd_w(fwd_mlp_post[t], dx, N_EMBD, MLP_DIM,
                         d_mlp_fc2[li]);

            fixed_t d_h1[MLP_DIM];
            for (int i = 0; i < MLP_DIM; i++) {
                /* d/dx(ReLU) = 1 for x > 0, 0 otherwise */
                d_h1[i] = fwd_mlp_pre[t][i] > 0 ? d_h2[i] : 0;
            }

            fixed_t d_xn_mlp[N_EMBD];
            memset(d_xn_mlp, 0, sizeof(d_xn_mlp));
            linear_bwd_x(mlp_fc1[li], d_h1, MLP_DIM, N_EMBD, d_xn_mlp);
            linear_bwd_w(fwd_xn_mlp[t], d_h1, MLP_DIM, N_EMBD,
                         d_mlp_fc1[li]);

            fixed_t d_x_mid[N_EMBD];
            memset(d_x_mid, 0, sizeof(d_x_mid));
            rmsnorm_bwd(fwd_x_mid[t], fwd_rms_mlp[t],
                        fwd_rms_ms_mlp[t], d_xn_mlp, N_EMBD, d_x_mid);
            for (int i = 0; i < N_EMBD; i++)
                dx[i] += d_x_mid[i];

            /* Wo backward */
            fixed_t d_ao[N_EMBD];
            memset(d_ao, 0, sizeof(d_ao));
            linear_bwd_x(attn_wo[li], dx, N_EMBD, N_EMBD, d_ao);
            linear_bwd_w(fwd_attn_out[t], dx, N_EMBD, N_EMBD,
                         d_attn_wo[li]);

            /* Attention backward: accumulate dk, dv; compute dq */
            fixed_t attn_scale = ATTN_SCALE;
            memset(bwd_dq[t], 0, sizeof(bwd_dq[t]));

            for (int h = 0; h < N_HEAD; h++) {
                int hs = h * HEAD_DIM;
                fixed_t d_aw[BLOCK_SIZE];
                memset(d_aw, 0, sizeof(d_aw));
                for (int j = 0; j < HEAD_DIM; j++) {
                    for (int s = 0; s <= t; s++) {
                        d_aw[s] += fp_mul(d_ao[hs+j],
                                          fwd_v[s][hs+j]);
                        bwd_dv[s][hs+j] +=
                            fp_mul(fwd_attn_w[t][h][s], d_ao[hs+j]);
                    }
                }
                fixed_t dot = 0;
                for (int s = 0; s <= t; s++)
                    dot += fp_mul(d_aw[s], fwd_attn_w[t][h][s]);
                fixed_t d_al[BLOCK_SIZE];
                for (int s = 0; s <= t; s++)
                    d_al[s] = fp_mul(fwd_attn_w[t][h][s],
                                     d_aw[s] - dot);
                for (int s = 0; s <= t; s++) {
                    for (int j = 0; j < HEAD_DIM; j++) {
                        bwd_dq[t][hs+j] += fp_mul(fp_mul(d_al[s],
                                     fwd_k[s][hs+j]), attn_scale);
                        bwd_dk[s][hs+j] +=
                            fp_mul(fp_mul(d_al[s],
                                   fwd_q[t][hs+j]), attn_scale);
                    }
                }
            }

            memcpy(bwd_d_res[t], dx, sizeof(dx));
        }
    }

    /* Phase 2: forward over positions — QKV weight grads, rmsnorm,
     * embedding grads */
    for (int t = 0; t < n; t++) {
        for (int li = 0; li < N_LAYER; li++) {
            fixed_t d_xn[N_EMBD];
            memset(d_xn, 0, sizeof(d_xn));

            linear_bwd_x(attn_wq[li], bwd_dq[t], N_EMBD, N_EMBD, d_xn);
            linear_bwd_w(fwd_xn_attn[t], bwd_dq[t], N_EMBD, N_EMBD,
                         d_attn_wq[li]);
            linear_bwd_x(attn_wk[li], bwd_dk[t], N_EMBD, N_EMBD, d_xn);
            linear_bwd_w(fwd_xn_attn[t], bwd_dk[t], N_EMBD, N_EMBD,
                         d_attn_wk[li]);
            linear_bwd_x(attn_wv[li], bwd_dv[t], N_EMBD, N_EMBD, d_xn);
            linear_bwd_w(fwd_xn_attn[t], bwd_dv[t], N_EMBD, N_EMBD,
                         d_attn_wv[li]);

            fixed_t d_x_in[N_EMBD];
            memset(d_x_in, 0, sizeof(d_x_in));
            rmsnorm_bwd(fwd_x_in[t], fwd_rms_attn[t],
                        fwd_rms_ms_attn[t], d_xn, N_EMBD, d_x_in);
            for (int i = 0; i < N_EMBD; i++)
                d_x_in[i] += bwd_d_res[t][i];

            fixed_t d_emb[N_EMBD];
            memset(d_emb, 0, sizeof(d_emb));
            rmsnorm_bwd(fwd_emb[t], fwd_rms_init[t],
                        fwd_rms_ms_init[t], d_x_in, N_EMBD, d_emb);

            int tok = tokens[t];
            for (int i = 0; i < N_EMBD; i++) {
                d_wte[tok * N_EMBD + i] += d_emb[i];
                d_wpe[t * N_EMBD + i] += d_emb[i];
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/*  Adam update (integer-only, running beta products)                   */
/* ------------------------------------------------------------------ */
static void adam_update(fixed_t *p, fixed_t *g, fixed_t *m, fixed_t *v,
                        int sz, fixed_t lr, fixed_t b1, fixed_t b2,
                        fixed_t eps, fixed_t inv_b1c, fixed_t inv_b2c) {
    fixed_t one_minus_b1 = FP_ONE - b1;
    fixed_t one_minus_b2 = FP_ONE - b2;
    /* Gradient clipping: clamp to [-1, 1] in fixed-point */
    fixed_t clip = FP_ONE;
    for (int i = 0; i < sz; i++) {
        fixed_t gi = g[i];
        if (gi > clip) gi = clip;
        if (gi < -clip) gi = -clip;
        m[i] = fp_mul(b1, m[i]) + fp_mul(one_minus_b1, gi);
        v[i] = fp_mul(b2, v[i]) + fp_mul(one_minus_b2, fp_mul(gi, gi));
        fixed_t m_hat = fp_mul(m[i], inv_b1c);
        fixed_t v_hat = fp_mul(v[i], inv_b2c);
        fixed_t denom = fp_sqrt_fast(v_hat) + eps;
        p[i] -= fp_div(fp_mul(lr, m_hat), denom);
        g[i] = 0;
    }
}

/* ------------------------------------------------------------------ */
/*  Weighted random choice (integer-only)                              */
/* ------------------------------------------------------------------ */
static int weighted_choice(const fixed_t *w, int n) {
    fixed_t total = 0;
    for (int i = 0; i < n; i++) total += w[i];
    if (total == 0) return 0;
    fixed_t r = fp_mul(fp_rng_uniform(), total);
    fixed_t cum = 0;
    for (int i = 0; i < n; i++) {
        cum += w[i];
        if (r < cum) return i;
    }
    return n - 1;
}

/* ------------------------------------------------------------------ */
/*  Inference forward (token-at-a-time with separate KV cache)         */
/* ------------------------------------------------------------------ */
static fixed_t inf_keys[N_LAYER][BLOCK_SIZE][N_EMBD];
static fixed_t inf_vals[N_LAYER][BLOCK_SIZE][N_EMBD];

static void inference_forward(int token_id, int pos, fixed_t *logits_out) {
    fixed_t x[N_EMBD], tmp[MLP_DIM > N_EMBD ? MLP_DIM : N_EMBD];

    for (int i = 0; i < N_EMBD; i++)
        x[i] = wte[token_id * N_EMBD + i] + wpe[pos * N_EMBD + i];
    rmsnorm_fwd(x, N_EMBD, x, NULL);

    for (int li = 0; li < N_LAYER; li++) {
        fixed_t xr[N_EMBD];
        memcpy(xr, x, sizeof(xr));

        fixed_t xn[N_EMBD];
        rmsnorm_fwd(x, N_EMBD, xn, NULL);

        fixed_t q[N_EMBD], k[N_EMBD], v[N_EMBD];
        linear_fwd(xn, attn_wq[li], N_EMBD, N_EMBD, q);
        linear_fwd(xn, attn_wk[li], N_EMBD, N_EMBD, k);
        linear_fwd(xn, attn_wv[li], N_EMBD, N_EMBD, v);
        memcpy(inf_keys[li][pos], k, sizeof(k));
        memcpy(inf_vals[li][pos], v, sizeof(v));

        int seq_len = pos + 1;
        fixed_t attn_scale = ATTN_SCALE;
        fixed_t ao[N_EMBD];
        for (int h = 0; h < N_HEAD; h++) {
            int hs = h * HEAD_DIM;
            fixed_t al[BLOCK_SIZE];
            for (int s = 0; s < seq_len; s++) {
                fixed_t dot = 0;
                for (int j = 0; j < HEAD_DIM; j++)
                    dot += fp_mul(q[hs + j], inf_keys[li][s][hs + j]);
                al[s] = fp_mul(dot, attn_scale);
            }
            fixed_t mx = al[0];
            for (int s = 1; s < seq_len; s++)
                if (al[s] > mx) mx = al[s];
            fixed_t sm = 0;
            for (int s = 0; s < seq_len; s++) {
                al[s] = fp_safe_exp(al[s] - mx);
                sm += al[s];
            }
            if (sm == 0) sm = 1;
            fixed_t inv_sm = fp_div(FP_ONE, sm);
            for (int s = 0; s < seq_len; s++)
                al[s] = fp_mul(al[s], inv_sm);
            for (int j = 0; j < HEAD_DIM; j++) {
                fixed_t acc = 0;
                for (int s = 0; s < seq_len; s++)
                    acc += fp_mul(al[s], inf_vals[li][s][hs + j]);
                ao[hs + j] = acc;
            }
        }

        linear_fwd(ao, attn_wo[li], N_EMBD, N_EMBD, tmp);
        for (int i = 0; i < N_EMBD; i++)
            x[i] = tmp[i] + xr[i];

        memcpy(xr, x, sizeof(xr));
        fixed_t xn_m[N_EMBD];
        rmsnorm_fwd(x, N_EMBD, xn_m, NULL);
        fixed_t h1[MLP_DIM];
        linear_fwd(xn_m, mlp_fc1[li], MLP_DIM, N_EMBD, h1);
        for (int i = 0; i < MLP_DIM; i++)
            h1[i] = h1[i] > 0 ? h1[i] : 0; /* ReLU (stable in fixed-point) */
        linear_fwd(h1, mlp_fc2[li], N_EMBD, MLP_DIM, tmp);
        for (int i = 0; i < N_EMBD; i++)
            x[i] = tmp[i] + xr[i];
    }

    linear_fwd(x, lm_head, vocab_size, N_EMBD, logits_out);
}

/* ------------------------------------------------------------------ */
/*  Main: training + inference — ZERO floating-point!                  */
/* ------------------------------------------------------------------ */
int main(void) {
    /* Initialize integer math (pi, CORDIC tables) */
    fp_math_init();

    printf("=== MicroGPT Integer-Only (Q16.48 Fixed-Point) ===\n");
    printf("Pi = "); fp_print(FP_PI, 15); printf("\n");

    load_dataset("input.txt");

    int *doc_order = (int *)malloc(num_docs * sizeof(int));
    for (int i = 0; i < num_docs; i++) doc_order[i] = i;
    fp_shuffle_ints(doc_order, num_docs);
    char (*docs_tmp)[MAX_DOC_LEN] = malloc((size_t)num_docs * MAX_DOC_LEN);
    for (int i = 0; i < num_docs; i++)
        memcpy(docs_tmp[i], docs[doc_order[i]], MAX_DOC_LEN);
    memcpy(docs, docs_tmp, (size_t)num_docs * MAX_DOC_LEN);
    free(docs_tmp);
    free(doc_order);

    printf("num docs: %d\n", num_docs);
    build_tokenizer();
    printf("vocab size: %d\n", vocab_size);
    init_params();

    /* Hyperparameters in fixed-point */
    fixed_t lr = FP_ONE / 100;            /* 0.01 */
    fixed_t b1 = FP_ONE * 9 / 10;        /* 0.9 */
    fixed_t b2 = FP_ONE * 95 / 100;      /* 0.95 */
    fixed_t eps = FP_ONE / 100000000;     /* 1e-8 */
    int num_steps = 5000;

    /* Running beta products for Adam bias correction (no powf!) */
    fixed_t beta1_pow = FP_ONE;  /* beta1^0 = 1 */
    fixed_t beta2_pow = FP_ONE;  /* beta2^0 = 1 */

    for (int step = 0; step < num_steps; step++) {
        char *doc = docs[step % num_docs];
        int doc_len = (int)strlen(doc);

        int tokens[MAX_DOC_LEN + 2];
        tokens[0] = BOS;
        for (int i = 0; i < doc_len; i++)
            tokens[i + 1] = char_to_id(doc[i]);
        tokens[doc_len + 1] = BOS;
        int n = BLOCK_SIZE < (doc_len + 1) ? BLOCK_SIZE : (doc_len + 1);

        fixed_t loss = gpt_forward(n, tokens);
        gpt_backward(n, tokens);

        /* Running beta products for bias correction */
        beta1_pow = fp_mul(beta1_pow, b1);
        beta2_pow = fp_mul(beta2_pow, b2);
        fixed_t b1c = FP_ONE - beta1_pow;
        fixed_t b2c = FP_ONE - beta2_pow;
        /* Precompute inverses once per step (not per parameter!) */
        fixed_t inv_b1c = fp_div(FP_ONE, b1c);
        fixed_t inv_b2c = fp_div(FP_ONE, b2c);

        /* Cosine LR schedule via CORDIC (integer-only) */
        fixed_t angle = fp_div(fp_mul(FP_PI, fp_from_int(step)),
                               fp_from_int(num_steps));
        fixed_t cos_val, sin_val;
        fp_sincos(angle, &cos_val, &sin_val);
        fixed_t lr_t = fp_mul(lr, fp_mul(FP_HALF, FP_ONE + cos_val));

        int es = vocab_size * N_EMBD, ps = BLOCK_SIZE * N_EMBD;
        int as = N_EMBD * N_EMBD, ms = MLP_DIM * N_EMBD;
        adam_update(wte, d_wte, adam_m_wte, adam_v_wte, es,
                    lr_t, b1, b2, eps, inv_b1c, inv_b2c);
        adam_update(wpe, d_wpe, adam_m_wpe, adam_v_wpe, ps,
                    lr_t, b1, b2, eps, inv_b1c, inv_b2c);
        adam_update(lm_head, d_lm_head, adam_m_lm, adam_v_lm, es,
                    lr_t, b1, b2, eps, inv_b1c, inv_b2c);
        for (int i = 0; i < N_LAYER; i++) {
            adam_update(attn_wq[i], d_attn_wq[i], adam_m_wq[i],
                        adam_v_wq[i], as, lr_t, b1, b2, eps, inv_b1c, inv_b2c);
            adam_update(attn_wk[i], d_attn_wk[i], adam_m_wk[i],
                        adam_v_wk[i], as, lr_t, b1, b2, eps, inv_b1c, inv_b2c);
            adam_update(attn_wv[i], d_attn_wv[i], adam_m_wv[i],
                        adam_v_wv[i], as, lr_t, b1, b2, eps, inv_b1c, inv_b2c);
            adam_update(attn_wo[i], d_attn_wo[i], adam_m_wo[i],
                        adam_v_wo[i], as, lr_t, b1, b2, eps, inv_b1c, inv_b2c);
            adam_update(mlp_fc1[i], d_mlp_fc1[i], adam_m_fc1[i],
                        adam_v_fc1[i], ms, lr_t, b1, b2, eps, inv_b1c, inv_b2c);
            adam_update(mlp_fc2[i], d_mlp_fc2[i], adam_m_fc2[i],
                        adam_v_fc2[i], ms, lr_t, b1, b2, eps, inv_b1c, inv_b2c);
        }

        /* Print loss (convert to double ONLY for display) */
        printf("step %4d / %4d | loss %.4f\n",
               step + 1, num_steps, fp_to_double(loss));
    }

    /* ---- Inference ---- */
    fixed_t temperature = FP_ONE / 2; /* 0.5 */
    printf("\n--- inference ---\n");
    for (int si = 0; si < 20; si++) {
        char sample[BLOCK_SIZE + 1];
        int slen = 0, token_id = BOS;
        for (int pos = 0; pos < BLOCK_SIZE; pos++) {
            fixed_t logits[MAX_CHARS + 1], probs[MAX_CHARS + 1];
            inference_forward(token_id, pos, logits);
            fixed_t inv_t = fp_div(FP_ONE, temperature);
            for (int i = 0; i < vocab_size; i++)
                logits[i] = fp_mul(logits[i], inv_t);
            softmax_fwd(logits, vocab_size, probs);
            token_id = weighted_choice(probs, vocab_size);
            if (token_id == BOS) break;
            if (token_id < num_uchars)
                sample[slen++] = uchars_arr[token_id];
        }
        sample[slen] = '\0';
        printf("sample %2d: %s\n", si + 1, sample);
        memset(inf_keys, 0, sizeof(inf_keys));
        memset(inf_vals, 0, sizeof(inf_vals));
    }

    /* cleanup */
    free(wte); free(d_wte); free(adam_m_wte); free(adam_v_wte);
    free(wpe); free(d_wpe); free(adam_m_wpe); free(adam_v_wpe);
    free(lm_head); free(d_lm_head); free(adam_m_lm); free(adam_v_lm);
    for (int i = 0; i < N_LAYER; i++) {
        free(attn_wq[i]); free(d_attn_wq[i]);
        free(adam_m_wq[i]); free(adam_v_wq[i]);
        free(attn_wk[i]); free(d_attn_wk[i]);
        free(adam_m_wk[i]); free(adam_v_wk[i]);
        free(attn_wv[i]); free(d_attn_wv[i]);
        free(adam_m_wv[i]); free(adam_v_wv[i]);
        free(attn_wo[i]); free(d_attn_wo[i]);
        free(adam_m_wo[i]); free(adam_v_wo[i]);
        free(mlp_fc1[i]); free(d_mlp_fc1[i]);
        free(adam_m_fc1[i]); free(adam_v_fc1[i]);
        free(mlp_fc2[i]); free(d_mlp_fc2[i]);
        free(adam_m_fc2[i]); free(adam_v_fc2[i]);
    }
    return 0;
}
