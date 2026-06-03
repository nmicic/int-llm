/*
 * Copyright (c) 2026 Nenad Mićić
 * SPDX-License-Identifier: Apache-2.0
 *
 * microgpt.c — Float32 1-layer GPT: training + inference in pure C.
 * Hand-written forward and backward passes, two-phase gradient accumulation.
 * Single file, no dependencies beyond libc + libm.
 *
 * Inspired by Andrej Karpathy's microgpt.py (a ~200-line Python char GPT):
 *   https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95
 * This is an independent from-scratch C rebuild, not a port. The config here
 * (N_EMBD=32, N_HEAD=4, N_LAYER=1, 14,656 params) is the float baseline that
 * pairs with the integer variant microgpt_int.c for the three-way benchmark.
 * A leaner speed-tuned variant (sub-20 ms/step) lives at:
 *   https://gist.github.com/nmicic/35316463f3c5e8e9fe8eb599b3842b58
 *
 * Compile: gcc -O3 -march=native -ffast-math -o gpt_float microgpt.c -lm
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ------------------------------------------------------------------ */
/*  xorshift64 PRNG (deterministic, seed = 42)                       */
/* ------------------------------------------------------------------ */
static unsigned long long rng_state = 42;

static unsigned long long rng_next(void) {
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 7;
  rng_state ^= rng_state << 17;
  return rng_state;
}

static double rng_uniform(void) {
  return (rng_next() >> 11) * (1.0 / 9007199254740992.0);
}

static float rng_gauss(float mean, float std) {
  double u1 = rng_uniform(), u2 = rng_uniform();
  if (u1 < 1e-30)
    u1 = 1e-30;
  return mean + std * (float)(sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2));
}

static void shuffle_ints(int *arr, int n) {
  for (int i = n - 1; i > 0; i--) {
    int j = (int)(rng_uniform() * (i + 1));
    int tmp = arr[i];
    arr[i] = arr[j];
    arr[j] = tmp;
  }
}

/* ------------------------------------------------------------------ */
/*  Dataset                                                           */
/* ------------------------------------------------------------------ */
#define MAX_DOCS 85000
#define MAX_DOC_LEN 512
#define MAX_CHARS 128

static char docs[MAX_DOCS][MAX_DOC_LEN];
static int num_docs = 0;

static void load_dataset(const char *path) {
  FILE *f = fopen(path, "r");
  if (!f) {
    fprintf(stderr, "Cannot open %s (run `make input` to download the demo dataset)\n", path);
    exit(1);
  }
  char line[256];
  while (fgets(line, sizeof(line), f) && num_docs < MAX_DOCS) {
    int len = (int)strlen(line);
    while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r'))
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
/*  Character tokenizer                                               */
/* ------------------------------------------------------------------ */
static char uchars_arr[MAX_CHARS];
static int vocab_size, BOS, num_uchars = 0;

static int char_to_id(char c) {
  for (int i = 0; i < num_uchars; i++)
    if (uchars_arr[i] == c)
      return i;
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
  for (int c = 0; c < 256; c++)
    if (seen[c])
      uchars_arr[num_uchars++] = (char)c;
  qsort(uchars_arr, num_uchars, sizeof(char), cmp_char);
  BOS = num_uchars;
  vocab_size = num_uchars + 1;
}

/* ------------------------------------------------------------------ */
/*  Model hyperparameters                                             */
/* ------------------------------------------------------------------ */
#define N_EMBD 32
#define N_HEAD 4
#define N_LAYER 1
#define BLOCK_SIZE 8
#define HEAD_DIM (N_EMBD / N_HEAD)
#define MLP_DIM (4 * N_EMBD)

/* ------------------------------------------------------------------ */
/*  Parameters, gradients, and Adam state (heap-allocated per matrix) */
/* ------------------------------------------------------------------ */
static float *wte, *d_wte, *m_wte, *v_wte;
static float *wpe, *d_wpe, *m_wpe, *v_wpe;
static float *lm_head, *d_lm_head, *m_lm_head, *v_lm_head;

static float *attn_wq[N_LAYER], *d_attn_wq[N_LAYER];
static float *attn_wk[N_LAYER], *d_attn_wk[N_LAYER];
static float *attn_wv[N_LAYER], *d_attn_wv[N_LAYER];
static float *attn_wo[N_LAYER], *d_attn_wo[N_LAYER];
static float *mlp_fc1[N_LAYER], *d_mlp_fc1[N_LAYER];
static float *mlp_fc2[N_LAYER], *d_mlp_fc2[N_LAYER];

static float *m_wq[N_LAYER], *v_wq[N_LAYER];
static float *m_wk[N_LAYER], *v_wk[N_LAYER];
static float *m_wv[N_LAYER], *v_wv[N_LAYER];
static float *m_wo[N_LAYER], *v_wo[N_LAYER];
static float *m_fc1[N_LAYER], *v_fc1[N_LAYER];
static float *m_fc2[N_LAYER], *v_fc2[N_LAYER];

static int num_params = 0;

static float *alloc_param(int n, float std) {
  float *p = (float *)calloc(n, sizeof(float));
  for (int i = 0; i < n; i++)
    p[i] = rng_gauss(0, std);
  num_params += n;
  return p;
}

static float *alloc_zero(int n) {
  return (float *)calloc(n, sizeof(float));
}

static void init_params(void) {
  int es = vocab_size * N_EMBD, ps = BLOCK_SIZE * N_EMBD;
  int as = N_EMBD * N_EMBD, ms = MLP_DIM * N_EMBD;

  wte = alloc_param(es, 0.02f);
  d_wte = alloc_zero(es);
  m_wte = alloc_zero(es);
  v_wte = alloc_zero(es);

  wpe = alloc_param(ps, 0.02f);
  d_wpe = alloc_zero(ps);
  m_wpe = alloc_zero(ps);
  v_wpe = alloc_zero(ps);

  lm_head = alloc_param(es, 0.02f);
  d_lm_head = alloc_zero(es);
  m_lm_head = alloc_zero(es);
  v_lm_head = alloc_zero(es);

  for (int i = 0; i < N_LAYER; i++) {
    attn_wq[i] = alloc_param(as, 0.02f);
    d_attn_wq[i] = alloc_zero(as);
    m_wq[i] = alloc_zero(as);
    v_wq[i] = alloc_zero(as);

    attn_wk[i] = alloc_param(as, 0.02f);
    d_attn_wk[i] = alloc_zero(as);
    m_wk[i] = alloc_zero(as);
    v_wk[i] = alloc_zero(as);

    attn_wv[i] = alloc_param(as, 0.02f);
    d_attn_wv[i] = alloc_zero(as);
    m_wv[i] = alloc_zero(as);
    v_wv[i] = alloc_zero(as);

    attn_wo[i] = alloc_param(as, 0.0f);
    d_attn_wo[i] = alloc_zero(as);
    m_wo[i] = alloc_zero(as);
    v_wo[i] = alloc_zero(as);

    mlp_fc1[i] = alloc_param(ms, 0.02f);
    d_mlp_fc1[i] = alloc_zero(ms);
    m_fc1[i] = alloc_zero(ms);
    v_fc1[i] = alloc_zero(ms);

    mlp_fc2[i] = alloc_param(ms, 0.0f);
    d_mlp_fc2[i] = alloc_zero(ms);
    m_fc2[i] = alloc_zero(ms);
    v_fc2[i] = alloc_zero(ms);
  }
  printf("num params: %d\n", num_params);
}

/* ------------------------------------------------------------------ */
/*  Forward-pass activation storage (whole-sequence)                  */
/*  NOTE: these arrays have no layer dimension — they are reused per  */
/*  layer iteration. Safe for N_LAYER==1; for multi-layer, each array */
/*  below (and the backward accumulators) would need a layer index.   */
/* ------------------------------------------------------------------ */
static float fwd_emb[BLOCK_SIZE][N_EMBD];
static float fwd_rms_init[BLOCK_SIZE];
static float fwd_x_in[BLOCK_SIZE][N_EMBD];
static float fwd_rms_attn[BLOCK_SIZE];
static float fwd_xn_attn[BLOCK_SIZE][N_EMBD];
static float fwd_q[BLOCK_SIZE][N_EMBD];
static float fwd_k[BLOCK_SIZE][N_EMBD];
static float fwd_v[BLOCK_SIZE][N_EMBD];
static float fwd_attn_w[BLOCK_SIZE][N_HEAD][BLOCK_SIZE];
static float fwd_attn_out[BLOCK_SIZE][N_EMBD];
static float fwd_x_mid[BLOCK_SIZE][N_EMBD];
static float fwd_rms_mlp[BLOCK_SIZE];
static float fwd_xn_mlp[BLOCK_SIZE][N_EMBD];
static float fwd_mlp_pre[BLOCK_SIZE][MLP_DIM];
static float fwd_mlp_post[BLOCK_SIZE][MLP_DIM];
static float fwd_x_out[BLOCK_SIZE][N_EMBD];
static float fwd_probs[BLOCK_SIZE][MAX_CHARS + 1];

/* Backward accumulators for two-phase gradient computation */
static float bwd_dk[BLOCK_SIZE][N_EMBD];
static float bwd_dv[BLOCK_SIZE][N_EMBD];
static float bwd_dq[BLOCK_SIZE][N_EMBD];
static float bwd_d_res[BLOCK_SIZE][N_EMBD];

/* ------------------------------------------------------------------ */
/*  Forward building blocks                                           */
/* ------------------------------------------------------------------ */
static inline void linear_fwd(const float *restrict x, const float *restrict w,
                              int nout, int nin, float *restrict out) {
  for (int r = 0; r < nout; r++) {
    float s = 0;
    const float *wr = w + r * nin;
    for (int c = 0; c < nin; c++)
      s += wr[c] * x[c];
    out[r] = s;
  }
}

static inline float rmsnorm_fwd(const float *x, int n, float *out) {
  float ms = 0;
  for (int i = 0; i < n; i++)
    ms += x[i] * x[i];
  ms /= n;
  float scale = 1.0f / sqrtf(ms + 1e-5f);
  for (int i = 0; i < n; i++)
    out[i] = x[i] * scale;
  return scale;
}

static inline void softmax_fwd(const float *logits, int n, float *probs) {
  float mx = logits[0];
  for (int i = 1; i < n; i++)
    if (logits[i] > mx)
      mx = logits[i];
  float sum = 0;
  for (int i = 0; i < n; i++) {
    probs[i] = expf(logits[i] - mx);
    sum += probs[i];
  }
  float inv = 1.0f / sum;
  for (int i = 0; i < n; i++)
    probs[i] *= inv;
}

/* ------------------------------------------------------------------ */
/*  Backward building blocks                                          */
/* ------------------------------------------------------------------ */
static inline void linear_bwd_x(const float *restrict w,
                                const float *restrict dout, int nout, int nin,
                                float *restrict dx) {
  for (int c = 0; c < nin; c++) {
    float s = 0;
    for (int r = 0; r < nout; r++)
      s += dout[r] * w[r * nin + c];
    dx[c] += s;
  }
}

static inline void linear_bwd_w(const float *restrict x,
                                const float *restrict dout, int nout, int nin,
                                float *restrict dw) {
  for (int r = 0; r < nout; r++) {
    float dr = dout[r];
    float *dwr = dw + r * nin;
    for (int c = 0; c < nin; c++)
      dwr[c] += dr * x[c];
  }
}

static inline void rmsnorm_bwd(const float *x, float scale, const float *dout,
                               int n, float *dx) {
  float dot = 0;
  for (int i = 0; i < n; i++)
    dot += dout[i] * x[i];
  float coeff = scale * scale * scale / n;
  for (int i = 0; i < n; i++)
    dx[i] += scale * dout[i] - coeff * x[i] * dot;
}

/* ------------------------------------------------------------------ */
/*  Training forward pass (whole-sequence, fills activation arrays)   */
/* ------------------------------------------------------------------ */
static float gpt_forward(int n, const int *tokens) {
  float logits[MAX_CHARS + 1];

  /* Embedding + initial RMSNorm */
  for (int t = 0; t < n; t++) {
    for (int i = 0; i < N_EMBD; i++)
      fwd_emb[t][i] = wte[tokens[t] * N_EMBD + i] + wpe[t * N_EMBD + i];
    fwd_rms_init[t] = rmsnorm_fwd(fwd_emb[t], N_EMBD, fwd_x_in[t]);
  }

  /* Transformer layer(s) */
  for (int li = 0; li < N_LAYER; li++) {
    /* Pre-attention RMSNorm + QKV projections for all positions */
    for (int t = 0; t < n; t++) {
      fwd_rms_attn[t] = rmsnorm_fwd(fwd_x_in[t], N_EMBD, fwd_xn_attn[t]);
      linear_fwd(fwd_xn_attn[t], attn_wq[li], N_EMBD, N_EMBD, fwd_q[t]);
      linear_fwd(fwd_xn_attn[t], attn_wk[li], N_EMBD, N_EMBD, fwd_k[t]);
      linear_fwd(fwd_xn_attn[t], attn_wv[li], N_EMBD, N_EMBD, fwd_v[t]);
    }

    /* Causal self-attention */
    float attn_scale = 1.0f / sqrtf((float)HEAD_DIM);
    for (int t = 0; t < n; t++) {
      for (int h = 0; h < N_HEAD; h++) {
        int hs = h * HEAD_DIM;
        float al[BLOCK_SIZE];
        for (int s = 0; s <= t; s++) {
          float dot = 0;
          for (int j = 0; j < HEAD_DIM; j++)
            dot += fwd_q[t][hs + j] * fwd_k[s][hs + j];
          al[s] = dot * attn_scale;
        }
        float mx = al[0];
        for (int s = 1; s <= t; s++)
          if (al[s] > mx)
            mx = al[s];
        float sm = 0;
        for (int s = 0; s <= t; s++) {
          al[s] = expf(al[s] - mx);
          sm += al[s];
        }
        float inv = 1.0f / sm;
        for (int s = 0; s <= t; s++) {
          al[s] *= inv;
          fwd_attn_w[t][h][s] = al[s];
        }
        for (int j = 0; j < HEAD_DIM; j++) {
          float acc = 0;
          for (int s = 0; s <= t; s++)
            acc += al[s] * fwd_v[s][hs + j];
          fwd_attn_out[t][hs + j] = acc;
        }
      }
    }

    /* Wo projection + attention residual */
    for (int t = 0; t < n; t++) {
      float wo_out[N_EMBD];
      linear_fwd(fwd_attn_out[t], attn_wo[li], N_EMBD, N_EMBD, wo_out);
      for (int i = 0; i < N_EMBD; i++)
        fwd_x_mid[t][i] = wo_out[i] + fwd_x_in[t][i];
    }

    /* MLP: RMSNorm -> fc1 -> x-squared -> fc2 + residual */
    for (int t = 0; t < n; t++) {
      fwd_rms_mlp[t] = rmsnorm_fwd(fwd_x_mid[t], N_EMBD, fwd_xn_mlp[t]);
      linear_fwd(fwd_xn_mlp[t], mlp_fc1[li], MLP_DIM, N_EMBD, fwd_mlp_pre[t]);
      for (int i = 0; i < MLP_DIM; i++)
        fwd_mlp_post[t][i] =
            fwd_mlp_pre[t][i] > 0 ? fwd_mlp_pre[t][i] * fwd_mlp_pre[t][i] : 0;
      float fc2_out[N_EMBD];
      linear_fwd(fwd_mlp_post[t], mlp_fc2[li], N_EMBD, MLP_DIM, fc2_out);
      for (int i = 0; i < N_EMBD; i++)
        fwd_x_out[t][i] = fc2_out[i] + fwd_x_mid[t][i];
    }
  }

  /* LM head -> softmax -> cross-entropy loss */
  float total_loss = 0;
  for (int t = 0; t < n; t++) {
    linear_fwd(fwd_x_out[t], lm_head, vocab_size, N_EMBD, logits);
    softmax_fwd(logits, vocab_size, fwd_probs[t]);
    total_loss += -logf(fwd_probs[t][tokens[t + 1]] + 1e-30f);
  }
  return total_loss / n;
}

/* ------------------------------------------------------------------ */
/*  Backward pass (two-phase)                                         */
/*  Phase 1 (reverse): loss -> lm_head -> MLP -> Wo -> attn scores    */
/*  Phase 2 (forward): QKV weight grads -> rmsnorm -> embedding grads */
/* ------------------------------------------------------------------ */
static void gpt_backward(int n, const int *tokens) {
  float inv_n = 1.0f / n;
  memset(bwd_dk, 0, sizeof(bwd_dk));
  memset(bwd_dv, 0, sizeof(bwd_dv));

  /* Phase 1: reverse over positions */
  for (int t = n - 1; t >= 0; t--) {
    int target = tokens[t + 1];

    float dl[MAX_CHARS + 1];
    for (int i = 0; i < vocab_size; i++)
      dl[i] = (fwd_probs[t][i] - (i == target ? 1.0f : 0.0f)) * inv_n;

    float dx[N_EMBD];
    memset(dx, 0, sizeof(dx));
    linear_bwd_x(lm_head, dl, vocab_size, N_EMBD, dx);
    linear_bwd_w(fwd_x_out[t], dl, vocab_size, N_EMBD, d_lm_head);

    for (int li = N_LAYER - 1; li >= 0; li--) {
      /* MLP backward */
      float d_h2[MLP_DIM];
      memset(d_h2, 0, sizeof(d_h2));
      linear_bwd_x(mlp_fc2[li], dx, N_EMBD, MLP_DIM, d_h2);
      linear_bwd_w(fwd_mlp_post[t], dx, N_EMBD, MLP_DIM, d_mlp_fc2[li]);

      float d_h1[MLP_DIM];
      for (int i = 0; i < MLP_DIM; i++)
        d_h1[i] =
            fwd_mlp_pre[t][i] > 0 ? 2.0f * fwd_mlp_pre[t][i] * d_h2[i] : 0;

      float d_xn_mlp[N_EMBD];
      memset(d_xn_mlp, 0, sizeof(d_xn_mlp));
      linear_bwd_x(mlp_fc1[li], d_h1, MLP_DIM, N_EMBD, d_xn_mlp);
      linear_bwd_w(fwd_xn_mlp[t], d_h1, MLP_DIM, N_EMBD, d_mlp_fc1[li]);

      float d_x_mid[N_EMBD];
      memset(d_x_mid, 0, sizeof(d_x_mid));
      rmsnorm_bwd(fwd_x_mid[t], fwd_rms_mlp[t], d_xn_mlp, N_EMBD, d_x_mid);
      for (int i = 0; i < N_EMBD; i++)
        dx[i] += d_x_mid[i];

      /* Wo backward */
      float d_ao[N_EMBD];
      memset(d_ao, 0, sizeof(d_ao));
      linear_bwd_x(attn_wo[li], dx, N_EMBD, N_EMBD, d_ao);
      linear_bwd_w(fwd_attn_out[t], dx, N_EMBD, N_EMBD, d_attn_wo[li]);

      /* Attention backward: accumulate dk, dv; compute dq */
      float attn_scale = 1.0f / sqrtf((float)HEAD_DIM);
      memset(bwd_dq[t], 0, sizeof(bwd_dq[t]));

      for (int h = 0; h < N_HEAD; h++) {
        int hs = h * HEAD_DIM;
        float d_aw[BLOCK_SIZE];
        memset(d_aw, 0, sizeof(d_aw));
        for (int j = 0; j < HEAD_DIM; j++) {
          for (int s = 0; s <= t; s++) {
            d_aw[s] += d_ao[hs + j] * fwd_v[s][hs + j];
            bwd_dv[s][hs + j] += fwd_attn_w[t][h][s] * d_ao[hs + j];
          }
        }
        float dot = 0;
        for (int s = 0; s <= t; s++)
          dot += d_aw[s] * fwd_attn_w[t][h][s];
        float d_al[BLOCK_SIZE];
        for (int s = 0; s <= t; s++)
          d_al[s] = fwd_attn_w[t][h][s] * (d_aw[s] - dot);
        for (int s = 0; s <= t; s++) {
          for (int j = 0; j < HEAD_DIM; j++) {
            bwd_dq[t][hs + j] += d_al[s] * fwd_k[s][hs + j] * attn_scale;
            bwd_dk[s][hs + j] += d_al[s] * fwd_q[t][hs + j] * attn_scale;
          }
        }
      }

      memcpy(bwd_d_res[t], dx, sizeof(dx));
    }
  }

  /* Phase 2: forward over positions — QKV weight grads, rmsnorm, embeddings */
  for (int t = 0; t < n; t++) {
    for (int li = 0; li < N_LAYER; li++) {
      float d_xn[N_EMBD];
      memset(d_xn, 0, sizeof(d_xn));

      linear_bwd_x(attn_wq[li], bwd_dq[t], N_EMBD, N_EMBD, d_xn);
      linear_bwd_w(fwd_xn_attn[t], bwd_dq[t], N_EMBD, N_EMBD, d_attn_wq[li]);
      linear_bwd_x(attn_wk[li], bwd_dk[t], N_EMBD, N_EMBD, d_xn);
      linear_bwd_w(fwd_xn_attn[t], bwd_dk[t], N_EMBD, N_EMBD, d_attn_wk[li]);
      linear_bwd_x(attn_wv[li], bwd_dv[t], N_EMBD, N_EMBD, d_xn);
      linear_bwd_w(fwd_xn_attn[t], bwd_dv[t], N_EMBD, N_EMBD, d_attn_wv[li]);

      float d_x_in[N_EMBD];
      memset(d_x_in, 0, sizeof(d_x_in));
      rmsnorm_bwd(fwd_x_in[t], fwd_rms_attn[t], d_xn, N_EMBD, d_x_in);
      for (int i = 0; i < N_EMBD; i++)
        d_x_in[i] += bwd_d_res[t][i];

      float d_emb[N_EMBD];
      memset(d_emb, 0, sizeof(d_emb));
      rmsnorm_bwd(fwd_emb[t], fwd_rms_init[t], d_x_in, N_EMBD, d_emb);

      int tok = tokens[t];
      for (int i = 0; i < N_EMBD; i++) {
        d_wte[tok * N_EMBD + i] += d_emb[i];
        d_wpe[t * N_EMBD + i] += d_emb[i];
      }
    }
  }
}

/* ------------------------------------------------------------------ */
/*  Adam optimizer                                                    */
/* ------------------------------------------------------------------ */
static void adam_update(float *p, float *g, float *m, float *v, int sz,
                        float lr, float b1, float b2, float eps, int step) {
  float b1c = 1.0f - powf(b1, step + 1);
  float b2c = 1.0f - powf(b2, step + 1);
  for (int i = 0; i < sz; i++) {
    m[i] = b1 * m[i] + (1 - b1) * g[i];
    v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
    p[i] -= lr * (m[i] / b1c) / (sqrtf(v[i] / b2c) + eps);
    g[i] = 0;
  }
}

/* ------------------------------------------------------------------ */
/*  Weighted random sampling                                          */
/* ------------------------------------------------------------------ */
static int weighted_choice(const float *w, int n) {
  float total = 0;
  for (int i = 0; i < n; i++)
    total += w[i];
  float r = (float)rng_uniform() * total, cum = 0;
  for (int i = 0; i < n; i++) {
    cum += w[i];
    if (r < cum)
      return i;
  }
  return n - 1;
}

/* ------------------------------------------------------------------ */
/*  Inference forward (token-at-a-time with KV cache)                 */
/* ------------------------------------------------------------------ */
static float inf_keys[N_LAYER][BLOCK_SIZE][N_EMBD];
static float inf_vals[N_LAYER][BLOCK_SIZE][N_EMBD];

static void inference_forward(int token, int pos, float *logits) {
  float x[N_EMBD], tmp[MLP_DIM > N_EMBD ? MLP_DIM : N_EMBD];

  for (int i = 0; i < N_EMBD; i++)
    x[i] = wte[token * N_EMBD + i] + wpe[pos * N_EMBD + i];
  rmsnorm_fwd(x, N_EMBD, x);

  for (int li = 0; li < N_LAYER; li++) {
    float xr[N_EMBD];
    memcpy(xr, x, sizeof(xr));

    float xn[N_EMBD];
    rmsnorm_fwd(x, N_EMBD, xn);

    float q[N_EMBD], k[N_EMBD], v[N_EMBD];
    linear_fwd(xn, attn_wq[li], N_EMBD, N_EMBD, q);
    linear_fwd(xn, attn_wk[li], N_EMBD, N_EMBD, k);
    linear_fwd(xn, attn_wv[li], N_EMBD, N_EMBD, v);
    memcpy(inf_keys[li][pos], k, sizeof(k));
    memcpy(inf_vals[li][pos], v, sizeof(v));

    int seq_len = pos + 1;
    float attn_scale = 1.0f / sqrtf((float)HEAD_DIM);
    float ao[N_EMBD];
    for (int h = 0; h < N_HEAD; h++) {
      int hs = h * HEAD_DIM;
      float al[BLOCK_SIZE];
      for (int s = 0; s < seq_len; s++) {
        float dot = 0;
        for (int j = 0; j < HEAD_DIM; j++)
          dot += q[hs + j] * inf_keys[li][s][hs + j];
        al[s] = dot * attn_scale;
      }
      float mx = al[0];
      for (int s = 1; s < seq_len; s++)
        if (al[s] > mx)
          mx = al[s];
      float sm = 0;
      for (int s = 0; s < seq_len; s++) {
        al[s] = expf(al[s] - mx);
        sm += al[s];
      }
      float inv = 1.0f / sm;
      for (int s = 0; s < seq_len; s++)
        al[s] *= inv;
      for (int j = 0; j < HEAD_DIM; j++) {
        float acc = 0;
        for (int s = 0; s < seq_len; s++)
          acc += al[s] * inf_vals[li][s][hs + j];
        ao[hs + j] = acc;
      }
    }

    linear_fwd(ao, attn_wo[li], N_EMBD, N_EMBD, tmp);
    for (int i = 0; i < N_EMBD; i++)
      x[i] = tmp[i] + xr[i];

    memcpy(xr, x, sizeof(xr));
    float xn_m[N_EMBD];
    rmsnorm_fwd(x, N_EMBD, xn_m);
    float h1[MLP_DIM];
    linear_fwd(xn_m, mlp_fc1[li], MLP_DIM, N_EMBD, h1);
    for (int i = 0; i < MLP_DIM; i++)
      h1[i] = h1[i] > 0 ? h1[i] * h1[i] : 0;
    linear_fwd(h1, mlp_fc2[li], N_EMBD, MLP_DIM, tmp);
    for (int i = 0; i < N_EMBD; i++)
      x[i] = tmp[i] + xr[i];
  }

  linear_fwd(x, lm_head, vocab_size, N_EMBD, logits);
}

/* ------------------------------------------------------------------ */
/*  Main: training loop + inference                                   */
/* ------------------------------------------------------------------ */
int main(void) {
  load_dataset("input.txt");

  /* Shuffle dataset (Fisher-Yates on index, then permute) */
  int *doc_order = (int *)malloc(num_docs * sizeof(int));
  for (int i = 0; i < num_docs; i++)
    doc_order[i] = i;
  shuffle_ints(doc_order, num_docs);
  char (*docs_shuf)[MAX_DOC_LEN] = malloc((size_t)num_docs * MAX_DOC_LEN);
  for (int i = 0; i < num_docs; i++)
    memcpy(docs_shuf[i], docs[doc_order[i]], MAX_DOC_LEN);
  memcpy(docs, docs_shuf, (size_t)num_docs * MAX_DOC_LEN);
  free(docs_shuf);
  free(doc_order);

  printf("num docs: %d\n", num_docs);
  build_tokenizer();
  printf("vocab size: %d\n", vocab_size);
  init_params();

  float lr = 1e-2f, b1 = 0.9f, b2 = 0.95f, eps = 1e-8f;
  int num_steps = 5000;

  for (int step = 0; step < num_steps; step++) {
    char *doc = docs[step % num_docs];
    int doc_len = (int)strlen(doc);

    int tokens[MAX_DOC_LEN + 2];
    tokens[0] = BOS;
    for (int i = 0; i < doc_len; i++)
      tokens[i + 1] = char_to_id(doc[i]);
    tokens[doc_len + 1] = BOS;
    int n = BLOCK_SIZE < (doc_len + 1) ? BLOCK_SIZE : (doc_len + 1);

    float loss = gpt_forward(n, tokens);
    gpt_backward(n, tokens);

    float lr_t =
        lr * 0.5f * (1.0f + cosf((float)M_PI * step / (float)num_steps));
    int es = vocab_size * N_EMBD, ps = BLOCK_SIZE * N_EMBD;
    int as = N_EMBD * N_EMBD, ms = MLP_DIM * N_EMBD;
    adam_update(wte, d_wte, m_wte, v_wte, es, lr_t, b1, b2, eps, step);
    adam_update(wpe, d_wpe, m_wpe, v_wpe, ps, lr_t, b1, b2, eps, step);
    adam_update(lm_head, d_lm_head, m_lm_head, v_lm_head, es, lr_t, b1, b2,
                eps, step);
    for (int i = 0; i < N_LAYER; i++) {
      adam_update(attn_wq[i], d_attn_wq[i], m_wq[i], v_wq[i], as, lr_t, b1,
                  b2, eps, step);
      adam_update(attn_wk[i], d_attn_wk[i], m_wk[i], v_wk[i], as, lr_t, b1,
                  b2, eps, step);
      adam_update(attn_wv[i], d_attn_wv[i], m_wv[i], v_wv[i], as, lr_t, b1,
                  b2, eps, step);
      adam_update(attn_wo[i], d_attn_wo[i], m_wo[i], v_wo[i], as, lr_t, b1,
                  b2, eps, step);
      adam_update(mlp_fc1[i], d_mlp_fc1[i], m_fc1[i], v_fc1[i], ms, lr_t, b1,
                  b2, eps, step);
      adam_update(mlp_fc2[i], d_mlp_fc2[i], m_fc2[i], v_fc2[i], ms, lr_t, b1,
                  b2, eps, step);
    }

    printf("step %4d / %4d | loss %.4f\n", step + 1, num_steps, loss);
  }

  /* Inference */
  float temperature = 0.5f;
  printf("\n--- inference ---\n");
  for (int si = 0; si < 20; si++) {
    char sample[BLOCK_SIZE + 1];
    int slen = 0, tok = BOS;
    for (int pos = 0; pos < BLOCK_SIZE; pos++) {
      float logits[MAX_CHARS + 1], probs[MAX_CHARS + 1];
      inference_forward(tok, pos, logits);
      float inv_t = 1.0f / temperature;
      for (int i = 0; i < vocab_size; i++)
        logits[i] *= inv_t;
      softmax_fwd(logits, vocab_size, probs);
      tok = weighted_choice(probs, vocab_size);
      if (tok == BOS)
        break;
      if (tok < num_uchars)
        sample[slen++] = uchars_arr[tok];
    }
    sample[slen] = '\0';
    printf("sample %2d: %s\n", si + 1, sample);
    memset(inf_keys, 0, sizeof(inf_keys));
    memset(inf_vals, 0, sizeof(inf_vals));
  }

  /* Cleanup */
  free(wte);
  free(d_wte);
  free(m_wte);
  free(v_wte);
  free(wpe);
  free(d_wpe);
  free(m_wpe);
  free(v_wpe);
  free(lm_head);
  free(d_lm_head);
  free(m_lm_head);
  free(v_lm_head);
  for (int i = 0; i < N_LAYER; i++) {
    free(attn_wq[i]);
    free(d_attn_wq[i]);
    free(m_wq[i]);
    free(v_wq[i]);
    free(attn_wk[i]);
    free(d_attn_wk[i]);
    free(m_wk[i]);
    free(v_wk[i]);
    free(attn_wv[i]);
    free(d_attn_wv[i]);
    free(m_wv[i]);
    free(v_wv[i]);
    free(attn_wo[i]);
    free(d_attn_wo[i]);
    free(m_wo[i]);
    free(v_wo[i]);
    free(mlp_fc1[i]);
    free(d_mlp_fc1[i]);
    free(m_fc1[i]);
    free(v_fc1[i]);
    free(mlp_fc2[i]);
    free(d_mlp_fc2[i]);
    free(m_fc2[i]);
    free(v_fc2[i]);
  }
  return 0;
}
