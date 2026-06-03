/* Copyright (c) 2026 Nenad Mićić
 * SPDX-License-Identifier: Apache-2.0
 *
 * fp_determinism.c — cross-machine bit-exact determinism gate for fp_math.h
 *
 * Evaluates the core Q16.48 functions
 * over a fixed, committed, deterministic input grid, then hashes the *raw int64
 * bit patterns* of every output with FNV-1a 64 (no -lm, no libcrypto).
 *
 * The claim being made falsifiable: the same input grid should produce the same
 * raw integer bits on supported compilers/architectures. If the combined hash
 * matches on a new platform, bit-identity is tested, not asserted. Per-function
 * sub-hashes localize any drift.
 *
 *   build:  gcc -O3 -march=native -Wall -Wextra -std=c11 -o fp_determinism fp_determinism.c
 *           (NO -lm — pure integer)
 *   run:    ./fp_determinism                  # print hashes
 *           ./fp_determinism tests/determinism_golden.txt   # compare, exit 1 on mismatch
 *
 * Determinism rules honored: no Math.random, no wall-clock, no float in compute.
 * printf / file IO is display + the gate's own check — not part of the hashed math.
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "fp_math.h"

/* ---- FNV-1a 64-bit over raw little-endian int64 bytes ---- */
#define FNV64_OFFSET 1469598103934665603ULL
#define FNV64_PRIME  1099511628211ULL

static uint64_t fnv_init(void) { return FNV64_OFFSET; }

static uint64_t fnv_mix_i64(uint64_t h, int64_t v) {
    uint64_t u = (uint64_t)v;
    for (int b = 0; b < 8; b++) {               /* little-endian byte order, fixed everywhere */
        h ^= (uint64_t)(u & 0xff);
        h *= FNV64_PRIME;
        u >>= 8;
    }
    return h;
}

/* ---- deterministic input grid ----------------------------------------- *
 * A few thousand fixed_t values spanning the usable Q16.48 range:
 *   - explicit special values (zero, ±one, ±half, near ±32767, tiny)
 *   - powers of two, positive and negative, from 2^-44 to 2^14
 *   - a fixed-seed xorshift sweep mapped into [-32000, 32000] (Q16.48)
 * All committed in code — no RNG state shared with fp_rng_*, no wall clock. */

#define GRID_RANDOM 4096
#define GRID_CAP    (GRID_RANDOM + 256)

static fixed_t grid[GRID_CAP];
static int     grid_n = 0;

static void grid_push(fixed_t v) {
    if (grid_n < GRID_CAP) grid[grid_n++] = v;
}

/* local generator for input construction — independent of fp_rng_* so seeding
 * the library RNG later cannot perturb the grid. Fixed seed, fixed algorithm. */
static uint64_t gen_state;
static uint64_t gen_next(void) {
    gen_state ^= gen_state << 13;
    gen_state ^= gen_state >> 7;
    gen_state ^= gen_state << 17;
    return gen_state;
}

static void build_grid(void) {
    grid_n = 0;
    gen_state = 0x9E3779B97F4A7C15ULL;          /* fixed seed */

    /* specials */
    grid_push(0);
    grid_push(FP_ONE);
    grid_push(-FP_ONE);
    grid_push(FP_HALF);
    grid_push(-FP_HALF);
    grid_push(1);                                /* one ULP */
    grid_push(-1);
    grid_push(FP_ONE + 1);
    grid_push(FP_ONE - 1);
    grid_push((fixed_t)32000 << FP_PRECISION);   /* near +max */
    grid_push(-((fixed_t)32000 << FP_PRECISION));
    grid_push((fixed_t)181 << FP_PRECISION);     /* near the x^2 overflow boundary */
    grid_push(fp_from_int(7));
    grid_push(fp_from_int(-7));
    grid_push(fp_from_int(2));
    grid_push(fp_from_int(10));

    /* powers of two, both signs */
    for (int e = -44; e <= 14; e++) {
        fixed_t p = (e >= 0) ? (FP_ONE << e) : (FP_ONE >> (-e));
        grid_push(p);
        grid_push(-p);
    }

    /* fixed-seed sweep into [-32000, 32000] Q16.48 */
    const fixed_t span = (fixed_t)32000 << FP_PRECISION;  /* fits int64 */
    for (int i = 0; i < GRID_RANDOM; i++) {
        uint64_t r = gen_next();
        fixed_t mag = (fixed_t)(r % (uint64_t)span);
        grid_push((r & 1) ? mag : -mag);
    }
}

/* clamp a grid value to a strictly-positive domain (for sqrt/log/inv_sqrt) */
static fixed_t pos_of(fixed_t x) {
    fixed_t a = fp_abs(x);
    if (a < 1024) a += 1024;                     /* keep well away from zero */
    return a;
}

/* map a grid value into [-20, 20] for exp so outputs stay meaningful */
static fixed_t exp_arg(fixed_t x) {
    const fixed_t forty = (fixed_t)40 << FP_PRECISION;
    fixed_t m = (fixed_t)(((uint64_t)x) % (uint64_t)forty);
    return m - ((fixed_t)20 << FP_PRECISION);
}

int main(int argc, char **argv) {
    fp_math_init();                              /* required before CORDIC sincos */
    build_grid();

    /* per-function sub-hashes */
    uint64_t h_mul = fnv_init(), h_div = fnv_init();
    uint64_t h_sqrt = fnv_init(), h_sqrtf = fnv_init(), h_isqrt = fnv_init();
    uint64_t h_exp = fnv_init(), h_log = fnv_init();
    uint64_t h_cos = fnv_init(), h_sin = fnv_init();
    uint64_t h_sig = fnv_init(), h_silu = fnv_init();
    uint64_t h_rng = fnv_init();

    for (int i = 0; i < grid_n; i++) {
        fixed_t x = grid[i];
        fixed_t y = grid[(i * 7 + 3) % grid_n];  /* deterministic partner */

        h_mul  = fnv_mix_i64(h_mul,  fp_mul(x, y));
        if (y != 0) h_div = fnv_mix_i64(h_div, fp_div(x, y));

        fixed_t xp = pos_of(x);
        h_sqrt   = fnv_mix_i64(h_sqrt,   fp_sqrt(xp));
        h_sqrtf  = fnv_mix_i64(h_sqrtf,  fp_sqrt_fast(xp));
        h_isqrt  = fnv_mix_i64(h_isqrt,  fp_inv_sqrt(xp));
        h_log    = fnv_mix_i64(h_log,    fp_log(xp));

        h_exp    = fnv_mix_i64(h_exp,    fp_exp(exp_arg(x)));

        fixed_t c, s;
        fp_sincos(x, &c, &s);
        h_cos = fnv_mix_i64(h_cos, c);
        h_sin = fnv_mix_i64(h_sin, s);

        h_sig  = fnv_mix_i64(h_sig,  fp_sigmoid(x));
        h_silu = fnv_mix_i64(h_silu, fp_silu(x));
    }

    /* fixed-seed RNG sequence — seed the library state explicitly, sweep raw + uniform */
    fp_rng_state = 0x123456789ABCDEFULL;
    for (int i = 0; i < 1024; i++) {
        h_rng = fnv_mix_i64(h_rng, (int64_t)fp_rng_next());
        h_rng = fnv_mix_i64(h_rng, fp_rng_uniform());
    }

    /* combined hash = FNV over the ordered list of sub-hashes */
    uint64_t combined = fnv_init();
    uint64_t subs[] = { h_mul, h_div, h_sqrt, h_sqrtf, h_isqrt, h_exp,
                        h_log, h_cos, h_sin, h_sig, h_silu, h_rng };
    for (size_t i = 0; i < sizeof(subs) / sizeof(subs[0]); i++)
        combined = fnv_mix_i64(combined, (int64_t)subs[i]);

    printf("grid_inputs=%d\n", grid_n);
    printf("sub:fp_mul=%016llx\n",      (unsigned long long)h_mul);
    printf("sub:fp_div=%016llx\n",      (unsigned long long)h_div);
    printf("sub:fp_sqrt=%016llx\n",     (unsigned long long)h_sqrt);
    printf("sub:fp_sqrt_fast=%016llx\n",(unsigned long long)h_sqrtf);
    printf("sub:fp_inv_sqrt=%016llx\n", (unsigned long long)h_isqrt);
    printf("sub:fp_exp=%016llx\n",      (unsigned long long)h_exp);
    printf("sub:fp_log=%016llx\n",      (unsigned long long)h_log);
    printf("sub:fp_sincos_cos=%016llx\n",(unsigned long long)h_cos);
    printf("sub:fp_sincos_sin=%016llx\n",(unsigned long long)h_sin);
    printf("sub:fp_sigmoid=%016llx\n",  (unsigned long long)h_sig);
    printf("sub:fp_silu=%016llx\n",     (unsigned long long)h_silu);
    printf("sub:fp_rng=%016llx\n",      (unsigned long long)h_rng);
    printf("DETERMINISM_HASH=%016llx\n",(unsigned long long)combined);

    /* optional golden comparison: argv[1] = path to file holding the expected hash */
    if (argc > 1) {
        FILE *f = fopen(argv[1], "r");
        if (!f) { fprintf(stderr, "determinism: cannot open golden '%s'\n", argv[1]); return 2; }
        char want[64] = {0};
        if (!fgets(want, sizeof(want), f)) { fclose(f); fprintf(stderr, "determinism: empty golden\n"); return 2; }
        fclose(f);
        want[strcspn(want, " \t\r\n")] = 0;
        char got[64];
        snprintf(got, sizeof(got), "%016llx", (unsigned long long)combined);
        if (strcmp(got, want) == 0) {
            printf("determinism: PASS (matches golden %s)\n", want);
            return 0;
        }
        fprintf(stderr, "determinism: MISMATCH got=%s want=%s\n", got, want);
        return 1;
    }
    return 0;
}
