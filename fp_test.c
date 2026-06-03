/*
 * Author: Nenad Mićić
 * LinkedIn: https://be.linkedin.com/in/nenadmicic
 *
 * Copyright (c) 2026 Nenad Mićić
 * SPDX-License-Identifier: Apache-2.0
 *
 * fp_test.c — Comprehensive Unit Tests & Performance Benchmarks for fp_math.h
 * =============================================================================
 *
 * Tests every primitive in the library for:
 *   - Correctness (exact values, known identities, edge cases)
 *   - Precision (error bounds relative to known constants)
 *   - Robustness (boundary values, zero, negative, overflow-adjacent inputs)
 *   - Performance (nanoseconds per call, throughput)
 *
 * Compile: gcc -O3 -march=native -o fp_test fp_test.c
 * Run:     ./fp_test
 *
 * Exit code 0 = all tests pass, nonzero = failure count.
 */

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "fp_math.h"

/* ================================================================== */
/*  Test framework                                                      */
/* ================================================================== */

static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

#define SECTION(name) printf("\n=== %s ===\n", name)

static void check(const char *name, int condition) {
    tests_run++;
    if (condition) {
        tests_passed++;
    } else {
        tests_failed++;
        printf("  FAIL: %s\n", name);
    }
}

/* Check that value is within 'ulps' least-significant-bits of expected */
static void check_close(const char *name, fixed_t got, fixed_t expected,
                         int64_t max_err) {
    int64_t err = (int64_t)(got > expected ? got - expected : expected - got);
    tests_run++;
    if (err <= max_err) {
        tests_passed++;
    } else {
        tests_failed++;
        printf("  FAIL: %s — got %lld, expected %lld, err %lld (max %lld)\n",
               name, (long long)got, (long long)expected,
               (long long)err, (long long)max_err);
    }
}

/* Check relative error: |got - expected| / |expected| < rel_err_fp
 * where rel_err_fp is in fixed-point (e.g. FP_ONE/1000 = 0.1%) */
static void check_rel(const char *name, fixed_t got, fixed_t expected,
                       fixed_t max_rel_err) {
    tests_run++;
    if (expected == 0) {
        if (fp_abs(got) <= max_rel_err) { tests_passed++; return; }
        tests_failed++;
        printf("  FAIL: %s — got %lld, expected 0, abs %lld > %lld\n",
               name, (long long)got, (long long)fp_abs(got),
               (long long)max_rel_err);
        return;
    }
    fixed_t abs_err = fp_abs(got - expected);
    fixed_t abs_exp = fp_abs(expected);
    /* rel_err = abs_err / abs_exp, compare with max_rel_err in fp */
    /* abs_err < max_rel_err * abs_exp / FP_ONE */
    int128_t threshold = ((int128_t)max_rel_err * abs_exp) >> FP_PRECISION;
    if (abs_err <= (int64_t)threshold || abs_err <= 2) {
        tests_passed++;
    } else {
        tests_failed++;
        double rel = (double)abs_err / (double)abs_exp;
        printf("  FAIL: %s — got %.10f, expected %.10f, rel_err %.2e\n",
               name, fp_to_double(got), fp_to_double(expected), rel);
    }
}

/* Performance timing */
static double time_ns(struct timespec *start, struct timespec *end) {
    return (double)(end->tv_sec - start->tv_sec) * 1e9 +
           (double)(end->tv_nsec - start->tv_nsec);
}

#define BENCH_ITERS 100000

#define BENCH_START(label) \
    { \
        const char *_bench_label = label; \
        struct timespec _ts0, _ts1; \
        volatile fixed_t _sink = 0; \
        clock_gettime(CLOCK_MONOTONIC, &_ts0);

#define BENCH_END() \
        clock_gettime(CLOCK_MONOTONIC, &_ts1); \
        double _ns = time_ns(&_ts0, &_ts1) / BENCH_ITERS; \
        printf("  %-28s %8.1f ns/call (%d iters)\n", \
               _bench_label, _ns, BENCH_ITERS); \
        (void)_sink; \
    }

/* ================================================================== */
/*  1. Basic Arithmetic Tests                                           */
/* ================================================================== */

static void test_arithmetic(void) {
    SECTION("1. Basic Arithmetic");

    /* fp_from_int */
    check("fp_from_int(0) == 0", fp_from_int(0) == FP_ZERO);
    check("fp_from_int(1) == FP_ONE", fp_from_int(1) == FP_ONE);
    check("fp_from_int(-1) == -FP_ONE", fp_from_int(-1) == -FP_ONE);
    check("fp_from_int(100) >> 48 == 100",
          (fp_from_int(100) >> FP_PRECISION) == 100);
    check("fp_from_int(32767) representable",
          (fp_from_int(32767) >> FP_PRECISION) == 32767);

    /* fp_mul — exact cases */
    check("3 * 4 = 12",
          fp_mul(fp_from_int(3), fp_from_int(4)) == fp_from_int(12));
    check("(-3) * 4 = -12",
          fp_mul(fp_from_int(-3), fp_from_int(4)) == fp_from_int(-12));
    check("(-3) * (-4) = 12",
          fp_mul(fp_from_int(-3), fp_from_int(-4)) == fp_from_int(12));
    check("0 * anything = 0",
          fp_mul(FP_ZERO, fp_from_int(99)) == FP_ZERO);
    check("1 * x = x",
          fp_mul(FP_ONE, fp_from_int(42)) == fp_from_int(42));

    /* fp_mul — fractional */
    fixed_t half = FP_HALF;
    check_close("0.5 * 0.5 = 0.25", fp_mul(half, half), FP_ONE / 4, 1);
    check_close("0.5 * 2 = 1", fp_mul(half, fp_from_int(2)), FP_ONE, 0);

    /* fp_div — exact cases */
    check("10 / 2 = 5",
          fp_div(fp_from_int(10), fp_from_int(2)) == fp_from_int(5));
    check("(-10) / 2 = -5",
          fp_div(fp_from_int(-10), fp_from_int(2)) == fp_from_int(-5));
    check_close("1 / 3 * 3 ≈ 1",
                fp_mul(fp_div(FP_ONE, fp_from_int(3)), fp_from_int(3)),
                FP_ONE, 2);
    check_close("7 / 7 = 1",
                fp_div(fp_from_int(7), fp_from_int(7)), FP_ONE, 0);

    /* fp_abs */
    check("abs(5) = 5", fp_abs(fp_from_int(5)) == fp_from_int(5));
    check("abs(-5) = 5", fp_abs(fp_from_int(-5)) == fp_from_int(5));
    check("abs(0) = 0", fp_abs(FP_ZERO) == FP_ZERO);

    /* fp_max, fp_min */
    check("max(3,5) = 5",
          fp_max(fp_from_int(3), fp_from_int(5)) == fp_from_int(5));
    check("min(3,5) = 3",
          fp_min(fp_from_int(3), fp_from_int(5)) == fp_from_int(3));
    check("max(-1,1) = 1",
          fp_max(fp_from_int(-1), fp_from_int(1)) == fp_from_int(1));
    check("min(-1,1) = -1",
          fp_min(fp_from_int(-1), fp_from_int(1)) == fp_from_int(-1));

    /* Commutativity */
    fixed_t a = fp_from_int(7), b = fp_from_int(13);
    check("mul commutativity", fp_mul(a, b) == fp_mul(b, a));

    /* Distributivity: a*(b+c) ≈ a*b + a*c */
    fixed_t c = fp_from_int(3);
    check_close("distributivity",
                fp_mul(a, b + c), fp_mul(a, b) + fp_mul(a, c), 2);

    printf("  arithmetic: %d/%d passed\n", tests_passed, tests_run);
}

/* ================================================================== */
/*  2. Square Root Tests                                                */
/* ================================================================== */

static void test_sqrt(void) {
    SECTION("2. Square Root (fp_sqrt, fp_sqrt_fast, isqrt128)");

    /* isqrt128 exact perfect squares */
    check("isqrt128(0) = 0", isqrt128(0) == 0);
    check("isqrt128(1) = 1", isqrt128(1) == 1);
    check("isqrt128(4) = 2", isqrt128(4) == 2);
    check("isqrt128(9) = 3", isqrt128(9) == 3);
    check("isqrt128(100) = 10", isqrt128(100) == 10);
    check("isqrt128(10000) = 100", isqrt128(10000) == 100);

    /* isqrt128 non-perfect squares (floor) */
    check("isqrt128(2) = 1", isqrt128(2) == 1);
    check("isqrt128(3) = 1", isqrt128(3) == 1);
    check("isqrt128(5) = 2", isqrt128(5) == 2);
    check("isqrt128(8) = 2", isqrt128(8) == 2);
    check("isqrt128(99) = 9", isqrt128(99) == 9);

    /* isqrt128 large values */
    uint128_t big = (uint128_t)1 << 100;
    check("isqrt128(2^100) = 2^50", isqrt128(big) == ((uint128_t)1 << 50));

    /* fp_sqrt — perfect squares */
    check("sqrt(0) = 0", fp_sqrt(FP_ZERO) == FP_ZERO);
    check_close("sqrt(1) = 1", fp_sqrt(FP_ONE), FP_ONE, 2);
    check_close("sqrt(4) = 2", fp_sqrt(fp_from_int(4)), fp_from_int(2), 2);
    check_close("sqrt(9) = 3", fp_sqrt(fp_from_int(9)), fp_from_int(3), 2);
    check_close("sqrt(16) = 4", fp_sqrt(fp_from_int(16)), fp_from_int(4), 2);
    check_close("sqrt(100) = 10", fp_sqrt(fp_from_int(100)), fp_from_int(10), 2);
    check_close("sqrt(10000) = 100",
                fp_sqrt(fp_from_int(10000)), fp_from_int(100), 2);

    /* fp_sqrt — known irrationals via identity: sqrt(x)^2 = x */
    fixed_t sqrt2 = fp_sqrt(fp_from_int(2));
    check_rel("sqrt(2)^2 ≈ 2", fp_mul(sqrt2, sqrt2), fp_from_int(2),
              FP_ONE / 1000000000);

    fixed_t sqrt_half = fp_sqrt(FP_HALF);
    check_rel("sqrt(0.5)^2 ≈ 0.5", fp_mul(sqrt_half, sqrt_half), FP_HALF,
              FP_ONE / 1000000000);

    /* Identity: sqrt(x)^2 = x */
    for (int v = 1; v <= 20; v++) {
        fixed_t x = fp_from_int(v);
        fixed_t s = fp_sqrt(x);
        char label[64];
        snprintf(label, sizeof(label), "sqrt(%d)^2 = %d", v, v);
        check_close(label, fp_mul(s, s), x, x / 10000000);
    }

    /* Edge cases */
    check("sqrt(negative) = 0", fp_sqrt(-FP_ONE) == 0);
    check("sqrt(tiny) >= 0", fp_sqrt(1) >= 0);

    /* fp_sqrt_fast — less precise but still close */
    for (int v = 1; v <= 20; v++) {
        fixed_t x = fp_from_int(v);
        fixed_t s_fast = fp_sqrt_fast(x);
        fixed_t s_precise = fp_sqrt(x);
        char label[64];
        snprintf(label, sizeof(label), "sqrt_fast(%d) within 0.01%%", v);
        check_rel(label, s_fast, s_precise, FP_ONE / 10000);
    }

    printf("  sqrt: %d/%d passed\n", tests_passed, tests_run);
}

/* ================================================================== */
/*  3. Inverse Square Root Tests                                        */
/* ================================================================== */

static void test_inv_sqrt(void) {
    SECTION("3. Inverse Square Root (fp_inv_sqrt)");

    /* 1/sqrt(1) = 1 */
    check_rel("inv_sqrt(1) = 1", fp_inv_sqrt(FP_ONE), FP_ONE, FP_ONE / 100000);

    /* 1/sqrt(4) = 0.5 */
    check_rel("inv_sqrt(4) = 0.5",
              fp_inv_sqrt(fp_from_int(4)), FP_HALF, FP_ONE / 100000);

    /* Identity: inv_sqrt(x) * sqrt(x) = 1 */
    int vals[] = {1, 2, 3, 4, 5, 8, 10, 16, 25, 100, 1000};
    for (int i = 0; i < 11; i++) {
        fixed_t x = fp_from_int(vals[i]);
        fixed_t product = fp_mul(fp_inv_sqrt(x), fp_sqrt(x));
        char label[64];
        snprintf(label, sizeof(label), "inv_sqrt(%d)*sqrt(%d)≈1", vals[i], vals[i]);
        check_rel(label, product, FP_ONE, FP_ONE / 10000);
    }

    /* Identity: inv_sqrt(x)^2 * x = 1 */
    for (int i = 0; i < 11; i++) {
        fixed_t x = fp_from_int(vals[i]);
        fixed_t is = fp_inv_sqrt(x);
        fixed_t product = fp_mul(fp_mul(is, is), x);
        char label[64];
        snprintf(label, sizeof(label), "inv_sqrt(%d)^2*%d≈1", vals[i], vals[i]);
        check_rel(label, product, FP_ONE, FP_ONE / 1000);
    }

    /* Fractional inputs */
    check_rel("inv_sqrt(0.25) = 2",
              fp_inv_sqrt(FP_ONE / 4), fp_from_int(2), FP_ONE / 10000);

    /* Edge cases */
    check("inv_sqrt(0) = 0", fp_inv_sqrt(0) == 0);
    check("inv_sqrt(negative) = 0", fp_inv_sqrt(-FP_ONE) == 0);

    printf("  inv_sqrt: %d/%d passed\n", tests_passed, tests_run);
}

/* ================================================================== */
/*  4. Exponential Tests                                                */
/* ================================================================== */

static void test_exp(void) {
    SECTION("4. Exponential (fp_exp, fp_safe_exp)");

    /* exp(0) = 1 */
    check_close("exp(0) = 1", fp_exp(FP_ZERO), FP_ONE, FP_ONE / 10000);

    /* exp(1) ≈ 2.71828... — verify via exp(1)*exp(-1) = 1 */
    fixed_t e_val = fp_exp(FP_ONE);
    check("exp(1) > 2.71", e_val > fp_from_int(2) + FP_ONE * 71 / 100);
    check("exp(1) < 2.72", e_val < fp_from_int(2) + FP_ONE * 72 / 100);

    /* exp(-1) ≈ 1/exp(1) */
    fixed_t inv_e = fp_exp(-FP_ONE);
    check_rel("exp(1)*exp(-1) ≈ 1", fp_mul(e_val, inv_e), FP_ONE, FP_ONE / 10000);

    /* exp(x) * exp(-x) = 1 */
    for (int v = 1; v <= 10; v++) {
        fixed_t x = fp_from_int(v);
        fixed_t product = fp_mul(fp_exp(x), fp_exp(-x));
        char label[64];
        snprintf(label, sizeof(label), "exp(%d)*exp(-%d)≈1", v, v);
        check_rel(label, product, FP_ONE, FP_ONE / 1000);
    }

    /* exp(a+b) = exp(a)*exp(b) */
    fixed_t a = fp_from_int(2), b = fp_from_int(3);
    fixed_t exp_sum = fp_exp(a + b);
    fixed_t exp_product = fp_mul(fp_exp(a), fp_exp(b));
    check_rel("exp(2+3) = exp(2)*exp(3)", exp_sum, exp_product, FP_ONE / 1000);

    /* Small values: exp(x) ≈ 1 + x for |x| << 1 */
    fixed_t tiny = FP_ONE / 1000; /* 0.001 */
    fixed_t exp_tiny = fp_exp(tiny);
    check_rel("exp(0.001) ≈ 1.001", exp_tiny, FP_ONE + tiny, FP_ONE / 10000);

    /* fp_safe_exp clamping — clamp at ±10 to prevent Q16.48 overflow.
     * exp(10) ≈ 22026 fits.  exp(10.4) ≈ 32860 overflows int64. */
    fixed_t large = fp_from_int(100);
    fixed_t exp_large = fp_safe_exp(large);
    check("safe_exp(100) clamped to exp(10)", exp_large == fp_exp(fp_from_int(10)));

    fixed_t very_neg = -fp_from_int(100);
    check("safe_exp(-100) = 0", fp_safe_exp(very_neg) == 0);

    /* Values within clamp range compute normally */
    check("safe_exp(5) = exp(5)",
          fp_safe_exp(fp_from_int(5)) == fp_exp(fp_from_int(5)));
    check("safe_exp(-5) = exp(-5)",
          fp_safe_exp(-fp_from_int(5)) == fp_exp(-fp_from_int(5)));
    /* exp(-11) should return 0 (clamped), not overflow garbage */
    check("safe_exp(-11) = 0", fp_safe_exp(-fp_from_int(11)) == 0);

    /* Monotonicity */
    fixed_t prev = fp_exp(-fp_from_int(5));
    int monotone = 1;
    for (int v = -4; v <= 10; v++) {
        fixed_t cur = fp_exp(fp_from_int(v));
        if (cur < prev) monotone = 0;
        prev = cur;
    }
    check("exp is monotonically increasing", monotone);

    printf("  exp: %d/%d passed\n", tests_passed, tests_run);
}

/* ================================================================== */
/*  5. Logarithm Tests                                                  */
/* ================================================================== */

static void test_log(void) {
    SECTION("5. Logarithm (fp_log, fp_safe_log)");

    /* log(1) = 0 */
    check_close("log(1) = 0", fp_log(FP_ONE), FP_ZERO, FP_ONE / 10000);

    /* log(e) = 1 */
    fixed_t e_val = fp_exp(FP_ONE);
    check_rel("log(e) ≈ 1", fp_log(e_val), FP_ONE, FP_ONE / 10000);

    /* log(exp(x)) = x  (round-trip) */
    for (int v = -5; v <= 10; v++) {
        fixed_t x = fp_from_int(v);
        fixed_t rt = fp_log(fp_exp(x));
        char label[64];
        snprintf(label, sizeof(label), "log(exp(%d)) = %d", v, v);
        check_rel(label, rt, x, FP_ONE / 1000);
    }

    /* exp(log(x)) = x (round-trip) */
    int vals[] = {1, 2, 3, 5, 10, 100, 1000};
    for (int i = 0; i < 7; i++) {
        fixed_t x = fp_from_int(vals[i]);
        fixed_t rt = fp_exp(fp_log(x));
        char label[64];
        snprintf(label, sizeof(label), "exp(log(%d)) = %d", vals[i], vals[i]);
        check_rel(label, rt, x, FP_ONE / 100);
    }

    /* log(a*b) = log(a) + log(b) */
    fixed_t la = fp_from_int(3), lb = fp_from_int(7);
    fixed_t log_product = fp_log(fp_mul(la, lb));
    fixed_t sum_logs = fp_log(la) + fp_log(lb);
    check_rel("log(3*7) = log(3)+log(7)", log_product, sum_logs, FP_ONE / 1000);

    /* log(x^n) = n*log(x) */
    fixed_t x2 = fp_from_int(2);
    fixed_t log_8 = fp_log(fp_from_int(8));
    fixed_t three_log_2 = fp_log(x2) * 3;
    check_rel("log(8) = 3*log(2)", log_8, three_log_2, FP_ONE / 1000);

    /* Edge cases */
    check("log(0) = sentinel", fp_log(FP_ZERO) == -fp_from_int(50));
    check("log(negative) = sentinel", fp_log(-FP_ONE) == -fp_from_int(50));

    /* fp_safe_log clamps small values */
    fixed_t sl = fp_safe_log(1); /* 1 ulp */
    check("safe_log(tiny) is finite and negative", sl < 0 && sl > -fp_from_int(50));

    /* Monotonicity */
    fixed_t prev = fp_log(fp_from_int(1));
    int monotone = 1;
    for (int v = 2; v <= 100; v++) {
        fixed_t cur = fp_log(fp_from_int(v));
        if (cur <= prev) monotone = 0;
        prev = cur;
    }
    check("log is monotonically increasing on [1,100]", monotone);

    printf("  log: %d/%d passed\n", tests_passed, tests_run);
}

/* ================================================================== */
/*  6. Pi & Arctan Tests                                                */
/* ================================================================== */

static void test_pi(void) {
    SECTION("6. Pi & Arctan (fp_compute_pi, fp_atan_taylor)");

    fp_math_init();

    /* Pi accuracy: 3.14159265358979323846... */
    /* In Q16.48: pi * 2^48 ≈ 884,279,719,003,555 */
    fixed_t pi_expected = (fixed_t)884279719003555LL;
    check_rel("pi ≈ 3.14159265359", FP_PI, pi_expected, FP_ONE / 1000000000000LL);

    /* Pi digits check via fp_print comparison */
    double pi_double = fp_to_double(FP_PI);
    check("pi > 3.14159", pi_double > 3.14159);
    check("pi < 3.14160", pi_double < 3.14160);

    /* arctan(0) = 0 */
    check("atan(0) = 0", fp_atan_taylor(FP_ZERO, 40) == FP_ZERO);

    /* arctan(1) = pi/4 — but Taylor at x=1 converges very slowly,
     * so we verify via Machin's formula which already computes pi correctly.
     * Instead test arctan(1/2) which converges fast. */
    fixed_t atan_half = fp_atan_taylor(FP_ONE / 2, 60);
    /* arctan(0.5) ≈ 0.46364760900... */
    check("atan(0.5) > 0.463", fp_to_double(atan_half) > 0.463);
    check("atan(0.5) < 0.464", fp_to_double(atan_half) < 0.464);

    /* arctan(1/sqrt(3)) = pi/6 */
    fixed_t inv_sqrt3 = fp_div(FP_ONE, fp_sqrt(fp_from_int(3)));
    fixed_t atan_is3 = fp_atan_taylor(inv_sqrt3, 60);
    fixed_t pi_over_6 = fp_div(FP_PI, fp_from_int(6));
    check_rel("atan(1/sqrt(3)) ≈ pi/6", atan_is3, pi_over_6, FP_ONE / 100000);

    /* arctan is odd: arctan(-x) = -arctan(x) */
    fixed_t x = FP_ONE / 5;
    fixed_t at_pos = fp_atan_taylor(x, 40);
    fixed_t at_neg = fp_atan_taylor(-x, 40);
    check("atan(-x) = -atan(x)", at_neg == -at_pos);

    /* Machin's identity: pi = 16*atan(1/5) - 4*atan(1/239) */
    fixed_t atan5 = fp_atan_taylor(FP_ONE / 5, 40);
    fixed_t atan239 = fp_atan_taylor(FP_ONE / 239, 15);
    fixed_t machin_pi = (atan5 << 4) - (atan239 << 2);
    check_rel("Machin identity gives pi", machin_pi, FP_PI, FP_ONE / 1000000000000LL);

    printf("  pi/atan: %d/%d passed\n", tests_passed, tests_run);
}

/* ================================================================== */
/*  7. CORDIC Sin/Cos Tests                                             */
/* ================================================================== */

static void test_sincos(void) {
    SECTION("7. Sin/Cos (fp_sincos via CORDIC)");

    fp_math_init();
    fixed_t c, s;

    /* sin(0) = 0, cos(0) = 1 */
    fp_sincos(FP_ZERO, &c, &s);
    check_rel("cos(0) = 1", c, FP_ONE, FP_ONE / 100000);
    check_close("sin(0) = 0", s, FP_ZERO, FP_ONE / 100000);

    /* sin(pi/2) = 1, cos(pi/2) = 0 */
    fp_sincos(FP_PI >> 1, &c, &s);
    check_close("cos(pi/2) ≈ 0", c, FP_ZERO, FP_ONE / 10000);
    check_rel("sin(pi/2) ≈ 1", s, FP_ONE, FP_ONE / 100000);

    /* sin(pi) = 0, cos(pi) = -1 */
    fp_sincos(FP_PI, &c, &s);
    check_rel("cos(pi) ≈ -1", c, -FP_ONE, FP_ONE / 100000);
    check_close("sin(pi) ≈ 0", s, FP_ZERO, FP_ONE / 10000);

    /* sin(-pi/2) = -1, cos(-pi/2) = 0 */
    fp_sincos(-(FP_PI >> 1), &c, &s);
    check_close("cos(-pi/2) ≈ 0", c, FP_ZERO, FP_ONE / 10000);
    check_rel("sin(-pi/2) ≈ -1", s, -FP_ONE, FP_ONE / 100000);

    /* sin(2*pi) = 0, cos(2*pi) = 1 */
    fp_sincos(FP_PI << 1, &c, &s);
    check_rel("cos(2*pi) ≈ 1", c, FP_ONE, FP_ONE / 10000);
    check_close("sin(2*pi) ≈ 0", s, FP_ZERO, FP_ONE / 10000);

    /* sin(pi/6) = 0.5, cos(pi/6) = sqrt(3)/2 */
    fixed_t pi6 = fp_div(FP_PI, fp_from_int(6));
    fp_sincos(pi6, &c, &s);
    check_rel("sin(pi/6) ≈ 0.5", s, FP_HALF, FP_ONE / 10000);
    fixed_t sqrt3_2 = fp_div(fp_sqrt(fp_from_int(3)), fp_from_int(2));
    check_rel("cos(pi/6) ≈ sqrt(3)/2", c, sqrt3_2, FP_ONE / 10000);

    /* sin(pi/4) = cos(pi/4) = 1/sqrt(2) */
    fixed_t pi4 = FP_PI >> 2;
    fp_sincos(pi4, &c, &s);
    fixed_t inv_sqrt2 = fp_inv_sqrt(fp_from_int(2));
    check_rel("sin(pi/4) ≈ 1/sqrt(2)", s, inv_sqrt2, FP_ONE / 10000);
    check_rel("cos(pi/4) ≈ 1/sqrt(2)", c, inv_sqrt2, FP_ONE / 10000);

    /* Pythagorean identity: sin^2 + cos^2 = 1 */
    int pyth_pass = 1;
    for (int deg = 0; deg < 360; deg += 5) {
        fixed_t theta = fp_div(fp_mul(FP_PI, fp_from_int(deg)),
                               fp_from_int(180));
        fp_sincos(theta, &c, &s);
        fixed_t sum_sq = fp_mul(c, c) + fp_mul(s, s);
        if (fp_abs(sum_sq - FP_ONE) > FP_ONE / 1000)
            pyth_pass = 0;
    }
    check("sin^2+cos^2=1 for 72 angles", pyth_pass);

    /* Odd/even: sin(-x) = -sin(x), cos(-x) = cos(x) */
    fixed_t theta = fp_div(FP_PI, fp_from_int(7));
    fixed_t cp, sp, cn, sn;
    fp_sincos(theta, &cp, &sp);
    fp_sincos(-theta, &cn, &sn);
    check_close("sin(-x) = -sin(x)", sn, -sp, FP_ONE / 10000);
    check_close("cos(-x) = cos(x)", cn, cp, FP_ONE / 10000);

    /* Large angle reduction */
    fixed_t big_theta = FP_PI * 10 + (FP_PI >> 2); /* 10*pi + pi/4 */
    fp_sincos(big_theta, &c, &s);
    fp_sincos(pi4, &cp, &sp);
    check_rel("sin(10pi+pi/4) = sin(pi/4)", s, sp, FP_ONE / 1000);
    check_rel("cos(10pi+pi/4) = cos(pi/4)", c, cp, FP_ONE / 1000);

    /* Negative large angle */
    big_theta = -FP_PI * 8 + (FP_PI >> 1);
    fp_sincos(big_theta, &c, &s);
    fixed_t c2, s2;
    fp_sincos(FP_PI >> 1, &c2, &s2);
    check_rel("cos(-8pi+pi/2) = cos(pi/2)", c, c2, FP_ONE / 1000);

    printf("  sincos: %d/%d passed\n", tests_passed, tests_run);
}

/* ================================================================== */
/*  8. CORDIC Tables & Init Tests                                       */
/* ================================================================== */

static void test_cordic_tables(void) {
    SECTION("8. CORDIC Tables & Init");

    fp_math_init();

    /* Angle[0] = pi/4 */
    check_rel("angle[0] = pi/4",
              fp_cordic_angles[0], FP_PI >> 2, FP_ONE / 1000000000);

    /* Angles are monotonically decreasing */
    int mono = 1;
    for (int i = 1; i < FP_CORDIC_N; i++) {
        if (fp_cordic_angles[i] >= fp_cordic_angles[i-1])
            mono = 0;
    }
    check("CORDIC angles monotonically decreasing", mono);

    /* Angles are all positive */
    int all_pos = 1;
    for (int i = 0; i < FP_CORDIC_N; i++) {
        if (fp_cordic_angles[i] <= 0) all_pos = 0;
    }
    check("CORDIC angles all positive", all_pos);

    /* For large i, angle[i] ≈ 2^(-i) (arctan(x) ≈ x for small x) */
    for (int i = 30; i < FP_CORDIC_N; i++) {
        fixed_t expected = FP_ONE >> i;
        char label[64];
        snprintf(label, sizeof(label), "angle[%d] ≈ 2^(-%d)", i, i);
        check_close(label, fp_cordic_angles[i], expected, 1);
    }

    /* CORDIC gain ≈ 0.6072529350 */
    fixed_t gain_expected = (fixed_t)170926505739103LL;
    check_rel("CORDIC gain ≈ 0.60725",
              fp_cordic_gain, gain_expected, FP_ONE / 100000000);

    /* gain * sqrt(prod(1 + 2^{-2i})) ≈ 1 */
    /* Already validated in fp_test but check identity here */
    fixed_t k_sq = FP_ONE;
    for (int i = 0; i < FP_CORDIC_N; i++) {
        fixed_t sv = (2 * i < 63) ? (FP_ONE >> (2 * i)) : 0;
        k_sq = fp_mul(k_sq, FP_ONE + sv);
    }
    fixed_t k_val = fp_sqrt(k_sq);
    fixed_t product = fp_mul(fp_cordic_gain, k_val);
    check_rel("gain * K = 1", product, FP_ONE, FP_ONE / 100000);

    /* Re-init is idempotent */
    fixed_t pi_before = FP_PI;
    fixed_t gain_before = fp_cordic_gain;
    fp_math_init();
    check("re-init idempotent (pi)", FP_PI == pi_before);
    check("re-init idempotent (gain)", fp_cordic_gain == gain_before);

    printf("  cordic_tables: %d/%d passed\n", tests_passed, tests_run);
}

/* ================================================================== */
/*  9. PRNG Tests                                                       */
/* ================================================================== */

static void test_rng(void) {
    SECTION("9. PRNG (xorshift64, uniform, gaussian, shuffle)");

    /* Reset RNG for reproducibility */
    fp_rng_state = 42;

    /* Deterministic: same seed → same sequence */
    unsigned long long s1 = fp_rng_next();
    unsigned long long s2 = fp_rng_next();
    fp_rng_state = 42;
    check("deterministic seq[0]", fp_rng_next() == s1);
    check("deterministic seq[1]", fp_rng_next() == s2);

    /* fp_rng_uniform in [0, FP_ONE) */
    fp_rng_state = 42;
    int in_range = 1;
    for (int i = 0; i < 10000; i++) {
        fixed_t u = fp_rng_uniform();
        if (u < 0 || u >= FP_ONE) in_range = 0;
    }
    check("uniform in [0, FP_ONE) for 10K samples", in_range);

    /* Uniform mean ≈ 0.5 (use int128 accumulator to avoid overflow) */
    fp_rng_state = 123;
    int128_t sum128 = 0;
    int N = 100000;
    for (int i = 0; i < N; i++)
        sum128 += fp_rng_uniform();
    fixed_t mean = (fixed_t)(sum128 / N);
    check_rel("uniform mean ≈ 0.5", mean, FP_HALF, FP_ONE / 50);

    /* Gaussian mean ≈ 0 when mean param = 0 */
    fp_rng_state = 456;
    sum128 = 0;
    N = 10000;
    for (int i = 0; i < N; i++)
        sum128 += fp_gaussian(FP_ZERO, FP_ONE);
    mean = (fixed_t)(sum128 / N);
    check_rel("gaussian(0,1) mean ≈ 0", mean, FP_ZERO, FP_ONE / 10);

    /* Gaussian with offset: mean should track */
    fp_rng_state = 789;
    sum128 = 0;
    fixed_t target_mean = fp_from_int(5);
    for (int i = 0; i < N; i++)
        sum128 += fp_gaussian(target_mean, FP_ONE);
    mean = (fixed_t)(sum128 / N);
    check_rel("gaussian(5,1) mean ≈ 5", mean, target_mean, FP_ONE / 5);

    /* Shuffle: all elements preserved */
    int arr[20];
    for (int i = 0; i < 20; i++) arr[i] = i;
    fp_rng_state = 1000;
    fp_shuffle_ints(arr, 20);
    int present[20] = {0};
    for (int i = 0; i < 20; i++) present[arr[i]] = 1;
    int all_present = 1;
    for (int i = 0; i < 20; i++)
        if (!present[i]) all_present = 0;
    check("shuffle preserves all elements", all_present);

    /* Shuffle actually permutes (probabilistic — failing is ~1/20! ≈ 0) */
    int is_identity = 1;
    for (int i = 0; i < 20; i++)
        if (arr[i] != i) is_identity = 0;
    check("shuffle actually permutes", !is_identity);

    /* Shuffle of 1 element is no-op */
    int single[] = {42};
    fp_shuffle_ints(single, 1);
    check("shuffle(1 element) unchanged", single[0] == 42);

    /* Shuffle of 0 elements doesn't crash */
    fp_shuffle_ints(NULL, 0);
    check("shuffle(0 elements) no crash", 1);

    printf("  rng: %d/%d passed\n", tests_passed, tests_run);
}

/* ================================================================== */
/*  10. Euler's Identity: e^(i*pi) + 1 = 0                             */
/* ================================================================== */

static void test_euler(void) {
    SECTION("10. Euler's Identity: exp(i*pi) + 1 = 0");

    fp_math_init();

    /* cos(pi) + i*sin(pi) should be (-1, 0) */
    fixed_t c, s;
    fp_sincos(FP_PI, &c, &s);

    /* e^(i*pi) = cos(pi) + i*sin(pi) = -1 + 0i
     * So e^(i*pi) + 1 = 0 */
    fixed_t real_part = c + FP_ONE; /* should be 0 */
    fixed_t imag_part = s;          /* should be 0 */

    check_close("Re(e^(i*pi) + 1) = 0", real_part, FP_ZERO, FP_ONE / 10000);
    check_close("Im(e^(i*pi)) = 0", imag_part, FP_ZERO, FP_ONE / 10000);

    /* Also verify via exp: |exp(i*theta)| = 1 for any theta */
    /* We can't compute complex exp directly, but cos^2+sin^2=1 serves */
    fixed_t mag_sq = fp_mul(c, c) + fp_mul(s, s);
    check_rel("|e^(i*pi)| = 1", mag_sq, FP_ONE, FP_ONE / 10000);

    printf("  euler: %d/%d passed\n", tests_passed, tests_run);
}

/* ================================================================== */
/*  11. Cross-function Consistency Tests                                */
/* ================================================================== */

static void test_consistency(void) {
    SECTION("11. Cross-function Consistency");

    fp_math_init();

    /* sqrt via exp/log: sqrt(x) = exp(log(x)/2) */
    for (int v = 2; v <= 10; v++) {
        fixed_t x = fp_from_int(v);
        fixed_t sqrt_direct = fp_sqrt(x);
        fixed_t sqrt_via_el = fp_exp(fp_div(fp_log(x), fp_from_int(2)));
        char label[64];
        snprintf(label, sizeof(label), "sqrt(%d) via exp(log/2)", v);
        check_rel(label, sqrt_via_el, sqrt_direct, FP_ONE / 100);
    }

    /* x^n via exp/log: x^3 = exp(3*log(x)) */
    fixed_t x = fp_from_int(2);
    fixed_t x_cubed = fp_mul(fp_mul(x, x), x);
    fixed_t x_cubed_el = fp_exp(fp_mul(fp_from_int(3), fp_log(x)));
    check_rel("2^3 via exp(3*log(2))", x_cubed_el, x_cubed, FP_ONE / 100);

    /* sin via exp: sin(x) = Im(exp(ix)) — tested via sincos */
    /* cos(a+b) = cos(a)cos(b) - sin(a)sin(b) */
    fixed_t ca, sa, cb, sb, cab, sab;
    fixed_t a = FP_PI / 5, b = FP_PI / 7;
    fp_sincos(a, &ca, &sa);
    fp_sincos(b, &cb, &sb);
    fp_sincos(a + b, &cab, &sab);
    fixed_t cos_add = fp_mul(ca, cb) - fp_mul(sa, sb);
    fixed_t sin_add = fp_mul(sa, cb) + fp_mul(ca, sb);
    check_rel("cos(a+b) = cos(a)cos(b)-sin(a)sin(b)", cab, cos_add, FP_ONE / 1000);
    check_rel("sin(a+b) = sin(a)cos(b)+cos(a)sin(b)", sab, sin_add, FP_ONE / 1000);

    /* Double angle: sin(2x) = 2*sin(x)*cos(x) */
    fixed_t c_2x, s_2x;
    fp_sincos(2 * a, &c_2x, &s_2x);
    fixed_t sin_double = 2 * fp_mul(sa, ca);
    check_rel("sin(2x) = 2*sin(x)*cos(x)", s_2x, sin_double, FP_ONE / 1000);

    /* log(exp(x)) round-trip through the full range */
    for (int v = -10; v <= 10; v++) {
        fixed_t xv = fp_from_int(v);
        fixed_t rt = fp_log(fp_exp(xv));
        char label[64];
        snprintf(label, sizeof(label), "log(exp(%d)) round-trip", v);
        check_rel(label, rt, xv, FP_ONE / 100);
    }

    printf("  consistency: %d/%d passed\n", tests_passed, tests_run);
}

/* ================================================================== */
/*  12. Edge Cases & Robustness                                         */
/* ================================================================== */

static void test_edge_cases(void) {
    SECTION("12. Edge Cases & Robustness");

    /* Division by very small numbers */
    fixed_t big = fp_div(FP_ONE, FP_ONE / 10000); /* 1 / 0.0001 = 10000 */
    check_rel("1 / 0.0001 = 10000", big, fp_from_int(10000), FP_ONE / 1000);

    /* Multiplication near limits (Q16.48 max int ≈ 32767) */
    fixed_t large = fp_from_int(100);
    fixed_t product = fp_mul(large, large); /* 100^2 = 10000 */
    check("100^2 = 10000",
          (product >> FP_PRECISION) == 10000LL);

    /* Moderately small * moderately large */
    fixed_t tiny_val = FP_ONE / 1000;
    fixed_t big_val = fp_from_int(1000);
    check_close("0.001 * 1000 ≈ 1", fp_mul(tiny_val, big_val), FP_ONE, FP_ONE / 100);

    /* fp_sqrt of very small values */
    fixed_t sqrt_tiny = fp_sqrt(FP_ONE / 10000); /* sqrt(0.0001) = 0.01 */
    check_rel("sqrt(0.0001) ≈ 0.01", sqrt_tiny, FP_ONE / 100, FP_ONE / 1000);

    /* fp_sqrt of largest safe value */
    fixed_t max_safe = fp_from_int(30000);
    fixed_t sqrt_max = fp_sqrt(max_safe);
    check("sqrt(30000) > 0", sqrt_max > 0);
    check_rel("sqrt(30000)^2 ≈ 30000",
              fp_mul(sqrt_max, sqrt_max), max_safe, max_safe / 10000);

    /* exp of zero */
    check("exp(0) = 1 exactly", fp_abs(fp_exp(FP_ZERO) - FP_ONE) < 100);

    /* log of very large value */
    fixed_t log_big = fp_log(fp_from_int(10000));
    /* ln(10000) ≈ 9.2103 */
    fixed_t expected = fp_from_int(9) + FP_ONE * 2103 / 10000;
    check_rel("log(10000) ≈ 9.21", log_big, expected, FP_ONE / 100);

    /* sincos with zero */
    fixed_t c, s;
    fp_sincos(FP_ZERO, &c, &s);
    check("sin(0) very close to 0", fp_abs(s) < FP_ONE / 10000);
    check("cos(0) very close to 1", fp_abs(c - FP_ONE) < FP_ONE / 10000);

    /* fp_abs of INT64_MIN would overflow — we don't claim to handle it,
     * but let's ensure normal negative values work */
    fixed_t neg = -fp_from_int(12345);
    check("abs(-12345) = 12345", fp_abs(neg) == fp_from_int(12345));

    /* mul with FP_HALF */
    check("x * 0.5 = x >> 1 (for even x)",
          fp_mul(fp_from_int(100), FP_HALF) == fp_from_int(50));

    printf("  edge_cases: %d/%d passed\n", tests_passed, tests_run);
}

/* ================================================================== */
/*  13. Performance Benchmarks                                          */
/* ================================================================== */

static void bench_all(void) {
    SECTION("13. Performance Benchmarks");

    fp_math_init();

    fixed_t x = fp_from_int(3) + FP_HALF; /* 3.5 */
    fixed_t y = fp_from_int(2) + FP_ONE / 3; /* ~2.333 */

    /* Arithmetic */
    BENCH_START("fp_mul")
    for (int i = 0; i < BENCH_ITERS; i++)
        _sink += fp_mul(x + i, y);
    BENCH_END()

    BENCH_START("fp_div")
    for (int i = 0; i < BENCH_ITERS; i++)
        _sink += fp_div(x + i, y);
    BENCH_END()

    /* Square roots */
    BENCH_START("fp_sqrt (precise)")
    for (int i = 0; i < BENCH_ITERS; i++)
        _sink += fp_sqrt(x + i);
    BENCH_END()

    BENCH_START("fp_sqrt_fast (approx)")
    for (int i = 0; i < BENCH_ITERS; i++)
        _sink += fp_sqrt_fast(x + i);
    BENCH_END()

    BENCH_START("isqrt128")
    for (int i = 0; i < BENCH_ITERS; i++)
        _sink += (fixed_t)isqrt128((uint128_t)(uint64_t)(x + i) << 48);
    BENCH_END()

    BENCH_START("fp_inv_sqrt")
    for (int i = 0; i < BENCH_ITERS; i++)
        _sink += fp_inv_sqrt(x + i);
    BENCH_END()

    /* Transcendentals */
    BENCH_START("fp_exp (positive)")
    for (int i = 0; i < BENCH_ITERS; i++)
        _sink += fp_exp(FP_ONE + (i & 0xFF));
    BENCH_END()

    BENCH_START("fp_exp (negative)")
    for (int i = 0; i < BENCH_ITERS; i++)
        _sink += fp_exp(-(FP_ONE + (i & 0xFF)));
    BENCH_END()

    BENCH_START("fp_safe_exp")
    for (int i = 0; i < BENCH_ITERS; i++)
        _sink += fp_safe_exp(FP_ONE * (i % 60 - 30));
    BENCH_END()

    BENCH_START("fp_log")
    for (int i = 0; i < BENCH_ITERS; i++)
        _sink += fp_log(FP_ONE + (i & 0xFFF));
    BENCH_END()

    BENCH_START("fp_safe_log")
    for (int i = 0; i < BENCH_ITERS; i++)
        _sink += fp_safe_log(FP_ONE / 2 + (i & 0xFFF));
    BENCH_END()

    /* Trigonometry */
    BENCH_START("fp_sincos")
    for (int i = 0; i < BENCH_ITERS; i++) {
        fixed_t c, s;
        fp_sincos(FP_PI * i / BENCH_ITERS, &c, &s);
        _sink += c + s;
    }
    BENCH_END()

    /* Arctan */
    BENCH_START("fp_atan_taylor(x, 40)")
    for (int i = 0; i < BENCH_ITERS; i++)
        _sink += fp_atan_taylor(FP_ONE / (5 + (i & 0xF)), 40);
    BENCH_END()

    /* PRNG */
    fp_rng_state = 42;
    BENCH_START("fp_rng_next")
    for (int i = 0; i < BENCH_ITERS; i++)
        _sink += (fixed_t)fp_rng_next();
    BENCH_END()

    BENCH_START("fp_rng_uniform")
    for (int i = 0; i < BENCH_ITERS; i++)
        _sink += fp_rng_uniform();
    BENCH_END()

    BENCH_START("fp_gaussian")
    for (int i = 0; i < BENCH_ITERS; i++)
        _sink += fp_gaussian(FP_ZERO, FP_ONE);
    BENCH_END()

    /* Composite operations (common in neural nets) */
    BENCH_START("rmsnorm pattern (sum+inv_sqrt)")
    for (int i = 0; i < BENCH_ITERS; i++) {
        fixed_t ms = fp_mul(x + i, x + i) + fp_mul(y, y);
        ms = ms / 32 + FP_ONE / 100000;
        _sink += fp_inv_sqrt(ms);
    }
    BENCH_END()

    BENCH_START("softmax pattern (exp+div)")
    for (int i = 0; i < BENCH_ITERS; i++) {
        fixed_t e1 = fp_safe_exp(FP_ONE * (i & 7));
        fixed_t e2 = fp_safe_exp(FP_ONE * ((i + 3) & 7));
        fixed_t sum = e1 + e2;
        _sink += fp_div(e1, sum);
    }
    BENCH_END()

    BENCH_START("adam pattern (sqrt+div)")
    for (int i = 0; i < BENCH_ITERS; i++) {
        fixed_t v_hat = fp_from_int(1) + (i & 0xFF);
        fixed_t denom = fp_sqrt_fast(v_hat) + FP_ONE / 100000000;
        _sink += fp_div(FP_ONE / 100, denom);
    }
    BENCH_END()

    /* Init cost */
    {
        struct timespec ts0, ts1;
        fp_math_initialized = 0; /* force re-init */
        clock_gettime(CLOCK_MONOTONIC, &ts0);
        fp_math_init();
        clock_gettime(CLOCK_MONOTONIC, &ts1);
        double ms = time_ns(&ts0, &ts1) / 1e6;
        printf("  %-28s %8.2f ms (one-time)\n", "fp_math_init", ms);
    }
}

/* ================================================================== */
/*  14. Numerical Stability Stress Tests                                */
/* ================================================================== */

static void test_stress(void) {
    SECTION("14. Numerical Stability Stress Tests");

    /* Repeated multiply/divide should preserve value */
    fixed_t x = fp_from_int(7);
    fixed_t val = x;
    for (int i = 0; i < 100; i++) {
        val = fp_mul(val, fp_from_int(3));
        val = fp_div(val, fp_from_int(3));
    }
    check_rel("100x mul(3)/div(3) preserves value", val, x, FP_ONE / 100);

    /* Repeated sqrt/square should converge to 1 */
    val = fp_from_int(100);
    for (int i = 0; i < 20; i++)
        val = fp_sqrt(val);
    for (int i = 0; i < 20; i++)
        val = fp_mul(val, val);
    check_rel("20x sqrt then 20x square ≈ original",
              val, fp_from_int(100), fp_from_int(1));

    /* Softmax stability: exp(x-max) / sum(exp(x_i-max)) should sum to 1 */
    fixed_t logits[10];
    fp_rng_state = 999;
    for (int i = 0; i < 10; i++)
        logits[i] = fp_gaussian(FP_ZERO, fp_from_int(3));

    fixed_t mx = logits[0];
    for (int i = 1; i < 10; i++)
        if (logits[i] > mx) mx = logits[i];

    fixed_t sum_exp = 0;
    fixed_t probs[10];
    for (int i = 0; i < 10; i++) {
        probs[i] = fp_safe_exp(logits[i] - mx);
        sum_exp += probs[i];
    }
    fixed_t prob_sum = 0;
    for (int i = 0; i < 10; i++) {
        probs[i] = fp_div(probs[i], sum_exp);
        prob_sum += probs[i];
    }
    check_rel("softmax probs sum to 1", prob_sum, FP_ONE, FP_ONE / 1000);

    /* All probs non-negative */
    int all_nn = 1;
    for (int i = 0; i < 10; i++)
        if (probs[i] < 0) all_nn = 0;
    check("softmax probs all >= 0", all_nn);

    /* Cross-entropy: -sum(p*log(q)) should be non-negative for valid distributions */
    fixed_t ce = 0;
    for (int i = 0; i < 10; i++) {
        if (probs[i] > 0)
            ce += -fp_mul(probs[i], fp_safe_log(probs[i]));
    }
    check("cross-entropy >= 0", ce >= 0);

    /* Gradient clipping simulation: large gradient * small lr = bounded update */
    fixed_t grad = fp_from_int(100);
    fixed_t lr = FP_ONE / 100;
    fixed_t update = fp_mul(lr, grad);
    check_close("100 * 0.01 ≈ 1", update, FP_ONE, FP_ONE / 100);

    /* Accumulated small values: sum of 1/N for N = 1000 */
    fixed_t harmonic = 0;
    for (int i = 1; i <= 1000; i++)
        harmonic += fp_div(FP_ONE, fp_from_int(i));
    /* H(1000) ≈ 7.4854... */
    check_rel("harmonic(1000) ≈ 7.485", harmonic,
              fp_from_int(7) + FP_ONE * 4854 / 10000, FP_ONE / 100);

    printf("  stress: %d/%d passed\n", tests_passed, tests_run);
}

/* ================================================================== */
/*  15. Sigmoid & SiLU Tests                                            */
/* ================================================================== */

static void test_sigmoid_silu(void) {
    SECTION("15. Sigmoid & SiLU (fp_sigmoid, fp_silu)");

    /* sigmoid(0) = 0.5 */
    check_rel("sigmoid(0) = 0.5", fp_sigmoid(FP_ZERO), FP_HALF, FP_ONE / 10000);

    /* sigmoid is bounded: 0 < sigmoid(x) < 1 for all finite x */
    int vals[] = {-10, -5, -1, 0, 1, 5, 10};
    for (int i = 0; i < 7; i++) {
        fixed_t x = fp_from_int(vals[i]);
        fixed_t s = fp_sigmoid(x);
        char label[64];
        snprintf(label, sizeof(label), "0 < sigmoid(%d) < 1", vals[i]);
        check(label, s > 0 && s < FP_ONE);
    }

    /* sigmoid(-x) = 1 - sigmoid(x) */
    for (int i = 0; i < 7; i++) {
        fixed_t x = fp_from_int(vals[i]);
        fixed_t s_pos = fp_sigmoid(x);
        fixed_t s_neg = fp_sigmoid(-x);
        char label[64];
        snprintf(label, sizeof(label), "sigmoid(-%d) = 1 - sigmoid(%d)", vals[i], vals[i]);
        check_rel(label, s_neg, FP_ONE - s_pos, FP_ONE / 1000);
    }

    /* sigmoid monotonically increasing */
    int mono = 1;
    fixed_t prev = fp_sigmoid(-fp_from_int(10));
    for (int v = -9; v <= 10; v++) {
        fixed_t cur = fp_sigmoid(fp_from_int(v));
        if (cur < prev) mono = 0;
        prev = cur;
    }
    check("sigmoid is monotonically increasing", mono);

    /* Known values (approximate):
     * sigmoid(-10) ≈ 0.0000454
     * sigmoid(-5)  ≈ 0.00669
     * sigmoid(-1)  ≈ 0.26894
     * sigmoid(1)   ≈ 0.73106
     * sigmoid(5)   ≈ 0.99331
     * sigmoid(10)  ≈ 0.99995 */
    fixed_t sig_1 = fp_sigmoid(FP_ONE);
    check("sigmoid(1) > 0.73", fp_to_double(sig_1) > 0.73);
    check("sigmoid(1) < 0.74", fp_to_double(sig_1) < 0.74);

    fixed_t sig_neg1 = fp_sigmoid(-FP_ONE);
    check("sigmoid(-1) > 0.26", fp_to_double(sig_neg1) > 0.26);
    check("sigmoid(-1) < 0.27", fp_to_double(sig_neg1) < 0.27);

    fixed_t sig_5 = fp_sigmoid(fp_from_int(5));
    check("sigmoid(5) > 0.99", fp_to_double(sig_5) > 0.99);

    /* SiLU tests */
    /* silu(0) = 0 * sigmoid(0) = 0 */
    check_close("silu(0) = 0", fp_silu(FP_ZERO), FP_ZERO, FP_ONE / 10000);

    /* silu(x) ≈ x for large positive x (sigmoid → 1) */
    fixed_t silu_10 = fp_silu(fp_from_int(10));
    check_rel("silu(10) ≈ 10", silu_10, fp_from_int(10), FP_ONE / 100);

    /* silu(x) ≈ 0 for large negative x (sigmoid → 0, x → -inf, product → 0) */
    fixed_t silu_neg10 = fp_silu(-fp_from_int(10));
    check("silu(-10) ≈ 0", fp_abs(silu_neg10) < FP_ONE / 100);

    /* silu has minimum at x ≈ -1.278 with value ≈ -0.278 */
    /* Verify it goes negative */
    fixed_t silu_neg1 = fp_silu(-FP_ONE);
    check("silu(-1) < 0", silu_neg1 < 0);
    check("silu(-1) > -0.5", fp_to_double(silu_neg1) > -0.5);

    /* silu(1) ≈ 0.731 */
    fixed_t silu_1 = fp_silu(FP_ONE);
    check("silu(1) > 0.73", fp_to_double(silu_1) > 0.73);
    check("silu(1) < 0.74", fp_to_double(silu_1) < 0.74);

    /* SiLU is smooth: verify continuity around 0 */
    fixed_t eps = FP_ONE / 100;
    fixed_t silu_eps = fp_silu(eps);
    fixed_t silu_neg_eps = fp_silu(-eps);
    check("silu continuous around 0",
          fp_abs(silu_eps - silu_neg_eps) < FP_ONE / 10);

    printf("  sigmoid/silu: %d/%d passed\n", tests_passed, tests_run);
}

/* ================================================================== */
/*  17. Safetensors Float16/BFloat16 Conversion Tests                   */
/* ================================================================== */

#include "safetensors.h"

static void test_float_conversion(void) {
    SECTION("17. Float16/BFloat16 → Q16.48 Conversion");

    /* float16: test known values */
    /* 0x0000 = +0.0 */
    check("f16 +0", float16_to_q1648(0x0000) == 0);
    /* 0x8000 = -0.0 */
    check("f16 -0", float16_to_q1648(0x8000) == 0);

    /* 0x3C00 = 1.0 */
    check_close("f16 1.0", float16_to_q1648(0x3C00), FP_ONE, 1);
    /* 0xBC00 = -1.0 */
    check_close("f16 -1.0", float16_to_q1648(0xBC00), -FP_ONE, 1);

    /* 0x4000 = 2.0 */
    check_close("f16 2.0", float16_to_q1648(0x4000), fp_from_int(2), 1);

    /* 0x3800 = 0.5 */
    check_close("f16 0.5", float16_to_q1648(0x3800), FP_HALF, 1);

    /* 0x4200 = 3.0 */
    check_close("f16 3.0", float16_to_q1648(0x4200), fp_from_int(3), 1);

    /* bfloat16: test known values */
    /* 0x0000 = +0.0 */
    check("bf16 +0", bfloat16_to_q1648(0x0000) == 0);

    /* 0x3F80 = 1.0 (same as float32 upper 16 bits) */
    check_close("bf16 1.0", bfloat16_to_q1648(0x3F80), FP_ONE, FP_ONE / 100);

    /* 0xBF80 = -1.0 */
    check_close("bf16 -1.0", bfloat16_to_q1648(0xBF80), -FP_ONE, FP_ONE / 100);

    /* 0x4000 = 2.0 */
    check_close("bf16 2.0", bfloat16_to_q1648(0x4000), fp_from_int(2), FP_ONE / 100);

    /* 0x3F00 = 0.5 */
    check_close("bf16 0.5", bfloat16_to_q1648(0x3F00), FP_HALF, FP_ONE / 100);

    /* Round-trip: float16 → Q16.48 → float32, check precision */
    /* For float16 with 10 mantissa bits, round-trip should be exact */
    uint16_t test_f16[] = {0x3C00, 0x4000, 0x3800, 0x4200, 0x4500, 0xC000};
    float expected_f32[] = {1.0f, 2.0f, 0.5f, 3.0f, 5.0f, -2.0f};
    for (int i = 0; i < 6; i++) {
        int64_t q = float16_to_q1648(test_f16[i]);
        float back = q1648_to_float32(q);
        char label[64];
        snprintf(label, sizeof(label), "f16 roundtrip %.1f", expected_f32[i]);
        check(label, fabsf(back - expected_f32[i]) < 0.001f);
    }

    printf("  float_conv: %d/%d passed\n", tests_passed, tests_run);
}

/* ================================================================== */
/*  Main                                                                */
/* ================================================================== */

int main(void) {
    printf("fp_math.h — Comprehensive Test Suite\n");
    printf("=====================================\n");
    printf("Q%d.%d fixed-point | FP_ONE = %lld | range ±%lld\n",
           64 - FP_PRECISION, FP_PRECISION,
           (long long)FP_ONE,
           (long long)(((int64_t)1 << (63 - FP_PRECISION)) - 1));

    test_arithmetic();
    test_sqrt();
    test_inv_sqrt();
    test_exp();
    test_log();
    test_pi();
    test_sincos();
    test_cordic_tables();
    test_rng();
    test_euler();
    test_consistency();
    test_edge_cases();
    test_stress();
    test_sigmoid_silu();
    test_float_conversion();
    bench_all();

    printf("\n=====================================\n");
    printf("TOTAL: %d tests, %d passed, %d failed\n",
           tests_run, tests_passed, tests_failed);

    if (tests_failed == 0)
        printf("ALL TESTS PASSED\n");
    else
        printf("*** %d FAILURES ***\n", tests_failed);

    return tests_failed;
}
