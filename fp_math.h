/*
 * Author: Nenad Mićić
 * LinkedIn: https://be.linkedin.com/in/nenadmicic
 *
 * Copyright (c) 2026 Nenad Mićić
 * SPDX-License-Identifier: Apache-2.0
 *
 * fp_math.h — Machine-Native Integer-Only Math Primitives
 * ========================================================
 *
 * A complete fixed-point math library providing every operation needed for
 * neural network training and inference, using ZERO floating-point operations.
 *
 * FORMAT
 *   Q16.48 fixed-point: int64_t with 48 fractional bits.
 *   - Range:      ±32767.999... (±2^15 - 1 ulp)
 *   - Resolution: 2^-48 ≈ 3.55e-15
 *   - FP_ONE = 2^48 = 281,474,976,710,656
 *
 * REQUIRES
 *   __int128 support (gcc/clang on 64-bit). All intermediates use 128-bit
 *   integer arithmetic for full-precision multiply and divide.
 *
 * INITIALIZATION
 *   Call fp_math_init() once before using CORDIC sin/cos. All other
 *   functions are stateless and need no init. Re-init is idempotent.
 *
 * API OVERVIEW
 *
 *   Arithmetic:
 *     fp_mul(a, b)            → a × b              (128-bit intermediate)
 *     fp_div(a, b)            → a ÷ b              (128-bit intermediate)
 *     fp_abs(a)               → |a|
 *     fp_from_int(n)          → n as Q16.48         (max ±32767)
 *     fp_max(a, b)            → max(a, b)
 *     fp_min(a, b)            → min(a, b)
 *
 *   Square Roots:
 *     fp_sqrt(x)              → √x                 (48-bit precision, Newton+CLZ)
 *     fp_sqrt_fast(x)         → √x approx          (~24-bit precision, 64-bit only)
 *     fp_inv_sqrt(x)          → 1/√x               (direct Newton, no sqrt+div)
 *     isqrt128(n)             → ⌊√n⌋               (128-bit bit-by-bit, exact)
 *
 *   Exponential & Logarithm:
 *     fp_exp(x)               → eˣ                 (dyadic refinement, k=14)
 *     fp_safe_exp(x)          → eˣ clamped          (softmax-safe, x ∈ [-50,30])
 *     fp_log(x)               → ln(x)              (Newton + CLZ initial guess)
 *     fp_safe_log(x)          → ln(x) clamped       (cross-entropy-safe)
 *     fp_exp_dyadic(x, k)     → eˣ with k rounds    (low-level, tunable precision)
 *
 *   Trigonometry (CORDIC — shifts and adds only):
 *     fp_sincos(θ, &c, &s)   → c=cos(θ), s=sin(θ)  (48 CORDIC iterations)
 *     fp_compute_pi()         → π                   (Machin's formula, integer Taylor)
 *     fp_atan_taylor(x, n)    → arctan(x)           (Taylor series, n terms)
 *     fp_atan_pow2(i)         → arctan(2⁻ⁱ)        (for CORDIC table init)
 *
 *   Constants (set by fp_math_init):
 *     FP_PI                   → 3.14159265358978...  (13+ correct digits)
 *     fp_cordic_gain          → 0.60725293500...     (CORDIC scale factor K⁻¹)
 *     fp_cordic_angles[48]    → arctan(2⁻ⁱ) table
 *
 *   PRNG (xorshift64):
 *     fp_rng_next()           → raw 64-bit random
 *     fp_rng_uniform()        → uniform in [0, 1)   (48-bit resolution)
 *     fp_gaussian(μ, σ)       → N(μ, σ²) sample     (CLT: sum 12 uniforms)
 *     fp_shuffle_ints(a, n)   → Fisher-Yates shuffle
 *
 *   Display:
 *     fp_print(x, decimals)   → prints decimal digits (integer extraction)
 *     fp_to_double(x)         → convert to double    (for display ONLY)
 *
 * ALGORITHMS
 *
 *   Multiplication:  (int128)a * b >> 48
 *   Division:        (int128)a << 48 / b
 *   Square root:     Newton-Raphson on isqrt(x << 48) with __builtin_clzll
 *                    initial guess. 5-7 iterations for 48-bit precision.
 *   Fast sqrt:       64-bit Newton only, r << 24 trick. ~24-bit precision.
 *   Inv sqrt:        Newton y*(3-x*y²)/2 with CLZ guess. No division needed.
 *   Exponential:     Dyadic limit: (1 + x/2^k)^(2^k) via k repeated squarings.
 *   Logarithm:       Newton y - 1 + x/exp(y) with CLZ-based log2 initial guess.
 *   Sin/Cos:         CORDIC vectoring mode, 48 iterations (shifts+adds).
 *   Pi:              Machin's formula: 16·atan(1/5) - 4·atan(1/239).
 *   Gaussian:        Central Limit Theorem: (Σ₁₂ uniform) - 6.
 *
 * PERFORMANCE (Apple M-series, -O3, 100K iterations)
 *
 *   fp_mul             ~1 ns     fp_exp           ~15 ns
 *   fp_div             ~7 ns     fp_log          ~325 ns
 *   fp_sqrt           ~60 ns     fp_sincos        ~49 ns
 *   fp_sqrt_fast      ~0.5 ns    fp_gaussian      ~18 ns
 *   fp_inv_sqrt       ~44 ns     fp_rng_next     ~1.5 ns
 *
 * THREAD SAFETY
 *   fp_math_init() is idempotent but not thread-safe. Call once at startup.
 *   All other functions are stateless EXCEPT fp_rng_* which share global state.
 *   For multi-threaded use, give each thread its own fp_rng_state.
 *
 * PROVENANCE
 *   Consolidated from: euler_identity.c, transcendentals_bitwise.c,
 *                      sqrt2_bitwise.c, e_bitwise.c, pi_bitwise.c
 */

#ifndef FP_MATH_H
#define FP_MATH_H

#include <stdint.h>

#ifdef FP_DEBUG
#include <stdio.h>
#include <stdlib.h>   /* abort() for the overflow guard */
#endif

/* ================================================================== */
/*  Configuration                                                      */
/* ================================================================== */

#ifndef __SIZEOF_INT128__
#error "fp_math.h requires __int128 support (gcc/clang on 64-bit)"
#endif

typedef __int128          int128_t;
typedef unsigned __int128 uint128_t;

#define FP_PRECISION 48

typedef int64_t fixed_t;

#define FP_ONE   ((fixed_t)1 << FP_PRECISION)
#define FP_HALF  ((fixed_t)1 << (FP_PRECISION - 1))
#undef  FP_ZERO  /* avoid collision with math.h FP_ZERO on glibc */
#define FP_ZERO  ((fixed_t)0)

/* ================================================================== */
/*  Basic Arithmetic                                                   */
/* ================================================================== */

static inline fixed_t fp_mul(fixed_t a, fixed_t b) {
    int128_t prod = (int128_t)a * b;
#ifdef FP_DEBUG
    /* The Q16.48 result is (prod >> 48); its integer part is (prod >> 96).
     * Overflow iff that integer part leaves the representable ±32767 range
     * (i.e. the result would not fit in int64). Trip loudly in dev builds. */
    int128_t ipart = prod >> (2 * FP_PRECISION);
    if (ipart > 32767 || ipart < -32768) {
        fprintf(stderr,
                "fp_mul overflow: a=%lld b=%lld -> integer part %lld out of [-32768,32767]\n",
                (long long)a, (long long)b, (long long)ipart);
        abort();
    }
#endif
    return (fixed_t)(prod >> FP_PRECISION);
}

static inline fixed_t fp_div(fixed_t a, fixed_t b) {
    return (fixed_t)(((int128_t)a << FP_PRECISION) / b);
}

static inline fixed_t fp_abs(fixed_t a) {
    return a >= 0 ? a : -a;
}

static inline fixed_t fp_from_int(int x) {
    return (fixed_t)x << FP_PRECISION;
}

static inline fixed_t fp_max(fixed_t a, fixed_t b) {
    return a > b ? a : b;
}

static inline fixed_t fp_min(fixed_t a, fixed_t b) {
    return a < b ? a : b;
}

/* ================================================================== */
/*  Display (integer-only digit extraction)                            */
/* ================================================================== */

#include <stdio.h>

static void fp_print(fixed_t x, int decimals) {
    if (x < 0) { printf("-"); x = -x; }
    int64_t int_part = x >> FP_PRECISION;
    printf("%lld.", (long long)int_part);
    int64_t frac_mask = ((int64_t)1 << FP_PRECISION) - 1;
    int64_t frac_part = x & frac_mask;
    for (int d = 0; d < decimals; d++) {
        frac_part *= 10;
        printf("%lld", (long long)(frac_part >> FP_PRECISION));
        frac_part &= frac_mask;
    }
}

static inline double fp_to_double(fixed_t x) {
    return (double)x / (double)FP_ONE;
}

/* ================================================================== */
/*  Integer Square Root (bit-by-bit, from sqrt2_bitwise.c)             */
/* ================================================================== */

static uint128_t isqrt128(uint128_t n) {
    if (n == 0) return 0;
    uint128_t x = 0;
    uint128_t bit = (uint128_t)1 << 126;
    while (bit > n) bit >>= 2;
    while (bit != 0) {
        if (n >= x + bit) {
            n -= x + bit;
            x = (x >> 1) + bit;
        } else {
            x >>= 1;
        }
        bit >>= 2;
    }
    return x;
}

/* Fast approximate sqrt — 64-bit only, no 128-bit division.
 * ~32 bits of precision. Perfect for Adam denominator. */
static fixed_t fp_sqrt_fast(fixed_t x) {
    if (x <= 0) return 0;
    uint64_t ux = (uint64_t)x;

    /* isqrt64 via Newton with CLZ initial guess */
    int bw = 64 - __builtin_clzll(ux);
    uint64_t r = (uint64_t)1 << ((bw + 1) / 2);
    for (int i = 0; i < 5; i++) {
        uint64_t nr = (r + ux / r) >> 1;
        if (nr >= r) break;
        r = nr;
    }
    /* r ≈ sqrt(x). Result in Q16.48 = sqrt(x) * 2^24 = r << 24.
     * This gives ~24 bits of fractional precision — plenty for Adam. */
    return (fixed_t)(r << 24);
}

/* Precise fixed-point square root via Newton-Raphson with CLZ initial guess.
 * Uses 128-bit arithmetic for full 48-bit precision. */
static fixed_t fp_sqrt(fixed_t x) {
    if (x <= 0) return 0;

    /* We want r = isqrt(x << 48). Use Newton: r = (r + n/r) / 2 */
    uint128_t n = (uint128_t)(uint64_t)x << FP_PRECISION;

    /* Initial guess via CLZ: find bit-width of n, then r0 ~ 2^(bw/2) */
    int bw;
    uint64_t hi = (uint64_t)(n >> 64);
    if (hi != 0)
        bw = 128 - __builtin_clzll(hi);
    else
        bw = 64 - __builtin_clzll((uint64_t)n);

    uint128_t r = (uint128_t)1 << ((bw + 1) / 2);

    /* Newton iterations: converges quadratically, 6 iterations gives
     * >48 bits of precision from any 1-bit initial guess */
    for (int i = 0; i < 7; i++) {
        if (r == 0) return 0;
        uint128_t nr = (r + n / r) >> 1;
        if (nr >= r) break; /* converged */
        r = nr;
    }

    /* Final correction: ensure r^2 <= n */
    while (r * r > n) r--;

    return (fixed_t)r;
}

/* ================================================================== */
/*  Inverse Square Root: 1/sqrt(x) via Newton y*(3 - x*y^2)/2         */
/* ================================================================== */

static fixed_t fp_inv_sqrt(fixed_t x) {
    if (x <= 0) return 0;

    /* Direct Newton for 1/sqrt(x): y = y * (3 - x*y^2) / 2.
     *
     * Range-reduce first: x = xn * 2^(2k) with xn in [2^48, 2^50) (real [1,4)),
     * then 1/sqrt(x) = (1/sqrt(xn)) * 2^(-k). Normalizing keeps every Newton
     * intermediate < ~2^99, so there is NO signed __int128 overflow on any
     * positive input. (The previous version relied on int128 not overflowing,
     * which it DID for tiny x — 1/sqrt(1 ULP) needs y ~ 2^72, so x*y^2 ~ 2^139.
     * That overflow was undefined behavior and diverged arm64/clang vs x86/gcc;
     * caught by `make determinism`.)
     *
     * When 1/sqrt(x) exceeds the Q16.48 range (x below ~2^-30 real), the true
     * value is unrepresentable: saturate to INT64_MAX deterministically. */
    int msb = 63 - __builtin_clzll((uint64_t)x);
    int k = (msb - 48) >> 1;                  /* floor((msb-48)/2); may be < 0 */
    int128_t xn = (k >= 0) ? ((int128_t)x >> (2 * k))
                           : ((int128_t)x << (-2 * k));

    /* Initial guess by sub-octave so xn*y0^2 < 3 (Newton basin) for all xn:
     *   xn in [1,2) (msb 48) -> y0 = 1.0 ;  xn in [2,4) (msb 49) -> y0 = 0.5 */
    int msb_n = 63 - __builtin_clzll((uint64_t)xn);
    int128_t y = (msb_n == 48) ? ((int128_t)1 << FP_PRECISION)
                               : ((int128_t)1 << (FP_PRECISION - 1));
    int128_t three = (int128_t)fp_from_int(3);
    for (int i = 0; i < 8; i++) {
        int128_t y2 = (y * y) >> FP_PRECISION;        /* <= ~2^48  */
        int128_t xy2 = (xn * y2) >> FP_PRECISION;     /* <= ~2^50  */
        int128_t factor = three - xy2;
        y = (y * factor >> FP_PRECISION) >> 1;        /* |y*factor| <= ~2^99 */
        if (y <= 0) { y = 1; break; }
    }

    /* Scale back by 2^(-k); saturate when out of Q16.48 range. */
    int128_t r = (k >= 0) ? (y >> k) : (y << (-k));
    if (r > (int128_t)INT64_MAX) r = INT64_MAX;
    if (r < 0) r = 0;
    return (fixed_t)r;
}

/* ================================================================== */
/*  Exponential: exp(x) as refinement invariant                        */
/* ================================================================== */

/* exp(x) = lim_{k->inf} (1 + x/2^k)^(2^k) via repeated squaring */
static fixed_t fp_exp_dyadic(fixed_t x, int k) {
    fixed_t base = FP_ONE + (x >> k);
    fixed_t result = base;
    for (int i = 0; i < k; i++)
        result = fp_mul(result, result);
    return result;
}

/* Full exp with negative handling.
 * k=14 gives ~42-bit accuracy — sufficient for softmax/loss. */
static fixed_t fp_exp(fixed_t x) {
    if (x >= 0) {
        return fp_exp_dyadic(x, 14);
    } else {
        fixed_t pos = fp_exp_dyadic(-x, 14);
        if (pos == 0) return 0;
        return fp_div(FP_ONE, pos);
    }
}

/* Safe exp for softmax/sigmoid: clamp to prevent overflow/underflow.
 * Q16.48 has 15 integer bits → max representable ~32767.
 * exp(10) ≈ 22026 fits.  exp(10.4) ≈ 32860 overflows int64.
 * Clamp positive to 10.  For x < -10, exp(x) < 4.5e-5 — negligible
 * for softmax; sigmoid handles this via its own saturation path. */
static fixed_t fp_safe_exp(fixed_t x) {
    fixed_t max_x = fp_from_int(10);
    if (x > max_x) x = max_x;
    if (x < -max_x) return 0;
    return fp_exp(x);
}

/* ================================================================== */
/*  Logarithm: log(x) via Newton's method                             */
/* ================================================================== */

/* Newton: y_{n+1} = y_n - 1 + x/exp(y_n)
 * Uses CLZ for initial guess: log(x) ≈ (bits - 48) * ln(2) */
static fixed_t fp_log(fixed_t x) {
    if (x <= 0) return -fp_from_int(50); /* sentinel for log(0) */

    /* CLZ-based initial guess: x in Q16.48, value = x/2^48.
     * log(x/2^48) = log2(x/2^48) * ln(2) = (log2(x) - 48) * ln(2).
     * log2(x) ≈ 63 - clz(x). So log(value) ≈ (63 - clz(x) - 48) * ln(2)
     * ln(2) ≈ 0.6931471805599453. In Q16.48 (round-to-nearest): ln2 = round(ln(2) * 2^48). */
    fixed_t ln2 = (fixed_t)195103586505167LL; /* ln(2) * 2^48, round-to-nearest = 0.69314718055994… */
    int lz = (x > 0) ? __builtin_clzll((uint64_t)x) : 63;
    int log2_approx = 63 - lz - FP_PRECISION; /* can be negative */
    fixed_t y = log2_approx * ln2;

    for (int i = 0; i < 10; i++) {
        fixed_t ey = fp_exp(y);
        if (ey == 0) break;
        fixed_t y_new = y - FP_ONE + fp_div(x, ey);
        fixed_t diff = fp_abs(y_new - y);
        if (diff < 2) break; /* converged to ~1 ulp */
        y = y_new;
    }
    return y;
}

/* Safe log for cross-entropy: clamp small inputs */
static fixed_t fp_safe_log(fixed_t x) {
    if (x <= 0) return -fp_from_int(50);
    /* Minimum representable positive: 1 ulp = 2^-48 ~ 3.5e-15
     * log(3.5e-15) ~ -33. We clamp to a sane minimum. */
    if (x < 4) x = 4; /* prevent extreme log values */
    return fp_log(x);
}

/* ================================================================== */
/*  Pi via Machin's Formula (pure integer, no fp_from_double!)         */
/* ================================================================== */

/* arctan(x) via Taylor: x - x^3/3 + x^5/5 - x^7/7 + ... */
static fixed_t fp_atan_taylor(fixed_t x, int max_terms) {
    fixed_t x_sq = fp_mul(x, x);
    fixed_t term = x;
    fixed_t sum = x;
    for (int n = 1; n <= max_terms; n++) {
        term = -fp_mul(term, x_sq);
        fixed_t contribution = term / (2 * n + 1);
        if (contribution == 0) break;
        sum += contribution;
    }
    return sum;
}

/* Machin: pi = 16*arctan(1/5) - 4*arctan(1/239)
 * Both converge fast. Pure integer. */
static fixed_t fp_compute_pi(void) {
    fixed_t x5 = FP_ONE / 5;
    fixed_t atan5 = fp_atan_taylor(x5, 40);

    fixed_t x239 = FP_ONE / 239;
    fixed_t atan239 = fp_atan_taylor(x239, 15);

    return (atan5 << 4) - (atan239 << 2);
}

/* ================================================================== */
/*  CORDIC Sin/Cos (shifts and adds only)                              */
/* ================================================================== */

#define FP_CORDIC_N 48

static fixed_t fp_cordic_angles[FP_CORDIC_N];
static fixed_t fp_cordic_gain; /* K^{-1} ~ 0.6072529... */
static fixed_t FP_PI;
static int fp_math_initialized = 0;

/* Compute arctan(2^{-i}) for i >= 1 via Taylor series (integer only) */
static fixed_t fp_atan_pow2(int i) {
    if (i >= 25) return FP_ONE >> i; /* arctan(x) ~ x for tiny x */
    fixed_t x = FP_ONE >> i;
    return fp_atan_taylor(x, 40);
}

/* Initialize CORDIC tables and pi — call once before use */
static void fp_math_init(void) {
    if (fp_math_initialized) return;

    /* Pi from Machin's formula */
    FP_PI = fp_compute_pi();

    /* CORDIC angle table */
    fp_cordic_angles[0] = FP_PI >> 2; /* arctan(1) = pi/4 */
    for (int i = 1; i < FP_CORDIC_N; i++)
        fp_cordic_angles[i] = fp_atan_pow2(i);

    /* CORDIC gain: K^{-1} = 1/prod(sqrt(1 + 2^{-2i}))
     * Compute K^2 = prod(1 + 2^{-2i}), then K^{-1} = 1/sqrt(K^2) */
    fixed_t k_sq = FP_ONE;
    for (int i = 0; i < FP_CORDIC_N; i++) {
        /* Guard: shifting int64 by >= 63 is UB. For 2*i >= 63,
         * 2^{-2i} < 2^{-62} which is below Q16.48 precision anyway. */
        fixed_t shift_val = (2 * i < 63) ? (FP_ONE >> (2 * i)) : 0;
        fixed_t factor = FP_ONE + shift_val;
        k_sq = fp_mul(k_sq, factor);
    }
    fixed_t k_val = fp_sqrt(k_sq);
    fp_cordic_gain = fp_div(FP_ONE, k_val);

    fp_math_initialized = 1;
}

/* CORDIC rotation: compute (cos(theta), sin(theta)) using shifts+adds */
static void fp_sincos(fixed_t theta, fixed_t *cos_out, fixed_t *sin_out) {
    fp_math_init();

    /* Reduce to [-pi, pi] */
    fixed_t two_pi = FP_PI << 1;
    while (theta > FP_PI) theta -= two_pi;
    while (theta < -FP_PI) theta += two_pi;

    /* Handle quadrants: CORDIC converges for |theta| < pi/2 */
    fixed_t pi_half = FP_PI >> 1;
    int negate_cos = 0;
    if (theta > pi_half) {
        theta = FP_PI - theta;
        negate_cos = 1;
    } else if (theta < -pi_half) {
        theta = -FP_PI - theta;
        negate_cos = 1;
    }

    fixed_t x = fp_cordic_gain;
    fixed_t y = 0;
    fixed_t z = theta;

    for (int i = 0; i < FP_CORDIC_N; i++) {
        fixed_t xs = x >> i;
        fixed_t ys = y >> i;
        if (z >= 0) {
            fixed_t xn = x - ys;
            fixed_t yn = y + xs;
            z -= fp_cordic_angles[i];
            x = xn; y = yn;
        } else {
            fixed_t xn = x + ys;
            fixed_t yn = y - xs;
            z += fp_cordic_angles[i];
            x = xn; y = yn;
        }
    }

    *cos_out = negate_cos ? -x : x;
    *sin_out = y;
}

/* ================================================================== */
/*  Sigmoid & SiLU (for SwiGLU activation in Llama-style models)       */
/* ================================================================== */

/* sigmoid(x) = 1 / (1 + exp(-x)) — reuses fp_safe_exp, fp_div.
 * Saturates early: sigmoid(x) < 4.5e-5 for x < -10 (≈ 0),
 * sigmoid(x) > 0.99995 for x > 10 (≈ 1). */
static fixed_t fp_sigmoid(fixed_t x) {
    if (x > fp_from_int(10)) return FP_ONE;
    if (x < -fp_from_int(10)) return 0;
    fixed_t exp_neg = fp_safe_exp(-x);
    return fp_div(FP_ONE, FP_ONE + exp_neg);
}

/* SiLU(x) = x * sigmoid(x) — the activation used in Llama's SwiGLU */
static fixed_t fp_silu(fixed_t x) {
    return fp_mul(x, fp_sigmoid(x));
}

/* ================================================================== */
/*  PRNG (xorshift64 — already integer)                                */
/* ================================================================== */

static unsigned long long fp_rng_state = 42;

static unsigned long long fp_rng_next(void) {
    fp_rng_state ^= fp_rng_state << 13;
    fp_rng_state ^= fp_rng_state >> 7;
    fp_rng_state ^= fp_rng_state << 17;
    return fp_rng_state;
}

/* Uniform random in [0, FP_ONE) — pure integer */
static fixed_t fp_rng_uniform(void) {
    /* Take top 48 bits of 64-bit random, scale to [0, FP_ONE) */
    return (fixed_t)(fp_rng_next() >> (64 - FP_PRECISION));
}

/* Gaussian via CLT: sum 12 uniforms - 6 ≈ N(0,1)
 * Simple, fast, integer-only, no transcendentals needed */
static fixed_t fp_gaussian(fixed_t mean, fixed_t std) {
    fixed_t sum = 0;
    for (int i = 0; i < 12; i++)
        sum += fp_rng_uniform();
    /* sum ~ N(6*FP_ONE, FP_ONE^2)
     * (sum - 6*FP_ONE) ~ N(0, FP_ONE) in fixed-point = N(0,1) */
    fixed_t z = sum - 6 * FP_ONE;
    return mean + fp_mul(std, z);
}

/* Shuffle array of ints using integer RNG */
static void fp_shuffle_ints(int *arr, int n) {
    for (int i = n - 1; i > 0; i--) {
        /* j in [0, i]: use modulo (biased but acceptable for shuffle) */
        int j = (int)(fp_rng_next() % (unsigned long long)(i + 1));
        int tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
    }
}

#endif /* FP_MATH_H */
