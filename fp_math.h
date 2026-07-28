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
 *   C99 and a 64-bit integer type — nothing else. All intermediates use
 *   128-bit integer arithmetic for full-precision multiply and divide;
 *   where the compiler provides __int128 (gcc/clang on 64-bit hosts)
 *   that is used directly, and everywhere else (Cortex-M, RV32, any
 *   32-bit target) a built-in two-limb software backend computes the
 *   SAME bits. The backend switch is invisible to callers: the fp_w128
 *   wide type and every fp_* function keep one name and one contract,
 *   and the astro-nav golden battery hashes bit-identically on both.
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
 *     isqrt128(n)             → ⌊√n⌋               (128-bit bit-by-bit, exact;
 *                                                  native-__int128 builds only)
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
 *   Activations:
 *     fp_sigmoid(x)           → 1/(1+e⁻ˣ)           (saturates at |x| > 10)
 *     fp_silu(x)              → x·sigmoid(x)         (SwiGLU activation)
 *
 *   PRNG (xorshift64):
 *     fp_rng_next()           → raw 64-bit random
 *     fp_rng_uniform()        → uniform in [0, 1)   (48-bit resolution)
 *     fp_gaussian(μ, σ)       → N(μ, σ²) sample     (CLT: sum 12 uniforms)
 *     fp_shuffle_ints(a, n)   → Fisher-Yates shuffle
 *
 *   Display:
 *     fp_print(x, decimals)   → prints decimal digits (needs FP_MATH_WITH_STDIO)
 *     fp_to_double(x)         → convert to double    (for display ONLY)
 *
 * CONFIGURATION MACROS (define before including)
 *   FP_DEBUG                — fp_mul overflow and fp_div zero-divisor guards
 *                             (fprintf + abort; pulls in stdio/stdlib)
 *   FP_MATH_WITH_STDIO      — enable fp_print (pulls in stdio)
 *   FP_MATH_INT128_ALIASES  — legacy int128_t/uint128_t typedef spellings
 *                             (native-__int128 builds only)
 *   FP_MATH_FORCE_PORTABLE  — use the two-limb software backend even
 *                             where __int128 exists (differential
 *                             testing: the host golden hash must not
 *                             move when this is defined)
 *
 * ALGORITHMS
 *
 *   Multiplication:  (int128)a * b >> 48
 *   Division:        sign(a) * ((uint128)|a| << 48 / b)   (shift on magnitude:
 *                    left-shifting a negative value is UB in C99/C11)
 *   Square root:     Newton-Raphson on isqrt(x << 48) with __builtin_clzll
 *                    initial guess. 5-7 iterations for 48-bit precision.
 *   Fast sqrt:       64-bit Newton only, r << 24 trick. ~24-bit precision.
 *   Inv sqrt:        Newton y*(3-x*y²)/2 with CLZ guess. No division needed.
 *   Exponential:     Dyadic limit: (1 + x/2^k)^(2^k) via k repeated squarings.
 *   Logarithm:       Newton y - 1 + x/exp(y) with CLZ-based log2 initial guess.
 *   Sin/Cos:         CORDIC rotation mode, 48 iterations (shifts+adds).
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
 * LINKAGE
 *   Header-only: every function is static inline, and the CORDIC tables,
 *   FP_PI, and fp_rng_state are static — each translation unit that includes
 *   this header gets its OWN copies. Call fp_math_init() (or any function
 *   that does, e.g. fp_sincos) per TU, and don't expect RNG streams or
 *   FP_PI pointer identity to be shared across TUs.
 *
 * PROVENANCE
 *   Consolidated from: euler_identity.c, transcendentals_bitwise.c,
 *                      sqrt2_bitwise.c, e_bitwise.c, pi_bitwise.c
 *   Lineage of this int-llm-32bit copy:
 *     int-llm/fp_math.h @ 8b16a00bb9126d2a99db9ad7683c3c064b9eb7fc
 *       (native-__int128-only original; requires a 64-bit host)
 *     -> astro-nav-int/fp_math.h @ 9db73509c7c06d4a7bcdcd706c77a5fde3e728c1
 *       (adds the portable two-limb wide backend, its modulo-2^128
 *        contract, and differential backend tests; hardware-validated
 *        bit-exact on Cortex-M0+/M33, RISC-V rv32, Xtensa LX6/LX7,
 *        ARMv6, and AVR — see astro-nav-int reviews/HW_tests/)
 *     -> vendored back here so microgpt_int runs on 32-bit targets.
 *   `make determinism` in this repo pins the same golden hash as the
 *   native-only original, on both backends.
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

/* Backend selection: native __int128 where the compiler has it, the
 * two-limb software backend everywhere else (or on demand, for
 * differential testing). Everything downstream keys off this macro. */
#if defined(__SIZEOF_INT128__) && !defined(FP_MATH_FORCE_PORTABLE)
#define FP_MATH_HAS_INT128 1
#else
#define FP_MATH_HAS_INT128 0
#endif

#if FP_MATH_HAS_INT128
/* Named fp_i128/fp_u128 rather than int128_t/uint128_t: the *_t names in
 * <stdint.h>'s namespace are reserved for the implementation, so a future
 * libc gaining int128_t would collide. Compile with FP_MATH_INT128_ALIASES
 * for the legacy spellings. These native typedefs (and isqrt128, whose
 * signature uses them) exist only on the native backend; portable code
 * uses the fp_w128 operations below, which exist on both. */
typedef __int128          fp_i128;
typedef unsigned __int128 fp_u128;

#ifdef FP_MATH_INT128_ALIASES
typedef fp_i128 int128_t;
typedef fp_u128 uint128_t;
#endif
#endif /* FP_MATH_HAS_INT128 */

#define FP_PRECISION 48

typedef int64_t fixed_t;

#define FP_ONE   ((fixed_t)1 << FP_PRECISION)
#define FP_HALF  ((fixed_t)1 << (FP_PRECISION - 1))
#undef  FP_ZERO  /* avoid collision with math.h FP_ZERO on glibc */
#define FP_ZERO  ((fixed_t)0)

/* ================================================================== */
/*  128-bit working type: fp_w128                                      */
/* ================================================================== */

/* Every 128-bit intermediate in this header (and in code built on it)
 * goes through the small operation set below, so one source line
 * serves both backends:
 *
 *   native   — fp_w128 IS __int128; each op is a one-line wrapper the
 *              compiler flattens to the exact instruction sequence the
 *              direct-operator code produced before this layer existed
 *              (the astro-nav golden hash pins that equivalence);
 *   portable — fp_w128 is a two-limb {lo, hi} struct in two's
 *              complement; ops use only C99 64-bit arithmetic (the
 *              64x64->128 multiply is four 32x32->64 partials, the
 *              128-bit divide is a CLZ-shortened restoring loop).
 *
 * The set is deliberately minimal: from/to 64-bit, widening multiply,
 * low-128 multiply, add/sub/neg, shifts, truncating divide, compare.
 *
 * CONTRACT (identical on both backends, and what the differential
 * fuzz gate --fuzz-w128 exercises):
 *   - add/sub/neg/mul/muls are two's-complement modulo 2^128: they
 *     wrap, never trap, on every input. The native wrappers route
 *     through fp_u128 to get that (signed __int128 overflow would be
 *     UB); the value-changing unsigned->signed cast is
 *     implementation-defined in C99, and gcc/clang — the only
 *     compilers that provide __int128 — both define it as
 *     two's-complement truncation, exactly the portable struct's
 *     semantics.
 *   - shift counts must be in [0, 127]; both backends are fully
 *     defined over that whole domain (fp_w_shl discards high bits,
 *     fp_w_asr fills with the sign).
 *   - divisors must be nonzero. -2^127 / -1 wraps to
 *     -2^127 like every other op (the native operator alone
 *     would make it UB, so it is peeled off explicitly).
 *   - fp_w_divs / fp_w_divs_pow2 truncate toward zero (an arithmetic
 *     right shift rounds toward minus infinity instead — the two
 *     differ on negative values, so both exist and callers pick the
 *     one they mean). */

#if FP_MATH_HAS_INT128

typedef fp_i128 fp_w128;

static inline fp_w128 fp_w_from_i64(int64_t v)         { return v; }
static inline fp_w128 fp_w_from_u64(uint64_t v)        { return (fp_i128)v; }
static inline int64_t fp_w_to_i64(fp_w128 a)           { return (int64_t)a; }
/* muls never wraps: |a*b| <= 2^126 fits any __int128, so the plain
 * signed multiply is defined for every input pair. */
static inline fp_w128 fp_w_muls(int64_t a, int64_t b)  { return (fp_i128)a * b; }
/* Everything below that CAN wrap goes through fp_u128 (see the
 * contract above): unsigned arithmetic wraps by definition where the
 * signed operators would be UB at INT128 extremes. */
static inline fp_w128 fp_w_mul(fp_w128 a, fp_w128 b)   { return (fp_i128)((fp_u128)a * (fp_u128)b); }
static inline fp_w128 fp_w_add(fp_w128 a, fp_w128 b)   { return (fp_i128)((fp_u128)a + (fp_u128)b); }
static inline fp_w128 fp_w_sub(fp_w128 a, fp_w128 b)   { return (fp_i128)((fp_u128)a - (fp_u128)b); }
static inline fp_w128 fp_w_neg(fp_w128 a)              { return (fp_i128)(0 - (fp_u128)a); }
/* Shift as unsigned: << on a negative signed value is UB (C99 6.5.7p4).
 * Counts are contract-limited to [0, 127]. asr keeps the native signed
 * shift: >> on a negative value is implementation-defined, and
 * gcc/clang define it as the arithmetic shift this op means. */
static inline fp_w128 fp_w_shl(fp_w128 a, int k)       { return (fp_i128)((fp_u128)a << k); }
static inline fp_w128 fp_w_asr(fp_w128 a, int k)       { return a >> k; }

static inline fp_w128 fp_w_divs(fp_w128 a, fp_w128 b)
{
    /* Peel b == -1 off: -2^127 / -1 is UB for the native operator
     * but must wrap to -2^127 (the portable backend's result, and
     * two's-complement negation's). For every other a, -a == a / -1. */
    if (b == (fp_i128)-1) return fp_w_neg(a);
    return a / b;
}

static inline fp_w128 fp_w_divs_pow2(fp_w128 a, int k)
{
    /* Truncation toward zero: shift the magnitude, restore the sign —
     * unsigned throughout, so k up to 127 and a == -2^127 stay
     * defined (the old form divided by (fp_i128)1 << k, which
     * overflows at k == 127). */
    fp_u128 mag = (a < 0) ? (0 - (fp_u128)a) : (fp_u128)a;
    fp_u128 q = mag >> k;
    return (a < 0) ? (fp_i128)(0 - q) : (fp_i128)q;
}

static inline int fp_w_cmp(fp_w128 a, fp_w128 b)
{
    return (a < b) ? -1 : (a > b) ? 1 : 0;
}

/* Bit length of a non-negative value (0 -> 0, 1 -> 1, 2..3 -> 2, ...). */
static inline int fp_w_bits(fp_w128 a)
{
    uint64_t hi = (uint64_t)((fp_u128)a >> 64);
    if (hi != 0) return 128 - __builtin_clzll(hi);
    uint64_t lo = (uint64_t)(fp_u128)a;
    return lo ? 64 - __builtin_clzll(lo) : 0;
}

#else /* portable two-limb backend */

typedef struct { uint64_t lo, hi; } fp_w128;   /* two's complement */

static inline fp_w128 fp_w_from_i64(int64_t v)
{
    fp_w128 r;
    r.lo = (uint64_t)v;
    r.hi = (v < 0) ? ~(uint64_t)0 : 0;   /* sign extension */
    return r;
}

static inline fp_w128 fp_w_from_u64(uint64_t v)
{
    fp_w128 r;
    r.lo = v;
    r.hi = 0;
    return r;
}

static inline int64_t fp_w_to_i64(fp_w128 a)
{
    return (int64_t)a.lo;   /* truncating cast, exactly like (int64_t)i128 */
}

static inline fp_w128 fp_w_add(fp_w128 a, fp_w128 b)
{
    fp_w128 r;
    r.lo = a.lo + b.lo;
    r.hi = a.hi + b.hi + (r.lo < a.lo);
    return r;
}

static inline fp_w128 fp_w_neg(fp_w128 a)
{
    fp_w128 r;
    r.lo = ~a.lo + 1;
    r.hi = ~a.hi + (a.lo == 0);
    return r;
}

static inline fp_w128 fp_w_sub(fp_w128 a, fp_w128 b)
{
    return fp_w_add(a, fp_w_neg(b));
}

/* Unsigned 64x64 -> 128 via four 32x32 -> 64 partial products: the one
 * real workhorse (RV32IM/Cortex-M3+ do the partials in hardware;
 * RV32I and Cortex-M0+ fall back to libgcc's integer multiply). */
static inline fp_w128 fp_w_umul64_(uint64_t a, uint64_t b)
{
    uint64_t a0 = a & 0xffffffffu, a1 = a >> 32;
    uint64_t b0 = b & 0xffffffffu, b1 = b >> 32;
    uint64_t p00 = a0 * b0;
    uint64_t p01 = a0 * b1;
    uint64_t p10 = a1 * b0;
    uint64_t p11 = a1 * b1;
    uint64_t mid = p01 + (p00 >> 32);       /* cannot carry: see below */
    uint64_t mid2 = mid + p10;              /* may carry into bit 64   */
    fp_w128 r;
    /* mid <= (2^32-1)^2/2^32 + (2^32-1) < 2^33 + 2^32 fits; the mid2
     * carry is the only one that can leave 64 bits. */
    r.lo = (mid2 << 32) | (p00 & 0xffffffffu);
    r.hi = p11 + (mid2 >> 32) + ((mid2 < mid) ? ((uint64_t)1 << 32) : 0);
    return r;
}

static inline fp_w128 fp_w_muls(int64_t a, int64_t b)
{
    uint64_t ua = (a < 0) ? -(uint64_t)a : (uint64_t)a;
    uint64_t ub = (b < 0) ? -(uint64_t)b : (uint64_t)b;
    fp_w128 p = fp_w_umul64_(ua, ub);
    return ((a < 0) != (b < 0)) ? fp_w_neg(p) : p;
}

/* Low 128 bits of the product — in two's complement the signed and
 * unsigned low halves coincide, so no sign handling is needed. */
static inline fp_w128 fp_w_mul(fp_w128 a, fp_w128 b)
{
    fp_w128 r = fp_w_umul64_(a.lo, b.lo);
    r.hi += a.lo * b.hi + a.hi * b.lo;
    return r;
}

/* Shift counts: the cross-backend contract is [0, 127] (see above).
 * This implementation happens to also return 0 for k >= 128, but the
 * native backend cannot, so no caller may rely on it. */
static inline fp_w128 fp_w_shl(fp_w128 a, int k)
{
    fp_w128 r;
    if (k == 0) return a;
    if (k < 64) {
        r.hi = (a.hi << k) | (a.lo >> (64 - k));
        r.lo = a.lo << k;
    } else {
        r.hi = (k < 128) ? (a.lo << (k - 64)) : 0;
        r.lo = 0;
    }
    return r;
}

/* Logical (zero-fill) right shift — internal helper. */
static inline fp_w128 fp_w_ushr_(fp_w128 a, int k)
{
    fp_w128 r;
    if (k == 0) return a;
    if (k < 64) {
        r.lo = (a.lo >> k) | (a.hi << (64 - k));
        r.hi = a.hi >> k;
    } else {
        r.lo = (k < 128) ? (a.hi >> (k - 64)) : 0;
        r.hi = 0;
    }
    return r;
}

static inline fp_w128 fp_w_asr(fp_w128 a, int k)
{
    uint64_t s = (a.hi >> 63) ? ~(uint64_t)0 : 0;
    fp_w128 r = fp_w_ushr_(a, k);
    if (k == 0) return r;
    if (k < 64) {
        r.hi |= s << (64 - k);
    } else {
        r.hi = s;
        if (k > 64 && k < 128) r.lo |= s << (128 - k);
        else if (k >= 128) r.lo = s;
    }
    return r;
}

static inline int fp_w_ucmp_(fp_w128 a, fp_w128 b)
{
    if (a.hi != b.hi) return (a.hi < b.hi) ? -1 : 1;
    if (a.lo != b.lo) return (a.lo < b.lo) ? -1 : 1;
    return 0;
}

static inline int fp_w_cmp(fp_w128 a, fp_w128 b)
{
    /* Flip the sign bit and the unsigned order becomes signed order. */
    fp_w128 au = a, bu = b;
    au.hi ^= (uint64_t)1 << 63;
    bu.hi ^= (uint64_t)1 << 63;
    return fp_w_ucmp_(au, bu);
}

static inline int fp_w_bits(fp_w128 a)
{
    if (a.hi != 0) return 128 - __builtin_clzll(a.hi);
    return a.lo ? 64 - __builtin_clzll(a.lo) : 0;
}

/* Unsigned restoring division, CLZ-shortened: walks only the dividend's
 * significant bits. Remainder stays below the divisor, so both fit the
 * same two limbs. Division by zero is undefined here exactly as it is
 * for the native operator. */
static inline fp_w128 fp_w_udivmod_(fp_w128 n, fp_w128 d, fp_w128 *rem_out)
{
    fp_w128 q = { 0, 0 }, r = { 0, 0 };
    int i = fp_w_bits(n);
    while (i-- > 0) {
        r = fp_w_shl(r, 1);
        r.lo |= (i < 64) ? ((n.lo >> i) & 1) : ((n.hi >> (i - 64)) & 1);
        if (fp_w_ucmp_(r, d) >= 0) {
            r = fp_w_sub(r, d);
            if (i < 64) q.lo |= (uint64_t)1 << i;
            else        q.hi |= (uint64_t)1 << (i - 64);
        }
    }
    if (rem_out) *rem_out = r;
    return q;
}

static inline fp_w128 fp_w_divs(fp_w128 a, fp_w128 b)
{
    /* -2^127 / -1: fp_w_neg of -2^127 is itself, the magnitudes divide as
     * 2^127 / 1, and the sign step negates back to -2^127 — the modulo
     * wrap the contract promises falls out with no special case. */
    int sa = (a.hi >> 63) != 0, sb = (b.hi >> 63) != 0;
    fp_w128 q = fp_w_udivmod_(sa ? fp_w_neg(a) : a,
                              sb ? fp_w_neg(b) : b, 0);
    return (sa != sb) ? fp_w_neg(q) : q;
}

static inline fp_w128 fp_w_divs_pow2(fp_w128 a, int k)
{
    /* Truncation toward zero: shift the magnitude, restore the sign. */
    if (a.hi >> 63)
        return fp_w_neg(fp_w_ushr_(fp_w_neg(a), k));
    return fp_w_ushr_(a, k);
}

#endif /* backend */

/* ================================================================== */
/*  Basic Arithmetic                                                   */
/* ================================================================== */

static inline fixed_t fp_mul(fixed_t a, fixed_t b) {
    fp_w128 prod = fp_w_muls(a, b);
#ifdef FP_DEBUG
    /* The Q16.48 result is (prod >> 48); its integer part is (prod >> 96).
     * Overflow iff that integer part leaves the representable ±32767 range
     * (i.e. the result would not fit in int64). Trip loudly in dev builds. */
    int64_t ipart = fp_w_to_i64(fp_w_asr(prod, 2 * FP_PRECISION));
    if (ipart > 32767 || ipart < -32768) {
        fprintf(stderr,
                "fp_mul overflow: a=%lld b=%lld -> integer part %lld out of [-32768,32767]\n",
                (long long)a, (long long)b, (long long)ipart);
        abort();
    }
#endif
    return (fixed_t)fp_w_to_i64(fp_w_asr(prod, FP_PRECISION));
}

static inline fixed_t fp_div(fixed_t a, fixed_t b) {
#ifdef FP_DEBUG
    if (b == 0) {
        fprintf(stderr, "fp_div by zero: a=%lld\n", (long long)a);
        abort();
    }
#endif
    /* Left-shifting a negative value is UB in C99/C11 (6.5.7p4) — UBSan
     * traps on it. Shift the magnitude as unsigned instead, then reapply
     * the sign. The unsigned negation also makes a == INT64_MIN safe.
     * Truncating division of the signed numerator is unchanged, so results
     * are bit-identical to the old shift wherever it happened to work. */
    uint64_t mag = (a < 0) ? -(uint64_t)a : (uint64_t)a;
    fp_w128 num = fp_w_shl(fp_w_from_u64(mag), FP_PRECISION);
    if (a < 0) num = fp_w_neg(num);
    return (fixed_t)fp_w_to_i64(fp_w_divs(num, fp_w_from_i64(b)));
}

static inline fixed_t fp_abs(fixed_t a) {
    return a >= 0 ? a : -a;
}

static inline fixed_t fp_from_int(int x) {
    /* Same UB rule as fp_div: never left-shift a negative. Widening to
     * 128 bits keeps x == -32768 exact (its magnitude shifted is 2^63,
     * which only fits back in int64 after the negation). */
    uint64_t mag = (x < 0) ? -(uint64_t)x : (uint64_t)x;
    fp_w128 r = fp_w_shl(fp_w_from_u64(mag), FP_PRECISION);
    if (x < 0) r = fp_w_neg(r);
    return (fixed_t)fp_w_to_i64(r);
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

/* fp_print drags in <stdio.h>, which freestanding/embedded builds may not
 * have (and which a no-libm audit should not see by surprise). Opt in with
 * -DFP_MATH_WITH_STDIO. fp_to_double stays available unconditionally: it
 * uses no libc, only a float cast, and is for display/testing ONLY. */
#ifdef FP_MATH_WITH_STDIO
#include <stdio.h>

static inline void fp_print(fixed_t x, int decimals) {
    /* Negate via unsigned magnitude: -x overflows when x == INT64_MIN. */
    uint64_t mag = (x < 0) ? -(uint64_t)x : (uint64_t)x;
    if (x < 0) printf("-");
    printf("%llu.", (unsigned long long)(mag >> FP_PRECISION));
    uint64_t frac_mask = ((uint64_t)1 << FP_PRECISION) - 1;
    uint64_t frac_part = mag & frac_mask;
    for (int d = 0; d < decimals; d++) {
        frac_part *= 10;
        printf("%llu", (unsigned long long)(frac_part >> FP_PRECISION));
        frac_part &= frac_mask;
    }
}
#endif /* FP_MATH_WITH_STDIO */

static inline double fp_to_double(fixed_t x) {
    return (double)x / (double)FP_ONE;
}

/* ================================================================== */
/*  Integer Square Root (bit-by-bit, from sqrt2_bitwise.c)             */
/* ================================================================== */

#if FP_MATH_HAS_INT128
/* Native-only: the fp_u128 signature cannot exist without compiler
 * __int128. Nothing in this header or astro-nav calls it — fp_sqrt
 * below is the library's square root — so the portable backend simply
 * omits it rather than inventing a second wide-typed public surface. */
static inline fp_u128 isqrt128(fp_u128 n) {
    if (n == 0) return 0;
    fp_u128 x = 0;
    fp_u128 bit = (fp_u128)1 << 126;
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
#endif /* FP_MATH_HAS_INT128 */

/* Fast approximate sqrt — 64-bit only, no 128-bit division.
 * ~24 fractional bits of precision. Perfect for Adam denominator. */
static inline fixed_t fp_sqrt_fast(fixed_t x) {
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
static inline fixed_t fp_sqrt(fixed_t x) {
    if (x <= 0) return 0;

    /* We want r = isqrt(x << 48). Use Newton: r = (r + n/r) / 2 */
    fp_w128 n = fp_w_shl(fp_w_from_u64((uint64_t)x), FP_PRECISION);

    /* Initial guess via CLZ: find bit-width of n, then r0 ~ 2^(bw/2).
     * n < 2^111, so r0 <= 2^56 and Newton descends monotonically from
     * there: r and n/r both stay within 64 bits for the whole loop. */
    int bw = fp_w_bits(n);
    uint64_t r = (uint64_t)1 << ((bw + 1) / 2);

    /* Newton iterations: converges quadratically, 6 iterations gives
     * >48 bits of precision from any 1-bit initial guess */
    for (int i = 0; i < 7; i++) {
        if (r == 0) return 0;
        uint64_t q = (uint64_t)fp_w_to_i64(fp_w_divs(n, fp_w_from_u64(r)));
        uint64_t nr = (r + q) >> 1;
        if (nr >= r) break; /* converged */
        r = nr;
    }

    /* Final correction: ensure r^2 <= n */
    while (fp_w_cmp(fp_w_muls((int64_t)r, (int64_t)r), n) > 0) r--;

    return (fixed_t)r;
}

/* ================================================================== */
/*  Inverse Square Root: 1/sqrt(x) via Newton y*(3 - x*y^2)/2         */
/* ================================================================== */

static inline fixed_t fp_inv_sqrt(fixed_t x) {
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
    /* xn normalizes into [2^48, 2^50): it and every per-iteration value
     * below except the raw products fit int64, so only the multiplies
     * go through the 128-bit working type. */
    int64_t xn = (k >= 0) ? (x >> (2 * k)) : (x << (-2 * k));

    /* Initial guess by sub-octave so xn*y0^2 < 3 (Newton basin) for all xn:
     *   xn in [1,2) (msb 48) -> y0 = 1.0 ;  xn in [2,4) (msb 49) -> y0 = 0.5 */
    int msb_n = 63 - __builtin_clzll((uint64_t)xn);
    int64_t y = (msb_n == 48) ? ((int64_t)1 << FP_PRECISION)
                              : ((int64_t)1 << (FP_PRECISION - 1));
    int64_t three = fp_from_int(3);
    for (int i = 0; i < 8; i++) {
        int64_t y2 = fp_w_to_i64(                     /* <= ~2^48  */
            fp_w_asr(fp_w_muls(y, y), FP_PRECISION));
        int64_t xy2 = fp_w_to_i64(                    /* <= ~2^50  */
            fp_w_asr(fp_w_muls(xn, y2), FP_PRECISION));
        int64_t factor = three - xy2;
        /* (y*factor >> 48) >> 1 == asr by 49: both shifts arithmetic. */
        y = fp_w_to_i64(                              /* |y*factor| <= ~2^99 */
            fp_w_asr(fp_w_muls(y, factor), FP_PRECISION + 1));
        if (y <= 0) { y = 1; break; }
    }

    /* Scale back by 2^(-k); saturate when out of Q16.48 range. */
    if (k >= 0) {
        int64_t r = y >> k;
        return (fixed_t)(r < 0 ? 0 : r);
    }
    fp_w128 r = fp_w_shl(fp_w_from_i64(y), -k);       /* y<<24 can pass 2^63 */
    if (fp_w_cmp(r, fp_w_from_i64(INT64_MAX)) > 0) return INT64_MAX;
    if (fp_w_cmp(r, fp_w_from_i64(0)) < 0) return 0;
    return (fixed_t)fp_w_to_i64(r);
}

/* ================================================================== */
/*  Exponential: exp(x) as refinement invariant                        */
/* ================================================================== */

/* exp(x) = lim_{k->inf} (1 + x/2^k)^(2^k) via repeated squaring */
static inline fixed_t fp_exp_dyadic(fixed_t x, int k) {
    fixed_t base = FP_ONE + (x >> k);
    fixed_t result = base;
    for (int i = 0; i < k; i++)
        result = fp_mul(result, result);
    return result;
}

/* Full exp with negative handling.
 * k=14 gives ~42-bit accuracy — sufficient for softmax/loss. */
static inline fixed_t fp_exp(fixed_t x) {
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
static inline fixed_t fp_safe_exp(fixed_t x) {
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
static inline fixed_t fp_log(fixed_t x) {
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
static inline fixed_t fp_safe_log(fixed_t x) {
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
static inline fixed_t fp_atan_taylor(fixed_t x, int max_terms) {
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
static inline fixed_t fp_compute_pi(void) {
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
static inline fixed_t fp_atan_pow2(int i) {
    if (i >= 25) return FP_ONE >> i; /* arctan(x) ~ x for tiny x */
    fixed_t x = FP_ONE >> i;
    return fp_atan_taylor(x, 40);
}

/* Initialize CORDIC tables and pi — call once before use */
static inline void fp_math_init(void) {
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

/* CORDIC rotation: compute (cos(angle), sin(angle)) using shifts+adds */
static inline void fp_sincos(fixed_t angle, fixed_t *cos_out, fixed_t *sin_out) {
    fp_math_init();

    /* Reduce to [-pi, pi] */
    fixed_t two_pi = FP_PI << 1;
    while (angle > FP_PI) angle -= two_pi;
    while (angle < -FP_PI) angle += two_pi;

    /* Handle quadrants: CORDIC converges for |angle| < pi/2 */
    fixed_t pi_half = FP_PI >> 1;
    int negate_cos = 0;
    if (angle > pi_half) {
        angle = FP_PI - angle;
        negate_cos = 1;
    } else if (angle < -pi_half) {
        angle = -FP_PI - angle;
        negate_cos = 1;
    }

    fixed_t x = fp_cordic_gain;
    fixed_t y = 0;
    fixed_t z = angle;

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
static inline fixed_t fp_sigmoid(fixed_t x) {
    if (x > fp_from_int(10)) return FP_ONE;
    if (x < -fp_from_int(10)) return 0;
    fixed_t exp_neg = fp_safe_exp(-x);
    return fp_div(FP_ONE, FP_ONE + exp_neg);
}

/* SiLU(x) = x * sigmoid(x) — the activation used in Llama's SwiGLU */
static inline fixed_t fp_silu(fixed_t x) {
    return fp_mul(x, fp_sigmoid(x));
}

/* ================================================================== */
/*  PRNG (xorshift64 — already integer)                                */
/* ================================================================== */

static unsigned long long fp_rng_state = 42;

static inline unsigned long long fp_rng_next(void) {
    fp_rng_state ^= fp_rng_state << 13;
    fp_rng_state ^= fp_rng_state >> 7;
    fp_rng_state ^= fp_rng_state << 17;
    return fp_rng_state;
}

/* Uniform random in [0, FP_ONE) — pure integer */
static inline fixed_t fp_rng_uniform(void) {
    /* Take top 48 bits of 64-bit random, scale to [0, FP_ONE) */
    return (fixed_t)(fp_rng_next() >> (64 - FP_PRECISION));
}

/* Gaussian via CLT: sum 12 uniforms - 6 ≈ N(0,1)
 * Simple, fast, integer-only, no transcendentals needed */
static inline fixed_t fp_gaussian(fixed_t mean, fixed_t std) {
    fixed_t sum = 0;
    for (int i = 0; i < 12; i++)
        sum += fp_rng_uniform();
    /* sum ~ N(6*FP_ONE, FP_ONE^2)
     * (sum - 6*FP_ONE) ~ N(0, FP_ONE) in fixed-point = N(0,1) */
    fixed_t z = sum - 6 * FP_ONE;
    return mean + fp_mul(std, z);
}

/* Shuffle array of ints using integer RNG */
static inline void fp_shuffle_ints(int *arr, int n) {
    for (int i = n - 1; i > 0; i--) {
        /* j in [0, i]: use modulo (biased but acceptable for shuffle) */
        int j = (int)(fp_rng_next() % (unsigned long long)(i + 1));
        int tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
    }
}

#endif /* FP_MATH_H */
