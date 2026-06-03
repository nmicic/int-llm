# `fp_math.h` — pure-integer Q16.48 math

> The spine of [int-llm](README.md). A single header, no dependencies, **zero floating point** —
> square root, exp, log, sin/cos, sigmoid, SiLU, a PRNG and Gaussian sampler, all in `int64_t`
> fixed point. The transformer on top of this repo is the headline; this file is the hidden gem.
> If you only take one thing from here for your own projects, take this.

`fp_math.h` is header-only and self-contained. Drop it in, `#include` it, and you have every
math primitive a small neural net needs without ever touching `float`, `double`, or `libm`. It
came **before** the rest of the repo — see [the seed visualizations](viz/) and
[Provenance](#provenance) at the end. The library is what grew out of convincing myself that you
can compute deterministic Q16.48 approximations of e, π, √2 and `e^(iπ)+1=0` with nothing but
integers, shifts and adds.

---

## The format: Q16.48

Everything is `typedef int64_t fixed_t;` interpreted as **Q16.48** — 16 integer bits, 48
fractional bits:

```
[63]    [62 .. 48]      [47 .. 0]
sign    integer part    fractional part
```

| | |
|---|---|
| `FP_PRECISION` | `48` |
| `FP_ONE` | `(int64_t)1 << 48` = `281474976710656` (the value `1.0`) |
| `FP_HALF` | `1 << 47` (the value `0.5`) |
| Range | ±32767.999… (±2¹⁵ − 1 ulp) |
| Resolution | 2⁻⁴⁸ ≈ 3.55 × 10⁻¹⁵ (≈ 14.4 decimal digits) |

For comparison: float32 carries ~7.2 decimal digits, bfloat16 ~2.4. **Q16.48 has more fractional
precision than float32**, so float16/bfloat16 model weights inside the Q16.48 range can preserve
their source mantissa exactly. The tradeoff is range: this is a ±32k format, not an
exponent-scaled one, which is exactly why the integer LLM clamps and range-reduces where it does.

### Requirements

- `__int128` (gcc/clang, 64-bit). Every multiply and divide uses a 128-bit intermediate for full
  precision, then shifts back to Q16.48. The header `#error`s out without it.
- Compiles with **no `-lm`** — there is no libm dependency, by design.

---

## Quick start

```c
#include "fp_math.h"

int main(void) {
    fp_math_init();                       // once, before CORDIC sin/cos (idempotent)

    fixed_t a = fp_from_int(3);           // 3.0
    fixed_t b = fp_div(FP_ONE, fp_from_int(4));   // 0.25
    fixed_t c = fp_mul(a, b);             // 0.75

    fixed_t e = fp_exp(FP_ONE);           // e^1
    fp_print(e, 12); printf("\n");        // 2.718281828...

    fixed_t s, co;
    fp_sincos(FP_PI >> 2, &co, &s);       // sin/cos of π/4
}
```

Only `fp_sincos` (and the `FP_PI` / CORDIC tables it uses) needs `fp_math_init()`. Everything else
is stateless — except the PRNG, which shares one global seed.

---

## API reference

Signatures are exact. "Cost" is from `fp_test.c`'s benchmark section on Apple M-series, `-O3`,
100K iterations — order-of-magnitude, not a guarantee.

### Arithmetic

| Function | Returns | Notes |
|---|---|---|
| `fixed_t fp_mul(fixed_t a, fixed_t b)` | a × b | 128-bit intermediate, `>> 48`. ~1 ns. Build with `-DFP_DEBUG` to trap overflow. |
| `fixed_t fp_div(fixed_t a, fixed_t b)` | a ÷ b | `(int128)a << 48 / b`. ~7 ns. |
| `fixed_t fp_abs(fixed_t a)` | \|a\| | |
| `fixed_t fp_from_int(int x)` | x as Q16.48 | `x << 48`. Max \|x\| = 32767. |
| `fixed_t fp_max(fixed_t a, fixed_t b)` | max(a, b) | |
| `fixed_t fp_min(fixed_t a, fixed_t b)` | min(a, b) | |

### Square roots

| Function | Returns | Notes |
|---|---|---|
| `fixed_t fp_sqrt(fixed_t x)` | √x | Newton-Raphson with `__builtin_clz` initial guess, 128-bit. Full 48-bit precision. ~60 ns. `x ≤ 0 → 0`. |
| `fixed_t fp_sqrt_fast(fixed_t x)` | √x (approx) | 64-bit only, `r << 24` trick. ~24 fractional bits — plenty for an Adam denominator. ~0.5 ns. |
| `fixed_t fp_inv_sqrt(fixed_t x)` | 1/√x | Direct Newton `y·(3 − x·y²)/2`, no sqrt-then-divide. Range-reduced so no `__int128` overflow on any positive input; saturates to `INT64_MAX` when 1/√x leaves range. ~44 ns. |
| `uint128_t isqrt128(uint128_t n)` | ⌊√n⌋ | Exact integer sqrt, bit-by-bit, for 128-bit operands. |

> `fp_inv_sqrt`'s range reduction is the subject of a real cross-machine determinism fix — an
> earlier version relied on `__int128` not overflowing, which it did for tiny `x`, diverging
> arm64/clang from x86/gcc until `make determinism` caught it. The fix is the reason that gate
> exists.

### Exponential & logarithm

| Function | Returns | Notes |
|---|---|---|
| `fixed_t fp_exp(fixed_t x)` | eˣ | Dyadic limit `(1 + x/2¹⁴)^(2¹⁴)` via 14 squarings; negatives via reciprocal. ~42-bit accuracy. ~15 ns. |
| `fixed_t fp_safe_exp(fixed_t x)` | eˣ, clamped | Clamps `x` to ≤ 10 (≈ 22026, fits), returns 0 for `x < −10`. Softmax/sigmoid-safe. |
| `fixed_t fp_log(fixed_t x)` | ln(x) | Newton `y − 1 + x/eʸ` with CLZ-based `log2` initial guess. ~40-bit. ~325 ns. `x ≤ 0` → sentinel `−50`. |
| `fixed_t fp_safe_log(fixed_t x)` | ln(x), clamped | Clamps tiny inputs to avoid extreme values. Cross-entropy-safe. |
| `fixed_t fp_exp_dyadic(fixed_t x, int k)` | eˣ, k rounds | Low-level, tunable precision (higher `k` = more accurate, more cost). |

### Trigonometry — CORDIC (shifts and adds only)

| Function | Returns | Notes |
|---|---|---|
| `void fp_math_init(void)` | — | Builds `FP_PI`, the CORDIC angle table, and the gain. Call once. Idempotent, not thread-safe. |
| `void fp_sincos(fixed_t θ, fixed_t *cos_out, fixed_t *sin_out)` | cos θ, sin θ | 48 CORDIC iterations, vectoring mode. Reduces to [−π, π] then per-quadrant. 48-bit. ~49 ns. Auto-inits. |
| `fixed_t fp_compute_pi(void)` | π | Machin: `16·atan(1/5) − 4·atan(1/239)`, integer Taylor arctan. 13+ correct digits. |
| `fixed_t fp_atan_taylor(fixed_t x, int max_terms)` | arctan(x) | `x − x³/3 + x⁵/5 − …`, stops early when terms vanish. |
| `fixed_t fp_atan_pow2(int i)` | arctan(2⁻ⁱ) | CORDIC table init helper. |

Constants set by `fp_math_init()`: `FP_PI` (≈ 3.14159265358978…), `fp_cordic_gain` (K⁻¹ ≈
0.60725293500…), `fp_cordic_angles[48]`.

### Activations (for SwiGLU / Llama-style MLPs)

| Function | Returns | Notes |
|---|---|---|
| `fixed_t fp_sigmoid(fixed_t x)` | 1 / (1 + e⁻ˣ) | Saturates: `→ 1` for `x > 10`, `→ 0` for `x < −10`. |
| `fixed_t fp_silu(fixed_t x)` | x · sigmoid(x) | The activation in Llama's SwiGLU. |

### PRNG & sampling

| Function | Returns | Notes |
|---|---|---|
| `unsigned long long fp_rng_next(void)` | raw 64-bit | xorshift64. ~1.5 ns. Global state `fp_rng_state` (default seed 42). |
| `fixed_t fp_rng_uniform(void)` | uniform [0, 1) | Top 48 bits of the raw word. |
| `fixed_t fp_gaussian(fixed_t mean, fixed_t std)` | N(μ, σ²) sample | Central Limit Theorem: Σ of 12 uniforms − 6. No transcendentals. ~18 ns. |
| `void fp_shuffle_ints(int *arr, int n)` | — | Fisher-Yates. |

For multi-threaded use, give each thread its own `fp_rng_state` — the generator's only shared
state.

### Display (I/O only — not computation)

| Function | Returns | Notes |
|---|---|---|
| `void fp_print(fixed_t x, int decimals)` | — | Prints decimal digits by integer extraction (`×10`, shift). |
| `double fp_to_double(fixed_t x)` | double | **Display only.** The one place a `double` appears — never feed it back into compute, or you break the integer guarantee. |

---

## Precision & cost at a glance

| Function | Precision | ~Cost |
|---|---|---|
| `fp_mul` | exact (to ulp) | 1 ns |
| `fp_div` | exact (to ulp) | 7 ns |
| `fp_sqrt` | 48-bit | 60 ns |
| `fp_sqrt_fast` | ~24-bit | 0.5 ns |
| `fp_inv_sqrt` | ~44-bit | 44 ns |
| `fp_exp` | ~42-bit | 15 ns |
| `fp_log` | ~40-bit | 325 ns |
| `fp_sincos` | 48-bit | 49 ns |
| `fp_gaussian` | — | 18 ns |
| `fp_rng_next` | — | 1.5 ns |

---

## The property that matters: bit-exact across machines

Because every operation is integer arithmetic with defined overflow behavior (and the one place
`__int128` could overflow is range-reduced away), `fp_math.h` produces **the same bits on the
tested platforms** — x86_64/gcc and arm64/clang included. That isn't asserted, it's *gated*:
`make determinism` hashes the raw integer outputs of the whole library over a fixed input grid
and compares against a committed golden. Matching the hash is the portability check. The full test suite
(`fp_test.c`, 335 checks: Euler's identity, `log(exp(x)) = x`, `sin²+cos² = 1`, …) is the
correctness side of that coin.

```bash
make test          # 335 library invariants
make determinism   # cross-machine bit-exact hash vs committed golden
```

---

## Provenance

This header was **consolidated from five standalone programs** that came first — each one a small
proof that a famous constant or function falls out of pure integer refinement:

```
euler_identity.c · transcendentals_bitwise.c · sqrt2_bitwise.c · e_bitwise.c · pi_bitwise.c
```

Their interactive companions live in [`viz/`](viz/): √2, e and π by dyadic refinement,
the transcendentals, and the full journey to **e<sup>iπ</sup> + 1 = 0** — the same identity that
ships as a passing test in `fp_test.c`, evaluated end to end in Q16.48 with no floating point. The
storyline is honest: *integer lattices → the most beautiful equation → an integer LLM.* The math
is the seed; this repo is what it grew into.

And a small open wish: if integer-only LLM training ever becomes the norm, and any of this nudged
it along — a citation, or just a quiet *thanks*, would make the experiment worth it.

## License

Apache-2.0 © 2026 Nenad Mićić. Same as the rest of the repo — take it,
use it, build on it.
