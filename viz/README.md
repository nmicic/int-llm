# Visualizations — the math foundation behind `fp_math.h`

These are the **seed visualizations** for [`int-llm`](../README.md). They predate the rest of
the repo: standalone HTML pages I built first, to convince myself that transcendental constants
and functions can be approximated deterministically with nothing but integers, shifts and adds. The
library `fp_math.h` is what grew out of them; the integer LLM is what that library turned out
to be good enough to run.

Open [`index.html`](index.html) for a launcher, or go straight to the pages:

## `math/` — what `fp_math.h` actually computes

| Visualization | Library function | Idea |
|---|---|---|
| [`sqrt2-dyadic-refinement.html`](math/sqrt2-dyadic-refinement.html) | `fp_sqrt`, `isqrt128` | √2 by bit-by-bit / Newton refinement |
| [`e-dyadic-refinement.html`](math/e-dyadic-refinement.html) | `fp_exp` | e as the dyadic limit `(1 + x/2^k)^(2^k)` |
| [`pi-dyadic-refinement.html`](math/pi-dyadic-refinement.html) | `fp_compute_pi` | π via Machin's formula, integer Taylor arctan |
| [`transcendentals-visualization.html`](math/transcendentals-visualization.html) | `fp_exp`, `fp_log`, `fp_sincos` | exp / log / sin / cos in fixed point |
| [`euler-complete-journey.html`](math/euler-complete-journey.html) | the lot, together | e<sup>iπ</sup> + 1 = 0 — the most beautiful equation, in pure integers |

The Euler page walks the same identity that ships as a passing unit test in `fp_test.c`:
`e^(iπ) + 1 = 0` evaluated end-to-end in Q16.48, no floating point anywhere.

## Notes

- Self-contained HTML — open locally in any browser, no build step, no network needed.
- Exploratory aids, **not** proofs. The library's correctness lives in `fp_test.c` and the
  cross-machine `make determinism` gate, not in these pages.
- The full API reference for the library these illustrate is in [`FP_MATH.md`](../FP_MATH.md).
