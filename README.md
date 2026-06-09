# int-llm

**Author:** Nenad Micic, Belgium — [LinkedIn](https://be.linkedin.com/in/nenadmicic)

Copyright (c) 2026 Nenad Mićić — Apache-2.0

**Bit-exact, deterministic LLM inference in pure-integer C.**

int-llm runs a real Llama-family checkpoint (TinyLlama-1.1B) through a fixed-point integer compute core — no `float`, no `double`, no `libm` — and produces **identical raw bitstreams across platforms and compilers**. The determinism gate hashes the integer outputs and matches a committed golden hash on x86_64/gcc and arm64/clang. Run it today, run it next year on different hardware: same bits.

The integer path also matches the float reference **token-for-token** (80/80 on the greedy verification gate), and a tiny character-level GPT *trains and samples* entirely in integer arithmetic. You can even convert the whole model into an all-integer `.mgw` weight file and run inference straight from that — the original floating-point weights never reappear in the compute path.

This is a feasibility demo, deliberately slow and simple. What it provides is something most LLM stacks don't: a **reproducible, exact reference** — an oracle — for anyone working on quantization, numerical drift, regression testing, or deterministic inference.

- `gpt_int`: integer-only Q16.48 character GPT — trains **and** samples from a small downloaded names dataset
- `gpt_float`: float32 baseline (for comparison)
- `llama_int`: integer-only Llama-family inference (TinyLlama-1.1B), CPU, dynamic `config.json`

The design rule is strict: machine-native representation stays in the core, and human-readable interpretation happens only at the boundary.

## Why Q16.48? Why build an oracle first?

The obvious objection: *Q16.48 in an `int64_t` is wastefully wide — why not INT8 like everyone else?*

Because the width is the point. **Q16.48 is the oracle, not the product.**

Q16.48 carries 48 fractional bits (resolution ≈ 3.55e-15) — enough to hold normal float16/bfloat16 model weights *exactly* inside its range. That has a powerful consequence: when the integer path disagrees with the float reference, the disagreement is a **bug**, never rounding ambiguity. Combined with the determinism gate, this gives a fixed ground truth that doesn't drift across machines, compilers, or time.

Lower-precision work desperately needs that ground truth. My own GPU experiments (see [`gpu/`](gpu/)) showed why: an INT8 tensor-core path with per-tensor quantization matched the reference on only 29/80 tokens. Without an exact oracle, there is no way to attribute that — quantization error? kernel bug? bad scaling scheme? With the oracle, every deviation is measurable and debuggable.

So the methodology is deliberate:

1. **First, build a path that is exact and reproducible** — wide fixed point, bit-identical across platforms, verified token-for-token against the float reference.
2. **Then descend the precision ladder** (Q16.16, Q8.24, INT8 SIMD, ...) with a working compass, measuring every step against fixed ground truth instead of against another approximation.

This repo is step 1, done properly. Step 2 is future work — and anyone exploring deterministic inference, fixed-point quantization, or cross-platform numerical reproducibility can reuse the oracle as-is.

If you only want the math, [`fp_math.h`](fp_math.h) is a self-contained, dependency-free Q16.48 integer math library (sqrt, exp, log, sin/cos via CORDIC, sigmoid, SiLU, deterministic PRNG) — see [`FP_MATH.md`](FP_MATH.md).

## Quick Start

Needs only a C compiler with `__int128` (gcc/clang, 64-bit). The integer paths have no dependencies — no math library.

### 1. 30 seconds — train a tiny GPT in pure integer

```bash
make input gpt_int && ./gpt_int
```

`make input` downloads Karpathy's public makemore names dataset into `input.txt` if it is missing. The repo does not ship a training corpus. The tiny demo then trains a small character-level GPT and samples from it — **all in Q16.48 fixed point, zero floating point**, in a couple of seconds. Once `input.txt` exists, this path is pure C with no model download.

### 2. Run an open LLM (TinyLlama-1.1B) integer-only

```bash
# a. Download TinyLlama-1.1B in HuggingFace format (~2.2 GB safetensors)
huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0 --local-dir models/tinyllama

# b. Build the integer Llama engine
make llama_int

# c. Generate text — integer-only inference (greedy)
./llama_int models/tinyllama --generate --prompt "What is the capital of France?" --max-new-tokens 16
```

`--prompt` uses the C-native tokenizer when `models/tinyllama/tokenizer.json` is present (it is, in the HF download); otherwise it falls back to a Python bridge that needs `transformers`. CPU-only and slow — that's expected; this is a feasibility demo, not a fast runtime. (Default is layer-streaming, ~0.2 tok/s on a desktop CPU; add `--cache-layers` to hold all weights in RAM for a faster, higher-memory run.)

The repo does not ship model weights; downloaded checkpoints remain under their upstream license.

### 3. (The fun part) Convert the whole model to integer weights, then run from those

```bash
# Translate every TinyLlama tensor into Q16.48 integers — one all-integer weight file
./llama_int models/tinyllama --export-native models/tinyllama.mgw

# Run inference straight from the integer weights — no float weight re-conversion at load
./llama_int models/tinyllama.mgw --native --generate --prompt "What is the capital of France?" --max-new-tokens 16

# (Optional) prove the all-integer-weight run still matches the float reference, token-for-token
./llama_int models/tinyllama.mgw --native --ref-dir models/tinyllama --verify
```

The `.mgw` file is the entire model as a flat array of `int64_t` Q16.48 values — the original float weights never reappear in the model loader or compute path. `--native` mmaps it (all layers resident, faster); `--native-stream` reads layers on demand for low-RAM machines. This is the "fully integer, weights and all" version of the demo: no floating-point weights or arithmetic in the model core.

### 4. (Optional) Verify the integer output matches the float reference, token-for-token

```bash
./llama_int models/tinyllama --dump-reference > gen_ref.py
python3 gen_ref.py            # writes models/tinyllama/reference_tokens.txt (needs torch + transformers)
./llama_int models/tinyllama --verify --cache-layers    # expect: 80/80 tokens match
```

### 5. (Optional) The reproducibility hook

```bash
make test            # fp_math.h unit tests
make determinism     # cross-machine bit-exact hash — identical on x86_64/gcc and arm64/clang
```

`make determinism` hashes the raw integer outputs of the core math over a fixed input grid and compares against a committed golden. Matching the golden is the project's reproducibility gate; it has been checked on x86_64/gcc and arm64/clang.

---

## How it works

Everything in the compute core is `int64_t` in **Q16.48** fixed point: 16 integer bits, 48 fractional bits, with `__int128` intermediates for full-precision multiply/divide. The integer compute core has no `float`, no `double`, and no `libm` dependency; floating point only appears at boundaries such as source-weight conversion, profiling/display, CLI argument conversion, and reference comparisons. Square root, exp, log, sin/cos, sigmoid and SiLU are all hand-rolled in `fp_math.h` (CORDIC, Newton, dyadic refinement).

> **The hidden gem:** `fp_math.h` is a self-contained, dependency-free integer math library that stands on its own — it's the seed the whole project grew from. If you want to lift just the math, start with **[`FP_MATH.md`](FP_MATH.md)** (full API reference) and the [`viz/`](viz/) gallery that shows e, π, √2 and `e^(iπ)+1=0` being computed in pure integers.

The design rule is one line: **machine-native representation stays in the core, human-readable interpretation happens only at the boundary.** For `llama_int` that boundary is the tokenizer; for `gpt_int` it's the few `printf`s that turn fixed-point back into decimal for display.

The Llama pipeline:

```text
prompt text
  -> tokenizer (tokenizer.h, or hf_tokenizer_bridge.py fallback)
  -> token ids
  -> llama_int integer core:
       embeddings -> N transformer layers (RMSNorm, GQA attention with RoPE,
       SwiGLU MLP, residuals) -> final norm -> lm_head logits -> next token
  -> generated token ids
  -> tokenizer -> decoded text
```

Weights load from a standard Hugging Face directory (`config.json` + safetensors). Tensors are converted from their source dtype (float16 / bfloat16) into Q16.48 at load time. For normal model weights inside Q16.48 range this preserves the source mantissa exactly; values outside range are not the target use case. Loading is **streaming by default** (one layer converted, used, freed — keeps memory small on a laptop) or `--cache-layers` (all layers resident, faster). For GQA models like TinyLlama the KV-cache is sized by `num_kv_heads` (4), not `num_heads` (32).

## What's proven — and what isn't

**Proven:**

- A real Llama-family checkpoint (TinyLlama-1.1B-Chat) runs end-to-end through a pure-integer inference core, prompt text to decoded text.
- The integer output matches the float reference **token-for-token** (80/80 across the benchmark prompts) under greedy decoding.
- The core fixed-point math is **bit-exact on the tested platforms**: the `make determinism` hash is identical on x86_64/gcc and arm64/clang. This is the headline property — deterministic bits, demonstrated by a gate rather than asserted.
- The tiny character GPT (`gpt_int`) both **trains and samples** entirely in integer arithmetic after downloading the small public names dataset.

**Not proven (and not claimed):**

- This is **slow on purpose** — a feasibility demo, not a fast runtime. No competitive throughput claims.
- No float-quality parity beyond the verified prompts; no production chat-quality evaluation.
- Larger models / longer contexts than the TinyLlama proof target aren't validated here.

> Aside: along the way we explored a geometric sign-code pre-filter for attention (a machine-native shortcut for the score computation). On this workload it gave no measurable benefit and is omitted from this artifact. The honest result is "didn't help here," kept out so the code that ships is only the code that earns its place.

### A parked experiment: GPU (`gpu/`)

There's a CUDA side-branch in [`gpu/`](gpu/) — kept as an honest **null result**, not part of the main build. A full FP16 TinyLlama decode on an RTX-class GPU reproduced the CPU's token output exactly, but FP16 is *floating point*, so it sits outside this project's zero-float thesis; the integer-on-GPU directions (INT8 / FP8 / INT64 kernels, CUDA-graph capture) were parked as negative-or-mixed. It needs `nvcc` + a GPU and is deliberately not wired into `make all`. See [`gpu/README.md`](gpu/README.md) for what was tried and why none of it earned a place in the integer core.

## Build and run

```bash
make all          # gpt_float, gpt_int, llama_int, fp_test, fp_determinism
make test         # fp_math.h unit tests (335)
make determinism  # cross-machine bit-exact hash vs committed golden
make regression   # hard-stop gate: unit tests + determinism
```

Integer targets build with **no `-lm`** — pure integer, no external math dependency. See **Quick Start** above for the TinyLlama generate/verify flow.

## Repo layout

| File | Purpose |
|------|---------|
| `microgpt_int.c` | Integer-only Q16.48 character GPT — trains + samples |
| `microgpt.c` | Float32 baseline (for comparison) |
| `llama_int.c` | Integer-only Llama-family inference (TinyLlama-1.1B) |
| `fp_math.h` | Header-only Q16.48 math library (sqrt, exp, log, sin/cos, sigmoid, SiLU, PRNG) — see [`FP_MATH.md`](FP_MATH.md) |
| `FP_MATH.md` | Full API reference for `fp_math.h` + the "integer lattices → Euler → integer LLM" story |
| `viz/` | Seed visualizations: e, π, √2 and `e^(iπ)+1=0` computed in pure integers (open `viz/index.html`) |
| `fp_determinism.c` | Cross-machine bit-exact determinism gate |
| `fp_test.c` | Unit tests for the fixed-point library |
| `safetensors.h` | Local safetensors loader (shape validation, dtype→Q16.48 conversion) |
| `tokenizer.h` | C-native tokenizer for Hugging Face `tokenizer.json` |
| `hf_tokenizer_bridge.py` | Python tokenizer fallback when the C path can't be used |
| `scripts/download_input.sh` | Downloads the public names dataset into `input.txt` for the tiny GPT demo |
| `gpu/` | Parked CUDA experiments (FP16/INT benchmarks) — honest null result, not in `make all`; see `gpu/README.md` |

## Acknowledgements

The character-GPT design is inspired by **Andrej Karpathy's** [microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) and guide — a ~200-line Python char GPT using the public makemore names dataset. The C here is an independent from-scratch rebuild, not a port; the baseline config (`N_EMBD=32, N_HEAD=4, N_LAYER=1`, 14,656 params) is the one that pairs cleanly with the integer variant for side-by-side comparison. A leaner speed-tuned variant of the float trainer (sub-20 ms/step) lives in [this gist](https://gist.github.com/nmicic/35316463f3c5e8e9fe8eb599b3842b58).

## License

Apache-2.0 © 2026 Nenad Mićić. See [LICENSE](LICENSE).
