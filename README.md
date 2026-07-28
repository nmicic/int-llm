# int-llm

**Author:** Nenad Micic, Belgium — [LinkedIn](https://be.linkedin.com/in/nenadmicic)

Copyright (c) 2026 Nenad Mićić — Apache-2.0

**Bit-exact, deterministic LLM inference in pure-integer C — from x86 servers down to 32-bit microcontrollers.**

int-llm runs a real Llama-family checkpoint (TinyLlama-1.1B) through a fixed-point integer compute core — no `float`, no `double`, no `libm` — and produces **bit-identical outputs across platforms and compilers**. The determinism gate hashes the integer outputs and matches a committed golden hash on x86_64/gcc, arm64/clang, and every recorded 32-bit target. Run it today, run it next year on different hardware: same bits.

The integer path also matches the float reference **token-for-token** (80/80 on the greedy verification gate), and a tiny character-level GPT *trains and samples* entirely in integer arithmetic. You can even convert the whole model into an all-integer `.mgw` weight file and run inference straight from that — the original floating-point weights never reappear in the compute path.

This is a feasibility demo, deliberately slow and simple. What it provides is something most LLM stacks don't: a **reproducible, exact reference** — an oracle — for anyone working on quantization, numerical drift, regression testing, or deterministic inference.

`fp_math.h` no longer requires `__int128`: it now carries two interchangeable wide-math backends — native `__int128` where the compiler has it, and a portable two-limb software backend (C99 + 64-bit integers only) everywhere else — Cortex-M, RV32, Xtensa, ARMv6 — computing the **same bits**. This was matured in the [astro-nav-int](https://github.com/nmicic/astro-nav-int) sibling project and folded back here. `make determinism` and `make determinism-portable` both reproduce the committed golden hash, `gpt_int` training on a 2014 Raspberry Pi 1 B+ (32-bit ARMv6, no `__int128`) is byte-identical to the same run on a 64-bit host, and the train-big/run-small loop closes on microcontrollers: train on a laptop (`./gpt_int --save model.mgw`), run inference-only from the same weight file on a $5 RP2040 — byte-for-byte the same samples (see [`validation/cpu/`](validation/cpu/)). `llama_int` still uses raw `__int128` accumulators directly and remains 64-bit-only for now.

- `gpt_int`: integer-only Q16.48 character GPT — trains **and** samples from a small downloaded names dataset; `--save`/`--load model.mgw` round-trips the trained weights as pure integers
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

Needs only a C99/C11 compiler with 64-bit integers — `__int128` is used when available, otherwise the portable two-limb backend kicks in automatically. Two exceptions still use `__int128` directly and need a 64-bit host: `llama_int` and the `fp_test` self-test — so on a compiler without `__int128`, build targets individually (`make gpt_int fp_determinism`) rather than `make all`. The integer paths have no dependencies — no math library.

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

The repo does not ship TinyLlama or other downloaded checkpoint weights; those remain under their upstream licenses. The only committed weights are the 115,576-byte [`model.mgw`](model.mgw) tiny-GPT demo model, trained from the public names dataset.

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
make determinism     # cross-machine bit-exact hash — identical across the recorded targets
```

`make determinism` hashes the raw integer outputs of the core math over a fixed input grid and compares against a committed golden. Matching the golden is the project's reproducibility gate; it has been checked on every platform recorded under [`validation/cpu/`](validation/cpu/), as well as arm64/clang.

---

## How it works

Everything in the compute core is `int64_t` in **Q16.48** fixed point: 16 integer bits and 48 fractional bits, with either native `__int128` or portable two-limb intermediates for full-precision multiply/divide. The integer compute core has no `float`, no `double`, and no `libm` dependency; floating point only appears at boundaries such as source-weight conversion, profiling/display, CLI argument conversion, and reference comparisons. Square root, exp, log, sin/cos, sigmoid and SiLU are all hand-rolled in `fp_math.h` (CORDIC, Newton, dyadic refinement).

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
- The core fixed-point math is **bit-exact on the tested platforms**: the `make determinism` hash is identical on x86_64/gcc (AMD and Intel), arm64/clang, 32-bit ARMv6 Linux, and six bare-metal microcontroller boards in seven ISA-mode configurations across three ISA families (ARM Cortex-M, RISC-V, Xtensa — the Pico 2 is tested in both its ARM and RISC-V modes) — see [`validation/cpu/`](validation/cpu/). This is the headline property — deterministic bits, demonstrated by a gate rather than asserted.
- The tiny character GPT (`gpt_int`) both **trains and samples** entirely in integer arithmetic after downloading the small public names dataset.

**Not proven (and not claimed):**

- This is **slow on purpose** — a feasibility demo, not a fast runtime. No competitive throughput claims.
- No float-quality parity beyond the verified prompts; no production chat-quality evaluation.
- Larger models / longer contexts than the TinyLlama proof target aren't validated here.
- The `.mgw` loaders perform basic format and tensor-shape checks but are **not hardened against adversarial files** — `--load` / `--native` expect a trusted file: the committed `model.mgw`, one you trained, or one you exported yourself.

> Aside: along the way we explored a geometric sign-code pre-filter for attention (a machine-native shortcut for the score computation). On this workload it gave no measurable benefit and is omitted from this artifact. The honest result is "didn't help here," kept out so the code that ships is only the code that earns its place.

### A parked experiment: GPU (`gpu/`)

There's a CUDA side-branch in [`gpu/`](gpu/) — kept as an honest **null result**, not part of the main build. A full FP16 TinyLlama decode on an RTX-class GPU reproduced the CPU's token output exactly, but FP16 is *floating point*, so it sits outside this project's zero-float thesis; the integer-on-GPU directions (INT8 / FP8 / INT64 kernels, CUDA-graph capture) were parked as negative-or-mixed. It needs `nvcc` + a GPU and is deliberately not wired into `make all`. See [`gpu/README.md`](gpu/README.md) for what was tried and why none of it earned a place in the integer core.

## Validated on real hardware (`validation/cpu/`)

The determinism gate and the tiny GPT are not just "portable in theory" — both have
been run on real silicon, with results recorded in [`validation/cpu/`](validation/cpu/)
(one folder per target, harness + raw captured transcript). Two checks per target:

- **determinism** — the target recomputes the full `fp_determinism` grid on the
  portable backend and must reproduce the committed golden hash `c0d933ea340452ec`.
- **microgpt inference** — the laptop-trained [`model.mgw`](model.mgw) (115,576 bytes,
  committed) is baked into flash; the firmware loads it zero-copy (`mgpt_load_mem`,
  weights stay in memory-mapped flash, never RAM) and must reproduce the training
  host's 20 sampled names **byte-for-byte**, PRNG stream included.

Every target passed both:

| target | ISA | determinism | 20 samples |
|---|---|---|---|
| XIAO RP2040 (Cortex-M0+ @ 133 MHz) | Armv6-M | 10.9 s | 6.7 s |
| Raspberry Pi Pico 2 (RP2350, ARM mode, Cortex-M33) | Armv8-M | 4.1 s | 3.1 s |
| Raspberry Pi Pico 2 (RP2350, RISC-V mode, Hazard3) | rv32imac | 5.4 s | 3.8 s |
| ESP32-C6 | rv32imac | 5.6 s | 2.0 s |
| Heltec V3 (ESP32-S3, LX7 @ 240 MHz) | Xtensa | 3.9 s | 1.1 s |
| LILYGO T-Beam (ESP32, LX6 @ 240 MHz) | Xtensa | 4.2 s | 2.6 s |
| XIAO nRF52840 (Cortex-M4F @ 64 MHz) | Armv7E-M | 13.9 s | 3.4 s |
| Raspberry Pi 1 B+ (ARMv6, 32-bit Linux) | ARMv6 | 0.7 s | 0.2 s |
| AMD Ryzen 7 7700 (Linux, gcc) | x86-64 | native + portable | byte-identical |
| Intel Core i7-7700 (Linux, gcc) | x86-64 | native + portable | byte-identical |

Same die, two ISAs: the Pico 2 reproduces the identical output in both its ARM
and RISC-V boot modes (two separate firmware builds, one chip). The three Linux
rows go further and rerun the **full 5000-step training**: the AMD, Intel, and
ARMv6 Pi 1 hosts each reproduce the committed `model.mgw` byte-for-byte (as
does the arm64 macOS host that trained it). The harness conventions follow
astro-nav-int's hardware test records; `validation/cpu/HOWTO.md` covers host
setup, flashing quirks, and how a run is provenance-stamped (host pin,
firmware/source SHA-256, monitor transcript).

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
| `model.mgw` | Committed pretrained tiny-GPT weights (Q16.48, 115,576 B) — the reference image for `--load` and the MCU harnesses |
| `validation/cpu/` | Hardware validation records: determinism + inference harnesses and raw transcripts per target |
| `gpu/` | Parked CUDA experiments (FP16/INT benchmarks) — honest null result, not in `make all`; see `gpu/README.md` |

## Acknowledgements

The character-GPT design is inspired by **Andrej Karpathy's** [microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) and guide — a ~200-line Python char GPT using the public makemore names dataset. The C here is an independent from-scratch rebuild, not a port; the baseline config (`N_EMBD=32, N_HEAD=4, N_LAYER=1`, 14,272 params) is the one that pairs cleanly with the integer variant for side-by-side comparison. A leaner speed-tuned variant of the float trainer (sub-20 ms/step) lives in [this gist](https://gist.github.com/nmicic/35316463f3c5e8e9fe8eb599b3842b58).

## License

Apache-2.0 © 2026 Nenad Mićić. See [LICENSE](LICENSE).
