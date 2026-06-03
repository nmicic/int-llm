# `gpu/` — a parked GPU experiment (honest null result)

This directory is **not part of the main build** and is not on the path to the project's
thesis. It's kept in the open because a negative result that someone actually measured is
worth more than a hand-wave. If you came here from the top-level README looking for "the
integer GPU runtime" — there isn't one that earned its place. Here's what was tried and why.

> Needs `nvcc` + an NVIDIA GPU (developed against CUDA 13, `sm_120` / RTX-class). Nothing
> here is wired into `make all`; each `.cu` has its own build line in its header comment.

## The one thing that worked — and why it doesn't count

A full **FP16** TinyLlama-1.1B decode on the GPU reproduces the CPU integer core's token
output **exactly** (the 80/80 oracle gate in `bench_full_fp16.cu`, longer runs in
`bench_long_fp16.cu`). So the GPU pipeline is correct.

But FP16 is **floating point**. The whole point of this project is *zero* float — Q16.48
integers end to end, bit-for-bit reproducible across machines. An FP16 GPU run, however fast
and however accurate, is on the wrong side of that line. It's a useful sanity check (the model
and the layer math are right), not a result that advances the integer thesis. So it stays here,
labelled, instead of in the README's "what's proven."

## The integer-on-GPU directions — parked as negative-or-mixed

The interesting question was whether *integer* arithmetic can be competitive on a GPU, where
the hardware is built for FP16/BF16 tensor cores. The honest answer from these benchmarks is
"not in a way that pays for itself here":

| File | What it measures | Outcome |
|------|------------------|---------|
| `bench_gemm.cu` | FP16 tensor-core vs INT8 tensor-core vs INT64 on TinyLlama linear-layer shapes | Establishes the FP16/INT8 tensor-core ceiling the integer paths have to beat |
| `bench_int64.cu` | Optimized INT64 / exact-Q16.48 (`__int128` carry-chain) GEMM kernels, block-size + split-K sweep | INT64 has no tensor-core path; can't close the gap against FP16/INT8 at decode shapes |
| `bench_int8_compare.cu` | INT8 (per-tensor absmax quant) decode vs FP16 baseline vs CPU oracle | Quantization changes tokens — not the exact-integer guarantee this project is about |
| `bench_fp8_compare.cu` | FP8 E4M3 weights/activations vs FP16 vs CPU oracle | Same story as INT8: lossy, off-thesis |
| `bench_cudacore_fair.cu` | FP32 vs INT32 fixed-point with **identical** bandwidth/tiling (CUDA cores only, no tensor cores) | At M=1 decode it's bandwidth-bound, so the ~4× integer instruction cost barely shows — but there's no win either, just parity |
| `bench_cudacore_p13b.cu` | The same fair comparison swept across M | Checks whether a compute-bound gap appears as M grows |
| `bench_graph_fp16.cu` | CUDA-graph capture to remove per-step launch overhead (FP16) | Measures a launch-overhead ceiling; real autoregressive decode can't be graphed (pos/seq_len change each step) |
| `bench_layer_fp16.cu` | One transformer layer end-to-end on GPU vs float32 reference tensors (drift) | Layer-level correctness harness for the FP16 path |
| `llama_gpu.cuh` | The reusable FP16 GPU runtime (load, KV-cache, forward, oracle gate) used by the benches above | — |
| `llama_gpu_generate.cu` | Small generation driver on top of `llama_gpu.cuh` | — |

The pattern: the GPU is a **tensor-core machine**. FP16/BF16/INT8/FP8 get the dedicated
hardware; exact INT64 Q16.48 does not, so it falls back to CUDA cores at a multiple of the
instruction cost. The lossy integer formats (INT8/FP8) *can* use tensor cores but stop being
exact — which is the one property this project refuses to give up. So on this hardware, for
this model, there was no integer GPU path that was both exact and faster than just doing it
right on the CPU. CUDA-graph capture and the fair CUDA-core comparison were the attempts to
find a win at the margins; they found parity, not an advantage.

## The numbers

Measured on an RTX-class Blackwell GPU (CUDA 13), TinyLlama-1.1B decode, against the CPU
integer core as oracle. "Oracle match" = greedy tokens identical to the CPU run across four
prompts (france / story / math / life, 20 tokens each = 80).

**Reduced-precision formats vs the FP16 baseline:**

| Format | Oracle match | tok/s | vs FP16 | Verdict |
|--------|--------------|-------|---------|---------|
| **FP16** (baseline) | **80/80** | 333–334 | 1.00× | Correct, but *floating point* — off-thesis |
| INT8 (per-tensor absmax) | 29/80 (36%) | 220 | **0.66× (slower)** | Parked — lossy *and* slower |
| FP8 (E4M3) | 36/80 (45%) | 394 | 1.18× (faster) | Parked — lossy; only sub-FP16 format to beat FP16 |

INT8 is *both* worse and slower: per-call cuBLASLt descriptor overhead (~50–70 µs/matmul vs
FP16's ~8–12 µs) plus a synchronous D2H copy for the activation scale erase the arithmetic
saving at M=1 decode, while the coarse per-tensor scale crushes small weights and the error
compounds over 22 layers (first token diverges by position 3–6). FP8's speedup is almost
entirely `lm_head` (1.95×, half-size weights); per-layer GEMM shapes are flat or worse
(`down_proj` 0.62×), and 3 mantissa bits (≈1 decimal digit) is a format-fundamental quality
ceiling, not a tuning problem.

**The decisive test — does *integer arithmetic itself* buy anything? (INT32 Q16.16 vs FP32,
CUDA cores only, identical tiling/bandwidth, no tensor cores):**

| GEMM shape | FP32 | INT32 Q16.16 | Ratio |
|------------|------|--------------|-------|
| q/o_proj | 98.3 µs | 96.4 µs | 0.98× |
| kv_proj / gate_up | 94.2 µs | 94.2 µs | 1.00× |
| down_proj | 255.3 µs | 254.1 µs | 1.00× |

Performance is **identical** (max numeric drift 2–4 × 10⁻⁴, consistent with Q16.16's 16
fractional bits). The reason is the whole point: at **M=1 decode the GPU is
memory-bandwidth-bound**, so the ~4× instruction-count gap between an FP32 FMA (1 instruction)
and an INT32→INT64 fixed-point multiply (~4 instructions) is completely hidden behind DRAM
latency. Integer arithmetic offers **no speed advantage on this hardware** — the CPU's Q16.48
win comes from the *wider, exact format* (48 fractional bits, bit-reproducible), not from
"integer being faster." And the only way to go *faster* on a GPU is the tensor cores, which
serve exactly the lossy formats (FP16/FP8/INT8) that break the exact-token guarantee. That
square — exact needs CUDA cores, fast needs tensor cores, and tensor cores aren't exact — is
why the whole branch is parked.

## Status

Parked. The code builds and runs (per the header build lines) and the FP16 gate passes, but
none of it changes the project's conclusion, so it isn't maintained alongside the integer core
and isn't covered by the cross-machine determinism gate. Treat it as a lab notebook: reproducible
measurements behind a result that came out "no."

Apache-2.0 © 2026 Nenad Mićić. Same license as the rest of the repo.
