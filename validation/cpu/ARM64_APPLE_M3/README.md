# CPU test: Apple M3 MacBook Air — arm64 macOS, native clang

Status: **PASS** (TinyLlama final-source validation, 2026-07-30)

This is the arm64/clang leg of the native host matrix and the local reference
host for the tiny-GPT byte comparisons. The recorded run below exercises the
stronger TinyLlama gate with the same final source and the same
8,800,406,496-byte native Q16.48 model used on the Pi 5, AMD, and Intel hosts.

The run is CPU-only. It uses the dependency-free C `llama_int` executable,
whose only dynamic library is `/usr/lib/libSystem.B.dylib`. It does not use
Metal, Accelerate, Core ML, or any GPU API; the Apple GPU was not used.

| check | result | runtime (informational) |
|---|---|---|
| warning-free arm64/clang build (`-Werror`) | PASS | — |
| TinyLlama four-prompt oracle | **80/80 tokens** | 447.86 s wall |
| prefill | 22 tokens | 98.754 s |
| decode | 76 tokens | 331.262 s |
| native-stream layer reads | 102 forward calls | 410.205 s |
| modeled peak resident memory | 1,513 MiB | — |

Each prompt matched all 20 reference tokens. Generation took 430.016 seconds;
layer loading accounted for 93% of measured layer time, while layer compute
used 31.895 seconds.

## Access-mode note

The 16 GB laptop had less than 1 GiB of free filesystem space and was running
its normal desktop workload. An initial `--native` mmap attempt was stopped
after 3,001 seconds because only about 196 seconds were user+system CPU time:
the 8.2 GiB mapping was repeatedly paged from storage. That incomplete attempt
is not a correctness result and is not included in the timing table.

The successful run used `--native-stream`, the intended low-memory policy for
a constrained laptop. It reads one layer at a time and kept the modeled peak
at 1.51 GiB. The same `.mgw` file and `--native` mmap mode are appropriate on
the less-loaded Pi 5 and the 64 GB Linux hosts, where that path already passed
80/80. Timing here characterizes this storage state, not Apple M3 compute
throughput.

Recorded transcript:

- `results/2026-07-30-apple-m3-tinyllama.txt`

## Hardware / Config

- System: MacBook Air, Apple M3
- CPU: 8 cores (4 performance + 4 efficiency), arm64
- Memory: 16 GB unified memory
- OS/toolchain: macOS 26.5.2, Apple clang 21.0.0
- Build: `-O3 -march=native -fwrapv -std=c11`, warnings as errors
- Runtime mode: `--native-stream`, normal APFS filesystem, no RAM disk
- GPU: not used
- TinyLlama model SHA-256:
  `7e8218d7f79a784f9d1868140fb16c3b9f5fbc45c19fb5c807ddcba5b41e32a8`
