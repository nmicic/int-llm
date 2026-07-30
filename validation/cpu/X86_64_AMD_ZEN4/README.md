# CPU test: x86-64 AMD (Zen 4) Linux server — native gcc

Status: **PASS** (native suite 2026-07-28; TinyLlama final-source
revalidation 2026-07-30)

Native-Linux target, no PlatformIO: `../native_check.sh` copies the sources
plus the committed `model.mgw` over ssh, builds with the distro gcc, and
byte-compares everything against the local (arm64 macOS) host — see
`HOWTO.md` §6. This is the cross-*vendor* leg of the x86 pair (compare
`X86_64_INTEL_KABYLAKE`): same distro, same gcc, different
microarchitecture.

| check | result | runtime (informational) |
|---|---|---|
| determinism, default backend (`__int128`) | `c0d933ea340452ec` (= golden) | 5 ms |
| determinism, `-DFP_MATH_FORCE_PORTABLE` | `c0d933ea340452ec` (= golden) | 19 ms |
| `--load model.mgw` inference, 20 samples | byte-identical to macOS host | 2 ms |
| full training stdout | byte-identical to macOS host | 2095 ms |
| `--save` after training | `.mgw` byte-identical to the committed `model.mgw` | — |

So an AMD/gcc/Linux box, an Intel/gcc/Linux box, an arm64/clang/macOS
laptop, and a 32-bit ARMv6 Pi all *train* to the identical weight file —
not just infer identically.

The final `llama_int.c` source was revalidated after the Pi 5 portability
fixes. TinyLlama-1.1B, exported to the 8,800,406,496-byte native `.mgw`,
ran from tmpfs with `--native --benchmark`: all four prompts matched the
float-reference oracle, **80/80 tokens**. The run took 51.8 seconds
wall-clock, about 1.55–1.66 generated token/s per prompt.

Recorded transcripts:

- `results/2026-07-28-x86-amd-zen4.txt`
- `results/2026-07-30-x86-amd-zen4-tinyllama.txt`

## Hardware / Config

- CPU: AMD Ryzen 7 7700 (8 cores, Zen 4), 61 GB RAM
- OS/toolchain: Ubuntu 24.04, gcc 13.3.0, `-O2 -fwrapv -std=c11`
- Both fp_math.h backends exercised: default picks native `__int128`,
  the portable two-limb build is forced with `-DFP_MATH_FORCE_PORTABLE`
