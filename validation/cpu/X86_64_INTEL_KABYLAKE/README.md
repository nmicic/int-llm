# CPU test: x86-64 Intel (Kaby Lake) Linux server — native gcc

Status: **PASS** (native suite 2026-07-28; TinyLlama final-source
revalidation 2026-07-30)

Native-Linux target, no PlatformIO: `../native_check.sh` copies the sources
plus the committed `model.mgw` over ssh, builds with the distro gcc, and
byte-compares everything against the local (arm64 macOS) host — see
`HOWTO.md` §6. This is the cross-*vendor* leg of the x86 pair (compare
`X86_64_AMD_ZEN4`): same distro, same gcc, different microarchitecture.

| check | result | runtime (informational) |
|---|---|---|
| determinism, default backend (`__int128`) | `c0d933ea340452ec` (= golden) | 8 ms |
| determinism, `-DFP_MATH_FORCE_PORTABLE` | `c0d933ea340452ec` (= golden) | 31 ms |
| `--load model.mgw` inference, 20 samples | byte-identical to macOS host | 3 ms |
| full training stdout | byte-identical to macOS host | 4441 ms |
| `--save` after training | `.mgw` byte-identical to the committed `model.mgw` | — |

The final `llama_int.c` source was revalidated after the Pi 5 portability
fixes. TinyLlama-1.1B, exported to the 8,800,406,496-byte native `.mgw`,
ran from tmpfs with `--native --benchmark`: all four prompts matched the
float-reference oracle, **80/80 tokens**. The run took 76.7 seconds
wall-clock, about 1.05–1.12 generated token/s per prompt.

Recorded transcripts:

- `results/2026-07-28-x86-intel-kabylake.txt`
- `results/2026-07-30-x86-intel-kabylake-tinyllama.txt`

## Hardware / Config

- CPU: Intel Core i7-7700 @ 3.60 GHz (4 cores + HT, Kaby Lake), 62 GB RAM
- OS/toolchain: Ubuntu 24.04, gcc 13.3.0, `-O2 -fwrapv -std=c11`
- Both fp_math.h backends exercised: default picks native `__int128`,
  the portable two-limb build is forced with `-DFP_MATH_FORCE_PORTABLE`
