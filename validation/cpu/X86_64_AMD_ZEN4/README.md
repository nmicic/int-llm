# CPU test: x86-64 AMD (Zen 4) Linux server — native gcc

Status: **PASS** (2026-07-28, tree `1b706ecf6c7f`)

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

Additional check on this host (not in the transcript): `llama_int` with
TinyLlama-1.1B exported to a native 8.8 GB `.mgw`, `--native --generate
--max-new-tokens 16` over the four preset prompts, weights served from a
tmpfs ramdisk — the generated token streams are identical to the Intel
box's and unchanged from the pre-merge build (~1.5 tok/s here).

Raw capture: `results/2026-07-28-x86-amd-zen4.txt`.

## Hardware / Config

- CPU: AMD Ryzen 7 7700 (8 cores, Zen 4), 61 GB RAM
- OS/toolchain: Ubuntu 24.04, gcc 13.3.0, `-O2 -fwrapv -std=c11`
- Both fp_math.h backends exercised: default picks native `__int128`,
  the portable two-limb build is forced with `-DFP_MATH_FORCE_PORTABLE`
