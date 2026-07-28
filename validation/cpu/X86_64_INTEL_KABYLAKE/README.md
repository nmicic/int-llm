# CPU test: x86-64 Intel (Kaby Lake) Linux server — native gcc

Status: **PASS** (2026-07-28, tree `1b706ecf6c7f`)

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

Additional check on this host (not in the transcript): `llama_int` with
TinyLlama-1.1B exported to a native 8.8 GB `.mgw`, `--native --generate
--max-new-tokens 16` over the four preset prompts, weights served from a
tmpfs ramdisk — the generated token streams are identical to the AMD
box's and unchanged from the pre-merge build (~1.0 tok/s here).

Raw capture: `results/2026-07-28-x86-intel-kabylake.txt`.

## Hardware / Config

- CPU: Intel Core i7-7700 @ 3.60 GHz (4 cores + HT, Kaby Lake), 62 GB RAM
- OS/toolchain: Ubuntu 24.04, gcc 13.3.0, `-O2 -fwrapv -std=c11`
- Both fp_math.h backends exercised: default picks native `__int128`,
  the portable two-limb build is forced with `-DFP_MATH_FORCE_PORTABLE`
