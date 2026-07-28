# CPU test: Raspberry Pi 1 Model B+ — 32-bit ARMv6 Linux, native gcc

Status: **PASS** (2026-07-28, tree `1b706ecf6c7f`)

The oldest and smallest *Linux* target: a 2014 single-core ARMv6 board
whose 32-bit userland gcc has **no `__int128`** — so unlike the x86
records, both determinism builds here (default and
`-DFP_MATH_FORCE_PORTABLE`) resolve to the portable two-limb backend;
the near-identical runtimes confirm it. Run via `../native_check.sh`
(sources + committed `model.mgw` over ssh, built with the distro gcc,
byte-compared against the arm64 macOS host — see `HOWTO.md` §6).

| check | result | runtime (informational) |
|---|---|---|
| determinism, default backend (= portable on ARMv6) | `c0d933ea340452ec` (= golden) | 746 ms |
| determinism, `-DFP_MATH_FORCE_PORTABLE` | `c0d933ea340452ec` (= golden) | 744 ms |
| `--load model.mgw` inference, 20 samples | byte-identical to macOS host | 229 ms (~11 ms/name, ~530 tok/s) |
| full 5000-step training, stdout | byte-identical to macOS host | 594,056 ms (~0.12 s/step) |
| full 5000-step training, saved `.mgw` | byte-identical to committed `model.mgw` | — |

Full training reproduces the 64-bit hosts byte-for-byte on this board,
recorded twice: the `native_check.sh --train` transcript above (`train:
PASS` + `train-save: PASS` against the exact committed sources), and an
independent operator-run timestamped 5000-step run excerpted in
`results/2026-07-28-pi1-bplus-train-ts.txt` (start/end timestamps,
source + model SHA-256s, `cmp` verdict, all 20 samples; ~0.26 s/step
there — the per-line timestamping pipeline roughly doubles the wall
clock on this single 700 MHz core). The committed `model.mgw` is itself
reproduced by training on arm64 macOS, x86-64 AMD, and x86-64 Intel —
this board makes it four hosts.

Raw capture: `results/2026-07-28-pi1-bplus.txt`.

## Hardware / Config

- Board: Raspberry Pi 1 Model B+ (BCM2835, ARM1176JZF-S @ 700 MHz,
  512 MB RAM), Raspbian, armv6l userland
- Toolchain: distro gcc 14.2, `-O2 -fwrapv -std=c11` — no cross
  toolchain, no PlatformIO
