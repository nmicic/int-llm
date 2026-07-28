# HW test: Seeed Studio XIAO RP2040 — fp_math.h determinism gate on Armv6-M

Status: **PASS** (2026-07-28, tree `8b16a00bb912-dirty`*)

Runs `fp_det_compute()` from `fp_determinism.c` (built with
`-DFP_DET_NO_MAIN`, portable two-limb backend — the RP2040's Cortex-M0+
has no `__int128`) and compares against the host pin, which prepare.sh
verifies against the committed golden `tests/determinism_golden.txt`.

| | hash | runtime (informational) |
|---|---|---|
| host reference (arm64 macOS, `__int128` backend) + committed golden | `c0d933ea340452ec` | — |
| **XIAO RP2040 Cortex-M0+ (Armv6-M)** | **`c0d933ea340452ec`** | 10851 ms |

\* dirty = the fp_math.h dual-backend vendoring + `FP_DET_NO_MAIN` split
were not yet committed when the test ran; the result file's firmware
SHA-256 pins the exact image.

Raw capture: `results/2026-07-28-xiao-rp2040-det.txt`.

## Hardware

- Board: Seeed Studio XIAO RP2040, RP2040 (2× Cortex-M0+ @ 133 MHz, no FPU),
  enumerates as `VID:PID 2E8A:000A`

## Config

- Platform: maxgerhardt `platform-raspberrypi` (arduino-pico core), pinned
  `#aa70b802be8851668053d4f09734e4089fe41932` (same as the astro-nav-int
  RP2040 targets), `board = seeed_xiao_rp2040`, `-O2 -fwrapv`
- Image: 63,332 B flash (3.0%), 43,960 B static RAM (16.8% — dominated by
  the 34 KB input grid)
