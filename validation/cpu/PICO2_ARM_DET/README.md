# HW test: Pico 2 (RP2350, ARM mode) — fp_math.h determinism gate

Status: **PASS** (2026-07-28, tree `8b16a00bb912-dirty`*)

Runs `fp_det_compute()` from `fp_determinism.c` (built with
`-DFP_DET_NO_MAIN`, portable two-limb backend — Cortex-M33 is 32-bit, no
`__int128`) and compares against the host pin, which prepare.sh verifies
against the committed golden `tests/determinism_golden.txt`.

| | hash | runtime (informational) |
|---|---|---|
| host reference (arm64 macOS, `__int128` backend) + committed golden | `c0d933ea340452ec` | — |
| XIAO RP2040 (Cortex-M0+ @ 133 MHz) | `c0d933ea340452ec` | 10851 ms |
| **Pico 2 (Cortex-M33 @ 150 MHz, ARM mode)** | **`c0d933ea340452ec`** | 4137 ms |

The same physical chip also passes in its RISC-V execution mode — see
`PICO2_RISCV_DET` (5448 ms on the Hazard3 cores).

\* dirty = the fp_math.h dual-backend vendoring + `FP_DET_NO_MAIN` split
were not yet committed when the test ran; the result file's firmware
SHA-256 pins the exact image.

Raw capture: `results/2026-07-28-pico2-arm-det.txt`.

## Hardware

- Board: Raspberry Pi Pico 2, VID:PID `2E8A:000F`,
  port `/dev/cu.usbmodem1101` — same physical board as
  astro-nav-int's `PI_PICO_RP2350` / `PI_PICO_RP2350_RISCV` targets
- SoC: RP2350, dual-ISA — 2× Cortex-M33 (Armv8-M, this target) *or*
  2× Hazard3 RISC-V (rv32imac), both @ 150 MHz; ARM mode has an FPU but
  the code path is pure integer

## Config

- Platform: maxgerhardt `platform-raspberrypi` pinned commit `aa70b802`
  (arduino-pico core; the registry mbed platform has no RP2350 support),
  `board = rpipico2`, `-O2 -fwrapv`
- Image: 61,656 B flash (1.5%), 44,784 B static RAM (8.5% — dominated by
  the 34 KB input grid)
