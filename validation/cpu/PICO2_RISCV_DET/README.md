# HW test: Pico 2 (RP2350, RISC-V mode) — fp_math.h determinism gate

Status: **PASS** (2026-07-28, tree `1b706ecf6c7f`)

Same physical board as `PICO2_ARM_DET`, switched to the RP2350's Hazard3
RISC-V execution mode (`board_build.mcu = rp2350-riscv`). Runs
`fp_det_compute()` on the portable two-limb backend (rv32imac: no
`__int128`, no FPU) and compares against the host pin, which prepare.sh
verifies against the committed golden `tests/determinism_golden.txt`.

| | hash | runtime (informational) |
|---|---|---|
| host reference (arm64 macOS, `__int128` backend) + committed golden | `c0d933ea340452ec` | — |
| Pico 2 same chip, ARM mode (Cortex-M33) | `c0d933ea340452ec` | 4137 ms |
| **Pico 2 (Hazard3 rv32imac @ 150 MHz, RISC-V mode)** | **`c0d933ea340452ec`** | 5448 ms |

One silicon die, two ISAs, one hash — the ARM/RISC-V pair on identical
transistors is about as clean a cross-ISA determinism check as it gets.

Raw capture: `results/2026-07-28-pico2-riscv-det.txt`.

## Hardware

- Board: Raspberry Pi Pico 2, VID:PID `2E8A:000F`
  — same physical board as the other `PICO2_*` targets
- SoC: RP2350, 2× Hazard3 RISC-V (rv32imac) @ 150 MHz in this mode, no FPU

## Config

- Platform: maxgerhardt `platform-raspberrypi` pinned commit `aa70b802`
  (arduino-pico core), `board = rpipico2`,
  `board_build.mcu = rp2350-riscv`, `-O2 -fwrapv`
- Image: 81,368 B flash (1.9%), 56,052 B static RAM (10.7%)
- After flashing across an ISA-mode switch the CDC port can take several
  seconds to re-enumerate; capture_serial.py's port retries cover it (see
  astro-nav-int's `PI_PICO_RP2350_RISCV` README)
