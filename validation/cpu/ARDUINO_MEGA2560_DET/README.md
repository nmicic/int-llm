# HW test: Arduino Mega 2560 — fp_math.h determinism gate on 8-bit AVR

Status: **PASS** (2026-07-29, tree `cf0bd4cc8589`)

The oldest, smallest-RAM, and only 8-bit target in the fleet — and a fourth
ISA family (AVR, after ARM Cortex-M, RISC-V, and Xtensa). Runs
`fp_det_compute()` from `fp_determinism.c` (built with `-DFP_DET_NO_MAIN`,
portable two-limb backend — every 64-bit operation is synthesized from
8-bit ALU instructions) and compares against the host pin, which prepare.sh
verifies against the committed golden `tests/determinism_golden.txt`.

Two properties of the ATmega2560 shape this harness, both handled by the
`FP_DET_EXTERNAL_GRID` mode:

- **8 KB SRAM** cannot hold the ~34 KB input grid, and the AVR's Harvard
  architecture means even a `const` array is copied into RAM at startup —
  so the grid is placed in the `__flash` address space (read via LPM
  instructions straight from program memory).
- **16-bit `size_t`** caps a single object at 32,767 bytes, so the
  generated table (`include/fp_det_grid.h`, emitted by the same
  `fp_determinism.c` built natively with `-DFP_DET_GRID_EMIT`) is two
  ~17 KB halves behind the `FP_DET_GRID_AT(i)` accessor.

The host reference is built in the same external-grid mode against the
committed golden, and the generated header's sha256 is pinned in the
transcript.

| | hash | runtime (informational) |
|---|---|---|
| host reference (arm64 macOS, `__int128` backend) + committed golden | `c0d933ea340452ec` | — |
| **Arduino Mega 2560 ATmega2560 (8-bit AVR)** | **`c0d933ea340452ec`** | 747901 ms (~12.5 min) |

Raw capture: `results/2026-07-29-mega2560-avr-det.txt`.

The GPT inference harness is not feasible on this part: microgpt's KV
cache + activations need ~13 KB of RAM against the ATmega2560's 8 KB
(and the 115 KB `.mgw` image would need far-flash accessors on top).
Determinism-only, by hardware limit.

## Hardware

- Board: Arduino Mega 2560, Microchip/Atmel ATmega2560 (8-bit AVR @
  16 MHz, Harvard architecture), 256 KB flash (248 KB usable) / 8 KB SRAM,
  enumerates as `VID:PID 2341:0010`

## Config

- Platform: registry `atmelavr` pinned `@ 5.3.0` (avr-gcc 7.3.0, Arduino
  AVR core), `board = megaatmega2560`, `-O2 -fwrapv`
- Image: 61,732 B flash (24.3% — includes the 33 KB grid in program
  memory), 800 B static RAM (**9.8%** of 8 KB)
