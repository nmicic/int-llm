# HW test: Arduino MKR Zero — fp_math.h determinism gate on Armv6-M (SAMD21)

Status: **PASS** (2026-07-29, tree `cf0bd4cc8589`)

Runs `fp_det_compute()` from `fp_determinism.c` (built with
`-DFP_DET_NO_MAIN`, portable two-limb backend — the SAMD21's Cortex-M0+
has no `__int128`) and compares against the host pin, which prepare.sh
verifies against the committed golden `tests/determinism_golden.txt`.

This is the first target whose SRAM cannot hold the determinism input grid:
the ~34 KB `fixed_t` array is larger than the SAMD21's entire 32 KB RAM.
The harness builds with `-DFP_DET_EXTERNAL_GRID`: prepare.sh compiles the
same copied `fp_determinism.c` natively with `-DFP_DET_GRID_EMIT`, which
prints the grid as a const table (`include/fp_det_grid.h`), and the firmware
reads it in place from memory-mapped flash. Emitter and consumer are one
source file, the generated header's sha256 is pinned in the transcript, and
the host reference is built in the same external-grid mode against the
committed golden — so the flash-resident grid is checked end-to-end, not
assumed. Verified natively first: default, emit, and external-grid builds
all reproduce `c0d933ea340452ec` on the host.

| | hash | runtime (informational) |
|---|---|---|
| host reference (arm64 macOS, `__int128` backend) + committed golden | `c0d933ea340452ec` | — |
| **Arduino MKR Zero Cortex-M0+ (Armv6-M)** | **`c0d933ea340452ec`** | 37249 ms |

Raw capture: `results/2026-07-29-mkrzero-samd21-det.txt`.

## Hardware

- Board: Arduino MKR Zero, Microchip SAMD21G18A (Cortex-M0+ @ 48 MHz, no
  FPU), 256 KB internal flash / 32 KB SRAM, enumerates as `VID:PID 2341:804F`

## Config

- Platform: registry `atmelsam` pinned `@ 8.3.0` (Arduino SAMD core),
  `board = mkrzero`, `-O2 -fwrapv`
- Image: 51,764 B flash (19.7% — includes the 33 KB const grid), 2,816 B
  static RAM (**8.6%** — the smallest RAM footprint of any DET target: the
  grid that dominates RAM elsewhere lives in flash here)
