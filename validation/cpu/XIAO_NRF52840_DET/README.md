# HW test: Seeed XIAO nRF52840 — fp_math.h determinism gate on Cortex-M4F

Status: **PASS** (2026-07-28, tree `1b706ecf6c7f`)

First Nordic silicon and first Cortex-M4F target in this repo — an
FPU-capable core running the FPU-less code: bit-equality with the golden
is the arbiter that no float contaminates the integer results. Runs
`fp_det_compute()` (built with `-DFP_DET_NO_MAIN`, portable two-limb
backend) against the host pin, which prepare.sh verifies against the
committed golden `tests/determinism_golden.txt`.

| | hash | runtime (informational) |
|---|---|---|
| host reference (arm64 macOS, `__int128` backend) + committed golden | `c0d933ea340452ec` | — |
| XIAO RP2040 (Cortex-M0+ @ 133 MHz) | `c0d933ea340452ec` | 10851 ms |
| **XIAO nRF52840 (Cortex-M4F @ 64 MHz)** | **`c0d933ea340452ec`** | 13883 ms |

Raw capture: `results/2026-07-28-xiao-nrf52840-det.txt`.

## Flashing (bootloader quirk — see astro-nav-int's XIAO_NRF52840_SENSE)

Factory/crashed firmware ignores the 1200-baud DFU touch, and PlatformIO
banners SUCCESS even when adafruit-nrfutil fails — trust only the
"Device programmed. Activating new firmware." lines. Operator double-taps
RST (mouse-double-click rhythm) → bootloader enumerates as `2886:0045`
→ `pio run -t upload` / run_test.sh does a real serial DFU. Once a
healthy TinyUSB firmware is on, re-flashing needs no button.

## Hardware

- Board: Seeed Studio XIAO nRF52840
  (application mode enumerates `2886:8044`, bootloader `2886:0045`) —
  NOT the same physical unit as astro-nav-int's XIAO_NRF52840_SENSE
- SoC: Nordic nRF52840, Cortex-M4F @ 64 MHz, 1 MB flash / 256 KB RAM,
  Adafruit UF2/DFU bootloader

## Config

- Platform: maxgerhardt `platform-nordicnrf52` fork pinned `#cac6fcf9`
  (registry platform lacks the board), Adafruit nRF52 core (Seeed
  variant), `board = xiaoble_adafruit`, `-O2 -fwrapv`
- `-DUSE_TINYUSB` required (`Serial` lives in TinyUSB; link fails
  without it); `main.cpp` guards `#include <Adafruit_TinyUSB.h>`
- Image: 65,504 B flash (8.1%), 42,380 B static RAM (17.8% — dominated
  by the 34 KB input grid)
