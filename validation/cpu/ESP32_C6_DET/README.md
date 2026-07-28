# HW test: ESP32-C6 devkit — fp_math.h determinism gate on RISC-V

Status: **PASS** (2026-07-28, tree `1b706ecf6c7f`)

Runs `fp_det_compute()` from `fp_determinism.c` (built with
`-DFP_DET_NO_MAIN`, portable two-limb backend — rv32imac has no `__int128`
and no FPU) and compares against the host pin, which prepare.sh verifies
against the committed golden `tests/determinism_golden.txt`.

| | hash | runtime (informational) |
|---|---|---|
| host reference (arm64 macOS, `__int128` backend) + committed golden | `c0d933ea340452ec` | — |
| **ESP32-C6 (RISC-V rv32imac @ 160 MHz)** | **`c0d933ea340452ec`** | 5631 ms |

Raw capture: `results/2026-07-28-esp32-c6-det.txt`.

## Hardware

- Board: ESP32-C6 devkit (DevKitC-1 form factor), chip revision v0.2, QFN40,
  the same physical board as astro-nav-int's
  `ESP32_C6` target
- SoC: ESP32-C6, 1× RV32IMAC @ 160 MHz + LP core, no FPU
- TWO serial ports for one board: `303A:1001` native USB-Serial/JTAG (do NOT
  capture here), `1A86:55D3` CH343 USB-UART bridge on UART0 (capture here)

## Config

- Platform: pioarduino `platform-espressif32` release 54.03.21-2 (pinned
  artifact; registry platform's Arduino core 2.x has no C6 support),
  Arduino core 3.x, `board = esp32-c6-devkitc-1`, `-O2 -fwrapv`
- `ARDUINO_USB_CDC_ON_BOOT=0` pins `Serial` to UART0 (see astro-nav-int's
  ESP32_C6 README for the native-CDC pitfalls)
- Image: 246,715 B flash (18.8% of the 1.3 MB app partition), 48,092 B
  static RAM (14.7% — dominated by the 34 KB input grid)
