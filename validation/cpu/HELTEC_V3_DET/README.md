# HW test: Heltec WiFi LoRa 32 V3 — fp_math.h determinism gate on Xtensa

Status: **PASS** (2026-07-28, tree `1b706ecf6c7f`)

First Xtensa target — the third ISA family (ARM, RISC-V, Xtensa) to
reproduce the golden hash in this repo. Runs `fp_det_compute()` from
`fp_determinism.c` (built with `-DFP_DET_NO_MAIN`, portable two-limb
backend — LX7 is 32-bit, no `__int128`) and compares against the host
pin, which prepare.sh verifies against the committed golden
`tests/determinism_golden.txt`.

| | hash | runtime (informational) |
|---|---|---|
| host reference (arm64 macOS, `__int128` backend) + committed golden | `c0d933ea340452ec` | — |
| Pico 2 ARM mode (Cortex-M33 @ 150 MHz), previous fastest | `c0d933ea340452ec` | 4137 ms |
| **Heltec V3 (ESP32-S3 Xtensa LX7 @ 240 MHz)** | **`c0d933ea340452ec`** | 3883 ms |

Raw capture: `results/2026-07-28-heltec-s3-det.txt`.

## Serial quirk

The board's USB-C goes through a CP2102 USB-UART bridge (`10C4:EA60`,
`/dev/cu.usbserial-0001`) to UART0 — the S3's native USB is NOT wired to
the connector, so there is only ONE port. `ARDUINO_USB_CDC_ON_BOOT=0`
keeps `Serial` on UART0, and the port must be passed to run_test.sh
explicitly (capture auto-detect only matches `usbmodem*`). See
astro-nav-int's `HELTEC_WIFI_LORA_32_V3` README.

## Hardware

- Board: Heltec WiFi LoRa 32 V3 (SX1262 LoRa + OLED, both unused) —
  same physical board as astro-nav-int's `HELTEC_WIFI_LORA_32_V3` target
- SoC: ESP32-S3, dual-core Xtensa LX7 @ 240 MHz, 8 MB flash / 320 KB SRAM

## Config

- Platform: pioarduino `platform-espressif32` release 54.03.21-2 (same
  pinned zip as the ESP32_C6 targets), Arduino core 3.x,
  `board = heltec_wifi_lora_32_V3`, `-O2 -fwrapv`
- Image: 310,930 B flash (9.3%), 55,168 B static RAM (16.8% — dominated
  by the 34 KB input grid)
