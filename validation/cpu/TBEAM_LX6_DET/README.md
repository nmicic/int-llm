# HW test: LILYGO T-Beam — fp_math.h determinism gate on Xtensa LX6

Status: **PASS** (2026-07-29, tree `cf0bd4cc8589`)

Second Xtensa target, but a different core generation than the Heltec V3:
the original dual-core LX6 (ESP32-D0WDQ6-V3) vs the S3's LX7. Runs
`fp_det_compute()` from `fp_determinism.c` (built with
`-DFP_DET_NO_MAIN`, portable two-limb backend) and compares against the
host pin, which prepare.sh verifies against the committed golden
`tests/determinism_golden.txt`.

| | hash | runtime (informational) |
|---|---|---|
| host reference (arm64 macOS, `__int128` backend) + committed golden | `c0d933ea340452ec` | — |
| Heltec V3 ESP32-S3 (Xtensa LX7 @ 240 MHz) | `c0d933ea340452ec` | 3883 ms |
| **T-Beam ESP32-D0WDQ6-V3 (Xtensa LX6 @ 240 MHz)** | **`c0d933ea340452ec`** | 4237 ms |

Raw capture: `results/2026-07-29-tbeam-lx6-det.txt`.

## Serial

Classic ESP32 has no native USB: the board's USB-C is a WCH CH9102
USB-UART bridge (`1A86:55D4`, `/dev/cu.usbserial-<REDACTED>`) to UART0,
so `Serial` needs no CDC flag — but the port name is `usbserial-*`, which
capture auto-detect does not match, so pass it explicitly. Flashing needs
no button: esptool auto-resets over the bridge's handshake lines and
works over any running firmware (including Meshtastic).

## Hardware

- Board: LILYGO T-Beam (normally runs Meshtastic — LoRa radio and GPS
  unused here; Meshtastic reflashed afterwards) — same
  physical board as astro-nav-int's `MESHTASTIC_ESP32_LX6` target
- SoC: ESP32-D0WDQ6-V3 (rev v3.1), dual-core Xtensa LX6 @ 240 MHz,
  4 MB flash / 320 KB SRAM

## Config

- Platform: pioarduino `platform-espressif32` release 54.03.21-2 (same
  pinned zip as the other Espressif targets), `board = esp32dev` (generic
  devkit definition — only UART0 and the CPU are used), `-O2 -fwrapv`
- Image: 306,302 B flash (23.4% of the 1.3 MB app partition), 55,920 B
  static RAM (17.1% — dominated by the 34 KB input grid)
