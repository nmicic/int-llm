# HW test: Heltec WiFi LoRa 32 V3 — microgpt_int INFERENCE on Xtensa

Status: **PASS** (2026-07-29, tree `cf0bd4cc8589`)

Same train-big/run-small loop as the other `*_GPT` targets, now on the
third ISA family: the model was trained on an arm64 Mac (`./gpt_int
--save model.mgw`, 5000 steps, Q16.48 integer-only), the 115,576-byte
`.mgw` image is baked into flash-mapped rodata by prepare.sh, and the S3
runs the full inference path (`mgpt_load_mem()` +
`mgpt_generate_sample()`, `-DMGPT_NO_TRAIN -DMGPT_NO_MAIN`, portable
two-limb backend) — reproducing the Mac training run's 20 sampled names
**byte-for-byte**, PRNG stream included. Fastest MCU in the fleet:

| | samples hash | runtime (informational) |
|---|---|---|
| host reference (arm64 macOS, `__int128` backend, same entry points) | `ff4bc4bf7d4fd99d` | — |
| ESP32-C6 (RISC-V rv32imac @ 160 MHz), previous fastest | `ff4bc4bf7d4fd99d` | 2047 ms |
| **Heltec V3 (ESP32-S3 Xtensa LX7 @ 240 MHz)** | **`ff4bc4bf7d4fd99d`** | 1064 ms (20 samples, ~53 ms/name) |

Samples: kayla, daia, lee, kayan, maha, kaia, ramiar, anall, ainale, kelel,
malana, arile, marion, avile, calan, kaylia, diaria, sarona, jahel, karin —
identical to the Mac's `./gpt_int --save` / `--load` output.

Raw capture: `results/2026-07-29-heltec-s3-gpt.txt`.

## Memory

- Image: 426,062 B flash (12.7%) — includes the whole 115 KB weight file
  in rodata; 26,464 B static RAM (**8.1%**)
- The weights never occupy RAM: on ESP32 const data lives in the
  flash-mapped rodata segment and `mgpt_load_mem()` points the weight
  slots straight into it (reads go through the flash cache)

## Serial quirk

USB-C → CP2102 UART bridge (`10C4:EA60`, `/dev/cu.usbserial-0001`) →
UART0; the S3's native USB is not on the connector. `Serial` pinned to
UART0 via `ARDUINO_USB_CDC_ON_BOOT=0`; pass the port explicitly.

## Hardware

- Board: Heltec WiFi LoRa 32 V3 (SX1262 LoRa + OLED, both unused) —
  same physical board as astro-nav-int's `HELTEC_WIFI_LORA_32_V3` target
- SoC: ESP32-S3, dual-core Xtensa LX7 @ 240 MHz, 8 MB flash / 320 KB SRAM

## Config

- Platform: pioarduino `platform-espressif32` release 54.03.21-2 (same
  pinned zip as the ESP32_C6 targets), Arduino core 3.x,
  `board = heltec_wifi_lora_32_V3`, `-O2 -fwrapv`
- Hash: FNV-1a 64 over each sample's characters + `'\n'`, in order —
  identical definition in `hostref/host_main.c` and `src/main.cpp`
