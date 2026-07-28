# HW test: ESP32-C6 devkit — microgpt_int INFERENCE on RISC-V

Status: **PASS** (2026-07-28, tree `8b16a00bb912-dirty`*)

Same train-big/run-small loop as `XIAO_RP2040_GPT`, now on RISC-V: the
model was trained on an arm64 Mac (`./gpt_int --save model.mgw`, 5000
steps, Q16.48 integer-only), the 115,576-byte `.mgw` image is baked into
the firmware by prepare.sh, and the C6 runs the full inference path
(`mgpt_load_mem()` + `mgpt_generate_sample()`, `-DMGPT_NO_TRAIN
-DMGPT_NO_MAIN`, portable two-limb backend) — reproducing the Mac training
run's 20 sampled names **byte-for-byte**, PRNG stream included.

| | samples hash | runtime (informational) |
|---|---|---|
| host reference (arm64 macOS, `__int128` backend, same entry points) | `ff4bc4bf7d4fd99d` | — |
| XIAO RP2040 (Cortex-M0+ @ 133 MHz) | `ff4bc4bf7d4fd99d` | 6717 ms |
| **ESP32-C6 (RISC-V rv32imac @ 160 MHz)** | **`ff4bc4bf7d4fd99d`** | 2047 ms (20 samples, ~100 ms/name) |

Samples: kayla, daia, lee, kayan, maha, kaia, ramiar, anall, ainale, kelel,
malana, arile, marion, avile, calan, kaylia, diaria, sarona, jahel, karin —
identical to the Mac's `./gpt_int --save` / `--load` output.

\* dirty = the fp_math.h dual-backend vendoring + .mgw save/load +
`MGPT_NO_MAIN`/`mgpt_load_mem` additions were not yet committed when the
test ran; the result file's firmware SHA-256 pins the exact image.

Raw capture: `results/2026-07-28-esp32-c6-gpt.txt` (capture attaches
mid-iteration; the hash covers all 20 samples).

## Memory

- Image: 361,919 B flash (27.6% of the 1.3 MB app partition) — includes the
  whole 115 KB weight file in rodata; 19,404 B static RAM (**5.9%**)
- The weights never occupy RAM: on ESP32 const data lives in the
  flash-mapped rodata segment and `mgpt_load_mem()` points the weight
  slots straight into it (reads go through the flash cache)

## Hardware

- Board: ESP32-C6 devkit (DevKitC-1 form factor), chip revision v0.2, QFN40,
  same physical board as astro-nav-int's
  `ESP32_C6` target
- SoC: ESP32-C6, 1× RV32IMAC @ 160 MHz + LP core, no FPU
- Capture on the CH343 UART bridge (`1A86:55D3`), not the native CDC

## Config

- Platform: pioarduino `platform-espressif32` release 54.03.21-2 (pinned
  artifact), Arduino core 3.x, `board = esp32-c6-devkitc-1`, `-O2 -fwrapv`,
  `ARDUINO_USB_CDC_ON_BOOT=0`
- Hash: FNV-1a 64 over each sample's characters + `'\n'`, in order —
  identical definition in `hostref/host_main.c` and `src/main.cpp`
