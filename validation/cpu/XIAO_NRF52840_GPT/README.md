# HW test: Seeed XIAO nRF52840 — microgpt_int INFERENCE on Cortex-M4F

Status: **PASS** (2026-07-29, tree `cf0bd4cc8589`)

Same train-big/run-small loop as the other `*_GPT` targets: Mac-trained
115,576-byte `.mgw` baked into flash rodata, firmware runs
`mgpt_load_mem()` + `mgpt_generate_sample()` (`-DMGPT_NO_TRAIN
-DMGPT_NO_MAIN`, portable two-limb backend) and reproduces the Mac's 20
sampled names **byte-for-byte**, PRNG stream included.

| | samples hash | runtime (informational) |
|---|---|---|
| host reference (arm64 macOS, `__int128` backend, same entry points) | `ff4bc4bf7d4fd99d` | — |
| XIAO RP2040 (Cortex-M0+ @ 133 MHz) | `ff4bc4bf7d4fd99d` | 6724 ms |
| **XIAO nRF52840 (Cortex-M4F @ 64 MHz)** | **`ff4bc4bf7d4fd99d`** | 3424 ms (20 samples, ~171 ms/name, ~36 tok/s) |

The 64 MHz M4F beats the 133 MHz M0+ ~2× — the M4's single-cycle
32×32→64 multiplier dominates this Q16.48 workload.

Samples: kayla, daia, lee, kayan, maha, kaia, ramiar, anall, ainale, kelel,
malana, arile, marion, avile, calan, kaylia, diaria, sarona, jahel, karin —
identical to the Mac's `./gpt_int --save` / `--load` output.

Raw capture: `results/2026-07-29-xiao-nrf52840-gpt.txt`.

## The crash this target exposed (fixed in microgpt_int.c)

First flash boot-looped with USB dead (board invisible to the host, only
a double-tap RST into the bootloader recovers it). Cause:
`inference_forward()` kept ~4 KB of work arrays on the stack, and the
Adafruit nRF52 core runs `loop()` in a FreeRTOS task with a hardcoded
4 KB stack (`LOOP_STACK_SZ` in the core's main.cpp — no build-flag
override). The overflow corrupted the RTOS. Fix: the work arrays are now
`static` in `microgpt_int.c` (each is fully rewritten before use per
call; nothing reentrant) — host output verified byte-identical after the
change. Every other board had given `loop()` ≥8 KB, which is why only
the Nordic core caught it.

## Memory

- Image: 179,728 B flash (22.2%) — includes the whole 115 KB weight file
  in rodata; 17,856 B static RAM (**7.5%**)
- Weights never occupy RAM: `mgpt_load_mem()` points the weight slots
  straight into the memory-mapped flash array

## Hardware / Config

Same board and platform as `XIAO_NRF52840_DET` (nRF52840 Cortex-M4F
@ 64 MHz, app `2886:8044` / bootloader `2886:0045`;
maxgerhardt nordicnrf52 `#cac6fcf9`,
`board = xiaoble_adafruit`, `-O2 -fwrapv -DUSE_TINYUSB`). Flashing quirks
in that README. Hash: FNV-1a 64 over each sample's characters + `'\n'`,
in order — identical definition in `hostref/host_main.c` and
`src/main.cpp`.
