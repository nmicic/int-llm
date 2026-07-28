# HW test: Seeed Studio XIAO RP2040 — microgpt_int INFERENCE on Armv6-M

Status: **PASS** (2026-07-28, tree `1b706ecf6c7f`)

The train-big/run-small loop, closed on a $5 microcontroller: the model was
trained on an arm64 Mac (`./gpt_int --save model.mgw`, 5000 steps, Q16.48
integer-only), the 115,576-byte `.mgw` image is baked into flash by
prepare.sh, and the firmware runs the full inference path
(`mgpt_load_mem()` + `mgpt_generate_sample()`, `-DMGPT_NO_TRAIN
-DMGPT_NO_MAIN`, portable two-limb backend) — reproducing the Mac training
run's 20 sampled names **byte-for-byte**, PRNG stream included.

| | samples hash | runtime (informational) |
|---|---|---|
| host reference (arm64 macOS, `__int128` backend, same entry points) | `ff4bc4bf7d4fd99d` | — |
| **XIAO RP2040 Cortex-M0+ (Armv6-M)** | **`ff4bc4bf7d4fd99d`** | 6724 ms (20 samples, ~336 ms/name) |

Samples: kayla, daia, lee, kayan, maha, kaia, ramiar, anall, ainale, kelel,
malana, arile, marion, avile, calan, kaylia, diaria, sarona, jahel, karin —
identical to the Mac's `./gpt_int --save` / `--load` output.

Raw capture: `results/2026-07-28-xiao-rp2040-gpt.txt` (capture attaches
mid-iteration, so the file starts at sample 3; the hash covers all 20).

## Memory

- Image: 187,932 B flash (9.0% of 2 MB) — includes the whole 115 KB weight
  file; 15,268 B static RAM (**5.8%** of 264 KB)
- The weights never occupy RAM: `mgpt_load_mem()` points the weight slots
  straight into the flash image and the M0+ reads them through XIP. RAM
  holds only the KV cache, activations, and the Arduino/USB stack.

## Hardware

- Board: Seeed Studio XIAO RP2040, RP2040 (2× Cortex-M0+ @ 133 MHz, no FPU),
  enumerates as `VID:PID 2E8A:000A`

## Config

- Platform: maxgerhardt `platform-raspberrypi` (arduino-pico core), pinned
  `#aa70b802be8851668053d4f09734e4089fe41932` (same as the astro-nav-int
  RP2040 targets), `board = seeed_xiao_rp2040`, `-O2 -fwrapv`
- Hash: FNV-1a 64 over each sample's characters + `'\n'`, in order —
  identical definition in `hostref/host_main.c` and `src/main.cpp`
