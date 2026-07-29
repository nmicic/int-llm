# HW test: Arduino MKR Zero — microgpt_int INFERENCE on Armv6-M (SAMD21)

Status: **PASS** (2026-07-29, tree `cf0bd4cc8589`)

The smallest-flash target in the fleet, and the literal check of the "code
plus weights fit a 256 KB-flash MCU" claim: the model was trained on an
arm64 Mac (`./gpt_int --save model.mgw`, 5000 steps, Q16.48 integer-only),
the 115,576-byte `.mgw` image is baked into the SAMD21's 256 KB internal
flash by prepare.sh, and the firmware runs the full inference path
(`mgpt_load_mem()` + `mgpt_generate_sample()`, `-DMGPT_NO_TRAIN
-DMGPT_NO_MAIN`, portable two-limb backend) — reproducing the Mac training
run's 20 sampled names **byte-for-byte**, PRNG stream included.

| | samples hash | runtime (informational) |
|---|---|---|
| host reference (arm64 macOS, `__int128` backend, same entry points) | `ff4bc4bf7d4fd99d` | — |
| **Arduino MKR Zero Cortex-M0+ (Armv6-M)** | **`ff4bc4bf7d4fd99d`** | 26377 ms (20 samples, ~1319 ms/name) |

Samples: kayla, daia, lee, kayan, maha, kaia, ramiar, anall, ainale, kelel,
malana, arile, marion, avile, calan, kaylia, diaria, sarona, jahel, karin —
identical to the Mac's `./gpt_int --save` / `--load` output.

Raw capture: `results/2026-07-29-mkrzero-samd21-gpt.txt`.

## Memory

- Image: 133,472 B flash (**50.9%** of the 256 KB part) — includes the whole
  115 KB weight file; 13,124 B static RAM (40.1% of 32 KB)
- The weights never occupy RAM: `mgpt_load_mem()` points the weight slots
  straight into the flash image, read in place from the SAMD21's internal
  memory-mapped flash. RAM holds only the KV cache, activations, and the
  Arduino core.

## Hardware

- Board: Arduino MKR Zero, Microchip SAMD21G18A (Cortex-M0+ @ 48 MHz, no
  FPU), 256 KB internal flash / 32 KB SRAM, enumerates as `VID:PID 2341:804F`

## Config

- Platform: registry `atmelsam` pinned `@ 8.3.0` (Arduino SAMD core),
  `board = mkrzero`, `-O2 -fwrapv`
- Hash: FNV-1a 64 over each sample's characters + `'\n'`, in order —
  identical definition in `hostref/host_main.c` and `src/main.cpp`
