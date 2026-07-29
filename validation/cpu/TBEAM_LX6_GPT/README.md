# HW test: LILYGO T-Beam — microgpt_int INFERENCE on Xtensa LX6

Status: **PASS** (2026-07-29, tree `cf0bd4cc8589`)

Same train-big/run-small loop as the other `*_GPT` targets, on the
original ESP32's dual-core LX6 (vs the Heltec V3's newer LX7): the model
was trained on an arm64 Mac (`./gpt_int --save model.mgw`, 5000 steps,
Q16.48 integer-only), the 115,576-byte `.mgw` image is baked into
flash-mapped rodata by prepare.sh, and the firmware runs the full
inference path (`mgpt_load_mem()` + `mgpt_generate_sample()`,
`-DMGPT_NO_TRAIN -DMGPT_NO_MAIN`, portable two-limb backend) —
reproducing the Mac training run's 20 sampled names **byte-for-byte**,
PRNG stream included.

| | samples hash | runtime (informational) |
|---|---|---|
| host reference (arm64 macOS, `__int128` backend, same entry points) | `ff4bc4bf7d4fd99d` | — |
| Heltec V3 ESP32-S3 (Xtensa LX7 @ 240 MHz) | `ff4bc4bf7d4fd99d` | 1064 ms |
| ESP32-C6 (RISC-V rv32imac @ 160 MHz) | `ff4bc4bf7d4fd99d` | 2047 ms |
| **T-Beam ESP32-D0WDQ6-V3 (Xtensa LX6 @ 240 MHz)** | **`ff4bc4bf7d4fd99d`** | 2591 ms (20 samples, ~130 ms/name, ~47 tok/s) |

Samples: kayla, daia, lee, kayan, maha, kaia, ramiar, anall, ainale, kelel,
malana, arile, marion, avile, calan, kaylia, diaria, sarona, jahel, karin —
identical to the Mac's `./gpt_int --save` / `--load` output. Note the LX6
@ 240 MHz is ~2.4× slower than the LX7 at the same clock and also behind
the 160 MHz C6 — the older core pays more for 64-bit multiply and flash
cache misses.

Raw capture: `results/2026-07-29-tbeam-lx6-gpt.txt`.

## Memory

- Image: 421,390 B flash (32.1% of the 1.3 MB app partition) — includes
  the whole 115 KB weight file in rodata; 27,216 B static RAM (**8.3%**)
- The weights never occupy RAM: on ESP32 const data lives in the
  flash-mapped rodata segment and `mgpt_load_mem()` points the weight
  slots straight into it (reads go through the flash cache)

## Serial / Hardware / Config

Same as `TBEAM_LX6_DET`: CH9102 bridge (`1A86:55D4`,
`/dev/cu.usbserial-<REDACTED>`, pass explicitly), no-button esptool
flashing, `board = esp32dev` on pioarduino 54.03.21-2, `-O2 -fwrapv`,
board is astro-nav-int's `MESHTASTIC_ESP32_LX6` (Meshtastic reflashed
afterwards). Hash: FNV-1a 64 over each sample's characters +
`'\n'`, in order — identical definition in `hostref/host_main.c` and
`src/main.cpp`.
