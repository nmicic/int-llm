# HW test: Pico 2 (RP2350, ARM mode) — microgpt_int INFERENCE

Status: **PASS** (2026-07-28, tree `8b16a00bb912-dirty`*)

Same train-big/run-small loop as `XIAO_RP2040_GPT`: the model was trained
on an arm64 Mac (`./gpt_int --save model.mgw`, 5000 steps, Q16.48
integer-only), the 115,576-byte `.mgw` image is baked into the firmware by
prepare.sh, and the Pico 2 runs the full inference path
(`mgpt_load_mem()` + `mgpt_generate_sample()`, `-DMGPT_NO_TRAIN
-DMGPT_NO_MAIN`, portable two-limb backend) — reproducing the Mac training
run's 20 sampled names **byte-for-byte**, PRNG stream included.

| | samples hash | runtime (informational) |
|---|---|---|
| host reference (arm64 macOS, `__int128` backend, same entry points) | `ff4bc4bf7d4fd99d` | — |
| XIAO RP2040 (Cortex-M0+ @ 133 MHz) | `ff4bc4bf7d4fd99d` | 6717 ms |
| ESP32-C6 (RISC-V rv32imac @ 160 MHz) | `ff4bc4bf7d4fd99d` | 2047 ms |
| **Pico 2 (Cortex-M33 @ 150 MHz, ARM mode)** | **`ff4bc4bf7d4fd99d`** | 3089 ms (20 samples, ~154 ms/name) |

Samples: kayla, daia, lee, kayan, maha, kaia, ramiar, anall, ainale, kelel,
malana, arile, marion, avile, calan, kaylia, diaria, sarona, jahel, karin —
identical to the Mac's `./gpt_int --save` / `--load` output. The same chip
also passes in its RISC-V execution mode — see `PICO2_RISCV_GPT`.

\* dirty = the fp_math.h dual-backend vendoring + .mgw save/load +
`MGPT_NO_MAIN`/`mgpt_load_mem` additions were not yet committed when the
test ran; the result file's firmware SHA-256 pins the exact image.

Raw capture: `results/2026-07-28-pico2-arm-gpt.txt`.

## Memory

- Image: 185,608 B flash (4.4%) — includes the whole 115 KB weight file in
  rodata; 15,364 B static RAM (**2.9%** of 520 KB)
- The weights never occupy RAM: `mgpt_load_mem()` points the weight slots
  straight into the flash array and reads go through XIP

## Hardware

- Board: Raspberry Pi Pico 2, VID:PID `2E8A:000F`
  — same physical board as the other `PICO2_*` targets
- SoC: RP2350, 2× Cortex-M33 (Armv8-M) @ 150 MHz in this mode

## Config

- Platform: maxgerhardt `platform-raspberrypi` pinned commit `aa70b802`
  (arduino-pico core), `board = rpipico2`, `-O2 -fwrapv`
- Hash: FNV-1a 64 over each sample's characters + `'\n'`, in order —
  identical definition in `hostref/host_main.c` and `src/main.cpp`
