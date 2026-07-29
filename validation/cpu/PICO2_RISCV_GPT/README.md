# HW test: Pico 2 (RP2350, RISC-V mode) — microgpt_int INFERENCE

Status: **PASS** (2026-07-29, tree `cf0bd4cc8589`)

Same physical board as `PICO2_ARM_GPT`, switched to the RP2350's Hazard3
RISC-V execution mode (`board_build.mcu = rp2350-riscv`). The Mac-trained
115,576-byte `.mgw` image is baked into flash by prepare.sh and the
firmware runs the full inference path (`mgpt_load_mem()` +
`mgpt_generate_sample()`, `-DMGPT_NO_TRAIN -DMGPT_NO_MAIN`, portable
two-limb backend) — reproducing the Mac training run's 20 sampled names
**byte-for-byte**, PRNG stream included.

| | samples hash | runtime (informational) |
|---|---|---|
| host reference (arm64 macOS, `__int128` backend, same entry points) | `ff4bc4bf7d4fd99d` | — |
| Pico 2 same chip, ARM mode (Cortex-M33) | `ff4bc4bf7d4fd99d` | 3090 ms |
| **Pico 2 (Hazard3 rv32imac @ 150 MHz, RISC-V mode)** | **`ff4bc4bf7d4fd99d`** | 3756 ms (20 samples, ~188 ms/name) |

Samples: kayla, daia, lee, kayan, maha, kaia, ramiar, anall, ainale, kelel,
malana, arile, marion, avile, calan, kaylia, diaria, sarona, jahel, karin —
identical to the Mac's `./gpt_int --save` / `--load` output, and to what
the very same die prints when running its ARM cores.

Raw capture: `results/2026-07-29-pico2-riscv-gpt.txt`.

## Memory

- Image: 205,032 B flash (4.9%) — includes the whole 115 KB weight file in
  rodata; 25,508 B static RAM (**4.9%** of 520 KB)
- The weights never occupy RAM: `mgpt_load_mem()` points the weight slots
  straight into the flash array and reads go through XIP

## Hardware

- Board: Raspberry Pi Pico 2, VID:PID `2E8A:000F`
  — same physical board as the other `PICO2_*` targets
- SoC: RP2350, 2× Hazard3 RISC-V (rv32imac) @ 150 MHz in this mode, no FPU

## Config

- Platform: maxgerhardt `platform-raspberrypi` pinned commit `aa70b802`
  (arduino-pico core), `board = rpipico2`,
  `board_build.mcu = rp2350-riscv`, `-O2 -fwrapv`
- Hash: FNV-1a 64 over each sample's characters + `'\n'`, in order —
  identical definition in `hostref/host_main.c` and `src/main.cpp`
- After flashing across an ISA-mode switch the CDC port can take several
  seconds to re-enumerate; capture_serial.py's port retries cover it
