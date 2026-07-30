# validation/cpu — int-llm on real hardware

Hardware validation records: one folder per target *and* per test, each with
the harness that produced the result and the raw captured transcript under
`<TARGET>/results/`. The layout and conventions follow astro-nav-int's
hardware test records; `HOWTO.md` covers host setup, flashing notes, and
troubleshooting. `run_test.sh <TARGET>` does prepare → build → flash →
capture for the MCU targets; `native_check.sh` is the ssh analogue for
Linux hosts.

Supplementary emulated-architecture checks live separately under
[`validation/qemu/`](../qemu/) and are excluded from every physical-board and
native-host count below.

Two harness types:

- `*_DET` — fp_math.h determinism gate: runs `fp_det_compute()` (the same
  grid + FNV-1a hashing as `make determinism`, built with
  `-DFP_DET_NO_MAIN`) on the target's portable two-limb backend and
  compares against a host pin computed at prepare time (which itself must
  match the committed golden `tests/determinism_golden.txt`). Targets whose
  SRAM cannot hold the ~34 KB input grid (SAMD21: 32 KB; ATmega2560: 8 KB)
  build with `-DFP_DET_EXTERNAL_GRID`: prepare.sh emits the grid as a const
  table (`fp_determinism.c` itself, built natively with
  `-DFP_DET_GRID_EMIT`) and the firmware reads it in place from flash —
  on the Harvard-architecture AVR via the `__flash` address space, as two
  halves (16-bit `size_t` caps single objects at 32 KB).
- `*_GPT` — microgpt_int INFERENCE: `prepare.sh` bakes the repo's committed
  `model.mgw` (trained on the host with `./gpt_int --save model.mgw`) into
  flash, and the firmware drives `mgpt_load_mem()` (zero-copy: weights are
  read through memory-mapped flash, never RAM) + `mgpt_generate_sample()`
  (`-DMGPT_NO_TRAIN -DMGPT_NO_MAIN`). PASS = the target reproduces the
  host training run's 20 samples byte-for-byte, PRNG stream included.

The native-Linux targets (`PI_1_MODEL_B_PLUS`, `PI_5_MODEL_B_16GB`,
`X86_64_*`) run both checks plus a full *training* byte-compare: the remote
host trains from the same `input.txt` and must produce the identical stdout
and the identical `.mgw` file as every other host.

## Results

| target | test | hash | runtime | result file |
|---|---|---|---|---|
| XIAO RP2040 Cortex-M0+ (Armv6-M) | determinism gate | `c0d933ea340452ec` (= golden) | 10851 ms | `XIAO_RP2040_DET/results/2026-07-29-xiao-rp2040-det.txt` |
| XIAO RP2040 Cortex-M0+ (Armv6-M) | microgpt inference, 20 samples | `ff4bc4bf7d4fd99d` (= host pin) | 6724 ms | `XIAO_RP2040_GPT/results/2026-07-29-xiao-rp2040-gpt.txt` |
| ESP32-C6 (RISC-V rv32imac, no FPU) | determinism gate | `c0d933ea340452ec` (= golden) | 5631 ms | `ESP32_C6_DET/results/2026-07-29-esp32-c6-det.txt` |
| ESP32-C6 (RISC-V rv32imac, no FPU) | microgpt inference, 20 samples | `ff4bc4bf7d4fd99d` (= host pin) | 2047 ms | `ESP32_C6_GPT/results/2026-07-29-esp32-c6-gpt.txt` |
| Pico 2 RP2350, ARM mode (Cortex-M33) | determinism gate | `c0d933ea340452ec` (= golden) | 4137 ms | `PICO2_ARM_DET/results/2026-07-29-pico2-arm-det.txt` |
| Pico 2 RP2350, ARM mode (Cortex-M33) | microgpt inference, 20 samples | `ff4bc4bf7d4fd99d` (= host pin) | 3090 ms | `PICO2_ARM_GPT/results/2026-07-29-pico2-arm-gpt.txt` |
| Pico 2 RP2350, RISC-V mode (Hazard3 rv32imac) | determinism gate | `c0d933ea340452ec` (= golden) | 5447 ms | `PICO2_RISCV_DET/results/2026-07-29-pico2-riscv-det.txt` |
| Pico 2 RP2350, RISC-V mode (Hazard3 rv32imac) | microgpt inference, 20 samples | `ff4bc4bf7d4fd99d` (= host pin) | 3756 ms | `PICO2_RISCV_GPT/results/2026-07-29-pico2-riscv-gpt.txt` |
| Heltec V3 ESP32-S3 (Xtensa LX7) | determinism gate | `c0d933ea340452ec` (= golden) | 3883 ms | `HELTEC_V3_DET/results/2026-07-29-heltec-s3-det.txt` |
| Heltec V3 ESP32-S3 (Xtensa LX7) | microgpt inference, 20 samples | `ff4bc4bf7d4fd99d` (= host pin) | 1064 ms | `HELTEC_V3_GPT/results/2026-07-29-heltec-s3-gpt.txt` |
| LILYGO T-Beam ESP32 (Xtensa LX6) | determinism gate | `c0d933ea340452ec` (= golden) | 4237 ms | `TBEAM_LX6_DET/results/2026-07-29-tbeam-lx6-det.txt` |
| LILYGO T-Beam ESP32 (Xtensa LX6) | microgpt inference, 20 samples | `ff4bc4bf7d4fd99d` (= host pin) | 2591 ms | `TBEAM_LX6_GPT/results/2026-07-29-tbeam-lx6-gpt.txt` |
| XIAO nRF52840 (Cortex-M4F) | determinism gate | `c0d933ea340452ec` (= golden) | 13883 ms | `XIAO_NRF52840_DET/results/2026-07-29-xiao-nrf52840-det.txt` |
| XIAO nRF52840 (Cortex-M4F) | microgpt inference, 20 samples | `ff4bc4bf7d4fd99d` (= host pin) | 3424 ms | `XIAO_NRF52840_GPT/results/2026-07-29-xiao-nrf52840-gpt.txt` |
| Arduino MKR Zero SAMD21 (Cortex-M0+ @ 48 MHz) | determinism gate (flash-resident grid) | `c0d933ea340452ec` (= golden) | 37249 ms | `ARDUINO_MKR_ZERO_DET/results/2026-07-29-mkrzero-samd21-det.txt` |
| Arduino MKR Zero SAMD21 (Cortex-M0+ @ 48 MHz) | microgpt inference, 20 samples | `ff4bc4bf7d4fd99d` (= host pin) | 26377 ms | `ARDUINO_MKR_ZERO_GPT/results/2026-07-29-mkrzero-samd21-gpt.txt` |
| Arduino Mega 2560 ATmega2560 (8-bit AVR @ 16 MHz) | determinism gate (flash-resident grid) | `c0d933ea340452ec` (= golden) | 747901 ms | `ARDUINO_MEGA2560_DET/results/2026-07-29-mega2560-avr-det.txt` |
| Raspberry Pi 1 B+ (ARMv6, 32-bit Linux) | det ×2 + inference + full training | golden + byte-identical | det 746 ms, train 594,056 ms | `PI_1_MODEL_B_PLUS/results/2026-07-28-pi1-bplus.txt` + timestamped training excerpt `2026-07-28-pi1-bplus-train-ts.txt` |
| Raspberry Pi 5 Model B 16 GB (Cortex-A76, aarch64 Linux) | det ×2 + inference + full training; TinyLlama native gate | golden + byte-identical; TinyLlama 80/80 | det 12 ms, train 8188 ms; TinyLlama 313.5 s | `PI_5_MODEL_B_16GB/results/2026-07-30-pi5-model-b-16gb.txt` + `2026-07-30-pi5-tinyllama.txt` |
| Apple M3 MacBook Air (arm64 macOS, clang; CPU only) | TinyLlama native-stream gate | TinyLlama 80/80 | 447.9 s wall, I/O-bound | `ARM64_APPLE_M3/results/2026-07-30-apple-m3-tinyllama.txt` |
| AMD Ryzen 7 7700 (x86-64 Linux, gcc) | det ×2 + inference + full training; TinyLlama native gate | golden + byte-identical; TinyLlama 80/80 | train 2095 ms; TinyLlama 51.8 s | `X86_64_AMD_ZEN4/results/2026-07-28-x86-amd-zen4.txt` + `2026-07-30-x86-amd-zen4-tinyllama.txt` |
| Intel i7-7700 (x86-64 Linux, gcc) | det ×2 + inference + full training; TinyLlama native gate | golden + byte-identical; TinyLlama 80/80 | train 4441 ms; TinyLlama 76.7 s | `X86_64_INTEL_KABYLAKE/results/2026-07-28-x86-intel-kabylake.txt` + `2026-07-30-x86-intel-kabylake-tinyllama.txt` |

Reference points: the same golden `c0d933ea340452ec` holds on arm64 macOS
(native `__int128` + forced-portable) and on every target above; the same
20 samples (kayla, daia, lee, …, karin) are what every host prints after
`./gpt_int --save model.mgw` and on every `--load` of that file. The
committed `model.mgw` itself has been reproduced bit-for-bit by training
on arm64 macOS, aarch64 Linux (Pi 5), x86-64 AMD, x86-64 Intel, and
32-bit ARMv6 (Pi 1).

MCU scoreboard: 8 boards, 9 ISA-mode targets (the Pico 2 runs both its ARM
and RISC-V modes), 4 ISA families (ARM Cortex-M, RISC-V, Xtensa, AVR),
17/17 PASS. Every board passes both harnesses except the 8-bit Mega 2560,
which is determinism-only by hardware limit (microgpt's ~13 KB of inference
state exceeds its 8 KB SRAM). The Linux hosts (Pi 1, Pi 5, AMD, Intel —
plus the arm64 macOS reference) are counted separately: five host systems
across three ISA classes. Inference throughput at 122 forward passes per
20-sample run ranges
from ~4.6 tok/s (SAMD21 M0+ @ 48 MHz) to ~115 tok/s (ESP32-S3 LX7).

Note on tree stamps: every record line carries the clean commit the tested
sources came from. All MCU records are from the 2026-07-29 campaign and
stamp `tree cf0bd4cc8589` (the commit that added the flash-resident grid
mode); the Pi 1, AMD, and Intel native-host records stamp `tree
1b706ecf6c7f` from 2026-07-28, while the Pi 5 record stamps
`tree af8336daa0b0` from 2026-07-30. `fp_math.h` and the native
training/inference path are unchanged; the later `fp_determinism.c` work adds
external-grid support, while its default native and forced-portable builds
still reproduce the same golden.
MCU transcripts additionally pin the exact firmware artifact and every
prepared source file with SHA-256; native transcripts record per-source
SHA-256 on both ends.
