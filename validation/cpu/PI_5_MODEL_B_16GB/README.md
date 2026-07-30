# CPU test: Raspberry Pi 5 Model B 16 GB — aarch64 Linux, native gcc

Status: **PASS** (2026-07-30)

This is the second Arm Linux host and the fifth independent training host.
`../native_check.sh` copied the exact committed tiny-GPT sources and
`model.mgw` to a random temporary directory, built them with the Pi's distro
gcc, and byte-compared the results against the arm64 macOS host.

| check | result | runtime (informational) |
|---|---|---|
| determinism, default backend (`__int128`) | `c0d933ea340452ec` (= golden) | 12 ms |
| determinism, `-DFP_MATH_FORCE_PORTABLE` | `c0d933ea340452ec` (= golden) | 40 ms |
| `--load model.mgw` inference, 20 samples | byte-identical to macOS host | 6 ms |
| full 5000-step training stdout | byte-identical to macOS host | 8188 ms |
| `--save` after training | `.mgw` byte-identical to the committed `model.mgw` | — |

The same host also ran the stronger TinyLlama gate. The 8,800,406,496-byte
native Q16.48 `tinyllama.mgw` stayed on the normal home filesystem and was
loaded with `--native` mmap; no RAM disk was used. All four greedy benchmark
prompts matched the committed float-reference oracle, **80/80 tokens**. The
run took 313.5 seconds wall-clock (about 0.26–0.27 generated token/s per
prompt), reported a modeled peak of 8,569.7 MiB, used no swap, and remained
unthrottled (`53.8 °C` before, `58.7 °C` after).

The Pi's `/tmp` is an 8.0 GiB tmpfs, slightly smaller than the 8.2 GiB model.
It is neither large enough nor needed: 16 GiB of RAM is sufficient for the
normal filesystem-backed mmap and page cache.

The first GCC 14 build exposed a missing feature-test declaration for
`realpath()`. Adding `_XOPEN_SOURCE=700` beside the existing
`_POSIX_C_SOURCE=200809L` in `llama_int.c` fixed the portable declaration;
the final source then built warning-free with `-Werror` on both Debian/GCC 14
and macOS/Clang before the recorded 80-token run.

Recorded transcripts:

- `results/2026-07-30-pi5-model-b-16gb.txt`
- `results/2026-07-30-pi5-tinyllama.txt`

## Hardware / Config

- Board: Raspberry Pi 5 Model B Rev 1.1, 16 GB RAM
- CPU: Broadcom BCM2712, four Cortex-A76 cores, aarch64, up to 2.4 GHz
- OS: Debian GNU/Linux 13 (trixie), Linux `6.18.34+rpt-rpi-2712`
- Toolchain: gcc 14.2.0, `-O3 -march=native -fwrapv -std=c11`
- TinyLlama model SHA-256:
  `7e8218d7f79a784f9d1868140fb16c3b9f5fbc45c19fb5c807ddcba5b41e32a8`
