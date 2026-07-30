# QEMU cross-architecture validation

This is a supplementary portability gate, separate from the real-hardware
records in [`validation/cpu/`](../cpu/). It cross-compiles static Linux
binaries and runs them with QEMU user-mode emulation. Emulated targets are
never counted as physical boards or native host systems.

For every target, [`run_matrix.sh`](run_matrix.sh) builds and runs:

1. `fp_determinism.c` with the compiler-selected backend;
2. the same gate with `FP_MATH_FORCE_PORTABLE`; and
3. the inference-only tiny GPT against the committed `model.mgw`.

The eight 64-bit targets whose compilers expose `__int128` also compile,
statically link, and start the full `llama_int` executable. The four 32-bit
targets skip that build because `llama_int.c` still uses raw `__int128`
directly; this is a documented project boundary, not a failed emulation test.

Both determinism builds must reproduce `c0d933ea340452ec`. On little-endian
targets, the tiny GPT must reproduce the exact 20-sample stdout SHA-256
`f62a78d3e878b341cd59289a919ec4991b8055eaf0b4d715e2bc58bba2865d63`.
The `.mgw` v1 format is intentionally host-native-endian, so big-endian
targets instead prove that the little-endian committed file is rejected
cleanly rather than misread.

## Matrix

| target | width / endian | compiler-selected backend | tiny-GPT file check |
|---|---|---|---|
| PowerPC64LE | 64-bit LE | native `__int128` | exact 20-sample output |
| RISC-V 64 | 64-bit LE | native `__int128` | exact 20-sample output |
| MIPS64EL | 64-bit LE | native `__int128` | exact 20-sample output |
| Alpha | 64-bit LE | native `__int128` | exact 20-sample output |
| i686 | 32-bit LE | portable two-limb | exact 20-sample output |
| SuperH-4 | 32-bit LE | portable two-limb | exact 20-sample output |
| IBM s390x | 64-bit BE | native `__int128` | expected endian rejection |
| PowerPC64 | 64-bit BE | native `__int128` | expected endian rejection |
| SPARC64 | 64-bit BE | native `__int128` | expected endian rejection |
| MIPS64 | 64-bit BE | native `__int128` | expected endian rejection |
| PA-RISC | 32-bit BE | portable two-limb | expected endian rejection |
| Motorola 68k | 32-bit BE | portable two-limb | expected endian rejection |

That is 12 ABI/endian configurations across 10 ISA families, including four
32-bit targets and six big-endian targets. The first recorded run passed all
12; see [`results/2026-07-30-qemu-user-matrix.txt`](results/2026-07-30-qemu-user-matrix.txt).

## Running it

On Ubuntu 24.04, install QEMU user mode and the selected cross-compilers:

```bash
sudo apt-get install qemu-user \
  gcc-powerpc64le-linux-gnu gcc-riscv64-linux-gnu \
  gcc-mips64el-linux-gnuabi64 gcc-alpha-linux-gnu \
  gcc-i686-linux-gnu gcc-sh4-linux-gnu gcc-s390x-linux-gnu \
  gcc-powerpc64-linux-gnu gcc-sparc64-linux-gnu \
  gcc-mips64-linux-gnuabi64 gcc-hppa-linux-gnu gcc-m68k-linux-gnu
```

Review the package-manager plan first: on Ubuntu, installing these cross
toolchains can remove the `gcc-multilib` and `g++-multilib` meta-packages.

Then run the whole matrix, or a named subset:

```bash
validation/qemu/run_matrix.sh
TARGETS="ppc64le s390x i686" validation/qemu/run_matrix.sh
```

The runner uses an isolated `mktemp` directory, compiles with warnings as
errors, pins all source/model hashes in its output, and removes its temporary
artifacts on exit.

This gate deliberately does not decode the 8.2 GiB TinyLlama model under
emulation. Full TinyLlama decode under QEMU TCG would mostly measure emulator
and storage overhead; the meaningful full-model evidence remains the native
real-host 80-token gate. The QEMU matrix targets arithmetic, compiler/ABI,
word-size, endianness, serialization rejection, small-model execution, and
warning-free full-runtime build/startup coverage where `__int128` exists.
