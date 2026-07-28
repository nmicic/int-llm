# ==============================================================================
# int-llm — Integer-Only LLM Stack in Pure C
# ==============================================================================
#
# Variants:
#   gpt_float  — Original float32 baseline (requires -lm)
#   gpt_int    — Integer-only Q16.48 fixed-point (NO -lm)
#
# Quick start:
#   make input      — download the tiny character dataset if missing
#   make all        — build all variants + test suite
#   make test       — run fp_math.h unit tests
#   make bench      — train float + integer variants, show timing comparison
#   make run-float  — train + infer with float baseline
#   make run-int    — train + infer with integer variant
#   make clean      — remove binaries
#
# Requirements:
#   - C99/C11 compiler with 64-bit integers — NOT necessarily a 64-bit platform.
#     fp_math.h uses native __int128 where the compiler has it (gcc/clang on
#     64-bit hosts) and a bit-identical two-limb software backend everywhere
#     else (Cortex-M, RV32, any 32-bit target). `make determinism-portable`
#     proves both backends hash identically. (`llama_int.c` and the legacy
#     `fp_test.c` self-test still use raw __int128 directly and need a
#     compiler that provides it.)
#   - run `make input` for the tiny GPT demo dataset
#   - No external dependencies (integer variants need NO math library)
#
# ==============================================================================

CC      = gcc
# -fwrapv defines signed overflow as two's-complement wraparound. REQUIRED for
# reproducible integer code: without it, signed overflow is UB and optimizer choices
# can differ across arch/compiler. `make determinism` guards the resulting bitstream.
# FP_MATH_WITH_STDIO enables fp_print (the header keeps stdio out of embedded
# builds by default); FP_MATH_INT128_ALIASES restores the legacy int128_t /
# uint128_t typedef spellings that fp_test.c and llama_int.c use.
CFLAGS  = -O3 -march=native -fwrapv -Wall -Wextra -Wno-unused-function -std=c11 -I. \
          -DFP_MATH_WITH_STDIO -DFP_MATH_INT128_ALIASES
LDFLAGS =

# Float variant needs libm
LDFLAGS_FLOAT = -lm

# Source files
SRC_FLOAT = microgpt.c
SRC_INT   = microgpt_int.c
SRC_TEST  = fp_test.c
SRC_DET   = fp_determinism.c
SRC_LLAMA = llama_int.c

# Headers
HEADERS = fp_math.h safetensors.h tokenizer.h

# Tiny character dataset used by the GPT demo. Kept out of git; downloaded on demand.
INPUT = input.txt

# Binaries
BIN_FLOAT  = gpt_float
BIN_INT    = gpt_int
BIN_TEST   = fp_test
BIN_DET    = fp_determinism
BIN_LLAMA  = llama_int
# Portable-backend builds: same sources, -DFP_MATH_FORCE_PORTABLE swaps the
# native __int128 backend for the two-limb software backend a 32-bit target
# would use. Running these on the 64-bit host is the differential gate.
BIN_DET_P  = fp_determinism_portable
BIN_INT_P  = gpt_int_portable
# Inference-only (-DMGPT_NO_TRAIN): loads a .mgw written by `gpt_int --save`
BIN_INT_I  = gpt_int_infer

# ==============================================================================
# Build targets
# ==============================================================================

.PHONY: all input clean test bench run-float run-int benchmark-all help llama regression determinism determinism-portable

all: $(BIN_FLOAT) $(BIN_INT) $(BIN_TEST) $(BIN_DET) $(BIN_LLAMA)
	@echo ""
	@echo "=== Build complete ==="
	@echo "  $(BIN_FLOAT)      — float32 baseline (uses -lm)"
	@echo "  $(BIN_INT)        — integer-only Q16.48 (NO -lm)"
	@echo "  $(BIN_TEST)       — fp_math.h unit tests"
	@echo "  $(BIN_DET)        — determinism hash gate"
	@echo "  $(BIN_LLAMA)      — integer-only Llama-family inference"
	@echo ""
	@echo "Run 'make test' to verify math invariants, 'make bench' to benchmark variants."

input: $(INPUT)

$(INPUT): scripts/download_input.sh
	@./scripts/download_input.sh $@

$(BIN_FLOAT): $(SRC_FLOAT)
	$(CC) $(CFLAGS) -ffast-math -o $@ $< $(LDFLAGS_FLOAT)

$(BIN_INT): $(SRC_INT) $(HEADERS)
	$(CC) $(CFLAGS) -o $@ $<

$(BIN_TEST): $(SRC_TEST) $(HEADERS)
	$(CC) $(CFLAGS) -o $@ $< -lm

$(BIN_LLAMA): $(SRC_LLAMA) $(HEADERS)
	$(CC) $(CFLAGS) -o $@ $<

# Determinism gate driver — pure integer, NO -lm
$(BIN_DET): $(SRC_DET) fp_math.h
	$(CC) $(CFLAGS) -o $@ $<

# Same driver, forced onto the two-limb portable backend (what a 32-bit
# target compiles). Must reproduce the identical golden hash.
$(BIN_DET_P): $(SRC_DET) fp_math.h
	$(CC) $(CFLAGS) -DFP_MATH_FORCE_PORTABLE -o $@ $<

# Integer GPT on the portable backend — differential build for 32-bit parity.
$(BIN_INT_P): $(SRC_INT) $(HEADERS)
	$(CC) $(CFLAGS) -DFP_MATH_FORCE_PORTABLE -o $@ $<

# Inference-only GPT: no dataset array (~43 MB), no gradients, no Adam —
# weights come from a .mgw file written by `./gpt_int --save`. This is the
# configuration for small 32-bit targets: train on a big host, load there.
$(BIN_INT_I): $(SRC_INT) $(HEADERS)
	$(CC) $(CFLAGS) -DMGPT_NO_TRAIN -o $@ $<

# ==============================================================================
# Test & benchmark targets
# ==============================================================================

test: $(BIN_TEST)
	@echo ""
	@echo "=== Running fp_math.h Unit Tests (335 tests) ==="
	@./$(BIN_TEST)

run-float: $(BIN_FLOAT) $(INPUT)
	@echo ""
	@echo "=== Float32 Baseline (5000 steps) ==="
	@time ./$(BIN_FLOAT)

run-int: $(BIN_INT) $(INPUT)
	@echo ""
	@echo "=== Integer-Only Q16.48 (5000 steps) ==="
	@time ./$(BIN_INT)

bench: $(BIN_FLOAT) $(BIN_INT) $(INPUT)
	@echo ""
	@echo "=================================================================="
	@echo "  int-llm Benchmark — 5000 training steps + 20 inference samples"
	@echo "=================================================================="
	@echo ""
	@echo "--- [1/2] Float32 Baseline ---"
	@time ./$(BIN_FLOAT) 2>&1 | grep -E "^(step (   1|1000|2000|3000|4000|5000) |num |vocab |sample|--- )"
	@echo ""
	@echo "--- [2/2] Integer-Only Q16.48 ---"
	@time ./$(BIN_INT) 2>&1 | grep -E "^(step (   1|1000|2000|3000|4000|5000) |num |vocab |sample|--- |=== |Pi )"
	@echo ""
	@echo "=================================================================="
	@echo "  Benchmark complete. Compare wall-clock times above."
	@echo "=================================================================="

# Full benchmark with all output (not filtered)
benchmark-all: $(BIN_FLOAT) $(BIN_INT) $(BIN_TEST) $(INPUT)
	@echo "=== [0/3] Unit Tests ==="
	@./$(BIN_TEST)
	@echo ""
	@echo "=== [1/3] Float32 Baseline ==="
	@time ./$(BIN_FLOAT) 2>&1 | tail -25
	@echo ""
	@echo "=== [2/3] Integer-Only Q16.48 ==="
	@time ./$(BIN_INT) 2>&1 | tail -25

llama: $(BIN_LLAMA)
	@echo ""
	@echo "=== Integer-Only Llama-Family Inference ==="
	@echo "Usage: ./$(BIN_LLAMA) <model_dir> [--verify|--generate|--benchmark] [--prompt TEXT]"

# Cross-machine bit-exact determinism gate.
# Builds + runs the fp_math.h determinism driver and compares its combined
# FNV-1a 64 hash against the committed golden. Exit non-zero on mismatch.
# Per-function sub-hashes are printed so any drift localizes to one function.
DET_GOLDEN = tests/determinism_golden.txt
determinism: $(BIN_DET) $(DET_GOLDEN)
	@echo ""
	@echo "=== Determinism Gate: fp_math.h bit-exact hash vs golden ==="
	@./$(BIN_DET) $(DET_GOLDEN)

# Backend-differential gate: the two-limb software backend (the code path a
# 32-bit target runs) must hash bit-identically to the same golden.
determinism-portable: $(BIN_DET_P) $(DET_GOLDEN)
	@echo ""
	@echo "=== Determinism Gate (portable two-limb backend) vs golden ==="
	@./$(BIN_DET_P) $(DET_GOLDEN)

# Regression gate — hard stop on any invariant or threshold violation.
# Use as CI gate or pre-commit check.
# Exit non-zero on failure.
regression: $(BIN_TEST) $(BIN_DET) $(BIN_DET_P) $(DET_GOLDEN)
	@echo ""
	@echo "=== Regression Gate: Unit Tests ==="
	@./$(BIN_TEST)
	@echo ""
	@echo "=== Regression Gate: Cross-machine Determinism Hash ==="
	@./$(BIN_DET) $(DET_GOLDEN)
	@echo ""
	@echo "=== Regression Gate: Portable-backend Determinism Hash ==="
	@./$(BIN_DET_P) $(DET_GOLDEN)
	@echo ""
	@echo "=== ALL REGRESSION GATES PASSED ==="

clean:
	rm -f $(BIN_FLOAT) $(BIN_INT) $(BIN_TEST) $(BIN_DET) $(BIN_LLAMA)
	rm -f $(BIN_DET_P) $(BIN_INT_P) $(BIN_INT_I)
	rm -f gpt gpt_int_instr gpt_int_prof

help:
	@echo ""
	@echo "int-llm — Integer-Only LLM Stack in Pure C"
	@echo "=========================================="
	@echo ""
	@echo "Targets:"
	@echo "  make input         Download input.txt for the tiny GPT demo if missing"
	@echo "  make all           Build all variants + test suite"
	@echo "  make test          Run fp_math.h unit tests"
	@echo "  make bench         Train float + integer variants with timing comparison"
	@echo "  make benchmark-all Full benchmark with tests + all output"
	@echo "  make run-float     Train + infer float baseline"
	@echo "  make run-int       Train + infer integer variant"
	@echo "  make determinism   Run cross-machine bit-exact determinism gate (vs golden hash)"
	@echo "  make determinism-portable  Run the same gate on the portable backend"
	@echo "  make gpt_int_infer Build the inference-only .mgw loader"
	@echo "  make regression    Run full regression gate (unit tests + determinism)"
	@echo "  make clean         Remove all binaries"
	@echo "  make help          Show this help"
	@echo ""
	@echo "Files:"
	@echo "  fp_math.h             Q16.48 fixed-point math library (header-only)"
	@echo "  microgpt.c            Float32 GPT baseline"
	@echo "  microgpt_int.c        Integer-only GPT (uses fp_math.h)"
	@echo "  fp_test.c             Unit tests for fixed-point math"
	@echo "  README.md             Project overview + Quick Start"
	@echo ""
	@echo "Quick start:"
	@echo "  make input && make all && make test && make bench"
	@echo ""
