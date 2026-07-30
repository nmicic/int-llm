#!/bin/sh
# Copyright (c) 2026 Nenad Mićić
# SPDX-License-Identifier: Apache-2.0
#
# Cross-compile the determinism gate and tiny-GPT inference loader as static
# Linux binaries, then execute them with QEMU user-mode emulation.
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH= cd -- "$script_dir/../.." && pwd)
tmpdir=$(mktemp -d "${TMPDIR:-/tmp}/intllm-qemu.XXXXXXXX")
trap 'rm -rf -- "$tmpdir"' EXIT HUP INT TERM

golden=c0d933ea340452ec
model_sha=466cfe9dba7b888cdaa23dedf4b10351826795793448c8e95dcb0f7a61ed33eb
infer_sha=f62a78d3e878b341cd59289a919ec4991b8055eaf0b4d715e2bc58bba2865d63
targets=${TARGETS:-"ppc64le riscv64 mips64el alpha i686 sh4 s390x ppc64 sparc64 mips64 hppa m68k"}

for tool in sha256sum sed cut grep; do
    command -v "$tool" > /dev/null 2>&1 || {
        echo "missing required host tool: $tool" >&2
        exit 2
    }
done

cp "$repo_root/fp_math.h" "$tmpdir/"
cp "$repo_root/fp_determinism.c" "$tmpdir/"
cp "$repo_root/microgpt_int.c" "$tmpdir/"
cp "$repo_root/safetensors.h" "$tmpdir/"
cp "$repo_root/tokenizer.h" "$tmpdir/"
cp "$repo_root/llama_int.c" "$tmpdir/"
cp "$repo_root/model.mgw" "$tmpdir/"
printf '%s\n' "$golden" > "$tmpdir/determinism_golden.txt"
cd "$tmpdir"

actual_model_sha=$(sha256sum model.mgw | cut -d ' ' -f 1)
if test "$actual_model_sha" != "$model_sha"; then
    echo "model.mgw SHA-256 mismatch: got $actual_model_sha" >&2
    exit 1
fi

run_target()
{
    label=$1
    cc=$2
    qemu=$3
    endian=$4
    expected_backend=$5

    command -v "$cc" > /dev/null 2>&1 || {
        echo "$label: missing cross-compiler $cc" >&2
        exit 2
    }
    command -v "$qemu" > /dev/null 2>&1 || {
        echo "$label: missing emulator $qemu" >&2
        exit 2
    }

    int128_size=$("$cc" -dM -E - < /dev/null |
        sed -n 's/^#define __SIZEOF_INT128__ //p')
    if test "$expected_backend" = native-int128; then
        test "$int128_size" = 16 || {
            echo "$label: compiler no longer exposes 128-bit integers" >&2
            exit 1
        }
    else
        test -z "$int128_size" || {
            echo "$label: compiler unexpectedly exposes 128-bit integers" >&2
            exit 1
        }
    fi

    echo "=== $label ($endian, $expected_backend) ==="
    "$cc" --version | sed -n '1p'
    "$qemu" --version | sed -n '1p'

    "$cc" -O3 -static -fwrapv -Wall -Wextra -Werror \
        -Wno-unused-function -std=c11 -I. \
        -DFP_MATH_WITH_STDIO -DFP_MATH_INT128_ALIASES \
        -o "$label-det-auto" fp_determinism.c
    "$cc" -O3 -static -fwrapv -Wall -Wextra -Werror \
        -Wno-unused-function -std=c11 -I. \
        -DFP_MATH_WITH_STDIO -DFP_MATH_INT128_ALIASES \
        -DFP_MATH_FORCE_PORTABLE \
        -o "$label-det-portable" fp_determinism.c
    "$cc" -O3 -static -fwrapv -Wall -Wextra -Werror \
        -Wno-unused-function -std=c11 -I. \
        -DFP_MATH_WITH_STDIO -DFP_MATH_INT128_ALIASES \
        -DMGPT_NO_TRAIN \
        -o "$label-gpt-infer" microgpt_int.c

    "$qemu" "./$label-det-auto" determinism_golden.txt \
        > "$label-det-auto.txt"
    "$qemu" "./$label-det-portable" determinism_golden.txt \
        > "$label-det-portable.txt"

    auto_hash=$(sed -n 's/^DETERMINISM_HASH=//p' "$label-det-auto.txt")
    portable_hash=$(sed -n 's/^DETERMINISM_HASH=//p' "$label-det-portable.txt")
    test "$auto_hash" = "$golden"
    test "$portable_hash" = "$golden"

    echo "auto_backend=$expected_backend hash=$auto_hash PASS"
    echo "forced_portable_hash=$portable_hash PASS"

    if test "$expected_backend" = native-int128; then
        "$cc" -O3 -static -fwrapv -Wall -Wextra -Werror \
            -Wno-unused-function -std=c11 -I. \
            -DFP_MATH_WITH_STDIO -DFP_MATH_INT128_ALIASES \
            -o "$label-llama-int" llama_int.c
        if "$qemu" "./$label-llama-int" \
            > "$label-llama-usage.txt" 2>&1; then
            echo "$label: llama_int without arguments unexpectedly succeeded" >&2
            exit 1
        fi
        grep -F "Usage:" "$label-llama-usage.txt" > /dev/null
        echo "llama_int_static_build_and_startup=PASS"
    else
        echo "llama_int_static_build=SKIP (target compiler has no __int128)"
    fi

    if test "$endian" = little-endian; then
        "$qemu" "./$label-gpt-infer" --load model.mgw \
            > "$label-gpt-infer.txt"
        output_sha=$(sha256sum "$label-gpt-infer.txt" | cut -d ' ' -f 1)
        test "$output_sha" = "$infer_sha"
        echo "microgpt_output_sha256=$output_sha PASS"
    else
        if "$qemu" "./$label-gpt-infer" --load model.mgw \
            > "$label-gpt-infer.txt" 2> "$label-gpt-infer.err"; then
            echo "$label: unexpectedly accepted little-endian model.mgw" >&2
            exit 1
        fi
        grep -F "not a native-endian MGW v1 file" \
            "$label-gpt-infer.err" > /dev/null
        echo "cross_endian_loader=PASS (little-endian MGW rejected)"
    fi
}

echo "source_sha256:"
sha256sum fp_math.h fp_determinism.c microgpt_int.c \
    safetensors.h tokenizer.h llama_int.c model.mgw

for label in $targets; do
    case "$label" in
        ppc64le)
            run_target "$label" powerpc64le-linux-gnu-gcc qemu-ppc64le \
                little-endian native-int128
            ;;
        riscv64)
            run_target "$label" riscv64-linux-gnu-gcc qemu-riscv64 \
                little-endian native-int128
            ;;
        mips64el)
            run_target "$label" mips64el-linux-gnuabi64-gcc qemu-mips64el \
                little-endian native-int128
            ;;
        alpha)
            run_target "$label" alpha-linux-gnu-gcc qemu-alpha \
                little-endian native-int128
            ;;
        i686)
            run_target "$label" i686-linux-gnu-gcc qemu-i386 \
                little-endian portable-two-limb
            ;;
        sh4)
            run_target "$label" sh4-linux-gnu-gcc qemu-sh4 \
                little-endian portable-two-limb
            ;;
        s390x)
            run_target "$label" s390x-linux-gnu-gcc qemu-s390x \
                big-endian native-int128
            ;;
        ppc64)
            run_target "$label" powerpc64-linux-gnu-gcc qemu-ppc64 \
                big-endian native-int128
            ;;
        sparc64)
            run_target "$label" sparc64-linux-gnu-gcc qemu-sparc64 \
                big-endian native-int128
            ;;
        mips64)
            run_target "$label" mips64-linux-gnuabi64-gcc qemu-mips64 \
                big-endian native-int128
            ;;
        hppa)
            run_target "$label" hppa-linux-gnu-gcc qemu-hppa \
                big-endian portable-two-limb
            ;;
        m68k)
            run_target "$label" m68k-linux-gnu-gcc qemu-m68k \
                big-endian portable-two-limb
            ;;
        *)
            echo "unknown TARGETS entry: $label" >&2
            exit 2
            ;;
    esac
done

echo "QEMU_MATRIX=PASS"
