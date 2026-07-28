#!/bin/sh
# Copy fp_math.h + fp_determinism.c from the repo root into this harness,
# stamp the tree state into include/build_info.h, and pin the expected
# combined hash into include/det_pin.h.
#
# Pin provenance: the pin is the HOST's DETERMINISM_HASH (native __int128
# backend), computed here at prepare time by building and running the same
# fp_determinism.c with its stdio main against the committed golden — so
# prepare itself fails if the host has drifted from tests/determinism_golden.txt,
# and the on-target compare is a genuine cross-backend (host __int128 vs
# portable two-limb) AND cross-ISA check.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../../.." && pwd)
mkdir -p "$HERE/src" "$HERE/include"
cp "$REPO/fp_determinism.c" "$HERE/src/fp_determinism.c"
cp "$REPO/fp_math.h"        "$HERE/include/fp_math.h"

COMMIT=$(git -C "$REPO" rev-parse --short=12 HEAD)
DIRTY=""
# Dirty is scoped to exactly the files copied above — an edited doc
# elsewhere must not taint the stamp.
if ! git -C "$REPO" diff --quiet HEAD -- fp_determinism.c fp_math.h 2>/dev/null; then
    DIRTY="-dirty"
fi
printf '#define BUILD_COMMIT "%s%s"\n' "$COMMIT" "$DIRTY" > "$HERE/include/build_info.h"

# Host reference: build the copied sources with the stdio main, run against
# the committed golden (fails prepare on mismatch), extract the hash.
HOSTDIR="$HERE/.hostref"
mkdir -p "$HOSTDIR"
cc -std=c11 -O2 -fwrapv -DFP_MATH_WITH_STDIO -I"$HERE/include" \
    "$HERE/src/fp_determinism.c" -o "$HOSTDIR/host_det"
HOST_OUT=$("$HOSTDIR/host_det" "$REPO/tests/determinism_golden.txt")
echo "$HOST_OUT" | tail -2
PIN_HASH=$(printf '%s\n' "$HOST_OUT" \
    | sed -n 's/^DETERMINISM_HASH=\([0-9a-f]\{16\}\)$/\1/p')
if [ -z "$PIN_HASH" ]; then
    echo "prepare: FAIL — host reference did not print DETERMINISM_HASH" >&2
    exit 1
fi
printf '#define PINNED_HASH 0x%sULL\n' "$PIN_HASH" > "$HERE/include/det_pin.h"

echo "prepared: sources from $REPO (tree $COMMIT$DIRTY, host pin 0x$PIN_HASH)"
