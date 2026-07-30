#!/bin/sh
# Native-Linux analogue of the MCU harnesses: copy the sources plus the
# committed model.mgw to a remote host over ssh, compile them there with the
# distro's native gcc (no cross toolchain, no PlatformIO), run both checks,
# and capture a provenance-stamped transcript under <TARGET>/results/.
#
# Checks (all fail-closed — the script exits non-zero unless every one PASSes):
#   det           fp_determinism on the remote's default backend must
#                 reproduce the committed golden tests/determinism_golden.txt
#   det-portable  same, with -DFP_MATH_FORCE_PORTABLE (two-limb backend)
#   gpt           microgpt_int built -DMGPT_NO_TRAIN, `--load model.mgw`;
#                 the 20-sample stream is copied back and byte-compared
#                 (cmp) against the same build/run on the local host
#   train         (only with --train) full training run; stdout is
#                 byte-compared against the local host's run, and the
#                 remote-trained .mgw against the committed model.mgw
#
# The gpt/train compares are cross-compiler and (when host and remote differ)
# cross-architecture byte-equality — the same contract the flash-image MCU
# targets prove, minus the serial capture.
#
# Usage: sh native_check.sh <user@host> <target-label> <TARGET_FOLDER> [--train]
#   e.g. sh native_check.sh pi@raspberrypi.local pi1-bplus PI_1_MODEL_B_PLUS
set -eu

USERHOST=$1; LABEL=$2; TARGET=$3; TRAIN=${4:-}
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
RESULTS="$HERE/$TARGET/results"
mkdir -p "$RESULTS"
STAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)
OUT="$RESULTS/$(date -u +%Y-%m-%d)-$LABEL.txt"
# unpredictable per-run remote workdir (never a fixed path we rm -rf)
REMOTE_DIR=$(ssh "$USERHOST" 'mktemp -d "${TMPDIR:-/tmp}/intllm-native-check.XXXXXX"')
[ -n "$REMOTE_DIR" ] || { echo "native_check: remote mktemp failed" >&2; exit 1; }

SOURCES="fp_math.h fp_determinism.c microgpt_int.c model.mgw"
GOLDEN=$(cat "$REPO/tests/determinism_golden.txt")

COMMIT=$(git -C "$REPO" rev-parse --short=12 HEAD)
DIRTY=""
git -C "$REPO" diff --quiet HEAD -- $SOURCES 2>/dev/null || DIRTY="-dirty"

CFLAGS="-O2 -fwrapv -std=c11 -DFP_MATH_WITH_STDIO"

# Local host reference for the byte-compares.
HOSTDIR=$(mktemp -d)
trap 'rm -rf "$HOSTDIR"; ssh "$USERHOST" "rm -rf \"$REMOTE_DIR\"" 2>/dev/null || true' EXIT
for f in $SOURCES; do cp "$REPO/$f" "$HOSTDIR/"; done
cc $CFLAGS -DMGPT_NO_TRAIN -o "$HOSTDIR/gpt_infer" "$HOSTDIR/microgpt_int.c"
( cd "$HOSTDIR" && ./gpt_infer --load model.mgw > host_samples.txt )
if [ "$TRAIN" = "--train" ]; then
    [ -f "$REPO/input.txt" ] || { echo "native_check: --train needs $REPO/input.txt (make input)" >&2; exit 1; }
    cp "$REPO/input.txt" "$HOSTDIR/"
    cc $CFLAGS -o "$HOSTDIR/gpt_full" "$HOSTDIR/microgpt_int.c"
    ( cd "$HOSTDIR" && ./gpt_full --save trained.mgw > host_train.txt )
fi

scp -q "$HOSTDIR"/fp_math.h "$HOSTDIR"/fp_determinism.c "$HOSTDIR"/microgpt_int.c \
       "$HOSTDIR"/model.mgw "$USERHOST:$REMOTE_DIR/"
[ "$TRAIN" = "--train" ] && scp -q "$HOSTDIR/input.txt" "$USERHOST:$REMOTE_DIR/"

ARCH=$(ssh "$USERHOST" 'uname -m')
REMOTE_CPU=$(ssh "$USERHOST" '
    cpu=$(sed -n "s/^model name[[:space:]]*:[[:space:]]*//p" /proc/cpuinfo | head -1)
    if [ -z "$cpu" ] && command -v lscpu >/dev/null 2>&1; then
        cpu=$(lscpu | sed -n "s/^Model name:[[:space:]]*//p" | head -1)
    fi
    if [ -z "$cpu" ] && [ -r /proc/device-tree/model ]; then
        cpu=$(tr -d "\0" < /proc/device-tree/model)
    fi
    printf "%s" "$cpu"
')

{
    echo "# run $STAMP"
    echo "# target-folder $TARGET tree $COMMIT$DIRTY"
    echo "# golden (committed): $GOLDEN"
    echo "# sources sha256 (local tree):"
    ( cd "$HOSTDIR" && shasum -a 256 fp_math.h fp_determinism.c microgpt_int.c model.mgw | sed 's/^/#   /' )
    echo "# remote: $(ssh "$USERHOST" 'uname -srvm')"
    echo "# remote cpu: $REMOTE_CPU"
    echo "# remote gcc: $(ssh "$USERHOST" 'gcc --version | head -1')"
    echo "# remote sha256:"
    ssh "$USERHOST" "cd $REMOTE_DIR && sha256sum fp_math.h fp_determinism.c microgpt_int.c model.mgw" | sed 's/^/#   /'
} > "$OUT"

fail=0

run_det () {  # $1 = extra cflags, $2 = check name
    ssh "$USERHOST" "cd $REMOTE_DIR && gcc $CFLAGS $1 -o fp_det_$2 fp_determinism.c && \
        s=\$(date +%s%N); ./fp_det_$2 > det_$2.txt; e=\$(date +%s%N); \
        echo \$(( (e - s) / 1000000 )) > det_$2.ms"
    H=$(ssh "$USERHOST" "sed -n 's/^DETERMINISM_HASH=//p' $REMOTE_DIR/det_$2.txt")
    MS=$(ssh "$USERHOST" "cat $REMOTE_DIR/det_$2.ms")
    echo "target $LABEL arch $ARCH tree $COMMIT$DIRTY PROFILE $2 hash $H ($MS ms, informational)" >> "$OUT"
    if [ "$H" = "$GOLDEN" ]; then
        echo "$2: PASS (matches committed golden)" >> "$OUT"
    else
        echo "$2: FAIL (got '$H', want '$GOLDEN')" >> "$OUT"; fail=1
    fi
}

run_det ""                        det
run_det "-DFP_MATH_FORCE_PORTABLE" det-portable

ssh "$USERHOST" "cd $REMOTE_DIR && gcc $CFLAGS -DMGPT_NO_TRAIN -o gpt_infer microgpt_int.c && \
    s=\$(date +%s%N); ./gpt_infer --load model.mgw > remote_samples.txt; e=\$(date +%s%N); \
    echo \$(( (e - s) / 1000000 )) > gpt.ms"
scp -q "$USERHOST:$REMOTE_DIR/remote_samples.txt" "$HOSTDIR/"
MS=$(ssh "$USERHOST" "cat $REMOTE_DIR/gpt.ms")
echo "target $LABEL arch $ARCH tree $COMMIT$DIRTY PROFILE gpt samples=$(grep -c '^sample' "$HOSTDIR/remote_samples.txt") ($MS ms, informational)" >> "$OUT"
if cmp -s "$HOSTDIR/remote_samples.txt" "$HOSTDIR/host_samples.txt"; then
    echo "gpt: PASS (--load model.mgw output byte-identical to local host run)" >> "$OUT"
else
    echo "gpt: FAIL (--load output differs from local host run)" >> "$OUT"; fail=1
fi

if [ "$TRAIN" = "--train" ]; then
    ssh "$USERHOST" "cd $REMOTE_DIR && gcc $CFLAGS -o gpt_full microgpt_int.c && \
        s=\$(date +%s%N); ./gpt_full --save trained.mgw > remote_train.txt; e=\$(date +%s%N); \
        echo \$(( (e - s) / 1000000 )) > train.ms"
    scp -q "$USERHOST:$REMOTE_DIR/remote_train.txt" "$HOSTDIR/"
    scp -q "$USERHOST:$REMOTE_DIR/trained.mgw" "$HOSTDIR/remote.mgw"
    MS=$(ssh "$USERHOST" "cat $REMOTE_DIR/train.ms")
    echo "target $LABEL arch $ARCH tree $COMMIT$DIRTY PROFILE train ($MS ms, informational)" >> "$OUT"
    if cmp -s "$HOSTDIR/remote_train.txt" "$HOSTDIR/host_train.txt"; then
        echo "train: PASS (full training stdout byte-identical to local host run)" >> "$OUT"
    else
        echo "train: FAIL (training stdout differs from local host run)" >> "$OUT"; fail=1
    fi
    if cmp -s "$HOSTDIR/remote.mgw" "$HOSTDIR/model.mgw"; then
        echo "train-save: PASS (remote-trained .mgw byte-identical to committed model.mgw)" >> "$OUT"
    else
        echo "train-save: FAIL (remote-trained .mgw differs from committed model.mgw)" >> "$OUT"; fail=1
    fi
fi

if [ "$fail" -eq 0 ]; then
    echo "native_check: OK target=$LABEL arch=$ARCH tree=$COMMIT$DIRTY" >> "$OUT"
else
    echo "native_check: FAIL target=$LABEL arch=$ARCH tree=$COMMIT$DIRTY" >> "$OUT"
fi
cat "$OUT"
exit "$fail"
