#!/bin/sh
# One-command hardware test run: prepare -> build -> flash -> capture.
#   sh run_test.sh <TARGET_FOLDER> [serial-port]
# e.g.
#   sh run_test.sh PICO2_RISCV_DET
#   sh run_test.sh PICO2_ARM_DET /dev/cu.usbmodem101   # explicit port (USB hub)
#
# Writes results/<date>-<target-name>.txt with a provenance header (firmware
# and prepared-source SHA-256s, env, platform pin, toolchain) followed by the
# captured transcript.
# Exits non-zero unless the capture proved a PASS from the expected target —
# the capture script writes the file itself, so there is no tee-in-a-pipeline
# exit-status hole.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
TARGET_DIR=${1:?usage: run_test.sh <target-folder> [port]}
PORT=${2:-}

cd "$HERE/$TARGET_DIR/harness"
sh prepare.sh
pio run
pio run -t upload

ENV=$(sed -n 's/^\[env:\(.*\)\]$/\1/p' platformio.ini | head -1)
NAME=$(grep -o 'HW_TEST_TARGET=[^ ]*' platformio.ini | head -1 | sed 's/.*=//; s/[\\"]//g')
# The flashable artifact differs per platform (uf2 for RP2 chips, zip DFU
# package for nRF52, hex/bin elsewhere); hash the first one present.
FW=""
for A in firmware.uf2 firmware.zip firmware.hex firmware.bin; do
    if [ -f ".pio/build/$ENV/$A" ]; then FW=".pio/build/$ENV/$A"; break; fi
done
[ -n "$FW" ] || { echo "run_test: no firmware artifact found" >&2; exit 1; }
SHA=$(shasum -a 256 "$FW" | cut -d' ' -f1)
PLATFORM_PIN=$(sed -n 's/^platform = //p' platformio.ini)

OUT="$HERE/$TARGET_DIR/results/$(date +%F)-$NAME.txt"
# truncate, don't append: one canonical run per file — a same-day rerun
# replaces the record instead of stacking partial/failed attempts under it
{
    echo "# run $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "# target-folder $TARGET_DIR env $ENV"
    echo "# firmware sha256 $SHA"
    echo "# platform $PLATFORM_PIN"
    # pin the exact source inputs (prepare.sh copies/generates them fresh,
    # so the firmware hash alone would not identify them on a dirty tree)
    for SRC in src/*.c src/*.cpp src/*.h include/*.h; do
        [ -f "$SRC" ] || continue
        echo "# source sha256 $(shasum -a 256 "$SRC" | cut -d' ' -f1)  $SRC"
    done
    # keep the operator's home directory out of the committed transcript
    pio system info 2>/dev/null | grep -E 'PlatformIO Core|Python' \
        | sed "s|$HOME|\$HOME|g; s/^/# /" || true
} > "$OUT"

# WINDOW env var overrides the capture listen window (seconds) — needed for
# slow 8-bit targets where one profile iteration takes tens of seconds.
python3 "$HERE/capture_serial.py" --expect-target "$NAME" --out "$OUT" \
    ${PORT:+--port "$PORT"} ${WINDOW:+--window "$WINDOW"}
echo "result: $OUT"
