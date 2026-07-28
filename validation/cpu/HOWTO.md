# HOWTO: run the hardware validation tests

From "I plugged in a board" to a recorded result. Companion to `README.md`
(conventions and the results table); any target folder (e.g.
`XIAO_RP2040_DET/`) is a working example. The layout and workflow follow
astro-nav-int's hardware test records (`embedded/hw/` there).

## 0. One-time host setup (macOS)

```sh
brew install platformio picotool esptool
python3 -m pip install pyserial   # capture_serial.py
```

- `pio` — builds firmware and manages toolchains per-project; nothing global.
- `picotool` — talks to RP2040/RP2350 boards over USB (identify, reboot into
  bootloader, flash). Only needed for Pico-family boards.
- `esptool` — identifies Espressif chips (`esptool --port <port> chip-id`).

## 1. Identify what you plugged in

```sh
pio device list          # serial ports + USB VID:PID
```

Vendor IDs seen in this folder's targets:

- `2E8A` Raspberry Pi: PID `0003`/`000F` = BOOTSEL bootloader (mass-storage
  drive mounted, ready to flash); a CDC PID = running firmware —
  `picotool info` names it.
- `303A` Espressif native USB; `1A86` WCH / `10C4` CP210x = a USB-UART
  bridge (often the only working monitor port — see the Heltec and T-Beam
  READMEs).
- `2886` Seeed Studio (XIAO family).

## 2. PlatformIO refresher

All commands run from a `<TARGET>/harness/` folder (where `platformio.ini`
is):

```sh
pio run                  # build (first run downloads platform + toolchain)
pio run -t upload        # build + flash (auto-detects the port)
pio device monitor       # serial terminal (Ctrl-C to exit)
```

`platformio.ini` is the whole config. Every env pins its platform (git
commit or release artifact) so a re-run rebuilds with the same toolchain,
and sets a unique `HW_TEST_TARGET` name — that name appears in the
firmware's output record and the capture verifies it.

## 3. Run a test end-to-end

One command from `validation/cpu/`:

```sh
sh run_test.sh <TARGET_FOLDER>                       # e.g. XIAO_RP2040_DET
sh run_test.sh <TARGET_FOLDER> /dev/cu.usbmodemXXX   # explicit port
WINDOW=60 sh run_test.sh <TARGET_FOLDER>             # widen capture window
```

It runs prepare → build → flash → capture, writes
`<TARGET>/results/<date>-<target-name>.txt` with a provenance header
(firmware and prepared-source SHA-256s, env, platform pin, host tool
versions) followed by the transcript, and exits non-zero unless the capture
proved a PASS **from the expected target name**. Capture-port identifiers
are redacted from the saved transcript. The capture script writes the result
file itself; do NOT hand-roll `capture | tee file; echo $?` — in a plain
shell `$?` after a pipeline is `tee`'s status and the check silently stops
being fail-closed.

Manual equivalent (what run_test.sh does):

```sh
cd validation/cpu/<TARGET>/harness
sh prepare.sh            # copy sources + stamp git HEAD + compute host pin
pio run
pio run -t upload
cd ..
python3 ../capture_serial.py --expect-target <name> --out results/<file>.txt
```

Always re-run `prepare.sh` after pulling — it is what makes the firmware's
`tree <commit>` stamp and its pinned expected value truthful.

## 4. How the pins work

The harness sources carry no committed expected values; `prepare.sh`
computes them at build time and fails closed:

- `*_DET`: prepare.sh builds `fp_determinism.c` on the host (native
  `__int128` backend), runs it, verifies the hash against the committed
  golden `tests/determinism_golden.txt` (aborting if the host has drifted),
  and pins it into `include/det_pin.h`. The on-target compare is therefore
  a genuine cross-backend (host `__int128` vs portable two-limb) and
  cross-ISA check anchored to the committed golden.
- `*_GPT`: prepare.sh bakes the repo's committed `model.mgw` into
  `include/model_mgw.h` (8-byte-aligned const array in flash rodata),
  builds the host reference (`hostref/host_main.c` + `microgpt_int.c` with
  `-DMGPT_NO_TRAIN -DMGPT_NO_MAIN`), runs it, and pins the FNV-1a 64 hash
  of the 20-sample stream into `include/gpt_pin.h`. The firmware calls
  `mgpt_load_mem()` (zero-copy: weights are read through memory-mapped
  flash, never RAM) + `mgpt_generate_sample()` and must reproduce the
  stream byte-for-byte, PRNG included.

The wrapper (`harness/src/main.cpp`) hardcodes nothing; it is essentially
identical across targets except where a core needs an extra include
(nRF52: TinyUSB).

## 5. Flashing: per-family notes

- **RP2040/RP2350 (arduino-pico core)**: `pio run -t upload` reboots the
  running firmware into the bootloader via a 1200-baud serial touch. If
  unknown firmware ignores it: `picotool reboot -f -u`, or hold BOOTSEL
  while plugging in and `picotool load -v -x .pio/build/<env>/firmware.uf2`.
  The Pico 2 runs both harnesses twice — once per ISA mode (`PICO2_ARM_*`
  vs `PICO2_RISCV_*`, `board_build.mcu = rp2350-riscv`).
- **Espressif**: esptool auto-resets over the serial handshake lines — no
  button, works over any firmware. ESP32-C6 and Heltec V3 capture quirk:
  monitor on the UART-bridge port with `-DARDUINO_USB_CDC_ON_BOOT=0`
  (C6) — the Heltec V3 only wires the CP2102 bridge anyway.
- **nRF52840 with Adafruit UF2/DFU bootloader**: factory or crashed
  firmware ignores the 1200-baud touch, and adafruit-nrfutil then prints
  "Target is not in DFU mode" — **while PlatformIO still banners
  SUCCESS**. Never trust the flasher's banner; trust only "Device
  programmed. Activating new firmware." and the capture's PASS/FAIL.
  Manual bootloader entry = double-tap RST (mouse-double-click rhythm),
  the board re-enumerates as the bootloader, and `pio run -t upload` then
  performs a real DFU. Once a healthy TinyUSB firmware is on, re-flashing
  needs no button. Also note the Adafruit core gives `loop()` a fixed
  4 KB FreeRTOS stack — see `XIAO_NRF52840_GPT/README.md` for the crash
  this caught.

## 6. Native Linux hosts (no PlatformIO)

`native_check.sh` is the ssh analogue of the MCU harnesses for anything
that runs Linux and gcc (a Raspberry Pi, an x86 server):

```sh
sh native_check.sh <user@host> <label> <TARGET_FOLDER> [--train]
# e.g.
sh native_check.sh pi@raspberrypi.local pi1-bplus PI_1_MODEL_B_PLUS
```

It copies the sources + committed `model.mgw` over ssh, builds with the
distro's native gcc, runs the determinism gate on both backends (default
and `-DFP_MATH_FORCE_PORTABLE`), runs `--load model.mgw` inference, and
byte-compares the output against the same build on the local host —
fail-closed, transcript into `<TARGET>/results/`. `--train` additionally
runs full training and byte-compares both the training stdout and the
saved `.mgw` against the committed one in the same run (a few seconds on
x86; about 10 minutes on the tested Pi 1).

## 7. Troubleshooting

- **No serial port after flashing**: give it a few seconds to re-enumerate
  (`capture_serial.py` retries an explicit port for 20 s). Still nothing →
  the sketch may have crashed before `Serial.begin`; reflash via the
  family's manual bootloader entry.
- **Port busy**: something else (a monitor, another capture) holds it.
- **capture FAILs with a record visible in the transcript**: check the
  `target` name — wrong firmware or wrong port.
- **IDE shows `'Arduino.h' file not found`** in `main.cpp`: harmless — the
  IDE indexer doesn't know PlatformIO's include paths; `pio run` is the
  truth.
- **nRF52840 link error `undefined reference to 'Serial'`**: `-DUSE_TINYUSB`
  missing from `build_flags`.
- **GPT firmware seems to hang**: the 20 sample lines print only after the
  first full generation pass; on slow targets give it `WINDOW=60`. The
  capture reads until the verdict line (it is not line-count-limited),
  so slow output is fine as long as the window covers it.
