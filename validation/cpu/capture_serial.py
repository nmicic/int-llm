#!/usr/bin/env python3
"""Fail-closed capture of the hardware profile test's serial output.

Reads the target's USB CDC port, parses the structured result record the
firmware prints:

    target <name> arch <isa> tree <commit> PROFILE <profile> hash <16 hex> (...)
    profile: PASS (...)

and exits 0 ONLY if a structured record was seen, a "profile: PASS" line
followed it, and (when --expect-target is given) the record's target name
matches. Everything read is echoed to stdout and, with --out, appended to the
result file — so the process that writes the transcript is the same one whose
exit code is checked (no tee-in-a-pipeline exit-status hole).

Profiles are named ("all", "core", ...) and carry no
case count — the PROFILE field in the record is a name, not a number.
"""
import argparse
import re
import sys
import time

import serial
import serial.tools.list_ports

RECORD_RE = re.compile(
    r"^target (?P<target>\S+) arch (?P<arch>\S+) tree (?P<tree>\S+) "
    r"PROFILE (?P<profile>\S+) hash (?P<hash>[0-9a-f]{16})"
)


def public_port_label(port):
    """Describe the port without recording a unit-specific USB identifier."""
    if "usbmodem" in port:
        return "usbmodem-<REDACTED>"
    if "usbserial" in port:
        return "usbserial-<REDACTED>"
    return "<REDACTED>"


def find_port(tries=40, delay=0.5):
    for _ in range(tries):
        ports = [p.device for p in serial.tools.list_ports.comports()
                 if "usbmodem" in p.device]
        if ports:
            return ports[0]
        time.sleep(delay)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", help="serial device; default: first usbmodem "
                    "(pass explicitly when several boards are attached)")
    ap.add_argument("--expect-target", help="required target name in the record")
    ap.add_argument("--out", help="append captured lines to this file")
    ap.add_argument("--window", type=float, default=15.0,
                    help="seconds to listen (default 15)")
    args = ap.parse_args()

    port = args.port or find_port()
    if port is None:
        print("FAIL: no usbmodem serial port found", file=sys.stderr)
        return 2

    sink = open(args.out, "a") if args.out else None

    def emit(line):
        print(line)
        if sink:
            sink.write(line + "\n")

    port_label = public_port_label(port)
    emit(f"port: {port_label}")
    # After a reflash the board re-enumerates; an explicitly-given port can
    # be briefly absent (seen on the Pico 2 ARM->RISC-V mode switch). Retry
    # the open instead of failing on the first attempt.
    s = None
    for _ in range(40):
        try:
            s = serial.Serial(port, 115200, timeout=8)
            break
        except serial.SerialException:
            time.sleep(0.5)
    if s is None:
        emit(f"FAIL: could not open {port_label} after 20 s")
        if sink:
            sink.close()
        return 2
    t0 = time.time()
    record = None
    saw_pass = False
    n = 0
    # Unlike the astro-nav-int original (which stopped after 4 lines — its
    # firmware prints only record + verdict), the GPT harness here prints 20
    # sample lines before the record, so read until the verdict line arrives
    # (or the window / a generous line cap runs out).
    while time.time() - t0 < args.window and n < 200:
        ln = s.readline().decode(errors="replace").strip()
        if not ln:
            continue
        n += 1
        emit(ln)
        m = RECORD_RE.match(ln)
        if m:
            record = m.groupdict()
        elif ln.startswith("profile: PASS") and record is not None:
            saw_pass = True
            break
        elif ln.startswith("profile: FAIL") and record is not None:
            break
    s.close()

    if record is None:
        emit("FAIL: no structured result record seen")
        rc = 1
    elif not saw_pass:
        emit("FAIL: record seen but no on-target PASS line")
        rc = 1
    elif args.expect_target and record["target"] != args.expect_target:
        emit(f"FAIL: target {record['target']!r} != expected "
             f"{args.expect_target!r} (wrong firmware or wrong port?)")
        rc = 1
    else:
        emit(f"capture: OK target={record['target']} arch={record['arch']} "
             f"tree={record['tree']} profile={record['profile']} "
             f"hash={record['hash']}")
        rc = 0
    if sink:
        sink.close()
    return rc


if __name__ == "__main__":
    sys.exit(main())
