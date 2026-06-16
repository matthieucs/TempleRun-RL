#!/usr/bin/env python3
"""Bring up the device for training: forward the gadget/state/action ports and inject the agent.

This replaces the hand-run `adb forward` block. The author's serial and tool paths are
NOT hardcoded — pass them via flags or environment variables.

Examples:
    python scripts/setup_device.py                       # single device, adb on PATH
    python scripts/setup_device.py --serial e4de25a1
    ANDROID_SERIAL=e4de25a1 ADB=D:/sdk/platform-tools/adb.exe python scripts/setup_device.py
    python scripts/setup_device.py --no-inject           # just forward ports
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCRIPT = REPO_ROOT / "frida" / "temple_run_hook.js"
GADGET_INJECT = REPO_ROOT / "frida" / "gadget_inject.py"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adb", default=os.environ.get("ADB", "adb"), help="path to adb (default: $ADB or 'adb')")
    p.add_argument("--serial", default=os.environ.get("ANDROID_SERIAL"),
                   help="device serial (default: $ANDROID_SERIAL; omit if only one device)")
    p.add_argument("--gadget-port", type=int, default=27042)
    p.add_argument("--state-port", type=int, default=9999)
    p.add_argument("--action-port", type=int, default=9998)
    p.add_argument("--script", default=str(DEFAULT_SCRIPT), help="Frida agent JS to inject")
    p.add_argument("--python", default=sys.executable, help="python used to run gadget_inject.py")
    p.add_argument("--no-inject", action="store_true", help="only set up adb forwards, do not inject")
    return p.parse_args()


def adb_base(args: argparse.Namespace) -> list[str]:
    cmd = [args.adb]
    if args.serial:
        cmd += ["-s", args.serial]
    return cmd


def check_device(args: argparse.Namespace) -> None:
    try:
        out = subprocess.run([args.adb, "devices"], capture_output=True, text=True, check=True).stdout
    except FileNotFoundError:
        sys.exit(f"adb not found at '{args.adb}'. Install platform-tools or pass --adb / set $ADB.")
    devices = [ln.split("\t")[0] for ln in out.splitlines()[1:] if "\tdevice" in ln]
    if not devices:
        sys.exit("No authorized device. Connect the phone, enable USB debugging, and accept the prompt.")
    if not args.serial and len(devices) > 1:
        sys.exit(f"Multiple devices ({devices}); pass --serial / set $ANDROID_SERIAL.")


def forward(args: argparse.Namespace) -> None:
    for port in (args.gadget_port, args.state_port, args.action_port):
        subprocess.run(adb_base(args) + ["forward", f"tcp:{port}", f"tcp:{port}"], check=True)
        print(f"forwarded tcp:{port} -> device tcp:{port}")


def inject(args: argparse.Namespace) -> None:
    cmd = [args.python, str(GADGET_INJECT), args.script, f"127.0.0.1:{args.gadget_port}"]
    print(f"injecting: {' '.join(cmd)}")
    print("(leave this running; you should see 'ready: state->9999 action<-9998')\n")
    # Stream the injector's output so the operator sees the gadget become ready.
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    check_device(args)
    forward(args)
    if args.no_inject:
        print("\nPorts forwarded. Skipping injection (--no-inject).")
        return
    inject(args)


if __name__ == "__main__":
    main()
