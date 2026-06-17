#!/usr/bin/env python3
"""Bring up the device for training: forward the gadget/state/action ports and inject the agent.

This replaces the hand-run `adb forward` block. The author's serial and tool paths are
NOT hardcoded — pass them via flags or environment variables.

For >1 phone, run this once per device with a distinct ``--device-index``: the in-phone
ports stay 27042/9999/9998, but the *host* ports are offset by ``index * 100`` so two
devices never collide on the host. Device i's env then connects to 9999+i*100 / 9998+i*100
(this matches ``templerun_rl.constants.device_ports(i)`` and ``train.py --num-envs``).

Examples:
    python scripts/setup_device.py                       # single device, adb on PATH
    python scripts/setup_device.py --serial e4de25a1
    ANDROID_SERIAL=e4de25a1 ADB=D:/sdk/platform-tools/adb.exe python scripts/setup_device.py
    python scripts/setup_device.py --no-inject           # just forward ports
    # two phones (run each in its own terminal; leave both running):
    python scripts/setup_device.py --serial PHONE_A --device-index 0
    python scripts/setup_device.py --serial PHONE_B --device-index 1
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

# Host-port stride between devices; mirrors templerun_rl.constants.DEVICE_PORT_STRIDE
# (kept inline so this bring-up script has no dependency on the installed package).
PORT_STRIDE = 100


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adb", default=os.environ.get("ADB", "adb"), help="path to adb (default: $ADB or 'adb')")
    p.add_argument("--serial", default=os.environ.get("ANDROID_SERIAL"),
                   help="device serial (default: $ANDROID_SERIAL; omit if only one device)")
    # These are the *in-phone* ports (same on every device); the host side is offset.
    p.add_argument("--gadget-port", type=int, default=27042, help="in-phone gadget port")
    p.add_argument("--state-port", type=int, default=9999, help="in-phone state port")
    p.add_argument("--action-port", type=int, default=9998, help="in-phone action port")
    p.add_argument("--device-index", type=int, default=0,
                   help=f"0-based device slot; host ports = in-phone port + index*{PORT_STRIDE}")
    p.add_argument("--port-offset", type=int, default=None,
                   help="raw host-port offset (overrides --device-index*stride if given)")
    p.add_argument("--script", default=str(DEFAULT_SCRIPT), help="Frida agent JS to inject")
    p.add_argument("--python", default=sys.executable, help="python used to run gadget_inject.py")
    p.add_argument("--no-inject", action="store_true", help="only set up adb forwards, do not inject")
    return p.parse_args()


def host_offset(args: argparse.Namespace) -> int:
    return args.port_offset if args.port_offset is not None else args.device_index * PORT_STRIDE


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
        sys.exit(f"Multiple devices ({devices}); pass --serial / set $ANDROID_SERIAL "
                 "(and a distinct --device-index per device).")


def forward(args: argparse.Namespace) -> None:
    """Map host port (in-phone port + offset) -> the fixed in-phone port, per device."""
    off = host_offset(args)
    for phone_port in (args.gadget_port, args.state_port, args.action_port):
        host_port = phone_port + off
        subprocess.run(adb_base(args) + ["forward", f"tcp:{host_port}", f"tcp:{phone_port}"], check=True)
        print(f"forwarded host tcp:{host_port} -> device tcp:{phone_port}")


def inject(args: argparse.Namespace) -> None:
    off = host_offset(args)
    host_gadget = args.gadget_port + off
    host_state, host_action = args.state_port + off, args.action_port + off
    cmd = [args.python, str(GADGET_INJECT), args.script, f"127.0.0.1:{host_gadget}"]
    print(f"injecting: {' '.join(cmd)}")
    print(f"(leave this running; agent prints 'ready'. This device's env uses "
          f"state_port={host_state} action_port={host_action})\n")
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
