#!/usr/bin/env python3
"""Keep N phones healthy during long real-time training (self-healing).

Real hardware drops out: the screen sleeps, HyperOS kills+relaunches the app, the
gadget session dies. This watchdog watches each device's state stream and, when one
goes silent, heals it:
  1. re-apply keep-awake (svc power stayon / appops + screen_off_timeout),
  2. wake + foreground the game (``am start``),
  3. if the app actually restarted (PID changed / gone), **re-inject** the Frida agent.

Paired with the env's auto-reconnecting ``StateClient`` and ``freeze_timeout``, a hiccup
makes training *pause* (the worker blocks waiting for frames) and then resume once the
watchdog restores the stream -- instead of crashing the whole ``SubprocVecEnv`` run.

Run it alongside training, after ``setup_device.py`` brought the devices up:
    python scripts/device/watchdog.py --serials SERIAL_A,SERIAL_B        # position = device index

Note: ``adb input keyevent WAKEUP`` is blocked on HyperOS, so a *physically* dark screen
may still need a tap; the keep-awake step prevents it from recurring. The watchdog logs
loudly when it cannot wake a device on its own.
"""
from __future__ import annotations

import sys as _sys, pathlib as _pl  # repo bootstrap: src/, cpp_sim/, scripts/*/, experiments/
_R = next(p for p in _pl.Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
_sys.path[:0] = [str(_R / "src"), str(_R / "cpp_sim"), *(str(d) for d in sorted((_R / "scripts").iterdir()) if d.is_dir()), str(_R / "experiments")]
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from templerun_rl import constants as C
from templerun_rl.client import StateClient

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCRIPT = REPO_ROOT / "frida" / "temple_run_hook.js"
GADGET_INJECT = REPO_ROOT / "frida" / "gadget_inject.py"
PKG = "com.imangi.templerun"
ACTIVITY = "com.imangi.unityactivity.ImangiUnityActivity"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adb", default=os.environ.get("ADB", "adb"), help="path to adb (default: $ADB or 'adb')")
    p.add_argument("--serials", required=True,
                   help="comma-separated device serials; list position = device index (0,1,...)")
    p.add_argument("--script", default=str(DEFAULT_SCRIPT), help="Frida agent JS to re-inject")
    p.add_argument("--python", default=sys.executable, help="python used to run gadget_inject.py")
    p.add_argument("--silent-secs", type=float, default=15.0,
                   help="stream silence (s) before a device is considered down and healed")
    p.add_argument("--poll-secs", type=float, default=5.0, help="health-check interval (s)")
    p.add_argument("--no-reinject", action="store_true",
                   help="re-apply keep-awake / wake only; never re-inject the agent")
    p.add_argument("--log-dir", default=str(REPO_ROOT / "runs"), help="where re-inject logs are written")
    return p.parse_args()


def sh(adb: str, serial: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([adb, "-s", serial, *args], capture_output=True, text=True)


def keep_awake(adb: str, serial: str) -> None:
    sh(adb, serial, "shell", "svc", "power", "stayon", "true")
    sh(adb, serial, "shell", "appops", "set", "com.android.shell", "android:write_settings", "allow")
    sh(adb, serial, "shell", "settings", "put", "system", "screen_off_timeout", "2147483647")


def pid_of(adb: str, serial: str) -> str | None:
    out = sh(adb, serial, "shell", "pidof", PKG).stdout.strip()
    return out or None


def wake_and_foreground(adb: str, serial: str) -> None:
    sh(adb, serial, "shell", "am", "start", "-n", f"{PKG}/{ACTIVITY}")


def reinject(args: argparse.Namespace, index: int) -> subprocess.Popen:
    gport = C.device_ports(index)["gadget_port"]
    log_path = os.path.join(args.log_dir, f"watchdog_inject_dev{index}.log")
    log = open(log_path, "ab")
    cmd = [args.python, str(GADGET_INJECT), args.script, f"127.0.0.1:{gport}"]
    return subprocess.Popen(cmd, stdout=log, stderr=log)


def main() -> None:
    args = parse_args()
    # Flush per line: this is a long-running monitor whose output is usually redirected to a
    # file/pipe (block-buffered by default), and its logs must appear promptly.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:  # noqa: BLE001
        pass
    serials = [s.strip() for s in args.serials.split(",") if s.strip()]
    if not serials:
        sys.exit("no serials given")
    os.makedirs(args.log_dir, exist_ok=True)

    clients: list[StateClient] = []
    for i, s in enumerate(serials):
        port = C.device_ports(i)["state_port"]
        c = StateClient(C.STATE_HOST, port)
        try:
            c.connect()  # auto-reconnects thereafter, so a not-yet-ready device recovers on its own
        except Exception as e:  # noqa: BLE001
            print(f"[watchdog] device {i} ({s}) state:{port} not reachable yet ({e}); will keep retrying")
        clients.append(c)

    last_pid = [pid_of(args.adb, s) for s in serials]
    injectors: list[subprocess.Popen | None] = [None] * len(serials)
    healing = [False] * len(serials)
    print(f"[watchdog] monitoring {len(serials)} device(s); silent > {args.silent_secs:.0f}s triggers healing. "
          f"Ctrl-C to stop.")

    try:
        while True:
            time.sleep(args.poll_secs)
            for i, s in enumerate(serials):
                stale = clients[i].staleness()
                if stale <= args.silent_secs:
                    if healing[i]:
                        print(f"[watchdog] device {i} ({s}) RECOVERED -- stream resumed.")
                        healing[i] = False
                    continue

                # --- device is silent: heal it ---
                healing[i] = True
                print(f"[watchdog] device {i} ({s}) silent {stale:.0f}s -> healing")
                keep_awake(args.adb, s)
                wake_and_foreground(args.adb, s)

                pid = pid_of(args.adb, s)
                restarted = (pid != last_pid[i]) or (pid is None)
                spawned_dead = injectors[i] is not None and injectors[i].poll() is not None
                last_pid[i] = pid

                if not args.no_reinject and (restarted or spawned_dead):
                    why = "app restarted" if restarted else "prior re-inject exited"
                    print(f"[watchdog] device {i} ({s}) {why} (pid={pid}) -> re-injecting agent")
                    if injectors[i] is not None and injectors[i].poll() is None:
                        injectors[i].terminate()
                    try:
                        injectors[i] = reinject(args, i)
                    except Exception as e:  # noqa: BLE001
                        print(f"[watchdog] device {i} ({s}) re-inject failed: {e}")
                elif pid is not None:
                    print(f"[watchdog] device {i} ({s}) app alive (pid={pid}) but paused -- "
                          f"if the screen is physically off, it may need a manual tap")
    except KeyboardInterrupt:
        print("\n[watchdog] stopping.")
    finally:
        for c in clients:
            c.close()
        for inj in injectors:
            if inj is not None and inj.poll() is None:
                inj.terminate()


if __name__ == "__main__":
    main()
