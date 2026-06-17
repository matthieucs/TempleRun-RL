#!/usr/bin/env python3
"""Sanity-check that N phones stream state *in parallel* before training across them.

This does NOT send actions or touch SB3 — it just opens one StateClient per device slot
(``device_ports(i)``) and watches their frame counters advance concurrently. If every
device's ``seq`` keeps rising over the window, the per-device forwards + gadgets are all
live and independent, and ``train.py --num-envs N`` will work.

Bring the devices up first (one terminal each):
    python scripts/setup_device.py --serial PHONE_A --device-index 0
    python scripts/setup_device.py --serial PHONE_B --device-index 1
Then:
    python scripts/multi_device_probe.py --num-devices 2 --seconds 5
"""
from __future__ import annotations

import argparse
import time

from templerun_rl import constants as C
from templerun_rl.client import StateClient


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num-devices", type=int, default=2)
    p.add_argument("--seconds", type=float, default=5.0)
    p.add_argument("--host", default=C.STATE_HOST)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    clients = []
    for i in range(args.num_devices):
        ports = C.device_ports(i)
        c = StateClient(args.host, ports["state_port"])
        try:
            c.connect()
        except Exception as e:  # noqa: BLE001
            raise SystemExit(
                f"device {i} (state_port={ports['state_port']}) not reachable: {e}\n"
                f"Run: python scripts/setup_device.py --serial <SERIAL> --device-index {i}"
            )
        clients.append((i, ports["state_port"], c))
        print(f"connected device {i} on state_port={ports['state_port']}")

    # Snapshot starting seq per device, then watch them advance.
    start = {i: c.latest()[1] for i, _, c in clients}
    print(f"\nwatching for {args.seconds:.0f}s (each device should keep producing frames)...\n")
    deadline = time.monotonic() + args.seconds
    while time.monotonic() < deadline:
        time.sleep(0.5)
        row = []
        for i, port, c in clients:
            state, seq = c.latest()
            frames = seq - start[i]
            score = (state or {}).get("score")
            row.append(f"dev{i}(:{port}) frames={frames:<4} score={score}")
        print("  " + " | ".join(row))

    print("\nresult:")
    ok = True
    for i, port, c in clients:
        _, seq = c.latest()
        frames = seq - start[i]
        verdict = "OK (streaming)" if frames > 0 else "STALLED (no new frames)"
        ok = ok and frames > 0
        print(f"  device {i} (:{port}): {frames} frames in {args.seconds:.0f}s -> {verdict}")
        c.close()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
