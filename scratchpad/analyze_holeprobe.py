#!/usr/bin/env python
"""Analyze the long-hole takeoff sweep (capture_calib --gap-override).

For each controlled jump fired near a wide gap, find the takeoff (cell_progress + velocity at the
`gapjump@` frame) and the OUTCOME (cleared the gap vs fell in within the next few cells). Separate
FLAT long-holes (path_type 12/13/15/16) from ZONE crevasses (33-36) — the sim models them differently.
Report the survivable takeoff-cp window per velocity band. This is the device calibration for the sim
wide-HOLE reach margin (the v35 multi-hole regression fix)."""
import json
import sys
from pathlib import Path

import numpy as np

CREV = set(range(33, 37))
FLATHOLE = {11, 12, 13, 14, 15, 16}


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def events(frames):
    """Each gapjump@ frame -> (takeoff_cp, velocity, gap_kind, cleared)."""
    out = []
    for i, f in enumerate(frames):
        cmd = f.get("narrow_cmd", "") or ""
        if not cmd.startswith("gapjump@"):
            continue
        cp = f.get("cell_progress")
        v = f.get("velocity")
        if cp is None or v is None:
            continue
        # gap kind = the path_type of the gap the jump is FOR (this or the next cell that is a gap)
        kind = None
        for g in frames[i:i + 6]:
            pt = int(g.get("path_type") or -1)
            if pt in CREV:
                kind = "crevasse"; break
            if pt in FLATHOLE:
                kind = "flathole"; break
        # outcome: died within the next ~10 frames (the gap span) with a fall?
        died = False
        for g in frames[i + 1:i + 12]:
            if g.get("is_dead"):
                died = True; break
            # left the gap region onto a clean straight -> survived
            pt = int(g.get("path_type") or -1)
            if pt in (1, 2) and not g.get("next_wide_gap") and g is not frames[i + 1]:
                break
        out.append((float(cp), float(v), kind, not died))
    return out


def main():
    caps = sys.argv[1:] or ["captures/holeprobe.jsonl"]
    ev = []
    for c in caps:
        if Path(c).exists():
            ev += events(load(c))
    print(f"{len(ev)} controlled gap jumps")
    for kind in ("flathole", "crevasse"):
        sub = [(cp, v, ok) for cp, v, k, ok in ev if k == kind]
        if not sub:
            print(f"\n[{kind}] none"); continue
        print(f"\n=== {kind} (n={len(sub)}) ===")
        for lo, hi in ((100, 150), (150, 180), (180, 215)):
            band = [(cp, ok) for cp, v, ok in sub if lo <= v < hi]
            if not band:
                continue
            clear = sorted(cp for cp, ok in band if ok)
            fall = sorted(cp for cp, ok in band if not ok)
            cw = f"[{min(clear):.2f},{max(clear):.2f}] med {np.median(clear):.2f}" if clear else "none"
            fl = f"{[round(x, 2) for x in fall]}" if fall else "none"
            print(f"  v{lo}-{hi}: n={len(band)} CLEARED cp {cw} (n={len(clear)}) | FELL cp {fl}")
        # overall survivable window
        clear = [cp for cp, v, ok in sub if ok]
        fall = [cp for cp, v, ok in sub if not ok]
        if clear and fall:
            print(f"  -> takeoff too-EARLY fatal below cp ~{np.percentile([c for c in clear],10):.2f}; "
                  f"fall cps {sorted(round(x,2) for x in fall)}")


if __name__ == "__main__":
    main()
