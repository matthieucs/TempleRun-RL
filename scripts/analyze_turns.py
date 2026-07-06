#!/usr/bin/env python3
"""Per-TURN outcome analysis for a capture (ideally from the turns-only forced track, hook >=16e).
Reports the metric that matters for turn reliability: of the junctions faced, how many the policy
TURNED (facing changed) and SURVIVED, plus the swipe cell_progress distribution (should cluster at
the ~0.44 safe center) and the death causes.

    .venv/bin/python scripts/analyze_turns.py captures/tilt_v3_turns.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

TL, TR = 4, 5


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("jsonl")
    a = p.parse_args()
    frames = [json.loads(l) for l in Path(a.jsonl).read_text().splitlines() if l.strip()]
    eps = defaultdict(list)
    for f in frames:
        eps[f.get("ep", 0)].append(f)

    junctions = executed = survived = 0
    swipe_cps: list = []
    double_bursts = 0
    water_deaths = 0
    dists = []
    death_cause = Counter()

    for e, fr in eps.items():
        dists.append(max(float(f.get("distance") or 0) for f in fr))
        # a junction = a maximal run of is_turn True; did the policy swipe in it + did facing change?
        i = 0
        burst = 0
        for j, f in enumerate(fr):
            if f.get("action") in (TL, TR):
                burst += 1
            elif burst:
                if burst >= 2:
                    double_bursts += 1
                burst = 0
        while i < len(fr):
            if fr[i].get("is_turn"):
                k = i
                fac0 = fr[i - 1].get("facing") if i > 0 else fr[i].get("facing")
                swiped_cp = None
                while k < len(fr) and (fr[k].get("is_turn") or (swiped_cp is not None and k < i + 6)):
                    if fr[k].get("action") in (TL, TR) and swiped_cp is None:
                        swiped_cp = float(fr[k].get("cell_progress") or 0.0)
                    k += 1
                junctions += 1
                if swiped_cp is not None:
                    swipe_cps.append(swiped_cp)
                # executed = facing changed within a few frames of the junction
                win = fr[i:i + 8]
                did = any(g.get("facing") != fac0 and g.get("facing") for g in win)
                executed += int(did)
                # survived this junction = not dead within the window
                if not any(g.get("is_dead") for g in win):
                    survived += 1
                i = k
            else:
                i += 1
        if fr[-1].get("is_dead"):
            tail = fr[-8:]
            if any(g.get("action") in (TL, TR) or g.get("is_turn") for g in tail):
                water_deaths += 1
                death_cause["turn"] += 1
            else:
                death_cause[f"other(pt{fr[-1].get('path_type')})"] += 1

    def pct(x, n):
        return f"{100.0 * x / max(1, n):.0f}%"

    print(f"episodes {len(eps)}  median dist {np.median(dists):.0f}")
    print(f"junctions faced      {junctions}")
    print(f"  turned (facing chg) {executed}  ({pct(executed, junctions)})")
    print(f"  survived junction   {survived}  ({pct(survived, junctions)})")
    print(f"double-turn bursts   {double_bursts}")
    print(f"turn/water deaths    {water_deaths} of {len(eps)} eps")
    if swipe_cps:
        s = np.array(swipe_cps)
        print(f"swipe cell_progress  med {np.median(s):.2f}  p10 {np.percentile(s, 10):.2f}  "
              f"p90 {np.percentile(s, 90):.2f}  (device-safe center ~0.44)")
    print(f"death causes         {dict(death_cause)}")


if __name__ == "__main__":
    main()
