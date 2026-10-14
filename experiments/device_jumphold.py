#!/usr/bin/env python3
"""On-device held-jump rate from a capture_calib JSONL: fraction of JUMP action bytes that
immediately follow another JUMP byte (the hold that causes the lethal high-ground landing-frame
re-jump). Compares v59 (before jump_hold_penalty) vs v60 (after). Also reports high-ground jump
deaths (dt=4 Ledge / dt=1 Fall while ground_height!=0 nearby) as the lethal-outcome metric."""
import json, sys
from pathlib import Path
from collections import defaultdict


def num(x, d=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


for path in sys.argv[1:]:
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    eps = defaultdict(list)
    for r in rows:
        eps[r.get("ep")].append(r)
    jumps = held = 0
    hi_jump_deaths = 0
    dists = []
    for ep, frames in eps.items():
        frames.sort(key=lambda r: r.get("step", 0))
        last = 6
        for f in frames:
            a = int(num(f.get("action"), 6))
            if a == 2:
                jumps += 1
                if last == 2:
                    held += 1
            last = a
        d = frames[-1]
        dists.append(num(d.get("distance")))
        dt = int(num(d.get("death_type"), 0))
        # high-ground jump death: died dt=4/1 with a raised/lowered ledge (|gh|>0) + airborne in last frames
        near_hi = any(abs(num(g.get("ground_height"))) > 0.5 for g in frames[-8:])
        air = any(g.get("is_jumping") for g in frames[-6:])
        if dt in (1, 4) and near_hi and air:
            hi_jump_deaths += 1
    pct = 100 * held / jumps if jumps else 0
    import statistics as S
    print(f"{Path(path).name:<24} held-jump%={pct:5.1f}% ({held}/{jumps})  "
          f"hi-ground jump-deaths={hi_jump_deaths}/{len(eps)}  median_dist={S.median(dists):.0f}")
