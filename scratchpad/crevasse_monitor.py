#!/usr/bin/env python3
"""WAVE 15 (USER-directed): MONITOR THE MID->HIGH CREVASSE per run.

USER (live): "You will still hit the mid to high transition problem. It works but it's not very
clean. The character often hits the wall which slows it down... I advise you to monitor this during
a run."

The wall-clip is the level-3 cap GRAZE: a non-fatal Stumble (velocity x0.85) fired when the jump arc
has already DESCENDED below the raised cap top at cap entry (device rule: world_y < ground_height).
It does not kill, so it never shows up in a death mix or a median -- it just quietly bleeds speed,
and a slower runner mistimes everything after it.

Reports per capture, split by the level the cap sits at (device: lvl-3 caps graze ~32%, lvl-2 ~0%):
  * GRAZE RATE per crevasse encounter                (the number the user is watching)
  * takeoff cell_progress at the approach            (EARLY takeoff => descending at the cap => graze;
                                                      human is LATE + speed-adaptive: 0.52 -> 0.65,
                                                      policies measured EARLY + FLAT: ~0.27-0.35)
  * arc clearance at cap entry (world_y - ground_height; negative = below the cap top)
  * velocity lost to grazes per km                   (the actual cost)

Usage: python scratchpad/crevasse_monitor.py <capture globs...>
"""
from __future__ import annotations
import glob, sys, statistics as S
from collections import defaultdict
from hole_landing_forensic import load_eps, cell_seq

CREV_PT = {33, 34, 35, 36}
CREV_OBST = 7


def analyse(paths, tag):
    enc = []
    km = 0.0
    for path in paths:
        hz = 30.0 if "human" in path else 10.0
        for epk, ep in load_eps(path):
            cells = cell_seq(ep)
            km += max(float(f.get("distance") or 0) for f in ep) / 1000.0
            takeoff = None
            for i, fr in enumerate(ep):
                jmp = bool(fr.get("is_jumping"))
                pj = bool(ep[i - 1].get("is_jumping")) if i else False
                if jmp and not pj:
                    src = ep[i - 1] if i else fr
                    takeoff = (float(src.get("cell_progress") or 0.0), float(src.get("velocity") or 0.0))
                is_crev = (fr.get("path_type") in CREV_PT) or (int(fr.get("obstacle") or 0) == CREV_OBST)
                prev_crev = i and ((ep[i - 1].get("path_type") in CREV_PT)
                                   or (int(ep[i - 1].get("obstacle") or 0) == CREV_OBST))
                if not is_crev or prev_crev:
                    continue
                # the CAP is the LAST cell of the crevasse structure and carries the NEW zone level —
                # reading the FIRST cell mis-attributes a mid->high transition to level 2.
                lvl = int(fr.get("path_level") or 2)
                for j in range(i, min(len(ep), i + 12)):
                    f2 = ep[j]
                    if (f2.get("path_type") in CREV_PT) or (int(f2.get("obstacle") or 0) == CREV_OBST):
                        lvl = max(lvl, int(f2.get("path_level") or 2))
                    else:
                        break
                # scan the crossing: clearance above the cap + whether a stumble fired + velocity drop
                clear_min, grazed, died = None, False, False
                v_in = float(fr.get("velocity") or 0.0)
                v_out = v_in
                for j in range(i, min(len(ep), i + int(1.2 * hz))):
                    f2 = ep[j]
                    # clearance only matters while AIRBORNE: on the ground world_y == ground_height,
                    # so including grounded frames pins the minimum at exactly 0 and hides the arc.
                    wy, gh = f2.get("world_y"), f2.get("ground_height")
                    if wy is not None and gh is not None and f2.get("is_jumping"):
                        c = float(wy) - float(gh)
                        clear_min = c if clear_min is None else min(clear_min, c)
                    if f2.get("is_stumbling") and not ep[max(0, j - 1)].get("is_stumbling"):
                        grazed = True
                    if f2.get("is_dead"):
                        died = True
                        break
                    v_out = float(f2.get("velocity") or v_out)
                enc.append({"lvl": lvl, "grazed": grazed, "died": died, "clear": clear_min,
                            "tk_cp": takeoff[0] if takeoff else None,
                            "v": takeoff[1] if takeoff else v_in,
                            "dv": v_in - v_out})
    print(f"\n=== {tag} ===  {len(enc)} crevasse encounters over {km:.1f} km")
    for lvl in sorted({e["lvl"] for e in enc}):
        xs = [e for e in enc if e["lvl"] == lvl]
        g = [e for e in xs if e["grazed"]]
        d = [e for e in xs if e["died"]]
        cl = [e["clear"] for e in xs if e["clear"] is not None]
        print(f"  level {lvl}: n={len(xs):4}  GRAZE {len(g):3} ({100*len(g)/len(xs):5.1f}%)  "
              f"kill {len(d):2} ({100*len(d)/len(xs):4.1f}%)"
              + (f"  arc clearance at cap: median {S.median(cl):+6.2f}" if cl else ""))
    tk = [(e["v"], e["tk_cp"]) for e in enc if e["tk_cp"] is not None]
    if tk:
        print("  TAKEOFF cell_progress by speed (human is LATE+adaptive 0.52->0.65; early => descending at the cap):")
        buckets = defaultdict(list)
        for v, cp in tk:
            buckets[int(v // 30) * 30].append(cp)
        for k in sorted(buckets):
            b = sorted(buckets[k])
            if len(b) >= 3:
                print(f"     v{k:3}-{k+29:3}  n={len(b):3}  median cp {b[len(b)//2]:.2f}")
    lost = sum(e["dv"] for e in enc if e["grazed"])
    print(f"  velocity bled to grazes: {lost:.0f} units over {km:.1f} km = {lost/max(1e-9,km):.1f} per km")


if __name__ == "__main__":
    gl = sys.argv[1:] or ["captures/w14_v78_holes.jsonl"]
    for g in gl:
        ps = sorted(glob.glob(g))
        if ps:
            analyse(ps, g)
