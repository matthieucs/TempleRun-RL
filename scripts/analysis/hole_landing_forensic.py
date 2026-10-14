#!/usr/bin/env python3
"""WAVE 14: the SINGLE-HOLE (and general) ARC-LANDING measurement.

Device deaths (v78 ep3 @612, v74 ep3 @1722, v74 r2ep0/ep1/ep3 ...) all show the same shape: the
runner is AIRBORNE when a must-jump cell arrives, its arc is already descending, and it drops in.
The sim only tests `_jump_t > 0` for a 1-cell hole (sim_env.py:1844-1854) -- an almost-expired
jump crosses for free. The wide-gap landing model (arc must land past the far edge) is gated to
`wide_gap` cells only.

This measures the REAL device rule: for every hole encounter, how many cells `d` separate the
TAKEOFF position from the hole cell, and did the runner survive? The kill threshold predicted by
the wide-gap span fit is  d > span(v) - 1 + grace,  span(v) = a*v + b.

Usage: python scripts/analysis/hole_landing_forensic.py [capture globs...]
"""
from __future__ import annotations
import sys as _sys, pathlib as _pl  # repo bootstrap: src/, cpp_sim/, scripts/*/, experiments/
_R = next(p for p in _pl.Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
_sys.path[:0] = [str(_R / "src"), str(_R / "cpp_sim"), *(str(d) for d in sorted((_R / "scripts").iterdir()) if d.is_dir()), str(_R / "experiments")]
import glob, json, sys
from collections import defaultdict

HOLE_PT = {11, 12, 13, 14, 15, 16}
LONG_PT = {12, 13, 15, 16}          # 2-cell HoleLong
SPAN_A, SPAN_B = 0.0204, -0.94      # device span fit (cells) from the wave-8 jump_arc forensic
GRACE = 0.35


def load_eps(path):
    eps = defaultdict(list)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            eps[d.get("ep", 0)].append(d)
    out = []
    for k in sorted(eps):
        fs = eps[k]
        i = 0
        while i < len(fs) and fs[i].get("is_dead"):
            i += 1
        j = i
        while j < len(fs) and not fs[j].get("is_dead"):
            j += 1
        if j - i > 30:
            out.append((k, fs[i:min(j + 1, len(fs))]))
    return out


def cell_seq(ep):
    """Assign a monotonic cell counter to each frame (cell_progress resets mark a new cell)."""
    idx, out, prev = 0, [], None
    for fr in ep:
        cp = float(fr.get("cell_progress") or 0.0)
        if prev is not None and cp < prev - 0.25:
            idx += 1
        out.append(idx)
        prev = cp
    return out


def is_hole(fr):
    """Hole cell, keyed on the OBSTACLE enum (works on BOTH device captures and sim rows -- the
    sim emits holes on generic straight path_types, so path_type 11-16 only exists on device)."""
    o = fr.get("obstacle")
    if o is not None:
        return int(o) == 1
    return fr.get("path_type") in HOLE_PT


def is_wide(fr):
    w = fr.get("wide_gap")
    if w is not None:
        return bool(w)
    return fr.get("path_type") in LONG_PT


def analyse(paths, tag):
    events = []
    for path in paths:
        for epk, ep in load_eps(path):
            cells = cell_seq(ep)
            # takeoff bookkeeping: frame index -> (cell + cp) at the last liftoff
            takeoff_pos, takeoff_i = None, None
            for i, fr in enumerate(ep):
                jmp = bool(fr.get("is_jumping"))
                pj = bool(ep[i - 1].get("is_jumping")) if i else False
                if jmp and not pj:                      # liftoff frame
                    # the takeoff happened between i-1 and i; use the previous frame's position
                    src = ep[i - 1] if i else fr
                    takeoff_pos = cells[max(0, i - 1)] + float(src.get("cell_progress") or 0.0)
                    takeoff_i = i
                pt = fr.get("path_type")
                if not is_hole(fr):
                    continue
                if i and is_hole(ep[i - 1]):
                    continue      # same cell (already counted) or 2nd cell of a HoleLong
                v = float(fr.get("velocity") or 0.0)
                d = (cells[i] - takeoff_pos) if takeoff_pos is not None and jmp else None
                # outcome: died within this hole crossing (next 6 frames)
                died = False
                for j in range(i, min(len(ep), i + 6)):
                    if ep[j].get("is_dead"):
                        died = int(ep[j].get("death_type") or 0) in (1, 4)
                        break
                    if not is_hole(ep[j]) and j > i + 1:
                        break
                events.append({"src": path.replace("\\", "/").split("/")[-1], "ep": epk,
                               "dist": float(fr.get("distance") or 0), "v": v, "pt": pt,
                               "long": is_wide(fr), "airborne": jmp, "d": d,
                               "wy": float(fr.get("world_y") or 0.0), "died": died,
                               "since_takeoff": (i - takeoff_i) if takeoff_i is not None else None})
    print(f"\n########## {tag}: {len(events)} hole encounters, "
          f"{sum(1 for e in events if e['died'])} deaths ##########")
    for label, sel in (("SINGLE hole (pt11/14)", lambda e: not e["long"]),
                       ("LONG hole (pt12/13/15/16)", lambda e: e["long"])):
        xs = [e for e in events if sel(e)]
        if not xs:
            continue
        dd = [e for e in xs if e["died"]]
        print(f"\n--- {label}: n={len(xs)}  deaths={len(dd)} ({100*len(dd)/len(xs):.1f}%) ---")
        # bucket by d (cells from takeoff to the hole cell)
        buckets = defaultdict(lambda: [0, 0])
        for e in xs:
            if e["d"] is None:
                key = "not airborne"
            else:
                key = f"d {max(0.0, e['d']):.1f}"[:5]
                key = f"d {int(e['d'] // 0.5) * 0.5:.1f}-{int(e['d'] // 0.5) * 0.5 + 0.5:.1f}"
            buckets[key][1] += 1
            buckets[key][0] += 1 if e["died"] else 0
        for k in sorted(buckets, key=lambda s: (s == "not airborne", s)):
            dth, n = buckets[k]
            print(f"    {k:14} n={n:4}  deaths={dth:3} ({100*dth/n:5.1f}%)")
        # predicted threshold check
        pred_bad = [e for e in xs if e["d"] is not None
                    and (SPAN_A * e["v"] + SPAN_B) < e["d"] + 1.0 - GRACE]
        pred_ok = [e for e in xs if e["d"] is not None
                   and (SPAN_A * e["v"] + SPAN_B) >= e["d"] + 1.0 - GRACE]
        def rate(g):
            return f"{sum(1 for e in g if e['died'])}/{len(g)} = {100*sum(1 for e in g if e['died'])/max(1,len(g)):.0f}%"
        print(f"    span-model says CLIP : {rate(pred_bad)}")
        print(f"    span-model says CLEAR: {rate(pred_ok)}")
        # THE WAVE-14 MECHANISM METRIC: fraction of crossings entered PAST THE ARC APEX
        # (frac = d/span > 0.5 = descending). That is the airborne trap. Device deaths all sit at
        # frac >= 0.51; every one of the 858 free crossings is <= 0.41. A policy that has LEARNED the
        # fix drives TRAP% toward 0 -- it takes off so it is still rising when the hole arrives.
        fr = [(e["d"] / max(1e-6, SPAN_A * e["v"] + SPAN_B)) for e in xs if e["d"] is not None]
        trap = [f for f in fr if f > 0.5]
        if fr:
            print(f"    TRAP%: {len(trap)}/{len(fr)} = {100*len(trap)/len(fr):.2f}% of crossings "
                  f"entered PAST APEX (frac>0.5)   median frac {sorted(fr)[len(fr)//2]:.2f}")
    dd = [e for e in events if e["died"]]
    if dd:
        print(f"\n  deaths detail (d = cells takeoff->hole, wy = world_y at hole entry):")
        for e in sorted(dd, key=lambda x: x["dist"]):
            ds = f"{e['d']:.2f}" if e["d"] is not None else "  -- "
            print(f"    {e['src'][:26]:26} ep{e['ep']} @{e['dist']:6.0f} pt{e['pt']:>2} "
                  f"v{e['v']:4.0f} airborne={int(e['airborne'])} d={ds} wy={e['wy']:7.1f} "
                  f"span={SPAN_A*e['v']+SPAN_B:.2f}")
    return events


if __name__ == "__main__":
    globs = sys.argv[1:] or ["captures/ab_v78*.jsonl", "captures/ab_v74*.jsonl",
                             "captures/ab_v70*.jsonl", "captures/best_run_1*.jsonl"]
    paths = [p for g in globs for p in sorted(glob.glob(g))]
    analyse(paths, "DEVICE " + ", ".join(globs))
