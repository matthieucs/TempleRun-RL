#!/usr/bin/env python3
"""TAKEOFF-TIMING DETECTOR — the mid->high crevasse, named and measured.

USER (2026-07-27, after reporting this for weeks): "The mid to high transition is not fixed. The
policy jumps way too early. I don't know why. Is it because it has no penalty to do so in the sim?
Have you modelled the case when the character takes a hit and is slowed down? This should be
penalized. It is clearly not."

Both halves of that are correct, and this script measures the first half.

THE PHYSICS. A jump spans span(v) = 0.0143*v - 0.11 cells, so the FASTER you run the FURTHER along
the track your arc reaches. A crevasse cap sits a fixed number of CELLS ahead. To still be near the
apex when you arrive over a RAISED (level-3) cap you must therefore take off LATER as speed rises.
Take off early and the arc is already descending at the cap -> the runner clips the wall -> Stumble
(x0.85 velocity). It rarely kills; it just bleeds speed, and a slower runner mistimes everything
after it.

WHAT IT REPORTS: takeoff cell_progress bucketed by speed, plus the SLOPE of that relationship. The
sign of the slope is the whole story:
    HUMAN    cp 0.47 -> 0.66 -> 0.82 as speed rises   (POSITIVE = delays takeoff = correct)
    POLICY   cp 0.79 -> 0.52 -> 0.37                  (NEGATIVE = INVERTED = clips the cap)
Also reports graze rate and the velocity actually bled per km, so the cost is visible even though
almost nothing dies.

    python scripts/takeoff_timing.py captures/w17_v92b_device.jsonl
    python scripts/takeoff_timing.py "captures/human_20260708_*.jsonl" --label HUMAN
"""
from __future__ import annotations
import argparse, glob, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scratchpad"))
from hole_landing_forensic import load_eps, cell_seq  # noqa: E402

CREV_PT = {33, 34, 35, 36}
CREV_OBST = 7


def collect(paths, hz):
    """One row per crevasse crossing: (level, speed, takeoff cp, grazed, velocity lost)."""
    rows, km = [], 0.0
    for p in paths:
        for _, ep in load_eps(p):
            cells = cell_seq(ep)
            km += max(float(f.get("distance") or 0) for f in ep) / 1000.0
            takeoff = None
            for i, fr in enumerate(ep):
                if fr.get("is_jumping") and not (i and ep[i - 1].get("is_jumping")):
                    src = ep[i - 1] if i else fr
                    takeoff = (float(src.get("cell_progress") or 0.0),
                               float(src.get("velocity") or 0.0), cells[max(0, i - 1)])
                crev = (fr.get("path_type") in CREV_PT) or (int(fr.get("obstacle") or 0) == CREV_OBST)
                prev_crev = i and ((ep[i - 1].get("path_type") in CREV_PT)
                                   or (int(ep[i - 1].get("obstacle") or 0) == CREV_OBST))
                if not crev or prev_crev or takeoff is None:
                    continue
                # the CAP carries the NEW zone level and is the LAST cell of the structure
                lvl = int(fr.get("path_level") or 2)
                for j in range(i, min(len(ep), i + 12)):
                    f2 = ep[j]
                    if (f2.get("path_type") in CREV_PT) or (int(f2.get("obstacle") or 0) == CREV_OBST):
                        lvl = max(lvl, int(f2.get("path_level") or 2))
                    else:
                        break
                grazed, v_in = False, float(fr.get("velocity") or 0.0)
                v_out = v_in
                for j in range(i, min(len(ep), i + int(1.2 * hz))):
                    f2 = ep[j]
                    if f2.get("is_stumbling") and not ep[max(0, j - 1)].get("is_stumbling"):
                        grazed = True
                    if f2.get("is_dead"):
                        break
                    v_out = float(f2.get("velocity") or v_out)
                rows.append((lvl, takeoff[1], takeoff[0], grazed, max(0.0, v_in - v_out)))
                takeoff = None
    return rows, km


def report(rows, km, label):
    lvl3 = [r for r in rows if r[0] >= 3]
    print(f"\n=== {label} ===  {len(rows)} crevasse crossings over {km:.1f} km "
          f"({len(lvl3)} onto a RAISED level-3 cap)")
    if not lvl3:
        print("  no level-3 caps in this capture")
        return
    grazed = [r for r in lvl3 if r[3]]
    bled = sum(r[4] for r in lvl3)
    print(f"  level-3 GRAZE rate {100*len(grazed)/len(lvl3):.1f}%  "
          f"({len(grazed)}/{len(lvl3)})   velocity bled {bled/max(1e-9,km):.1f} per km")
    buckets = defaultdict(list)
    for _, v, cp, _, _ in lvl3:
        buckets[int(v // 30) * 30].append(cp)
    print("  TAKEOFF cell_progress by speed (must RISE with speed — the arc spans more cells):")
    pts = []
    for k in sorted(buckets):
        b = sorted(buckets[k])
        if len(b) >= 4:
            med = b[len(b) // 2]
            pts.append((k + 15, med))
            print(f"     v{k:3}-{k+29:3}  n={len(b):3}  median cp {med:.2f}")
    # Regress on EVERY crossing, not on 3 bucket medians. The top bucket holds only n~7 (the velocity
    # cap is 210), so a 3-point fit swung between -0.0035 and +0.0000 across arms that differ little —
    # the same underpowered-verdict trap as an 8-episode device median. Report a standard error so
    # "inverted" is only claimed when the sign is actually resolved.
    xs = [r[1] for r in lvl3]
    ys = [r[2] for r in lvl3]
    n = len(xs)
    if n >= 8:
        mx, my = sum(xs) / n, sum(ys) / n
        den = sum((x - mx) ** 2 for x in xs)
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else 0.0
        inter = my - slope * mx
        resid = [y - (inter + slope * x) for x, y in zip(xs, ys)]
        s2 = sum(r * r for r in resid) / max(1, n - 2)
        se = (s2 / den) ** 0.5 if den else float("inf")
        sig = abs(slope) > 2 * se
        verdict = ("CORRECT (delays takeoff at speed)" if slope > 0 and sig else
                   "** INVERTED — takes off EARLIER the faster it runs = the wall clip **"
                   if slope < 0 and sig else
                   "NOT RESOLVED at this n (slope within 2 s.e. of zero)")
        print(f"  slope d(cp)/d(v) = {slope:+.5f} +- {se:.5f} (1 s.e., n={n} crossings)  -> {verdict}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("captures", nargs="+")
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    for g in a.captures:
        paths = sorted(glob.glob(g))
        if not paths:
            continue
        hz = 30.0 if "human" in g else 10.0
        rows, km = collect(paths, hz)
        report(rows, km, a.label or g)


if __name__ == "__main__":
    main()
