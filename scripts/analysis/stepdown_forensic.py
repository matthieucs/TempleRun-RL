#!/usr/bin/env python3
"""STEP-DOWN (mid->low) OVER-JUMP — the user's second live report, measured.

USER (2026-07-27, watching v102): "The problem is also the mid to low transition when you have an
obstacle right ahead. The policy learnt to jump when it goes from the mid to the low ground. It does
not need to. When there is an obstacle, either you jump early or you don't jump, but the policy
can't do that right now."

The claim decomposes into three measurable parts, and each is answered here:

  (1) DOES IT NEED TO JUMP?  Split every crevasse crossing by level delta (step DOWN vs step UP) and
      report the death rate when the runner crossed it GROUNDED. If a grounded step-down is survived
      on device, the jump is optional there and any policy that always jumps has learnt a superstition.
  (2) DOES THE POLICY DO IT ANYWAY?  P(jump | step-down) for each policy vs the HUMAN. The human is
      the control: they see the same track and choose freely.
  (3) DOES IT COST ANYTHING?  Death rate for step-downs that have a hazard within 1-4 cells after the
      cap -- the "obstacle right ahead" case -- split by whether the runner jumped the step-down.
      The user's mechanic is that the habitual mid-timed jump is simultaneously too late to clear the
      obstacle and too committed to react to it.

    python scripts/analysis/stepdown_forensic.py captures/w17_v102_device.jsonl --label v102
    python scripts/analysis/stepdown_forensic.py "captures/human_20260708_*.jsonl" --label HUMAN
"""
from __future__ import annotations
import sys as _sys, pathlib as _pl  # repo bootstrap: src/, cpp_sim/, scripts/*/, experiments/
_R = next(p for p in _pl.Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
_sys.path[:0] = [str(_R / "src"), str(_R / "cpp_sim"), *(str(d) for d in sorted((_R / "scripts").iterdir()) if d.is_dir()), str(_R / "experiments")]
import argparse, glob, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from hole_landing_forensic import load_eps, cell_seq  # noqa: E402

CREV_PT = {33, 34, 35, 36}
CREV_OBST = 7
MUST_JUMP = {1, 3, 4, 7}
MUST_SLIDE = {5, 6}
NAME = {0: "-", 1: "Hole", 2: "Narrow", 3: "Stumble", 4: "TreeJump", 5: "TreeSlide",
        6: "Flame", 7: "Crevasse"}


def crossings(path, hz):
    """One row per crevasse structure crossed."""
    rows, km = [], 0.0
    for _, ep in load_eps(path):
        cells = cell_seq(ep)
        km += max(float(f.get("distance") or 0) for f in ep) / 1000.0
        # per-cell summary
        cell_obst, cell_lvl, cell_first = {}, {}, {}
        for i, fr in enumerate(ep):
            c = cells[i]
            o = int(fr.get("obstacle") or 0)
            pt = fr.get("path_type")
            is_cr = (pt in CREV_PT) or (o == CREV_OBST)
            if c not in cell_obst or (not cell_obst[c] and (o or is_cr)):
                cell_obst[c] = CREV_OBST if is_cr else o
            cell_lvl.setdefault(c, int(fr.get("path_level") or 2))
            cell_lvl[c] = int(fr.get("path_level") or 2)
            cell_first.setdefault(c, i)
        death_cell = cells[-1] if ep[-1].get("is_dead") else None

        ordered = sorted(cell_obst)
        k = 0
        while k < len(ordered):
            c = ordered[k]
            if cell_obst.get(c) != CREV_OBST:
                k += 1
                continue
            start = c
            end = c
            while end + 1 in cell_obst and cell_obst[end + 1] == CREV_OBST:
                end += 1
            # level BEFORE the structure vs AFTER it
            before = cell_lvl.get(start - 1, cell_lvl.get(start, 2))
            after = cell_lvl.get(end + 1, cell_lvl.get(end, 2))
            delta = after - before
            i0, i1 = cell_first.get(start, 0), cell_first.get(end, 0)
            # was the runner AIRBORNE at any point across the structure?
            seg = range(i0, min(len(ep), cell_first.get(end + 1, len(ep))))
            jumped = any(ep[j].get("is_jumping") for j in seg)
            # what sits right after the cap?
            nxt, nxt_gap = 0, None
            for d in range(1, 6):
                o = cell_obst.get(end + d, 0)
                if o in MUST_JUMP or o in MUST_SLIDE:
                    nxt, nxt_gap = o, d
                    break
            killed = death_cell is not None and start <= death_cell <= end + 5
            fr = ep[i0]
            rows.append(dict(delta=delta, before=before, after=after, cells=end - start + 1,
                             jumped=jumped, nxt=nxt, nxt_gap=nxt_gap, killed=killed,
                             v=float(fr.get("velocity") or 0.0),
                             dist=float(fr.get("distance") or 0.0)))
            k = ordered.index(end) + 1 if end in ordered else k + 1
    return rows, km


def pct(a, b):
    return f"{100*a/b:5.1f}%" if b else "    - "


def report(rows, km, label):
    print(f"\n=== {label} ===  {len(rows)} crevasse crossings over {km:.1f} km")
    if not rows:
        return
    down = [r for r in rows if r["delta"] < 0]
    up = [r for r in rows if r["delta"] > 0]
    flat = [r for r in rows if r["delta"] == 0]
    print(f"  step DOWN {len(down)}   step UP {len(up)}   same-level {len(flat)}")
    for nm, grp in (("STEP-DOWN", down), ("STEP-UP", up)):
        if not grp:
            continue
        j = [r for r in grp if r["jumped"]]
        g = [r for r in grp if not r["jumped"]]
        print(f"  {nm:10} P(jump) {pct(len(j), len(grp))}"
              f"   deaths: jumped {pct(sum(r['killed'] for r in j), len(j))} (n={len(j)})"
              f"   GROUNDED {pct(sum(r['killed'] for r in g), len(g))} (n={len(g)})")
    # the user's case: a step-down with a hazard right after
    tight = [r for r in down if r["nxt_gap"] is not None and r["nxt_gap"] <= 4]
    if tight:
        j = [r for r in tight if r["jumped"]]
        g = [r for r in tight if not r["jumped"]]
        print(f"  STEP-DOWN + hazard within 4 cells: n={len(tight)}"
              f"  ({100*len(tight)/max(1,len(down)):.0f}% of step-downs)")
        print(f"     jumped   n={len(j):3}  deaths {pct(sum(r['killed'] for r in j), len(j))}")
        print(f"     grounded n={len(g):3}  deaths {pct(sum(r['killed'] for r in g), len(g))}")
        c = Counter((NAME.get(r["nxt"], r["nxt"]), r["nxt_gap"]) for r in tight)
        print("     what follows (type, gap):  "
              + "  ".join(f"{k[0]}@{k[1]} x{v}" for k, v in c.most_common(6)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("captures", nargs="+")
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    for g in a.captures:
        paths = sorted(glob.glob(g))
        if not paths:
            continue
        rows, km = [], 0.0
        for p in paths:
            r, k = crossings(p, 30.0 if "human" in p else 10.0)
            rows += r
            km += k
        report(rows, km, a.label or g)


if __name__ == "__main__":
    main()
