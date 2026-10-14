#!/usr/bin/env python3
"""Does a step-DOWN jump fly FURTHER than a flat jump? (the v104 post-mortem hypothesis)

v103 and v104 both failed to move P(jump | step-down to low) off ~98.5%, and the reason is visible
in their own rollouts: in the SIM a step-down jump with a hazard within 4 cells kills 0.0% (n=41),
while on DEVICE the same situation kills 50.0% (v102, n=4) and 16.7% (v92b, n=6). The sim does not
reproduce the COST, so there is no gradient to unlearn the habit -- removing the death and the bonus
only made the jump reward-NEUTRAL.

THE HYPOTHESIS. On the phone a step-down jump leaves from HIGH ground and lands on LOW ground, so
the runner falls further and stays airborne LONGER -- the arc is stretched and sails into whatever
follows the cap. The sim uses a fixed jump_duration regardless of the height drop, so its arc lands
early and safely. If true, the fix is geometry (extend the airtime by the drop), not shaping.

WHAT THIS MEASURES, on device captures only: airtime in SECONDS and arc length in CELLS for every
jump, split by whether the runner crossed a level-DOWN / level-UP / no level change while airborne.
The human set is 30 Hz (0.033 s resolution) and is the one to trust; policy sets are 10 Hz.

    python experiments/stepdown_arc.py "captures/human_20260708_*.jsonl" --label HUMAN
"""
from __future__ import annotations
import sys as _sys, pathlib as _pl  # repo bootstrap: src/, cpp_sim/, scripts/*/, experiments/
_R = next(p for p in _pl.Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
_sys.path[:0] = [str(_R / "src"), str(_R / "cpp_sim"), *(str(d) for d in sorted((_R / "scripts").iterdir()) if d.is_dir()), str(_R / "experiments")]
import argparse, glob, sys
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).parent))
from hole_landing_forensic import load_eps, cell_seq  # noqa: E402

CREV_PT = {33, 34, 35, 36}


def jumps(path, hz):
    """One row per airborne phase: (kind, airtime_s, cells_spanned, v)."""
    rows = []
    for _, ep in load_eps(path):
        cells = cell_seq(ep)
        lvl = {}
        for i, fr in enumerate(ep):
            lvl[cells[i]] = int(fr.get("path_level") or 2)
        i = 0
        n = len(ep)
        while i < n:
            if ep[i].get("is_jumping") and not (i and ep[i - 1].get("is_jumping")):
                j = i
                while j + 1 < n and ep[j + 1].get("is_jumping"):
                    j += 1
                c0, c1 = cells[i], cells[j]
                # level before takeoff vs after landing
                before = lvl.get(max(0, c0 - 1), lvl.get(c0, 2))
                after = lvl.get(c1 + 1, lvl.get(c1, 2))
                crossed_crev = any((ep[k].get("path_type") in CREV_PT)
                                   or int(ep[k].get("obstacle") or 0) == 7
                                   for k in range(i, j + 1))
                if not crossed_crev:
                    kind = "flat (no crevasse)"
                elif after < before:
                    kind = f"step-DOWN {before}->{after}"
                elif after > before:
                    kind = f"step-UP {before}->{after}"
                else:
                    kind = "crevasse, same level"
                rows.append((kind, (j - i + 1) / hz, c1 - c0,
                             float(ep[i].get("velocity") or 0.0)))
                i = j + 1
            else:
                i += 1
    return rows


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
        rows = []
        for p in paths:
            rows += jumps(p, hz)
        print(f"\n=== {a.label or g} ===  {len(rows)} airborne phases @ {hz:.0f} Hz")
        groups = {}
        for k, t, c, v in rows:
            groups.setdefault(k.split()[0] if k.startswith(("step", "flat")) else k, []).append((k, t, c, v))
        for key in sorted(groups, key=lambda x: -len(groups[x])):
            g2 = groups[key]
            if len(g2) < 4:
                continue
            print(f"  {key:22} n={len(g2):4}  airtime median {median(t for _,t,_,_ in g2):5.3f}s"
                  f"   cells spanned median {median(c for _,_,c,_ in g2):4.1f}"
                  f"   v median {median(v for _,_,_,v in g2):5.0f}")
        # the exact contrast that matters
        down = [r for r in rows if r[0].startswith("step-DOWN")]
        flat = [r for r in rows if r[0].startswith("flat")]
        if len(down) >= 4 and len(flat) >= 4:
            dt, ft = median(t for _, t, _, _ in down), median(t for _, t, _, _ in flat)
            print(f"\n  >>> step-DOWN airtime {dt:.3f}s vs FLAT {ft:.3f}s = {dt/max(1e-9,ft):.2f}x"
                  f"   ({(dt-ft)*1000:+.0f} ms)")


if __name__ == "__main__":
    main()
