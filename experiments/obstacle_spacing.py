#!/usr/bin/env python3
"""WAVE 14: inter-obstacle SPACING, device vs sim, same estimator.

The airborne-trap (arrive at a hole still descending from a previous jump) needs the next
must-jump obstacle to sit INSIDE the jump arc (~2.1-3.1 cells). The sim's compound tails are
placed `gap_cells`(2) clear cells AFTER a 2-cell wide-gap structure => the 2nd obstacle sits ~4-5
cells out, BEYOND the arc, so the runner always lands first and re-jumps => d<=1 always, the trap
is unreachable in-sim. This measures the real device spacing distribution to calibrate the fix.

Usage:
  python experiments/obstacle_spacing.py device "captures/ab_v78*.jsonl" ...
  python experiments/obstacle_spacing.py sim experiments/_sim_holes.jsonl
"""
from __future__ import annotations
import sys as _sys, pathlib as _pl  # repo bootstrap: src/, cpp_sim/, scripts/*/, experiments/
_R = next(p for p in _pl.Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
_sys.path[:0] = [str(_R / "src"), str(_R / "cpp_sim"), *(str(d) for d in sorted((_R / "scripts").iterdir()) if d.is_dir()), str(_R / "experiments")]
import glob, sys
from collections import Counter
from hole_landing_forensic import load_eps, cell_seq

MUST_JUMP = {1, 3, 4, 7}     # Hole, TreeStumble, TreeJump, LevelChange
HOLE = 1


def obst_of(fr):
    o = fr.get("obstacle")
    return int(o) if o is not None else 0


PAIR = Counter()


def spacings(paths):
    gaps, gaps_to_hole = [], []
    for path in paths:
        for epk, ep in load_eps(path):
            cells = cell_seq(ep)
            seq = {}                       # cell index -> obstacle (first frame on the cell wins)
            for i, fr in enumerate(ep):
                seq.setdefault(cells[i], obst_of(fr))
            idxs = sorted(k for k, v in seq.items() if v in MUST_JUMP)
            for a, b in zip(idxs, idxs[1:]):
                # collapse multi-cell structures (a 2-cell HoleLong / crevasse is ONE obstacle)
                if b - a <= 1 and seq[a] == seq[b]:
                    continue
                gaps.append(b - a)
                if seq[b] == HOLE:
                    gaps_to_hole.append(b - a)
                if b - a <= 3:
                    PAIR[(seq[a], seq[b], b - a)] += 1
    return gaps, gaps_to_hole


def report(tag, gaps, gaps_to_hole):
    print(f"\n=== {tag} ===")
    for name, g in (("must-jump -> must-jump", gaps), ("must-jump -> HOLE", gaps_to_hole)):
        if not g:
            continue
        c = Counter(g)
        tot = len(g)
        inside = sum(n for k, n in c.items() if k <= 3)
        print(f"  {name}: n={tot}  median={sorted(g)[tot//2]}  "
              f"<=3 cells (INSIDE the jump arc) = {inside} ({100*inside/tot:.1f}%)")
        print("     gap cells: " + "  ".join(f"{k}:{c[k]}" for k in sorted(c) if k <= 10))


NAME = {1: "Hole", 3: "TreeStumble", 4: "TreeJump", 7: "LevelChange"}

if __name__ == "__main__":
    mode = sys.argv[1]
    paths = [p for g in sys.argv[2:] for p in sorted(glob.glob(g))]
    report(mode.upper() + " " + ",".join(sys.argv[2:]), *spacings(paths))
    print("\n  INSIDE-ARC pairs (prev -> next, gap cells), most common first:")
    for (a, b, g), n in PAIR.most_common(14):
        print(f"     {NAME.get(a,a):12} -> {NAME.get(b,b):12}  gap {g}   n={n}")
