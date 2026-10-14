#!/usr/bin/env python3
"""WAVE 14: are ONE-cell crevasses harder than TWO-cell ones on DEVICE?

The sim modelled every crevasse as 2 cells (gap + raised ledge, apex check on the ledge). Device is
67.5% one-cell (n=409). Making the 1-cell form carry the apex check on its ONLY cell shifts the
required takeoff a full cell earlier -- if the device does NOT punish 1-cell crevasses more than
2-cell ones, that model is too harsh and needs softening (offset/cap), not just a probability.

Reports, per structure length: encounters, deaths (dt2 Ledge / dt1 Fall at or within 4 frames of the
structure) and stumble (graze) rate.
"""
from __future__ import annotations
import sys as _sys, pathlib as _pl  # repo bootstrap: src/, cpp_sim/, scripts/*/, experiments/
_R = next(p for p in _pl.Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
_sys.path[:0] = [str(_R / "src"), str(_R / "cpp_sim"), *(str(d) for d in sorted((_R / "scripts").iterdir()) if d.is_dir()), str(_R / "experiments")]
import glob, sys
from collections import defaultdict
from hole_landing_forensic import load_eps, cell_seq

LEVELCHANGE = 7


def main(globs):
    paths = [p for g in globs for p in sorted(glob.glob(g))]
    stats = defaultdict(lambda: {"n": 0, "died": 0, "stumb": 0, "dt": defaultdict(int)})
    for path in paths:
        for epk, ep in load_eps(path):
            cells = cell_seq(ep)
            first = {}
            obst = {}
            for i, fr in enumerate(ep):
                c = cells[i]
                obst.setdefault(c, int(fr.get("obstacle") or 0))
                first.setdefault(c, i)
            ks = sorted(obst)
            i = 0
            while i < len(ks):
                if obst[ks[i]] != LEVELCHANGE:
                    i += 1
                    continue
                j = i
                while j + 1 < len(ks) and ks[j + 1] == ks[j] + 1 and obst[ks[j + 1]] == LEVELCHANGE:
                    j += 1
                length = j - i + 1
                s = stats[length]
                s["n"] += 1
                fi = first[ks[i]]
                end = min(len(ep), first[ks[j]] + 8)
                died = False
                for q in range(fi, end):
                    if ep[q].get("is_dead"):
                        died = True
                        s["dt"][int(ep[q].get("death_type") or 0)] += 1
                        break
                    if ep[q].get("is_stumbling") and not ep[max(0, q - 1)].get("is_stumbling"):
                        s["stumb"] += 1
                s["died"] += int(died)
                i = j + 1
    print("DEVICE crevasse outcome by structure length:")
    for L in sorted(stats):
        s = stats[L]
        print(f"  {L}-cell: n={s['n']:4}  deaths={s['died']:3} ({100*s['died']/max(1,s['n']):4.1f}%)  "
              f"stumbles={s['stumb']:3} ({100*s['stumb']/max(1,s['n']):4.1f}%)  dt={dict(s['dt'])}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["captures/ab_v7*.jsonl", "captures/w14_v78_holes.jsonl",
                          "captures/ab_v66*.jsonl", "captures/best_run_1*.jsonl"])
