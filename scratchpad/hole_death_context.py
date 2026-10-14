#!/usr/bin/env python3
"""WAVE 14 opener: dump the frame-exact context of every device death (v78/v74 era) so the
HOLE death mechanic can be classified. Prints the last N frames before death with the fields
that matter for a jump/gap decision. Usage:
    python scratchpad/hole_death_context.py captures/ab_v78_r1.jsonl [more...]
"""
from __future__ import annotations
import glob, json, sys
from collections import defaultdict

ACT = {0: "SL", 1: "SR", 2: "JMP", 3: "SLD", 4: "TL", 5: "TR", 6: "-"}
DT = {1: "Fall", 2: "Tree", 3: "Slide", 4: "Ledge", 5: "Burnt"}


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


N = int(sys.argv[-1]) if sys.argv[-1].isdigit() else 16
paths = [p for g in sys.argv[1:] if not g.isdigit() for p in sorted(glob.glob(g))]
for path in paths:
    for epk, ep in load_eps(path):
        last = ep[-1]
        if not last.get("is_dead"):
            continue
        dt = int(last.get("death_type") or 0)
        print(f"\n=== {path.split('/')[-1]} ep{epk}  dist={float(last.get('distance') or 0):.0f}  "
              f"death={DT.get(dt, dt)}({dt})  pt={last.get('path_type')}  lvl={last.get('path_level')} ===")
        print("   step  dist   pt lvl obst  cp    v    lane  mj nmj n2mj n3mj  jmp sld  wy    vy    act")
        for fr in ep[-N:]:
            print(f"  {fr.get('step',0):5} {float(fr.get('distance') or 0):6.0f} "
                  f"{str(fr.get('path_type')):>4} {str(fr.get('path_level')):>3} "
                  f"{str(fr.get('obstacle')):>4} {float(fr.get('cell_progress') or 0):.2f} "
                  f"{float(fr.get('velocity') or 0):5.0f} {float(fr.get('lane') or 0):6.2f} "
                  f"{int(bool(fr.get('must_jump')))}  {int(bool(fr.get('next_must_jump')))}  "
                  f"{int(bool(fr.get('next2_must_jump')))}    {int(bool(fr.get('next3_must_jump')))}   "
                  f"{int(bool(fr.get('is_jumping')))}   {int(bool(fr.get('is_sliding')))}  "
                  f"{float(fr.get('world_y') or 0):6.1f} {float(fr.get('vel_y') or 0):6.1f}  "
                  f"{ACT.get(fr.get('action'), fr.get('action'))}")
